// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Implementation of spectral clustering for leaflet classification.

use std::{
    cmp::Ordering,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::lanczos::{Hermitian, Order};
use getset::Getters;
use groan_rs::{prelude::Dimension, system::System};
use hashbrown::{HashMap, HashSet};
use nalgebra::{DMatrix, SymmetricEigen};
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::{
    errors::{AnalysisError, ClusterError},
    PANIC_MESSAGE,
};

use super::{common::macros::group_name, pbc::PBCHandler};

/// Number of iterations for the Lanczos algorithm.
const LANCZOS_ITERATIONS: usize = 300;
/// Number of clusters to identify.
const N_CLUSTERS: usize = 2;
/// Sigma value for construction of the similarity matrix when the algorithm is sloppy.
const SLOPPY_SIGMA: f32 = 0.5;
/// Sigma value for construction of the similarity matrix when the algorithm is precise.
const PRECISE_SIGMA: f32 = 1.0;
/// Distance cut-off used for the sloppy algorithm (in nm).
const DISTANCE_CUTOFF: f32 = 6.0;

/// Maximal number of times the sloppy method can fail in a row before we switch to the precise method.
const MAX_SLOPPY_FAILS: u8 = 3;

/// Precise assignment will always be performed below this number of headgroups.
const PRECISE_LOWER_LIMIT: usize = 1000;
/// Precise assignment will never be performed above this number of headgroups (unless the sloppy method fails 3 times).
const PRECISE_UPPER_LIMIT: usize = 5000;

/// Relative number of lipids molecules that must remain in the same leaflet
/// between two consecutive trajectory frames analyzed by the same thread
/// for reliable leaflet match.
const CLUSTER_CLASSIFICATION_LIMIT: f32 = 0.8;

/// [`TIMEOUT`] in seconds.
const TIMEOUT_SECONDS: u64 = 120;
/// Global soft timeout duration for spin-lock used when fetching data for cluster assignment.
/// After this time a warning is logged.
static TIMEOUT: Lazy<Duration> = Lazy::new(|| Duration::from_secs(TIMEOUT_SECONDS));

/// [`HARD_TIMEOUT`] in seconds.
const HARD_TIMEOUT_SECONDS: u64 = 720;
/// Global HARD timeout duration for spin-lock used when fetching data for cluster assignment.
/// After this time a PANIC is raised.
static HARD_TIMEOUT: Lazy<Duration> = Lazy::new(|| Duration::from_secs(HARD_TIMEOUT_SECONDS));

#[derive(Debug, Clone, Getters)]
pub(crate) struct SystemClusterClassification {
    /// Converts between global index of an atom and its index in the similarity matrix.
    converter: HashMap<usize, usize>,
    /// Clusters assigned for the first frame. Shared among all threads.
    reference_clusters: Arc<Mutex<Option<Clusters>>>,
    /// Clusters assigned for the current frame.
    #[getset(get = "pub(super)")]
    clusters: Option<Clusters>,
    /// Number of fails that were encountered using the sloppy method in a row.
    sloppy_fails: u8,
    /// Flip leaflet assignment.
    flip: bool,
}

impl SystemClusterClassification {
    /// Create a new structure for clustering.
    pub(super) fn new(system: &System, flip: bool) -> Self {
        let mut converter = HashMap::new();
        for (i, atom) in system
            .group_iter(group_name!("ClusterHeads"))
            .unwrap_or_else(|_| panic!("Group `ClusterHeads` should exist.  {}", PANIC_MESSAGE))
            .enumerate()
        {
            converter.insert(atom.get_index(), i);
        }

        SystemClusterClassification {
            converter,
            reference_clusters: Arc::new(Mutex::new(None)),
            clusters: None,
            sloppy_fails: 0,
            flip,
        }
    }

    /// Assign lipid headgroups into individual leaflets using spectral clustering.
    ///
    /// ## Algorithmic details
    /// Spectral clustering works by creating a graph structure from the provided headgroup
    /// atoms (atoms are nodes) and assigning weights to the individual edges based on
    /// the distance between the atoms (between 0 and 1, inclusive).
    /// The closer the atoms are to each other, the higher the weight of the edge connecting them.
    ///
    /// Once this 'similarity matrix' is constructed, we calculate a normalized Laplacian from it
    /// and then perform eigendecomposition of the Laplacian. We use the two smallest eigenvectors
    /// with non-zero eigenvalues and perform k-means clustering on them.
    ///
    /// From the k-means clustering, we obtain the assignments of individual headgroup atoms
    /// to clusters.
    ///
    /// Ideally, the 'similarity matrix' should be constructed by calculating the distances
    /// between all pairs of atoms, and a complete eigendecomposition should be performed for the
    /// Laplacian. We call this method the 'precise' method.
    ///
    /// The 'precise' method is however VERY computationally expensive for large systems
    /// (it scales as `O(n^3)`, where `n` is the number of particles).
    ///
    /// It is in many cases sufficient to only calculate the distances for the 'similarity matrix'
    /// between nearby atoms (using CellGrid) and then only calculate several smallest eigenvectors
    /// using the Lanczos method. We call this method the 'sloppy' method.
    /// The 'sloppy' method has some problems:
    /// - It is heuristic and may fail even for simple geometries depending on the random seed used.
    /// - It will very likely fail for more complex geometries. Fortunately, the failure is typically
    ///   catastrophic, as in it is very easy to recognize when it happens.
    /// - For small systems, the Lanczos method performed at sufficient precision can be
    ///   slower than full eigendecomposition.
    ///
    /// This function attempts to intelligently select the appropriate method to balance
    /// computational expense and accuracy. It also attempts to validate that the
    /// assignment did not catastrophically fail.
    ///
    /// For the first analyzed frame, we aim to perform 'precise' clustering since we
    /// need robust reference clusters. Only when clustering more than [`PRECISE_UPPER_LIMIT`]
    /// atoms (lipid molecules) do we perform 'sloppy' clustering. To validate it (remember: 'sloppy' clustering is heuristic),
    /// we perform it up to three times until reaching two identical results in two independent runs.
    /// If all three runs provide different results, we return an error.
    /// When performing the 'precise' clustering, we assume that it is correct. If this assumption proves to be
    /// wrong, we will most likely get to know in the following frame.
    ///
    /// For all other analyzed frames, we try to perform 'sloppy' clustering unless the system is very small
    /// (fewer than [`PRECISE_LOWER_LIMIT`] lipid molecules), in which case it will actually be faster to
    /// perform 'precise' clustering.
    /// For most systems, we perform up to three 'sloppy' clustering runs until we reach a 'match' with
    /// clusters identified in the previous trajectory frame analyzed by the same thread
    /// (or in the first, reference frame, if this is the first frame analyzed by this thread).
    /// To see what constitutes a 'match' between clusters, see [`SystemClusterClassification::classify_clusters`].
    /// Simply put, clusters 'match' if they do not differ too much from each other.
    /// This is heuristic again and may fail if the frames are not sufficiently correlated, but this should be extremely rare.
    ///
    /// In case none of the three 'sloppy' clustering runs returns a 'match', we switch to 'precise' clustering,
    /// unless the system is too large (larger than [`PRECISE_UPPER_LIMIT`], in which case we return an error).
    /// We only perform one 'precise' clustering and try to match the clusters again. If this fails, we return an error.
    ///
    /// If 'sloppy' clustering fails in [`MAX_SLOPPY_FAILS`] consecutive analyzed frames, we switch to precise clustering
    /// permanently.
    ///
    /// When performing 'precise' clustering (small systems or if 'sloppy' clustering fails too much),
    /// we always perform only one 'precise' run since it is not heuristic. We attempt to match the results
    /// to clusters from the previous frame, and if this fails, we immediately return an error.
    pub(super) fn cluster<'a>(
        &mut self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
        frame_index: usize,
    ) -> Result<(), AnalysisError> {
        if frame_index == 0 {
            if self.converter.len() > PRECISE_UPPER_LIMIT {
                self.sloppy_cluster_frame_one(system, pbc)?;
            } else {
                self.precise_cluster_frame_one(system, pbc)?;
            }

            // print information about cluster classification
            self.clusters
                .as_ref()
                .expect(PANIC_MESSAGE)
                .log_info(self.flip);
        } else if self.converter.len() > PRECISE_LOWER_LIMIT && self.sloppy_fails < MAX_SLOPPY_FAILS
        {
            // system is large and sloppy method did not fail often enough
            let matrix =
                self.create_similarity_matrix(system, pbc, DISTANCE_CUTOFF, SLOPPY_SIGMA)?;
            let laplacian = Self::create_normalized_laplacian(&matrix);
            let n = matrix.shape().0;

            // try sloppy method up to three times (sloppy is heuristic)
            for _ in 0..3 {
                if let Some(valid_cluster) = self.sloppy_clustering(&laplacian, n, frame_index) {
                    self.clusters = Some(valid_cluster);
                    // reset sloppy fails counter
                    self.sloppy_fails = 0;
                    return Ok(());
                }
            }

            // still no luck with assignment?, increase the sloppy fails counter and use the precise method
            self.sloppy_fails += 1;

            // do not perform precise clustering if the system is very large
            if self.converter.len() > PRECISE_UPPER_LIMIT {
                return Err(AnalysisError::ClusterError(
                    ClusterError::CouldNotMatchLeaflets(
                        (CLUSTER_CLASSIFICATION_LIMIT * 100.0) as u8,
                    ),
                ));
            }

            let matrix =
                self.create_similarity_matrix(system, pbc, f32::INFINITY, PRECISE_SIGMA)?;
            let laplacian = Self::create_normalized_laplacian(&matrix);
            let n = matrix.shape().0;

            match self.precise_clustering(laplacian, n, frame_index) {
                Some(x) => self.clusters = Some(x),
                None => {
                    return Err(AnalysisError::ClusterError(
                        ClusterError::CouldNotMatchLeaflets(
                            (CLUSTER_CLASSIFICATION_LIMIT * 100.0) as u8,
                        ),
                    ))
                }
            }
        } else {
            // system is small => use precise method
            let matrix =
                self.create_similarity_matrix(system, pbc, f32::INFINITY, PRECISE_SIGMA)?;
            let laplacian = Self::create_normalized_laplacian(&matrix);
            let n = matrix.shape().0;

            match self.precise_clustering(laplacian, n, frame_index) {
                Some(x) => self.clusters = Some(x),
                None => {
                    return Err(AnalysisError::ClusterError(
                        ClusterError::CouldNotMatchLeaflets(
                            (CLUSTER_CLASSIFICATION_LIMIT * 100.0) as u8,
                        ),
                    ))
                }
            }
        }

        Ok(())
    }

    /// Perform sloppy leaflet assignment for the first frame. Try it at least two times and check for matches.
    fn sloppy_cluster_frame_one<'a>(
        &mut self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<(), AnalysisError> {
        let matrix = self.create_similarity_matrix(system, pbc, DISTANCE_CUTOFF, SLOPPY_SIGMA)?;
        let laplacian = Self::create_normalized_laplacian(&matrix);
        let n = matrix.shape().0;

        let cluster1 = self.sloppy_clustering(&laplacian, n, 0);
        let cluster2 = self.sloppy_clustering(&laplacian, n, 0);

        match (cluster1, cluster2) {
            (None, _) | (_, None) => {
                return Err(AnalysisError::ClusterError(
                    ClusterError::SloppyFirstFrameFail,
                ))
            }
            (Some(x), Some(y)) if x == y => {
                // assignment successful
                self.clusters = Some(x);
            }
            (Some(x), Some(y)) if x != y => {
                // assignment mismatch => run one more time
                let cluster3 = self.sloppy_clustering(&laplacian, n, 0).ok_or(
                    AnalysisError::ClusterError(ClusterError::SloppyFirstFrameFail),
                )?;
                if x == cluster3 {
                    self.clusters = Some(x);
                } else if y == cluster3 {
                    self.clusters = Some(y);
                } else {
                    return Err(AnalysisError::ClusterError(
                        ClusterError::SloppyFirstFrameFail,
                    ));
                }
            }
            _ => unreachable!("FATAL GORDER ERROR | SystemClusterClassification::sloppy_cluster_frame_one | Unreachable pattern reached."),
        }

        // update shared data
        self.update_shared();

        Ok(())
    }

    /// Perform precise leaflet assignment for the first frame.
    fn precise_cluster_frame_one<'a>(
        &mut self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<(), AnalysisError> {
        let matrix = self.create_similarity_matrix(system, pbc, f32::INFINITY, PRECISE_SIGMA)?;
        let laplacian = Self::create_normalized_laplacian(&matrix);
        let n = matrix.shape().0;
        let clusters = self
            .precise_clustering(laplacian, n, 0)
            .unwrap_or_else(||
                panic!("FATAL GORDER ERROR | SystemClusterClassification::precise_cluster_frame_one | Could not classify clusters in frame 0. {}", 
                PANIC_MESSAGE)
            );

        self.clusters = Some(clusters);

        // update shared data
        self.update_shared();

        Ok(())
    }

    /// Copy the current clusters assignment to shared data.
    fn update_shared(&mut self) {
        let mut shared_clusters = self.reference_clusters.lock();
        *shared_clusters = self.clusters.clone();
        // Defensive check
        assert!(shared_clusters.is_some(), "FATAL GORDER ERROR | SystemClusterClassification::sloppy_cluster_frame_one | Shared clusters is None, after assigning. {}", PANIC_MESSAGE);
    }

    /// Get clusters assignment from shared data.
    fn get_from_shared(&self) -> Clusters {
        let start_time = Instant::now();
        let mut warning_logged = false;

        // spin-lock: waiting for the requested frame to become available
        loop {
            let shared_clusters = self.reference_clusters.lock();
            if let Some(clusters) = shared_clusters.clone() {
                return clusters;
            }

            // defensive check for a deadlock
            if start_time.elapsed() > *TIMEOUT {
                if !warning_logged {
                    colog_warn!("DEADLOCKED? Thread has been waiting for shared clustering data (frame '0') for more than {} seconds.
This may be due to extreme system size, resource contention or a bug. 
Ensure that your CPU is not oversubscribed and that you have not lost access to the trajectory file.
If `gorder` is causing oversubscription, reduce the number of threads used for the analysis.
If other computationally intensive software is running alongside `gorder`, consider terminating it.
If the issue persists, please report it by opening an issue at `github.com/Ladme/gorder/issues` or sending an email to `ladmeb@gmail.com`.
(Note: If no progress is made, this thread will terminate in {} seconds to prevent resource exhaustion.)",
                    TIMEOUT_SECONDS,
                    HARD_TIMEOUT_SECONDS - TIMEOUT_SECONDS,
                );
                    warning_logged = true;
                }

                if start_time.elapsed() > *HARD_TIMEOUT {
                    panic!("FATAL GORDER ERROR | SystemClusterClassification::get_from_shared | Deadlock. Could not get shared clusters for leaflet assignment. Spent more than `{}` seconds inside the spin-lock. {}",
                    HARD_TIMEOUT_SECONDS, PANIC_MESSAGE)
                }
            }

            // shared data unlock here
        }
    }

    /// Perform one sloppy clustering without checking the validity of the clusters.
    fn sloppy_clustering(
        &self,
        laplacian: &DMatrix<f32>,
        n: usize,
        frame_index: usize,
    ) -> Option<Clusters> {
        let embedding =
            Self::calc_and_embed_eigenvectors_lanczos(laplacian, n, LANCZOS_ITERATIONS.min(n));
        let assignments = Self::k_means(&embedding, N_CLUSTERS);
        let (c1, c2, min_index, min_index_cluster) = self.process_assignments(assignments);
        self.classify_clusters(c1, c2, min_index, min_index_cluster, frame_index)
    }

    /// Perform one precise clustering without checking the validity of the clusters.
    fn precise_clustering(
        &self,
        laplacian: DMatrix<f32>,
        n: usize,
        frame_index: usize,
    ) -> Option<Clusters> {
        let embedding = Self::calc_and_embed_eigenvectors_full(laplacian, n);
        let assignments = Self::k_means(&embedding, N_CLUSTERS);
        let (c1, c2, min_index, min_index_cluster) = self.process_assignments(assignments);
        self.classify_clusters(c1, c2, min_index, min_index_cluster, frame_index)
    }

    /// Create a dense or sparse similarity matrix depending on cut-off.
    fn create_similarity_matrix<'a>(
        &self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
        cutoff: f32,
        sigma: f32,
    ) -> Result<DMatrix<f32>, AnalysisError> {
        let n_atoms = system
            .group_get_n_atoms(group_name!("ClusterHeads"))
            .expect(PANIC_MESSAGE);
        let mut matrix = DMatrix::zeros(n_atoms, n_atoms);

        for (i, atom1) in system
            .group_iter(group_name!("ClusterHeads"))
            .expect(PANIC_MESSAGE)
            .enumerate()
        {
            let position1 = atom1
                .get_position()
                .ok_or(AnalysisError::UndefinedPosition(atom1.get_index()))?;

            if cutoff.is_infinite() {
                for (j, atom2) in system
                    .group_iter(group_name!("ClusterHeads"))
                    .expect(PANIC_MESSAGE)
                    .enumerate()
                {
                    let dist = pbc
                        .atoms_distance(
                            system,
                            atom1.get_index(),
                            atom2.get_index(),
                            Dimension::XYZ,
                        )
                        .map_err(AnalysisError::AtomError)?;

                    matrix[(i, j)] = (-sigma * dist * dist).exp();
                }
            } else {
                for (atom2, dist) in pbc.nearby_atoms(system, position1.clone(), cutoff) {
                    let j = *self.converter
                    .get(&atom2.get_index()).unwrap_or_else(||
                        panic!("FATAL GORDER ERROR | SystemClusterClassification::create_similarity_matrix | Atom index `{}` not in converter map. {}", 
                        atom2.get_index(), PANIC_MESSAGE));

                    matrix[(i, j)] = (-sigma * dist * dist).exp();
                }
            }
        }

        Ok(matrix)
    }

    /// Calculate eigenvectors using the Lanczos method and create an embedding for them.
    fn calc_and_embed_eigenvectors_lanczos(
        laplacian: &DMatrix<f32>,
        items: usize,
        iterations: usize,
    ) -> DMatrix<f32> {
        let eigen = laplacian.eigsh(iterations.min(items), Order::Smallest);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // create embedding from the collected eigenvectors (skip the first eigenvector since it is zero)
        let mut embedding = DMatrix::zeros(items, N_CLUSTERS);
        for (k, _) in eigenvalues.iter().enumerate().skip(1).take(N_CLUSTERS) {
            for row in 0..items {
                embedding[(row, k - 1)] = eigenvectors[(row, k)];
            }
        }

        // normalize each row to unit length
        for mut row in embedding.row_iter_mut() {
            let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }

        embedding
    }

    /// Calculate eigenvectors using full eigendecomposition and create an embedding for them.
    fn calc_and_embed_eigenvectors_full(laplacian: DMatrix<f32>, items: usize) -> DMatrix<f32> {
        let eigen = SymmetricEigen::new(laplacian);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // select N_CLUSTERS smallest eigenvectors (skipping the first eigenvector which is zero)
        let mut indices: Vec<usize> = (0..items).collect();
        indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());
        let selected_indices = &indices[1..(N_CLUSTERS + 1)];

        // create embedding
        let mut embedding = DMatrix::zeros(items, N_CLUSTERS);
        for (k, &i) in selected_indices.iter().enumerate() {
            for row in 0..items {
                embedding[(row, k)] = eigenvectors[(row, i)];
            }
        }

        // normalize each row to unit length
        for mut row in embedding.row_iter_mut() {
            let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }

        embedding
    }

    /// Process assignments into two clusters.
    fn process_assignments(
        &self,
        assignments: Vec<usize>,
    ) -> (HashSet<usize>, HashSet<usize>, usize, i8) {
        let mut cluster1 = HashSet::new();
        let mut cluster2 = HashSet::new();
        let mut min_index = usize::MAX;
        let mut min_index_cluster = 0;

        for (&abs_index, &matrix_index) in self.converter.iter() {
            let cluster = assignments.get(matrix_index).expect(PANIC_MESSAGE);

            match cluster {
                0 => { cluster1.insert(abs_index); },
                1 => { cluster2.insert(abs_index); },
                x => panic!(
                    "FATAL GORDER ERROR | SystemClusterClassification::process_assignments | Invalid cluster index `{}`. {}",
                    x,
                    PANIC_MESSAGE
                ),
            }

            if abs_index < min_index {
                min_index = abs_index;
                min_index_cluster = *cluster as i8;
            }
        }

        (cluster1, cluster2, min_index, min_index_cluster)
    }

    /// Determine which cluster is `upper` and `which` is lower.
    ///
    /// In the first frame, the clusters are classified as follows:
    /// - the more populated leaflet is the `upper` leaflet,
    /// - if both leaflets are equally populated, the `upper` leaflet is the one containing
    ///   a reference atom with the lowest index (typically the first analyzed lipid).
    ///
    /// In the other frames, the clusters are classified by trying to match them with the
    /// clusters from the previous frame analyzed by this thread.
    ///
    /// In membranes with lipid flip-flop, the match is heuristic and may in extremely
    /// rare cases be incorrect:
    /// - Matching will succeed, if less than 20% of lipids have changed leaflet between two analyzed frames.
    /// - Matching will fail (returning None), if 20-80% of lipids have changed leaflet between two analyzed frames.
    /// - Matching will fail silently and the results will be incorrect if over 80% of lipids have
    ///   changed leaflet between two analyzed frames! This should be basically unphysical, so it's not
    ///   a big concern.
    fn classify_clusters(
        &self,
        cluster1: HashSet<usize>,
        cluster2: HashSet<usize>,
        min_index: usize,
        min_index_cluster: i8,
        frame_index: usize,
    ) -> Option<Clusters> {
        // only for the first frame
        if frame_index == 0 {
            return Some(Clusters::classify_ab_initio(
                cluster1,
                cluster2,
                min_index,
                min_index_cluster,
            ));
        }

        let previous_clusters = match &self.clusters {
            Some(x) => x,
            // load data from shared storage
            None => &self.get_from_shared(),
        };

        Clusters::classify_by_match(previous_clusters, cluster1, cluster2, min_index)
    }

    /// Create normalized Laplacian for the similarity matrix.
    fn create_normalized_laplacian(similarity_matrix: &DMatrix<f32>) -> DMatrix<f32> {
        let n = similarity_matrix.nrows();

        // compute degree matrix (diagonal)
        let degrees: Vec<f32> = (0..n).map(|i| similarity_matrix.row(i).sum()).collect();

        // create D^(-1/2)
        let d_neg_sqrt: Vec<f32> = degrees
            .iter()
            .map(|&x| if x > 1e-10 { 1.0 / x.sqrt() } else { 0.0 })
            .collect();

        // create normalized Laplacian: I - D^(-1/2) * W * D^(-1/2)
        let mut laplacian = DMatrix::identity(n, n);

        for i in 0..n {
            for j in 0..n {
                let w_ij = similarity_matrix[(i, j)];
                if w_ij != 0.0 {
                    laplacian[(i, j)] -= d_neg_sqrt[i] * w_ij * d_neg_sqrt[j];
                }
            }
        }

        laplacian
    }

    /// Perform k-means clustering.
    fn k_means(data: &nalgebra::DMatrix<f32>, k: usize) -> Vec<usize> {
        assert!(
        k > 0,
        "FATAL GORDER ERROR | SystemClusterClassification::k_means | Number of clusters must be greater than 0. {}",
        PANIC_MESSAGE
    );
        assert!(
            data.nrows() >= k,
            "FATAL GORDER ERROR | SystemClusterClassification::k_means | More clusters than data points. {}",
            PANIC_MESSAGE
        );

        let (n_samples, n_features) = (data.nrows(), data.ncols());
        let mut centroids = nalgebra::DMatrix::zeros(k, n_features);
        let mut labels = vec![0; n_samples];
        let mut prev_labels = vec![usize::MAX; n_samples];

        // initialize centroids with first k samples
        for i in 0..k {
            for j in 0..n_features {
                centroids[(i, j)] = data[(i, j)];
            }
        }

        // main optimization loop
        for _ in 0..100 {
            // assign labels
            for (sample_idx, label) in labels.iter_mut().enumerate().take(n_samples) {
                let mut best_cluster = 0;
                let mut min_distance = f32::INFINITY;

                for cluster_idx in 0..k {
                    let distance =
                        Self::euclidean_distance(data, sample_idx, &centroids, cluster_idx);

                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = cluster_idx;
                    }
                }

                *label = best_cluster;
            }

            // check convergence
            if labels == prev_labels {
                break;
            }

            // update centroids
            let mut counts = vec![0; k];
            centroids.fill(0.0);

            for sample_idx in 0..n_samples {
                let cluster = labels[sample_idx];

                for j in 0..n_features {
                    centroids[(cluster, j)] += data[(sample_idx, j)];
                }

                counts[cluster] += 1;
            }

            // normalize and handle empty clusters
            for cluster in 0..k {
                let count = counts[cluster];
                if count > 0 {
                    for j in 0..n_features {
                        centroids[(cluster, j)] /= count as f32;
                    }
                } else {
                    // handle empty cluster by using the first data point
                    for j in 0..n_features {
                        centroids[(cluster, j)] = data[(0, j)];
                    }
                }
            }

            prev_labels.clone_from(&labels);
        }

        labels
    }

    /// Calculate Euclidean distance between rows in matrices.
    fn euclidean_distance(
        matrix1: &nalgebra::DMatrix<f32>,
        row1: usize,
        matrix2: &nalgebra::DMatrix<f32>,
        row2: usize,
    ) -> f32 {
        let n_features = matrix1.ncols();
        let mut sum = 0.0;

        for j in 0..n_features {
            let diff = matrix1[(row1, j)] - matrix2[(row2, j)];
            sum += diff * diff;
        }

        sum.sqrt()
    }
}

/// Leaflets assigned using clustering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Clusters {
    /// Upper leaflet.
    pub(super) upper: HashSet<usize>,
    /// Lower leaflet.
    pub(super) lower: HashSet<usize>,
    /// Smallest index of the headgroup-representing atoms.
    pub(super) min_index: usize,
}

impl Clusters {
    /// Classify clusters as leaflets using the properties of the clusters.
    /// This should only be performed for the first frame.
    fn classify_ab_initio(
        cluster1: HashSet<usize>,
        cluster2: HashSet<usize>,
        min_index: usize,
        min_index_cluster: i8,
    ) -> Self {
        match cluster1.len().cmp(&cluster2.len()) {
            // more populated cluster is `upper`
            Ordering::Less => Clusters {
                upper: cluster2,
                lower: cluster1,
                min_index,
            },
            Ordering::Greater => Clusters {
                upper: cluster1,
                lower: cluster2,
                min_index,
            },
            Ordering::Equal => {
                // if both clusters are equally populated, cluster containing `min_index` is `upper`
                if min_index_cluster == 0 {
                    Clusters {
                        upper: cluster1,
                        lower: cluster2,
                        min_index,
                    }
                } else {
                    Clusters {
                        upper: cluster2,
                        lower: cluster1,
                        min_index,
                    }
                }
            }
        }
    }

    /// Classsify clusters as leaflets by trying to match them to reference clusters.
    /// Returns `None`, if the match fails.
    fn classify_by_match(
        reference: &Clusters,
        cluster1: HashSet<usize>,
        cluster2: HashSet<usize>,
        min_index: usize,
    ) -> Option<Self> {
        let overlap_cluster1_upper =
            cluster1.intersection(&reference.upper).count() as f32 / cluster1.len() as f32;
        let overlap_cluster1_lower =
            cluster1.intersection(&reference.lower).count() as f32 / cluster1.len() as f32;

        if overlap_cluster1_upper < CLUSTER_CLASSIFICATION_LIMIT
            && overlap_cluster1_lower < CLUSTER_CLASSIFICATION_LIMIT
        {
            return None;
        }

        if overlap_cluster1_upper < overlap_cluster1_lower {
            Some(Clusters {
                upper: cluster2,
                lower: cluster1,
                min_index,
            })
        } else {
            Some(Clusters {
                upper: cluster1,
                lower: cluster2,
                min_index,
            })
        }
    }

    /// Log information about which cluster was assigned as the `upper` leaflet and which as the `lower` leaflet.
    fn log_info(&self, flip: bool) {
        let main_leaflet = if flip { "lower" } else { "upper" };

        if self.upper.len() > self.lower.len() {
            colog_info!("Clustering leaflet classification: classifying the more populated leaflet as '{}'.", main_leaflet);
        } else {
            colog_info!("Clustering leaflet classification: classifying the leaflet containing lipid with reference atom index '{}' as '{}'.", self.min_index, main_leaflet);
        }
    }
}

#[cfg(test)]
mod tests_clusters {
    use super::*;

    #[test]
    fn test_ab_initio_unequal_populations() {
        let cluster1 = HashSet::from([13, 18, 24, 27, 29, 33, 156, 17, 14]);
        let cluster2 = HashSet::from([1, 4, 8, 146, 158, 123, 1453, 13]);

        let min_index = 1;
        let min_index_cluster = 1;

        let clusters = Clusters::classify_ab_initio(
            cluster1.clone(),
            cluster2.clone(),
            min_index,
            min_index_cluster,
        );

        assert_eq!(clusters.upper, cluster1);
        assert_eq!(clusters.lower, cluster2);
        assert_eq!(clusters.min_index, min_index);
    }

    #[test]
    fn test_ab_initio_equal_populations() {
        let cluster1 = HashSet::from([13, 18, 24, 27, 29, 33, 156, 17, 14]);
        let cluster2 = HashSet::from([1, 4, 8, 146, 158, 123, 1453, 13, 19]);

        let min_index = 1;
        let min_index_cluster = 1;

        let clusters = Clusters::classify_ab_initio(
            cluster1.clone(),
            cluster2.clone(),
            min_index,
            min_index_cluster,
        );

        assert_eq!(clusters.upper, cluster2);
        assert_eq!(clusters.lower, cluster1);
        assert_eq!(clusters.min_index, min_index);
    }

    #[test]
    fn test_matching_perfect() {
        let cluster1 = HashSet::from([13, 18, 24, 27, 29, 33, 156, 17, 14]);
        let cluster2 = HashSet::from([1, 4, 8, 146, 158, 123, 1453, 13, 19]);

        let min_index = 1;
        let min_index_cluster = 1;

        let clusters = Clusters::classify_ab_initio(
            cluster1.clone(),
            cluster2.clone(),
            min_index,
            min_index_cluster,
        );

        let clusters2 =
            Clusters::classify_by_match(&clusters, cluster1, cluster2, min_index).unwrap();

        assert_eq!(clusters, clusters2);
    }

    #[test]
    fn test_matching_small_mismatch() {
        let cluster1 = HashSet::from([13, 18, 24, 27, 29, 33, 156, 17, 14]);
        let cluster2 = HashSet::from([1, 4, 8, 146, 158, 123, 1453, 13, 19]);

        let min_index = 1;
        let min_index_cluster = 1;

        let clusters =
            Clusters::classify_ab_initio(cluster1, cluster2, min_index, min_index_cluster);

        let cluster1 = HashSet::from([13, 18, 24, 27, 29, 33, 156, 17, 14, 1]);
        let cluster2 = HashSet::from([4, 8, 146, 158, 123, 1453, 13, 19]);

        let clusters2 =
            Clusters::classify_by_match(&clusters, cluster1.clone(), cluster2.clone(), min_index)
                .unwrap();

        assert_eq!(clusters2.upper, cluster2);
        assert_eq!(clusters2.lower, cluster1);

        let clusters3 =
            Clusters::classify_by_match(&clusters, cluster2.clone(), cluster1.clone(), min_index)
                .unwrap();

        assert_eq!(clusters3, clusters2);
    }

    #[test]
    fn test_matching_large_mismatch() {
        let cluster1 = HashSet::from([13, 18, 24, 27, 29, 33, 156, 17, 14]);
        let cluster2 = HashSet::from([1, 4, 8, 146, 158, 123, 1453, 13, 19]);

        let min_index = 1;
        let min_index_cluster = 1;

        let clusters =
            Clusters::classify_ab_initio(cluster1, cluster2, min_index, min_index_cluster);

        let cluster1 = HashSet::from([13, 18, 24, 27, 17, 14, 1, 13, 19]);
        let cluster2 = HashSet::from([4, 8, 146, 158, 123, 29, 33, 156, 1453]);

        assert!(Clusters::classify_by_match(&clusters, cluster1, cluster2, min_index).is_none());
    }
}

#[cfg(test)]
mod tests_classify {
    use crate::analysis::pbc::{NoPBC, PBC3D};

    use super::*;

    fn test_sloppy(tpr: &str, heads: &str, handle_pbc: bool, expected: Vec<usize>) {
        let mut system = System::from_file(tpr).unwrap();
        system
            .group_create(group_name!("ClusterHeads"), heads)
            .unwrap();

        let clustering = SystemClusterClassification::new(&system, false);
        let matrix = if handle_pbc {
            clustering
                .create_similarity_matrix(
                    &system,
                    &PBC3D::new(system.get_box().unwrap()),
                    DISTANCE_CUTOFF,
                    SLOPPY_SIGMA,
                )
                .unwrap()
        } else {
            clustering
                .create_similarity_matrix(&system, &NoPBC, DISTANCE_CUTOFF, SLOPPY_SIGMA)
                .unwrap()
        };

        let laplacian = SystemClusterClassification::create_normalized_laplacian(&matrix);
        let n = matrix.shape().0;

        // try 3 times
        let mut i = 0;
        loop {
            if i == 3 {
                panic!("Could not reach match after 3 tries.");
            }

            let embedding = SystemClusterClassification::calc_and_embed_eigenvectors_lanczos(
                &laplacian,
                n,
                LANCZOS_ITERATIONS,
            );
            let assignments = SystemClusterClassification::k_means(&embedding, N_CLUSTERS);
            let assignments_reversed: Vec<usize> = assignments
                .iter()
                .map(|x| match x {
                    0 => 1,
                    1 => 0,
                    _ => panic!("Invalid cluster number."),
                })
                .collect();

            if assignments == expected || assignments_reversed == expected {
                break;
            }

            i += 1;
        }
    }

    fn test_precise(tpr: &str, heads: &str, handle_pbc: bool, expected: Vec<usize>) {
        let mut system = System::from_file(tpr).unwrap();
        system
            .group_create(group_name!("ClusterHeads"), heads)
            .unwrap();

        let clustering = SystemClusterClassification::new(&system, false);
        let matrix = if handle_pbc {
            clustering
                .create_similarity_matrix(
                    &system,
                    &PBC3D::new(system.get_box().unwrap()),
                    f32::INFINITY,
                    PRECISE_SIGMA,
                )
                .unwrap()
        } else {
            clustering
                .create_similarity_matrix(&system, &NoPBC, f32::INFINITY, PRECISE_SIGMA)
                .unwrap()
        };

        let laplacian = SystemClusterClassification::create_normalized_laplacian(&matrix);
        let n = matrix.shape().0;

        let embedding = SystemClusterClassification::calc_and_embed_eigenvectors_full(laplacian, n);
        let assignments = SystemClusterClassification::k_means(&embedding, N_CLUSTERS);
        let assignments_reversed: Vec<usize> = assignments
            .iter()
            .map(|x| match x {
                0 => 1,
                1 => 0,
                _ => panic!("Invalid cluster number."),
            })
            .collect();

        assert!(assignments == expected || assignments_reversed == expected);
    }

    fn test_sloppy_trajectory(
        tpr: &str,
        xtc: &str,
        heads: &str,
        handle_pbc: bool,
        expected: Vec<usize>,
        step: usize,
    ) {
        let mut system = System::from_file(tpr).unwrap();
        system
            .group_create(group_name!("ClusterHeads"), heads)
            .unwrap();

        let clustering = SystemClusterClassification::new(&system, false);

        for frame in system.xtc_iter(xtc).unwrap().with_step(step).unwrap() {
            let frame = frame.unwrap();
            let matrix = if handle_pbc {
                clustering
                    .create_similarity_matrix(
                        frame,
                        &PBC3D::new(frame.get_box().unwrap()),
                        DISTANCE_CUTOFF,
                        SLOPPY_SIGMA,
                    )
                    .unwrap()
            } else {
                clustering
                    .create_similarity_matrix(frame, &NoPBC, DISTANCE_CUTOFF, SLOPPY_SIGMA)
                    .unwrap()
            };

            let laplacian = SystemClusterClassification::create_normalized_laplacian(&matrix);
            let n = matrix.shape().0;

            let mut i = 0;
            loop {
                if i == 3 {
                    panic!("Could not reach match after 3 tries.");
                }

                let embedding = SystemClusterClassification::calc_and_embed_eigenvectors_lanczos(
                    &laplacian,
                    n,
                    LANCZOS_ITERATIONS,
                );
                let assignments = SystemClusterClassification::k_means(&embedding, N_CLUSTERS);
                let assignments_reversed: Vec<usize> = assignments
                    .iter()
                    .map(|x| match x {
                        0 => 1,
                        1 => 0,
                        _ => panic!("Invalid cluster number."),
                    })
                    .collect();

                if assignments == expected || assignments_reversed == expected {
                    break;
                }

                i += 1;
            }
        }
    }

    fn test_precise_trajectory(
        tpr: &str,
        xtc: &str,
        heads: &str,
        handle_pbc: bool,
        expected: Vec<usize>,
        step: usize,
    ) {
        let mut system = System::from_file(tpr).unwrap();
        system
            .group_create(group_name!("ClusterHeads"), heads)
            .unwrap();

        let clustering = SystemClusterClassification::new(&system, false);

        for frame in system.xtc_iter(xtc).unwrap().with_step(step).unwrap() {
            let frame = frame.unwrap();

            let matrix = if handle_pbc {
                clustering
                    .create_similarity_matrix(
                        frame,
                        &PBC3D::new(frame.get_box().unwrap()),
                        f32::INFINITY,
                        PRECISE_SIGMA,
                    )
                    .unwrap()
            } else {
                clustering
                    .create_similarity_matrix(frame, &NoPBC, f32::INFINITY, PRECISE_SIGMA)
                    .unwrap()
            };

            let laplacian = SystemClusterClassification::create_normalized_laplacian(&matrix);
            let n = matrix.shape().0;

            let embedding =
                SystemClusterClassification::calc_and_embed_eigenvectors_full(laplacian, n);
            let assignments = SystemClusterClassification::k_means(&embedding, N_CLUSTERS);
            let assignments_reversed: Vec<usize> = assignments
                .iter()
                .map(|x| match x {
                    0 => 1,
                    1 => 0,
                    _ => panic!("Invalid cluster number."),
                })
                .collect();

            assert!(assignments == expected || assignments_reversed == expected);
        }
    }

    #[allow(dead_code)]
    fn output_clusters_as_gro(
        system: &mut System,
        assignments: Vec<usize>,
        converter: &HashMap<usize, usize>,
    ) {
        let mut cluster1 = Vec::new();
        let mut cluster2 = Vec::new();

        for (&real_index, &matrix_index) in converter.iter() {
            match assignments[matrix_index] {
                0 => cluster1.push(real_index),
                1 => cluster2.push(real_index),
                _ => panic!("Invalid cluster number."),
            }
        }

        system
            .group_create_from_indices("Cluster1", cluster1)
            .unwrap();
        system
            .group_create_from_indices("Cluster2", cluster2)
            .unwrap();

        system
            .group_write_gro("Cluster1", "cluster1.gro", false)
            .unwrap();
        system
            .group_write_gro("Cluster2", "cluster2.gro", false)
            .unwrap();
    }

    fn expected_assignments_flat_cg() -> Vec<usize> {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
    }

    fn expected_assignments_flat_aa() -> Vec<usize> {
        vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        ]
    }

    fn expected_assignments_vesicle() -> Vec<usize> {
        vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ]
    }

    fn expected_assignments_scrambling_cg() -> Vec<usize> {
        vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
    }

    fn expected_assignments_buckled_aa() -> Vec<usize> {
        vec![
            0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
            0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0,
            1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
            1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1,
            0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1,
            0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
            1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1,
            0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,
            1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1,
            1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1,
            0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0,
            1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
            1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
            1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1,
            0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1,
            1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
            0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
    }

    fn expected_assignments_buckled_cg() -> Vec<usize> {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
    }

    #[test]
    fn test_flat_cg_sloppy_pbc() {
        test_sloppy(
            "tests/files/cg.tpr",
            "name PO4",
            true,
            expected_assignments_flat_cg(),
        );
    }

    #[test]
    fn test_flat_cg_sloppy_nopbc() {
        test_sloppy(
            "tests/files/cg.tpr",
            "name PO4",
            false,
            expected_assignments_flat_cg(),
        );
    }

    #[test]
    fn test_flat_cg_precise_pbc() {
        test_precise(
            "tests/files/cg.tpr",
            "name PO4",
            true,
            expected_assignments_flat_cg(),
        )
    }

    #[test]
    fn test_flat_cg_precise_nopbc() {
        test_precise(
            "tests/files/cg.tpr",
            "name PO4",
            false,
            expected_assignments_flat_cg(),
        )
    }

    #[test]
    fn test_flat_cg_sloppy_trajectory_pbc() {
        test_sloppy_trajectory(
            "tests/files/cg.tpr",
            "tests/files/cg.xtc",
            "name PO4",
            true,
            expected_assignments_flat_cg(),
            5,
        );
    }

    #[test]
    fn test_flat_cg_precise_trajectory_pbc() {
        test_precise_trajectory(
            "tests/files/cg.tpr",
            "tests/files/cg.xtc",
            "name PO4",
            true,
            expected_assignments_flat_cg(),
            5,
        );
    }

    #[test]
    fn test_flat_aa_sloppy_pbc() {
        test_sloppy(
            "tests/files/pcpepg.tpr",
            "name P",
            true,
            expected_assignments_flat_aa(),
        );
    }

    #[test]
    fn test_flat_aa_sloppy_nopbc() {
        test_sloppy(
            "tests/files/pcpepg.tpr",
            "name P",
            false,
            expected_assignments_flat_aa(),
        );
    }

    #[test]
    fn test_flat_aa_precise_pbc() {
        test_precise(
            "tests/files/pcpepg.tpr",
            "name P",
            true,
            expected_assignments_flat_aa(),
        )
    }

    #[test]
    fn test_flat_aa_precise_nopbc() {
        test_precise(
            "tests/files/pcpepg.tpr",
            "name P",
            false,
            expected_assignments_flat_aa(),
        )
    }

    #[test]
    fn test_vesicle_sloppy_pbc() {
        test_sloppy(
            "tests/files/vesicle.tpr",
            "name PO4",
            true,
            expected_assignments_vesicle(),
        );
    }

    #[test]
    fn test_vesicle_sloppy_nopbc() {
        test_sloppy(
            "tests/files/vesicle.tpr",
            "name PO4",
            false,
            expected_assignments_vesicle(),
        );
    }

    #[test]
    fn test_vesicle_precise_pbc() {
        test_precise(
            "tests/files/vesicle.tpr",
            "name PO4",
            true,
            expected_assignments_vesicle(),
        );
    }

    #[test]
    fn test_vesicle_precise_nopbc() {
        test_precise(
            "tests/files/vesicle.tpr",
            "name PO4",
            false,
            expected_assignments_vesicle(),
        );
    }

    #[test]
    fn test_vesicle_sloppy_trajectory_pbc() {
        test_sloppy_trajectory(
            "tests/files/vesicle.tpr",
            "tests/files/vesicle.xtc",
            "name PO4",
            true,
            expected_assignments_vesicle(),
            5,
        );
    }

    #[test]
    fn test_vesicle_sloppy_trajectory_nopbc() {
        test_sloppy_trajectory(
            "tests/files/vesicle.tpr",
            "tests/files/vesicle_centered.xtc",
            "name PO4",
            false,
            expected_assignments_vesicle(),
            5,
        );
    }

    #[test]
    fn test_scrambling_sloppy_pbc() {
        test_sloppy(
            "tests/files/scrambling/cg_scrambling.tpr",
            "name PO4",
            true,
            expected_assignments_scrambling_cg(),
        );
    }

    #[test]
    fn test_scrambling_sloppy_nopbc() {
        test_sloppy(
            "tests/files/scrambling/cg_scrambling.tpr",
            "name PO4",
            false,
            expected_assignments_scrambling_cg(),
        );
    }

    #[test]
    fn test_scrambling_precise_pbc() {
        test_precise(
            "tests/files/scrambling/cg_scrambling.tpr",
            "name PO4",
            true,
            expected_assignments_scrambling_cg(),
        );
    }

    #[test]
    fn test_scrambling_precise_nopbc() {
        test_precise(
            "tests/files/scrambling/cg_scrambling.tpr",
            "name PO4",
            false,
            expected_assignments_scrambling_cg(),
        );
    }

    #[test]
    fn test_buckled_aa_sloppy_pbc() {
        test_sloppy(
            "tests/files/aa_buckled.tpr",
            "name P",
            true,
            expected_assignments_buckled_aa(),
        );
    }

    #[test]
    fn test_buckled_aa_sloppy_nopbc() {
        test_sloppy(
            "tests/files/aa_buckled.tpr",
            "name P",
            false,
            expected_assignments_buckled_aa(),
        );
    }

    #[test]
    fn test_buckled_aa_precise_pbc() {
        test_precise(
            "tests/files/aa_buckled.tpr",
            "name P",
            true,
            expected_assignments_buckled_aa(),
        )
    }

    #[test]
    fn test_buckled_aa_precise_nopbc() {
        test_precise(
            "tests/files/aa_buckled.tpr",
            "name P",
            false,
            expected_assignments_buckled_aa(),
        )
    }

    #[test]
    fn test_buckled_cg_sloppy_pbc() {
        test_sloppy(
            "tests/files/cg_buckled.tpr",
            "name PO4",
            true,
            expected_assignments_buckled_cg(),
        )
    }

    #[test]
    fn test_buckled_cg_sloppy_nopbc() {
        test_sloppy(
            "tests/files/cg_buckled.tpr",
            "name PO4",
            false,
            expected_assignments_buckled_cg(),
        )
    }

    #[test]
    fn test_buckled_cg_precise_pbc() {
        test_precise(
            "tests/files/cg_buckled.tpr",
            "name PO4",
            true,
            expected_assignments_buckled_cg(),
        )
    }

    #[test]
    fn test_buckled_cg_precise_nopbc() {
        test_precise(
            "tests/files/cg_buckled.tpr",
            "name PO4",
            false,
            expected_assignments_buckled_cg(),
        )
    }

    #[test]
    fn test_buckled_cg_precise_trajectory_pbc() {
        test_precise_trajectory(
            "tests/files/cg_buckled.tpr",
            "tests/files/cg_buckled.xtc",
            "name PO4",
            true,
            expected_assignments_buckled_cg(),
            5,
        );
    }

    #[test]
    fn test_buckled_cg_precise_trajectory_nopbc() {
        test_precise_trajectory(
            "tests/files/cg_buckled.tpr",
            "tests/files/cg_buckled.xtc",
            "name PO4",
            false,
            expected_assignments_buckled_cg(),
            5,
        );
    }
}
