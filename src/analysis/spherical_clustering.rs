// Released under MIT License.
// Copyright (c) 2024-2025 Ladislav Bartos

//! Implementation of Gaussian Mixture Model for leaflet classification.

use std::f32::consts::PI;

use getset::Getters;
use groan_rs::{
    prelude::{AtomIterator, Dimension},
    system::System,
};
use hashbrown::HashSet;
use statistical::{mean, variance};

use crate::{
    analysis::{clustering::Clusters, common::macros::group_name, pbc::PBCHandler},
    errors::AnalysisError,
    PANIC_MESSAGE,
};

/// Maximal number of iterations when performing the GMM fit.
const GMM_MAX_ITERATIONS: usize = 50;

/// Convergence tolerance when performing the GMM fit.
const GMM_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Clone, Getters)]
pub(crate) struct SystemSphericalClusterClassification {
    /// Clusters assigned for the current frame.
    #[getset(get = "pub(super)")]
    clusters: Option<Clusters>,
}

impl SystemSphericalClusterClassification {
    /// Create a new structure for spherical clustering.
    pub(super) fn new() -> Self {
        SystemSphericalClusterClassification { clusters: None }
    }

    /// Assign lipid headgroups into individual leaflets using 2-Component Gaussian Mixture.
    pub(super) fn cluster<'a>(
        &mut self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<(), AnalysisError> {
        let vesicle_center = pbc.group_get_center(system, group_name!("ClusterHeads"))
            .unwrap_or_else(|_| panic!("FATAL GORDER ERROR | SystemSphericalClusterClassification::cluster | Could not get center of geometry of `ClusterHeads`. {}", PANIC_MESSAGE));

        let distances = system
            .group_iter(group_name!("ClusterHeads"))
            .expect(PANIC_MESSAGE)
            .map(|atom| {
                pbc.distance(
                    atom.get_position().expect(PANIC_MESSAGE),
                    &vesicle_center,
                    Dimension::XYZ,
                )
            })
            .collect::<Vec<f32>>();

        let (_, responsibilities, _) =
            fit_gmm_1d_two_components(&distances, GMM_MAX_ITERATIONS, GMM_TOLERANCE);

        let clusters = Clusters::from_responsibilities(
            &responsibilities,
            &distances,
            system
                .group_iter(group_name!("ClusterHeads"))
                .expect(PANIC_MESSAGE),
        );

        self.clusters = Some(clusters);

        Ok(())
    }
}

/// Parameters of a 1D, two-component Gaussian Mixture Model.
#[derive(Debug, Clone, Copy)]
struct GmmParams {
    /// Mixture weight for component A.
    /// The weight for component B is `1.0 - weight_a`.
    weight_a: f32,

    /// Mean of component A.
    mean_a: f32,

    /// Variance of component A.
    var_a: f32,

    /// Mean of component B.
    mean_b: f32,

    /// Variance of component B.
    var_b: f32,
}

/// Natural logarithm of the 1D Gaussian probability density at `x`.
#[inline(always)]
fn log_gaussian(x: f32, mean: f32, variance: f32) -> f32 {
    let diff = x - mean;
    -0.5 * ((2.0 * PI).ln() + variance.ln() + diff * diff / variance)
}

/// Numerically stable log(exp(a) + exp(b)).
#[inline(always)]
fn log_sum_exp(a: f32, b: f32) -> f32 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Initializes GMM parameters from data.
///
/// The means are initialized from the 25th and 75th percentiles of the data,
/// variances are initialized from the global variance, and mixture weights
/// are set to 0.5.
fn initialize_params(data: &[f32]) -> GmmParams {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let mean_a = sorted[n / 4];
    let mean_b = sorted[(3 * n) / 4];

    let global_mean = mean(data);
    let mut global_var = variance(data, Some(global_mean));
    if !global_var.is_finite() || global_var <= 0.0 {
        global_var = 1.0;
    }

    GmmParams {
        weight_a: 0.5,
        mean_a,
        var_a: global_var,
        mean_b,
        var_b: global_var,
    }
}

/// Fits a two-component 1D Gaussian Mixture Model using the EM algorithm.
///
/// # Arguments
/// - `data` - Input observations
/// - `max_iters` - Maximum number of EM iterations
/// - `tolerance` - Convergence threshold on average log-likelihood
///
/// # Returns
/// - Fitted GMM parameters
/// - Responsibilities for component A
/// - Final average log-likelihood
fn fit_gmm_1d_two_components(
    data: &[f32],
    max_iters: usize,
    tolerance: f32,
) -> (GmmParams, Vec<f32>, f32) {
    let n = data.len();
    let n_f = n as f32;

    let var_floor = 1e-6;
    let weight_floor = 1e-4;

    // create an initial guess for the mixture parameters
    let mut params = initialize_params(data);
    params.var_a = params.var_a.max(var_floor);
    params.var_b = params.var_b.max(var_floor);

    // stores responsibilities (probabilities that datapoints belong to cluster A)
    let mut resp_a = vec![0.5; n];
    // log-likelihood from previous iteration
    let mut prev_avg_ll = f32::NEG_INFINITY;

    // repeat the EM updated until convergence or iteration limits
    for _ in 0..max_iters {
        // accumulates total log-likelihood
        let mut loglik_sum = 0.0;

        // precompute logarithms
        let log_weight_a = params.weight_a.ln();
        let log_weight_b = (1.0 - params.weight_a).ln();

        // for each datapoint, evaluate the likelihoods of the datapoint
        // belonging to cluster A and cluster B
        for (i, &x) in data.iter().enumerate() {
            let log_joint_a = log_weight_a + log_gaussian(x, params.mean_a, params.var_a);
            let log_joint_b = log_weight_b + log_gaussian(x, params.mean_b, params.var_b);

            let log_px = log_sum_exp(log_joint_a, log_joint_b);
            loglik_sum += log_px;

            resp_a[i] = (log_joint_a - log_px).exp();
        }

        // average log-likelihood per data point
        let avg_ll = loglik_sum / n_f;

        // stop iteration if the improvement in likelihood
        // is smaller than the requested tolerance
        if (avg_ll - prev_avg_ll).abs() < tolerance {
            prev_avg_ll = avg_ll;
            break;
        }
        prev_avg_ll = avg_ll;

        // effective number of points assigned to cluster A
        let sum_resp_a: f32 = resp_a.iter().sum();
        // effective number of points assigned to cluster B
        let sum_resp_b = n_f - sum_resp_a;

        // guard against numerical collapse
        let sum_resp_a = sum_resp_a.max(1e-6);
        let sum_resp_b = sum_resp_b.max(1e-6);

        // update mixture weight
        params.weight_a = (sum_resp_a / n_f).clamp(weight_floor, 1.0 - weight_floor);

        // update means
        let mut mean_a_num = 0.0;
        let mut mean_b_num = 0.0;
        for (&x, &r) in data.iter().zip(resp_a.iter()) {
            mean_a_num += r * x;
            mean_b_num += (1.0 - r) * x;
        }

        // update variances
        params.mean_a = mean_a_num / sum_resp_a;
        params.mean_b = mean_b_num / sum_resp_b;
        let mut var_a_num = 0.0;
        let mut var_b_num = 0.0;
        for (&x, &r) in data.iter().zip(resp_a.iter()) {
            let da = x - params.mean_a;
            let db = x - params.mean_b;
            var_a_num += r * da * da;
            var_b_num += (1.0 - r) * db * db;
        }

        params.var_a = (var_a_num / sum_resp_a).max(var_floor);
        params.var_b = (var_b_num / sum_resp_b).max(var_floor);
    }

    (params, resp_a, prev_avg_ll)
}

impl Clusters {
    /// Assing headgroups into clusters based on their responsibilities.
    fn from_responsibilities(resp_a: &[f32], distances: &[f32], atoms: AtomIterator) -> Self {
        let mut cluster1 = HashSet::with_capacity(resp_a.len());
        let mut cluster1_dist_sum = 0.0;
        let mut cluster2 = HashSet::with_capacity(resp_a.len());
        let mut cluster2_dist_sum = 0.0;
        for (&r, (&d, atom)) in resp_a.iter().zip(distances.iter().zip(atoms)) {
            if r < 0.5 {
                cluster1.insert(atom.get_index());
                cluster1_dist_sum += d;
            } else {
                cluster2.insert(atom.get_index());
                cluster2_dist_sum += d;
            }
        }

        let cluster1_av_dist = cluster1_dist_sum / cluster1.len() as f32;
        let cluster2_av_dist = cluster2_dist_sum / cluster2.len() as f32;

        if cluster1_av_dist > cluster2_av_dist {
            Self {
                upper: cluster1, // upper = outer
                lower: cluster2, // lower = inner
                min_index: 0,    // unused, so we can set anything
            }
        } else {
            Self {
                upper: cluster2,
                lower: cluster1,
                min_index: 0, // unused, so we can set anything
            }
        }
    }
}

#[cfg(test)]
mod tests_clusters {
    use groan_rs::prelude::{Atom, SimBox};

    use super::*;

    fn fake_system() -> System {
        let atoms = vec![
            Atom::new(1, "POPC", 1, "P"),
            Atom::new(2, "POPC", 12, "P"),
            Atom::new(3, "POPC", 23, "P"),
            Atom::new(4, "POPC", 34, "P"),
            Atom::new(5, "POPC", 45, "P"),
        ];

        let mut system = System::new("Fake system", atoms, Some(SimBox::from([20.0, 20.0, 20.0])));

        system.group_create("Atoms", "all").unwrap();

        system
    }

    #[test]
    fn test_clusters_from_responsibilities() {
        let system = fake_system();
        let atoms = system.group_iter("Atoms").unwrap();
        let responsibilities = vec![0.9998, 0.1, 0.42, 0.834, 0.932];
        let distances = vec![10.5, 1.3, 2.8, 7.8, 8.4];

        let clusters = Clusters::from_responsibilities(&responsibilities, &distances, atoms);

        assert_eq!(clusters.upper.len(), 3);
        for index in [0usize, 3, 4] {
            assert!(clusters.upper.contains(&index));
        }

        assert_eq!(clusters.lower.len(), 2);
        for index in [1, 2] {
            assert!(clusters.lower.contains(&index));
        }
    }
}

#[cfg(test)]
mod tests_clustering {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    use crate::analysis::pbc::PBC3D;

    use super::*;

    fn generate_two_cluster_points(
        n: usize,
        mean_a: f32,
        std_a: f32,
        mean_b: f32,
        std_b: f32,
        p_a: f32,
        seed: u64,
    ) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);

        let dist_a = Normal::new(mean_a, std_a).unwrap();
        let dist_b = Normal::new(mean_b, std_b).unwrap();

        (0..n)
            .map(|_| {
                let u: f32 = rng.random();
                if u < p_a {
                    dist_a.sample(&mut rng) as f32
                } else {
                    dist_b.sample(&mut rng) as f32
                }
            })
            .collect()
    }

    #[test]
    fn test_fit_gmm() {
        for seed in [424242, 67676767, 12345678, 1111111, 999999] {
            let data = generate_two_cluster_points(50, 5.0, 1.0, 20.0, 2.0, 0.5, seed);
            let (_, resp, _) = fit_gmm_1d_two_components(&data, GMM_MAX_ITERATIONS, GMM_TOLERANCE);

            for (&d, &r) in data.iter().zip(resp.iter()) {
                if (d - 5.0).abs() < (d - 10.0).abs() {
                    assert!(r > 0.5);
                } else {
                    assert!(r < 0.5);
                }
            }
        }
    }

    #[test]
    fn test_cluster() {
        let mut system = System::from_file("tests/files/vesicle.tpr").unwrap();
        system
            .group_create(group_name!("ClusterHeads"), "name PO4")
            .unwrap();

        system.read_ndx("tests/files/vesicle.ndx").unwrap();

        let pbc = PBC3D::new(system.get_box().unwrap());

        let mut classifier = SystemSphericalClusterClassification::new();
        classifier.cluster(&system, &pbc).unwrap();

        let clusters = classifier.clusters().as_ref().unwrap();
        assert_eq!(
            clusters.upper.len(),
            system.group_get_n_atoms("UpperLeaflet").unwrap()
        );
        assert_eq!(
            clusters.lower.len(),
            system.group_get_n_atoms("LowerLeaflet").unwrap()
        );

        for atom in system.group_iter("UpperLeaflet").unwrap() {
            assert!(clusters.upper.contains(&atom.get_index()))
        }
        for atom in system.group_iter("LowerLeaflet").unwrap() {
            assert!(clusters.lower.contains(&atom.get_index()));
        }
    }
}
