// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Implementation of a structure describing the topology of the entire system.

use crate::analysis::spherical_clustering::SystemSphericalClusterClassification;
use crate::input::{Analysis, EstimateError, LeafletClassification, OrderMap};
use crate::PANIC_MESSAGE;

use super::clustering::SystemClusterClassification;
use super::geometry::{GeometrySelection, GeometrySelectionType};
use super::leaflets::{
    MoleculeLeafletClassification, SystemLeafletClassification, SystemLeafletClassificationType,
};
use super::normal::MoleculeMembraneNormal;
use super::pbc::PBCHandler;
use crate::errors::{AnalysisError, ErrorEstimationError, TopologyError};
use crate::presentation::converter::{MolConvert, ResultsConverter};
use getset::{CopyGetters, Getters, MutGetters};
use groan_rs::prelude::{Atom, Vector3D};
use groan_rs::system::{ParallelTrajData, System};
use hashbrown::HashSet;
use indexmap::IndexMap;
use molecule::{handle_moltypes, MoleculeTypes};
use std::ops::Add;

pub(crate) mod atom;
pub(crate) mod bond;
pub(crate) mod classify;
pub(crate) mod molecule;
pub(crate) mod uatom;

/// Structure describing the topology of the system.
#[derive(Debug, Clone, CopyGetters, Getters, MutGetters)]
pub(crate) struct SystemTopology {
    /// ID of a thread working with this SystemTopology.
    #[getset(get = "pub(super)")]
    thread_id: usize,
    /// Total number of threads used in the analysis.
    n_threads: usize,
    /// Index of the current frame from the start of the iteration.
    #[getset(get_copy = "pub(super)")]
    frame: usize,
    /// Step size of the analysis.
    #[getset(get_copy = "pub(super)")]
    step_size: usize,
    /// Total number of frames analyzed by this thread.
    #[getset(get_copy = "pub(crate)")]
    total_frames: usize,
    /// All molecule types for which order parameters are calculated.
    #[getset(get = "pub(crate)", get_mut = "pub(super)")]
    molecule_types: MoleculeTypes,
    /// Parameters of error estimation.
    #[getset(get = "pub(super)")]
    estimate_error: Option<EstimateError>,
    /// Structure for geometry selection.
    #[getset(get = "pub(super)")]
    geometry: GeometrySelectionType,
    /// Structure for system-level leaflet classification.
    #[getset(get = "pub(super)", get_mut = "pub(super)")]
    leaflet_classification: Option<SystemLeafletClassification>,
    /// Should PBC be handled?
    #[getset(get_copy = "pub(super)")]
    handle_pbc: bool,
}

impl SystemTopology {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        system: &System,
        molecule_types: MoleculeTypes,
        estimate_error: Option<EstimateError>,
        step_size: usize,
        n_threads: usize,
        geometry: GeometrySelectionType,
        leaflets: Option<&LeafletClassification>,
        handle_pbc: bool,
    ) -> SystemTopology {
        let leaflet_classification = match leaflets {
            Some(LeafletClassification::Global(x)) => Some(SystemLeafletClassification::new(
                SystemLeafletClassificationType::MembraneCenter(Vector3D::new(0.0, 0.0, 0.0)),
                x.frequency(),
                step_size,
            )),
            Some(LeafletClassification::Clustering(x)) => Some(SystemLeafletClassification::new(
                SystemLeafletClassificationType::Clustering(SystemClusterClassification::new(
                    system,
                    x.flip(),
                )),
                x.frequency(),
                step_size,
            )),
            Some(LeafletClassification::SphericalClustering(x)) => {
                Some(SystemLeafletClassification::new(
                    SystemLeafletClassificationType::SphericalClustering(
                        SystemSphericalClusterClassification::new(),
                    ),
                    x.frequency(),
                    step_size,
                ))
            }
            _ => None,
        };

        SystemTopology {
            thread_id: 0,
            frame: step_size, // will be modified during initialization
            n_threads,
            step_size,
            total_frames: 0,
            molecule_types,
            estimate_error,
            geometry,
            leaflet_classification,
            handle_pbc,
        }
    }

    /// Convert the topology into a results structure.
    #[inline(always)]
    pub(super) fn convert<O: MolConvert>(self, analysis: Analysis) -> O {
        let converter = ResultsConverter::<O>::new(analysis);
        converter.convert_topology(self)
    }

    /// Initialize reading of a new frame.
    #[inline]
    pub(super) fn init_new_frame<'a>(
        &mut self,
        frame: &'a System,
        pbc_handler: &mut impl PBCHandler<'a>,
    ) {
        pbc_handler.init_new_frame();
        self.geometry.init_new_frame(frame, pbc_handler);
        handle_moltypes!(self.molecule_types_mut(), x => x.iter_mut().for_each(|mol| mol.init_new_frame()));
    }

    /// Increase the frame counter by `step_size`.
    #[inline(always)]
    pub(super) fn increase_frame_counter(&mut self) {
        self.frame += self.step_size * self.n_threads;
        self.total_frames += 1;
    }

    /// Check that all frames for manual leaflet classification and membrane normal specification were used.
    #[inline(always)]
    pub(super) fn validate_run(
        &self,
        step: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.validate_leaflet_classification(step)?;
        self.validate_normals_specification()?;
        Ok(())
    }

    /// Report basic information about the result of the error estimation.
    #[inline(always)]
    pub(super) fn error_info(&self) -> Result<(), ErrorEstimationError> {
        let Some(estimate_error) = self.estimate_error() else {
            return Ok(());
        };
        let n_blocks = estimate_error.n_blocks();
        handle_moltypes!(self.molecule_types(), x => {
            for mol in x.iter() {
                if !mol.log_error_info(n_blocks)? {
                    break;
                }
            }
        });

        Ok(())
    }

    /// Log information about leaflet assignment for the first frame.
    /// This function should only be called once.
    #[cold]
    pub(super) fn log_first_frame_leaflet_assignment_info(&self) {
        let mut upper = IndexMap::new();
        let mut lower = IndexMap::new();

        handle_moltypes!(self.molecule_types(), x => {
            for mol in x.iter() {
                if let Some(x) = mol.leaflet_classification().as_ref() {
                    let stats = x.statistics();
                    // we do not want `LIPID: 0` in the output
                    if stats.0 > 0 {
                        upper.insert(mol.name(), stats.0);
                    }

                    if stats.1 > 0 {
                        lower.insert(mol.name(), stats.1);
                    }
                }
            }
        });

        fn indexmap2string(map: &IndexMap<&String, usize>) -> String {
            map.into_iter()
                .map(|(name, number)| (name, number.to_string()))
                .fold(String::new(), |mut acc, (name, number)| {
                    if !acc.is_empty() {
                        acc.push_str(", ");
                    }
                    acc.push_str(name);
                    acc.push_str(": ");
                    acc.push_str(&number);
                    acc
                })
        }

        if !upper.is_empty() {
            colog_info!(
                "Upper leaflet in the first analyzed frame: {}",
                indexmap2string(&upper)
            );
        }

        if !lower.is_empty() {
            colog_info!(
                "Lower leaflet in the first analyzed frame: {}",
                indexmap2string(&lower)
            );
        }
    }

    /// Log the total number of frames analyzed by this thread.
    pub(super) fn log_total_analyzed_frames(&self) {
        colog_info!(
            "Trajectory reading completed. Analyzed {} trajectory frames.",
            self.total_frames,
        );
    }
}

impl Add<SystemTopology> for SystemTopology {
    type Output = Self;

    #[inline]
    fn add(self, rhs: SystemTopology) -> Self::Output {
        Self {
            thread_id: self.thread_id, // at this point, the thread_id does not matter
            n_threads: self.n_threads,
            frame: self.frame.max(rhs.frame), // get the higher frame, although it does not actually matter since the frame number is wrong at this point
            step_size: self.step_size,
            total_frames: self.total_frames + rhs.total_frames,
            molecule_types: self.molecule_types + rhs.molecule_types,
            estimate_error: self.estimate_error,
            geometry: self.geometry,
            leaflet_classification: self.leaflet_classification,
            handle_pbc: self.handle_pbc,
        }
    }
}

impl ParallelTrajData for SystemTopology {
    fn reduce(mut data: Vec<Self>) -> Self {
        // sort the data by `thread_id` - `groan_rs` does not guarantee the order
        // of data returned from the individual threads; although there is probably no reason
        // for `groan` to return a different order than a simple sequential, we should make sure
        data.sort_by(|a, b| a.thread_id.cmp(&b.thread_id));

        let mut iter = data.into_iter();
        let first = iter.next().unwrap_or_else(|| {
            panic!(
                "FATAL GORDER ERROR | SystemTopology::reduce | Vector should not be empty. {}",
                PANIC_MESSAGE
            )
        });

        iter.fold(first, |acc, top| acc + top)
    }

    fn initialize(&mut self, thread_id: usize) {
        self.thread_id = thread_id;
        self.frame *= thread_id;
    }
}

/// Trait implemented by structures which can be used to calculate order parameters.
/// For CG and AA systems, use `OrderBonds`.
/// For UA systems, use `UAOrderAtoms`.
pub(crate) trait OrderCalculable: Sized + Add<Output = Self> {
    /// A HashSet of basic building blocks: either bonds (AA, CG) or heavy atoms (UA).
    type ElementSet;

    /// Create a new instance of the structure.
    fn new<'a>(
        system: &System,
        elements: &Self::ElementSet,
        min_index: usize,
        classify_leaflets: bool,
        ordermap: Option<&OrderMap>,
        errors: bool,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Result<Self, TopologyError>;

    /// Insert new instances of the elements into the structure.
    /// Only requires `min_index` and the elements are then reconstructed based on the reference structure.
    fn insert(&mut self, min_index: usize);

    /// Perform the analysis for a single frame.
    fn analyze_frame<'a, Geom: GeometrySelection>(
        &mut self,
        frame: &'a System,
        pbc_handler: &'a impl PBCHandler<'a>,
        frame_index: usize,
        geometry: &Geom,
        leaflet: &mut Option<MoleculeLeafletClassification>,
        normal: &mut MoleculeMembraneNormal,
    ) -> Result<(), AnalysisError>;

    /// Initialize the analysis of a new frame.
    fn init_new_frame(&mut self);

    /// Get the number of molecules for which data are (to be) collected.
    fn n_molecules(&self) -> usize;

    /// Returns `true` if there is no bond type or atom type stored in the structure.
    fn is_empty(&self) -> bool;

    /// Get basic timewise order data for the first molecule (block size and the number of frames).
    fn get_timewise_info(&self, n_blocks: usize) -> Option<(usize, usize)>;
}

/// Checks that `min_index` is not higher than index of an atom involved in bonding.
/// Checks that there is no self-bonding.
/// Panics if any of these checks fails.
fn bonds_sanity_check(bonds: &HashSet<(usize, usize)>, min_index: usize) {
    for &(index1, index2) in bonds {
        for index in [index1, index2] {
            if index < min_index {
                panic!(
                    "FATAL GORDER ERROR | topology::bonds_sanity_check | Atom index '{}' is lower than minimum index '{}'. {}",
                    index1, min_index, PANIC_MESSAGE
                );
            }
        }

        if index1 == index2 {
            panic!(
                "FATAL GORDER ERROR | topology::bonds_sanity_check | Bond between the same atom (index: '{}'). {}",
                index1, PANIC_MESSAGE
            );
        }
    }
}

/// Get atoms corresponding to the provided absolute indices,
/// panicking if these indices are out of range.
fn get_atoms_from_bond(system: &System, index1: usize, index2: usize) -> (&Atom, &Atom) {
    let atom1 = system
        .get_atom(index1)
        .unwrap_or_else(|_| panic!("FATAL GORDER ERROR | topology::get_atoms_from_bond | Index '{}' does not correspond to an existing atom. {}", index1, PANIC_MESSAGE));

    let atom2 = system
        .get_atom(index2)
        .unwrap_or_else(|_| panic!("FATAL GORDER ERROR | topology::get_atoms_from_bond | Index '{}' does not correspond to an existing atom. {}", index2, PANIC_MESSAGE));

    (atom1, atom2)
}

#[cfg(test)]
mod tests {

    use super::*;

    // this tests whether the SystemTopology structures correctly initialize and iterate through frames
    // this is important for shared data to be properly read in leaflet assignment
    #[test]
    fn test_simulated_iteration() {
        let system = System::new("System", vec![], None);
        for n_frames in 0..301 {
            for step in [1, 2, 3, 4, 5, 7, 10, 15, 20, 100] {
                let expected_visited_frames = (0..n_frames).step_by(step).collect::<Vec<usize>>();

                for n_threads in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128] {
                    let mut visited_frames = Vec::new();

                    let mut threads = (0..n_threads)
                        .map(|i| {
                            let mut top = SystemTopology::new(
                                &system,
                                MoleculeTypes::BondBased(vec![]),
                                None,
                                step,
                                n_threads,
                                GeometrySelectionType::default(),
                                None,
                                true,
                            );
                            top.initialize(i);
                            top
                        })
                        .collect::<Vec<SystemTopology>>();

                    loop {
                        let mut all_done = true;

                        for top in threads.iter_mut() {
                            if top.frame < n_frames {
                                visited_frames.push(top.frame);
                                all_done = false;
                            }
                            top.increase_frame_counter();
                        }

                        if all_done {
                            break;
                        }
                    }

                    /*println!(
                        "n_frames {} step {} n_threads {}",
                        n_frames, step, n_threads
                    );
                    println!("VISITED {:?}", visited_frames);
                    println!("EXPECTED {:?}\n", expected_visited_frames);*/
                    assert_eq!(visited_frames.len(), expected_visited_frames.len());
                    for (val, exp) in visited_frames.iter().zip(expected_visited_frames.iter()) {
                        assert_eq!(val, exp);
                    }
                }
            }
        }
    }
}
