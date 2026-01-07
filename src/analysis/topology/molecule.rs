// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for constructing and working with topology of a molecule type.

use std::ops::Add;

use getset::{Getters, MutGetters};
use groan_rs::{structures::group::Group, system::System};
use hashbrown::HashSet;

use super::{
    super::{
        geometry::GeometrySelection, leaflets::MoleculeLeafletClassification,
        normal::MoleculeMembraneNormal, pbc::PBCHandler,
    },
    atom::OrderAtoms,
    bond::{BondTopology, OrderBonds},
    bonds_sanity_check, get_atoms_from_bond,
    uatom::UAOrderAtoms,
    OrderCalculable, SystemTopology,
};
use crate::{analysis::geometry::GeometrySelectionType, errors::ErrorEstimationError};
use crate::{
    errors::{AnalysisError, TopologyError},
    input::OrderMap,
    PANIC_MESSAGE,
};

/// Specifies what structures are molecule types based on.
#[derive(Debug, Clone)]
pub(crate) enum MoleculeTypes {
    /// For AA and CG systems.
    BondBased(Vec<MoleculeType<OrderBonds>>),
    /// For UA systems.
    AtomBased(Vec<MoleculeType<UAOrderAtoms>>),
}

/// Macro helping with handling of distinct MoleculeTypes.
macro_rules! handle_moltypes {
    ($self:expr, $x:ident => $code:expr) => {
        use crate::analysis::topology::molecule::MoleculeTypes;

        match $self {
            MoleculeTypes::AtomBased($x) => $code,
            MoleculeTypes::BondBased($x) => $code,
        }
    };
}
pub(crate) use handle_moltypes;

impl MoleculeTypes {
    /// Analyze a single trajectory frame.
    pub(crate) fn analyze_frame<'a>(
        &mut self,
        frame: &'a System,
        topology: &'a mut SystemTopology,
        pbc_handler: &'a impl PBCHandler<'a>,
    ) -> Result<(), AnalysisError> {
        let frame_index = topology.frame();
        // perform system level leaflet-classification, if needed
        if let Some(classification) = topology.leaflet_classification_mut() {
            classification.run(frame, pbc_handler, frame_index)?;
        }

        handle_moltypes!(self, x => {
            for molecule in x.iter_mut() {
                // assign molecules to leaflets
                if let Some(classifier) = molecule.leaflet_classification_mut() {
                    classifier.assign_lipids(frame, pbc_handler, frame_index, topology.leaflet_classification())?;
                }

                // calculate order parameters
                match topology.geometry() {
                    GeometrySelectionType::None(x) => {
                        molecule.analyze_frame(frame, pbc_handler, frame_index, x)?
                    }
                    GeometrySelectionType::Cuboid(x) => {
                        molecule.analyze_frame(frame, pbc_handler, frame_index, x)?
                    }
                    GeometrySelectionType::Cylinder(x) => {
                        molecule.analyze_frame(frame, pbc_handler, frame_index, x)?
                    }
                    GeometrySelectionType::Sphere(x) => {
                        molecule.analyze_frame(frame, pbc_handler, frame_index, x)?
                    }
                }

                // store dynamic membrane normals, if requested
                molecule.membrane_normal_mut().store_normals();
            }
        });

        Ok(())
    }

    /// Get the number of molecule types.
    #[inline(always)]
    pub(crate) fn n_molecule_types(&self) -> usize {
        match self {
            Self::AtomBased(x) => x.len(),
            Self::BondBased(x) => x.len(),
        }
    }
}

impl From<Vec<MoleculeType<OrderBonds>>> for MoleculeTypes {
    fn from(value: Vec<MoleculeType<OrderBonds>>) -> Self {
        Self::BondBased(value)
    }
}

impl From<Vec<MoleculeType<UAOrderAtoms>>> for MoleculeTypes {
    fn from(value: Vec<MoleculeType<UAOrderAtoms>>) -> Self {
        Self::AtomBased(value)
    }
}

impl Add<MoleculeTypes> for MoleculeTypes {
    type Output = MoleculeTypes;

    fn add(self, rhs: MoleculeTypes) -> Self::Output {
        match (self, rhs) {
            (Self::AtomBased(x), Self::AtomBased(y)) => MoleculeTypes::AtomBased(
                x
                    .into_iter()
                    .zip(y)
                    .map(|(a, b)| a + b)
                    .collect::<Vec<MoleculeType<UAOrderAtoms>>>()
            ),
            (Self::BondBased(x), Self::BondBased(y)) => MoleculeTypes::BondBased(
                x
                    .into_iter()
                    .zip(y)
                    .map(|(a, b)| a + b)
                    .collect::<Vec<MoleculeType<OrderBonds>>>()
            ),
            (Self::AtomBased(_), Self::BondBased(_)) | (Self::BondBased(_), Self::AtomBased(_)) => panic!(
                "FATAL GORDER ERROR | MoleculeTypes::add | Cannot add atom-based and bond-based structures. {}", PANIC_MESSAGE),
        }
    }
}

/// Represents a specific type of molecule with all its atoms and bonds.
/// Contains indices of all atoms that are part of molecules of this type.
#[derive(Debug, Clone, Getters, MutGetters)]
pub(crate) struct MoleculeType<O: OrderCalculable> {
    /// Name of the molecule.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    name: String,
    /// Topology of the molecule (set of all bonds).
    #[getset(get = "pub(crate)")]
    topology: MoleculeTopology,
    /// Relative indices of all atoms of the molecule
    #[getset(get = "pub(crate)")]
    reference_atoms: Vec<usize>,
    /// List of all bonds or united-atom atoms for which order parameter should be calculated.
    #[getset(get = "pub(crate)")]
    order_structure: O,
    /// List of all atoms for which order parameter should be calculated (all atoms for CG, heavy atoms for AA).
    #[getset(get = "pub(crate)")]
    order_atoms: OrderAtoms,
    /// Either a statically assigned or dynamically computed membrane normal for all molecules of this type.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    membrane_normal: MoleculeMembraneNormal,
    /// Method to use to assign this molecule to membrane leaflet.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    leaflet_classification: Option<MoleculeLeafletClassification>,
}

impl<O: OrderCalculable> MoleculeType<O> {
    /// Initialize reading of a new frame.
    #[inline(always)]
    pub(crate) fn init_new_frame(&mut self) {
        self.membrane_normal.init_new_frame();
        self.order_structure.init_new_frame();
    }

    /// Get the number of molecules of this molecule type.
    #[inline(always)]
    pub(crate) fn n_molecules(&self) -> usize {
        self.order_structure.n_molecules()
    }

    /// Report basic information about the result of the error estimation for this molecule.
    /// Returns `true` if this should be performed for the next molecule. Else returns `false`.
    pub(crate) fn log_error_info(&self, n_blocks: usize) -> Result<bool, ErrorEstimationError> {
        if self.order_structure.is_empty() {
            return Ok(true); // molecule has no bonds/atoms; continue with the next one
        }

        let (block_size, n_frames) = match self.order_structure.get_timewise_info(n_blocks) {
            Some((x, y)) => (x, y),
            None => return Ok(false), // if timewise data is not calculated for this molecule type, it is not calculated for any
        };

        if block_size < 1 {
            return Err(ErrorEstimationError::NotEnoughData(n_frames, n_blocks));
        }

        if block_size < 10 {
            colog_warn!("Error estimation: you probably do not have enough data for reasonable error estimation ({} frames might be too little).",
                n_frames);
        }

        colog_info!(
                "Error estimation: collected {} blocks, each consisting of {} trajectory frames (total: {} frames).",
                n_blocks, block_size, n_blocks * block_size
            );

        if n_frames != n_blocks * block_size {
            colog_info!(
                "Error estimation: data from {} frame(s) could not be distributed into blocks and will be excluded from error estimation.",
                n_frames - n_blocks * block_size,
            );
        }

        Ok(false) // only perform the logging for a single molecule
    }

    /// Insert new elements into the molecule.
    /// We only need to provide the minimum index of the new molecule and the rest will be constructed
    /// using the previously stored information.
    pub(crate) fn insert(
        &mut self,
        system: &System,
        min_index: usize,
    ) -> Result<(), TopologyError> {
        self.order_structure.insert(min_index);

        // reconstruct all atoms of the molecule
        let molecule = Group::from_indices(
            self.reference_atoms.iter().map(|a| a + min_index).collect(),
            usize::MAX,
        );

        if let Some(classifier) = self.leaflet_classification.as_mut() {
            classifier.insert(&molecule, system)?;
        }

        self.membrane_normal.insert(&molecule, system)?;

        Ok(())
    }
}

impl MoleculeType<OrderBonds> {
    /// Create new molecule type for AA or CG order calculation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_bond_based<'a>(
        system: &System,
        name: String,
        topology: &MoleculeTopology,
        reference_atoms: &[usize],
        order_bonds: &HashSet<(usize, usize)>,
        order_atoms: &[usize],
        min_index: usize,
        leaflet_classification: Option<MoleculeLeafletClassification>,
        ordermap_params: Option<&OrderMap>,
        errors: bool,
        pbc_handler: &impl PBCHandler<'a>,
        membrane_normal: MoleculeMembraneNormal,
    ) -> Result<Self, TopologyError> {
        Ok(Self {
            name,
            topology: topology.clone(),
            order_structure: OrderBonds::new(
                system,
                order_bonds,
                min_index,
                leaflet_classification.is_some(),
                ordermap_params,
                errors,
                pbc_handler,
            )?,
            reference_atoms: reference_atoms.to_owned(),
            order_atoms: OrderAtoms::new(system, order_atoms, min_index),
            leaflet_classification,
            membrane_normal,
        })
    }

    /// Calculate order parameters for bonds of a single molecule type from a single simulation frame.
    #[inline(always)]
    pub(crate) fn analyze_frame<'a, Geom: GeometrySelection>(
        &mut self,
        frame: &'a System,
        pbc_handler: &'a impl PBCHandler<'a>,
        frame_index: usize,
        geometry: &Geom,
    ) -> Result<(), AnalysisError> {
        self.order_structure.analyze_frame(
            frame,
            pbc_handler,
            frame_index,
            geometry,
            &mut self.leaflet_classification,
            &mut self.membrane_normal,
        )
    }
}

impl MoleculeType<UAOrderAtoms> {
    /// Create new molecule type for UA order calculation.
    #[allow(clippy::too_many_arguments, clippy::ptr_arg)]
    pub(crate) fn new_atom_based<'a>(
        system: &System,
        name: String,
        topology: &MoleculeTopology,
        reference_atoms: &[usize],
        order_atoms: &Vec<usize>,
        min_index: usize,
        leaflet_classification: Option<MoleculeLeafletClassification>,
        ordermap_params: Option<&OrderMap>,
        errors: bool,
        pbc_handler: &impl PBCHandler<'a>,
        membrane_normal: MoleculeMembraneNormal,
    ) -> Result<Self, TopologyError> {
        Ok(Self {
            name,
            topology: topology.clone(),
            order_structure: UAOrderAtoms::new(
                system,
                order_atoms,
                min_index,
                leaflet_classification.is_some(),
                ordermap_params,
                errors,
                pbc_handler,
            )?,
            reference_atoms: reference_atoms.to_owned(),
            order_atoms: OrderAtoms::new(system, order_atoms, min_index),
            leaflet_classification,
            membrane_normal,
        })
    }

    /// Calculate order parameters for atoms of a single molecule type from a single simulation frame.
    #[inline(always)]
    pub(super) fn analyze_frame<'a, Geom: GeometrySelection>(
        &mut self,
        frame: &'a System,
        pbc_handler: &'a impl PBCHandler<'a>,
        frame_index: usize,
        geometry: &Geom,
    ) -> Result<(), AnalysisError> {
        self.order_structure.analyze_frame(
            frame,
            pbc_handler,
            frame_index,
            geometry,
            &mut self.leaflet_classification,
            &mut self.membrane_normal,
        )
    }
}

impl<O: OrderCalculable> Add<MoleculeType<O>> for MoleculeType<O> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            name: self.name,
            topology: self.topology,
            order_structure: self.order_structure + rhs.order_structure,
            reference_atoms: self.reference_atoms,
            order_atoms: self.order_atoms,
            leaflet_classification: self.leaflet_classification,
            membrane_normal: self.membrane_normal + rhs.membrane_normal,
        }
    }
}

/// Collection of all bond types in a molecule describing its topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MoleculeTopology {
    pub(super) bonds: HashSet<BondTopology>,
}

impl MoleculeTopology {
    /// Create new molecule topology from a set of bonds (absolute indices) and the minimum index in the molecule.
    ///
    /// ## Panics
    /// - Panics if the `min_index` is higher than any index inside the `bonds` set.
    /// - Panics if there is a bond connecting the same atom (e.g. 14-14) in the `bonds` set.
    /// - Panics if an index in the `bonds` set does not correspond to an existing atom.
    pub(crate) fn new(
        system: &System,
        bonds: &HashSet<(usize, usize)>,
        min_index: usize,
    ) -> MoleculeTopology {
        bonds_sanity_check(bonds, min_index);

        let mut converted_bonds = HashSet::new();
        for &(index1, index2) in bonds {
            let (atom1, atom2) = get_atoms_from_bond(system, index1, index2);
            let bond = BondTopology::new(index1 - min_index, atom1, index2 - min_index, atom2);

            if !converted_bonds.insert(bond) {
                panic!(
                    "FATAL GORDER ERROR | MoleculeTopology::new | Bond between atoms '{}' and '{}' defined multiple times. {}",
                    index1, index2, PANIC_MESSAGE
                );
            }
        }

        MoleculeTopology {
            bonds: converted_bonds,
        }
    }

    #[allow(unused)]
    pub(crate) fn new_raw(bonds: HashSet<BondTopology>) -> Self {
        Self { bonds }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        analysis::{
            geometry::NoSelection,
            pbc::PBC3D,
            topology::{classify::MoleculesClassifier, MoleculeTypes, SystemTopology},
        },
        input::{Analysis, AnalysisType},
    };

    use super::*;

    fn expected_bonds(system: &System) -> [BondTopology; 5] {
        [
            BondTopology::new(
                44,
                system.get_atom(169).unwrap(),
                45,
                system.get_atom(170).unwrap(),
            ),
            BondTopology::new(
                44,
                system.get_atom(169).unwrap(),
                46,
                system.get_atom(171).unwrap(),
            ),
            BondTopology::new(
                88,
                system.get_atom(213).unwrap(),
                89,
                system.get_atom(214).unwrap(),
            ),
            BondTopology::new(
                88,
                system.get_atom(213).unwrap(),
                90,
                system.get_atom(215).unwrap(),
            ),
            BondTopology::new(
                121,
                system.get_atom(246).unwrap(),
                122,
                system.get_atom(247).unwrap(),
            ),
        ]
    }

    #[test]
    fn molecule_topology_new() {
        let system = System::from_file("tests/files/pcpepg.tpr").unwrap();
        let bonds = [(169, 170), (169, 171), (213, 214), (213, 215), (246, 247)];
        let bonds_set = HashSet::from(bonds);
        let topology = MoleculeTopology::new(&system, &bonds_set, 125);

        let expected_bonds = expected_bonds(&system);

        // bonds can be in any order inside `order_bonds`
        for bond in topology.bonds.iter() {
            if !expected_bonds.iter().any(|expected| bond == expected) {
                panic!("Expected bond not found.")
            }
        }
    }

    #[test]
    fn n_molecules_aa() {
        let mut system = System::from_file("tests/files/pcpepg.tpr").unwrap();
        crate::analysis::common::create_group(
            &mut system,
            "HeavyAtoms",
            "@membrane and element name carbon",
        )
        .unwrap();
        crate::analysis::common::create_group(
            &mut system,
            "Hydrogens",
            "@membrane and element name hydrogen",
        )
        .unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
            .unwrap();

        let molecules =
            MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap();

        let topology = SystemTopology::new(
            &system,
            molecules,
            None,
            1,
            1,
            crate::analysis::geometry::GeometrySelectionType::None(NoSelection {}),
            None,
            true,
        );

        let expected = [131, 128, 15];
        let molecule_types = match topology.molecule_types() {
            MoleculeTypes::AtomBased(_) => panic!("Molecule types should be bond-based."),
            MoleculeTypes::BondBased(x) => x,
        };
        for (i, molecule) in molecule_types.iter().enumerate() {
            assert_eq!(molecule.n_molecules(), expected[i]);
        }
    }

    #[test]
    fn n_molecules_cg() {
        let mut system = System::from_file("tests/files/cg.tpr").unwrap();
        crate::analysis::common::create_group(&mut system, "Beads", "@membrane").unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .build()
            .unwrap();

        let molecules =
            MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap();

        let topology = SystemTopology::new(
            &system,
            molecules,
            None,
            1,
            1,
            crate::analysis::geometry::GeometrySelectionType::None(NoSelection {}),
            None,
            true,
        );

        let expected = [242, 242, 24];
        let molecule_types = match topology.molecule_types() {
            MoleculeTypes::AtomBased(_) => panic!("Molecule types should be bond-based."),
            MoleculeTypes::BondBased(x) => x,
        };
        for (i, molecule) in molecule_types.iter().enumerate() {
            assert_eq!(molecule.n_molecules(), expected[i]);
        }
    }
}
