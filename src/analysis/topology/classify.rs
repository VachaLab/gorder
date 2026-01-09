// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for classifying molecules in the molecular system.

use std::time::{Duration, Instant};

use groan_rs::{structures::group::Group, system::System};
use hashbrown::{HashMap, HashSet};
use once_cell::sync::Lazy;

use crate::{
    analysis::{
        common::macros::group_name, leaflets::MoleculeLeafletClassification,
        normal::MoleculeMembraneNormal, pbc::PBCHandler, spinner::Spinner,
    },
    errors::TopologyError,
    input::{Analysis, AnalysisType, OrderMap},
    PANIC_MESSAGE,
};

use super::{
    bond::OrderBonds,
    molecule::{MoleculeTopology, MoleculeType, MoleculeTypes},
    uatom::UAOrderAtoms,
    OrderCalculable,
};

/// Time in ms after which the progress of molecule classification will be logged.
static MOLECULE_CLASSIFICATION_TIME_LIMIT: Lazy<u64> = Lazy::new(|| {
    std::env::var("GORDER_MOLECULE_CLASSIFICATION_TIME_LIMIT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(500)
});

/// Helper structure for classifying molecules.
#[derive(Debug, Clone)]
#[allow(private_interfaces)]
pub(crate) struct MoleculesClassifier;

impl MoleculesClassifier {
    /// Classify molecules in the system based on their topology and return a list of molecule types,
    /// along with information about the atoms that form each molecule.
    pub(crate) fn classify<'a>(
        system: &'a System,
        analysis_options: &Analysis,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Result<MoleculeTypes, TopologyError> {
        match analysis_options.analysis_type() {
            AnalysisType::AAOrder {
                heavy_atoms: _,
                hydrogens: _,
            } => {
                let classifier = BondBasedClassifier {
                    order_group_1: group_name!("HeavyAtoms"),
                    order_group_2: group_name!("Hydrogens"),
                    visited_atoms: HashSet::new(),
                    molecules: Vec::new(),
                };
                Ok(classifier
                    .classify(system, analysis_options, pbc_handler)?
                    .into())
            }
            AnalysisType::CGOrder { beads: _ } => {
                let classifier = BondBasedClassifier {
                    order_group_1: group_name!("Beads"),
                    order_group_2: group_name!("Beads"),
                    visited_atoms: HashSet::new(),
                    molecules: Vec::new(),
                };
                Ok(classifier
                    .classify(system, analysis_options, pbc_handler)?
                    .into())
            }
            AnalysisType::UAOrder {
                saturated: _,
                unsaturated: _,
                ignore: _,
            } => {
                let classifier = AtomBasedClassifier {
                    order_group: group_name!("SatUnsat"),
                    visited_atoms: HashSet::new(),
                    molecules: Vec::new(),
                };

                Ok(classifier
                    .classify(system, analysis_options, pbc_handler)?
                    .into())
            }
        }
    }
}

/// Trait implemented by structures that can be used to classify molecules-
trait MoleculesClassify: Sized {
    type MolType: OrderCalculable;
    type MolConstructor: MolConstruct;

    /// Get the name of the group corresponding to the order atoms (heavy atoms for AA).
    fn order_group(&self) -> &str;

    /// Get a mutable reference to the molecule types stored in the classifier.
    fn molecules_mut(&mut self) -> &mut Vec<MoleculeType<Self::MolType>>;

    /// Get an immutable reference to the molecule types stored in the classifier.
    fn molecules(&self) -> &[MoleculeType<Self::MolType>];

    /// Extract molecules from the classifier and cosume the classifier.
    fn extract_molecules(self) -> Vec<MoleculeType<Self::MolType>>;

    /// Add an atom index to the set of visited indices.
    /// Returns `true` if the index was not previously in the set.
    /// Returns `false` if the index was previously in the set.
    fn add2visited(&mut self, index: usize) -> bool;

    /// Create a molecule constructor from topology.
    fn constructor_from_topology(
        &self,
        system: &System,
        molecule_topology: &MoleculeTopology,
        min_index: usize,
    ) -> Self::MolConstructor;

    /// Create a new molecule type.
    #[allow(clippy::too_many_arguments)]
    fn molecule_from_constructor<'a>(
        &self,
        system: &System,
        constructor: &Self::MolConstructor,
        leaflet_classification: Option<MoleculeLeafletClassification>,
        ordermap_params: Option<&OrderMap>,
        errors: bool,
        pbc_handler: &impl PBCHandler<'a>,
        membrane_normal: MoleculeMembraneNormal,
    ) -> Result<MoleculeType<Self::MolType>, TopologyError>;

    /// Classify molecules in the system based on their topology and returns a list of molecule types,
    /// along with information about the atoms that form each molecule.
    fn classify<'a>(
        mut self,
        system: &'a System,
        analysis_options: &Analysis,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Result<Vec<MoleculeType<Self::MolType>>, TopologyError> {
        let start = Instant::now();
        let n_atoms = system
            .group_get_n_atoms(self.order_group())
            .expect(PANIC_MESSAGE);
        let mut spinner: Option<Spinner> = None;

        for (i, atom) in system
            .group_iter(self.order_group())
            .expect(PANIC_MESSAGE)
            .enumerate()
        {
            let index = atom.get_index();

            // print progress if the molecule classification takes too long
            if i % 1000 == 0 {
                if start.elapsed() >= Duration::from_millis(*MOLECULE_CLASSIFICATION_TIME_LIMIT)
                    && spinner.is_none()
                {
                    log::info!("Detecting molecule types...");
                    spinner = Some(Spinner::new(analysis_options.silent()));
                }

                if let Some(spin) = spinner.as_mut() {
                    spin.tick(100 * i / n_atoms);
                }
            }

            if !self.add2visited(index) {
                continue;
            }

            // get the topology of the molecule
            let bonds = self.get_bonds(system, index);
            let min_index = Self::get_min_index(&bonds);
            let topology = MoleculeTopology::new(system, &bonds, min_index);

            // check whether this molecule type already exists
            if let Some(existing_molecule_type) = self
                .molecules_mut()
                .iter_mut()
                .find(|m| *m.topology() == topology)
            {
                // add a new molecule to this molecule type
                existing_molecule_type.insert(system, min_index)?;
            } else {
                // create new molecule type
                let constructor = self.constructor_from_topology(system, &topology, min_index);

                // construct a leaflet classifier
                let leaflet_classifier = if let Some(params) = analysis_options.leaflets() {
                    let mut leaflet_classifier = MoleculeLeafletClassification::new(
                        params,
                        analysis_options.membrane_normal(),
                        analysis_options.n_threads(),
                        analysis_options.step(),
                    )
                    .map_err(TopologyError::ConfigError)?;

                    leaflet_classifier.insert(constructor.get_molecule(), system)?;

                    Some(leaflet_classifier)
                } else {
                    None
                };

                // construct a membrane normal structure
                let mut normal = MoleculeMembraneNormal::from(analysis_options.membrane_normal());
                normal.insert(constructor.get_molecule(), system)?;

                let molecule = self.molecule_from_constructor(
                    system,
                    &constructor,
                    leaflet_classifier,
                    analysis_options.map().as_ref(),
                    analysis_options.estimate_error().is_some(),
                    pbc_handler,
                    normal,
                )?;

                self.molecules_mut().push(molecule);
            }
        }

        if let Some(spin) = spinner.as_ref() {
            spin.done();
        }

        // rename molecules that share names
        self.solve_name_conflicts();

        if self.sanity_check_molecules() {
            Ok(self.extract_molecules())
        } else {
            Ok(Vec::new())
        }
    }

    /// Get the minimum index of an atom involved in the specified bonds.
    fn get_min_index(bonds: &HashSet<(usize, usize)>) -> usize {
        bonds
            .iter()
            // we only have to check atom1, since the index of the other atom in the pair is always higher
            .map(|&(a1, _)| a1)
            .min()
            .unwrap_or(usize::MAX)
    }

    /// Get all bonds of a molecule by breadth-first traversal.
    fn get_bonds(&mut self, system: &System, start_index: usize) -> HashSet<(usize, usize)> {
        system
            .molecule_bonds_iter(start_index)
            .expect(PANIC_MESSAGE)
            .map(|(a1, a2)| {
                let (a1_index, a2_index) = (a1.get_index(), a2.get_index());
                self.add2visited(a1_index);
                self.add2visited(a2_index);
                (a1_index, a2_index)
            })
            .collect::<HashSet<(usize, usize)>>()
    }

    fn solve_name_conflicts(&mut self) {
        let mut counts: HashMap<String, usize> = HashMap::new();

        // count the number of individual names
        for name in self.molecules().iter().map(|mol| mol.name()) {
            *counts.entry(name.clone()).or_insert(0) += 1;
        }

        // remove entries with the count of 1
        counts.retain(|_, &mut count| count > 1);

        if counts.is_empty() {
            return;
        }

        for (name, count) in counts.iter() {
            colog_warn!("There are {} types of entities consisting of residue(s) '{}' that are actually different molecule types and will be treated as such.", count, name.replace("-", " "));
        }

        // rename molecules in conflict
        for mol in self.molecules_mut().iter_mut().rev() {
            let name = mol.name_mut();
            if let Some(count) = counts.get_mut(name) {
                *name = format!("{}{}", name, count);
                *count -= 1;
            }
        }
    }

    /// Check whether there are any molecules that can be analyzed.
    fn sanity_check_molecules(&self) -> bool {
        // if no molecules are detected, end the analysis
        if self.molecules().is_empty() {
            log::warn!("No molecules suitable for analysis detected.");
            return false;
        }

        // if only empty molecules are detected, end the analysis
        for mol in self.molecules().iter() {
            if mol.order_structure().is_empty() {
                log::warn!("No bonds/atoms suitable for analysis detected.");
                return false;
            }
        }

        true
    }
}

/// Constructs a list of bond-based molecules.
#[derive(Debug, Clone)]
struct BondBasedClassifier {
    order_group_1: &'static str,
    order_group_2: &'static str,
    visited_atoms: HashSet<usize>,
    molecules: Vec<MoleculeType<OrderBonds>>,
}

impl MoleculesClassify for BondBasedClassifier {
    type MolType = OrderBonds;
    type MolConstructor = BondBasedConstructor;

    #[inline(always)]
    fn add2visited(&mut self, index: usize) -> bool {
        self.visited_atoms.insert(index)
    }

    #[inline(always)]
    fn order_group(&self) -> &str {
        self.order_group_1
    }

    #[inline(always)]
    fn molecules(&self) -> &[MoleculeType<Self::MolType>] {
        &self.molecules
    }

    #[inline(always)]
    fn molecules_mut(&mut self) -> &mut Vec<MoleculeType<Self::MolType>> {
        &mut self.molecules
    }

    #[inline(always)]
    fn extract_molecules(self) -> Vec<MoleculeType<Self::MolType>> {
        self.molecules
    }

    /// Extract all atom indices, indices of order atoms, residue names, and indices of order bonds from molecule topology.
    fn constructor_from_topology(
        &self,
        system: &System,
        molecule_topology: &MoleculeTopology,
        min_index: usize,
    ) -> BondBasedConstructor {
        let mut residues = Vec::new();
        let mut order_atoms = HashSet::new();
        let mut all_atom_indices = Vec::new();
        let mut order_bonds = HashSet::new();

        for bond in molecule_topology.bonds.iter() {
            let (atom_type_1, atom_type_2) = (bond.atom1(), bond.atom2());

            let (a1_index, a2_index) = (
                atom_type_1.relative_index() + min_index,
                atom_type_2.relative_index() + min_index,
            );

            // collect atom indices
            for a_index in [a1_index, a2_index] {
                if system
                    .group_isin(self.order_group_1, a_index)
                    .expect(PANIC_MESSAGE)
                {
                    order_atoms.insert(a_index);
                }

                all_atom_indices.push(a_index);
            }

            // collect order bonds
            if ((system
                .group_isin(self.order_group_1, a1_index)
                .expect(PANIC_MESSAGE)
                && system
                    .group_isin(self.order_group_2, a2_index)
                    .expect(PANIC_MESSAGE))
                || (system
                    .group_isin(self.order_group_2, a1_index)
                    .expect(PANIC_MESSAGE)
                    && system
                        .group_isin(self.order_group_1, a2_index)
                        .expect(PANIC_MESSAGE)))
                && !order_bonds.insert((a1_index, a2_index))
            {
                panic!(
                    "FATAL GORDER ERROR | BondBasedClassifier::constructor_from_topology | Order bond between '{}' and '{}' encountered multiple times in a molecule. {}",
                    a1_index, a2_index, PANIC_MESSAGE
                );
            }
        }

        // collect residue names
        // we have to collect them from the group since the names have to be ordered
        let molecule = Group::from_indices(all_atom_indices, usize::MAX);
        for index in molecule.get_atoms().iter() {
            let atom = system.get_atom(index).expect(PANIC_MESSAGE);
            let res = atom.get_residue_name();
            if !residues.contains(res) {
                residues.push(res.clone());
            }
        }
        BondBasedConstructor {
            min_index,
            molecule_topology: molecule_topology.clone(),
            molecule,
            order_atoms: order_atoms.into_iter().collect(),
            order_bonds,
            residues,
        }
    }

    #[inline]
    fn molecule_from_constructor<'a>(
        &self,
        system: &System,
        constructor: &Self::MolConstructor,
        leaflet_classification: Option<MoleculeLeafletClassification>,
        ordermap_params: Option<&OrderMap>,
        errors: bool,
        pbc_handler: &impl PBCHandler<'a>,
        membrane_normal: MoleculeMembraneNormal,
    ) -> Result<MoleculeType<OrderBonds>, TopologyError> {
        MoleculeType::new_bond_based(
            system,
            constructor.construct_molecule_name(),
            &constructor.molecule_topology,
            &constructor
                .molecule
                .get_atoms()
                .iter()
                .map(|a| a - constructor.min_index)
                .collect::<Vec<usize>>(),
            &constructor.order_bonds,
            &constructor.order_atoms,
            constructor.min_index,
            leaflet_classification,
            ordermap_params,
            errors,
            pbc_handler,
            membrane_normal,
        )
    }
}

/// Constructs a list of atom-based molecules.
#[derive(Debug, Clone)]
struct AtomBasedClassifier {
    order_group: &'static str,
    visited_atoms: HashSet<usize>,
    molecules: Vec<MoleculeType<UAOrderAtoms>>,
}

impl MoleculesClassify for AtomBasedClassifier {
    type MolType = UAOrderAtoms;
    type MolConstructor = AtomBasedConstructor;

    #[inline(always)]
    fn add2visited(&mut self, index: usize) -> bool {
        self.visited_atoms.insert(index)
    }

    #[inline(always)]
    fn order_group(&self) -> &str {
        self.order_group
    }

    #[inline(always)]
    fn molecules(&self) -> &[MoleculeType<Self::MolType>] {
        &self.molecules
    }

    #[inline(always)]
    fn molecules_mut(&mut self) -> &mut Vec<MoleculeType<Self::MolType>> {
        &mut self.molecules
    }

    #[inline(always)]
    fn extract_molecules(self) -> Vec<MoleculeType<Self::MolType>> {
        self.molecules
    }

    fn constructor_from_topology(
        &self,
        system: &System,
        molecule_topology: &MoleculeTopology,
        min_index: usize,
    ) -> Self::MolConstructor {
        let mut residues = Vec::new();
        let mut order_atoms = HashSet::new();
        let mut all_atom_indices = Vec::new();

        for bond in molecule_topology.bonds.iter() {
            let (atom_type_1, atom_type_2) = (bond.atom1(), bond.atom2());

            let (a1_index, a2_index) = (
                atom_type_1.relative_index() + min_index,
                atom_type_2.relative_index() + min_index,
            );

            // collect atom indices
            for a_index in [a1_index, a2_index] {
                if system
                    .group_isin(self.order_group, a_index)
                    .expect(PANIC_MESSAGE)
                {
                    order_atoms.insert(a_index);
                }

                all_atom_indices.push(a_index);
            }
        }

        // collect residue names
        // we have to collect them from the group since the names have to be ordered
        let molecule = Group::from_indices(all_atom_indices, usize::MAX);
        for index in molecule.get_atoms().iter() {
            let atom = system.get_atom(index).expect(PANIC_MESSAGE);
            let res = atom.get_residue_name();
            if !residues.contains(res) {
                residues.push(res.clone());
            }
        }
        AtomBasedConstructor {
            min_index,
            molecule_topology: molecule_topology.clone(),
            molecule,
            order_atoms: order_atoms.into_iter().collect(),
            residues,
        }
    }

    #[inline]
    fn molecule_from_constructor<'a>(
        &self,
        system: &System,
        constructor: &Self::MolConstructor,
        leaflet_classification: Option<MoleculeLeafletClassification>,
        ordermap_params: Option<&OrderMap>,
        errors: bool,
        pbc_handler: &impl PBCHandler<'a>,
        membrane_normal: MoleculeMembraneNormal,
    ) -> Result<MoleculeType<Self::MolType>, TopologyError> {
        MoleculeType::new_atom_based(
            system,
            constructor.construct_molecule_name(),
            &constructor.molecule_topology,
            &constructor
                .molecule
                .get_atoms()
                .iter()
                .map(|a| a - constructor.min_index)
                .collect::<Vec<usize>>(),
            &constructor.order_atoms,
            constructor.min_index,
            leaflet_classification,
            ordermap_params,
            errors,
            pbc_handler,
            membrane_normal,
        )
    }
}

/// Helper trait implemented by structures used when classifying molecules.
trait MolConstruct {
    /// Create name of the molecule from its residue names.
    fn construct_molecule_name(&self) -> String;

    /// Get the group corresponding to all the atoms of the molecule.
    fn get_molecule(&self) -> &Group;
}

/// Helper for constructing bond-based molecules.
#[derive(Debug, Clone)]
struct BondBasedConstructor {
    min_index: usize,
    molecule_topology: MoleculeTopology,
    molecule: Group,
    order_atoms: Vec<usize>,
    order_bonds: HashSet<(usize, usize)>,
    residues: Vec<String>,
}

impl MolConstruct for BondBasedConstructor {
    #[inline(always)]
    fn construct_molecule_name(&self) -> String {
        self.residues.join("-")
    }

    #[inline(always)]
    fn get_molecule(&self) -> &Group {
        &self.molecule
    }
}

/// Helper for constructing atom-based molecules.
#[derive(Debug, Clone)]
struct AtomBasedConstructor {
    min_index: usize,
    molecule_topology: MoleculeTopology,
    molecule: Group,
    order_atoms: Vec<usize>,
    residues: Vec<String>,
}

impl MolConstruct for AtomBasedConstructor {
    #[inline(always)]
    fn construct_molecule_name(&self) -> String {
        self.residues.join("-")
    }

    #[inline(always)]
    fn get_molecule(&self) -> &Group {
        &self.molecule
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        analysis::{
            common::create_group,
            pbc::PBC3D,
            topology::{
                atom::{AtomType, OrderAtoms},
                bond::BondTopology,
            },
        },
        input::{AnalysisType, LeafletClassification},
    };

    use super::*;

    fn pope_topology() -> MoleculeTopology {
        let string_representation = "0 POPE N 3 POPE HN3
0 POPE N 2 POPE HN2
0 POPE N 4 POPE C12
0 POPE N 1 POPE HN1
4 POPE C12 7 POPE C11
4 POPE C12 6 POPE H12B
4 POPE C12 5 POPE H12A
7 POPE C11 14 POPE O12
7 POPE C11 9 POPE H11B
7 POPE C11 8 POPE H11A
10 POPE P 14 POPE O12
10 POPE P 11 POPE O13
10 POPE P 13 POPE O11
10 POPE P 12 POPE O14
13 POPE O11 15 POPE C1
15 POPE C1 17 POPE HB
15 POPE C1 18 POPE C2
15 POPE C1 16 POPE HA
18 POPE C2 19 POPE HS
18 POPE C2 20 POPE O21
18 POPE C2 26 POPE C3
20 POPE O21 21 POPE C21
21 POPE C21 22 POPE O22
21 POPE C21 23 POPE C22
23 POPE C22 24 POPE H2R
23 POPE C22 25 POPE H2S
23 POPE C22 35 POPE C23
26 POPE C3 29 POPE O31
26 POPE C3 27 POPE HX
26 POPE C3 28 POPE HY
29 POPE O31 30 POPE C31
30 POPE C31 31 POPE O32
30 POPE C31 32 POPE C32
32 POPE C32 33 POPE H2X
32 POPE C32 82 POPE C33
32 POPE C32 34 POPE H2Y
35 POPE C23 37 POPE H3S
35 POPE C23 38 POPE C24
35 POPE C23 36 POPE H3R
38 POPE C24 39 POPE H4R
38 POPE C24 41 POPE C25
38 POPE C24 40 POPE H4S
41 POPE C25 43 POPE H5S
41 POPE C25 42 POPE H5R
41 POPE C25 44 POPE C26
44 POPE C26 45 POPE H6R
44 POPE C26 46 POPE H6S
44 POPE C26 47 POPE C27
47 POPE C27 50 POPE C28
47 POPE C27 48 POPE H7R
47 POPE C27 49 POPE H7S
50 POPE C28 51 POPE H8R
50 POPE C28 52 POPE H8S
50 POPE C28 53 POPE C29
53 POPE C29 54 POPE H91
53 POPE C29 55 POPE C210
55 POPE C210 56 POPE H101
55 POPE C210 57 POPE C211
57 POPE C211 58 POPE H11R
57 POPE C211 59 POPE H11S
57 POPE C211 60 POPE C212
60 POPE C212 62 POPE H12S
60 POPE C212 61 POPE H12R
60 POPE C212 63 POPE C213
63 POPE C213 65 POPE H13S
63 POPE C213 64 POPE H13R
63 POPE C213 66 POPE C214
66 POPE C214 67 POPE H14R
66 POPE C214 68 POPE H14S
66 POPE C214 69 POPE C215
69 POPE C215 71 POPE H15S
69 POPE C215 70 POPE H15R
69 POPE C215 72 POPE C216
72 POPE C216 75 POPE C217
72 POPE C216 74 POPE H16S
72 POPE C216 73 POPE H16R
75 POPE C217 76 POPE H17R
75 POPE C217 78 POPE C218
75 POPE C217 77 POPE H17S
78 POPE C218 79 POPE H18R
78 POPE C218 80 POPE H18S
78 POPE C218 81 POPE H18T
82 POPE C33 84 POPE H3Y
82 POPE C33 85 POPE C34
82 POPE C33 83 POPE H3X
85 POPE C34 86 POPE H4X
85 POPE C34 87 POPE H4Y
85 POPE C34 88 POPE C35
88 POPE C35 89 POPE H5X
88 POPE C35 91 POPE C36
88 POPE C35 90 POPE H5Y
91 POPE C36 92 POPE H6X
91 POPE C36 94 POPE C37
91 POPE C36 93 POPE H6Y
94 POPE C37 95 POPE H7X
94 POPE C37 96 POPE H7Y
94 POPE C37 97 POPE C38
97 POPE C38 100 POPE C39
97 POPE C38 99 POPE H8Y
97 POPE C38 98 POPE H8X
100 POPE C39 101 POPE H9X
100 POPE C39 103 POPE C310
100 POPE C39 102 POPE H9Y
103 POPE C310 104 POPE H10X
103 POPE C310 106 POPE C311
103 POPE C310 105 POPE H10Y
106 POPE C311 109 POPE C312
106 POPE C311 108 POPE H11Y
106 POPE C311 107 POPE H11X
109 POPE C312 110 POPE H12X
109 POPE C312 111 POPE H12Y
109 POPE C312 112 POPE C313
112 POPE C313 115 POPE C314
112 POPE C313 113 POPE H13X
112 POPE C313 114 POPE H13Y
115 POPE C314 117 POPE H14Y
115 POPE C314 116 POPE H14X
115 POPE C314 118 POPE C315
118 POPE C315 121 POPE C316
118 POPE C315 119 POPE H15X
118 POPE C315 120 POPE H15Y
121 POPE C316 124 POPE H16Z
121 POPE C316 123 POPE H16Y
121 POPE C316 122 POPE H16X";

        string2topology(string_representation)
    }

    fn pope_order_bonds() -> HashSet<BondTopology> {
        let string_representation = "4 POPE C12 5 POPE H12A
4 POPE C12 6 POPE H12B
7 POPE C11 8 POPE H11A
7 POPE C11 9 POPE H11B
15 POPE C1 17 POPE HB
15 POPE C1 16 POPE HA
18 POPE C2 19 POPE HS
23 POPE C22 24 POPE H2R
23 POPE C22 25 POPE H2S
26 POPE C3 27 POPE HX
26 POPE C3 28 POPE HY
32 POPE C32 33 POPE H2X
32 POPE C32 34 POPE H2Y
35 POPE C23 36 POPE H3R
35 POPE C23 37 POPE H3S
38 POPE C24 39 POPE H4R
38 POPE C24 40 POPE H4S
41 POPE C25 43 POPE H5S
41 POPE C25 42 POPE H5R
44 POPE C26 45 POPE H6R
44 POPE C26 46 POPE H6S
47 POPE C27 49 POPE H7S
47 POPE C27 48 POPE H7R
50 POPE C28 52 POPE H8S
50 POPE C28 51 POPE H8R
53 POPE C29 54 POPE H91
55 POPE C210 56 POPE H101
57 POPE C211 58 POPE H11R
57 POPE C211 59 POPE H11S
60 POPE C212 62 POPE H12S
60 POPE C212 61 POPE H12R
63 POPE C213 65 POPE H13S
63 POPE C213 64 POPE H13R
66 POPE C214 67 POPE H14R
66 POPE C214 68 POPE H14S
69 POPE C215 70 POPE H15R
69 POPE C215 71 POPE H15S
72 POPE C216 74 POPE H16S
72 POPE C216 73 POPE H16R
75 POPE C217 77 POPE H17S
75 POPE C217 76 POPE H17R
78 POPE C218 81 POPE H18T
78 POPE C218 80 POPE H18S
78 POPE C218 79 POPE H18R
82 POPE C33 83 POPE H3X
82 POPE C33 84 POPE H3Y
85 POPE C34 86 POPE H4X
85 POPE C34 87 POPE H4Y
88 POPE C35 90 POPE H5Y
88 POPE C35 89 POPE H5X
91 POPE C36 93 POPE H6Y
91 POPE C36 92 POPE H6X
94 POPE C37 95 POPE H7X
94 POPE C37 96 POPE H7Y
97 POPE C38 99 POPE H8Y
97 POPE C38 98 POPE H8X
100 POPE C39 101 POPE H9X
100 POPE C39 102 POPE H9Y
103 POPE C310 105 POPE H10Y
103 POPE C310 104 POPE H10X
106 POPE C311 108 POPE H11Y
106 POPE C311 107 POPE H11X
109 POPE C312 111 POPE H12Y
109 POPE C312 110 POPE H12X
112 POPE C313 113 POPE H13X
112 POPE C313 114 POPE H13Y
115 POPE C314 117 POPE H14Y
115 POPE C314 116 POPE H14X
118 POPE C315 120 POPE H15Y
118 POPE C315 119 POPE H15X
121 POPE C316 123 POPE H16Y
121 POPE C316 122 POPE H16X
121 POPE C316 124 POPE H16Z";

        string2bonds(string_representation)
    }

    fn pope_order_atoms() -> OrderAtoms {
        let string_representation = "4 POPE C12
7 POPE C11
15 POPE C1
18 POPE C2
21 POPE C21
23 POPE C22
26 POPE C3
30 POPE C31
32 POPE C32
35 POPE C23
38 POPE C24
41 POPE C25
44 POPE C26
47 POPE C27
50 POPE C28
53 POPE C29
55 POPE C210
57 POPE C211
60 POPE C212
63 POPE C213
66 POPE C214
69 POPE C215
72 POPE C216
75 POPE C217
78 POPE C218
82 POPE C33
85 POPE C34
88 POPE C35
91 POPE C36
94 POPE C37
97 POPE C38
100 POPE C39
103 POPE C310
106 POPE C311
109 POPE C312
112 POPE C313
115 POPE C314
118 POPE C315
121 POPE C316";

        string2atoms(string_representation)
    }

    fn popc_topology() -> MoleculeTopology {
        let string_representation = "0 POPC N 4 POPC C13
0 POPC N 8 POPC C14
0 POPC N 12 POPC C15
0 POPC N 1 POPC C12
1 POPC C12 2 POPC H12A
1 POPC C12 3 POPC H12B
1 POPC C12 16 POPC C11
4 POPC C13 6 POPC H13B
4 POPC C13 7 POPC H13C
4 POPC C13 5 POPC H13A
8 POPC C14 11 POPC H14C
8 POPC C14 9 POPC H14A
8 POPC C14 10 POPC H14B
12 POPC C15 15 POPC H15C
12 POPC C15 13 POPC H15A
12 POPC C15 14 POPC H15B
16 POPC C11 18 POPC H11B
16 POPC C11 17 POPC H11A
16 POPC C11 22 POPC O12
19 POPC P 23 POPC O11
19 POPC P 22 POPC O12
19 POPC P 20 POPC O13
19 POPC P 21 POPC O14
23 POPC O11 24 POPC C1
24 POPC C1 27 POPC C2
24 POPC C1 25 POPC HA
24 POPC C1 26 POPC HB
27 POPC C2 35 POPC C3
27 POPC C2 29 POPC O21
27 POPC C2 28 POPC HS
29 POPC O21 30 POPC C21
30 POPC C21 32 POPC C22
30 POPC C21 31 POPC O22
32 POPC C22 34 POPC H2S
32 POPC C22 44 POPC C23
32 POPC C22 33 POPC H2R
35 POPC C3 36 POPC HX
35 POPC C3 38 POPC O31
35 POPC C3 37 POPC HY
38 POPC O31 39 POPC C31
39 POPC C31 40 POPC O32
39 POPC C31 41 POPC C32
41 POPC C32 91 POPC C33
41 POPC C32 43 POPC H2Y
41 POPC C32 42 POPC H2X
44 POPC C23 47 POPC C24
44 POPC C23 46 POPC H3S
44 POPC C23 45 POPC H3R
47 POPC C24 50 POPC C25
47 POPC C24 48 POPC H4R
47 POPC C24 49 POPC H4S
50 POPC C25 52 POPC H5S
50 POPC C25 51 POPC H5R
50 POPC C25 53 POPC C26
53 POPC C26 54 POPC H6R
53 POPC C26 55 POPC H6S
53 POPC C26 56 POPC C27
56 POPC C27 59 POPC C28
56 POPC C27 58 POPC H7S
56 POPC C27 57 POPC H7R
59 POPC C28 62 POPC C29
59 POPC C28 60 POPC H8R
59 POPC C28 61 POPC H8S
62 POPC C29 64 POPC C210
62 POPC C29 63 POPC H91
64 POPC C210 65 POPC H101
64 POPC C210 66 POPC C211
66 POPC C211 67 POPC H11R
66 POPC C211 68 POPC H11S
66 POPC C211 69 POPC C212
69 POPC C212 70 POPC H12R
69 POPC C212 71 POPC H12S
69 POPC C212 72 POPC C213
72 POPC C213 75 POPC C214
72 POPC C213 73 POPC H13R
72 POPC C213 74 POPC H13S
75 POPC C214 78 POPC C215
75 POPC C214 76 POPC H14R
75 POPC C214 77 POPC H14S
78 POPC C215 81 POPC C216
78 POPC C215 80 POPC H15S
78 POPC C215 79 POPC H15R
81 POPC C216 84 POPC C217
81 POPC C216 82 POPC H16R
81 POPC C216 83 POPC H16S
84 POPC C217 87 POPC C218
84 POPC C217 85 POPC H17R
84 POPC C217 86 POPC H17S
87 POPC C218 89 POPC H18S
87 POPC C218 90 POPC H18T
87 POPC C218 88 POPC H18R
91 POPC C33 94 POPC C34
91 POPC C33 92 POPC H3X
91 POPC C33 93 POPC H3Y
94 POPC C34 95 POPC H4X
94 POPC C34 97 POPC C35
94 POPC C34 96 POPC H4Y
97 POPC C35 99 POPC H5Y
97 POPC C35 100 POPC C36
97 POPC C35 98 POPC H5X
100 POPC C36 103 POPC C37
100 POPC C36 102 POPC H6Y
100 POPC C36 101 POPC H6X
103 POPC C37 106 POPC C38
103 POPC C37 105 POPC H7Y
103 POPC C37 104 POPC H7X
106 POPC C38 107 POPC H8X
106 POPC C38 109 POPC C39
106 POPC C38 108 POPC H8Y
109 POPC C39 110 POPC H9X
109 POPC C39 111 POPC H9Y
109 POPC C39 112 POPC C310
112 POPC C310 115 POPC C311
112 POPC C310 113 POPC H10X
112 POPC C310 114 POPC H10Y
115 POPC C311 117 POPC H11Y
115 POPC C311 118 POPC C312
115 POPC C311 116 POPC H11X
118 POPC C312 121 POPC C313
118 POPC C312 119 POPC H12X
118 POPC C312 120 POPC H12Y
121 POPC C313 124 POPC C314
121 POPC C313 122 POPC H13X
121 POPC C313 123 POPC H13Y
124 POPC C314 127 POPC C315
124 POPC C314 126 POPC H14Y
124 POPC C314 125 POPC H14X
127 POPC C315 128 POPC H15X
127 POPC C315 129 POPC H15Y
127 POPC C315 130 POPC C316
130 POPC C316 131 POPC H16X
130 POPC C316 133 POPC H16Z
130 POPC C316 132 POPC H16Y";

        string2topology(string_representation)
    }

    fn popc_order_bonds() -> HashSet<BondTopology> {
        let string_representation = "1 POPC C12 3 POPC H12B
1 POPC C12 2 POPC H12A
4 POPC C13 5 POPC H13A
4 POPC C13 7 POPC H13C
4 POPC C13 6 POPC H13B
8 POPC C14 10 POPC H14B
8 POPC C14 11 POPC H14C
8 POPC C14 9 POPC H14A
12 POPC C15 14 POPC H15B
12 POPC C15 13 POPC H15A
12 POPC C15 15 POPC H15C
16 POPC C11 18 POPC H11B
16 POPC C11 17 POPC H11A
24 POPC C1 26 POPC HB
24 POPC C1 25 POPC HA
27 POPC C2 28 POPC HS
32 POPC C22 33 POPC H2R
32 POPC C22 34 POPC H2S
35 POPC C3 37 POPC HY
35 POPC C3 36 POPC HX
41 POPC C32 42 POPC H2X
41 POPC C32 43 POPC H2Y
44 POPC C23 46 POPC H3S
44 POPC C23 45 POPC H3R
47 POPC C24 49 POPC H4S
47 POPC C24 48 POPC H4R
50 POPC C25 51 POPC H5R
50 POPC C25 52 POPC H5S
53 POPC C26 55 POPC H6S
53 POPC C26 54 POPC H6R
56 POPC C27 57 POPC H7R
56 POPC C27 58 POPC H7S
59 POPC C28 60 POPC H8R
59 POPC C28 61 POPC H8S
62 POPC C29 63 POPC H91
64 POPC C210 65 POPC H101
66 POPC C211 67 POPC H11R
66 POPC C211 68 POPC H11S
69 POPC C212 71 POPC H12S
69 POPC C212 70 POPC H12R
72 POPC C213 74 POPC H13S
72 POPC C213 73 POPC H13R
75 POPC C214 76 POPC H14R
75 POPC C214 77 POPC H14S
78 POPC C215 80 POPC H15S
78 POPC C215 79 POPC H15R
81 POPC C216 83 POPC H16S
81 POPC C216 82 POPC H16R
84 POPC C217 85 POPC H17R
84 POPC C217 86 POPC H17S
87 POPC C218 90 POPC H18T
87 POPC C218 89 POPC H18S
87 POPC C218 88 POPC H18R
91 POPC C33 93 POPC H3Y
91 POPC C33 92 POPC H3X
94 POPC C34 95 POPC H4X
94 POPC C34 96 POPC H4Y
97 POPC C35 98 POPC H5X
97 POPC C35 99 POPC H5Y
100 POPC C36 101 POPC H6X
100 POPC C36 102 POPC H6Y
103 POPC C37 104 POPC H7X
103 POPC C37 105 POPC H7Y
106 POPC C38 107 POPC H8X
106 POPC C38 108 POPC H8Y
109 POPC C39 110 POPC H9X
109 POPC C39 111 POPC H9Y
112 POPC C310 114 POPC H10Y
112 POPC C310 113 POPC H10X
115 POPC C311 116 POPC H11X
115 POPC C311 117 POPC H11Y
118 POPC C312 120 POPC H12Y
118 POPC C312 119 POPC H12X
121 POPC C313 122 POPC H13X
121 POPC C313 123 POPC H13Y
124 POPC C314 125 POPC H14X
124 POPC C314 126 POPC H14Y
127 POPC C315 128 POPC H15X
127 POPC C315 129 POPC H15Y
130 POPC C316 133 POPC H16Z
130 POPC C316 131 POPC H16X
130 POPC C316 132 POPC H16Y";

        string2bonds(string_representation)
    }

    fn popc_order_atoms() -> OrderAtoms {
        let string_representation = "1 POPC C12
4 POPC C13
8 POPC C14
12 POPC C15
16 POPC C11
24 POPC C1
27 POPC C2
30 POPC C21
32 POPC C22
35 POPC C3
39 POPC C31
41 POPC C32
44 POPC C23
47 POPC C24
50 POPC C25
53 POPC C26
56 POPC C27
59 POPC C28
62 POPC C29
64 POPC C210
66 POPC C211
69 POPC C212
72 POPC C213
75 POPC C214
78 POPC C215
81 POPC C216
84 POPC C217
87 POPC C218
91 POPC C33
94 POPC C34
97 POPC C35
100 POPC C36
103 POPC C37
106 POPC C38
109 POPC C39
112 POPC C310
115 POPC C311
118 POPC C312
121 POPC C313
124 POPC C314
127 POPC C315
130 POPC C316";

        string2atoms(string_representation)
    }

    fn popg_topology() -> MoleculeTopology {
        let string_representation = "0 POPG C13 3 POPG OC3
0 POPG C13 5 POPG C12
0 POPG C13 2 POPG H13B
0 POPG C13 1 POPG H13A
3 POPG OC3 4 POPG HO3
5 POPG C12 6 POPG H12A
5 POPG C12 7 POPG OC2
5 POPG C12 9 POPG C11
7 POPG OC2 8 POPG HO2
9 POPG C11 11 POPG H11B
9 POPG C11 10 POPG H11A
9 POPG C11 15 POPG O12
12 POPG P 14 POPG O14
12 POPG P 16 POPG O11
12 POPG P 13 POPG O13
12 POPG P 15 POPG O12
16 POPG O11 17 POPG C1
17 POPG C1 20 POPG C2
17 POPG C1 19 POPG HB
17 POPG C1 18 POPG HA
20 POPG C2 22 POPG O21
20 POPG C2 21 POPG HS
20 POPG C2 28 POPG C3
22 POPG O21 23 POPG C21
23 POPG C21 24 POPG O22
23 POPG C21 25 POPG C22
25 POPG C22 37 POPG C23
25 POPG C22 26 POPG H2R
25 POPG C22 27 POPG H2S
28 POPG C3 29 POPG HX
28 POPG C3 31 POPG O31
28 POPG C3 30 POPG HY
31 POPG O31 32 POPG C31
32 POPG C31 33 POPG O32
32 POPG C31 34 POPG C32
34 POPG C32 84 POPG C33
34 POPG C32 36 POPG H2Y
34 POPG C32 35 POPG H2X
37 POPG C23 39 POPG H3S
37 POPG C23 40 POPG C24
37 POPG C23 38 POPG H3R
40 POPG C24 43 POPG C25
40 POPG C24 41 POPG H4R
40 POPG C24 42 POPG H4S
43 POPG C25 46 POPG C26
43 POPG C25 45 POPG H5S
43 POPG C25 44 POPG H5R
46 POPG C26 49 POPG C27
46 POPG C26 47 POPG H6R
46 POPG C26 48 POPG H6S
49 POPG C27 52 POPG C28
49 POPG C27 51 POPG H7S
49 POPG C27 50 POPG H7R
52 POPG C28 54 POPG H8S
52 POPG C28 53 POPG H8R
52 POPG C28 55 POPG C29
55 POPG C29 56 POPG H91
55 POPG C29 57 POPG C210
57 POPG C210 59 POPG C211
57 POPG C210 58 POPG H101
59 POPG C211 62 POPG C212
59 POPG C211 61 POPG H11S
59 POPG C211 60 POPG H11R
62 POPG C212 63 POPG H12R
62 POPG C212 64 POPG H12S
62 POPG C212 65 POPG C213
65 POPG C213 67 POPG H13S
65 POPG C213 68 POPG C214
65 POPG C213 66 POPG H13R
68 POPG C214 69 POPG H14R
68 POPG C214 70 POPG H14S
68 POPG C214 71 POPG C215
71 POPG C215 72 POPG H15R
71 POPG C215 74 POPG C216
71 POPG C215 73 POPG H15S
74 POPG C216 77 POPG C217
74 POPG C216 75 POPG H16R
74 POPG C216 76 POPG H16S
77 POPG C217 79 POPG H17S
77 POPG C217 80 POPG C218
77 POPG C217 78 POPG H17R
80 POPG C218 82 POPG H18S
80 POPG C218 81 POPG H18R
80 POPG C218 83 POPG H18T
84 POPG C33 85 POPG H3X
84 POPG C33 86 POPG H3Y
84 POPG C33 87 POPG C34
87 POPG C34 88 POPG H4X
87 POPG C34 90 POPG C35
87 POPG C34 89 POPG H4Y
90 POPG C35 93 POPG C36
90 POPG C35 92 POPG H5Y
90 POPG C35 91 POPG H5X
93 POPG C36 96 POPG C37
93 POPG C36 95 POPG H6Y
93 POPG C36 94 POPG H6X
96 POPG C37 97 POPG H7X
96 POPG C37 98 POPG H7Y
96 POPG C37 99 POPG C38
99 POPG C38 101 POPG H8Y
99 POPG C38 102 POPG C39
99 POPG C38 100 POPG H8X
102 POPG C39 105 POPG C310
102 POPG C39 104 POPG H9Y
102 POPG C39 103 POPG H9X
105 POPG C310 108 POPG C311
105 POPG C310 107 POPG H10Y
105 POPG C310 106 POPG H10X
108 POPG C311 110 POPG H11Y
108 POPG C311 109 POPG H11X
108 POPG C311 111 POPG C312
111 POPG C312 113 POPG H12Y
111 POPG C312 112 POPG H12X
111 POPG C312 114 POPG C313
114 POPG C313 116 POPG H13Y
114 POPG C313 115 POPG H13X
114 POPG C313 117 POPG C314
117 POPG C314 118 POPG H14X
117 POPG C314 120 POPG C315
117 POPG C314 119 POPG H14Y
120 POPG C315 123 POPG C316
120 POPG C315 122 POPG H15Y
120 POPG C315 121 POPG H15X
123 POPG C316 124 POPG H16X
123 POPG C316 125 POPG H16Y
123 POPG C316 126 POPG H16Z";

        string2topology(string_representation)
    }

    fn popg_order_bonds() -> HashSet<BondTopology> {
        let string_representation = "0 POPG C13 2 POPG H13B
0 POPG C13 1 POPG H13A
5 POPG C12 6 POPG H12A
9 POPG C11 10 POPG H11A
9 POPG C11 11 POPG H11B
17 POPG C1 19 POPG HB
17 POPG C1 18 POPG HA
20 POPG C2 21 POPG HS
25 POPG C22 27 POPG H2S
25 POPG C22 26 POPG H2R
28 POPG C3 29 POPG HX
28 POPG C3 30 POPG HY
34 POPG C32 35 POPG H2X
34 POPG C32 36 POPG H2Y
37 POPG C23 39 POPG H3S
37 POPG C23 38 POPG H3R
40 POPG C24 41 POPG H4R
40 POPG C24 42 POPG H4S
43 POPG C25 45 POPG H5S
43 POPG C25 44 POPG H5R
46 POPG C26 48 POPG H6S
46 POPG C26 47 POPG H6R
49 POPG C27 50 POPG H7R
49 POPG C27 51 POPG H7S
52 POPG C28 54 POPG H8S
52 POPG C28 53 POPG H8R
55 POPG C29 56 POPG H91
57 POPG C210 58 POPG H101
59 POPG C211 60 POPG H11R
59 POPG C211 61 POPG H11S
62 POPG C212 63 POPG H12R
62 POPG C212 64 POPG H12S
65 POPG C213 66 POPG H13R
65 POPG C213 67 POPG H13S
68 POPG C214 69 POPG H14R
68 POPG C214 70 POPG H14S
71 POPG C215 73 POPG H15S
71 POPG C215 72 POPG H15R
74 POPG C216 75 POPG H16R
74 POPG C216 76 POPG H16S
77 POPG C217 79 POPG H17S
77 POPG C217 78 POPG H17R
80 POPG C218 83 POPG H18T
80 POPG C218 82 POPG H18S
80 POPG C218 81 POPG H18R
84 POPG C33 86 POPG H3Y
84 POPG C33 85 POPG H3X
87 POPG C34 88 POPG H4X
87 POPG C34 89 POPG H4Y
90 POPG C35 91 POPG H5X
90 POPG C35 92 POPG H5Y
93 POPG C36 94 POPG H6X
93 POPG C36 95 POPG H6Y
96 POPG C37 97 POPG H7X
96 POPG C37 98 POPG H7Y
99 POPG C38 100 POPG H8X
99 POPG C38 101 POPG H8Y
102 POPG C39 103 POPG H9X
102 POPG C39 104 POPG H9Y
105 POPG C310 106 POPG H10X
105 POPG C310 107 POPG H10Y
108 POPG C311 109 POPG H11X
108 POPG C311 110 POPG H11Y
111 POPG C312 113 POPG H12Y
111 POPG C312 112 POPG H12X
114 POPG C313 115 POPG H13X
114 POPG C313 116 POPG H13Y
117 POPG C314 119 POPG H14Y
117 POPG C314 118 POPG H14X
120 POPG C315 121 POPG H15X
120 POPG C315 122 POPG H15Y
123 POPG C316 124 POPG H16X
123 POPG C316 126 POPG H16Z
123 POPG C316 125 POPG H16Y";

        string2bonds(string_representation)
    }

    fn popg_order_atoms() -> OrderAtoms {
        let string_representation = "0 POPG C13
5 POPG C12
9 POPG C11
17 POPG C1
20 POPG C2
23 POPG C21
25 POPG C22
28 POPG C3
32 POPG C31
34 POPG C32
37 POPG C23
40 POPG C24
43 POPG C25
46 POPG C26
49 POPG C27
52 POPG C28
55 POPG C29
57 POPG C210
59 POPG C211
62 POPG C212
65 POPG C213
68 POPG C214
71 POPG C215
74 POPG C216
77 POPG C217
80 POPG C218
84 POPG C33
87 POPG C34
90 POPG C35
93 POPG C36
96 POPG C37
99 POPG C38
102 POPG C39
105 POPG C310
108 POPG C311
111 POPG C312
114 POPG C313
117 POPG C314
120 POPG C315
123 POPG C316";

        string2atoms(string_representation)
    }

    fn string2topology(string: &str) -> MoleculeTopology {
        let bonds = string2bonds(string);
        MoleculeTopology::new_raw(bonds)
    }

    fn string2bonds(string: &str) -> HashSet<BondTopology> {
        let string_representation = string.trim().split("\n");

        let mut bonds = HashSet::new();
        for bond in string_representation.into_iter() {
            let split: Vec<&str> = bond.split_whitespace().collect();
            let index1 = split[0].parse::<usize>().unwrap();
            let resname1 = split[1];
            let atomname1 = split[2];

            let index2 = split[3].parse::<usize>().unwrap();
            let resname2 = split[4];
            let atomname2 = split[5];

            let atom1 = AtomType::new_raw(index1, resname1, atomname1);
            let atom2 = AtomType::new_raw(index2, resname2, atomname2);

            let bond = BondTopology::new_from_types(atom1, atom2);
            bonds.insert(bond);
        }

        bonds
    }

    fn string2atoms(string: &str) -> OrderAtoms {
        let string_representation = string.trim().split("\n");

        let mut atoms = Vec::new();
        for atom in string_representation.into_iter() {
            let split: Vec<&str> = atom.split_whitespace().collect();
            let relative_index = split[0].parse::<usize>().unwrap();
            let residue_name = split[1];
            let atom_name = split[2];

            atoms.push(AtomType::new_raw(relative_index, residue_name, atom_name));
        }

        OrderAtoms::new_raw(atoms)
    }

    #[test]
    fn test_classify_molecules() {
        let mut system = System::from_file("tests/files/pcpepg.tpr").unwrap();

        create_group(
            &mut system,
            "HeavyAtoms",
            "@membrane and element name carbon",
        )
        .unwrap();
        create_group(
            &mut system,
            "Hydrogens",
            "@membrane and element name hydrogen",
        )
        .unwrap();

        create_group(&mut system, "Heads", "name P").unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::local("@membrane", "name P", 2.5))
            .build()
            .unwrap();

        let molecules =
            match MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap()
            {
                MoleculeTypes::BondBased(x) => x,
                _ => panic!("MoleculeTypes should be bond-based."),
            };
        let expected_names = ["POPE", "POPC", "POPG"];
        let expected_topology = [pope_topology(), popc_topology(), popg_topology()];
        let expected_order_bond_topologies =
            [pope_order_bonds(), popc_order_bonds(), popg_order_bonds()];
        let expected_n_instances = [131, 128, 15];
        let expected_bond_instances = [
            vec![
                (7, 8),
                (132, 133),
                (257, 258),
                (382, 383),
                (507, 508),
                (632, 633),
                (757, 758),
                (882, 883),
                (1007, 1008),
                (1132, 1133),
                (1257, 1258),
                (1382, 1383),
                (1507, 1508),
                (1632, 1633),
                (1757, 1758),
                (1882, 1883),
                (2007, 2008),
                (2132, 2133),
                (2257, 2258),
                (2382, 2383),
                (2507, 2508),
                (2632, 2633),
                (2757, 2758),
                (2882, 2883),
                (3007, 3008),
                (3132, 3133),
                (3257, 3258),
                (3382, 3383),
                (3507, 3508),
                (3632, 3633),
                (3757, 3758),
                (3882, 3883),
                (4007, 4008),
                (4132, 4133),
                (4257, 4258),
                (4382, 4383),
                (4507, 4508),
                (4632, 4633),
                (4757, 4758),
                (4882, 4883),
                (5007, 5008),
                (5132, 5133),
                (5257, 5258),
                (5382, 5383),
                (5507, 5508),
                (5632, 5633),
                (5757, 5758),
                (5882, 5883),
                (6007, 6008),
                (6132, 6133),
                (6257, 6258),
                (6382, 6383),
                (6507, 6508),
                (6632, 6633),
                (6757, 6758),
                (6882, 6883),
                (7007, 7008),
                (7132, 7133),
                (7257, 7258),
                (7382, 7383),
                (7507, 7508),
                (7632, 7633),
                (7757, 7758),
                (7882, 7883),
                (8007, 8008),
                (8132, 8133),
                (8257, 8258),
                (8382, 8383),
                (8507, 8508),
                (8632, 8633),
                (8757, 8758),
                (8882, 8883),
                (9007, 9008),
                (9132, 9133),
                (9257, 9258),
                (9382, 9383),
                (9507, 9508),
                (9632, 9633),
                (9757, 9758),
                (9882, 9883),
                (10007, 10008),
                (10132, 10133),
                (10257, 10258),
                (10382, 10383),
                (10507, 10508),
                (10632, 10633),
                (10757, 10758),
                (10882, 10883),
                (11007, 11008),
                (11132, 11133),
                (11257, 11258),
                (11382, 11383),
                (11507, 11508),
                (11632, 11633),
                (11757, 11758),
                (11882, 11883),
                (12007, 12008),
                (12132, 12133),
                (12257, 12258),
                (12382, 12383),
                (12507, 12508),
                (12632, 12633),
                (12757, 12758),
                (12882, 12883),
                (13007, 13008),
                (13132, 13133),
                (13257, 13258),
                (13382, 13383),
                (13507, 13508),
                (13632, 13633),
                (13757, 13758),
                (13882, 13883),
                (14007, 14008),
                (14132, 14133),
                (14257, 14258),
                (14382, 14383),
                (14507, 14508),
                (14632, 14633),
                (14757, 14758),
                (14882, 14883),
                (15007, 15008),
                (15132, 15133),
                (15257, 15258),
                (15382, 15383),
                (15507, 15508),
                (15632, 15633),
                (15757, 15758),
                (15882, 15883),
                (16007, 16008),
                (16132, 16133),
                (16257, 16258),
            ],
            vec![
                (16422, 16424),
                (16556, 16558),
                (16690, 16692),
                (16824, 16826),
                (16958, 16960),
                (17092, 17094),
                (17226, 17228),
                (17360, 17362),
                (17494, 17496),
                (17628, 17630),
                (17762, 17764),
                (17896, 17898),
                (18030, 18032),
                (18164, 18166),
                (18298, 18300),
                (18432, 18434),
                (18566, 18568),
                (18700, 18702),
                (18834, 18836),
                (18968, 18970),
                (19102, 19104),
                (19236, 19238),
                (19370, 19372),
                (19504, 19506),
                (19638, 19640),
                (19772, 19774),
                (19906, 19908),
                (20040, 20042),
                (20174, 20176),
                (20308, 20310),
                (20442, 20444),
                (20576, 20578),
                (20710, 20712),
                (20844, 20846),
                (20978, 20980),
                (21112, 21114),
                (21246, 21248),
                (21380, 21382),
                (21514, 21516),
                (21648, 21650),
                (21782, 21784),
                (21916, 21918),
                (22050, 22052),
                (22184, 22186),
                (22318, 22320),
                (22452, 22454),
                (22586, 22588),
                (22720, 22722),
                (22854, 22856),
                (22988, 22990),
                (23122, 23124),
                (23256, 23258),
                (23390, 23392),
                (23524, 23526),
                (23658, 23660),
                (23792, 23794),
                (23926, 23928),
                (24060, 24062),
                (24194, 24196),
                (24328, 24330),
                (24462, 24464),
                (24596, 24598),
                (24730, 24732),
                (24864, 24866),
                (24998, 25000),
                (25132, 25134),
                (25266, 25268),
                (25400, 25402),
                (25534, 25536),
                (25668, 25670),
                (25802, 25804),
                (25936, 25938),
                (26070, 26072),
                (26204, 26206),
                (26338, 26340),
                (26472, 26474),
                (26606, 26608),
                (26740, 26742),
                (26874, 26876),
                (27008, 27010),
                (27142, 27144),
                (27276, 27278),
                (27410, 27412),
                (27544, 27546),
                (27678, 27680),
                (27812, 27814),
                (27946, 27948),
                (28080, 28082),
                (28214, 28216),
                (28348, 28350),
                (28482, 28484),
                (28616, 28618),
                (28750, 28752),
                (28884, 28886),
                (29018, 29020),
                (29152, 29154),
                (29286, 29288),
                (29420, 29422),
                (29554, 29556),
                (29688, 29690),
                (29822, 29824),
                (29956, 29958),
                (30090, 30092),
                (30224, 30226),
                (30358, 30360),
                (30492, 30494),
                (30626, 30628),
                (30760, 30762),
                (30894, 30896),
                (31028, 31030),
                (31162, 31164),
                (31296, 31298),
                (31430, 31432),
                (31564, 31566),
                (31698, 31700),
                (31832, 31834),
                (31966, 31968),
                (32100, 32102),
                (32234, 32236),
                (32368, 32370),
                (32502, 32504),
                (32636, 32638),
                (32770, 32772),
                (32904, 32906),
                (33038, 33040),
                (33172, 33174),
                (33306, 33308),
                (33440, 33442),
            ],
            vec![
                (33620, 33621),
                (33747, 33748),
                (33874, 33875),
                (34001, 34002),
                (34128, 34129),
                (34255, 34256),
                (34382, 34383),
                (34509, 34510),
                (34636, 34637),
                (34763, 34764),
                (34890, 34891),
                (35017, 35018),
                (35144, 35145),
                (35271, 35272),
                (35398, 35399),
            ],
        ];
        let expected_order_atoms = [pope_order_atoms(), popc_order_atoms(), popg_order_atoms()];
        let expected_heads = [
            vec![
                10, 135, 260, 385, 510, 635, 760, 885, 1010, 1135, 1260, 1385, 1510, 1635, 1760,
                1885, 2010, 2135, 2260, 2385, 2510, 2635, 2760, 2885, 3010, 3135, 3260, 3385, 3510,
                3635, 3760, 3885, 4010, 4135, 4260, 4385, 4510, 4635, 4760, 4885, 5010, 5135, 5260,
                5385, 5510, 5635, 5760, 5885, 6010, 6135, 6260, 6385, 6510, 6635, 6760, 6885, 7010,
                7135, 7260, 7385, 7510, 7635, 7760, 7885, 8010, 8135, 8260, 8385, 8510, 8635, 8760,
                8885, 9010, 9135, 9260, 9385, 9510, 9635, 9760, 9885, 10010, 10135, 10260, 10385,
                10510, 10635, 10760, 10885, 11010, 11135, 11260, 11385, 11510, 11635, 11760, 11885,
                12010, 12135, 12260, 12385, 12510, 12635, 12760, 12885, 13010, 13135, 13260, 13385,
                13510, 13635, 13760, 13885, 14010, 14135, 14260, 14385, 14510, 14635, 14760, 14885,
                15010, 15135, 15260, 15385, 15510, 15635, 15760, 15885, 16010, 16135, 16260,
            ],
            vec![
                16394, 16528, 16662, 16796, 16930, 17064, 17198, 17332, 17466, 17600, 17734, 17868,
                18002, 18136, 18270, 18404, 18538, 18672, 18806, 18940, 19074, 19208, 19342, 19476,
                19610, 19744, 19878, 20012, 20146, 20280, 20414, 20548, 20682, 20816, 20950, 21084,
                21218, 21352, 21486, 21620, 21754, 21888, 22022, 22156, 22290, 22424, 22558, 22692,
                22826, 22960, 23094, 23228, 23362, 23496, 23630, 23764, 23898, 24032, 24166, 24300,
                24434, 24568, 24702, 24836, 24970, 25104, 25238, 25372, 25506, 25640, 25774, 25908,
                26042, 26176, 26310, 26444, 26578, 26712, 26846, 26980, 27114, 27248, 27382, 27516,
                27650, 27784, 27918, 28052, 28186, 28320, 28454, 28588, 28722, 28856, 28990, 29124,
                29258, 29392, 29526, 29660, 29794, 29928, 30062, 30196, 30330, 30464, 30598, 30732,
                30866, 31000, 31134, 31268, 31402, 31536, 31670, 31804, 31938, 32072, 32206, 32340,
                32474, 32608, 32742, 32876, 33010, 33144, 33278, 33412,
            ],
            vec![
                33539, 33666, 33793, 33920, 34047, 34174, 34301, 34428, 34555, 34682, 34809, 34936,
                35063, 35190, 35317,
            ],
        ];
        let relative_indices = [(7, 8), (47, 49), (93, 94)];

        for (i, molecule) in molecules.into_iter().enumerate() {
            assert_eq!(molecule.name(), expected_names[i]);
            assert_eq!(molecule.topology(), &expected_topology[i]);
            let order_bonds = molecule.order_structure();
            let order_bond_topologies: HashSet<BondTopology> = order_bonds
                .bond_types()
                .iter()
                .map(|x| x.bond_topology().clone())
                .collect();

            assert_eq!(order_bond_topologies, expected_order_bond_topologies[i]);

            for (b, bond_instances) in order_bonds
                .bond_types()
                .iter()
                .map(|x| (x, x.bonds().clone()))
            {
                assert_eq!(bond_instances.len(), expected_n_instances[i]);
                if b.atom1_index() == relative_indices[i].0
                    && b.atom2_index() == relative_indices[i].1
                {
                    assert_eq!(bond_instances, expected_bond_instances[i]);
                }
            }

            assert_eq!(molecule.order_atoms(), &expected_order_atoms[i]);

            if let Some(MoleculeLeafletClassification::Local(x, _)) =
                molecule.leaflet_classification()
            {
                assert_eq!(x.heads(), &expected_heads[i]);
                assert_eq!(x.radius(), 2.5);
            } else {
                panic!("Incorrect MoleculeLeafletClassification.")
            }
        }
    }

    fn pope_cg_bonds() -> HashSet<BondTopology> {
        let string_representation = "
0 POPE NH3 1 POPE PO4
1 POPE PO4 2 POPE GL1
2 POPE GL1 4 POPE C1A
2 POPE GL1 3 POPE GL2
3 POPE GL2 8 POPE C1B
4 POPE C1A 5 POPE D2A
5 POPE D2A 6 POPE C3A
6 POPE C3A 7 POPE C4A
8 POPE C1B 9 POPE C2B
9 POPE C2B 10 POPE C3B
10 POPE C3B 11 POPE C4B";

        string2bonds(string_representation)
    }

    fn pope_cg_order_atoms() -> OrderAtoms {
        let string_representation = "
0 POPE NH3
1 POPE PO4
2 POPE GL1
3 POPE GL2
4 POPE C1A
5 POPE D2A
6 POPE C3A
7 POPE C4A
8 POPE C1B
9 POPE C2B
10 POPE C3B
11 POPE C4B";

        string2atoms(string_representation)
    }

    fn popg_cg_bonds() -> HashSet<BondTopology> {
        let string_representation = "
0 POPG GL0 1 POPG PO4
1 POPG PO4 2 POPG GL1
2 POPG GL1 4 POPG C1A
2 POPG GL1 3 POPG GL2
3 POPG GL2 8 POPG C1B
4 POPG C1A 5 POPG D2A
5 POPG D2A 6 POPG C3A
6 POPG C3A 7 POPG C4A
8 POPG C1B 9 POPG C2B
9 POPG C2B 10 POPG C3B
10 POPG C3B 11 POPG C4B";

        string2bonds(string_representation)
    }

    fn popg_cg_order_atoms() -> OrderAtoms {
        let string_representation = "
0 POPG GL0
1 POPG PO4
2 POPG GL1
3 POPG GL2
4 POPG C1A
5 POPG D2A
6 POPG C3A
7 POPG C4A
8 POPG C1B
9 POPG C2B
10 POPG C3B
11 POPG C4B";

        string2atoms(string_representation)
    }

    #[test]
    fn test_classify_molecules_cg() {
        let mut system = System::from_file("tests/files/pepg_cg.tpr").unwrap();

        create_group(&mut system, "Beads", "resname POPE POPG").unwrap();
        create_group(&mut system, "Heads", "name PO4").unwrap();
        create_group(&mut system, "Methyls", "name C4A C4B").unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::individual(
                "name PO4",
                "name C4A C4B",
            ))
            .build()
            .unwrap();

        let molecules =
            match MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap()
            {
                MoleculeTypes::BondBased(x) => x,
                _ => panic!("MoleculeTypes should be bond-based."),
            };
        let expected_names = ["POPE", "POPG"];
        let expected_topology = [
            MoleculeTopology::new_raw(pope_cg_bonds()),
            MoleculeTopology::new_raw(popg_cg_bonds()),
        ];
        let expected_order_bond_topologies = [pope_cg_bonds(), popg_cg_bonds()];
        let expected_n_instances = [216, 72];
        let expected_bond_instances = [
            vec![
                (2, 4),
                (14, 16),
                (26, 28),
                (38, 40),
                (50, 52),
                (62, 64),
                (74, 76),
                (86, 88),
                (98, 100),
                (110, 112),
                (122, 124),
                (134, 136),
                (146, 148),
                (158, 160),
                (170, 172),
                (182, 184),
                (194, 196),
                (206, 208),
                (218, 220),
                (230, 232),
                (242, 244),
                (254, 256),
                (266, 268),
                (278, 280),
                (290, 292),
                (302, 304),
                (314, 316),
                (326, 328),
                (338, 340),
                (350, 352),
                (362, 364),
                (374, 376),
                (386, 388),
                (398, 400),
                (410, 412),
                (422, 424),
                (434, 436),
                (446, 448),
                (458, 460),
                (470, 472),
                (482, 484),
                (494, 496),
                (506, 508),
                (518, 520),
                (530, 532),
                (542, 544),
                (554, 556),
                (566, 568),
                (578, 580),
                (590, 592),
                (602, 604),
                (614, 616),
                (626, 628),
                (638, 640),
                (650, 652),
                (662, 664),
                (674, 676),
                (686, 688),
                (698, 700),
                (710, 712),
                (722, 724),
                (734, 736),
                (746, 748),
                (758, 760),
                (770, 772),
                (782, 784),
                (794, 796),
                (806, 808),
                (818, 820),
                (830, 832),
                (842, 844),
                (854, 856),
                (866, 868),
                (878, 880),
                (890, 892),
                (902, 904),
                (914, 916),
                (926, 928),
                (938, 940),
                (950, 952),
                (962, 964),
                (974, 976),
                (986, 988),
                (998, 1000),
                (1010, 1012),
                (1022, 1024),
                (1034, 1036),
                (1046, 1048),
                (1058, 1060),
                (1070, 1072),
                (1082, 1084),
                (1094, 1096),
                (1106, 1108),
                (1118, 1120),
                (1130, 1132),
                (1142, 1144),
                (1154, 1156),
                (1166, 1168),
                (1178, 1180),
                (1190, 1192),
                (1202, 1204),
                (1214, 1216),
                (1226, 1228),
                (1238, 1240),
                (1250, 1252),
                (1262, 1264),
                (1274, 1276),
                (1286, 1288),
                (1730, 1732),
                (1742, 1744),
                (1754, 1756),
                (1766, 1768),
                (1778, 1780),
                (1790, 1792),
                (1802, 1804),
                (1814, 1816),
                (1826, 1828),
                (1838, 1840),
                (1850, 1852),
                (1862, 1864),
                (1874, 1876),
                (1886, 1888),
                (1898, 1900),
                (1910, 1912),
                (1922, 1924),
                (1934, 1936),
                (1946, 1948),
                (1958, 1960),
                (1970, 1972),
                (1982, 1984),
                (1994, 1996),
                (2006, 2008),
                (2018, 2020),
                (2030, 2032),
                (2042, 2044),
                (2054, 2056),
                (2066, 2068),
                (2078, 2080),
                (2090, 2092),
                (2102, 2104),
                (2114, 2116),
                (2126, 2128),
                (2138, 2140),
                (2150, 2152),
                (2162, 2164),
                (2174, 2176),
                (2186, 2188),
                (2198, 2200),
                (2210, 2212),
                (2222, 2224),
                (2234, 2236),
                (2246, 2248),
                (2258, 2260),
                (2270, 2272),
                (2282, 2284),
                (2294, 2296),
                (2306, 2308),
                (2318, 2320),
                (2330, 2332),
                (2342, 2344),
                (2354, 2356),
                (2366, 2368),
                (2378, 2380),
                (2390, 2392),
                (2402, 2404),
                (2414, 2416),
                (2426, 2428),
                (2438, 2440),
                (2450, 2452),
                (2462, 2464),
                (2474, 2476),
                (2486, 2488),
                (2498, 2500),
                (2510, 2512),
                (2522, 2524),
                (2534, 2536),
                (2546, 2548),
                (2558, 2560),
                (2570, 2572),
                (2582, 2584),
                (2594, 2596),
                (2606, 2608),
                (2618, 2620),
                (2630, 2632),
                (2642, 2644),
                (2654, 2656),
                (2666, 2668),
                (2678, 2680),
                (2690, 2692),
                (2702, 2704),
                (2714, 2716),
                (2726, 2728),
                (2738, 2740),
                (2750, 2752),
                (2762, 2764),
                (2774, 2776),
                (2786, 2788),
                (2798, 2800),
                (2810, 2812),
                (2822, 2824),
                (2834, 2836),
                (2846, 2848),
                (2858, 2860),
                (2870, 2872),
                (2882, 2884),
                (2894, 2896),
                (2906, 2908),
                (2918, 2920),
                (2930, 2932),
                (2942, 2944),
                (2954, 2956),
                (2966, 2968),
                (2978, 2980),
                (2990, 2992),
                (3002, 3004),
                (3014, 3016),
            ],
            vec![
                (1299, 1304),
                (1311, 1316),
                (1323, 1328),
                (1335, 1340),
                (1347, 1352),
                (1359, 1364),
                (1371, 1376),
                (1383, 1388),
                (1395, 1400),
                (1407, 1412),
                (1419, 1424),
                (1431, 1436),
                (1443, 1448),
                (1455, 1460),
                (1467, 1472),
                (1479, 1484),
                (1491, 1496),
                (1503, 1508),
                (1515, 1520),
                (1527, 1532),
                (1539, 1544),
                (1551, 1556),
                (1563, 1568),
                (1575, 1580),
                (1587, 1592),
                (1599, 1604),
                (1611, 1616),
                (1623, 1628),
                (1635, 1640),
                (1647, 1652),
                (1659, 1664),
                (1671, 1676),
                (1683, 1688),
                (1695, 1700),
                (1707, 1712),
                (1719, 1724),
                (3027, 3032),
                (3039, 3044),
                (3051, 3056),
                (3063, 3068),
                (3075, 3080),
                (3087, 3092),
                (3099, 3104),
                (3111, 3116),
                (3123, 3128),
                (3135, 3140),
                (3147, 3152),
                (3159, 3164),
                (3171, 3176),
                (3183, 3188),
                (3195, 3200),
                (3207, 3212),
                (3219, 3224),
                (3231, 3236),
                (3243, 3248),
                (3255, 3260),
                (3267, 3272),
                (3279, 3284),
                (3291, 3296),
                (3303, 3308),
                (3315, 3320),
                (3327, 3332),
                (3339, 3344),
                (3351, 3356),
                (3363, 3368),
                (3375, 3380),
                (3387, 3392),
                (3399, 3404),
                (3411, 3416),
                (3423, 3428),
                (3435, 3440),
                (3447, 3452),
            ],
        ];
        let expected_order_atoms = [pope_cg_order_atoms(), popg_cg_order_atoms()];
        let expected_heads = [
            vec![
                1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121, 133, 145, 157, 169, 181, 193, 205,
                217, 229, 241, 253, 265, 277, 289, 301, 313, 325, 337, 349, 361, 373, 385, 397,
                409, 421, 433, 445, 457, 469, 481, 493, 505, 517, 529, 541, 553, 565, 577, 589,
                601, 613, 625, 637, 649, 661, 673, 685, 697, 709, 721, 733, 745, 757, 769, 781,
                793, 805, 817, 829, 841, 853, 865, 877, 889, 901, 913, 925, 937, 949, 961, 973,
                985, 997, 1009, 1021, 1033, 1045, 1057, 1069, 1081, 1093, 1105, 1117, 1129, 1141,
                1153, 1165, 1177, 1189, 1201, 1213, 1225, 1237, 1249, 1261, 1273, 1285, 1729, 1741,
                1753, 1765, 1777, 1789, 1801, 1813, 1825, 1837, 1849, 1861, 1873, 1885, 1897, 1909,
                1921, 1933, 1945, 1957, 1969, 1981, 1993, 2005, 2017, 2029, 2041, 2053, 2065, 2077,
                2089, 2101, 2113, 2125, 2137, 2149, 2161, 2173, 2185, 2197, 2209, 2221, 2233, 2245,
                2257, 2269, 2281, 2293, 2305, 2317, 2329, 2341, 2353, 2365, 2377, 2389, 2401, 2413,
                2425, 2437, 2449, 2461, 2473, 2485, 2497, 2509, 2521, 2533, 2545, 2557, 2569, 2581,
                2593, 2605, 2617, 2629, 2641, 2653, 2665, 2677, 2689, 2701, 2713, 2725, 2737, 2749,
                2761, 2773, 2785, 2797, 2809, 2821, 2833, 2845, 2857, 2869, 2881, 2893, 2905, 2917,
                2929, 2941, 2953, 2965, 2977, 2989, 3001, 3013,
            ],
            vec![
                1297, 1309, 1321, 1333, 1345, 1357, 1369, 1381, 1393, 1405, 1417, 1429, 1441, 1453,
                1465, 1477, 1489, 1501, 1513, 1525, 1537, 1549, 1561, 1573, 1585, 1597, 1609, 1621,
                1633, 1645, 1657, 1669, 1681, 1693, 1705, 1717, 3025, 3037, 3049, 3061, 3073, 3085,
                3097, 3109, 3121, 3133, 3145, 3157, 3169, 3181, 3193, 3205, 3217, 3229, 3241, 3253,
                3265, 3277, 3289, 3301, 3313, 3325, 3337, 3349, 3361, 3373, 3385, 3397, 3409, 3421,
                3433, 3445,
            ],
        ];
        let expected_methyls = [
            vec![
                [7, 11],
                [19, 23],
                [31, 35],
                [43, 47],
                [55, 59],
                [67, 71],
                [79, 83],
                [91, 95],
                [103, 107],
                [115, 119],
                [127, 131],
                [139, 143],
                [151, 155],
                [163, 167],
                [175, 179],
                [187, 191],
                [199, 203],
                [211, 215],
                [223, 227],
                [235, 239],
                [247, 251],
                [259, 263],
                [271, 275],
                [283, 287],
                [295, 299],
                [307, 311],
                [319, 323],
                [331, 335],
                [343, 347],
                [355, 359],
                [367, 371],
                [379, 383],
                [391, 395],
                [403, 407],
                [415, 419],
                [427, 431],
                [439, 443],
                [451, 455],
                [463, 467],
                [475, 479],
                [487, 491],
                [499, 503],
                [511, 515],
                [523, 527],
                [535, 539],
                [547, 551],
                [559, 563],
                [571, 575],
                [583, 587],
                [595, 599],
                [607, 611],
                [619, 623],
                [631, 635],
                [643, 647],
                [655, 659],
                [667, 671],
                [679, 683],
                [691, 695],
                [703, 707],
                [715, 719],
                [727, 731],
                [739, 743],
                [751, 755],
                [763, 767],
                [775, 779],
                [787, 791],
                [799, 803],
                [811, 815],
                [823, 827],
                [835, 839],
                [847, 851],
                [859, 863],
                [871, 875],
                [883, 887],
                [895, 899],
                [907, 911],
                [919, 923],
                [931, 935],
                [943, 947],
                [955, 959],
                [967, 971],
                [979, 983],
                [991, 995],
                [1003, 1007],
                [1015, 1019],
                [1027, 1031],
                [1039, 1043],
                [1051, 1055],
                [1063, 1067],
                [1075, 1079],
                [1087, 1091],
                [1099, 1103],
                [1111, 1115],
                [1123, 1127],
                [1135, 1139],
                [1147, 1151],
                [1159, 1163],
                [1171, 1175],
                [1183, 1187],
                [1195, 1199],
                [1207, 1211],
                [1219, 1223],
                [1231, 1235],
                [1243, 1247],
                [1255, 1259],
                [1267, 1271],
                [1279, 1283],
                [1291, 1295],
                [1735, 1739],
                [1747, 1751],
                [1759, 1763],
                [1771, 1775],
                [1783, 1787],
                [1795, 1799],
                [1807, 1811],
                [1819, 1823],
                [1831, 1835],
                [1843, 1847],
                [1855, 1859],
                [1867, 1871],
                [1879, 1883],
                [1891, 1895],
                [1903, 1907],
                [1915, 1919],
                [1927, 1931],
                [1939, 1943],
                [1951, 1955],
                [1963, 1967],
                [1975, 1979],
                [1987, 1991],
                [1999, 2003],
                [2011, 2015],
                [2023, 2027],
                [2035, 2039],
                [2047, 2051],
                [2059, 2063],
                [2071, 2075],
                [2083, 2087],
                [2095, 2099],
                [2107, 2111],
                [2119, 2123],
                [2131, 2135],
                [2143, 2147],
                [2155, 2159],
                [2167, 2171],
                [2179, 2183],
                [2191, 2195],
                [2203, 2207],
                [2215, 2219],
                [2227, 2231],
                [2239, 2243],
                [2251, 2255],
                [2263, 2267],
                [2275, 2279],
                [2287, 2291],
                [2299, 2303],
                [2311, 2315],
                [2323, 2327],
                [2335, 2339],
                [2347, 2351],
                [2359, 2363],
                [2371, 2375],
                [2383, 2387],
                [2395, 2399],
                [2407, 2411],
                [2419, 2423],
                [2431, 2435],
                [2443, 2447],
                [2455, 2459],
                [2467, 2471],
                [2479, 2483],
                [2491, 2495],
                [2503, 2507],
                [2515, 2519],
                [2527, 2531],
                [2539, 2543],
                [2551, 2555],
                [2563, 2567],
                [2575, 2579],
                [2587, 2591],
                [2599, 2603],
                [2611, 2615],
                [2623, 2627],
                [2635, 2639],
                [2647, 2651],
                [2659, 2663],
                [2671, 2675],
                [2683, 2687],
                [2695, 2699],
                [2707, 2711],
                [2719, 2723],
                [2731, 2735],
                [2743, 2747],
                [2755, 2759],
                [2767, 2771],
                [2779, 2783],
                [2791, 2795],
                [2803, 2807],
                [2815, 2819],
                [2827, 2831],
                [2839, 2843],
                [2851, 2855],
                [2863, 2867],
                [2875, 2879],
                [2887, 2891],
                [2899, 2903],
                [2911, 2915],
                [2923, 2927],
                [2935, 2939],
                [2947, 2951],
                [2959, 2963],
                [2971, 2975],
                [2983, 2987],
                [2995, 2999],
                [3007, 3011],
                [3019, 3023],
            ],
            vec![
                [1303, 1307],
                [1315, 1319],
                [1327, 1331],
                [1339, 1343],
                [1351, 1355],
                [1363, 1367],
                [1375, 1379],
                [1387, 1391],
                [1399, 1403],
                [1411, 1415],
                [1423, 1427],
                [1435, 1439],
                [1447, 1451],
                [1459, 1463],
                [1471, 1475],
                [1483, 1487],
                [1495, 1499],
                [1507, 1511],
                [1519, 1523],
                [1531, 1535],
                [1543, 1547],
                [1555, 1559],
                [1567, 1571],
                [1579, 1583],
                [1591, 1595],
                [1603, 1607],
                [1615, 1619],
                [1627, 1631],
                [1639, 1643],
                [1651, 1655],
                [1663, 1667],
                [1675, 1679],
                [1687, 1691],
                [1699, 1703],
                [1711, 1715],
                [1723, 1727],
                [3031, 3035],
                [3043, 3047],
                [3055, 3059],
                [3067, 3071],
                [3079, 3083],
                [3091, 3095],
                [3103, 3107],
                [3115, 3119],
                [3127, 3131],
                [3139, 3143],
                [3151, 3155],
                [3163, 3167],
                [3175, 3179],
                [3187, 3191],
                [3199, 3203],
                [3211, 3215],
                [3223, 3227],
                [3235, 3239],
                [3247, 3251],
                [3259, 3263],
                [3271, 3275],
                [3283, 3287],
                [3295, 3299],
                [3307, 3311],
                [3319, 3323],
                [3331, 3335],
                [3343, 3347],
                [3355, 3359],
                [3367, 3371],
                [3379, 3383],
                [3391, 3395],
                [3403, 3407],
                [3415, 3419],
                [3427, 3431],
                [3439, 3443],
                [3451, 3455],
            ],
        ];
        let relative_indices = [(2, 4), (3, 8)];

        assert_eq!(molecules.len(), 2);

        for (i, molecule) in molecules.into_iter().enumerate() {
            assert_eq!(molecule.name(), expected_names[i]);
            assert_eq!(molecule.topology(), &expected_topology[i]);
            let order_bonds = molecule.order_structure();
            let order_bond_topologies: HashSet<BondTopology> = order_bonds
                .bond_types()
                .iter()
                .map(|x| x.bond_topology().clone())
                .collect();
            assert_eq!(order_bond_topologies, expected_order_bond_topologies[i]);

            for (b, bond_instances) in order_bonds
                .bond_types()
                .iter()
                .map(|x| (x, x.bonds().clone()))
            {
                assert_eq!(bond_instances.len(), expected_n_instances[i]);
                if b.atom1_index() == relative_indices[i].0
                    && b.atom2_index() == relative_indices[i].1
                {
                    assert_eq!(bond_instances, expected_bond_instances[i]);
                }
            }

            assert_eq!(molecule.order_atoms(), &expected_order_atoms[i]);

            if let Some(MoleculeLeafletClassification::Individual(x, _)) =
                molecule.leaflet_classification()
            {
                assert_eq!(x.heads(), &expected_heads[i]);
                assert_eq!(x.methyls(), &expected_methyls[i]);
            } else {
                panic!("Incorrect MoleculeLeafletClassification.")
            }
        }
    }

    #[test]
    fn test_classify_molecules_shared_name() {
        let mut system = System::from_file("tests/files/same_name.tpr").unwrap();
        create_group(&mut system, "Beads", "resname POPC").unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/same_name.tpr")
            .trajectory("tests/files/same_name.xtc")
            .analysis_type(AnalysisType::cgorder("resname POPC"))
            .build()
            .unwrap();

        let molecules =
            match MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap()
            {
                MoleculeTypes::BondBased(x) => x,
                _ => panic!("MoleculeTypes should be bond-based."),
            };
        let expected_names = ["POPC1", "POPC2"];
        let expected_n_instances = [2, 1];

        for (i, molecule) in molecules.into_iter().enumerate() {
            assert_eq!(molecule.name(), expected_names[i]);

            for bond_instances in molecule
                .order_structure()
                .bond_types()
                .iter()
                .map(|x| x.bonds().clone())
            {
                assert_eq!(bond_instances.len(), expected_n_instances[i]);
            }
        }
    }

    #[test]
    fn test_classify_molecules_multiple_residues() {
        let mut system = System::from_file("tests/files/multiple_resid.tpr").unwrap();
        create_group(&mut system, "Beads", "resname POPC POPE").unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/multiple_resid.tpr")
            .trajectory("tests/files/multiple_resid.xtc")
            .analysis_type(AnalysisType::cgorder("resname POPC POPE"))
            .build()
            .unwrap();

        let molecules =
            match MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap()
            {
                MoleculeTypes::BondBased(x) => x,
                _ => panic!("MoleculeTypes should be bond-based."),
            };

        let expected_names = ["POPC-POPE", "POPC"];
        let expected_n_instances = [2, 1];

        for (i, molecule) in molecules.into_iter().enumerate() {
            assert_eq!(molecule.name(), expected_names[i]);

            for bond_instances in molecule
                .order_structure()
                .bond_types()
                .iter()
                .map(|x| x.bonds().clone())
            {
                assert_eq!(bond_instances.len(), expected_n_instances[i]);
            }
        }
    }

    #[test]
    fn test_classify_molecules_shared_name_multiple_residues() {
        let mut system = System::from_file("tests/files/multiple_resid_same_name.tpr").unwrap();
        create_group(&mut system, "Beads", "resname POPC POPE").unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/multiple_resid_same_name.tpr")
            .trajectory("tests/files/multiple_resid_same_name.xtc")
            .analysis_type(AnalysisType::cgorder("resname POPC POPE"))
            .build()
            .unwrap();

        let molecules =
            match MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap()
            {
                MoleculeTypes::BondBased(x) => x,
                _ => panic!("MoleculeTypes should be bond-based."),
            };

        let expected_names = ["POPC-POPE1", "POPC-POPE2", "POPC"];
        let expected_n_instances = [1, 1, 1];

        for (i, molecule) in molecules.into_iter().enumerate() {
            assert_eq!(molecule.name(), expected_names[i]);

            for bond_instances in molecule
                .order_structure()
                .bond_types()
                .iter()
                .map(|x| x.bonds().clone())
            {
                assert_eq!(bond_instances.len(), expected_n_instances[i]);
            }
        }
    }

    #[test]
    fn test_classify_molecules_cyclic_molecule() {
        let mut system = System::from_file("tests/files/cyclic.tpr").unwrap();
        create_group(&mut system, "Beads", "resname POPC").unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cyclic.tpr")
            .trajectory("tests/files/cyclic.xtc")
            .analysis_type(AnalysisType::cgorder("resname POPC"))
            .build()
            .unwrap();

        let molecules =
            match MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap()
            {
                MoleculeTypes::BondBased(x) => x,
                _ => panic!("MoleculeTypes should be bond-based."),
            };

        assert_eq!(molecules.len(), 1);
        assert_eq!(molecules[0].order_structure().bond_types().len(), 14);
    }
}
