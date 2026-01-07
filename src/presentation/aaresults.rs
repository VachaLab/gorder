// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for presenting the results of the analysis of atomistic order parameters.

use super::convergence::Convergence;
use super::{
    AAOrder, BondResults, MoleculeResults, OrderCollection, OrderResults, PublicMoleculeResults,
    PublicOrderResults,
};
use crate::analysis::topology::atom::AtomType;
use crate::analysis::topology::bond::OrderBonds;
use crate::input::Analysis;
use crate::presentation::leaflets::LeafletsData;
use crate::presentation::normals::NormalsData;
use crate::presentation::OrderMapsCollection;
use getset::Getters;
use indexmap::IndexMap;
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

/// Results of the atomistic order parameters calculation.
#[derive(Debug, Clone, Getters)]
pub struct AAOrderResults {
    /// Results for individual molecules of the system.
    molecules: IndexMap<String, AAMoleculeResults>,
    /// Average order parameter calculated from all bond types of all molecule types.
    #[getset(get = "pub")]
    average_order: OrderCollection,
    /// Average order parameter maps calculated from all bond types of all molecule type.
    #[getset(get = "pub")]
    average_ordermaps: OrderMapsCollection,
    /// Leaflet classification data collected during the analysis.
    /// Only collected, if explicitly requested.
    #[getset(get = "pub")]
    leaflets_data: Option<LeafletsData>,
    /// Membrane normals collected during the analysis.
    /// Only collected, if explicitly requested.
    #[getset(get = "pub")]
    normals_data: Option<NormalsData>,
    /// Parameters of the analysis.
    analysis: Analysis,
    /// Total number of analyzed frames.
    n_analyzed_frames: usize,
}

impl Serialize for AAOrderResults {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.molecules.len() + 1))?;

        // serialize "average order" field
        map.serialize_entry("average order", &self.average_order)?;

        // serialize individual molecules
        for (key, value) in &self.molecules {
            map.serialize_entry(key, value)?;
        }

        map.end()
    }
}

impl PublicOrderResults for AAOrderResults {
    type MoleculeResults = AAMoleculeResults;

    fn molecules(&self) -> impl Iterator<Item = &AAMoleculeResults> {
        self.molecules.values()
    }

    fn get_molecule(&self, name: &str) -> Option<&AAMoleculeResults> {
        self.molecules.get(name)
    }

    fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    fn n_analyzed_frames(&self) -> usize {
        self.n_analyzed_frames
    }
}

impl OrderResults for AAOrderResults {
    type OrderType = AAOrder;
    type MoleculeBased = OrderBonds;

    fn empty(analysis: Analysis) -> Self {
        Self {
            average_order: OrderCollection::default(),
            average_ordermaps: OrderMapsCollection::default(),
            molecules: IndexMap::new(),
            leaflets_data: None,
            normals_data: None,
            analysis,
            n_analyzed_frames: 0,
        }
    }

    fn new(
        molecules: IndexMap<String, AAMoleculeResults>,
        average_order: OrderCollection,
        average_ordermaps: OrderMapsCollection,
        leaflets_data: Option<LeafletsData>,
        normals_data: Option<NormalsData>,
        analysis: Analysis,
        n_analyzed_frames: usize,
    ) -> Self {
        Self {
            molecules,
            average_order,
            average_ordermaps,
            leaflets_data,
            normals_data,
            analysis,
            n_analyzed_frames,
        }
    }

    /// Get the maximal number of bonds per heavy atoms in the system.
    /// Returns 0 if there are no molecules.
    fn max_bonds(&self) -> usize {
        self.molecules
            .values()
            .map(|mol| mol.max_bonds())
            .max()
            .unwrap_or(0)
    }

    fn average_ordermaps(&self) -> &OrderMapsCollection {
        &self.average_ordermaps
    }

    fn leaflets_data(&self) -> &Option<LeafletsData> {
        self.leaflets_data()
    }

    fn normals_data(&self) -> &Option<NormalsData> {
        self.normals_data()
    }
}

/// Atomistic order parameters calculated for a single molecule type.
#[derive(Debug, Clone, Serialize, Getters)]
pub struct AAMoleculeResults {
    /// Name of the molecule type.
    #[serde(skip)]
    molecule: String,
    /// Average order parameter calculated from all bond types of this molecule type.
    #[serde(rename = "average order")]
    #[getset(get = "pub")]
    average_order: OrderCollection,
    /// Average order parameter maps calculated from all bond types of this molecule type.
    #[serde(skip)]
    #[getset(get = "pub")]
    average_ordermaps: OrderMapsCollection,
    /// Order parameters calculated for individual atom types.
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    #[serde(rename = "order parameters")]
    #[getset(get = "pub(super)")]
    order: IndexMap<AtomType, AAAtomResults>,
    /// Data about convergence of the order parameters.
    #[serde(skip)]
    convergence: Option<Convergence>,
}

impl AAMoleculeResults {
    pub(super) fn new(
        molecule: &str,
        average_order: OrderCollection,
        average_ordermaps: OrderMapsCollection,
        order: IndexMap<AtomType, AAAtomResults>,
        convergence: Option<Convergence>,
    ) -> Self {
        Self {
            molecule: molecule.to_owned(),
            average_order,
            average_ordermaps,
            order,
            convergence,
        }
    }

    /// Get the results for all the atoms of the molecule.
    pub fn atoms(&self) -> impl Iterator<Item = &AAAtomResults> {
        self.order.values()
    }

    /// Get results for an atom with the specified relative index.
    /// O(n) complexity.
    /// Returns `None` if such atom does not exist or is not a heavy atom with calculated order parameters.
    pub fn get_atom(&self, relative_index: usize) -> Option<&AAAtomResults> {
        self.order.iter().find_map(|(atom, results)| {
            if atom.relative_index() == relative_index {
                Some(results)
            } else {
                None
            }
        })
    }

    /// Get results for a bond involving atoms with the specified relative indices.
    /// O(n) complexity.
    /// Returns `None` if such bond does not exist. The order of atoms does not matter.
    pub fn get_bond(
        &self,
        relative_index_1: usize,
        relative_index_2: usize,
    ) -> Option<&BondResults> {
        if let Some(atom) = self.get_atom(relative_index_1) {
            atom.get_bond(relative_index_2)
        } else if let Some(atom) = self.get_atom(relative_index_2) {
            atom.get_bond(relative_index_1)
        } else {
            None
        }
    }
}

impl PublicMoleculeResults for AAMoleculeResults {
    fn molecule(&self) -> &str {
        &self.molecule
    }

    fn convergence(&self) -> Option<&Convergence> {
        self.convergence.as_ref()
    }
}

impl MoleculeResults for AAMoleculeResults {
    #[inline(always)]
    fn max_bonds(&self) -> usize {
        self.order
            .values()
            .map(|x| x.bonds.len())
            .max()
            .unwrap_or(0)
    }
}

/// Atomistic order parameters calculated for a single atom type and for involved bond types.
#[derive(Debug, Clone, Serialize, Getters)]
pub struct AAAtomResults {
    /// Name of the atom type.
    #[serde(skip)]
    #[getset(get = "pub")]
    atom: AtomType,
    /// Name of the molecule this atom is part of.
    #[serde(skip)]
    #[getset(get = "pub")]
    molecule: String,
    /// Order parameters calculated for this atom.
    #[serde(flatten)]
    #[getset(get = "pub")]
    order: OrderCollection,
    /// Ordermaps calculated for this atom.
    #[serde(skip)]
    #[getset(get = "pub")]
    ordermaps: OrderMapsCollection,
    /// Order parameters calculated for bond types that this atom type is involved in.
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    bonds: IndexMap<AtomType, BondResults>,
}

impl AAAtomResults {
    pub(super) fn new(
        atom: AtomType,
        molecule: &str,
        order: OrderCollection,
        ordermaps: OrderMapsCollection,
        bonds: IndexMap<AtomType, BondResults>,
    ) -> Self {
        Self {
            atom,
            molecule: molecule.to_owned(),
            order,
            ordermaps,
            bonds,
        }
    }

    /// Check whether the atom has any bonds associated with it.
    #[inline(always)]
    pub(super) fn is_empty(&self) -> bool {
        self.bonds.is_empty()
    }

    /// Get results for all the bonds of the atom.
    pub fn bonds(&self) -> impl Iterator<Item = &BondResults> {
        self.bonds.values()
    }

    /// Get results for a bond between this atom and the atom with the specified relative index.
    /// O(n) complexity.
    /// Returns `None` if such bond exists.
    pub fn get_bond(&self, relative_index: usize) -> Option<&BondResults> {
        self.bonds.iter().find_map(|(atom, results)| {
            if atom.relative_index() == relative_index {
                Some(results)
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{prelude::Order, presentation::BondTopology};

    use super::*;

    fn create_example_bond1() -> BondResults {
        let atom1 = AtomType::new_raw(3, "POPC", "C22");
        let atom2 = AtomType::new_raw(5, "POPC", "H2A");

        let order = OrderCollection::new(
            Some(Order::from([0.1278, 0.0063])),
            Some(Order::from([0.2342, 0.0102])),
            Some(Order::from([0.7621, 0.0098])),
        );

        let maps = OrderMapsCollection::new(None, None, None);

        BondResults::new(
            &BondTopology::new_from_types(atom1, atom2),
            "POPC",
            order,
            maps,
        )
    }

    fn create_example_bond2() -> BondResults {
        let atom1 = AtomType::new_raw(3, "POPC", "C22");
        let atom2 = AtomType::new_raw(6, "POPC", "H2B");

        let order = OrderCollection::new(
            Some(Order::from([0.2278, 0.0073])),
            Some(Order::from([0.3342, 0.0112])),
            Some(Order::from([0.8621, 0.0108])),
        );

        let maps = OrderMapsCollection::new(None, None, None);

        BondResults::new(
            &BondTopology::new_from_types(atom1, atom2),
            "POPC",
            order,
            maps,
        )
    }

    fn create_example_atom() -> AAAtomResults {
        let order = OrderCollection::new(
            Some(Order::from([0.4231, 0.0032])),
            Some(Order::from([0.1234, 0.0017])),
            Some(Order::from([0.2863, 0.0123])),
        );
        let maps = OrderMapsCollection::new(None, None, None);
        let bonds = IndexMap::from([
            (AtomType::new_raw(5, "POPC", "H2A"), create_example_bond1()),
            (AtomType::new_raw(6, "POPC", "H2B"), create_example_bond2()),
        ]);
        AAAtomResults::new(
            AtomType::new_raw(3, "POPC", "C22"),
            "POPC",
            order,
            maps,
            bonds,
        )
    }

    #[test]
    fn atom_results_get_bond() {
        let atom = create_example_atom();
        let bond1 = atom.get_bond(5).unwrap();

        let (a1, a2) = bond1.atoms();
        assert_eq!(a1.atom_name(), "C22");
        assert_eq!(a1.relative_index(), 3);
        assert_eq!(a1.residue_name(), "POPC");

        assert_eq!(a2.atom_name(), "H2A");
        assert_eq!(a2.relative_index(), 5);
        assert_eq!(a2.residue_name(), "POPC");

        assert_eq!(bond1.molecule(), "POPC");

        assert_relative_eq!(
            bond1.order().total().unwrap().value(),
            0.1278,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            bond1.order().upper().unwrap().value(),
            0.2342,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            bond1.order().lower().unwrap().value(),
            0.7621,
            epsilon = 1e-4
        );

        assert_relative_eq!(
            bond1.order().total().unwrap().error().unwrap(),
            0.0063,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            bond1.order().upper().unwrap().error().unwrap(),
            0.0102,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            bond1.order().lower().unwrap().error().unwrap(),
            0.0098,
            epsilon = 1e-4
        );

        let bond2 = atom.get_bond(6).unwrap();

        let (a1, a2) = bond2.atoms();
        assert_eq!(a1.atom_name(), "C22");
        assert_eq!(a1.relative_index(), 3);
        assert_eq!(a1.residue_name(), "POPC");

        assert_eq!(a2.atom_name(), "H2B");
        assert_eq!(a2.relative_index(), 6);
        assert_eq!(a2.residue_name(), "POPC");

        assert_eq!(bond1.molecule(), "POPC");

        assert_relative_eq!(
            bond2.order().total().unwrap().value(),
            0.2278,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            bond2.order().upper().unwrap().value(),
            0.3342,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            bond2.order().lower().unwrap().value(),
            0.8621,
            epsilon = 1e-4
        );

        assert_relative_eq!(
            bond2.order().total().unwrap().error().unwrap(),
            0.0073,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            bond2.order().upper().unwrap().error().unwrap(),
            0.0112,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            bond2.order().lower().unwrap().error().unwrap(),
            0.0108,
            epsilon = 1e-4
        );

        assert!(atom.get_bond(3).is_none());
        assert!(atom.get_bond(7).is_none());
    }
}
