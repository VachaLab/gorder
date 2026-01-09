// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for presenting the results of the analysis of coarse-grained order parameters.

use getset::Getters;
use indexmap::IndexMap;
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

use super::convergence::Convergence;
use super::{
    BondResults, CGOrder, MoleculeResults, OrderCollection, OrderResults, PublicMoleculeResults,
    PublicOrderResults,
};
use crate::analysis::topology::bond::{BondTopology, OrderBonds};
use crate::input::Analysis;
use crate::presentation::leaflets::LeafletsData;
use crate::presentation::normals::NormalsData;
use crate::presentation::OrderMapsCollection;

/// Results of the coarse-grained order parameters calculation.
#[derive(Debug, Clone, Getters)]
pub struct CGOrderResults {
    /// Results for individual molecules of the system.
    molecules: IndexMap<String, CGMoleculeResults>,
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

impl Serialize for CGOrderResults {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.molecules.len() + 1))?;

        // Serialize "average order" field
        map.serialize_entry("average order", &self.average_order)?;

        // Serialize individual molecules
        for (key, value) in &self.molecules {
            map.serialize_entry(key, value)?;
        }

        map.end()
    }
}

impl PublicOrderResults for CGOrderResults {
    type MoleculeResults = CGMoleculeResults;

    fn molecules(&self) -> impl Iterator<Item = &CGMoleculeResults> {
        self.molecules.values()
    }

    fn get_molecule(&self, name: &str) -> Option<&CGMoleculeResults> {
        self.molecules.get(name)
    }

    fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    fn n_analyzed_frames(&self) -> usize {
        self.n_analyzed_frames
    }
}

impl OrderResults for CGOrderResults {
    type OrderType = CGOrder;
    type MoleculeBased = OrderBonds;

    fn empty(analysis: Analysis) -> Self {
        Self {
            molecules: IndexMap::new(),
            average_order: OrderCollection::default(),
            average_ordermaps: OrderMapsCollection::default(),
            leaflets_data: None,
            normals_data: None,
            analysis,
            n_analyzed_frames: 0,
        }
    }

    fn new(
        molecules: IndexMap<String, CGMoleculeResults>,
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

/// Coarse-grained order parameters calculated for a single molecule type.
#[derive(Debug, Clone, Serialize, Getters)]
pub struct CGMoleculeResults {
    /// Name of the molecule.
    #[serde(skip)]
    molecule: String,
    /// Average order parameter for all bond types of the molecule.
    #[serde(rename = "average order")]
    #[getset(get = "pub")]
    average_order: OrderCollection,
    /// Average order parameter maps calculated from all bond types of this molecule type.
    #[serde(skip)]
    #[getset(get = "pub")]
    average_ordermaps: OrderMapsCollection,
    /// Order parameters calculated for specific bonds.
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    #[serde(rename = "order parameters")]
    #[getset(get = "pub(super)")]
    order: IndexMap<BondTopology, BondResults>,
    /// Data about convergence of the order parameters.
    #[serde(skip)]
    convergence: Option<Convergence>,
}

impl CGMoleculeResults {
    pub(super) fn new(
        name: &str,
        average_order: OrderCollection,
        average_ordermaps: OrderMapsCollection,
        order: IndexMap<BondTopology, BondResults>,
        convergence: Option<Convergence>,
    ) -> Self {
        CGMoleculeResults {
            molecule: name.to_owned(),
            average_order,
            average_ordermaps,
            order,
            convergence,
        }
    }

    /// Get results for all the bonds of the molecule.
    pub fn bonds(&self) -> impl Iterator<Item = &BondResults> {
        self.order.values()
    }

    /// Get results for a bond involving atoms with the specified relative indices.
    /// O(n) complexity.
    /// Returns `None` if such bond does not exist. The order of atoms does not matter.
    pub fn get_bond(
        &self,
        relative_index_1: usize,
        relative_index_2: usize,
    ) -> Option<&BondResults> {
        self.order.iter().find_map(|(bond, results)| {
            let (a1, a2) = (bond.atom1().relative_index(), bond.atom2().relative_index());
            if (a1, a2) == (relative_index_1, relative_index_2)
                || (a2, a1) == (relative_index_1, relative_index_2)
            {
                Some(results)
            } else {
                None
            }
        })
    }
}

impl Serialize for BondTopology {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let formatted_string = format!(
            "{} {} ({}) - {} {} ({})",
            self.atom1().residue_name(),
            self.atom1().atom_name(),
            self.atom1().relative_index(),
            self.atom2().residue_name(),
            self.atom2().atom_name(),
            self.atom2().relative_index()
        );
        serializer.serialize_str(&formatted_string)
    }
}

impl MoleculeResults for CGMoleculeResults {}

impl PublicMoleculeResults for CGMoleculeResults {
    fn molecule(&self) -> &str {
        &self.molecule
    }

    fn convergence(&self) -> Option<&Convergence> {
        self.convergence.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use indexmap::IndexMap;

    use crate::prelude::{AtomType, BondResults, Order, OrderCollection, OrderMapsCollection};

    use super::{BondTopology, CGMoleculeResults};

    fn create_example_bond1() -> BondResults {
        let atom1 = AtomType::new_raw(3, "POPC", "NC3");
        let atom2 = AtomType::new_raw(5, "POPC", "PO4");

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
        let atom1 = AtomType::new_raw(3, "POPC", "NC3");
        let atom2 = AtomType::new_raw(6, "POPC", "GL1");

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

    fn create_example_molecule() -> CGMoleculeResults {
        let atom1 = AtomType::new_raw(3, "POPC", "NC3");
        let atom2 = AtomType::new_raw(5, "POPC", "PO4");
        let atom3 = AtomType::new_raw(6, "POPC", "GL1");

        let bond_topology1 = BondTopology::new_from_types(atom1.clone(), atom2);
        let bond1 = create_example_bond1();
        let bond_topology2 = BondTopology::new_from_types(atom1, atom3);
        let bond2 = create_example_bond2();

        let bonds = IndexMap::from([(bond_topology1, bond1), (bond_topology2, bond2)]);

        let average_order = OrderCollection::new(None, None, None);
        let average_maps = OrderMapsCollection::new(None, None, None);

        CGMoleculeResults::new("POPC", average_order, average_maps, bonds, None)
    }

    #[test]
    fn molecule_results_get_bond() {
        let mol = create_example_molecule();

        let bond1 = mol.get_bond(3, 5).unwrap();
        let (a1, a2) = bond1.atoms();
        assert_eq!(a1.atom_name(), "NC3");
        assert_eq!(a1.relative_index(), 3);
        assert_eq!(a1.residue_name(), "POPC");

        assert_eq!(a2.atom_name(), "PO4");
        assert_eq!(a2.relative_index(), 5);
        assert_eq!(a2.residue_name(), "POPC");

        let bond1 = mol.get_bond(5, 3).unwrap();
        let (a1, a2) = bond1.atoms();
        assert_eq!(a1.atom_name(), "NC3");
        assert_eq!(a1.relative_index(), 3);
        assert_eq!(a1.residue_name(), "POPC");

        assert_eq!(a2.atom_name(), "PO4");
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

        let bond2 = mol.get_bond(3, 6).unwrap();
        let (a1, a2) = bond2.atoms();
        assert_eq!(a1.atom_name(), "NC3");
        assert_eq!(a1.relative_index(), 3);
        assert_eq!(a1.residue_name(), "POPC");

        assert_eq!(a2.atom_name(), "GL1");
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

        assert!(mol.get_bond(5, 6).is_none());
        assert!(mol.get_bond(3, 7).is_none());
        assert!(mol.get_bond(1, 9).is_none());
    }
}
