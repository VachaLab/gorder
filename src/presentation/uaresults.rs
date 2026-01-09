// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for presenting the results of the analysis of united-atom order parameters.

use getset::Getters;
use indexmap::IndexMap;
use serde::{ser::SerializeMap, Serialize, Serializer};

use crate::{
    analysis::topology::uatom::UAOrderAtoms,
    input::Analysis,
    prelude::{AtomType, Convergence},
    presentation::{leaflets::LeafletsData, normals::NormalsData},
};

use super::{
    MoleculeResults, OrderCollection, OrderMapsCollection, OrderResults, PublicMoleculeResults,
    PublicOrderResults, UAOrder,
};

/// Results of the united-atom order parameters calculation.
#[derive(Debug, Clone, Getters)]
pub struct UAOrderResults {
    /// Results for individual molecules of the system.
    molecules: IndexMap<String, UAMoleculeResults>,
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

impl Serialize for UAOrderResults {
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

impl PublicOrderResults for UAOrderResults {
    type MoleculeResults = UAMoleculeResults;

    fn molecules(&self) -> impl Iterator<Item = &UAMoleculeResults> {
        self.molecules.values()
    }

    fn get_molecule(&self, name: &str) -> Option<&UAMoleculeResults> {
        self.molecules.get(name)
    }

    fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    fn n_analyzed_frames(&self) -> usize {
        self.n_analyzed_frames
    }
}

impl OrderResults for UAOrderResults {
    type OrderType = UAOrder;
    type MoleculeBased = UAOrderAtoms;

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
        molecules: IndexMap<String, UAMoleculeResults>,
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

/// United-atom order parameters calculated for a single molecule type.
#[derive(Debug, Clone, Serialize, Getters)]
pub struct UAMoleculeResults {
    /// Name of the molecule type.
    #[serde(skip)]
    molecule: String,
    /// Average order parameter calculated from all virtual bonds of this molecule type.
    #[serde(rename = "average order")]
    #[getset(get = "pub")]
    average_order: OrderCollection,
    /// Average order parameter maps calculated from all virtual bonds of this molecule type.
    #[serde(skip)]
    #[getset(get = "pub")]
    average_ordermaps: OrderMapsCollection,
    /// Order parameters calculated for individual atom types.
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    #[serde(rename = "order parameters")]
    #[getset(get = "pub(super)")]
    order: IndexMap<AtomType, UAAtomResults>,
    /// Data about convergence of the order parameters.
    #[serde(skip)]
    convergence: Option<Convergence>,
}

impl PublicMoleculeResults for UAMoleculeResults {
    fn molecule(&self) -> &str {
        &self.molecule
    }

    fn convergence(&self) -> Option<&Convergence> {
        self.convergence.as_ref()
    }
}

impl MoleculeResults for UAMoleculeResults {
    #[inline(always)]
    fn max_bonds(&self) -> usize {
        self.order
            .values()
            .map(|x| x.bonds.len())
            .max()
            .unwrap_or(0)
    }
}

impl UAMoleculeResults {
    pub(super) fn new(
        molecule: &str,
        average_order: OrderCollection,
        average_ordermaps: OrderMapsCollection,
        order: IndexMap<AtomType, UAAtomResults>,
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
    pub fn atoms(&self) -> impl Iterator<Item = &UAAtomResults> {
        self.order.values()
    }

    /// Get results for an atom with the specified relative index.
    /// O(n) complexity.
    /// Returns `None` if such atom does not exist or is not a united-atom with calculated order parameters.
    pub fn get_atom(&self, relative_index: usize) -> Option<&UAAtomResults> {
        self.order.iter().find_map(|(atom, results)| {
            if atom.relative_index() == relative_index {
                Some(results)
            } else {
                None
            }
        })
    }
}

/// United-atom order parameters calculated for a single atom type and for involved virtual bond types.
#[derive(Debug, Clone, Serialize, Getters)]
pub struct UAAtomResults {
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
    /// Order parameters calculated for virtual bonds that this atom type is involved in.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    bonds: Vec<UABondResults>,
}

impl UAAtomResults {
    pub(super) fn new(
        atom: AtomType,
        molecule: &str,
        order: OrderCollection,
        ordermaps: OrderMapsCollection,
        bonds: Vec<UABondResults>,
    ) -> Self {
        Self {
            atom,
            molecule: molecule.to_owned(),
            order,
            ordermaps,
            bonds,
        }
    }

    /// Get results for all the bonds of the atom.
    pub fn bonds(&self) -> impl Iterator<Item = &UABondResults> {
        self.bonds.iter()
    }
}

/// Order parameters calculated for a single united-atom C-H bond.
#[derive(Debug, Clone, Serialize, Getters)]
pub struct UABondResults {
    /// Name of the molecule this bond belongs to.
    #[serde(skip)]
    #[getset(get = "pub")]
    molecule: String,
    /// Order parameters calculated for this bond.
    #[serde(flatten)]
    #[getset(get = "pub")]
    order: OrderCollection,
    /// Ordermaps calculated for this bond.
    #[serde(skip)]
    #[getset(get = "pub")]
    ordermaps: OrderMapsCollection,
}

impl UABondResults {
    pub(super) fn new(
        molecule: &str,
        order: OrderCollection,
        ordermaps: OrderMapsCollection,
    ) -> Self {
        Self {
            molecule: molecule.to_owned(),
            order,
            ordermaps,
        }
    }
}
