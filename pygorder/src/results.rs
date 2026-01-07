// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

use std::sync::Arc;

use gorder_core::prelude::AAAtomResults as RsAtomResults;
use gorder_core::prelude::BondResults as RsBondResults;
use gorder_core::prelude::Convergence as RsConvergence;
use gorder_core::prelude::GridMapF32;
use gorder_core::prelude::Order as RsOrder;
use gorder_core::prelude::OrderCollection as RsOrderCollection;
use gorder_core::prelude::OrderMapsCollection as RsMapsCollection;
use gorder_core::prelude::PublicMoleculeResults;
use gorder_core::prelude::UAAtomResults as RsUAAtomResults;
use gorder_core::prelude::UAMoleculeResults;
use gorder_core::prelude::Vector3D;
use gorder_core::prelude::{
    AAMoleculeResults, AnalysisResults as RsResults, CGMoleculeResults, PublicOrderResults,
};
use gorder_core::Leaflet as RsLeaflet;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;
use pyo3_stub_gen::derive::gen_stub_pymethods;

use crate::APIError;
use crate::AtomType;
use crate::WriteError;

/// Container for all results of an analysis.
///
/// Provides access to overall results, per-molecule results, average order
/// parameters, average order maps, and optionally collected leaflet classification data.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct AnalysisResults(pub(crate) Arc<RsResults>);

#[gen_stub_pymethods]
#[pymethods]
impl AnalysisResults {
    /// Write the results into output files.
    ///
    /// Raises
    /// ------
    /// WriteError
    ///     If writing fails due to file system or internal errors.
    pub fn write(&self) -> PyResult<()> {
        if let Err(e) = self.0.write() {
            Err(WriteError::new_err(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Get the total number of analyzed frames.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of frames processed in the analysis.
    pub fn n_analyzed_frames(&self) -> usize {
        self.0.n_analyzed_frames()
    }

    /// Get results for all individual molecule types.
    ///
    /// Returns
    /// -------
    /// List[MoleculeResults]
    ///     A list of results for each molecule type.
    pub fn molecules(&self) -> Vec<MoleculeResults> {
        match self.0.as_ref() {
            RsResults::AA(x) => x
                .molecules()
                .map(|x| MoleculeResults {
                    results: self.0.clone(),
                    name: x.molecule().to_owned(),
                })
                .collect::<Vec<MoleculeResults>>(),
            RsResults::CG(x) => x
                .molecules()
                .map(|x| MoleculeResults {
                    results: self.0.clone(),
                    name: x.molecule().to_owned(),
                })
                .collect::<Vec<MoleculeResults>>(),
            RsResults::UA(x) => x
                .molecules()
                .map(|x| MoleculeResults {
                    results: self.0.clone(),
                    name: x.molecule().to_owned(),
                })
                .collect::<Vec<MoleculeResults>>(),
        }
    }

    /// Get results for a molecule type with the specified name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the molecule type.
    ///
    /// Returns
    /// -------
    /// MoleculeResults
    ///     Results for the specified molecule type.
    ///
    /// Raises
    /// ------
    /// APIError
    ///     If no molecule type with the given name exists.
    pub fn get_molecule(&self, name: &str) -> PyResult<MoleculeResults> {
        match self.0.as_ref() {
            RsResults::AA(x) => x
                .get_molecule(name)
                .ok_or_else(|| APIError::new_err("molecule with the given name does not exist"))
                .map(|x| MoleculeResults {
                    results: self.0.clone(),
                    name: x.molecule().to_owned(),
                }),
            RsResults::CG(x) => x
                .get_molecule(name)
                .ok_or_else(|| APIError::new_err("molecule with the given name does not exist"))
                .map(|x| MoleculeResults {
                    results: self.0.clone(),
                    name: x.molecule().to_owned(),
                }),
            RsResults::UA(x) => x
                .get_molecule(name)
                .ok_or_else(|| APIError::new_err("molecule with the given name does not exist"))
                .map(|x| MoleculeResults {
                    results: self.0.clone(),
                    name: x.molecule().to_owned(),
                }),
        }
    }

    /// Get average order parameters across all bond types of all molecules.
    ///
    /// Returns
    /// -------
    /// OrderCollection
    ///     Collection of average order parameters.
    pub fn average_order(&self) -> OrderCollection {
        OrderCollection {
            results: self.0.clone(),
            molecule: None,
            identifier: OrderIdentifier::Average,
        }
    }

    /// Get average order parameter maps across all bond types of all molecules.
    ///
    /// Returns
    /// -------
    /// OrderMapsCollection
    ///     Collection of average order maps.
    pub fn average_ordermaps(&self) -> OrderMapsCollection {
        OrderMapsCollection {
            results: self.0.clone(),
            molecule: None,
            identifier: OrderIdentifier::Average,
        }
    }

    /// Get collected leaflet classification data.
    ///
    /// Returns
    /// -------
    /// LeafletsData
    ///     Leaflet classification data if stored; otherwise `None`.
    ///
    /// Notes
    /// -----
    /// Leaflet classification data are not stored by default. To store them,
    /// you must explicitly request collection during analysis.
    pub fn leaflets_data(&self) -> Option<LeafletsData> {
        let leaflets_data_available = match self.0.as_ref() {
            RsResults::AA(x) => x.leaflets_data().is_some(),
            RsResults::CG(x) => x.leaflets_data().is_some(),
            RsResults::UA(x) => x.leaflets_data().is_some(),
        };

        if !leaflets_data_available {
            return None;
        }

        Some(LeafletsData {
            results: self.0.clone(),
        })
    }

    /// Get collected membrane normals.
    ///
    /// Returns
    /// -------
    /// NormalsData
    ///     Membrane normals data if stored; otherwise `None`.
    ///
    /// Notes
    /// -----
    /// Membrane normals are not stored by default. To store them,
    /// you must explicitly request collection during analysis.
    /// Collecting membrane normals is only supported when they are dynamically calculated.
    pub fn normals_data(&self) -> Option<NormalsData> {
        let normals_data_available = match self.0.as_ref() {
            RsResults::AA(x) => x.normals_data().is_some(),
            RsResults::CG(x) => x.normals_data().is_some(),
            RsResults::UA(x) => x.normals_data().is_some(),
        };

        if !normals_data_available {
            return None;
        }

        Some(NormalsData {
            results: self.0.clone(),
        })
    }
}

/// Results of the analysis for a single molecule type.
///
/// Provides access to average order parameters, average order maps, and per-atom or
/// per-bond results, as well as convergence data when available.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct MoleculeResults {
    results: Arc<RsResults>,
    name: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl MoleculeResults {
    /// Get the name of the molecule type.
    ///
    /// Returns
    /// -------
    /// str
    ///     Name of the molecule type.
    pub fn molecule(&self) -> String {
        self.name.to_owned()
    }

    /// Get average order parameters for this molecule type.
    ///
    /// Returns
    /// -------
    /// OrderCollection
    ///     Collection of average order parameters across all bond types.
    pub fn average_order(&self) -> OrderCollection {
        OrderCollection {
            results: self.results.clone(),
            molecule: Some(self.name.clone()),
            identifier: OrderIdentifier::Average,
        }
    }

    /// Get average order parameter maps for this molecule type.
    ///
    /// Returns
    /// -------
    /// OrderMapsCollection
    ///     Collection of average order maps across all bond types.
    pub fn average_ordermaps(&self) -> OrderMapsCollection {
        OrderMapsCollection {
            results: self.results.clone(),
            molecule: Some(self.name.clone()),
            identifier: OrderIdentifier::Average,
        }
    }

    /// Get results for each heavy atom type of the molecule.
    ///
    /// Returns
    /// -------
    /// List[AtomResults]
    ///     Per-atom type results.
    ///
    /// Raises
    /// ------
    /// APIError
    ///     If results are obtained for a coarse-grained system (atom-level results unavailable).
    pub fn atoms(&self) -> PyResult<Vec<AtomResults>> {
        match self.results.as_ref() {
            RsResults::AA(x) => {
                Ok(x.get_molecule(&self.name).unwrap().atoms().map(|atom| AtomResults {
                    results: self.results.clone(),
                    molecule: self.name.clone(),
                    atom: atom.atom().relative_index(),
                })
                .collect())
            }
            RsResults::UA(x) => {
                Ok(x.get_molecule(&self.name).unwrap().atoms().map(|atom| AtomResults {
                    results: self.results.clone(),
                    molecule: self.name.clone(),
                    atom: atom.atom().relative_index(),
                })
                .collect())
            }
            RsResults::CG(_) => Err(APIError::new_err(
                "results for individual atoms are not available for coarse-grained order parameters; you want `bonds`",
            )),
        }
    }

    /// Get results for each bond type of the molecule.
    ///
    /// Returns
    /// -------
    /// List[BondResults]
    ///     Per-bond type results.
    pub fn bonds(&self) -> Vec<BondResults> {
        match self.results.as_ref() {
            RsResults::AA(x) => x
                .get_molecule(&self.name)
                .unwrap()
                .atoms()
                .flat_map(|atom| atom.bonds())
                .map(|bond| BondResults::new(self.results.clone(), bond, self.name.clone()))
                .collect(),
            RsResults::CG(x) => x
                .get_molecule(&self.name)
                .unwrap()
                .bonds()
                .map(|bond| BondResults::new(self.results.clone(), bond, self.name.clone()))
                .collect(),
            RsResults::UA(x) => x
                .get_molecule(&self.name)
                .unwrap()
                .atoms()
                .flat_map(|atom| {
                    atom.bonds().enumerate().map(|(hydrogen, _)| {
                        BondResults::new_ua(
                            self.results.clone(),
                            atom.atom().relative_index(),
                            hydrogen,
                            self.name.clone(),
                        )
                    })
                })
                .collect(),
        }
    }

    /// Get results for a heavy atom type with the specified relative index.
    ///
    /// Parameters
    /// ----------
    /// relative_index : int
    ///     Relative index (zero-based) of the atom type within the molecule type.
    ///
    /// Returns
    /// -------
    /// AtomResults
    ///     Results for the specified atom type.
    ///
    /// Raises
    /// ------
    /// APIError
    ///     If the atom type does not exist or results are obtained for a coarse-grained system.
    pub fn get_atom(&self, relative_index: usize) -> PyResult<AtomResults> {
        match self.results.as_ref() {
            RsResults::AA(x) => x
                .get_molecule(&self.name)
                .unwrap()
                .get_atom(relative_index)
                .ok_or_else(|| {
                    APIError::new_err("atom with the given relative index does not exist")
                })
                .map(|atom| AtomResults {
                    results: self.results.clone(),
                    molecule: self.name.clone(),
                    atom: atom.atom().relative_index(),
                }),
            RsResults::UA(x) => x
                .get_molecule(&self.name)
                .unwrap()
                .get_atom(relative_index)
                .ok_or_else(|| {
                    APIError::new_err("atom with the given relative index does not exist")
                })
                .map(|atom| AtomResults {
                    results: self.results.clone(),
                    molecule: self.name.clone(),
                    atom: atom.atom().relative_index(),
                }),
            RsResults::CG(_) => Err(APIError::new_err(
                "results for individual atoms are not available for coarse-grained order parameters; you want `get_bond`"
            ))
        }
    }

    /// Get results for a bond type involving atom types with the specified relative indices.
    ///
    /// Parameters
    /// ----------
    /// relative_index_1 : int
    /// relative_index_2 : int
    ///     Relative indices (zero-based) of the bonded atom types.
    ///
    /// Returns
    /// -------
    /// BondResults
    ///     Results for the specified bond type.
    ///
    /// Raises
    /// ------
    /// APIError
    ///     If the bond type does not exist or the results are obtained for a united-atom system.
    pub fn get_bond(
        &self,
        relative_index_1: usize,
        relative_index_2: usize,
    ) -> PyResult<BondResults> {
        match self.results.as_ref() {
            RsResults::AA(x) => x
                .get_molecule(&self.name)
                .unwrap()
                .get_bond(relative_index_1, relative_index_2)
                .ok_or_else(|| {
                    APIError::new_err("bond specified by the given relative indices does not exist")
                })
                .map(|bond| BondResults::new(self.results.clone(), bond, self.name.clone())),
            RsResults::CG(x) => x
                .get_molecule(&self.name)
                .unwrap()
                .get_bond(relative_index_1, relative_index_2)
                .ok_or_else(|| {
                    APIError::new_err("bond specified by the given relative indices does not exist")
                })
                .map(|bond| BondResults::new(self.results.clone(), bond, self.name.clone())),
            RsResults::UA(_) => Err(APIError::new_err(
                "united-atom results for individual bonds cannot be accesed by using relative indices because the hydrogen atoms are virtual and do not have assigned indices",
            )),
        }
    }

    /// Get convergence data for the molecule.
    ///
    /// Returns
    /// -------
    /// Convergence
    ///     Convergence data if available; otherwise `None`.
    pub fn convergence(&self) -> Option<Convergence> {
        match self.results.as_ref() {
            RsResults::AA(x) => {
                x.get_molecule(&self.name)
                    .unwrap()
                    .convergence()
                    .map(|_| Convergence {
                        results: self.results.clone(),
                        molecule: self.name.clone(),
                    })
            }
            RsResults::CG(x) => {
                x.get_molecule(&self.name)
                    .unwrap()
                    .convergence()
                    .map(|_| Convergence {
                        results: self.results.clone(),
                        molecule: self.name.clone(),
                    })
            }
            RsResults::UA(x) => {
                x.get_molecule(&self.name)
                    .unwrap()
                    .convergence()
                    .map(|_| Convergence {
                        results: self.results.clone(),
                        molecule: self.name.clone(),
                    })
            }
        }
    }
}

/// Results of the analysis for a single atom type.
///
/// Provides access to per-atom type order parameters, order maps, and the bond results
/// associated with this atom type.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct AtomResults {
    results: Arc<RsResults>,
    molecule: String,
    atom: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl AtomResults {
    /// Get the type of the atom for which these results were calculated.
    ///
    /// Returns
    /// -------
    /// AtomType
    ///     The atom type for which these results were calculated.
    #[gen_stub(override_return_type(type_repr = "gorder.AtomType"))]
    pub fn atom(&self) -> AtomType {
        match self.results.as_ref() {
            RsResults::AA(_) => AtomType(self.get_atom_aa_results().atom().clone()),
            RsResults::UA(_) => AtomType(self.get_atom_ua_results().atom().clone()),
            RsResults::CG(_) => unreachable!(
                "FATAL GORDER ERROR | AtomResults::atom | AtomResults should not exist for CG."
            ),
        }
    }

    /// Get the name of the molecule type for this atom type.
    ///
    /// Returns
    /// -------
    /// str
    ///     Name of the molecule containing this atom type.
    pub fn molecule(&self) -> String {
        self.molecule.clone()
    }

    /// Get the results for each bond type of this atom type.
    ///
    /// Returns
    /// -------
    /// List[BondResults]
    ///     Results for all bond types involving this atom type.
    pub fn bonds(&self) -> Vec<BondResults> {
        match self.results.as_ref() {
            RsResults::AA(_) => self
                .get_atom_aa_results()
                .bonds()
                .map(|bond| BondResults::new(self.results.clone(), bond, self.molecule.clone()))
                .collect(),
            RsResults::UA(_) => self
                .get_atom_ua_results()
                .bonds()
                .enumerate()
                .map(|(hydrogen, _)| {
                    BondResults::new_ua(
                        self.results.clone(),
                        self.atom,
                        hydrogen,
                        self.molecule.clone(),
                    )
                })
                .collect(),
            RsResults::CG(_) => unreachable!(
                "FATAL GORDER ERROR | AtomResults::bonds | AtomResults should not exist for CG."
            ),
        }
    }

    /// Get the results for a bond types between this atom type and a hydrogen type (AA) or virtual hydrogen type (UA).
    ///
    /// Parameters
    /// ----------
    /// relative_index : int
    ///     Relative index (zero-based) of the bonded hydrogen atom type.
    ///
    /// Returns
    /// -------
    /// BondResults
    ///     Results for the specified bond type.
    ///
    /// Raises
    /// ------
    /// APIError
    ///     If the specified (virtual) hydrogen type does not exist.
    pub fn get_bond(&self, relative_index: usize) -> PyResult<BondResults> {
        match self.results.as_ref() {
            RsResults::AA(_) => self
                .get_atom_aa_results()
                .get_bond(relative_index)
                .ok_or_else(|| {
                    APIError::new_err(
                        "heavy atom is not bonded to hydrogen with the given relative index",
                    )
                })
                .map(|bond| BondResults::new(self.results.clone(), bond, self.molecule.clone())),
            RsResults::UA(_) => self
                .get_atom_ua_results()
                .bonds()
                .nth(relative_index)
                .ok_or_else(|| {
                    APIError::new_err("carbon does not have a virtual hydrogen with this index")
                })
                .map(|_| {
                    BondResults::new_ua(
                        self.results.clone(),
                        self.atom,
                        relative_index,
                        self.molecule.clone(),
                    )
                }),
            RsResults::CG(_) => unreachable!(
                "FATAL GORDER ERROR | AtomResults::get_bond | AtomResults should not exist for CG."
            ),
        }
    }

    /// Get order parameters calculated for this atom type.
    ///
    /// Returns
    /// -------
    /// OrderCollection
    ///     Collection of order parameters for this atom type.
    pub fn order(&self) -> OrderCollection {
        OrderCollection {
            results: self.results.clone(),
            molecule: Some(self.molecule.clone()),
            identifier: OrderIdentifier::Atom(self.atom),
        }
    }

    /// Get order maps calculated for this atom type.
    ///
    /// Returns
    /// -------
    /// OrderMapsCollection
    ///     Collection of order maps for this atom type.
    pub fn ordermaps(&self) -> OrderMapsCollection {
        OrderMapsCollection {
            results: self.results.clone(),
            molecule: Some(self.molecule.clone()),
            identifier: OrderIdentifier::Atom(self.atom),
        }
    }
}

impl AtomResults {
    /// Helper method for obtaining reference to the results for this AA atom.
    fn get_atom_aa_results(&self) -> &RsAtomResults {
        match self.results.as_ref() {
            RsResults::AA(x) => x
                .get_molecule(&self.molecule)
                .unwrap()
                .get_atom(self.atom)
                .unwrap(),
            RsResults::CG(_) | RsResults::UA(_) => unreachable!(
                "FATAL GORDER ERROR | AtomResults::get_atom_results | Results should be atomistic."
            ),
        }
    }

    /// Helper method for obtaining reference to the results for this UA atom.
    fn get_atom_ua_results(&self) -> &RsUAAtomResults {
        match self.results.as_ref() {
            RsResults::AA(_) | RsResults::CG(_) => unreachable!("FATAL GORDER ERROR | AtomResults::get_atom_ua_results | Results should be united-atom."),
            RsResults::UA(x) => x.get_molecule(&self.molecule).unwrap().get_atom(self.atom).unwrap(),
        }
    }
}

/// Results of the analysis for a single bond type.
///
/// Provides access to the molecule name, the atom types involved (if available),
/// and the order parameters and order maps for this bond type.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct BondResults {
    results: Arc<RsResults>,
    molecule: String,
    // Relative indices of the involved atoms (CG, AA) OR
    // relative index of the involved atom and the bond index (UA).
    bond: (usize, usize),
}

#[gen_stub_pymethods]
#[pymethods]
impl BondResults {
    /// Get the name of the molecule type for this bond type.
    ///
    /// Returns
    /// -------
    /// str
    ///     Name of the molecule containing this bond type.
    pub fn molecule(&self) -> String {
        self.molecule.to_owned()
    }

    /// Get the atom types involved in this bond type.
    ///
    /// Returns
    /// -------
    /// Tuple[AtomType, AtomType]
    ///     The two atom types forming this bond type.
    ///
    /// Raises
    /// ------
    /// APIError
    ///     If the bond type is a virtual united-atom bond type (UA), where only one real atom type exists.
    #[gen_stub(override_return_type(
        type_repr = "builtins.tuple[gorder.AtomType, gorder.AtomType]"
    ))]
    pub fn atoms(&self) -> Result<(AtomType, AtomType), PyErr> {
        match self.results.as_ref() {
            RsResults::AA(_) | RsResults::CG(_) => {
                let atoms = self.get_bond_results().atoms();
                Ok((AtomType(atoms.0.clone()), AtomType(atoms.1.clone())))
            }
            RsResults::UA(_) => Err(APIError::new_err(
                "cannot access information about atoms in a virtual united-atom bond; the bond only involves one real atom"))
        }
    }

    /// Get order parameters calculated for this bond type.
    ///
    /// Returns
    /// -------
    /// OrderCollection
    ///     Collection of order parameters for this bond type.
    pub fn order(&self) -> OrderCollection {
        let identifier = match self.results.as_ref() {
            RsResults::AA(_) | RsResults::CG(_) => OrderIdentifier::Bond(self.bond.0, self.bond.1),
            RsResults::UA(_) => OrderIdentifier::VirtualBond(self.bond.0, self.bond.1),
        };

        OrderCollection {
            results: self.results.clone(),
            molecule: Some(self.molecule.clone()),
            identifier,
        }
    }

    /// Get order maps calculated for this bond type.
    ///
    /// Returns
    /// -------
    /// OrderMapsCollection
    ///     Collection of order maps for this bond type.
    pub fn ordermaps(&self) -> OrderMapsCollection {
        let identifier = match self.results.as_ref() {
            RsResults::AA(_) | RsResults::CG(_) => OrderIdentifier::Bond(self.bond.0, self.bond.1),
            RsResults::UA(_) => OrderIdentifier::VirtualBond(self.bond.0, self.bond.1),
        };

        OrderMapsCollection {
            results: self.results.clone(),
            molecule: Some(self.molecule.clone()),
            identifier,
        }
    }
}

impl BondResults {
    /// Create a new BondResults wrapper.
    fn new(all_results: Arc<RsResults>, bond_results: &RsBondResults, molecule: String) -> Self {
        BondResults {
            results: all_results,
            molecule,
            bond: (
                bond_results.atoms().0.relative_index(),
                bond_results.atoms().1.relative_index(),
            ),
        }
    }

    /// Create a new BondResults wrapper for the united-atom bond.
    fn new_ua(
        all_results: Arc<RsResults>,
        carbon_index: usize,
        hydrogen_index: usize,
        molecule: String,
    ) -> Self {
        BondResults {
            results: all_results,
            molecule,
            bond: (carbon_index, hydrogen_index),
        }
    }

    /// Helper method for obtaining reference to the results for this bond.
    /// Panics, if used for UA BondResults.
    fn get_bond_results(&self) -> &RsBondResults {
        match self.results.as_ref() {
            RsResults::AA(x) => x
                .get_molecule(&self.molecule)
                .unwrap()
                .get_bond(self.bond.0, self.bond.1)
                .unwrap(),
            RsResults::CG(x) => x
                .get_molecule(&self.molecule)
                .unwrap()
                .get_bond(self.bond.0, self.bond.1)
                .unwrap(),
            RsResults::UA(_) => unreachable!("FATAL GORDER ERROR | BondResults::get_bond_results | This method cannot be used for united-atom results."),
        }
    }
}

/// Helper enum for identifying leaflet for which order parameters / ordermaps should be collected.
#[derive(Debug, Clone, Copy)]
enum Leaflet {
    Upper,
    Lower,
    Total,
}

impl Leaflet {
    #[inline]
    fn get_order(&self, collection: &RsOrderCollection) -> Option<Order> {
        match self {
            Self::Upper => collection.upper().map(Order),
            Self::Lower => collection.lower().map(Order),
            Self::Total => collection.total().map(Order),
        }
    }

    #[inline]
    fn get_ordermap<'a>(&self, collection: &'a RsMapsCollection) -> Option<&'a GridMapF32> {
        match self {
            Self::Upper => collection.upper().as_ref(),
            Self::Lower => collection.lower().as_ref(),
            Self::Total => collection.total().as_ref(),
        }
    }
}

/// Helper enum for identifying bond or
/// atom for which order parameters / ordermaps should be collected.
#[derive(Debug, Clone)]
enum OrderIdentifier {
    Average,
    Bond(usize, usize),
    Atom(usize),
    VirtualBond(usize, usize),
}

impl OrderIdentifier {
    #[inline]
    fn get_order_aa(&self, mol_results: &AAMoleculeResults, leaflet: Leaflet) -> Option<Order> {
        match self {
            Self::Average => leaflet.get_order(mol_results.average_order()),
            Self::Bond(x, y) => leaflet.get_order(mol_results.get_bond(*x, *y)?.order()),
            Self::Atom(x) => leaflet.get_order(mol_results.get_atom(*x)?.order()),
            Self::VirtualBond(_, _) => unreachable!("FATAL GORDER ERROR | OrderIdentifier::get_order_aa | Virtual bond identifier cannot be used for AA."),
        }
    }

    #[inline]
    fn get_order_cg(&self, mol_results: &CGMoleculeResults, leaflet: Leaflet) -> Option<Order> {
        match self {
            Self::Average => leaflet.get_order(mol_results.average_order()),
            Self::Bond(x, y) => leaflet.get_order(mol_results.get_bond(*x, *y)?.order()),
            Self::Atom(_) => unreachable!("FATAL GORDER ERROR | OrderIdentifier::get_order_cg | Atom identifier cannot be used for CG."),
            Self::VirtualBond(_, _) => unreachable!("FATAL GORDER ERROR | OrderIdentifier::get_order_cg | Virtual bond identifier cannot be used for CG."),
        }
    }

    #[inline]
    fn get_order_ua(&self, mol_results: &UAMoleculeResults, leaflet: Leaflet) -> Option<Order> {
        match self {
            Self::Average => leaflet.get_order(mol_results.average_order()),
            Self::Bond(_, _) => unreachable!("FATAL GORDER ERROR | OrderIdentifier::get_order_ua | Bond identifier cannot be used for UA."),
            Self::Atom(x) => leaflet.get_order(mol_results.get_atom(*x)?.order()),
            Self::VirtualBond(x, y) => leaflet.get_order(mol_results.get_atom(*x)?.bonds().nth(*y)?.order()),
        }
    }

    #[inline]
    fn get_ordermap_aa<'a>(
        &self,
        mol_results: &'a AAMoleculeResults,
        leaflet: Leaflet,
    ) -> Option<&'a GridMapF32> {
        match self {
            Self::Average => leaflet.get_ordermap(mol_results.average_ordermaps()),
            Self::Bond(x, y) => leaflet.get_ordermap(mol_results.get_bond(*x, *y)?.ordermaps()),
            Self::Atom(x) => leaflet.get_ordermap(mol_results.get_atom(*x)?.ordermaps()),
            Self::VirtualBond(_, _) => {
                unreachable!("FATAL GORDER ERROR | OrderIdentifier::get_ordermap_aa | Virtual bond identifier cannot be used for AA.")
            }
        }
    }

    #[inline]
    fn get_ordermap_cg<'a>(
        &self,
        mol_results: &'a CGMoleculeResults,
        leaflet: Leaflet,
    ) -> Option<&'a GridMapF32> {
        match self {
            Self::Average => leaflet.get_ordermap(mol_results.average_ordermaps()),
            Self::Bond(x, y) => {
                leaflet.get_ordermap(mol_results.get_bond(*x, *y)?.ordermaps())
            }
            Self::Atom(_) => panic!("FATAL GORDER ERROR | OrderIdentifier::get_ordermap_cg | Atom identifier cannot be used for  CG."),
            Self::VirtualBond(_, _) => {
                unreachable!("FATAL GORDER ERROR | OrderIdentifier::get_ordermap_cg | Virtual bond identifier cannot be used for CG.")
            }
        }
    }

    #[inline]
    fn get_ordermap_ua<'a>(
        &self,
        mol_results: &'a UAMoleculeResults,
        leaflet: Leaflet,
    ) -> Option<&'a GridMapF32> {
        match self {
            Self::Average => leaflet.get_ordermap(mol_results.average_ordermaps()),
            Self::Bond(_, _) => unreachable!("FATAL GORDER ERROR | OrderIdentifier::get_ordermap_ua | Bond identifier cannot be used for UA."),
            Self::Atom(x) => leaflet.get_ordermap(mol_results.get_atom(*x)?.ordermaps()),
            Self::VirtualBond(x, y) => leaflet.get_ordermap(mol_results.get_atom(*x)?.bonds().nth(*y)?.ordermaps()),
        }
    }
}

/// Order parameters for a single object (atom type, bond type, molecule type, system)
/// calculated for the full membrane, the upper leaflet, and the lower leaflet.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct OrderCollection {
    results: Arc<RsResults>,
    molecule: Option<String>, // `None` for average order for the entire system
    identifier: OrderIdentifier,
}

#[gen_stub_pymethods]
#[pymethods]
impl OrderCollection {
    /// Get the order parameter calculated from the whole membrane.
    ///
    /// Returns
    /// -------
    /// Order
    ///     Order parameter for the whole membrane. Returns `None` if not available.
    pub fn total(&self) -> Option<Order> {
        self.get_order(Leaflet::Total)
    }

    /// Get the order parameter calculated from the upper leaflet.
    ///
    /// Returns
    /// -------
    /// Order
    ///     Order parameter for the upper leaflet. Returns `None` if not available.
    pub fn upper(&self) -> Option<Order> {
        self.get_order(Leaflet::Upper)
    }

    /// Get the order parameter calculated from the lower leaflet.
    ///
    /// Returns
    /// -------
    /// Order
    ///     Order parameter for the lower leaflet. Returns `None` if not available.
    pub fn lower(&self) -> Option<Order> {
        self.get_order(Leaflet::Lower)
    }
}

impl OrderCollection {
    /// Helper method for getting order parameters for a leaflet.
    fn get_order(&self, leaflet: Leaflet) -> Option<Order> {
        match &self.molecule {
            None => match self.results.as_ref() {
                RsResults::AA(results) => leaflet.get_order(results.average_order()),
                RsResults::CG(results) => leaflet.get_order(results.average_order()),
                RsResults::UA(results) => leaflet.get_order(results.average_order()),
            },
            Some(mol) => match self.results.as_ref() {
                RsResults::AA(results) => self
                    .identifier
                    .get_order_aa(results.get_molecule(mol)?, leaflet),
                RsResults::CG(results) => self
                    .identifier
                    .get_order_cg(results.get_molecule(mol)?, leaflet),
                RsResults::UA(results) => self
                    .identifier
                    .get_order_ua(results.get_molecule(mol)?, leaflet),
            },
        }
    }
}

/// Single order parameter value, optionally with its estimated error.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct Order(pub(crate) RsOrder);

#[gen_stub_pymethods]
#[pymethods]
impl Order {
    /// Get the value of the order parameter (mean from the analyzed frames).
    ///
    /// Returns
    /// -------
    /// float
    ///     Mean value of the order parameter.
    pub fn value(&self) -> f32 {
        self.0.value()
    }

    /// Get the estimated error for this order parameter.
    ///
    /// Returns
    /// -------
    /// float
    ///     Estimated error of the order parameter, or `None` if not available.
    pub fn error(&self) -> Option<f32> {
        self.0.error()
    }
}

/// Order maps for a single object (atom type, bond type, molecule type, system)
/// calculated for the full membrane, the upper leaflet, and the lower leaflet.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
#[derive(Clone)]
pub struct OrderMapsCollection {
    results: Arc<RsResults>,
    molecule: Option<String>, // `None` for average order for the entire system
    identifier: OrderIdentifier,
}

#[gen_stub_pymethods]
#[pymethods]
impl OrderMapsCollection {
    /// Get the order map calculated from the whole membrane.
    ///
    /// Returns
    /// -------
    /// Map
    ///     Order map for the whole membrane, or `None` if not available.
    pub fn total(&self) -> Option<Map> {
        self.get_ordermap(Leaflet::Total).map(|_| Map {
            collection: self.clone(),
            leaflet: Leaflet::Total,
        })
    }

    /// Get the order map calculated from the upper leaflet.
    ///
    /// Returns
    /// -------
    /// Map
    ///     Order map for the upper leaflet, or `None` if not available.
    pub fn upper(&self) -> Option<Map> {
        self.get_ordermap(Leaflet::Upper).map(|_| Map {
            collection: self.clone(),
            leaflet: Leaflet::Upper,
        })
    }

    /// Get the order map calculated from the lower leaflet.
    ///
    /// Returns
    /// -------
    /// Map
    ///     Order map for the lower leaflet, or `None` if not available.
    pub fn lower(&self) -> Option<Map> {
        self.get_ordermap(Leaflet::Lower).map(|_| Map {
            collection: self.clone(),
            leaflet: Leaflet::Lower,
        })
    }
}

impl OrderMapsCollection {
    /// Helper method for getting ordermaps for a leaflet.
    fn get_ordermap(&self, leaflet: Leaflet) -> Option<&GridMapF32> {
        match &self.molecule {
            None => match self.results.as_ref() {
                RsResults::AA(results) => leaflet.get_ordermap(results.average_ordermaps()),
                RsResults::CG(results) => leaflet.get_ordermap(results.average_ordermaps()),
                RsResults::UA(results) => leaflet.get_ordermap(results.average_ordermaps()),
            },
            Some(mol) => match self.results.as_ref() {
                RsResults::AA(results) => self
                    .identifier
                    .get_ordermap_aa(results.get_molecule(mol)?, leaflet),
                RsResults::CG(results) => self
                    .identifier
                    .get_ordermap_cg(results.get_molecule(mol)?, leaflet),
                RsResults::UA(results) => self
                    .identifier
                    .get_ordermap_ua(results.get_molecule(mol)?, leaflet),
            },
        }
    }
}

/// Map of order parameters.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct Map {
    collection: OrderMapsCollection,
    leaflet: Leaflet,
}

#[gen_stub_pymethods]
#[pymethods]
impl Map {
    /// Get the span of the map along the first dimension of the map.
    ///
    /// Returns
    /// -------
    /// Tuple[float, float]
    ///     Minimum and maximum coordinates along the first dimension of the map.
    pub fn span_x(&self) -> (f32, f32) {
        let ordermap = self.collection.get_ordermap(self.leaflet).unwrap();
        ordermap.span_x()
    }

    /// Get the span of the map along the second dimension of the map.
    ///
    /// Returns
    /// -------
    /// Tuple[float, float]
    ///     Minimum and maximum coordinates along the second dimension of the map.
    pub fn span_y(&self) -> (f32, f32) {
        let ordermap = self.collection.get_ordermap(self.leaflet).unwrap();
        ordermap.span_y()
    }

    /// Get the dimensions of a single grid tile in the map.
    ///
    /// Returns
    /// -------
    /// Tuple[float, float]
    ///     Width and height of a single tile of the map.
    pub fn tile_dim(&self) -> (f32, f32) {
        let ordermap = self.collection.get_ordermap(self.leaflet).unwrap();
        ordermap.tile_dim()
    }

    /// Get the order parameter at the specified coordinates.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///     Coordinate along the first dimension of the map.
    /// y : float
    ///     Coordinate along the second dimension of the map.
    ///
    /// Returns
    /// -------
    /// float
    ///     Order parameter at the given coordinates, or `None` if out of bounds.
    pub fn get_at(&self, x: f32, y: f32) -> Option<f32> {
        let ordermap = self.collection.get_ordermap(self.leaflet).unwrap();
        ordermap.get_at_convert(x, y)
    }

    /// Extract the order map into NumPy arrays.
    ///
    /// Returns
    /// -------
    /// Tuple[np.ndarray[float32], np.ndarray[float32], np.ndarray[float32]]
    ///     A tuple of NumPy arrays:
    ///     - The first array (1D) contains positions of the grid tiles along the first dimension of the map (typically `x`).
    ///     - The second array (1D) contains positions of the grid tiles along the second dimension of the map (typically `y`).
    ///     - The third array (2D) contains the calculated order parameters.
    #[allow(clippy::type_complexity)]
    pub fn extract(
        &self,
        py: Python<'_>,
    ) -> (
        Py<numpy::PyArray1<f32>>,
        Py<numpy::PyArray1<f32>>,
        Py<numpy::PyArray2<f32>>,
    ) {
        let ordermap = self.collection.get_ordermap(self.leaflet).unwrap();
        (
            Self::x_positions(ordermap, py),
            Self::y_positions(ordermap, py),
            Self::values_array(ordermap, py),
        )
    }
}

impl Map {
    fn x_positions(ordermap: &GridMapF32, py: Python<'_>) -> Py<numpy::PyArray1<f32>> {
        let start = ordermap.span_x().0;
        let step = ordermap.tile_dim().0;
        let n = ordermap.n_tiles_x();

        let x_coords: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();

        numpy::PyArray1::from_vec(py, x_coords).into()
    }

    fn y_positions(ordermap: &GridMapF32, py: Python<'_>) -> Py<numpy::PyArray1<f32>> {
        let start = ordermap.span_y().0;
        let step = ordermap.tile_dim().1;
        let n = ordermap.n_tiles_y();

        let y_coords: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();

        numpy::PyArray1::from_vec(py, y_coords).into()
    }

    fn values_array(ordermap: &GridMapF32, py: Python<'_>) -> Py<numpy::PyArray2<f32>> {
        let mut converted_values: Vec<f32> =
            ordermap.extract_convert().map(|(_, _, val)| val).collect();

        let n_x = ordermap.n_tiles_x();
        let n_y = ordermap.n_tiles_y();

        let mut values_2d = Vec::with_capacity(n_x);
        for chunk in converted_values.chunks_mut(n_y) {
            values_2d.push(chunk.to_vec());
        }

        numpy::PyArray2::from_vec2(py, &values_2d).expect(
            "FATAL GORDER ERROR | OrderMap::values_array | Could not convert ndarray to numpy array.").into()
    }
}

/// Stores information about the convergence of order parameter calculations
/// for a single molecule.
///
/// Provides cumulative averages over time for the full membrane and for
/// individual leaflets.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct Convergence {
    results: Arc<RsResults>,
    molecule: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl Convergence {
    /// Get the indices of trajectory frames for which order parameters were calculated.
    ///
    /// The first analyzed frame is assigned an index of 1. For example,
    /// if the analysis starts at 200 ns, the frame at or just after 200 ns
    /// is indexed as 1.
    ///
    /// Returns
    /// -------
    /// List[int]
    ///     Indices of the analyzed frames.
    pub fn frames(&self) -> Vec<usize> {
        self.get_convergence().frames().clone()
    }

    /// Get cumulative average order parameters for the entire membrane.
    ///
    /// Each element represents the cumulative average up to that frame.
    ///
    /// Returns
    /// -------
    /// List[float]
    ///     Cumulative averages for the full membrane, or `None` if not available.
    pub fn total(&self) -> Option<Vec<f32>> {
        self.get_convergence().total().clone()
    }

    /// Get cumulative average order parameters for the upper leaflet.
    ///
    /// Returns
    /// -------
    /// List[float]
    ///     Cumulative averages for the upper leaflet, or `None` if not available.
    pub fn upper(&self) -> Option<Vec<f32>> {
        self.get_convergence().upper().clone()
    }

    /// Get cumulative average order parameters for the lower leaflet.
    ///
    /// Returns
    /// -------
    /// List[float]
    ///     Cumulative averages for the lower leaflet, or `None` if not available.
    pub fn lower(&self) -> Option<Vec<f32>> {
        self.get_convergence().lower().clone()
    }
}

impl Convergence {
    /// Helper method for obtaining reference to the Convergence structure.
    fn get_convergence(&self) -> &RsConvergence {
        match self.results.as_ref() {
            RsResults::AA(results) => results
                .get_molecule(&self.molecule)
                .unwrap()
                .convergence()
                .unwrap(),
            RsResults::CG(results) => results
                .get_molecule(&self.molecule)
                .unwrap()
                .convergence()
                .unwrap(),
            RsResults::UA(results) => results
                .get_molecule(&self.molecule)
                .unwrap()
                .convergence()
                .unwrap(),
        }
    }
}

/// Stores collected leaflet classification data.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct LeafletsData {
    pub(crate) results: Arc<RsResults>,
}

#[gen_stub_pymethods]
#[pymethods]
impl LeafletsData {
    /// Get leaflet classification data for the specified molecule type.
    ///
    /// Parameters
    /// ----------
    /// molecule : str
    ///     Name of the molecule type to query.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_frames, n_molecules), dtype=uint8
    ///     A 2D array where rows correspond to analyzed trajectory frames and columns to
    ///     individual molecules. Values are `1` for molecules in the upper leaflet
    ///     and `0` for molecules in the lower leaflet. Returns `None` if no leaflet
    ///     classification data exists for the specified molecule type.
    ///
    /// Notes
    /// -----
    /// This is a potentially expensive operation, as it involves copying and
    /// converting from the internal Rust representation into a NumPy array.
    pub fn get_molecule(&self, molecule: &str, py: Python<'_>) -> Option<Py<numpy::PyArray2<u8>>> {
        let molecule_data = match self.results.as_ref() {
            RsResults::AA(x) => x.leaflets_data().as_ref().unwrap().get_molecule(molecule),
            RsResults::CG(x) => x.leaflets_data().as_ref().unwrap().get_molecule(molecule),
            RsResults::UA(x) => x.leaflets_data().as_ref().unwrap().get_molecule(molecule),
        }?;

        let converted_data = convert_leaflets(molecule_data);
        Some(numpy::PyArray2::from_vec2(py, &converted_data).expect(
                "FATAL GORDER ERROR | LeafletsData::get_molecule | Could not convert Vec<Vec> to numpy array.").into())
    }

    /// Get the indices of trajectory frames for which leaflet classification was performed.
    ///
    /// The first analyzed frame is assigned an index of 1. For example,
    /// if the analysis starts at 200 ns, the frame at or just after 200 ns
    /// is indexed as 1.
    ///
    /// Returns
    /// -------
    /// List[int]
    ///     Indices of the frames for which leaflet classification was performed.
    pub fn frames(&self) -> Vec<usize> {
        match self.results.as_ref() {
            RsResults::AA(x) => x.leaflets_data().as_ref().unwrap().frames().clone(),
            RsResults::CG(x) => x.leaflets_data().as_ref().unwrap().frames().clone(),
            RsResults::UA(x) => x.leaflets_data().as_ref().unwrap().frames().clone(),
        }
    }
}

/// Helper function for converting from Leaflet to u8.
fn leaflet_to_u8(l: &RsLeaflet) -> u8 {
    match l {
        RsLeaflet::Upper => 1,
        RsLeaflet::Lower => 0,
    }
}

/// Helper function for converting leaflets data.
fn convert_leaflets(input: &[Vec<RsLeaflet>]) -> Vec<Vec<u8>> {
    input
        .iter()
        .map(|inner| inner.iter().map(leaflet_to_u8).collect())
        .collect()
}

/// Stores collected membrane normals.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.results")]
pub struct NormalsData {
    pub(crate) results: Arc<RsResults>,
}

#[gen_stub_pymethods]
#[pymethods]
impl NormalsData {
    /// Get collected membrane normals for the specified molecule type.
    ///
    /// Parameters
    /// ----------
    /// molecule : str
    ///     Name of the molecule type to query.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_frames, n_molecules, 3), dtype=float32
    ///     A 2D array where rows correspond to analyzed trajectory frames and columns to
    ///     individual molecules. Each entry is a 3D vector.
    ///     If the membrane normal was not calculated for a given molecule in a frame,
    ///     the corresponding vector is (NaN, NaN, NaN).
    ///     Returns `None` if no membrane normals have been collected.
    ///
    /// Notes
    /// -----
    /// This is a potentially expensive operation, as it involves copying and
    /// converting from the internal Rust representation into a NumPy array.
    pub fn get_molecule(&self, molecule: &str, py: Python<'_>) -> Option<Py<numpy::PyArray3<f32>>> {
        let molecule_data = match self.results.as_ref() {
            RsResults::AA(x) => x.normals_data().as_ref().unwrap().get_molecule(molecule),
            RsResults::CG(x) => x.normals_data().as_ref().unwrap().get_molecule(molecule),
            RsResults::UA(x) => x.normals_data().as_ref().unwrap().get_molecule(molecule),
        }?;

        let converted_data = convert_normals(molecule_data);
        Some(numpy::PyArray3::from_vec3(py, &converted_data).expect(
                "FATAL GORDER ERROR | NormalsData::get_molecule | Could not convert Vec<Vec<Vec>> to numpy array.").into())
    }

    /// Get the indices of trajectory frames which were analyzed.
    ///
    /// The first analyzed frame is assigned an index of 1. For example,
    /// if the analysis starts at 200 ns, the frame at or just after 200 ns
    /// is indexed as 1.
    ///
    /// Returns
    /// -------
    /// List[int]
    ///     Indices of the analyzed frames.
    pub fn frames(&self) -> Vec<usize> {
        match self.results.as_ref() {
            RsResults::AA(x) => x.normals_data().as_ref().unwrap().frames().clone(),
            RsResults::CG(x) => x.normals_data().as_ref().unwrap().frames().clone(),
            RsResults::UA(x) => x.normals_data().as_ref().unwrap().frames().clone(),
        }
    }
}

fn convert_normals(input: &[Vec<Vector3D>]) -> Vec<Vec<Vec<f32>>> {
    input
        .iter()
        .map(|inner| inner.iter().map(|vec| vec![vec.x, vec.y, vec.z]).collect())
        .collect()
}
