// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for calculations of order parameters from united-atom simulations.

use std::ops::Add;
use std::{f32::consts::PI, ops::Deref};

use super::common::macros::group_name;
use super::common::GORDER_GROUP_PREFIX;
use super::{
    calc_sch, geometry::GeometrySelection, leaflets::MoleculeLeafletClassification,
    normal::MoleculeMembraneNormal, pbc::PBCHandler, topology::bond::VirtualBondType,
};
use crate::analysis::common::{
    prepare_geometry_selection, prepare_membrane_normal_calculation, read_trajectory,
};
use crate::analysis::index::read_ndx_file;
use crate::analysis::pbc::{NoPBC, PBC3D};
use crate::analysis::structure;
use crate::analysis::topology::bond::BondLike;
use crate::analysis::topology::classify::MoleculesClassifier;
use crate::analysis::topology::SystemTopology;
use crate::errors::TopologyError;
use crate::input::{Analysis, OrderMap};
use crate::prelude::AnalysisResults;
use crate::presentation::uaresults::UAOrderResults;
use crate::presentation::OrderResults;
use crate::{errors::AnalysisError, prelude::AtomType, PANIC_MESSAGE};
use groan_rs::prelude::Atom;
use groan_rs::{prelude::Vector3D, system::System};
use nalgebra::{Rotation3, Unit};

/// Tetrahedral angle.
const TETRAHEDRAL_ANGLE: f32 = 1.910633;
/// Half of the tetrahedral angle.
const TETRAHEDRAL_ANGLE_HALF: f32 = 0.9553165;
/// Length of the C-H bond in nm.
const BOND_LENGTH: f32 = 0.109;
/// Rotation angle used for predicting hydrogens in the CH3 group.
const CH3_ANGLE: f32 = 2.0943952;

/// Calculate the united-atom order parameters.
pub(super) fn analyze_united(
    analysis: Analysis,
) -> Result<AnalysisResults, Box<dyn std::error::Error + Send + Sync>> {
    let mut system = structure::read_structure_and_topology(&analysis)?;

    if let Some(ndx) = analysis.index() {
        read_ndx_file(&mut system, ndx)?;
    }

    prepare_groups(&mut system, &analysis)?;

    // prepare system for leaflet classification
    if let Some(leaflet) = analysis.leaflets() {
        leaflet.prepare_system(&mut system)?;
    }

    // prepare system for dynamic normal calculation, if needed
    prepare_membrane_normal_calculation(analysis.membrane_normal(), &mut system)?;

    // prepare system for geometry selection
    let geom = prepare_geometry_selection(
        analysis.geometry().as_ref(),
        &mut system,
        analysis.handle_pbc(),
    )?;
    geom.info();

    // get the relevant molecules
    macro_rules! classify_molecules_with_pbc {
        ($pbc:expr) => {
            MoleculesClassifier::classify(&system, &analysis, $pbc)
        };
    }

    let molecules = match analysis.handle_pbc() {
        true => classify_molecules_with_pbc!(&PBC3D::from_system(&system))?,
        false => classify_molecules_with_pbc!(&NoPBC)?,
    };

    // check that there are molecules to analyze
    if molecules.n_molecule_types() == 0 {
        return Ok(AnalysisResults::UA(UAOrderResults::empty(analysis)));
    }

    let mut data = SystemTopology::new(
        &system,
        molecules,
        analysis.estimate_error().clone(),
        analysis.step(),
        analysis.n_threads(),
        geom,
        analysis.leaflets().as_ref(),
        analysis.handle_pbc(),
    );

    data.info()?;

    // finalize the manual leaflet classification
    if let Some(classification) = analysis.leaflets() {
        data.finalize_manual_leaflet_classification(classification)?;
    }

    // finalize the membrane normal specification
    data.finalize_manual_membrane_normals(analysis.membrane_normal())?;

    if let Some(error_estimation) = analysis.estimate_error() {
        error_estimation.info();
    }

    let result = read_trajectory(
        &system,
        data,
        analysis.trajectory(),
        analysis.n_threads(),
        analysis.begin(),
        analysis.end(),
        analysis.step(),
        analysis.silent(),
    )?;

    result.validate_run(analysis.step())?;
    result.log_total_analyzed_frames();

    // print basic info about error estimation
    result.error_info()?;

    Ok(AnalysisResults::UA(
        result.convert::<UAOrderResults>(analysis),
    ))
}

/// Prepare groups for UA analysis.
fn prepare_groups(system: &mut System, analysis: &Analysis) -> Result<(), TopologyError> {
    // create groups
    if let Some(saturated) = analysis.saturated() {
        super::common::create_group(system, "Saturated", saturated)?;

        colog_info!(
            "Detected {} saturated carbons using a query '{}'.",
            system
                .group_get_n_atoms(group_name!("Saturated"))
                .expect(PANIC_MESSAGE),
            saturated,
        );
    }

    if let Some(unsaturated) = analysis.unsaturated() {
        super::common::create_group(system, "Unsaturated", unsaturated)?;

        colog_info!(
            "Detected {} unsaturated carbons using a query '{}'.",
            system
                .group_get_n_atoms(group_name!("Unsaturated"))
                .expect(PANIC_MESSAGE),
            unsaturated,
        );

        // check for overlap
        if let Some(sat) = analysis.saturated() {
            super::common::check_groups_overlap(
                system,
                "Saturated",
                sat,
                "Unsaturated",
                unsaturated,
            )?;
        }
    }

    // create a merged saturated-unsaturated group
    match (
        analysis.saturated().is_some(),
        analysis.unsaturated().is_some(),
    ) {
        (true, true) => super::common::create_group(
            system,
            "SatUnsat",
            &format!(
                "{}Saturated {}Unsaturated",
                GORDER_GROUP_PREFIX, GORDER_GROUP_PREFIX
            ),
        )?,
        (true, false) => super::common::create_group(system, "SatUnsat", group_name!("Saturated"))?,
        (false, true) => {
            super::common::create_group(system, "SatUnsat", group_name!("Unsaturated"))?
        }
        (false, false) => return Err(TopologyError::NoUACarbons),
    }

    if let Some(ignore) = analysis.ignore() {
        super::common::create_group(system, "Ignore", ignore)?;

        colog_info!(
            "Detected {} atoms to ignore using a query '{}'.",
            system
                .group_get_n_atoms(group_name!("Ignore"))
                .expect(PANIC_MESSAGE),
            ignore,
        );

        // check for overlap
        if let Some(sat) = analysis.saturated() {
            super::common::check_groups_overlap(system, "Saturated", sat, "Ignore", ignore)?;
        }

        if let Some(unsat) = analysis.unsaturated() {
            super::common::check_groups_overlap(system, "Unsaturated", unsat, "Ignore", ignore)?;
        }
    }

    // helper group that includes all atoms that are part of the same molecule as any saturated or unsatured carbon
    // but Ignore atoms are still ignored
    // we need to include these atoms in the Master group so that positions of helper atoms are updated as well
    let query = if analysis.ignore().is_some() {
        format!(
            "(molwith {}SatUnsat) and not {}Ignore",
            GORDER_GROUP_PREFIX, GORDER_GROUP_PREFIX
        )
    } else {
        format!("molwith {}SatUnsat", GORDER_GROUP_PREFIX)
    };

    super::common::create_group(system, "Helper", &query)?;

    Ok(())
}

/// Enum representing a type of united atom.
/// Contains indices of all atoms of this type and its helper atoms.
#[derive(Debug, Clone)]
#[allow(private_interfaces)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum UAOrderAtomType {
    CH3(CH3Atom),
    CH2(CH2Atom),
    CH1Unsaturated(CH1UnsaturatedAtom),
    CH1Saturated(CH1SaturatedAtom),
}

impl UAOrderAtomType {
    /// Analyze order parameters for this atom type.
    pub(crate) fn analyze_frame<'a, Geom: GeometrySelection>(
        &mut self,
        frame: &'a System,
        leaflet_classification: &mut Option<MoleculeLeafletClassification>,
        pbc_handler: &'a impl PBCHandler<'a>,
        membrane_normal: &mut MoleculeMembraneNormal,
        frame_index: usize,
        geometry: &Geom,
    ) -> Result<(), AnalysisError> {
        macro_rules! impl_analyze_frame {
            ($enum:ident => $($variant:ident),+) => {
                match self {
                    $(
                        Self::$variant(x) => x.analyze_frame(
                            frame,
                            leaflet_classification,
                            pbc_handler,
                            membrane_normal,
                            frame_index,
                            geometry,
                        ),
                    )+
                }
            };
        }

        impl_analyze_frame!(Self => CH3, CH2, CH1Unsaturated, CH1Saturated)
    }

    /// Get the type of the atom.
    pub(crate) fn get_type(&self) -> &AtomType {
        match self {
            Self::CH3(x) => &x.atom_type,
            Self::CH2(x) => &x.atom_type,
            Self::CH1Saturated(x) => &x.atom_type,
            Self::CH1Unsaturated(x) => &x.atom_type,
        }
    }

    /// Initialize reading of a new simulation frame.
    #[inline(always)]
    pub(crate) fn init_new_frame(&mut self) {
        match self {
            Self::CH3(x) => {
                x.bonds[0].init_new_frame();
                x.bonds[1].init_new_frame();
                x.bonds[2].init_new_frame();
            }
            Self::CH2(x) => {
                x.bonds[0].init_new_frame();
                x.bonds[1].init_new_frame();
            }
            Self::CH1Saturated(x) => {
                x.bond.init_new_frame();
            }
            Self::CH1Unsaturated(x) => {
                x.bond.init_new_frame();
            }
        }
    }

    /// Get the number of molecules.
    #[inline(always)]
    pub(crate) fn n_molecules(&self) -> usize {
        match self {
            Self::CH3(x) => x.atoms.len(),
            Self::CH2(x) => x.atoms.len(),
            Self::CH1Saturated(x) => x.atoms.len(),
            Self::CH1Unsaturated(x) => x.atoms.len(),
        }
    }

    /// Get basic information about timewise analysis.
    #[inline(always)]
    pub(crate) fn get_timewise_info(&self, n_blocks: usize) -> Option<(usize, usize)> {
        match self {
            Self::CH3(x) => x.bonds[0].get_timewise_info(n_blocks),
            Self::CH2(x) => x.bonds[0].get_timewise_info(n_blocks),
            Self::CH1Saturated(x) => x.bond.get_timewise_info(n_blocks),
            Self::CH1Unsaturated(x) => x.bond.get_timewise_info(n_blocks),
        }
    }

    /// Copy the virtual bonds of the atom into a vector.
    pub(crate) fn extract_bonds(&self) -> Vec<VirtualBondType> {
        match self {
            Self::CH3(x) => x.bonds.to_vec(),
            Self::CH2(x) => x.bonds.to_vec(),
            Self::CH1Saturated(x) => vec![x.bond.clone()],
            Self::CH1Unsaturated(x) => vec![x.bond.clone()],
        }
    }
}

impl Add<UAOrderAtomType> for UAOrderAtomType {
    type Output = UAOrderAtomType;

    fn add(self, rhs: UAOrderAtomType) -> Self::Output {
        match (self, rhs) {
            (Self::CH3(x), Self::CH3(y)) => Self::CH3(x + y),
            (Self::CH2(x), Self::CH2(y)) => Self::CH2(x + y),
            (Self::CH1Saturated(x), Self::CH1Saturated(y)) => Self::CH1Saturated(x + y),
            (Self::CH1Unsaturated(x), Self::CH1Unsaturated(y)) => Self::CH1Unsaturated(x + y),
            _ => panic!(
                "FATAL GORDER ERROR | UAOrderAtomType::Add | Unmatching atom types summed. {}",
                PANIC_MESSAGE
            ),
        }
    }
}

trait UAAtom<const HYDROGENS: usize, const INDICES: usize> {
    type Indices: AtomIndices<INDICES>;

    /// Predict positions of N hydrogens for molecule with the given index.
    fn predict_hydrogens<'a>(
        indices: &Self::Indices,
        frame: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; HYDROGENS], AnalysisError>;

    /// Iterate over the atom triplets/quadruplets of the UA atom.
    fn indices_iter(&self) -> impl Iterator<Item = &Self::Indices>;

    /// Get mutable reference to UA bond type with the given index.
    /// Should panic if the index is out of range.
    fn get_bond(&mut self, index: usize) -> &mut VirtualBondType;

    /// Use the predicted hydrogen positions to calculate order parameters and bond positions.
    /// If a bond is not inside the specified geometry, return `None` for it.
    fn calculate_sch<'a, Geom: GeometrySelection>(
        target: &Vector3D,
        hydrogens: [Vector3D; HYDROGENS],
        pbc: &'a impl PBCHandler<'a>,
        normal: &Vector3D,
        geometry: &Geom,
    ) -> ([Option<f32>; HYDROGENS], [Vector3D; HYDROGENS]) {
        let results: [(Option<f32>, Vector3D); HYDROGENS] = std::array::from_fn(|i| {
            let hydrogen = &hydrogens[i];
            let vec = pbc.vector_to(target, hydrogen);
            let bond_pos = hydrogen + (&vec / 2.0);
            let sch_value = pbc
                .inside(&bond_pos, geometry)
                .then(|| calc_sch(&vec, normal));

            (sch_value, bond_pos)
        });

        let sch = std::array::from_fn(|i| results[i].0);
        let bond_positions = std::array::from_fn(|i| results[i].1.clone());

        (sch, bond_positions)
    }

    /// Calculate order parameters for a single atom type in a single trajectory frame.
    fn analyze_frame<'a, Geom: GeometrySelection>(
        &mut self,
        frame: &'a System,
        leaflet_classification: &mut Option<MoleculeLeafletClassification>,
        pbc_handler: &'a impl PBCHandler<'a>,
        membrane_normal: &mut MoleculeMembraneNormal,
        frame_index: usize,
        geometry: &Geom,
    ) -> Result<(), AnalysisError> {
        let self_ptr = self as *mut Self;

        for (molecule_index, triplet) in self.indices_iter().enumerate() {
            let normal =
                membrane_normal.get_normal(frame_index, molecule_index, frame, pbc_handler)?;

            let hydrogens = Self::predict_hydrogens(triplet, frame, pbc_handler)?;
            let target_pos = triplet.get_target(frame)?;
            let (sch, bond_pos) =
                Self::calculate_sch(target_pos, hydrogens, pbc_handler, normal, geometry);

            for (i, (order, pos)) in sch.into_iter().zip(bond_pos.into_iter()).enumerate() {
                match order {
                    // safety: self lives the entire method
                    // we are modifying different part of the structure than is iterated by `triplets_iter`
                    Some(x) => unsafe { &mut *self_ptr }.get_bond(i).add_order(
                        molecule_index,
                        x,
                        &pos,
                        leaflet_classification,
                        frame_index,
                    ),
                    None => continue,
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
enum CarbonKind {
    Saturated,
    Unsaturated,
}

#[derive(Debug, Clone)]
enum UAType {
    CH3(AtomTriplet),
    CH2(AtomTriplet),
    CH1Unsat(AtomTriplet),
    CH1Sat(AtomQuadruplet),
}

impl UAOrderAtomType {
    /// Create a new `UAOrderAtomType` structure.
    pub(crate) fn new<'a>(
        system: &System,
        index: usize,
        min_index: usize,
        classify_leaflets: bool,
        ordermap: Option<&OrderMap>,
        pbc_handler: &impl PBCHandler<'a>,
        errors: bool,
    ) -> Result<Option<Self>, TopologyError> {
        // index cannot be lower than `min_index`
        if index < min_index {
            panic!(
                "FATAL GORDER ERROR | UAOrderAtomType::new | Atom index '{}' is lower than minimum index '{}'. {}",
                index, min_index, PANIC_MESSAGE
            );
        }

        let atom = system
            .get_atom(index)
            .unwrap_or_else(|_|
                panic!("FATAL GORDER ERROR | UAOrderAtomType::new | Index '{}' does not correspond to an existing atom. {}", index, PANIC_MESSAGE));

        let atom_type = AtomType::new(index - min_index, atom);
        let bond = VirtualBondType::new(classify_leaflets, ordermap, pbc_handler, errors)?;

        match Self::get_atom_type(system, atom) {
            Some(a) => match a {
                UAType::CH3(x) => Ok(Some(Self::CH3(CH3Atom {
                    atom_type,
                    atoms: vec![x],
                    bonds: [bond.clone(), bond.clone(), bond],
                }))),

                UAType::CH2(x) => Ok(Some(Self::CH2(CH2Atom {
                    atom_type,
                    atoms: vec![x],
                    bonds: [bond.clone(), bond],
                }))),

                UAType::CH1Sat(x) => Ok(Some(Self::CH1Saturated(CH1SaturatedAtom {
                    atom_type,
                    atoms: vec![x],
                    bond,
                }))),

                UAType::CH1Unsat(x) => Ok(Some(Self::CH1Unsaturated(CH1UnsaturatedAtom {
                    atom_type,
                    atoms: vec![x],
                    bond,
                }))),
            },
            None => Ok(None),
        }
    }

    /// Get the relative index of this atom type.
    #[inline(always)]
    pub(crate) fn get_relative_index(&self) -> usize {
        match self {
            UAOrderAtomType::CH3(x) => x.atom_type.relative_index(),
            UAOrderAtomType::CH2(x) => x.atom_type.relative_index(),
            UAOrderAtomType::CH1Unsaturated(x) => x.atom_type.relative_index(),
            UAOrderAtomType::CH1Saturated(x) => x.atom_type.relative_index(),
        }
    }

    /// Insert a new atom to the atom type.
    pub(crate) fn insert(&mut self, index: usize) {
        match self {
            UAOrderAtomType::CH3(x) => {
                let triplet = x.atoms.first().expect(PANIC_MESSAGE);
                x.atoms.push(Self::triplet_from_reference(triplet, index));
            }
            UAOrderAtomType::CH2(x) => {
                let triplet = x.atoms.first().expect(PANIC_MESSAGE);
                x.atoms.push(Self::triplet_from_reference(triplet, index));
            }
            UAOrderAtomType::CH1Unsaturated(x) => {
                let triplet = x.atoms.first().expect(PANIC_MESSAGE);
                x.atoms.push(Self::triplet_from_reference(triplet, index));
            }
            UAOrderAtomType::CH1Saturated(x) => {
                let quadruplet = x.atoms.first().expect(PANIC_MESSAGE);
                x.atoms
                    .push(Self::quadruplet_from_reference(quadruplet, index));
            }
        }
    }

    /// Get helper atoms based on a reference triplet and a target index.
    #[inline]
    fn triplet_from_reference(triplet: &AtomTriplet, target: usize) -> AtomTriplet {
        let (h1, t, h2) = (
            triplet.helper1 as isize,
            triplet.target as isize,
            triplet.helper2 as isize,
        );

        AtomTriplet {
            helper1: (target as isize + (h1 - t)) as usize,
            target,
            helper2: (target as isize + (h2 - t)) as usize,
        }
    }

    /// Get helper atoms based on a reference quadruplet and target index.
    #[inline]
    fn quadruplet_from_reference(quadruplet: &AtomQuadruplet, target: usize) -> AtomQuadruplet {
        let (h1, h2, h3, t) = (
            quadruplet.helper1 as isize,
            quadruplet.helper2 as isize,
            quadruplet.helper3 as isize,
            quadruplet.target as isize,
        );

        AtomQuadruplet {
            helper1: (target as isize + (h1 - t)) as usize,
            helper2: (target as isize + (h2 - t)) as usize,
            helper3: (target as isize + (h3 - t)) as usize,
            target,
        }
    }

    /// Get the type of the heavy atom. Returns None, if the atom has no hydrogens to predict.
    fn get_atom_type(system: &System, atom: &Atom) -> Option<UAType> {
        let carbon = Self::classify_carbon(system, atom.get_index());

        let bonded_atoms = atom
            .get_bonded()
            .iter()
            .filter(|&a| !system.group_isin(group_name!("Ignore"), a).unwrap_or(false))
            .collect::<Vec<usize>>();

        if bonded_atoms.len() > 4 {
            colog_warn!("Atom number {} is bonded to {} atoms which is more than the expected maximum of {} atoms.", 
                atom.get_index() + 1, bonded_atoms.len(), 4);
        }

        let missing_h = 4usize.saturating_sub(bonded_atoms.len());
        match (carbon, missing_h) {
            (_, 0) | (CarbonKind::Unsaturated, 1) => None,

            (CarbonKind::Saturated, 1) => Some(UAType::CH1Sat(AtomQuadruplet {
                helper1: bonded_atoms[0],
                helper2: bonded_atoms[1],
                helper3: bonded_atoms[2],
                target: atom.get_index(),
            })),

            (CarbonKind::Saturated, 2) => Some(UAType::CH2(AtomTriplet {
                helper1: bonded_atoms[0],
                target: atom.get_index(),
                helper2: bonded_atoms[1],
            })),

            (CarbonKind::Saturated, 3) => {
                let helper1 = bonded_atoms[0];
                let helper1_atom = system.get_atom(helper1).expect(PANIC_MESSAGE);
                let helper2 = match helper1_atom
                    .get_bonded()
                    .iter()
                    .find(|&i| i != atom.get_index())
                {
                    Some(x) => x,
                    None => {
                        colog_warn!(
                            "Atom {} of residue {} was identified as being a {} carbon. However, it is in an isolated chain of {} carbons and hydrogens therefore cannot be predicted for it. Ignoring.", 
                            atom.get_atom_name(), atom.get_residue_name(), "methyl", 2);
                        return None;
                    }
                };

                Some(UAType::CH3(AtomTriplet {
                    helper1: bonded_atoms[0],
                    target: atom.get_index(),
                    helper2,
                }))
            }

            (CarbonKind::Saturated, x) => {
                colog_warn!(
                    "Atom {} of residue {} is a {} carbon and has {} missing hydrogens. This is unsupported. Ignoring.",
                    atom.get_atom_name(),
                    atom.get_residue_name(),
                    "saturated",
                    x
                );

                None
            }

            (CarbonKind::Unsaturated, 2) => Some(UAType::CH1Unsat(AtomTriplet {
                helper1: bonded_atoms[0],
                target: atom.get_index(),
                helper2: bonded_atoms[1],
            })),

            (CarbonKind::Unsaturated, x) => {
                colog_warn!(
                    "Atom {} of residue {} is an {} carbon and has {} missing hydrogens. This is unsupported. Ignoring.",
                    atom.get_atom_name(),
                    atom.get_residue_name(),
                    "unsaturated",
                    x - 1,
                );

                None
            }
        }
    }

    /// Get the type of carbon (saturated, unsaturated) for `index`.
    fn classify_carbon(system: &System, index: usize) -> CarbonKind {
        let saturated = system
            .group_isin(group_name!("Saturated"), index)
            .unwrap_or(false);

        let unsaturated = system
            .group_isin(group_name!("Unsaturated"), index)
            .unwrap_or(false);

        match (saturated, unsaturated) {
            (true, false) => CarbonKind::Saturated,
            (false, true) => CarbonKind::Unsaturated,
            (true, true) => panic!(
                "FATAL GORDER ERROR | UAOrderAtomType::classify_carbon | Atom `{}` is both satured and unsatured but this should have been handled before. {}", 
                index, PANIC_MESSAGE),
            (false, false) => panic!(
                "FATAL GORDER ERROR | UAOrderAtomType::classify_carbon | Atom `{}` is neither saturated nor unsaturated. How did it get here? {}",
                index, PANIC_MESSAGE),
        }
    }
}

/// Methyl atom type.
#[derive(Debug, Clone)]
struct CH3Atom {
    atom_type: AtomType,
    atoms: Vec<AtomTriplet>,
    bonds: [VirtualBondType; 3],
}

impl UAAtom<3, 3> for CH3Atom {
    type Indices = AtomTriplet;

    #[inline(always)]
    fn indices_iter(&self) -> impl Iterator<Item = &AtomTriplet> {
        self.atoms.iter()
    }

    #[inline(always)]
    fn get_bond(&mut self, index: usize) -> &mut VirtualBondType {
        self.bonds.get_mut(index).unwrap_or_else(|| {
            panic!(
                "FATAL GORDER ERROR | CH3Atom::get_bond | Bond index `{}` out of range. {}",
                index, PANIC_MESSAGE
            )
        })
    }

    #[inline(always)]
    fn predict_hydrogens<'a>(
        triplet: &AtomTriplet,
        frame: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; 3], AnalysisError> {
        triplet.predict_hydrogens_ch3(frame, pbc)
    }
}

impl Add<CH3Atom> for CH3Atom {
    type Output = CH3Atom;

    fn add(self, rhs: CH3Atom) -> Self::Output {
        let mut left_iter = self.bonds.into_iter();
        let mut right_iter = rhs.bonds.into_iter();

        CH3Atom {
            atom_type: self.atom_type,
            atoms: self.atoms,
            bonds: std::array::from_fn(|_| {
                let a = left_iter.next().unwrap();
                let b = right_iter.next().unwrap();
                a + b
            }),
        }
    }
}

/// Methylene atom type.
#[derive(Debug, Clone)]
struct CH2Atom {
    atom_type: AtomType,
    atoms: Vec<AtomTriplet>,
    bonds: [VirtualBondType; 2],
}

impl UAAtom<2, 3> for CH2Atom {
    type Indices = AtomTriplet;

    #[inline(always)]
    fn indices_iter(&self) -> impl Iterator<Item = &AtomTriplet> {
        self.atoms.iter()
    }

    #[inline(always)]
    fn get_bond(&mut self, index: usize) -> &mut VirtualBondType {
        self.bonds.get_mut(index).unwrap_or_else(|| {
            panic!(
                "FATAL GORDER ERROR | CH2Atom::get_bond | Bond index `{}` out of range. {}",
                index, PANIC_MESSAGE
            )
        })
    }

    #[inline(always)]
    fn predict_hydrogens<'a>(
        triplet: &AtomTriplet,
        frame: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; 2], AnalysisError> {
        triplet.predict_hydrogens_ch2(frame, pbc)
    }
}

impl Add<CH2Atom> for CH2Atom {
    type Output = CH2Atom;

    fn add(self, rhs: CH2Atom) -> Self::Output {
        let mut left_iter = self.bonds.into_iter();
        let mut right_iter = rhs.bonds.into_iter();

        CH2Atom {
            atom_type: self.atom_type,
            atoms: self.atoms,
            bonds: std::array::from_fn(|_| {
                let a = left_iter.next().unwrap();
                let b = right_iter.next().unwrap();
                a + b
            }),
        }
    }
}

/// Methine atom type.
#[derive(Debug, Clone)]
struct CH1UnsaturatedAtom {
    atom_type: AtomType,
    atoms: Vec<AtomTriplet>,
    bond: VirtualBondType,
}

impl UAAtom<1, 3> for CH1UnsaturatedAtom {
    type Indices = AtomTriplet;

    #[inline(always)]
    fn indices_iter(&self) -> impl Iterator<Item = &AtomTriplet> {
        self.atoms.iter()
    }

    #[inline(always)]
    fn get_bond(&mut self, index: usize) -> &mut VirtualBondType {
        if index != 0 {
            panic!(
                "FATAL GORDER ERROR | CH1UnsaturatedAtom::get_bond | Bond index `{}` out of range. {}",
                index, PANIC_MESSAGE
            )
        }

        &mut self.bond
    }

    #[inline(always)]
    fn predict_hydrogens<'a>(
        triplet: &AtomTriplet,
        frame: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; 1], AnalysisError> {
        triplet.predict_hydrogen_ch1_unsaturated(frame, pbc)
    }
}

impl Add<CH1UnsaturatedAtom> for CH1UnsaturatedAtom {
    type Output = CH1UnsaturatedAtom;

    fn add(self, rhs: CH1UnsaturatedAtom) -> Self::Output {
        CH1UnsaturatedAtom {
            atom_type: self.atom_type,
            atoms: self.atoms,
            bond: self.bond + rhs.bond,
        }
    }
}

/// Methanetriyl atom type.
#[derive(Debug, Clone)]
struct CH1SaturatedAtom {
    atom_type: AtomType,
    atoms: Vec<AtomQuadruplet>,
    bond: VirtualBondType,
}

impl UAAtom<1, 4> for CH1SaturatedAtom {
    type Indices = AtomQuadruplet;

    #[inline(always)]
    fn indices_iter(&self) -> impl Iterator<Item = &AtomQuadruplet> {
        self.atoms.iter()
    }

    #[inline(always)]
    fn get_bond(&mut self, index: usize) -> &mut VirtualBondType {
        if index != 0 {
            panic!(
                "FATAL GORDER ERROR | CH1SaturatedAtom::get_bond | Bond index `{}` out of range. {}",
                index, PANIC_MESSAGE
            )
        }

        &mut self.bond
    }

    #[inline(always)]
    fn predict_hydrogens<'a>(
        triplet: &AtomQuadruplet,
        frame: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; 1], AnalysisError> {
        triplet.predict_hydrogen_ch1_saturated(frame, pbc)
    }
}

impl Add<CH1SaturatedAtom> for CH1SaturatedAtom {
    type Output = CH1SaturatedAtom;

    fn add(self, rhs: CH1SaturatedAtom) -> Self::Output {
        CH1SaturatedAtom {
            atom_type: self.atom_type,
            atoms: self.atoms,
            bond: self.bond + rhs.bond,
        }
    }
}

/// Trait implemented by structures storing indices of atoms used in hydrogen prediction.
trait AtomIndices<const N: usize> {
    /// Get positions of the atoms which indices are stored in the structure.
    fn unpack2pos<'a>(&self, system: &'a System) -> Result<[&'a Vector3D; N], AnalysisError>;

    /// Get the position of the target atom.
    fn get_target<'a>(&self, system: &'a System) -> Result<&'a Vector3D, AnalysisError>;
}

/// Structure storing indices of two helper atoms and a target atom.
#[derive(Debug, Clone)]
struct AtomTriplet {
    helper1: usize,
    target: usize,
    helper2: usize,
}

impl AtomIndices<3> for AtomTriplet {
    fn unpack2pos<'a>(&self, system: &'a System) -> Result<[&'a Vector3D; 3], AnalysisError> {
        let positions: Result<Vec<_>, _> = [self.helper1, self.target, self.helper2]
            .into_iter()
            .map(|index| {
                // SAFETY: indices must be valid
                let atom = unsafe { system.get_atom_unchecked(index) };
                atom.get_position()
                    .ok_or_else(|| AnalysisError::UndefinedPosition(index))
            })
            .collect();

        match positions {
            Ok(positions) => Ok([positions[0], positions[1], positions[2]]),
            Err(e) => Err(e),
        }
    }

    #[inline(always)]
    fn get_target<'a>(&self, system: &'a System) -> Result<&'a Vector3D, AnalysisError> {
        // SAFETY: index must be valid
        unsafe { system.get_atom_unchecked(self.target) }
            .get_position()
            .ok_or_else(|| AnalysisError::UndefinedPosition(self.target))
    }
}

impl AtomTriplet {
    /// Predict positions of hydrogens of a methyl group.
    /// Adapted from https://buildh.readthedocs.io/en/latest/algorithms_Hbuilding.html#building-ch3.
    fn predict_hydrogens_ch3<'a>(
        &self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; 3], AnalysisError> {
        let [helper1, target, helper2] = self.unpack2pos(system)?;

        let th1 = pbc.vector_to(target, helper1);
        let th2 = pbc.vector_to(target, helper2);

        let rot_axis = th2.cross(&th1);
        let rotation1 =
            Rotation3::from_axis_angle(&Unit::new_normalize(rot_axis), TETRAHEDRAL_ANGLE);

        let hydrogen1_vec = th1.clone().rotate(&rotation1.into());
        let mut hydrogen1 = target.clone();
        hydrogen1.shift(hydrogen1_vec.clone(), BOND_LENGTH);
        pbc.wrap(&mut hydrogen1);

        let normalized_th1 = Unit::new_normalize(*th1.deref());

        let rotation2 = Rotation3::from_axis_angle(&normalized_th1, CH3_ANGLE);
        let hydrogen2_vec = hydrogen1_vec.clone().rotate(&rotation2.into());
        let mut hydrogen2 = target.clone();
        hydrogen2.shift(hydrogen2_vec, BOND_LENGTH);
        pbc.wrap(&mut hydrogen2);

        let rotation3 = Rotation3::from_axis_angle(&normalized_th1, -CH3_ANGLE);
        let hydrogen3_vec = hydrogen1_vec.clone().rotate(&rotation3.into());
        let mut hydrogen3 = target.clone();
        hydrogen3.shift(hydrogen3_vec, BOND_LENGTH);
        pbc.wrap(&mut hydrogen3);

        Ok([hydrogen1, hydrogen2, hydrogen3])
    }

    /// Predict positions of hydrogens of a methylene group.
    /// Adapted from https://buildh.readthedocs.io/en/latest/algorithms_Hbuilding.html#building-ch2.
    fn predict_hydrogens_ch2<'a>(
        &self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; 2], AnalysisError> {
        let [helper1, target, helper2] = self.unpack2pos(system)?;

        let th1 = pbc.vector_to(target, helper1).to_unit();
        let th2 = pbc.vector_to(target, helper2).to_unit();
        let plane_normal = th2.cross(&th1);
        let rot_axis = (th1 - th2).to_unit();
        let rot_vec = Vector3D::from(plane_normal.cross(&rot_axis));

        let rotation_positive = Rotation3::from_axis_angle(
            &Unit::new_normalize(*rot_axis.deref()),
            TETRAHEDRAL_ANGLE_HALF,
        );

        let rotation_negative = Rotation3::from_axis_angle(
            &Unit::new_normalize(*rot_axis.deref()),
            -TETRAHEDRAL_ANGLE_HALF,
        );

        let hydrogen1_vec = rot_vec.clone().rotate(&rotation_positive.into());
        let hydrogen2_vec = rot_vec.rotate(&rotation_negative.into());

        let mut hydrogen1 = target.clone();
        hydrogen1.shift(hydrogen1_vec, BOND_LENGTH);
        pbc.wrap(&mut hydrogen1);

        let mut hydrogen2 = target.clone();
        hydrogen2.shift(hydrogen2_vec, BOND_LENGTH);
        pbc.wrap(&mut hydrogen2);

        Ok([hydrogen1, hydrogen2])
    }

    /// Predict position of a hydrogen of a methine group.
    /// Adapted from https://buildh.readthedocs.io/en/latest/algorithms_Hbuilding.html#building-ch-on-a-double-bond.
    fn predict_hydrogen_ch1_unsaturated<'a>(
        &self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; 1], AnalysisError> {
        let [helper1, target, helper2] = self.unpack2pos(system)?;

        let th1 = pbc.vector_to(target, helper1);
        let th2 = pbc.vector_to(target, helper2);
        let gamma = th1.angle(&th2);
        let rot_axis = th1.cross(&th2);

        let rotation =
            Rotation3::from_axis_angle(&Unit::new_normalize(rot_axis), PI - (gamma / 2.0));

        let hydrogen_vec = th2.clone().rotate(&rotation.into());
        let mut hydrogen = target.clone();
        hydrogen.shift(hydrogen_vec, BOND_LENGTH);
        pbc.wrap(&mut hydrogen);

        Ok([hydrogen])
    }
}

/// Structure storing indices of three helper atoms and a target atom.
#[derive(Debug, Clone)]
struct AtomQuadruplet {
    helper1: usize,
    helper2: usize,
    helper3: usize,
    target: usize,
}

impl AtomIndices<4> for AtomQuadruplet {
    fn unpack2pos<'a>(&self, system: &'a System) -> Result<[&'a Vector3D; 4], AnalysisError> {
        let positions: Result<Vec<_>, _> = [self.helper1, self.helper2, self.helper3, self.target]
            .into_iter()
            .map(|index| {
                // SAFETY: indices must be valid
                let atom = unsafe { system.get_atom_unchecked(index) };
                atom.get_position()
                    .ok_or_else(|| AnalysisError::UndefinedPosition(index))
            })
            .collect();

        match positions {
            Ok(positions) => Ok([positions[0], positions[1], positions[2], positions[3]]),
            Err(e) => Err(e),
        }
    }

    #[inline(always)]
    fn get_target<'a>(&self, system: &'a System) -> Result<&'a Vector3D, AnalysisError> {
        // SAFETY: index must be valid
        unsafe { system.get_atom_unchecked(self.target) }
            .get_position()
            .ok_or_else(|| AnalysisError::UndefinedPosition(self.target))
    }
}

impl AtomQuadruplet {
    /// Predict position of a hydrogen of a methanetriyl group.
    /// Adapted from https://buildh.readthedocs.io/en/latest/algorithms_Hbuilding.html#building-ch.
    fn predict_hydrogen_ch1_saturated<'a>(
        &self,
        system: &'a System,
        pbc: &'a impl PBCHandler<'a>,
    ) -> Result<[Vector3D; 1], AnalysisError> {
        let [h1, h2, h3, t] = self.unpack2pos(system)?;

        let th1 = pbc.vector_to(t, h1).to_unit();
        let th2 = pbc.vector_to(t, h2).to_unit();
        let th3 = pbc.vector_to(t, h3).to_unit();

        let hydrogen_vec = (th1 + th2 + th3).invert();
        let mut hydrogen = t.clone();
        hydrogen.shift(hydrogen_vec, BOND_LENGTH);
        pbc.wrap(&mut hydrogen);

        Ok([hydrogen])
    }
}

#[cfg(test)]
mod tests_predict {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_predict_ch2() {
        let system = System::from_file("tests/files/ua.tpr").unwrap();
        let pbc = PBC3D::from_system(&system);

        let triplet = AtomTriplet {
            helper1: 38,
            target: 39,
            helper2: 40,
        };

        let [h1, h2] = triplet.predict_hydrogens_ch2(&system, &pbc).unwrap();

        assert_relative_eq!(h1.x, 2.3435528);
        assert_relative_eq!(h1.y, 2.1503785);
        assert_relative_eq!(h1.z, 2.1272178);

        assert_relative_eq!(h2.x, 2.35857);
        assert_relative_eq!(h2.y, 2.3045487);
        assert_relative_eq!(h2.z, 2.039533);
    }

    #[test]
    fn test_predict_ch3() {
        let system = System::from_file("tests/files/ua.tpr").unwrap();
        let pbc = PBC3D::from_system(&system);

        let triplet = AtomTriplet {
            helper1: 48,
            target: 49,
            helper2: 47,
        };

        let [h1, h2, h3] = triplet.predict_hydrogens_ch3(&system, &pbc).unwrap();

        assert_relative_eq!(h1.x, 3.3708375);
        assert_relative_eq!(h1.y, 2.7527616);
        assert_relative_eq!(h1.z, 2.257202);

        assert_relative_eq!(h2.x, 3.254057);
        assert_relative_eq!(h2.y, 2.8633823);
        assert_relative_eq!(h2.z, 2.3334126);

        assert_relative_eq!(h3.x, 3.3182635);
        assert_relative_eq!(h3.y, 2.8995805);
        assert_relative_eq!(h3.z, 2.1713943);
    }

    #[test]
    fn test_predict_unsat_ch1() {
        let system = System::from_file("tests/files/ua.tpr").unwrap();
        let pbc = PBC3D::from_system(&system);

        let triplet = AtomTriplet {
            helper1: 22,
            target: 23,
            helper2: 24,
        };

        let [h] = triplet
            .predict_hydrogen_ch1_unsaturated(&system, &pbc)
            .unwrap();

        assert_relative_eq!(h.x, 1.0985602);
        assert_relative_eq!(h.y, 2.994375);
        assert_relative_eq!(h.z, 2.7727659);
    }

    #[test]
    fn test_predict_sat_ch1() {
        let system = System::from_file("tests/files/ua.tpr").unwrap();
        let pbc = PBC3D::from_system(&system);

        let quadruplet = AtomQuadruplet {
            helper1: 11,
            helper2: 31,
            helper3: 13,
            target: 12,
        };

        let [h] = quadruplet
            .predict_hydrogen_ch1_saturated(&system, &pbc)
            .unwrap();

        assert_relative_eq!(h.x, 1.5022101);
        assert_relative_eq!(h.y, 2.6938448);
        assert_relative_eq!(h.z, 1.7839708);
    }
}

#[cfg(test)]
mod tests_groups {
    use crate::input::AnalysisType;

    use super::*;

    #[test]
    fn test_fail_saturated_unsaturated_overlap() {
        let mut system = System::from_file("tests/files/ua.tpr").unwrap();
        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34) or (resname POPS and name r'^C' and not name C6 C18 C39)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .build()
            .unwrap();

        match prepare_groups(&mut system, &analysis) {
            Ok(_) => panic!("Function should have failed."),
            Err(TopologyError::AtomsOverlap {
                n_overlapping,
                name1,
                query1,
                name2,
                query2,
            }) => {
                assert_eq!(n_overlapping, 256);
                assert_eq!(name1, "Saturated");
                assert_eq!(query1, "(resname POPC and name r'^C' and not name C15 C34) or (resname POPS and name r'^C' and not name C6 C18 C39)");
                assert_eq!(name2, "Unsaturated");
                assert_eq!(
                    query2,
                    "(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"
                );
            }
            Err(e) => panic!("Unexpected error type `{}` returned.", e),
        }
    }

    #[test]
    fn test_fail_saturated_ignore_overlap() {
        let mut system = System::from_file("tests/files/ua.tpr").unwrap();
        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                None,
                Some("resname POPC"),
            ))
            .build()
            .unwrap();

        match prepare_groups(&mut system, &analysis) {
            Ok(_) => panic!("Function should have failed."),
            Err(TopologyError::AtomsOverlap {
                n_overlapping,
                name1,
                query1,
                name2,
                query2,
            }) => {
                assert_eq!(n_overlapping, 3876);
                assert_eq!(name1, "Saturated");
                assert_eq!(query1, "(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)");
                assert_eq!(name2, "Ignore");
                assert_eq!(query2, "resname POPC");
            }
            Err(e) => panic!("Unexpected error type `{}` returned.", e),
        }
    }

    #[test]
    fn test_fail_unsaturated_ignore_overlap() {
        let mut system = System::from_file("tests/files/ua.tpr").unwrap();
        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                None,
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                Some("name C24"),
            ))
            .build()
            .unwrap();

        match prepare_groups(&mut system, &analysis) {
            Ok(_) => panic!("Function should have failed."),
            Err(TopologyError::AtomsOverlap {
                n_overlapping,
                name1,
                query1,
                name2,
                query2,
            }) => {
                assert_eq!(n_overlapping, 102);
                assert_eq!(name1, "Unsaturated");
                assert_eq!(
                    query1,
                    "(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"
                );
                assert_eq!(name2, "Ignore");
                assert_eq!(query2, "name C24");
            }
            Err(e) => panic!("Unexpected error type `{}` returned.", e),
        }
    }

    #[test]
    fn test_fail_missing_saturated_unsaturated() {
        let mut system = System::from_file("tests/files/ua.tpr").unwrap();
        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(None, None, Some("resname SOL")))
            .build()
            .unwrap();

        match prepare_groups(&mut system, &analysis) {
            Ok(_) => panic!("Function should have failed."),
            Err(TopologyError::NoUACarbons) => (),
            Err(e) => panic!("Unexpected error type `{}` returned.", e),
        }
    }
}
