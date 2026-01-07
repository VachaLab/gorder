// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for working with order united-atoms.

use std::ops::Add;

use super::OrderCalculable;
use crate::{
    analysis::{
        geometry::GeometrySelection, leaflets::MoleculeLeafletClassification,
        normal::MoleculeMembraneNormal, pbc::PBCHandler, uaorder::UAOrderAtomType,
    },
    errors::{AnalysisError, TopologyError},
    input::OrderMap,
};
use getset::{Getters, MutGetters};
use groan_rs::system::System;

/// Collection of all united atoms for which the order parameters should be calculated.
#[derive(Debug, Clone, Getters, MutGetters)]
pub(crate) struct UAOrderAtoms {
    #[getset(get = "pub(crate)", get_mut = "pub(super)")]
    atom_types: Vec<UAOrderAtomType>,
}

impl OrderCalculable for UAOrderAtoms {
    type ElementSet = Vec<usize>;

    fn new<'a>(
        system: &System,
        atoms: &Vec<usize>,
        min_index: usize,
        classify_leaflets: bool,
        ordermap: Option<&OrderMap>,
        errors: bool,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Result<Self, TopologyError> {
        let mut order_atoms = Vec::new();
        let mut sorted_atoms = atoms.clone();
        sorted_atoms.sort();

        for &index in sorted_atoms.iter() {
            let ua_atom = match UAOrderAtomType::new(
                system,
                index,
                min_index,
                classify_leaflets,
                ordermap,
                pbc_handler,
                errors,
            )? {
                Some(x) => x,
                None => continue,
            };

            order_atoms.push(ua_atom);
        }

        Ok(UAOrderAtoms {
            atom_types: order_atoms,
        })
    }

    fn insert(&mut self, min_index: usize) {
        for atom in self.atom_types.iter_mut() {
            let relative_index = atom.get_relative_index();
            atom.insert(relative_index + min_index);
        }
    }

    #[inline(always)]
    fn analyze_frame<'a, Geom: GeometrySelection>(
        &mut self,
        frame: &'a System,
        pbc_handler: &'a impl PBCHandler<'a>,
        frame_index: usize,
        geometry: &Geom,
        leaflet: &mut Option<MoleculeLeafletClassification>,
        normal: &mut MoleculeMembraneNormal,
    ) -> Result<(), AnalysisError> {
        for atom in self.atom_types_mut() {
            atom.analyze_frame(frame, leaflet, pbc_handler, normal, frame_index, geometry)?;
        }

        Ok(())
    }

    #[inline(always)]
    fn init_new_frame(&mut self) {
        self.atom_types_mut()
            .iter_mut()
            .for_each(|atom| atom.init_new_frame());
    }

    #[inline(always)]
    fn n_molecules(&self) -> usize {
        self.atom_types()
            .first()
            .map(|atom_type| atom_type.n_molecules())
            .unwrap_or(0)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.atom_types().is_empty()
    }

    #[inline(always)]
    fn get_timewise_info(&self, n_blocks: usize) -> Option<(usize, usize)> {
        if let Some(atom) = self.atom_types().first() {
            atom.get_timewise_info(n_blocks)
        } else {
            None
        }
    }
}

impl Add<UAOrderAtoms> for UAOrderAtoms {
    type Output = Self;

    fn add(self, rhs: UAOrderAtoms) -> Self::Output {
        UAOrderAtoms {
            atom_types: self
                .atom_types
                .into_iter()
                .zip(rhs.atom_types)
                .map(|(a, b)| a + b)
                .collect::<Vec<UAOrderAtomType>>(),
        }
    }
}
