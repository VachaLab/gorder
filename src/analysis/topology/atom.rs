// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for working with atoms and atom types.

use std::fmt;

use getset::{CopyGetters, Getters, MutGetters};
use groan_rs::{prelude::Atom, system::System};

use crate::PANIC_MESSAGE;

/// Collection of all atom types for which order parameters should be calculated.
/// In case of coarse-grained order parameters, this involves all specified atoms.
/// In case of atomistic order parameters, this only involves heavy atoms.
#[derive(Debug, Clone, PartialEq, Eq, MutGetters, Getters)]
pub(crate) struct OrderAtoms {
    /// Ordered by the increasing relative index.
    #[getset(get = "pub(crate)", get_mut = "pub(super)")]
    atoms: Vec<AtomType>,
}

impl OrderAtoms {
    /// Create a new list of atoms involved in the order calculations.
    pub(crate) fn new(system: &System, atoms: &[usize], minimum_index: usize) -> Self {
        let mut converted_atoms = atoms
            .iter()
            .map(|&x| AtomType::new(x - minimum_index, system.get_atom(x).expect(PANIC_MESSAGE)))
            .collect::<Vec<AtomType>>();

        converted_atoms.sort_by(|a, b| a.relative_index.cmp(&b.relative_index));

        OrderAtoms {
            atoms: converted_atoms,
        }
    }

    #[allow(unused)]
    #[inline(always)]
    pub(crate) fn new_raw(atoms: Vec<AtomType>) -> Self {
        OrderAtoms { atoms }
    }
}

/// Type of atom. Specific to a particular molecule.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Getters, MutGetters, CopyGetters)]
pub struct AtomType {
    /// Relative index of the atom in a molecule.
    #[getset(get_copy = "pub", get_mut = "pub(super)")]
    relative_index: usize,
    /// Name of the residue of the atom.
    #[getset(get = "pub", get_mut = "pub(super)")]
    residue_name: String,
    /// Name of the atom.
    #[getset(get = "pub", get_mut = "pub(super)")]
    atom_name: String,
}

impl AtomType {
    /// Create a new atom type.
    #[inline(always)]
    pub(crate) fn new(relative_index: usize, atom: &Atom) -> AtomType {
        AtomType {
            relative_index,
            residue_name: atom.get_residue_name().to_owned(),
            atom_name: atom.get_atom_name().to_owned(),
        }
    }

    /// Create a new atom type from raw parts.
    #[allow(unused)]
    #[inline(always)]
    pub(crate) fn new_raw(relative_index: usize, residue_name: &str, atom_name: &str) -> AtomType {
        AtomType {
            relative_index,
            residue_name: residue_name.to_owned(),
            atom_name: atom_name.to_owned(),
        }
    }
}

impl fmt::Display for AtomType {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}-{}-{}",
            self.residue_name(),
            self.atom_name(),
            self.relative_index()
        )
    }
}
