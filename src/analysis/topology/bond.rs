// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for working with order bonds.

use std::ops::Add;

use crate::{
    analysis::{
        calc_sch,
        geometry::GeometrySelection,
        leaflets::MoleculeLeafletClassification,
        normal::MoleculeMembraneNormal,
        order::{merge_option_order, AnalysisOrder},
        ordermap::{merge_option_maps, Map},
        pbc::PBCHandler,
        timewise::AddExtend,
    },
    errors::{AnalysisError, TopologyError},
    input::OrderMap,
    Leaflet, PANIC_MESSAGE,
};

use super::{atom::AtomType, bonds_sanity_check, get_atoms_from_bond, OrderCalculable};
use getset::{CopyGetters, Getters, MutGetters};
use groan_rs::{
    prelude::{Atom, Vector3D},
    system::System,
};
use hashbrown::HashSet;

/// Collection of all bonds for which the order parameters should be calculated.
#[derive(Debug, Clone, Getters, MutGetters)]
pub(crate) struct OrderBonds {
    #[getset(get = "pub(crate)", get_mut = "pub(super)")]
    bond_types: Vec<BondType>,
}

impl OrderCalculable for OrderBonds {
    type ElementSet = HashSet<(usize, usize)>;

    /// Create a new `OrderBonds` structure from a set of bonds (absolute indices) and the minimum index of the molecule.
    ///
    /// ## Panics
    /// - Panics if the `min_index` is higher than any index inside the `bonds` set.
    /// - Panics if there is a bond connecting the same atom (e.g. 14-14) in the `bonds` set.
    /// - Panics if an index in the `bonds` set does not correspond to an existing atom.
    fn new<'a>(
        system: &System,
        bonds: &HashSet<(usize, usize)>,
        min_index: usize,
        classify_leaflets: bool,
        ordermap: Option<&OrderMap>,
        errors: bool,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Result<Self, TopologyError> {
        bonds_sanity_check(bonds, min_index);

        let mut order_bonds = Vec::new();
        for &(index1, index2) in bonds.iter() {
            let (atom1, atom2) = get_atoms_from_bond(system, index1, index2);
            let bond = BondType::new(
                index1,
                atom1,
                index2,
                atom2,
                min_index,
                classify_leaflets,
                ordermap,
                pbc_handler,
                errors,
            )?;
            order_bonds.push(bond)
        }

        // sort order bonds so that atoms with lower indices come first
        order_bonds.sort_by(|b1, b2| {
            b1.atom1_index()
                .cmp(&b2.atom1_index())
                .then_with(|| b1.atom2_index().cmp(&b2.atom2_index()))
        });

        Ok(OrderBonds {
            bond_types: order_bonds,
        })
    }

    /// Insert new instances of bonds to an already constructed list of order bonds.
    fn insert(&mut self, min_index: usize) {
        for bond in self.bond_types.iter_mut() {
            let reference_topology = bond.bond_topology();
            let (atom_type_1, atom_type_2) =
                (reference_topology.atom1(), reference_topology.atom2());

            let abs_index_1 = atom_type_1.relative_index() + min_index;
            let abs_index_2 = atom_type_2.relative_index() + min_index;

            bond.insert(abs_index_1, abs_index_2);
        }
    }

    #[inline]
    fn analyze_frame<'a, Geom: GeometrySelection>(
        &mut self,
        frame: &'a System,
        pbc_handler: &'a impl PBCHandler<'a>,
        frame_index: usize,
        geometry: &Geom,
        leaflet: &mut Option<MoleculeLeafletClassification>,
        normal: &mut MoleculeMembraneNormal,
    ) -> Result<(), AnalysisError> {
        self.bond_types.iter_mut().try_for_each(|bond_type| {
            bond_type.analyze_frame(frame, leaflet, pbc_handler, normal, frame_index, geometry)
        })?;

        Ok(())
    }

    #[inline(always)]
    fn init_new_frame(&mut self) {
        self.bond_types
            .iter_mut()
            .for_each(|bond| bond.init_new_frame());
    }

    #[inline(always)]
    fn n_molecules(&self) -> usize {
        self.bond_types
            .first()
            .map(|bond_type| bond_type.bonds.len())
            .unwrap_or(0)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.bond_types.is_empty()
    }

    #[inline]
    fn get_timewise_info(&self, n_blocks: usize) -> Option<(usize, usize)> {
        if let Some(bond) = self.bond_types.first() {
            bond.total()
                .timewise()
                .map(|timewise| (timewise.block_size(n_blocks), timewise.n_frames()))
        } else {
            None
        }
    }
}

impl Add<OrderBonds> for OrderBonds {
    type Output = OrderBonds;

    #[inline(always)]
    fn add(self, rhs: OrderBonds) -> Self::Output {
        OrderBonds {
            bond_types: self
                .bond_types
                .into_iter()
                .zip(rhs.bond_types)
                .map(|(a, b)| a + b)
                .collect::<Vec<BondType>>(),
        }
    }
}

/// Trait implemented by `Bond`-like structures such as `BondType` or `VirtualBondType`.
pub(crate) trait BondLike {
    /// Get mutable reference to the order parameter calculated for the full membrane.
    fn get_total(&mut self) -> &mut AnalysisOrder<AddExtend>;
    /// Get mutable reference to the order parameter calculated for the upper leaflet.
    fn get_upper(&mut self) -> Option<&mut AnalysisOrder<AddExtend>>;
    /// Get mutable reference to the order parameter calculated for the lower leaflet.
    fn get_lower(&mut self) -> Option<&mut AnalysisOrder<AddExtend>>;

    /// Get mutable reference to the order parameter map calculated for the full membrane.
    fn get_total_map(&mut self) -> Option<&mut Map>;
    /// Get mutable reference to the order parameter map calculated for the upper leaflet.
    fn get_upper_map(&mut self) -> Option<&mut Map>;
    /// Get mutable reference to the order parameter map calculated for the lower leaflet.
    fn get_lower_map(&mut self) -> Option<&mut Map>;

    /// Add the calculated order parameter to the already collected data.
    fn add_order(
        &mut self,
        molecule_index: usize,
        sch: f32,
        bond_pos: &Vector3D,
        leaflet_classification: &mut Option<MoleculeLeafletClassification>,
        frame_index: usize,
    ) {
        *self.get_total() += sch;

        if let Some(map) = self.get_total_map() {
            map.add_order(sch, bond_pos);
        }

        // get the assignment of molecule (assignment is performed earlier)
        if let Some(classifier) = leaflet_classification.as_mut() {
            match classifier.get_assigned_leaflet(molecule_index, frame_index) {
                Leaflet::Upper => {
                    *self.get_upper().expect(PANIC_MESSAGE) += sch;
                    if let Some(map) = self.get_upper_map() {
                        map.add_order(sch, bond_pos);
                    }
                }
                Leaflet::Lower => {
                    *self.get_lower().expect(PANIC_MESSAGE) += sch;
                    if let Some(map) = self.get_lower_map() {
                        map.add_order(sch, bond_pos);
                    }
                }
            }
        }
    }
}

/// Structure describing a type of bond.
/// Contains indices of all atoms that are involved in this bond type.
#[derive(Debug, Clone, CopyGetters, Getters, MutGetters)]
pub(crate) struct BondType {
    /// Atom types involved in this bond type.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    bond_topology: BondTopology,
    /// List of all bond instances of this bond type (absolute indices of atoms).
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    bonds: Vec<(usize, usize)>,
    /// Order parameter for this bond type calculated using lipids in the entire membrane.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    total: AnalysisOrder<AddExtend>,
    /// Order parameter for this bond type calculated using lipids in the upper leaflet.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    upper: Option<AnalysisOrder<AddExtend>>,
    /// Order parameter for this bond type calculated using lipids in the lower leaflet.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    lower: Option<AnalysisOrder<AddExtend>>,
    /// Order parameter map of this bond calculated using lipids in the entire membrane.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    total_map: Option<Map>,
    /// Order parameter map of this bond calculated using lipids in the upper leaflet.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    upper_map: Option<Map>,
    /// Order parameter map of this bond calculated using lipids in the lower leaflet.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    lower_map: Option<Map>,
}

impl BondLike for BondType {
    #[inline(always)]
    fn get_total(&mut self) -> &mut AnalysisOrder<AddExtend> {
        &mut self.total
    }

    #[inline(always)]
    fn get_upper(&mut self) -> Option<&mut AnalysisOrder<AddExtend>> {
        self.upper.as_mut()
    }

    #[inline(always)]
    fn get_lower(&mut self) -> Option<&mut AnalysisOrder<AddExtend>> {
        self.lower.as_mut()
    }

    #[inline(always)]
    fn get_total_map(&mut self) -> Option<&mut Map> {
        self.total_map.as_mut()
    }

    #[inline(always)]
    fn get_upper_map(&mut self) -> Option<&mut Map> {
        self.upper_map.as_mut()
    }

    #[inline(always)]
    fn get_lower_map(&mut self) -> Option<&mut Map> {
        self.lower_map.as_mut()
    }
}

impl BondType {
    /// Create a new bond for the calculation of order parameters.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<'a>(
        abs_index_1: usize,
        atom_1: &Atom,
        abs_index_2: usize,
        atom_2: &Atom,
        min_index: usize,
        classify_leaflets: bool,
        ordermap: Option<&OrderMap>,
        pbc_handler: &impl PBCHandler<'a>,
        errors: bool,
    ) -> Result<Self, TopologyError> {
        let bond_topology = BondTopology::new(
            abs_index_1 - min_index,
            atom_1,
            abs_index_2 - min_index,
            atom_2,
        );
        let real_bond = if abs_index_1 < abs_index_2 {
            (abs_index_1, abs_index_2)
        } else {
            (abs_index_2, abs_index_1)
        };

        let optional_map = if let Some(map_params) = ordermap {
            Some(
                Map::new(map_params.to_owned(), pbc_handler)
                    .map_err(TopologyError::OrderMapError)?,
            )
        } else {
            None
        };

        let (leaflet_order, leaflet_map) = if classify_leaflets {
            (
                Some(AnalysisOrder::new(0.0, 0, errors)),
                optional_map.clone(),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            bond_topology,
            bonds: vec![real_bond],
            total: AnalysisOrder::new(0.0, 0, errors),
            upper: leaflet_order.clone(),
            lower: leaflet_order,
            total_map: optional_map,
            upper_map: leaflet_map.clone(),
            lower_map: leaflet_map,
        })
    }

    /// Insert new real bond to the current order bond.
    #[inline(always)]
    pub(crate) fn insert(&mut self, abs_index_1: usize, abs_index_2: usize) {
        if abs_index_1 < abs_index_2 {
            self.bonds.push((abs_index_1, abs_index_2));
        } else {
            self.bonds.push((abs_index_2, abs_index_1));
        }
    }

    /// Get the first atom of this bond type from BondTopology.
    #[inline(always)]
    pub(crate) fn atom1(&self) -> &AtomType {
        self.bond_topology().atom1()
    }

    /// Get the second atom of this bond type from BondTopology.
    #[inline(always)]
    pub(crate) fn atom2(&self) -> &AtomType {
        self.bond_topology().atom2()
    }

    /// Get the relative index of the first atom of this bond type.
    #[inline(always)]
    pub(crate) fn atom1_index(&self) -> usize {
        self.atom1().relative_index()
    }

    /// Get the relative index of the second atom of this bond type.
    #[inline(always)]
    pub(crate) fn atom2_index(&self) -> usize {
        self.atom2().relative_index()
    }

    /// Does this bond involve a specific atom type?
    #[inline(always)]
    pub(crate) fn contains(&self, atom: &AtomType) -> bool {
        self.bond_topology().contains(atom)
    }

    /// Return the other atom involved in this bond.
    /// If the provided atom is not involved in the bond, return `None`.
    #[inline(always)]
    pub(crate) fn get_other_atom(&self, atom: &AtomType) -> Option<&AtomType> {
        self.bond_topology().get_other_atom(atom)
    }

    /// Initialize reading of a new simulation frame.
    #[inline(always)]
    fn init_new_frame(&mut self) {
        self.total.init_new_frame();
        if let Some(x) = self.upper.as_mut() {
            x.init_new_frame()
        }
        if let Some(x) = self.lower.as_mut() {
            x.init_new_frame()
        }
    }

    /// Calculate the current order parameter for this bond type.
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

        for (molecule_index, (index1, index2)) in self.bonds.iter().enumerate() {
            let atom1 = unsafe { frame.get_atom_unchecked(*index1) };
            let atom2 = unsafe { frame.get_atom_unchecked(*index2) };

            let pos1 = atom1
                .get_position()
                .ok_or_else(|| AnalysisError::UndefinedPosition(atom1.get_index()))?;

            let pos2 = atom2
                .get_position()
                .ok_or_else(|| AnalysisError::UndefinedPosition(atom2.get_index()))?;

            let vec = pbc_handler.vector_to(pos1, pos2);

            // get the coordinates of the bond
            let bond_pos = pos1 + (&vec / 2.0);
            // check whether the bond is inside the geometric shape
            if !pbc_handler.inside(&bond_pos, geometry) {
                continue;
            }

            // get the membrane normal for a molecule
            let normal =
                membrane_normal.get_normal(frame_index, molecule_index, frame, pbc_handler)?;

            let sch = calc_sch(&vec, normal);

            // safety: self lives the entire method
            // we are modifying different part of the structure than is iterated by `bonds.iter()`
            unsafe { &mut *self_ptr }.add_order(
                molecule_index,
                sch,
                &bond_pos,
                leaflet_classification,
                frame_index,
            )
        }

        Ok(())
    }
}

impl Add<BondType> for BondType {
    type Output = BondType;

    #[inline]
    fn add(self, rhs: BondType) -> Self::Output {
        BondType {
            bond_topology: self.bond_topology,
            bonds: self.bonds,
            total: self.total + rhs.total,
            upper: merge_option_order(self.upper, rhs.upper),
            lower: merge_option_order(self.lower, rhs.lower),
            total_map: merge_option_maps(self.total_map, rhs.total_map),
            upper_map: merge_option_maps(self.upper_map, rhs.upper_map),
            lower_map: merge_option_maps(self.lower_map, rhs.lower_map),
        }
    }
}

/// Atom types involved in a specific bond type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Getters, MutGetters)]
pub(crate) struct BondTopology {
    #[getset(get = "pub(crate)", get_mut = "pub(super)")]
    atom1: AtomType,
    #[getset(get = "pub(crate)", get_mut = "pub(super)")]
    atom2: AtomType,
}

impl BondTopology {
    /// Construct a new BondTopology.
    /// The provided atoms can be in any order. In the constructed structure, the atom
    /// with the smaller index will always be the `atom1`.
    #[inline]
    pub(super) fn new(
        relative_index_1: usize,
        atom_1: &Atom,
        relative_index_2: usize,
        atom_2: &Atom,
    ) -> BondTopology {
        if relative_index_1 < relative_index_2 {
            BondTopology {
                atom1: AtomType::new(relative_index_1, atom_1),
                atom2: AtomType::new(relative_index_2, atom_2),
            }
        } else {
            BondTopology {
                atom1: AtomType::new(relative_index_2, atom_2),
                atom2: AtomType::new(relative_index_1, atom_1),
            }
        }
    }

    /// Construct a new BondTopology using directly already constructed atom types.
    #[allow(unused)]
    #[inline(always)]
    pub(crate) fn new_from_types(atom_type1: AtomType, atom_type2: AtomType) -> BondTopology {
        BondTopology {
            atom1: atom_type1,
            atom2: atom_type2,
        }
    }

    /// Does this bond type involve the provided atom type?
    #[inline(always)]
    fn contains(&self, atom: &AtomType) -> bool {
        self.atom1() == atom || self.atom2() == atom
    }

    /// Return the other atom involved in this bond.
    /// If the provided atom is not involved in the bond, return `None`.
    #[inline(always)]
    fn get_other_atom(&self, atom: &AtomType) -> Option<&AtomType> {
        if self.atom1() == atom {
            Some(self.atom2())
        } else if self.atom2() == atom {
            Some(self.atom1())
        } else {
            None
        }
    }
}

/// Order parameters and ordermaps calculated for a single bond between a real atom and a virtual atom.
#[derive(Debug, Clone, Getters)]
pub(crate) struct VirtualBondType {
    #[getset(get = "pub(crate)")]
    total: AnalysisOrder<AddExtend>,
    #[getset(get = "pub(crate)")]
    upper: Option<AnalysisOrder<AddExtend>>,
    #[getset(get = "pub(crate)")]
    lower: Option<AnalysisOrder<AddExtend>>,
    #[getset(get = "pub(crate)")]
    total_map: Option<Map>,
    #[getset(get = "pub(crate)")]
    upper_map: Option<Map>,
    #[getset(get = "pub(crate)")]
    lower_map: Option<Map>,
}

impl VirtualBondType {
    pub(crate) fn new<'a>(
        classify_leaflets: bool,
        ordermap: Option<&OrderMap>,
        pbc_handler: &impl PBCHandler<'a>,
        errors: bool,
    ) -> Result<Self, TopologyError> {
        let optional_map = if let Some(map_params) = ordermap {
            Some(
                Map::new(map_params.to_owned(), pbc_handler)
                    .map_err(TopologyError::OrderMapError)?,
            )
        } else {
            None
        };

        let (leaflet_order, leaflet_map) = if classify_leaflets {
            (
                Some(AnalysisOrder::new(0.0, 0, errors)),
                optional_map.clone(),
            )
        } else {
            (None, None)
        };

        Ok(VirtualBondType {
            total: AnalysisOrder::new(0.0, 0, errors),
            upper: leaflet_order.clone(),
            lower: leaflet_order,
            total_map: optional_map,
            upper_map: leaflet_map.clone(),
            lower_map: leaflet_map,
        })
    }

    /// Initialize reading of a new simulation frame.
    #[inline(always)]
    pub(crate) fn init_new_frame(&mut self) {
        self.total.init_new_frame();
        if let Some(x) = self.upper.as_mut() {
            x.init_new_frame()
        }
        if let Some(x) = self.lower.as_mut() {
            x.init_new_frame()
        }
    }

    #[inline(always)]
    pub(crate) fn get_timewise_info(&self, n_blocks: usize) -> Option<(usize, usize)> {
        self.total
            .timewise()
            .map(|timewise| (timewise.block_size(n_blocks), timewise.n_frames()))
    }
}

impl BondLike for VirtualBondType {
    #[inline(always)]
    fn get_total(&mut self) -> &mut AnalysisOrder<AddExtend> {
        &mut self.total
    }

    #[inline(always)]
    fn get_upper(&mut self) -> Option<&mut AnalysisOrder<AddExtend>> {
        self.upper.as_mut()
    }

    #[inline(always)]
    fn get_lower(&mut self) -> Option<&mut AnalysisOrder<AddExtend>> {
        self.lower.as_mut()
    }

    #[inline(always)]
    fn get_total_map(&mut self) -> Option<&mut Map> {
        self.total_map.as_mut()
    }

    #[inline(always)]
    fn get_upper_map(&mut self) -> Option<&mut Map> {
        self.upper_map.as_mut()
    }

    #[inline(always)]
    fn get_lower_map(&mut self) -> Option<&mut Map> {
        self.lower_map.as_mut()
    }
}

impl Add<VirtualBondType> for VirtualBondType {
    type Output = VirtualBondType;

    #[inline]
    fn add(self, rhs: VirtualBondType) -> Self::Output {
        VirtualBondType {
            total: self.total + rhs.total,
            upper: merge_option_order(self.upper, rhs.upper),
            lower: merge_option_order(self.lower, rhs.lower),
            total_map: merge_option_maps(self.total_map, rhs.total_map),
            upper_map: merge_option_maps(self.upper_map, rhs.upper_map),
            lower_map: merge_option_maps(self.lower_map, rhs.lower_map),
        }
    }
}

#[cfg(test)]
mod tests {
    use groan_rs::prelude::SimBox;

    use crate::{
        analysis::{pbc::PBC3D, topology::atom::OrderAtoms},
        input::{GridSpan, Plane},
    };

    use super::*;

    #[test]
    fn test_bond_type_new() {
        let atom1 = Atom::new(1, "POPE", 1, "N");
        let atom2 = Atom::new(1, "POPE", 6, "HN");

        let bond1 = BondTopology::new(0, &atom1, 5, &atom2);
        let bond2 = BondTopology::new(5, &atom2, 0, &atom1);

        assert_eq!(bond1, bond2);
    }

    #[test]
    fn test_bond_new() {
        let atom1 = Atom::new(17, "POPE", 456, "N");
        let atom2 = Atom::new(17, "POPE", 461, "HN");

        let bond1 = BondType::new(
            455,
            &atom1,
            460,
            &atom2,
            455,
            false,
            None,
            &PBC3D::new(&SimBox::from([10.0, 10.0, 10.0])),
            false,
        )
        .unwrap();
        let bond2 = BondType::new(
            460,
            &atom2,
            455,
            &atom1,
            455,
            false,
            None,
            &PBC3D::new(&SimBox::from([10.0, 10.0, 10.0])),
            false,
        )
        .unwrap();

        assert_eq!(bond1.bond_topology, bond2.bond_topology);
        assert_eq!(bond1.bonds.len(), 1);
        assert_eq!(bond2.bonds.len(), 1);
        assert_eq!(bond1.bonds[0], bond2.bonds[0]);
    }

    #[test]
    fn test_bond_new_with_leaflet_classification() {
        let atom1 = Atom::new(17, "POPE", 456, "N");
        let atom2 = Atom::new(17, "POPE", 461, "HN");

        let bond = BondType::new(
            455,
            &atom1,
            460,
            &atom2,
            455,
            true,
            None,
            &PBC3D::new(&SimBox::from([10.0, 10.0, 10.0])),
            false,
        )
        .unwrap();

        assert!(bond.lower.is_some());
        assert!(bond.upper.is_some());
    }

    #[test]
    fn test_bond_new_with_ordermap() {
        let atom1 = Atom::new(17, "POPE", 456, "N");
        let atom2 = Atom::new(17, "POPE", 461, "HN");

        let ordermap_params = OrderMap::builder()
            .dim([GridSpan::Auto, GridSpan::Auto])
            .output_directory("ordermaps")
            .plane(Plane::XY)
            .build()
            .unwrap();
        let bond = BondType::new(
            455,
            &atom1,
            460,
            &atom2,
            455,
            false,
            Some(&ordermap_params),
            &PBC3D::new(&SimBox::from([10.0, 10.0, 10.0])),
            false,
        )
        .unwrap();

        assert!(bond.lower.is_none());
        assert!(bond.upper.is_none());
        assert!(bond.total_map.is_some());
        assert!(bond.lower_map.is_none());
        assert!(bond.upper_map.is_none());
    }

    #[test]
    fn test_bond_new_with_leaflet_classification_and_ordermap() {
        let atom1 = Atom::new(17, "POPE", 456, "N");
        let atom2 = Atom::new(17, "POPE", 461, "HN");

        let ordermap_params = OrderMap::builder()
            .dim([GridSpan::Auto, GridSpan::Auto])
            .output_directory("ordermaps")
            .plane(Plane::XY)
            .build()
            .unwrap();
        let bond = BondType::new(
            455,
            &atom1,
            460,
            &atom2,
            455,
            true,
            Some(&ordermap_params),
            &PBC3D::new(&SimBox::from([10.0, 10.0, 10.0])),
            false,
        )
        .unwrap();

        assert!(bond.lower.is_some());
        assert!(bond.upper.is_some());
        assert!(bond.total_map.is_some());
        assert!(bond.lower_map.is_some());
        assert!(bond.upper_map.is_some());
    }

    #[test]
    fn test_bond_add() {
        let atom1 = Atom::new(17, "POPE", 456, "N");
        let atom2 = Atom::new(17, "POPE", 461, "HN");

        let mut bond = BondType::new(
            455,
            &atom1,
            460,
            &atom2,
            455,
            false,
            None,
            &PBC3D::new(&SimBox::from([10.0, 10.0, 10.0])),
            false,
        )
        .unwrap();

        bond.insert(1354, 1359);
        bond.insert(1676, 1671);

        assert_eq!(bond.bonds.len(), 3);
        assert_eq!(bond.bonds[0], (455, 460));
        assert_eq!(bond.bonds[1], (1354, 1359));
        assert_eq!(bond.bonds[2], (1671, 1676));
    }

    #[test]
    fn order_atoms_new() {
        let system = System::from_file("tests/files/pcpepg.tpr").unwrap();
        let atoms = [133, 127, 163, 156, 145];
        let order_atoms = OrderAtoms::new(&system, &atoms, 125);

        let expected_atoms = [
            AtomType::new(2, system.get_atom(127).unwrap()),
            AtomType::new(8, system.get_atom(133).unwrap()),
            AtomType::new(20, system.get_atom(145).unwrap()),
            AtomType::new(31, system.get_atom(156).unwrap()),
            AtomType::new(38, system.get_atom(163).unwrap()),
        ];

        for (atom, expected) in order_atoms.atoms().iter().zip(expected_atoms.iter()) {
            assert_eq!(atom, expected);
        }
    }

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
    fn order_bonds_new() {
        let system = System::from_file("tests/files/pcpepg.tpr").unwrap();
        let bonds = [(169, 170), (169, 171), (213, 214), (213, 215), (246, 247)];
        let bonds_set = HashSet::from(bonds);
        let order_bonds = OrderBonds::new(
            &system,
            &bonds_set,
            125,
            false,
            None,
            false,
            &PBC3D::new(&SimBox::from([10.0, 10.0, 10.0])),
        )
        .unwrap();

        let expected_bonds = expected_bonds(&system);

        // bonds can be in any order inside `order_bonds`
        for bond in order_bonds.bond_types.iter() {
            let expected = expected_bonds
                .iter()
                .enumerate()
                .find(|(_, expected)| &bond.bond_topology == *expected);

            if let Some((i, _)) = expected {
                assert_eq!(bond.bonds.len(), 1);
                assert_eq!(bond.bonds[0], bonds[i]);
            } else {
                panic!("Expected bond not found.");
            }
        }
    }

    #[test]
    fn order_bonds_add() {
        let system = System::from_file("tests/files/pcpepg.tpr").unwrap();
        let bonds = [(169, 170), (169, 171), (213, 214), (213, 215), (246, 247)];
        let bonds_set = HashSet::from(bonds);
        let mut order_bonds = OrderBonds::new(
            &system,
            &bonds_set,
            125,
            false,
            None,
            false,
            &PBC3D::new(&SimBox::from([10.0, 10.0, 10.0])),
        )
        .unwrap();

        let new_bonds = [(919, 920), (919, 921), (963, 964), (963, 965), (996, 997)];
        order_bonds.insert(875);

        let expected_bonds = expected_bonds(&system);
        for bond in order_bonds.bond_types.iter() {
            let expected = expected_bonds
                .iter()
                .enumerate()
                .find(|(_, expected)| &bond.bond_topology == *expected);

            if let Some((i, _)) = expected {
                assert_eq!(bond.bonds.len(), 2);
                assert_eq!(bond.bonds[0], bonds[i]);
                assert_eq!(bond.bonds[1], new_bonds[i]);
            } else {
                panic!("Expected bond not found.");
            }
        }
    }
}
