// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Contains methods for working with and without periodic boundary conditions.

use groan_rs::{
    errors::{AtomError, GroupError},
    prelude::{
        Atom, AtomIterable, AtomIteratorWithBox, CellGrid, CellNeighbors, Cylinder, Dimension,
        ImmutableAtomIterable, SimBox, Sphere, Vector3D,
    },
    system::System,
};
use once_cell::sync::OnceCell;

use crate::{errors::AnalysisError, PANIC_MESSAGE};

use super::{common::macros::group_name, geometry::GeometrySelection};

/// Trait implemented by structures handling (or intentionally not handling) PBC.
pub(crate) trait PBCHandler<'a> {
    /// Get the geometric center of a group.
    fn group_get_center(&self, system: &System, group: &str) -> Result<Vector3D, GroupError>;

    /// Calculate the local membrane centers for all lipid molecules.
    fn calc_local_membrane_centers(
        &'a self,
        system: &'a System,
        heads: &[usize],
        radius: f32,
        membrane_normal: Dimension,
    ) -> Result<Vec<Vector3D>, AnalysisError>;

    /// Get the cloud of lipid heads for local membrane normal calculations.
    fn get_heads_cloud(
        &'a self,
        system: &'a System,
        reference: &Vector3D,
        radius: f32,
    ) -> Result<Vec<Vector3D>, AnalysisError>;

    /// Calculate distance between two points in the specified dimensions.
    fn distance(&self, point1: &Vector3D, point2: &Vector3D, dim: Dimension) -> f32;

    /// Calculate distance between two atoms of target indices.
    fn atoms_distance(
        &self,
        system: &System,
        index1: usize,
        index2: usize,
        dim: Dimension,
    ) -> Result<f32, AtomError>;

    /// Check if a point is inside a geometric shape.
    fn inside<Geom: GeometrySelection>(&self, point: &Vector3D, shape: &Geom) -> bool;

    /// Calculate shortest vector connecting point1 with point2.
    fn vector_to(&self, point1: &Vector3D, point2: &Vector3D) -> Vector3D;

    /// Wrap the point into the simulation box. Or not, if PBC are ignored.
    fn wrap(&self, point: &mut Vector3D);

    /// Get the reference position and length in the corresponding dimension for infinite shape.
    fn get_infinite_span(&self, pos: &mut f32) -> f32;

    /// Get the simulation box center or PANIC if PBC are ignored.
    fn get_box_center(&self) -> Vector3D;

    /// Get refrence to the simulation box. Returns None, if PBC are ignored.
    fn get_simbox(&self) -> Option<&SimBox>;

    /// Construct a cylinder for local center of geometry calculation.
    fn cylinder_for_local_center(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        orientation: Dimension,
    ) -> Cylinder;

    /// Initialize reading of the new frame.
    fn init_new_frame(&mut self);

    /// Get atoms that are closer to the `reference` than `distance`.
    /// Returns an iterator over atoms and their distances from reference.
    fn nearby_atoms(
        &'a self,
        system: &'a System,
        reference: Vector3D,
        distance: f32,
    ) -> impl Iterator<Item = (&'a Atom, f32)>;
}

/// PBCHandler that ignores all periodic boundary conditions.
#[derive(Debug, Clone)]
pub(crate) struct NoPBC;

impl<'a> PBCHandler<'a> for NoPBC {
    #[inline(always)]
    fn group_get_center(&self, system: &System, group: &str) -> Result<Vector3D, GroupError> {
        system.group_get_center_naive(group)
    }

    fn calc_local_membrane_centers(
        &'a self,
        system: &'a System,
        heads: &[usize],
        radius: f32,
        membrane_normal: Dimension,
    ) -> Result<Vec<Vector3D>, AnalysisError> {
        let membrane_iterator = system
            .group_iter(group_name!("Membrane"))
            .unwrap_or_else(|_|
                panic!("FATAL GORDER ERROR | NoPBC::calc_local_membrane_centers | Could not get the `Membrane` group. {}", PANIC_MESSAGE)
            );

        let mut centers = Vec::with_capacity(heads.len());
        for &index in heads.iter() {
            let position = unsafe { system.get_atom_unchecked(index) }
                .get_position()
                .ok_or(AnalysisError::UndefinedPosition(index))?;

            let cylinder =
                self.cylinder_for_local_center(position.x, position.y, radius, membrane_normal);

            let center = membrane_iterator
                .clone()
                .filter_geometry_naive(cylinder)
                .get_center_naive()
                .map_err(|_| AnalysisError::InvalidLocalMembraneCenter(index))?;

            if center.x.is_nan() || center.y.is_nan() || center.z.is_nan() {
                return Err(AnalysisError::InvalidLocalMembraneCenter(index));
            }

            centers.push(center);
        }

        Ok(centers)
    }

    fn get_heads_cloud(
        &self,
        system: &System,
        reference: &Vector3D,
        radius: f32,
    ) -> Result<Vec<Vector3D>, AnalysisError> {
        let sphere = Sphere::new(reference.clone(), radius);

        system
            .group_iter(group_name!("NormalHeads"))
            .unwrap_or_else(|_|
                panic!("FATAL GORDER ERROR | NoPBC::get_heads_cloud | Could not get NormalHeads group. {}", PANIC_MESSAGE))
            .filter_geometry_naive(sphere)
            .map(|atom| atom
                .get_position()
                .ok_or_else(|| AnalysisError::UndefinedPosition(atom.get_index()))
                .cloned()
            )
            .collect::<Result<Vec<_>, _>>()
    }

    #[inline(always)]
    fn distance(&self, point1: &Vector3D, point2: &Vector3D, dim: Dimension) -> f32 {
        point1.distance_naive(point2, dim)
    }

    #[inline(always)]
    fn atoms_distance(
        &self,
        system: &System,
        index1: usize,
        index2: usize,
        dim: Dimension,
    ) -> Result<f32, AtomError> {
        let atom1 = system.get_atom(index1)?;
        let atom2 = system.get_atom(index2)?;

        atom1.distance_naive(atom2, dim)
    }

    #[inline(always)]
    fn inside<Geom: GeometrySelection>(&self, point: &Vector3D, shape: &Geom) -> bool {
        shape.inside_naive(point)
    }

    #[inline(always)]
    fn vector_to(&self, point1: &Vector3D, point2: &Vector3D) -> Vector3D {
        point2 - point1
    }

    #[inline(always)]
    fn wrap(&self, _point: &mut Vector3D) {} // do nothing; no wrapping is performed if PBC are ignored

    #[inline(always)]
    fn get_infinite_span(&self, pos: &mut f32) -> f32 {
        *pos = f32::MIN;
        f32::INFINITY
    }

    #[inline(always)]
    fn get_box_center(&self) -> Vector3D {
        panic!("FATAL GORDER ERROR | NoPBC::get_box_center | PBC are ignored. Can't get box center. {}", PANIC_MESSAGE);
    }

    #[inline(always)]
    fn get_simbox(&self) -> Option<&SimBox> {
        None
    }

    #[inline(always)]
    fn cylinder_for_local_center(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        orientation: Dimension,
    ) -> Cylinder {
        Cylinder::new(
            Vector3D::new(x, y, f32::MIN),
            radius,
            f32::INFINITY,
            orientation,
        )
    }

    #[inline(always)]
    fn init_new_frame(&mut self) {}

    fn nearby_atoms(
        &'a self,
        system: &'a System,
        reference: Vector3D,
        distance: f32,
    ) -> impl Iterator<Item = (&'a Atom, f32)> {
        system
            .group_iter(group_name!("ClusterHeads"))
            .unwrap_or_else(|_| panic!("FATAL GORDER ERROR | NoPBC::nearby_atoms | Group `ClusterHeads` should exist. {}", PANIC_MESSAGE))
            .filter_map(move |atom| {
                let pos2 = atom.get_position();
                if let Some(pos2) = pos2 {
                    let dist = reference.distance_naive(pos2, Dimension::XYZ);
                    if dist < distance {
                        Some((atom, dist))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
    }
}

/// PBCHandler that assumes periodic boundary conditions in all three dimensions.
#[derive(Debug, Clone)]
pub(crate) struct PBC3D<'a> {
    simbox: &'a SimBox,
    /// Cell grid for membrane atoms.
    membrane_grid: OnceCell<CellGrid<'a>>,
    /// Cell grid for normal heads atoms.
    heads_grid: OnceCell<CellGrid<'a>>,
    /// Cell grid for clustering atoms.
    cluster_grid: OnceCell<CellGrid<'a>>,
}

impl<'a> PBCHandler<'a> for PBC3D<'a> {
    #[inline(always)]
    fn group_get_center(&self, system: &System, group: &str) -> Result<Vector3D, GroupError> {
        system.group_get_center(group)
    }

    fn calc_local_membrane_centers(
        &'a self,
        system: &'a System,
        heads: &[usize],
        radius: f32,
        membrane_normal: Dimension,
    ) -> Result<Vec<Vector3D>, AnalysisError> {
        let cell_grid = self.membrane_grid.get_or_init(|| {
            CellGrid::new(system, group_name!("Membrane"), radius)
            .unwrap_or_else(|e|
                panic!("FATAL GORDER ERROR | PBC3D::calc_local_membrane_centers | Could not construct a cell grid `{}`. {}", e, PANIC_MESSAGE)
            )
        });

        let neighbors = match membrane_normal {
            Dimension::X => CellNeighbors::new(.., -1..=1, -1..=1),
            Dimension::Y => CellNeighbors::new(-1..=1, .., -1..=1),
            Dimension::Z => CellNeighbors::new(-1..=1, -1..=1, ..),
            x => panic!("FATAL GORDER ERROR | PBC3D::calc_local_membrane_centers | Invalid dimension `{}`. {}", x, PANIC_MESSAGE),
        };

        let mut centers = Vec::with_capacity(heads.len());
        for &index in heads.iter() {
            let position = unsafe { system.get_atom_unchecked(index) }
                .get_position()
                .ok_or(AnalysisError::UndefinedPosition(index))?;

            let cylinder =
                self.cylinder_for_local_center(position.x, position.y, radius, membrane_normal);

            let membrane_iterator = cell_grid.neighbors_iter(position.clone(), neighbors.clone());

            let center = membrane_iterator
                .filter_geometry(cylinder)
                .get_center()
                .map_err(|_| AnalysisError::InvalidLocalMembraneCenter(index))?;

            if center.x.is_nan() || center.y.is_nan() || center.z.is_nan() {
                return Err(AnalysisError::InvalidLocalMembraneCenter(index));
            }

            centers.push(center);
        }

        Ok(centers)
    }

    /// The cloud of positions is made whole, i.e., it is not broken at PBC.
    fn get_heads_cloud(
        &'a self,
        system: &'a System,
        reference: &Vector3D,
        radius: f32,
    ) -> Result<Vec<Vector3D>, AnalysisError> {
        let cell_grid = self.heads_grid.get_or_init(|| {
            CellGrid::new(system, group_name!("NormalHeads"), radius)
            .unwrap_or_else(|e|
                panic!("FATAL GORDER ERROR | PBC3D::get_heads_cloud | Could not construct a cell grid `{}`. {}", e, PANIC_MESSAGE)
            )
        });

        let sphere = Sphere::new(reference.clone(), radius);

        let mut positions = Vec::new();
        for atom in cell_grid
            .neighbors_iter(reference.clone(), CellNeighbors::default())
            .filter_geometry(sphere)
        {
            // select images of atoms that are closest to the reference atom
            // if this is not done, the positions might be broken at box boundaries which breaks the PCA
            let position = atom
                .get_position()
                .ok_or_else(|| AnalysisError::UndefinedPosition(atom.get_index()))?;
            let vector = reference.vector_to(position, self.simbox);
            positions.push(reference + vector);
        }

        Ok(positions)
    }

    #[inline(always)]
    fn distance(&self, point1: &Vector3D, point2: &Vector3D, dim: Dimension) -> f32 {
        point1.distance(point2, dim, self.simbox)
    }

    #[inline(always)]
    fn atoms_distance(
        &self,
        system: &System,
        index1: usize,
        index2: usize,
        dim: Dimension,
    ) -> Result<f32, AtomError> {
        let atom1 = system.get_atom(index1)?;
        let atom2 = system.get_atom(index2)?;

        atom1.distance(atom2, dim, self.simbox)
    }

    #[inline(always)]
    fn inside<Geom: GeometrySelection>(&self, point: &Vector3D, shape: &Geom) -> bool {
        shape.inside(point, self.simbox)
    }

    #[inline(always)]
    fn vector_to(&self, point1: &Vector3D, point2: &Vector3D) -> Vector3D {
        // this calculation introduces minor numerical errors compared to the NoPBC version
        // as a result, the results when using PBC3D and NoPBC handlers will differ,
        // even when calculating order parameters for lipids near the box center
        // which are not affected by numerical errors introduced by making the molecules whole
        // these discrepancies become more noticeable with shorter trajectories
        point1.vector_to(point2, self.simbox)
    }

    #[inline(always)]
    fn wrap(&self, point: &mut Vector3D) {
        point.wrap(self.simbox)
    }

    #[inline(always)]
    fn get_infinite_span(&self, pos: &mut f32) -> f32 {
        *pos = 0.0;
        f32::INFINITY
    }

    #[inline(always)]
    fn get_box_center(&self) -> Vector3D {
        Vector3D::new(
            self.simbox.x / 2.0f32,
            self.simbox.y / 2.0f32,
            self.simbox.z / 2.0f32,
        )
    }

    #[inline(always)]
    fn get_simbox(&self) -> Option<&SimBox> {
        Some(self.simbox)
    }

    #[inline(always)]
    fn cylinder_for_local_center(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        orientation: Dimension,
    ) -> Cylinder {
        Cylinder::new(Vector3D::new(x, y, 0.0), radius, f32::INFINITY, orientation)
    }

    #[inline(always)]
    fn init_new_frame(&mut self) {
        self.membrane_grid = OnceCell::new();
        self.heads_grid = OnceCell::new();
        self.cluster_grid = OnceCell::new();
    }

    fn nearby_atoms(
        &'a self,
        system: &'a System,
        reference: Vector3D,
        distance: f32,
    ) -> impl Iterator<Item = (&'a Atom, f32)> {
        let cell_grid = self.cluster_grid.get_or_init(|| {
            CellGrid::new(system, group_name!("ClusterHeads"), distance)
            .unwrap_or_else(|e|
                panic!("FATAL GORDER ERROR | PBC3D::nearby_atoms | Could not construct a cell grid `{}`. {}", e, PANIC_MESSAGE)
            )
        });

        cell_grid
            .neighbors_iter(reference.clone(), CellNeighbors::default())
            .filter_map(move |atom| {
                match atom.distance_from_point(
                    &reference,
                    Dimension::XYZ,
                    system.get_box().expect(PANIC_MESSAGE),
                ) {
                    Ok(x) if x < distance => Some((atom, x)),
                    Ok(_) => None,
                    Err(_) => None,
                }
            })
    }
}

impl<'a> PBC3D<'a> {
    pub(crate) fn new(simbox: &'a SimBox) -> Self {
        Self {
            simbox,
            membrane_grid: OnceCell::new(),
            heads_grid: OnceCell::new(),
            cluster_grid: OnceCell::new(),
        }
    }

    /// Create a periodic boundary handler for a system.
    /// Assumes that the simulation box is defined and orthogonal.
    pub(super) fn from_system(system: &'a System) -> Self {
        Self {
            simbox: system.get_box()
            .unwrap_or_else(||
                panic!("FATAL GORDER ERROR | PBC3D::from_system | Simulation box is undefined but this should have been handled before.")
            ),
            membrane_grid: OnceCell::new(),
            heads_grid: OnceCell::new(),
            cluster_grid: OnceCell::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_filter_geometry_get_pos_pbc3d() {
        let mut system = System::from_file("tests/files/pcpepg.tpr").unwrap();
        system
            .group_create("xxxGorderReservedxxx-NormalHeads", "name P")
            .unwrap();

        let pbc = PBC3D::new(system.get_box().unwrap());

        for radius in [1.5, 2.0, 2.5, 3.0, 3.5, 5.0] {
            for x in [7.5, 8.0, 8.5, 9.0, 10.0] {
                for y in [7.5, 8.0, 8.5, 9.0, 10.0] {
                    for z in [3.0, 3.5, 5.0, 5.5, 6.0] {
                        let reference = Vector3D::new(x, y, z);
                        let positions = pbc.get_heads_cloud(&system, &reference, radius).unwrap();

                        for pos in positions {
                            assert!(reference.distance_naive(&pos, Dimension::XYZ) <= radius);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_group_filter_geometry_get_pos_nopbc() {
        let mut system = System::from_file("tests/files/pcpepg.tpr").unwrap();
        system
            .group_create("xxxGorderReservedxxx-NormalHeads", "name P")
            .unwrap();

        let pbc = NoPBC;

        for radius in [1.5, 2.0, 2.5, 3.0, 3.5, 5.0] {
            for x in [7.5, 8.0, 8.5, 9.0, 10.0] {
                for y in [7.5, 8.0, 8.5, 9.0, 10.0] {
                    for z in [3.0, 3.5, 5.0, 5.5, 6.0] {
                        let reference = Vector3D::new(x, y, z);
                        let positions = pbc.get_heads_cloud(&system, &reference, radius).unwrap();

                        for pos in positions {
                            assert!(reference.distance_naive(&pos, Dimension::XYZ) <= radius);
                        }
                    }
                }
            }
        }
    }
}
