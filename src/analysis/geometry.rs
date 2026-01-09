// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Handles dynamic geometry selection during the analysis run.

use groan_rs::{
    prelude::{Cylinder, NaiveShape, Rectangular, Shape, SimBox, Sphere, Vector3D},
    system::System,
};

use crate::{
    errors::TopologyError,
    input::{
        geometry::{CuboidSelection, CylinderSelection, GeomReference, SphereSelection},
        Axis, Geometry,
    },
    PANIC_MESSAGE,
};

use super::{common::macros::group_name, pbc::PBCHandler};

/// Enum encompassing all possible geometry selections.
#[derive(Debug, Clone)]
pub(crate) enum GeometrySelectionType {
    None(NoSelection),
    Cuboid(CuboidAnalysis),
    Cylinder(CylinderAnalysis),
    Sphere(SphereAnalysis),
}

impl Default for GeometrySelectionType {
    fn default() -> Self {
        GeometrySelectionType::None(NoSelection::default())
    }
}

impl GeometrySelectionType {
    /// Construct a geometry selection type from the input geometry.
    pub(super) fn from_geometry<'a>(
        geometry: Option<&Geometry>,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Self {
        match geometry {
            None => GeometrySelectionType::None(NoSelection::default()),
            Some(Geometry::Cuboid(cuboid)) => {
                GeometrySelectionType::Cuboid(CuboidAnalysis::new(cuboid, pbc_handler))
            }
            Some(Geometry::Cylinder(cylinder)) => {
                GeometrySelectionType::Cylinder(CylinderAnalysis::new(cylinder, pbc_handler))
            }
            Some(Geometry::Sphere(sphere)) => {
                GeometrySelectionType::Sphere(SphereAnalysis::new(sphere, pbc_handler))
            }
        }
    }

    /// Log basic information about the performed geometry selection.
    pub(super) fn info(&self) {
        match self {
            GeometrySelectionType::None(_) => (),
            GeometrySelectionType::Cuboid(cuboid) => {
                colog_info!(
                    "Will only consider bonds located inside a {}:
  x-dimension: from {} nm to {} nm
  y-dimension: from {} nm to {} nm
  z-dimension: from {} nm to {} nm
  relative to {}",
                    "cuboid",
                    cuboid.properties.xdim()[0],
                    cuboid.properties.xdim()[1],
                    cuboid.properties.ydim()[0],
                    cuboid.properties.ydim()[1],
                    cuboid.properties.zdim()[0],
                    cuboid.properties.zdim()[1],
                    cuboid.properties.reference(),
                );
            }
            GeometrySelectionType::Cylinder(cylinder) => {
                colog_info!(
                    "Will only consider bonds located inside a {}:
  radius: {} nm
  oriented along the {} axis 
  going from {} nm to {} nm along the {} axis
  relative to {}",
                    "cylinder",
                    cylinder.properties.radius(),
                    cylinder.properties.orientation(),
                    cylinder.properties.span()[0],
                    cylinder.properties.span()[1],
                    cylinder.properties.orientation(),
                    cylinder.properties.reference()
                )
            }
            GeometrySelectionType::Sphere(sphere) => {
                colog_info!(
                    "Will only consider bonds located inside a {}:
  radius: {} nm
  center: {}",
                    "sphere",
                    sphere.properties.radius(),
                    sphere.properties.reference()
                )
            }
        }
    }

    /// Initialize the reading of a new frame (calculate and set new reference position if needed).
    #[inline]
    pub(super) fn init_new_frame<'a>(
        &mut self,
        system: &System,
        pbc_handler: &impl PBCHandler<'a>,
    ) {
        match self {
            GeometrySelectionType::None(_) => (),
            GeometrySelectionType::Cuboid(x) => x.init_reference(system, pbc_handler),
            GeometrySelectionType::Cylinder(x) => x.init_reference(system, pbc_handler),
            GeometrySelectionType::Sphere(x) => x.init_reference(system, pbc_handler),
        }
    }
}

/// Trait implemented by all structures that can be used for geometry selection.
pub(crate) trait GeometrySelection: Send + Sync {
    type Shape: Shape + NaiveShape;
    type Properties;

    /// Create the structure from the provided properties.
    fn new<'a>(properties: &Self::Properties, pbc_handler: &impl PBCHandler<'a>) -> Self;

    /// Get the reference point of the geometry selection.
    fn reference(&self) -> &GeomReference;

    /// Get the properties of the geometry selection.
    fn properties(&self) -> &Self::Properties;

    /// Get the inner shape of the geometry selection.
    fn shape(&self) -> &Self::Shape;

    /// Set the inner shape of the geometry selection.
    fn set_shape(&mut self, shape: Self::Shape);

    /// Construct the inner shape of the geometry selection with the given point being used as reference.
    fn construct_shape<'a>(
        properties: &Self::Properties,
        point: Vector3D,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Self::Shape;

    /// Prepare the system for geometry selection, i.e. construct the required groups.
    fn prepare_system(&self, system: &mut System) -> Result<(), TopologyError> {
        match self.reference() {
            GeomReference::Point(_) | GeomReference::Center => Ok(()), // nothing to do
            GeomReference::Selection(query) => {
                // construct a group for geometry reference
                super::common::create_group(system, "GeomReference", query)
            }
        }
    }

    /// Is the point inside the geometry? Take PBC into consideration.
    #[inline(always)]
    fn inside(&self, point: &Vector3D, simbox: &SimBox) -> bool {
        self.shape().inside(point, simbox)
    }

    /// Is the point inside the geometry? Ignore PBC.
    #[inline(always)]
    fn inside_naive(&self, point: &Vector3D) -> bool {
        self.shape().inside_naive(point)
    }

    /// Calculate and set the reference position for this frame.
    fn init_reference<'a>(&mut self, system: &System, pbc_handler: &impl PBCHandler<'a>) {
        let reference_point = match self.reference() {
            GeomReference::Point(_) => return, // nothing to do, reference position is fixed
            GeomReference::Selection(_) => {
                // calculate the center of geometry
                pbc_handler
                    .group_get_center(system, group_name!("GeomReference"))
                    .unwrap_or_else(|_| panic!("FATAL GORDER ERROR | GeometrySelection::init_reference | Group specifying geometry reference should exist. {}", PANIC_MESSAGE))
            }
            GeomReference::Center => {
                // get the box center
                pbc_handler.get_box_center()
            }
        };

        let shape = Self::construct_shape(self.properties(), reference_point, pbc_handler);

        self.set_shape(shape);
    }
}

/// No geometry selection requested. Order parameters will be calculated for all bonds, no matter where they are.
#[derive(Debug, Clone, Default)]
pub(crate) struct NoSelection {}

impl GeometrySelection for NoSelection {
    type Shape = Sphere; // arbitrary shape
    type Properties = (); // arbitrary properties

    #[inline(always)]
    fn new<'a>(_geometry: &Self::Properties, _pbc_handler: &impl PBCHandler<'a>) -> Self {
        Self {}
    }

    #[inline(always)]
    fn reference(&self) -> &GeomReference {
        panic!(
            "FATAL GORDER ERROR | NoSelection::reference | This method should never be called. {}",
            PANIC_MESSAGE
        );
    }

    #[inline(always)]
    fn properties(&self) -> &Self::Properties {
        panic!(
            "FATAL GORDER ERROR | NoSelection::properties | This method should never be called. {}",
            PANIC_MESSAGE
        );
    }

    #[inline(always)]
    fn shape(&self) -> &Self::Shape {
        panic!(
            "FATAL GORDER ERROR | NoSelection::shape | This method should never be called. {}",
            PANIC_MESSAGE
        );
    }

    #[inline(always)]
    fn set_shape(&mut self, _shape: Self::Shape) {}

    #[inline(always)]
    fn construct_shape<'a>(
        _properties: &Self::Properties,
        _point: Vector3D,
        _pbc_handler: &impl PBCHandler<'a>,
    ) -> Self::Shape {
        panic!("FATAL GORDER ERROR | NoSelection::construct_shape | This method should never be called. {}", PANIC_MESSAGE);
    }

    #[inline(always)]
    fn init_reference<'a>(&mut self, _system: &System, _pbc_handler: &impl PBCHandler<'a>) {}

    #[inline(always)]
    fn inside(&self, _point: &Vector3D, _simbox: &SimBox) -> bool {
        true
    }

    #[inline(always)]
    fn inside_naive(&self, _point: &Vector3D) -> bool {
        true
    }

    #[inline(always)]
    fn prepare_system(&self, _system: &mut System) -> Result<(), TopologyError> {
        Ok(())
    }
}

/// Cuboid geometry selection.
#[derive(Debug, Clone)]
pub(crate) struct CuboidAnalysis {
    properties: CuboidSelection,
    shape: Rectangular,
}

impl GeometrySelection for CuboidAnalysis {
    type Shape = Rectangular;
    type Properties = CuboidSelection;

    fn new<'a>(properties: &CuboidSelection, pbc_handler: &impl PBCHandler<'a>) -> Self {
        let reference_point = match properties.reference() {
            // fixed value
            GeomReference::Point(x) => x.clone(),
            // we set any value; reference will be set for each frame inside `init_reference`
            GeomReference::Selection(_) | GeomReference::Center => Vector3D::default(),
        };

        let shape = CuboidAnalysis::construct_shape(properties, reference_point, pbc_handler);

        CuboidAnalysis {
            properties: properties.clone(),
            shape,
        }
    }

    #[inline(always)]
    fn reference(&self) -> &GeomReference {
        self.properties.reference()
    }

    #[inline(always)]
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }

    #[inline(always)]
    fn properties(&self) -> &Self::Properties {
        &self.properties
    }

    fn construct_shape<'a>(
        properties: &Self::Properties,
        mut point: Vector3D,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Self::Shape {
        #[inline(always)]
        fn compute_dimension<'a>(
            dim: [f32; 2],
            reference_component: &mut f32,
            pbc_handler: &impl PBCHandler<'a>,
        ) -> f32 {
            match dim {
                [f32::NEG_INFINITY, f32::INFINITY] => {
                    pbc_handler.get_infinite_span(reference_component)
                }
                [min, max] => {
                    *reference_component += min;
                    max - min
                }
            }
        }

        let x = compute_dimension(properties.xdim(), &mut point.x, pbc_handler);
        let y = compute_dimension(properties.ydim(), &mut point.y, pbc_handler);
        let z = compute_dimension(properties.zdim(), &mut point.z, pbc_handler);

        pbc_handler.wrap(&mut point);

        Rectangular::new(point, x, y, z)
    }

    #[inline(always)]
    fn set_shape(&mut self, shape: Self::Shape) {
        self.shape = shape;
    }
}

/// Cylindrical geometry selection.
#[derive(Debug, Clone)]
pub(crate) struct CylinderAnalysis {
    properties: CylinderSelection,
    shape: Cylinder,
}

impl GeometrySelection for CylinderAnalysis {
    type Shape = Cylinder;
    type Properties = CylinderSelection;

    fn new<'a>(properties: &CylinderSelection, pbc_handler: &impl PBCHandler<'a>) -> Self {
        let reference_point = match properties.reference() {
            // fixed value
            GeomReference::Point(x) => x.clone(),
            // we set any value; reference will be set for each frame inside `init_reference`
            GeomReference::Selection(_) | GeomReference::Center => Vector3D::default(),
        };

        let shape = CylinderAnalysis::construct_shape(properties, reference_point, pbc_handler);

        CylinderAnalysis {
            properties: properties.clone(),
            shape,
        }
    }

    #[inline(always)]
    fn properties(&self) -> &Self::Properties {
        &self.properties
    }

    #[inline(always)]
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }

    #[inline(always)]
    fn set_shape(&mut self, shape: Self::Shape) {
        self.shape = shape;
    }

    #[inline(always)]
    fn reference(&self) -> &GeomReference {
        self.properties.reference()
    }

    fn construct_shape<'a>(
        properties: &Self::Properties,
        mut point: Vector3D,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Self::Shape {
        let height = match properties.span() {
            [f32::NEG_INFINITY, f32::INFINITY] => match properties.orientation() {
                Axis::X => pbc_handler.get_infinite_span(&mut point.x),
                Axis::Y => pbc_handler.get_infinite_span(&mut point.y),
                Axis::Z => pbc_handler.get_infinite_span(&mut point.z),
            },
            _ => {
                match properties.orientation() {
                    Axis::X => point.x += properties.span()[0],
                    Axis::Y => point.y += properties.span()[0],
                    Axis::Z => point.z += properties.span()[0],
                }
                properties.span()[1] - properties.span()[0]
            }
        };

        pbc_handler.wrap(&mut point);

        Cylinder::new(
            point,
            properties.radius(),
            height,
            properties.orientation().into(),
        )
    }
}

/// Spherical geometry selection.
#[derive(Debug, Clone)]
pub(crate) struct SphereAnalysis {
    properties: SphereSelection,
    shape: Sphere,
}

impl GeometrySelection for SphereAnalysis {
    type Shape = Sphere;
    type Properties = SphereSelection;

    fn new<'a>(properties: &Self::Properties, pbc_handler: &impl PBCHandler<'a>) -> Self {
        let reference_point = match properties.reference() {
            // fixed value
            GeomReference::Point(x) => x.clone(),
            // we set any value; reference will be set for each frame inside `init_reference`
            GeomReference::Selection(_) | GeomReference::Center => Vector3D::default(),
        };

        let shape = SphereAnalysis::construct_shape(properties, reference_point, pbc_handler);

        SphereAnalysis {
            properties: properties.clone(),
            shape,
        }
    }

    #[inline(always)]
    fn properties(&self) -> &Self::Properties {
        &self.properties
    }

    #[inline(always)]
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }

    #[inline(always)]
    fn set_shape(&mut self, shape: Self::Shape) {
        self.shape = shape;
    }

    #[inline(always)]
    fn reference(&self) -> &GeomReference {
        self.properties.reference()
    }

    #[inline(always)]
    fn construct_shape<'a>(
        properties: &Self::Properties,
        mut point: Vector3D,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Self::Shape {
        pbc_handler.wrap(&mut point);
        Sphere::new(point, properties.radius())
    }
}

#[cfg(test)]
mod tests_cuboid {
    use approx::assert_relative_eq;
    use rand::prelude::*;

    use crate::analysis::pbc::PBC3D;

    use super::*;

    #[test]
    fn test_construct_shape_origin() {
        let cuboid =
            match Geometry::cuboid("@protein", [2.5, 3.1], [-1.5, 3.5], [-1.0, 1.0]).unwrap() {
                Geometry::Cuboid(x) => x,
                _ => panic!("Invalid geometry."),
            };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape = CuboidAnalysis::construct_shape(&cuboid, Vector3D::new(0.0, 0.0, 0.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 2.5);
        assert_relative_eq!(position.y, 4.5);
        assert_relative_eq!(position.z, 7.0);

        assert_relative_eq!(shape.get_x(), 0.6);
        assert_relative_eq!(shape.get_y(), 5.0);
        assert_relative_eq!(shape.get_z(), 2.0);
    }

    #[test]
    fn test_construct_shape_simple() {
        let cuboid =
            match Geometry::cuboid("@protein", [2.5, 3.1], [-1.5, 3.5], [-1.0, 1.0]).unwrap() {
                Geometry::Cuboid(x) => x,
                _ => panic!("Invalid geometry."),
            };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape = CuboidAnalysis::construct_shape(&cuboid, Vector3D::new(8.0, 5.5, 2.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 0.5);
        assert_relative_eq!(position.y, 4.0);
        assert_relative_eq!(position.z, 1.0);

        assert_relative_eq!(shape.get_x(), 0.6);
        assert_relative_eq!(shape.get_y(), 5.0);
        assert_relative_eq!(shape.get_z(), 2.0);
    }

    #[test]
    fn test_construct_shape_infinity() {
        let cuboid = match Geometry::cuboid(
            "@protein",
            [2.5, 3.1],
            [f32::NEG_INFINITY, f32::INFINITY],
            [-1.0, 1.0],
        )
        .unwrap()
        {
            Geometry::Cuboid(x) => x,
            _ => panic!("Invalid geometry."),
        };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape = CuboidAnalysis::construct_shape(&cuboid, Vector3D::new(15.0, 5.5, 1.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 7.5);
        assert_relative_eq!(position.y, 0.0);
        assert_relative_eq!(position.z, 0.0);

        assert_relative_eq!(shape.get_x(), 0.6);
        assert_relative_eq!(shape.get_y(), f32::INFINITY);
        assert_relative_eq!(shape.get_z(), 2.0);

        let point = Vector3D::new(7.8, -124.4, 1.5);
        assert!(shape.inside(&point, &simbox));
        let point = Vector3D::new(7.8, 124.4, 1.5);
        assert!(shape.inside(&point, &simbox));
    }

    #[test]
    fn test_inside_random() {
        let mut rng = StdRng::seed_from_u64(1288746347198273);
        let simbox = SimBox::from([10.0, 10.0, 10.0]);
        let pbc = PBC3D::new(&simbox);

        for i in 0..100 {
            let xmin: f32 = rng.random_range(0.0..10.0);
            let xmax = rng.random_range(0.0..10.0);

            let ymin: f32 = rng.random_range(0.0..10.0);
            let ymax = rng.random_range(0.0..10.0);

            let zmin: f32 = rng.random_range(0.0..10.0);
            let zmax = rng.random_range(0.0..10.0);

            let mut xrange = [xmin, xmax];
            xrange.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if i % 8 == 0 {
                xrange = [f32::NEG_INFINITY, f32::INFINITY];
            }

            let mut yrange = [ymin, ymax];
            yrange.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if i % 10 == 0 {
                yrange = [f32::NEG_INFINITY, f32::INFINITY];
            }

            let mut zrange = [zmin, zmax];
            zrange.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if i % 6 == 0 {
                zrange = [f32::NEG_INFINITY, f32::INFINITY];
            }

            let cuboid = match Geometry::cuboid("@protein", xrange, yrange, zrange).unwrap() {
                Geometry::Cuboid(x) => x,
                _ => panic!("Invalid geometry."),
            };

            let shape = CuboidAnalysis::construct_shape(&cuboid, Vector3D::default(), &pbc);

            for _ in 0..1000 {
                let pos_x = rng.random_range(0.0..10.0);
                let pos_y = rng.random_range(0.0..10.0);
                let pos_z = rng.random_range(0.0..10.0);

                let point = Vector3D::new(pos_x, pos_y, pos_z);

                let is_inside = point.x > cuboid.xdim()[0]
                    && point.x < cuboid.xdim()[1]
                    && point.y > cuboid.ydim()[0]
                    && point.y < cuboid.ydim()[1]
                    && point.z > cuboid.zdim()[0]
                    && point.z < cuboid.zdim()[1];

                assert_eq!(is_inside, shape.inside(&point, &simbox));
            }
        }
    }

    #[test]
    fn test_prepare_system_init_reference() {
        let mut system = System::from_file("tests/files/pcpepg.tpr").unwrap();

        // fixed point
        let cuboid = match Geometry::cuboid(
            Vector3D::new(3.0, 1.0, 2.0),
            [-1.0, 2.0],
            [1.5, 2.5],
            [f32::NEG_INFINITY, f32::INFINITY],
        )
        .unwrap()
        {
            Geometry::Cuboid(x) => x,
            _ => panic!("Invalid geometry."),
        };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = CuboidAnalysis::new(&cuboid, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);
        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, 2.0);
        assert_relative_eq!(point.y, 2.5);
        assert_relative_eq!(point.z, 0.0);

        // center of geometry
        let cuboid = match Geometry::cuboid(
            "@membrane",
            [-1.0, 2.0],
            [1.5, 2.5],
            [f32::NEG_INFINITY, f32::INFINITY],
        )
        .unwrap()
        {
            Geometry::Cuboid(x) => x,
            _ => panic!("Invalid geometry."),
        };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = CuboidAnalysis::new(&cuboid, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);

        let mut membrane_center = system
            .group_get_center(group_name!("GeomReference"))
            .unwrap();
        membrane_center.x -= 1.0;
        membrane_center.y += 1.5;
        membrane_center.wrap(system.get_box().unwrap());

        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, membrane_center.x);
        assert_relative_eq!(point.y, membrane_center.y);
        assert_relative_eq!(point.z, 0.0);

        // box center
        let cuboid = match Geometry::cuboid(
            GeomReference::center(),
            [-1.0, 2.0],
            [1.5, 2.5],
            [f32::NEG_INFINITY, f32::INFINITY],
        )
        .unwrap()
        {
            Geometry::Cuboid(x) => x,
            _ => panic!("Invalid geometry."),
        };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = CuboidAnalysis::new(&cuboid, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);

        let box_center = system.get_box_center().unwrap();
        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, box_center.x - 1.0);
        assert_relative_eq!(point.y, box_center.y + 1.5);
        assert_relative_eq!(point.z, 0.0);
    }
}

#[cfg(test)]
mod tests_cylinder {
    use approx::assert_relative_eq;
    use groan_rs::prelude::Dimension;

    use crate::analysis::pbc::PBC3D;

    use super::*;

    #[test]
    fn test_construct_shape_origin() {
        let cylinder = match Geometry::cylinder("@protein", 2.5, [-1.5, 3.5], Axis::Y).unwrap() {
            Geometry::Cylinder(x) => x,
            _ => panic!("Invalid geometry."),
        };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape =
            CylinderAnalysis::construct_shape(&cylinder, Vector3D::new(0.0, 0.0, 0.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 0.0);
        assert_relative_eq!(position.y, 4.5);
        assert_relative_eq!(position.z, 0.0);

        assert_relative_eq!(shape.get_radius(), 2.5);
        assert_relative_eq!(shape.get_height(), 5.0);
        assert_eq!(shape.get_orientation(), Dimension::Y);
    }

    #[test]
    fn test_construct_shape_simple() {
        let cylinder = match Geometry::cylinder("@protein", 2.5, [-1.5, 3.5], Axis::X).unwrap() {
            Geometry::Cylinder(x) => x,
            _ => panic!("Invalid geometry."),
        };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape =
            CylinderAnalysis::construct_shape(&cylinder, Vector3D::new(8.0, 5.5, 2.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 6.5);
        assert_relative_eq!(position.y, 5.5);
        assert_relative_eq!(position.z, 2.0);

        assert_relative_eq!(shape.get_radius(), 2.5);
        assert_relative_eq!(shape.get_height(), 5.0);
        assert_eq!(shape.get_orientation(), Dimension::X);
    }

    #[test]
    fn test_construct_shape_infinity() {
        let cylinder =
            match Geometry::cylinder("@protein", 2.5, [f32::NEG_INFINITY, f32::INFINITY], Axis::Z)
                .unwrap()
            {
                Geometry::Cylinder(x) => x,
                _ => panic!("Invalid geometry."),
            };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape =
            CylinderAnalysis::construct_shape(&cylinder, Vector3D::new(8.0, 5.5, 2.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 8.0);
        assert_relative_eq!(position.y, 5.5);
        assert_relative_eq!(position.z, 0.0);

        assert_relative_eq!(shape.get_radius(), 2.5);
        assert_relative_eq!(shape.get_height(), f32::INFINITY);
        assert_eq!(shape.get_orientation(), Dimension::Z);

        let point = Vector3D::new(7.8, 1.1, -932.2);
        assert!(shape.inside(&point, &simbox));
        let point = Vector3D::new(7.8, 1.1, 923.1);
        assert!(shape.inside(&point, &simbox));
    }

    #[test]
    fn test_prepare_system_init_reference() {
        let mut system = System::from_file("tests/files/pcpepg.tpr").unwrap();

        // fixed point
        let cylinder =
            match Geometry::cylinder(Vector3D::new(3.0, 1.0, 2.0), 3.5, [1.5, 2.5], Axis::Y)
                .unwrap()
            {
                Geometry::Cylinder(x) => x,
                _ => panic!("Invalid geometry."),
            };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = CylinderAnalysis::new(&cylinder, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);
        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, 3.0);
        assert_relative_eq!(point.y, 2.5);
        assert_relative_eq!(point.z, 2.0);

        // center of geometry
        let cylinder = match Geometry::cylinder("@membrane", 3.5, [1.5, 2.5], Axis::Z).unwrap() {
            Geometry::Cylinder(x) => x,
            _ => panic!("Invalid geometry."),
        };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = CylinderAnalysis::new(&cylinder, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);

        let mut membrane_center = system
            .group_get_center(group_name!("GeomReference"))
            .unwrap();
        membrane_center.z += 1.5;
        membrane_center.wrap(system.get_box().unwrap());

        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, membrane_center.x);
        assert_relative_eq!(point.y, membrane_center.y);
        assert_relative_eq!(point.z, membrane_center.z);

        // box center
        let cylinder = match Geometry::cylinder(
            GeomReference::center(),
            3.5,
            [f32::NEG_INFINITY, f32::INFINITY],
            Axis::X,
        )
        .unwrap()
        {
            Geometry::Cylinder(x) => x,
            _ => panic!("Invalid geometry."),
        };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = CylinderAnalysis::new(&cylinder, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);

        let box_center = system.get_box_center().unwrap();
        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, 0.0);
        assert_relative_eq!(point.y, box_center.y);
        assert_relative_eq!(point.z, box_center.z);
    }
}

#[cfg(test)]
mod tests_sphere {
    use approx::assert_relative_eq;

    use crate::analysis::pbc::PBC3D;

    use super::*;

    #[test]
    fn test_construct_shape_origin() {
        let sphere = match Geometry::sphere("@protein", 1.8).unwrap() {
            Geometry::Sphere(x) => x,
            _ => panic!("Invalid geometry."),
        };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape = SphereAnalysis::construct_shape(&sphere, Vector3D::new(0.0, 0.0, 0.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 0.0);
        assert_relative_eq!(position.y, 0.0);
        assert_relative_eq!(position.z, 0.0);

        assert_relative_eq!(shape.get_radius(), 1.8);
    }

    #[test]
    fn test_construct_shape_simple() {
        let sphere = match Geometry::sphere("@protein", 2.3).unwrap() {
            Geometry::Sphere(x) => x,
            _ => panic!("Invalid geometry."),
        };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape = SphereAnalysis::construct_shape(&sphere, Vector3D::new(2.0, 7.0, 5.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 2.0);
        assert_relative_eq!(position.y, 1.0);
        assert_relative_eq!(position.z, 5.0);

        assert_relative_eq!(shape.get_radius(), 2.3);
    }

    #[test]
    fn test_construct_shape_infinity() {
        let sphere = match Geometry::sphere("@protein", f32::INFINITY).unwrap() {
            Geometry::Sphere(x) => x,
            _ => panic!("Invalid geometry."),
        };

        let simbox = SimBox::from([10.0, 6.0, 8.0]);
        let pbc = PBC3D::new(&simbox);

        let shape = SphereAnalysis::construct_shape(&sphere, Vector3D::new(2.0, 7.0, 5.0), &pbc);

        let position = shape.get_position();
        assert_relative_eq!(position.x, 2.0);
        assert_relative_eq!(position.y, 1.0);
        assert_relative_eq!(position.z, 5.0);

        assert_relative_eq!(shape.get_radius(), f32::INFINITY);

        let point = Vector3D::new(-19347.2, 9784.1, 9372.0);
        assert!(shape.inside(&point, &simbox));
    }

    #[test]
    fn test_prepare_system_init_reference() {
        let mut system = System::from_file("tests/files/pcpepg.tpr").unwrap();

        // fixed point
        let sphere = match Geometry::sphere(Vector3D::new(3.0, 1.0, 2.0), 1.5).unwrap() {
            Geometry::Sphere(x) => x,
            _ => panic!("Invalid geometry."),
        };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = SphereAnalysis::new(&sphere, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);
        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, 3.0);
        assert_relative_eq!(point.y, 1.0);
        assert_relative_eq!(point.z, 2.0);

        // center of geometry
        let sphere = match Geometry::sphere("@membrane", 1.5).unwrap() {
            Geometry::Sphere(x) => x,
            _ => panic!("Invalid geometry."),
        };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = SphereAnalysis::new(&sphere, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);

        let membrane_center = system
            .group_get_center(group_name!("GeomReference"))
            .unwrap();
        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, membrane_center.x);
        assert_relative_eq!(point.y, membrane_center.y);
        assert_relative_eq!(point.z, membrane_center.z);

        // box center
        let sphere = match Geometry::sphere(GeomReference::center(), 1.5).unwrap() {
            Geometry::Sphere(x) => x,
            _ => panic!("Invalid geometry."),
        };
        let simbox = system.get_box().unwrap().clone();
        let pbc = PBC3D::new(&simbox);
        let mut geometry = SphereAnalysis::new(&sphere, &pbc);
        geometry.prepare_system(&mut system).unwrap();
        geometry.init_reference(&system, &pbc);

        let box_center = system.get_box_center().unwrap();
        let shape = geometry.shape();
        let point = shape.get_position();
        assert_relative_eq!(point.x, box_center.x);
        assert_relative_eq!(point.y, box_center.y);
        assert_relative_eq!(point.z, box_center.z);
    }
}
