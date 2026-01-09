// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Contains the implementation of the `Geometry` structure and its methods.

use std::fmt::Display;

use getset::{CopyGetters, Getters};
use groan_rs::prelude::Vector3D;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;

use crate::errors::GeometryConfigError;

use super::Axis;

/// Specification of the geometric shape in which bonds should be positioned to be considered.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum Geometry {
    Cuboid(CuboidSelection),
    Cylinder(CylinderSelection),
    Sphere(SphereSelection),
}

impl Geometry {
    /// Construct a cuboid.
    /// Returns an error in case any of the dimensions is invalid.
    ///
    /// ## Parameters
    /// - `reference` - reference point relative to which the
    ///    position of the cuboid is specified
    /// - `xdim` - span of the cuboid along the x-axis
    /// - `ydim` - span of the cuboid along the y-axis
    /// - `zdim` - span of the cuboid along the z-axis
    pub fn cuboid(
        reference: impl Into<GeomReference>,
        xdim: [f32; 2],
        ydim: [f32; 2],
        zdim: [f32; 2],
    ) -> Result<Self, GeometryConfigError> {
        let geometry = Self::Cuboid(CuboidSelection {
            reference: reference.into(),
            xdim,
            ydim,
            zdim,
        });

        geometry.validate()?;
        Ok(geometry)
    }

    /// Construct a cylinder.
    /// Returns an error in case the radius is negative or the span is invalid.
    ///
    /// ## Parameters
    /// - `reference` - reference point relative to which the position
    ///    and size of the cylinder is specified
    /// - `radius` - radius of the cylinder
    /// - `span` - span of the cylinder along its main axis
    /// - `orientation` - orientation of the main axis of the cylinder
    pub fn cylinder(
        reference: impl Into<GeomReference>,
        radius: f32,
        span: [f32; 2],
        orientation: Axis,
    ) -> Result<Self, GeometryConfigError> {
        let geometry = Self::Cylinder(CylinderSelection {
            reference: reference.into(),
            radius,
            span,
            orientation,
        });

        geometry.validate()?;
        Ok(geometry)
    }

    /// Construct a sphere.
    /// Returns an error in case the radius is negative.
    ///
    /// ## Parameters
    /// - `reference` - center of the sphere
    /// - `radius` - radius of the sphere
    pub fn sphere(
        reference: impl Into<GeomReference>,
        radius: f32,
    ) -> Result<Self, GeometryConfigError> {
        let geometry = Self::Sphere(SphereSelection {
            reference: reference.into(),
            radius,
        });

        geometry.validate()?;
        Ok(geometry)
    }

    /// Check that the geometry selection makes sense.
    pub(super) fn validate(&self) -> Result<(), GeometryConfigError> {
        match self {
            Self::Cuboid(x) => {
                for dim in [x.xdim, x.ydim, x.zdim] {
                    if dim[0] > dim[1] {
                        return Err(GeometryConfigError::InvalidDimension(dim[0], dim[1]));
                    }
                }
            }
            Self::Cylinder(x) => {
                if x.radius < 0.0 {
                    return Err(GeometryConfigError::InvalidRadius(x.radius));
                }

                if x.span[0] > x.span[1] {
                    return Err(GeometryConfigError::InvalidSpan(x.span[0], x.span[1]));
                }
            }
            Self::Sphere(x) => {
                if x.radius < 0.0 {
                    return Err(GeometryConfigError::InvalidRadius(x.radius));
                }
            }
        }

        Ok(())
    }

    /// Returns `true` if the geometry requires box information.
    pub(crate) fn uses_box_center(&self) -> bool {
        match self {
            Self::Cuboid(cuboid) => cuboid.reference.uses_box_center(),
            Self::Cylinder(cylinder) => cylinder.reference.uses_box_center(),
            Self::Sphere(sphere) => sphere.reference.uses_box_center(),
        }
    }
}

/// Represents a cuboid used to select bonds located within its bounds.
#[derive(Debug, Clone, Deserialize, Serialize, Getters, CopyGetters)]
#[serde(deny_unknown_fields)]
pub struct CuboidSelection {
    /// The reference point for the cuboid's dimensions. This can be an absolute position or
    /// a selection of atoms whose center of geometry defines the reference point.
    /// Defaults to [0, 0, 0] (the origin of the coordinate system) if not specified.
    #[getset(get = "pub")]
    #[serde(default)]
    reference: GeomReference,
    /// The extent of the cuboid along the x-axis, relative to the reference point [in nm].
    /// If not specified, the cuboid is considered infinitely large along the x-axis.
    #[getset(get_copy = "pub")]
    #[serde(default = "infinite_dim")]
    #[serde(alias = "x", alias = "dim_x")]
    xdim: [f32; 2],
    /// The extent of the cuboid along the y-axis, relative to the reference point [in nm].
    /// If not specified, the cuboid is considered infinitely large along the y-axis.
    #[getset(get_copy = "pub")]
    #[serde(default = "infinite_dim")]
    #[serde(alias = "y", alias = "dim_y")]
    ydim: [f32; 2],
    /// The extent of the cuboid along the z-axis, relative to the reference point [in nm].
    /// If not specified, the cuboid is considered infinitely large along the z-axis.
    #[getset(get_copy = "pub")]
    #[serde(default = "infinite_dim")]
    #[serde(alias = "z", alias = "dim_z")]
    zdim: [f32; 2],
}

/// Represents a cylinder used to select bonds located within its bounds.
#[derive(Debug, Clone, Deserialize, Serialize, Getters, CopyGetters)]
#[serde(deny_unknown_fields)]
pub struct CylinderSelection {
    /// The center of the cylinder, serving as its reference point.
    /// This can be an absolute position or a selection of atoms
    /// whose center of geometry defines the reference point.
    /// Defaults to [0, 0, 0] (the origin of the coordinate system) if not specified.
    #[getset(get = "pub")]
    #[serde(default)]
    reference: GeomReference,
    /// The radius of the cylinder [in nm], defining its circular cross-section.
    #[getset(get_copy = "pub")]
    radius: f32,
    /// Defines the cylinder's extent along its specified axis (`orientation`), from a minimum to a maximum value.
    /// If not specified, the cylinder is considered to have infinite height
    #[getset(get_copy = "pub")]
    #[serde(default = "infinite_dim")]
    span: [f32; 2],
    /// The spatial orientation of the cylinder, determining its alignment in 3D space.
    #[getset(get_copy = "pub")]
    orientation: Axis,
}

fn infinite_dim() -> [f32; 2] {
    [f32::NEG_INFINITY, f32::INFINITY]
}

/// Represents a sphere used to select bonds located within its bounds.
#[derive(Debug, Clone, Deserialize, Serialize, Getters, CopyGetters)]
#[serde(deny_unknown_fields)]
pub struct SphereSelection {
    /// The reference point corresponding to the center of the sphere.
    /// This can be an absolute position or a selection of atoms
    /// whose center of geometry defines the reference point.
    /// Defaults to [0, 0, 0] (the origin of the coordinate system) if not specified.
    #[getset(get = "pub")]
    #[serde(default, alias = "center")]
    reference: GeomReference,
    /// Radius of the sphere [in nm].
    #[getset(get_copy = "pub")]
    radius: f32,
}

/// Coordinates of a point, a GSL selection query, or request to use box center.
#[derive(Debug, Clone)]
pub enum GeomReference {
    Point(Vector3D),
    Selection(String),
    Center,
}

impl Serialize for GeomReference {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            GeomReference::Point(point) => point.serialize(serializer),
            GeomReference::Selection(selection) => selection.serialize(serializer),
            GeomReference::Center => serializer.serialize_str("!Center"),
        }
    }
}

impl<'de> Deserialize<'de> for GeomReference {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Value = Deserialize::deserialize(deserializer)?;

        if let Value::Tagged(tagged) = &value {
            if tagged.tag == "!Center" {
                return Ok(GeomReference::Center);
            }
        }

        if let Ok(point) = serde_yaml::from_value::<Vector3D>(value.clone()) {
            return Ok(GeomReference::Point(point));
        }

        if let Ok(selection) = serde_yaml::from_value::<String>(value) {
            return Ok(GeomReference::Selection(selection));
        }

        Err(serde::de::Error::custom(
            "Could not match any variant of GeomReference",
        ))
    }
}

impl From<Vector3D> for GeomReference {
    fn from(value: Vector3D) -> Self {
        Self::Point(value)
    }
}

impl From<[f32; 3]> for GeomReference {
    fn from(value: [f32; 3]) -> Self {
        Self::Point(value.into())
    }
}

impl From<&str> for GeomReference {
    fn from(value: &str) -> Self {
        GeomReference::Selection(value.to_owned())
    }
}

impl From<String> for GeomReference {
    fn from(value: String) -> Self {
        GeomReference::Selection(value)
    }
}

impl Default for GeomReference {
    fn default() -> Self {
        Self::Point(Vector3D::default())
    }
}

impl Display for GeomReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Point(x) => write!(f, "static point at {}", x),
            Self::Selection(x) => write!(f, "dynamic center of geometry of '{}'", x),
            Self::Center => write!(f, "dynamic center of the simulation box"),
        }
    }
}

impl GeomReference {
    /// Use the origin of the box [0, 0, 0] as a geometric reference.
    #[inline(always)]
    pub fn origin() -> Self {
        GeomReference::Point(Vector3D::default())
    }

    /// Use the box center as a geometric reference. The box center is updated
    /// every analyzed frame.
    #[inline(always)]
    pub fn center() -> Self {
        GeomReference::Center
    }

    /// Returns `true` if the geometric reference should be placed into the center of the simulation box.
    #[inline(always)]
    pub(crate) fn uses_box_center(&self) -> bool {
        match self {
            GeomReference::Point(_) | GeomReference::Selection(_) => false,
            GeomReference::Center => true,
        }
    }
}

#[cfg(test)]
mod pass_tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn geom_reference_default() {
        let default = GeomReference::default();
        match default {
            GeomReference::Point(p) => {
                assert_relative_eq!(p.x, 0.0);
                assert_relative_eq!(p.y, 0.0);
                assert_relative_eq!(p.z, 0.0);
            }
            _ => panic!("Unexpected geometry reference."),
        }
        let origin = GeomReference::origin();
        match origin {
            GeomReference::Point(p) => {
                assert_relative_eq!(p.x, 0.0);
                assert_relative_eq!(p.y, 0.0);
                assert_relative_eq!(p.z, 0.0);
            }
            _ => panic!("Unexpected geometry reference."),
        }
    }

    #[test]
    fn test_cuboid() {
        let geom = Geometry::cuboid(
            Vector3D::new(1.4, 3.2, 1.7),
            [1.4, 2.1],
            [f32::NEG_INFINITY, f32::INFINITY],
            [-0.4, 2.8],
        )
        .unwrap();

        match geom {
            Geometry::Cuboid(cuboid) => {
                match cuboid.reference {
                    GeomReference::Point(p) => {
                        assert_relative_eq!(p.x, 1.4);
                        assert_relative_eq!(p.y, 3.2);
                        assert_relative_eq!(p.z, 1.7);
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(cuboid.xdim[0], 1.4);
                assert_relative_eq!(cuboid.xdim[1], 2.1);

                assert_relative_eq!(cuboid.ydim[0], f32::NEG_INFINITY);
                assert_relative_eq!(cuboid.ydim[1], f32::INFINITY);

                assert_relative_eq!(cuboid.zdim[0], -0.4);
                assert_relative_eq!(cuboid.zdim[1], 2.8);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_cylinder() {
        let geom = Geometry::cylinder([1.4, 3.2, 1.7], 2.5, [0.3, 1.6], Axis::X).unwrap();

        match geom {
            Geometry::Cylinder(cylinder) => {
                match cylinder.reference {
                    GeomReference::Point(p) => {
                        assert_relative_eq!(p.x, 1.4);
                        assert_relative_eq!(p.y, 3.2);
                        assert_relative_eq!(p.z, 1.7);
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(cylinder.radius, 2.5);
                assert_relative_eq!(cylinder.span[0], 0.3);
                assert_relative_eq!(cylinder.span[1], 1.6);

                assert_eq!(cylinder.orientation, Axis::X);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_sphere() {
        let geom = Geometry::sphere("name BB", 4.2).unwrap();

        match geom {
            Geometry::Sphere(sphere) => {
                match sphere.reference {
                    GeomReference::Selection(query) => assert_eq!(query, "name BB"),
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(sphere.radius, 4.2);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_cuboid_yaml_all_default() {
        let string = "!Cuboid";
        match serde_yaml::from_str(string).unwrap() {
            Geometry::Cuboid(cuboid) => {
                match cuboid.reference {
                    GeomReference::Point(p) => {
                        assert_relative_eq!(p.x, 0.0);
                        assert_relative_eq!(p.y, 0.0);
                        assert_relative_eq!(p.z, 0.0);
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(cuboid.xdim[0], f32::NEG_INFINITY);
                assert_relative_eq!(cuboid.xdim[1], f32::INFINITY);
                assert_relative_eq!(cuboid.ydim[0], f32::NEG_INFINITY);
                assert_relative_eq!(cuboid.ydim[1], f32::INFINITY);
                assert_relative_eq!(cuboid.zdim[0], f32::NEG_INFINITY);
                assert_relative_eq!(cuboid.zdim[1], f32::INFINITY);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_cuboid_yaml_some_default() {
        let string = "!Cuboid
  reference: \"Backbone\"
  dim_x: [ 4.3, 8.6 ]
  ydim: [ -10.0, 5.4 ]";

        match serde_yaml::from_str(string).unwrap() {
            Geometry::Cuboid(cuboid) => {
                match cuboid.reference {
                    GeomReference::Selection(s) => {
                        assert_eq!(s, "Backbone");
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(cuboid.xdim[0], 4.3);
                assert_relative_eq!(cuboid.xdim[1], 8.6);
                assert_relative_eq!(cuboid.ydim[0], -10.0);
                assert_relative_eq!(cuboid.ydim[1], 5.4);
                assert_relative_eq!(cuboid.zdim[0], f32::NEG_INFINITY);
                assert_relative_eq!(cuboid.zdim[1], f32::INFINITY);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_cuboid_yaml_none_default() {
        let string = "!Cuboid
  reference: [ 0.0, 1.0, 0.5 ]
  dim_x: [ 4.3, 8.6 ]
  ydim: [ -10.0, 5.4 ]
  z: [ 3.1, 8.5 ]";

        match serde_yaml::from_str(string).unwrap() {
            Geometry::Cuboid(cuboid) => {
                match cuboid.reference {
                    GeomReference::Point(p) => {
                        assert_relative_eq!(p.x, 0.0);
                        assert_relative_eq!(p.y, 1.0);
                        assert_relative_eq!(p.z, 0.5);
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(cuboid.xdim[0], 4.3);
                assert_relative_eq!(cuboid.xdim[1], 8.6);
                assert_relative_eq!(cuboid.ydim[0], -10.0);
                assert_relative_eq!(cuboid.ydim[1], 5.4);
                assert_relative_eq!(cuboid.zdim[0], 3.1);
                assert_relative_eq!(cuboid.zdim[1], 8.5);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_cylinder_yaml_all_default() {
        let string = "!Cylinder
  orientation: z
  radius: 4.0";
        match serde_yaml::from_str(string).unwrap() {
            Geometry::Cylinder(cylinder) => {
                match cylinder.reference {
                    GeomReference::Point(p) => {
                        assert_relative_eq!(p.x, 0.0);
                        assert_relative_eq!(p.y, 0.0);
                        assert_relative_eq!(p.z, 0.0);
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(cylinder.span[0], f32::NEG_INFINITY);
                assert_relative_eq!(cylinder.span[1], f32::INFINITY);
                assert_relative_eq!(cylinder.radius, 4.0);
                assert_eq!(cylinder.orientation, Axis::Z);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_cylinder_yaml_some_default() {
        let string = "!Cylinder
  orientation: z
  radius: 4.0
  span: [-5.4, 7]";
        match serde_yaml::from_str(string).unwrap() {
            Geometry::Cylinder(cylinder) => {
                match cylinder.reference {
                    GeomReference::Point(p) => {
                        assert_relative_eq!(p.x, 0.0);
                        assert_relative_eq!(p.y, 0.0);
                        assert_relative_eq!(p.z, 0.0);
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(cylinder.span[0], -5.4);
                assert_relative_eq!(cylinder.span[1], 7.0);
                assert_relative_eq!(cylinder.radius, 4.0);
                assert_eq!(cylinder.orientation, Axis::Z);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_cylinder_yaml_none_default() {
        let string = "!Cylinder
  orientation: z
  radius: 4.0
  span: [-5.4, 7]
  reference: \"@protein and name CA\"";
        match serde_yaml::from_str(string).unwrap() {
            Geometry::Cylinder(cylinder) => {
                match cylinder.reference {
                    GeomReference::Selection(s) => {
                        assert_eq!(s, "@protein and name CA");
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(cylinder.span[0], -5.4);
                assert_relative_eq!(cylinder.span[1], 7.0);
                assert_relative_eq!(cylinder.radius, 4.0);
                assert_eq!(cylinder.orientation, Axis::Z);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_sphere_yaml_some_default() {
        let string = "!Sphere
  radius: 7.4";
        match serde_yaml::from_str(string).unwrap() {
            Geometry::Sphere(sphere) => {
                match sphere.reference {
                    GeomReference::Point(p) => {
                        assert_relative_eq!(p.x, 0.0);
                        assert_relative_eq!(p.y, 0.0);
                        assert_relative_eq!(p.z, 0.0);
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(sphere.radius, 7.4);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }

    #[test]
    fn test_sphere_yaml_none_default() {
        let string = "!Sphere
  radius: 7.4
  reference: [ 4, 3, 2.5 ]";
        match serde_yaml::from_str(string).unwrap() {
            Geometry::Sphere(sphere) => {
                match sphere.reference {
                    GeomReference::Point(p) => {
                        assert_relative_eq!(p.x, 4.0);
                        assert_relative_eq!(p.y, 3.0);
                        assert_relative_eq!(p.z, 2.5);
                    }
                    _ => panic!("Unexpected geometry reference."),
                }

                assert_relative_eq!(sphere.radius, 7.4);
            }
            _ => panic!("Unexpected geometry constructed."),
        }
    }
}

#[cfg(test)]
mod fail_tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_cuboid_fail_dimx() {
        match Geometry::cuboid(
            GeomReference::default(),
            [3.0, 2.8],
            [0.1, 4.3],
            [-0.4, 3.2],
        ) {
            Ok(_) => panic!("Function should have failed."),
            Err(GeometryConfigError::InvalidDimension(x, y)) => {
                assert_relative_eq!(x, 3.0);
                assert_relative_eq!(y, 2.8);
            }
            Err(e) => panic!("Unexpected error type: {}", e),
        }
    }

    #[test]
    fn test_cuboid_fail_dimy() {
        match Geometry::cuboid(
            GeomReference::default(),
            [3.0, 3.8],
            [0.1, -4.3],
            [-0.4, 3.2],
        ) {
            Ok(_) => panic!("Function should have failed."),
            Err(GeometryConfigError::InvalidDimension(x, y)) => {
                assert_relative_eq!(x, 0.1);
                assert_relative_eq!(y, -4.3);
            }
            Err(e) => panic!("Unexpected error type: {}", e),
        }
    }

    #[test]
    fn test_cuboid_fail_dimz() {
        match Geometry::cuboid(
            GeomReference::default(),
            [3.0, 3.8],
            [0.1, 4.3],
            [-0.4, -0.8],
        ) {
            Ok(_) => panic!("Function should have failed."),
            Err(GeometryConfigError::InvalidDimension(x, y)) => {
                assert_relative_eq!(x, -0.4);
                assert_relative_eq!(y, -0.8);
            }
            Err(e) => panic!("Unexpected error type: {}", e),
        }
    }

    #[test]
    fn test_cylinder_fail_radius() {
        match Geometry::cylinder(
            GeomReference::origin(),
            -1.0,
            [f32::NEG_INFINITY, f32::INFINITY],
            Axis::Y,
        ) {
            Ok(_) => panic!("Function should have failed."),
            Err(GeometryConfigError::InvalidRadius(x)) => {
                assert_relative_eq!(x, -1.0);
            }
            Err(e) => panic!("Unexpected error type: {}", e),
        }
    }

    #[test]
    fn test_cylinder_fail_height() {
        match Geometry::cylinder(GeomReference::origin(), 1.0, [0.4, 0.0], Axis::Y) {
            Ok(_) => panic!("Function should have failed."),
            Err(GeometryConfigError::InvalidSpan(x, y)) => {
                assert_relative_eq!(x, 0.4);
                assert_relative_eq!(y, 0.0);
            }
            Err(e) => panic!("Unexpected error type: {}", e),
        }
    }

    #[test]
    fn test_sphere_fail_radius() {
        match Geometry::sphere(GeomReference::origin(), -1.0) {
            Ok(_) => panic!("Function should have failed."),
            Err(GeometryConfigError::InvalidRadius(x)) => {
                assert_relative_eq!(x, -1.0);
            }
            Err(e) => panic!("Unexpected error type: {}", e),
        }
    }
}
