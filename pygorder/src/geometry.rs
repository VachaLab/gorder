// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

use gorder_core::prelude::Vector3D;
use pyo3::prelude::*;

use gorder_core::input::GeomReference as RsReference;
use gorder_core::input::Geometry as RsGeometry;
use pyo3_stub_gen::derive::gen_stub_pyclass;
use pyo3_stub_gen::derive::gen_stub_pymethods;

use crate::string2axis;
use crate::ConfigError;

macro_rules! try_extract {
    ($obj:expr, $( $t:ty ),*) => {
        $(
            if let Ok(geometry_type) = $obj.extract::<$t>() {
                return Ok(Self(geometry_type.0));
            }
        )*
    };
}

/// Helper structure for geometry selection.
#[derive(Clone)]
pub struct Geometry(pub(crate) RsGeometry);

impl<'source> FromPyObject<'source, '_> for Geometry {
    type Error = PyErr;

    fn extract(obj: Borrowed<'source, '_, PyAny>) -> PyResult<Self> {
        try_extract!(obj, Cuboid, Cylinder, Sphere);

        Err(ConfigError::new_err(
            "expected an instance of Cuboid, Cylinder, or Sphere",
        ))
    }
}

/// Calculate order parameters inside a cuboid.
///
/// Parameters
/// ----------
/// xdim : Optional[Sequence[float]]
///     Span of the cuboid along the x-axis [nm]. Defaults to infinite if not specified.
/// ydim : Optional[Sequence[float]]
///     Span of the cuboid along the y-axis [nm]. Defaults to infinite if not specified.
/// zdim : Optional[Sequence[float]]
///     Span of the cuboid along the z-axis [nm]. Defaults to infinite if not specified.
/// reference : Optional[Union[Sequence[float],str]]
///     Reference point for the cuboid position. Defaults to [0.0, 0.0, 0.0].
/// invert : Optional[bool]
///     Should the geometry selection be inverted? Defaults to False.
///
/// Raises
/// ------
/// ConfigError
///     If any dimension is invalid.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.geometry")]
#[derive(Clone)]
pub struct Cuboid(RsGeometry);

#[gen_stub_pymethods]
#[pymethods]
impl Cuboid {
    #[new]
    #[pyo3(signature = (
        xdim = [f32::NEG_INFINITY, f32::INFINITY],
        ydim = [f32::NEG_INFINITY, f32::INFINITY],
        zdim = [f32::NEG_INFINITY, f32::INFINITY],
        reference = None,
        invert = false))]
    pub fn new(
        xdim: [f32; 2],
        ydim: [f32; 2],
        zdim: [f32; 2],
        #[gen_stub(override_type(
            type_repr = "typing.Union[typing.Sequence[builtins.float], builtins.str, None]", imports = ("typing")
        ))]
        reference: Option<Bound<'_, PyAny>>,
        invert: bool,
    ) -> PyResult<Self> {
        let converted_ref = if let Some(reference) = reference {
            GeomReference::extract(reference.as_borrowed())?
        } else {
            GeomReference(RsReference::Point(Vector3D::default()))
        };

        Ok(Self(
            RsGeometry::cuboid(converted_ref.0, xdim, ydim, zdim)
                .map_err(|e| ConfigError::new_err(e.to_string()))?
                .with_invert(invert),
        ))
    }
}

/// Calculate order parameters inside a cylinder.
///
/// Parameters
/// ----------
/// radius : float
///     Radius of the cylinder [nm].
/// orientation : str
///     Orientation of the cylinder's main axis. Expected values are x, y, or z.
/// span : Optional[Sequence[float]]
///     Span along the main axis [nm]. Defaults to infinite if not specified.
/// reference : Optional[Union[Sequence[float],str]]
///     Reference point for position and size. Defaults to [0.0, 0.0, 0.0].
/// invert : Optional[bool]
///     Should the geometry selection be inverted? Defaults to False.
///
/// Raises
/// ------
/// ConfigError
///     If `radius` is not positive, `span` is invalid, or `orientation` is not recognized.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.geometry")]
#[derive(Clone)]
pub struct Cylinder(RsGeometry);

#[gen_stub_pymethods]
#[pymethods]
impl Cylinder {
    #[new]
    #[pyo3(signature = (
        radius,
        orientation,
        span = [f32::NEG_INFINITY, f32::INFINITY],
        reference = None,
        invert = false))]
    pub fn new<'a>(
        radius: f32,
        orientation: &str,
        span: [f32; 2],
        #[gen_stub(override_type(
            type_repr = "typing.Union[typing.Sequence[builtins.float], builtins.str, None]", imports = ("typing")
        ))]
        reference: Option<Bound<'a, PyAny>>,
        invert: bool,
    ) -> PyResult<Self> {
        let converted_ref = if let Some(reference) = reference {
            GeomReference::extract(reference.as_borrowed())?
        } else {
            GeomReference(RsReference::Point(Vector3D::default()))
        };

        Ok(Self(
            RsGeometry::cylinder(converted_ref.0, radius, span, string2axis(orientation)?)
                .map_err(|e| ConfigError::new_err(e.to_string()))?
                .with_invert(invert),
        ))
    }
}

/// Calculate order parameters inside a sphere.
///
/// Parameters
/// ----------
/// radius : float
///     Radius of the sphere [nm].
/// reference : Optional[Union[Sequence[float],str]]
///     Center of the sphere. Defaults to [0.0, 0.0, 0.0].
/// invert : Optional[bool]
///     Should the geometry selection be inverted? Defaults to False.
///
/// Raises
/// ------
/// ConfigError
///     If `radius` is not positive.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.geometry")]
#[derive(Clone)]
pub struct Sphere(RsGeometry);

#[gen_stub_pymethods]
#[pymethods]
impl Sphere {
    #[new]
    #[pyo3(signature = (
        radius,
        reference = None,
        invert = false))]
    pub fn new<'a>(
        radius: f32,
        #[gen_stub(override_type(
            type_repr = "typing.Union[typing.Sequence[builtins.float], builtins.str, None]", imports = ("typing")
        ))]
        reference: Option<Bound<'a, PyAny>>,
        invert: bool,
    ) -> PyResult<Self> {
        let converted_ref = if let Some(reference) = reference {
            GeomReference::extract(reference.as_borrowed())?
        } else {
            GeomReference(RsReference::Point(Vector3D::default()))
        };

        Ok(Self(
            RsGeometry::sphere(converted_ref.0, radius)
                .map_err(|e| ConfigError::new_err(e.to_string()))?
                .with_invert(invert),
        ))
    }
}

#[derive(Clone)]
pub struct GeomReference(RsReference);

impl<'source> FromPyObject<'source, '_> for GeomReference {
    type Error = PyErr;

    fn extract(obj: Borrowed<'source, '_, PyAny>) -> PyResult<Self> {
        // try to extract as Vector3D
        if let Ok(pos) = obj.extract::<[f32; 3]>() {
            return Ok(Self(RsReference::Point(Vector3D::new(
                pos[0], pos[1], pos[2],
            ))));
        }

        // try to extract as a string
        if let Ok(s) = obj.extract::<String>() {
            let s_lower = s.to_lowercase();
            if &s_lower == "center" {
                return Ok(Self(RsReference::Center));
            }

            return Ok(Self(RsReference::Selection(s)));
        }

        Err(ConfigError::new_err(
            "invalid type for GeomReference constructor: expected a list or string",
        ))
    }
}

impl From<[f32; 3]> for GeomReference {
    fn from(value: [f32; 3]) -> Self {
        Self(RsReference::Point(Vector3D::new(
            value[0], value[1], value[2],
        )))
    }
}
