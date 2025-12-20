// Released under MIT License.
// Copyright (c) 2024-2025 Ladislav Bartos

use gorder_core::input::DynamicNormal as RsDynamic;
use gorder_core::input::MembraneNormal as RsNormal;
use gorder_core::prelude::Vector3D;
use hashbrown::HashMap;
use numpy::ndarray;
use numpy::ndarray::ArrayView3;
use numpy::PyArray3;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyclass;
use pyo3_stub_gen::derive::gen_stub_pymethods;

use crate::string2axis;
use crate::Collect;
use crate::ConfigError;

/// Structure describing the direction of the membrane normal
/// or properties necessary for its calculation.
#[derive(Clone)]
pub struct MembraneNormal(pub(crate) RsNormal);

impl<'source> FromPyObject<'source, '_> for MembraneNormal {
    type Error = PyErr;

    fn extract(obj: Borrowed<'source, '_, PyAny>) -> PyResult<Self> {
        // try to extract as DynamicNormal
        if let Ok(dyn_norm) = obj.extract::<DynamicNormal>() {
            return Ok(MembraneNormal(RsNormal::Dynamic(dyn_norm.0)));
        }
        // try to extract as a string
        if let Ok(s) = obj.extract::<String>() {
            let s_lower = s.to_lowercase();
            if s_lower == "x" || s_lower == "y" || s_lower == "z" {
                return Ok(MembraneNormal(RsNormal::Static(string2axis(&s_lower)?)));
            } else {
                return Ok(MembraneNormal(RsNormal::FromFile(s)));
            }
        }
        // try to extract as a dictionary
        if let Ok(map) = extract_map(&obj) {
            return Ok(MembraneNormal(RsNormal::FromMap(map)));
        }

        Err(ConfigError::new_err(
            "invalid type for MembraneNormal constructor: expected a str, DynamicNormal, or Mapping",
        ))
    }
}

/// Request a dynamic local membrane normal calculation.
///
/// Parameters
/// ----------
/// heads : str
///     Selection query specifying reference atoms representing lipid headgroups
///     (typically phosphorus atoms or phosphate beads). Must be exactly one
///     atom/bead per lipid molecule.
/// radius : float
///     Radius of the sphere used to select nearby lipids for membrane normal
///     estimation in nm. Recommended value is half the membrane thickness.
///     Must be greater than 0. The default value is 2.0 (nm).
/// collect : Optional[Union[bool, str]], default=False
///     Determines whether dynamic membrane normals are saved and exported.
///     By default (`False`), normals are not saved.
///     If `True`, normals are saved internally and accessible via the Python API, but not written to a file.
///     If a string is provided, normals are saved and written to the specified output file.
///
/// Raises
/// ------
/// ConfigError
///     If `radius` is not positive.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.membrane_normal")]
#[derive(Clone)]
pub struct DynamicNormal(RsDynamic);

#[gen_stub_pymethods]
#[pymethods]
impl DynamicNormal {
    #[new]
    #[pyo3(signature = (heads, radius = 2.0, collect = None))]
    pub fn new<'a>(
        heads: &str,
        radius: f32,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[builtins.bool, builtins.str]]", imports=("typing")
        ))]
        collect: Option<Bound<'a, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self(add_collect(
            RsDynamic::new(heads, radius).map_err(|e| ConfigError::new_err(e.to_string()))?,
            collect,
        )?))
    }
}

/// Attempt to add request for data collection to dynamic membrane normals.
fn add_collect<'a>(normals: RsDynamic, collect: Option<Bound<'a, PyAny>>) -> PyResult<RsDynamic> {
    if let Some(collect) = collect {
        return Ok(normals.with_collect(Collect::extract(collect.as_borrowed())?.0));
    }

    Ok(normals)
}

/// Converts a three-dimensional numpy array into a Vec<Vec<Vector3D>>.
/// The numpy array must have shape [outer, inner, 3].
fn extract_nested_vector(py_obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Vector3D>>> {
    // try to downcast the input to a PyArray3 of f32
    let array = py_obj.cast::<PyArray3<f32>>().map_err(|_| {
        ConfigError::new_err("expected a 3D numpy array for dynamic membrane normals")
    })?;
    let view: ArrayView3<f32> = unsafe { array.as_array() };

    let shape = view.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(ConfigError::new_err(
            "expected a 3D numpy array with shape (n_frames, n_molecules, 3)",
        ));
    }
    let outer = shape[0];
    let inner = shape[1];
    let mut result = Vec::with_capacity(outer);
    for i in 0..outer {
        let mut inner_vec = Vec::with_capacity(inner);
        for j in 0..inner {
            // slice the row [i, j, :]
            let slice = view.slice(ndarray::s![i, j, ..]);
            // create a Vector3D from the slice
            let vec3 = Vector3D::new(slice[0], slice[1], slice[2]);
            inner_vec.push(vec3);
        }
        result.push(inner_vec);
    }
    Ok(result)
}

/// Converts a Python dictionary whose keys are strings and values are 3D numpy arrays
/// into a hashbrown::HashMap<String, Vec<Vec<Vector3D>>>.
fn extract_map(py_obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, Vec<Vec<Vector3D>>>> {
    let dict = py_obj.cast::<PyDict>().map_err(|_| {
        ConfigError::new_err(
            "expected a dictionary using molecule types as keys and 3D numpy arrays with shape (n_frames, n_molecules, 3) as values",
        )
    })?;
    let mut map = HashMap::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let nested_vec = extract_nested_vector(&value)?;
        map.insert(key_str, nested_vec);
    }
    Ok(map)
}
