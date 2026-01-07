// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

use gorder_core::input::Collect as RsCollect;
use gorder_core::input::{Axis, Plane};
use gorder_core::prelude::AtomType as RsAtomType;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::process;
use std::sync::Once;

create_exception!(
    exceptions,
    AnalysisError,
    PyException,
    "Exception that can be raised when analyzing the trajectory."
);
create_exception!(
    exceptions,
    WriteError,
    PyException,
    "Exception that can be raised when writing the results into output files."
);
create_exception!(
    exceptions,
    ConfigError,
    PyException,
    "Exception that can be raised when constructing the config Analysis class."
);
create_exception!(
    exceptions,
    APIError,
    PyException,
    "Exception that can be raised when accessing the results programmatically."
);

#[macro_export]
macro_rules! apply_if_some {
    ($builder:expr, $( $var:ident => $method:ident ),+ $(,)?) => {
        $(
            if let Some(x) = $var { $builder.$method(x); }
        )+
    };
}

#[macro_export]
macro_rules! apply_inner_if_some {
    ($builder:expr, $( $var:ident => $method:ident ),+ $(,)?) => {
        $(
            if let Some(x) = $var { $builder.$method(x.0); }
        )+
    };
}

mod analysis;
mod estimate_error;
mod geometry;
mod leaflets;
mod normal;
mod ordermap;
mod results;

/// Convert string to axis or return a python error if the conversion fails.
fn string2axis(string: impl AsRef<str>) -> PyResult<Axis> {
    match string.as_ref().to_lowercase().as_str() {
        "x" => Ok(Axis::X),
        "y" => Ok(Axis::Y),
        "z" => Ok(Axis::Z),
        _ => Err(ConfigError::new_err(format!(
            "'{}' is not a valid axis",
            string.as_ref()
        ))),
    }
}

/// Convert string to plane or return a python error if the conversion failes.
fn string2plane(string: impl AsRef<str>) -> PyResult<Plane> {
    match string.as_ref().to_lowercase().as_str() {
        "xy" => Ok(Plane::XY),
        "xz" => Ok(Plane::XZ),
        "yz" => Ok(Plane::YZ),
        _ => Err(ConfigError::new_err(format!(
            "'{}' is not a valid plane",
            string.as_ref()
        ))),
    }
}

/// Represents an atom type within a molecule type.
///
/// Provides access to the atom's name, its relative position within the molecule type,
/// and the residue it belongs to.
#[gen_stub_pyclass]
#[pyclass]
pub struct AtomType(pub(crate) RsAtomType);

#[gen_stub_pymethods]
#[pymethods]
impl AtomType {
    /// Get the name of the atom type.
    ///
    /// Returns
    /// -------
    /// str
    ///     Name of the atom type.
    pub fn atom_name(&self) -> String {
        self.0.atom_name().clone()
    }

    /// Get the relative index of the atom type in its molecule type.
    ///
    /// Returns
    /// -------
    /// int
    ///     Zero-based index of the atom within its molecule type.
    pub fn relative_index(&self) -> usize {
        self.0.relative_index()
    }

    /// Get the name of the residue this atom belongs to.
    ///
    /// Returns
    /// -------
    /// str
    ///     Name of the residue containing this atom.
    pub fn residue_name(&self) -> String {
        self.0.residue_name().clone()
    }
}

/// Handles specifying how data should be collected.
#[derive(Clone)]
pub struct Collect(RsCollect);

impl<'source> FromPyObject<'source, '_> for Collect {
    type Error = PyErr;

    fn extract(obj: Borrowed<'source, '_, PyAny>) -> PyResult<Self> {
        // try to extract as boolean
        if let Ok(boolean) = obj.extract::<bool>() {
            return Ok(Collect(RsCollect::Boolean(boolean)));
        }
        // try to extract as a string
        if let Ok(s) = obj.extract::<String>() {
            return Ok(Collect(RsCollect::File(s)));
        }

        Err(ConfigError::new_err(
            "invalid type for Collect constructor: expected a bool or str",
        ))
    }
}

static INIT: Once = Once::new();

#[pymodule]
fn gorder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    INIT.call_once(|| {
        // initialize logging
        colog::init();

        // set up Ctrl+C handler
        ctrlc::set_handler(|| {
            process::exit(1);
        })
        .unwrap_or_else(|e| panic!("FATAL GORDER ERROR | python::analysis::Analysis | Could not set up the CTRL-C handler: {}.", e));
    });

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // global classes
    m.add_class::<analysis::Analysis>()?;
    m.add_class::<leaflets::Frequency>()?;
    m.add_class::<AtomType>()?;

    // module: analysis_types
    let analysis_types = PyModule::new(m.py(), "analysis_types")?;
    analysis_types.add_class::<analysis::AAOrder>()?;
    analysis_types.add_class::<analysis::CGOrder>()?;
    analysis_types.add_class::<analysis::UAOrder>()?;
    m.add_submodule(&analysis_types)?;

    // module: membrane_normal
    let membrane_normal = PyModule::new(m.py(), "membrane_normal")?;
    membrane_normal.add_class::<normal::DynamicNormal>()?;
    m.add_submodule(&membrane_normal)?;

    // module: leaflets
    let leaflets = PyModule::new(m.py(), "leaflets")?;
    leaflets.add_class::<leaflets::GlobalClassification>()?;
    leaflets.add_class::<leaflets::LocalClassification>()?;
    leaflets.add_class::<leaflets::IndividualClassification>()?;
    leaflets.add_class::<leaflets::ClusteringClassification>()?;
    leaflets.add_class::<leaflets::SphericalClusteringClassification>()?;
    leaflets.add_class::<leaflets::ManualClassification>()?;
    leaflets.add_class::<leaflets::NdxClassification>()?;
    m.add_submodule(&leaflets)?;

    // module: estimate_error
    let estimate_error = PyModule::new(m.py(), "estimate_error")?;
    estimate_error.add_class::<estimate_error::EstimateError>()?;
    m.add_submodule(&estimate_error)?;

    // module: geometry
    let geometry = PyModule::new(m.py(), "geometry")?;
    geometry.add_class::<geometry::Cuboid>()?;
    geometry.add_class::<geometry::Cylinder>()?;
    geometry.add_class::<geometry::Sphere>()?;
    m.add_submodule(&geometry)?;

    // module: ordermap
    let ordermap = PyModule::new(m.py(), "ordermap")?;
    ordermap.add_class::<ordermap::OrderMap>()?;
    m.add_submodule(&ordermap)?;

    // module: results
    let results = PyModule::new(m.py(), "results")?;
    results.add_class::<results::AnalysisResults>()?;
    results.add_class::<results::MoleculeResults>()?;
    results.add_class::<results::AtomResults>()?;
    results.add_class::<results::BondResults>()?;
    results.add_class::<results::OrderCollection>()?;
    results.add_class::<results::OrderMapsCollection>()?;
    results.add_class::<results::Order>()?;
    results.add_class::<results::Map>()?;
    results.add_class::<results::Convergence>()?;
    results.add_class::<results::LeafletsData>()?;
    m.add_submodule(&results)?;

    // module: exceptions
    let exceptions = PyModule::new(m.py(), "exceptions")?;
    exceptions.add("ConfigError", m.py().get_type::<ConfigError>())?;
    exceptions.add("AnalysisError", m.py().get_type::<AnalysisError>())?;
    exceptions.add("WriteError", m.py().get_type::<WriteError>())?;
    exceptions.add("APIError", m.py().get_type::<APIError>())?;
    m.add_submodule(&exceptions)?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);
