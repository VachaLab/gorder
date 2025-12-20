// Released under MIT License.
// Copyright (c) 2024-2025 Ladislav Bartos

use std::sync::Arc;

use gorder_core::input::analysis::TrajectoryInput as RsTrajectoryInput;
use gorder_core::input::AnalysisType as RsAnalysisType;
use gorder_core::prelude::Analysis as RsAnalysis;
use gorder_core::prelude::AnalysisBuilder as RsAnalysisBuilder;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;
use pyo3_stub_gen::derive::gen_stub_pymethods;

use crate::estimate_error::EstimateError;
use crate::geometry::Geometry;
use crate::leaflets::LeafletClassification;
use crate::normal::MembraneNormal;
use crate::ordermap::OrderMap;
use crate::results::AnalysisResults;
use crate::{AnalysisError, ConfigError};

/// Request the calculation of atomistic order parameters.
///
/// Parameters
/// ----------
/// heavy_atoms : str
///     Selection query specifying the heavy atoms to be used in the analysis (typically carbon atoms in lipid tails).
/// hydrogens : str
///     Selection query specifiying the hydrogen atoms to be used in the analysis (only those bonded to heavy atoms will be considered).
///
/// Notes
/// ------
/// - Atoms should be specified using the `groan selection language <https://ladme.github.io/gsl-guide>`_.
/// - Order parameters are calculated for bonds between `heavy_atoms` and `hydrogens`. These bonds are detected automatically.
/// - The order parameters for heavy atoms are determined by averaging the order parameters of the corresponding bonds.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.analysis_types")]
#[derive(Clone)]
pub struct AAOrder(RsAnalysisType);

#[gen_stub_pymethods]
#[pymethods]
impl AAOrder {
    #[new]
    pub fn new(heavy_atoms: &str, hydrogens: &str) -> Self {
        Self(RsAnalysisType::aaorder(heavy_atoms, hydrogens))
    }
}

/// Request the calculation of coarse-grained order parameters.
///
/// Parameters
/// ----------
/// beads : str
///     Selection query specifying the coarse-grained beads to be used in the analysis.
///
/// Notes
/// -----
/// - Beads should be specified using the `groan selection language <https://ladme.github.io/gsl-guide>`_.
/// - Order parameters are calculated for bonds between individual `beads`. These bonds are detected automatically.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.analysis_types")]
#[derive(Clone)]
pub struct CGOrder(RsAnalysisType);

#[gen_stub_pymethods]
#[pymethods]
impl CGOrder {
    #[new]
    pub fn new(beads: &str) -> Self {
        Self(RsAnalysisType::cgorder(beads))
    }
}

/// Request the calculation of united-atom order parameters.
///
/// Parameters
/// ----------
/// saturated : Optional[str], default=None
///     Selection query specifying saturated carbons which order parameters should be calculated.
/// unsaturated : Optional[str], default=None
///     Selection query specifying unsaturated carbons which order parameters should be calculated.
/// ignore : Optional[str], default=None
///     Selection query specifying atoms to be completely ignored when performing the analysis.
///
/// Notes
/// -----
/// - To specify atoms, use the `groan selection language <https://ladme.github.io/gsl-guide>`_.
/// - The positions of hydrogens will be predicted for the respective carbons and order parameters will be calculated
///   for the individual carbon-hydrogen bonds.
/// - Only carbons are supported. If you need to predict hydrogens for other elements, look elsewhere!
/// - When calculating the number of bonds, `gorder` does not distinguish between single and double bonds.
///   This means it will attempt to add one hydrogen to a carboxyl atom if specified.
///   A simple solution to this issue is to exclude such atoms from the analysis.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.analysis_types")]
#[derive(Clone)]
pub struct UAOrder(RsAnalysisType);

#[gen_stub_pymethods]
#[pymethods]
impl UAOrder {
    #[new]
    #[pyo3(signature = (
        saturated = None,
        unsaturated = None,
        ignore = None))]
    pub fn new(saturated: Option<&str>, unsaturated: Option<&str>, ignore: Option<&str>) -> Self {
        Self(RsAnalysisType::uaorder(saturated, unsaturated, ignore))
    }
}

#[derive(Clone)]
pub struct AnalysisType(RsAnalysisType);

impl<'source> FromPyObject<'source, '_> for AnalysisType {
    type Error = PyErr;

    fn extract(obj: Borrowed<'source, '_, PyAny>) -> PyResult<Self> {
        if let Ok(analysis_type) = obj.extract::<AAOrder>() {
            return Ok(AnalysisType(analysis_type.0));
        }

        if let Ok(analysis_type) = obj.extract::<CGOrder>() {
            return Ok(AnalysisType(analysis_type.0));
        }

        if let Ok(analysis_type) = obj.extract::<UAOrder>() {
            return Ok(AnalysisType(analysis_type.0));
        }

        Err(ConfigError::new_err(
            "expected an instance of AAOrder, CGOrder, or UAOrder as AnalysisType",
        ))
    }
}

#[derive(Clone)]
pub struct TrajectoryInput(RsTrajectoryInput);

impl<'source> FromPyObject<'source, '_> for TrajectoryInput {
    type Error = PyErr;

    fn extract(obj: Borrowed<'source, '_, PyAny>) -> PyResult<Self> {
        if let Ok(string) = obj.extract::<String>() {
            return Ok(TrajectoryInput(string.into()));
        }

        if let Ok(list) = obj.extract::<Vec<String>>() {
            return Ok(TrajectoryInput(list.into()));
        }

        Err(ConfigError::new_err(
            "expected a string or a list of strings",
        ))
    }
}

/// Class describing all the parameters of the analysis.
///
/// Parameters
/// ----------
/// structure : str
///     Path to a TPR (recommended), PDB, GRO, or PQR file containing the structure and topology of the system.
/// trajectory : Union[str, Sequence[str]]
///     Path to an XTC (recommended), TRR, or GRO trajectory file to be analyzed.
///     You can provide multiple XTC or TRR trajectories and these will be seamlessly concatenated.
/// analysis_type : Union[AAOrder, CGOrder, UAOrder]
///     Type of analysis to perform (e.g., AAOrder or CGOrder).
/// bonds : Optional[str], default=None
///     Path to a file containing bonding information. If specified, this overrides bonds from the structure file.
/// index : Optional[str], default=None
///     Path to an NDX file specifying groups in the system.
/// output_yaml : Optional[str], default=None
///     Path to an output YAML file containing the full analysis results.
/// output_tab : Optional[str], default=None
///     Path to an output TABLE file with human-readable results.
/// output_xvg : Optional[str], default=None
///     Filename pattern for output XVG files storing results. Each molecule type gets a separate file.
/// output_csv : Optional[str], default=None
///     Path to an output CSV file containing analysis results.
/// membrane_normal : Optional[Union[str, Mapping[str, ndarray[float32]], DynamicNormal]], default=None
///     Direction of the membrane normal.
///     Allowed values are `x`, `y`, `z`, path to file, dictionary specifying manual membrane normals or an instance of `DynamicNormal`.
///     Defaults to the z-axis if not specified.
/// begin : Optional[float], default=None
///     Starting time of the trajectory analysis in picoseconds (ps). Defaults to the beginning of the trajectory.
/// end : Optional[float], default=None
///     Ending time of the trajectory analysis in picoseconds (ps). Defaults to the end of the trajectory.
/// step : Optional[int], default=None
///     Step size for analysis. Every Nth frame will be analyzed. Defaults to 1.
/// min_samples : Optional[int], default=None
///     Minimum number of samples required for each heavy atom or bond type to compute its order parameter. Defaults to 1.
/// n_threads : Optional[int], default=None
///     Number of threads to use for analysis. Defaults to 1.
/// leaflets : Optional[Union[GlobalClassification, LocalClassification, IndividualClassification, ClusteringClassification, SphericalClusteringClassification, ManualClassification, NdxClassification]], default=None
///     Defines how lipids are assigned to membrane leaflets. If provided, order parameters are calculated per leaflet.
/// ordermap : Optional[OrderMap], default=None
///     Specifies parameters for ordermap calculations. If not provided, ordermaps are not generated.
/// estimate_error : Optional[EstimateError], default=None
///     Enables error estimation for each bond if specified.
/// geometry : Optional[Union[Cuboid, Cylinder, Sphere]], default=None
///     Defines a specific region in the simulation box for order parameter calculations. Defaults to the entire system.
/// handle_pbc : Optional[bool], default=True
///     If False, ignores periodic boundary conditions (PBC). Defaults to True.
/// silent : Optional[bool], default=False
///     If True, suppresses standard output messages during analysis.
/// overwrite : Optional[bool], default=False
///     If True, overwrites existing output files and directories without backups.
#[gen_stub_pyclass]
#[pyclass(module = "gorder")]
pub struct Analysis(RsAnalysis);

#[gen_stub_pymethods]
#[pymethods]
impl Analysis {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        structure,
        trajectory,
        analysis_type,
        bonds=None,
        index=None,
        output_yaml=None,
        output_tab=None,
        output_xvg=None,
        output_csv=None,
        membrane_normal=None,
        begin=None,
        end=None,
        step=None,
        min_samples=None,
        n_threads=None,
        leaflets=None,
        ordermap=None,
        estimate_error=None,
        geometry=None,
        handle_pbc=None,
        silent=None,
        overwrite=None))]
    pub fn new<'a>(
        structure: &str,
        #[gen_stub(override_type(
            type_repr = "typing.Union[builtins.str, typing.Sequence[str]]", imports=("typing")
        ))]
        trajectory: Bound<'a, PyAny>,
        #[gen_stub(override_type(
            type_repr = "typing.Union[gorder.analysis_types.AAOrder, gorder.analysis_types.CGOrder, gorder.analysis_types.UAOrder]"
        ))]
        analysis_type: Bound<'a, PyAny>,
        bonds: Option<&str>,
        index: Option<&str>,
        output_yaml: Option<&str>,
        output_tab: Option<&str>,
        output_xvg: Option<&str>,
        output_csv: Option<&str>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[builtins.str, typing.Mapping[builtins.str, numpy.typing.NDArray[numpy.float32]], gorder.membrane_normal.DynamicNormal]]", imports=("typing", "numpy")
        ))]
        membrane_normal: Option<Bound<'a, PyAny>>,
        begin: Option<f32>,
        end: Option<f32>,
        step: Option<usize>,
        min_samples: Option<usize>,
        n_threads: Option<usize>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[gorder.leaflets.GlobalClassification, gorder.leaflets.LocalClassification, gorder.leaflets.IndividualClassification, gorder.leaflets.ClusteringClassification, gorder.leaflets.SphericalClusteringClassification, gorder.leaflets.ManualClassification, gorder.leaflets.NdxClassification]]", imports=("typing")
        ))]
        leaflets: Option<Bound<'a, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[gorder.ordermap.OrderMap]"))]
        ordermap: Option<OrderMap>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[gorder.estimate_error.EstimateError]"
        ))]
        estimate_error: Option<EstimateError>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[gorder.geometry.Cuboid, gorder.geometry.Cylinder, gorder.geometry.Sphere]]"
        ))]
        geometry: Option<Bound<'a, PyAny>>,
        handle_pbc: Option<bool>,
        silent: Option<bool>,
        overwrite: Option<bool>,
    ) -> PyResult<Self> {
        // convert to Rust
        let trajectory = TrajectoryInput::extract(trajectory.as_borrowed())?;
        let analysis_type = AnalysisType::extract(analysis_type.as_borrowed())?;
        let membrane_normal = membrane_normal
            .map(|normal| MembraneNormal::extract(normal.as_borrowed()))
            .transpose()?;
        let leaflets = leaflets
            .map(|method| LeafletClassification::extract(method.as_borrowed()))
            .transpose()?;
        let geometry = geometry
            .map(|shape| Geometry::extract(shape.as_borrowed()))
            .transpose()?;

        let mut builder: RsAnalysisBuilder = RsAnalysis::builder();
        builder
            .structure(structure)
            .trajectory(trajectory.0)
            .analysis_type(analysis_type.0);

        apply_if_some!(
            builder,
            bonds           => bonds,
            index           => index,
            output_yaml     => output_yaml,
            output_tab      => output_tab,
            output_xvg      => output_xvg,
            output_csv      => output_csv,
            begin           => begin,
            end             => end,
            step            => step,
            min_samples     => min_samples,
            n_threads       => n_threads,
            handle_pbc      => handle_pbc,
        );

        apply_inner_if_some!(
            builder,
            membrane_normal => membrane_normal,
            leaflets        => leaflets,
            estimate_error  => estimate_error,
            geometry        => geometry,
            ordermap        => map,
        );

        if let Some(true) = silent {
            builder.silent();
        }

        if let Some(true) = overwrite {
            builder.overwrite();
        }

        let inner = builder
            .build()
            .map_err(|e| ConfigError::new_err(e.to_string()))?;
        Ok(Analysis(inner))
    }

    /// Run the analysis.
    ///
    /// Executes the configured analysis on the input data and returns the results.
    ///
    /// Returns
    /// -------
    /// AnalysisResults
    ///     Results of the analysis.
    ///
    /// Raises
    /// ------
    /// AnalysisError
    ///     If the analysis fails.
    #[gen_stub(override_return_type(type_repr = "gorder.results.AnalysisResults"))]
    pub fn run(&mut self) -> PyResult<AnalysisResults> {
        if self.0.silent() {
            log::set_max_level(log::LevelFilter::Error);
        }

        match self.0.clone().run() {
            Err(e) => Err(AnalysisError::new_err(e.to_string())),
            Ok(x) => Ok(AnalysisResults(Arc::new(x))),
        }
    }

    /// Read analysis options from a YAML configuration file.
    ///
    /// Parameters
    /// ----------
    /// file : str
    ///     Path to the YAML configuration file.
    ///
    /// Returns
    /// -------
    /// Analysis
    ///     Analysis instance initialized from the file.
    ///
    /// Raises
    /// ------
    /// ConfigError
    ///     If the file cannot be read or parsed.
    #[staticmethod]
    pub fn from_file(file: String) -> PyResult<Self> {
        let analysis =
            RsAnalysis::from_file(file).map_err(|e| ConfigError::new_err(e.to_string()))?;
        Ok(Analysis(analysis))
    }
}
