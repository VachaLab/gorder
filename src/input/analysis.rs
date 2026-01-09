// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Contains the implementation of the main `Analysis` structure and its methods.

use std::fmt;
use std::fs::read_to_string;
use std::path::Path;

use derive_builder::Builder;
use getset::{CopyGetters, Getters};
use getset::{MutGetters, Setters};
use groan_rs::files::FileType;
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::errors::ConfigError;

use super::membrane_normal::MembraneNormal;
use super::OrderMap;
use super::{Axis, EstimateError};
use super::{Geometry, LeafletClassification};

/// Type of analysis to perform.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum AnalysisType {
    AAOrder {
        heavy_atoms: String,
        hydrogens: String,
    },
    CGOrder {
        #[serde(alias = "atoms")]
        beads: String,
    },
    UAOrder {
        saturated: Option<String>,
        unsaturated: Option<String>,
        ignore: Option<String>,
    },
}

impl AnalysisType {
    /// Request calculation of atomistic order parameters.
    ///
    /// ## Arguments
    /// - `heavy_atoms` - specification of heavy atoms to be used in the analysis (usually carbon atoms of lipid tails)
    /// - `hydrogens` - specification of hydrogens to be used in the analysis (only atoms connected to heavy atoms will be used)
    ///
    /// ## Notes
    /// - To specify the atoms, use the [groan selection language](https://ladme.github.io/gsl-guide).
    /// - Order parameters will be calculated for bonds between the `heavy_atoms` and `hydrogens`. These bonds are detected automatically.
    /// - Order parameters for heavy atoms are then calculated by averaging the order parameters of the relevant bonds.
    pub fn aaorder(heavy_atoms: &str, hydrogens: &str) -> Self {
        Self::AAOrder {
            heavy_atoms: heavy_atoms.to_owned(),
            hydrogens: hydrogens.to_owned(),
        }
    }

    /// Request calculation of coarse-grained order parameters.
    ///
    /// ## Arguments
    /// - `beads` - specification of coarse-grained beads
    ///
    /// ## Notes
    /// - To specify the beads, use the [groan selection language](https://ladme.github.io/gsl-guide).
    /// - Order parameters will be calculated for bonds between the individual `beads`. These bonds are detected automatically.
    pub fn cgorder(beads: &str) -> Self {
        Self::CGOrder {
            beads: beads.to_owned(),
        }
    }

    /// Request calculation of united-atom order parameters.
    ///
    /// ## Arguments
    /// - `saturated` - specification of saturated carbons, i.e., carbons with (4 - N_bonds) hydrogens bonded to them
    /// - `unsaturated` - specification of unsaturated carbons, i.e., carbons with one hydrogen bonded to them
    /// - `ignore` - specification of atoms to ignore when calculating the number of bonds to carbons
    ///
    /// ## Notes
    /// - To specify atoms, use the [groan selection language](https://ladme.github.io/gsl-guide).
    /// - The positions of hydrogens will be predicted for the respective carbons and order parameters will be calculated
    ///   for the individual carbon-hydrogen bonds.
    /// - Only carbons are supported. If you need to predict hydrogens for other elements, look elsewhere!
    /// - When calculating the number of bonds, `gorder` does not distinguish between single and double bonds.
    ///   This means it will attempt to add one hydrogen to a carboxyl atom if specified.
    ///   A simple solution to this issue is to exclude such atoms from the analysis.
    pub fn uaorder(
        saturated: Option<&str>,
        unsaturated: Option<&str>,
        ignore: Option<&str>,
    ) -> Self {
        Self::UAOrder {
            saturated: saturated.map(|x| x.to_owned()),
            unsaturated: unsaturated.map(|x| x.to_owned()),
            ignore: ignore.map(|x| x.to_owned()),
        }
    }

    /// Get the type of analysis as a string.
    pub fn name(&self) -> &str {
        match self {
            Self::AAOrder {
                heavy_atoms: _,
                hydrogens: _,
            } => "atomistic order parameters",
            Self::CGOrder { beads: _ } => "coarse-grained order parameters",
            Self::UAOrder {
                saturated: _,
                unsaturated: _,
                ignore: _,
            } => "united-atom order parameters",
        }
    }
}

/// Structure holding all the information necessary to perform the analysis.
#[derive(
    Debug, Clone, Builder, Getters, CopyGetters, Setters, MutGetters, Deserialize, Serialize,
)]
#[serde(deny_unknown_fields)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct Analysis {
    /// Path to a TPR (recommended), PDB, GRO, or PQR file containing the structure (and topology) of the system.
    #[builder(setter(into))]
    #[getset(get = "pub")]
    structure: String,

    /// Optional path to a file containing information about bonds.
    /// If specified, overrides any bonds in the structure file.
    #[builder(setter(into, strip_option), default)]
    #[getset(get = "pub")]
    bonds: Option<String>,

    /// Path to an XTC (recommended), TRR, or GRO file containing the trajectory to be analyzed.
    /// Multiple XTC or TRR trajectories can be provided and these will be concatenated.
    #[getset(get = "pub")]
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    #[builder(setter(custom))]
    trajectory: Vec<String>,

    /// Optional path to an NDX file containing the groups associated with the system.
    #[builder(setter(into, strip_option), default)]
    #[getset(get = "pub")]
    index: Option<String>,

    /// Optional path to an output YAML file where the full results of the analysis will be saved.
    /// YAML file is the only output file containing the full results of the analysis.
    #[builder(setter(into, strip_option), default)]
    #[getset(get = "pub")]
    #[serde(alias = "output", alias = "output_yml")]
    output_yaml: Option<String>,

    /// Optional path to an output TABLE file where the analysis results will be saved
    /// in a human-readable format.
    #[builder(setter(into, strip_option), default)]
    #[getset(get = "pub")]
    output_tab: Option<String>,

    /// Optional filename pattern for the output XVG files where analysis results will be stored.
    /// A separate XVG file will be generated for each detected molecule type, with the molecule
    /// type name appended to the pattern.
    ///
    /// Example: 'order.xvg' may become 'order_POPC.xvg'.
    #[builder(setter(into, strip_option), default)]
    #[getset(get = "pub")]
    output_xvg: Option<String>,

    /// Optional path to an output CSV file where the analysis results will be stored.
    #[builder(setter(into, strip_option), default)]
    #[getset(get = "pub")]
    output_csv: Option<String>,

    /// Type of analysis to be performed (AAOrder, CGOrder, or UAOrder).
    #[getset(get = "pub")]
    #[serde(alias = "type")]
    analysis_type: AnalysisType,

    /// Direction of the membrane normal. Defaults to 'z axis' if not provided.
    ///
    /// You can either provide `Axis`, `DynamicNormal`, or `MembraneNormal`.
    /// The former two types will automatically get converted into `MembraneNormal`.
    #[builder(setter(into), default = "Axis::Z.into()")]
    #[serde(
        default = "default_membrane_normal",
        deserialize_with = "deserialize_membrane_normal"
    )]
    #[getset(get = "pub")]
    membrane_normal: MembraneNormal,

    /// Starting time of the trajectory analysis in picoseconds (ps).
    /// Defaults to the beginning of the trajectory if not specified.
    #[builder(default = "0.0")]
    #[serde(default = "default_begin", alias = "start")]
    #[getset(get_copy = "pub")]
    begin: f32,

    /// Ending time of the trajectory analysis in picoseconds (ps).
    /// Defaults to the end of the trajectory if not specified.
    #[builder(default = "f32::INFINITY")]
    #[serde(default = "default_end")]
    #[getset(get_copy = "pub")]
    end: f32,

    /// Step size for the analysis. Every Nth frame of the trajectory will be analyzed.
    /// Defaults to 1 if not specified.
    #[builder(default = "1")]
    #[serde(default = "default_one")]
    #[getset(get_copy = "pub")]
    step: usize,

    /// Minimum number of samples required for each heavy atom to calculate its order parameter.
    /// Defaults to 1 if not specified.
    #[builder(default = "1")]
    #[serde(default = "default_one")]
    #[getset(get_copy = "pub")]
    min_samples: usize,

    /// Number of threads to use for the analysis. Defaults to 1 if not specified.
    #[builder(default = "1")]
    #[serde(default = "default_one")]
    #[getset(get_copy = "pub")]
    n_threads: usize,

    /// Specifies how lipids are assigned to membrane leaflets. If provided, order parameters
    /// will be calculated for each leaflet as well as for the entire membrane. If not provided,
    /// only the overall membrane values will be calculated.
    #[builder(setter(strip_option), default)]
    #[getset(get = "pub")]
    leaflets: Option<LeafletClassification>,

    /// Optional specification for the properties of an ordermap to be calculated.
    /// If not provided, no ordermap will be calculated.
    #[builder(setter(strip_option), default)]
    #[serde(alias = "maps", alias = "ordermap", alias = "ordermaps")]
    #[serde(deserialize_with = "deserialize_order_map", default)]
    #[getset(get = "pub", get_mut = "pub(crate)")]
    map: Option<OrderMap>,

    /// Optional specification of calculation error estimation.
    /// If provided, calculation error will be provided for each bond.
    #[builder(setter(strip_option), default)]
    #[serde(deserialize_with = "deserialize_estimate_error", default)]
    #[getset(get = "pub")]
    estimate_error: Option<EstimateError>,

    /// Specifies an optional region within the simulation box for calculating order parameters.
    /// If not provided, the calculations include the entire system.
    /// When specified, only bonds within the defined geometry are considered.
    #[builder(setter(strip_option), default)]
    #[serde(default)]
    #[getset(get = "pub")]
    geometry: Option<Geometry>,

    /// If false, periodic boundary conditions will be ignored. This is useful if you are working
    /// with PBC that are not supported by `gorder`. Note that `gorder` only supports
    /// orthogonal simulation boxes with PBC applied in all three dimensions.
    /// If not specified, defaults to `true`.
    #[builder(default = "true")]
    #[serde(default = "default_true")]
    #[getset(get_copy = "pub")]
    handle_pbc: bool,

    /// If true, suppress all output to the standard output during the analysis.
    #[builder(setter(custom), default = "false")]
    #[serde(default = "default_false")]
    #[getset(get_copy = "pub", set = "pub")]
    silent: bool,

    /// If true, overwrite existing output files and directories without creating backups.
    #[builder(setter(custom), default = "false")]
    #[serde(default = "default_false")]
    #[getset(get_copy = "pub", set = "pub")]
    overwrite: bool,
}

fn default_membrane_normal() -> MembraneNormal {
    Axis::Z.into()
}

fn default_begin() -> f32 {
    0.0
}

fn default_end() -> f32 {
    f32::INFINITY
}

fn default_one() -> usize {
    1
}

fn default_false() -> bool {
    false
}

fn default_true() -> bool {
    true
}

fn validate_step(step: usize) -> Result<(), ConfigError> {
    if step == 0 {
        Err(ConfigError::InvalidStep)
    } else {
        Ok(())
    }
}

fn validate_min_samples(samples: usize) -> Result<(), ConfigError> {
    if samples == 0 {
        Err(ConfigError::InvalidMinSamples)
    } else {
        Ok(())
    }
}

fn validate_n_threads(n_threads: usize) -> Result<(), ConfigError> {
    if n_threads == 0 {
        Err(ConfigError::InvalidNThreads)
    } else {
        Ok(())
    }
}

fn validate_begin_end(begin: f32, end: f32) -> Result<(), ConfigError> {
    if begin > end {
        Err(ConfigError::InvalidBeginEnd)
    } else {
        Ok(())
    }
}

fn validate_structure_format(file: &str) -> Result<(), ConfigError> {
    match FileType::from_name(file) {
        FileType::TPR | FileType::GRO | FileType::PDB | FileType::PQR => Ok(()),
        _ => Err(ConfigError::InvalidStructureFormat(file.to_owned())),
    }
}

fn validate_trajectory_format(files: &[String]) -> Result<(), ConfigError> {
    if files.is_empty() {
        return Err(ConfigError::NoTrajectoryFile);
    }

    let mut last: Option<(FileType, &String)> = None;
    for file in files {
        let file_type = FileType::from_name(file);
        match file_type {
            FileType::XTC | FileType::TRR => (),
            FileType::GRO => {
                // trajectory concatenation is not supported for these formats
                if files.len() > 1 {
                    return Err(ConfigError::TrajCatNotSupported);
                }
            }
            FileType::PDB | FileType::NC | FileType::DCD | FileType::LAMMPSTRJ => {
                return Err(ConfigError::DeprecationError("since 'gorder' v0.7, PDB, NC, DCD, and LAMPPSTR trajectories are not supported; if you want to use them, switch to version 0.6".into()))
            }
            _ => return Err(ConfigError::InvalidTrajectoryFormat(file.to_owned())),
        }

        // check that all the trajectory files have the same format
        if let Some((unwrapped_type, unwrapped_name)) = last {
            if file_type != unwrapped_type {
                return Err(ConfigError::InconsistentTrajectoryFormat(
                    file.clone(),
                    unwrapped_name.clone(),
                ));
            }
        }

        last = Some((file_type, file));
    }

    Ok(())
}

pub fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVecVisitor;

    impl<'de> Visitor<'de> for StringOrVecVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or a sequence of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            match glob::glob(value) {
                Ok(paths) => {
                    let files: Vec<_> = paths
                        .filter_map(Result::ok)
                        .map(|path| path.to_string_lossy().into_owned())
                        .collect();

                    if files.is_empty() {
                        Ok(vec![value.to_owned()])
                    } else {
                        Ok(files)
                    }
                }
                Err(e) => Err(de::Error::custom(e.to_string())),
            }
        }

        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrVecVisitor)
}

fn deserialize_membrane_normal<'de, D>(deserializer: D) -> Result<MembraneNormal, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_yaml::Value::deserialize(deserializer)?;

    if let Ok(axis) = Axis::deserialize(&value) {
        return Ok(axis.into());
    }

    MembraneNormal::deserialize(&value).map_err(D::Error::custom)
}

fn deserialize_order_map<'de, D>(deserializer: D) -> Result<Option<OrderMap>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: serde_yaml::Value = Deserialize::deserialize(deserializer)?;

    match value {
        serde_yaml::Value::String(keyword) if keyword == "default" || keyword == "true" => {
            Ok(Some(OrderMap::default()))
        }
        serde_yaml::Value::Null => Ok(None),
        serde_yaml::Value::Bool(true) => Ok(Some(OrderMap::default())),
        serde_yaml::Value::Bool(false) => Err(serde::de::Error::custom("Invalid value 'false' for 'order_map'. If you do not want to calculate ordermaps, just omit this field.")),
        serde_yaml::Value::Mapping(_) => serde_yaml::from_value(value)
            .map(Some)
            .map_err(serde::de::Error::custom),
        _ => Err(serde::de::Error::custom(
            "Invalid value for 'order_map'. Expected 'default', 'true', 'null', or a valid structure.",
        )),
    }
}

fn deserialize_estimate_error<'de, D>(deserializer: D) -> Result<Option<EstimateError>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: serde_yaml::Value = Deserialize::deserialize(deserializer)?;

    match value {
        serde_yaml::Value::String(keyword) if keyword == "default" => {
            Ok(Some(EstimateError::default()))
        }
        serde_yaml::Value::Null => Ok(None),
        serde_yaml::Value::Bool(true) => Ok(Some(EstimateError::default())),
        serde_yaml::Value::Bool(false) => Err(serde::de::Error::custom("Invalid value 'false' for 'estimate_error'. If you do not want to calculate error, just omit this field.")),
        serde_yaml::Value::Mapping(_) => serde_yaml::from_value(value)
            .map(Some)
            .map_err(serde::de::Error::custom),
        _ => Err(serde::de::Error::custom(
            "Invalid value for 'estimate_error'. Expected 'default', 'true', 'null', or a valid structure.",
        )),
    }
}

impl Analysis {
    /// Start providing the analysis parameters.
    pub fn builder() -> AnalysisBuilder {
        AnalysisBuilder::default()
    }

    /// Read parameters of the analysis from an input YAML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Analysis, ConfigError> {
        let string = read_to_string(&path).map_err(|_| {
            ConfigError::CouldNotOpenConfig(path.as_ref().to_str().unwrap().to_owned())
        })?;
        let analysis: Analysis = serde_yaml::from_str(&string).map_err(|e| {
            ConfigError::CouldNotParseConfig(path.as_ref().to_str().unwrap().to_owned(), e)
        })?;

        analysis.validate()?;
        Ok(analysis)
    }

    /// Check that the Analysis structure is valid. Used after deserialization from config yaml file.
    fn validate(&self) -> Result<(), ConfigError> {
        validate_step(self.step)?;
        validate_min_samples(self.min_samples)?;
        validate_n_threads(self.n_threads)?;
        validate_begin_end(self.begin, self.end)?;
        validate_structure_format(&self.structure)?;
        validate_trajectory_format(&self.trajectory)?;
        self.membrane_normal.validate()?;

        // check the validity of the order map, if present
        if let Some(ref map) = self.map {
            map.validate().map_err(ConfigError::InvalidOrderMap)?;
        }

        // check the validity of error estimation
        if let Some(ref error) = self.estimate_error {
            error
                .validate()
                .map_err(ConfigError::InvalidErrorEstimation)?;
        }

        // check the validity of the geometry selection
        if let Some(ref geometry) = self.geometry {
            geometry.validate().map_err(ConfigError::InvalidGeometry)?;
        }

        Ok(())
    }

    /// Get the heavy atoms specified for the analysis.
    /// If the calculation of coarse-grained or united-atom order parameters is requested, returns None.
    pub fn heavy_atoms(&self) -> Option<&String> {
        match &self.analysis_type {
            AnalysisType::CGOrder { beads: _ } => None,
            AnalysisType::AAOrder {
                heavy_atoms,
                hydrogens: _,
            } => Some(heavy_atoms),
            AnalysisType::UAOrder {
                saturated: _,
                unsaturated: _,
                ignore: _,
            } => None,
        }
    }

    /// Get the hydrogens specified for the analysis.
    /// If the calculation of coarse-grained or united-atom order parameters is requested, returns None.
    pub fn hydrogens(&self) -> Option<&String> {
        match &self.analysis_type {
            AnalysisType::CGOrder { beads: _ } => None,
            AnalysisType::AAOrder {
                heavy_atoms: _,
                hydrogens,
            } => Some(hydrogens),
            AnalysisType::UAOrder {
                saturated: _,
                unsaturated: _,
                ignore: _,
            } => None,
        }
    }

    /// Get the beads specified for the analysis.
    /// If the calculation of atomistic or united-atom order parameters is requested, returns None.
    pub fn beads(&self) -> Option<&String> {
        match &self.analysis_type {
            AnalysisType::CGOrder { beads } => Some(beads),
            AnalysisType::AAOrder {
                heavy_atoms: _,
                hydrogens: _,
            } => None,
            AnalysisType::UAOrder {
                saturated: _,
                unsaturated: _,
                ignore: _,
            } => None,
        }
    }

    /// Get the saturated carbons specified for the analysis.
    /// If the calculation of atomistic or coarse-grained order parameters is requested, returns None.
    /// Also returns None, if no saturated carbons have been specified.
    pub fn saturated(&self) -> Option<&String> {
        match &self.analysis_type {
            AnalysisType::CGOrder { beads: _ } => None,
            AnalysisType::AAOrder {
                heavy_atoms: _,
                hydrogens: _,
            } => None,
            AnalysisType::UAOrder {
                saturated,
                unsaturated: _,
                ignore: _,
            } => saturated.as_ref(),
        }
    }

    /// Get the unsaturated carbons specified for the analysis.
    /// If the calculation of atomistic or coarse-grained order parameters is requested, returns None.
    /// Also returns None, if no unsaturated carbons have been specified.
    pub fn unsaturated(&self) -> Option<&String> {
        match &self.analysis_type {
            AnalysisType::CGOrder { beads: _ } => None,
            AnalysisType::AAOrder {
                heavy_atoms: _,
                hydrogens: _,
            } => None,
            AnalysisType::UAOrder {
                saturated: _,
                unsaturated,
                ignore: _,
            } => unsaturated.as_ref(),
        }
    }

    /// Get the atoms to be ignored during the analysis.
    /// If the calculation of atomistic or coarse-grained order parameters is requested, returns None.
    /// Also returns None, if no atoms to ignore have been specified.
    pub fn ignore(&self) -> Option<&String> {
        match &self.analysis_type {
            AnalysisType::CGOrder { beads: _ } => None,
            AnalysisType::AAOrder {
                heavy_atoms: _,
                hydrogens: _,
            } => None,
            AnalysisType::UAOrder {
                saturated: _,
                unsaturated: _,
                ignore,
            } => ignore.as_ref(),
        }
    }

    /// Alias for [`Analysis::beads`].
    #[inline(always)]
    pub fn atoms(&self) -> Option<&String> {
        self.beads()
    }

    /// Alias for `output_yaml`.
    #[inline(always)]
    pub fn output(&self) -> &Option<String> {
        &self.output_yaml
    }
}

/// Helper enum for providing input trajectory file(s).
#[derive(Debug, Clone)]
pub enum TrajectoryInput {
    Single(String),
    Multiple(Vec<String>),
}

impl TrajectoryInput {
    fn expand<T>(value: T) -> Self
    where
        T: Into<String>,
    {
        let string_pattern = value.into();
        match glob::glob(&string_pattern) {
            Ok(paths) => {
                let files: Vec<_> = paths
                    .filter_map(Result::ok)
                    .map(|path| path.to_string_lossy().into_owned())
                    .collect();

                // if there is no match, just use the original pattern
                if files.is_empty() {
                    Self::Single(string_pattern)
                } else {
                    Self::Multiple(files)
                }
            }
            Err(e) => panic!(
                "FATAL GORDER ERROR | TrajectoryInput::expand | Trajectory glob pattern is invalid. Error: {}. (This may be your fault.)", e)
        }
    }
}

impl From<&str> for TrajectoryInput {
    fn from(value: &str) -> Self {
        Self::expand(value)
    }
}

impl From<String> for TrajectoryInput {
    fn from(value: String) -> Self {
        Self::expand(value)
    }
}

impl From<&String> for TrajectoryInput {
    fn from(value: &String) -> Self {
        Self::expand(value)
    }
}

impl<T> From<Vec<T>> for TrajectoryInput
where
    T: Into<String>,
{
    fn from(value: Vec<T>) -> Self {
        Self::Multiple(value.into_iter().map(|x| x.into()).collect())
    }
}

impl<T> From<&[T]> for TrajectoryInput
where
    T: Into<String> + Clone,
{
    fn from(value: &[T]) -> Self {
        Self::Multiple(value.iter().cloned().map(|x| x.into()).collect())
    }
}

impl AnalysisBuilder {
    /// Path to an XTC (recommended), TRR, GRO, PDB, NC, DCD, or LAMMPSTRJ containing the trajectory to be analyzed.
    /// Multiple XTC or TRR trajectories can be provided and these will be concatenated.
    pub fn trajectory(&mut self, trajectory: impl Into<TrajectoryInput>) -> &mut Self {
        match trajectory.into() {
            TrajectoryInput::Single(x) => self.trajectory = Some(vec![x]),
            TrajectoryInput::Multiple(x) => self.trajectory = Some(x),
        }
        self
    }

    /// Be silent. Print nothing to the standard output during the analysis.
    #[inline(always)]
    pub fn silent(&mut self) -> &mut Self {
        self.silent = Some(true);
        self
    }

    /// Do not make backups. Overwrite all output files and directories.
    #[inline(always)]
    pub fn overwrite(&mut self) -> &mut Self {
        self.overwrite = Some(true);
        self
    }

    /// Alias for `output_yaml`.
    #[inline(always)]
    pub fn output(&mut self, value: &str) -> &mut Self {
        self.output_yaml(value)
    }

    /// Alias for `map`.
    #[inline(always)]
    pub fn maps(&mut self, value: OrderMap) -> &mut Self {
        self.map(value)
    }

    /// Alias for `map`.
    #[inline(always)]
    pub fn ordermap(&mut self, value: OrderMap) -> &mut Self {
        self.map(value)
    }

    /// Alias for `map`.
    #[inline(always)]
    pub fn ordermaps(&mut self, value: OrderMap) -> &mut Self {
        self.map(value)
    }

    /// Validate the process of analysis building.
    fn validate(&self) -> Result<(), String> {
        // check that step, min_samples and n_threads are not zero
        if let Some(step) = self.step {
            validate_step(step).map_err(|e| e.to_string())?;
        }

        if let Some(min_samples) = self.min_samples {
            validate_min_samples(min_samples).map_err(|e| e.to_string())?;
        }

        if let Some(n_threads) = self.n_threads {
            validate_n_threads(n_threads).map_err(|e| e.to_string())?;
        }

        // check that start is not larger than end
        if let (Some(begin), Some(end)) = (self.begin, self.end) {
            validate_begin_end(begin, end).map_err(|e| e.to_string())?;
        }

        // check that structure file has a supported format
        if let Some(structure) = &self.structure {
            validate_structure_format(structure).map_err(|e| e.to_string())?;
        }

        // check that the trajectory file has a supported format
        if let Some(trajectory) = &self.trajectory {
            validate_trajectory_format(trajectory).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests_yaml {
    use approx::assert_relative_eq;

    use super::*;
    use crate::errors::{ErrorEstimationError, GeometryConfigError};
    use crate::input::frequency::Frequency;
    use crate::input::geometry::GeomReference;
    use crate::input::Collect;
    use crate::{
        errors::OrderMapConfigError,
        input::{ordermap::Plane, GridSpan},
    };

    fn compare_as_paths(path1: &str, path2: &str) {
        assert_eq!(Path::new(path1), Path::new(path2));
    }

    #[test]
    fn analysis_yaml_pass_basic() {
        let analysis = Analysis::from_file("tests/files/inputs/basic.yaml").unwrap();

        assert_eq!(analysis.structure(), "tests/files/pcpepg.tpr");
        compare_as_paths(&analysis.trajectory()[0], "tests/files/pcpepg.xtc");
        assert!(analysis.index().is_none());
        assert!(analysis.output().is_none());
        assert!(analysis.output_tab().is_none());
        assert!(analysis.output_xvg().is_none());
        assert!(analysis.output_csv().is_none());
        assert!(matches!(
            analysis.membrane_normal(),
            MembraneNormal::Static(Axis::Z)
        ));
        assert_eq!(
            analysis.heavy_atoms().unwrap(),
            "@membrane and element name carbon"
        );
        assert_eq!(
            analysis.hydrogens().unwrap(),
            "@membrane and element name hydrogen"
        );
        assert!(analysis.atoms().is_none());
        assert_eq!(analysis.begin(), 0.0);
        assert!(analysis.end().is_infinite());
        assert_eq!(analysis.step(), 1);
        assert_eq!(analysis.min_samples(), 1);
        assert_eq!(analysis.n_threads(), 1);
        assert!(analysis.leaflets().is_none());
        assert!(analysis.map().is_none());
        assert!(analysis.geometry().is_none());
        assert!(analysis.handle_pbc());
        assert!(!analysis.silent());
        assert!(!analysis.overwrite());
    }

    #[test]
    fn analysis_yaml_pass_full() {
        let analysis = Analysis::from_file("tests/files/inputs/full.yaml").unwrap();

        assert_eq!(analysis.structure(), "tests/files/cg.tpr");
        compare_as_paths(&analysis.trajectory()[0], "tests/files/cg.xtc");
        assert_eq!(analysis.index().as_ref().unwrap(), "index.ndx");
        assert_eq!(analysis.output().as_ref().unwrap(), "order.yaml");
        assert_eq!(analysis.output_tab().as_ref().unwrap(), "order.dat");
        assert_eq!(analysis.output_xvg().as_ref().unwrap(), "order.xvg");
        assert_eq!(analysis.output_csv().as_ref().unwrap(), "order.csv");
        assert!(matches!(
            analysis.membrane_normal(),
            MembraneNormal::Static(Axis::X)
        ));
        assert!(analysis.heavy_atoms().is_none());
        assert!(analysis.hydrogens().is_none());
        assert_eq!(analysis.atoms().unwrap(), "@membrane");
        assert_eq!(analysis.begin(), 100.0);
        assert_eq!(analysis.end(), 10_000.0);
        assert_eq!(analysis.step(), 5);
        assert_eq!(analysis.min_samples(), 10);
        assert_eq!(analysis.n_threads(), 4);
        let leaflets = analysis.leaflets().as_ref().unwrap();
        match leaflets {
            LeafletClassification::Global(x) => {
                assert_eq!(x.heads(), "name P");
                assert_eq!(x.membrane(), "@membrane");
                assert!(matches!(x.frequency(), Frequency::Once));
                assert_eq!(x.collect(), &Collect::File(String::from("leaflets.yaml")));
            }
            _ => panic!("Incorrect leaflet classification type returned."),
        }

        let map = analysis.map().as_ref().unwrap();

        assert_eq!(map.output_directory().as_ref().unwrap(), "ordermaps");
        matches!(
            map.dim_x(),
            GridSpan::Manual {
                start: 0.5,
                end: 10.5
            }
        );
        matches!(map.dim_y(), GridSpan::Auto);
        assert_eq!(map.min_samples(), 10);
        assert_eq!(map.bin_size_x(), 0.05);
        assert_eq!(map.bin_size_y(), 0.02);
        assert_eq!(map.plane().unwrap(), Plane::XY);

        let ee = analysis.estimate_error().as_ref().unwrap();
        assert_eq!(ee.n_blocks(), 10);
        assert_eq!(ee.output_convergence().unwrap(), "convergence.xvg");

        match analysis.geometry().as_ref().unwrap() {
            Geometry::Cylinder(c) => {
                match c.reference() {
                    GeomReference::Selection(x) => assert_eq!(x, "@protein and name BB"),
                    _ => panic!("Invalid geometric reference."),
                }
                assert_relative_eq!(c.radius(), 3.5);
                assert_relative_eq!(c.span()[0], 2.3);
                assert_relative_eq!(c.span()[1], 5.1);
                assert_eq!(c.orientation(), Axis::Z);
            }
            _ => panic!("Incorrect geometry type returned."),
        }

        assert!(!analysis.handle_pbc());
        assert!(analysis.overwrite());
        assert!(analysis.silent());
    }

    #[test]
    fn analysis_yaml_pass_default_ordermap() {
        let analysis = Analysis::from_file("tests/files/inputs/default_ordermap.yaml").unwrap();

        let ordermap = analysis.map.unwrap();
        assert!(ordermap.output_directory().is_none());
        assert_relative_eq!(ordermap.bin_size_x(), 0.1);
        assert_relative_eq!(ordermap.bin_size_y(), 0.1);
        assert_eq!(ordermap.min_samples(), 1);
        matches!(ordermap.dim_x(), GridSpan::Auto);
        matches!(ordermap.dim_y(), GridSpan::Auto);
    }

    #[test]
    fn analysis_yaml_pass_true_ordermap() {
        let analysis = Analysis::from_file("tests/files/inputs/true_ordermap.yaml").unwrap();

        let ordermap = analysis.map.unwrap();
        assert!(ordermap.output_directory().is_none());
        assert_relative_eq!(ordermap.bin_size_x(), 0.1);
        assert_relative_eq!(ordermap.bin_size_y(), 0.1);
        assert_eq!(ordermap.min_samples(), 1);
        matches!(ordermap.dim_x(), GridSpan::Auto);
        matches!(ordermap.dim_y(), GridSpan::Auto);
    }

    #[test]
    fn analysis_yaml_pass_default_estimate_error() {
        let analysis =
            Analysis::from_file("tests/files/inputs/default_estimate_error.yaml").unwrap();

        let ee = analysis.estimate_error.unwrap();
        assert_eq!(ee.n_blocks(), 5);
        assert!(ee.output_convergence().is_none());
    }

    #[test]
    fn analysis_yaml_pass_true_estimate_error() {
        let analysis = Analysis::from_file("tests/files/inputs/true_estimate_error.yaml").unwrap();

        let ee = analysis.estimate_error.unwrap();
        assert_eq!(ee.n_blocks(), 5);
        assert!(ee.output_convergence().is_none());
    }

    #[test]
    fn analysis_yaml_pass_dynamic_membrane_normal() {
        let analysis =
            Analysis::from_file("tests/files/inputs/dynamic_membrane_normal.yaml").unwrap();

        match analysis.membrane_normal() {
            MembraneNormal::Dynamic(dynamic) => {
                assert_eq!(dynamic.heads(), "name PO4");
                assert_relative_eq!(dynamic.radius(), 2.5);
            }
            _ => {
                panic!("Invalid membrane normal returned.")
            }
        }
    }

    #[test]
    fn analysis_yaml_pass_membrane_normal_from_file() {
        let analysis =
            Analysis::from_file("tests/files/inputs/membrane_normal_from_file.yaml").unwrap();

        match analysis.membrane_normal() {
            MembraneNormal::FromFile(file) => {
                assert_eq!(file, "normals.yaml");
            }
            _ => panic!("Invalid membrane normal returned."),
        }
    }

    #[test]
    fn analysis_yaml_pass_membrane_normal_inline() {
        let analysis =
            Analysis::from_file("tests/files/inputs/membrane_normal_inline.yaml").unwrap();

        match analysis.membrane_normal() {
            MembraneNormal::FromMap(map) => {
                assert!(map.get("POPC").is_some());
                assert_relative_eq!(
                    map.get("POPC").unwrap().first().unwrap().get(1).unwrap().y,
                    3.0
                );
                assert_relative_eq!(
                    map.get("POPC").unwrap().get(1).unwrap().first().unwrap().x,
                    2.0
                );
                assert_relative_eq!(
                    map.get("POPC").unwrap().get(1).unwrap().get(2).unwrap().z,
                    8.0
                );
            }
            _ => panic!("Invalid membrane normal returned."),
        }
    }

    #[test]
    fn analysis_yaml_fail_incomplete() {
        match Analysis::from_file("tests/files/inputs/incomplete.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::CouldNotParseConfig(_, e)) => {
                assert!(e.to_string().contains("missing field"))
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    fn check_analysis_error(file: &str, expected_variant: &str) {
        match Analysis::from_file(file) {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(e) => match e {
                ConfigError::InvalidStep if expected_variant == "InvalidStep" => (),
                ConfigError::InvalidMinSamples if expected_variant == "InvalidMinSamples" => (),
                ConfigError::InvalidNThreads if expected_variant == "InvalidNThreads" => (),
                ConfigError::InvalidBeginEnd if expected_variant == "InvalidBeginEnd" => (),
                ConfigError::InvalidStructureFormat(_)
                    if expected_variant == "InvalidStructureFormat" => {}
                ConfigError::InvalidTrajectoryFormat(_)
                    if expected_variant == "InvalidTrajectoryFormat" => {}
                ConfigError::InvalidDynamicNormalRadius(_)
                    if expected_variant == "InvalidDynamicNormalRadius" => {}
                _ => panic!("Unexpected error type returned: {}", e),
            },
        }
    }

    #[test]
    fn analysis_yaml_fail_zero_step() {
        check_analysis_error("tests/files/inputs/zero_step.yaml", "InvalidStep");
    }

    #[test]
    fn analysis_yaml_fail_zero_min_samples() {
        check_analysis_error(
            "tests/files/inputs/zero_min_samples.yaml",
            "InvalidMinSamples",
        );
    }

    #[test]
    fn analysis_yaml_fail_zero_n_threads() {
        check_analysis_error("tests/files/inputs/zero_n_threads.yaml", "InvalidNThreads");
    }

    #[test]
    fn analysis_yaml_fail_start_higher_than_end() {
        check_analysis_error("tests/files/inputs/begin_higher.yaml", "InvalidBeginEnd");
    }

    #[test]
    fn analysis_yaml_fail_invalid_structure_format() {
        check_analysis_error(
            "tests/files/inputs/invalid_structure_format.yaml",
            "InvalidStructureFormat",
        );
    }

    #[test]
    fn analysis_yaml_fail_invalid_trajectory_format() {
        check_analysis_error(
            "tests/files/inputs/invalid_trajectory_format.yaml",
            "InvalidTrajectoryFormat",
        );
    }

    #[test]
    fn analysis_yaml_fail_invalid_dynamic_normal_radius() {
        // radius 0
        check_analysis_error(
            "tests/files/inputs/invalid_dynamic_normal_radius1.yaml",
            "InvalidDynamicNormalRadius",
        );
        // radius -2
        check_analysis_error(
            "tests/files/inputs/invalid_dynamic_normal_radius2.yaml",
            "InvalidDynamicNormalRadius",
        );
    }

    #[test]
    fn analysis_yaml_fail_ordermap_zero_min_samples() {
        match Analysis::from_file("tests/files/inputs/ordermap_zero_min_samples.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidOrderMap(OrderMapConfigError::InvalidMinSamples)) => (),
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_ordermap_invalid_dim_x() {
        match Analysis::from_file("tests/files/inputs/ordermap_invalid_dim_x.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidOrderMap(OrderMapConfigError::InvalidGridSpan(x, y))) => {
                assert_relative_eq!(x, 10.0);
                assert_relative_eq!(y, 7.5);
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_ordermap_invalid_dim_y() {
        match Analysis::from_file("tests/files/inputs/ordermap_invalid_dim_y.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidOrderMap(OrderMapConfigError::InvalidGridSpan(x, y))) => {
                assert_relative_eq!(x, 10.0);
                assert_relative_eq!(y, 7.5);
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_ordermap_invalid_bin_size_x() {
        match Analysis::from_file("tests/files/inputs/ordermap_invalid_bin_size_x.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidOrderMap(OrderMapConfigError::InvalidBinSize(x))) => {
                assert_relative_eq!(x, 0.0);
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_ordermap_invalid_bin_size_y() {
        match Analysis::from_file("tests/files/inputs/ordermap_invalid_bin_size_y.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidOrderMap(OrderMapConfigError::InvalidBinSize(x))) => {
                assert_relative_eq!(x, -0.1);
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_ordermap_unknown_keyword() {
        match Analysis::from_file("tests/files/inputs/ordermap_unknown_keyword.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(e) => assert!(e.to_string().contains(
                "Invalid value for 'order_map'. Expected 'default', 'true', 'null', or a valid structure."
            )),
        }
    }

    #[test]
    fn analysis_yaml_fail_estimate_error_invalid_n_blocks() {
        match Analysis::from_file("tests/files/inputs/estimate_error_invalid_n_blocks.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidErrorEstimation(ErrorEstimationError::NotEnoughBlocks(x))) => {
                assert_eq!(x, 1);
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_estimate_error_unknown_keyword() {
        match Analysis::from_file("tests/files/inputs/estimate_error_unknown_keyword.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(e) => assert!(e.to_string().contains(
                "Invalid value for 'estimate_error'. Expected 'default', 'true', 'null', or a valid structure."
            )),
        }
    }

    #[test]
    fn analysis_yaml_fail_zero_frequency() {
        match Analysis::from_file("tests/files/inputs/leaflets_zero_frequency.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(e) => assert!(e
                .to_string()
                .contains("invalid value: integer `0`, expected a nonzero usize")),
        }
    }

    #[test]
    fn analysis_yaml_fail_geometry_cuboid_invalid_dimension() {
        match Analysis::from_file("tests/files/inputs/cuboid_invalid_dimension.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidGeometry(GeometryConfigError::InvalidDimension(x, y))) => {
                assert_relative_eq!(x, 6.4);
                assert_relative_eq!(y, 4.2);
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_geometry_cylinder_radius() {
        match Analysis::from_file("tests/files/inputs/cylinder_negative_radius.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidGeometry(GeometryConfigError::InvalidRadius(x))) => {
                assert_relative_eq!(x, -1.7);
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_geometry_cylinder_height() {
        match Analysis::from_file("tests/files/inputs/cylinder_invalid_span.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidGeometry(GeometryConfigError::InvalidSpan(x, y))) => {
                assert_relative_eq!(x, -1.7);
                assert_relative_eq!(y, -1.9)
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_geometry_sphere_radius() {
        match Analysis::from_file("tests/files/inputs/sphere_negative_radius.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidGeometry(GeometryConfigError::InvalidRadius(x))) => {
                assert_relative_eq!(x, -1.7);
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_pass_multiple_trajectories_list() {
        let analysis =
            Analysis::from_file("tests/files/inputs/multiple_trajectories_list.yaml").unwrap();

        assert_eq!(analysis.structure(), "tests/files/pcpepg.tpr");
        compare_as_paths(&analysis.trajectory()[0], "tests/files/split/pcpepg1.xtc");
        compare_as_paths(&analysis.trajectory()[1], "tests/files/split/pcpepg2.xtc");
        compare_as_paths(&analysis.trajectory()[2], "tests/files/split/pcpepg3.xtc");
        compare_as_paths(&analysis.trajectory()[3], "tests/files/split/pcpepg4.xtc");
        compare_as_paths(&analysis.trajectory()[4], "tests/files/split/pcpepg5.xtc");
        assert_eq!(
            analysis.heavy_atoms().unwrap(),
            "@membrane and element name carbon"
        );
        assert_eq!(
            analysis.hydrogens().unwrap(),
            "@membrane and element name hydrogen"
        );
    }

    #[test]
    fn analysis_yaml_pass_multiple_trajectories_glob() {
        let analysis =
            Analysis::from_file("tests/files/inputs/multiple_trajectories_glob.yaml").unwrap();

        assert_eq!(analysis.structure(), "tests/files/pcpepg.tpr");
        compare_as_paths(&analysis.trajectory()[0], "tests/files/split/pcpepg1.xtc");
        compare_as_paths(&analysis.trajectory()[1], "tests/files/split/pcpepg2.xtc");
        compare_as_paths(&analysis.trajectory()[2], "tests/files/split/pcpepg3.xtc");
        compare_as_paths(&analysis.trajectory()[3], "tests/files/split/pcpepg4.xtc");
        compare_as_paths(&analysis.trajectory()[4], "tests/files/split/pcpepg5.xtc");
        assert_eq!(
            analysis.heavy_atoms().unwrap(),
            "@membrane and element name carbon"
        );
        assert_eq!(
            analysis.hydrogens().unwrap(),
            "@membrane and element name hydrogen"
        );
    }

    #[test]
    fn analysis_yaml_fail_no_trajectories() {
        match Analysis::from_file("tests/files/inputs/no_trajectories.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::NoTrajectoryFile) => (),
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_inconsistent_trajectory_formats() {
        match Analysis::from_file("tests/files/inputs/inconsistent_trajectories.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InconsistentTrajectoryFormat(x, y)) => {
                assert_eq!(y, String::from("tests/files/split/pcpepg2.xtc"));
                assert_eq!(x, String::from("tests/files/pcpepg.trr"));
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_unsupported_concatenation() {
        match Analysis::from_file("tests/files/inputs/unsupported_cat.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::TrajCatNotSupported) => (),
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_yaml_fail_ordermaps_working_directory() {
        match Analysis::from_file("tests/files/inputs/fail_ordermaps_working_dir.yaml") {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(ConfigError::InvalidOrderMap(OrderMapConfigError::InvalidOutputDirectory(x))) => {
                assert_eq!(x, String::from("."))
            }
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }
}

#[cfg(test)]
mod tests_builder {

    use approx::assert_relative_eq;
    use groan_rs::prelude::Vector3D;
    use hashbrown::HashMap;

    use crate::input::geometry::GeomReference;
    use crate::input::ordermap::Plane;
    use crate::input::Collect;
    use crate::input::{DynamicNormal, Frequency};

    use super::super::GridSpan;
    use super::*;

    fn compare_as_paths(path1: &str, path2: &str) {
        assert_eq!(Path::new(path1), Path::new(path2));
    }

    #[test]
    fn analysis_builder_pass_basic() {
        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
            .unwrap();

        assert_eq!(analysis.structure(), "tests/files/pcpepg.tpr");
        compare_as_paths(&analysis.trajectory()[0], "tests/files/pcpepg.xtc");
        assert!(analysis.index().is_none());
        assert!(analysis.output().is_none());
        assert!(analysis.output_tab().is_none());
        assert!(analysis.output_xvg().is_none());
        assert!(analysis.output_csv().is_none());
        assert!(matches!(
            analysis.membrane_normal(),
            MembraneNormal::Static(Axis::Z)
        ));
        assert_eq!(
            analysis.heavy_atoms().unwrap(),
            "@membrane and element name carbon"
        );
        assert_eq!(
            analysis.hydrogens().unwrap(),
            "@membrane and element name hydrogen"
        );
        assert!(analysis.atoms().is_none());
        assert_eq!(analysis.begin(), 0.0);
        assert!(analysis.end().is_infinite());
        assert_eq!(analysis.step(), 1);
        assert_eq!(analysis.min_samples(), 1);
        assert_eq!(analysis.n_threads(), 1);
        assert!(analysis.leaflets().is_none());
        assert!(analysis.map().is_none());
        assert!(analysis.geometry().is_none());
        assert!(!analysis.silent());
        assert!(!analysis.overwrite());
        assert!(analysis.handle_pbc());
    }

    #[test]
    fn analysis_builder_pass_full() {
        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .index("index.ndx")
            .output("order.yaml")
            .output_tab("order.dat")
            .output_xvg("order.xvg")
            .output_csv("order.csv")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .membrane_normal(Axis::X)
            .begin(100.0)
            .end(10_000.0)
            .step(5)
            .min_samples(10)
            .n_threads(4)
            .leaflets(
                LeafletClassification::global("@membrane", "name P")
                    .with_frequency(Frequency::once())
                    .with_collect("leaflets.yaml"),
            )
            .map(
                OrderMap::builder()
                    .output_directory("ordermaps")
                    .dim([
                        GridSpan::Manual {
                            start: 0.5,
                            end: 10.5,
                        },
                        GridSpan::Auto,
                    ])
                    .min_samples(10)
                    .bin_size([0.05, 0.02])
                    .plane(Plane::XY)
                    .build()
                    .unwrap(),
            )
            .estimate_error(EstimateError::new(Some(10), Some("convergence.xvg")).unwrap())
            .geometry(Geometry::cylinder("@protein and name BB", 3.5, [2.3, 5.1], Axis::Z).unwrap())
            .handle_pbc(false)
            .overwrite()
            .silent()
            .build()
            .unwrap();

        assert_eq!(analysis.structure(), "tests/files/cg.tpr");
        compare_as_paths(&analysis.trajectory()[0], "tests/files/cg.xtc");
        assert_eq!(analysis.index().as_ref().unwrap(), "index.ndx");
        assert_eq!(analysis.output().as_ref().unwrap(), "order.yaml");
        assert_eq!(analysis.output_tab().as_ref().unwrap(), "order.dat");
        assert_eq!(analysis.output_xvg().as_ref().unwrap(), "order.xvg");
        assert_eq!(analysis.output_csv().as_ref().unwrap(), "order.csv");
        assert!(matches!(
            analysis.membrane_normal(),
            MembraneNormal::Static(Axis::X)
        ));
        assert!(analysis.heavy_atoms().is_none());
        assert!(analysis.hydrogens().is_none());
        assert_eq!(analysis.atoms().unwrap(), "@membrane");
        assert_eq!(analysis.begin(), 100.0);
        assert_eq!(analysis.end(), 10_000.0);
        assert_eq!(analysis.step(), 5);
        assert_eq!(analysis.min_samples(), 10);
        assert_eq!(analysis.n_threads(), 4);

        let leaflets = analysis.leaflets().as_ref().unwrap();
        match leaflets {
            LeafletClassification::Global(x) => {
                assert_eq!(x.heads(), "name P");
                assert_eq!(x.membrane(), "@membrane");
                assert!(matches!(x.frequency(), Frequency::Once));
                assert_eq!(x.collect(), &Collect::File(String::from("leaflets.yaml")));
            }
            _ => panic!("Incorrect leaflet classification type returned."),
        }

        let map = analysis.map().as_ref().unwrap();

        assert_eq!(map.output_directory().as_ref().unwrap(), "ordermaps");
        matches!(
            map.dim_x(),
            GridSpan::Manual {
                start: 0.5,
                end: 10.5
            }
        );
        matches!(map.dim_y(), GridSpan::Auto);
        assert_eq!(map.min_samples(), 10);
        assert_eq!(map.bin_size_x(), 0.05);
        assert_eq!(map.bin_size_y(), 0.02);
        assert_eq!(map.plane().unwrap(), Plane::XY);

        let ee = analysis.estimate_error().as_ref().unwrap();

        assert_eq!(ee.n_blocks(), 10);
        assert_eq!(ee.output_convergence().unwrap(), "convergence.xvg");

        match analysis.geometry().as_ref().unwrap() {
            Geometry::Cylinder(c) => {
                match c.reference() {
                    GeomReference::Selection(x) => assert_eq!(x, "@protein and name BB"),
                    _ => panic!("Invalid geometric reference."),
                }
                assert_relative_eq!(c.radius(), 3.5);
                assert_relative_eq!(c.span()[0], 2.3);
                assert_relative_eq!(c.span()[1], 5.1);
                assert_eq!(c.orientation(), Axis::Z);
            }
            _ => panic!("Incorrect geometry type returned."),
        }

        assert!(!analysis.handle_pbc());
        assert!(analysis.overwrite());
        assert!(analysis.silent());
    }

    #[test]
    fn analysis_builder_pass_default_ordermap() {
        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .map(OrderMap::default())
            .build()
            .unwrap();

        let ordermap = analysis.map.unwrap();
        assert!(ordermap.output_directory().is_none());
        assert_relative_eq!(ordermap.bin_size_x(), 0.1);
        assert_relative_eq!(ordermap.bin_size_y(), 0.1);
        assert_eq!(ordermap.min_samples(), 1);
        matches!(ordermap.dim_x(), GridSpan::Auto);
        matches!(ordermap.dim_y(), GridSpan::Auto);
    }

    #[test]
    fn analysis_builder_pass_dynamic_membrane_normal() {
        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .membrane_normal(DynamicNormal::new("name PO4", 2.5).unwrap())
            .build()
            .unwrap();

        match analysis.membrane_normal() {
            MembraneNormal::Dynamic(dynamic) => {
                assert_eq!(dynamic.heads(), "name PO4");
                assert_relative_eq!(dynamic.radius(), 2.5);
            }
            _ => {
                panic!("Invalid membrane normal returned.")
            }
        }
    }

    #[test]
    fn analysis_builder_pass_membrane_normal_from_file() {
        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .membrane_normal("normals.yaml")
            .build()
            .unwrap();

        match analysis.membrane_normal() {
            MembraneNormal::FromFile(file) => {
                assert_eq!(file, "normals.yaml");
            }
            _ => panic!("Invalid membrane normal returned."),
        }
    }

    #[test]
    fn analysis_builder_pass_membrane_normal_from_map() {
        let mut map = HashMap::new();
        map.insert(
            "POPC".to_owned(),
            vec![
                vec![
                    Vector3D::new(1.0, 2.0, 3.0),
                    Vector3D::new(2.0, 3.0, 4.0),
                    Vector3D::new(5.0, 6.0, 7.0),
                ],
                vec![
                    Vector3D::new(2.0, 3.0, 4.0),
                    Vector3D::new(3.0, 4.0, 5.0),
                    Vector3D::new(6.0, 7.0, 8.0),
                ],
            ],
        );

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .membrane_normal(map)
            .build()
            .unwrap();

        match analysis.membrane_normal() {
            MembraneNormal::FromMap(map) => {
                assert!(map.get("POPC").is_some());
                assert_relative_eq!(
                    map.get("POPC").unwrap().first().unwrap().get(1).unwrap().y,
                    3.0
                );
                assert_relative_eq!(
                    map.get("POPC").unwrap().get(1).unwrap().first().unwrap().x,
                    2.0
                );
                assert_relative_eq!(
                    map.get("POPC").unwrap().get(1).unwrap().get(2).unwrap().z,
                    8.0
                );
            }
            _ => panic!("Invalid membrane normal returned."),
        }
    }

    #[test]
    fn analysis_builder_fail_incomplete() {
        match Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output("order.yaml")
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::UninitializedField(_)) => (),
            Err(_) => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn analysis_builder_fail_zero_step() {
        match Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .step(0)
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(_) => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn analysis_builder_fail_zero_min_samples() {
        match Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .min_samples(0)
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(_) => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn analysis_builder_fail_zero_threads() {
        match Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .n_threads(0)
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(_) => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn analysis_builder_fail_start_higher_than_end() {
        match Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .begin(100_000.0)
            .end(50_000.0)
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(_) => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn analysis_builder_fail_invalid_structure_format() {
        match Analysis::builder()
            .structure("tests/files/pcpepg.ndx")
            .trajectory("tests/files/pcpepg.xtc")
            .output("order.yaml")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(_) => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn analysis_builder_fail_invalid_trajectory_format() {
        match Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/cg_order_leaflets.tab")
            .output("order.yaml")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(_) => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn analysis_builder_pass_multiple_trajectories_list() {
        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory(vec![
                "tests/files/split/pcpepg1.xtc",
                "tests/files/split/pcpepg2.xtc",
                "tests/files/split/pcpepg3.xtc",
                "tests/files/split/pcpepg4.xtc",
                "tests/files/split/pcpepg5.xtc",
            ])
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
            .unwrap();

        assert_eq!(analysis.structure(), "tests/files/pcpepg.tpr");
        compare_as_paths(&analysis.trajectory()[0], "tests/files/split/pcpepg1.xtc");
        compare_as_paths(&analysis.trajectory()[1], "tests/files/split/pcpepg2.xtc");
        compare_as_paths(&analysis.trajectory()[2], "tests/files/split/pcpepg3.xtc");
        compare_as_paths(&analysis.trajectory()[3], "tests/files/split/pcpepg4.xtc");
        compare_as_paths(&analysis.trajectory()[4], "tests/files/split/pcpepg5.xtc");
        assert_eq!(
            analysis.heavy_atoms().unwrap(),
            "@membrane and element name carbon"
        );
        assert_eq!(
            analysis.hydrogens().unwrap(),
            "@membrane and element name hydrogen"
        );
    }

    #[test]
    fn analysis_builder_pass_multiple_trajectories_glob() {
        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/split/pcpepg?.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
            .unwrap();

        assert_eq!(analysis.structure(), "tests/files/pcpepg.tpr");
        compare_as_paths(&analysis.trajectory()[0], "tests/files/split/pcpepg1.xtc");
        compare_as_paths(&analysis.trajectory()[1], "tests/files/split/pcpepg2.xtc");
        compare_as_paths(&analysis.trajectory()[2], "tests/files/split/pcpepg3.xtc");
        compare_as_paths(&analysis.trajectory()[3], "tests/files/split/pcpepg4.xtc");
        compare_as_paths(&analysis.trajectory()[4], "tests/files/split/pcpepg5.xtc");
        assert_eq!(
            analysis.heavy_atoms().unwrap(),
            "@membrane and element name carbon"
        );
        assert_eq!(
            analysis.hydrogens().unwrap(),
            "@membrane and element name hydrogen"
        );
    }

    #[test]
    fn analysis_builder_fail_no_trajectories() {
        let no_trajectories: Vec<&str> = vec![];

        match Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory(no_trajectories)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_builder_fail_inconsistent_trajectory_formats() {
        match Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory(vec![
                "tests/files/split/pcpepg1.xtc",
                "tests/files/split/pcpepg2.xtc",
                "tests/files/pcpepg.trr",
            ])
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }

    #[test]
    fn analysis_builder_fail_unsupported_concatenation() {
        match Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory(vec!["tests/files/pcpepg.gro", "tests/files/pcpepg.gro"])
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .build()
        {
            Ok(_) => panic!("Should have failed, but succeeded."),
            Err(AnalysisBuilderError::ValidationError(_)) => (),
            Err(e) => panic!("Unexpected error type returned: {}", e),
        }
    }
}
