// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Contains structures and methods for the construction of maps of order parameters.

use std::{fmt, path::PathBuf};

use derive_builder::Builder;
use getset::{CopyGetters, Getters, Setters};
use groan_rs::prelude::{SimBox, Vector3D};
use serde::{Deserialize, Serialize};

use crate::{
    errors::{GridSpanError, OrderMapConfigError},
    PANIC_MESSAGE,
};

/// Orientation of the order map. Should correspond to the plane in which the membrane is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum Plane {
    #[default]
    #[serde(alias = "xy")]
    XY,
    #[serde(alias = "xz")]
    XZ,
    #[serde(alias = "yz")]
    YZ,
}

impl Plane {
    /// Get the dimensions of the map from the simulation box dimensions.
    #[inline(always)]
    pub(crate) fn dimensions_from_simbox(&self, simbox: &SimBox) -> (f32, f32) {
        match self {
            Plane::XY => (simbox.x, simbox.y),
            Plane::XZ => (simbox.x, simbox.z),
            Plane::YZ => (simbox.z, simbox.y),
        }
    }

    /// Get projection of the position to the plane.
    #[inline(always)]
    pub(crate) fn projection2plane(&self, position: &Vector3D) -> (f32, f32) {
        match self {
            Plane::XY => (position.x, position.y),
            Plane::XZ => (position.x, position.z),
            Plane::YZ => (position.z, position.y),
        }
    }

    /// Get labels for the axes of the ordermap.
    #[inline(always)]
    pub(crate) fn get_labels(&self) -> (char, char) {
        match self {
            Plane::XY => ('x', 'y'),
            Plane::XZ => ('x', 'z'),
            Plane::YZ => ('z', 'y'),
        }
    }
}

impl fmt::Display for Plane {
    /// Print Plane.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::XY => write!(f, "xy"),
            Self::XZ => write!(f, "xz"),
            Self::YZ => write!(f, "yz"),
        }
    }
}

/// Parameters for constructing ordermaps.
#[derive(Debug, Clone, Builder, Getters, CopyGetters, Setters, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct OrderMap {
    /// Directory where the output files containing the individual ordermaps will be saved.
    #[builder(setter(into, strip_option), default)]
    #[serde(alias = "output_dir")]
    #[getset(get = "pub")]
    output_directory: Option<String>,

    /// Minimum number of samples required in a grid tile to calculate the order parameter.
    /// The default value is 1.
    #[builder(default = "1")]
    #[serde(default = "default_min_samples")]
    #[getset(get_copy = "pub")]
    min_samples: usize,

    /// Span of the grid along the axes.
    /// The first span corresponds to the x-axis (if the map is in the xy or xz plane) or the z-axis (if the map is in the yz plane).
    /// The second span corresponds to the y-axis (if the map is in the xy or yz plane) or the z-axis (if the map is in the xz plane).
    /// If not specified, the span is derived from the simulation box size of the input structure.
    #[builder(default)]
    #[serde(default = "default_gridspan")]
    #[getset(get_copy = "pub")]
    dim: [GridSpan; 2],

    /// Size of the grid bin along the axes.
    /// The first bin dimension corresponds to the x-axis (if the map is in the xy or xz plane) or the z-axis (if the map is in the yz plane).
    /// The second bin dimension corresponds to the y-axis (if the map is in the xy or yz plane) or the z-axis (if the map is in the xz plane).
    /// The default value is 0.1 x 0.1 nm if not specified.
    #[builder(default = "[0.1, 0.1]")]
    #[serde(default = "default_bin_size")]
    #[getset(get_copy = "pub")]
    bin_size: [f32; 2],

    /// Plane in which the ordermaps should be constructed.
    /// If not specified, the plane is assumed to be perpendicular to the membrane normal.
    #[builder(setter(strip_option), default)]
    #[getset(get_copy = "pub", set = "pub(crate)")]
    plane: Option<Plane>,
}

impl Default for OrderMap {
    /// Set the default parameters for the ordermaps calculation.
    /// Warning: this sets NO output directory for the ordermaps, i.e. ordermaps will not be written out
    /// and will only be available using public API.
    fn default() -> Self {
        Self::builder().build().unwrap_or_else(|e| {
            panic!(
                "FATAL GORDER ERROR | OrderMap::default | Could not build default OrderMap (`{}`)",
                e
            )
        })
    }
}

impl OrderMap {
    /// Get the bin size of the map along the primary axis.
    pub(crate) fn bin_size_x(&self) -> f32 {
        self.bin_size[0]
    }

    /// Get the bin size of the map along the secondary axis.
    pub(crate) fn bin_size_y(&self) -> f32 {
        self.bin_size[1]
    }

    /// Get the span of the grid along the primary axis.
    pub(crate) fn dim_x(&self) -> GridSpan {
        self.dim[0]
    }

    /// Get the span of the grid along the secondary axis.
    pub(crate) fn dim_y(&self) -> GridSpan {
        self.dim[1]
    }
}

fn default_bin_size() -> [f32; 2] {
    [0.1, 0.1]
}

fn default_min_samples() -> usize {
    1
}

fn default_gridspan() -> [GridSpan; 2] {
    [GridSpan::Auto, GridSpan::Auto]
}

fn validate_min_samples(samples: usize) -> Result<(), OrderMapConfigError> {
    if samples == 0 {
        Err(OrderMapConfigError::InvalidMinSamples)
    } else {
        Ok(())
    }
}

fn validate_gridspan(gridspan: &[GridSpan; 2]) -> Result<(), OrderMapConfigError> {
    for span in gridspan {
        if let GridSpan::Manual { start, end } = span {
            if start > end {
                return Err(OrderMapConfigError::InvalidGridSpan(*start, *end));
            }
        }
    }

    Ok(())
}

fn validate_bin_size(size: [f32; 2]) -> Result<(), OrderMapConfigError> {
    for dimension in size {
        if dimension <= 0.0 {
            return Err(OrderMapConfigError::InvalidBinSize(dimension));
        }
    }

    Ok(())
}

/// Check that the output directory for ordermaps is not the current directory.
/// If this is not checked, the user could delete their entire working directory on accident.
fn validate_output_directory(directory: Option<&String>) -> Result<(), OrderMapConfigError> {
    if let Some(directory) = directory {
        if PathBuf::from(directory)
            .canonicalize()
            .unwrap_or(PathBuf::from(directory)) // if the directory does not exist, we are fine
            == std::env::current_dir()
                .expect(PANIC_MESSAGE)
                .canonicalize()
                .expect(PANIC_MESSAGE)
        {
            return Err(OrderMapConfigError::InvalidOutputDirectory(
                directory.clone(),
            ));
        }
    }

    Ok(())
}

impl OrderMap {
    /// Start providing ordermap parameters.
    pub fn builder() -> OrderMapBuilder {
        OrderMapBuilder::default()
    }

    /// Check that the OrderMap is valid. This is used after deserializing the structure from the yaml config file.
    pub(crate) fn validate(&self) -> Result<(), OrderMapConfigError> {
        validate_min_samples(self.min_samples)?;
        validate_gridspan(&self.dim)?;
        validate_bin_size(self.bin_size)?;
        validate_output_directory(self.output_directory().as_ref())?;
        Ok(())
    }
}

impl OrderMapBuilder {
    fn validate(&self) -> Result<(), String> {
        if let Some(directory) = &self.output_directory {
            validate_output_directory(directory.as_ref()).map_err(|e| e.to_string())?;
        }

        if let Some(min_samples) = self.min_samples {
            validate_min_samples(min_samples).map_err(|e| e.to_string())?;
        }

        if let Some(bin) = self.bin_size {
            validate_bin_size(bin).map_err(|e| e.to_string())?;
        }

        // no need to validate GridSpan as that is guaranteed to be valid

        Ok(())
    }
}

/// Specifies the span of an ordermap grid.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum GridSpan {
    /// Span should be obtained from the input structure file.
    #[default]
    Auto,
    /// Span is directly provided.
    Manual { start: f32, end: f32 },
}

impl GridSpan {
    /// Create a new valid `GridSpan` structure from the provided floats.
    ///
    /// This performs sanity checks, making sure that start is not higher than end.
    pub fn manual(start: f32, end: f32) -> Result<GridSpan, GridSpanError> {
        if start > end {
            return Err(GridSpanError::Invalid(start, end));
        }

        Ok(GridSpan::Manual { start, end })
    }
}

#[cfg(test)]
mod test_grid_span {
    use super::*;

    #[test]
    fn test_manual_pass() {
        GridSpan::manual(0.0, 10.0).unwrap();
    }

    #[test]
    fn test_manual_fail_invalid() {
        let error = GridSpan::manual(10.0, 0.0).unwrap_err();
        assert!(matches!(error, GridSpanError::Invalid(x, y) if x == 10.0 && y == 0.0));
    }
}

#[cfg(test)]
mod test_ordermap {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_ordermap_pass_basic() {
        let map = OrderMap::builder().build().unwrap();

        assert!(map.output_directory().is_none());
        assert_relative_eq!(map.bin_size_x(), 0.1);
        assert_relative_eq!(map.bin_size_y(), 0.1);
        assert_eq!(map.min_samples(), 1);
        matches!(map.dim_x(), GridSpan::Auto);
        matches!(map.dim_y(), GridSpan::Auto);
    }

    #[test]
    fn test_ordermap_pass_default() {
        let map = OrderMap::default();

        assert!(map.output_directory().is_none());
        assert_relative_eq!(map.bin_size_x(), 0.1);
        assert_relative_eq!(map.bin_size_y(), 0.1);
        assert_eq!(map.min_samples(), 1);
        matches!(map.dim_x(), GridSpan::Auto);
        matches!(map.dim_y(), GridSpan::Auto);
    }

    #[test]
    fn test_ordermap_pass_full() {
        let map = OrderMap::builder()
            .output_directory("ordermaps")
            .bin_size([0.2, 0.01])
            .dim([GridSpan::manual(-5.0, 10.0).unwrap(), GridSpan::Auto])
            .min_samples(10)
            .build()
            .unwrap();

        assert_eq!(map.output_directory().as_ref().unwrap(), "ordermaps");
        assert_relative_eq!(map.bin_size_x(), 0.2);
        assert_relative_eq!(map.bin_size_y(), 0.01);
        assert_eq!(map.min_samples(), 10);
        matches!(
            map.dim_x(),
            GridSpan::Manual {
                start: 0.5,
                end: 10.5
            }
        );
        matches!(map.dim_y(), GridSpan::Auto);
    }

    #[test]
    fn test_ordermap_fail_zero_min_samples() {
        match OrderMap::builder()
            .output_directory("ordermaps")
            .min_samples(0)
            .build()
        {
            Ok(_) => panic!("Function should have failed but it succeeded."),
            Err(OrderMapBuilderError::ValidationError(x)) => assert!(x.contains("min_samples")),
            Err(e) => panic!("Unexpected error type returned {}", e),
        }
    }

    #[test]
    fn test_ordermap_fail_invalid_bin_size_x() {
        match OrderMap::builder()
            .output_directory("ordermaps")
            .bin_size([0.0, 0.1])
            .build()
        {
            Ok(_) => panic!("Function should have failed but it succeeded."),
            Err(OrderMapBuilderError::ValidationError(x)) => {
                assert!(x.contains("invalid bin size"))
            }
            Err(e) => panic!("Unexpected error type returned {}", e),
        }
    }

    #[test]
    fn test_ordermap_fail_invalid_bin_size_y() {
        match OrderMap::builder()
            .output_directory("ordermaps")
            .bin_size([0.1, -0.3])
            .build()
        {
            Ok(_) => panic!("Function should have failed but it succeeded."),
            Err(OrderMapBuilderError::ValidationError(x)) => {
                assert!(x.contains("invalid bin size"))
            }
            Err(e) => panic!("Unexpected error type returned {}", e),
        }
    }

    #[test]
    fn test_ordermap_fail_working_directory1() {
        match OrderMap::builder().output_directory(".").build() {
            Ok(_) => panic!("Function should have failed but it succeeded."),
            Err(OrderMapBuilderError::ValidationError(x)) => {
                assert!(x.contains("output directory specified for saving ordermaps cannot be the current directory"));
            }
            Err(e) => panic!("Unexpected error type returned {}", e),
        }
    }

    #[test]
    fn test_ordermap_fail_working_directory2() {
        match OrderMap::builder()
            .output_directory("tests/../src/..")
            .build()
        {
            Ok(_) => panic!("Function should have failed but it succeeded."),
            Err(OrderMapBuilderError::ValidationError(x)) => {
                assert!(x.contains("output directory specified for saving ordermaps cannot be the current directory"));
            }
            Err(e) => panic!("Unexpected error type returned {}", e),
        }
    }
}
