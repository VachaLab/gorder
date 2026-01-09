// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! This module contains structures and methods for specifying parameters of the analysis.

pub mod analysis;
pub mod axis;
pub mod estimate_error;
pub mod frequency;
pub mod geometry;
pub mod leaflets;
pub mod membrane_normal;
pub mod ordermap;

pub use analysis::{Analysis, AnalysisType};
pub use axis::Axis;
pub use estimate_error::EstimateError;
pub use frequency::Frequency;
pub use geometry::{GeomReference, Geometry};
pub use leaflets::LeafletClassification;
pub use membrane_normal::{DynamicNormal, MembraneNormal};
pub use ordermap::{GridSpan, OrderMap, Plane};
use serde::{Deserialize, Deserializer, Serialize};

/// Helper struct specifying whether some data should be collected or not
/// and where they should be exported to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum Collect {
    Boolean(bool),
    File(String),
}

impl From<bool> for Collect {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<&str> for Collect {
    fn from(value: &str) -> Self {
        Self::File(value.to_owned())
    }
}

impl Default for Collect {
    fn default() -> Self {
        Self::Boolean(false)
    }
}

impl<'de> Deserialize<'de> for Collect {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Collect::File(s))
    }
}
