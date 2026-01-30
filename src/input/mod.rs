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

use std::fmt;

pub use analysis::{Analysis, AnalysisType};
pub use axis::Axis;
pub use estimate_error::EstimateError;
pub use frequency::Frequency;
pub use geometry::{GeomReference, Geometry};
pub use leaflets::LeafletClassification;
pub use membrane_normal::{DynamicNormal, MembraneNormal};
pub use ordermap::{GridSpan, OrderMap, Plane};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize,
};

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
        struct CollectVisitor;

        impl<'de> Visitor<'de> for CollectVisitor {
            type Value = Collect;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a boolean or a string")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Collect::Boolean(v))
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Collect::File(v.to_owned()))
            }

            fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Collect::File(v))
            }
        }

        deserializer.deserialize_any(CollectVisitor)
    }
}

#[cfg(test)]
mod tests_collect {
    use super::*;
    use serde_yaml;

    #[derive(Debug, PartialEq, Deserialize)]
    struct TestStruct {
        collect: Collect,
    }

    #[test]
    fn test_collect_deserialize() {
        let yaml = "collect: false";
        let test: TestStruct = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(test.collect, Collect::Boolean(false));

        let yaml = "collect: true";
        let test: TestStruct = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(test.collect, Collect::Boolean(true));

        let yaml = "collect: \"path/to/file\"";
        let test: TestStruct = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(test.collect, Collect::File("path/to/file".to_owned()));

        let yaml = "collect: file";
        let test: TestStruct = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(test.collect, Collect::File("file".to_owned()));
    }
}
