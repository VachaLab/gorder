// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! This module contains structures and methods for specifying membrane normal.

use std::fmt::{self, Display};

use colored::Colorize;
use getset::{CopyGetters, Getters};
use groan_rs::prelude::Vector3D;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::{errors::ConfigError, input::Collect};

use super::Axis;

/// Structure describing the direction of the membrane normal
/// or properties necessary for its calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum MembraneNormal {
    /// Membrane normal is oriented along a dimension of the simulation box: x, y, or z axis.
    Static(Axis),
    /// Membrane normal should be calculated dynamically for each molecule based on the shape of the membrane.
    Dynamic(DynamicNormal),
    /// Membrane normals for individual molecules should be read from a yaml file.
    FromFile(String),
    /// Membrane normals for individual molecules should be set from a map.
    #[serde(alias = "Inline")]
    FromMap(HashMap<String, Vec<Vec<Vector3D>>>),
}

impl MembraneNormal {
    /// Check the validity of the membrane normal.
    pub(super) fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::Static(_) | Self::FromFile(_) | Self::FromMap(_) => Ok(()),
            Self::Dynamic(dynamic) => DynamicNormal::check_radius(dynamic.radius),
        }
    }
}

impl Display for MembraneNormal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(x) => write!(
                f,
                "Membrane normal expected to be oriented along the {} axis.",
                x.to_string().cyan()
            ),
            Self::Dynamic(_) => write!(
                f,
                "Membrane normal will be {} calculated for each molecule.",
                "dynamically".cyan(),
            ),
            Self::FromFile(x) => {
                write!(
                    f,
                    "Membrane normals for each molecule will be read from a file '{}'.",
                    x.cyan(),
                )
            }
            Self::FromMap(_) => {
                write!(
                    f,
                    "Membrane normals are provided manually for each molecule."
                )
            }
        }
    }
}

impl From<Axis> for MembraneNormal {
    fn from(value: Axis) -> Self {
        Self::Static(value)
    }
}

impl From<DynamicNormal> for MembraneNormal {
    fn from(value: DynamicNormal) -> Self {
        Self::Dynamic(value)
    }
}

impl From<&str> for MembraneNormal {
    fn from(value: &str) -> Self {
        Self::FromFile(value.to_owned())
    }
}

impl From<HashMap<String, Vec<Vec<Vector3D>>>> for MembraneNormal {
    fn from(value: HashMap<String, Vec<Vec<Vector3D>>>) -> Self {
        Self::FromMap(value)
    }
}

/// Structure describing properties of the dynamic local membrane normal calculation.
#[derive(Debug, Clone, Getters, CopyGetters, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicNormal {
    /// Reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead).
    /// There must only be one such atom/bead per lipid molecule.
    #[getset(get = "pub")]
    heads: String,
    /// Radius of the sphere for selecting nearby lipids for membrane normal estimation.
    /// The default value is 2 nm. The recommended value is half the membrane thickness.
    #[getset(get_copy = "pub")]
    #[serde(default = "default_dynamic_radius")]
    radius: f32,
    /// Should the dynamic membrane normals be collected, stored and exported into an output file?
    #[getset(get = "pub")]
    #[serde(default, alias = "export")]
    collect: Collect,
}

impl DynamicNormal {
    /// Request a dynamic local membrane normal calculation.
    ///
    /// ## Parameters
    /// - `heads`: reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead);
    ///    there must only be one such atom/bead per lipid molecule
    /// - `radius`: radius of the sphere for selecting nearby lipids for membrane normal estimation;
    ///    the recommended value is half the membrane thickness
    pub fn new(heads: &str, radius: f32) -> Result<DynamicNormal, ConfigError> {
        Self::check_radius(radius)?;

        Ok(DynamicNormal {
            heads: heads.to_owned(),
            radius,
            collect: Default::default(),
        })
    }

    /// Check that the radius specified for the dynamic membrane normal estimation is valid.
    fn check_radius(radius: f32) -> Result<(), ConfigError> {
        if radius <= 0.0 {
            Err(ConfigError::InvalidDynamicNormalRadius(radius.to_string()))
        } else {
            Ok(())
        }
    }

    /// Collect and store the dynamic membrane normals.
    /// If `true`, the normals are collected but only accessible using API.
    /// If a string is provided, the data will be exported into the output file.
    pub fn with_collect(mut self, collect: impl Into<Collect>) -> Self {
        self.collect = collect.into();
        self
    }
}

fn default_dynamic_radius() -> f32 {
    2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_deserialize() {
        match serde_yaml::from_str("!Static x").unwrap() {
            MembraneNormal::Static(axis) => assert_eq!(axis, Axis::X),
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str("!Static y").unwrap() {
            MembraneNormal::Static(axis) => assert_eq!(axis, Axis::Y),
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str("!Static z").unwrap() {
            MembraneNormal::Static(axis) => assert_eq!(axis, Axis::Z),
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str("!Dynamic { heads: \"name P\" }").unwrap() {
            MembraneNormal::Dynamic(dynamic) => {
                assert_eq!(dynamic.heads(), "name P");
                assert_relative_eq!(dynamic.radius, 2.0);
                assert_eq!(dynamic.collect(), &Collect::Boolean(false));
            }
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str("!Dynamic { heads: \"name P\", export: normals.yaml }").unwrap()
        {
            MembraneNormal::Dynamic(dynamic) => {
                assert_eq!(dynamic.heads(), "name P");
                assert_relative_eq!(dynamic.radius, 2.0);
                assert_eq!(
                    dynamic.collect(),
                    &Collect::File(String::from("normals.yaml"))
                );
            }
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str("!Dynamic { heads: \"name P\", collect: normals.yaml }").unwrap()
        {
            MembraneNormal::Dynamic(dynamic) => {
                assert_eq!(dynamic.heads(), "name P");
                assert_relative_eq!(dynamic.radius, 2.0);
                assert_eq!(
                    dynamic.collect(),
                    &Collect::File(String::from("normals.yaml"))
                );
            }
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str(
            "!Dynamic
heads: \"name P\"
radius: 3.5",
        )
        .unwrap()
        {
            MembraneNormal::Dynamic(dynamic) => {
                assert_eq!(dynamic.heads(), "name P");
                assert_relative_eq!(dynamic.radius, 3.5);
            }
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str("!FromFile normals.yaml").unwrap() {
            MembraneNormal::FromFile(x) => assert_eq!(x, "normals.yaml"),
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str(
            "!Inline 
POPC:
  - [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
  - [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]",
        )
        .unwrap()
        {
            MembraneNormal::FromMap(x) => {
                assert!(x.get("POPC").is_some());
                assert_relative_eq!(
                    x.get("POPC").unwrap().first().unwrap().get(1).unwrap().y,
                    3.0
                );
                assert_relative_eq!(
                    x.get("POPC").unwrap().get(1).unwrap().first().unwrap().x,
                    2.0
                );
                assert_relative_eq!(
                    x.get("POPC").unwrap().get(1).unwrap().get(2).unwrap().z,
                    8.0
                );
            }
            _ => panic!("Incorrect membrane normal parsed."),
        }

        match serde_yaml::from_str(
            "!FromMap
POPC:
  - [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
  - [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]",
        )
        .unwrap()
        {
            MembraneNormal::FromMap(x) => {
                assert!(x.get("POPC").is_some());
                assert_relative_eq!(
                    x.get("POPC").unwrap().first().unwrap().get(1).unwrap().y,
                    3.0
                );
                assert_relative_eq!(
                    x.get("POPC").unwrap().get(1).unwrap().first().unwrap().x,
                    2.0
                );
                assert_relative_eq!(
                    x.get("POPC").unwrap().get(1).unwrap().get(2).unwrap().z,
                    8.0
                );
            }
            _ => panic!("Incorrect membrane normal parsed."),
        }
    }

    #[test]
    fn test_serialize() {
        assert_eq!(
            serde_yaml::to_string(&MembraneNormal::Static(Axis::Z)).unwrap(),
            "!Static Z\n"
        );

        assert_eq!(
            serde_yaml::to_string(&MembraneNormal::Dynamic(
                DynamicNormal::new("name P", 3.0).unwrap()
            ))
            .unwrap(),
            "!Dynamic
heads: name P
radius: 3.0
collect: false\n"
        );

        assert_eq!(
            serde_yaml::to_string(&MembraneNormal::FromFile("normals.yaml".to_owned())).unwrap(),
            "!FromFile normals.yaml\n"
        );

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
        let from_map = serde_yaml::to_string(&MembraneNormal::FromMap(map)).unwrap();
        let expected = "!FromMap
POPC:
- - - 1.0
    - 2.0
    - 3.0
  - - 2.0
    - 3.0
    - 4.0
  - - 5.0
    - 6.0
    - 7.0
- - - 2.0
    - 3.0
    - 4.0
  - - 3.0
    - 4.0
    - 5.0
  - - 6.0
    - 7.0
    - 8.0
";
        assert_eq!(from_map, expected);
    }

    #[test]
    fn test_validate() {
        let normal: MembraneNormal = DynamicNormal::new("name P", 1.5).unwrap().into();
        assert!(normal.validate().is_ok());

        let normal: MembraneNormal = Axis::Z.into();
        assert!(normal.validate().is_ok());

        let normal: MembraneNormal =
            serde_yaml::from_str("!Dynamic { heads: name P, radius: 0.0 }").unwrap();
        assert!(matches!(
            normal.validate(),
            Err(ConfigError::InvalidDynamicNormalRadius(_))
        ));

        let normal: MembraneNormal =
            serde_yaml::from_str("!Dynamic { heads: name P, radius: -2.0 }").unwrap();
        assert!(matches!(
            normal.validate(),
            Err(ConfigError::InvalidDynamicNormalRadius(_))
        ));
    }

    #[test]
    fn test_new() {
        let normal = DynamicNormal::new("name P", 1.5).unwrap();
        assert_eq!(normal.heads(), "name P");
        assert_relative_eq!(normal.radius(), 1.5);

        assert!(DynamicNormal::new("name P", 0.0).is_err());
        assert!(DynamicNormal::new("name P", -2.0).is_err());
    }
}
