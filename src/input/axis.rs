// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Contains the implementation of the `Axis` structure and its methods.

use std::fmt;

use groan_rs::prelude::{Dimension, Vector3D};
use serde::{Deserialize, Serialize};

use crate::PANIC_MESSAGE;

use super::ordermap::Plane;

/// Represents the X, Y, or Z axis.
#[derive(Debug, Clone, PartialEq, Eq, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum Axis {
    #[serde(alias = "x")]
    X,
    #[serde(alias = "y")]
    Y,
    #[serde(alias = "z")]
    Z,
}

impl Axis {
    /// Return a plane perpendicular to this axis.
    pub(crate) fn perpendicular(&self) -> Plane {
        match self {
            Axis::X => Plane::YZ,
            Axis::Y => Plane::XZ,
            Axis::Z => Plane::XY,
        }
    }
}

impl fmt::Display for Axis {
    /// Print Axis.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::X => write!(f, "x"),
            Self::Y => write!(f, "y"),
            Self::Z => write!(f, "z"),
        }
    }
}

impl From<Axis> for Vector3D {
    /// Convert from Axis to Vector3D.
    fn from(value: Axis) -> Self {
        match value {
            Axis::X => Vector3D::new(1.0, 0.0, 0.0),
            Axis::Y => Vector3D::new(0.0, 1.0, 0.0),
            Axis::Z => Vector3D::new(0.0, 0.0, 1.0),
        }
    }
}

impl From<Axis> for Dimension {
    /// Convert from Axis to Dimension.
    fn from(value: Axis) -> Self {
        match value {
            Axis::X => Dimension::X,
            Axis::Y => Dimension::Y,
            Axis::Z => Dimension::Z,
        }
    }
}

impl From<Dimension> for Axis {
    /// Convert from Dimension to Axis.
    fn from(value: Dimension) -> Self {
        match value {
            Dimension::X => Axis::X,
            Dimension::Y => Axis::Y,
            Dimension::Z => Axis::Z,
            unknown => panic!("FATAL GORDER ERROR | Axis::from::<Dimension> | Dimension '{}' could not be converted to Axis. {}", unknown, PANIC_MESSAGE),
        }
    }
}

impl From<&str> for Axis {
    /// Convert from string to Axis.
    fn from(value: &str) -> Self {
        match value {
            "x" => Axis::X,
            "y" => Axis::Y,
            "z" => Axis::Z,
            unknown => panic!(
                "FATAL GORDER ERROR | Axis::from::<&str> | String '{}' could not be converted to Axis. {}",
                unknown,
                PANIC_MESSAGE
            ),
        }
    }
}

impl From<&String> for Axis {
    fn from(value: &String) -> Self {
        Axis::from(value.as_str())
    }
}
