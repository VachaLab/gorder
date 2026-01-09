// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

use std::{num::NonZeroUsize, ops::Mul};

use serde::{Deserialize, Serialize};

use crate::{errors::FrequencyError, PANIC_MESSAGE};

/// Frequency of some action being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum Frequency {
    /// Perform the action every N analyzed trajectory frames.
    Every(NonZeroUsize),
    /// Only perform the action once (using the first frame).
    Once,
}

impl Frequency {
    /// Perform the action once.
    #[inline(always)]
    pub fn once() -> Self {
        Frequency::Once
    }

    /// Perform the action every N frames.
    /// Returns an error if `n_frames` is 0.
    #[inline(always)]
    pub fn every(n_frames: usize) -> Result<Self, FrequencyError> {
        Ok(Frequency::Every(
            NonZeroUsize::try_from(n_frames).map_err(|_| FrequencyError::EveryZero)?,
        ))
    }
}

impl Default for Frequency {
    /// Default frequency is every frame.
    #[inline(always)]
    fn default() -> Self {
        Frequency::Every(NonZeroUsize::try_from(1).expect(PANIC_MESSAGE))
    }
}

impl Mul<NonZeroUsize> for Frequency {
    type Output = Frequency;

    #[inline(always)]
    fn mul(self, rhs: NonZeroUsize) -> Frequency {
        match self {
            Frequency::Every(n) => Frequency::every(n.get() * rhs.get()).expect(PANIC_MESSAGE),
            Frequency::Once => Frequency::once(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_create() {
        let freq = Frequency::once();
        assert!(matches!(freq, Frequency::Once));

        let freq = Frequency::every(5).unwrap();
        match freq {
            Frequency::Every(x) => assert_eq!(x.get(), 5),
            _ => panic!("Invalid frequency type returned."),
        }

        let freq = Frequency::every(1).unwrap();
        match freq {
            Frequency::Every(x) => assert_eq!(x.get(), 1),
            _ => panic!("Invalid frequency type returned."),
        }

        match Frequency::every(0) {
            Ok(_) => panic!("Function should have failed."),
            Err(FrequencyError::EveryZero) => (),
        }
    }

    #[test]
    fn test_frequency_from_yaml() {
        let freq: Frequency = serde_yaml::from_str("!Once").unwrap();
        assert!(matches!(freq, Frequency::Once));

        let freq: Frequency = serde_yaml::from_str("!Every 5").unwrap();
        match freq {
            Frequency::Every(x) => assert_eq!(x.get(), 5),
            _ => panic!("Invalid frequency type returned."),
        }

        let freq: Frequency = serde_yaml::from_str("!Every 1").unwrap();
        match freq {
            Frequency::Every(x) => assert_eq!(x.get(), 1),
            _ => panic!("Invalid frequency type returned."),
        }

        match serde_yaml::from_str::<Frequency>("!Every 0") {
            Ok(_) => panic!("Function should have failed."),
            Err(e) => assert_eq!(
                e.to_string(),
                "invalid value: integer `0`, expected a nonzero usize"
            ),
        }
    }

    #[test]
    fn test_frequency_mul() {
        let freq = Frequency::every(7).unwrap() * NonZeroUsize::new(4).unwrap();
        match freq {
            Frequency::Every(x) => assert_eq!(x.get(), 28),
            _ => panic!("Invalid frequency type returned."),
        }

        let freq = Frequency::once() * NonZeroUsize::new(4).unwrap();
        assert!(matches!(freq, Frequency::Once));
    }
}
