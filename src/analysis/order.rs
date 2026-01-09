// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Contains implementation of structures for storing and calculating order parameters.

use crate::analysis::timewise::{AddSum, TimeWiseAddTreatment, TimeWiseData};
use crate::PANIC_MESSAGE;
use getset::{CopyGetters, Getters};
use serde::Deserialize;
use std::num::NonZeroUsize;
use std::ops::{Add, AddAssign, Div};

const PRECISION: f64 = 1_000_000.0;

/// Value of the order parameter x PRECISION rounded to the nearest integer.
/// Avoids issues with floating point precision.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Deserialize, Default)]
#[serde(transparent)]
pub(crate) struct OrderValue(i64);

impl From<f32> for OrderValue {
    fn from(value: f32) -> Self {
        // we do not check for overflow here as `value` should never be larger than 1.0
        OrderValue((value as f64 * PRECISION).round() as i64)
    }
}

impl From<OrderValue> for f32 {
    fn from(value: OrderValue) -> Self {
        ((value.0 as f64) / PRECISION) as f32
    }
}

impl Div<usize> for OrderValue {
    type Output = OrderValue;

    fn div(self, rhs: usize) -> Self::Output {
        OrderValue(self.0 / TryInto::<i64>::try_into(rhs)
            .unwrap_or_else(|e| panic!("FATAL GORDER ERROR | OrderValue::div | Conversion of usize to i64 failed. {}. Value of '{}' cannot be converted. {}", e, rhs, PANIC_MESSAGE))
        )
    }
}

impl Add<OrderValue> for OrderValue {
    type Output = OrderValue;

    fn add(self, rhs: OrderValue) -> Self::Output {
        OrderValue(
            self.0.checked_add(rhs.0)
                .unwrap_or_else(|| panic!("FATAL GORDER ERROR | OrderValue::add | OrderValue overflowed. Tried adding '{}' and '{}'. {}", self.0, rhs.0, PANIC_MESSAGE))
        )
    }
}

impl AddAssign<OrderValue> for OrderValue {
    fn add_assign(&mut self, rhs: OrderValue) {
        self.0 = self.0.checked_add(rhs.0)
            .unwrap_or_else(|| panic!("FATAL GORDER ERROR | OrderValue::add_assign | OrderValue overflowed. Tried adding '{}' and '{}'. {}", self.0, rhs.0, PANIC_MESSAGE));
    }
}

impl AddAssign<f32> for OrderValue {
    fn add_assign(&mut self, rhs: f32) {
        self.add_assign(OrderValue::from(rhs))
    }
}

/// Structure for calculating order parameters from the simulation.
#[derive(Debug, Clone, CopyGetters, Getters)]
pub(crate) struct AnalysisOrder<T: TimeWiseAddTreatment> {
    /// Cumulative order parameter calculated over the analysis.
    #[getset(get_copy = "pub(super)")]
    order: OrderValue,
    /// Number of samples collected for this order parameter.
    #[getset(get_copy = "pub(super)")]
    n_samples: usize,
    /// Data for time-wise analysis.
    timewise: Option<TimeWiseData<T>>,
}

impl<T: TimeWiseAddTreatment> AnalysisOrder<T> {
    #[inline(always)]
    pub(crate) fn new(order: f32, n_samples: usize, timewise: bool) -> AnalysisOrder<T> {
        let timewise = if timewise {
            Some(TimeWiseData::default())
        } else {
            None
        };

        AnalysisOrder {
            order: OrderValue::from(order),
            n_samples,
            timewise,
        }
    }

    /// Calculate average order from the collected data.
    ///
    /// Return `f32::NAN` if the number of samples is lower than the required minimal number.
    #[inline(always)]
    pub(crate) fn calc_order(&self, min_samples: NonZeroUsize) -> f32 {
        if self.n_samples < min_samples.into() {
            f32::NAN
        } else {
            f32::from(self.order / self.n_samples)
        }
    }

    /// Initialize analysis of a new frame. This only does something if `timewise` calculation is requested.
    #[inline(always)]
    pub(crate) fn init_new_frame(&mut self) {
        if let Some(x) = self.timewise.as_mut() {
            x.next_frame();
        }
    }

    /// Data for time-wise analysis.
    pub(super) fn timewise(&self) -> Option<&TimeWiseData<T>> {
        self.timewise.as_ref()
    }

    /// Estimate error for this order parameter.
    /// Returns `None` if no information necessary for the error estimation is available.
    /// Returns NaN if the total number of samples is below `min_samples`.
    pub(crate) fn estimate_error(&self, n_blocks: usize, min_samples: NonZeroUsize) -> Option<f32> {
        let error = self
            .timewise
            .as_ref()
            .and_then(|data| data.estimate_error(n_blocks));

        if error.is_some() && self.n_samples < min_samples.into() {
            Some(f32::NAN)
        } else {
            error
        }
    }

    /// Switch between two ways of treating addition in `timewise`.
    pub(crate) fn switch_type<U: TimeWiseAddTreatment>(self) -> AnalysisOrder<U> {
        AnalysisOrder {
            order: self.order,
            n_samples: self.n_samples,
            timewise: self.timewise.map(|x| x.switch_type()),
        }
    }

    /// Computes the prefix average of the order parameter values, accounting for the number
    /// of samples used in each frame. Frames with more samples contribute proportionally
    /// more to the average.
    /// Returns an empty vector if `timewise` is None or empty.
    pub(crate) fn order_prefix_average(&self) -> Vec<f32> {
        if let Some(timewise) = self.timewise() {
            timewise.prefix_average()
        } else {
            Vec::new()
        }
    }
}

impl<T: TimeWiseAddTreatment> Add<AnalysisOrder<T>> for AnalysisOrder<T> {
    type Output = AnalysisOrder<T>;

    #[inline(always)]
    fn add(self, rhs: AnalysisOrder<T>) -> Self::Output {
        let timewise = match (self.timewise, rhs.timewise) {
            (Some(x), Some(y)) => Some(x + y),
            _ => None,
        };

        AnalysisOrder {
            order: self.order + rhs.order,
            n_samples: self.n_samples + rhs.n_samples,
            timewise,
        }
    }
}

impl<T: TimeWiseAddTreatment> AddAssign<f32> for AnalysisOrder<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32) {
        self.order += rhs;
        self.n_samples += 1;

        if let Some(x) = self.timewise.as_mut() {
            *x += rhs;
        }
    }
}

impl<T: TimeWiseAddTreatment, U: TimeWiseAddTreatment> AddAssign<AnalysisOrder<U>>
    for AnalysisOrder<T>
{
    #[inline]
    fn add_assign(&mut self, rhs: AnalysisOrder<U>) {
        self.order += rhs.order;
        self.n_samples += rhs.n_samples;

        match (&mut self.timewise, rhs.timewise) {
            (None, Some(rhs_timewise)) => {
                self.timewise = Some(rhs_timewise.switch_type());
            }
            (Some(lhs_timewise), Some(rhs_timewise)) => {
                *lhs_timewise += rhs_timewise;
            }
            _ => (),
        }
    }
}

/// Helper function for merging optional order parameters.
#[inline]
pub(super) fn merge_option_order<T: TimeWiseAddTreatment>(
    lhs: Option<AnalysisOrder<T>>,
    rhs: Option<AnalysisOrder<T>>,
) -> Option<AnalysisOrder<T>> {
    match (lhs, rhs) {
        (Some(x), Some(y)) => Some(x + y),
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(x)) => Some(x),
    }
}

/// Helper function for adding an optional order parameter to another optional order parameter.
#[inline(always)]
pub(crate) fn add_option_order<T: TimeWiseAddTreatment>(
    lhs: &mut Option<AnalysisOrder<AddSum>>,
    rhs: Option<AnalysisOrder<T>>,
) {
    if let Some(rhs_value) = rhs {
        match lhs {
            Some(lhs_value) => *lhs_value += rhs_value,
            None => *lhs = Some(rhs_value.switch_type::<AddSum>()),
        }
    }
}
