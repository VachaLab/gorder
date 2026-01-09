// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for timewise analysis.

use crate::analysis::common::interleave_vectors;
use crate::PANIC_MESSAGE;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign};

use super::order::OrderValue;

/// Trait implemented by structures that specify what `Add` operation should be used for
/// a `TimeWiseData` structure.
pub(crate) trait TimeWiseAddTreatment: Clone + Debug {
    fn add_timewise<T: TimeWiseAddTreatment>(
        lhs: TimeWiseData<T>,
        rhs: TimeWiseData<T>,
    ) -> TimeWiseData<T>;
    fn add_assign_timewise<T: TimeWiseAddTreatment, U: TimeWiseAddTreatment>(
        lhs: &mut TimeWiseData<T>,
        rhs: TimeWiseData<U>,
    );
}

/// Specifies that when merging `TimeWiseData`, the collected values should be interleaved,
/// i.e., the length of `order` and `n_samples` should increase.
#[derive(Debug, Clone)]
pub(crate) struct AddExtend {}
impl TimeWiseAddTreatment for AddExtend {
    /// Note that this is not a commutative operation.
    #[inline]
    fn add_timewise<T: TimeWiseAddTreatment>(
        lhs: TimeWiseData<T>,
        rhs: TimeWiseData<T>,
    ) -> TimeWiseData<T> {
        if lhs.order.is_empty() {
            return rhs;
        }

        if rhs.order.is_empty() {
            return lhs;
        }

        let order = interleave_vectors(&lhs.order, &rhs.order, lhs.n_threads, rhs.n_threads);
        let n_samples =
            interleave_vectors(&lhs.n_samples, &rhs.n_samples, lhs.n_threads, rhs.n_threads);

        TimeWiseData::new(order, n_samples, lhs.n_threads + rhs.n_threads)
    }

    #[inline]
    fn add_assign_timewise<T: TimeWiseAddTreatment, U: TimeWiseAddTreatment>(
        lhs: &mut TimeWiseData<T>,
        rhs: TimeWiseData<U>,
    ) {
        if lhs.order.is_empty() {
            lhs.order = rhs.order;
            lhs.n_samples = rhs.n_samples;
            return;
        }

        if rhs.order.is_empty() {
            return;
        }

        let order = interleave_vectors(&lhs.order, &rhs.order, lhs.n_threads, rhs.n_threads);
        let n_samples =
            interleave_vectors(&lhs.n_samples, &rhs.n_samples, lhs.n_threads, rhs.n_threads);

        lhs.order = order;
        lhs.n_samples = n_samples;
        lhs.n_threads += rhs.n_threads;
    }
}

/// Specified that when merging `TimeWiseData`, the collected values should be summed,
/// i.e., the length of `order` and `n_samples` should remain the same.
#[derive(Debug, Clone)]
pub(crate) struct AddSum {}
impl TimeWiseAddTreatment for AddSum {
    #[inline]
    fn add_timewise<T: TimeWiseAddTreatment>(
        lhs: TimeWiseData<T>,
        rhs: TimeWiseData<T>,
    ) -> TimeWiseData<T> {
        if lhs.order.is_empty() {
            return rhs;
        }

        let order = lhs
            .order
            .into_iter()
            .zip(rhs.order)
            .map(|(x, y)| x + y)
            .collect::<Vec<OrderValue>>();
        let n_samples = lhs
            .n_samples
            .into_iter()
            .zip(rhs.n_samples)
            .map(|(x, y)| x + y)
            .collect::<Vec<usize>>();

        TimeWiseData::new(order, n_samples, lhs.n_threads)
    }

    #[inline]
    fn add_assign_timewise<T: TimeWiseAddTreatment, U: TimeWiseAddTreatment>(
        lhs: &mut TimeWiseData<T>,
        rhs: TimeWiseData<U>,
    ) {
        if lhs.order.is_empty() {
            lhs.order = rhs.order;
            lhs.n_samples = rhs.n_samples;
            return;
        }

        lhs.order
            .iter_mut()
            .zip(rhs.order.iter())
            .for_each(|(x, y)| *x += *y);
        lhs.n_samples
            .iter_mut()
            .zip(rhs.n_samples.iter())
            .for_each(|(x, y)| *x += y);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TimeWiseData<T: TimeWiseAddTreatment> {
    /// Order parameters calculated independently for each trajectory frame.
    order: Vec<OrderValue>,
    /// Number of samples collected from each trajectory frame.
    n_samples: Vec<usize>,
    /// Number of threads that the data have been collected by.
    n_threads: usize,
    add_treatment: PhantomData<T>,
}

impl<T: TimeWiseAddTreatment> Default for TimeWiseData<T> {
    #[inline(always)]
    fn default() -> Self {
        TimeWiseData {
            n_threads: 1,
            order: Vec::new(),
            n_samples: Vec::new(),
            add_treatment: PhantomData,
        }
    }
}

impl<T: TimeWiseAddTreatment> Add<TimeWiseData<T>> for TimeWiseData<T> {
    type Output = TimeWiseData<T>;
    #[inline(always)]
    fn add(self, rhs: TimeWiseData<T>) -> Self::Output {
        T::add_timewise(self, rhs)
    }
}

impl<T: TimeWiseAddTreatment, U: TimeWiseAddTreatment> AddAssign<TimeWiseData<U>>
    for TimeWiseData<T>
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: TimeWiseData<U>) {
        T::add_assign_timewise(self, rhs);
    }
}

impl<T: TimeWiseAddTreatment> TimeWiseData<T> {
    #[inline(always)]
    pub(crate) fn new(order: Vec<OrderValue>, n_samples: Vec<usize>, n_threads: usize) -> Self {
        Self {
            order,
            n_samples,
            n_threads,
            add_treatment: PhantomData,
        }
    }

    /// Initiate reading of the next frame.
    #[inline(always)]
    pub(super) fn next_frame(&mut self) {
        self.order.push(OrderValue::from(0.0));
        self.n_samples.push(0);
    }

    /// Estimate the calculation error from the order parameters calculated for the individual blocks.
    /// Returns samples standard deviation or None if the structure is empty, or if the number of samples
    /// in a block is zero.
    pub(super) fn estimate_error(&self, n_blocks: usize) -> Option<f32> {
        if self.order.is_empty() {
            return None;
        }

        // error estimation requires at least 2 blocks
        if n_blocks < 2 {
            panic!("FATAL GORDER ERROR | TimeWiseData::estimate_error | Cannot estimate the error from '{}' blocks. {}", n_blocks, PANIC_MESSAGE);
        }

        let block_size = self.order.len() / n_blocks;

        let mut blocks_order = vec![OrderValue::from(0.0); n_blocks];
        let mut blocks_samples = vec![0; n_blocks];

        for (i, (o, s)) in self.order.iter().zip(&self.n_samples).enumerate() {
            let block_id = i / block_size;
            if block_id < n_blocks {
                blocks_order[block_id] += *o;
                blocks_samples[block_id] += s;
            }
        }

        let mut orders: Vec<f32> = Vec::new();
        for (o, s) in blocks_order.into_iter().zip(blocks_samples) {
            if s > 0 {
                orders.push((o / s).into())
            } else {
                // can't estimate error if the number of samples in any block is 0
                return Some(f32::NAN);
            }
        }

        match std::panic::catch_unwind(|| statistical::standard_deviation(&orders, None)) {
                Ok(result) => Some(result),
                Err(e) => panic!(
                    "FATAL GORDER ERROR | TimeWiseData::estimate_error | Standard deviation calculation failed: {:?}. {}",
                    e, PANIC_MESSAGE
                ),
            }
    }

    /// Get the size of each block for error estimation.
    #[inline(always)]
    pub(super) fn block_size(&self, n_blocks: usize) -> usize {
        self.order.len() / n_blocks
    }

    /// Get the number of frames read.
    #[inline(always)]
    pub(super) fn n_frames(&self) -> usize {
        self.order.len()
    }

    /// Unpack the `TimeWiseData` structure into its individual fields.
    pub(super) fn unpack(self) -> (Vec<OrderValue>, Vec<usize>, usize) {
        (self.order, self.n_samples, self.n_threads)
    }

    /// Convert between two ways to treat addition.
    pub(super) fn switch_type<U: TimeWiseAddTreatment>(self) -> TimeWiseData<U> {
        let (order, samples, threads) = self.unpack();
        TimeWiseData::new(order, samples, threads)
    }

    /// Computes the prefix average of the order parameter values, accounting for the number
    /// of samples used in each frame. Frames with more samples contribute proportionally
    /// more to the average.
    pub(super) fn prefix_average(&self) -> Vec<f32> {
        self.order
            .iter()
            .zip(self.n_samples.iter())
            .scan((OrderValue::default(), 0), |state, (&order, &samples)| {
                let (current_sum, current_n_samples) = state;
                *current_sum += order;
                *current_n_samples += samples;
                if *current_n_samples == 0 {
                    Some(f32::NAN)
                } else {
                    Some((*current_sum / *current_n_samples).into())
                }
            })
            .collect()
    }
}

impl<T: TimeWiseAddTreatment> AddAssign<f32> for TimeWiseData<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32) {
        *self.order.last_mut().expect(PANIC_MESSAGE) += rhs;
        *self.n_samples.last_mut().expect(PANIC_MESSAGE) += 1;
    }
}

#[allow(clippy::excessive_precision)]
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn timewise_merge_simple() {
        let data1: TimeWiseData<AddExtend> = TimeWiseData {
            n_threads: 1,
            order: vec![
                OrderValue::from(1.1),
                OrderValue::from(2.2),
                OrderValue::from(3.3),
                OrderValue::from(4.4),
            ],
            n_samples: vec![1, 2, 3, 4],
            add_treatment: PhantomData,
        };

        let data2 = TimeWiseData {
            n_threads: 1,
            order: vec![
                OrderValue::from(21.1),
                OrderValue::from(22.2),
                OrderValue::from(23.3),
                OrderValue::from(24.4),
            ],
            n_samples: vec![21, 22, 23, 24],
            add_treatment: PhantomData,
        };

        let data_sum = data1 + data2;

        let expected_order: Vec<OrderValue> = [1.1, 21.1, 2.2, 22.2, 3.3, 23.3, 4.4, 24.4]
            .into_iter()
            .map(|x| x.into())
            .collect();
        let expected_n_samples = [1, 21, 2, 22, 3, 23, 4, 24];

        assert_eq!(data_sum.n_threads, 2);
        assert_eq!(data_sum.order.len(), expected_order.len());
        assert_eq!(data_sum.n_samples.len(), expected_n_samples.len());

        for (x, y) in data_sum.order.iter().zip(expected_order.iter()) {
            assert_eq!(x, y);
        }

        for (x, y) in data_sum.n_samples.iter().zip(expected_n_samples.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn timewise_merge_incomplete() {
        let data1 = TimeWiseData::<AddExtend>::new(
            vec![
                OrderValue::from(1.1),
                OrderValue::from(2.2),
                OrderValue::from(3.3),
                OrderValue::from(4.4),
            ],
            vec![1, 2, 3, 4],
            1,
        );

        let data2 = TimeWiseData::new(
            vec![
                OrderValue::from(21.1),
                OrderValue::from(22.2),
                OrderValue::from(23.3),
            ],
            vec![21, 22, 23],
            1,
        );

        let data_sum = data1 + data2;

        let expected_order: Vec<OrderValue> = [1.1, 21.1, 2.2, 22.2, 3.3, 23.3, 4.4]
            .into_iter()
            .map(|x| x.into())
            .collect();
        let expected_n_samples = [1, 21, 2, 22, 3, 23, 4];

        assert_eq!(data_sum.n_threads, 2);
        assert_eq!(data_sum.order.len(), expected_order.len());
        assert_eq!(data_sum.n_samples.len(), expected_n_samples.len());

        for (x, y) in data_sum.order.iter().zip(expected_order.iter()) {
            assert_eq!(x, y);
        }

        for (x, y) in data_sum.n_samples.iter().zip(expected_n_samples.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn timewise_merge_two_and_one() {
        let data1 = TimeWiseData::<AddExtend>::new(
            vec![
                OrderValue::from(1.1),
                OrderValue::from(11.1),
                OrderValue::from(2.2),
                OrderValue::from(12.2),
                OrderValue::from(3.3),
                OrderValue::from(13.3),
                OrderValue::from(4.4),
                OrderValue::from(14.4),
            ],
            vec![1, 11, 2, 12, 3, 13, 4, 14],
            2,
        );

        let data2 = TimeWiseData::new(
            vec![
                OrderValue::from(21.1),
                OrderValue::from(22.2),
                OrderValue::from(23.3),
                OrderValue::from(24.4),
            ],
            vec![21, 22, 23, 24],
            1,
        );

        let data_sum = data1 + data2;

        let expected_order: Vec<OrderValue> = [
            1.1, 11.1, 21.1, 2.2, 12.2, 22.2, 3.3, 13.3, 23.3, 4.4, 14.4, 24.4,
        ]
        .into_iter()
        .map(|x| x.into())
        .collect();
        let expected_n_samples = [1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24];

        assert_eq!(data_sum.n_threads, 3);
        assert_eq!(data_sum.order.len(), expected_order.len());
        assert_eq!(data_sum.n_samples.len(), expected_n_samples.len());

        for (x, y) in data_sum.order.iter().zip(expected_order.iter()) {
            assert_eq!(x, y);
        }

        for (x, y) in data_sum.n_samples.iter().zip(expected_n_samples.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn timewise_merge_two_and_one_incomplete() {
        let data1 = TimeWiseData::<AddExtend>::new(
            vec![
                OrderValue::from(1.1),
                OrderValue::from(11.1),
                OrderValue::from(2.2),
                OrderValue::from(12.2),
                OrderValue::from(3.3),
            ],
            vec![1, 11, 2, 12, 3],
            2,
        );

        let data2 = TimeWiseData::new(
            vec![
                OrderValue::from(21.1),
                OrderValue::from(22.2),
                OrderValue::from(23.3),
                OrderValue::from(24.4),
            ],
            vec![21, 22, 23, 24],
            1,
        );

        let data_sum = data1 + data2;

        let expected_order: Vec<OrderValue> = [1.1, 11.1, 21.1, 2.2, 12.2, 22.2, 3.3, 23.3, 24.4]
            .into_iter()
            .map(|x| x.into())
            .collect();
        let expected_n_samples = [1, 11, 21, 2, 12, 22, 3, 23, 24];

        assert_eq!(data_sum.n_threads, 3);
        assert_eq!(data_sum.order.len(), expected_order.len());
        assert_eq!(data_sum.n_samples.len(), expected_n_samples.len());

        for (x, y) in data_sum.order.iter().zip(expected_order.iter()) {
            assert_eq!(x, y);
        }

        for (x, y) in data_sum.n_samples.iter().zip(expected_n_samples.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn timewise_merge_one_and_two() {
        let data1 = TimeWiseData::<AddExtend>::new(
            vec![
                OrderValue::from(21.1),
                OrderValue::from(22.2),
                OrderValue::from(23.3),
                OrderValue::from(24.4),
            ],
            vec![21, 22, 23, 24],
            1,
        );

        let data2 = TimeWiseData::new(
            vec![
                OrderValue::from(1.1),
                OrderValue::from(11.1),
                OrderValue::from(2.2),
                OrderValue::from(12.2),
                OrderValue::from(3.3),
                OrderValue::from(13.3),
                OrderValue::from(4.4),
                OrderValue::from(14.4),
            ],
            vec![1, 11, 2, 12, 3, 13, 4, 14],
            2,
        );

        let data_sum = data1 + data2;

        let expected_order: Vec<OrderValue> = [
            21.1, 1.1, 11.1, 22.2, 2.2, 12.2, 23.3, 3.3, 13.3, 24.4, 4.4, 14.4,
        ]
        .into_iter()
        .map(|x| x.into())
        .collect();
        let expected_n_samples = [21, 1, 11, 22, 2, 12, 23, 3, 13, 24, 4, 14];

        assert_eq!(data_sum.n_threads, 3);
        assert_eq!(data_sum.order.len(), expected_order.len());
        assert_eq!(data_sum.n_samples.len(), expected_n_samples.len());

        for (x, y) in data_sum.order.iter().zip(expected_order.iter()) {
            assert_eq!(x, y);
        }

        for (x, y) in data_sum.n_samples.iter().zip(expected_n_samples.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn timewise_merge_four_and_three() {
        let data1 = TimeWiseData::<AddExtend>::new(
            vec![
                OrderValue::from(1.0),
                OrderValue::from(2.0),
                OrderValue::from(3.0),
                OrderValue::from(4.0),
                OrderValue::from(1.1),
                OrderValue::from(2.1),
                OrderValue::from(3.1),
                OrderValue::from(4.1),
                OrderValue::from(1.2),
                OrderValue::from(2.2),
                OrderValue::from(3.2),
                OrderValue::from(4.2),
            ],
            vec![10, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42],
            4,
        );

        let data2 = TimeWiseData::new(
            vec![
                OrderValue::from(5.0),
                OrderValue::from(6.0),
                OrderValue::from(7.0),
                OrderValue::from(5.1),
                OrderValue::from(6.1),
                OrderValue::from(7.1),
                OrderValue::from(5.2),
                OrderValue::from(6.2),
                OrderValue::from(7.2),
            ],
            vec![50, 60, 70, 51, 61, 71, 52, 62, 72],
            3,
        );

        let data_sum = data1 + data2;

        let expected_order: Vec<OrderValue> = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 1.2, 2.2, 3.2,
            4.2, 5.2, 6.2, 7.2,
        ]
        .into_iter()
        .map(|x| x.into())
        .collect();
        let expected_n_samples = [
            10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 71, 12, 22, 32, 42, 52, 62, 72,
        ];

        assert_eq!(data_sum.n_threads, 7);
        assert_eq!(data_sum.order.len(), expected_order.len());
        assert_eq!(data_sum.n_samples.len(), expected_n_samples.len());

        for (x, y) in data_sum.order.iter().zip(expected_order.iter()) {
            assert_eq!(x, y);
        }

        for (x, y) in data_sum.n_samples.iter().zip(expected_n_samples.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn estimate_error() {
        let order = [
            10.0, 15.0, 18.0, 12.0, 14.0, 15.0, 16.0, 20.0, 21.0, 18.0, 9.0, 11.0, 13.0, 14.0,
            19.0, 16.0, 17.0,
        ];

        let data = TimeWiseData::<AddExtend>::new(
            order
                .into_iter()
                .map(|x| x.into())
                .collect::<Vec<OrderValue>>(),
            vec![
                10, 12, 15, 11, 13, 11, 11, 17, 18, 15, 8, 10, 12, 13, 17, 14, 15,
            ],
            3,
        );

        // blocks: 1.16216, 1.1714, 1.2391, 1.1515, 1.0952 (last two numbers ignored)

        let error = data.estimate_error(5).unwrap();
        assert_relative_eq!(error, 0.0514468);
    }

    #[test]
    fn estimate_error_empty_structure() {
        let data: TimeWiseData<AddExtend> = TimeWiseData::new(Vec::new(), Vec::new(), 1);
        assert!(data.estimate_error(5).is_none());
    }

    #[test]
    fn test_prefix_average() {
        let order = [10.0, 12.0, 15.0, 10.0, 9.0, 12.0, 98432.0];
        let samples = vec![13, 15, 20, 12, 11, 14, 98432];
        let data =
            TimeWiseData::<AddSum>::new(order.into_iter().map(|x| x.into()).collect(), samples, 1);

        let prefix_average = data.prefix_average();
        let expected = [
            0.769230769,
            0.785714286,
            0.770833333,
            0.783333333,
            0.788732394,
            0.8,
            0.999827441,
        ];

        assert_eq!(prefix_average.len(), expected.len());

        for (val, exp) in prefix_average.into_iter().zip(expected.into_iter()) {
            assert_relative_eq!(val, exp, epsilon = 1e-5);
        }
    }
}
