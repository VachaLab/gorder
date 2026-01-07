// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Contains structures and methods for the construction of maps of order parameters.

use std::ops::{Add, AddAssign};

use getset::{Getters, MutGetters};
use groan_rs::{
    errors::GridMapError,
    prelude::{GridMap, Vector3D},
    structures::gridmap::DataOrder,
};

use super::{order::OrderValue, pbc::PBCHandler};
use crate::{errors::OrderMapConfigError, input::GridSpan, input::OrderMap, PANIC_MESSAGE};

/// Order parameter map. Stores order parameters calculated for a specific bond/atom
/// and for each tile of a grid covering the membrane.
#[derive(Debug, Clone, Getters, MutGetters)]
pub(crate) struct Map {
    /// Calculation parameters.
    #[getset(get = "pub(crate)")]
    params: OrderMap,
    /// Cumulative order parameters calculated over the analysis for each tile of the grid.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    values: GridMap<OrderValue, f32, fn(&OrderValue) -> f32>,
    /// Number of samples calculated for each tile of the grid.
    #[getset(get = "pub(crate)", get_mut = "pub(crate)")]
    samples: GridMap<usize, usize, fn(&usize) -> usize>,
}

/// Convert `OrderValue` to f32.
fn from_order_value(value: &OrderValue) -> f32 {
    f32::from(*value)
}

impl Map {
    /// Construct a new ordermap.
    pub(crate) fn new<'a>(
        params: OrderMap,
        pbc_handler: &impl PBCHandler<'a>,
    ) -> Result<Map, OrderMapConfigError> {
        let binx = params.bin_size_x();
        let biny = params.bin_size_y();

        let plane = params.plane().unwrap_or_else(|| {
            panic!(
                "FATAL GORDER ERROR | Map::new | Ordermap plane should already be set. {}",
                PANIC_MESSAGE
            )
        });

        let simbox = pbc_handler.get_simbox();

        let (auto_x, auto_y) = if matches!(params.dim_x(), GridSpan::Auto)
            || matches!(params.dim_y(), GridSpan::Auto)
        {
            match simbox {
                None => return Err(OrderMapConfigError::InvalidBoxAuto),
                Some(sbox) => plane.dimensions_from_simbox(sbox),
            }
        } else {
            (0.0, 0.0) // automatic span will not be used, so we set the values to 0
        };

        let (xmin, xmax) = match params.dim_x() {
            GridSpan::Auto => (0.0, auto_x),
            GridSpan::Manual { start, end } => (start, end),
        };

        let (ymin, ymax) = match params.dim_y() {
            GridSpan::Auto => (0.0, auto_y),
            GridSpan::Manual { start, end } => (start, end),
        };

        let values = match GridMap::new((xmin, xmax), (ymin, ymax), (binx, biny), from_order_value as fn(&OrderValue) -> f32) {
            Ok(x) => x,
            Err(GridMapError::InvalidGridTile) => return Err(OrderMapConfigError::BinTooLarge((binx, biny), (xmax, ymax))),
            Err(e) => panic!("FATAL GORDER ERROR | Map::new | Could not create gridmap for values. Unexpected error {}. {}", e, PANIC_MESSAGE),
        };

        let samples = GridMap::new(
            (xmin, xmax),
            (ymin, ymax),
            (binx, biny),
            usize::clone as fn(&usize) -> usize,
        ).unwrap_or_else(|e|
            panic!("FATAL GORDER ERROR | Map::new | Could not create gridmap for samples. Unexpected error: {}. {}", e, PANIC_MESSAGE));

        Ok(Map {
            params,
            values,
            samples,
        })
    }

    /// Add sampled order parameter to the correct position. Ignore if out of range.
    #[inline(always)]
    pub(crate) fn add_order(&mut self, order: f32, bond_pos: &Vector3D) {
        let (x, y) = self
            .params()
            .plane()
            .expect(PANIC_MESSAGE)
            .projection2plane(bond_pos);

        if let (Some(valbin), Some(samplebin)) =
            (self.values.get_mut_at(x, y), self.samples.get_mut_at(x, y))
        {
            *valbin += order;
            *samplebin += 1;
        }
    }
}

impl Add<Map> for Map {
    type Output = Map;

    fn add(self, rhs: Map) -> Self::Output {
        let joined_values_map = merge_grid_maps(
            self.values,
            rhs.values,
            from_order_value as fn(&OrderValue) -> f32,
        );

        let joined_samples_map = merge_grid_maps(
            self.samples,
            rhs.samples,
            usize::clone as fn(&usize) -> usize,
        );

        Map {
            params: self.params,
            values: joined_values_map,
            samples: joined_samples_map,
        }
    }
}

impl AddAssign<Map> for Map {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Map) {
        // not a very efficient implementation, but will do
        *self = self.clone() + rhs;
    }
}

/// Helper function for merging optional Maps.
#[inline]
pub(super) fn merge_option_maps(lhs: Option<Map>, rhs: Option<Map>) -> Option<Map> {
    match (lhs, rhs) {
        (Some(x), Some(y)) => Some(x + y),
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(x)) => Some(x),
    }
}

/// Helper function for adding an optional ordermap to another optional ordermap.
#[inline]
pub(crate) fn add_option_map(lhs: &mut Option<Map>, rhs: Option<Map>) {
    if let Some(rhs_value) = rhs {
        match lhs {
            Some(lhs_value) => *lhs_value += rhs_value,
            None => *lhs = Some(rhs_value),
        }
    }
}

/// Helper function to merge two GridMaps and construct a new GridMap.
fn merge_grid_maps<T, U>(
    lhs: GridMap<T, U, fn(&T) -> U>,
    rhs: GridMap<T, U, fn(&T) -> U>,
    clone_fn: fn(&T) -> U,
) -> GridMap<T, U, fn(&T) -> U>
where
    T: Add<Output = T> + Copy + Default + std::fmt::Debug,
    U: std::fmt::Display,
{
    let merged = lhs
        .extract_raw()
        .zip(rhs.extract_raw())
        .map(|((_, _, &x), (_, _, &y))| x + y)
        .collect::<Vec<T>>();

    GridMap::from_vec(
        lhs.span_x(),
        lhs.span_y(),
        lhs.tile_dim(),
        merged,
        DataOrder::default(),
        clone_fn,
    )
    .unwrap_or_else(|_| {
        panic!(
            "FATAL GORDER ERROR | ordermap::merge_grid_maps | Could not construct merged map. {}",
            PANIC_MESSAGE
        )
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use groan_rs::prelude::SimBox;

    use crate::{
        analysis::{
            order::AnalysisOrder,
            pbc::{NoPBC, PBC3D},
        },
        input::ordermap::Plane,
    };

    use super::*;

    #[test]
    fn new_map_auto() {
        let simbox = SimBox::from([10.0, 5.0, 7.0]);
        let pbc = PBC3D::new(&simbox);
        let params = OrderMap::builder()
            .output_directory("ordermaps")
            .bin_size([0.05, 0.2])
            .plane(Plane::XY)
            .build()
            .unwrap();

        let map = Map::new(params, &pbc).unwrap();

        assert_eq!(map.samples.span_x(), (0.0, 10.0));
        assert_eq!(map.samples.span_y(), (0.0, 5.0));
        assert_eq!(map.samples.tile_dim(), (0.05, 0.2));

        assert_eq!(map.values.span_x(), (0.0, 10.0));
        assert_eq!(map.values.span_y(), (0.0, 5.0));
        assert_eq!(map.values.tile_dim(), (0.05, 0.2));
    }

    #[test]
    fn new_map_manual() {
        let simbox = SimBox::from([10.0, 5.0, 7.0]);
        let pbc = PBC3D::new(&simbox);
        let params = OrderMap::builder()
            .output_directory("ordermaps")
            .dim([
                GridSpan::Manual {
                    start: -4.0,
                    end: 8.0,
                },
                GridSpan::Manual {
                    start: 1.5,
                    end: 4.5,
                },
            ])
            .plane(Plane::XY)
            .build()
            .unwrap();

        let map = Map::new(params, &pbc).unwrap();

        assert_eq!(map.samples.span_x(), (-4.0, 8.0));
        assert_eq!(map.samples.span_y(), (1.5, 4.5));
        assert_eq!(map.samples.tile_dim(), (0.1, 0.1));

        assert_eq!(map.values.span_x(), (-4.0, 8.0));
        assert_eq!(map.values.span_y(), (1.5, 4.5));
        assert_eq!(map.values.tile_dim(), (0.1, 0.1));
    }

    #[test]
    fn new_map_fail_bin_size() {
        let simbox = SimBox::from([10.0, 3.0, 6.0]);
        let pbc = PBC3D::new(&simbox);
        let params = OrderMap::builder()
            .output_directory("ordermaps")
            .bin_size([1.0, 5.0])
            .plane(Plane::XY)
            .build()
            .unwrap();

        match Map::new(params, &pbc) {
            Ok(_) => panic!("Function should have failed."),
            Err(OrderMapConfigError::BinTooLarge((a, b), (x, y))) => {
                assert_eq!(a, 1.0);
                assert_eq!(b, 5.0);
                assert_eq!(x, 10.0);
                assert_eq!(y, 3.0);
            }
            Err(e) => panic!("Unexpected error type `{}` returned.", e),
        }
    }

    #[test]
    fn new_map_manual_nopbc() {
        let params = OrderMap::builder()
            .output_directory("ordermaps")
            .dim([
                GridSpan::Manual {
                    start: -4.0,
                    end: 8.0,
                },
                GridSpan::Manual {
                    start: 1.5,
                    end: 4.5,
                },
            ])
            .plane(Plane::XY)
            .build()
            .unwrap();

        let map = Map::new(params, &NoPBC).unwrap();

        assert_eq!(map.samples.span_x(), (-4.0, 8.0));
        assert_eq!(map.samples.span_y(), (1.5, 4.5));
        assert_eq!(map.samples.tile_dim(), (0.1, 0.1));

        assert_eq!(map.values.span_x(), (-4.0, 8.0));
        assert_eq!(map.values.span_y(), (1.5, 4.5));
        assert_eq!(map.values.tile_dim(), (0.1, 0.1));
    }

    #[test]
    fn new_map_auto_nopbc_fail() {
        for dim in [
            [
                GridSpan::Manual {
                    start: 1.5,
                    end: 4.5,
                },
                GridSpan::Auto,
            ],
            [
                GridSpan::Auto,
                GridSpan::Manual {
                    start: 1.5,
                    end: 4.5,
                },
            ],
            [GridSpan::Auto, GridSpan::Auto],
        ] {
            let params = OrderMap::builder()
                .output_directory("ordermaps")
                .dim(dim)
                .plane(Plane::XY)
                .build()
                .unwrap();

            match Map::new(params, &NoPBC) {
                Ok(_) => panic!("Function should have failed."),
                Err(OrderMapConfigError::InvalidBoxAuto) => (),
                Err(e) => panic!("Unexpected error `{}` returned.", e),
            }
        }
    }

    #[test]
    fn merge_order() {
        let order1: AnalysisOrder<super::super::timewise::AddExtend> =
            AnalysisOrder::new(0.78, 31, false);
        let order2 = AnalysisOrder::new(0.43, 14, false);
        let merged = order1 + order2;

        assert_relative_eq!(f32::from(merged.order()), 1.21);
        assert_eq!(merged.n_samples(), 45);
    }

    #[test]
    fn merge_map() {
        let values = vec![1.0, 2.5, 3.0, 4.2, 5.3, 6.1, 7.3, 8.9]
            .into_iter()
            .map(OrderValue::from)
            .collect::<Vec<OrderValue>>();
        let map1_values = GridMap::from_vec(
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 0.5),
            values,
            DataOrder::RowMajor,
            from_order_value as fn(&OrderValue) -> f32,
        )
        .unwrap();

        let samples = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let map1_samples = GridMap::from_vec(
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 0.5),
            samples,
            DataOrder::RowMajor,
            usize::to_owned as fn(&usize) -> usize,
        )
        .unwrap();

        let map1 = Map {
            params: OrderMap::builder()
                .output_directory("ordermaps")
                .build()
                .unwrap(),
            values: map1_values,
            samples: map1_samples,
        };

        let values = vec![0.7, 1.4, 2.1, 1.4, 2.3, 3.1, 3.3, 1.9]
            .into_iter()
            .map(OrderValue::from)
            .collect::<Vec<OrderValue>>();
        let map2_values = GridMap::from_vec(
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 0.5),
            values,
            DataOrder::RowMajor,
            from_order_value as fn(&OrderValue) -> f32,
        )
        .unwrap();

        let samples = vec![2, 1, 4, 3, 2, 1, 0, 2];
        let map2_samples = GridMap::from_vec(
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 0.5),
            samples,
            DataOrder::RowMajor,
            usize::to_owned as fn(&usize) -> usize,
        )
        .unwrap();

        let map2 = Map {
            params: OrderMap::builder()
                .output_directory("ordermaps")
                .build()
                .unwrap(),
            values: map2_values,
            samples: map2_samples,
        };

        let map = map1 + map2;

        let expected_values = [1.7, 3.9, 5.1, 5.6, 7.6, 9.2, 10.6, 10.8];
        let expected_samples = [3, 3, 7, 7, 7, 7, 7, 10];

        let values = map
            .values
            .extract_raw()
            .map(|(_, _, &x)| x.into())
            .collect::<Vec<f32>>();
        let samples = map
            .samples
            .extract_raw()
            .map(|(_, _, &x)| x)
            .collect::<Vec<usize>>();

        assert_eq!(values.len(), expected_values.len());
        assert_eq!(samples.len(), expected_samples.len());

        for (got, exp) in values.iter().zip(expected_values.iter()) {
            assert_relative_eq!(got, exp);
        }

        for (got, exp) in samples.iter().zip(expected_samples.iter()) {
            assert_eq!(got, exp);
        }
    }

    #[test]
    fn add_assign_map() {
        let values = vec![1.0, 2.5, 3.0, 4.2, 5.3, 6.1, 7.3, 8.9]
            .into_iter()
            .map(OrderValue::from)
            .collect::<Vec<OrderValue>>();
        let map1_values = GridMap::from_vec(
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 0.5),
            values,
            DataOrder::RowMajor,
            from_order_value as fn(&OrderValue) -> f32,
        )
        .unwrap();

        let samples = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let map1_samples = GridMap::from_vec(
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 0.5),
            samples,
            DataOrder::RowMajor,
            usize::to_owned as fn(&usize) -> usize,
        )
        .unwrap();

        let mut map1 = Map {
            params: OrderMap::builder()
                .output_directory("ordermaps")
                .build()
                .unwrap(),
            values: map1_values,
            samples: map1_samples,
        };

        let values = vec![0.7, 1.4, 2.1, 1.4, 2.3, 3.1, 3.3, 1.9]
            .into_iter()
            .map(OrderValue::from)
            .collect::<Vec<OrderValue>>();
        let map2_values = GridMap::from_vec(
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 0.5),
            values,
            DataOrder::RowMajor,
            from_order_value as fn(&OrderValue) -> f32,
        )
        .unwrap();

        let samples = vec![2, 1, 4, 3, 2, 1, 0, 2];
        let map2_samples = GridMap::from_vec(
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 0.5),
            samples,
            DataOrder::RowMajor,
            usize::to_owned as fn(&usize) -> usize,
        )
        .unwrap();

        let map2 = Map {
            params: OrderMap::builder()
                .output_directory("ordermaps")
                .build()
                .unwrap(),
            values: map2_values,
            samples: map2_samples,
        };

        map1 += map2;

        let expected_values = [1.7, 3.9, 5.1, 5.6, 7.6, 9.2, 10.6, 10.8];
        let expected_samples = [3, 3, 7, 7, 7, 7, 7, 10];

        let values = map1
            .values
            .extract_raw()
            .map(|(_, _, &x)| x.into())
            .collect::<Vec<f32>>();
        let samples = map1
            .samples
            .extract_raw()
            .map(|(_, _, &x)| x)
            .collect::<Vec<usize>>();

        assert_eq!(values.len(), expected_values.len());
        assert_eq!(samples.len(), expected_samples.len());

        for (got, exp) in values.iter().zip(expected_values.iter()) {
            assert_relative_eq!(got, exp);
        }

        for (got, exp) in samples.iter().zip(expected_samples.iter()) {
            assert_eq!(got, exp);
        }
    }
}
