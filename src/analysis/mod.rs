// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! This module contains the implementation of the analysis logic.

use groan_rs::prelude::Vector3D;

use crate::errors::OrderMapConfigError;
use crate::input::membrane_normal::MembraneNormal;
use crate::input::{Analysis, AnalysisType};
use crate::presentation::AnalysisResults;

mod aaorder;
mod cgorder;
mod clustering;
mod common;
pub(crate) mod geometry;
mod index;
pub(crate) mod leaflets;
pub(crate) mod normal;
pub(crate) mod order;
pub(crate) mod ordermap;
pub(crate) mod pbc;
mod spherical_clustering;
mod spinner;
mod structure;
pub(crate) mod timewise;
pub(crate) mod topology;
mod uaorder;

impl Analysis {
    /// Perform the analysis.
    pub fn run(mut self) -> Result<AnalysisResults, Box<dyn std::error::Error + Send + Sync>> {
        self.init_ordermap(self.membrane_normal().clone())?;
        self.info();

        match self.analysis_type() {
            AnalysisType::AAOrder {
                heavy_atoms: _,
                hydrogens: _,
            } => aaorder::analyze_atomistic(self),
            AnalysisType::CGOrder { beads: _ } => cgorder::analyze_coarse_grained(self),
            AnalysisType::UAOrder {
                saturated: _,
                unsaturated: _,
                ignore: _,
            } => uaorder::analyze_united(self),
        }
    }

    /// Finish the ordermap plane initialization.
    fn init_ordermap(
        &mut self,
        membrane_normal: MembraneNormal,
    ) -> Result<(), OrderMapConfigError> {
        if let Some(map) = self.map_mut().as_mut() {
            if map.plane().is_some() {
                return Ok(());
            }

            return match membrane_normal {
                MembraneNormal::Static(axis) => {
                    map.set_plane(Some(axis.perpendicular()));
                    Ok(())
                }
                MembraneNormal::Dynamic(_)
                | MembraneNormal::FromFile(_)
                | MembraneNormal::FromMap(_) => Err(OrderMapConfigError::InvalidPlaneAuto),
            };
        }

        Ok(())
    }
}

/// Calculate instantenous value of order parameter of a bond defined by a vector going from atom1 to atom2.
#[inline(always)]
pub(super) fn calc_sch(vector: &Vector3D, membrane_normal: &Vector3D) -> f32 {
    let angle = vector.angle(membrane_normal);
    let cos = angle.cos();
    (1.5 * cos * cos) - 0.5
}

#[cfg(test)]
mod tests {
    use crate::input::{Axis, DynamicNormal, OrderMap};
    use crate::prelude::Plane;

    use super::*;
    use approx::assert_relative_eq;
    use groan_rs::prelude::{Dimension, SimBox};
    use hashbrown::HashMap;

    #[test]
    fn test_calc_sch() {
        let pos1 = Vector3D::new(1.7, 2.1, 9.7);
        let pos2 = Vector3D::new(1.9, 2.4, 0.8);

        let simbox = SimBox::from([10.0, 10.0, 10.0]);

        assert_relative_eq!(
            calc_sch(&(pos1.vector_to(&pos2, &simbox)), &Dimension::Z.into()),
            0.8544775
        );
    }

    #[test]
    fn test_init_ordermap() {
        let mut analysis = Analysis::builder()
            .structure("md.tpr")
            .trajectory("md.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .ordermaps(OrderMap::builder().plane(Plane::XZ).build().unwrap())
            .build()
            .unwrap();

        analysis.init_ordermap(Axis::Z.into()).unwrap();
        assert_eq!(analysis.map().as_ref().unwrap().plane().unwrap(), Plane::XZ);

        analysis
            .init_ordermap(DynamicNormal::new("name PO4", 2.0).unwrap().into())
            .unwrap();
        assert_eq!(analysis.map().as_ref().unwrap().plane().unwrap(), Plane::XZ);

        let mut analysis = Analysis::builder()
            .structure("md.tpr")
            .trajectory("md.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .ordermaps(OrderMap::default())
            .build()
            .unwrap();

        match analysis.init_ordermap(DynamicNormal::new("name PO4", 2.0).unwrap().into()) {
            Ok(_) => panic!("Function should have failed."),
            Err(OrderMapConfigError::InvalidPlaneAuto) => (),
            Err(e) => panic!("Unexpected error type `{}` returned.", e),
        }

        analysis.init_ordermap(Axis::Z.into()).unwrap();
        assert_eq!(analysis.map().as_ref().unwrap().plane().unwrap(), Plane::XY);

        let mut analysis = Analysis::builder()
            .structure("md.tpr")
            .trajectory("md.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .ordermaps(OrderMap::default())
            .build()
            .unwrap();

        match analysis.init_ordermap("normals.yaml".into()) {
            Ok(_) => panic!("Function should have failed."),
            Err(OrderMapConfigError::InvalidPlaneAuto) => (),
            Err(e) => panic!("Unexpected error type `{}` returned.", e),
        }

        let mut map = HashMap::new();
        map.insert(
            "POPE".to_owned(),
            vec![
                vec![Vector3D::new(1.0, 2.0, 3.0)],
                vec![Vector3D::new(2.0, 3.0, 4.0)],
            ],
        );
        match analysis.init_ordermap(map.into()) {
            Ok(_) => panic!("Function should have failed."),
            Err(OrderMapConfigError::InvalidPlaneAuto) => (),
            Err(e) => panic!("Unexpected error type `{}` returned.", e),
        }
    }
}
