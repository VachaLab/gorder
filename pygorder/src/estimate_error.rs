// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

use pyo3::prelude::*;

use gorder_core::input::EstimateError as RsEstimateError;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::ConfigError;

/// Parameters for estimating the error of the analysis.
///
/// Parameters
/// ----------
/// n_blocks : int, default=5
///     Number of blocks to divide the trajectory for error estimation.
///     Must be at least 2. It is recommended not to modify this value to
///     artificially reduce error estimates.
/// output_convergence : Optional[str], default=None
///     Filename for writing convergence data. If omitted, convergence analysis
///     is still performed but results are not written into a file.
///
/// Raises
/// ------
/// ConfigError
///     If `n_blocks` is less than 2.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.estimate_error")]
#[derive(Clone)]
pub struct EstimateError(pub(crate) RsEstimateError);

#[gen_stub_pymethods]
#[pymethods]
impl EstimateError {
    #[new]
    #[pyo3(signature = (n_blocks = 5, output_convergence = None))]
    pub fn new(n_blocks: usize, output_convergence: Option<&str>) -> PyResult<Self> {
        Ok(Self(
            RsEstimateError::new(Some(n_blocks), output_convergence)
                .map_err(|e| ConfigError::new_err(e.to_string()))?,
        ))
    }
}
