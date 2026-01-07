// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! # gorder: Everything you will ever need for lipid order calculations
//!
//! A crate for calculating lipid order parameters from Gromacs simulations.
//! `gorder` can calculate atomistic, coarse-grained, as well as united-atom lipid order parameters.
//!
//! **It is recommended to first read the [gorder manual](https://ladme.github.io/gorder-manual/) to understand the capabilities
//! of `gorder` and then refer to this documentation for details about the Rust API.**
//!
//! ## Usage
//!
//! Run:
//!
//! ```bash
//! $ cargo add gorder
//! ```
//!
//! Import the crate in your Rust code:
//!
//! ```rust
//! use gorder::prelude::*;
//! ```
//!
//! `gorder` is also available as a command-line tool. You can install it using:
//!
//! ```bash
//! $ cargo install gorder
//! ```
//!
//! ## Quick examples
//!
//! Basic analysis of atomistic lipid order parameters:
//!
//! ```no_run
//! use gorder::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     // Construct the analysis
//!     let analysis = Analysis::builder()
//!         .structure("system.tpr")                   // Structure file
//!         .trajectory("md.xtc")                      // Trajectory file to analyze
//!         .output("order.yaml")                      // Output YAML file
//!         .analysis_type(AnalysisType::aaorder(      // Type of analysis to perform
//!             "@membrane and element name carbon",   // Selection of heavy atoms
//!             "@membrane and element name hydrogen", // Selection of hydrogens
//!         ))
//!         .build()?;                                 // Build the analysis
//!
//!     // Activate colog for logging (requires the `colog` crate)
//!     colog::init();
//!
//!     // Run the analysis and write the output
//!     analysis.run()?.write()?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ***
//!
//! Basic analysis of coarse-grained lipid order parameters:
//!
//! ```no_run
//! use gorder::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     // Construct the analysis
//!     let analysis = Analysis::builder()
//!         .structure("system.tpr")                   // Structure file
//!         .trajectory("md.xtc")                      // Trajectory file to analyze
//!         .output("order.yaml")                      // Output YAML file
//!         .analysis_type(AnalysisType::cgorder(      // Type of analysis to perform
//!             "@membrane",                           // Selection of beads
//!         ))
//!         .build()?;                                 // Build the analysis
//!
//!     // Activate colog for logging (requires the `colog` crate)
//!     colog::init();
//!
//!     // Run the analysis and write the output
//!     analysis.run()?.write()?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ***
//!
//! Basic analysis of united-atom lipid order parameters:
//!
//! ```no_run
//! use gorder::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     // Construct the analysis
//!     let analysis = Analysis::builder()
//!         .structure("system.tpr")                   // Structure file
//!         .trajectory("md.xtc")                      // Trajectory file to analyze
//!         .output("order.yaml")                      // Output YAML file
//!         .analysis_type(AnalysisType::uaorder(      // Type of analysis to perform
//!             Some("element name carbon and not name C15 C34 C24 C25"),  // Selection of satured carbons
//!             Some("name C24 C25"),                  // Selection of unsaturated carbons
//!             None,                                  // Selection of atoms to ignore
//!         ))
//!         .build()?;                                 // Build the analysis
//!
//!     // Activate colog for logging (requires the `colog` crate)
//!     colog::init();
//!
//!     // Run the analysis and write the output
//!     analysis.run()?.write()?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ***
//!
//! The [`Analysis`](crate::prelude::Analysis) structure includes many optional fields.
//!
//! ```no_run
//! use gorder::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     // Construct the analysis
//!     let analysis = Analysis::builder()
//!         .structure("system.tpr")                   // Structure file
//!         .bonds("bonds.bnd")                        // Topology file containing bonds (not needed with TPR structure file)
//!         .trajectory("md.xtc")                      // Trajectory file to analyze
//!         .index("index.ndx")                        // Input NDX file
//!         .output_yaml("order.yaml")                 // Output YAML file
//!         .output_tab("order.tab")                   // Output table file
//!         .output_xvg("order.xvg")                   // Pattern for output XVG files
//!         .output_csv("order.csv")                   // Output CSV file
//!         .analysis_type(AnalysisType::cgorder(      // Type of analysis to perform
//!             "@membrane",                           // Selection of beads
//!         ))
//!         .membrane_normal(Axis::Z)                  // Membrane normal
//!         .begin(100_000.0)                          // Start time of analysis
//!         .end(200_000.0)                            // End time of analysis
//!         .step(5)                                   // Analyze every Nth frame
//!         .min_samples(100)                          // Minimum required samples
//!         .n_threads(4)                              // Number of threads to use
//!         .leaflets(                                 // Calculate order for individual leaflets
//!             LeafletClassification::global(         // Method for classifying lipids into leaflets
//!                 "@membrane",                       // Lipids for membrane center
//!                 "name PO4"                         // Lipid heads selection
//!             )
//!             .with_frequency(Frequency::once())     // Frequency of classification
//!         )
//!         .ordermaps(                                // Construct maps of order parameters
//!             OrderMap::builder()
//!                 .output_directory("ordermaps")     // Directory for order maps
//!                 .dim([                             // Dimensions of the map
//!                     GridSpan::Manual {             // X-dimension span
//!                         start: 5.0,                // Start at 5 nm
//!                         end: 10.0,                 // End at 10 nm
//!                     },
//!                     GridSpan::Auto,                // Auto span for Y-dimension
//!                 ])
//!                 .bin_size([0.05, 0.2])             // Grid bin size
//!                 .min_samples(30)                   // Minimum samples per bin
//!                 .plane(Plane::XY)                  // Orientation of the map
//!                 .build()?
//!         )  
//!         .estimate_error(EstimateError::new(        // Estimate error for calculations
//!             Some(10),                              // Number of blocks for averaging
//!             Some("convergence.xvg")                // Output file for convergence
//!         )?)
//!         .geometry(Geometry::cylinder(              // Only consider bonds inside a cylinder
//!             "@protein",                            // Reference position for the cylinder
//!             3.0,                                   // Radius of the cylinder
//!             [-2.0, 2.0],                           // Span of the cylinder relative to reference
//!             Axis::Z                                // Orientation of the main cylinder axis
//!         )?)               
//!         .handle_pbc(true)                          // Handle periodic boundary conditions?
//!         .build()?;                                 // Build the analysis
//!
//!     // Activate colog for logging (requires the `colog` crate)
//!     colog::init();
//!
//!     // Run the analysis and write the output
//!     analysis.run()?.write()?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Detailed usage
//!
//! **It is recommended to first read the [gorder manual](https://ladme.github.io/gorder-manual/) to understand the capabilities
//! of `gorder` and then refer to this documentation for details about the Rust API.**
//!
//! Performing lipid order parameter calculations using the `gorder` crate consists of three main steps:
//!
//! 1. Constructing the [`Analysis`](crate::prelude::Analysis) structure.
//! 2. Running the analysis.
//! 3. Inspecting the results.
//!
//! ### Step 1: Constructing the `Analysis` structure
//!
//! Start by including the prelude of the `gorder` crate:
//!
//! ```no_run
//! use gorder::prelude::*;
//! ```
//!
//! The [`Analysis`](crate::prelude::Analysis) structure is constructed using the "builder" pattern.
//! First, initiate the builder:
//!
//! ```no_run
//! # use gorder::prelude::*;
//! #
//! # let analysis =
//! Analysis::builder()
//! # ;
//! ```
//!
//! Then, add your options to it:
//!
//! ```no_run
//! # use gorder::prelude::*;
//! #
//! # let builder = Analysis::builder()
//! .structure("system.tpr")
//! .trajectory("md.xtc")
//! .analysis_type(AnalysisType::aaorder(
//!     "@membrane and element name carbon",
//!     "@membrane and element name hydrogen")
//! )
//! # ;
//! ```
//! > *When using the `gorder` application, specifying the output YAML file is mandatory.
//! > However, when you are using `gorder` as a crate, specifying the output file is optional,
//! > as you might not require any output to be generated.*
//!
//! Finally, assemble the `Analysis`:
//!
//! ```no_run
//! # use gorder::prelude::*;
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! # let analysis = Analysis::builder()
//! # .structure("system.tpr")
//! # .trajectory("md.xtc")
//! # .analysis_type(AnalysisType::aaorder(
//! #     "@membrane and element name carbon",
//! #     "@membrane and element name hydrogen")
//! # )
//! .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! Alternatively, you can construct the `Analysis` using an input YAML file that is also used by the CLI version of `gorder`:
//!
//! ```no_run
//! # use gorder::prelude::*;
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let analysis = Analysis::from_file("analysis.yaml")?;
//! # Ok(())
//! # }
//! ```
//!
//! See the [gorder manual](https://ladme.github.io/gorder-manual/) for the format of this input YAML file.
//!
//! To learn more about the various input parameters for `Analysis`, refer to:
//! 1. [`AnalysisBuilder`](crate::prelude::AnalysisBuilder) for an overview of the builder.
//! 2. [`AnalysisType`](crate::prelude::AnalysisType) for the types of analysis.
//! 3. [`OrderMapBuilder`](crate::prelude::OrderMapBuilder) and [`OrderMap`](crate::prelude::OrderMap) for specifying order parameter maps.
//! 4. [`LeafletClassification`](crate::prelude::LeafletClassification) for leaflet classification.
//! 5. [`MembraneNormal`](crate::prelude::MembraneNormal), [`Axis`](crate::prelude::Axis), and [`DynamicNormal`](crate::prelude::DynamicNormal) for membrane normal specification.
//! 6. [`EstimateError`](crate::prelude::EstimateError) for error estimation.
//! 7. [`Geometry`](crate::prelude::Geometry) for geometry selection.
//!
//! ### Step 2: Running the analysis
//!
//! Once the `Analysis` structure is ready, running the analysis is straightforward:
//!
//! ```no_run
//! # use gorder::prelude::*;
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! # let analysis = Analysis::builder()
//! # .structure("system.tpr")
//! # .trajectory("md.xtc")
//! # .analysis_type(AnalysisType::aaorder(
//! #     "@membrane and element name carbon",
//! #     "@membrane and element name hydrogen")
//! # ).build()?;
//! #
//! let results = analysis.run()?;
//! # Ok(())
//! # }
//! ```
//!
//! It is also recommened to initialize some logging crate, if you want to see information about the progress of the analysis.
//!
//! ```no_run
//! # use gorder::prelude::*;
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! # let analysis = Analysis::builder()
//! # .structure("system.tpr")
//! # .trajectory("md.xtc")
//! # .analysis_type(AnalysisType::aaorder(
//! #     "@membrane and element name carbon",
//! #     "@membrane and element name hydrogen")
//! # ).build()?;
//! #
//! colog::init();
//! let results = analysis.run()?;
//! # Ok(())
//! # }
//! ```
//!
//! The [`Analysis::run`](crate::prelude::Analysis::run) method returns an
//! [`AnalysisResults`](crate::prelude::AnalysisResults) enum containing the results.
//!
//! ### Step 3: Inspecting the results
//!
//! The simplest way to inspect results is by writing output files. These files must be specified during the construction of the `Analysis` structure:
//!
//! 1. [`AnalysisBuilder::output_yaml`](`crate::prelude::AnalysisBuilder::output_yaml`) for YAML file,
//! 2. [`AnalysisBuilder::output_csv`](`crate::prelude::AnalysisBuilder::output_csv`) for CSV file,
//! 3. [`AnalysisBuilder::output_xvg`](`crate::prelude::AnalysisBuilder::output_xvg`) for XVG files,
//! 4. [`AnalysisBuilder::output_tab`](`crate::prelude::AnalysisBuilder::output_tab`) for files in "table" format.
//!
//! To write the output files, call:
//!
//! ```no_run
//! # use gorder::prelude::*;
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! # let analysis = Analysis::builder()
//! # .structure("system.tpr")
//! # .trajectory("md.xtc")
//! # .analysis_type(AnalysisType::aaorder(
//! #     "@membrane and element name carbon",
//! #     "@membrane and element name hydrogen")
//! # ).build()?;
//! #
//! # let results = analysis.run()?;
//! results.write()?;
//! # Ok(())
//! # }
//! ```
//!
//! Alternatively, results can be extracted programmatically. Match the [`AnalysisResults`](crate::prelude::AnalysisResults) enum to access the results:
//!
//! ```no_run
//! # use gorder::prelude::*;
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! # let analysis = Analysis::builder()
//! # .structure("system.tpr")
//! # .trajectory("md.xtc")
//! # .analysis_type(AnalysisType::aaorder(
//! #     "@membrane and element name carbon",
//! #     "@membrane and element name hydrogen")
//! # ).build()?;
//! #
//! # let results = analysis.run()?;
//! let aa_results = match results {
//!     AnalysisResults::AA(aa_results) => aa_results,
//!     _ => panic!("Expected atomistic results."),
//! };
//! # Ok(())
//! # }
//! ```
//!
//! Then, inspect the results as needed. Refer to:
//! 1. [`AAOrderResults`](crate::prelude::AAOrderResults) for atomistic results.
//! 2. [`CGOrderResults`](crate::prelude::CGOrderResults) for coarse-grained results.
//! 3. [`UAOrderResults`](crate::prelude::UAOrderResults) for united-atom results.

// `doc_overindented_list_items` lint does not work correctly...
#![allow(clippy::doc_overindented_list_items)]

/// Version of the `gorder` crate.
pub const GORDER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Message that should be added to every panic.
pub(crate) const PANIC_MESSAGE: &str =
    "\n\n\n            >>> THIS SHOULD NOT HAVE HAPPENED! PLEASE REPORT THIS ERROR <<<
(open an issue at 'github.com/Ladme/gorder/issues' or write an e-mail to 'ladmeb@gmail.com')\n\n";

/// Log colored info message.
#[macro_export]
macro_rules! colog_info {
    ($msg:expr) => {
        log::info!($msg)
    };
    ($msg:expr, $($arg:expr),+ $(,)?) => {{
        use colored::Colorize;
        log::info!($msg, $( $arg.to_string().cyan() ),+)
    }};
}

/// Log colored warning message.
#[macro_export]
macro_rules! colog_warn {
    ($msg:expr) => {
        log::warn!($msg)
    };
    ($msg:expr, $($arg:expr),+ $(,)?) => {{
        use colored::Colorize;
        log::warn!($msg, $( $arg.to_string().yellow() ),+)
    }};
}

/// Specifies the leaflet a lipid is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Leaflet {
    #[serde(alias = "1")]
    Upper,
    #[serde(alias = "0")]
    Lower,
}

impl Display for Leaflet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Leaflet::Upper => write!(f, "upper"),
            Leaflet::Lower => write!(f, "lower"),
        }
    }
}

mod analysis;
pub mod errors;
pub mod input;
mod lanczos;
pub mod presentation;

use std::fmt::Display;

use serde::{Deserialize, Serialize};

/// This module contains re-exported public structures of the `gorder` crate.
pub mod prelude {
    pub use crate::Leaflet;

    pub use super::input::{
        analysis::AnalysisBuilder, ordermap::OrderMapBuilder, Analysis, AnalysisType, Axis,
        DynamicNormal, EstimateError, Frequency, GeomReference, Geometry, GridSpan,
        LeafletClassification, MembraneNormal, OrderMap, Plane,
    };

    pub use super::analysis::topology::atom::AtomType;

    pub use super::presentation::{
        aaresults::{AAAtomResults, AAMoleculeResults, AAOrderResults},
        cgresults::{CGMoleculeResults, CGOrderResults},
        convergence::Convergence,
        uaresults::{UAAtomResults, UABondResults, UAMoleculeResults, UAOrderResults},
        AnalysisResults, BondResults, GridMapF32, Order, OrderCollection, OrderMapsCollection,
        PublicMoleculeResults, PublicOrderResults,
    };

    pub use groan_rs::prelude::Vector3D;
}
