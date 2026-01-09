// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! This module contains structures and methods for presenting the results of the analysis.

use crate::analysis::topology::atom::AtomType;
use crate::analysis::topology::bond::BondTopology;
use crate::analysis::topology::molecule::MoleculeTypes;
use crate::analysis::topology::OrderCalculable;
use crate::input::{Collect, MembraneNormal};
use crate::input::{Frequency, LeafletClassification, Plane};
use crate::presentation::aaresults::AAOrderResults;
use crate::presentation::cgresults::CGOrderResults;
use crate::presentation::csv_presenter::{CsvPresenter, CsvProperties, CsvWrite};
use crate::presentation::leaflets::LeafletsData;
use crate::presentation::normals::NormalsData;
use crate::presentation::ordermaps_presenter::MapWrite;
use crate::presentation::ordermaps_presenter::{OrderMapPresenter, OrderMapProperties};
use crate::presentation::tab_presenter::{TabPresenter, TabProperties, TabWrite};
use crate::presentation::xvg_presenter::{XvgPresenter, XvgProperties, XvgWrite};
use crate::presentation::yaml_presenter::{YamlPresenter, YamlProperties, YamlWrite};
use crate::{
    analysis::topology::SystemTopology, errors::WriteError, input::Analysis, PANIC_MESSAGE,
};
use colored::Colorize;
use convergence::{ConvPresenter, ConvProperties, Convergence};
use getset::{CopyGetters, Getters};
use groan_rs::prelude::GridMap;
use indexmap::IndexMap;
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use std::fmt::Debug;
use std::fmt::Write as fmtWrite;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
use strum_macros::Display;
use uaresults::UAOrderResults;

macro_rules! write_result {
    ($dst:expr, $($arg:tt)*) => {
        write!($dst, $($arg)*).map_err(|e| WriteError::CouldNotWriteResults(e))?
    };
}

pub mod aaresults;
pub mod cgresults;
pub(crate) mod converter;
mod csv_presenter;
//pub(crate) mod ordermap;
pub mod convergence;
mod leaflets;
mod normals;
pub mod ordermaps_presenter;
mod tab_presenter;
pub mod uaresults;
mod xvg_presenter;
mod yaml_presenter;

/// Enum representing any of the types of results that can be returned by the `gorder`.
#[derive(Debug, Clone)]
pub enum AnalysisResults {
    AA(AAOrderResults),
    CG(CGOrderResults),
    UA(UAOrderResults),
}

impl AnalysisResults {
    /// Write the results of the analysis into output files.
    pub fn write(&self) -> Result<(), WriteError> {
        match self {
            AnalysisResults::AA(x) => x.write_all_results(),
            AnalysisResults::CG(x) => x.write_all_results(),
            AnalysisResults::UA(x) => x.write_all_results(),
        }
    }

    /// Get the total number of analyzed frames.
    pub fn n_analyzed_frames(&self) -> usize {
        match self {
            AnalysisResults::AA(x) => x.n_analyzed_frames(),
            AnalysisResults::CG(x) => x.n_analyzed_frames(),
            AnalysisResults::UA(x) => x.n_analyzed_frames(),
        }
    }

    /// Get the analysis options.
    pub fn analysis(&self) -> &Analysis {
        match self {
            AnalysisResults::AA(x) => x.analysis(),
            AnalysisResults::CG(x) => x.analysis(),
            AnalysisResults::UA(x) => x.analysis(),
        }
    }
}

/// Type alias for a gridmap of f32 values.
pub type GridMapF32 = GridMap<f32, f32, fn(&f32) -> f32>;

/// Public trait implemented by results-containing structures.
pub trait PublicOrderResults {
    #[allow(private_bounds)]
    type MoleculeResults: MoleculeResults;

    /// Get the results for all molecules.
    fn molecules(&self) -> impl Iterator<Item = &Self::MoleculeResults>;

    /// Get the results for a molecule with the specified name.
    /// O(1) complexity.
    /// Returns `None` if such molecule does not exist.
    fn get_molecule(&self, name: &str) -> Option<&Self::MoleculeResults>;

    /// Get the parameters of the analysis.
    fn analysis(&self) -> &Analysis;

    /// Get the total number of analyzed frames.
    fn n_analyzed_frames(&self) -> usize;
}

/// Trait implemented by all structures providing the full results of the analysis.
pub(crate) trait OrderResults:
    Debug + Clone + CsvWrite + TabWrite + YamlWrite + PublicOrderResults
{
    type OrderType: OrderType;
    type MoleculeBased: OrderCalculable;

    /// Create an empty `OrderResults` structure (i.e., without any molecules).
    fn empty(analysis: Analysis) -> Self;

    /// Create a new `OrderResults` structure.
    fn new(
        molecules: IndexMap<String, Self::MoleculeResults>,
        average_order: OrderCollection,
        average_ordermaps: OrderMapsCollection,
        leaflets_data: Option<LeafletsData>,
        normals_data: Option<NormalsData>,
        analysis: Analysis,
        n_analyzed_frames: usize,
    ) -> Self;

    /// Return the maximal number of bonds for heavy atoms in the system.
    /// Only makes sense for atomistic order. Returns 0 by default.
    #[inline(always)]
    fn max_bonds(&self) -> usize {
        0
    }

    /// Get reference to average ordermaps calculated for the entire membrane.
    fn average_ordermaps(&self) -> &OrderMapsCollection;

    /// Get reference to the leaflets data, if there are any.
    fn leaflets_data(&self) -> &Option<LeafletsData>;

    /// Get reference to the collected membrane normals, if there are any.
    fn normals_data(&self) -> &Option<NormalsData>;

    /// Write results of the analysis into the output files.
    fn write_all_results(&self) -> Result<(), WriteError> {
        if self.molecules().count() == 0 {
            log::warn!("Nothing to write.");
            return Ok(());
        }

        let analysis = self.analysis();
        let errors = analysis.estimate_error().is_some();
        let leaflets = analysis.leaflets().is_some();
        let input_structure = analysis.structure();
        let input_trajectory = analysis.trajectory();
        let overwrite = analysis.overwrite();

        if let Some(yaml) = analysis.output_yaml() {
            YamlPresenter::new(self, YamlProperties::new(input_structure, input_trajectory))
                .write(yaml, overwrite)?;
        }

        if let Some(tab) = analysis.output_tab() {
            TabPresenter::new(
                self,
                TabProperties::new(self, input_structure, input_trajectory, errors, leaflets),
            )
            .write(tab, overwrite)?;
        }

        if let Some(xvg) = analysis.output_xvg() {
            XvgPresenter::new(
                self,
                XvgProperties::new(input_structure, input_trajectory, leaflets),
            )
            .write(xvg, overwrite)?;
        }

        if let Some(csv) = analysis.output_csv() {
            CsvPresenter::new(self, CsvProperties::new(self, errors, leaflets))
                .write(csv, overwrite)?;
        }

        if let Some(estimate_error) = analysis.estimate_error() {
            if let Some(convergence) = estimate_error.output_convergence() {
                ConvPresenter::new(
                    self,
                    ConvProperties::new(input_structure, input_trajectory, leaflets),
                )
                .write(convergence, overwrite)?;
            }
        }

        if let Some(map) = analysis.map() {
            if let Some(output_dir) = map.output_directory() {
                OrderMapPresenter::new(
                    self,
                    OrderMapProperties::new(map.plane().unwrap_or(Plane::XY)),
                )
                .write(output_dir, overwrite)?;
            }
        }

        if let Some(data) = self.leaflets_data() {
            // only export data, if an output file is provided
            if let Collect::File(filename) = analysis
                .leaflets()
                .as_ref()
                .expect(PANIC_MESSAGE)
                .get_collect()
            {
                data.export(filename, analysis.trajectory(), analysis.overwrite())?;
            }
        }

        if let Some(data) = self.normals_data() {
            // only export data, if an output file is provided
            if let MembraneNormal::Dynamic(params) = analysis.membrane_normal() {
                if let Collect::File(filename) = params.collect() {
                    data.export(filename, analysis.trajectory(), analysis.overwrite())?;
                }
            }
        }

        Ok(())
    }
}

/// Trait implemented by all structures providing the results of the analysis for a single molecule type.
pub(crate) trait MoleculeResults:
    Debug + Clone + CsvWrite + TabWrite + XvgWrite + MapWrite + PublicMoleculeResults
{
    /// Return the maximal number of bonds for heavy atoms in the molecule.
    /// Only makes sense for atomistic order. Returns 0 by default.
    #[inline(always)]
    fn max_bonds(&self) -> usize {
        0
    }
}

/// Public trait implemented by molecule results.
pub trait PublicMoleculeResults {
    /// Data about convergence of the order parameters.
    fn convergence(&self) -> Option<&Convergence>;

    /// Get the name of the molecule
    fn molecule(&self) -> &str;
}

/// All supported output file formats.
#[derive(Debug, Clone, Display)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum OutputFormat {
    #[strum(serialize = "yaml")]
    YAML,
    #[strum(serialize = "csv")]
    CSV,
    #[strum(serialize = "xvg")]
    XVG,
    #[strum(serialize = "tab")]
    TAB,
    #[strum(serialize = "map")]
    MAP,
    #[strum(serialize = "convergence")]
    CONV, // convergence data file (xvg format)
}

/// Specifies whether a file with the same name existed or not
/// and whether it has been overwritten or backed up.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub(crate) enum FileStatus {
    New,
    Backup,
    Overwrite,
}

impl FileStatus {
    /// Log information about a file and what has been performed with it.
    fn info(self, format: OutputFormat, filename: &str) {
        match self {
            Self::New => colog_info!(
                "Written order parameters into {} file '{}'.",
                format,
                filename
            ),
            Self::Backup => colog_info!(
                "Backed up an already existing file '{}' and saved order parameters.",
                filename,
            ),
            Self::Overwrite => colog_warn!(
                "Overwritten an already existing file '{}' with order parameters.",
                filename,
            ),
        }
    }

    /// Log information abbout a directory and what has been performed with it.
    fn info_dir(self, dirname: &str) {
        match self {
            Self::New => colog_info!("Written ordermaps into a directory '{}'.", dirname),
            Self::Backup => colog_info!(
                "Backed up an already existing directory '{}' and saved ordermaps.",
                dirname,
            ),
            Self::Overwrite => colog_warn!(
                "Overwritten an already existing directory '{}' with ordermaps.",
                dirname,
            ),
        }
    }

    /// Log information about a file and what has been performed with it.
    /// Includes custom specification of what the file contains instead of the default 'order parameters'.
    fn info_custom(self, format: OutputFormat, filename: &str, content: &str) {
        match self {
            Self::New => log::info!(
                "Written {} into {} file '{}'.",
                content,
                format.to_string().cyan(),
                filename.to_string().cyan()
            ),
            Self::Backup => log::info!(
                "Backed up an already existing file '{}' and saved {}.",
                filename.to_string().cyan(),
                content,
            ),
            Self::Overwrite => log::warn!(
                "Overwritten an already existing file '{}' with {}.",
                filename.to_string().yellow(),
                content,
            ),
        }
    }
}

/// Trait implemented by all structures that store properties of Presenters.
pub(crate) trait PresenterProperties: Debug + Clone {
    /// Is the data for leaflets available?
    fn leaflets(&self) -> bool;
}

/// Trait implemented by all structures presenting results of the analysis.
pub(crate) trait Presenter<'a, R: OrderResults>: Debug + Clone {
    /// Structure describing properties of the presenter.
    type Properties: PresenterProperties;

    /// Create a new presenter structure.
    fn new(results: &'a R, properties: Self::Properties) -> Self;

    /// Get the format of the output file that this Presenter creates.
    fn file_format(&self) -> OutputFormat;

    /// Write the results into an open output file.
    fn write_results(&self, writer: &mut impl Write) -> Result<(), WriteError>;

    /// Write empty (missing) order parameter into the output file.
    fn write_empty_order(
        writer: &mut impl Write,
        properties: &Self::Properties,
    ) -> Result<(), WriteError>;

    /// Write empty (missing) order parameters for an entire collection.
    fn write_empty_bond_collection(
        writer: &mut impl Write,
        properties: &Self::Properties,
    ) -> Result<(), WriteError> {
        if properties.leaflets() {
            for _ in 0..3 {
                Self::write_empty_order(writer, properties)?;
            }
        } else {
            Self::write_empty_order(writer, properties)?;
        }

        Ok(())
    }

    /// Create (and potentially back up) an output file, open it and write the results into it.
    fn write(&self, filename: impl AsRef<Path>, overwrite: bool) -> Result<(), WriteError> {
        let file_status = Self::try_backup(&filename, overwrite)?;
        let mut writer = Self::create_and_open(&filename)?;
        self.write_results(&mut writer)?;
        file_status.info(
            self.file_format(),
            filename.as_ref().to_str().expect(PANIC_MESSAGE),
        );
        Ok(())
    }

    /// Create and open a file for buffered writing.
    #[inline(always)]
    fn create_and_open(filename: &impl AsRef<Path>) -> Result<BufWriter<File>, WriteError> {
        let file = File::create(filename.as_ref())
            .map_err(|_| WriteError::CouldNotCreateFile(Box::from(filename.as_ref())))?;

        Ok(BufWriter::new(file))
    }

    /// Back up an output file, if it is necessary and if it is requested.
    #[inline(always)]
    fn try_backup(filename: &impl AsRef<Path>, overwrite: bool) -> Result<FileStatus, WriteError> {
        if filename.as_ref().exists() {
            if !overwrite {
                backitup::backup(filename.as_ref())
                    .map_err(|_| WriteError::CouldNotBackupFile(Box::from(filename.as_ref())))?;

                Ok(FileStatus::Backup)
            } else {
                Ok(FileStatus::Overwrite)
            }
        } else {
            Ok(FileStatus::New)
        }
    }

    /// Write header into the output file specifying version of the library used and some other basic info.
    fn write_header(
        writer: &mut impl Write,
        structure: &str,
        trajectory: &[String],
    ) -> Result<(), WriteError> {
        if trajectory.len() == 1 {
            write_result!(writer, "# Order parameters calculated with 'gorder v{}' using a structure file '{}' and a trajectory file '{}'.\n",
            crate::GORDER_VERSION, structure, trajectory.first().expect(PANIC_MESSAGE));
        } else {
            write_result!(writer, "# Order parameters calculated with 'gorder v{}' using a structure file '{}' and trajectory files '{}'.\n",
            crate::GORDER_VERSION, structure, trajectory.join(" "));
        }

        Ok(())
    }
}

/// Single order parameter value, optionally with its estimated error.
#[derive(Debug, Clone, Copy, CopyGetters)]
pub struct Order {
    /// Value of the order parameter (mean from the analyzed frames).
    #[getset(get_copy = "pub")]
    value: f32,
    /// Estimated error for this order parameter (standard deviation of N blocks).
    #[getset(get_copy = "pub")]
    error: Option<f32>,
}

impl From<f32> for Order {
    #[inline(always)]
    fn from(value: f32) -> Self {
        Order { value, error: None }
    }
}

impl From<[f32; 2]> for Order {
    #[inline(always)]
    fn from(value: [f32; 2]) -> Self {
        Order {
            value: value[0],
            error: Some(value[1]),
        }
    }
}

impl Serialize for Order {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(error) = self.error {
            // serialize as a dictionary if `error` is present
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("mean", &self.value.round_to_4())?;
            map.serialize_entry("error", &error.round_to_4())?;
            map.end()
        } else {
            // serialize as a single float rounded to 4 decimal places
            serializer.serialize_f64(self.value.round_to_4())
        }
    }
}

// Helper trait for rounding a float to 4 decimal places
trait RoundTo4 {
    fn round_to_4(self) -> f64;
}

impl RoundTo4 for f32 {
    fn round_to_4(self) -> f64 {
        (self as f64 * 10_000.0).round() / 10_000.0
    }
}

/// Collection of (up to) 3 order parameters: for the full membrane, the upper leaflet,
/// and the lower leaflet.
#[derive(Debug, Clone, Default, Serialize, Getters)]
pub struct OrderCollection {
    /// Order parameter for the full membrane.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[getset(get = "pub")]
    total: Option<Order>,
    /// Order parameter for the upper leaflet.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[getset(get = "pub")]
    upper: Option<Order>,
    /// Order parameter for the lower leaflet.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[getset(get = "pub")]
    lower: Option<Order>,
}

impl OrderCollection {
    fn new(total: Option<Order>, upper: Option<Order>, lower: Option<Order>) -> Self {
        Self {
            total,
            upper,
            lower,
        }
    }
}

/// Collection of (up to) 3 ordermaps: for the full membrane, the upper leaflet,
/// and the lower leaflet.
#[derive(Debug, Clone, Default, Getters)]
pub struct OrderMapsCollection {
    /// Ordermap for the full membrane.
    #[getset(get = "pub")]
    total: Option<GridMapF32>,
    /// Ordermap for the upper leaflet.
    #[getset(get = "pub")]
    upper: Option<GridMapF32>,
    /// Ordermap for the lower leaflet.
    #[getset(get = "pub")]
    lower: Option<GridMapF32>,
}

impl OrderMapsCollection {
    pub(super) fn new(
        total: Option<GridMapF32>,
        upper: Option<GridMapF32>,
        lower: Option<GridMapF32>,
    ) -> Self {
        Self {
            total,
            upper,
            lower,
        }
    }
}

/// Order parameters calculated for a single bond.
#[derive(Debug, Clone, Serialize, Getters)]
pub struct BondResults {
    /// Name of the bond.
    #[serde(skip)]
    #[getset(get = "pub(super)")] // intentionally not public
    bond: BondTopology,
    /// Name of the molecule this bond belongs to.
    #[serde(skip)]
    #[getset(get = "pub")]
    molecule: String,
    /// Order parameters calculated for this bond.
    #[serde(flatten)]
    #[getset(get = "pub")]
    order: OrderCollection,
    /// Ordermaps calculated for this bond.
    #[serde(skip)]
    #[getset(get = "pub")]
    ordermaps: OrderMapsCollection,
}

impl BondResults {
    /// Atom types involved in the bond.
    pub fn atoms(&self) -> (&AtomType, &AtomType) {
        (self.bond.atom1(), self.bond.atom2())
    }
}

/// Empty struct used as a marker.
#[derive(Default, Debug, Clone)]
pub(crate) struct AAOrder {}
/// Empty struct used as a marker.
#[derive(Default, Debug, Clone)]
pub(crate) struct CGOrder {}
/// Empty struct used as a marker.
#[derive(Default, Debug, Clone)]
pub(crate) struct UAOrder {}

/// Trait implemented only by `AAOrder`, `CGOrder`, and `UAOrder` structs.
pub(crate) trait OrderType: Debug + Clone {
    /// Used to convert an order parameter to its final value depending on the analysis type.
    /// Atomistic order parameters are reported as -S_CH.
    /// Coarse grained order parameters are reported as S.
    fn convert(order: f32, error: Option<f32>) -> Order;

    /// String to use as a label for z-axis in the ordermap.
    fn zlabel() -> &'static str;

    /// String to use as a label for y-axis in xvg files.
    fn xvg_ylabel() -> &'static str;

    /// Colorbar range for ordermaps.
    fn zrange() -> (f32, f32);
}

impl OrderType for AAOrder {
    #[inline(always)]
    fn convert(order: f32, error: Option<f32>) -> Order {
        Order {
            value: -order,
            error,
        }
    }

    #[inline(always)]
    fn zlabel() -> &'static str {
        "order parameter ($-S_{CH}$)"
    }

    #[inline(always)]
    fn xvg_ylabel() -> &'static str {
        "-Sch"
    }

    #[inline(always)]
    fn zrange() -> (f32, f32) {
        (-1.0, 0.5)
    }
}

impl OrderType for CGOrder {
    #[inline(always)]
    fn convert(order: f32, error: Option<f32>) -> Order {
        Order {
            value: order,
            error,
        }
    }

    #[inline(always)]
    fn zlabel() -> &'static str {
        "order parameter ($S$)"
    }

    #[inline(always)]
    fn xvg_ylabel() -> &'static str {
        "S"
    }

    #[inline(always)]
    fn zrange() -> (f32, f32) {
        (-0.5, 1.0)
    }
}

impl OrderType for UAOrder {
    #[inline(always)]
    fn convert(order: f32, error: Option<f32>) -> Order {
        Order {
            value: -order,
            error,
        }
    }

    #[inline(always)]
    fn zlabel() -> &'static str {
        "order parameter ($-S_{CH}$)"
    }

    #[inline(always)]
    fn xvg_ylabel() -> &'static str {
        "-Sch"
    }

    #[inline(always)]
    fn zrange() -> (f32, f32) {
        (-1.0, 0.5)
    }
}

impl BondResults {
    #[inline(always)]
    fn new(
        bond: &BondTopology,
        molecule: &str,
        order: OrderCollection,
        ordermaps: OrderMapsCollection,
    ) -> Self {
        Self {
            bond: bond.clone(),
            molecule: molecule.to_owned(),
            order,
            ordermaps,
        }
    }
}

impl Serialize for AtomType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let formatted_string = format!(
            "{} {} ({})",
            self.residue_name(),
            self.atom_name(),
            self.relative_index()
        );
        serializer.serialize_str(&formatted_string)
    }
}

impl Analysis {
    /// Print basic information about the analysis for the user.
    pub(crate) fn info(&self) {
        colog_info!("Will calculate {}.", self.analysis_type().name());
        log::info!("{}", self.membrane_normal());
        if self.map().is_some() {
            colog_info!(
                "Will construct ordermaps in the {} plane.",
                self.map().as_ref().unwrap().plane().expect(PANIC_MESSAGE)
            );
        }
        if let Some(leaflets) = self.leaflets() {
            leaflets.info();
        }
    }
}

impl LeafletClassification {
    /// Print basic information about the leaflet classification.
    #[inline(always)]
    fn info(&self) {
        let ndx_files = match self {
            LeafletClassification::FromNdx(params) => {
                params.compact_display_ndx().expect(PANIC_MESSAGE)
            }
            _ => String::new(),
        };
        if let Some(normal) = self.get_membrane_normal() {
            log::info!(
                "Will classify lipids into membrane leaflets {} using the '{}' method.
Note: membrane normal for leaflet classification assumed to be oriented along the {} axis.{}",
                self.get_frequency().to_string().cyan(),
                self.to_string().cyan(),
                normal.to_string().cyan(),
                ndx_files,
            )
        } else {
            log::info!(
                "Will classify lipids into membrane leaflets {} using the '{}' method.{}",
                self.get_frequency().to_string().cyan(),
                self.to_string().cyan(),
                ndx_files,
            )
        }
    }
}

impl std::fmt::Display for Frequency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Frequency::Every(n) if n.get() == 1 => write!(f, "every analyzed frame"),
            Frequency::Every(n) if n.get() == 2 => write!(f, "every 2nd analyzed frame"),
            Frequency::Every(n) if n.get() == 3 => write!(f, "every 3rd analyzed frame"),
            Frequency::Every(n) => write!(f, "every {}th analyzed frame", n),
            Frequency::Once => write!(f, "once at the start of the analysis"),
        }
    }
}

impl SystemTopology {
    /// Print basic information about the system topology for the user.
    #[inline(always)]
    pub(crate) fn info(&self) -> std::fmt::Result {
        let mut string = String::new();

        write!(
            string,
            "Detected {} relevant molecule type(s):",
            self.molecule_types().n_molecule_types().to_string().cyan()
        )?;

        match self.molecule_types() {
            MoleculeTypes::BondBased(x) => {
                for molecule in x.iter() {
                    write!(
                        string,
                        "\n  Molecule type {}: {} order bonds, {} molecules.",
                        molecule.name().cyan(),
                        molecule
                            .order_structure()
                            .bond_types()
                            .len()
                            .to_string()
                            .cyan(),
                        molecule.order_structure().n_molecules().to_string().cyan(),
                    )?
                }
            }
            MoleculeTypes::AtomBased(x) => {
                for molecule in x.iter() {
                    write!(
                        string,
                        "\n  Molecule type {}: {} order atoms, {} molecules.",
                        molecule.name().cyan(),
                        molecule
                            .order_structure()
                            .atom_types()
                            .len()
                            .to_string()
                            .cyan(),
                        molecule.order_structure().n_molecules().to_string().cyan(),
                    )?
                }
            }
        }

        log::info!("{}", string);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_display() {
        let freq = Frequency::once();
        assert_eq!(freq.to_string(), "once at the start of the analysis");

        let freq = Frequency::every(1).unwrap();
        assert_eq!(freq.to_string(), "every analyzed frame");

        let freq = Frequency::every(2).unwrap();
        assert_eq!(freq.to_string(), "every 2nd analyzed frame");

        let freq = Frequency::every(3).unwrap();
        assert_eq!(freq.to_string(), "every 3rd analyzed frame");

        let freq = Frequency::every(10).unwrap();
        assert_eq!(freq.to_string(), "every 10th analyzed frame");
    }
}
