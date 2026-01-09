// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! This module contains error types that can be returned by the `gorder` crate.

use std::path::Path;

use colored::{ColoredString, Colorize};
use groan_rs::errors::{AtomError, ParseNdxError, SelectError};
use thiserror::Error;

use crate::input::Frequency;

fn path_to_yellow(path: &Path) -> ColoredString {
    path.to_str().unwrap().yellow()
}

/// Errors that can occur when creating a `GridSpan` structure.
#[derive(Error, Debug)]
pub enum GridSpanError {
    #[error("{} the first coordinate for the grid span ('{}' nm) is higher than the second coordinate for the grid span ('{}' nm)", "error:".red().bold(), .0.to_string().yellow(), .1.to_string().yellow()
    )]
    Invalid(f32, f32),
}

/// Errors that can occur when constructing a geometry selection.
#[derive(Error, Debug)]
pub enum GeometryConfigError {
    #[error("{} the first value for dimension ('{}' nm) is higher than the second value for dimension ('{}' nm)", "error".red().bold(), .0.to_string().yellow(), .1.to_string().yellow())]
    InvalidDimension(f32, f32),

    #[error(
        "{} the specified radius for the geometry selection is '{}' but it must be non-negative", "error".red().bold(), .0.to_string().yellow()
    )]
    InvalidRadius(f32),

    #[error(
        "{} the first value for span ('{}' nm) is higher than the second value for span ('{}' nm)", "error".red().bold(), .0.to_string().yellow(), .1.to_string().yellow()
    )]
    InvalidSpan(f32, f32),

    #[error("{} cannot use dynamic center of simulation box as the reference position since periodic boundary conditions are ignored", "error:".red().bold())]
    InvalidBoxCenter,
}

/// Errors that can occur when creating a `Frequency` structure.
#[derive(Error, Debug)]
pub enum FrequencyError {
    #[error("{} action cannot be performed once every '{}' frames (frequency has to be at least 1)", "error:".red().bold(), "0".yellow())]
    EveryZero,
}

/// Errors that can occur when analyzing system topology.
#[derive(Error, Debug)]
pub enum TopologyError {
    #[error("{} {}", .0, 
        if matches!(.0, SelectError::GroupNotFound(_)) {
            format!("({} one of your atom selection queries uses a name for a group not defined in your system; maybe an ndx file is missing?)", "hint:".blue().bold()) 
        } else {
            String::from("")
        }
    )]
    InvalidQuery(SelectError),

    #[error("{} group '{}' is empty ({} {})", "error:".red().bold(), .group.yellow(), "hint:".blue().bold(), .hint)]
    EmptyGroup { group: String, hint: String },

    #[error("{} {} atoms are part of both '{}' (query: '{}') and '{}' (query: '{}')", "error:".red().bold(), .n_overlapping.to_string().yellow(), .name1.yellow(), .query1.yellow(), .name2.yellow(), .query2.yellow()
    )]
    AtomsOverlap {
        n_overlapping: usize,
        name1: String,
        query1: String,
        name2: String,
        query2: String,
    },

    #[error("{} molecule starting with atom index '{}' contains multiple head group atoms", "error:".red().bold(), .0.to_string().yellow()
    )]
    MultipleHeads(usize),

    #[error("{} molecule starting with atom index '{}' contains no head group atom", "error:".red().bold(), .0.to_string().yellow()
    )]
    NoHead(usize),

    #[error("{} molecule starting with atom index '{}' contains no methyl group atom", "error:".red().bold(), .0.to_string().yellow()
    )]
    NoMethyl(usize),

    #[error("{} molecule starting with atom index '{}' contains a number of methyl group atoms ('{}') not consistent with other molecules ('{}')", "error:".red().bold(), .0.to_string().yellow(), .1.to_string().yellow(), .2.to_string().yellow()
    )]
    InconsistentNumberOfMethyls(usize, usize, usize),

    #[error("{} system has undefined simulation box", "error:".red().bold())]
    UndefinedBox,

    #[error("{} the simulation box is not orthogonal ({} consider setting '{}' to {} but make sure that your lipid molecules are whole)", 
    "error:".red().bold(), "hint:".blue().bold(), "handle_pbc".bright_blue(), "false".bright_blue())]
    NotOrthogonalBox,

    #[error("{} all dimensions of the simulation box are zero", "error:".red().bold())]
    ZeroBox,

    #[error("{} no carbons for the calculation of united-atom order parameters were specified", "error:".red().bold())]
    NoUACarbons,

    #[error("{} clustering leaflet classification has been requested but only '{}' headgroup atom has been provided; need at least '{}' atoms",
    "error:".red().bold(), .0.to_string().yellow(), "2".yellow())]
    NotEnoughAtomsToCluster(usize),

    #[error("{}", .0)]
    OrderMapError(OrderMapConfigError),

    /// Used for (other) configuration errors that are detected while constructing the system topology.
    #[error("{}", .0)]
    ConfigError(ConfigError),
}

/// Errors that can occur while analyzing the trajectory.
#[derive(Error, Debug)]
pub enum AnalysisError {
    #[error("{} system has undefined simulation box ({} consider setting '{}' to {} but make sure that your lipid molecules are whole)", 
    "error:".red().bold(), "hint:".blue().bold(), "handle_pbc".bright_blue(), "false".bright_blue())]
    UndefinedBox,

    #[error("{} the simulation box is not orthogonal ({} consider setting '{}' to {} but make sure that your lipid molecules are whole)", 
    "error:".red().bold(), "hint:".blue().bold(), "handle_pbc".bright_blue(), "false".bright_blue())]
    NotOrthogonalBox,

    #[error("{} all dimensions of the simulation box are zero ({} consider setting '{}' to {} but make sure that your lipid molecules are whole)", 
    "error:".red().bold(), "hint:".blue().bold(), "handle_pbc".bright_blue(), "false".bright_blue())]
    ZeroBox,

    #[error("{} atom with atom index '{}' has an undefined position", "error:".red().bold(), .0.to_string().yellow()
    )]
    UndefinedPosition(usize),

    #[error("{} could not calculate global membrane center", "error:".red().bold())]
    InvalidGlobalMembraneCenter,

    #[error("{} could not calculate local membrane center for molecule with a head identifier index '{}'", "error:".red().bold(), .0.to_string().yellow()
    )]
    InvalidLocalMembraneCenter(usize),

    /// Used when there is an error in the manual leaflet classification using the leaflet assignment file.
    #[error("{}", .0)]
    ManualLeafletError(ManualLeafletClassificationError),

    /// Used when there is an error in the manual leaflet classification using NDX files.
    #[error("{}", .0)]
    NdxLeafletError(NdxLeafletClassificationError),

    /// Used when there is an error in the dynamic membrane normal calculation.
    #[error("{}", .0)]
    DynamicNormalError(DynamicNormalError),

    /// Used when there is an error in manual membrane normal assignment.
    #[error("{}", .0)]
    ManualNormalError(ManualNormalError),

    /// Used when an error while working with an atom occurs.
    #[error("{}", .0)]
    AtomError(AtomError),

    /// Used when there is an error in cluster-based leaflet assignment.
    #[error("{}", .0)]
    ClusterError(ClusterError),
}

/// Errors that can occur when calculating dynamic membrane normals.
#[derive(Error, Debug)]
pub enum DynamicNormalError {
    #[error("{} not enough points for dynamic local membrane normal calculation: got '{}', need at least '{}' points 
({} try increasing the '{}' in the '{}' section of your input configuration file)", 
    "error:".red().bold(), .0.to_string().yellow(), "3".yellow(),
    "hint:".blue().bold(), "radius".bright_blue(), "membrane_normal".bright_blue())]
    NotEnoughPoints(usize),

    #[error("{} could not perform Singular Value Decomposition for dynamic local membrane normal calculation", "error:".red().bold())]
    SVDFailed,
}

/// Errors that can occur when setting membrane normals manually.
#[derive(Error, Debug)]
pub enum ManualNormalError {
    #[error("{} could not get membrane normals for frame number '{}'
{} membrane normals were provided for '{}' frames which can accommodate 
at most '{}' trajectory frames at the current analysis step of '{}'",
    "error:".red().bold(),
    (.frame_index + 1).to_string().yellow(),
    "details:".yellow().bold(),
    .available_for.to_string().yellow(),
    (.available_for * .step).to_string().yellow(),
    .step.to_string().yellow())]
    FrameNotFound {
        frame_index: usize,
        step: usize,
        available_for: usize,
    },

    #[error("{} could not open the normals file '{}'", "error:".red().bold(), .0.yellow())]
    FileNotFound(String),

    #[error("{} could not understand the contents of the normals file '{}' ({})", "error:".red().bold(), .0.yellow(), .1)]
    CouldNotParse(String, serde_yaml::Error),

    #[error("{} molecule type '{}' not found in the manual normals structure", "error:".red().bold(), .0.yellow())]
    MoleculeTypeNotFound(String),

    #[error("{} no membrane normals provided for molecule type '{}'", "error:".red().bold(), .0.yellow())]
    NoNormals(String),

    #[error("{} inconsistent number of molecules specified in the normals structure: expected '{}' molecules of type '{}', got '{}' molecules in frame '{}'", 
    "error:".red().bold(), .expected.to_string().yellow(), .molecule.yellow(), .got.to_string().yellow(), .frame.to_string().yellow())]
    InconsistentNumberOfMolecules {
        expected: usize,
        molecule: String,
        got: usize,
        frame: usize,
    },

    #[error("{} molecule type '{}' specified in the normals structure not found in the system (detected molecule types are: '{}')", 
    "error:".red().bold(), .0.yellow(), .1.join(" ").yellow())]
    UnknownMoleculeType(String, Vec<String>),

    #[error("{} number of frames specified in the normals structure ('{}') is not consistent with the number of analyzed frames ('{}')",
    "error:".red().bold(), .used_frames.to_string().yellow(), .analyzed_frames.to_string().yellow())]
    UnexpectedNumberOfFrames {
        used_frames: usize,
        analyzed_frames: usize,
    },
}

/// Errors that can occur while writing the results.
#[derive(Error, Debug)]
pub enum WriteError {
    #[error("{} could not create file '{}'", "error:".red().bold(), path_to_yellow(.0))]
    CouldNotCreateFile(Box<Path>),

    #[error("{} could not create a backup for file '{}'", "error:".red().bold(), path_to_yellow(.0)
    )]
    CouldNotBackupFile(Box<Path>),

    #[error("{} could not write results in yaml format (serde_yaml error: `{}`)", "error:".red().bold(), .0.to_string()
    )]
    CouldNotWriteYaml(serde_yaml::Error),

    #[error("{} could not export analysis options in yaml format (serde_yaml error: `{}`)", "error:".red().bold(), .0.to_string()
    )]
    CouldNotExportAnalysis(serde_yaml::Error),

    #[error("{} could not write results to the output file ({})", "error:".red().bold(), .0)]
    CouldNotWriteResults(std::io::Error),

    #[error("{}", .0)]
    CouldNotWriteOrderMap(OrderMapWriteError),

    #[error("{} could not write a line to the output file '{}'", "error:".red().bold(), path_to_yellow(.0))]
    CouldNotWriteLine(Box<Path>),

    #[error("{} could not create plotting script at '{}'", "error:".red().bold(), path_to_yellow(.0))]
    CouldNotCreatePlotScript(Box<Path>),
}

/// Errors that can occur while writing the order maps.
#[derive(Error, Debug)]
pub enum OrderMapWriteError {
    #[error("{} could not create directory '{}'", "error:".red().bold(), path_to_yellow(.0))]
    CouldNotCreateDirectory(Box<Path>),

    #[error("{} could not create a backup for directory '{}'", "error:".red().bold(), path_to_yellow(.0)
    )]
    CouldNotBackupDirectory(Box<Path>),

    #[error("{} could not remove an existing directory '{}'", "error:".red().bold(), path_to_yellow(.0)
    )]
    CouldNotRemoveDirectory(Box<Path>),

    #[error("{} could not create file '{}'", "error:".red().bold(), path_to_yellow(.0))]
    CouldNotCreateFile(Box<Path>),

    #[error("{} could not write line into '{}'", "error:".red().bold(), path_to_yellow(.0))]
    CouldNotWriteLine(Box<Path>),
}

/// Errors that can occur when constructing an `Analysis` structure from the provided configuration.
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("{} could not open the configuration file '{}'", "error:".red().bold(), .0.yellow())]
    CouldNotOpenConfig(String),

    #[error("{} could not understand the contents of the configuration file '{}' ({})", "error:".red().bold(), .0.yellow(), .1
    )]
    CouldNotParseConfig(String, serde_yaml::Error),

    /// Only used in the `gorder` application.
    #[error("{} no yaml output file specified in the configuration file '{}' ({} add '{}' to your configuration file)", "error:".red().bold(), .0.yellow(), "hint:".blue().bold(), "output: output.yaml".bright_blue())]
    NoYamlOutput(String),

    // Only used in the `gorder` application.
    #[error("{} no output directory for ordermaps specified in the configuration file '{}'", "error:".red().bold(), .0.yellow())]
    NoOrdermapsOutput(String),

    #[error("{} the specified value of '{}' is invalid (must be positive)", "error:".red().bold(), "step".yellow()
    )]
    InvalidStep,

    #[error("{} the specified value of '{}' is invalid (must be positive)", "error:".red().bold(), "min_samples".yellow()
    )]
    InvalidMinSamples,

    #[error("{} the specified value of '{}' is invalid (must be positive)", "error:".red().bold(), "n_threads".yellow()
    )]
    InvalidNThreads,

    #[error("{} invalid values of '{}' and '{}' (begin is higher than end)",
            "error:".red().bold(),
            "begin".yellow(),
            "end".yellow())]
    InvalidBeginEnd,

    #[error("{}", .0)]
    InvalidOrderMap(OrderMapConfigError),

    #[error("{}", .0)]
    InvalidErrorEstimation(ErrorEstimationError),

    #[error("{}", .0)]
    InvalidGeometry(GeometryConfigError),

    #[error("{} the input structure file '{}' does not contain topology information ({} provide a `bonds` file)", "error:".red().bold(), .0.yellow(), "hint:".blue().bold())]
    NoTopology(String),

    #[error("{} cannot parse topology from the provided PDB file '{}' - non-unique atom numbers make the CONECT information ambiguous (see: https://www.wwpdb.org/documentation/file-format-content/format33/sect10.html)",
    "error:".red().bold(), .0.yellow())]
    InvalidPdbTopology(String),

    #[error("{} the provided structure file '{}' has an unknown, invalid, or unsupported format", "error:".red().bold(), .0.yellow())]
    InvalidStructureFormat(String),

    #[error("{} the provided trajectory file '{}' has an unknown, invalid, or unsupported format", "error:".red().bold(), .0.yellow())]
    InvalidTrajectoryFormat(String),

    #[error("{} the provided trajectory files '{}' and '{}' have inconsistent file format", "error:".red().bold(), .0.yellow(), .1.yellow())]
    InconsistentTrajectoryFormat(String, String),

    #[error("{} trajectory concatenation is only supported for XTC and TRR files; please provide only one trajectory file", "error:".red().bold())]
    TrajCatNotSupported,

    #[error("{} no trajectory file has been provided", "error:".red().bold())]
    NoTrajectoryFile,

    #[error("{} static global membrane normal is not used but leaflet classification requires it
({} add '{}' to the '{}' section of your input configuration file or, if analyzing a vesicle, 
 assign the lipids into leaflets using the clustering method or manually)", 
    "error:".red().bold(), "hint:".blue().bold(), "membrane_normal".bright_blue(), "leaflets".bright_blue())]
    MissingMembraneNormal,

    #[error("{} the specified radius for dynamic membrane normal calculation must be larger than 0, not '{}'
({} the recommended value for '{}' is roughly half of the membrane thickness)", 
    "error:".red().bold(), .0.yellow(),
    "hint:".blue().bold(), "radius".bright_blue())]
    InvalidDynamicNormalRadius(String),

    #[error("{} {}", "error:".red().bold(), .0)]
    DeprecationError(String),
}

/// Errors that can occur when constructing an `OrderMap` structure from the provided configuration.
#[derive(Error, Debug)]
pub enum OrderMapConfigError {
    #[error("{} the specified value of '{}' inside '{}' is invalid (must be positive)", 
            "error:".red().bold(), 
            "min_samples".yellow(), 
            "ordermap".yellow())]
    InvalidMinSamples,

    #[error("{} invalid span of '{}': minimum ('{}') is higher than maximum ('{}')",
            "error:".red().bold(),
            "ordermap".yellow(),
            .0.to_string().yellow(), .1.to_string().yellow())]
    InvalidGridSpan(f32, f32),

    #[error("{} invalid bin size of '{}': value is '{}', must be positive", 
            "error:".red().bold(), 
            "ordermap".yellow(), 
            .0.to_string().yellow())]
    InvalidBinSize(f32),

    #[error("{} invalid bin size of '{}': bin size of '{}x{}' is larger than grid span of '{}x{}'",
            "error:".red().bold(),
            "ordermap".yellow(),
            .0.0.to_string().yellow(),
            .0.1.to_string().yellow(),
            .1.0.to_string().yellow(),
            .1.1.to_string().yellow())]
    BinTooLarge((f32, f32), (f32, f32)),

    #[error("{} simulation box and periodic boundary conditions are ignored => unable to automatically set ordermap dimensions ({} set ordermap dimensions manually)",
    "error:".red().bold(), "hint:".blue().bold())]
    InvalidBoxAuto,

    #[error("{} membrane normal is not a static global dimension => unable to automatically set ordermap plane ({} set ordermap plane manually)",
    "error:".red().bold(), "hint:".blue().bold())]
    InvalidPlaneAuto,

    #[error("{} output directory specified for saving ordermaps cannot be the current directory (provided path: '{}')", "error:".red().bold(), .0.yellow())]
    InvalidOutputDirectory(String),
}

/// Errors that can occur when estimating the error of the calculation.
#[derive(Error, Debug)]
pub enum ErrorEstimationError {
    #[error("{} number of blocks for error estimation must be at least 2, not '{}'",
    "error:".red().bold(), .0.to_string().yellow(),
    )]
    NotEnoughBlocks(usize),

    #[error("{} read '{}' trajectory frame(s) which is fewer than the number of blocks ('{}')",
    "error:".red().bold(), .0.to_string().yellow(), .1.to_string().yellow())]
    NotEnoughData(usize, usize),
}

/// Errors that can occur when reading bonds from an external topology file.
#[derive(Error, Debug)]
pub enum BondsError {
    #[error("{} could not open the bonds file '{}'", "error:".red().bold(), .0.yellow())]
    FileNotFound(String),

    #[error("{} could not read line in the bonds file '{}'", "error:".red().bold(), .0.yellow())]
    CouldNotReadLine(String),

    #[error("{} could read '{}' as an atom serial number", "error:".red().bold(), .0.yellow())]
    CouldNotParse(String),

    #[error("{} atom with serial number '{}' claims to be bonded to itself which does not make sense", "error:".red().bold(), .0.to_string().yellow())]
    SelfBonding(usize),

    #[error("{} atom with serial number '{}' does not exist (the system only contains '{}' atoms)", "error:".red().bold(), .0.to_string().yellow(), .1.to_string().yellow())]
    AtomNotFound(usize, usize),
}

/// Errors that can occur when working manual leaflet assignment using NDX files.
#[derive(Error, Debug)]
pub enum NdxLeafletClassificationError {
    #[error("{}", .0)]
    CouldNotParse(ParseNdxError),

    #[error("{} could not get ndx file for frame number '{}' [expected index of the ndx file: '{}']
(total number of specified ndx files is '{}'; maybe the assignment frequency is incorrect?)", 
    "error:".red().bold(), (.0 + 1).to_string().yellow(), .1.to_string().yellow(), .2.to_string().yellow())]
    FrameNotFound(usize, usize, usize),

    #[error("{} group name '{}' specified in an ndx file '{}' is invalid and cannot be used ({} following characters are not allowed in group names: \'\"&|!@()<>=)",
"error:".red().bold(), .0.yellow(), .1.yellow(), "hint:".blue().bold())]
    InvalidName(String, String),

    #[error("{} group '{}' is defined multiple times in an ndx file '{}'", "error:".red().bold(), .0.yellow(), .1.yellow())]
    DuplicateName(String, String),

    #[error("{} group '{}' for selecting {} molecules was not found in the ndx file '{}'", 
    "error:".red().bold(), .0.yellow(), .1.yellow(), .2.yellow())]
    GroupNotFound(String, String, String),

    #[error("{} could not find leaflet assignment for molecule index '{}' (head index '{}')
({} head identifier index '{}' is missing from both specified ndx groups)",
    "error:".red().bold(), .0.to_string().yellow(), .1.to_string().yellow(),
    "hint:".blue().bold(), .1.to_string().yellow())]
    AssignmentNotFound(usize, usize),

    #[error("{} number of ndx files provided ('{}') is not consistent with the number of analyzed frames ('{}')
(leaflet assignment was supposed to be performed {}, therefore there should be exactly '{}' ndx file(s) provided)",
    "error:".red().bold(), .ndx_files.to_string().yellow(), .analyzed_frames.to_string().yellow(), 
    .frequency.to_string().yellow(), .expected_ndx_files.to_string().yellow())]
    UnexpectedNumberOfNdxFiles {
        ndx_files: usize,
        analyzed_frames: usize,
        frequency: Frequency,
        expected_ndx_files: usize,
    },
}

/// Errors that can occur when working with manual leaflet assignment from a leaflet assignment file.
#[derive(Error, Debug)]
pub enum ManualLeafletClassificationError {
    #[error("{} could not open the leaflet assignment file '{}'", "error:".red().bold(), .0.yellow())]
    FileNotFound(String),

    #[error("{} could not understand the contents of the leaflet assignment file '{}' ({})", "error:".red().bold(), .0.yellow(), .1)]
    CouldNotParse(String, serde_yaml::Error),

    #[error("{} molecule type '{}' not found in the leaflet assignment structure", "error:".red().bold(), .0.yellow())]
    MoleculeTypeNotFound(String),

    #[error("{} could not get leaflet assignment for frame number '{}' [expected index in the leaflet assignment structure: '{}']
(total number of frames in the leaflet assignment structure is '{}'; maybe the assignment frequency is incorrect?)", 
"error:".red().bold(), (.0 + 1).to_string().yellow(), .1.to_string().yellow(), .2.to_string().yellow())]
    FrameNotFound(usize, usize, usize),

    #[error("{} inconsistent number of molecules specified in the leaflet assignment: expected '{}' molecules of type '{}', got '{}' molecules in assignment frame '{}'", 
    "error:".red().bold(), .expected.to_string().yellow(), .molecule.yellow(), .got.to_string().yellow(), .frame.to_string().yellow())]
    InconsistentNumberOfMolecules {
        expected: usize,
        molecule: String,
        got: usize,
        frame: usize,
    },

    #[error("{} no leaflet assignment data provided for molecule type '{}'", "error:".red().bold(), .0.yellow())]
    EmptyAssignment(String),

    #[error("{} number of frames specified in the leaflet assignment structure ('{}') is not consistent with the number of analyzed frames ('{}')
(leaflet assignment was supposed to be performed {}, therefore there should be exactly '{}' frame(s) specified in the leaflet assignment structure)",
"error:".red().bold(), .assignment_frames.to_string().yellow(), .analyzed_frames.to_string().yellow(), 
.frequency.to_string().yellow(), .expected_assignment_frames.to_string().yellow())]
    UnexpectedNumberOfFrames {
        assignment_frames: usize,
        analyzed_frames: usize,
        frequency: Frequency,
        expected_assignment_frames: usize,
    },

    #[error("{} molecule type '{}' specified in the leaflet assignment structure not found in the system (detected molecule types are: '{}')", 
    "error:".red().bold(), .0.yellow(), .1.join(" ").yellow())]
    UnknownMoleculeType(String, Vec<String>),
}

/// Errors that can occur when assigning lipids into leaflets using a clustering method.
#[derive(Error, Debug)]
pub enum ClusterError {
    #[error("{} clustering leaflet classification failed
{} when comparing current frame to previous frame, the previously identified leaflets show >{} lipid composition change
{} this may be caused by either of several issues:
  - leaflets identified incorrectly => consider manual leaflet assignment,
  - too rapid flip-flop => increase classification frequency or reduce the number of threads used,
  - frames too far apart => increase classification frequency or reduce the number of threads used",
"error:".red().bold(), "details:".yellow().bold(), format!("{}%", .0).yellow(), "hint:".blue().bold())]
    CouldNotMatchLeaflets(u8),

    #[error("{} could not obtain consistent leaflet assignment using the clustering method for the first frame; assign the lipids to leaflets manually", "error:".red().bold())]
    SloppyFirstFrameFail,
}
