// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for storing and writing the convergence analysis.

use std::io::Write;

use getset::Getters;

use super::{OrderResults, OutputFormat, Presenter, PresenterProperties};
use crate::errors::WriteError;
use crate::presentation::{OrderType, PublicMoleculeResults};
use crate::PANIC_MESSAGE;

/// Stores information about the convergence of calculations, i.e.
/// the average order parameter for this molecule collected in time.
/// Helps determine when the simulation reaches a steady state and the order parameters stabilize.
#[derive(Debug, Clone, Getters)]
pub struct Convergence {
    /// Indices of trajectory frames where order parameters were calculated.
    /// The first analyzed frame is assigned an index of 1. For instance, if the
    /// analysis begins at 200 ns, the frame at or just after 200 ns is indexed as 1.
    #[getset(get = "pub")]
    frames: Vec<usize>,

    /// Cumulative average order parameters for this molecule, calculated across the entire membrane.
    /// - Each value in the vector represents the cumulative average for a molecule up to a specific frame:
    ///   - The first element corresponds to the value from the first frame.
    ///   - The second element is the average from the first two frames.
    ///   - The third element is the average from the first three frames, and so on.
    /// - The last element is the overall average across all frames.
    #[getset(get = "pub")]
    total: Option<Vec<f32>>,

    /// Cumulative average order parameters for this molecule, calculated only for the upper leaflet.
    /// Follows the same format and logic as `total`.
    #[getset(get = "pub")]
    upper: Option<Vec<f32>>,

    /// Cumulative average order parameters for this molecule, calculated only for the lower leaflet.
    /// Follows the same format and logic as `total`.
    #[getset(get = "pub")]
    lower: Option<Vec<f32>>,
}

impl Convergence {
    pub(super) fn new(
        frames: Vec<usize>,
        total: Option<Vec<f32>>,
        upper: Option<Vec<f32>>,
        lower: Option<Vec<f32>>,
    ) -> Convergence {
        Convergence {
            frames,
            total,
            upper,
            lower,
        }
    }
}

/// Structure handling the writing of convergence data.
#[derive(Debug, Clone)]
pub(super) struct ConvPresenter<'a, R: OrderResults> {
    /// Results of the analysis.
    results: &'a R,
    /// Parameters necessary for the writing of the convergence data.
    properties: ConvProperties,
}

/// Structure containing parameters necessary for the writing of the convergence file.
#[derive(Debug, Clone)]
pub(crate) struct ConvProperties {
    /// Name of the input structure file.
    structure: String,
    /// Names of the input trajectory files.
    trajectory: Vec<String>,
    /// Are results for individual leaflets available?
    leaflets: bool,
}

/// Trait implemented by all structures for which convergence data can be written.
pub(crate) trait ConvWrite {
    /// Write the convergence data from the structureinto an output file.
    fn write_convergence(
        &self,
        writer: &mut impl Write,
        properties: &ConvProperties,
    ) -> Result<(), WriteError>;
}

impl PresenterProperties for ConvProperties {
    fn leaflets(&self) -> bool {
        self.leaflets
    }
}

impl ConvProperties {
    pub(super) fn new(structure: &str, trajectory: &[String], leaflets: bool) -> Self {
        Self {
            structure: structure.to_owned(),
            trajectory: trajectory.to_vec(),
            leaflets,
        }
    }
}

impl<'a, R: OrderResults> Presenter<'a, R> for ConvPresenter<'a, R> {
    type Properties = ConvProperties;

    fn new(results: &'a R, properties: ConvProperties) -> Self {
        Self {
            results,
            properties,
        }
    }

    fn file_format(&self) -> OutputFormat {
        OutputFormat::CONV
    }

    fn write_results(&self, writer: &mut impl Write) -> Result<(), WriteError> {
        self.results.write_convergence(writer, &self.properties)
    }

    fn write_empty_order(
        writer: &mut impl Write,
        _properties: &ConvProperties,
    ) -> Result<(), WriteError> {
        write_result!(writer, " NaN ");
        Ok(())
    }
}

impl<R: OrderResults> ConvWrite for R {
    fn write_convergence(
        &self,
        writer: &mut impl Write,
        properties: &ConvProperties,
    ) -> Result<(), WriteError> {
        ConvPresenter::<R>::write_header(writer, &properties.structure, &properties.trajectory)?;

        write_metadata(writer, R::OrderType::xvg_ylabel())?;

        let n_lines_per_mol = if properties.leaflets { 3 } else { 1 };
        write_legend(writer, self, n_lines_per_mol, properties.leaflets)?;

        write_result!(writer, "@TYPE xy\n");

        write_data(writer, self, properties.leaflets)?;

        Ok(())
    }
}

/// Write metadata for the convergence XVG file.
fn write_metadata(writer: &mut impl Write, ylabel: &str) -> Result<(), WriteError> {
    write_result!(
        writer,
        "@    title \"Convergence of average order parameters for individual molecule types\"\n",
    );
    write_result!(
        writer,
        "@    xaxis label \"Frame number\"\n@    yaxis label \"{}\"\n",
        ylabel
    );
    Ok(())
}

/// Write legend for the convergence XVG file.
fn write_legend<R: OrderResults>(
    writer: &mut impl Write,
    results: &R,
    n_lines_per_mol: usize,
    leaflets: bool,
) -> Result<(), WriteError> {
    for (i, mol) in results.molecules().enumerate() {
        let name = mol.molecule();
        let legend = if leaflets {
            format!("{} full", name)
        } else {
            name.to_owned()
        };

        write_result!(
            writer,
            "@    s{} legend \"{}\"\n",
            i * n_lines_per_mol,
            legend
        );

        if leaflets {
            write_result!(
                writer,
                "@    s{} legend \"{} upper\"\n",
                i * n_lines_per_mol + 1,
                name
            );
            write_result!(
                writer,
                "@    s{} legend \"{} lower\"\n",
                i * n_lines_per_mol + 2,
                name
            );
        }
    }
    Ok(())
}

/// Write convergence data for all molecules.
fn write_data<R: OrderResults>(
    writer: &mut impl Write,
    results: &R,
    leaflets: bool,
) -> Result<(), WriteError> {
    let mut i = 0;
    loop {
        for (m, molecule) in results.molecules().enumerate() {
            let convergence = molecule.convergence().unwrap_or_else(|| {
                panic!(
                    "FATAL GORDER ERROR | convergence::write_data | Convergence for molecule '{}' should not be None.",
                    molecule.molecule()
                )
            });

            if m == 0 {
                let frame = match convergence.frames().get(i) {
                    Some(f) => f,
                    None => return Ok(()),
                };

                write_result!(writer, "{:<4} ", *frame);
            }

            let total = get_convergence_value(convergence.total().as_ref(), i);
            write_result!(writer, "{: >8.4} ", total);

            if leaflets {
                let upper = get_convergence_value(convergence.upper().as_ref(), i);
                let lower = get_convergence_value(convergence.lower().as_ref(), i);

                write_result!(writer, "{: >8.4} {: >8.4} ", upper, lower);
            }
        }

        write_result!(writer, "\n");
        i += 1;
    }
}

fn get_convergence_value(data: Option<&Vec<f32>>, index: usize) -> f32 {
    data.and_then(|v| v.get(index).copied())
        .unwrap_or_else(|| panic!("FATAL GORDER ERROR | convergence::get_convergence_value | Value should be available. {}", PANIC_MESSAGE))
}
