// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for writing xvg files.

use crate::errors::WriteError;
use crate::presentation::aaresults::{AAAtomResults, AAMoleculeResults, AAOrderResults};
use crate::presentation::cgresults::CGMoleculeResults;
use crate::presentation::{
    BondResults, Order, OrderCollection, OrderResults, OutputFormat, Presenter,
    PresenterProperties, PublicMoleculeResults,
};
use crate::PANIC_MESSAGE;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::uaresults::{UAAtomResults, UABondResults, UAMoleculeResults};

/// Structure handling the writing of an xvg output.
#[derive(Debug, Clone)]
pub(super) struct XvgPresenter<'a, R: OrderResults> {
    /// Results of the analysis.
    results: &'a R,
    /// Parameters necessary for the writing of the xvg file.
    properties: XvgProperties,
}

/// Structure containing parameters necessary for the writing of the xvg file.
#[derive(Debug, Clone)]
pub(crate) struct XvgProperties {
    /// Name of the input structure file.
    structure: String,
    /// Names of the input trajectory files.
    trajectory: Vec<String>,
    /// Are results for individual leaflets available?
    leaflets: bool,
}

/// Trait implemented by all structures that can be written in an xvg format.
pub(crate) trait XvgWrite {
    /// Write the structure in an xvg format into an output file.
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError>;
}

impl PresenterProperties for XvgProperties {
    fn leaflets(&self) -> bool {
        self.leaflets
    }
}

impl XvgProperties {
    pub(super) fn new(structure: &str, trajectory: &[String], leaflets: bool) -> Self {
        Self {
            structure: structure.to_owned(),
            trajectory: trajectory.to_vec(),
            leaflets,
        }
    }
}

#[inline(always)]
fn strip_extension(file_path: &Path) -> PathBuf {
    if let Some(stem) = file_path.file_stem() {
        if let Some(parent) = file_path.parent() {
            return parent.join(stem);
        }
    }
    file_path.to_path_buf()
}

impl<'a, R: OrderResults> Presenter<'a, R> for XvgPresenter<'a, R> {
    type Properties = XvgProperties;

    fn new(results: &'a R, properties: XvgProperties) -> Self {
        Self {
            results,
            properties,
        }
    }

    fn file_format(&self) -> OutputFormat {
        OutputFormat::XVG
    }

    #[allow(unused)]
    fn write_results(&self, writer: &mut impl Write) -> Result<(), WriteError> {
        panic!("FATAL GORDER ERROR | XvgPresenter::write_results | This method should never be called. {}", PANIC_MESSAGE);
    }

    fn write_empty_order(
        writer: &mut impl Write,
        _properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        write_result!(writer, " NaN ");
        Ok(())
    }

    /// Write all xvg output files.
    fn write(&self, file_pattern: impl AsRef<Path>, overwrite: bool) -> Result<(), WriteError> {
        let extension = file_pattern
            .as_ref()
            .extension()
            .and_then(|x| x.to_str())
            .map(Some)
            .unwrap_or(None);

        let path_buf = strip_extension(file_pattern.as_ref());
        let file_path = path_buf.to_str().expect(PANIC_MESSAGE);

        // all molecule names must be unique
        let names: Vec<String> = self
            .results
            .molecules()
            .map(|x| x.molecule().to_owned())
            .collect();

        for (i, mol) in self.results.molecules().enumerate() {
            let filename = match extension {
                Some(x) => format!("{}_{}.{}", file_path, names[i], x),
                None => format!("{}_{}", file_path, names[i]),
            };

            let file_status = Self::try_backup(&filename, overwrite)?;
            let mut writer = Self::create_and_open(&filename)?;
            XvgPresenter::<AAOrderResults>::write_header(
                &mut writer,
                &self.properties.structure,
                &self.properties.trajectory,
            )?;
            mol.write_xvg(&mut writer, &self.properties)?;
            file_status.info(self.file_format(), &filename);
        }

        Ok(())
    }
}

impl XvgWrite for Order {
    /// Write xvg data for a single order parameter.
    #[inline(always)]
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        _properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        write_result!(writer, "{: >8.4} ", self.value);
        Ok(())
    }
}

impl XvgWrite for OrderCollection {
    /// Write xvg data for a single order collection.
    #[inline]
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        if !properties.leaflets {
            if let Some(order) = &self.total {
                order.write_xvg(writer, properties)?;
            } else {
                // the specific OrderResults type does not matter
                XvgPresenter::<AAOrderResults>::write_empty_order(writer, properties)?;
            }
        } else {
            for order in [&self.total, &self.upper, &self.lower] {
                match order {
                    Some(x) => x.write_xvg(writer, properties)?,
                    None => XvgPresenter::<AAOrderResults>::write_empty_order(writer, properties)?,
                }
            }
        }

        Ok(())
    }
}

impl XvgWrite for BondResults {
    /// Write xvg data for a single bond.
    #[inline(always)]
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        self.order.write_xvg(writer, properties)
    }
}

impl XvgWrite for UABondResults {
    /// Write xvg data for a single united-atom bond.
    #[inline(always)]
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        self.order().write_xvg(writer, properties)
    }
}

impl XvgWrite for AAMoleculeResults {
    /// Write xvg data for a single atomistic molecule.
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        write_aa_ua_header(writer, properties, "Atomistic", self.molecule())?;

        for (i, atom) in self.order().values().enumerate() {
            write_result!(writer, "# Atom {}:\n", atom.atom().atom_name());
            write_result!(writer, "{:<4} ", i + 1);
            atom.write_xvg(writer, properties)?;
        }

        Ok(())
    }
}

impl XvgWrite for AAAtomResults {
    /// Write xvg data for a single heavy atom.
    #[inline(always)]
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        self.order().write_xvg(writer, properties)?;
        write_result!(writer, "\n");
        Ok(())
    }
}

impl XvgWrite for CGMoleculeResults {
    /// Write xvg data for a single coarse-grained molecule.
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        write_result!(
            writer,
            "@    title \"Coarse-grained order parameters for molecule type {}\"\n",
            self.molecule()
        );
        write_result!(
            writer,
            "@    xaxis label \"Bond\"\n@    yaxis label \"S\"\n"
        );

        write_result!(writer, "@    s0 legend \"Full membrane\"\n");
        if properties.leaflets {
            write_result!(writer, "@    s1 legend \"Upper leaflet\"\n");
            write_result!(writer, "@    s2 legend \"Lower leaflet\"\n");
        }

        write_result!(writer, "@TYPE xy\n");

        for (i, bond) in self.order().values().enumerate() {
            write_result!(
                writer,
                "# Bond {} - {}:\n",
                bond.bond().atom1().atom_name(),
                bond.bond().atom2().atom_name()
            );
            write_result!(writer, "{:<4} ", i + 1);
            bond.write_xvg(writer, properties)?;
            write_result!(writer, "\n");
        }

        Ok(())
    }
}

impl XvgWrite for UAMoleculeResults {
    /// Write xvg data for a single united-atom molecule.
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        write_aa_ua_header(writer, properties, "United-atom", self.molecule())?;

        for (i, atom) in self.order().values().enumerate() {
            write_result!(writer, "# Atom {}:\n", atom.atom().atom_name());
            write_result!(writer, "{:<4} ", i + 1);
            atom.write_xvg(writer, properties)?;
        }

        Ok(())
    }
}

impl XvgWrite for UAAtomResults {
    /// Write xvg data for a single united-atom.
    #[inline(always)]
    fn write_xvg(
        &self,
        writer: &mut impl Write,
        properties: &XvgProperties,
    ) -> Result<(), WriteError> {
        self.order().write_xvg(writer, properties)?;
        write_result!(writer, "\n");
        Ok(())
    }
}

fn write_aa_ua_header(
    writer: &mut impl Write,
    properties: &XvgProperties,
    order_type: &str,
    molecule_name: &str,
) -> Result<(), WriteError> {
    write_result!(
        writer,
        "@    title \"{} order parameters for molecule type {}\"\n",
        order_type,
        molecule_name,
    );
    write_result!(
        writer,
        "@    xaxis label \"Atom\"\n@    yaxis label \"-Sch\"\n"
    );

    write_result!(writer, "@    s0 legend \"Full membrane\"\n");
    if properties.leaflets {
        write_result!(writer, "@    s1 legend \"Upper leaflet\"\n");
        write_result!(writer, "@    s2 legend \"Lower leaflet\"\n");
    }

    write_result!(writer, "@TYPE xy\n");

    Ok(())
}
