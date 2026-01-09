// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for writing csv files.

use crate::errors::WriteError;
use crate::presentation::aaresults::{AAAtomResults, AAMoleculeResults, AAOrderResults};
use crate::presentation::cgresults::{CGMoleculeResults, CGOrderResults};
use crate::presentation::{
    BondResults, Order, OrderCollection, OrderResults, OutputFormat, Presenter,
    PresenterProperties, PublicMoleculeResults, PublicOrderResults,
};
use std::io::Write;

use super::uaresults::{UAAtomResults, UABondResults, UAMoleculeResults, UAOrderResults};

/// Structure handling the writing of a csv output.
#[derive(Debug, Clone)]
pub(super) struct CsvPresenter<'a, R: OrderResults> {
    /// Results of the analysis.
    results: &'a R,
    /// Parameters necessary for the writing of the csv file.
    properties: CsvProperties,
}

/// Structure containing parameters necessary for the writing of the csv file.
#[derive(Debug, Clone)]
pub(crate) struct CsvProperties {
    /// Are errors calculated?
    errors: bool,
    /// Are results for individual leaflets available?
    leaflets: bool,
    /// Maximal number of hydrogen bonds per heavy atom. (Only used for AA order.)
    max_bonds: usize,
}

/// Trait implemented by all structures that can be written in csv format.
pub(crate) trait CsvWrite {
    /// Write the structure in a csv format into an output file.
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError>;
}

impl PresenterProperties for CsvProperties {
    fn leaflets(&self) -> bool {
        self.leaflets
    }
}

impl CsvProperties {
    /// Create new structure capturing properties of the csv format.
    pub(super) fn new(results: &impl OrderResults, errors: bool, leaflets: bool) -> Self {
        Self {
            errors,
            leaflets,
            max_bonds: results.max_bonds(),
        }
    }
}

impl<'a, R: OrderResults> Presenter<'a, R> for CsvPresenter<'a, R> {
    type Properties = CsvProperties;

    fn new(results: &'a R, properties: CsvProperties) -> Self {
        Self {
            results,
            properties,
        }
    }

    fn file_format(&self) -> OutputFormat {
        OutputFormat::CSV
    }

    fn write_results(&self, writer: &mut impl Write) -> Result<(), WriteError> {
        self.results.write_csv(writer, &self.properties)
    }

    fn write_empty_order(
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        if properties.errors {
            write_result!(writer, ",,")
        } else {
            write_result!(writer, ",")
        }

        Ok(())
    }
}

impl CsvWrite for Order {
    /// Write csv data for a single order parameter.
    #[inline(always)]
    fn write_csv(
        &self,
        writer: &mut impl Write,
        _properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        match self.error {
            Some(e) => write_result!(writer, ",{:.4},{:.4}", self.value, e),
            None => write_result!(writer, ",{:.4}", self.value),
        }

        Ok(())
    }
}

impl CsvWrite for OrderCollection {
    /// Write csv data for a single order collection.
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        if !properties.leaflets {
            if let Some(order) = &self.total {
                order.write_csv(writer, properties)?;
            } else {
                // the specific OrderResults type does not matter
                CsvPresenter::<AAOrderResults>::write_empty_order(writer, properties)?;
            }
        } else {
            for order in [&self.total, &self.upper, &self.lower] {
                match order {
                    Some(x) => x.write_csv(writer, properties)?,
                    None => CsvPresenter::<AAOrderResults>::write_empty_order(writer, properties)?,
                }
            }
        }

        Ok(())
    }
}

impl CsvWrite for BondResults {
    /// Write csv data for a single bond.
    #[inline(always)]
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        self.order.write_csv(writer, properties)
    }
}

/// Write CSV header for all-atom or united-atom order parameters.
fn csv_header_aa_ua(writer: &mut impl Write, properties: &CsvProperties) -> Result<(), WriteError> {
    write_result!(writer, "molecule,residue,atom,relative index");

    match (properties.leaflets, properties.errors) {
        (true, true) => write_result!(
            writer,
            ",total full membrane,total full membrane error,\
                 total upper leaflet,total upper leaflet error,\
                 total lower leaflet,total lower leaflet error"
        ),
        (true, false) => write_result!(
            writer,
            ",total full membrane,total upper leaflet,total lower leaflet"
        ),
        (false, true) => write_result!(writer, ",total,total error"),
        (false, false) => write_result!(writer, ",total"),
    };

    for i in 1..=properties.max_bonds {
        match (properties.leaflets, properties.errors) {
            (true, true) => write_result!(
                writer,
                ",hydrogen #{} full membrane,hydrogen #{} full membrane error,\
                 hydrogen #{} upper leaflet,hydrogen #{} upper leaflet error,\
                 hydrogen #{} lower leaflet,hydrogen #{} lower leaflet error",
                i,
                i,
                i,
                i,
                i,
                i
            ),
            (true, false) => write_result!(
                writer,
                ",hydrogen #{} full membrane,\
                 hydrogen #{} upper leaflet,\
                 hydrogen #{} lower leaflet",
                i,
                i,
                i
            ),
            (false, true) => write_result!(writer, ",hydrogen #{},hydrogen #{} error", i, i),
            (false, false) => write_result!(writer, ",hydrogen #{}", i),
        };
    }

    write_result!(writer, "\n");
    Ok(())
}

impl CsvWrite for AAOrderResults {
    /// Write csv data for atomistic order parameters.
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        csv_header_aa_ua(writer, properties)?;

        self.molecules()
            .try_for_each(|mol| mol.write_csv(writer, properties))?;
        Ok(())
    }
}

impl CsvWrite for AAMoleculeResults {
    /// Write csv data for a single atomistic molecule.
    #[inline]
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        for atom in self.order().values() {
            write_result!(writer, "{},", self.molecule());
            atom.write_csv(writer, properties)?;
        }

        Ok(())
    }
}

impl CsvWrite for AAAtomResults {
    /// Write csv data for a single heavy atom.
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        write_result!(
            writer,
            "{},{},{}",
            self.atom().residue_name(),
            self.atom().atom_name(),
            self.atom().relative_index()
        );

        self.order().write_csv(writer, properties)?;
        let mut bonds = self.bonds();
        for _ in 0..properties.max_bonds {
            match bonds.next() {
                Some(bond) => bond.write_csv(writer, properties)?,
                None => {
                    CsvPresenter::<AAOrderResults>::write_empty_bond_collection(writer, properties)?
                }
            }
        }

        write_result!(writer, "\n");
        Ok(())
    }
}

impl CsvWrite for CGOrderResults {
    /// Write csv data for coarse-grained order parameters.
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        write_result!(writer, "molecule,atom 1,atom 2");

        match (properties.leaflets, properties.errors) {
            (true, true) => write_result!(
                writer,
                ",full membrane,full membrane error,\
             upper leaflet,upper leaflet error,\
             lower leaflet,lower leaflet error\n"
            ),
            (true, false) => write_result!(writer, ",full membrane,upper leaflet,lower leaflet\n"),
            (false, true) => write_result!(writer, ",full membrane,full membrane error\n"),
            (false, false) => write_result!(writer, ",full membrane\n"),
        }

        self.molecules()
            .try_for_each(|mol| mol.write_csv(writer, properties))
    }
}

impl CsvWrite for CGMoleculeResults {
    /// Write csv data for a single coarse-grained molecule.
    #[inline(always)]
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        for bond in self.order().values() {
            write_result!(
                writer,
                "{},{},{}",
                bond.molecule(),
                bond.bond().atom1().atom_name(),
                bond.bond().atom2().atom_name()
            );
            bond.write_csv(writer, properties)?;
            write_result!(writer, "\n");
        }
        Ok(())
    }
}

impl CsvWrite for UAOrderResults {
    /// Write csv data for united-atom order parameters.
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        csv_header_aa_ua(writer, properties)?;

        self.molecules()
            .try_for_each(|mol| mol.write_csv(writer, properties))?;
        Ok(())
    }
}

impl CsvWrite for UAMoleculeResults {
    /// Write csv data for a single united-atom molecule.
    #[inline]
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        for atom in self.order().values() {
            write_result!(writer, "{},", self.molecule());
            atom.write_csv(writer, properties)?;
        }

        Ok(())
    }
}

impl CsvWrite for UAAtomResults {
    /// Write csv data for a single united atom.
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        write_result!(
            writer,
            "{},{},{}",
            self.atom().residue_name(),
            self.atom().atom_name(),
            self.atom().relative_index()
        );

        self.order().write_csv(writer, properties)?;
        let mut bonds = self.bonds();
        for _ in 0..properties.max_bonds {
            match bonds.next() {
                Some(bond) => bond.write_csv(writer, properties)?,
                None => {
                    CsvPresenter::<AAOrderResults>::write_empty_bond_collection(writer, properties)?
                }
            }
        }

        write_result!(writer, "\n");
        Ok(())
    }
}

impl CsvWrite for UABondResults {
    /// Write csv data for a single united-atom bond.
    #[inline(always)]
    fn write_csv(
        &self,
        writer: &mut impl Write,
        properties: &CsvProperties,
    ) -> Result<(), WriteError> {
        self.order().write_csv(writer, properties)
    }
}
