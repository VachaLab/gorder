// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Structures and methods for writing tab files.

use crate::errors::WriteError;
use crate::presentation::aaresults::{AAAtomResults, AAMoleculeResults, AAOrderResults};
use crate::presentation::cgresults::{CGMoleculeResults, CGOrderResults};
use crate::presentation::uaresults::UAOrderResults;
use crate::presentation::{
    BondResults, MoleculeResults, Order, OrderCollection, OrderResults, OutputFormat, Presenter,
    PresenterProperties, PublicMoleculeResults,
};
use crate::PANIC_MESSAGE;
use hashbrown::HashMap;
use std::io::Write;

use super::uaresults::{UAAtomResults, UABondResults, UAMoleculeResults};
use super::PublicOrderResults;

/// Structure handling the writing of a table output.
#[derive(Debug, Clone)]
pub(super) struct TabPresenter<'a, R: OrderResults> {
    /// Results of the analysis.
    results: &'a R,
    /// Parameters necessary for the writing of the table file.
    properties: TabProperties,
}

/// Structure containing parameters necessary for the writing of the table file.
#[derive(Debug, Clone)]
pub(crate) struct TabProperties {
    /// Name of the input structure file.
    structure: String,
    /// Names of the input trajectory files.
    trajectory: Vec<String>,
    /// Are errors calculated?
    errors: bool,
    /// Are results for individual leaflets available?
    leaflets: bool,
    /// Maximal number of hydrogen bonds per heavy atom for each molecule. (Only used for AA order.)
    max_bonds: HashMap<String, usize>,
}

/// Trait implemented by all structures that can be written in table format.
pub(crate) trait TabWrite {
    /// Write the structure in a table format into an output file.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError>;
}

impl PresenterProperties for TabProperties {
    fn leaflets(&self) -> bool {
        self.leaflets
    }
}

impl TabProperties {
    /// Create new structure capturing properties of the table format.
    pub(super) fn new(
        results: &impl OrderResults,
        structure: &str,
        trajectory: &[String],
        errors: bool,
        leaflets: bool,
    ) -> Self {
        // individual value of `max_bonds` for each molecule
        let max_bonds = results
            .molecules()
            .map(|x| (x.molecule().to_owned(), x.max_bonds()))
            .collect::<HashMap<String, usize>>();

        Self {
            structure: structure.to_owned(),
            trajectory: trajectory.to_vec(),
            errors,
            leaflets,
            max_bonds,
        }
    }

    /// Returns the maximal number of bonds (with hydrogens) of any heavy atom in a specific molecule.
    /// Panics if the molecule does not exist!
    fn max_bonds_for_molecule(&self, molecule: &str) -> usize {
        *self.max_bonds.get(molecule).unwrap_or_else(|| panic!(
            "FATAL GORDER ERROR | TabProperties::max_bonds_for_molecule | Could not find `max_bonds` for molecule '{}'. {}",
            molecule, PANIC_MESSAGE
        ))
    }
}

impl<'a, R: OrderResults> Presenter<'a, R> for TabPresenter<'a, R> {
    type Properties = TabProperties;

    fn new(results: &'a R, properties: TabProperties) -> Self {
        Self {
            results,
            properties,
        }
    }

    fn file_format(&self) -> OutputFormat {
        OutputFormat::TAB
    }

    fn write_results(&self, writer: &mut impl Write) -> Result<(), WriteError> {
        self.results.write_tab(writer, &self.properties)
    }

    fn write_empty_order(
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        if properties.errors {
            write_result!(writer, " {:17} ", "");
        } else {
            write_result!(writer, " {:8} ", "");
        }

        Ok(())
    }
}

impl TabWrite for Order {
    /// Write table data for a single order parameter.
    #[inline(always)]
    fn write_tab(
        &self,
        writer: &mut impl Write,
        _properties: &TabProperties,
    ) -> Result<(), WriteError> {
        match (self.value.is_nan(), self.error) {
            (false, Some(e)) => write_result!(writer, " {: >7.4} ± {: ^7.4} ", self.value, e),
            (_, None) => write_result!(writer, " {: ^8.4} ", self.value),
            (true, Some(_)) => write_result!(writer, " {: ^17.4} ", f32::NAN),
        }

        Ok(())
    }
}

impl TabWrite for OrderCollection {
    /// Write tab data for a single order collection.
    #[inline]
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        if !properties.leaflets {
            if let Some(order) = &self.total {
                order.write_tab(writer, properties)?;
            } else {
                // the specific OrderResults type does not matter
                TabPresenter::<AAOrderResults>::write_empty_order(writer, properties)?;
            }
        } else {
            for order in [&self.total, &self.upper, &self.lower] {
                match order {
                    Some(x) => x.write_tab(writer, properties)?,
                    None => TabPresenter::<AAOrderResults>::write_empty_order(writer, properties)?,
                }
            }
        }

        Ok(())
    }
}

impl TabWrite for BondResults {
    /// Write tab data for a single bond.
    #[inline(always)]
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        self.order.write_tab(writer, properties)
    }
}

impl TabWrite for UABondResults {
    /// Write tab data for a single united-atom bond.
    #[inline(always)]
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        self.order().write_tab(writer, properties)
    }
}

impl TabWrite for AAOrderResults {
    /// Write table data for atomistic order parameters.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        TabPresenter::<AAOrderResults>::write_header(
            writer,
            &properties.structure,
            &properties.trajectory,
        )?;

        self.molecules()
            .try_for_each(|mol| mol.write_tab(writer, properties))?;

        write_aa_ua_full_header(writer, properties)?;
        self.average_order().write_tab(writer, properties)?;
        write_result!(writer, "|\n");
        Ok(())
    }
}

impl TabWrite for AAMoleculeResults {
    /// Write table data for a single atomistic molecule.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        write_aa_ua_molecule_header(writer, properties, self.molecule())?;

        self.order()
            .values()
            .try_for_each(|atom| atom.write_tab(writer, properties))?;

        write_result!(writer, "AVERAGE  ");
        self.average_order().write_tab(writer, properties)?;
        write_result!(writer, "|\n");

        Ok(())
    }
}

impl TabWrite for AAAtomResults {
    /// Write tab data for a single heavy atom.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        write_result!(writer, "{:<8} ", self.atom().atom_name());
        self.order().write_tab(writer, properties)?;
        write_result!(writer, "|");

        let max_bonds = properties.max_bonds_for_molecule(self.molecule());
        let mut bonds = self.bonds();
        for _ in 0..max_bonds {
            match bonds.next() {
                Some(bond) => bond.write_tab(writer, properties)?,
                None => {
                    TabPresenter::<AAOrderResults>::write_empty_bond_collection(writer, properties)?
                }
            }
            write_result!(writer, "|");
        }
        write_result!(writer, "\n");

        Ok(())
    }
}

impl TabWrite for CGOrderResults {
    /// Write table data for coarse-grained order parameters.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        TabPresenter::<CGOrderResults>::write_header(
            writer,
            &properties.structure,
            &properties.trajectory,
        )?;

        self.molecules()
            .try_for_each(|mol| mol.write_tab(writer, properties))?;

        write_result!(writer, "\nAll molecule types\n");
        write_cg_molecule_header(writer, properties)?;
        write_result!(writer, "AVERAGE         ");
        self.average_order().write_tab(writer, properties)?;
        write_result!(writer, "|\n");
        Ok(())
    }
}

impl TabWrite for CGMoleculeResults {
    /// Write table data for a single coarse-grained molecule.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        write_result!(writer, "\nMolecule type {}\n", self.molecule());

        write_cg_molecule_header(writer, properties)?;

        for bond in self.order().values() {
            let name = format!(
                "{} - {}",
                bond.bond.atom1().atom_name(),
                bond.bond.atom2().atom_name(),
            );
            write_result!(writer, "{:<16}", name);
            bond.write_tab(writer, properties)?;
            write_result!(writer, "|\n");
        }

        write_result!(writer, "AVERAGE         ");
        self.average_order().write_tab(writer, properties)?;
        write_result!(writer, "|\n");

        Ok(())
    }
}

impl TabWrite for UAOrderResults {
    /// Write table data for atomistic order parameters.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        TabPresenter::<UAOrderResults>::write_header(
            writer,
            &properties.structure,
            &properties.trajectory,
        )?;

        self.molecules()
            .try_for_each(|mol| mol.write_tab(writer, properties))?;

        write_aa_ua_full_header(writer, properties)?;
        self.average_order().write_tab(writer, properties)?;
        write_result!(writer, "|\n");
        Ok(())
    }
}

impl TabWrite for UAMoleculeResults {
    /// Write table data for a single united-atom molecule.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        write_aa_ua_molecule_header(writer, properties, self.molecule())?;

        self.order()
            .values()
            .try_for_each(|atom| atom.write_tab(writer, properties))?;

        write_result!(writer, "AVERAGE  ");
        self.average_order().write_tab(writer, properties)?;
        write_result!(writer, "|\n");

        Ok(())
    }
}

impl TabWrite for UAAtomResults {
    /// Write tab data for a single united-atom.
    fn write_tab(
        &self,
        writer: &mut impl Write,
        properties: &TabProperties,
    ) -> Result<(), WriteError> {
        write_result!(writer, "{:<8} ", self.atom().atom_name());
        self.order().write_tab(writer, properties)?;
        write_result!(writer, "|");

        let max_bonds = properties.max_bonds_for_molecule(self.molecule());
        let mut bonds = self.bonds();
        for _ in 0..max_bonds {
            match bonds.next() {
                Some(bond) => bond.write_tab(writer, properties)?,
                None => {
                    TabPresenter::<AAOrderResults>::write_empty_bond_collection(writer, properties)?
                }
            }
            write_result!(writer, "|");
        }
        write_result!(writer, "\n");

        Ok(())
    }
}

fn write_cg_molecule_header(
    writer: &mut impl Write,
    properties: &TabProperties,
) -> Result<(), WriteError> {
    match (properties.leaflets, properties.errors) {
        (true, true) => {
            write_result!(
                writer,
                "                        FULL              UPPER              LOWER       |\n"
            )
        }
        (true, false) => {
            write_result!(writer, "                   FULL     UPPER     LOWER   |\n")
        }
        (false, true) => write_result!(writer, "                        FULL       |\n"),
        (false, false) => write_result!(writer, "                   FULL   |\n"),
    }

    Ok(())
}

fn write_aa_ua_molecule_header(
    writer: &mut impl Write,
    properties: &TabProperties,
    molecule_name: &str,
) -> Result<(), WriteError> {
    write_result!(writer, "\nMolecule type {}\n", molecule_name);
    write_result!(writer, "{:9}", " ");
    let width = match (properties.leaflets, properties.errors) {
        (true, true) => 55,
        (true, false) => 28,
        (false, true) => 17,
        (false, false) => 8,
    };

    let max_bonds = properties.max_bonds_for_molecule(molecule_name);

    write_result!(writer, " {: ^width$} |", "TOTAL", width = width);
    for i in 1..=max_bonds {
        let hydrogen = if properties.leaflets || properties.errors {
            format!("HYDROGEN #{}", i)
        } else {
            format!("H #{}", i)
        };

        write_result!(writer, " {: ^width$} |", hydrogen, width = width);
    }
    write_result!(writer, "\n");

    if properties.leaflets {
        write_result!(writer, "         ");
        for _ in 0..=max_bonds {
            if properties.errors {
                write_result!(
                    writer,
                    "        FULL              UPPER              LOWER       |"
                );
            } else {
                write_result!(writer, "   FULL     UPPER     LOWER   |");
            }
        }
        write_result!(writer, "\n");
    }

    Ok(())
}

fn write_aa_ua_full_header(
    writer: &mut impl Write,
    properties: &TabProperties,
) -> Result<(), WriteError> {
    write_result!(writer, "\nAll molecule types\n");
    write_result!(writer, "         ");
    if properties.leaflets {
        if properties.errors {
            write_result!(
                writer,
                "        FULL              UPPER              LOWER       |\n"
            );
        } else {
            write_result!(writer, "   FULL     UPPER     LOWER   |\n");
        }
    } else if properties.errors {
        write_result!(writer, " {: ^17} |\n", "TOTAL");
    } else {
        write_result!(writer, " {: ^8} |\n", "TOTAL");
    }
    write_result!(writer, "AVERAGE  ");
    Ok(())
}
