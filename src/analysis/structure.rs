// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Handles reading the input structure and topology and setting up the system.

use crate::{
    errors::{BondsError, ConfigError},
    input::{
        Analysis, AnalysisType, GeomReference, Geometry, LeafletClassification, MembraneNormal,
    },
    PANIC_MESSAGE,
};
use colored::Colorize;
use groan_rs::{
    errors::{ElementError, ParsePdbConnectivityError},
    files::FileType,
    prelude::Elements,
    system::System,
};
use hashbrown::HashSet;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

/// Read the input structure and topology. Handles box validation for the structure file.
pub(super) fn read_structure_and_topology(
    analysis: &Analysis,
) -> Result<System, Box<dyn std::error::Error + Send + Sync>> {
    let file_type = FileType::from_name(analysis.structure());
    let mut system = System::from_file_with_format(analysis.structure(), file_type)?;

    if analysis.handle_pbc() {
        // check simulation box
        super::common::check_box(&system)?;
    } else {
        // log warning in case handle_pbc is false
        log::warn!("Periodic boundary conditions ignored. Lipid molecules must be made whole!")
    }

    if let Some(bonds_file) = analysis.bonds() {
        // if `bonds` file is provided, read it no matter the input file type
        colog_info!("Read molecular structure from '{}'.", analysis.structure());
        read_bonds(&mut system, bonds_file)?;
        colog_info!("Read topology from bonds file '{}'.", bonds_file);

        if file_type != FileType::TPR {
            maybe_guess_elements(analysis, &mut system)?;
        }
        Ok(system)
    } else if file_type == FileType::PDB {
        // attempt to read bonds from a PDB file
        match system.add_bonds_from_pdb(analysis.structure()) {
            Ok(_) => (),
            Err(ParsePdbConnectivityError::NoBondsWarning(_)) => {
                return Err(Box::from(ConfigError::NoTopology(
                    analysis.structure().to_owned(),
                )))
            }
            Err(ParsePdbConnectivityError::DuplicateAtomNumbers) => {
                return Err(Box::from(ConfigError::InvalidPdbTopology(
                    analysis.structure().to_owned(),
                )))
            }
            Err(e) => return Err(Box::from(e)),
        }

        colog_info!(
            "Read molecular structure and topology from '{}'.",
            analysis.structure()
        );

        maybe_guess_elements(analysis, &mut system)?;
        Ok(system)
    } else if file_type == FileType::TPR {
        // no further checks required for a TPR file
        colog_info!(
            "Read molecular structure and topology from '{}'.",
            analysis.structure()
        );
        Ok(system)
    } else {
        // not a TPR file, not a PDB file, and no `bonds` file? => error
        return Err(Box::from(ConfigError::NoTopology(
            analysis.structure().to_owned(),
        )));
    }
}

/// Read bonds into a system from an external `bonds_file`. Removes all previously set bonds.
fn read_bonds(system: &mut System, bonds_file: &str) -> Result<(), BondsError> {
    system.clear_bonds();

    // read the bonds
    let bonds = parse_bonds_file(system.get_n_atoms(), bonds_file)?;
    // defensive check: the number of bonds must correspond to the number of atoms
    assert_eq!(
        system.get_n_atoms(),
        bonds.len(),
        "FATAL GORDER ERROR | structure::read_bonds | Unmatching number of atoms. {}",
        PANIC_MESSAGE
    );

    // add the bonds into system
    for (bonded, atom) in bonds.iter().zip(system.atoms_iter_mut()) {
        unsafe { atom.set_bonded(bonded.into_iter().cloned().collect()) }
    }

    Ok(())
}

/// Parse bonds from the input file. Returns a hashset of bonded atoms for each atom of the system.
///
/// ## Notes
/// - If a serial number could not be read, `CouldNotParse` error is returned.
/// - If a serial number is too high (does not correspond to any atom), `AtomNotFound` error is returned.
/// - If a bond is between the same atom, `SelfBonding` error is returned.
/// - Comments can be marked by `#`. Comments are not parsed.
/// - Empty lines are ignored. Duplicate bonds are ignored.
/// - Bond between atom 13 a 15 can be specified either as `13 15\n` or as `15 13\n` or by both variants in the same file.
fn parse_bonds_file(n_atoms: usize, bonds_file: &str) -> Result<Vec<HashSet<usize>>, BondsError> {
    let mut bonds = vec![HashSet::new(); n_atoms];

    let file =
        File::open(bonds_file).map_err(|_| BondsError::FileNotFound(bonds_file.to_owned()))?;
    let buffer = BufReader::new(file);

    for line in buffer.lines() {
        let line = line.map_err(|_| BondsError::CouldNotReadLine(bonds_file.to_owned()))?;

        let split = decomment(&line).split_whitespace().collect::<Vec<&str>>();
        if split.len() < 2 {
            continue;
        }

        let target_atom: usize = parse_string_to_atom_number(split.first().expect(PANIC_MESSAGE))?;
        if target_atom > n_atoms {
            return Err(BondsError::AtomNotFound(target_atom, n_atoms));
        }

        for bonded in split.into_iter().skip(1) {
            let atom = parse_string_to_atom_number(bonded)?;

            if atom == target_atom {
                return Err(BondsError::SelfBonding(atom));
            }

            if atom > n_atoms {
                return Err(BondsError::AtomNotFound(atom, n_atoms));
            }

            // set the bonded atom for the target atom
            bonds
                .get_mut(target_atom - 1)
                .expect(PANIC_MESSAGE)
                .insert(atom - 1);

            // set the bonded atom for the other atom
            bonds
                .get_mut(atom - 1)
                .expect(PANIC_MESSAGE)
                .insert(target_atom - 1);
        }
    }

    Ok(bonds)
}

/// Parses string to an atom number returning the proper error if the parsing fails.
fn parse_string_to_atom_number(string: &str) -> Result<usize, BondsError> {
    string
        .parse()
        .map_err(|_| BondsError::CouldNotParse(string.to_owned()))
}

/// Decomments a string.
fn decomment(input: &str) -> &str {
    if let Some(pos) = input.find('#') {
        &input[..pos]
    } else {
        input
    }
}

/// Check whether a query contains any of the supported element keywords.
// We should ideally compare the parsed `Selection` tree instead of the raw query.
// If there is e.g. a group named `element`, this will lead to false positives.
// But false positives are not really an issue in this case, so let's go with the simpler option.
fn has_element(query: &str) -> bool {
    query.contains("element") || query.contains("elname") || query.contains("elsymbol")
}

/// Check whether target geometry reference uses an element keyword.
fn reference_has_element(reference: &GeomReference) -> bool {
    match reference {
        GeomReference::Selection(x) => has_element(x),
        GeomReference::Center | GeomReference::Point(_) => false,
    }
}

/// Check whether it is necessary to guess elements.
fn should_guess_elements(analysis: &Analysis) -> bool {
    let guess = match analysis.analysis_type() {
        AnalysisType::AAOrder {
            heavy_atoms: x,
            hydrogens: y,
        } => has_element(x) || has_element(y),
        AnalysisType::CGOrder { beads: x } => has_element(x),
        AnalysisType::UAOrder {
            saturated: x,
            unsaturated: y,
            ignore: z,
        } => {
            x.as_ref().map(|a| has_element(a)).unwrap_or(false)
                || y.as_ref().map(|a| has_element(a)).unwrap_or(false)
                || z.as_ref().map(|a| has_element(a)).unwrap_or(false)
        }
    } || match analysis.leaflets() {
        None => false,
        Some(LeafletClassification::Global(x)) => {
            has_element(x.heads()) || has_element(x.membrane())
        }
        Some(LeafletClassification::Individual(x)) => {
            has_element(x.heads()) || has_element(x.methyls())
        }
        Some(LeafletClassification::Local(x)) => {
            has_element(x.heads()) || has_element(x.membrane())
        }
        Some(LeafletClassification::FromNdx(x)) => has_element(x.heads()),
        Some(LeafletClassification::FromFile(_)) | Some(LeafletClassification::FromMap(_)) => false,
        Some(LeafletClassification::Clustering(x)) => has_element(x.heads()),
        Some(LeafletClassification::SphericalClustering(x)) => has_element(x.heads()),
    } || match analysis.geometry() {
        Some(Geometry::Cuboid(x)) => reference_has_element(x.reference()),
        Some(Geometry::Cylinder(x)) => reference_has_element(x.reference()),
        Some(Geometry::Sphere(x)) => reference_has_element(x.reference()),
        None => false,
    } || match analysis.membrane_normal() {
        MembraneNormal::Static(_) | MembraneNormal::FromFile(_) | MembraneNormal::FromMap(_) => {
            false
        }
        MembraneNormal::Dynamic(dynamic) => has_element(dynamic.heads()),
    };

    guess
}

/// Guess elements if this is requested in an input query.
#[allow(clippy::format_in_format_args)]
fn maybe_guess_elements(analysis: &Analysis, system: &mut System) -> Result<(), ElementError> {
    if should_guess_elements(analysis) {
        match system.guess_elements(Elements::default()) {
            Ok(_) => {
                log::info!("Assigned elements to atoms without warnings.");
                Ok(())
            }
            Err(ElementError::ElementGuessWarning(e)) => {
                log::warn!(
                    "When guessing elements, following concerns have been raised:\n{}\n{}",
                    e,
                    format!(
                        "({} if this is an issue, provide a TPR file instead or use atom selection queries without 'element')",
                        "hint:".blue().bold()
                    )
                );
                Ok(())
            }
            Err(e) => Err(e),
        }
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::input::Axis;

    use super::*;

    #[test]
    fn parse_bonds_simple() {
        let bonds = parse_bonds_file(20, "tests/files/bonds/simple.bnd").unwrap();

        assert_eq!(bonds[0], [1].into());
        assert_eq!(bonds[1], [0, 2, 3, 4, 6, 18].into());
        assert_eq!(bonds[2], [1, 6, 13].into());
        assert_eq!(bonds[3], [1].into());
        assert_eq!(bonds[4], [1, 8].into());
        assert_eq!(bonds[5], [7, 8, 10].into());
        assert_eq!(bonds[6], [1, 2].into());
        assert_eq!(bonds[7], [5].into());
        assert_eq!(bonds[8], [4, 5, 10, 14, 19, 18, 12, 13].into());
        assert_eq!(bonds[9], [].into());
        assert_eq!(bonds[10], [5, 8].into());
        assert_eq!(bonds[11], [12, 14].into());
        assert_eq!(bonds[12], [8, 11, 13].into());
        assert_eq!(bonds[13], [2, 8, 12].into());
        assert_eq!(bonds[14], [8, 11, 16, 17].into());
        assert_eq!(bonds[15], [].into());
        assert_eq!(bonds[16], [14].into());
        assert_eq!(bonds[17], [14].into());
        assert_eq!(bonds[18], [1, 8, 19].into());
        assert_eq!(bonds[19], [8, 18].into());
    }

    #[test]
    fn parse_bonds_complex() {
        let bonds = parse_bonds_file(20, "tests/files/bonds/complex.bnd").unwrap();

        assert_eq!(bonds[0], [1].into());
        assert_eq!(bonds[1], [0, 2, 3, 4, 6, 18].into());
        assert_eq!(bonds[2], [1, 6, 13].into());
        assert_eq!(bonds[3], [1].into());
        assert_eq!(bonds[4], [1, 8].into());
        assert_eq!(bonds[5], [7, 8, 10].into());
        assert_eq!(bonds[6], [1, 2].into());
        assert_eq!(bonds[7], [5].into());
        assert_eq!(bonds[8], [4, 5, 10, 14, 19, 18, 12, 13].into());
        assert_eq!(bonds[9], [].into());
        assert_eq!(bonds[10], [5, 8].into());
        assert_eq!(bonds[11], [12, 14].into());
        assert_eq!(bonds[12], [8, 11, 13].into());
        assert_eq!(bonds[13], [2, 8, 12].into());
        assert_eq!(bonds[14], [8, 11, 16, 17].into());
        assert_eq!(bonds[15], [].into());
        assert_eq!(bonds[16], [14].into());
        assert_eq!(bonds[17], [14].into());
        assert_eq!(bonds[18], [1, 8, 19].into());
        assert_eq!(bonds[19], [8, 18].into());
    }

    #[test]
    fn parse_bonds_fail_nonexistent_file() {
        match parse_bonds_file(20, "nonexistent/file.bnd") {
            Ok(_) => panic!("Function should have failed."),
            Err(BondsError::FileNotFound(x)) => assert_eq!(x, "nonexistent/file.bnd"),
            Err(e) => panic!("Unexpected error returned: `{}`", e),
        }
    }

    #[test]
    fn parse_bonds_fail_could_not_parse_serial_1() {
        match parse_bonds_file(20, "tests/files/bonds/could_not_parse_1.bnd") {
            Ok(_) => panic!("Function should have failed."),
            Err(BondsError::CouldNotParse(x)) => assert_eq!(x, "BOND"),
            Err(e) => panic!("Unexpected error returned: `{}`", e),
        }
    }

    #[test]
    fn parse_bonds_fail_could_not_parse_serial_2() {
        match parse_bonds_file(20, "tests/files/bonds/could_not_parse_2.bnd") {
            Ok(_) => panic!("Function should have failed."),
            Err(BondsError::CouldNotParse(x)) => assert_eq!(x, "2O"),
            Err(e) => panic!("Unexpected error returned: `{}`", e),
        }
    }

    #[test]
    fn parse_bonds_fail_atom_not_found_1() {
        match parse_bonds_file(10, "tests/files/bonds/simple.bnd") {
            Ok(_) => panic!("Function should have failed."),
            Err(BondsError::AtomNotFound(x, y)) => {
                assert_eq!(x, 11);
                assert_eq!(y, 10);
            }
            Err(e) => panic!("Unexpected error returned: `{}`", e),
        }
    }

    #[test]
    fn parse_bonds_fail_atom_not_found_2() {
        match parse_bonds_file(1, "tests/files/bonds/simple.bnd") {
            Ok(_) => panic!("Function should have failed."),
            Err(BondsError::AtomNotFound(x, y)) => {
                assert_eq!(x, 2);
                assert_eq!(y, 1);
            }
            Err(e) => panic!("Unexpected error returned: `{}`", e),
        }
    }

    #[test]
    fn parse_bonds_fail_selfbonding() {
        match parse_bonds_file(20, "tests/files/bonds/self-bonding.bnd") {
            Ok(_) => panic!("Function should have failed."),
            Err(BondsError::SelfBonding(x)) => {
                assert_eq!(x, 16);
            }
            Err(e) => panic!("Unexpected error returned: `{}`", e),
        }
    }

    #[test]
    fn test_should_guess_elements() {
        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("element name carbon"))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::aaorder("elname carbon", "name r'^H'"))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::aaorder(
                "name r'^C'",
                "@membrane and element symbol H",
            ))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::global("@membrane", "elsymbol P"))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::global(
                "@membrane and elname carbon",
                "name PO4",
            ))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::local(
                "@membrane",
                "not element name carbon",
                2.5,
            ))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::local(
                "element symbol C",
                "name PO4",
                2.5,
            ))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::individual(
                "element name phosphate",
                "name C4A C4B",
            ))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::individual(
                "name PO4",
                "elname carbon",
            ))
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(
                Geometry::cuboid("element name oxygen", [1.0, 2.0], [-2.0, 2.0], [-1.0, 3.0])
                    .unwrap(),
            )
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(
                Geometry::cylinder(
                    "element symbol O",
                    3.0,
                    [f32::NEG_INFINITY, f32::INFINITY],
                    Axis::Z,
                )
                .unwrap(),
            )
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(Geometry::sphere("elname oxygen", 3.0).unwrap())
            .build()
            .unwrap();

        assert!(should_guess_elements(&analysis));
    }

    #[test]
    fn test_should_not_guess_elements() {
        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .build()
            .unwrap();

        assert!(!should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane",
                "@membrane and name r'^H'",
            ))
            .build()
            .unwrap();

        assert!(!should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .build()
            .unwrap();

        assert!(!should_guess_elements(&analysis));

        let analysis = Analysis::builder()
            .structure("system.gro")
            .trajectory("system.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .geometry(Geometry::sphere("@protein", 3.0).unwrap())
            .build()
            .unwrap();

        assert!(!should_guess_elements(&analysis));
    }
}
