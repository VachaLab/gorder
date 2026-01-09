// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Reading NDX file into the global system.

use std::collections::HashSet;
use std::fmt::Write;

use colored::Colorize;
use groan_rs::{errors::ParseNdxError, system::System};

use crate::PANIC_MESSAGE;

fn unpack_set(set: &HashSet<String>) -> String {
    let mut output = String::new();
    let len = set.len();

    for (i, key) in set.iter().enumerate() {
        writeln!(output).expect(PANIC_MESSAGE);
        write!(output, "> {}", key.yellow()).expect(PANIC_MESSAGE);

        if i >= 9 && len != i + 1 {
            write!(
                output,
                "... and {} more...",
                (len - i - 1).to_string().yellow()
            )
            .expect(PANIC_MESSAGE);
            break;
        }
    }

    output
}

/// Read an ndx file handling the returned warnings and propagating errors.
pub(super) fn read_ndx_file(system: &mut System, ndx: &str) -> Result<(), ParseNdxError> {
    match system.read_ndx(ndx) {
        Ok(_) => (),
        Err(ParseNdxError::DuplicateGroupsWarning(x)) => colog_warn!(
            "Duplicate group(s) detected in the ndx file '{}': {}",
            ndx,
            unpack_set(&x).yellow()
        ),
        Err(ParseNdxError::InvalidNamesWarning(x)) => {
            colog_warn!(
                "Group(s) with invalid name(s) detected in the ndx file '{}': {}",
                ndx,
                unpack_set(&x).yellow()
            )
        }
        Err(e) => return Err(e),
    }

    colog_info!(
        "Read {} group(s) from ndx file '{}'.",
        system.get_n_groups() - 2,
        ndx
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_ndx() {
        let mut system = System::from_file("tests/files/cg.tpr").unwrap();
        read_ndx_file(&mut system, "tests/files/cg.ndx").unwrap();
        assert_eq!(system.get_n_groups(), 3);
    }

    #[test]
    fn test_duplicate_groups() {
        let mut system = System::from_file("tests/files/cg.tpr").unwrap();
        read_ndx_file(&mut system, "tests/files/cg_duplicate.ndx").unwrap();
        assert_eq!(system.get_n_groups(), 3);
    }

    #[test]
    fn test_invalid_name() {
        let mut system = System::from_file("tests/files/cg.tpr").unwrap();
        read_ndx_file(&mut system, "tests/files/cg_invalid.ndx").unwrap();
        assert_eq!(system.get_n_groups(), 2);
    }
}
