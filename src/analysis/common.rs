// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Implementations of common function used by both AAOrder and CGOrder calculations.

use super::geometry::{GeometrySelection, GeometrySelectionType};
use super::pbc::{NoPBC, PBC3D};
use super::topology::molecule::MoleculeTypes;
use super::topology::SystemTopology;
use crate::errors::TopologyError;
use crate::errors::{AnalysisError, GeometryConfigError};
use crate::input::{Geometry, MembraneNormal};
use crate::PANIC_MESSAGE;
use colored::Colorize;
use groan_rs::files::FileType;
use groan_rs::prelude::{
    GroReader, GroupXtcReader, OrderedAtomIterator, ProgressPrinter, SimBox, TrrReader,
};
use groan_rs::{errors::GroupError, structures::group::Group, system::System};

/// A prefix used as an identifier for gorder groups.
pub(super) const GORDER_GROUP_PREFIX: &str = "xxxGorderReservedxxx-";

#[macro_use]
pub(crate) mod macros {
    macro_rules! group_name {
        ($group:expr) => {
            concat!("xxxGorderReservedxxx-", $group)
        };
    }

    pub(crate) use group_name;
}

/// Return a (hopefully) useful hint that can help solving an empty group error.
fn get_hint(group: &str) -> String {
    let (yaml_name, yaml_type) = match group {
        "HeavyAtoms" => ("heavy_atoms".bright_blue(), "analysis_type".bright_blue()),
        "Hydrogens" => ("hydrogens".bright_blue(), "analysis_type".bright_blue()),
        "Beads" => ("beads".bright_blue(), "analysis_type".bright_blue()),
        "Membrane" => ("membrane".bright_blue(), "leaflets".bright_blue()),
        "Heads" => ("heads".bright_blue(), "leaflets".bright_blue()),
        "NormalHeads" => ("heads".bright_blue(), "membrane_normal".bright_blue()),
        "ClusterHeads" => ("heads".bright_blue(), "leaflets".bright_blue()),
        "Methyls" => ("methyls".bright_blue(), "leaflets".bright_blue()),
        "GeomReference" => ("reference".bright_blue(), "geometry".bright_blue()),
        "Saturated" => ("saturated".bright_blue(), "analysis_type".bright_blue()),
        "Unsaturated" => ("unsaturated".bright_blue(), "analysis_type".bright_blue()),
        "Ignore" => ("ignore".bright_blue(), "analysis_type".bright_blue()),
        // unknown group name; this should not happen, but it's not important so we will pretend it's okay
        _ => return String::from("a query specifying the group selects no atoms"),
    };

    format!(
        "the query specified for '{}' inside '{}' selects no atoms; is the query correct?",
        yaml_name, yaml_type
    )
}

/// Create a group while handling all potential errors. Check that the group is not empty.
/// Also adds the atoms of the newly created group into the master group.
pub(super) fn create_group(
    system: &mut System,
    group: &str,
    query: &str,
) -> Result<(), TopologyError> {
    let group_name = format!("{}{}", GORDER_GROUP_PREFIX, group);

    match system.group_create(&group_name, query) {
        Ok(_) | Err(GroupError::AlreadyExistsWarning(_)) => (),
        Err(GroupError::InvalidQuery(e)) => {
            return Err(TopologyError::InvalidQuery(e))
        }
        Err(e) => panic!(
            "FATAL GORDER ERROR | common::create_group | Unexpected error `{}` returned when selecting '{}' using the query '{}'. {}",
            e, group, query, PANIC_MESSAGE
        ),
    }

    if system.group_isempty(&group_name).unwrap_or_else(|_| {
        panic!(
            "FATAL GORDER ERROR | common::create_group | Group '{}' should exist. {}",
            group, PANIC_MESSAGE,
        )
    }) {
        let hint = get_hint(group);
        Err(TopologyError::EmptyGroup {
            group: group.to_owned(),
            hint,
        })
    } else {
        // add group to the master group
        if !system.group_exists(group_name!("Master")) {
            system
                .group_create(group_name!("Master"), "not all")
                .expect(PANIC_MESSAGE);
        }
        system
            .group_extend(group_name!("Master"), &group_name)
            .unwrap_or_else(|e| {
                panic!("FATAL GORDER ERROR | common::create_group | Could not extend Master group: `{}`", e)
            });
        Ok(())
    }
}

/// Check overlap between two groups and return an error if they overlap.
pub(super) fn check_groups_overlap(
    system: &System,
    group1_name: &str,
    group1_query: &str,
    group2_name: &str,
    group2_query: &str,
) -> Result<(), TopologyError> {
    let n_overlapping = system
        .group_iter(&format!("{}{}", GORDER_GROUP_PREFIX, group1_name))
        .expect(PANIC_MESSAGE)
        .intersection(
            system
                .group_iter(&format!("{}{}", GORDER_GROUP_PREFIX, group2_name))
                .expect(PANIC_MESSAGE),
        )
        .count();

    if n_overlapping > 0 {
        return Err(TopologyError::AtomsOverlap {
            n_overlapping,
            name1: group1_name.to_owned(),
            query1: group1_query.to_owned(),
            name2: group2_name.to_owned(),
            query2: group2_query.to_owned(),
        });
    }

    Ok(())
}

/// Construct a geometry selection structure and prepare the system for geometry selection.
pub(super) fn prepare_geometry_selection(
    geometry: Option<&Geometry>,
    system: &mut System,
    handle_pbc: bool,
) -> Result<GeometrySelectionType, Box<dyn std::error::Error + Send + Sync>> {
    let geom = match handle_pbc {
        true => {
            let simbox = system.get_box().expect(PANIC_MESSAGE);
            let pbc = PBC3D::new(simbox);
            GeometrySelectionType::from_geometry(geometry, &pbc)
        }
        false => {
            // sanity check that the geometry selection does not use box center as a reference
            match geometry {
                Some(x) if x.uses_box_center() => {
                    return Err(Box::from(GeometryConfigError::InvalidBoxCenter))
                }
                Some(_) | None => (),
            }
            GeometrySelectionType::from_geometry(geometry, &NoPBC)
        }
    };

    match &geom {
        GeometrySelectionType::None(_) => (),
        GeometrySelectionType::Cuboid(x) => x.prepare_system(system)?,
        GeometrySelectionType::Cylinder(x) => x.prepare_system(system)?,
        GeometrySelectionType::Sphere(x) => x.prepare_system(system)?,
    }

    Ok(geom)
}

/// Prepare the system for dynamic membrane normal calculation, if this is needed.
pub(super) fn prepare_membrane_normal_calculation(
    membrane_normal: &MembraneNormal,
    system: &mut System,
) -> Result<(), TopologyError> {
    match membrane_normal {
        MembraneNormal::Static(_) | MembraneNormal::FromFile(_) | MembraneNormal::FromMap(_) => {
            Ok(())
        } // do nothing
        MembraneNormal::Dynamic(params) => create_group(system, "NormalHeads", params.heads()),
    }
}

/// Check that the simulation box is valid. Return the box, if it is.
pub(super) fn check_box(frame: &System) -> Result<&SimBox, AnalysisError> {
    let simbox = frame.get_box().ok_or(AnalysisError::UndefinedBox)?;

    if !simbox.is_orthogonal() {
        return Err(AnalysisError::NotOrthogonalBox);
    }

    if simbox.is_zero() {
        return Err(AnalysisError::ZeroBox);
    }

    Ok(simbox)
}

/// Calculate order parameters in a single simulation frame.
pub(super) fn analyze_frame(
    frame: &System,
    data: &mut SystemTopology,
) -> Result<(), AnalysisError> {
    let molecules = data.molecule_types_mut() as *mut MoleculeTypes;

    if data.handle_pbc() {
        // check the validity of the simulation box
        let simbox = check_box(frame)?;
        let mut pbc = PBC3D::new(simbox);
        // initialize the reading of the next frame
        data.init_new_frame(frame, &mut pbc);

        // safety: we are modifying a different part of the `data` structure
        unsafe { &mut *molecules }.analyze_frame(frame, data, &pbc)?
    } else {
        let mut pbc = NoPBC;
        // initialize the reading of the next frame
        data.init_new_frame(frame, &mut pbc);

        // safety: we are modifying a different part of the `data` structure
        unsafe { &mut *molecules }.analyze_frame(frame, data, &pbc)?
    };

    // print information about leaflet assignment for quick sanity check by the user
    // only do this for the first frame (this also guarantees that only one thread prints this information)
    if data.frame() == 0 {
        data.log_first_frame_leaflet_assignment_info();
    }

    // increase the frame counter
    data.increase_frame_counter();

    Ok(())
}

/// Perform the analysis.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_trajectory(
    system: &System,
    topology: SystemTopology,
    trajectory: &[String],
    n_threads: usize,
    begin: f32,
    end: f32,
    step: usize,
    silent: bool,
) -> Result<SystemTopology, Box<dyn std::error::Error + Send + Sync>> {
    // get the format of the trajectory files from the first file;
    // this assumes that it has been validated before that all trajectories have the same format
    let format = FileType::from_name(trajectory.first().unwrap_or_else(||
        panic!("FATAL GORDER ERROR | common::read_trajectory | At least one trajectory file should have been provided. {}", PANIC_MESSAGE))
    );

    let progress_printer = if silent {
        None
    } else {
        Some(ProgressPrinter::new().with_print_freq(100 / n_threads))
    };

    if trajectory.len() == 1 {
        colog_info!(
            "Will read trajectory file '{}' (start: {} ps, end: {} ps, step: {}).",
            trajectory.first().expect(PANIC_MESSAGE),
            begin,
            end,
            step
        );
    } else {
        colog_info!(
            "Will read trajectory files '{}' (start: {} ps, end: {} ps, step: {}).",
            trajectory.join(" "),
            begin,
            end,
            step,
        )
    }

    colog_info!("Performing the analysis using {} thread(s)...", n_threads);

    match format {
        FileType::XTC => if trajectory.len() == 1 {
            system.traj_iter_map_reduce::<GroupXtcReader, SystemTopology, AnalysisError>(
                trajectory.first().expect(PANIC_MESSAGE),
                n_threads,
                analyze_frame,
                topology,
                Some(group_name!("Master")),
                Some(begin),
                Some(end),
                Some(step),
                progress_printer,
            )} else {
            system.traj_iter_cat_map_reduce::<GroupXtcReader, SystemTopology, AnalysisError>(
                trajectory,
                n_threads,
                analyze_frame,
                topology,
                Some(group_name!("Master")),
                Some(begin),
                Some(end),
                Some(step),
                progress_printer,
            )}
        FileType::TRR => if trajectory.len() == 1 {
            system.traj_iter_map_reduce::<TrrReader, SystemTopology, AnalysisError>(
                trajectory.first().expect(PANIC_MESSAGE),
                n_threads,
                analyze_frame,
                topology,
                None,
                Some(begin),
                Some(end),
                Some(step),
                progress_printer,
            )} else {
            system.traj_iter_cat_map_reduce::<TrrReader, SystemTopology, AnalysisError>(
                trajectory,
                n_threads,
                analyze_frame,
                topology,
                None,
                Some(begin),
                Some(end),
                Some(step),
                progress_printer,
            )}
        FileType::GRO => system
            .traj_iter_map_reduce::<GroReader, SystemTopology, AnalysisError>(
                trajectory.first().expect(PANIC_MESSAGE),
                n_threads,
                analyze_frame,
                topology,
                None,
                Some(begin),
                Some(end),
                Some(step),
                progress_printer,
            ),
        _ => panic!("FATAL GORDER ERROR | common::read_trajectory | Unexpected trajectory file format `{}`.", format),
    }
}

/// Get index of an atom that represents the head of the given lipid molecule.
pub(super) fn get_reference_head(
    molecule: &Group,
    system: &System,
    group_to_search: &'static str,
) -> Result<usize, TopologyError> {
    let mut atoms = Vec::new();
    for index in molecule.get_atoms().iter() {
        if system
            .group_isin(group_to_search, index)
            .expect(PANIC_MESSAGE)
        {
            atoms.push(index);
        }
    }

    if atoms.is_empty() {
        return Err(TopologyError::NoHead(
            molecule
                .get_atoms()
                .first()
                .unwrap_or_else(|| panic!("FATAL GORDER ERROR | common::get_reference_head | No atoms detected inside a molecule. {}", PANIC_MESSAGE))));
    }

    if atoms.len() > 1 {
        return Err(TopologyError::MultipleHeads(
            molecule.get_atoms().first().expect(PANIC_MESSAGE),
        ));
    }

    Ok(*atoms.first().expect(PANIC_MESSAGE))
}

/// Returns a new vector by interleaving elements from two slices.
/// Up to `n` items are taken from `vec1`, followed by up to `m` items from `vec2`,
/// repeating this pattern until both slices are exhausted.
pub(super) fn interleave_vectors<T: Clone>(vec1: &[T], vec2: &[T], n: usize, m: usize) -> Vec<T> {
    let mut result = Vec::with_capacity(vec1.len() + vec2.len());
    let mut iter1 = vec1.iter();
    let mut iter2 = vec2.iter();

    loop {
        for _ in 0..n {
            if let Some(value) = iter1.next() {
                result.push(value.clone());
            }
        }

        for _ in 0..m {
            if let Some(value) = iter2.next() {
                result.push(value.clone());
            }
        }

        if iter1.len() == 0 && iter2.len() == 0 {
            break;
        }
    }

    result
}
