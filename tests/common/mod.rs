// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Functions used in various integration tests.

use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::PathBuf,
};

use approx::assert_relative_eq;
use gorder::prelude::Vector3D;
use indexmap::IndexMap;

/// Test utility. Diff the contents of two files without the first `skip` lines.
#[allow(dead_code)]
pub(super) fn diff_files_ignore_first(file1: &str, file2: &str, skip: usize) -> bool {
    let content1 = read_file_without_first_lines(file1, skip);
    let content2 = read_file_without_first_lines(file2, skip);
    content1 == content2
}

fn read_file_without_first_lines(file: &str, skip: usize) -> Vec<String> {
    let reader = BufReader::new(File::open(file).unwrap());
    reader
        .lines()
        .skip(skip) // skip the first line
        .map(|line| line.unwrap())
        .collect()
}

/// Test utility. Assert that two order files (not csv) match each other.
#[allow(dead_code)]
pub(super) fn assert_eq_order(a: &str, b: &str, skip: usize) {
    let (file_a, file_b) = match (File::open(a), File::open(b)) {
        (Ok(f1), Ok(f2)) => (f1, f2),
        _ => panic!("One or both files do not exist."),
    };

    let mut lines_a = BufReader::new(file_a).lines().skip(skip);
    let mut lines_b = BufReader::new(file_b).lines().skip(skip);

    loop {
        match (lines_a.next(), lines_b.next()) {
            (Some(Ok(line_a)), Some(Ok(line_b))) => assert_lines(&line_a, &line_b),
            (None, None) => break,
            _ => panic!("Files have different number of lines"),
        }
    }
}

/// Test utility. Assert that two membrane normal files match each other.
#[allow(dead_code)]
pub(super) fn assert_eq_normals(a: &str, b: &str) {
    let content_a = std::fs::read_to_string(a).unwrap();
    let content_b = std::fs::read_to_string(b).unwrap();

    let normals_a: IndexMap<String, Vec<Vec<Vector3D>>> = serde_yaml::from_str(&content_a).unwrap();
    let normals_b: IndexMap<String, Vec<Vec<Vector3D>>> = serde_yaml::from_str(&content_b).unwrap();

    assert_eq!(normals_a.len(), normals_b.len());
    for (moltype_a, moltype_b) in normals_a.keys().zip(normals_b.keys()) {
        //assert_eq!(moltype_a, moltype_b);
        assert_eq!(
            normals_a.get(moltype_a).unwrap().len(),
            normals_b.get(moltype_b).unwrap().len()
        );

        for (frame_a, frame_b) in normals_a
            .get(moltype_a)
            .unwrap()
            .iter()
            .zip(normals_b.get(moltype_b).unwrap().iter())
        {
            assert_eq!(frame_a.len(), frame_b.len());
            for (mol_a, mol_b) in frame_a.iter().zip(frame_b.iter()) {
                if mol_a.x.is_nan() && mol_a.y.is_nan() && mol_a.z.is_nan() {
                    assert!(mol_b.x.is_nan());
                    assert!(mol_b.y.is_nan());
                    assert!(mol_b.z.is_nan());
                    continue;
                }

                assert_relative_eq!(mol_a.x, mol_b.x, epsilon = 1e-5);
                assert_relative_eq!(mol_a.y, mol_b.y, epsilon = 1e-5);
                assert_relative_eq!(mol_a.z, mol_b.z, epsilon = 1e-5);
            }
        }
    }
}

/// Test utility. Assert that two order csv files match each other.
#[allow(dead_code)]
pub(super) fn assert_eq_csv(a: &str, b: &str, skip: usize) {
    let (file_a, file_b) = match (File::open(a), File::open(b)) {
        (Ok(f1), Ok(f2)) => (f1, f2),
        _ => panic!("One or both files do not exist."),
    };

    let mut lines_a = BufReader::new(file_a).lines().skip(skip);
    let mut lines_b = BufReader::new(file_b).lines().skip(skip);

    loop {
        match (lines_a.next(), lines_b.next()) {
            (Some(Ok(line_a)), Some(Ok(line_b))) => assert_lines_csv(&line_a, &line_b),
            (None, None) => break,
            _ => panic!("Files have different number of lines"),
        }
    }
}

fn assert_lines(line_a: &str, line_b: &str) {
    let mut line_a_split = line_a.split_whitespace();
    let mut line_b_split = line_b.split_whitespace();

    loop {
        match (line_a_split.next(), line_b_split.next()) {
            (Some(item_a), Some(item_b)) => assert_items(item_a, item_b),
            (None, None) => break,
            _ => panic!("Lines do not match: {} vs. {}", line_a, line_b),
        }
    }
}

fn assert_lines_csv(line_a: &str, line_b: &str) {
    let mut line_a_split = line_a.split(",");
    let mut line_b_split = line_b.split(",");

    loop {
        match (line_a_split.next(), line_b_split.next()) {
            (Some(item_a), Some(item_b)) => assert_items(item_a, item_b),
            (None, None) => break,
            _ => panic!("Lines do not match: {} vs. {}", line_a, line_b),
        }
    }
}

fn assert_items(item_a: &str, item_b: &str) {
    match (item_a.parse::<f32>(), item_b.parse::<f32>()) {
        (Ok(z1), Ok(z2)) if z1.is_nan() && z2.is_nan() => (),
        (Ok(z1), Ok(z2)) => assert_relative_eq!(z1, z2, epsilon = 2e-4),
        (Err(_), Err(_)) => assert_eq!(
            item_a, item_b,
            "Items do not match: {} vs {}",
            item_a, item_b
        ),
        _ => panic!("Invalid or mismatched items: {} vs {}", item_a, item_b),
    }
}

/// Test utility. Assert that two ordermap files match each other.
#[allow(dead_code)]
pub(super) fn assert_eq_maps(a: &str, b: &str, skip: usize) {
    let (file_a, file_b) = match (File::open(a), File::open(b)) {
        (Ok(f1), Ok(f2)) => (f1, f2),
        _ => panic!("One or both files do not exist."),
    };

    let mut lines_a = BufReader::new(file_a).lines().skip(skip);
    let mut lines_b = BufReader::new(file_b).lines().skip(skip);

    loop {
        match (lines_a.next(), lines_b.next()) {
            (Some(Ok(line_a)), Some(Ok(line_b))) => {
                let is_data = line_a
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f32>().ok())
                    .is_some();

                if is_data {
                    let p: Vec<_> = line_a.split_whitespace().collect();
                    let q: Vec<_> = line_b.split_whitespace().collect();
                    assert_eq!(p.len(), 3, "Data line must have 3 columns");
                    assert_eq!(q.len(), 3, "Data line must have 3 columns");

                    assert_eq!(p[0], q[0], "First columns differ");
                    assert_eq!(p[1], q[1], "Second columns differ");

                    assert_items(p[2], q[2]);
                } else {
                    assert_eq!(line_a, line_b, "Non-data lines differ");
                }
            }
            (None, None) => break,
            _ => panic!("Files have different number of lines"),
        }
    }
}

/// Test utility. Compares the content of all files in a directory except the excluded files.
#[allow(dead_code)]
pub(super) fn read_and_compare_files(dir: &str, exclude_paths: &[PathBuf], expected_content: &str) {
    let mut count = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() || exclude_paths.contains(&path) {
            continue;
        }

        count += 1;

        let mut file_content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut file_content)
            .unwrap();

        assert_eq!(file_content, expected_content);
    }

    assert_eq!(count, 6);
}
