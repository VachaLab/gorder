// Released under MIT License.
// Copyright (c) 2024-2025 Ladislav Bartos

//! Integration tests for the calculation of coarse-grained order parameters.

mod common;

use std::{
    fs::{read_to_string, File},
    io::Read,
    path::{Path, PathBuf},
};

use approx::assert_relative_eq;
use gorder::prelude::*;
use hashbrown::HashMap;
use indexmap::IndexMap;
use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

use common::{assert_eq_csv, assert_eq_maps, assert_eq_order, read_and_compare_files};

use crate::common::{assert_eq_normals, diff_files_ignore_first};

#[test]
fn test_cg_order_basic_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_basic.yaml", 1);
}

#[test]
fn test_cg_order_basic_concatenated_yaml_multiple_threads() {
    for n_threads in [1, 2, 3, 8, 64, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/split/cg*.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_basic.yaml", 1);
    }
}

#[test]
fn test_cg_order_basic_ndx_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .index("tests/files/cg.ndx")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("Membrane"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_basic.yaml", 1);
}

#[test]
fn test_cg_order_basic_table() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_table, "tests/files/cg_order_basic.tab", 1);
}

#[test]
fn test_cg_order_basic_xvg() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let pattern = format!("{}/order.xvg", path_to_dir);

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_xvg(pattern)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    for molecule in ["POPC", "POPE", "POPG"] {
        let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
        let path_expected = format!("tests/files/cg_order_basic_{}.xvg", molecule);

        assert_eq_order(&path, &path_expected, 1);
    }
}

#[test]
fn test_cg_order_basic_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(path_to_csv, "tests/files/cg_order_basic.csv", 0);
}

#[test]
fn test_cg_order_basic_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_basic.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_yaml() {
    for method in [
        LeafletClassification::global("@membrane", "name PO4"),
        LeafletClassification::local("@membrane", "name PO4", 2.5),
        LeafletClassification::individual("name PO4", "name C4A C4B"),
        LeafletClassification::clustering("name PO4"),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_yaml_only_upper() {
    for method in [
        LeafletClassification::global("@membrane", "name PO4"),
        LeafletClassification::local("@membrane", "name PO4", 2.5),
        LeafletClassification::individual("name PO4", "name C4A C4B"),
        LeafletClassification::clustering("name PO4"),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("resid 1 to 254"))
            .leaflets(method.with_frequency(Frequency::once()))
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/cg_order_leaflets_only_upper.yaml",
            1,
        );
    }
}

#[test]
fn test_cg_order_leaflets_yaml_alt_traj() {
    for trajectory in ["tests/files/cg.trr", "tests/files/cg_traj.gro"] {
        for n_threads in [1, 3, 8] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/cg.tpr")
                .trajectory(trajectory)
                .output(path_to_output)
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .leaflets(
                    LeafletClassification::individual("name PO4", "name C4A C4B")
                        .with_frequency(Frequency::once()),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_cg_order_leaflets_yaml_trr_concatenated() {
    for n_threads in [1, 3, 8] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/split/cg*.trr")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(
                LeafletClassification::individual("name PO4", "name C4A C4B")
                    .with_frequency(Frequency::once()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_yaml_from_gro() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.gro")
        .bonds("tests/files/cg.bnd")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
}

#[test]
fn test_cg_order_leaflets_yaml_from_gro_min_bonds() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.gro")
        .bonds("tests/files/cg_min.bnd")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
}

#[test]
fn test_cg_order_leaflets_yaml_from_pdb() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.pdb")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
}

#[test]
fn test_cg_order_leaflets_yaml_from_pqr() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.pqr")
        .bonds("tests/files/cg.bnd")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
}

// this test not only checks that a TPR file is redefined by a bond file but also checks the behavior of gorder when molecules with duplicate residue names are present
#[test]
fn test_cg_order_leaflets_yaml_from_tpr_redefined_bonds() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .bonds("tests/files/cg_redefined.bnd")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_redefined_bonds.yaml",
        1,
    );
}

#[test]
fn test_cg_order_leaflets_yaml_from_pdb_redefined_bonds() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.pdb")
        .bonds("tests/files/cg_redefined.bnd")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_redefined_bonds.yaml",
        1,
    );
}

#[test]
fn test_cg_order_leaflets_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64, 128] {
        for method in [
            LeafletClassification::global("@membrane", "name PO4"),
            LeafletClassification::local("@membrane", "name PO4", 2.5),
            LeafletClassification::individual("name PO4", "name C4A C4B"),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/cg.tpr")
                .trajectory("tests/files/cg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .leaflets(method)
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_cg_order_leaflets_yaml_multiple_threads_various_frequencies() {
    for n_threads in [1, 2, 5, 8, 128] {
        for method in [
            LeafletClassification::global("@membrane", "name PO4"),
            LeafletClassification::local("@membrane", "name PO4", 2.5),
            LeafletClassification::individual("name PO4", "name C4A C4B"),
        ] {
            for freq in [
                Frequency::every(4).unwrap(),
                Frequency::every(20).unwrap(),
                Frequency::every(200).unwrap(),
                Frequency::once(),
            ] {
                let output = NamedTempFile::new().unwrap();
                let path_to_output = output.path().to_str().unwrap();

                let analysis = Analysis::builder()
                    .structure("tests/files/cg.tpr")
                    .trajectory("tests/files/cg.xtc")
                    .output(path_to_output)
                    .analysis_type(AnalysisType::cgorder("@membrane"))
                    .leaflets(method.clone().with_frequency(freq))
                    .n_threads(n_threads)
                    .silent()
                    .overwrite()
                    .build()
                    .unwrap();

                analysis.run().unwrap().write().unwrap();

                assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
            }
        }
    }
}

#[test]
fn test_cg_order_leaflets_clustering_yaml_multiple_threads_various_frequencies() {
    for n_threads in [1, 2, 5, 8, 128] {
        for freq in [
            Frequency::every(1).unwrap(),
            Frequency::every(4).unwrap(),
            Frequency::every(200).unwrap(),
            Frequency::once(),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/cg.tpr")
                .trajectory("tests/files/cg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .leaflets(LeafletClassification::clustering("name PO4").with_frequency(freq))
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_cg_order_leaflets_table() {
    for method in [
        LeafletClassification::global("@membrane", "name PO4"),
        LeafletClassification::local("@membrane", "name PO4", 2.5),
        LeafletClassification::individual("name PO4", "name C4A C4B"),
    ] {
        let output_table = NamedTempFile::new().unwrap();
        let path_to_table = output_table.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output_tab(path_to_table)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_table, "tests/files/cg_order_leaflets.tab", 1);
    }
}

#[test]
fn test_cg_order_leaflets_xvg() {
    for method in [
        LeafletClassification::global("@membrane", "name PO4"),
        LeafletClassification::local("@membrane", "name PO4", 2.5),
        LeafletClassification::individual("name PO4", "name C4A C4B"),
    ] {
        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let pattern = format!("{}/order.xvg", path_to_dir);

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output_xvg(pattern)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        for molecule in ["POPC", "POPE", "POPG"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!("tests/files/cg_order_leaflets_{}.xvg", molecule);

            assert_eq_order(&path, &path_expected, 1);
        }
    }
}

#[test]
fn test_cg_order_leaflets_csv() {
    for method in [
        LeafletClassification::global("@membrane", "name PO4"),
        LeafletClassification::local("@membrane", "name PO4", 2.5),
        LeafletClassification::individual("name PO4", "name C4A C4B"),
    ] {
        let output_csv = NamedTempFile::new().unwrap();
        let path_to_csv = output_csv.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output_csv(path_to_csv)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_csv(path_to_csv, "tests/files/cg_order_leaflets.csv", 0);
    }
}

#[test]
fn test_cg_order_limit_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .min_samples(5000)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_limit.yaml", 1);
}

#[test]
fn test_cg_order_leaflets_limit_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .min_samples(2000)
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_leaflets_limit.yaml",
        1,
    );
}

#[test]
fn test_cg_order_leaflets_limit_tab() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .min_samples(2000)
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_table, "tests/files/cg_order_leaflets_limit.tab", 1);
}

#[test]
fn test_cg_order_leaflets_limit_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .min_samples(2000)
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(path_to_csv, "tests/files/cg_order_leaflets_limit.csv", 0);
}

#[test]
fn test_cg_order_leaflets_clustering_fail_not_enough_atoms() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::clustering("resid 1 and name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("clustering leaflet classification has been requested but only")),
    }

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::clustering("not all"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e.to_string().contains("is empty")),
    }
}

#[test]
fn test_cg_order_begin_end_step_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .begin(352_000.0)
        .end(358_000.0)
        .step(5)
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = analysis.run().unwrap();
    assert_eq!(results.n_analyzed_frames(), 13);
    results.write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_begin_end_step.yaml",
        1,
    );
}

#[test]
fn test_cg_order_begin_end_step_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .begin(352_000.0)
            .end(358_000.0)
            .step(5)
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let results = analysis.run().unwrap();
        assert_eq!(results.n_analyzed_frames(), 13);
        results.write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/cg_order_begin_end_step.yaml",
            1,
        );
    }
}

#[test]
fn test_cg_order_begin_end_step_concatenated_yaml_multiple_threads() {
    for n_threads in [1, 2, 3, 8, 64, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory(vec![
                "tests/files/split/cg1.xtc",
                "tests/files/split/cg2.xtc",
                "tests/files/split/cg3.xtc",
                "tests/files/split/cg4.xtc",
                "tests/files/split/cg5.xtc",
            ])
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .begin(352_000.0)
            .end(358_000.0)
            .step(5)
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let results = analysis.run().unwrap();
        assert_eq!(results.n_analyzed_frames(), 13);
        results.write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/cg_order_begin_end_step.yaml",
            1,
        );
    }
}

#[test]
fn test_cg_order_begin_end_step_yaml_leaflets_multiple_threads_various_frequencies() {
    for n_threads in [1, 2, 5, 8, 128] {
        for method in [
            LeafletClassification::global("@membrane", "name PO4"),
            LeafletClassification::local("@membrane", "name PO4", 2.5),
            LeafletClassification::individual("name PO4", "name C4A C4B"),
        ] {
            for freq in [
                Frequency::every(2).unwrap(),
                Frequency::every(10).unwrap(),
                Frequency::once(),
            ] {
                let output = NamedTempFile::new().unwrap();
                let path_to_output = output.path().to_str().unwrap();

                let analysis = Analysis::builder()
                    .structure("tests/files/cg.tpr")
                    .trajectory("tests/files/cg.xtc")
                    .output(path_to_output)
                    .analysis_type(AnalysisType::cgorder("@membrane"))
                    .begin(352_000.0)
                    .end(358_000.0)
                    .step(5)
                    .leaflets(method.clone().with_frequency(freq))
                    .n_threads(n_threads)
                    .silent()
                    .overwrite()
                    .build()
                    .unwrap();

                let results = analysis.run().unwrap();
                assert_eq!(results.n_analyzed_frames(), 13);
                results.write().unwrap();

                assert_eq_order(
                    path_to_output,
                    "tests/files/cg_order_begin_end_step.yaml",
                    1,
                );
            }
        }
    }
}

#[test]
fn test_cg_order_begin_end_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .begin(352_000.0)
        .end(358_000.0)
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = analysis.run().unwrap();
    assert_eq!(results.n_analyzed_frames(), 61);
    results.write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_begin_end.yaml", 1);
}

#[test]
fn test_cg_order_begin_end_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .begin(352_000.0)
            .end(358_000.0)
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let results = analysis.run().unwrap();
        assert_eq!(results.n_analyzed_frames(), 61);
        results.write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_begin_end.yaml", 1);
    }
}

#[test]
fn test_cg_order_no_molecules() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output("THIS_FILE_SHOULD_NOT_BE_CREATED_CG_1")
        .analysis_type(AnalysisType::cgorder("@ion"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert!(!Path::new("THIS_FILE_SHOULD_NOT_BE_CREATED_CG_1").exists());
}

#[test]
fn test_cg_order_empty_molecules() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output("THIS_FILE_SHOULD_NOT_BE_CREATED_CG_2")
        .analysis_type(AnalysisType::cgorder("name PO4 C1A"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert!(!Path::new("THIS_FILE_SHOULD_NOT_BE_CREATED_CG_2").exists());
}

macro_rules! create_file_for_backup {
    ($path:expr) => {{
        File::create($path)
            .unwrap()
            .write_all("This file will be backed up.".as_bytes())
            .unwrap()
    }};
}

#[test]
fn test_cg_order_basic_all_formats_backup() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let file_paths = [
        format!("{}/order.yaml", path_to_dir),
        format!("{}/order.tab", path_to_dir),
        format!("{}/order.csv", path_to_dir),
        format!("{}/order_POPC.xvg", path_to_dir),
        format!("{}/order_POPE.xvg", path_to_dir),
        format!("{}/order_POPG.xvg", path_to_dir),
    ];

    for path in &file_paths {
        create_file_for_backup!(path);
    }

    let xvg_pattern = format!("{}/order.xvg", path_to_dir);

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_yaml(&file_paths[0])
        .output_tab(&file_paths[1])
        .output_csv(&file_paths[2])
        .output_xvg(&xvg_pattern)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .silent()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(&file_paths[0], "tests/files/cg_order_basic.yaml", 1);
    assert_eq_order(&file_paths[1], "tests/files/cg_order_basic.tab", 1);
    assert_eq_csv(&file_paths[2], "tests/files/cg_order_basic.csv", 0);
    assert_eq_order(&file_paths[3], "tests/files/cg_order_basic_POPC.xvg", 1);
    assert_eq_order(&file_paths[4], "tests/files/cg_order_basic_POPE.xvg", 1);
    assert_eq_order(&file_paths[5], "tests/files/cg_order_basic_POPG.xvg", 1);

    read_and_compare_files(
        path_to_dir,
        &file_paths
            .iter()
            .map(|s| Path::new(s).to_path_buf())
            .collect::<Vec<_>>(),
        "This file will be backed up.",
    );
}

#[test]
fn test_cg_order_maps_basic() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "resname POPC and name C1B C2B C3B C4B",
        ))
        .map(
            OrderMap::builder()
                .bin_size([1.0, 1.0])
                .output_directory(path_to_dir)
                .min_samples(10)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    let expected_file_names = [
        "ordermap_POPC-C1B-8--POPC-C2B-9_full.dat",
        "ordermap_POPC-C2B-9--POPC-C3B-10_full.dat",
        "ordermap_POPC-C3B-10--POPC-C4B-11_full.dat",
        "ordermap_average_full.dat",
    ];

    for file in expected_file_names {
        let real_file = format!("{}/POPC/{}", path_to_dir, file);
        let test_file = format!("tests/files/ordermaps_cg/{}", file);
        assert_eq_maps(&real_file, &test_file, 2);
    }

    // check the script
    let real_script = format!("{}/plot.py", path_to_dir);
    assert!(common::diff_files_ignore_first(
        &real_script,
        "scripts/plot.py",
        0
    ));

    // full map for the entire system is the same as for POPC
    let real_file = format!("{}/ordermap_average_full.dat", path_to_dir);
    let test_file = "tests/files/ordermaps_cg/ordermap_average_full.dat";
    assert_eq_maps(&real_file, test_file, 2);

    assert_eq_order(path_to_output, "tests/files/cg_order_small.yaml", 1);
}

#[test]
fn test_cg_order_maps_leaflets() {
    for method in [
        LeafletClassification::global("@membrane", "name PO4"),
        LeafletClassification::local("@membrane", "name PO4", 2.5),
        LeafletClassification::individual("name PO4", "name C4A C4B"),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder(
                "resname POPC and name C1B C2B C3B C4B",
            ))
            .leaflets(method)
            .map(
                OrderMap::builder()
                    .bin_size([1.0, 1.0])
                    .output_directory(path_to_dir)
                    .min_samples(10)
                    .build()
                    .unwrap(),
            )
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        let expected_file_names = [
            "ordermap_POPC-C1B-8--POPC-C2B-9_full.dat",
            "ordermap_POPC-C1B-8--POPC-C2B-9_upper.dat",
            "ordermap_POPC-C2B-9--POPC-C3B-10_lower.dat",
            "ordermap_POPC-C3B-10--POPC-C4B-11_full.dat",
            "ordermap_POPC-C3B-10--POPC-C4B-11_upper.dat",
            "ordermap_POPC-C1B-8--POPC-C2B-9_lower.dat",
            "ordermap_POPC-C2B-9--POPC-C3B-10_full.dat",
            "ordermap_POPC-C2B-9--POPC-C3B-10_upper.dat",
            "ordermap_POPC-C3B-10--POPC-C4B-11_lower.dat",
            "ordermap_average_full.dat",
            "ordermap_average_upper.dat",
            "ordermap_average_lower.dat",
        ];

        for file in expected_file_names {
            let real_file = format!("{}/POPC/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps_cg/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        // full maps for the entire system are the same as for POPC
        for file in [
            "ordermap_average_full.dat",
            "ordermap_average_upper.dat",
            "ordermap_average_lower.dat",
        ] {
            let real_file = format!("{}/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps_cg/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        // check the script
        let real_script = format!("{}/plot.py", path_to_dir);
        assert!(common::diff_files_ignore_first(
            &real_script,
            "scripts/plot.py",
            0
        ));

        assert_eq_order(
            path_to_output,
            "tests/files/cg_order_leaflets_small.yaml",
            1,
        );
    }
}

#[test]
fn test_cg_order_maps_leaflets_full() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .map(
            OrderMap::builder()
                .bin_size([1.0, 1.0])
                .output_directory(path_to_dir)
                .min_samples(10)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    // only testing the maps for the entire system
    for file in [
        "ordermap_average_full.dat",
        "ordermap_average_upper.dat",
        "ordermap_average_lower.dat",
    ] {
        let real_file = format!("{}/{}", path_to_dir, file);
        let test_file = format!("tests/files/ordermaps_cg/full/{}", file);
        assert_eq_maps(&real_file, &test_file, 2);
    }

    // check the script
    let real_script = format!("{}/plot.py", path_to_dir);
    assert!(common::diff_files_ignore_first(
        &real_script,
        "scripts/plot.py",
        0
    ));
}

#[test]
fn test_cg_order_maps_basic_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder(
                "resname POPC and name C1B C2B C3B C4B",
            ))
            .n_threads(n_threads)
            .map(
                OrderMap::builder()
                    .bin_size([1.0, 1.0])
                    .output_directory(path_to_dir)
                    .min_samples(10)
                    .build()
                    .unwrap(),
            )
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        let expected_file_names = [
            "ordermap_POPC-C1B-8--POPC-C2B-9_full.dat",
            "ordermap_POPC-C2B-9--POPC-C3B-10_full.dat",
            "ordermap_POPC-C3B-10--POPC-C4B-11_full.dat",
            "ordermap_average_full.dat",
        ];

        for file in expected_file_names {
            let real_file = format!("{}/POPC/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps_cg/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        // full map for the entire system is the same as for POPC
        let real_file = format!("{}/ordermap_average_full.dat", path_to_dir);
        let test_file = "tests/files/ordermaps_cg/ordermap_average_full.dat";
        assert_eq_maps(&real_file, test_file, 2);

        // check the script
        let real_script = format!("{}/plot.py", path_to_dir);
        assert!(common::diff_files_ignore_first(
            &real_script,
            "scripts/plot.py",
            0
        ));

        assert_eq_order(path_to_output, "tests/files/cg_order_small.yaml", 1);
    }
}

#[test]
fn test_cg_order_maps_basic_backup() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let outer_directory = TempDir::new().unwrap();
    let path_to_outer_dir = outer_directory.path().to_str().unwrap();

    let directory = TempDir::new_in(path_to_outer_dir).unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let backup_file = format!("{}/to_backup.txt", path_to_dir);
    create_file_for_backup!(&backup_file);

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "resname POPC and name C1B C2B C3B C4B",
        ))
        .map(
            OrderMap::builder()
                .bin_size([1.0, 1.0])
                .output_directory(path_to_dir)
                .min_samples(10)
                .build()
                .unwrap(),
        )
        .silent()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    let expected_file_names = [
        "ordermap_POPC-C1B-8--POPC-C2B-9_full.dat",
        "ordermap_POPC-C2B-9--POPC-C3B-10_full.dat",
        "ordermap_POPC-C3B-10--POPC-C4B-11_full.dat",
        "ordermap_average_full.dat",
    ];

    for file in expected_file_names {
        let real_file = format!("{}/POPC/{}", path_to_dir, file);
        let test_file = format!("tests/files/ordermaps_cg/{}", file);
        assert_eq_maps(&real_file, &test_file, 2);
    }

    // check the script
    let real_script = format!("{}/plot.py", path_to_dir);
    assert!(common::diff_files_ignore_first(
        &real_script,
        "scripts/plot.py",
        0
    ));

    assert_eq_order(path_to_output, "tests/files/cg_order_small.yaml", 1);

    // check backed up directory
    let directories = std::fs::read_dir(path_to_outer_dir)
        .unwrap()
        .map(|x| x.unwrap().path())
        .filter(|x| x.is_dir() && x.to_str().unwrap() != path_to_dir)
        .collect::<Vec<PathBuf>>();

    assert_eq!(directories.len(), 1);

    let file_path = format!("{}/to_backup.txt", directories[0].display());
    let mut file_content = String::new();
    File::open(&file_path)
        .unwrap()
        .read_to_string(&mut file_content)
        .unwrap();

    assert_eq!(file_content, "This file will be backed up.");
}

#[test]
fn test_cg_order_error_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_error.yaml", 1);
}

#[test]
fn test_cg_order_error_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .n_threads(n_threads)
            .estimate_error(EstimateError::default())
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_error.yaml", 1);
    }
}

#[test]
fn test_cg_order_error_leaflets_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_error_leaflets.yaml",
        1,
    );
}

#[test]
fn test_cg_order_error_leaflets_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .n_threads(n_threads)
            .estimate_error(EstimateError::default())
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/cg_order_error_leaflets.yaml",
            1,
        );
    }
}

#[test]
fn test_cg_order_error_tab() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_table, "tests/files/cg_order_error.tab", 1);
}

#[test]
fn test_cg_order_error_leaflets_tab() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_table, "tests/files/cg_order_error_leaflets.tab", 1);
}

#[test]
fn test_cg_order_error_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(path_to_csv, "tests/files/cg_order_error.csv", 0);
}

#[test]
fn test_cg_order_error_leaflets_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(path_to_csv, "tests/files/cg_order_error_leaflets.csv", 0);
}

#[test]
fn test_cg_order_error_xvg() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let pattern = format!("{}/order.xvg", path_to_dir);

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_xvg(pattern)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    for molecule in ["POPC", "POPE", "POPG"] {
        let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
        // same files as when `estimate_error` is not provided - xvg files do not show error
        let path_expected = format!("tests/files/cg_order_basic_{}.xvg", molecule);

        assert_eq_order(&path, &path_expected, 1);
    }
}

#[test]
fn test_cg_order_error_leaflets_xvg() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let pattern = format!("{}/order.xvg", path_to_dir);

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_xvg(pattern)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    for molecule in ["POPC", "POPE", "POPG"] {
        let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
        // same files as when `estimate_error` is not provided - xvg files do not show error
        let path_expected = format!("tests/files/cg_order_leaflets_{}.xvg", molecule);

        assert_eq_order(&path, &path_expected, 1);
    }
}

#[test]
fn test_cg_order_error_limit() {
    let output = NamedTempFile::new().unwrap();
    let path_to_yaml = output.path().to_str().unwrap();

    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_yaml(path_to_yaml)
        .output_tab(path_to_table)
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::default())
        .min_samples(5000)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_yaml, "tests/files/cg_order_error_limit.yaml", 1);
    assert_eq_order(path_to_table, "tests/files/cg_order_error_limit.tab", 1);
    assert_eq_csv(path_to_csv, "tests/files/cg_order_error_limit.csv", 0);
}

#[test]
fn test_cg_order_error_leaflets_limit() {
    let output = NamedTempFile::new().unwrap();
    let path_to_yaml = output.path().to_str().unwrap();

    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_yaml(path_to_yaml)
        .output_tab(path_to_table)
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::default())
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .min_samples(2000)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_yaml,
        "tests/files/cg_order_error_leaflets_limit.yaml",
        1,
    );

    assert_eq_order(
        path_to_table,
        "tests/files/cg_order_error_leaflets_limit.tab",
        1,
    );

    assert_eq_csv(
        path_to_csv,
        "tests/files/cg_order_error_leaflets_limit.csv",
        0,
    );
}

#[test]
fn test_cg_order_convergence() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::new(None, Some(path_to_output)).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_convergence.xvg", 1);
}

#[test]
fn test_cg_order_leaflets_convergence() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .estimate_error(EstimateError::new(None, Some(path_to_output)).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_leaflets_convergence.xvg",
        1,
    );
}

#[test]
fn test_cg_order_convergence_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .estimate_error(EstimateError::new(None, Some(path_to_output)).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_convergence.xvg", 1);
    }
}

#[test]
fn test_cg_order_convergence_step() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::new(None, Some(path_to_output)).unwrap())
        .step(5)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_convergence_s5.xvg", 1);
}

#[test]
fn test_cg_order_leaflets_scrambling_various_methods_and_frequencies() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    for n_threads in [1, 2, 5, 8, 128] {
        for method in [
            LeafletClassification::global("@membrane", "name PO4"),
            LeafletClassification::local("@membrane", "name PO4", 3.0),
            LeafletClassification::individual("name PO4", "name C4A C4B"),
        ] {
            for freq in [
                Frequency::every(1).unwrap(),
                Frequency::every(10).unwrap(),
                Frequency::once(),
            ] {
                let analysis = Analysis::builder()
                    .structure("tests/files/scrambling/cg_scrambling.tpr")
                    .trajectory("tests/files/scrambling/cg_scrambling.xtc")
                    .analysis_type(AnalysisType::cgorder("@membrane"))
                    .output_yaml(path_to_output)
                    .leaflets(method.clone().with_frequency(freq))
                    .n_threads(n_threads)
                    .silent()
                    .overwrite()
                    .build()
                    .unwrap();

                analysis.run().unwrap().write().unwrap();

                let test_file = match (method.clone(), freq) {
                    (LeafletClassification::Global(_), Frequency::Every(n)) if n.get() == 1 => {
                        "order_global.yaml"
                    }
                    (LeafletClassification::Global(_), Frequency::Every(n)) if n.get() == 10 => {
                        "order_global_every_10.yaml"
                    }
                    (LeafletClassification::Local(_), Frequency::Every(n)) if n.get() == 1 => {
                        "order_local.yaml"
                    }
                    (LeafletClassification::Local(_), Frequency::Every(n)) if n.get() == 10 => {
                        "order_local_every_10.yaml"
                    }
                    (LeafletClassification::Individual(_), Frequency::Every(n)) if n.get() == 1 => {
                        "order_individual.yaml"
                    }
                    (LeafletClassification::Individual(_), Frequency::Every(n))
                        if n.get() == 10 =>
                    {
                        "order_individual_every_10.yaml"
                    }
                    (_, Frequency::Once) => "order_once.yaml",
                    _ => unreachable!("Unexpected method-frequency combination."),
                };

                assert_eq_order(
                    path_to_output,
                    &format!("tests/files/scrambling/{}", test_file),
                    1,
                );
            }
        }
    }
}

/* This test fails on MacOS and Windows. TODO: fix
#[test]
fn test_cg_order_leaflets_scrambling_clustering() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    for freq in [
        Frequency::every(1).unwrap(),
        Frequency::every(10).unwrap(),
        Frequency::once(),
    ] {
        let analysis = Analysis::builder()
            .structure("tests/files/scrambling/cg_scrambling.tpr")
            .trajectory("tests/files/scrambling/cg_scrambling.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .output_yaml(path_to_output)
            .leaflets(
                LeafletClassification::clustering("name PO4")
                    .clone()
                    .with_frequency(freq),
            )
            .n_threads(4)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        let test_file = match freq {
            Frequency::Every(n) if n.get() == 1 => "order_clustering.yaml",
            Frequency::Every(n) if n.get() == 10 => "order_clustering_every_10.yaml",
            Frequency::Once => "order_once.yaml",
            _ => unreachable!("Unexpected frequency."),
        };

        assert_eq_order(
            path_to_output,
            &format!("tests/files/scrambling/{}", test_file),
            1,
        );
    }
}*/

#[test]
fn test_cg_order_leaflets_scrambling_from_file() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    for n_threads in [1, 2, 3, 5, 8, 128] {
        for (leaflets_file, freq) in ["every.yaml", "every10.yaml", "once.yaml"].into_iter().zip(
            [
                Frequency::every(1).unwrap(),
                Frequency::every(10).unwrap(),
                Frequency::once(),
            ]
            .into_iter(),
        ) {
            let analysis = Analysis::builder()
                .structure("tests/files/scrambling/cg_scrambling.tpr")
                .trajectory("tests/files/scrambling/cg_scrambling.xtc")
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .output_yaml(path_to_output)
                .leaflets(
                    LeafletClassification::from_file(&format!(
                        "tests/files/scrambling/leaflets_{}",
                        leaflets_file
                    ))
                    .with_frequency(freq),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            let test_file = match leaflets_file {
                "every.yaml" => "order_global.yaml",
                "every10.yaml" => "order_global_every_10.yaml",
                "once.yaml" => "order_once.yaml",
                _ => panic!("Unexpected leaflets file provided."),
            };

            assert_eq_order(
                path_to_output,
                &format!("tests/files/scrambling/{}", test_file),
                1,
            );
        }
    }
}

#[test]
fn test_cg_order_leaflets_scrambling_export() {
    for n_threads in [1, 2, 3, 5, 8, 128] {
        for freq in [
            Frequency::every(1).unwrap(),
            Frequency::every(10).unwrap(),
            Frequency::once(),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let output_leaflets = NamedTempFile::new().unwrap();
            let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/scrambling/cg_scrambling.tpr")
                .trajectory("tests/files/scrambling/cg_scrambling.xtc")
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .output_yaml(path_to_output)
                .leaflets(
                    LeafletClassification::global("@membrane", "name PO4")
                        .with_frequency(freq)
                        .with_collect(path_to_output_leaflets),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            let (test_order_file, test_leaflets_file) = match freq {
                Frequency::Every(n) if n.get() == 1 => ("order_global.yaml", "leaflets_every.yaml"),
                Frequency::Every(n) if n.get() == 10 => {
                    ("order_global_every_10.yaml", "leaflets_every10.yaml")
                }
                Frequency::Once => ("order_once.yaml", "leaflets_once.yaml"),
                _ => panic!("Unexpected frequency specified."),
            };

            assert_eq_order(
                path_to_output,
                &format!("tests/files/scrambling/{}", test_order_file),
                1,
            );

            assert!(diff_files_ignore_first(
                path_to_output_leaflets,
                &format!("tests/files/scrambling/{}", test_leaflets_file),
                1,
            ));
        }
    }
}

#[test]
fn test_cg_order_leaflets_scrambling_export_and_load() {
    for n_threads in [1, 2, 5, 8, 64] {
        for freq in [
            Frequency::every(1).unwrap(),
            Frequency::every(10).unwrap(),
            Frequency::once(),
        ] {
            for step in [1, 3, 7, 64] {
                // export the leaflet classification
                let output_orig = NamedTempFile::new().unwrap();
                let path_to_output_orig = output_orig.path().to_str().unwrap();

                let output_leaflets = NamedTempFile::new().unwrap();
                let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

                let analysis = Analysis::builder()
                    .structure("tests/files/scrambling/cg_scrambling.tpr")
                    .trajectory("tests/files/scrambling/cg_scrambling.xtc")
                    .analysis_type(AnalysisType::cgorder("@membrane"))
                    .output_yaml(path_to_output_orig)
                    .leaflets(
                        LeafletClassification::global("@membrane", "name PO4")
                            .with_frequency(freq)
                            .with_collect(path_to_output_leaflets),
                    )
                    .step(step)
                    .n_threads(n_threads)
                    .silent()
                    .overwrite()
                    .build()
                    .unwrap();

                analysis.run().unwrap().write().unwrap();

                // rerun the analysis using the exported leaflet classification
                let output_recalc = NamedTempFile::new().unwrap();
                let path_to_output_recalc = output_recalc.path().to_str().unwrap();

                let analysis = Analysis::builder()
                    .structure("tests/files/scrambling/cg_scrambling.tpr")
                    .trajectory("tests/files/scrambling/cg_scrambling.xtc")
                    .analysis_type(AnalysisType::cgorder("@membrane"))
                    .output_yaml(path_to_output_recalc)
                    .leaflets(
                        LeafletClassification::from_file(path_to_output_leaflets)
                            .with_frequency(freq),
                    )
                    .step(step)
                    .n_threads(1) // always one thread
                    .silent()
                    .overwrite()
                    .build()
                    .unwrap();

                analysis.run().unwrap().write().unwrap();

                // order parameters should match exactly
                assert!(diff_files_ignore_first(
                    path_to_output_orig,
                    path_to_output_recalc,
                    1,
                ));
            }
        }
    }
}

#[test]
fn test_cg_order_leaflets_scrambling_from_map() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    for n_threads in [1, 2, 3, 5, 8, 128] {
        for (leaflets_file, freq) in ["every.yaml", "every10.yaml", "once.yaml"].into_iter().zip(
            [
                Frequency::every(1).unwrap(),
                Frequency::every(10).unwrap(),
                Frequency::once(),
            ]
            .into_iter(),
        ) {
            let filename = format!("tests/files/scrambling/leaflets_{}", leaflets_file);
            let mut file = File::open(&filename).unwrap();
            let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
                serde_yaml::from_reader(&mut file).unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/scrambling/cg_scrambling.tpr")
                .trajectory("tests/files/scrambling/cg_scrambling.xtc")
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .output_yaml(path_to_output)
                .leaflets(LeafletClassification::from_map(assignment).with_frequency(freq))
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            let test_file = match leaflets_file {
                "every.yaml" => "order_global.yaml",
                "every10.yaml" => "order_global_every_10.yaml",
                "once.yaml" => "order_once.yaml",
                _ => panic!("Unexpected leaflets file provided."),
            };

            assert_eq_order(
                path_to_output,
                &format!("tests/files/scrambling/{}", test_file),
                1,
            );
        }
    }
}

#[test]
fn test_cg_order_leaflets_scrambling_from_ndx() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    for n_threads in [1, 2, 3, 5, 8, 128] {
        for (ndx, freq) in [
            glob::glob("tests/files/scrambling/ndx/leaflets_frame_*.ndx")
                .unwrap()
                .filter_map(Result::ok)
                .map(|path| path.to_str().unwrap().to_owned())
                .collect(),
            vec![
                "tests/files/scrambling/ndx/leaflets_frame_000.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_010.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_020.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_030.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_040.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_050.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_060.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_070.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_080.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_090.ndx".to_owned(),
                "tests/files/scrambling/ndx/leaflets_frame_100.ndx".to_owned(),
            ],
            vec!["tests/files/scrambling/ndx/leaflets_frame_000.ndx".to_owned()],
        ]
        .into_iter()
        .zip(
            [
                Frequency::every(1).unwrap(),
                Frequency::every(10).unwrap(),
                Frequency::once(),
            ]
            .into_iter(),
        ) {
            let ndx_ref: Vec<&str> = ndx.iter().map(|s| s.as_str()).collect();

            let analysis = Analysis::builder()
                .structure("tests/files/scrambling/cg_scrambling.tpr")
                .trajectory("tests/files/scrambling/cg_scrambling.xtc")
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .output_yaml(path_to_output)
                .leaflets(
                    LeafletClassification::from_ndx(&ndx_ref, "name PO4", "Upper", "Lower")
                        .with_frequency(freq),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            let test_file = match freq {
                Frequency::Every(x) if x.get() == 1 => "order_global.yaml",
                Frequency::Every(x) if x.get() == 10 => "order_global_every_10.yaml",
                Frequency::Once => "order_once.yaml",
                _ => panic!("Unexpected frequency."),
            };

            assert_eq_order(
                path_to_output,
                &format!("tests/files/scrambling/{}", test_file),
                1,
            );
        }
    }
}

#[test]
fn test_cg_order_leaflets_asymmetric_multiple_threads() {
    for n_threads in [1, 2, 5, 8] {
        let output_yaml = NamedTempFile::new().unwrap();
        let path_to_yaml = output_yaml.path().to_str().unwrap();

        let output_tab = NamedTempFile::new().unwrap();
        let path_to_tab = output_tab.path().to_str().unwrap();

        let output_csv = NamedTempFile::new().unwrap();
        let path_to_csv = output_csv.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let xvg_pattern = format!("{}/order.xvg", path_to_dir);

        let analysis = Analysis::builder()
            .structure("tests/files/asymmetric/cg_asym.tpr")
            .trajectory("tests/files/asymmetric/cg_asym.xtc")
            .output_yaml(path_to_yaml)
            .output_tab(path_to_tab)
            .output_csv(path_to_csv)
            .output_xvg(xvg_pattern)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_yaml,
            "tests/files/asymmetric/cg_order_asymmetric.yaml",
            1,
        );

        assert_eq_order(
            path_to_tab,
            "tests/files/asymmetric/cg_order_asymmetric.tab",
            1,
        );

        assert_eq_csv(
            path_to_csv,
            "tests/files/asymmetric/cg_order_asymmetric.csv",
            0,
        );

        for molecule in ["POPE", "POPG"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!(
                "tests/files/asymmetric/cg_order_asymmetric_{}.xvg",
                molecule
            );

            assert_eq_order(&path, &path_expected, 1);
        }
    }
}

#[test]
fn test_cg_order_leaflets_asymmetric_error_multiple_threads() {
    for n_threads in [1, 2, 5, 8] {
        let output_yaml = NamedTempFile::new().unwrap();
        let path_to_yaml = output_yaml.path().to_str().unwrap();

        let output_tab = NamedTempFile::new().unwrap();
        let path_to_tab = output_tab.path().to_str().unwrap();

        let output_csv = NamedTempFile::new().unwrap();
        let path_to_csv = output_csv.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let xvg_pattern = format!("{}/order.xvg", path_to_dir);

        let analysis = Analysis::builder()
            .structure("tests/files/asymmetric/cg_asym.tpr")
            .trajectory("tests/files/asymmetric/cg_asym.xtc")
            .output_yaml(path_to_yaml)
            .output_tab(path_to_tab)
            .output_csv(path_to_csv)
            .output_xvg(xvg_pattern)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .estimate_error(EstimateError::default())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_yaml,
            "tests/files/asymmetric/cg_order_asymmetric_errors.yaml",
            1,
        );

        assert_eq_order(
            path_to_tab,
            "tests/files/asymmetric/cg_order_asymmetric_errors.tab",
            1,
        );

        assert_eq_csv(
            path_to_csv,
            "tests/files/asymmetric/cg_order_asymmetric_errors.csv",
            0,
        );

        for molecule in ["POPE", "POPG"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!(
                "tests/files/asymmetric/cg_order_asymmetric_{}.xvg",
                molecule
            );

            assert_eq_order(&path, &path_expected, 1);
        }
    }
}

#[test]
fn test_cg_order_leaflets_asymmetric_ordermaps_multiple_threads() {
    for n_threads in [1, 2, 5, 8] {
        let analysis = Analysis::builder()
            .structure("tests/files/asymmetric/cg_asym.tpr")
            .trajectory("tests/files/asymmetric/cg_asym.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane and name C1B C2B C3B C4B"))
            .leaflets(LeafletClassification::global("@membrane", "name PO4"))
            .ordermaps(OrderMap::builder().bin_size([1.0, 1.0]).build().unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let result = match analysis.run().unwrap() {
            AnalysisResults::CG(x) => x,
            _ => panic!("Incorrect results type returned."),
        };

        for molecule in result.molecules() {
            for bond in molecule.bonds() {
                let total = bond.ordermaps().total().as_ref().unwrap();
                let upper = bond.ordermaps().upper().as_ref().unwrap();
                let lower = bond.ordermaps().lower().as_ref().unwrap();

                if molecule.molecule() == "POPE" {
                    for ((_, _, t), (_, _, u)) in
                        total.extract_convert().zip(upper.extract_convert())
                    {
                        assert_relative_eq!(t, u, epsilon = 2e-4);
                    }

                    for (_, _, l) in lower.extract_convert() {
                        assert!(l.is_nan());
                    }
                } else if molecule.molecule() == "POPG" {
                    for ((_, _, t), (_, _, l)) in
                        total.extract_convert().zip(lower.extract_convert())
                    {
                        assert_relative_eq!(t, l, epsilon = 2e-4);
                    }

                    for (_, _, u) in upper.extract_convert() {
                        assert!(u.is_nan());
                    }
                } else {
                    panic!("Unexpected molecule.")
                }
            }
        }
    }
}

#[test]
fn test_cg_order_geometry_cuboid_full() {
    for reference in [
        GeomReference::default(),
        Vector3D::new(2.0, 2.0, 3.0).into(),
        "@membrane".into(),
        GeomReference::center(),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(
                Geometry::cuboid(
                    reference,
                    [f32::NEG_INFINITY, f32::INFINITY],
                    [f32::NEG_INFINITY, f32::INFINITY],
                    [f32::NEG_INFINITY, f32::INFINITY],
                )
                .unwrap(),
            )
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_basic.yaml", 1);
    }
}

#[test]
fn test_cg_order_geometry_cylinder_full() {
    for reference in [
        GeomReference::default(),
        Vector3D::new(2.0, 2.0, 3.0).into(),
        "@membrane".into(),
        GeomReference::center(),
    ] {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/cg.tpr")
                .trajectory("tests/files/cg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .geometry(
                    Geometry::cylinder(
                        reference.clone(),
                        f32::INFINITY,
                        [f32::NEG_INFINITY, f32::INFINITY],
                        axis,
                    )
                    .unwrap(),
                )
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(path_to_output, "tests/files/cg_order_basic.yaml", 1);
        }
    }
}

#[test]
fn test_cg_order_geometry_sphere_full() {
    for reference in [
        GeomReference::default(),
        Vector3D::new(2.0, 2.0, 3.0).into(),
        "@membrane".into(),
        GeomReference::center(),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(Geometry::sphere(reference, f32::INFINITY).unwrap())
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_basic.yaml", 1);
    }
}

#[test]
fn test_cg_order_geometry_cuboid_box_center_square_multiple_threads() {
    for n_threads in [1, 2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(
                Geometry::cuboid(
                    GeomReference::center(),
                    [-8.0, -2.0],
                    [2.0, 8.0],
                    [f32::NEG_INFINITY, f32::INFINITY],
                )
                .unwrap(),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_cuboid_square.yaml", 1);
    }
}

#[test]
fn test_cg_order_geometry_cylinder_static_multiple_threads() {
    for n_threads in [1, 2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(
                Geometry::cylinder(
                    Vector3D::new(2.0, 1.0, 0.0),
                    3.25,
                    [f32::NEG_INFINITY, f32::INFINITY],
                    Axis::Z,
                )
                .unwrap(),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_cylinder.yaml", 1);
    }
}

#[test]
fn test_cg_order_geometry_sphere_dynamic_multiple_threads() {
    for n_threads in [1, 2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(Geometry::sphere("resid 1", 2.5).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_sphere.yaml", 1);
    }
}

#[test]
fn test_cg_order_geometry_cuboid_z() {
    let analysis_geometry = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder(
            "@membrane and name PO4 NC3 NH3 GL0 GL1 GL2 C1A C1B",
        ))
        .geometry(
            Geometry::cuboid(
                "@membrane",
                [f32::NEG_INFINITY, f32::INFINITY],
                [f32::NEG_INFINITY, f32::INFINITY],
                [0.0, 3.5],
            )
            .unwrap(),
        )
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let analysis_leaflets = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder(
            "@membrane and name PO4 NC3 NH3 GL0 GL1 GL2 C1A C1B",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results_geometry = analysis_geometry.run().unwrap();
    let results_leaflets = analysis_leaflets.run().unwrap();

    let results_geometry = match results_geometry {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    let results_leaflets = match results_leaflets {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    for (mol, mol2) in results_geometry
        .molecules()
        .zip(results_leaflets.molecules())
    {
        for (bond, bond2) in mol.bonds().zip(mol2.bonds()) {
            assert_relative_eq!(
                bond.order().total().unwrap().value(),
                bond.order().upper().unwrap().value()
            );
            assert!(bond.order().lower().unwrap().value().is_nan());

            assert_relative_eq!(
                bond.order().total().unwrap().value(),
                bond2.order().upper().unwrap().value()
            );
        }
    }
}

#[test]
fn test_cg_order_geometry_cylinder_z() {
    let analysis_geometry = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder(
            "@membrane and name PO4 NC3 NH3 GL0 GL1 GL2 C1A C1B",
        ))
        .geometry(Geometry::cylinder("@membrane", f32::INFINITY, [0.0, 3.5], Axis::Z).unwrap())
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let analysis_leaflets = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder(
            "@membrane and name PO4 NC3 NH3 GL0 GL1 GL2 C1A C1B",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results_geometry = analysis_geometry.run().unwrap();
    let results_leaflets = analysis_leaflets.run().unwrap();

    let results_geometry = match results_geometry {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    let results_leaflets = match results_leaflets {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    for (mol, mol2) in results_geometry
        .molecules()
        .zip(results_leaflets.molecules())
    {
        for (bond, bond2) in mol.bonds().zip(mol2.bonds()) {
            assert_relative_eq!(
                bond.order().total().unwrap().value(),
                bond.order().upper().unwrap().value()
            );
            assert!(bond.order().lower().unwrap().value().is_nan());

            assert_relative_eq!(
                bond.order().total().unwrap().value(),
                bond2.order().upper().unwrap().value()
            );
        }
    }
}

#[test]
fn test_cg_order_leaflets_from_file_once_multiple_threads() {
    for n_threads in [1, 2, 5, 8, 16, 32] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output_yaml(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(
                LeafletClassification::from_file("tests/files/inputs/leaflets_files/cg_once.yaml")
                    .with_frequency(Frequency::once()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_from_map_once() {
    let mut file = File::open("tests/files/inputs/leaflets_files/cg_once.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_yaml(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::from_map(assignment).with_frequency(Frequency::once()))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
}

#[test]
fn test_cg_order_leaflets_from_file_every20_multiple_threads() {
    for n_threads in [1, 2, 5, 8, 16, 32] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output_yaml(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(
                LeafletClassification::from_file(
                    "tests/files/inputs/leaflets_files/cg_every20.yaml",
                )
                .with_frequency(Frequency::every(20).unwrap()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_from_map_every20() {
    let mut file = File::open("tests/files/inputs/leaflets_files/cg_every20.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_yaml(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(
            LeafletClassification::from_map(assignment)
                .with_frequency(Frequency::every(20).unwrap()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
}

#[test]
fn test_cg_order_leaflets_from_file_every_multiple_threads() {
    for n_threads in [1, 2, 5, 8, 16, 32] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output_yaml(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/cg_every.yaml",
            ))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_from_map_every() {
    let mut file = File::open("tests/files/inputs/leaflets_files/cg_every.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .output_yaml(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::from_map(assignment))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
}

#[test]
fn test_cg_order_leaflets_from_file_fail_missing_molecule_type() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/cg_missing_moltype.yaml",
            )
            .with_frequency(Frequency::once()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Run should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("not found in the leaflet assignment")),
    }
}

#[test]
fn test_cg_order_leaflets_from_map_fail_unexpected_molecule_type() {
    let mut file =
        File::open("tests/files/inputs/leaflets_files/cg_unexpected_moltype.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::from_map(assignment).with_frequency(Frequency::once()))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Run should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("specified in the leaflet assignment structure not found in the system")),
    }
}

#[test]
fn test_cg_order_leaflets_from_file_fail_nonexistent() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/cg_nonexistent.yaml",
            )
            .with_frequency(Frequency::once()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Run should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("could not open the leaflet assignment file")),
    }
}

#[test]
fn test_cg_order_leaflets_from_file_fail_invalid() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_invalid.yaml", // intentionally reading file for AA
            )
            .with_frequency(Frequency::once()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Run should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("could not understand the contents of the leaflet assignment file")),
    }
}

#[test]
fn test_cg_order_leaflets_from_map_fail_invalid_number_of_molecules() {
    let mut file = File::open("tests/files/inputs/leaflets_files/cg_invalid_number.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::from_map(assignment).with_frequency(Frequency::once()))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Run should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("inconsistent number of molecules specified in the leaflet assignment")),
    }
}

#[test]
fn test_cg_order_leaflets_from_file_fail_empty_assignment() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(
            LeafletClassification::from_file("tests/files/inputs/leaflets_files/cg_empty.yaml")
                .with_frequency(Frequency::once()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Run should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("no leaflet assignment data provided for molecule type")),
    }
}

#[test]
fn test_cg_order_leaflets_from_map_fail_too_many_frames() {
    let mut file = File::open("tests/files/inputs/leaflets_files/cg_every20.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(
            LeafletClassification::from_map(assignment)
                .with_frequency(Frequency::every(5).unwrap()),
        )
        .step(5)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Run should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("number of frames specified in the leaflet assignment structure")),
    }
}

#[test]
fn test_cg_order_leaflets_from_file_not_enough_frames() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(
            LeafletClassification::from_file("tests/files/inputs/leaflets_files/cg_every20.yaml")
                .with_frequency(Frequency::every(16).unwrap()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Run should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("could not get leaflet assignment for frame")),
    }
}

#[test]
fn test_cg_order_leaflets_no_pbc_multiple_threads() {
    for n_threads in [1, 3, 8, 32] {
        for structure in ["tests/files/cg.tpr", "tests/files/cg_nobox.pdb"] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure(structure)
                .trajectory("tests/files/cg_whole_nobox.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .leaflets(LeafletClassification::global("@membrane", "name PO4"))
                .handle_pbc(false)
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(
                path_to_output,
                "tests/files/cg_order_leaflets_nopbc.yaml",
                1,
            );
        }
    }
}

#[test]
fn test_cg_order_error_leaflets_no_pbc_multiple_threads() {
    for n_threads in [1, 3, 8, 32] {
        for structure in ["tests/files/cg.tpr", "tests/files/cg_nobox.pdb"] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure(structure)
                .trajectory("tests/files/cg_whole_nobox.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::cgorder("@membrane"))
                .leaflets(LeafletClassification::global("@membrane", "name PO4"))
                .handle_pbc(false)
                .estimate_error(EstimateError::default())
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(
                path_to_output,
                "tests/files/cg_order_error_leaflets_nopbc.yaml",
                1,
            );
        }
    }
}

#[test]
fn test_cg_order_maps_leaflets_no_pbc() {
    for method in [
        LeafletClassification::global("@membrane", "name PO4"),
        LeafletClassification::local("@membrane", "name PO4", 2.5),
        LeafletClassification::individual("name PO4", "name C4A C4B"),
    ] {
        for structure in ["tests/files/cg.tpr", "tests/files/cg_nobox.pdb"] {
            let directory = TempDir::new().unwrap();
            let path_to_dir = directory.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure(structure)
                .trajectory("tests/files/cg_whole_nobox.xtc")
                .analysis_type(AnalysisType::cgorder(
                    "resname POPC and name C1B C2B C3B C4B",
                ))
                .leaflets(method.clone())
                .map(
                    OrderMap::builder()
                        .bin_size([1.0, 1.0])
                        .output_directory(path_to_dir)
                        .min_samples(10)
                        .dim([
                            GridSpan::manual(0.0, 13.0).unwrap(),
                            GridSpan::manual(0.0, 13.0).unwrap(),
                        ])
                        .build()
                        .unwrap(),
                )
                .handle_pbc(false)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            let expected_file_names = [
                "ordermap_average_full.dat",
                "ordermap_average_upper.dat",
                "ordermap_average_lower.dat",
            ];

            for file in expected_file_names {
                let real_file = format!("{}/POPC/{}", path_to_dir, file);
                let test_file = format!("tests/files/ordermaps_cg_nopbc/{}", file);
                assert_eq_maps(&real_file, &test_file, 2);
            }

            // full maps for the entire system are the same as for POPC
            for file in [
                "ordermap_average_full.dat",
                "ordermap_average_upper.dat",
                "ordermap_average_lower.dat",
            ] {
                let real_file = format!("{}/{}", path_to_dir, file);
                let test_file = format!("tests/files/ordermaps_cg_nopbc/{}", file);
                assert_eq_maps(&real_file, &test_file, 2);
            }

            // check the script
            let real_script = format!("{}/plot.py", path_to_dir);
            assert!(common::diff_files_ignore_first(
                &real_script,
                "scripts/plot.py",
                0
            ));
        }
    }
}

#[test]
fn test_cg_order_geometry_cuboid_ordermaps_no_pbc() {
    for structure in ["tests/files/cg.tpr", "tests/files/cg_nobox.pdb"] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure(structure)
            .trajectory("tests/files/cg_whole_nobox.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(
                Geometry::cuboid(
                    [-2.0, 11.0, 0.0],
                    [-2.0, 8.0],
                    [f32::NEG_INFINITY, f32::INFINITY],
                    [f32::NEG_INFINITY, f32::INFINITY],
                )
                .unwrap(),
            )
            .map(
                OrderMap::builder()
                    .bin_size([1.0, 1.0])
                    .output_directory(path_to_dir)
                    .dim([
                        GridSpan::manual(0.0, 12.7).unwrap(),
                        GridSpan::manual(0.0, 14.0).unwrap(),
                    ])
                    .build()
                    .unwrap(),
            )
            .handle_pbc(false)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_cuboid_nopbc.yaml", 1);

        let real_file = format!("{}/{}", path_to_dir, "ordermap_average_full.dat");
        let test_file = "tests/files/ordermaps_cg_nopbc/cuboid.dat";
        assert_eq_maps(&real_file, test_file, 2);
    }
}

#[test]
fn test_cg_order_geometry_cylinder_ordermaps_no_pbc() {
    for structure in ["tests/files/cg.tpr", "tests/files/cg_nobox.pdb"] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure(structure)
            .trajectory("tests/files/cg_whole_nobox.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(
                Geometry::cylinder(
                    [10.0, 12.0, 0.0],
                    4.0,
                    [f32::NEG_INFINITY, f32::INFINITY],
                    Axis::Z,
                )
                .unwrap(),
            )
            .map(
                OrderMap::builder()
                    .bin_size([1.0, 1.0])
                    .output_directory(path_to_dir)
                    .dim([
                        GridSpan::manual(0.0, 12.7).unwrap(),
                        GridSpan::manual(0.0, 12.7).unwrap(),
                    ])
                    .build()
                    .unwrap(),
            )
            .handle_pbc(false)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/cg_order_cylinder_nopbc.yaml",
            1,
        );

        let real_file = format!("{}/{}", path_to_dir, "ordermap_average_full.dat");
        let test_file = "tests/files/ordermaps_cg_nopbc/cylinder.dat";
        assert_eq_maps(&real_file, test_file, 2);
    }
}

#[test]
fn test_cg_order_geometry_sphere_ordermaps_no_pbc() {
    for structure in ["tests/files/cg.tpr", "tests/files/cg_nobox.pdb"] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure(structure)
            .trajectory("tests/files/cg_whole_nobox.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .geometry(Geometry::sphere([10.0, 12.0, 5.5], 4.0).unwrap())
            .map(
                OrderMap::builder()
                    .bin_size([1.0, 1.0])
                    .output_directory(path_to_dir)
                    .dim([
                        GridSpan::manual(0.0, 12.7).unwrap(),
                        GridSpan::manual(0.0, 12.7).unwrap(),
                    ])
                    .build()
                    .unwrap(),
            )
            .handle_pbc(false)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_sphere_nopbc.yaml", 1);

        let real_file = format!("{}/{}", path_to_dir, "ordermap_average_full.dat");
        let test_file = "tests/files/ordermaps_cg_nopbc/sphere.dat";
        assert_eq_maps(&real_file, test_file, 2);
    }
}

#[test]
fn test_cg_order_leaflets_dynamic_membrane_normal_yaml() {
    for n_threads in [1, 3, 8, 32] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(
                LeafletClassification::individual("name PO4", "name C4A C4B")
                    .with_membrane_normal(Axis::Z)
                    .with_frequency(Frequency::once()),
            )
            .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/cg_order_leaflets_dynamic.yaml",
            1,
        );
    }
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normal_yaml() {
    for n_threads in [1, 3, 8, 16] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/vesicle.tpr")
            .trajectory("tests/files/vesicle.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder(
                "name C1A D2A C3A C4A C1B C2B C3B C4B",
            ))
            .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_vesicle.yaml", 1);
    }
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normal_spherical_clustering_yaml() {
    for n_threads in [1, 3, 8, 16] {
        for freq in [
            Frequency::once(),
            Frequency::every(1).unwrap(),
            Frequency::every(5).unwrap(),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/vesicle.tpr")
                .trajectory("tests/files/vesicle.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::cgorder(
                    "name C1A D2A C3A C4A C1B C2B C3B C4B",
                ))
                .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
                .leaflets(
                    LeafletClassification::spherical_clustering("name PO4").with_frequency(freq),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(
                path_to_output,
                "tests/files/cg_order_vesicle_leaflets.yaml",
                1,
            );
        }
    }
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normal_spherical_clustering_yaml_flip() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .leaflets(LeafletClassification::spherical_clustering("name PO4").with_flip(true))
        .n_threads(1)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_vesicle_leaflets_flipped.yaml",
        1,
    );
}

#[test]
fn test_cg_order_centered_vesicle_dynamic_membrane_normal_spherical_clustering_yaml() {
    for n_threads in [1, 3, 8, 16] {
        for pbc in [true, false] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/vesicle.tpr")
                .trajectory("tests/files/vesicle_centered.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::cgorder(
                    "name C1A D2A C3A C4A C1B C2B C3B C4B",
                ))
                .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
                .leaflets(LeafletClassification::spherical_clustering("name PO4"))
                .handle_pbc(pbc)
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(
                path_to_output,
                "tests/files/cg_order_vesicle_leaflets.yaml",
                1,
            );
        }
    }
}

#[test]
fn test_cg_order_vesicle_membrane_normals_from_file_yaml() {
    for n_threads in [1, 3, 8, 16] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/vesicle.tpr")
            .trajectory("tests/files/vesicle.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder(
                "name C1A D2A C3A C4A C1B C2B C3B C4B",
            ))
            .membrane_normal("tests/files/normals_vesicle.yaml")
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_vesicle.yaml", 1);
    }
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normals_export() {
    for n_threads in [1, 3, 8, 16] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let output_normals = NamedTempFile::new().unwrap();
        let path_to_output_normals = output_normals.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/vesicle.tpr")
            .trajectory("tests/files/vesicle.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder(
                "name C1A D2A C3A C4A C1B C2B C3B C4B",
            ))
            .membrane_normal(
                DynamicNormal::new("name PO4", 2.0)
                    .unwrap()
                    .with_collect(path_to_output_normals),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_vesicle.yaml", 1);
        assert_eq_normals(path_to_output_normals, "tests/files/normals_vesicle.yaml");
    }
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normals_leflets_export() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let output_normals = NamedTempFile::new().unwrap();
    let path_to_output_normals = output_normals.path().to_str().unwrap();

    let output_leaflets = NamedTempFile::new().unwrap();
    let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(
            DynamicNormal::new("name PO4", 2.0)
                .unwrap()
                .with_collect(path_to_output_normals),
        )
        .leaflets(
            LeafletClassification::clustering("name PO4")
                .with_frequency(Frequency::once())
                .with_collect(path_to_output_leaflets),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_vesicle_leaflets.yaml",
        1,
    );
    assert_eq_normals(path_to_output_normals, "tests/files/normals_vesicle.yaml");
    assert!(diff_files_ignore_first(
        path_to_output_leaflets,
        "tests/files/leaflets_vesicle.yaml",
        1
    ));
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normals_spherical_clustering_leflets_export() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let output_normals = NamedTempFile::new().unwrap();
    let path_to_output_normals = output_normals.path().to_str().unwrap();

    let output_leaflets = NamedTempFile::new().unwrap();
    let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(
            DynamicNormal::new("name PO4", 2.0)
                .unwrap()
                .with_collect(path_to_output_normals),
        )
        .leaflets(
            LeafletClassification::spherical_clustering("name PO4")
                .with_collect(path_to_output_leaflets),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_vesicle_leaflets.yaml",
        1,
    );
    assert_eq_normals(path_to_output_normals, "tests/files/normals_vesicle.yaml");
    assert!(diff_files_ignore_first(
        path_to_output_leaflets,
        "tests/files/leaflets_vesicle_all.yaml",
        1
    ));
}

#[test]
fn test_cg_order_vesicle_membrane_normals_from_file_fail_unmatching_molecules() {
    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal("tests/files/normals_unmatching.yaml")
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("inconsistent number of molecules specified in the normals structure")),
    }
}

#[test]
fn test_cg_order_vesicle_normals_from_file_fail_missing_molecule_type() {
    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal("tests/files/normals_missing.yaml")
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("not found in the manual normals structure")),
    }
}

#[test]
fn test_cg_order_vesicle_normals_from_file_fail_empty_molecule_type() {
    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal("tests/files/normals_empty.yaml")
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("no membrane normals provided for molecule type")),
    }
}

#[test]
fn test_cg_order_vesicle_membrane_normals_from_file_fail_too_many_frames() {
    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal("tests/files/normals_vesicle.yaml")
        .begin(2600000.0)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("number of frames specified in the normals structure")),
    }
}

#[test]
fn test_cg_order_vesicle_membrane_normals_from_file_fail_nonexistent_file() {
    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal("tests/files/normals_nonexistent.yaml")
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e.to_string().contains("could not open the normals file")),
    }
}

#[test]
fn test_cg_order_vesicle_membrane_normals_from_map_yaml() {
    let string = read_to_string("tests/files/normals_vesicle.yaml").unwrap();
    let normals_map: HashMap<String, Vec<Vec<Vector3D>>> = serde_yaml::from_str(&string).unwrap();

    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(MembraneNormal::from(normals_map))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_vesicle.yaml", 1);
}

#[test]
fn test_cg_order_vesicle_leaflets_dynamic_membrane_normal_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .leaflets(
            LeafletClassification::from_ndx(
                &["tests/files/vesicle.ndx"],
                "name PO4",
                "UpperLeaflet",
                "LowerLeaflet",
            )
            .with_frequency(Frequency::once()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_vesicle_leaflets.yaml",
        1,
    );
}

#[test]
fn test_cg_order_vesicle_leaflets_clustering_dynamic_membrane_normal_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .leaflets(LeafletClassification::clustering("name PO4").with_frequency(Frequency::once()))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_vesicle_leaflets.yaml",
        1,
    );
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normal_centered_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle_centered.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_vesicle_centered.yaml", // very small differences compared to `cg_order_vesicle.yaml`
        1,
    );
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normal_centered_nopbc_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle_centered.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .handle_pbc(false)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_vesicle_centered.yaml",
        1,
    );
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normal_centered_clustering_nopbc_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle_centered.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .leaflets(LeafletClassification::clustering("name PO4").with_frequency(Frequency::once()))
        .handle_pbc(false)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_vesicle_leaflets_centered.yaml",
        1,
    );
}

#[test]
fn test_cg_order_buckled_dynamic_membrane_normal_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg_buckled.tpr")
        .trajectory("tests/files/cg_buckled.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/cg_order_buckled.yaml", 1);
}

#[test]
fn test_cg_order_buckled_leaflets_clustering_dynamic_membrane_normal_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/cg_buckled.tpr")
        .trajectory("tests/files/cg_buckled.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .leaflets(LeafletClassification::clustering("name PO4"))
        .n_threads(4)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/cg_order_buckled_leaflets.yaml",
        1,
    );
}

#[test]
fn test_cg_order_fail_dynamic_undefined_ordermap_plane() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .map(
            OrderMap::builder()
                .bin_size([1.0, 1.0])
                .min_samples(5)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("unable to automatically set ordermap plane")),
    }
}

#[test]
fn test_cg_order_fail_dynamic_undefined_leaflet_normal() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .membrane_normal(DynamicNormal::new("name PO4", 2.0).unwrap())
        .leaflets(LeafletClassification::individual(
            "name PO4",
            "name C4A C4B",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e.to_string().contains("leaflet classification requires it")),
    }
}

#[test]
fn test_cg_order_fail_dynamic_multiple_heads() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .membrane_normal(DynamicNormal::new("name PO4 NC3", 2.0).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e.to_string().contains("multiple head group atoms")),
    }
}

#[test]
fn test_cg_order_fail_dynamic_no_head() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .membrane_normal(DynamicNormal::new("name W", 2.0).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e.to_string().contains("no head group atom")),
    }
}

#[test]
fn test_cg_order_leaflets_from_ndx_once_multiple_threads() {
    for n_threads in [1, 3, 5, 8, 16, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(
                LeafletClassification::from_ndx(
                    &["tests/files/ndx/cg_leaflets.ndx"],
                    "name PO4",
                    "Upper",
                    "Lower",
                )
                .with_frequency(Frequency::once()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_from_ndx_every_multiple_threads() {
    let mut ndx = [
        "tests/files/ndx/cg_leaflets.ndx",
        "tests/files/ndx/cg_leaflets_all.ndx",
    ]
    .repeat(50);
    ndx.push("tests/files/ndx/cg_leaflets.ndx");

    for n_threads in [1, 3, 5, 8, 16, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(LeafletClassification::from_ndx(
                &ndx, "name PO4", "Upper", "Lower",
            ))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_from_ndx_every20_multiple_threads() {
    let ndx = vec![
        "tests/files/ndx/cg_leaflets.ndx",
        "tests/files/ndx/cg_leaflets_all.ndx",
        "tests/files/ndx/cg_leaflets_duplicate_irrelevant.ndx",
        "tests/files/ndx/cg_leaflets_invalid_irrelevant.ndx",
        "tests/files/ndx/cg_leaflets.ndx",
        "tests/files/ndx/cg_leaflets_all.ndx",
    ];

    for n_threads in [1, 3, 5, 8, 16, 128] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(
                LeafletClassification::from_ndx(&ndx, "name PO4", "Upper", "Lower")
                    .with_frequency(Frequency::every(20).unwrap()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/cg_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_cg_order_leaflets_from_ndx_partial() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    for (freq, ndx) in [
        Frequency::once(),
        Frequency::every(1).unwrap(),
        Frequency::every(20).unwrap(),
    ]
    .into_iter()
    .zip([
        vec!["tests/files/ndx/cg_leaflets.ndx"],
        ["tests/files/ndx/cg_leaflets.ndx"].repeat(101),
        ["tests/files/ndx/cg_leaflets.ndx"].repeat(6),
    ]) {
        let analysis = Analysis::builder()
            .structure("tests/files/cg.tpr")
            .trajectory("tests/files/cg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::cgorder(
                "resname POPC and name C1B C2B C3B C4B",
            ))
            .leaflets(
                LeafletClassification::from_ndx(&ndx, "name PO4", "Upper", "Lower")
                    .with_frequency(freq),
            )
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/cg_order_leaflets_small.yaml",
            1,
        );
    }
}

#[test]
fn test_cg_order_basic_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 101);
    assert_eq!(results.analysis().structure(), "tests/files/cg.tpr");

    assert_eq!(results.molecules().count(), 3);

    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPE").is_some());
    assert!(results.get_molecule("POPG").is_some());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.2962,
        epsilon = 2e-4
    );
    assert!(results.average_order().upper().is_none());
    assert!(results.average_order().lower().is_none());

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_molecule_names = ["POPC", "POPE", "POPG"];
    let expected_average_orders = [0.2943, 0.2972, 0.3059];
    let expected_bond_orders = [0.3682, 0.3759, 0.3789];

    for (i, molecule) in results.molecules().enumerate() {
        assert_eq!(molecule.molecule(), expected_molecule_names[i]);

        let average_order = molecule.average_order();
        assert_relative_eq!(
            average_order.total().unwrap().value(),
            expected_average_orders[i],
            epsilon = 2e-4
        );
        assert!(average_order.total().unwrap().error().is_none());
        assert!(average_order.upper().is_none());
        assert!(average_order.lower().is_none());

        let average_maps = molecule.average_ordermaps();
        assert!(average_maps.total().is_none());
        assert!(average_maps.upper().is_none());
        assert!(average_maps.lower().is_none());

        // bonds
        assert_eq!(molecule.bonds().count(), 11);

        let bond = molecule.get_bond(4, 5).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.atom_name(), "C1A");
        assert_eq!(a1.relative_index(), 4);
        assert_eq!(a1.residue_name(), expected_molecule_names[i]);
        assert_eq!(a2.atom_name(), "D2A");
        assert_eq!(a2.relative_index(), 5);
        assert_eq!(a2.residue_name(), expected_molecule_names[i]);

        let order = bond.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_bond_orders[i],
            epsilon = 2e-4
        );
        assert!(order.total().unwrap().error().is_none());
        assert!(order.upper().is_none());
        assert!(order.lower().is_none());

        let maps = bond.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // the same bond
        let bond = molecule.get_bond(5, 4).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), 4);
        assert_eq!(a2.relative_index(), 5);

        // nonexistent bond
        assert!(molecule.get_bond(1, 3).is_none());
        assert!(molecule.get_bond(15, 16).is_none());
    }
}

#[test]
fn test_cg_order_error_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 101);
    assert!(results
        .analysis()
        .estimate_error()
        .as_ref()
        .unwrap()
        .output_convergence()
        .is_none());

    assert_eq!(results.molecules().count(), 3);

    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPE").is_some());
    assert!(results.get_molecule("POPG").is_some());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.2962,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().total().unwrap().error().unwrap(),
        0.0050,
        epsilon = 2e-4
    );
    assert!(results.average_order().upper().is_none());
    assert!(results.average_order().lower().is_none());

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_molecule_names = ["POPC", "POPE", "POPG"];
    let expected_average_orders = [0.2943, 0.2972, 0.3059];
    let expected_average_errors = [0.0067, 0.0052, 0.0089];

    let expected_bond_orders = [0.3682, 0.3759, 0.3789];
    let expected_bond_errors = [0.0125, 0.0164, 0.0159];

    let expected_convergence_frames = (1..=101).collect::<Vec<usize>>();
    let expected_convergence_values = [
        [0.2756, 0.2902, 0.2943],
        [0.2830, 0.2995, 0.2972],
        [0.3198, 0.3066, 0.3059],
    ];

    for (i, molecule) in results.molecules().enumerate() {
        assert_eq!(molecule.molecule(), expected_molecule_names[i]);

        let average_order = molecule.average_order();
        assert_relative_eq!(
            average_order.total().unwrap().value(),
            expected_average_orders[i],
            epsilon = 2e-4
        );
        assert_relative_eq!(
            average_order.total().unwrap().error().unwrap(),
            expected_average_errors[i],
            epsilon = 2e-4
        );
        assert!(average_order.upper().is_none());
        assert!(average_order.lower().is_none());

        let average_maps = molecule.average_ordermaps();
        assert!(average_maps.total().is_none());
        assert!(average_maps.upper().is_none());
        assert!(average_maps.lower().is_none());

        // convergence (available even if `output_convergence` is not specified)
        let convergence = molecule.convergence().unwrap();
        assert_eq!(
            convergence.frames().len(),
            expected_convergence_frames.len()
        );
        for (val, exp) in convergence
            .frames()
            .iter()
            .zip(expected_convergence_frames.iter())
        {
            assert_eq!(val, exp);
        }

        for (j, frame) in [0, 50, 100].into_iter().enumerate() {
            assert_relative_eq!(
                *convergence.total().as_ref().unwrap().get(frame).unwrap(),
                expected_convergence_values.get(i).unwrap().get(j).unwrap(),
                epsilon = 2e-4
            );
        }

        assert!(convergence.upper().is_none());
        assert!(convergence.lower().is_none());

        // bonds
        assert_eq!(molecule.bonds().count(), 11);

        let bond = molecule.get_bond(4, 5).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.atom_name(), "C1A");
        assert_eq!(a1.relative_index(), 4);
        assert_eq!(a1.residue_name(), expected_molecule_names[i]);
        assert_eq!(a2.atom_name(), "D2A");
        assert_eq!(a2.relative_index(), 5);
        assert_eq!(a2.residue_name(), expected_molecule_names[i]);

        let order = bond.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_bond_orders[i],
            epsilon = 2e-4
        );
        assert_relative_eq!(
            order.total().unwrap().error().unwrap(),
            expected_bond_errors[i],
            epsilon = 2e-4
        );
        assert!(order.upper().is_none());
        assert!(order.lower().is_none());

        let maps = bond.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // the same bond
        let bond = molecule.get_bond(5, 4).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), 4);
        assert_eq!(a2.relative_index(), 5);

        // nonexistent bond
        assert!(molecule.get_bond(1, 3).is_none());
        assert!(molecule.get_bond(15, 16).is_none());
    }
}

#[test]
fn test_cg_order_leaflets_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 101);
    assert_eq!(results.molecules().count(), 3);

    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPE").is_some());
    assert!(results.get_molecule("POPG").is_some());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.2962,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().value(),
        0.2971,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().lower().unwrap().value(),
        0.2954,
        epsilon = 2e-4
    );

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_molecule_names = ["POPC", "POPE", "POPG"];
    let expected_average_orders = [0.2943, 0.2972, 0.3059];
    let expected_average_upper = [0.2965, 0.2965, 0.3085];
    let expected_average_lower = [0.2920, 0.2980, 0.3033];

    let expected_bond_orders = [0.3682, 0.3759, 0.3789];
    let expected_bond_upper = [0.3647, 0.3713, 0.4129];
    let expected_bond_lower = [0.3717, 0.3806, 0.3449];

    for (i, molecule) in results.molecules().enumerate() {
        assert_eq!(molecule.molecule(), expected_molecule_names[i]);

        let average_order = molecule.average_order();
        assert_relative_eq!(
            average_order.total().unwrap().value(),
            expected_average_orders[i],
            epsilon = 2e-4
        );
        assert!(average_order.total().unwrap().error().is_none());

        assert_relative_eq!(
            average_order.upper().unwrap().value(),
            expected_average_upper[i],
            epsilon = 2e-4
        );
        assert!(average_order.upper().unwrap().error().is_none());

        assert_relative_eq!(
            average_order.lower().unwrap().value(),
            expected_average_lower[i],
            epsilon = 2e-4
        );
        assert!(average_order.lower().unwrap().error().is_none());

        let average_maps = molecule.average_ordermaps();
        assert!(average_maps.total().is_none());
        assert!(average_maps.upper().is_none());
        assert!(average_maps.lower().is_none());

        // bonds
        assert_eq!(molecule.bonds().count(), 11);

        let bond = molecule.get_bond(4, 5).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.atom_name(), "C1A");
        assert_eq!(a1.relative_index(), 4);
        assert_eq!(a1.residue_name(), expected_molecule_names[i]);
        assert_eq!(a2.atom_name(), "D2A");
        assert_eq!(a2.relative_index(), 5);
        assert_eq!(a2.residue_name(), expected_molecule_names[i]);

        let order = bond.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_bond_orders[i],
            epsilon = 2e-4
        );
        assert!(order.total().unwrap().error().is_none());

        assert_relative_eq!(
            order.upper().unwrap().value(),
            expected_bond_upper[i],
            epsilon = 2e-4
        );
        assert!(order.upper().unwrap().error().is_none());

        assert_relative_eq!(
            order.lower().unwrap().value(),
            expected_bond_lower[i],
            epsilon = 2e-4
        );
        assert!(order.lower().unwrap().error().is_none());

        let maps = bond.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // the same bond
        let bond = molecule.get_bond(5, 4).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), 4);
        assert_eq!(a2.relative_index(), 5);

        // nonexistent bond
        assert!(molecule.get_bond(1, 3).is_none());
        assert!(molecule.get_bond(15, 16).is_none());
    }
}

#[test]
fn test_cg_order_error_leaflets_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder("@membrane"))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 101);
    assert!(results
        .analysis()
        .estimate_error()
        .as_ref()
        .unwrap()
        .output_convergence()
        .is_none());

    assert_eq!(results.molecules().count(), 3);

    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPE").is_some());
    assert!(results.get_molecule("POPG").is_some());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.2962,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().total().unwrap().error().unwrap(),
        0.0050,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().value(),
        0.2971,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().error().unwrap(),
        0.0049,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().lower().unwrap().value(),
        0.2954,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().lower().unwrap().error().unwrap(),
        0.0056,
        epsilon = 2e-4
    );

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_molecule_names = ["POPC", "POPE", "POPG"];

    for (i, molecule) in results.molecules().enumerate() {
        assert_eq!(molecule.molecule(), expected_molecule_names[i]);

        let average_order = molecule.average_order();
        assert!(average_order.total().unwrap().error().is_some());
        assert!(average_order.upper().unwrap().error().is_some());
        assert!(average_order.lower().unwrap().error().is_some());

        let average_maps = molecule.average_ordermaps();
        assert!(average_maps.total().is_none());
        assert!(average_maps.upper().is_none());
        assert!(average_maps.lower().is_none());

        // convergence (available even if `output_convergence` is not specified)
        let convergence = molecule.convergence().unwrap();
        assert_eq!(convergence.frames().len(), 101);
        assert!(convergence.total().is_some());
        assert!(convergence.upper().is_some());
        assert!(convergence.lower().is_some());

        // bonds
        assert_eq!(molecule.bonds().count(), 11);

        let bond = molecule.get_bond(4, 5).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.atom_name(), "C1A");
        assert_eq!(a1.relative_index(), 4);
        assert_eq!(a1.residue_name(), expected_molecule_names[i]);
        assert_eq!(a2.atom_name(), "D2A");
        assert_eq!(a2.relative_index(), 5);
        assert_eq!(a2.residue_name(), expected_molecule_names[i]);

        let order = bond.order();
        assert!(order.total().unwrap().error().is_some());
        assert!(order.upper().unwrap().error().is_some());
        assert!(order.lower().unwrap().error().is_some());

        let maps = bond.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // the same bond
        let bond = molecule.get_bond(5, 4).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), 4);
        assert_eq!(a2.relative_index(), 5);

        // nonexistent bond
        assert!(molecule.get_bond(1, 3).is_none());
        assert!(molecule.get_bond(15, 16).is_none());
    }
}

#[test]
fn test_cg_order_ordermaps_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder(
            "resname POPC and name C1B C2B C3B C4B",
        ))
        .map(
            OrderMap::builder()
                .bin_size([1.0, 1.0])
                .min_samples(10)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 101);
    assert_eq!(results.molecules().count(), 1);

    // average ordermaps for the entire system
    assert!(results.average_ordermaps().total().is_some());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    // average ordermaps for the entire molecule
    let molecule = results.get_molecule("POPC").unwrap();
    let map = molecule.average_ordermaps().total().as_ref().unwrap();
    assert!(molecule.average_ordermaps().upper().is_none());
    assert!(molecule.average_ordermaps().lower().is_none());

    let span_x = map.span_x();
    let span_y = map.span_y();
    let bin = map.tile_dim();

    assert_relative_eq!(span_x.0, 0.0);
    assert_relative_eq!(span_x.1, 12.747616);
    assert_relative_eq!(span_y.0, 0.0);
    assert_relative_eq!(span_y.1, 12.747616);
    assert_relative_eq!(bin.0, 1.0);
    assert_relative_eq!(bin.1, 1.0);

    assert_relative_eq!(
        map.get_at_convert(1.0, 8.0).unwrap(),
        0.3590,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(7.0, 0.0).unwrap(),
        0.3765,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(13.0, 11.0).unwrap(),
        0.4296,
        epsilon = 2e-4
    );

    // ordermaps for a selected bond
    let bond = molecule.get_bond(9, 10).unwrap();
    let map = bond.ordermaps().total().as_ref().unwrap();
    assert!(bond.ordermaps().upper().is_none());
    assert!(bond.ordermaps().lower().is_none());

    assert_relative_eq!(
        map.get_at_convert(1.0, 8.0).unwrap(),
        0.3967,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(7.0, 0.0).unwrap(),
        0.3213,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(13.0, 11.0).unwrap(),
        0.4104,
        epsilon = 2e-4
    );
}

#[test]
fn test_cg_order_ordermaps_leaflets_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/cg.tpr")
        .trajectory("tests/files/cg.xtc")
        .analysis_type(AnalysisType::cgorder(
            "resname POPC and name C1B C2B C3B C4B",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name PO4"))
        .map(
            OrderMap::builder()
                .bin_size([1.0, 1.0])
                .min_samples(10)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::CG(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 101);
    assert_eq!(results.molecules().count(), 1);

    // average ordermaps for the entire molecule
    let molecule = results.get_molecule("POPC").unwrap();
    let total = molecule.average_ordermaps().total().as_ref().unwrap();

    let span_x = total.span_x();
    let span_y = total.span_y();
    let bin = total.tile_dim();

    assert_relative_eq!(span_x.0, 0.0);
    assert_relative_eq!(span_x.1, 12.747616);
    assert_relative_eq!(span_y.0, 0.0);
    assert_relative_eq!(span_y.1, 12.747616);
    assert_relative_eq!(bin.0, 1.0);
    assert_relative_eq!(bin.1, 1.0);

    assert_relative_eq!(
        total.get_at_convert(1.0, 8.0).unwrap(),
        0.3590,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        total.get_at_convert(13.0, 11.0).unwrap(),
        0.4296,
        epsilon = 2e-4
    );

    let upper = molecule.average_ordermaps().upper().as_ref().unwrap();

    assert_relative_eq!(
        upper.get_at_convert(1.0, 8.0).unwrap(),
        0.3418,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        upper.get_at_convert(13.0, 11.0).unwrap(),
        0.4051,
        epsilon = 2e-4
    );

    let lower = molecule.average_ordermaps().lower().as_ref().unwrap();

    assert_relative_eq!(
        lower.get_at_convert(1.0, 8.0).unwrap(),
        0.3662,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        lower.get_at_convert(13.0, 11.0).unwrap(),
        0.4506,
        epsilon = 2e-4
    );

    // ordermaps for a selected bond
    let bond = molecule.get_bond(9, 10).unwrap();
    let total = bond.ordermaps().total().as_ref().unwrap();

    assert_relative_eq!(
        total.get_at_convert(1.0, 8.0).unwrap(),
        0.3967,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        total.get_at_convert(13.0, 11.0).unwrap(),
        0.4104,
        epsilon = 2e-4
    );

    let upper = bond.ordermaps().upper().as_ref().unwrap();

    assert_relative_eq!(
        upper.get_at_convert(1.0, 8.0).unwrap(),
        0.3573,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        upper.get_at_convert(13.0, 11.0).unwrap(),
        0.4807,
        epsilon = 2e-4
    );

    let lower = bond.ordermaps().lower().as_ref().unwrap();

    assert_relative_eq!(
        lower.get_at_convert(1.0, 8.0).unwrap(),
        0.4118,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        lower.get_at_convert(13.0, 11.0).unwrap(),
        0.3563,
        epsilon = 2e-4
    );
}

#[test]
fn test_cg_order_vesicle_dynamic_membrane_normals_collect_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/vesicle.tpr")
        .trajectory("tests/files/vesicle.xtc")
        .analysis_type(AnalysisType::cgorder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B",
        ))
        .membrane_normal(
            DynamicNormal::new("name PO4", 2.0)
                .unwrap()
                .with_collect(true),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = analysis.run().unwrap();
    results.write().unwrap(); // should not write out anything

    let normals = match &results {
        AnalysisResults::CG(x) => x.normals_data().as_ref().unwrap(),
        _ => panic!("Invalid results type returned."),
    };

    let reference_content = std::fs::read_to_string("tests/files/normals_vesicle.yaml").unwrap();
    let reference_normals: IndexMap<String, Vec<Vec<Vector3D>>> =
        serde_yaml::from_str(&reference_content).unwrap();

    for moltype in reference_normals.keys() {
        for (frame_a, frame_b) in reference_normals
            .get(moltype)
            .unwrap()
            .iter()
            .zip(normals.get_molecule(moltype).unwrap().iter())
        {
            assert_eq!(frame_a.len(), frame_b.len());
            for (mol_a, mol_b) in frame_a.iter().zip(frame_b.iter()) {
                assert_relative_eq!(mol_a.x, mol_b.x, epsilon = 1e-5);
                assert_relative_eq!(mol_a.y, mol_b.y, epsilon = 1e-5);
                assert_relative_eq!(mol_a.z, mol_b.z, epsilon = 1e-5);
            }
        }
    }
}

#[test]
fn test_cg_order_leaflets_scrambling_flip_rust_api() {
    for leaflets in [
        LeafletClassification::global("@membrane", "name PO4"),
        LeafletClassification::local("@membrane", "name PO4", 3.0),
        LeafletClassification::individual("name PO4", "name C4A C4B"),
        LeafletClassification::clustering("name PO4").with_frequency(Frequency::every(10).unwrap()),
    ] {
        // unflipped analysis run
        let analysis = Analysis::builder()
            .structure("tests/files/scrambling/cg_scrambling.tpr")
            .trajectory("tests/files/scrambling/cg_scrambling.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(leaflets.clone().with_collect(true))
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let unflipped_results = match analysis.run().unwrap() {
            AnalysisResults::CG(x) => x,
            _ => panic!("Invalid results."),
        };

        // flipped analysis run
        let analysis = Analysis::builder()
            .structure("tests/files/scrambling/cg_scrambling.tpr")
            .trajectory("tests/files/scrambling/cg_scrambling.xtc")
            .analysis_type(AnalysisType::cgorder("@membrane"))
            .leaflets(leaflets.with_collect(true).with_flip(true))
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let flipped_results = match analysis.run().unwrap() {
            AnalysisResults::CG(x) => x,
            _ => panic!("Invalid results."),
        };

        // compare assignment data
        let unflipped_leaflets = unflipped_results
            .leaflets_data()
            .as_ref()
            .unwrap()
            .get_molecule("POPC")
            .unwrap();
        let flipped_leaflets = flipped_results
            .leaflets_data()
            .as_ref()
            .unwrap()
            .get_molecule("POPC")
            .unwrap();

        assert_eq!(unflipped_leaflets.len(), flipped_leaflets.len());
        for (unflipped_frame, flipped_frame) in
            unflipped_leaflets.iter().zip(flipped_leaflets.iter())
        {
            assert_eq!(unflipped_frame.len(), flipped_frame.len());
            for (unflipped, flipped) in unflipped_frame.iter().zip(flipped_frame.iter()) {
                assert_ne!(unflipped, flipped);
            }
        }

        // compare order parameters
        let unflipped_order = unflipped_results.get_molecule("POPC").unwrap();
        let flipped_order = flipped_results.get_molecule("POPC").unwrap();

        assert_eq!(
            unflipped_order.bonds().count(),
            flipped_order.bonds().count()
        );

        for (unflipped_bond, flipped_bond) in unflipped_order.bonds().zip(flipped_order.bonds()) {
            assert_relative_eq!(
                unflipped_bond.order().total().unwrap().value(),
                flipped_bond.order().total().unwrap().value()
            );
            assert_relative_eq!(
                unflipped_bond.order().upper().unwrap().value(),
                flipped_bond.order().lower().unwrap().value()
            );
            assert_relative_eq!(
                unflipped_bond.order().lower().unwrap().value(),
                flipped_bond.order().upper().unwrap().value(),
            )
        }
    }
}
