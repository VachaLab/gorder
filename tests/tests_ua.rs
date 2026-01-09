// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Integration tests for the calculation of united-atom order parameters.

mod common;

use std::path::Path;

use approx::assert_relative_eq;
use gorder::prelude::*;
use tempfile::{NamedTempFile, TempDir};

use common::{assert_eq_csv, assert_eq_maps, assert_eq_order};

use crate::common::{assert_eq_normals, diff_files_ignore_first};

#[test]
fn test_ua_order_basic() {
    for n_threads in [1, 2, 3, 4, 8, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let output_table = NamedTempFile::new().unwrap();
        let path_to_table = output_table.path().to_str().unwrap();

        let output_csv = NamedTempFile::new().unwrap();
        let path_to_csv = output_csv.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let pattern = format!("{}/order.xvg", path_to_dir);

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .output_yaml(path_to_yaml)
            .output_tab(path_to_table)
            .output_csv(path_to_csv)
            .output_xvg(&pattern)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_basic.yaml", 1);

        assert_eq_order(path_to_table, "tests/files/ua_order_basic.tab", 1);

        assert_eq_csv(path_to_csv, "tests/files/ua_order_basic.csv", 0);

        for molecule in ["POPC", "POPS"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!("tests/files/ua_order_basic_{}.xvg", molecule);

            assert_eq_order(&path, &path_expected, 1);
        }
    }
}

#[test]
fn test_ua_order_basic_saturated_only() {
    let output = NamedTempFile::new().unwrap();
    let path_to_yaml = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/ua.tpr")
        .trajectory("tests/files/ua.xtc")
        .output_yaml(path_to_yaml)
        .analysis_type(AnalysisType::uaorder(
            Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
            None,
            None
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_yaml, "tests/files/ua_order_basic_saturated.yaml", 1);
}

#[test]
fn test_ua_order_basic_unsaturated_only() {
    let output = NamedTempFile::new().unwrap();
    let path_to_yaml = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/ua.tpr")
        .trajectory("tests/files/ua.xtc")
        .output_yaml(path_to_yaml)
        .analysis_type(AnalysisType::uaorder(
            None,
            Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
            None,
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_yaml,
        "tests/files/ua_order_basic_unsaturated.yaml",
        1,
    );
}

#[test]
fn test_ua_order_from_aa_ignore() {
    let output = NamedTempFile::new().unwrap();
    let path_to_yaml = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_yaml(path_to_yaml)
        .analysis_type(AnalysisType::uaorder(
            Some("@membrane and element name carbon and not name C29 C210 C21 C31"),
            Some("@membrane and name C29 C210"),
            Some("element name hydrogen"),
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_yaml, "tests/files/ua_order_from_aa.yaml", 1);
}

#[test]
fn test_ua_order_leaflets() {
    for n_threads in [1, 2, 3, 8, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name r'^P'"),
            LeafletClassification::local("@membrane", "name r'^P'", 2.5),
            LeafletClassification::individual(
                "name r'^P'",
                "(resname POPC and name CA2 C50) or (resname POPS and name C36 C55)",
            ),
        ] {
            for freq in [
                Frequency::every(1).unwrap(),
                Frequency::every(5).unwrap(),
                Frequency::every(100).unwrap(),
                Frequency::once(),
            ] {
                let output = NamedTempFile::new().unwrap();
                let path_to_yaml = output.path().to_str().unwrap();

                let output_table = NamedTempFile::new().unwrap();
                let path_to_table = output_table.path().to_str().unwrap();

                let output_csv = NamedTempFile::new().unwrap();
                let path_to_csv = output_csv.path().to_str().unwrap();

                let directory = TempDir::new().unwrap();
                let path_to_dir = directory.path().to_str().unwrap();

                let pattern = format!("{}/order.xvg", path_to_dir);

                let analysis = Analysis::builder()
                .structure("tests/files/ua.tpr")
                .trajectory("tests/files/ua.xtc")
                .output_yaml(path_to_yaml)
                .output_tab(path_to_table)
                .output_csv(path_to_csv)
                .output_xvg(&pattern)
                .analysis_type(AnalysisType::uaorder(
                    Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                    Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                    None
                ))
                .leaflets(method.clone().with_frequency(freq))
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

                analysis.run().unwrap().write().unwrap();

                assert_eq_order(path_to_yaml, "tests/files/ua_order_leaflets.yaml", 1);

                assert_eq_order(path_to_table, "tests/files/ua_order_leaflets.tab", 1);

                assert_eq_csv(path_to_csv, "tests/files/ua_order_leaflets.csv", 0);

                for molecule in ["POPC", "POPS"] {
                    let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
                    let path_expected = format!("tests/files/ua_order_leaflets_{}.xvg", molecule);

                    assert_eq_order(&path, &path_expected, 1);
                }
            }
        }
    }
}

#[test]
fn test_ua_order_leaflets_clustering() {
    for n_threads in [1, 2, 8, 64] {
        for freq in [
            Frequency::every(1).unwrap(),
            Frequency::every(5).unwrap(),
            Frequency::every(100).unwrap(),
            Frequency::once(),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_yaml = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/ua.tpr")
                .trajectory("tests/files/ua.xtc")
                .output_yaml(path_to_yaml)
                .analysis_type(AnalysisType::uaorder(
                    Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                    Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                    None
                ))
                .leaflets(LeafletClassification::clustering("name r'^P'").with_frequency(freq))
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(
                path_to_yaml,
                "tests/files/ua_order_leaflets_flipped.yaml",
                1,
            );
        }
    }
}

#[test]
fn test_ua_order_leaflets_export() {
    for n_threads in [1, 2, 3, 8, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name r'^P'"),
            LeafletClassification::local("@membrane", "name r'^P'", 2.5),
            LeafletClassification::individual(
                "name r'^P'",
                "(resname POPC and name CA2 C50) or (resname POPS and name C36 C55)",
            ),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_yaml = output.path().to_str().unwrap();

            let output_leaflets = NamedTempFile::new().unwrap();
            let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/ua.tpr")
                .trajectory("tests/files/ua.xtc")
                .output_yaml(path_to_yaml)
                .analysis_type(AnalysisType::uaorder(
                    Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                    Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                    None
                ))
                .leaflets(method.clone().with_frequency(Frequency::once()).with_collect(path_to_output_leaflets))
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(path_to_yaml, "tests/files/ua_order_leaflets.yaml", 1);

            assert!(diff_files_ignore_first(
                path_to_output_leaflets,
                "tests/files/ua_leaflets_once.yaml",
                1,
            ));
        }
    }
}

#[test]
fn test_ua_order_begin_end_step() {
    for n_threads in [1, 2, 4, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .output_yaml(path_to_yaml)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .n_threads(n_threads)
            .leaflets(LeafletClassification::global("@membrane", "name r'^P'"))
            .begin(199200.0)
            .end(199800.0)
            .step(3)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let results = analysis.run().unwrap();

        assert_eq!(results.n_analyzed_frames(), 11);
        results.write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_begin_end_step.yaml", 1);
    }
}

#[test]
fn test_ua_order_nothing_to_analyze() {
    let analysis = Analysis::builder()
        .structure("tests/files/ua.tpr")
        .trajectory("tests/files/ua.xtc")
        .output("THIS_FILE_SHOULD_NOT_BE_CREATED_UA_1")
        .analysis_type(AnalysisType::uaorder(Some("@water"), Some("name Cs"), None))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert!(!Path::new("THIS_FILE_SHOULD_NOT_BE_CREATED_UA_1").exists());
}

#[test]
fn test_ua_order_maps_basic() {
    for n_threads in [1, 3, 8, 32] {
        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                Some("resname POPC and name C50 C20 C13"),
                Some("resname POPC and name C24"),
                None,
            ))
            .map(
                OrderMap::builder()
                    .bin_size([0.5, 2.0])
                    .output_directory(path_to_dir)
                    .min_samples(5)
                    .build()
                    .unwrap(),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        let expected_file_names = [
            "ordermap_POPC-C13-12_full.dat",
            "ordermap_POPC-C13-12--POPC-H1-12_full.dat",
            "ordermap_POPC-C20-19_full.dat",
            "ordermap_POPC-C20-19--POPC-H1-19_full.dat",
            "ordermap_POPC-C20-19--POPC-H2-19_full.dat",
            "ordermap_POPC-C24-23_full.dat",
            "ordermap_POPC-C24-23--POPC-H1-23_full.dat",
            "ordermap_POPC-C50-49_full.dat",
            "ordermap_POPC-C50-49--POPC-H1-49_full.dat",
            "ordermap_POPC-C50-49--POPC-H2-49_full.dat",
            "ordermap_POPC-C50-49--POPC-H3-49_full.dat",
            "ordermap_average_full.dat",
        ];

        for file in expected_file_names {
            let real_file = format!("{}/POPC/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps_ua/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        // full map for the entire system is the same as for POPC
        let real_file = format!("{}/ordermap_average_full.dat", path_to_dir);
        let test_file = "tests/files/ordermaps_ua/ordermap_average_full.dat";
        assert_eq_maps(&real_file, test_file, 2);

        // check the script
        let real_script = format!("{}/plot.py", path_to_dir);
        assert!(common::diff_files_ignore_first(
            &real_script,
            "scripts/plot.py",
            0
        ));
    }
}

#[test]
fn test_ua_order_maps_leaflets() {
    for n_threads in [1, 3, 8, 32] {
        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                Some("resname POPC and name C50 C20 C13"),
                Some("resname POPC and name C24"),
                None,
            ))
            .map(
                OrderMap::builder()
                    .bin_size([0.5, 2.0])
                    .output_directory(path_to_dir)
                    .min_samples(5)
                    .build()
                    .unwrap(),
            )
            .leaflets(LeafletClassification::global("@membrane", "name r'^P'"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        let expected_file_names = [
            "ordermap_POPC-C13-12_upper.dat",
            "ordermap_POPC-C20-19--POPC-H2-19_upper.dat",
            "ordermap_POPC-C50-49_full.dat",
            "ordermap_POPC-C50-49--POPC-H3-49_full.dat",
            "ordermap_POPC-C20-19_full.dat",
            "ordermap_POPC-C20-19_upper.dat",
            "ordermap_POPC-C50-49_lower.dat",
            "ordermap_POPC-C50-49--POPC-H3-49_lower.dat",
            "ordermap_POPC-C20-19_lower.dat",
            "ordermap_POPC-C24-23_full.dat",
            "ordermap_POPC-C50-49--POPC-H1-49_full.dat",
            "ordermap_POPC-C50-49--POPC-H3-49_upper.dat",
            "ordermap_POPC-C13-12_full.dat",
            "ordermap_POPC-C20-19--POPC-H1-19_full.dat",
            "ordermap_POPC-C24-23_lower.dat",
            "ordermap_POPC-C50-49--POPC-H1-49_lower.dat",
            "ordermap_POPC-C50-49_upper.dat",
            "ordermap_POPC-C13-12_lower.dat",
            "ordermap_POPC-C20-19--POPC-H1-19_lower.dat",
            "ordermap_POPC-C24-23--POPC-H1-23_full.dat",
            "ordermap_POPC-C50-49--POPC-H1-49_upper.dat",
            "ordermap_POPC-C13-12--POPC-H1-12_full.dat",
            "ordermap_POPC-C20-19--POPC-H1-19_upper.dat",
            "ordermap_POPC-C24-23--POPC-H1-23_lower.dat",
            "ordermap_POPC-C50-49--POPC-H2-49_full.dat",
            "ordermap_POPC-C13-12--POPC-H1-12_lower.dat",
            "ordermap_POPC-C20-19--POPC-H2-19_full.dat",
            "ordermap_POPC-C24-23--POPC-H1-23_upper.dat",
            "ordermap_POPC-C50-49--POPC-H2-49_lower.dat",
            "ordermap_POPC-C13-12--POPC-H1-12_upper.dat",
            "ordermap_POPC-C20-19--POPC-H2-19_lower.dat",
            "ordermap_POPC-C24-23_upper.dat",
            "ordermap_POPC-C50-49--POPC-H2-49_upper.dat",
            "ordermap_average_full.dat",
            "ordermap_average_upper.dat",
            "ordermap_average_lower.dat",
        ];

        for file in expected_file_names {
            let real_file = format!("{}/POPC/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps_ua/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        // full map for the entire system is the same as for POPC
        let real_file = format!("{}/ordermap_average_full.dat", path_to_dir);
        let test_file = "tests/files/ordermaps_ua/ordermap_average_full.dat";
        assert_eq_maps(&real_file, test_file, 2);

        // check the script
        let real_script = format!("{}/plot.py", path_to_dir);
        assert!(common::diff_files_ignore_first(
            &real_script,
            "scripts/plot.py",
            0
        ));
    }
}

#[test]
fn test_ua_order_error_convergence() {
    for n_threads in [1, 3, 8, 32] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let output_table = NamedTempFile::new().unwrap();
        let path_to_table = output_table.path().to_str().unwrap();

        let output_csv = NamedTempFile::new().unwrap();
        let path_to_csv = output_csv.path().to_str().unwrap();

        let output_convergence = NamedTempFile::new().unwrap();
        let path_to_convergence = output_convergence.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let pattern = format!("{}/order.xvg", path_to_dir);

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .output_yaml(path_to_yaml)
            .output_tab(path_to_table)
            .output_csv(path_to_csv)
            .output_xvg(&pattern)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .estimate_error(EstimateError::new(None, Some(path_to_convergence)).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_error.yaml", 1);

        assert_eq_order(path_to_table, "tests/files/ua_order_error.tab", 1);

        assert_eq_csv(path_to_csv, "tests/files/ua_order_error.csv", 0);

        for molecule in ["POPC", "POPS"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!("tests/files/ua_order_basic_{}.xvg", molecule);

            assert_eq_order(&path, &path_expected, 1);
        }

        assert_eq_order(
            path_to_convergence,
            "tests/files/ua_order_convergence.xvg",
            1,
        );
    }
}

#[test]
fn test_ua_order_error_leaflets_convergence() {
    for n_threads in [1, 3, 8, 32] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let output_table = NamedTempFile::new().unwrap();
        let path_to_table = output_table.path().to_str().unwrap();

        let output_csv = NamedTempFile::new().unwrap();
        let path_to_csv = output_csv.path().to_str().unwrap();

        let output_convergence = NamedTempFile::new().unwrap();
        let path_to_convergence = output_convergence.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let pattern = format!("{}/order.xvg", path_to_dir);

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .output_yaml(path_to_yaml)
            .output_tab(path_to_table)
            .output_csv(path_to_csv)
            .output_xvg(&pattern)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .estimate_error(EstimateError::new(None, Some(path_to_convergence)).unwrap())
            .leaflets(LeafletClassification::global("@membrane", "name r'^P'"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_leaflets_error.yaml", 1);
        assert_eq_order(path_to_table, "tests/files/ua_order_leaflets_error.tab", 1);
        assert_eq_csv(path_to_csv, "tests/files/ua_order_leaflets_error.csv", 0);

        for molecule in ["POPC", "POPS"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!("tests/files/ua_order_leaflets_{}.xvg", molecule);

            assert_eq_order(&path, &path_expected, 1);
        }

        assert_eq_order(
            path_to_convergence,
            "tests/files/ua_order_leaflets_convergence.xvg",
            1,
        );
    }
}

#[test]
fn test_ua_order_cylinder_center() {
    for n_threads in [1, 2, 4, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .output_yaml(path_to_yaml)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .geometry(Geometry::cylinder(GeomReference::center(), 2.5, [f32::NEG_INFINITY, f32::INFINITY], Axis::Z).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_cylinder_center.yaml", 1);
    }
}

#[test]
fn test_ua_order_cuboid_point() {
    for n_threads in [1, 2, 4, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .output_yaml(path_to_yaml)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .geometry(Geometry::cuboid(Vector3D::new(1.5, 2.5, 0.0), [-1.0, 2.0], [0.0, 1.0], [f32::NEG_INFINITY, f32::INFINITY]).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_cuboid_point.yaml", 1);
    }
}

#[test]
fn test_ua_order_leaflets_no_pbc() {
    for n_threads in [1, 2, 4, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/ua_nobox.pdb")
            .trajectory("tests/files/ua_whole_nobox.xtc")
            .output_yaml(path_to_yaml)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .leaflets(LeafletClassification::global("@membrane", "name r'^P'").with_membrane_normal(Axis::Z))
            .handle_pbc(false)
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_leaflets_nopbc.yaml", 1);
    }
}

#[test]
fn test_ua_order_dynamic_normals() {
    for n_threads in [1, 2, 4, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .output_yaml(path_to_yaml)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .membrane_normal(DynamicNormal::new("name r'^P'", 2.0).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_dynamic_normals.yaml", 1);
    }
}

#[test]
fn test_ua_order_dynamic_normals_export() {
    for n_threads in [1, 2, 4, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_yaml = output.path().to_str().unwrap();

        let output_normals = NamedTempFile::new().unwrap();
        let path_to_output_normals = output_normals.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .output_yaml(path_to_yaml)
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .membrane_normal(DynamicNormal::new("name r'^P'", 2.0).unwrap().with_collect(path_to_output_normals))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_yaml, "tests/files/ua_order_dynamic_normals.yaml", 1);
        assert_eq_normals(path_to_output_normals, "tests/files/ua_normals.yaml");
    }
}

#[test]
fn test_ua_order_basic_rust_api() {
    let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .silent()
            .overwrite()
            .build()
            .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::UA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert_eq!(results.analysis().structure(), "tests/files/ua.tpr");

    assert_eq!(results.molecules().count(), 2);

    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPS").is_some());
    assert!(results.get_molecule("POPG").is_none());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.1169,
        epsilon = 2e-4
    );
    assert!(results.average_order().upper().is_none());
    assert!(results.average_order().lower().is_none());

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_average_orders = [0.1101, 0.1470];
    let expected_atom_numbers = [40, 37];
    let expected_molecule_names = ["POPC", "POPS"];

    let expected_atom_indices = [23, 45];
    let expected_atom_names = ["C24", "C46"];
    let expected_atom_order = [0.0978, 0.2221];

    let expected_bond_numbers = [1, 2];

    let expected_bond_orders = [vec![0.0978], vec![0.2084, 0.2359]];

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

        // atoms
        assert_eq!(molecule.atoms().count(), expected_atom_numbers[i]);

        let atom = molecule.get_atom(expected_atom_indices[i]).unwrap();
        let atom_type = atom.atom();
        assert_eq!(atom_type.atom_name(), expected_atom_names[i]);
        assert_eq!(atom_type.relative_index(), expected_atom_indices[i]);
        assert_eq!(atom_type.residue_name(), expected_molecule_names[i]);
        assert_eq!(atom.molecule(), expected_molecule_names[i]);

        let order = atom.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_atom_order[i],
            epsilon = 2e-4
        );
        assert!(order.total().unwrap().error().is_none());
        assert!(order.upper().is_none());
        assert!(order.lower().is_none());

        let maps = atom.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bonds
        assert_eq!(atom.bonds().count(), expected_bond_numbers[i]);

        for (b, bond) in atom.bonds().enumerate() {
            assert_eq!(bond.molecule(), expected_molecule_names[i]);
            assert_relative_eq!(
                bond.order().total().unwrap().value(),
                expected_bond_orders[i][b],
                epsilon = 2e-4
            );
            assert!(bond.order().total().unwrap().error().is_none());
            assert!(bond.order().upper().is_none());
            assert!(bond.order().lower().is_none());
            assert!(bond.ordermaps().total().is_none());
            assert!(bond.ordermaps().upper().is_none());
            assert!(bond.ordermaps().lower().is_none());
        }

        // nonexistent atoms
        assert!(molecule.get_atom(145).is_none());
        assert!(molecule.get_atom(7).is_none());
    }
}

#[test]
fn test_ua_order_error_rust_api() {
    let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .estimate_error(EstimateError::default())
            .silent()
            .overwrite()
            .build()
            .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::UA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert_eq!(results.analysis().structure(), "tests/files/ua.tpr");

    assert_eq!(results.molecules().count(), 2);

    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPS").is_some());
    assert!(results.get_molecule("POPG").is_none());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.1169,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().total().unwrap().error().unwrap(),
        0.0027,
        epsilon = 2e-4
    );
    assert!(results.average_order().upper().is_none());
    assert!(results.average_order().lower().is_none());

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_average_orders = [0.1101, 0.1470];
    let expected_average_errors = [0.0019, 0.0106];
    let expected_atom_numbers = [40, 37];
    let expected_molecule_names = ["POPC", "POPS"];

    let expected_atom_indices = [23, 45];
    let expected_atom_names = ["C24", "C46"];
    let expected_atom_order = [0.0978, 0.2221];
    let expected_atom_errors = [0.0070, 0.0241];

    let expected_bond_numbers = [1, 2];

    let expected_bond_orders = [vec![0.0978], vec![0.2084, 0.2359]];
    let expected_bond_errors = [vec![0.0070], vec![0.0262, 0.0441]];

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

        // atoms
        assert_eq!(molecule.atoms().count(), expected_atom_numbers[i]);

        let atom = molecule.get_atom(expected_atom_indices[i]).unwrap();
        let atom_type = atom.atom();
        assert_eq!(atom_type.atom_name(), expected_atom_names[i]);
        assert_eq!(atom_type.relative_index(), expected_atom_indices[i]);
        assert_eq!(atom_type.residue_name(), expected_molecule_names[i]);
        assert_eq!(atom.molecule(), expected_molecule_names[i]);

        let order = atom.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_atom_order[i],
            epsilon = 2e-4
        );
        assert_relative_eq!(
            order.total().unwrap().error().unwrap(),
            expected_atom_errors[i],
            epsilon = 2e-4
        );
        assert!(order.upper().is_none());
        assert!(order.lower().is_none());

        let maps = atom.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bonds
        assert_eq!(atom.bonds().count(), expected_bond_numbers[i]);

        for (b, bond) in atom.bonds().enumerate() {
            assert_eq!(bond.molecule(), expected_molecule_names[i]);
            assert_relative_eq!(
                bond.order().total().unwrap().value(),
                expected_bond_orders[i][b],
                epsilon = 2e-4
            );
            assert_relative_eq!(
                bond.order().total().unwrap().error().unwrap(),
                expected_bond_errors[i][b],
                epsilon = 2e-4
            );
            assert!(bond.order().upper().is_none());
            assert!(bond.order().lower().is_none());
            assert!(bond.ordermaps().total().is_none());
            assert!(bond.ordermaps().upper().is_none());
            assert!(bond.ordermaps().lower().is_none());
        }

        // nonexistent atoms
        assert!(molecule.get_atom(145).is_none());
        assert!(molecule.get_atom(7).is_none());
    }
}

#[test]
fn test_ua_order_leaflets_rust_api() {
    let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .leaflets(LeafletClassification::global("@membrane", "name r'^P'"))
            .silent()
            .overwrite()
            .build()
            .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::UA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert_eq!(results.analysis().structure(), "tests/files/ua.tpr");

    assert_eq!(results.molecules().count(), 2);

    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPS").is_some());
    assert!(results.get_molecule("POPG").is_none());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.1169,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().value(),
        0.1151,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().lower().unwrap().value(),
        0.1186,
        epsilon = 2e-4
    );

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_average_orders = [0.1101, 0.1470];
    let expected_average_upper = [0.1075, 0.1491];
    let expected_average_lower = [0.1128, 0.1449];
    let expected_atom_numbers = [40, 37];
    let expected_molecule_names = ["POPC", "POPS"];

    let expected_atom_indices = [23, 45];
    let expected_atom_names = ["C24", "C46"];
    let expected_atom_order = [0.0978, 0.2221];
    let expected_atom_upper = [0.1088, 0.2204];
    let expected_atom_lower = [0.0869, 0.2239];

    let expected_bond_numbers = [1, 2];

    let expected_bond_orders = [vec![0.0978], vec![0.2084, 0.2359]];
    let expected_bond_upper = [vec![0.1088], vec![0.1986, 0.2421]];
    let expected_bond_lower = [vec![0.0869], vec![0.2181, 0.2296]];

    for (i, molecule) in results.molecules().enumerate() {
        assert_eq!(molecule.molecule(), expected_molecule_names[i]);

        let average_order = molecule.average_order();
        assert_relative_eq!(
            average_order.total().unwrap().value(),
            expected_average_orders[i],
            epsilon = 2e-4
        );
        assert_relative_eq!(
            average_order.upper().unwrap().value(),
            expected_average_upper[i],
            epsilon = 2e-4
        );
        assert_relative_eq!(
            average_order.lower().unwrap().value(),
            expected_average_lower[i],
            epsilon = 2e-4
        );
        assert!(average_order.total().unwrap().error().is_none());
        assert!(average_order.upper().unwrap().error().is_none());
        assert!(average_order.lower().unwrap().error().is_none());

        let average_maps = molecule.average_ordermaps();
        assert!(average_maps.total().is_none());
        assert!(average_maps.upper().is_none());
        assert!(average_maps.lower().is_none());

        // atoms
        assert_eq!(molecule.atoms().count(), expected_atom_numbers[i]);

        let atom = molecule.get_atom(expected_atom_indices[i]).unwrap();
        let atom_type = atom.atom();
        assert_eq!(atom_type.atom_name(), expected_atom_names[i]);
        assert_eq!(atom_type.relative_index(), expected_atom_indices[i]);
        assert_eq!(atom_type.residue_name(), expected_molecule_names[i]);
        assert_eq!(atom.molecule(), expected_molecule_names[i]);

        let order = atom.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_atom_order[i],
            epsilon = 2e-4
        );
        assert_relative_eq!(
            order.upper().unwrap().value(),
            expected_atom_upper[i],
            epsilon = 2e-4
        );
        assert_relative_eq!(
            order.lower().unwrap().value(),
            expected_atom_lower[i],
            epsilon = 2e-4
        );
        assert!(order.total().unwrap().error().is_none());
        assert!(order.upper().unwrap().error().is_none());
        assert!(order.lower().unwrap().error().is_none());

        let maps = atom.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bonds
        assert_eq!(atom.bonds().count(), expected_bond_numbers[i]);

        for (b, bond) in atom.bonds().enumerate() {
            assert_eq!(bond.molecule(), expected_molecule_names[i]);
            assert_relative_eq!(
                bond.order().total().unwrap().value(),
                expected_bond_orders[i][b],
                epsilon = 2e-4
            );
            assert_relative_eq!(
                bond.order().upper().unwrap().value(),
                expected_bond_upper[i][b],
                epsilon = 2e-4
            );
            assert_relative_eq!(
                bond.order().lower().unwrap().value(),
                expected_bond_lower[i][b],
                epsilon = 2e-4
            );
            assert!(bond.order().total().unwrap().error().is_none());
            assert!(bond.order().upper().unwrap().error().is_none());
            assert!(bond.order().lower().unwrap().error().is_none());
            assert!(bond.ordermaps().total().is_none());
            assert!(bond.ordermaps().upper().is_none());
            assert!(bond.ordermaps().lower().is_none());
        }

        // nonexistent atoms
        assert!(molecule.get_atom(145).is_none());
        assert!(molecule.get_atom(7).is_none());
    }
}

#[test]
fn test_ua_order_error_leaflets_rust_api() {
    let analysis = Analysis::builder()
            .structure("tests/files/ua.tpr")
            .trajectory("tests/files/ua.xtc")
            .analysis_type(AnalysisType::uaorder(
                Some("(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"),
                Some("(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"),
                None
            ))
            .leaflets(LeafletClassification::global("@membrane", "name r'^P'"))
            .estimate_error(EstimateError::default())
            .silent()
            .overwrite()
            .build()
            .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::UA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert_eq!(results.analysis().structure(), "tests/files/ua.tpr");

    assert_eq!(results.molecules().count(), 2);

    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPS").is_some());
    assert!(results.get_molecule("POPG").is_none());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.1169,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().value(),
        0.1151,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().lower().unwrap().value(),
        0.1186,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().total().unwrap().error().unwrap(),
        0.0027,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().error().unwrap(),
        0.0031,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().lower().unwrap().error().unwrap(),
        0.0031,
        epsilon = 2e-4
    );

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_atom_numbers = [40, 37];
    let expected_molecule_names = ["POPC", "POPS"];

    let expected_atom_indices = [23, 45];
    let expected_atom_names = ["C24", "C46"];

    let expected_bond_numbers = [1, 2];

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

        // atoms
        assert_eq!(molecule.atoms().count(), expected_atom_numbers[i]);

        let atom = molecule.get_atom(expected_atom_indices[i]).unwrap();
        let atom_type = atom.atom();
        assert_eq!(atom_type.atom_name(), expected_atom_names[i]);
        assert_eq!(atom_type.relative_index(), expected_atom_indices[i]);
        assert_eq!(atom_type.residue_name(), expected_molecule_names[i]);
        assert_eq!(atom.molecule(), expected_molecule_names[i]);

        let order = atom.order();
        assert!(order.total().unwrap().error().is_some());
        assert!(order.upper().unwrap().error().is_some());
        assert!(order.lower().unwrap().error().is_some());

        let maps = atom.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bonds
        assert_eq!(atom.bonds().count(), expected_bond_numbers[i]);

        for bond in atom.bonds() {
            assert_eq!(bond.molecule(), expected_molecule_names[i]);
            assert!(bond.order().total().unwrap().error().is_some());
            assert!(bond.order().upper().unwrap().error().is_some());
            assert!(bond.order().lower().unwrap().error().is_some());
            assert!(bond.ordermaps().total().is_none());
            assert!(bond.ordermaps().upper().is_none());
            assert!(bond.ordermaps().lower().is_none());
        }

        // nonexistent atoms
        assert!(molecule.get_atom(145).is_none());
        assert!(molecule.get_atom(7).is_none());
    }
}

#[test]
fn test_ua_order_ordermaps_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/ua.tpr")
        .trajectory("tests/files/ua.xtc")
        .analysis_type(AnalysisType::uaorder(
            Some("resname POPC and name C50 C20 C13"),
            Some("resname POPC and name C24"),
            None,
        ))
        .map(
            OrderMap::builder()
                .bin_size([0.5, 2.0])
                .min_samples(5)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::UA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
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
    assert_relative_eq!(span_x.1, 6.53265);
    assert_relative_eq!(span_y.0, 0.0);
    assert_relative_eq!(span_y.1, 6.53265);
    assert_relative_eq!(bin.0, 0.5);
    assert_relative_eq!(bin.1, 2.0);

    assert_relative_eq!(
        map.get_at_convert(2.0, 6.0).unwrap(),
        0.0127,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(4.3, 0.1).unwrap(),
        0.1286,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(6.4, 2.2).unwrap(),
        0.0839,
        epsilon = 2e-4
    );

    // ordermaps for a selected atom
    let atom = molecule.get_atom(49).unwrap();
    let map = atom.ordermaps().total().as_ref().unwrap();
    assert!(atom.ordermaps().upper().is_none());
    assert!(atom.ordermaps().lower().is_none());

    assert_relative_eq!(
        map.get_at_convert(2.0, 6.0).unwrap(),
        0.0349,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(4.3, 0.1).unwrap(),
        -0.0160,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(6.4, 2.2).unwrap(),
        -0.0084,
        epsilon = 2e-4
    );

    // ordermaps for a selected bond
    let bond = atom.bonds().nth(1).unwrap();
    let map = bond.ordermaps().total().as_ref().unwrap();
    assert!(bond.ordermaps().upper().is_none());
    assert!(bond.ordermaps().lower().is_none());

    assert_relative_eq!(
        map.get_at_convert(2.0, 6.0).unwrap(),
        0.1869,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(4.3, 0.1).unwrap(),
        0.0962,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(6.4, 2.2).unwrap(),
        0.0358,
        epsilon = 2e-4
    );
}

#[test]
fn test_ua_order_leaflets_ordermaps_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/ua.tpr")
        .trajectory("tests/files/ua.xtc")
        .analysis_type(AnalysisType::uaorder(
            Some("resname POPC and name C50 C20 C13"),
            Some("resname POPC and name C24"),
            None,
        ))
        .map(
            OrderMap::builder()
                .bin_size([0.5, 2.0])
                .min_samples(5)
                .build()
                .unwrap(),
        )
        .leaflets(LeafletClassification::global("@membrane", "name r'^P'"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::UA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert_eq!(results.molecules().count(), 1);

    // average ordermaps for the entire system
    assert!(results.average_ordermaps().total().is_some());
    assert!(results.average_ordermaps().upper().is_some());
    assert!(results.average_ordermaps().lower().is_some());

    // average ordermaps for the entire molecule
    let molecule = results.get_molecule("POPC").unwrap();
    let total = molecule.average_ordermaps().total().as_ref().unwrap();
    let upper = molecule.average_ordermaps().upper().as_ref().unwrap();
    let lower = molecule.average_ordermaps().lower().as_ref().unwrap();

    let span_x = total.span_x();
    let span_y = total.span_y();
    let bin = total.tile_dim();

    assert_relative_eq!(span_x.0, 0.0);
    assert_relative_eq!(span_x.1, 6.53265);
    assert_relative_eq!(span_y.0, 0.0);
    assert_relative_eq!(span_y.1, 6.53265);
    assert_relative_eq!(bin.0, 0.5);
    assert_relative_eq!(bin.1, 2.0);

    assert_relative_eq!(
        total.get_at_convert(2.1, 5.8).unwrap(),
        0.0127,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        upper.get_at_convert(2.1, 5.8).unwrap(),
        0.0499,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        lower.get_at_convert(2.1, 5.8).unwrap(),
        -0.0036,
        epsilon = 2e-4
    );

    // ordermaps for a selected atom
    let atom = molecule.get_atom(49).unwrap();
    let total = atom.ordermaps().total().as_ref().unwrap();
    let upper = atom.ordermaps().upper().as_ref().unwrap();
    let lower = atom.ordermaps().lower().as_ref().unwrap();

    assert_relative_eq!(
        total.get_at_convert(2.1, 5.8).unwrap(),
        0.0349,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        upper.get_at_convert(2.1, 5.8).unwrap(),
        0.0450,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        lower.get_at_convert(2.1, 5.8).unwrap(),
        0.0272,
        epsilon = 2e-4
    );

    // ordermaps for a selected bond
    let bond = atom.bonds().nth(1).unwrap();
    let total = bond.ordermaps().total().as_ref().unwrap();
    let upper = bond.ordermaps().upper().as_ref().unwrap();
    let lower = bond.ordermaps().lower().as_ref().unwrap();

    assert_relative_eq!(
        total.get_at_convert(2.1, 5.8).unwrap(),
        0.1869,
        epsilon = 2e-4
    );
    assert!(upper.get_at_convert(6.4, 0.0).unwrap().is_nan());
    assert!(lower.get_at_convert(6.4, 6.0).unwrap().is_nan());
}
