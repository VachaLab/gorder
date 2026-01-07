// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Integration tests for the calculation of atomistic order parameters.

mod common;

use std::{
    fs::{read_to_string, File},
    io::Read,
    path::{Path, PathBuf},
};

use approx::assert_relative_eq;
use gorder::prelude::*;
use hashbrown::HashMap;
use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

use common::{assert_eq_csv, assert_eq_maps, assert_eq_order, read_and_compare_files};

use crate::common::{assert_eq_normals, diff_files_ignore_first};

#[test]
fn test_aa_order_basic_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_basic.yaml", 1);
}

#[test]
fn test_aa_order_basic_concatenated_yaml_multiple_threads() {
    for n_threads in [1, 2, 3, 8, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory(vec![
                "tests/files/split/pcpepg1.xtc",
                "tests/files/split/pcpepg2.xtc",
                "tests/files/split/pcpepg3.xtc",
                "tests/files/split/pcpepg4.xtc",
                "tests/files/split/pcpepg5.xtc",
            ])
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_basic.yaml", 1);
    }
}

#[test]
fn test_aa_order_basic_ndx_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .index("tests/files/pcpepg.ndx")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder("HeavyAtoms", "Hydrogens"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_basic.yaml", 1);
}

#[test]
fn test_aa_order_basic_fail_overlap() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon or serial 876 to 1234",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Function should have failed."),
        Err(e) => assert!(e.to_string().contains("are part of both")),
    }
}

#[test]
fn test_aa_order_basic_table() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_table, "tests/files/aa_order_basic.tab", 1);
}

#[test]
fn test_aa_order_basic_xvg() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let pattern = format!("{}/order.xvg", path_to_dir);

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_xvg(pattern)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    for molecule in ["POPC", "POPE", "POPG"] {
        let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
        let path_expected = format!("tests/files/aa_order_basic_{}.xvg", molecule);

        assert_eq_order(&path, &path_expected, 1);
    }
}

#[test]
fn test_aa_order_basic_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(path_to_csv, "tests/files/aa_order_basic.csv", 0);
}

#[test]
fn test_aa_order_basic_xvg_weird_names() {
    for name in ["order", ".this.is.a.weird.name.xvg"] {
        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let pattern = format!("{}/{}", path_to_dir, name);

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_xvg(pattern)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        for molecule in ["POPC", "POPE", "POPG"] {
            let path = if name.contains(".xvg") {
                format!("{}/.this.is.a.weird.name_{}.xvg", path_to_dir, molecule)
            } else {
                format!("{}/order_{}", path_to_dir, molecule)
            };

            let path_expected = format!("tests/files/aa_order_basic_{}.xvg", molecule);
            assert_eq_order(&path, &path_expected, 1);
        }
    }
}

#[test]
fn test_aa_order_basic_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_basic.yaml", 1);
    }
}

#[test]
fn test_aa_order_basic_table_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output_table = NamedTempFile::new().unwrap();
        let path_to_table = output_table.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_tab(path_to_table)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_table, "tests/files/aa_order_basic.tab", 1);
    }
}

#[test]
fn test_aa_order_leaflets_yaml() {
    for method in [
        LeafletClassification::global("@membrane", "name P"),
        LeafletClassification::local("@membrane", "name P", 2.5),
        LeafletClassification::individual("name P", "name C218 C316"),
        LeafletClassification::clustering("name P"),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_aa_order_leaflets_yaml_multiple_threads_binary_presice_comparison() {
    let output_reference = NamedTempFile::new().unwrap();
    let path_to_output_reference = output_reference.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output_reference)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    for n_threads in [2, 3, 5, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        // precise comparison
        assert!(diff_files_ignore_first(
            path_to_output,
            path_to_output_reference,
            1
        ));
    }
}

/* deprecated since v0.7
#[test]
fn test_aa_order_leaflets_yaml_alt_traj() {
    for traj in ["tests/files/pcpepg.nc", "tests/files/pcpepg.dcd"] {
        for n_threads in [1, 3, 8] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory(traj)
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(
                    LeafletClassification::individual("name P", "name C218 C316")
                        .with_frequency(Frequency::once()),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            if traj == "tests/files/pcpepg.lammpstrj" {
                assert!(diff_files_ignore_first(
                    path_to_output,
                    "tests/files/aa_order_leaflets_lammps.yaml",
                    1
                ));
            } else {
                assert!(diff_files_ignore_first(
                    path_to_output,
                    "tests/files/aa_order_leaflets.yaml",
                    1
                ));
            }
        }
    }
}*/

#[test]
fn test_aa_order_leaflets_yaml_from_gro() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.gro")
        .bonds("tests/files/pcpepg.bnd")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_yaml_from_gro_min_bonds() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.gro")
        .bonds("tests/files/pcpepg_min.bnd")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_yaml_from_pdb() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.pdb")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_yaml_from_pqr() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.pqr")
        .bonds("tests/files/pcpepg.bnd")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.5),
            LeafletClassification::individual("name P", "name C218 C316"),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(method)
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_yaml_multiple_threads_various_frequencies() {
    for n_threads in [1, 2, 5, 8, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.5),
            LeafletClassification::individual("name P", "name C218 C316"),
        ] {
            for freq in [
                Frequency::every(4).unwrap(),
                Frequency::every(20).unwrap(),
                Frequency::every(100).unwrap(),
                Frequency::once(),
            ] {
                let output = NamedTempFile::new().unwrap();
                let path_to_output = output.path().to_str().unwrap();

                let analysis = Analysis::builder()
                    .structure("tests/files/pcpepg.tpr")
                    .trajectory("tests/files/pcpepg.xtc")
                    .output(path_to_output)
                    .analysis_type(AnalysisType::aaorder(
                        "@membrane and element name carbon",
                        "@membrane and element name hydrogen",
                    ))
                    .leaflets(method.clone().with_frequency(freq))
                    .n_threads(n_threads)
                    .silent()
                    .overwrite()
                    .build()
                    .unwrap();

                analysis.run().unwrap().write().unwrap();

                assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
            }
        }
    }
}

#[test]
fn test_aa_order_leaflets_yaml_multiple_threads_frequency_once_export() {
    for n_threads in [1, 2, 5, 8, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.5),
            LeafletClassification::individual("name P", "name C218 C316"),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let output_leaflets = NamedTempFile::new().unwrap();
            let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(
                    method
                        .clone()
                        .with_frequency(Frequency::Once)
                        .with_collect(path_to_output_leaflets),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert!(diff_files_ignore_first(
                path_to_output_leaflets,
                "tests/files/aa_leaflets_once.yaml",
                1,
            ));

            assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_yaml_multiple_threads_frequency_every5_export() {
    for n_threads in [1, 2, 5, 8, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.5),
            LeafletClassification::individual("name P", "name C218 C316"),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let output_leaflets = NamedTempFile::new().unwrap();
            let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(
                    method
                        .clone()
                        .with_frequency(Frequency::every(5).unwrap())
                        .with_collect(path_to_output_leaflets),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert!(diff_files_ignore_first(
                path_to_output_leaflets,
                "tests/files/aa_leaflets_every5.yaml",
                1,
            ));

            assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_yaml_multiple_threads_frequency_every1_export() {
    for n_threads in [1, 2, 5, 8, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.5),
            LeafletClassification::individual("name P", "name C218 C316"),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let output_leaflets = NamedTempFile::new().unwrap();
            let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(method.clone().with_collect(path_to_output_leaflets))
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert!(diff_files_ignore_first(
                path_to_output_leaflets,
                "tests/files/aa_leaflets_every1.yaml",
                1,
            ));

            assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_yaml_geometry_export() {
    let output_leaflets = NamedTempFile::new().unwrap();
    let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::global("@membrane", "name P")
                .with_collect(path_to_output_leaflets),
        )
        .geometry(
            Geometry::cylinder(
                GeomReference::center(),
                1.0,
                [f32::NEG_INFINITY, f32::INFINITY],
                Axis::Z,
            )
            .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    // assignment for all lipids should be exported even though only lipids in cylinder are used to calculate order parameters
    assert!(diff_files_ignore_first(
        path_to_output_leaflets,
        "tests/files/aa_leaflets_every1.yaml",
        1,
    ));
}

#[test]
fn test_aa_order_leaflets_yaml_multiple_threads_concatenation_frequency_every1_export() {
    for n_threads in [1, 2, 5, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let output_leaflets = NamedTempFile::new().unwrap();
        let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory(vec![
                "tests/files/split/pcpepg1.xtc",
                "tests/files/split/pcpepg2.xtc",
                "tests/files/split/pcpepg3.xtc",
                "tests/files/split/pcpepg4.xtc",
                "tests/files/split/pcpepg5.xtc",
            ])
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(
                LeafletClassification::global("@membrane", "name P")
                    .with_collect(path_to_output_leaflets),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert!(diff_files_ignore_first(
            path_to_output_leaflets,
            "tests/files/aa_leaflets_every1.yaml",
            1,
        ));

        assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_aa_order_leaflets_yaml_clustering_frequency_once_export() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let output_leaflets = NamedTempFile::new().unwrap();
    let path_to_output_leaflets = output_leaflets.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::clustering("name P")
                .with_frequency(Frequency::Once)
                .with_collect(path_to_output_leaflets),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert!(diff_files_ignore_first(
        path_to_output_leaflets,
        "tests/files/aa_leaflets_once.yaml",
        1,
    ));

    assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_yaml_clustering_multiple_threads_various_frequencies() {
    for n_threads in [1, 2, 8, 64] {
        for freq in [
            Frequency::every(1).unwrap(),
            Frequency::every(4).unwrap(),
            Frequency::every(20).unwrap(),
            Frequency::every(100).unwrap(),
            Frequency::once(),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(LeafletClassification::clustering("name P").with_frequency(freq))
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_yaml_different_membrane_normals() {
    for (input_traj, normal) in [
        "tests/files/pcpepg_switched_xy.xtc",
        "tests/files/pcpepg_switched_xz.xtc",
        "tests/files/pcpepg_switched_yz.xtc",
    ]
    .into_iter()
    .zip([Axis::Z, Axis::X, Axis::Y].into_iter())
    {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.5),
            LeafletClassification::individual("name P", "name C218 C316"),
            LeafletClassification::clustering("name P"),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory(input_traj)
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(method)
                .membrane_normal(normal)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_table() {
    for method in [
        LeafletClassification::global("@membrane", "name P"),
        LeafletClassification::local("@membrane", "name P", 2.5),
        LeafletClassification::individual("name P", "name C218 C316"),
    ] {
        let output_table = NamedTempFile::new().unwrap();
        let path_to_table = output_table.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_tab(path_to_table)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_table, "tests/files/aa_order_leaflets.tab", 1);
    }
}

#[test]
fn test_aa_order_leaflets_xvg() {
    for method in [
        LeafletClassification::global("@membrane", "name P"),
        LeafletClassification::local("@membrane", "name P", 2.5),
        LeafletClassification::individual("name P", "name C218 C316"),
    ] {
        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let pattern = format!("{}/order.xvg", path_to_dir);

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_xvg(pattern)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        for molecule in ["POPC", "POPE", "POPG"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!("tests/files/aa_order_leaflets_{}.xvg", molecule);

            assert_eq_order(&path, &path_expected, 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_csv() {
    for method in [
        LeafletClassification::global("@membrane", "name P"),
        LeafletClassification::local("@membrane", "name P", 2.5),
        LeafletClassification::individual("name P", "name C218 C316"),
    ] {
        let output_csv = NamedTempFile::new().unwrap();
        let path_to_csv = output_csv.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_csv(path_to_csv)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_csv(path_to_csv, "tests/files/aa_order_leaflets.csv", 0);
    }
}

#[test]
fn test_aa_order_leaflets_yaml_supershort() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg_selected.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_selected.yaml", 1);
}

#[test]
fn test_aa_order_one_different_hydrogen_numbers_table() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::aaorder(
            "(resname POPC and name C29 C210) or (resname POPE and element name carbon)",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_table,
        "tests/files/aa_order_different_hydrogen_numbers.tab",
        1,
    );
}

#[test]
fn test_aa_order_one_different_hydrogen_numbers_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::aaorder(
            "(resname POPC and name C29 C210) or (resname POPE and element name carbon)",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(
        path_to_csv,
        "tests/files/aa_order_different_hydrogen_numbers.csv",
        0,
    );
}

#[test]
fn test_aa_order_limit_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .min_samples(2000)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_limit.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_limit_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .min_samples(500)
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_leaflets_limit.yaml",
        1,
    );
}

#[test]
fn test_aa_order_leaflets_limit_tab() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .min_samples(500)
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_table, "tests/files/aa_order_leaflets_limit.tab", 1);
}

#[test]
fn test_aa_order_leaflets_limit_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .min_samples(500)
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(path_to_csv, "tests/files/aa_order_leaflets_limit.csv", 0);
}

#[test]
fn test_aa_order_begin_end_step_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .begin(450_200.0)
        .end(450_400.0)
        .step(3)
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = analysis.run().unwrap();
    assert_eq!(results.n_analyzed_frames(), 4);
    results.write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_begin_end_step.yaml",
        1,
    );
}

#[test]
fn test_aa_order_begin_end_step_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .begin(450_200.0)
            .end(450_400.0)
            .step(3)
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let results = analysis.run().unwrap();
        assert_eq!(results.n_analyzed_frames(), 4);
        results.write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/aa_order_begin_end_step.yaml",
            1,
        );
    }
}

#[test]
fn test_aa_order_concatenated_begin_end_step_yaml_multiple_threads() {
    for n_threads in [1, 2, 3, 8, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/split/pcpepg?.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .begin(450_200.0)
            .end(450_400.0)
            .step(3)
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let results = analysis.run().unwrap();
        assert_eq!(results.n_analyzed_frames(), 4);
        results.write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/aa_order_begin_end_step.yaml",
            1,
        );
    }
}

#[test]
fn test_aa_order_step_yaml_leaflets_multiple_threads_various_frequencies() {
    for n_threads in [1, 2, 5, 8, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.5),
            LeafletClassification::individual("name P", "name C218 C316"),
        ] {
            for freq in [
                Frequency::every(1).unwrap(),
                Frequency::every(5).unwrap(),
                Frequency::every(30).unwrap(),
                Frequency::once(),
            ] {
                let output = NamedTempFile::new().unwrap();
                let path_to_output = output.path().to_str().unwrap();

                let analysis = Analysis::builder()
                    .structure("tests/files/pcpepg.tpr")
                    .trajectory("tests/files/pcpepg.xtc")
                    .output(path_to_output)
                    .analysis_type(AnalysisType::aaorder(
                        "@membrane and element name carbon",
                        "@membrane and element name hydrogen",
                    ))
                    .step(5)
                    .leaflets(method.clone().with_frequency(freq))
                    .n_threads(n_threads)
                    .silent()
                    .overwrite()
                    .build()
                    .unwrap();

                let results = analysis.run().unwrap();
                assert_eq!(results.n_analyzed_frames(), 11);
                results.write().unwrap();

                assert_eq_order(path_to_output, "tests/files/aa_order_step.yaml", 1);
            }
        }
    }
}

#[test]
fn test_aa_order_begin_end_step_yaml_leaflets_multiple_threads_various_frequencies() {
    for n_threads in [1, 2, 5, 8, 64] {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.5),
            LeafletClassification::individual("name P", "name C218 C316"),
        ] {
            for freq in [
                Frequency::every(2).unwrap(),
                Frequency::every(10).unwrap(),
                Frequency::once(),
            ] {
                let output = NamedTempFile::new().unwrap();
                let path_to_output = output.path().to_str().unwrap();

                let analysis = Analysis::builder()
                    .structure("tests/files/pcpepg.tpr")
                    .trajectory("tests/files/pcpepg.xtc")
                    .output(path_to_output)
                    .analysis_type(AnalysisType::aaorder(
                        "@membrane and element name carbon",
                        "@membrane and element name hydrogen",
                    ))
                    .begin(450_200.0)
                    .end(450_400.0)
                    .step(3)
                    .leaflets(method.clone().with_frequency(freq))
                    .n_threads(n_threads)
                    .silent()
                    .overwrite()
                    .build()
                    .unwrap();

                let results = analysis.run().unwrap();
                assert_eq!(results.n_analyzed_frames(), 4);
                results.write().unwrap();

                assert_eq_order(
                    path_to_output,
                    "tests/files/aa_order_begin_end_step.yaml",
                    1,
                );
            }
        }
    }
}

#[test]
fn test_aa_order_begin_end_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .begin(450_200.0)
        .end(450_400.0)
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = analysis.run().unwrap();
    assert_eq!(results.n_analyzed_frames(), 11);
    results.write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_begin_end.yaml", 1);
}

#[test]
fn test_aa_order_begin_end_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .begin(450_200.0)
            .end(450_400.0)
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let results = analysis.run().unwrap();
        assert_eq!(results.n_analyzed_frames(), 11);
        results.write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_begin_end.yaml", 1);
    }
}

#[test]
fn test_aa_order_no_molecules() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output("THIS_FILE_SHOULD_NOT_BE_CREATED_1")
        .analysis_type(AnalysisType::aaorder(
            "@ion",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert!(!Path::new("THIS_FILE_SHOULD_NOT_BE_CREATED_1").exists());
}

#[test]
fn test_aa_order_empty_molecules() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output("THIS_FILE_SHOULD_NOT_BE_CREATED_2")
        .analysis_type(AnalysisType::aaorder(
            "@water and element symbol O",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert!(!Path::new("THIS_FILE_SHOULD_NOT_BE_CREATED_2").exists());
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
fn test_aa_order_basic_all_formats_backup() {
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
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_yaml(&file_paths[0])
        .output_tab(&file_paths[1])
        .output_csv(&file_paths[2])
        .output_xvg(&xvg_pattern)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(&file_paths[0], "tests/files/aa_order_basic.yaml", 1);
    assert_eq_order(&file_paths[1], "tests/files/aa_order_basic.tab", 1);
    assert_eq_csv(&file_paths[2], "tests/files/aa_order_basic.csv", 0);
    assert_eq_order(&file_paths[3], "tests/files/aa_order_basic_POPC.xvg", 1);
    assert_eq_order(&file_paths[4], "tests/files/aa_order_basic_POPE.xvg", 1);
    assert_eq_order(&file_paths[5], "tests/files/aa_order_basic_POPG.xvg", 1);

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
fn test_aa_order_maps_basic() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "resname POPC and name C22 C24 C218",
            "@membrane and element name hydrogen",
        ))
        .map(
            OrderMap::builder()
                .bin_size([0.1, 4.0])
                .output_directory(path_to_dir)
                .min_samples(5)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    let expected_file_names = [
        "ordermap_POPC-C218-87--POPC-H18R-88_full.dat",
        "ordermap_POPC-C218-87--POPC-H18S-89_full.dat",
        "ordermap_POPC-C218-87--POPC-H18T-90_full.dat",
        "ordermap_POPC-C218-87_full.dat",
        "ordermap_POPC-C22-32--POPC-H2R-33_full.dat",
        "ordermap_POPC-C22-32--POPC-H2S-34_full.dat",
        "ordermap_POPC-C22-32_full.dat",
        "ordermap_POPC-C24-47--POPC-H4R-48_full.dat",
        "ordermap_POPC-C24-47--POPC-H4S-49_full.dat",
        "ordermap_POPC-C24-47_full.dat",
        "ordermap_average_full.dat",
    ];

    for file in expected_file_names {
        let real_file = format!("{}/POPC/{}", path_to_dir, file);
        let test_file = format!("tests/files/ordermaps/{}", file);
        assert_eq_maps(&real_file, &test_file, 2);
    }

    // full map for the entire system is the same as for POPC
    let real_file = format!("{}/ordermap_average_full.dat", path_to_dir);
    let test_file = "tests/files/ordermaps/ordermap_average_full.dat";
    assert_eq_maps(&real_file, test_file, 2);

    // check the script
    let real_script = format!("{}/plot.py", path_to_dir);
    assert!(common::diff_files_ignore_first(
        &real_script,
        "scripts/plot.py",
        0
    ));

    assert_eq_order(path_to_output, "tests/files/aa_order_small.yaml", 1);
}

#[test]
fn test_aa_order_maps_leaflets() {
    for method in [
        LeafletClassification::global("@membrane", "name P"),
        LeafletClassification::local("@membrane", "name P", 2.0),
        LeafletClassification::individual("name P", "name C218 C316"),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "resname POPC and name C22 C24 C218",
                "@membrane and element name hydrogen",
            ))
            .leaflets(method)
            .map(
                OrderMap::builder()
                    .bin_size([0.1, 4.0])
                    .output_directory(path_to_dir)
                    .min_samples(5)
                    .build()
                    .unwrap(),
            )
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        let expected_file_names = [
            "ordermap_POPC-C218-87_lower.dat",
            "ordermap_POPC-C218-87--POPC-H18S-89_upper.dat",
            "ordermap_POPC-C22-32_lower.dat",
            "ordermap_POPC-C22-32--POPC-H2S-34_upper.dat",
            "ordermap_POPC-C24-47--POPC-H4R-48_upper.dat",
            "ordermap_POPC-C218-87--POPC-H18R-88_lower.dat",
            "ordermap_POPC-C218-87--POPC-H18T-90_lower.dat",
            "ordermap_POPC-C22-32--POPC-H2R-33_lower.dat",
            "ordermap_POPC-C22-32_full.dat",
            "ordermap_POPC-C24-47--POPC-H4S-49_lower.dat",
            "ordermap_POPC-C218-87--POPC-H18R-88_full.dat",
            "ordermap_POPC-C218-87--POPC-H18T-90_full.dat",
            "ordermap_POPC-C22-32--POPC-H2R-33_full.dat",
            "ordermap_POPC-C22-32_full.dat",
            "ordermap_POPC-C24-47--POPC-H4S-49_full.dat",
            "ordermap_POPC-C218-87--POPC-H18R-88_upper.dat",
            "ordermap_POPC-C218-87--POPC-H18T-90_upper.dat",
            "ordermap_POPC-C22-32--POPC-H2R-33_upper.dat",
            "ordermap_POPC-C24-47_lower.dat",
            "ordermap_POPC-C24-47--POPC-H4S-49_upper.dat",
            "ordermap_POPC-C218-87--POPC-H18S-89_lower.dat",
            "ordermap_POPC-C218-87_full.dat",
            "ordermap_POPC-C22-32--POPC-H2S-34_lower.dat",
            "ordermap_POPC-C24-47--POPC-H4R-48_lower.dat",
            "ordermap_POPC-C24-47_full.dat",
            "ordermap_POPC-C218-87--POPC-H18S-89_full.dat",
            "ordermap_POPC-C218-87_upper.dat",
            "ordermap_POPC-C22-32--POPC-H2S-34_full.dat",
            "ordermap_POPC-C24-47--POPC-H4R-48_full.dat",
            "ordermap_POPC-C24-47_upper.dat",
            "ordermap_average_full.dat",
            "ordermap_average_upper.dat",
            "ordermap_average_lower.dat",
        ];

        for file in expected_file_names {
            let real_file = format!("{}/POPC/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        // full maps for the entire system are the same as for POPC
        for file in [
            "ordermap_average_full.dat",
            "ordermap_average_upper.dat",
            "ordermap_average_lower.dat",
        ] {
            let real_file = format!("{}/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps/{}", file);
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
            "tests/files/aa_order_leaflets_small.yaml",
            1,
        );
    }
}

#[test]
fn test_aa_order_maps_leaflets_full() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .map(
            OrderMap::builder()
                .bin_size([0.1, 4.0])
                .output_directory(path_to_dir)
                .min_samples(5)
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
        let test_file = format!("tests/files/ordermaps/full/{}", file);
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
fn test_aa_order_maps_leaflets_different_membrane_normals() {
    for (input_traj, (normal, structure)) in [
        "tests/files/pcpepg_switched_xz.xtc",
        "tests/files/pcpepg_switched_yz.xtc",
    ]
    .into_iter()
    .zip(
        [Axis::X, Axis::Y].into_iter().zip(
            [
                "tests/files/pcpepg_switched_xz.tpr",
                "tests/files/pcpepg_switched_yz.tpr",
            ]
            .into_iter(),
        ),
    ) {
        for method in [
            LeafletClassification::global("@membrane", "name P"),
            LeafletClassification::local("@membrane", "name P", 2.0),
            LeafletClassification::individual("name P", "name C218 C316"),
        ] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let directory = TempDir::new().unwrap();
            let path_to_dir = directory.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure(structure)
                .trajectory(input_traj)
                .membrane_normal(normal)
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "resname POPC and name C22 C24 C218",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(method)
                .map(
                    OrderMap::builder()
                        .bin_size([0.1, 4.0])
                        .output_directory(path_to_dir)
                        .min_samples(5)
                        .build()
                        .unwrap(),
                )
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            let expected_file_names = [
                "ordermap_POPC-C218-87_lower.dat",
                "ordermap_POPC-C218-87--POPC-H18S-89_upper.dat",
                "ordermap_POPC-C22-32_lower.dat",
                "ordermap_POPC-C22-32--POPC-H2S-34_upper.dat",
                "ordermap_POPC-C24-47--POPC-H4R-48_upper.dat",
                "ordermap_POPC-C218-87--POPC-H18R-88_lower.dat",
                "ordermap_POPC-C218-87--POPC-H18T-90_lower.dat",
                "ordermap_POPC-C22-32--POPC-H2R-33_lower.dat",
                "ordermap_POPC-C22-32_full.dat",
                "ordermap_POPC-C24-47--POPC-H4S-49_lower.dat",
                "ordermap_POPC-C218-87--POPC-H18R-88_full.dat",
                "ordermap_POPC-C218-87--POPC-H18T-90_full.dat",
                "ordermap_POPC-C22-32--POPC-H2R-33_full.dat",
                "ordermap_POPC-C22-32_upper.dat",
                "ordermap_POPC-C24-47--POPC-H4S-49_full.dat",
                "ordermap_POPC-C218-87--POPC-H18R-88_upper.dat",
                "ordermap_POPC-C218-87--POPC-H18T-90_upper.dat",
                "ordermap_POPC-C22-32--POPC-H2R-33_upper.dat",
                "ordermap_POPC-C24-47_lower.dat",
                "ordermap_POPC-C24-47--POPC-H4S-49_upper.dat",
                "ordermap_POPC-C218-87--POPC-H18S-89_lower.dat",
                "ordermap_POPC-C218-87_full.dat",
                "ordermap_POPC-C22-32--POPC-H2S-34_lower.dat",
                "ordermap_POPC-C24-47--POPC-H4R-48_lower.dat",
                "ordermap_POPC-C24-47_full.dat",
                "ordermap_POPC-C218-87--POPC-H18S-89_full.dat",
                "ordermap_POPC-C218-87_upper.dat",
                "ordermap_POPC-C22-32--POPC-H2S-34_full.dat",
                "ordermap_POPC-C24-47--POPC-H4R-48_full.dat",
                "ordermap_POPC-C24-47_upper.dat",
                "ordermap_average_full.dat",
                "ordermap_average_upper.dat",
                "ordermap_average_lower.dat",
            ];

            for file in expected_file_names {
                let real_file = format!("{}/POPC/{}", path_to_dir, file);
                let test_file = format!("tests/files/ordermaps/{}", file);
                assert_eq_maps(&real_file, &test_file, 4);
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
                "tests/files/aa_order_leaflets_small.yaml",
                1,
            );
        }
    }
}

#[test]
fn test_aa_order_maps_basic_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "resname POPC and name C22 C24 C218",
                "@membrane and element name hydrogen",
            ))
            .n_threads(n_threads)
            .map(
                OrderMap::builder()
                    .bin_size([0.1, 4.0])
                    .output_directory(path_to_dir)
                    .min_samples(5)
                    .build()
                    .unwrap(),
            )
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        let expected_file_names = [
            "ordermap_POPC-C218-87--POPC-H18R-88_full.dat",
            "ordermap_POPC-C218-87--POPC-H18S-89_full.dat",
            "ordermap_POPC-C218-87--POPC-H18T-90_full.dat",
            "ordermap_POPC-C218-87_full.dat",
            "ordermap_POPC-C22-32--POPC-H2R-33_full.dat",
            "ordermap_POPC-C22-32--POPC-H2S-34_full.dat",
            "ordermap_POPC-C22-32_full.dat",
            "ordermap_POPC-C24-47--POPC-H4R-48_full.dat",
            "ordermap_POPC-C24-47--POPC-H4S-49_full.dat",
            "ordermap_POPC-C24-47_full.dat",
            "ordermap_average_full.dat",
        ];

        for file in expected_file_names {
            let real_file = format!("{}/POPC/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        // full map for the entire system is the same as for POPC
        let real_file = format!("{}/ordermap_average_full.dat", path_to_dir);
        let test_file = "tests/files/ordermaps/ordermap_average_full.dat";
        assert_eq_maps(&real_file, test_file, 2);

        // check the script
        let real_script = format!("{}/plot.py", path_to_dir);
        assert!(common::diff_files_ignore_first(
            &real_script,
            "scripts/plot.py",
            0
        ));

        assert_eq_order(path_to_output, "tests/files/aa_order_small.yaml", 1);
    }
}

#[test]
fn test_aa_order_maps_basic_weird_molecules() {
    // calculation of ordermaps for system with molecules sharing their name and being composed of multiple residues
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/multiple_resid_same_name.tpr")
        .trajectory("tests/files/multiple_resid_same_name.xtc")
        .analysis_type(AnalysisType::aaorder(
            "resname POPC POPE and name C1A C3A C1B C3B",
            "resname POPC POPE and name D2A C4A C2B C4B",
        ))
        .map(
            OrderMap::builder()
                .bin_size([0.1, 4.0])
                .output_directory(path_to_dir)
                .min_samples(1)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    let expected_file_names = [
        "POPC-POPE1/ordermap_POPC-C1A-4--POPC-D2A-5_full.dat",
        "POPC-POPE1/ordermap_POPC-D2A-5--POPE-C3A-6_full.dat",
        "POPC-POPE1/ordermap_POPE-C3A-6--POPE-C4A-7_full.dat",
        "POPC-POPE1/ordermap_POPE-C1B-8--POPE-C2B-9_full.dat",
        "POPC-POPE1/ordermap_POPE-C2B-9--POPE-C3B-10_full.dat",
        "POPC-POPE1/ordermap_POPE-C3B-10--POPE-C4B-11_full.dat",
        "POPC-POPE1/ordermap_POPC-C1A-4_full.dat",
        "POPC-POPE1/ordermap_POPE-C3A-6_full.dat",
        "POPC-POPE1/ordermap_POPE-C1B-8_full.dat",
        "POPC-POPE1/ordermap_POPE-C3B-10_full.dat",
        "POPC-POPE1/ordermap_average_full.dat",
        "POPC-POPE2/ordermap_POPC-C1A-4--POPC-D2A-5_full.dat",
        "POPC-POPE2/ordermap_POPC-D2A-5--POPE-C3A-6_full.dat",
        "POPC-POPE2/ordermap_POPE-C3A-6--POPE-C4A-7_full.dat",
        "POPC-POPE2/ordermap_POPE-C3B-10--POPE-C4B-11_full.dat",
        "POPC-POPE2/ordermap_POPC-C1A-4_full.dat",
        "POPC-POPE2/ordermap_POPE-C3A-6_full.dat",
        "POPC-POPE2/ordermap_POPE-C3B-10_full.dat",
        "POPC-POPE2/ordermap_POPC-C1A-4--POPC-D2A-5_full.dat",
        "POPC-POPE2/ordermap_average_full.dat",
        "POPC/ordermap_POPC-D2A-5--POPC-C3A-6_full.dat",
        "POPC/ordermap_POPC-C3A-6--POPC-C4A-7_full.dat",
        "POPC/ordermap_POPC-C1B-8--POPC-C2B-9_full.dat",
        "POPC/ordermap_POPC-C2B-9--POPC-C3B-10_full.dat",
        "POPC/ordermap_POPC-C3B-10--POPC-C4B-11_full.dat",
        "POPC/ordermap_POPC-C1A-4_full.dat",
        "POPC/ordermap_POPC-C3A-6_full.dat",
        "POPC/ordermap_POPC-C1B-8_full.dat",
        "POPC/ordermap_POPC-C3B-10_full.dat",
        "POPC/ordermap_average_full.dat",
    ];

    for file in expected_file_names {
        let real_file = format!("{}/{}", path_to_dir, file);
        assert!(Path::new(&real_file).exists());
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
fn test_aa_order_maps_basic_backup() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let outer_directory = TempDir::new().unwrap();
    let path_to_outer_dir = outer_directory.path().to_str().unwrap();

    let directory = TempDir::new_in(path_to_outer_dir).unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let backup_file = format!("{}/to_backup.txt", path_to_dir);
    create_file_for_backup!(&backup_file);

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "resname POPC and name C22 C24 C218",
            "@membrane and element name hydrogen",
        ))
        .map(
            OrderMap::builder()
                .bin_size([0.1, 4.0])
                .output_directory(path_to_dir)
                .min_samples(5)
                .build()
                .unwrap(),
        )
        .silent()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    let expected_file_names = [
        "ordermap_POPC-C218-87--POPC-H18R-88_full.dat",
        "ordermap_POPC-C218-87--POPC-H18S-89_full.dat",
        "ordermap_POPC-C218-87--POPC-H18T-90_full.dat",
        "ordermap_POPC-C218-87_full.dat",
        "ordermap_POPC-C22-32--POPC-H2R-33_full.dat",
        "ordermap_POPC-C22-32--POPC-H2S-34_full.dat",
        "ordermap_POPC-C22-32_full.dat",
        "ordermap_POPC-C24-47--POPC-H4R-48_full.dat",
        "ordermap_POPC-C24-47--POPC-H4S-49_full.dat",
        "ordermap_POPC-C24-47_full.dat",
        "ordermap_average_full.dat",
    ];

    for file in expected_file_names {
        let real_file = format!("{}/POPC/{}", path_to_dir, file);
        let test_file = format!("tests/files/ordermaps/{}", file);
        assert_eq_maps(&real_file, &test_file, 2);
    }

    // check the script
    let real_script = format!("{}/plot.py", path_to_dir);
    assert!(common::diff_files_ignore_first(
        &real_script,
        "scripts/plot.py",
        0
    ));

    assert_eq_order(path_to_output, "tests/files/aa_order_small.yaml", 1);

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
fn test_aa_order_maps_basic_different_plane() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "resname POPC and name C22 C24 C218",
            "@membrane and element name hydrogen",
        ))
        .map(
            OrderMap::builder()
                .bin_size([4.0, 0.1])
                .output_directory(path_to_dir)
                .min_samples(5)
                .plane(Plane::XZ)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    // only test one file
    let real_file = format!("{}/POPC/ordermap_POPC-C218-87_full.dat", path_to_dir);
    let test_file = "tests/files/ordermaps/ordermap_xz.dat";
    assert_eq_maps(&real_file, test_file, 2);

    // check the script
    let real_script = format!("{}/plot.py", path_to_dir);
    assert!(common::diff_files_ignore_first(
        &real_script,
        "scripts/plot.py",
        0
    ));

    assert_eq_order(path_to_output, "tests/files/aa_order_small.yaml", 1);
}

#[test]
fn test_aa_order_error_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_error.yaml", 1);
}

#[test]
fn test_aa_order_error_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .n_threads(n_threads)
            .estimate_error(EstimateError::default())
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_error.yaml", 1);
    }
}

#[test]
fn test_aa_order_error_leaflets_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_error_leaflets.yaml",
        1,
    );
}

#[test]
fn test_aa_order_error_leaflets_yaml_multiple_threads() {
    for n_threads in [2, 3, 5, 8, 12, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .n_threads(n_threads)
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .estimate_error(EstimateError::default())
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/aa_order_error_leaflets.yaml",
            1,
        );
    }
}

#[test]
fn test_aa_order_error_tab() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_table, "tests/files/aa_order_error.tab", 1);
}

#[test]
fn test_aa_order_error_leaflets_tab() {
    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_tab(path_to_table)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_table, "tests/files/aa_order_error_leaflets.tab", 1);
}

#[test]
fn test_aa_order_error_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(path_to_csv, "tests/files/aa_order_error.csv", 0);
}

#[test]
fn test_aa_order_error_leaflets_csv() {
    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_csv(path_to_csv, "tests/files/aa_order_error_leaflets.csv", 0);
}

#[test]
fn test_aa_order_error_xvg() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let pattern = format!("{}/order.xvg", path_to_dir);

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_xvg(pattern)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    for molecule in ["POPC", "POPE", "POPG"] {
        let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
        // same files as when `estimate_error` is not provided - xvg files do not show error
        let path_expected = format!("tests/files/aa_order_basic_{}.xvg", molecule);

        assert_eq_order(&path, &path_expected, 1);
    }
}

#[test]
fn test_aa_order_error_leaflets_xvg() {
    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let pattern = format!("{}/order.xvg", path_to_dir);

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_xvg(pattern)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    for molecule in ["POPC", "POPE", "POPG"] {
        let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
        // same files as when `estimate_error` is not provided - xvg files do not show error
        let path_expected = format!("tests/files/aa_order_leaflets_{}.xvg", molecule);

        assert_eq_order(&path, &path_expected, 1);
    }
}

#[test]
fn test_aa_order_error_limit() {
    let output = NamedTempFile::new().unwrap();
    let path_to_yaml = output.path().to_str().unwrap();

    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_yaml(path_to_yaml)
        .output_tab(path_to_table)
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::default())
        .min_samples(2000)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_yaml, "tests/files/aa_order_error_limit.yaml", 1);
    assert_eq_order(path_to_table, "tests/files/aa_order_error_limit.tab", 1);
    assert_eq_csv(path_to_csv, "tests/files/aa_order_error_limit.csv", 0);
}

#[test]
fn test_aa_order_error_leaflets_limit() {
    let output = NamedTempFile::new().unwrap();
    let path_to_yaml = output.path().to_str().unwrap();

    let output_table = NamedTempFile::new().unwrap();
    let path_to_table = output_table.path().to_str().unwrap();

    let output_csv = NamedTempFile::new().unwrap();
    let path_to_csv = output_csv.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_yaml(path_to_yaml)
        .output_tab(path_to_table)
        .output_csv(path_to_csv)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .estimate_error(EstimateError::default())
        .min_samples(500)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_yaml,
        "tests/files/aa_order_error_leaflets_limit.yaml",
        1,
    );

    assert_eq_order(
        path_to_table,
        "tests/files/aa_order_error_leaflets_limit.tab",
        1,
    );

    assert_eq_csv(
        path_to_csv,
        "tests/files/aa_order_error_leaflets_limit.csv",
        0,
    );
}

#[test]
fn test_aa_order_error_blocks_10_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::new(Some(10), None).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_error_blocks10.yaml",
        1,
    );
}

#[test]
fn test_aa_order_error_blocks_too_many() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::new(Some(100), None).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed but it succeeded."),
        Err(e) => {
            assert!(e.to_string().contains("fewer than the number of blocks"))
        }
    }
}

#[test]
fn test_aa_order_convergence() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::new(None, Some(path_to_output)).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_convergence.xvg", 1);
}

#[test]
fn test_aa_order_leaflets_convergence() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .estimate_error(EstimateError::new(None, Some(path_to_output)).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_leaflets_convergence.xvg",
        1,
    );
}

#[test]
fn test_aa_order_convergence_multiple_threads() {
    for n_threads in [2, 5, 12, 32] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .estimate_error(EstimateError::new(None, Some(path_to_output)).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_convergence.xvg", 1);
    }
}

#[test]
fn test_aa_order_convergence_step() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::new(None, Some(path_to_output)).unwrap())
        .step(5)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_convergence_s5.xvg", 1);
}

#[test]
fn test_aa_order_leaflets_asymmetric_multiple_threads() {
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
            .structure("tests/files/asymmetric/aa_asym.tpr")
            .trajectory("tests/files/asymmetric/aa_asym.xtc")
            .output_yaml(path_to_yaml)
            .output_tab(path_to_tab)
            .output_csv(path_to_csv)
            .output_xvg(xvg_pattern)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_yaml,
            "tests/files/asymmetric/aa_order_asymmetric.yaml",
            1,
        );

        assert_eq_order(
            path_to_tab,
            "tests/files/asymmetric/aa_order_asymmetric.tab",
            1,
        );

        assert_eq_csv(
            path_to_csv,
            "tests/files/asymmetric/aa_order_asymmetric.csv",
            0,
        );

        for molecule in ["POPC", "POPE"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!(
                "tests/files/asymmetric/aa_order_asymmetric_{}.xvg",
                molecule
            );

            assert_eq_order(&path, &path_expected, 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_asymmetric_error_multiple_threads() {
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
            .structure("tests/files/asymmetric/aa_asym.tpr")
            .trajectory("tests/files/asymmetric/aa_asym.xtc")
            .output_yaml(path_to_yaml)
            .output_tab(path_to_tab)
            .output_csv(path_to_csv)
            .output_xvg(xvg_pattern)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .estimate_error(EstimateError::default())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_yaml,
            "tests/files/asymmetric/aa_order_asymmetric_errors.yaml",
            1,
        );

        assert_eq_order(
            path_to_tab,
            "tests/files/asymmetric/aa_order_asymmetric_errors.tab",
            1,
        );

        assert_eq_csv(
            path_to_csv,
            "tests/files/asymmetric/aa_order_asymmetric_errors.csv",
            0,
        );

        for molecule in ["POPC", "POPE"] {
            let path = format!("{}/order_{}.xvg", path_to_dir, molecule);
            let path_expected = format!(
                "tests/files/asymmetric/aa_order_asymmetric_{}.xvg",
                molecule
            );

            assert_eq_order(&path, &path_expected, 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_asymmetric_ordermaps_multiple_threads() {
    for n_threads in [1, 2, 5, 8] {
        let analysis = Analysis::builder()
            .structure("tests/files/asymmetric/aa_asym.tpr")
            .trajectory("tests/files/asymmetric/aa_asym.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and name C22 C24 C218",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .ordermaps(
                OrderMap::builder()
                    .bin_size([2.0, 2.0])
                    .dim([
                        GridSpan::manual(2.0, 8.0).unwrap(),
                        GridSpan::manual(2.0, 8.0).unwrap(),
                    ])
                    .build()
                    .unwrap(),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        let result = match analysis.run().unwrap() {
            AnalysisResults::AA(x) => x,
            _ => panic!("Incorrect results type returned."),
        };

        for molecule in result.molecules() {
            for atom in molecule.atoms() {
                let total = atom.ordermaps().total().as_ref().unwrap();
                let upper = atom.ordermaps().upper().as_ref().unwrap();
                let lower = atom.ordermaps().lower().as_ref().unwrap();

                if molecule.molecule() == "POPC" {
                    for ((_, _, t), (_, _, u)) in
                        total.extract_convert().zip(upper.extract_convert())
                    {
                        assert_relative_eq!(t, u, epsilon = 2e-4);
                    }

                    for (_, _, l) in lower.extract_convert() {
                        assert!(l.is_nan());
                    }
                } else if molecule.molecule() == "POPE" {
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

                for bond in atom.bonds() {
                    let total = bond.ordermaps().total().as_ref().unwrap();
                    let upper = bond.ordermaps().upper().as_ref().unwrap();
                    let lower = bond.ordermaps().lower().as_ref().unwrap();

                    if molecule.molecule() == "POPC" {
                        for ((_, _, t), (_, _, u)) in
                            total.extract_convert().zip(upper.extract_convert())
                        {
                            assert_relative_eq!(t, u, epsilon = 2e-4);
                        }

                        for (_, _, l) in lower.extract_convert() {
                            assert!(l.is_nan());
                        }
                    } else if molecule.molecule() == "POPE" {
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
}

#[test]
fn test_aa_order_geometry_cuboid_full() {
    for reference in [
        GeomReference::default(),
        Vector3D::new(2.0, 2.0, 3.0).into(),
        "@membrane".into(),
        GeomReference::center(),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
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

        assert_eq_order(path_to_output, "tests/files/aa_order_basic.yaml", 1);
    }
}

#[test]
fn test_aa_order_geometry_cylinder_full() {
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
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
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

            assert_eq_order(path_to_output, "tests/files/aa_order_basic.yaml", 1);
        }
    }
}

#[test]
fn test_aa_order_geometry_sphere_full() {
    for reference in [
        GeomReference::default(),
        Vector3D::new(2.0, 2.0, 3.0).into(),
        "@membrane".into(),
        GeomReference::center(),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .geometry(Geometry::sphere(reference, f32::INFINITY).unwrap())
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_basic.yaml", 1);
    }
}

#[test]
fn test_aa_order_geometry_cuboid_static_square_multiple_threads() {
    for n_threads in [1, 2, 3, 5, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "resname POPC and name C22 C24 C218",
                "@membrane and element name hydrogen",
            ))
            .geometry(
                Geometry::cuboid(
                    Vector3D::new(8.0, 2.0, 0.0),
                    [-2.0, 4.0],
                    [-4.0, 1.0],
                    [f32::NEG_INFINITY, f32::INFINITY],
                )
                .unwrap(),
            )
            .ordermap(
                OrderMap::builder()
                    .output_directory(path_to_dir)
                    .bin_size([0.5, 0.5])
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
            "ordermap_POPC-C218-87--POPC-H18R-88_full.dat",
            "ordermap_POPC-C218-87--POPC-H18S-89_full.dat",
            "ordermap_POPC-C218-87--POPC-H18T-90_full.dat",
            "ordermap_POPC-C218-87_full.dat",
            "ordermap_POPC-C22-32--POPC-H2R-33_full.dat",
            "ordermap_POPC-C22-32--POPC-H2S-34_full.dat",
            "ordermap_POPC-C22-32_full.dat",
            "ordermap_POPC-C24-47--POPC-H4R-48_full.dat",
            "ordermap_POPC-C24-47--POPC-H4S-49_full.dat",
            "ordermap_POPC-C24-47_full.dat",
            "ordermap_average_full.dat",
        ];

        for file in expected_file_names {
            let real_file = format!("{}/POPC/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps_cuboid/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        assert_eq_order(path_to_output, "tests/files/aa_order_cuboid_square.yaml", 1);
    }
}

#[test]
fn test_aa_order_geometry_cylinder_static_multiple_threads() {
    for n_threads in [1, 2, 3, 5, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "resname POPC and name C22 C24 C218",
                "@membrane and element name hydrogen",
            ))
            .geometry(
                Geometry::cylinder(
                    Vector3D::new(8.0, 2.0, 0.0),
                    2.5,
                    [f32::NEG_INFINITY, f32::INFINITY],
                    Axis::Z,
                )
                .unwrap(),
            )
            .ordermap(
                OrderMap::builder()
                    .output_directory(path_to_dir)
                    .bin_size([0.5, 0.5])
                    .min_samples(1)
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
            "ordermap_POPC-C218-87--POPC-H18R-88_full.dat",
            "ordermap_POPC-C218-87--POPC-H18S-89_full.dat",
            "ordermap_POPC-C218-87--POPC-H18T-90_full.dat",
            "ordermap_POPC-C218-87_full.dat",
            "ordermap_POPC-C22-32--POPC-H2R-33_full.dat",
            "ordermap_POPC-C22-32--POPC-H2S-34_full.dat",
            "ordermap_POPC-C22-32_full.dat",
            "ordermap_POPC-C24-47--POPC-H4R-48_full.dat",
            "ordermap_POPC-C24-47--POPC-H4S-49_full.dat",
            "ordermap_POPC-C24-47_full.dat",
            "ordermap_average_full.dat",
        ];

        for file in expected_file_names {
            let real_file = format!("{}/POPC/{}", path_to_dir, file);
            let test_file = format!("tests/files/ordermaps_cylinder/{}", file);
            assert_eq_maps(&real_file, &test_file, 2);
        }

        assert_eq_order(path_to_output, "tests/files/aa_order_cylinder.yaml", 1);
    }
}

#[test]
fn test_aa_order_geometry_sphere_static_multiple_threads() {
    for n_threads in [1, 2, 3, 5, 8, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .geometry(Geometry::sphere(Vector3D::new(8.0, 2.0, 4.5), 2.5).unwrap())
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_sphere_static.yaml", 1);
    }
}

#[test]
fn test_aa_order_geometry_cuboid_box_center_patch() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .geometry(
            Geometry::cuboid(
                GeomReference::center(),
                [-1.0, 3.0],
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

    assert_eq_order(path_to_output, "tests/files/aa_order_cuboid_patch.yaml", 1);
}

#[test]
fn test_aa_order_geometry_cylinder_box_center_x() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .geometry(Geometry::cylinder(GeomReference::center(), 3.0, [-1.0, 3.0], Axis::X).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_cylinder_x.yaml", 1);
}

#[test]
fn test_aa_order_geometry_sphere_box_center() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .geometry(Geometry::sphere(GeomReference::center(), 2.5).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_sphere_center.yaml", 1);
}

#[test]
fn test_aa_order_geometry_cuboid_dynamic() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .geometry(Geometry::cuboid("resid 1", [-1.0, 3.0], [1.0, 4.0], [-3.0, 3.0]).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_cuboid_dynamic.yaml",
        1,
    );
}

#[test]
fn test_aa_order_geometry_cylinder_dynamic() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .geometry(
            Geometry::cylinder("resid 1", 2.1, [f32::NEG_INFINITY, f32::INFINITY], Axis::Y)
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_cylinder_dynamic.yaml",
        1,
    );
}

#[test]
fn test_aa_order_geometry_sphere_dynamic() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .geometry(Geometry::sphere("resid 1", 2.5).unwrap())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_sphere_dynamic.yaml",
        1,
    );
}

#[test]
fn test_aa_order_geometry_cuboid_z() {
    let analysis_geometry = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and name C11 C12 C13 C14 C15 C1 C2 C3 C22 C32 C23 C33",
            "@membrane and element name hydrogen",
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
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let analysis_leaflets = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and name C11 C12 C13 C14 C15 C1 C2 C3 C22 C32 C23 C33",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results_geometry = analysis_geometry.run().unwrap();
    let results_leaflets = analysis_leaflets.run().unwrap();

    let results_geometry = match results_geometry {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    let results_leaflets = match results_leaflets {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    for (mol, mol2) in results_geometry
        .molecules()
        .zip(results_leaflets.molecules())
    {
        for (atom, atom2) in mol.atoms().zip(mol2.atoms()) {
            assert_relative_eq!(
                atom.order().total().unwrap().value(),
                atom.order().upper().unwrap().value()
            );
            assert!(atom.order().lower().unwrap().value().is_nan());

            assert_relative_eq!(
                atom.order().total().unwrap().value(),
                atom2.order().upper().unwrap().value()
            );

            for (bond, bond2) in atom.bonds().zip(atom2.bonds()) {
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
}

#[test]
fn test_aa_order_geometry_cylinder_z() {
    let analysis_geometry = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and name C11 C12 C13 C14 C15 C1 C2 C3 C22 C32 C23 C33",
            "@membrane and element name hydrogen",
        ))
        .geometry(Geometry::cylinder("@membrane", f32::INFINITY, [0.0, 3.5], Axis::Z).unwrap())
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let analysis_leaflets = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and name C11 C12 C13 C14 C15 C1 C2 C3 C22 C32 C23 C33",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results_geometry = analysis_geometry.run().unwrap();
    let results_leaflets = analysis_leaflets.run().unwrap();

    let results_geometry = match results_geometry {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    let results_leaflets = match results_leaflets {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    for (mol, mol2) in results_geometry
        .molecules()
        .zip(results_leaflets.molecules())
    {
        for (atom, atom2) in mol.atoms().zip(mol2.atoms()) {
            assert_relative_eq!(
                atom.order().total().unwrap().value(),
                atom.order().upper().unwrap().value()
            );
            assert!(atom.order().lower().unwrap().value().is_nan());

            assert_relative_eq!(
                atom.order().total().unwrap().value(),
                atom2.order().upper().unwrap().value()
            );

            for (bond, bond2) in atom.bonds().zip(atom2.bonds()) {
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
}

#[test]
fn test_aa_order_leaflets_from_file_once_multiple_threads() {
    for n_threads in [1, 2, 5, 8, 16] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_yaml(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(
                LeafletClassification::from_file(
                    "tests/files/inputs/leaflets_files/pcpepg_once.yaml",
                )
                .with_frequency(Frequency::once()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_aa_order_leaflets_from_map_once() {
    let mut file = File::open("tests/files/inputs/leaflets_files/pcpepg_once.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_yaml(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::from_map(assignment).with_frequency(Frequency::once()))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_from_file_once_asymmetric() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/asymmetric/aa_asym.tpr")
        .trajectory("tests/files/asymmetric/aa_asym.xtc")
        .output_yaml(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file("tests/files/inputs/leaflets_files/asym_once.yaml")
                .with_frequency(Frequency::once()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/asymmetric/aa_order_asymmetric.yaml",
        1,
    );
}

#[test]
fn test_aa_order_leaflets_from_file_every10_multiple_threads() {
    for n_threads in [1, 2, 5, 8, 16] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_yaml(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(
                LeafletClassification::from_file(
                    "tests/files/inputs/leaflets_files/pcpepg_every10.yaml",
                )
                .with_frequency(Frequency::every(10).unwrap()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_aa_order_leaflets_from_map_every10() {
    let mut file = File::open("tests/files/inputs/leaflets_files/pcpepg_every10.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_yaml(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_map(assignment)
                .with_frequency(Frequency::every(10).unwrap()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_from_file_every10_stepping() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_yaml(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_every10.yaml",
            )
            .with_frequency(Frequency::every(2).unwrap()),
        )
        .step(5)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_step.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_from_file_every_multiple_threads() {
    for n_threads in [1, 2, 5, 8, 16] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_yaml(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_every.yaml",
            ))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_aa_order_leaflets_from_map_every() {
    let mut file = File::open("tests/files/inputs/leaflets_files/pcpepg_every.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .output_yaml(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::from_map(assignment))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
}

#[test]
fn test_aa_order_leaflets_from_file_every_begin_end_step_multiple_threads() {
    for n_threads in [1, 2, 5, 8, 16] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output_yaml(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_every_begin_end_step.yaml",
            ))
            .n_threads(n_threads)
            .begin(450_200.0)
            .end(450_400.0)
            .step(3)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/aa_order_begin_end_step.yaml",
            1,
        );
    }
}

#[test]
fn test_aa_order_leaflets_from_file_fail_missing_molecule_type() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_missing_moltype.yaml",
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
fn test_aa_order_leaflets_from_map_fail_missing_molecule_type() {
    let mut file =
        File::open("tests/files/inputs/leaflets_files/pcpepg_missing_moltype.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::from_map(assignment).with_frequency(Frequency::once()))
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
fn test_aa_order_leaflets_from_file_fail_unexpected_molecule_type() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_unexpected_moltype.yaml",
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
            .contains("specified in the leaflet assignment structure not found in the system")),
    }
}

#[test]
fn test_aa_order_leaflets_from_map_fail_unexpected_molecule_type() {
    let mut file =
        File::open("tests/files/inputs/leaflets_files/pcpepg_unexpected_moltype.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
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
fn test_aa_order_leaflets_from_file_fail_nonexistent() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_nonexistent.yaml",
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
fn test_aa_order_leaflets_from_file_fail_invalid() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_invalid.yaml",
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
fn test_aa_order_leaflets_from_file_fail_invalid_number_of_molecules() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_invalid_number.yaml",
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
            .contains("inconsistent number of molecules specified in the leaflet assignment")),
    }
}

#[test]
fn test_aa_order_leaflets_from_map_fail_invalid_number_of_molecules() {
    let mut file =
        File::open("tests/files/inputs/leaflets_files/pcpepg_invalid_number.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
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
fn test_aa_order_leaflets_from_file_fail_empty_assignment() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file("tests/files/inputs/leaflets_files/pcpepg_empty.yaml")
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
fn test_aa_order_leaflets_from_map_fail_empty_assignment() {
    let mut file = File::open("tests/files/inputs/leaflets_files/pcpepg_empty.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::from_map(assignment).with_frequency(Frequency::once()))
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
fn test_aa_order_leaflets_from_file_too_many_frames() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_every10.yaml",
            )
            .with_frequency(Frequency::every(2).unwrap()),
        )
        .step(10)
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
fn test_aa_order_leaflets_from_map_too_many_frames() {
    let mut file = File::open("tests/files/inputs/leaflets_files/pcpepg_every10.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_map(assignment)
                .with_frequency(Frequency::every(2).unwrap()),
        )
        .step(10)
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
fn test_aa_order_leaflets_from_file_not_enough_frames() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_file(
                "tests/files/inputs/leaflets_files/pcpepg_every10.yaml",
            )
            .with_frequency(Frequency::every(8).unwrap()),
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
fn test_aa_order_leaflets_from_map_not_enough_frames() {
    let mut file = File::open("tests/files/inputs/leaflets_files/pcpepg_every10.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_map(assignment)
                .with_frequency(Frequency::every(8).unwrap()),
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
fn test_aa_order_leaflets_no_pbc_multiple_threads() {
    for n_threads in [1, 3, 8, 16] {
        for structure in ["tests/files/pcpepg.tpr", "tests/files/pcpepg_nobox.pdb"] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure(structure)
                .trajectory("tests/files/pcpepg_whole_nobox.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(LeafletClassification::global("@membrane", "name P"))
                .handle_pbc(false)
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            assert_eq_order(
                path_to_output,
                "tests/files/aa_order_leaflets_nopbc.yaml",
                1,
            );
        }
    }
}

#[test]
fn test_aa_order_leaflets_clustering_no_pbc() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg_whole_nobox.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::clustering("name P"))
        .handle_pbc(false)
        .n_threads(4)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_leaflets_nopbc.yaml",
        1,
    );
}

#[test]
fn test_aa_order_error_leaflets_no_pbc_multiple_threads() {
    for n_threads in [1, 3, 8, 16] {
        for structure in ["tests/files/pcpepg.tpr", "tests/files/pcpepg_nobox.pdb"] {
            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure(structure)
                .trajectory("tests/files/pcpepg_whole_nobox.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(LeafletClassification::global("@membrane", "name P"))
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
                "tests/files/aa_order_error_leaflets_nopbc.yaml",
                1,
            );
        }
    }
}

#[test]
fn test_aa_order_maps_leaflets_no_pbc() {
    for method in [
        LeafletClassification::global("@membrane", "name P"),
        LeafletClassification::local("@membrane", "name P", 2.0),
        LeafletClassification::individual("name P", "name C218 C316"),
    ] {
        for structure in ["tests/files/pcpepg.tpr", "tests/files/pcpepg_nobox.pdb"] {
            let directory = TempDir::new().unwrap();
            let path_to_dir = directory.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure(structure)
                .trajectory("tests/files/pcpepg_whole_nobox.xtc")
                .analysis_type(AnalysisType::aaorder(
                    "resname POPC and name C22 C24 C218",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(method.clone())
                .map(
                    OrderMap::builder()
                        .bin_size([0.1, 4.0])
                        .output_directory(path_to_dir)
                        .min_samples(5)
                        .dim([
                            GridSpan::manual(0.0, 9.0).unwrap(),
                            GridSpan::manual(0.0, 8.0).unwrap(),
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
                let test_file = format!("tests/files/ordermaps_nopbc/{}", file);
                assert_eq_maps(&real_file, &test_file, 2);
            }

            // full maps for the entire system are the same as for POPC
            for file in [
                "ordermap_average_full.dat",
                "ordermap_average_upper.dat",
                "ordermap_average_lower.dat",
            ] {
                let real_file = format!("{}/{}", path_to_dir, file);
                let test_file = format!("tests/files/ordermaps_nopbc/{}", file);
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
fn test_aa_order_geometry_cuboid_ordermaps_no_pbc() {
    for structure in ["tests/files/pcpepg.tpr", "tests/files/pcpepg_nobox.pdb"] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure(structure)
            .trajectory("tests/files/pcpepg_whole_nobox.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .geometry(
                Geometry::cuboid(
                    [1.5, -3.0, 3.0],
                    [f32::NEG_INFINITY, f32::INFINITY],
                    [-1.0, 4.0],
                    [-2.0, 2.0],
                )
                .unwrap(),
            )
            .map(
                OrderMap::builder()
                    .bin_size([1.0, 1.0])
                    .output_directory(path_to_dir)
                    .dim([
                        GridSpan::manual(0.0, 9.0).unwrap(),
                        GridSpan::manual(0.0, 9.0).unwrap(),
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

        assert_eq_order(path_to_output, "tests/files/aa_order_cuboid_nopbc.yaml", 1);

        let real_file = format!("{}/{}", path_to_dir, "ordermap_average_full.dat");
        let test_file = "tests/files/ordermaps_nopbc/cuboid.dat";
        assert_eq_maps(&real_file, test_file, 2);
    }
}

#[test]
fn test_aa_order_geometry_cylinder_ordermaps_no_pbc() {
    for structure in ["tests/files/pcpepg.tpr", "tests/files/pcpepg_nobox.pdb"] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure(structure)
            .trajectory("tests/files/pcpepg_whole_nobox.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .geometry(Geometry::cylinder([8.0, 2.0, 0.0], 3.5, [-2.0, 3.0], Axis::X).unwrap())
            .map(
                OrderMap::builder()
                    .bin_size([1.0, 1.0])
                    .output_directory(path_to_dir)
                    .dim([
                        GridSpan::manual(0.0, 9.0).unwrap(),
                        GridSpan::manual(0.0, 9.0).unwrap(),
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
            "tests/files/aa_order_cylinder_nopbc.yaml",
            1,
        );

        let real_file = format!("{}/{}", path_to_dir, "ordermap_average_full.dat");
        let test_file = "tests/files/ordermaps_nopbc/cylinder.dat";
        assert_eq_maps(&real_file, test_file, 2);
    }
}

#[test]
fn test_aa_order_geometry_sphere_ordermaps_no_pbc() {
    for structure in ["tests/files/pcpepg.tpr", "tests/files/pcpepg_nobox.pdb"] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure(structure)
            .trajectory("tests/files/pcpepg_whole_nobox.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .geometry(Geometry::sphere([8.0, 2.0, 3.9], 3.5).unwrap())
            .map(
                OrderMap::builder()
                    .bin_size([1.0, 1.0])
                    .output_directory(path_to_dir)
                    .dim([
                        GridSpan::manual(0.0, 9.0).unwrap(),
                        GridSpan::manual(0.0, 9.0).unwrap(),
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

        assert_eq_order(path_to_output, "tests/files/aa_order_sphere_nopbc.yaml", 1);

        let real_file = format!("{}/{}", path_to_dir, "ordermap_average_full.dat");
        let test_file = "tests/files/ordermaps_nopbc/sphere.dat";
        assert_eq_maps(&real_file, test_file, 2);
    }
}

#[test]
fn test_aa_order_geometry_no_pbc_fail_box_center() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg_whole_nobox.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .geometry(
            Geometry::cylinder(
                GeomReference::center(),
                2.5,
                [f32::NEG_INFINITY, f32::INFINITY],
                Axis::Z,
            )
            .unwrap(),
        )
        .handle_pbc(false)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("cannot use dynamic center of simulation box as the reference position")),
    }
}

#[test]
fn test_aa_order_maps_leaflets_no_pbc_fail_autodim() {
    for structure in ["tests/files/pcpepg.tpr", "tests/files/pcpepg_nobox.pdb"] {
        let directory = TempDir::new().unwrap();
        let path_to_dir = directory.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure(structure)
            .trajectory("tests/files/pcpepg_whole_nobox.xtc")
            .analysis_type(AnalysisType::aaorder(
                "resname POPC and name C22 C24 C218",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::global("@membrane", "name P"))
            .map(
                OrderMap::builder()
                    .bin_size([0.1, 4.0])
                    .output_directory(path_to_dir)
                    .min_samples(5)
                    .build()
                    .unwrap(),
            )
            .handle_pbc(false)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        match analysis.run() {
            Ok(_) => panic!("Analysis should have failed."),
            Err(e) => assert!(e
                .to_string()
                .contains("simulation box and periodic boundary conditions are ignored")),
        }
    }
}

#[test]
fn test_aa_order_basic_yaml_nobox_xtc_fail() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg_whole_nobox.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("all dimensions of the simulation box are zero")),
    }
}

#[test]
fn test_aa_order_basic_yaml_nobox_pdb_fail() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg_nobox.pdb")
        .trajectory("tests/files/pcpepg.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("system has undefined simulation box")),
    }
}

#[test]
fn test_aa_order_leaflets_yaml_shifted_trajectory() {
    let mut file = File::open("tests/files/inputs/leaflets_files/pcpepg_once.yaml").unwrap();
    let assignment: HashMap<String, Vec<Vec<Leaflet>>> =
        serde_yaml::from_reader(&mut file).unwrap();

    for method in [
        LeafletClassification::global("@membrane", "name P"),
        LeafletClassification::local("@membrane", "name P", 2.5),
        LeafletClassification::individual("name P", "name C218 C316"),
        LeafletClassification::from_map(assignment).with_frequency(Frequency::once()),
    ] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg_shifted.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(method)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/aa_order_leaflets_shifted.yaml",
            1,
        );
    }
}

#[test]
fn test_aa_order_leaflets_dynamic_membrane_normal_yaml() {
    for n_threads in [1, 3, 8, 16] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .membrane_normal(DynamicNormal::new("name P", 2.0).unwrap())
            .leaflets(
                LeafletClassification::individual("name P", "name C218 C316")
                    .with_membrane_normal(Axis::Z)
                    .with_frequency(Frequency::once()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(
            path_to_output,
            "tests/files/aa_order_leaflets_dynamic.yaml",
            1,
        );
    }
}

#[test]
fn test_aa_order_buckled_dynamic_membrane_normal_ordermaps_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let directory = TempDir::new().unwrap();
    let path_to_dir = directory.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/aa_buckled.tpr")
        .trajectory("tests/files/aa_buckled.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .membrane_normal(DynamicNormal::new("name P", 2.0).unwrap())
        .map(
            OrderMap::builder()
                .bin_size([1.0, 1.0])
                .output_directory(path_to_dir)
                .min_samples(5)
                .plane(Plane::XY)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(path_to_output, "tests/files/aa_order_buckled.yaml", 1);

    let real_file = format!("{}/ordermap_average_full.dat", path_to_dir);
    let test_file = "tests/files/ordermaps_buckled/ordermap_average_full.dat";
    assert_eq_maps(&real_file, test_file, 2);

    let real_file = format!("{}/POPC/ordermap_average_full.dat", path_to_dir);
    assert_eq_maps(&real_file, test_file, 2);

    // check the script
    let real_script = format!("{}/plot.py", path_to_dir);
    assert!(common::diff_files_ignore_first(
        &real_script,
        "scripts/plot.py",
        0
    ));
}

#[test]
fn test_aa_order_buckled_leaflets_clustering_yaml() {
    let output = NamedTempFile::new().unwrap();
    let path_to_output = output.path().to_str().unwrap();

    let analysis = Analysis::builder()
        .structure("tests/files/aa_buckled.tpr")
        .trajectory("tests/files/aa_buckled.xtc")
        .output(path_to_output)
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .membrane_normal(DynamicNormal::new("name P", 2.0).unwrap())
        .leaflets(LeafletClassification::clustering("name P"))
        .n_threads(8)
        .silent()
        .overwrite()
        .build()
        .unwrap();

    analysis.run().unwrap().write().unwrap();

    assert_eq_order(
        path_to_output,
        "tests/files/aa_order_buckled_leaflets.yaml",
        1,
    );
}

#[test]
fn test_aa_order_buckled_membrane_normals_from_file_from_map_min_yaml() {
    let string = read_to_string("tests/files/normals_aa_buckled_min.yaml").unwrap();
    let normals_map: HashMap<String, Vec<Vec<Vector3D>>> = serde_yaml::from_str(&string).unwrap();

    let mut output_paths = Vec::new();
    let mut outputs = Vec::new();
    for normals in [
        MembraneNormal::from(DynamicNormal::new("name P", 2.0).unwrap()),
        MembraneNormal::from(normals_map),
        MembraneNormal::from("tests/files/normals_aa_buckled_min.yaml"),
    ] {
        let output = NamedTempFile::new().unwrap();
        outputs.push(output);
        let path_to_output = outputs.last().unwrap().path().to_str().unwrap();
        output_paths.push(path_to_output.to_owned());

        let analysis = Analysis::builder()
            .structure("tests/files/aa_buckled.tpr")
            .trajectory("tests/files/aa_buckled.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .membrane_normal(normals)
            .end(1630000.0)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();
    }

    assert_eq_order(&output_paths[0], &output_paths[1], 1);
    assert_eq_order(&output_paths[1], &output_paths[2], 1);
}

#[test]
fn test_aa_order_buckled_membrane_normals_export() {
    for n_threads in [1, 2, 3, 8] {
        let output_normals = NamedTempFile::new().unwrap();
        let path_to_output_normals = output_normals.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/aa_buckled.tpr")
            .trajectory("tests/files/aa_buckled.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .membrane_normal(
                DynamicNormal::new("name P", 2.0)
                    .unwrap()
                    .with_collect(path_to_output_normals),
            )
            .end(1630000.0)
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_normals(
            path_to_output_normals,
            "tests/files/normals_aa_buckled_min.yaml",
        );
    }
}

#[test]
fn test_aa_order_dynamic_normals_multiple_threads_various_steps_export_and_load() {
    for n_threads in [1, 3, 8, 64] {
        for step in [1, 3, 7] {
            // export the normals
            let output_orig = NamedTempFile::new().unwrap();
            let path_to_output_orig = output_orig.path().to_str().unwrap();

            let output_normals = NamedTempFile::new().unwrap();
            let path_to_output_normals = output_normals.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output_orig)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .membrane_normal(
                    DynamicNormal::new("name P", 2.0)
                        .unwrap()
                        .with_collect(path_to_output_normals),
                )
                .n_threads(n_threads)
                .step(step)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            // load the exported normals
            let output_recalc = NamedTempFile::new().unwrap();
            let path_to_output_recalc = output_recalc.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output_recalc)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .membrane_normal(path_to_output_normals)
                .n_threads(1) // always one thread
                .step(step)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            analysis.run().unwrap().write().unwrap();

            // order parameters should match
            assert_eq_order(path_to_output_orig, path_to_output_recalc, 1);
        }
    }
}

#[test]
fn test_aa_order_dynamic_normals_export_incomplete() {
    for n_threads in [1, 2, 3, 8] {
        let output_normals = NamedTempFile::new().unwrap();
        let path_to_output_normals = output_normals.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .membrane_normal(
                DynamicNormal::new("name P", 2.0)
                    .unwrap()
                    .with_collect(path_to_output_normals),
            )
            .geometry(
                Geometry::cylinder(
                    GeomReference::Center,
                    2.5,
                    [f32::NEG_INFINITY, f32::INFINITY],
                    Axis::Z,
                )
                .unwrap(),
            )
            .step(10)
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_normals(
            path_to_output_normals,
            "tests/files/normals_incomplete.yaml",
        );
    }
}

#[test]
fn test_aa_order_buckled_membrane_normals_from_file_fail_unknown_molecule() {
    let analysis = Analysis::builder()
        .structure("tests/files/aa_buckled.tpr")
        .trajectory("tests/files/aa_buckled.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .membrane_normal("tests/files/normals_aa_buckled_unknown_mol.yaml")
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("specified in the normals structure not found in the system")),
    }
}

#[test]
fn test_aa_order_buckled_membrane_normals_from_file_fail_missing_frames() {
    let analysis = Analysis::builder()
        .structure("tests/files/aa_buckled.tpr")
        .trajectory("tests/files/aa_buckled.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .membrane_normal("tests/files/normals_aa_buckled_min.yaml")
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed."),
        Err(e) => assert!(e
            .to_string()
            .contains("could not get membrane normals for frame")),
    }
}

#[test]
fn test_aa_order_fail_dynamic_undefined_ordermap_plane() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .membrane_normal(DynamicNormal::new("name P", 2.0).unwrap())
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
fn test_aa_order_fail_dynamic_undefined_leaflet_normal() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .membrane_normal(DynamicNormal::new("name P", 2.0).unwrap())
        .leaflets(LeafletClassification::individual(
            "name P",
            "name C218 C316",
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
fn test_aa_order_fail_dynamic_multiple_heads() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .membrane_normal(DynamicNormal::new("name P O13", 2.0).unwrap())
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
fn test_aa_order_fail_dynamic_no_head() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .membrane_normal(DynamicNormal::new("name OH2", 2.0).unwrap())
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
fn test_aa_order_leaflets_from_ndx_once_multiple_threads() {
    for n_threads in [1, 3, 5, 8, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(
                LeafletClassification::from_ndx(
                    &["tests/files/ndx/pcpepg_leaflets.ndx"],
                    "name P",
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

        assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_aa_order_leaflets_from_ndx_every_multiple_threads() {
    let mut ndx = [
        "tests/files/ndx/pcpepg_leaflets.ndx",
        "tests/files/ndx/pcpepg_leaflets_all.ndx",
    ]
    .repeat(25);
    ndx.push("tests/files/ndx/pcpepg_leaflets.ndx");

    for n_threads in [1, 3, 5, 8, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(LeafletClassification::from_ndx(
                &ndx, "name P", "Upper", "Lower",
            ))
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_aa_order_leaflets_from_ndx_every10_multiple_threads() {
    let ndx = vec![
        "tests/files/ndx/pcpepg_leaflets.ndx",
        "tests/files/ndx/pcpepg_leaflets_all.ndx",
        "tests/files/ndx/pcpepg_leaflets_duplicate_irrelevant.ndx",
        "tests/files/ndx/pcpepg_leaflets_invalid_irrelevant.ndx",
        "tests/files/ndx/pcpepg_leaflets.ndx",
        "tests/files/ndx/pcpepg_leaflets_all.ndx",
    ];

    for n_threads in [1, 3, 5, 8, 16, 64] {
        let output = NamedTempFile::new().unwrap();
        let path_to_output = output.path().to_str().unwrap();

        let analysis = Analysis::builder()
            .structure("tests/files/pcpepg.tpr")
            .trajectory("tests/files/pcpepg.xtc")
            .output(path_to_output)
            .analysis_type(AnalysisType::aaorder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ))
            .leaflets(
                LeafletClassification::from_ndx(&ndx, "name P", "Upper", "Lower")
                    .with_frequency(Frequency::every(10).unwrap()),
            )
            .n_threads(n_threads)
            .silent()
            .overwrite()
            .build()
            .unwrap();

        analysis.run().unwrap().write().unwrap();

        assert_eq_order(path_to_output, "tests/files/aa_order_leaflets.yaml", 1);
    }
}

#[test]
fn test_aa_order_leaflets_from_ndx_step_multiple_threads() {
    let ndx = ["tests/files/ndx/pcpepg_leaflets.ndx"];
    for n_threads in [1, 2, 5, 8, 64] {
        for (repeat, freq) in [11, 3, 1, 1].into_iter().zip([
            Frequency::every(1).unwrap(),
            Frequency::every(5).unwrap(),
            Frequency::every(30).unwrap(),
            Frequency::once(),
        ]) {
            let ndx = ndx.repeat(repeat);

            let output = NamedTempFile::new().unwrap();
            let path_to_output = output.path().to_str().unwrap();

            let analysis = Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .output(path_to_output)
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .step(5)
                .leaflets(
                    LeafletClassification::from_ndx(&ndx, "name P", "Upper", "Lower")
                        .with_frequency(freq),
                )
                .n_threads(n_threads)
                .silent()
                .overwrite()
                .build()
                .unwrap();

            let results = analysis.run().unwrap();
            assert_eq!(results.n_analyzed_frames(), 11);
            results.write().unwrap();

            assert_eq_order(path_to_output, "tests/files/aa_order_step.yaml", 1);
        }
    }
}

#[test]
fn test_aa_order_leaflets_from_ndx_fail_missing_ndx() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_ndx(
                &[
                    "tests/files/ndx/pcpepg_leaflets.ndx",
                    "tests/files/ndx/pcpepg_leaflets.ndx",
                    "tests/files/ndx/pcpepg_leaflets.ndx",
                    "tests/files/ndx/pcpepg_leaflets.ndx",
                    "tests/files/ndx/pcpepg_leaflets.ndx",
                ],
                "name P",
                "Upper",
                "Lower",
            )
            .with_frequency(Frequency::every(10).unwrap()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed but it succeeded."),
        Err(e) => {
            assert!(e.to_string().contains("could not get ndx file for frame"))
        }
    }
}

#[test]
fn test_aa_order_leaflets_from_ndx_fail_too_many_ndx() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(
            LeafletClassification::from_ndx(
                &[
                    "tests/files/ndx/pcpepg_leaflets.ndx",
                    "tests/files/ndx/pcpepg_leaflets_all.ndx",
                ],
                "name P",
                "Upper",
                "Lower",
            )
            .with_frequency(Frequency::once()),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    match analysis.run() {
        Ok(_) => panic!("Analysis should have failed but it succeeded."),
        Err(e) => {
            assert!(e
                .to_string()
                .contains("is not consistent with the number of analyzed frames"))
        }
    }
}

#[test]
fn test_aa_order_basic_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert_eq!(results.analysis().structure(), "tests/files/pcpepg.tpr");
    assert!(results.leaflets_data().is_none());

    assert_eq!(results.molecules().count(), 3);

    assert!(results.get_molecule("POPE").is_some());
    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPG").is_some());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.1423,
        epsilon = 2e-4
    );
    assert!(results.average_order().upper().is_none());
    assert!(results.average_order().lower().is_none());

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_average_orders = [0.1455, 0.1378, 0.1561];
    let expected_atom_numbers = [37, 40, 38];
    let expected_molecule_names = ["POPE", "POPC", "POPG"];

    let expected_atom_indices = [32, 41, 34];
    let expected_atom_names = ["C32", "C32", "C32"];
    let expected_atom_order = [0.2226, 0.2363, 0.2247];

    let expected_bond_numbers = [2, 2, 2];

    let expected_atom2_indices = [34, 43, 36];
    let expected_atom2_names = ["H2Y", "H2Y", "H2Y"];
    let expected_atom2_order = [0.2040, 0.2317, 0.2020];

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

        let bond = atom.get_bond(expected_atom2_indices[i]).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.atom_name(), expected_atom_names[i]);
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a1.residue_name(), expected_molecule_names[i]);
        assert_eq!(a2.atom_name(), expected_atom2_names[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);
        assert_eq!(a2.residue_name(), expected_molecule_names[i]);
        assert_eq!(bond.molecule(), expected_molecule_names[i]);

        let order = bond.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_atom2_order[i],
            epsilon = 2e-4
        );
        assert!(order.total().unwrap().error().is_none());
        assert!(order.upper().is_none());
        assert!(order.lower().is_none());

        let maps = bond.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bond directly from molecule
        let bond = molecule
            .get_bond(expected_atom_indices[i], expected_atom2_indices[i])
            .unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);

        let bond = molecule
            .get_bond(expected_atom2_indices[i], expected_atom_indices[i])
            .unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);

        // nonexistent atom
        assert!(molecule.get_atom(145).is_none());
        // nonexistent bond
        assert!(molecule.get_bond(7, 19).is_none());
        assert!(molecule.get_bond(145, 189).is_none());
    }
}

#[test]
fn test_aa_order_error_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert_eq!(results.molecules().count(), 3);

    assert!(results.get_molecule("POPE").is_some());
    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPG").is_some());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.1423,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().total().unwrap().error().unwrap(),
        0.0026,
        epsilon = 2e-4
    );
    assert!(results.average_order().upper().is_none());
    assert!(results.average_order().lower().is_none());

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_average_orders = [0.1455, 0.1378, 0.1561];
    let expected_average_errors = [0.0029, 0.0036, 0.0112];
    let expected_atom_numbers = [37, 40, 38];
    let expected_molecule_names = ["POPE", "POPC", "POPG"];

    let expected_atom_indices = [32, 41, 34];
    let expected_atom_names = ["C32", "C32", "C32"];
    let expected_atom_order = [0.2226, 0.2363, 0.2247];
    let expected_atom_errors = [0.0087, 0.0071, 0.0574];

    let expected_bond_numbers = [2, 2, 2];

    let expected_atom2_indices = [34, 43, 36];
    let expected_atom2_names = ["H2Y", "H2Y", "H2Y"];
    let expected_atom2_order = [0.2040, 0.2317, 0.2020];
    let expected_atom2_errors = [0.0125, 0.0091, 0.0656];

    let expected_convergence_frames = (1..=51).collect::<Vec<usize>>();
    let expected_convergence_values = [
        [0.1494, 0.1460, 0.1455],
        [0.1422, 0.1353, 0.1378],
        [0.1572, 0.1507, 0.1561],
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

        for (j, frame) in [0, 25, 50].into_iter().enumerate() {
            assert_relative_eq!(
                *convergence.total().as_ref().unwrap().get(frame).unwrap(),
                expected_convergence_values.get(i).unwrap().get(j).unwrap(),
                epsilon = 2e-4
            );
        }

        assert!(convergence.upper().is_none());
        assert!(convergence.lower().is_none());

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

        let bond = atom.get_bond(expected_atom2_indices[i]).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.atom_name(), expected_atom_names[i]);
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a1.residue_name(), expected_molecule_names[i]);
        assert_eq!(a2.atom_name(), expected_atom2_names[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);
        assert_eq!(a2.residue_name(), expected_molecule_names[i]);
        assert_eq!(bond.molecule(), expected_molecule_names[i]);

        let order = bond.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_atom2_order[i],
            epsilon = 2e-4
        );
        assert_relative_eq!(
            order.total().unwrap().error().unwrap(),
            expected_atom2_errors[i],
            epsilon = 2e-4
        );
        assert!(order.upper().is_none());
        assert!(order.lower().is_none());

        let maps = bond.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bond directly from molecule
        let bond = molecule
            .get_bond(expected_atom_indices[i], expected_atom2_indices[i])
            .unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);

        // nonexistent atom
        assert!(molecule.get_atom(145).is_none());
        // nonexistent bond
        assert!(molecule.get_bond(7, 19).is_none());
        assert!(molecule.get_bond(145, 189).is_none());
    }
}

#[test]
fn test_aa_order_leaflets_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert_eq!(results.molecules().count(), 3);
    assert!(results.leaflets_data().is_none());

    assert!(results.get_molecule("POPE").is_some());
    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPG").is_some());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.1423,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().value(),
        0.1411,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().lower().unwrap().value(),
        0.1434,
        epsilon = 2e-4
    );

    assert!(results.average_ordermaps().total().is_none());
    assert!(results.average_ordermaps().upper().is_none());
    assert!(results.average_ordermaps().lower().is_none());

    let expected_average_orders = [0.1455, 0.1378, 0.1561];
    let expected_average_upper = [0.1492, 0.1326, 0.1522];
    let expected_average_lower = [0.1419, 0.1431, 0.1606];
    let expected_atom_numbers = [37, 40, 38];
    let expected_molecule_names = ["POPE", "POPC", "POPG"];

    let expected_atom_indices = [32, 41, 34];
    let expected_atom_names = ["C32", "C32", "C32"];
    let expected_atom_order = [0.2226, 0.2363, 0.2247];
    let expected_atom_upper = [0.2131, 0.2334, 0.2484];
    let expected_atom_lower = [0.2319, 0.2391, 0.1976];

    let expected_bond_numbers = [2, 2, 2];

    let expected_atom2_indices = [34, 43, 36];
    let expected_atom2_names = ["H2Y", "H2Y", "H2Y"];
    let expected_atom2_order = [0.2040, 0.2317, 0.2020];
    let expected_atom2_upper = [0.1876, 0.2507, 0.2254];
    let expected_atom2_lower = [0.2203, 0.2126, 0.1752];

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

        assert_relative_eq!(
            order.upper().unwrap().value(),
            expected_atom_upper[i],
            epsilon = 2e-4
        );
        assert!(order.upper().unwrap().error().is_none());

        assert_relative_eq!(
            order.lower().unwrap().value(),
            expected_atom_lower[i],
            epsilon = 2e-4
        );
        assert!(order.lower().unwrap().error().is_none());

        let maps = atom.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bonds
        assert_eq!(atom.bonds().count(), expected_bond_numbers[i]);

        let bond = atom.get_bond(expected_atom2_indices[i]).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.atom_name(), expected_atom_names[i]);
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a1.residue_name(), expected_molecule_names[i]);
        assert_eq!(a2.atom_name(), expected_atom2_names[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);
        assert_eq!(a2.residue_name(), expected_molecule_names[i]);
        assert_eq!(bond.molecule(), expected_molecule_names[i]);

        let order = bond.order();
        assert_relative_eq!(
            order.total().unwrap().value(),
            expected_atom2_order[i],
            epsilon = 2e-4
        );
        assert!(order.total().unwrap().error().is_none());

        assert_relative_eq!(
            order.upper().unwrap().value(),
            expected_atom2_upper[i],
            epsilon = 2e-4
        );
        assert!(order.upper().unwrap().error().is_none());

        assert_relative_eq!(
            order.lower().unwrap().value(),
            expected_atom2_lower[i],
            epsilon = 2e-4
        );
        assert!(order.lower().unwrap().error().is_none());

        let maps = bond.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bond directly from molecule
        let bond = molecule
            .get_bond(expected_atom_indices[i], expected_atom2_indices[i])
            .unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);

        // nonexistent atom
        assert!(molecule.get_atom(145).is_none());
        // nonexistent bond
        assert!(molecule.get_bond(7, 19).is_none());
        assert!(molecule.get_bond(145, 189).is_none());
    }
}

#[test]
fn test_aa_order_error_leaflets_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .estimate_error(EstimateError::default())
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    assert_eq!(results.n_analyzed_frames(), 51);
    assert!(results
        .analysis()
        .estimate_error()
        .as_ref()
        .unwrap()
        .output_convergence()
        .is_none());

    assert_eq!(results.molecules().count(), 3);

    assert!(results.get_molecule("POPE").is_some());
    assert!(results.get_molecule("POPC").is_some());
    assert!(results.get_molecule("POPG").is_some());
    assert!(results.get_molecule("POPA").is_none());

    assert_relative_eq!(
        results.average_order().total().unwrap().value(),
        0.1423,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().total().unwrap().error().unwrap(),
        0.0026,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().value(),
        0.1411,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().upper().unwrap().error().unwrap(),
        0.0024,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        results.average_order().lower().unwrap().value(),
        0.1434,
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

    let expected_atom_numbers = [37, 40, 38];
    let expected_molecule_names = ["POPE", "POPC", "POPG"];

    let expected_atom_indices = [32, 41, 34];
    let expected_atom_names = ["C32", "C32", "C32"];

    let expected_bond_numbers = [2, 2, 2];

    let expected_atom2_indices = [34, 43, 36];
    let expected_atom2_names = ["H2Y", "H2Y", "H2Y"];

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
        assert_eq!(convergence.frames().len(), 51);
        assert!(convergence.total().is_some());
        assert!(convergence.upper().is_some());
        assert!(convergence.lower().is_some());

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

        let bond = atom.get_bond(expected_atom2_indices[i]).unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.atom_name(), expected_atom_names[i]);
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a1.residue_name(), expected_molecule_names[i]);
        assert_eq!(a2.atom_name(), expected_atom2_names[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);
        assert_eq!(a2.residue_name(), expected_molecule_names[i]);
        assert_eq!(bond.molecule(), expected_molecule_names[i]);

        let order = bond.order();
        assert!(order.total().unwrap().error().is_some());
        assert!(order.upper().unwrap().error().is_some());
        assert!(order.lower().unwrap().error().is_some());

        let maps = bond.ordermaps();
        assert!(maps.total().is_none());
        assert!(maps.upper().is_none());
        assert!(maps.lower().is_none());

        // bond directly from molecule
        let bond = molecule
            .get_bond(expected_atom_indices[i], expected_atom2_indices[i])
            .unwrap();
        let (a1, a2) = bond.atoms();
        assert_eq!(a1.relative_index(), expected_atom_indices[i]);
        assert_eq!(a2.relative_index(), expected_atom2_indices[i]);

        // nonexistent atom
        assert!(molecule.get_atom(145).is_none());
        // nonexistent bond
        assert!(molecule.get_bond(7, 19).is_none());
        assert!(molecule.get_bond(145, 189).is_none());
    }
}

#[test]
fn test_aa_order_ordermaps_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "resname POPC and name C22 C24 C218",
            "@membrane and element name hydrogen",
        ))
        .map(
            OrderMap::builder()
                .bin_size([0.1, 4.0])
                .min_samples(5)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::AA(x) => x,
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
    assert_relative_eq!(span_x.1, 9.15673);
    assert_relative_eq!(span_y.0, 0.0);
    assert_relative_eq!(span_y.1, 9.15673);
    assert_relative_eq!(bin.0, 0.1);
    assert_relative_eq!(bin.1, 4.0);

    assert_relative_eq!(
        map.get_at_convert(0.6, 8.0).unwrap(),
        0.1653,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(4.3, 0.0).unwrap(),
        0.1340,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(9.2, 4.0).unwrap(),
        0.1990,
        epsilon = 2e-4
    );

    // ordermaps for a selected atom
    let atom = molecule.get_atom(47).unwrap();
    let map = atom.ordermaps().total().as_ref().unwrap();
    assert!(atom.ordermaps().upper().is_none());
    assert!(atom.ordermaps().lower().is_none());

    assert_relative_eq!(
        map.get_at_convert(0.6, 8.0).unwrap(),
        0.2224,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(4.3, 0.0).unwrap(),
        0.1532,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(9.2, 4.0).unwrap(),
        0.0982,
        epsilon = 2e-4
    );

    // ordermaps for a selected bond
    let bond = atom.get_bond(49).unwrap();
    let map = bond.ordermaps().total().as_ref().unwrap();
    assert!(bond.ordermaps().upper().is_none());
    assert!(bond.ordermaps().lower().is_none());

    assert_relative_eq!(
        map.get_at_convert(0.6, 8.0).unwrap(),
        0.2901,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        map.get_at_convert(4.3, 0.0).unwrap(),
        0.1163,
        epsilon = 2e-4
    );
    assert!(map.get_at_convert(9.2, 4.0).unwrap().is_nan());
}

#[test]
fn test_aa_order_ordermaps_leaflets_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "resname POPC and name C22 C24 C218",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P"))
        .map(
            OrderMap::builder()
                .bin_size([0.1, 4.0])
                .min_samples(5)
                .build()
                .unwrap(),
        )
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::AA(x) => x,
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

    let span_x = total.span_x();
    let span_y = total.span_y();
    let bin = total.tile_dim();

    assert_relative_eq!(span_x.0, 0.0);
    assert_relative_eq!(span_x.1, 9.15673);
    assert_relative_eq!(span_y.0, 0.0);
    assert_relative_eq!(span_y.1, 9.15673);
    assert_relative_eq!(bin.0, 0.1);
    assert_relative_eq!(bin.1, 4.0);

    assert_relative_eq!(
        total.get_at_convert(0.6, 8.0).unwrap(),
        0.1653,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        total.get_at_convert(9.2, 4.0).unwrap(),
        0.1990,
        epsilon = 2e-4
    );

    let upper = molecule.average_ordermaps().upper().as_ref().unwrap();

    assert_relative_eq!(
        upper.get_at_convert(0.6, 8.0).unwrap(),
        0.1347,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        upper.get_at_convert(9.2, 4.0).unwrap(),
        0.3196,
        epsilon = 2e-4
    );

    let lower = molecule.average_ordermaps().lower().as_ref().unwrap();

    assert_relative_eq!(
        lower.get_at_convert(0.6, 8.0).unwrap(),
        0.2104,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        lower.get_at_convert(9.2, 4.0).unwrap(),
        0.1106,
        epsilon = 2e-4
    );

    // ordermaps for a selected atom
    let atom = molecule.get_atom(47).unwrap();
    let total = atom.ordermaps().total().as_ref().unwrap();

    assert_relative_eq!(
        total.get_at_convert(0.6, 8.0).unwrap(),
        0.2224,
        epsilon = 2e-4
    );
    assert_relative_eq!(
        total.get_at_convert(9.2, 4.0).unwrap(),
        0.0982,
        epsilon = 2e-4
    );

    let upper = atom.ordermaps().upper().as_ref().unwrap();

    assert_relative_eq!(
        upper.get_at_convert(0.6, 8.0).unwrap(),
        0.2039,
        epsilon = 2e-4
    );
    assert!(upper.get_at_convert(9.2, 4.0).unwrap().is_nan());

    let lower = atom.ordermaps().lower().as_ref().unwrap();

    assert_relative_eq!(
        lower.get_at_convert(0.6, 8.0).unwrap(),
        0.2540,
        epsilon = 2e-4
    );
    assert!(lower.get_at_convert(9.2, 4.0).unwrap().is_nan());

    // ordermaps for a selected bond
    let bond = atom.get_bond(49).unwrap();
    let total = bond.ordermaps().total().as_ref().unwrap();

    assert_relative_eq!(
        total.get_at_convert(0.6, 8.0).unwrap(),
        0.2901,
        epsilon = 2e-4
    );
    assert!(total.get_at_convert(9.2, 4.0).unwrap().is_nan());

    let upper = bond.ordermaps().upper().as_ref().unwrap();

    assert_relative_eq!(
        upper.get_at_convert(0.6, 8.0).unwrap(),
        0.3584,
        epsilon = 2e-4
    );
    assert!(upper.get_at_convert(9.2, 4.0).unwrap().is_nan());

    let lower = bond.ordermaps().lower().as_ref().unwrap();

    assert_relative_eq!(
        lower.get_at_convert(0.6, 8.0).unwrap(),
        0.1715,
        epsilon = 2e-4
    );
    assert!(lower.get_at_convert(9.2, 4.0).unwrap().is_nan());
}

#[test]
fn test_aa_order_leaflets_every1_collect_rust_api() {
    let analysis = Analysis::builder()
        .structure("tests/files/pcpepg.tpr")
        .trajectory("tests/files/pcpepg.xtc")
        .analysis_type(AnalysisType::aaorder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen",
        ))
        .leaflets(LeafletClassification::global("@membrane", "name P").with_collect(true))
        .silent()
        .overwrite()
        .build()
        .unwrap();

    let results = match analysis.run().unwrap() {
        AnalysisResults::AA(x) => x,
        _ => panic!("Incorrect results type returned."),
    };

    let leaflets_data = results.leaflets_data().as_ref().unwrap();

    assert_eq!(leaflets_data.frames(), &((1..52).collect::<Vec<usize>>()));

    let pope_data = leaflets_data.get_molecule("POPE").unwrap();
    assert_eq!(pope_data.len(), 51);
    for frame in pope_data.iter() {
        assert_eq!(frame.len(), 131);
        for (i, lipid) in frame.iter().enumerate() {
            if i < 65 {
                assert_eq!(lipid, &Leaflet::Upper);
            } else {
                assert_eq!(lipid, &Leaflet::Lower);
            }
        }
    }

    let popc_data = leaflets_data.get_molecule("POPC").unwrap();
    assert_eq!(popc_data.len(), 51);
    for frame in popc_data.iter() {
        assert_eq!(frame.len(), 128);
        for (i, lipid) in frame.iter().enumerate() {
            if i < 64 {
                assert_eq!(lipid, &Leaflet::Upper);
            } else {
                assert_eq!(lipid, &Leaflet::Lower);
            }
        }
    }

    let popg_data = leaflets_data.get_molecule("POPG").unwrap();
    assert_eq!(popg_data.len(), 51);
    for frame in popg_data.iter() {
        assert_eq!(frame.len(), 15);
        for (i, lipid) in frame.iter().enumerate() {
            if i < 8 {
                assert_eq!(lipid, &Leaflet::Upper);
            } else {
                assert_eq!(lipid, &Leaflet::Lower);
            }
        }
    }
}
