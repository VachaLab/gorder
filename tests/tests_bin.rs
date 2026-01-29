// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Tests of the binary application.

mod common;

use assert_cmd::Command;
use common::{assert_eq_csv, assert_eq_maps, assert_eq_order};

use crate::common::diff_files_ignore_first;

#[test]
fn test_bin_aa_order_basic_yaml() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/basic_aa.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order("temp_aa_order.yaml", "tests/files/aa_order_basic.yaml", 1);

    std::fs::remove_file("temp_aa_order.yaml").unwrap();
}

#[test]
fn test_bin_ua_order_basic_yaml() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/basic_ua.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order("temp_ua_order.yaml", "tests/files/ua_order_basic.yaml", 1);

    std::fs::remove_file("temp_ua_order.yaml").unwrap();
}

#[test]
fn test_bin_ua_order_from_aa_yaml() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/ua_from_aa.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_ua_order_from_aa.yaml",
        "tests/files/ua_order_from_aa.yaml",
        1,
    );

    std::fs::remove_file("temp_ua_order_from_aa.yaml").unwrap();
}

#[test]
fn test_bin_cg_order_leaflets_yaml_tab() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args(["tests/files/inputs/leaflets_cg.yaml", "--overwrite"])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_order.yaml",
        "tests/files/cg_order_leaflets.yaml",
        1,
    );

    assert_eq_order("temp_cg_order.tab", "tests/files/cg_order_leaflets.tab", 1);

    std::fs::remove_file("temp_cg_order.yaml").unwrap();
    std::fs::remove_file("temp_cg_order.tab").unwrap();
}

#[test]
fn test_bin_cg_order_maps() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args(["tests/files/inputs/maps_cg.yaml", "--overwrite"])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_order_maps.yaml",
        "tests/files/cg_order_small.yaml",
        1,
    );

    std::fs::remove_file("temp_cg_order_maps.yaml").unwrap();

    let expected_file_names = [
        "ordermap_POPC-C1B-8--POPC-C2B-9_full.dat",
        "ordermap_POPC-C2B-9--POPC-C3B-10_full.dat",
        "ordermap_POPC-C3B-10--POPC-C4B-11_full.dat",
        "ordermap_average_full.dat",
    ];

    for file in expected_file_names {
        let real_file = format!("temp_cg_ordermaps/POPC/{}", file);
        let test_file = format!("tests/files/ordermaps_cg/{}", file);
        assert_eq_maps(&real_file, &test_file, 2);
    }

    std::fs::remove_dir_all("temp_cg_ordermaps").unwrap();
}

#[test]
fn test_bin_estimate_error() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args(["tests/files/inputs/estimate_error_cg.yaml", "--overwrite"])
        .assert()
        .success()
        .stdout("");

    assert_eq_order("temp_cg_ee.yaml", "tests/files/cg_order_error.yaml", 1);
    assert_eq_order("temp_cg_ee.tab", "tests/files/cg_order_error.tab", 1);
    assert_eq_csv("temp_cg_ee.csv", "tests/files/cg_order_error.csv", 0);

    std::fs::remove_file("temp_cg_ee.yaml").unwrap();
    std::fs::remove_file("temp_cg_ee.tab").unwrap();
    std::fs::remove_file("temp_cg_ee.csv").unwrap();
}

#[test]
fn test_bin_concatenate_estimate_error() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/concatenate_estimate_error_cg.yaml",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order("temp_cg_cat_ee.yaml", "tests/files/cg_order_error.yaml", 1);
    assert_eq_order("temp_cg_cat_ee.tab", "tests/files/cg_order_error.tab", 1);
    assert_eq_csv("temp_cg_cat_ee.csv", "tests/files/cg_order_error.csv", 0);

    std::fs::remove_file("temp_cg_cat_ee.yaml").unwrap();
    std::fs::remove_file("temp_cg_cat_ee.tab").unwrap();
    std::fs::remove_file("temp_cg_cat_ee.csv").unwrap();
}

#[test]
fn test_bin_cg_order_maps_export_config() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/maps_cg_for_export_config.yaml",
            "--overwrite",
            "--export-config",
            "temp_analysis_out.yaml",
        ])
        .assert()
        .success()
        .stdout("");

    // remove everything and rerun the analysis using output config to check that it works
    std::fs::remove_file("temp_cg_order_maps_for_export_config.yaml").unwrap();
    std::fs::remove_dir_all("temp_cg_ordermaps_for_export_config").unwrap();

    Command::cargo_bin("gorder")
        .unwrap()
        .args(["temp_analysis_out.yaml", "--overwrite"])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_order_maps_for_export_config.yaml",
        "tests/files/cg_order_small.yaml",
        1,
    );

    std::fs::remove_file("temp_cg_order_maps_for_export_config.yaml").unwrap();

    let expected_file_names = [
        "ordermap_POPC-C1B-8--POPC-C2B-9_full.dat",
        "ordermap_POPC-C2B-9--POPC-C3B-10_full.dat",
        "ordermap_POPC-C3B-10--POPC-C4B-11_full.dat",
        "ordermap_average_full.dat",
    ];

    for file in expected_file_names {
        let real_file = format!("temp_cg_ordermaps_for_export_config/POPC/{}", file);
        let test_file = format!("tests/files/ordermaps_cg/{}", file);
        assert_eq_maps(&real_file, &test_file, 2);
    }

    std::fs::remove_dir_all("temp_cg_ordermaps_for_export_config").unwrap();
    std::fs::remove_file("temp_analysis_out.yaml").unwrap();
}

#[test]
fn test_bin_cg_geometry_selection() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/cylinder.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_order_cylinder.yaml",
        "tests/files/cg_order_cylinder.yaml",
        1,
    );

    std::fs::remove_file("temp_cg_order_cylinder.yaml").unwrap();
}

#[test]
fn test_bin_aa_inverted_cuboid_selection() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/inverted_cuboid.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_aa_order_inverted_cuboid.yaml",
        "tests/files/aa_order_cuboid_square_inverted.yaml",
        1,
    );

    std::fs::remove_file("temp_aa_order_inverted_cuboid.yaml").unwrap();
}

#[test]
fn test_bin_aa_leaflets_every_export() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/leaflets_every_export.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_aa_order_leaflets_with_export.yaml",
        "tests/files/aa_order_leaflets.yaml",
        1,
    );

    assert!(diff_files_ignore_first(
        "temp_leaflets_exported_every.yaml",
        "tests/files/aa_leaflets_every1.yaml",
        1,
    ));

    std::fs::remove_file("temp_aa_order_leaflets_with_export.yaml").unwrap();
    std::fs::remove_file("temp_leaflets_exported_every.yaml").unwrap();
}

#[test]
fn test_bin_aa_leaflets_from_file() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/leaflets_from_file.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_aa_order_leaflets_from_file.yaml",
        "tests/files/aa_order_leaflets.yaml",
        1,
    );

    std::fs::remove_file("temp_aa_order_leaflets_from_file.yaml").unwrap();
}

#[test]
fn test_bin_cg_leaflets_from_map() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/leaflets_from_map.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_order_leaflets_from_map.yaml",
        "tests/files/cg_order_leaflets.yaml",
        1,
    );

    std::fs::remove_file("temp_cg_order_leaflets_from_map.yaml").unwrap();
}

#[test]
fn test_bin_aa_leaflets_no_pbc() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/aa_leaflets_no_pbc.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_aa_order_leaflets_no_pbc.yaml",
        "tests/files/aa_order_leaflets_nopbc.yaml",
        1,
    );

    std::fs::remove_file("temp_aa_order_leaflets_no_pbc.yaml").unwrap();
}

#[test]
fn test_bin_cg_vesicle_dynamic() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/vesicle_dynamic_membrane_normal.yaml",
            // silent and overwrite part of the config file
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_order_vesicle_dynamic_membrane_normal.yaml",
        "tests/files/cg_order_vesicle.yaml",
        1,
    );

    std::fs::remove_file("temp_cg_order_vesicle_dynamic_membrane_normal.yaml").unwrap();
}

#[test]
fn test_bin_ua_order_dynamic_yaml() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/ua_dynamic.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_ua_order_dynamic.yaml",
        "tests/files/ua_order_dynamic_normals.yaml",
        1,
    );

    std::fs::remove_file("temp_ua_order_dynamic.yaml").unwrap();
}

// testing single NDX file
#[test]
fn test_bin_aa_leaflets_once_ndx() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/leaflets_aa_ndx.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_aa_order_leaflets_ndx.yaml",
        "tests/files/aa_order_leaflets.yaml",
        1,
    );

    std::fs::remove_file("temp_aa_order_leaflets_ndx.yaml").unwrap();
}

// testing glob expansion
#[test]
fn test_bin_cg_leaflets_every_ndx() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/leaflets_cg_scrambling_ndx.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_order_scrambling_leaflets_ndx.yaml",
        "tests/files/scrambling/order_global.yaml",
        1,
    );

    std::fs::remove_file("temp_cg_order_scrambling_leaflets_ndx.yaml").unwrap();
}

// testing explicitly provided NDX files
#[test]
fn test_bin_cg_leaflets_every20_ndx() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/leaflets_cg_every20_ndx.yaml",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_leaflets_every20_ndx.yaml",
        "tests/files/cg_order_leaflets.yaml",
        1,
    );

    std::fs::remove_file("temp_cg_leaflets_every20_ndx.yaml").unwrap();
}

#[test]
fn test_bin_cg_inline_manual_normals() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/inline_manual_normals.yaml",
            "--overwrite",
            "--silent",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_cg_inline_manual_normals.yaml",
        "tests/files/cg_order_vesicle.yaml",
        1,
    );

    std::fs::remove_file("temp_cg_inline_manual_normals.yaml").unwrap();
}

#[test]
fn test_bin_aa_clustering() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/aa_clustering.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .success()
        .stdout("");

    assert_eq_order(
        "temp_aa_buckled_clustering.yaml",
        "tests/files/aa_order_buckled_leaflets.yaml",
        1,
    );

    std::fs::remove_file("temp_aa_buckled_clustering.yaml").unwrap();
}

#[test]
fn test_bin_cg_leaflets_fail_nonexistent_traj() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/cg_fail_nonexistent_traj.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .failure()
        .stdout("")
        .stderr(
            "[E] error: file 'cg.xtc' was not found or could not be read as a trajectory file\n",
        );
}

#[test]
fn test_bin_aa_leaflets_fail_no_ndx() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/leaflets_ndx_fail.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .failure()
        .stdout("")
        .stderr("[E] error: file \'tests/files/nonexistent*.ndx\' was not found\n");
}

#[test]
fn test_bin_aa_order_writing_fail() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/writing_fail.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .failure()
        .stdout("")
        .stderr(
            "[E] error: could not create file \'this_directory_does_not_exist/temp_aa_order.yaml\'\n",
        );
}

#[test]
fn test_bin_aa_order_fail() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/atom_overlap_error.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .failure()
        .stdout("")
        .stderr("[E] error: 217 atoms are part of both \'HeavyAtoms\' (query: \'@membrane and element name carbon or serial 876 to 1234\') and \'Hydrogens\' (query: \'@membrane and element name hydrogen\')\n");
}

#[test]
fn test_bin_missing_output_fail() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/basic.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .failure()
        .stdout("")
        .stderr("[E] error: no yaml output file specified in the configuration file \'tests/files/inputs/basic.yaml\' (hint: add \'output: output.yaml\' to your configuration file)\n");
}

#[test]
fn test_bin_missing_maps_output_fail() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/default_ordermap.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .failure()
        .stdout("")
        .stderr("[E] error: no output directory for ordermaps specified in the configuration file \'tests/files/inputs/default_ordermap.yaml\'\n");
}

#[test]
fn test_bin_output_config_writing_fails() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/basic_aa_config_fails.yaml",
            "--silent",
            "--overwrite",
            "--export-config",
            "this_directory_does_not_exist/analysis_out.yaml",
        ])
        .assert()
        .success()
        .stdout("")
        .stderr(
            "[E] Analysis completed successfully, but exporting the analysis options failed!
 |      error: could not create file \'this_directory_does_not_exist/analysis_out.yaml\'\n",
        );

    assert_eq_order(
        "temp_aa_order_config_fails.yaml",
        "tests/files/aa_order_basic.yaml",
        1,
    );

    std::fs::remove_file("temp_aa_order_config_fails.yaml").unwrap();
}

#[test]
fn test_bin_ua_no_carbons_fail() {
    Command::cargo_bin("gorder")
        .unwrap()
        .args([
            "tests/files/inputs/ua_no_carbons.yaml",
            "--silent",
            "--overwrite",
        ])
        .assert()
        .failure()
        .stdout("")
        .stderr("[E] error: no carbons for the calculation of united-atom order parameters were specified\n");
}
