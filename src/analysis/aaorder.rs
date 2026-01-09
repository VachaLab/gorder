// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Contains the implementation of the calculation of the atomistic order parameters.

use super::{common::macros::group_name, topology::SystemTopology};
use crate::analysis::common::{
    prepare_geometry_selection, prepare_membrane_normal_calculation, read_trajectory,
};
use crate::analysis::index::read_ndx_file;
use crate::analysis::pbc::{NoPBC, PBC3D};
use crate::analysis::structure;
use crate::analysis::topology::classify::MoleculesClassifier;
use crate::presentation::aaresults::AAOrderResults;
use crate::presentation::{AnalysisResults, OrderResults};
use crate::{input::Analysis, PANIC_MESSAGE};

/// Calculate the atomistic order parameters.
pub(super) fn analyze_atomistic(
    analysis: Analysis,
) -> Result<AnalysisResults, Box<dyn std::error::Error + Send + Sync>> {
    let mut system = structure::read_structure_and_topology(&analysis)?;

    if let Some(ndx) = analysis.index() {
        read_ndx_file(&mut system, ndx)?;
    }

    super::common::create_group(
        &mut system,
        "HeavyAtoms",
        analysis.heavy_atoms().as_ref().unwrap_or_else(||
            panic!("FATAL GORDER ERROR | aaorder::analyze_atomistic | Selection of heavy atoms should be provided. {}", PANIC_MESSAGE)),
    )?;

    colog_info!(
        "Detected {} heavy atoms using a query '{}'.",
        system
            .group_get_n_atoms(group_name!("HeavyAtoms"))
            .expect(PANIC_MESSAGE),
        analysis.heavy_atoms().as_ref().expect(PANIC_MESSAGE)
    );

    super::common::create_group(
        &mut system,
        "Hydrogens",
        analysis.hydrogens().as_ref().unwrap_or_else(||
            panic!("FATAL GORDER ERROR | aaorder::analyze_atomistic | Selection of hydrogens should be provided. {}", PANIC_MESSAGE)),
    )?;

    colog_info!(
        "Detected {} hydrogen atoms using a query '{}'.",
        system
            .group_get_n_atoms(group_name!("Hydrogens"))
            .expect(PANIC_MESSAGE),
        analysis.hydrogens().as_ref().expect(PANIC_MESSAGE)
    );

    super::common::check_groups_overlap(
        &system,
        "HeavyAtoms",
        analysis.heavy_atoms().expect(PANIC_MESSAGE),
        "Hydrogens",
        analysis.hydrogens().expect(PANIC_MESSAGE),
    )?;

    // prepare system for leaflet classification
    if let Some(leaflet) = analysis.leaflets() {
        leaflet.prepare_system(&mut system)?;
    }

    // prepare system for dynamic normal calculation, if needed
    prepare_membrane_normal_calculation(analysis.membrane_normal(), &mut system)?;

    // prepare system for geometry selection
    let geom = prepare_geometry_selection(
        analysis.geometry().as_ref(),
        &mut system,
        analysis.handle_pbc(),
    )?;
    geom.info();

    // get the relevant molecules
    macro_rules! classify_molecules_with_pbc {
        ($pbc:expr) => {
            MoleculesClassifier::classify(&system, &analysis, $pbc)
        };
    }

    let molecules = match analysis.handle_pbc() {
        true => classify_molecules_with_pbc!(&PBC3D::from_system(&system))?,
        false => classify_molecules_with_pbc!(&NoPBC)?,
    };

    // check that there are molecules to analyze
    if molecules.n_molecule_types() == 0 {
        return Ok(AnalysisResults::AA(AAOrderResults::empty(analysis)));
    }

    let mut data = SystemTopology::new(
        &system,
        molecules,
        analysis.estimate_error().clone(),
        analysis.step(),
        analysis.n_threads(),
        geom,
        analysis.leaflets().as_ref(),
        analysis.handle_pbc(),
    );

    data.info()?;

    // finalize the manual leaflet classification
    if let Some(classification) = analysis.leaflets() {
        data.finalize_manual_leaflet_classification(classification)?;
    }

    // finalize the membrane normal specification
    data.finalize_manual_membrane_normals(analysis.membrane_normal())?;

    if let Some(error_estimation) = analysis.estimate_error() {
        error_estimation.info();
    }

    let result = read_trajectory(
        &system,
        data,
        analysis.trajectory(),
        analysis.n_threads(),
        analysis.begin(),
        analysis.end(),
        analysis.step(),
        analysis.silent(),
    )?;

    result.validate_run(analysis.step())?;
    result.log_total_analyzed_frames();

    // print basic info about error estimation
    result.error_info()?;

    Ok(AnalysisResults::AA(
        result.convert::<AAOrderResults>(analysis),
    ))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use groan_rs::system::System;

    use super::*;
    use crate::{
        analysis::{
            common::analyze_frame,
            geometry::GeometrySelectionType,
            topology::{
                bond::{BondType, OrderBonds},
                molecule::{MoleculeType, MoleculeTypes},
            },
        },
        input::{leaflets::LeafletClassification, AnalysisType},
    };

    fn prepare_data_for_tests(
        leaflet_classification: Option<LeafletClassification>,
    ) -> (System, SystemTopology) {
        let mut system = System::from_file("tests/files/pcpepg.tpr").unwrap();

        system
            .group_create(
                group_name!("HeavyAtoms"),
                "@membrane and element name carbon",
            )
            .unwrap();

        system
            .group_create(
                group_name!("Hydrogens"),
                "@membrane and element name hydrogen",
            )
            .unwrap();

        let analysis = if let Some(leaflet) = &leaflet_classification {
            leaflet.prepare_system(&mut system).unwrap();
            Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .leaflets(leaflet.clone())
                .build()
                .unwrap()
        } else {
            Analysis::builder()
                .structure("tests/files/pcpepg.tpr")
                .trajectory("tests/files/pcpepg.xtc")
                .analysis_type(AnalysisType::aaorder(
                    "@membrane and element name carbon",
                    "@membrane and element name hydrogen",
                ))
                .build()
                .unwrap()
        };

        let molecules =
            MoleculesClassifier::classify(&system, &analysis, &PBC3D::from_system(&system))
                .unwrap();

        (
            system.clone(),
            SystemTopology::new(
                &system,
                molecules,
                None,
                1,
                1,
                GeometrySelectionType::default(),
                leaflet_classification.as_ref(),
                true,
            ),
        )
    }

    fn expected_total_orders() -> [Vec<f32>; 3] {
        [
            vec![
                11.389852, 5.878675, -13.541706, -11.096258, 27.989347, 25.653976, 19.82676,
                12.905666, 10.441483, 22.06909, 1.591241, 27.316359, 30.524328, 24.914053,
                31.363665, 28.719316, 30.966375, 28.598515, 30.24056, 22.511251, 31.064238,
                22.65085, 28.63437, 14.978867, 9.811054, 9.882707, 12.23798, 11.099977, 11.654344,
                19.075573, 18.81576, 23.57949, 19.751907, 18.586905, 23.117905, 20.328497,
                23.12013, 19.377028, 21.543617, 5.945527, 20.534143, 4.208125, 6.06384, 2.432906,
                19.9338, 23.64509, 26.96149, 24.438528, 28.552176, 25.008694, 27.270033, 36.337536,
                28.786331, 36.51211, 34.370853, 31.867886, 32.787514, 31.19441, 26.743471,
                29.923025, 25.999725, 29.702312, 28.175154, 25.971518, 24.937567, 23.082615,
                25.887962, 11.512761, 19.913324, 9.057475, 5.056282, 6.974994, 6.667744,
            ],
            vec![
                7.632992, 11.820058, -6.716706, 0.968587, -1.911412, 1.691883, -5.185786, 2.068528,
                -6.920565, -1.894261, 5.227475, 3.660791, -9.327749, 23.51873, 29.831762,
                19.633087, 13.166886, 16.53596, 20.407572, 4.83182, 34.152702, 29.234467, 27.19902,
                29.976748, 27.20112, 29.56742, 26.33664, 32.356857, 26.892092, 25.80453, 25.827114,
                27.20555, 16.994942, 19.094893, 11.55925, 4.492817, 4.772768, 8.061026, 17.412918,
                13.769701, 13.628062, 16.356264, 18.402756, 15.024005, 22.206396, 14.288318,
                17.65203, 13.681941, 20.933535, 9.603966, 16.412039, 0.705084, 2.477132, 29.031408,
                21.52936, 28.327082, 28.09423, 29.071302, 23.746443, 34.364952, 27.89767,
                30.205156, 30.772783, 32.497437, 24.186447, 31.099823, 27.633347, 32.11146,
                18.132109, 30.3674, 20.990023, 27.911135, 20.481102, 15.411962, 15.656943,
                14.946928, 13.218304, 9.822734, 14.016404, 5.663187, 0.980849, 2.227933,
            ],
            vec![
                3.29417, -3.428375, -1.080091, -2.759262, -0.782757, 3.254287, 1.775321, 1.837404,
                -0.511983, 2.644462, 3.72656, -3.031709, 3.253082, 2.484887, -0.706625, 2.211916,
                1.536808, 1.938944, 3.429004, 2.227355, 4.541319, -1.467586, 4.356756, 0.622792,
                0.695665, 2.421673, 0.950128, 2.054902, 1.246198, 0.663714, 2.06197, 0.855577,
                2.98418, 1.386018, 3.973852, 3.595663, 4.880532, 4.837658, 3.861673, 4.990142,
                4.761697, 2.181142, 1.610492, 1.312807, 0.471691, 3.684109, 2.418208, 4.102426,
                3.594907, 3.366313, 3.225462, 1.853718, 3.553464, 2.327006, 2.906338, 3.51264,
                1.606181, 2.882711, 1.589037, 3.930573, 1.550469, 2.114951, 3.555313, 4.157771,
                3.780659, 3.278489, 2.51102, 2.231766, 1.352664, -0.352376, 0.36294, -0.380075,
                -1.870841, 0.96486,
            ],
        ]
    }

    fn expected_upper_orders() -> [Vec<f32>; 3] {
        [
            vec![
                5.291137, 0.789349, -7.450514, -4.314344, 14.737599, 9.509797, 12.143463,
                10.080934, 7.871229, 11.660121, -7.185095, 13.638057, 13.475295, 12.030589,
                14.926608, 14.842074, 13.062988, 13.427266, 9.725433, 11.22098, 12.190741,
                11.216098, 13.531442, 7.979978, 3.702543, 5.203975, 1.599453, 5.209001, 4.91488,
                5.659866, 10.398188, 8.781143, 10.999677, 5.941546, 9.561797, 9.174995, 12.355704,
                11.32586, 12.095253, 4.116473, 15.033503, 5.295498, 3.611082, 6.175383, 10.740083,
                10.916917, 15.405568, 12.284265, 16.144745, 12.567817, 14.028614, 16.80673,
                14.612877, 17.943262, 16.08202, 17.858238, 17.36128, 18.055008, 16.81815,
                13.963133, 13.338045, 11.941948, 10.077958, 13.209061, 8.454679, 9.680064,
                10.611863, 6.245687, 9.020741, 3.877773, 3.018459, 6.515419, 0.038673,
            ],
            vec![
                0.095618, 6.556378, -2.672594, 1.187651, 1.936592, 4.416539, -5.379504, 1.276909,
                -6.00049, -2.63293, 4.988412, 5.378988, -7.967307, 12.298703, 16.69686, 15.49187,
                6.273861, 10.584631, 14.711886, 0.548411, 15.736787, 13.422438, 15.179039,
                17.256449, 15.52683, 14.964476, 14.486416, 16.268179, 13.833229, 13.754546,
                12.476432, 12.782988, 10.746774, 6.481171, 6.889213, -1.440165, 0.35717, -0.591434,
                5.778335, 4.959051, 4.514744, 8.965904, 5.025961, 7.80603, 8.344823, 6.207133,
                6.249355, 7.985627, 8.485929, 5.660625, 5.742283, 2.854231, 0.813643, 15.208017,
                8.857445, 17.16443, 11.207312, 17.699806, 11.005369, 20.546412, 14.422904,
                20.429096, 13.450131, 20.360537, 13.40167, 18.521997, 14.826643, 17.555248,
                6.645715, 17.487207, 6.455177, 14.336545, 7.260629, 5.329075, 7.360892, 8.053697,
                6.724807, 5.174401, 5.207442, -1.368811, -2.373745, 5.366388,
            ],
            vec![
                1.666534, -2.013375, 0.011224, -1.861104, 0.271434, 1.622315, 1.144751, 1.300498,
                -1.627284, 0.104286, 1.582823, -0.236736, 2.074151, 1.163585, -1.903353, -0.120156,
                1.125026, 0.421596, 2.512546, 2.56527, 2.948156, -0.718106, 2.639498, 0.237125,
                1.106276, 1.246939, 0.192422, 1.76711, 1.93911, -0.373426, 0.597813, -0.173237,
                0.741756, 0.940684, 2.107792, 2.10336, 2.976827, 2.745061, 1.8227, 2.995421,
                2.407649, 0.381791, 0.43358, 0.721784, -0.32691, 1.645782, 2.339347, 1.253812,
                2.845095, 0.792993, 2.821195, 1.246402, 2.055948, 1.303927, 1.832217, 1.894172,
                1.946451, 1.026144, 1.983396, 1.81565, 2.070712, 1.114288, 2.034042, 2.08145,
                1.640327, 2.534143, 0.738018, 1.356258, -0.066966, -0.555389, -0.32461, -0.891659,
                -0.824119, -0.168321,
            ],
        ]
    }

    fn expected_lower_orders() -> [Vec<f32>; 3] {
        [
            vec![
                6.098715, 5.089326, -6.091192, -6.781914, 13.251749, 16.144178, 7.683297, 2.824732,
                2.570254, 10.408968, 8.776336, 13.678302, 17.049032, 12.883464, 16.437057,
                13.877243, 17.903387, 15.171249, 20.515127, 11.290271, 18.873497, 11.434752,
                15.102929, 6.998889, 6.108511, 4.678732, 10.638527, 5.890976, 6.739464, 13.415707,
                8.417572, 14.798347, 8.752231, 12.645358, 13.556108, 11.153501, 10.764426,
                8.051167, 9.448365, 1.829054, 5.50064, -1.087373, 2.452758, -3.742477, 9.193716,
                12.728174, 11.555922, 12.154264, 12.407431, 12.440876, 13.241419, 19.530804,
                14.173455, 18.56885, 18.288834, 14.009647, 15.426231, 13.139402, 9.925323,
                15.959893, 12.661681, 17.760365, 18.097195, 12.762457, 16.482887, 13.40255,
                15.2761, 5.267074, 10.892584, 5.179702, 2.037823, 0.459575, 6.629071,
            ],
            vec![
                7.537374, 5.26368, -4.044112, -0.219064, -3.848004, -2.724656, 0.193718, 0.791619,
                -0.920075, 0.738669, 0.239063, -1.718197, -1.360442, 11.220028, 13.134903,
                4.141218, 6.893025, 5.951329, 5.695686, 4.283409, 18.415915, 15.812028, 12.019982,
                12.720299, 11.67429, 14.602943, 11.850224, 16.08868, 13.058862, 12.049984,
                13.350682, 14.422562, 6.248167, 12.613722, 4.670037, 5.932982, 4.415598, 8.65246,
                11.634583, 8.81065, 9.113318, 7.390361, 13.376794, 7.217975, 13.861573, 8.081185,
                11.402676, 5.696314, 12.447606, 3.943341, 10.669756, -2.149147, 1.663489,
                13.823391, 12.671914, 11.162651, 16.886917, 11.371497, 12.741074, 13.818541,
                13.474767, 9.776061, 17.322653, 12.1369, 10.784778, 12.577826, 12.806703,
                14.556215, 11.486393, 12.880192, 14.534846, 13.574589, 13.220473, 10.082887,
                8.296051, 6.893231, 6.493497, 4.648333, 8.808962, 7.031998, 3.354594, -3.138455,
            ],
            vec![
                1.627636, -1.415, -1.091315, -0.898158, -1.054191, 1.631972, 0.63057, 0.536906,
                1.115301, 2.540176, 2.143737, -2.794973, 1.178931, 1.321302, 1.196728, 2.332072,
                0.411782, 1.517348, 0.916458, -0.337915, 1.593163, -0.74948, 1.717258, 0.385667,
                -0.410611, 1.174734, 0.757706, 0.287792, -0.692912, 1.03714, 1.464157, 1.028814,
                2.242424, 0.445334, 1.86606, 1.492303, 1.903705, 2.092597, 2.038973, 1.994721,
                2.354048, 1.799351, 1.176912, 0.591023, 0.798601, 2.038327, 0.078861, 2.848614,
                0.749812, 2.57332, 0.404267, 0.607316, 1.497516, 1.023079, 1.074121, 1.618468,
                -0.34027, 1.856567, -0.394359, 2.114923, -0.520243, 1.000663, 1.521271, 2.076321,
                2.140332, 0.744346, 1.773002, 0.875508, 1.41963, 0.203013, 0.68755, 0.511584,
                -1.046722, 1.133181,
            ],
        ]
    }

    fn collect_bond_data<T, F>(molecule: &MoleculeType<OrderBonds>, func: F) -> Vec<T>
    where
        F: Fn(&BondType) -> T,
    {
        molecule
            .order_structure()
            .bond_types()
            .iter()
            .map(func)
            .collect()
    }

    #[test]
    fn test_aaorder_analyze_frame_basic() {
        let (system, mut data) = prepare_data_for_tests(None);

        analyze_frame(&system, &mut data).unwrap();
        let expected_total_orders = expected_total_orders();

        let molecule_types = match data.molecule_types() {
            MoleculeTypes::AtomBased(_) => panic!("Molecule types should be bond-based."),
            MoleculeTypes::BondBased(x) => x,
        };

        for (m, molecule) in molecule_types.iter().enumerate() {
            let n_instances = molecule.order_structure().bond_types()[0].bonds().len();
            let orders = molecule
                .order_structure()
                .bond_types()
                .iter()
                .map(|b| b.total().order().into())
                .collect::<Vec<f32>>();

            let samples = molecule
                .order_structure()
                .bond_types()
                .iter()
                .map(|b| b.total().n_samples())
                .collect::<Vec<usize>>();

            assert_eq!(orders.len(), expected_total_orders[m].len());
            for (real, expected) in orders.iter().zip(expected_total_orders[m].iter()) {
                assert_relative_eq!(-real, expected, epsilon = 1e-5);
            }

            for sample in samples {
                assert_eq!(sample, n_instances);
            }
        }
    }

    #[test]
    fn test_aaorder_analyze_frame_leaflets() {
        let classifier = LeafletClassification::global("@membrane", "name P");

        let (system, mut data) = prepare_data_for_tests(Some(classifier));

        analyze_frame(&system, &mut data).unwrap();
        let expected_total_orders = expected_total_orders();
        let expected_upper_orders = expected_upper_orders();
        let expected_lower_orders = expected_lower_orders();
        let expected_total_samples = [131, 128, 15];
        let expected_upper_samples = [65, 64, 8];
        let expected_lower_samples = [66, 64, 7];

        let molecule_types = match data.molecule_types() {
            MoleculeTypes::AtomBased(_) => panic!("Molecule types should be bond-based."),
            MoleculeTypes::BondBased(x) => x,
        };

        for (m, molecule) in molecule_types.iter().enumerate() {
            let total_orders: Vec<f32> = collect_bond_data(molecule, |b| b.total().order().into());
            let upper_orders =
                collect_bond_data(molecule, |b| b.upper().as_ref().unwrap().order().into());
            let lower_orders =
                collect_bond_data(molecule, |b| b.lower().as_ref().unwrap().order().into());
            let total_samples = collect_bond_data(molecule, |b| b.total().n_samples());
            let upper_samples =
                collect_bond_data(molecule, |b| b.upper().as_ref().unwrap().n_samples());
            let lower_samples =
                collect_bond_data(molecule, |b| b.lower().as_ref().unwrap().n_samples());

            for (order, samples, expected_order, expected_samples) in [
                (
                    &total_orders,
                    &total_samples,
                    &expected_total_orders[m],
                    expected_total_samples[m],
                ),
                (
                    &upper_orders,
                    &upper_samples,
                    &expected_upper_orders[m],
                    expected_upper_samples[m],
                ),
                (
                    &lower_orders,
                    &lower_samples,
                    &expected_lower_orders[m],
                    expected_lower_samples[m],
                ),
            ] {
                assert_eq!(order.len(), expected_order.len());
                for (real, expected) in order.iter().zip(expected_order.iter()) {
                    assert_relative_eq!(-real, expected, epsilon = 1e-5);
                }

                for &sample in samples {
                    assert_eq!(sample, expected_samples);
                }
            }
        }
    }
}
