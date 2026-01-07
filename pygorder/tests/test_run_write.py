"""
Released under MIT License.
Copyright (c) 2024-2026 Ladislav Bartos
"""

import gorder, tempfile, shutil, os, yaml, pytest
import numpy as np


def read_file_without_first_lines(file_path: str, skip: int) -> list[str]:
    """Reads a file and returns its contents as a list of lines, skipping the first `skip` lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()[skip:]


def diff_files_ignore_first(file1: str, file2: str, skip: int) -> bool:
    """Compares two files, ignoring the first `skip` lines."""
    content1 = read_file_without_first_lines(file1, skip)
    content2 = read_file_without_first_lines(file2, skip)
    return content1 == content2


def read_leaflets_yaml(filename):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    result = {}
    for key, value in data.items():
        converted_values = [
            [1 if v == "Upper" else 0 if v == "Lower" else v for v in sublist]
            for sublist in value
        ]
        arr = np.array(converted_values, dtype=np.uint8)
        result[key] = arr

    return result


def read_normals_yaml(filename):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    result = {}
    for key, value in data.items():
        arr = np.array(value, dtype=np.float32)
        result[key] = arr
    return result


def test_cg_order_basic_yaml():
    for n_threads in [1, 2, 3, 4, 8, 32]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        analysis = gorder.Analysis(
            structure="../tests/files/cg.tpr",
            trajectory="../tests/files/cg.xtc",
            analysis_type=gorder.analysis_types.CGOrder("@membrane"),
            output_yaml=temp_file_path,
            silent=True,
            overwrite=True,
            n_threads=n_threads,
        )

        analysis.run().write()

        try:
            assert diff_files_ignore_first(
                temp_file_path, "../tests/files/cg_order_basic.yaml", 1
            ), "Files do not match!"
        finally:
            shutil.rmtree(temp_file_path, ignore_errors=True)


def test_aa_order_basic_yaml():
    for n_threads in [1, 2, 3, 4, 8, 32]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        analysis = gorder.Analysis(
            structure="../tests/files/pcpepg.tpr",
            trajectory="../tests/files/pcpepg.xtc",
            analysis_type=gorder.analysis_types.AAOrder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ),
            output_yaml=temp_file_path,
            silent=True,
            overwrite=True,
            n_threads=n_threads,
        )

        analysis.run().write()

        try:
            assert diff_files_ignore_first(
                temp_file_path, "../tests/files/aa_order_basic.yaml", 1
            ), "Files do not match!"
        finally:
            shutil.rmtree(temp_file_path, ignore_errors=True)


def test_aa_order_basic_from_file_yaml():
    analysis = gorder.Analysis.from_file(
        "../tests/files/inputs/basic_aa_for_python.yaml"
    )
    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            "temp_aa_order_py.yaml", "../tests/files/aa_order_basic.yaml", 1
        ), "Files do not match!"
    finally:
        os.remove("temp_aa_order_py.yaml")


def test_from_file_fail():
    with pytest.raises(gorder.exceptions.ConfigError) as excinfo:
        gorder.Analysis.from_file("../tests/files/inputs/cylinder_negative_radius.yaml")
    assert "error the specified radius for the geometry selection is" in str(
        excinfo.value
    )


def test_ua_order_basic_yaml():
    for n_threads in [1, 2, 3, 4, 8, 32]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        analysis = gorder.Analysis(
            structure="../tests/files/ua.tpr",
            trajectory="../tests/files/ua.xtc",
            analysis_type=gorder.analysis_types.UAOrder(
                saturated="(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)",
                unsaturated="(resname POPC and name C24 C25) or (resname POPS and name C27 C28)",
            ),
            output_yaml=temp_file_path,
            silent=True,
            overwrite=True,
            n_threads=n_threads,
        )

        analysis.run().write()

        try:
            assert diff_files_ignore_first(
                temp_file_path, "../tests/files/ua_order_basic.yaml", 1
            ), "Files do not match!"
        finally:
            shutil.rmtree(temp_file_path, ignore_errors=True)


def test_aa_order_basic_concatenated_yaml():
    for trajectory in [
        "../tests/files/split/pcpepg?.xtc",
        [
            "../tests/files/split/pcpepg1.xtc",
            "../tests/files/split/pcpepg2.xtc",
            "../tests/files/split/pcpepg3.xtc",
            "../tests/files/split/pcpepg4.xtc",
            "../tests/files/split/pcpepg5.xtc",
        ],
    ]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        analysis = gorder.Analysis(
            structure="../tests/files/pcpepg.tpr",
            trajectory=trajectory,
            analysis_type=gorder.analysis_types.AAOrder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ),
            output_yaml=temp_file_path,
            silent=True,
            overwrite=True,
        )

        analysis.run().write()

        try:
            assert diff_files_ignore_first(
                temp_file_path, "../tests/files/aa_order_basic.yaml", 1
            ), "Files do not match!"
        finally:
            shutil.rmtree(temp_file_path, ignore_errors=True)


def test_all_outputs():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file_yaml:
        yaml_path = temp_file_yaml.name
    with tempfile.NamedTemporaryFile(delete=False) as temp_file_tab:
        tab_path = temp_file_tab.name
    with tempfile.NamedTemporaryFile(delete=False) as temp_file_csv:
        csv_path = temp_file_csv.name

    temp_dir = tempfile.mkdtemp()
    dir_path = os.path.abspath(temp_dir)

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "@membrane and element name carbon", "@membrane and element name hydrogen"
        ),
        output_yaml=yaml_path,
        output_tab=tab_path,
        output_csv=csv_path,
        output_xvg=f"{dir_path}/order.xvg",
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            yaml_path, "../tests/files/aa_order_basic.yaml", 1
        ), "Files do not match!"
        assert diff_files_ignore_first(
            tab_path, "../tests/files/aa_order_basic.tab", 1
        ), "Files do not match!"
        assert diff_files_ignore_first(
            csv_path, "../tests/files/aa_order_basic.csv", 0
        ), "Files do not match!"
        assert diff_files_ignore_first(
            f"{dir_path}/order_POPC.xvg", "../tests/files/aa_order_basic_POPC.xvg", 1
        ), "Files do not match!"
        assert diff_files_ignore_first(
            f"{dir_path}/order_POPE.xvg", "../tests/files/aa_order_basic_POPE.xvg", 1
        ), "Files do not match!"
        assert diff_files_ignore_first(
            f"{dir_path}/order_POPG.xvg", "../tests/files/aa_order_basic_POPG.xvg", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(yaml_path, ignore_errors=True)
        shutil.rmtree(tab_path, ignore_errors=True)
        shutil.rmtree(csv_path, ignore_errors=True)
        shutil.rmtree(dir_path, ignore_errors=True)


def test_leaflets():
    manual_dict = read_leaflets_yaml(
        "../tests/files/inputs/leaflets_files/cg_every.yaml"
    )

    for leaflets in [
        gorder.leaflets.GlobalClassification("@membrane", "name PO4"),
        gorder.leaflets.LocalClassification("@membrane", "name PO4", radius=2.5),
        gorder.leaflets.IndividualClassification("name PO4", "name C4A C4B"),
        gorder.leaflets.ClusteringClassification("name PO4"),
        gorder.leaflets.ManualClassification(
            "../tests/files/inputs/leaflets_files/cg_every.yaml"
        ),
        gorder.leaflets.ManualClassification(manual_dict),
        gorder.leaflets.NdxClassification(
            ndx=["../tests/files/ndx/cg_leaflets.ndx"] * 101,
            heads="name PO4",
            upper_leaflet="Upper",
            lower_leaflet="Lower",
        ),
    ]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        analysis = gorder.Analysis(
            structure="../tests/files/cg.tpr",
            trajectory="../tests/files/cg.xtc",
            analysis_type=gorder.analysis_types.CGOrder("@membrane"),
            leaflets=leaflets,
            output_yaml=temp_file_path,
            silent=True,
            overwrite=True,
        )

        analysis.run().write()

        try:
            assert diff_files_ignore_first(
                temp_file_path, "../tests/files/cg_order_leaflets.yaml", 1
            ), "Files do not match!"
        finally:
            shutil.rmtree(temp_file_path, ignore_errors=True)


def test_leaflets_clustering_once():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("@membrane"),
        leaflets=gorder.leaflets.ClusteringClassification(
            "name PO4", frequency=gorder.Frequency.once()
        ),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_leaflets.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_leaflets_spherical_clustering_vesicle():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/vesicle.tpr",
        trajectory="../tests/files/vesicle.xtc",
        analysis_type=gorder.analysis_types.CGOrder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B"
        ),
        membrane_normal=gorder.membrane_normal.DynamicNormal("name PO4", radius=2.0),
        leaflets=gorder.leaflets.SphericalClusteringClassification("name PO4"),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_vesicle_leaflets.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_ua_leaflets():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            saturated="(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)",
            unsaturated="(resname POPC and name C24 C25) or (resname POPS and name C27 C28)",
        ),
        leaflets=gorder.leaflets.GlobalClassification("@membrane", "name r'^P'"),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/ua_order_leaflets.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_scrambling_leaflets():
    files = [
        [
            "order_once.yaml",
            "order_global.yaml",
            "order_global_every_10.yaml",
        ],  # global
        ["order_once.yaml", "order_local.yaml", "order_local_every_10.yaml"],  # local
        [
            "order_once.yaml",
            "order_individual.yaml",
            "order_individual_every_10.yaml",
        ],  # individual
        [
            "order_once.yaml",
            "order_manual.yaml",
            "order_global_every_10.yaml",
        ],  # from file
        [
            "order_once.yaml",
            "order_manual.yaml",
            "order_global_every_10.yaml",
        ],  # from dict
        [
            "order_once.yaml",
            "order_manual_ndx.yaml",
            "order_global_every_10.yaml",
        ],  # from NDX
    ]
    manual_classification = ["once.yaml", "every.yaml", "every10.yaml"]
    dicts = [
        read_leaflets_yaml(f"../tests/files/scrambling/leaflets_{x}")
        for x in manual_classification
    ]
    ndxs = [
        ["../tests/files/scrambling/ndx/leaflets_frame_000.ndx"],
        [
            f"../tests/files/scrambling/ndx/leaflets_frame_{x:03d}.ndx"
            for x in range(0, 101)
        ],
        [
            f"../tests/files/scrambling/ndx/leaflets_frame_{x:03d}.ndx"
            for x in range(0, 101, 10)
        ],
    ]

    for j, freq in enumerate(
        [gorder.Frequency.once(), gorder.Frequency.every(1), gorder.Frequency.every(10)]
    ):
        for i, leaflets in enumerate(
            [
                gorder.leaflets.GlobalClassification(
                    "@membrane", "name PO4", frequency=freq
                ),
                gorder.leaflets.LocalClassification(
                    "@membrane", "name PO4", radius=3.0, frequency=freq
                ),
                gorder.leaflets.IndividualClassification(
                    "name PO4", "name C4A C4B", frequency=freq
                ),
                gorder.leaflets.ManualClassification(
                    f"../tests/files/scrambling/leaflets_{manual_classification[j]}",
                    frequency=freq,
                ),
                gorder.leaflets.ManualClassification(dicts[j], frequency=freq),
                gorder.leaflets.NdxClassification(
                    ndxs[j], "name PO4", "Upper", "Lower", frequency=freq
                ),
            ]
        ):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name

            analysis = gorder.Analysis(
                structure="../tests/files/scrambling/cg_scrambling.tpr",
                trajectory="../tests/files/scrambling/cg_scrambling.xtc",
                analysis_type=gorder.analysis_types.CGOrder("@membrane"),
                leaflets=leaflets,
                output_yaml=temp_file_path,
                silent=True,
                overwrite=True,
            )

            analysis.run().write()

            try:
                assert diff_files_ignore_first(
                    temp_file_path, f"../tests/files/scrambling/{files[i][j]}", 1
                ), "Files do not match!"
            finally:
                shutil.rmtree(temp_file_path, ignore_errors=True)


def test_scrambling_leaflets_export():
    output_files = [
        "order_once.yaml",
        "order_global.yaml",
        "order_global_every_10.yaml",
    ]
    leaflets_files = [
        "leaflets_once.yaml",
        "leaflets_every.yaml",
        "leaflets_every10.yaml",
    ]

    for i, freq in enumerate(
        [gorder.Frequency.once(), gorder.Frequency.every(1), gorder.Frequency.every(10)]
    ):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        with tempfile.NamedTemporaryFile(delete=False) as temp_file_leaflets:
            temp_file_leaflets_path = temp_file_leaflets.name

        analysis = gorder.Analysis(
            structure="../tests/files/scrambling/cg_scrambling.tpr",
            trajectory="../tests/files/scrambling/cg_scrambling.xtc",
            analysis_type=gorder.analysis_types.CGOrder("@membrane"),
            leaflets=gorder.leaflets.GlobalClassification(
                "@membrane", "name PO4", frequency=freq, collect=temp_file_leaflets_path
            ),
            output_yaml=temp_file_path,
            silent=True,
            overwrite=True,
        )

        analysis.run().write()

        try:
            assert diff_files_ignore_first(
                temp_file_path, f"../tests/files/scrambling/{output_files[i]}", 1
            ), "Files do not match!"
            assert diff_files_ignore_first(
                temp_file_leaflets_path,
                f"../tests/files/scrambling/{leaflets_files[i]}",
                1,
            ), "Leaflet files do not match!"
        finally:
            shutil.rmtree(temp_file_path, ignore_errors=True)
            shutil.rmtree(temp_file_leaflets_path, ignore_errors=True)


def test_ndx():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        index="../tests/files/cg.ndx",
        analysis_type=gorder.analysis_types.CGOrder("Membrane"),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_basic.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_gro_bonds():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.gro",
        bonds="../tests/files/pcpepg.bnd",
        trajectory="../tests/files/pcpepg.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "@membrane and element name carbon", "@membrane and element name hydrogen"
        ),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/aa_order_basic.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_x_normal():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg_switched_xz.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "@membrane and element name carbon", "@membrane and element name hydrogen"
        ),
        output_yaml=temp_file_path,
        leaflets=gorder.leaflets.GlobalClassification(
            "@membrane", "name P", frequency=gorder.Frequency.once()
        ),
        membrane_normal="x",
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/aa_order_leaflets.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_dynamic_normals():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/vesicle.tpr",
        trajectory="../tests/files/vesicle.xtc",
        analysis_type=gorder.analysis_types.CGOrder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B"
        ),
        output_yaml=temp_file_path,
        membrane_normal=gorder.membrane_normal.DynamicNormal("name PO4"),
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_vesicle.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_dynamic_normals_leaflets():
    manual_dict = read_leaflets_yaml(
        "../tests/files/inputs/leaflets_files/pcpepg_every.yaml"
    )

    for leaflets in [
        gorder.leaflets.GlobalClassification(
            "@membrane", "name P", membrane_normal="z"
        ),
        gorder.leaflets.LocalClassification(
            "@membrane", "name P", radius=2.5, membrane_normal="z"
        ),
        gorder.leaflets.IndividualClassification(
            "name P", "name C218 C316", membrane_normal="z"
        ),
        gorder.leaflets.ManualClassification(
            "../tests/files/inputs/leaflets_files/pcpepg_every.yaml"
        ),
        gorder.leaflets.ManualClassification(manual_dict),
        gorder.leaflets.NdxClassification(
            ndx=["../tests/files/ndx/pcpepg_leaflets.ndx"] * 51,
            heads="name P",
            upper_leaflet="Upper",
            lower_leaflet="Lower",
        ),
    ]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        analysis = gorder.Analysis(
            structure="../tests/files/pcpepg.tpr",
            trajectory="../tests/files/pcpepg.xtc",
            analysis_type=gorder.analysis_types.AAOrder(
                "@membrane and element name carbon",
                "@membrane and element name hydrogen",
            ),
            leaflets=leaflets,
            membrane_normal=gorder.membrane_normal.DynamicNormal("name P", 2.0),
            output_yaml=temp_file_path,
            silent=True,
            overwrite=True,
        )

        analysis.run().write()

        try:
            assert diff_files_ignore_first(
                temp_file_path, "../tests/files/aa_order_leaflets_dynamic.yaml", 1
            ), "Files do not match!"
        finally:
            shutil.rmtree(temp_file_path, ignore_errors=True)


def test_manual_normals():
    dict = read_normals_yaml("../tests/files/normals_vesicle.yaml")
    for input in ["../tests/files/normals_vesicle.yaml", dict]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        analysis = gorder.Analysis(
            structure="../tests/files/vesicle.tpr",
            trajectory="../tests/files/vesicle.xtc",
            analysis_type=gorder.analysis_types.CGOrder(
                "name C1A D2A C3A C4A C1B C2B C3B C4B"
            ),
            output_yaml=temp_file_path,
            membrane_normal=input,
            silent=True,
            overwrite=True,
        )

        analysis.run().write()

        try:
            assert diff_files_ignore_first(
                temp_file_path, "../tests/files/cg_order_vesicle.yaml", 1
            ), "Files do not match!"
        finally:
            shutil.rmtree(temp_file_path, ignore_errors=True)


def test_begin_end_step():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "@membrane and element name carbon", "@membrane and element name hydrogen"
        ),
        leaflets=gorder.leaflets.GlobalClassification(
            "@membrane", "name P", frequency=gorder.Frequency.once()
        ),
        begin=450200.0,
        end=450400.0,
        step=3,
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/aa_order_begin_end_step.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_min_samples():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "@membrane and element name carbon", "@membrane and element name hydrogen"
        ),
        min_samples=2000,
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/aa_order_limit.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_estimate_error():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("@membrane"),
        estimate_error=gorder.estimate_error.EstimateError(),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_error.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_ua_estimate_error_leaflets():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            saturated="(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)",
            unsaturated="(resname POPC and name C24 C25) or (resname POPS and name C27 C28)",
        ),
        estimate_error=gorder.estimate_error.EstimateError(),
        leaflets=gorder.leaflets.LocalClassification("@membrane", "name r'^P'", 2.5),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/ua_order_leaflets_error.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_convergence():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("@membrane"),
        estimate_error=gorder.estimate_error.EstimateError(
            output_convergence=temp_file_path
        ),
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_convergence.xvg", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_geometry_cuboid():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("@membrane"),
        geometry=gorder.geometry.Cuboid(reference="center", xdim=[-8, -2], ydim=[2, 8]),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_cuboid_square.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_geometry_cylinder():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "resname POPC and name C22 C24 C218", "@membrane and element name hydrogen"
        ),
        geometry=gorder.geometry.Cylinder(
            reference=[8, 2, 0], radius=2.5, orientation="z"
        ),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/aa_order_cylinder.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_ua_geometry_cylinder_center():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            saturated="(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)",
            unsaturated="(resname POPC and name C24 C25) or (resname POPS and name C27 C28)",
        ),
        geometry=gorder.geometry.Cylinder(
            reference="center", radius=2.5, orientation="z"
        ),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/ua_order_cylinder_center.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_geometry_sphere():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("@membrane"),
        geometry=gorder.geometry.Sphere(reference="resid 1", radius=2.5),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_sphere.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_ignore_pbc():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg_whole_nobox.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "@membrane and element name carbon", "@membrane and element name hydrogen"
        ),
        output_yaml=temp_file_path,
        leaflets=gorder.leaflets.GlobalClassification(
            "@membrane", "name P", frequency=gorder.Frequency.once()
        ),
        handle_pbc=False,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/aa_order_leaflets_nopbc.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_ordermaps_basic():
    temp_dir = tempfile.mkdtemp()
    dir_path = os.path.abspath(temp_dir)

    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder(
            "resname POPC and name C1B C2B C3B C4B"
        ),
        ordermap=gorder.ordermap.OrderMap(dir_path, bin_size=[1, 1], min_samples=10),
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    expected_file_names = [
        "ordermap_POPC-C1B-8--POPC-C2B-9_full.dat",
        "ordermap_POPC-C2B-9--POPC-C3B-10_full.dat",
        "ordermap_POPC-C3B-10--POPC-C4B-11_full.dat",
        "ordermap_average_full.dat",
    ]

    try:
        for file in expected_file_names:
            assert diff_files_ignore_first(
                f"{dir_path}/POPC/{file}", f"../tests/files/ordermaps_cg/{file}", 2
            ), "Files do not match!"

        assert diff_files_ignore_first(
            f"{dir_path}/ordermap_average_full.dat",
            "../tests/files/ordermaps_cg/ordermap_average_full.dat",
            2,
        ), "Files do not match!"
        assert diff_files_ignore_first(
            f"{dir_path}/plot.py", "../scripts/plot.py", 0
        ), "Files do not match!"

    finally:
        shutil.rmtree(dir_path, ignore_errors=True)


def test_ordermaps_leaflets_nopbc_manual_everything():
    temp_dir = tempfile.mkdtemp()
    dir_path = os.path.abspath(temp_dir)

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg_whole_nobox.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "resname POPC and name C22 C24 C218", "@membrane and element name hydrogen"
        ),
        leaflets=gorder.leaflets.IndividualClassification("name P", "name C218 C316"),
        ordermap=gorder.ordermap.OrderMap(
            dir_path,
            bin_size=[0.1, 4.0],
            min_samples=5,
            dim=[[0, 9], [0, 8]],
            plane="xy",
        ),
        handle_pbc=False,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    expected_file_names = [
        "ordermap_average_full.dat",
        "ordermap_average_upper.dat",
        "ordermap_average_lower.dat",
    ]

    try:
        for file in expected_file_names:
            assert diff_files_ignore_first(
                f"{dir_path}/POPC/{file}", f"../tests/files/ordermaps_nopbc/{file}", 2
            ), "Files do not match!"

        assert diff_files_ignore_first(
            f"{dir_path}/ordermap_average_full.dat",
            "../tests/files/ordermaps_nopbc/ordermap_average_full.dat",
            2,
        ), "Files do not match!"
        assert diff_files_ignore_first(
            f"{dir_path}/plot.py", "../scripts/plot.py", 0
        ), "Files do not match!"

    finally:
        shutil.rmtree(dir_path, ignore_errors=True)


def test_ua_order_satured_only():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            saturated="(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)"
        ),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/ua_order_basic_saturated.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_ua_order_unsatured_only():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            unsaturated="(resname POPC and name C24 C25) or (resname POPS and name C27 C28)"
        ),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/ua_order_basic_unsaturated.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_ua_order_from_aa():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            saturated="@membrane and element name carbon and not name C29 C210 C21 C31",
            unsaturated="@membrane and name C29 C210",
            ignore="element name hydrogen",
        ),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/ua_order_from_aa.yaml", 1
        ), "Files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)


def test_dynamic_normals_export():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_file_normals:
        temp_file_normals_path = temp_file_normals.name

    analysis = gorder.Analysis(
        structure="../tests/files/vesicle.tpr",
        trajectory="../tests/files/vesicle.xtc",
        analysis_type=gorder.analysis_types.CGOrder(
            "name C1A D2A C3A C4A C1B C2B C3B C4B"
        ),
        output_yaml=temp_file_path,
        membrane_normal=gorder.membrane_normal.DynamicNormal(
            "name PO4", 2.0, collect=temp_file_normals_path
        ),
        silent=True,
        overwrite=True,
    )

    analysis.run().write()

    try:
        assert diff_files_ignore_first(
            temp_file_path, "../tests/files/cg_order_vesicle.yaml", 1
        ), "Order files do not match!"
        assert diff_files_ignore_first(
            temp_file_normals_path, f"../tests/files/normals_vesicle.yaml", 1
        ), "Normals files do not match!"
    finally:
        shutil.rmtree(temp_file_path, ignore_errors=True)
        shutil.rmtree(temp_file_normals_path, ignore_errors=True)


def test_ua_order_fail_no_carbons():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(),
        output_yaml=temp_file_path,
        silent=True,
        overwrite=True,
    )

    with pytest.raises(gorder.exceptions.AnalysisError) as excinfo:
        analysis.run()
    assert (
        "no carbons for the calculation of united-atom order parameters were specified"
        in str(excinfo.value)
    )
