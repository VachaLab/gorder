"""
Released under MIT License.
Copyright (c) 2024-2026 Ladislav Bartos
"""

# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalIterable=false

import gorder, pytest, math

def compare_orders(x: float, y: float) -> bool:
    return math.isclose(round(x, 4), round(y, 4), rel_tol = 1e-4)

def compare_normals(x: list[float], y: list[float]) -> bool:
    """
    Expects normals not containing any NaN values.
    """

    for (val1, val2) in zip(x, y):
        if not math.isclose(round(val1, 5), round(val2, 5), rel_tol = 1e-4):
            return False
    
    return True

def normal_is_nan(x: list[float]) -> bool:
    return math.isnan(x[0]) and math.isnan(x[1]) and math.isnan(x[2])

def test_aa_order_basic():
    analysis = gorder.Analysis(
        structure = "../tests/files/pcpepg.tpr",
        trajectory = "../tests/files/pcpepg.xtc",
        analysis_type = gorder.analysis_types.AAOrder("@membrane and element name carbon", "@membrane and element name hydrogen"),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 3
    assert results.normals_data() is None
    
    assert compare_orders(results.average_order().total().value(), 0.1423)
    assert results.average_order().total().error() is None
    assert results.average_order().upper() is None
    assert results.average_order().lower() is None
    
    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    with pytest.raises(gorder.exceptions.APIError) as excinfo:
        results.get_molecule("POPA")
    assert "molecule with the given name does not exist" in str(excinfo.value)

    expected_average_orders = [0.1455, 0.1378, 0.1561]
    expected_atom_numbers = [37, 40, 38]
    expected_molecule_names = ["POPE", "POPC", "POPG"]

    expected_atom_indices = [32, 41, 34]
    expected_atom_names = ["C32", "C32", "C32"]
    expected_atom_order = [0.2226, 0.2363, 0.2247]

    expected_bond_numbers = [2, 2, 2]

    expected_atom2_indices = [34, 43, 36]
    expected_atom2_names = ["H2Y", "H2Y", "H2Y"]
    expected_atom2_order = [0.2040, 0.2317, 0.2020]

    for (i, molecule) in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        average_order = molecule.average_order()
        assert compare_orders(average_order.total().value(), expected_average_orders[i])
        assert average_order.total().error() is None
        assert average_order.upper() is None
        assert average_order.lower() is None

        average_maps = molecule.average_ordermaps()
        assert average_maps.total() is None
        assert average_maps.upper() is None
        assert average_maps.lower() is None

        # ATOMS
        assert len(molecule.atoms()) == expected_atom_numbers[i]

        atom = molecule.get_atom(expected_atom_indices[i])
        atom_type = atom.atom()
        assert atom_type.atom_name() == expected_atom_names[i]
        assert atom_type.relative_index() == expected_atom_indices[i]
        assert atom_type.residue_name() == expected_molecule_names[i]
        assert atom.molecule() == expected_molecule_names[i]

        order = atom.order()
        assert compare_orders(order.total().value(), expected_atom_order[i])
        assert order.total().error() is None
        assert order.upper() is None
        assert order.lower() is None

        maps = atom.ordermaps()
        assert maps.total() is None
        assert maps.upper() is None
        assert maps.lower() is None

        # BONDS
        assert len(atom.bonds()) == expected_bond_numbers[i]

        bond = atom.get_bond(expected_atom2_indices[i])
        a1, a2 = bond.atoms()
        assert a1.atom_name() == expected_atom_names[i]
        assert a1.relative_index() == expected_atom_indices[i]
        assert a1.residue_name() == expected_molecule_names[i]
        assert a2.atom_name() == expected_atom2_names[i]
        assert a2.relative_index() == expected_atom2_indices[i]
        assert a2.residue_name() == expected_molecule_names[i]
        assert bond.molecule() == expected_molecule_names[i]

        order = bond.order()
        assert compare_orders(order.total().value(), expected_atom2_order[i])
        assert order.total().error() is None
        assert order.upper() is None
        assert order.lower() is None

        maps = bond.ordermaps()
        assert maps.total() is None
        assert maps.upper() is None
        assert maps.lower() is None

        # BOND FROM MOLECULE
        bond = molecule.get_bond(expected_atom_indices[i], expected_atom2_indices[i])
        a1, a2 = bond.atoms()
        assert a1.relative_index() == expected_atom_indices[i]
        assert a2.relative_index() == expected_atom2_indices[i]

        bond = molecule.get_bond(expected_atom2_indices[i], expected_atom_indices[i])
        a1, a2 = bond.atoms()
        assert a1.relative_index() == expected_atom_indices[i]
        assert a2.relative_index() == expected_atom2_indices[i]

        # NONEXISTENT ATOM
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(145)
        assert "atom with the given relative index does not exist" in str(excinfo.value)

        # NONEXISTENT BOND
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(7, 19)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(145, 189)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

def test_cg_order_basic():
    analysis = gorder.Analysis(
        structure = "../tests/files/cg.tpr",
        trajectory = "../tests/files/cg.xtc",
        analysis_type = gorder.analysis_types.CGOrder("@membrane"),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 101
    assert len(results.molecules()) == 3
    
    assert compare_orders(results.average_order().total().value(), 0.2962)
    assert results.average_order().total().error() is None
    assert results.average_order().upper() is None
    assert results.average_order().lower() is None
    
    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    with pytest.raises(gorder.exceptions.APIError) as excinfo:
        results.get_molecule("POPA")
    assert "molecule with the given name does not exist" in str(excinfo.value)

    expected_molecule_names = ["POPC", "POPE", "POPG"]
    expected_average_orders = [0.2943, 0.2972, 0.3059]
    expected_bond_orders = [0.3682, 0.3759, 0.3789]

    for (i, molecule) in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        average_order = molecule.average_order()
        assert compare_orders(average_order.total().value(), expected_average_orders[i])
        assert average_order.total().error() is None
        assert average_order.upper() is None
        assert average_order.lower() is None

        average_maps = molecule.average_ordermaps()
        assert average_maps.total() is None
        assert average_maps.upper() is None
        assert average_maps.lower() is None

        # BONDS
        assert len(molecule.bonds()) == 11

        bond = molecule.get_bond(4, 5)
        a1, a2 = bond.atoms()
        assert a1.atom_name() == "C1A"
        assert a1.relative_index() == 4
        assert a1.residue_name() == expected_molecule_names[i]
        assert a2.atom_name() == "D2A"
        assert a2.relative_index() == 5
        assert a2.residue_name() == expected_molecule_names[i]

        order = bond.order()
        assert compare_orders(order.total().value(), expected_bond_orders[i])
        assert order.total().error() is None
        assert order.upper() is None
        assert order.lower() is None

        maps = bond.ordermaps()
        assert maps.total() is None
        assert maps.upper() is None
        assert maps.lower() is None

        # THE SAME BOND
        bond = molecule.get_bond(5, 4)
        a1, a2 = bond.atoms()
        assert a1.relative_index() == 4
        assert a2.relative_index() == 5

        # NONEXISTENT BOND
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(1, 3)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(15, 16)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

        # ATTEMPTING TO ACCESS ATOMS
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.atoms()
        assert "results for individual atoms are not available for coarse-grained order parameters" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(3)
        assert "results for individual atoms are not available for coarse-grained order parameters" in str(excinfo.value)

def test_ua_order_basic():
    analysis = gorder.Analysis(
        structure = "../tests/files/ua.tpr",
        trajectory = "../tests/files/ua.xtc",
        analysis_type = gorder.analysis_types.UAOrder(
            saturated = "(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)",
            unsaturated = "(resname POPC and name C24 C25) or (resname POPS and name C27 C28)",
        ),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 2

    results.get_molecule("POPC")
    results.get_molecule("POPS")

    with pytest.raises(gorder.exceptions.APIError) as excinfo:
        results.get_molecule("POPG")
    assert "molecule with the given name does not exist" in str(excinfo.value)

    assert compare_orders(results.average_order().total().value(), 0.1169)
    assert results.average_order().total().error() is None
    assert results.average_order().upper() is None
    assert results.average_order().lower() is None

    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_average_orders = [0.1101, 0.1470]
    expected_atom_numbers = [40, 37]
    expected_molecule_names = ["POPC", "POPS"]

    expected_atom_indices = [23, 45]
    expected_atom_names = ["C24", "C46"]
    expected_atom_order = [0.0978, 0.2221]

    expected_mol_bond_numbers = [82, 72]
    expected_bond_numbers = [1, 2]
    expected_bond_orders = [[0.0978], [0.2084, 0.2359]]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        average_order = molecule.average_order()
        assert compare_orders(average_order.total().value(), expected_average_orders[i])
        assert average_order.total().error() is None
        assert average_order.upper() is None
        assert average_order.lower() is None

        average_maps = molecule.average_ordermaps()
        assert average_maps.total() is None
        assert average_maps.upper() is None
        assert average_maps.lower() is None

        # ATOMS
        assert len(molecule.atoms()) == expected_atom_numbers[i]

        atom = molecule.get_atom(expected_atom_indices[i])
        atom_type = atom.atom()
        assert atom_type.atom_name() == expected_atom_names[i]
        assert atom_type.relative_index() == expected_atom_indices[i]
        assert atom_type.residue_name() == expected_molecule_names[i]
        assert atom.molecule() == expected_molecule_names[i]

        order = atom.order()
        assert compare_orders(order.total().value(), expected_atom_order[i])
        assert order.total().error() is None
        assert order.upper() is None
        assert order.lower() is None

        maps = atom.ordermaps()
        assert maps.total() is None
        assert maps.upper() is None
        assert maps.lower() is None

        # BONDS
        assert len(molecule.bonds()) == expected_mol_bond_numbers[i]
        assert len(atom.bonds()) == expected_bond_numbers[i]

        for b, bond in enumerate(atom.bonds()):
            assert compare_orders(bond.order().total().value(), expected_bond_orders[i][b])
            assert bond.order().total().error() is None
            assert bond.order().upper() is None
            assert bond.order().lower() is None

            maps = bond.ordermaps()
            assert maps.total() is None
            assert maps.upper() is None
            assert maps.lower() is None

        # NONEXISTENT ATOMS
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(145)
        assert "atom with the given relative index does not exist" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(7)
        assert "atom with the given relative index does not exist" in str(excinfo.value)

        # ACCESSING BONDS
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(49, 1)
        assert "united-atom results for individual bonds cannot be accesed by using relative indices" in str(excinfo.value)

def test_aa_order_error():
    analysis = gorder.Analysis(
        structure = "../tests/files/pcpepg.tpr",
        trajectory = "../tests/files/pcpepg.xtc",
        analysis_type = gorder.analysis_types.AAOrder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen"
        ),
        estimate_error = gorder.estimate_error.EstimateError(),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 3

    avg_order = results.average_order()
    assert compare_orders(avg_order.total().value(), 0.1423)
    assert compare_orders(avg_order.total().error(), 0.0026)
    assert avg_order.upper() is None
    assert avg_order.lower() is None

    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_average_orders = [0.1455, 0.1378, 0.1561]
    expected_average_errors = [0.0029, 0.0036, 0.0112]
    expected_atom_numbers = [37, 40, 38]
    expected_molecule_names = ["POPE", "POPC", "POPG"]

    expected_atom_indices = [32, 41, 34]
    expected_atom_names = ["C32", "C32", "C32"]
    expected_atom_order = [0.2226, 0.2363, 0.2247]
    expected_atom_errors = [0.0087, 0.0071, 0.0574]

    expected_bond_numbers = [2, 2, 2]

    expected_atom2_indices = [34, 43, 36]
    expected_atom2_names = ["H2Y", "H2Y", "H2Y"]
    expected_atom2_order = [0.2040, 0.2317, 0.2020]
    expected_atom2_errors = [0.0125, 0.0091, 0.0656]

    expected_convergence_frames = list(range(1, 52))
    expected_convergence_values = [
        [0.1494, 0.1460, 0.1455],
        [0.1422, 0.1353, 0.1378],
        [0.1572, 0.1507, 0.1561],
    ]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        avg_order = molecule.average_order()
        assert compare_orders(avg_order.total().value(), expected_average_orders[i])
        assert compare_orders(avg_order.total().error(), expected_average_errors[i])
        assert avg_order.upper() is None
        assert avg_order.lower() is None

        # CONVERGENCE
        convergence = molecule.convergence()
        assert convergence.frames() == expected_convergence_frames
        
        sample_frames = [0, 25, 50]
        extracted_conv = convergence.total()
        for j, frame in enumerate(sample_frames):
            conv_value = extracted_conv[frame]
            expected = expected_convergence_values[i][j]
            assert compare_orders(conv_value, expected)

        assert convergence.upper() is None
        assert convergence.lower() is None

        # ATOMS
        assert len(molecule.atoms()) == expected_atom_numbers[i]
        atom = molecule.get_atom(expected_atom_indices[i])
        atom_type = atom.atom()
        assert atom_type.atom_name() == expected_atom_names[i]
        assert atom_type.relative_index() == expected_atom_indices[i]
        assert atom_type.residue_name() == expected_molecule_names[i]
        assert atom.molecule() == expected_molecule_names[i]

        order = atom.order()
        assert compare_orders(order.total().value(), expected_atom_order[i])
        assert compare_orders(order.total().error(), expected_atom_errors[i])
        assert order.upper() is None
        assert order.lower() is None

        maps = atom.ordermaps()
        assert maps.total() is None
        assert maps.upper() is None
        assert maps.lower() is None

        # BONDS
        assert len(atom.bonds()) == expected_bond_numbers[i]
        bond = atom.get_bond(expected_atom2_indices[i])
        a1, a2 = bond.atoms()
        assert a1.atom_name() == expected_atom_names[i]
        assert a1.relative_index() == expected_atom_indices[i]
        assert a1.residue_name() == expected_molecule_names[i]
        assert a2.atom_name() == expected_atom2_names[i]
        assert a2.relative_index() == expected_atom2_indices[i]
        assert a2.residue_name() == expected_molecule_names[i]
        assert bond.molecule() == expected_molecule_names[i]
        
        bond_order = bond.order()
        assert compare_orders(bond_order.total().value(), expected_atom2_order[i])
        assert compare_orders(bond_order.total().error(), expected_atom2_errors[i])
        assert bond_order.upper() is None
        assert bond_order.lower() is None

        bond_maps = bond.ordermaps()
        assert bond_maps.total() is None
        assert bond_maps.upper() is None
        assert bond_maps.lower() is None

        # BOND FROM MOLECULE
        bond = molecule.get_bond(expected_atom_indices[i], expected_atom2_indices[i])
        a1, a2 = bond.atoms()
        assert a1.relative_index() == expected_atom_indices[i]
        assert a2.relative_index() == expected_atom2_indices[i]

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(145)
        assert "atom with the given relative index does not exist" in str(excinfo.value)
            
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(7, 19)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

def test_cg_order_error():
    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("@membrane"),
        estimate_error=gorder.estimate_error.EstimateError(),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 101
    assert len(results.molecules()) == 3

    avg_order = results.average_order()
    assert compare_orders(avg_order.total().value(), 0.2962)
    assert compare_orders(avg_order.total().error(), 0.0050)
    assert avg_order.upper() is None
    assert avg_order.lower() is None

    # Test ordermaps absence
    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_molecule_names = ["POPC", "POPE", "POPG"]
    expected_average_orders = [0.2943, 0.2972, 0.3059]
    expected_average_errors = [0.0067, 0.0052, 0.0089]
    expected_bond_orders = [0.3682, 0.3759, 0.3789]
    expected_bond_errors = [0.0125, 0.0164, 0.0159]
    expected_convergence_values = [
        [0.2756, 0.2902, 0.2943],
        [0.2830, 0.2995, 0.2972],
        [0.3198, 0.3066, 0.3059],
    ]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        # AVERAGE
        avg_order = molecule.average_order()
        assert compare_orders(avg_order.total().value(), expected_average_orders[i])
        assert compare_orders(avg_order.total().error(), expected_average_errors[i])
        assert avg_order.upper() is None
        assert avg_order.lower() is None

        avg_maps = molecule.average_ordermaps()
        assert avg_maps.total() is None
        assert avg_maps.upper() is None
        assert avg_maps.lower() is None

        # CONVERGENCE
        convergence = molecule.convergence()
        assert len(convergence.frames()) == 101
        assert convergence.upper() is None
        assert convergence.lower() is None

        # Check specific convergence points
        sample_indices = [0, 50, 100]
        for j, idx in enumerate(sample_indices):
            conv_value = convergence.total()[idx]
            expected = expected_convergence_values[i][j]
            assert compare_orders(conv_value, expected)

        # BONDS
        assert len(molecule.bonds()) == 11
        bond = molecule.get_bond(4, 5)
        a1, a2 = bond.atoms()
        assert a1.atom_name() == "C1A"
        assert a1.relative_index() == 4
        assert a1.residue_name() == expected_molecule_names[i]
        assert a2.atom_name() == "D2A"
        assert a2.relative_index() == 5
        assert a2.residue_name() == expected_molecule_names[i]

        order = bond.order()
        assert compare_orders(order.total().value(), expected_bond_orders[i])
        assert compare_orders(order.total().error(), expected_bond_errors[i])
        assert order.upper() is None
        assert order.lower() is None

        reverse_bond = molecule.get_bond(5, 4)
        a1_rev, a2_rev = reverse_bond.atoms()
        assert a1_rev.relative_index() == 4
        assert a2_rev.relative_index() == 5

        # NONEXISTENT BONDS
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(1, 3)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(15, 16)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

def test_ua_order_error():
    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            saturated="(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)",
            unsaturated="(resname POPC and name C24 C25) or (resname POPS and name C27 C28)",
        ),
        estimate_error=gorder.estimate_error.EstimateError(),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 2

    results.get_molecule("POPC")
    results.get_molecule("POPS")
    
    with pytest.raises(gorder.exceptions.APIError) as excinfo:
        results.get_molecule("POPG")
    assert "molecule with the given name does not exist" in str(excinfo.value)

    avg_order = results.average_order().total()
    assert compare_orders(avg_order.value(), 0.1169)
    assert compare_orders(avg_order.error(), 0.0027)
    assert results.average_order().upper() is None
    assert results.average_order().lower() is None

    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_average_orders = [0.1101, 0.1470]
    expected_average_errors = [0.0019, 0.0106]
    expected_atom_numbers = [40, 37]
    expected_molecule_names = ["POPC", "POPS"]
    
    expected_atom_indices = [23, 45]
    expected_atom_names = ["C24", "C46"]
    expected_atom_order = [0.0978, 0.2221]
    expected_atom_errors = [0.0070, 0.0241]

    expected_mol_bond_numbers = [82, 72]
    expected_bond_numbers = [1, 2]
    expected_bond_orders = [[0.0978], [0.2084, 0.2359]]
    expected_bond_errors = [[0.0070], [0.0262, 0.0441]]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        mol_order = molecule.average_order().total()
        assert compare_orders(mol_order.value(), expected_average_orders[i])
        assert compare_orders(mol_order.error(), expected_average_errors[i])
        assert molecule.average_order().upper() is None
        assert molecule.average_order().lower() is None

        assert molecule.average_ordermaps().total() is None
        assert molecule.average_ordermaps().upper() is None
        assert molecule.average_ordermaps().lower() is None

        # ATOMS
        assert len(molecule.atoms()) == expected_atom_numbers[i]
        atom = molecule.get_atom(expected_atom_indices[i])
        
        assert atom.atom().atom_name() == expected_atom_names[i]
        assert atom.atom().relative_index() == expected_atom_indices[i]
        assert atom.molecule() == expected_molecule_names[i]

        atom_order = atom.order().total()
        assert compare_orders(atom_order.value(), expected_atom_order[i])
        assert compare_orders(atom_order.error(), expected_atom_errors[i])
        assert atom.order().upper() is None
        assert atom.order().lower() is None

        assert atom.ordermaps().total() is None
        assert atom.ordermaps().upper() is None
        assert atom.ordermaps().lower() is None

        # BONDS
        assert len(molecule.bonds()) == expected_mol_bond_numbers[i]
        assert len(atom.bonds()) == expected_bond_numbers[i]

        for b_idx, bond in enumerate(atom.bonds()):
            bond_order = bond.order().total()
            assert compare_orders(bond_order.value(), expected_bond_orders[i][b_idx])
            assert compare_orders(bond_order.error(), expected_bond_errors[i][b_idx])
            assert bond.order().upper() is None
            assert bond.order().lower() is None
            
            assert bond.ordermaps().total() is None
            assert bond.ordermaps().upper() is None
            assert bond.ordermaps().lower() is None

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(145)
        assert "atom with the given relative index does not exist" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(1, 2)
        assert "united-atom results for individual bonds cannot be accesed by using relative indices" in str(excinfo.value)

def test_aa_order_leaflets():
    analysis = gorder.Analysis(
        structure = "../tests/files/pcpepg.tpr",
        trajectory = "../tests/files/pcpepg.xtc",
        analysis_type = gorder.analysis_types.AAOrder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen"
        ),
        leaflets = gorder.leaflets.GlobalClassification("@membrane", "name P"),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 3

    avg_order = results.average_order()
    assert compare_orders(avg_order.total().value(), 0.1423)
    assert compare_orders(avg_order.upper().value(), 0.1411)
    assert compare_orders(avg_order.lower().value(), 0.1434)
    assert avg_order.total().error() is None
    assert avg_order.upper().error() is None
    assert avg_order.lower().error() is None

    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_average_orders = [0.1455, 0.1378, 0.1561]
    expected_average_upper = [0.1492, 0.1326, 0.1522]
    expected_average_lower = [0.1419, 0.1431, 0.1606]
    expected_atom_numbers = [37, 40, 38]
    expected_molecule_names = ["POPE", "POPC", "POPG"]

    expected_atom_indices = [32, 41, 34]
    expected_atom_names = ["C32", "C32", "C32"]
    expected_atom_order = [0.2226, 0.2363, 0.2247]
    expected_atom_upper = [0.2131, 0.2334, 0.2484]
    expected_atom_lower = [0.2319, 0.2391, 0.1976]

    expected_bond_numbers = [2, 2, 2]

    expected_atom2_indices = [34, 43, 36]
    expected_atom2_names = ["H2Y", "H2Y", "H2Y"]
    expected_atom2_order = [0.2040, 0.2317, 0.2020]
    expected_atom2_upper = [0.1876, 0.2507, 0.2254]
    expected_atom2_lower = [0.2203, 0.2126, 0.1752]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        # AVERAGE
        avg_order = molecule.average_order()
        assert compare_orders(avg_order.total().value(), expected_average_orders[i])
        assert compare_orders(avg_order.upper().value(), expected_average_upper[i])
        assert compare_orders(avg_order.lower().value(), expected_average_lower[i])
        assert avg_order.total().error() is None
        assert avg_order.upper().error() is None
        assert avg_order.lower().error() is None

        avg_maps = molecule.average_ordermaps()
        assert avg_maps.total() is None
        assert avg_maps.upper() is None
        assert avg_maps.lower() is None

        # ATOMS
        assert len(molecule.atoms()) == expected_atom_numbers[i]
        atom = molecule.get_atom(expected_atom_indices[i])
        atom_type = atom.atom()
        assert atom_type.atom_name() == expected_atom_names[i]
        assert atom_type.relative_index() == expected_atom_indices[i]
        assert atom_type.residue_name() == expected_molecule_names[i]
        assert atom.molecule() == expected_molecule_names[i]

        order = atom.order()
        assert compare_orders(order.total().value(), expected_atom_order[i])
        assert compare_orders(order.upper().value(), expected_atom_upper[i])
        assert compare_orders(order.lower().value(), expected_atom_lower[i])
        assert order.total().error() is None
        assert order.upper().error() is None
        assert order.lower().error() is None

        maps = atom.ordermaps()
        assert maps.total() is None
        assert maps.upper() is None
        assert maps.lower() is None

        # BONDS
        assert len(atom.bonds()) == expected_bond_numbers[i]
        bond = atom.get_bond(expected_atom2_indices[i])
        a1, a2 = bond.atoms()
        assert a1.atom_name() == expected_atom_names[i]
        assert a1.relative_index() == expected_atom_indices[i]
        assert a1.residue_name() == expected_molecule_names[i]
        assert a2.atom_name() == expected_atom2_names[i]
        assert a2.relative_index() == expected_atom2_indices[i]
        assert a2.residue_name() == expected_molecule_names[i]
        assert bond.molecule() == expected_molecule_names[i]

        bond_order = bond.order()
        assert compare_orders(bond_order.total().value(), expected_atom2_order[i])
        assert compare_orders(bond_order.upper().value(), expected_atom2_upper[i])
        assert compare_orders(bond_order.lower().value(), expected_atom2_lower[i])
        assert bond_order.total().error() is None
        assert bond_order.upper().error() is None
        assert bond_order.lower().error() is None

        bond_maps = bond.ordermaps()
        assert bond_maps.total() is None
        assert bond_maps.upper() is None
        assert bond_maps.lower() is None

        # BOND FROM MOL
        bond = molecule.get_bond(expected_atom_indices[i], expected_atom2_indices[i])
        a1, a2 = bond.atoms()
        assert a1.relative_index() == expected_atom_indices[i]
        assert a2.relative_index() == expected_atom2_indices[i]

        # ERROR HANDLING
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(145)
        assert "atom with the given relative index does not exist" in str(excinfo.value)
            
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(7, 19)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

def test_cg_order_leaflets():
    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("@membrane"),
        leaflets=gorder.leaflets.GlobalClassification("@membrane", "name PO4"),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 101
    assert len(results.molecules()) == 3

    avg_order = results.average_order()
    assert compare_orders(avg_order.total().value(), 0.2962)
    assert compare_orders(avg_order.upper().value(), 0.2971)
    assert compare_orders(avg_order.lower().value(), 0.2954)
    assert avg_order.total().error() is None
    assert avg_order.upper().error() is None
    assert avg_order.lower().error() is None

    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_molecule_names = ["POPC", "POPE", "POPG"]
    expected_average_orders = [0.2943, 0.2972, 0.3059]
    expected_average_upper = [0.2965, 0.2965, 0.3085]
    expected_average_lower = [0.2920, 0.2980, 0.3033]
    expected_bond_orders = [0.3682, 0.3759, 0.3789]
    expected_bond_upper = [0.3647, 0.3713, 0.4129]
    expected_bond_lower = [0.3717, 0.3806, 0.3449]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        # AVERAGE
        avg_order = molecule.average_order()
        assert compare_orders(avg_order.total().value(), expected_average_orders[i])
        assert compare_orders(avg_order.upper().value(), expected_average_upper[i])
        assert compare_orders(avg_order.lower().value(), expected_average_lower[i])
        assert avg_order.total().error() is None
        assert avg_order.upper().error() is None
        assert avg_order.lower().error() is None

        avg_maps = molecule.average_ordermaps()
        assert avg_maps.total() is None
        assert avg_maps.upper() is None
        assert avg_maps.lower() is None

        # BONDS
        assert len(molecule.bonds()) == 11
        bond = molecule.get_bond(4, 5)
        a1, a2 = bond.atoms()
        assert a1.atom_name() == "C1A"
        assert a1.relative_index() == 4
        assert a1.residue_name() == expected_molecule_names[i]
        assert a2.atom_name() == "D2A"
        assert a2.relative_index() == 5
        assert a2.residue_name() == expected_molecule_names[i]

        order = bond.order()
        assert compare_orders(order.total().value(), expected_bond_orders[i])
        assert compare_orders(order.upper().value(), expected_bond_upper[i])
        assert compare_orders(order.lower().value(), expected_bond_lower[i])
        assert order.total().error() is None
        assert order.upper().error() is None
        assert order.lower().error() is None

        reverse_bond = molecule.get_bond(5, 4)
        a1_rev, a2_rev = reverse_bond.atoms()
        assert a1_rev.relative_index() == 4
        assert a2_rev.relative_index() == 5

        # NONEXISTENT BONDS
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(1, 3)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(15, 16)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

def test_ua_order_leaflets():
    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            saturated="(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)",
            unsaturated="(resname POPC and name C24 C25) or (resname POPS and name C27 C28)",
        ),
        leaflets=gorder.leaflets.GlobalClassification("@membrane", "name r'^P'"),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 2

    results.get_molecule("POPC")
    results.get_molecule("POPS")
    
    with pytest.raises(gorder.exceptions.APIError) as excinfo:
        results.get_molecule("POPG")
    assert "molecule with the given name does not exist" in str(excinfo.value)

    avg_order = results.average_order()
    assert compare_orders(avg_order.total().value(), 0.1169)
    assert compare_orders(avg_order.upper().value(), 0.1151)
    assert compare_orders(avg_order.lower().value(), 0.1186)
    
    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_average_orders = [0.1101, 0.1470]
    expected_average_upper = [0.1075, 0.1491]
    expected_average_lower = [0.1128, 0.1449]
    expected_atom_numbers = [40, 37]
    expected_molecule_names = ["POPC", "POPS"]

    expected_atom_indices = [23, 45]
    expected_atom_names = ["C24", "C46"]
    expected_atom_order = [0.0978, 0.2221]
    expected_atom_upper = [0.1088, 0.2204]
    expected_atom_lower = [0.0869, 0.2239]

    expected_mol_bond_numbers = [82, 72]
    expected_bond_numbers = [1, 2]
    expected_bond_orders = [[0.0978], [0.2084, 0.2359]]
    expected_bond_upper = [[0.1088], [0.1986, 0.2421]]
    expected_bond_lower = [[0.0869], [0.2181, 0.2296]]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        mol_order = molecule.average_order()
        assert compare_orders(mol_order.total().value(), expected_average_orders[i])
        assert compare_orders(mol_order.upper().value(), expected_average_upper[i])
        assert compare_orders(mol_order.lower().value(), expected_average_lower[i])
        
        assert molecule.average_ordermaps().total() is None
        assert molecule.average_ordermaps().upper() is None
        assert molecule.average_ordermaps().lower() is None

        assert len(molecule.atoms()) == expected_atom_numbers[i]
        atom = molecule.get_atom(expected_atom_indices[i])
        
        assert atom.atom().atom_name() == expected_atom_names[i]
        assert atom.atom().relative_index() == expected_atom_indices[i]
        assert atom.molecule() == expected_molecule_names[i]

        atom_order = atom.order()
        assert compare_orders(atom_order.total().value(), expected_atom_order[i])
        assert compare_orders(atom_order.upper().value(), expected_atom_upper[i])
        assert compare_orders(atom_order.lower().value(), expected_atom_lower[i])
        
        assert atom.ordermaps().total() is None
        assert atom.ordermaps().upper() is None
        assert atom.ordermaps().lower() is None

        assert len(molecule.bonds()) == expected_mol_bond_numbers[i]
        assert len(atom.bonds()) == expected_bond_numbers[i]

        for b_idx, bond in enumerate(atom.bonds()):
            assert compare_orders(bond.order().total().value(), expected_bond_orders[i][b_idx])
            assert compare_orders(bond.order().upper().value(), expected_bond_upper[i][b_idx])
            assert compare_orders(bond.order().lower().value(), expected_bond_lower[i][b_idx])
            
            assert bond.ordermaps().total() is None
            assert bond.ordermaps().upper() is None
            assert bond.ordermaps().lower() is None

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(145)
        assert "atom with the given relative index does not exist" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(1, 2)
        assert "united-atom results for individual bonds cannot be accesed by using relative indices" in str(excinfo.value)

def test_aa_order_error_leaflets():
    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen"
        ),
        leaflets=gorder.leaflets.GlobalClassification("@membrane", "name P"),
        estimate_error=gorder.estimate_error.EstimateError(),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 3

    avg_order = results.average_order()
    assert compare_orders(avg_order.total().value(), 0.1423)
    assert compare_orders(avg_order.total().error(), 0.0026)
    assert compare_orders(avg_order.upper().value(), 0.1411)
    assert compare_orders(avg_order.upper().error(), 0.0024)
    assert compare_orders(avg_order.lower().value(), 0.1434)
    assert compare_orders(avg_order.lower().error(), 0.0031)

    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_atom_numbers = [37, 40, 38]
    expected_molecule_names = ["POPE", "POPC", "POPG"]

    expected_atom_indices = [32, 41, 34]
    expected_atom_names = ["C32", "C32", "C32"]

    expected_bond_numbers = [2, 2, 2]

    expected_atom2_indices = [34, 43, 36]
    expected_atom2_names = ["H2Y", "H2Y", "H2Y"]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        avg_order = molecule.average_order()
        assert avg_order.total().error() is not None
        assert avg_order.upper().error() is not None
        assert avg_order.lower().error() is not None

        # Test ordermaps absence
        avg_maps = molecule.average_ordermaps()
        assert avg_maps.total() is None
        assert avg_maps.upper() is None
        assert avg_maps.lower() is None

        convergence = molecule.convergence()
        assert len(convergence.frames()) == 51
        assert convergence.total() is not None
        assert convergence.upper() is not None
        assert convergence.lower() is not None

        assert len(molecule.atoms()) == expected_atom_numbers[i]
        atom = molecule.get_atom(expected_atom_indices[i])
        atom_type = atom.atom()
        assert atom_type.atom_name() == expected_atom_names[i]
        assert atom_type.relative_index() == expected_atom_indices[i]
        assert atom_type.residue_name() == expected_molecule_names[i]
        assert atom.molecule() == expected_molecule_names[i]

        order = atom.order()
        assert order.total().error() is not None
        assert order.upper().error() is not None
        assert order.lower().error() is not None

        maps = atom.ordermaps()
        assert maps.total() is None
        assert maps.upper() is None
        assert maps.lower() is None

        assert len(atom.bonds()) == expected_bond_numbers[i]
        bond = atom.get_bond(expected_atom2_indices[i])
        a1, a2 = bond.atoms()
        assert a1.atom_name() == expected_atom_names[i]
        assert a1.relative_index() == expected_atom_indices[i]
        assert a1.residue_name() == expected_molecule_names[i]
        assert a2.atom_name() == expected_atom2_names[i]
        assert a2.relative_index() == expected_atom2_indices[i]
        assert a2.residue_name() == expected_molecule_names[i]
        assert bond.molecule() == expected_molecule_names[i]

        bond_order = bond.order()
        assert bond_order.total().error() is not None
        assert bond_order.upper().error() is not None
        assert bond_order.lower().error() is not None

        bond_maps = bond.ordermaps()
        assert bond_maps.total() is None
        assert bond_maps.upper() is None
        assert bond_maps.lower() is None

        bond = molecule.get_bond(expected_atom_indices[i], expected_atom2_indices[i])
        a1, a2 = bond.atoms()
        assert a1.relative_index() == expected_atom_indices[i]
        assert a2.relative_index() == expected_atom2_indices[i]

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_atom(145)
        assert "atom with the given relative index does not exist" in str(excinfo.value)
            
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(7, 19)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

def test_cg_order_error_leaflets():
    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("@membrane"),
        leaflets=gorder.leaflets.GlobalClassification("@membrane", "name PO4"),
        estimate_error=gorder.estimate_error.EstimateError(),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 101
    assert len(results.molecules()) == 3

    avg_order = results.average_order()
    assert compare_orders(avg_order.total().value(), 0.2962)
    assert compare_orders(avg_order.total().error(), 0.0050)
    assert compare_orders(avg_order.upper().value(), 0.2971)
    assert compare_orders(avg_order.upper().error(), 0.0049)
    assert compare_orders(avg_order.lower().value(), 0.2954)
    assert compare_orders(avg_order.lower().error(), 0.0056)

    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_molecule_names = ["POPC", "POPE", "POPG"]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        # AVERAGE
        avg_order = molecule.average_order()
        assert avg_order.total().error() is not None
        assert avg_order.upper().error() is not None
        assert avg_order.lower().error() is not None

        avg_maps = molecule.average_ordermaps()
        assert avg_maps.total() is None
        assert avg_maps.upper() is None
        assert avg_maps.lower() is None

        convergence = molecule.convergence()
        assert len(convergence.frames()) == 101
        assert convergence.total() is not None
        assert convergence.upper() is not None
        assert convergence.lower() is not None

        # BONDS
        assert len(molecule.bonds()) == 11
        bond = molecule.get_bond(4, 5)
        a1, a2 = bond.atoms()
        assert a1.atom_name() == "C1A"
        assert a1.relative_index() == 4
        assert a1.residue_name() == expected_molecule_names[i]
        assert a2.atom_name() == "D2A"
        assert a2.relative_index() == 5
        assert a2.residue_name() == expected_molecule_names[i]

        bond_order = bond.order()
        assert bond_order.total().error() is not None
        assert bond_order.upper().error() is not None
        assert bond_order.lower().error() is not None

        reverse_bond = molecule.get_bond(5, 4)
        a1_rev, a2_rev = reverse_bond.atoms()
        assert a1_rev.relative_index() == 4
        assert a2_rev.relative_index() == 5

        # NONEXISTENT BONDS
        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(1, 3)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(15, 16)
        assert "bond specified by the given relative indices does not exist" in str(excinfo.value)

def test_ua_order_error_leaflets():
    analysis = gorder.Analysis(
        structure="../tests/files/ua.tpr",
        trajectory="../tests/files/ua.xtc",
        analysis_type=gorder.analysis_types.UAOrder(
            saturated="(resname POPC and name r'^C' and not name C15 C34 C24 C25) or (resname POPS and name r'^C' and not name C6 C18 C39 C27 C28)",
            unsaturated="(resname POPC and name C24 C25) or (resname POPS and name C27 C28)",
        ),
        leaflets=gorder.leaflets.GlobalClassification("@membrane", "name r'^P'"),
        estimate_error=gorder.estimate_error.EstimateError(),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 2

    assert results.get_molecule("POPC") is not None
    assert results.get_molecule("POPS") is not None
    
    with pytest.raises(gorder.exceptions.APIError):
        results.get_molecule("POPG")

    avg_order = results.average_order()
    assert compare_orders(avg_order.total().value(), 0.1169)
    assert compare_orders(avg_order.upper().value(), 0.1151)
    assert compare_orders(avg_order.lower().value(), 0.1186)
    assert compare_orders(avg_order.total().error(), 0.0027)
    assert compare_orders(avg_order.upper().error(), 0.0031)
    assert compare_orders(avg_order.lower().error(), 0.0031)

    assert results.average_ordermaps().total() is None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    expected_atom_numbers = [40, 37]
    expected_molecule_names = ["POPC", "POPS"]
    expected_atom_indices = [23, 45]
    expected_atom_names = ["C24", "C46"]
    expected_bond_numbers = [1, 2]

    for i, molecule in enumerate(results.molecules()):
        assert molecule.molecule() == expected_molecule_names[i]

        mol_order = molecule.average_order()
        assert mol_order.total().error() is not None
        assert mol_order.upper().error() is not None
        assert mol_order.lower().error() is not None

        assert molecule.average_ordermaps().total() is None
        assert molecule.average_ordermaps().upper() is None
        assert molecule.average_ordermaps().lower() is None

        assert len(molecule.atoms()) == expected_atom_numbers[i]
        atom = molecule.get_atom(expected_atom_indices[i])
        
        assert atom.atom().atom_name() == expected_atom_names[i]
        assert atom.atom().relative_index() == expected_atom_indices[i]

        atom_order = atom.order()
        assert atom_order.total().error() is not None
        assert atom_order.upper().error() is not None
        assert atom_order.lower().error() is not None

        assert atom.ordermaps().total() is None
        assert atom.ordermaps().upper() is None
        assert atom.ordermaps().lower() is None

        assert len(atom.bonds()) == expected_bond_numbers[i]
        for bond in atom.bonds():
            assert bond.order().total().error() is not None
            assert bond.order().upper().error() is not None
            assert bond.order().lower().error() is not None
            
            assert bond.ordermaps().total() is None
            assert bond.ordermaps().upper() is None
            assert bond.ordermaps().lower() is None

        with pytest.raises(gorder.exceptions.APIError):
            molecule.get_atom(145)
        with pytest.raises(gorder.exceptions.APIError):
            molecule.get_atom(7)

        with pytest.raises(gorder.exceptions.APIError) as excinfo:
            molecule.get_bond(1, 2)
        assert "united-atom results for individual bonds cannot be accesed by using relative indices" in str(excinfo.value)

def test_aa_order_ordermaps():
    analysis = gorder.Analysis(
        structure = "../tests/files/pcpepg.tpr",
        trajectory = "../tests/files/pcpepg.xtc",
        analysis_type = gorder.analysis_types.AAOrder(
            "resname POPC and name C22 C24 C218", 
            "@membrane and element name hydrogen"
        ),
        ordermap = gorder.ordermap.OrderMap(bin_size = [0.1, 4.0], min_samples = 5),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 1

    # SYSTEM
    assert results.average_order().total() is not None
    assert results.average_order().upper() is None
    assert results.average_order().lower() is None
    
    map = results.average_ordermaps().total()
    assert compare_orders(map.get_at(0.6, 8.0), 0.1653)
    assert compare_orders(map.get_at(4.3, 0.0), 0.1340)
    assert compare_orders(map.get_at(9.2, 4.0), 0.1990)
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    # MOLECULE
    molecule = results.get_molecule("POPC")
    map = molecule.average_ordermaps().total()
    assert molecule.average_ordermaps().upper() is None
    assert molecule.average_ordermaps().lower() is None

    span_x = map.span_x()
    span_y = map.span_y()
    bin = map.tile_dim()

    assert math.isclose(span_x[0], 0.0, rel_tol = 1e-5)
    assert math.isclose(span_x[1], 9.15673, rel_tol = 1e-5)
    assert math.isclose(span_y[0], 0.0, rel_tol = 1e-5)
    assert math.isclose(span_y[1], 9.15673, rel_tol = 1e-5)
    assert math.isclose(bin[0], 0.1, rel_tol = 1e-5)
    assert math.isclose(bin[1], 4.0, rel_tol = 1e-5)

    assert compare_orders(map.get_at(0.6, 8.0), 0.1653)
    assert compare_orders(map.get_at(4.3, 0.0), 0.1340)
    assert compare_orders(map.get_at(9.2, 4.0), 0.1990)

    # ATOM
    atom = molecule.get_atom(47)
    map = atom.ordermaps().total()
    assert atom.ordermaps().upper() is None
    assert atom.ordermaps().lower() is None

    assert compare_orders(map.get_at(0.6, 8.0), 0.2224)
    assert compare_orders(map.get_at(4.3, 0.0), 0.1532)
    assert compare_orders(map.get_at(9.2, 4.0), 0.0982)

    # BOND
    bond = atom.get_bond(49)
    map = bond.ordermaps().total()
    assert bond.ordermaps().upper() is None
    assert bond.ordermaps().lower() is None

    assert compare_orders(map.get_at(0.6, 8.0), 0.2901)
    assert compare_orders(map.get_at(4.3, 0.0), 0.1163)
    assert math.isnan(map.get_at(9.2, 4.0))

    # EXTRACT CHECK
    (extracted_x, extracted_y, extracted_values) = map.extract()
    assert len(extracted_x) == 93
    assert len(extracted_y) == 3
   
    for (real, expected) in zip(extracted_x, [x / 10 for x in range(0, 93)]):
        assert math.isclose(real, expected, rel_tol = 1e-5)
   
    for (real, expected) in zip(extracted_y, [0.0, 4.0, 8.0]):
        assert math.isclose(real, expected, rel_tol = 1e-5)
   
    for (xi, x) in enumerate(extracted_x):
        for (yi, y) in enumerate(extracted_y):
            get = map.get_at(x, y)
            ext = extracted_values[xi][yi]
            if math.isnan(get) and math.isnan(ext):
                continue
            assert compare_orders(map.get_at(x, y), extracted_values[xi][yi])

def test_cg_order_ordermaps():
    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("resname POPC and name C1B C2B C3B C4B"),
        ordermap=gorder.ordermap.OrderMap(
            bin_size=[1.0, 1.0],
            min_samples=10
        ),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 101
    assert len(results.molecules()) == 1

    # SYSTEM
    assert results.average_order().total() is not None
    assert results.average_order().upper() is None
    assert results.average_order().lower() is None
    
    map = results.average_ordermaps().total()
    assert compare_orders(map.get_at(1.0, 8.0), 0.3590)
    assert compare_orders(map.get_at(7.0, 0.0), 0.3765)
    assert compare_orders(map.get_at(13.0, 11.0), 0.4296)
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    # MOLECULE
    molecule = results.get_molecule("POPC")
    mol_maps = molecule.average_ordermaps()
    
    total_map = mol_maps.total()
    assert mol_maps.upper() is None
    assert mol_maps.lower() is None

    span_x = total_map.span_x()
    span_y = total_map.span_y()
    bin_size = total_map.tile_dim()
    
    assert math.isclose(span_x[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_x[1], 12.747616, rel_tol=1e-5)
    assert math.isclose(span_y[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_y[1], 12.747616, rel_tol=1e-5)
    assert math.isclose(bin_size[0], 1.0, rel_tol=1e-5)
    assert math.isclose(bin_size[1], 1.0, rel_tol=1e-5)

    assert compare_orders(total_map.get_at(1.0, 8.0), 0.3590)
    assert compare_orders(total_map.get_at(7.0, 0.0), 0.3765)
    assert compare_orders(total_map.get_at(13.0, 11.0), 0.4296)

    # BOND
    bond = molecule.get_bond(9, 10)
    bond_maps = bond.ordermaps()
    
    bond_total = bond_maps.total()
    assert bond_maps.upper() is None
    assert bond_maps.lower() is None

    assert compare_orders(bond_total.get_at(1.0, 8.0), 0.3967)
    assert compare_orders(bond_total.get_at(7.0, 0.0), 0.3213)
    assert compare_orders(bond_total.get_at(13.0, 11.0), 0.4104)

    # EXTRACT CHECK
    (extracted_x, extracted_y, extracted_values) = map.extract()
    assert len(extracted_x) == 14
    assert len(extracted_y) == 14
   
    for (real, expected) in zip(extracted_x, [x for x in range(0, 14)]):
        assert math.isclose(real, expected, rel_tol = 1e-5)
   
    for (real, expected) in zip(extracted_y, [y for y in range(0, 14)]):
        assert math.isclose(real, expected, rel_tol = 1e-5)
   
    for (xi, x) in enumerate(extracted_x):
        for (yi, y) in enumerate(extracted_y):
            get = map.get_at(x, y)
            ext = extracted_values[xi][yi]
            if math.isnan(get) and math.isnan(ext):
                continue
            assert compare_orders(map.get_at(x, y), extracted_values[xi][yi])

def test_ua_order_ordermaps():
    analysis = gorder.Analysis(
        structure = "../tests/files/ua.tpr",
        trajectory = "../tests/files/ua.xtc",
        analysis_type = gorder.analysis_types.UAOrder(
            saturated = "resname POPC and name C50 C20 C13",
            unsaturated = "resname POPC and name C24"
        ),
        ordermap = gorder.ordermap.OrderMap(bin_size = [0.5, 2.0], min_samples = 5),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 1

    assert results.average_ordermaps().total() is not None
    assert results.average_ordermaps().upper() is None
    assert results.average_ordermaps().lower() is None

    molecule = results.get_molecule("POPC")
    map = molecule.average_ordermaps().total()
    
    span_x = map.span_x()
    span_y = map.span_y()
    bin = map.tile_dim()

    assert math.isclose(span_x[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_x[1], 6.53265, rel_tol=1e-5)
    assert math.isclose(span_y[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_y[1], 6.53265, rel_tol=1e-5)
    assert math.isclose(bin[0], 0.5, rel_tol=1e-5)
    assert math.isclose(bin[1], 2.0, rel_tol=1e-5)

    assert compare_orders(map.get_at(2.0, 6.0), 0.0127)
    assert compare_orders(map.get_at(4.3, 0.1), 0.1286)
    assert compare_orders(map.get_at(6.4, 2.2), 0.0839)

    atom = molecule.get_atom(49)
    atom_map = atom.ordermaps().total()
    
    assert compare_orders(atom_map.get_at(2.0, 6.0), 0.0349)
    assert compare_orders(atom_map.get_at(4.3, 0.1), -0.0160)
    assert compare_orders(atom_map.get_at(6.4, 2.2), -0.0084)

    bond = atom.bonds()[1]
    bond_map = bond.ordermaps().total()
    
    assert compare_orders(bond_map.get_at(2.0, 6.0), 0.1869)
    assert compare_orders(bond_map.get_at(4.3, 0.1), 0.0962)
    assert compare_orders(bond_map.get_at(6.4, 2.2), 0.0358)

    (extracted_x, extracted_y, extracted_values) = bond_map.extract()
    assert len(extracted_x) == 14
    assert len(extracted_y) == 4
    
    for x in extracted_x:
        assert math.isclose(x % 0.5, 0.0, abs_tol = 1e-5)
    
    expected_y = [0.0, 2.0, 4.0, 6.0]
    for real, expected in zip(extracted_y, expected_y):
        assert math.isclose(real, expected, rel_tol = 1e-5)
    
    for xi, x in enumerate(extracted_x):
        for yi, y in enumerate(extracted_y):
            map_val = bond_map.get_at(x, y)
            extracted_val = extracted_values[xi][yi]
            if math.isnan(map_val) and math.isnan(extracted_val):
                continue
            assert compare_orders(map_val, extracted_val)

def test_aa_order_ordermaps_leaflets():
    analysis = gorder.Analysis(
        structure="../tests/files/pcpepg.tpr",
        trajectory="../tests/files/pcpepg.xtc",
        analysis_type=gorder.analysis_types.AAOrder(
            "resname POPC and name C22 C24 C218",
            "@membrane and element name hydrogen"
        ),
        leaflets=gorder.leaflets.GlobalClassification("@membrane", "name P"),
        ordermap=gorder.ordermap.OrderMap(
            bin_size=[0.1, 4.0],
            min_samples=5
        ),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 1

    # SYSTEM
    sys_maps = results.average_ordermaps()
    assert compare_orders(sys_maps.total().get_at(0.6, 8.0), 0.1653)
    assert compare_orders(sys_maps.total().get_at(9.2, 4.0), 0.1990)
    assert compare_orders(sys_maps.upper().get_at(0.6, 8.0), 0.1347)
    assert compare_orders(sys_maps.upper().get_at(9.2, 4.0), 0.3196)
    assert compare_orders(sys_maps.lower().get_at(0.6, 8.0), 0.2104)
    assert compare_orders(sys_maps.lower().get_at(9.2, 4.0), 0.1106)

    # MOLECULE
    molecule = results.get_molecule("POPC")
    mol_maps = molecule.average_ordermaps()
    
    total_map = mol_maps.total()
    span_x = total_map.span_x()
    span_y = total_map.span_y()
    bin_size = total_map.tile_dim()
    
    assert math.isclose(span_x[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_x[1], 9.15673, rel_tol=1e-5)
    assert math.isclose(span_y[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_y[1], 9.15673, rel_tol=1e-5)
    assert math.isclose(bin_size[0], 0.1, rel_tol=1e-5)
    assert math.isclose(bin_size[1], 4.0, rel_tol=1e-5)
    
    assert compare_orders(total_map.get_at(0.6, 8.0), 0.1653)
    assert compare_orders(total_map.get_at(9.2, 4.0), 0.1990)
    upper_map = mol_maps.upper()
    assert compare_orders(upper_map.get_at(0.6, 8.0), 0.1347)
    assert compare_orders(upper_map.get_at(9.2, 4.0), 0.3196)
    lower_map = mol_maps.lower()
    assert compare_orders(lower_map.get_at(0.6, 8.0), 0.2104)
    assert compare_orders(lower_map.get_at(9.2, 4.0), 0.1106)

    # ATOM
    atom = molecule.get_atom(47)
    atom_maps = atom.ordermaps()

    atom_total = atom_maps.total()
    assert compare_orders(atom_total.get_at(0.6, 8.0), 0.2224)
    assert compare_orders(atom_total.get_at(9.2, 4.0), 0.0982)
    atom_upper = atom_maps.upper()
    assert compare_orders(atom_upper.get_at(0.6, 8.0), 0.2039)
    assert math.isnan(atom_upper.get_at(9.2, 4.0))
    atom_lower = atom_maps.lower()
    assert compare_orders(atom_lower.get_at(0.6, 8.0), 0.2540)
    assert math.isnan(atom_lower.get_at(9.2, 4.0))

    # BOND
    bond = atom.get_bond(49)
    bond_maps = bond.ordermaps()
    
    bond_total = bond_maps.total()
    assert compare_orders(bond_total.get_at(0.6, 8.0), 0.2901)
    assert math.isnan(bond_total.get_at(9.2, 4.0))
    bond_upper = bond_maps.upper()
    assert compare_orders(bond_upper.get_at(0.6, 8.0), 0.3584)
    assert math.isnan(bond_upper.get_at(9.2, 4.0))
    bond_lower = bond_maps.lower()
    assert compare_orders(bond_lower.get_at(0.6, 8.0), 0.1715)
    assert math.isnan(bond_lower.get_at(9.2, 4.0))

def test_cg_order_ordermaps_leaflets():
    analysis = gorder.Analysis(
        structure="../tests/files/cg.tpr",
        trajectory="../tests/files/cg.xtc",
        analysis_type=gorder.analysis_types.CGOrder("resname POPC and name C1B C2B C3B C4B"),
        leaflets=gorder.leaflets.GlobalClassification("@membrane", "name PO4"),
        ordermap=gorder.ordermap.OrderMap(
            bin_size=[1.0, 1.0],
            min_samples=10
        ),
        silent=True,
        overwrite=True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 101
    assert len(results.molecules()) == 1

    # SYSTEM
    sys_maps = results.average_ordermaps()
    assert compare_orders(sys_maps.total().get_at(1.0, 8.0), 0.3590)
    assert compare_orders(sys_maps.total().get_at(13.0, 11.0), 0.4296)
    assert compare_orders(sys_maps.upper().get_at(1.0, 8.0), 0.3418)
    assert compare_orders(sys_maps.upper().get_at(13.0, 11.0), 0.4051)
    assert compare_orders(sys_maps.lower().get_at(1.0, 8.0), 0.3662)
    assert compare_orders(sys_maps.lower().get_at(13.0, 11.0), 0.4506)

    # MOLECULE
    molecule = results.get_molecule("POPC")
    mol_maps = molecule.average_ordermaps()

    total_map = mol_maps.total()
    span_x = total_map.span_x()
    span_y = total_map.span_y()
    bin_size = total_map.tile_dim()
    
    assert math.isclose(span_x[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_x[1], 12.747616, rel_tol=1e-5)
    assert math.isclose(span_y[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_y[1], 12.747616, rel_tol=1e-5)
    assert math.isclose(bin_size[0], 1.0, rel_tol=1e-5)
    assert math.isclose(bin_size[1], 1.0, rel_tol=1e-5)

    assert compare_orders(total_map.get_at(1.0, 8.0), 0.3590)
    assert compare_orders(total_map.get_at(13.0, 11.0), 0.4296)
    upper_map = mol_maps.upper()
    assert compare_orders(upper_map.get_at(1.0, 8.0), 0.3418)
    assert compare_orders(upper_map.get_at(13.0, 11.0), 0.4051)
    lower_map = mol_maps.lower()
    assert compare_orders(lower_map.get_at(1.0, 8.0), 0.3662)
    assert compare_orders(lower_map.get_at(13.0, 11.0), 0.4506)

    # BOND
    bond = molecule.get_bond(9, 10)
    bond_maps = bond.ordermaps()
    
    bond_total = bond_maps.total()
    assert compare_orders(bond_total.get_at(1.0, 8.0), 0.3967)
    assert compare_orders(bond_total.get_at(13.0, 11.0), 0.4104)
    bond_upper = bond_maps.upper()
    assert compare_orders(bond_upper.get_at(1.0, 8.0), 0.3573)
    assert compare_orders(bond_upper.get_at(13.0, 11.0), 0.4807)
    bond_lower = bond_maps.lower()
    assert compare_orders(bond_lower.get_at(1.0, 8.0), 0.4118)
    assert compare_orders(bond_lower.get_at(13.0, 11.0), 0.3563)

def test_ua_order_leaflets_ordermaps():
    analysis = gorder.Analysis(
        structure = "../tests/files/ua.tpr",
        trajectory = "../tests/files/ua.xtc",
        analysis_type = gorder.analysis_types.UAOrder(
            saturated = "resname POPC and name C50 C20 C13",
            unsaturated = "resname POPC and name C24"
        ),
        ordermap = gorder.ordermap.OrderMap(bin_size=[0.5, 2.0], min_samples=5),
        leaflets = gorder.leaflets.GlobalClassification("@membrane", "name r'^P'"),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.n_analyzed_frames() == 51
    assert len(results.molecules()) == 1

    assert results.average_ordermaps().total() is not None
    assert results.average_ordermaps().upper() is not None
    assert results.average_ordermaps().lower() is not None

    molecule = results.get_molecule("POPC")
    total_map = molecule.average_ordermaps().total()
    upper_map = molecule.average_ordermaps().upper()
    lower_map = molecule.average_ordermaps().lower()

    span_x = total_map.span_x()
    span_y = total_map.span_y()
    bin = total_map.tile_dim()

    assert math.isclose(span_x[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_x[1], 6.53265, rel_tol=1e-5)
    assert math.isclose(span_y[0], 0.0, rel_tol=1e-5)
    assert math.isclose(span_y[1], 6.53265, rel_tol=1e-5)
    assert math.isclose(bin[0], 0.5, rel_tol=1e-5)
    assert math.isclose(bin[1], 2.0, rel_tol=1e-5)

    assert compare_orders(total_map.get_at(2.1, 5.8), 0.0127)
    assert compare_orders(upper_map.get_at(2.1, 5.8), 0.0499)
    assert compare_orders(lower_map.get_at(2.1, 5.8), -0.0036)

    atom = molecule.get_atom(49)
    atom_total = atom.ordermaps().total()
    atom_upper = atom.ordermaps().upper()
    atom_lower = atom.ordermaps().lower()

    assert compare_orders(atom_total.get_at(2.1, 5.8), 0.0349)
    assert compare_orders(atom_upper.get_at(2.1, 5.8), 0.0450)
    assert compare_orders(atom_lower.get_at(2.1, 5.8), 0.0272)

    bond = atom.bonds()[1]
    bond_total = bond.ordermaps().total()
    bond_upper = bond.ordermaps().upper()
    bond_lower = bond.ordermaps().lower()

    assert compare_orders(bond_total.get_at(2.1, 5.8), 0.1869)
    assert math.isnan(bond_upper.get_at(6.4, 0.0))
    assert math.isnan(bond_lower.get_at(6.4, 6.0))

    # EXTRACT CHECK
    (ext_x, ext_y, ext_vals) = bond_total.extract()
    assert len(ext_x) == 14
    assert len(ext_y) == 4
    
    for x in ext_x:
        assert math.isclose(x % 0.5, 0.0, abs_tol=1e-5)
    
    expected_y = [0.0, 2.0, 4.0, 6.0]
    for real, expected in zip(ext_y, expected_y):
        assert math.isclose(real, expected, rel_tol=1e-5)
    
    for xi, x in enumerate(ext_x):
        for yi, y in enumerate(ext_y):
            map_val = bond_total.get_at(x, y)
            ext_val = ext_vals[xi][yi]
            if math.isnan(map_val) and math.isnan(ext_val):
                continue
            assert compare_orders(map_val, ext_val)

def test_aa_order_leaflets_collect():
    analysis = gorder.Analysis(
        structure = "../tests/files/pcpepg.tpr",
        trajectory = "../tests/files/pcpepg.xtc",
        analysis_type = gorder.analysis_types.AAOrder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen"
        ),
        leaflets = gorder.leaflets.GlobalClassification("@membrane", "name P", collect = True),
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.leaflets_data().frames() == [x for x in range(1, 52)]

    pope_data = results.leaflets_data().get_molecule("POPE")
    assert len(pope_data) == 51
    for frame in pope_data:
        assert len(frame) == 131
        for (i, lipid) in enumerate(frame):
            if i < 65:
                assert lipid == 1
            else:
                assert lipid == 0
    
    popc_data = results.leaflets_data().get_molecule("POPC")
    assert len(popc_data) == 51
    for frame in popc_data:
        assert len(frame) == 128
        for (i, lipid) in enumerate(frame):
            if i < 64:
                assert lipid == 1
            else:
                assert lipid == 0
    
    popg_data = results.leaflets_data().get_molecule("POPG")
    assert len(popg_data) == 51
    for frame in popg_data:
        assert len(frame) == 15
        for (i, lipid) in enumerate(frame):
            if i < 8:
                assert lipid == 1
            else:
                assert lipid == 0

def test_aa_order_dynamic_normals_collect():
    analysis = gorder.Analysis(
        structure = "../tests/files/pcpepg.tpr",
        trajectory = "../tests/files/pcpepg.xtc",
        analysis_type = gorder.analysis_types.AAOrder(
            "@membrane and element name carbon",
            "@membrane and element name hydrogen"
        ),
        membrane_normal = gorder.membrane_normal.DynamicNormal(
            "name P", 2.0, collect = True
        ),
        geometry = gorder.geometry.Cylinder(reference = "center", radius = 2.5, orientation = "z"),
        step = 10,
        silent = True,
        overwrite = True,
    )

    results = analysis.run()

    assert results.normals_data().frames() == [1, 11, 21, 31, 41, 51]

    pope_data = results.normals_data().get_molecule("POPE")
    assert len(pope_data) == 6

    for frame in pope_data:
        assert len(frame) == 131

    assert normal_is_nan(pope_data[0][0])
    assert compare_normals(pope_data[4][2], [0.038475, 0.171717, 0.984395])
        
    
    popc_data = results.normals_data().get_molecule("POPC")
    assert len(popc_data) == 6
    for frame in popc_data:
        assert len(frame) == 128

    assert normal_is_nan(popc_data[2][-1])
    assert compare_normals(popc_data[2][4], [0.156903, 0.041018, 0.986762])
    
    popg_data = results.normals_data().get_molecule("POPG")
    assert len(popg_data) == 6
    for frame in popg_data:
        assert len(frame) == 15
    
    assert compare_normals(popg_data[5][-2], [0.069389, 0.018346, 0.997421])

def test_aa_order_scrambling_leaflets_flip():
    for (leaflets_unflipped, leaflets_flipped) in [
        (gorder.leaflets.GlobalClassification("@membrane", "name PO4", collect = True),
         gorder.leaflets.GlobalClassification("@membrane", "name PO4", collect = True, flip = True)),
        (gorder.leaflets.LocalClassification("@membrane", "name PO4", 2.5, collect = True),
         gorder.leaflets.LocalClassification("@membrane", "name PO4", 2.5, collect = True, flip = True)),
        (gorder.leaflets.IndividualClassification("name PO4", "name C4A C4B", collect = True),
         gorder.leaflets.IndividualClassification("name PO4", "name C4A C4B", collect = True, flip = True)),
        (gorder.leaflets.ClusteringClassification("name PO4", frequency = gorder.Frequency.every(10), collect = True),
         gorder.leaflets.ClusteringClassification("name PO4", frequency = gorder.Frequency.every(10), collect = True, flip = True))
    ]:
            
        analysis_unflipped = gorder.Analysis(
            structure = "../tests/files/cg.tpr",
            trajectory = "../tests/files/cg.xtc",
            analysis_type = gorder.analysis_types.CGOrder("@membrane"),
            leaflets = leaflets_unflipped,
            silent = True,
            overwrite = True,
        )

        results_unflipped = analysis_unflipped.run()

        analysis_flipped = gorder.Analysis(
            structure = "../tests/files/cg.tpr",
            trajectory = "../tests/files/cg.xtc",
            analysis_type = gorder.analysis_types.CGOrder("@membrane"),
            leaflets = leaflets_flipped,
            silent = True,
            overwrite = True,
        )

        results_flipped = analysis_flipped.run()

        # compare leaflet assignment data
        leaflets_unflipped = results_unflipped.leaflets_data().get_molecule("POPC")
        leaflets_flipped = results_flipped.leaflets_data().get_molecule("POPC")

        assert len(leaflets_unflipped) == len(leaflets_flipped)

        for (frame_unflipped, frame_flipped) in zip(leaflets_unflipped, leaflets_flipped):
            assert len(frame_unflipped) == len(frame_flipped)
            for (leaflet_unflipped, leaflet_flipped) in zip(frame_unflipped, frame_flipped):
                assert leaflet_unflipped != leaflet_flipped
        
        # compare order parameters
        order_unflipped = results_unflipped.get_molecule("POPC")
        order_flipped = results_flipped.get_molecule("POPC")

        assert len(order_unflipped.bonds()) == len(order_flipped.bonds())

        for (bond_unflipped, bond_flipped) in zip(order_unflipped.bonds(), order_flipped.bonds()):
            assert compare_orders(bond_unflipped.order().total().value(), bond_flipped.order().total().value())
            assert bond_unflipped.order().upper() is not None
            assert bond_unflipped.order().lower() is not None
            assert bond_flipped.order().upper() is not None
            assert bond_flipped.order().lower() is not None
            assert compare_orders(bond_unflipped.order().upper().value(), bond_flipped.order().lower().value())
            assert compare_orders(bond_unflipped.order().lower().value(), bond_flipped.order().upper().value())