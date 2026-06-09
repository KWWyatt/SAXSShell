from __future__ import annotations

from pathlib import Path

from saxshell.bondanalysis import BondAnalysisWorkflow, BondPairDefinition
from saxshell.exafs import artemis_gds_overview_path
from saxshell.exafs.mapping import (
    build_gds_mapping_document,
    default_absorber_element,
    discover_bondanalysis_results,
    discover_representative_structures,
    gds_registry_entries_for_stoichiometry,
    load_bondanalysis_result,
    load_structure_preview,
    scattering_path_events_from_preview,
    write_gds_mapping_file,
    write_padded_cif_from_structure,
)
from saxshell.fullrmc.project_model import ensure_rmcsetup_structure
from saxshell.fullrmc.representatives import (
    DistributionSelectionEntry,
    DistributionSelectionMetadata,
    RepresentativeSelectionEntry,
    RepresentativeSelectionMetadata,
    RepresentativeSelectionSettings,
    save_representative_selection_metadata,
)
from saxshell.saxs.project_manager import DreamBestFitSelection


def _write_xyz_structure(
    path: Path,
    atoms: list[tuple[str, float, float, float]],
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.4f} {y_coord:.4f} {z_coord:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x_coord: float,
    y_coord: float,
    z_coord: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:<4} {residue_name:>3} X"
        f"{residue_number:4d}    "
        f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
        f"  1.00  0.00          {element:>2}"
    )


def _write_pdb_structure(
    path: Path,
    atoms: list[tuple[str, str, int, float, float, float, str]],
) -> None:
    lines = [
        _pdb_atom_line(
            serial=index,
            atom_name=atom_name,
            residue_name=residue_name,
            residue_number=residue_number,
            x_coord=x_coord,
            y_coord=y_coord,
            z_coord=z_coord,
            element=element,
        )
        for index, (
            atom_name,
            residue_name,
            residue_number,
            x_coord,
            y_coord,
            z_coord,
            element,
        ) in enumerate(atoms, start=1)
    ]
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_project_representative_metadata(project_dir: Path) -> Path:
    source_path = (
        project_dir
        / "rmcsetup"
        / "representative_structures"
        / "nosolv"
        / "PbI2"
        / "PbI2__representative__frame_0001_AAA.xyz"
    )
    _write_xyz_structure(
        source_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("I", -3.2, 0.0, 0.0),
        ],
    )
    paths = ensure_rmcsetup_structure(project_dir)
    selection = DreamBestFitSelection(
        run_name="demo",
        run_relative_path="dream/demo",
    )
    distribution_entry = DistributionSelectionEntry(
        param="w0",
        structure="PbI2",
        motif="no_motif",
        selected_weight=1.0,
        vary=True,
        cluster_count=1,
        source_dir=str(source_path.parent),
        source_file=str(source_path),
        source_file_name=source_path.name,
        source_kind="representative",
        is_active=True,
    )
    metadata = RepresentativeSelectionMetadata(
        selection_mode="first_file",
        selection=selection,
        distribution_selection=DistributionSelectionMetadata(
            selection_mode="first_file",
            selection=selection,
            run_dir=str(project_dir / "dream" / "demo"),
            updated_at="2026-06-02T00:00:00",
            entries=[distribution_entry],
        ),
        settings=RepresentativeSelectionSettings(selection_mode="first_file"),
        updated_at="2026-06-02T00:00:00",
        representative_entries=[
            RepresentativeSelectionEntry(
                structure="PbI2",
                motif="no_motif",
                param="w0",
                selected_weight=1.0,
                cluster_count=1,
                source_dir=str(source_path.parent),
                source_file=str(source_path),
                source_file_name=source_path.name,
                atom_count=3,
                element_counts={"Pb": 1, "I": 2},
                source_solvent_mode="nosolv",
            )
        ],
        missing_bins=[],
        invalid_bins=[],
    )
    save_representative_selection_metadata(
        paths.representative_selection_path,
        metadata,
    )
    return source_path


def _write_bondanalysis_results(project_dir: Path, clusters_dir: Path) -> Path:
    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir(parents=True)
    _write_xyz_structure(
        pbi2_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("I", -3.2, 0.0, 0.0),
        ],
    )
    output_dir = project_dir / "analysis" / "bondanalysis" / "demo"
    workflow = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 3.5)],
        output_dir=output_dir,
        generate_preview_plots=False,
    )
    result = workflow.run()
    return result.output_dir


def test_exafs_mapping_discovers_representatives_and_bondanalysis_variables(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    representative_path = _write_project_representative_metadata(project_dir)
    bondanalysis_output = _write_bondanalysis_results(
        project_dir,
        tmp_path / "clusters",
    )

    representatives = discover_representative_structures(project_dir)
    bondanalysis_results = discover_bondanalysis_results(project_dir)

    assert representatives[0].stoichiometry == "PbI2"
    assert representatives[0].source_file == representative_path.resolve()
    assert bondanalysis_results[0].output_dir == bondanalysis_output.resolve()

    result_index = load_bondanalysis_result(bondanalysis_results[0])
    entries = gds_registry_entries_for_stoichiometry(result_index, "PbI2")

    assert entries
    assert any(entry["distribution_label"] == "Pb-I" for entry in entries)
    assert "ba_bond_all_pb_i_center" in entries[-1]["set_rows"]


def test_exafs_mapping_builds_gds_with_selected_bondanalysis_variables(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_i.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("I", -3.2, 0.0, 0.0),
        ],
    )
    registry_entries = [
        {
            "set_rows": (
                "set ba_bond_all_pb_i_center = 3.2 ; "
                "set ba_bond_all_pb_i_sigma2 = 0.001"
            )
        }
    ]

    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=3.5,
    )
    document = build_gds_mapping_document(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=3.5,
        gds_registry_entries=registry_entries,
    )
    output_path = write_gds_mapping_file(
        tmp_path / "mapped.gds",
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=3.5,
        gds_registry_entries=registry_entries,
    )
    text = output_path.read_text(encoding="utf-8")
    overview_text = artemis_gds_overview_path(output_path).read_text(
        encoding="utf-8"
    )

    assert len(preview.paths) == 2
    assert preview.paths[0].label == "Pb1-I1"
    assert preview.absorber_indices == (1,)
    assert preview.atom_labels == ("Pb1", "I1", "I2")
    assert [bond.label for bond in preview.dynamic_bonds] == [
        "R_{Pb1-I1}",
        "R_{Pb1-I2}",
    ]
    assert document.parameters[0].name == "ba_bond_all_pb_i_center"
    assert all(
        line and not line.startswith(("#", "!", "%"))
        for line in text.splitlines()
    )
    assert "set ba_bond_all_pb_i_sigma2 = 0.001" in text
    assert "Prepended bond-analysis registry parameters: 2" in overview_text
    assert "guess dr_pb_i_01 = 0" in text
    assert text.index("set ba_bond_all_pb_i_center = 3.2") < text.index(
        "guess amp = 0.9"
    )


def test_exafs_mapping_preview_excludes_hydrogen_paths(tmp_path: Path) -> None:
    structure_path = tmp_path / "pb_i_h.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("H", 1.1, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
        ],
    )

    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=3.5,
    )

    assert preview.elements == ("Pb", "I")
    assert preview.atom_indices == (1, 3)
    assert preview.atom_labels == ("Pb1", "I1")
    assert [path.label for path in preview.paths] == ["Pb1-I1"]
    assert [bond.label for bond in preview.dynamic_bonds] == ["R_{Pb1-I1}"]
    assert preview.static_bonds == ()
    assert preview.angles == ()


def test_exafs_mapping_writes_padded_cif_from_representative_pdb(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_i_cluster.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("Pb1", "PBI", 1, 1.00, 2.00, 3.00, "Pb"),
            ("I1", "PBI", 1, 4.00, 6.00, 8.00, "I"),
        ],
    )

    output_path = write_padded_cif_from_structure(
        structure_path,
        padding_angstrom=20.0,
    )
    text = output_path.read_text(encoding="utf-8")

    assert output_path == tmp_path / "pb_i_cluster_padded_20A.cif"
    assert "_cell_length_a 43" in text
    assert "_cell_length_b 44" in text
    assert "_cell_length_c 45" in text
    assert "Pb1 Pb 0.46511628 0.45454545 0.44444444 1" in text
    assert "I1 I 0.53488372 0.54545455 0.55555556 1" in text


def test_exafs_mapping_preview_labels_static_solvent_geometry(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_dmso_fragment.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 2.3, 0.0, 0.0),
            ("S", 3.0, 1.2, 0.0),
            ("C", 4.7, 1.2, 0.0),
            ("H", 5.3, 1.2, 0.0),
        ],
    )

    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=3.6,
    )

    assert {path.label for path in preview.paths} == {"Pb1-O1", "Pb1-S1"}
    assert [bond.label for bond in preview.dynamic_bonds] == ["R_{Pb1-O1}"]
    assert {bond.label for bond in preview.static_bonds} == {
        "b_{O1-S1}",
        "b_{S1-C1}",
    }
    assert {angle.atom_triplet_label for angle in preview.angles} == {
        "Pb1-O1-S1",
        "O1-S1-C1",
    }
    assert all("H1" not in bond.label for bond in preview.dynamic_bonds)
    assert all("H1" not in bond.label for bond in preview.static_bonds)
    assert all(
        "H1" not in angle.atom_triplet_label for angle in preview.angles
    )


def test_exafs_mapping_scattering_path_events_include_build_geometry(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_dmso_path_events.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 2.3, 0.0, 0.0),
            ("S", 3.0, 1.2, 0.0),
            ("C", 4.7, 1.2, 0.0),
        ],
    )

    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=3.6,
    )
    events = scattering_path_events_from_preview(preview)
    sulfur_event = next(
        event for event in events if event.scatterer_atom_label == "S1"
    )

    assert sulfur_event.label == "Pb1-O1-S1"
    assert sulfur_event.path_key == (1, 3)
    assert sulfur_event.degeneracy == 1.0
    assert sulfur_event.total_path_length_angstrom > (
        sulfur_event.effective_distance_angstrom
    )
    assert any("R_{Pb1-O1}" in bond for bond in sulfur_event.bond_lengths)
    assert any("b_{O1-S1}" in bond for bond in sulfur_event.bond_lengths)
    assert sulfur_event.angles == ("Pb1-O1-S1=120.26 deg",)
    assert sulfur_event.dihedrals == ()


def test_exafs_mapping_labels_dmf_dihedral_planes_in_pdb(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_dmf_dihedral.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("Pb1", "PBI", 1, 0.50, 0.50, 0.00, "Pb"),
            ("O1", "DMF", 1, 1.00, 0.00, 0.00, "O"),
            ("C1", "DMF", 1, 2.00, 0.00, 0.00, "C"),
            ("N1", "DMF", 1, 2.00, 1.00, 0.00, "N"),
            ("C2", "DMF", 1, 2.00, 2.00, 1.00, "C"),
            ("C3", "DMF", 1, 2.00, 2.00, -1.00, "C"),
        ],
    )

    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=2.4,
    )

    assert {
        "Pb1-O11-C11-N11",
        "O11-C11-N11-C12",
        "O11-C11-N11-C13",
    } == {dihedral.label for dihedral in preview.dihedrals}
    assert {
        "Pb1-O11-C11",
        "O11-C11-N11",
        "C11-N11-C12",
        "C11-N11-C13",
        "C12-N11-C13",
    } <= {angle.atom_triplet_label for angle in preview.angles}
    assert all(
        dihedral.plane1_indices == dihedral.atom_indices[:3]
        and dihedral.plane2_indices == dihedral.atom_indices[1:]
        for dihedral in preview.dihedrals
    )


def test_exafs_mapping_labels_dmso_degenerate_dihedrals_in_pdb(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_dmso_dihedral.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("Pb1", "PBI", 1, 0.50, 0.50, 0.00, "Pb"),
            ("O1", "DMS", 1, 1.00, 0.00, 0.00, "O"),
            ("S1", "DMS", 1, 2.00, 0.00, 0.00, "S"),
            ("C1", "DMS", 1, 2.00, 1.00, 0.80, "C"),
            ("C2", "DMS", 1, 2.00, 1.00, -0.80, "C"),
        ],
    )

    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=2.4,
    )

    assert {dihedral.label for dihedral in preview.dihedrals} == {
        "Pb1-O11-S11-C11",
        "Pb1-O11-S11-C12",
    }
    assert {"Pb1-O11-S11", "O11-S11-C11", "O11-S11-C12"} <= {
        angle.atom_triplet_label for angle in preview.angles
    }


def test_exafs_mapping_pair_cutoff_excludes_far_solvent_geometry(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_far_dmf.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("Pb1", "PBI", 1, 0.00, 0.00, 0.00, "Pb"),
            ("I1", "PBI", 1, 3.00, 0.00, 0.00, "I"),
            ("O1", "DMF", 1, 5.00, 0.00, 0.00, "O"),
            ("C1", "DMF", 1, 6.20, 0.50, 0.00, "C"),
            ("N1", "DMF", 1, 7.40, 0.50, 0.80, "N"),
        ],
    )

    filtered_preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        absorber_atom_index=1,
        max_distance_angstrom=6.0,
        pair_cutoff_distances_angstrom={("Pb", "O"): 3.5},
    )
    permissive_preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        absorber_atom_index=1,
        max_distance_angstrom=6.0,
        pair_cutoff_distances_angstrom={("Pb", "O"): 5.5},
    )
    filtered_document = build_gds_mapping_document(
        structure_path,
        absorber_element="Pb",
        absorber_atom_index=1,
        max_distance_angstrom=6.0,
        pair_cutoff_distances_angstrom={("Pb", "O"): 3.5},
    )

    assert [path.label for path in filtered_preview.paths] == ["Pb1-I1"]
    assert len(filtered_preview.dynamic_bonds) == 1
    assert filtered_preview.dynamic_bonds[0].label == "R_{Pb1-I1}"
    assert filtered_preview.static_bonds == ()
    assert filtered_preview.angles == ()
    assert filtered_preview.dihedrals == ()
    assert "Pb1-O11-C11" in {
        angle.atom_triplet_label for angle in permissive_preview.angles
    }
    assert "Pb1-O11-C11-N11" in {
        dihedral.label for dihedral in permissive_preview.dihedrals
    }
    filtered_text = filtered_document.to_text().lower()
    assert "guess dr_pb_i_01" in filtered_text
    assert "pb_o" not in filtered_text
    assert "pb_c" not in filtered_text


def test_exafs_mapping_selected_path_pairs_filter_generic_gds(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_i_o_selection.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.1, 0.0, 0.0),
            ("O", 2.3, 0.0, 0.0),
        ],
    )

    document = build_gds_mapping_document(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=3.5,
        included_path_pairs=((1, 2),),
    )
    text = document.to_text().lower()

    assert "guess dr_pb_i_01" in text
    assert "pb_o" not in text


def test_exafs_mapping_solvent_pair_cutoff_is_absorber_dependent(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb2_absorber_dependent_dmso.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("Pb1", "PBI", 1, 0.00, 0.00, 0.00, "Pb"),
            ("Pb2", "PBI", 1, 10.00, 0.00, 0.00, "Pb"),
            ("O1", "DMS", 2, 12.30, 0.00, 0.00, "O"),
            ("S1", "DMS", 2, 13.50, 0.60, 0.00, "S"),
            ("C1", "DMS", 2, 14.60, 1.30, 0.00, "C"),
        ],
    )

    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=15.0,
        pair_cutoff_distances_angstrom={("Pb", "O"): 3.36},
    )

    path_labels = {path.label for path in preview.paths}
    assert "Pb1-O11" not in path_labels
    assert "Pb1-S11" not in path_labels
    assert "Pb2-O11" in path_labels
    assert "Pb2-S11" in path_labels


def test_exafs_mapping_pb_dmso_build_uses_pair_oxygen_cutoff(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_two_close_one_far_dmso.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.1, 0.0, 0.0),
            ("O", 2.3, 0.0, 0.0),
            ("S", 3.0, 1.2, 0.0),
            ("O", 0.0, 2.4, 0.0),
            ("S", 1.2, 3.0, 0.0),
            ("O", 5.2, 0.0, 0.0),
            ("S", 6.0, 1.2, 0.0),
        ],
    )

    try:
        build_gds_mapping_document(
            structure_path,
            mode="pb_dmso",
            absorber_atom_index=1,
            max_distance_angstrom=6.0,
            pair_cutoff_distances_angstrom={("Pb", "O"): 3.5},
        )
    except ValueError as exc:
        message = str(exc)
    else:
        message = ""

    assert "Requested 3 oxygen atoms, but only 2 were found" in message
    document = build_gds_mapping_document(
        structure_path,
        mode="pb_dmso",
        absorber_atom_index=1,
        max_distance_angstrom=6.0,
        pair_cutoff_distances_angstrom={("Pb", "O"): 5.5},
    )
    assert "set reff_pbo_3 = 5.2" in document.to_text()


def test_exafs_mapping_pb_dmso_selected_paths_omit_unchecked_terminal(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_three_dmso_path_selection.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.1, 0.0, 0.0),
            ("O", 2.3, 0.0, 0.0),
            ("S", 3.0, 1.2, 0.0),
            ("O", 0.0, 2.4, 0.0),
            ("S", 1.2, 3.0, 0.0),
            ("O", 0.0, 0.0, 2.5),
            ("S", 1.2, 0.0, 3.1),
        ],
    )

    document = build_gds_mapping_document(
        structure_path,
        mode="pb_dmso",
        absorber_atom_index=1,
        max_distance_angstrom=6.0,
        pair_cutoff_distances_angstrom={("Pb", "O"): 3.5},
        included_path_pairs=((1, 3),),
    )
    text = document.to_text().lower()

    assert "reff_pbo" not in text
    assert "reff_pbs" not in text
    assert "cn_s" not in text
    assert "delr_s" not in text
    assert [hint.shell_label for hint in document.path_hints] == ["pb_o_1"]


def test_exafs_mapping_pb_dmf_template_builds_geometric_paths(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_three_dmf.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("O", 2.5, 0.0, 0.0),
            ("C", 3.0, 1.0, 0.0),
            ("N", 3.2, 2.0, 0.6),
            ("O", 0.0, 2.6, 0.0),
            ("C", -1.0, 3.1, 0.0),
            ("N", -2.0, 3.2, 0.6),
            ("O", 0.0, 0.0, 2.7),
            ("C", 0.8, 0.0, 3.2),
            ("N", 1.5, 0.5, 3.6),
        ],
    )

    document = build_gds_mapping_document(
        structure_path,
        mode="pb_dmf",
        absorber_atom_index=1,
        max_distance_angstrom=6.0,
        pair_cutoff_distances_angstrom={("Pb", "O"): 3.5},
    )
    text = document.to_text()

    assert "def cn_c1 = cn_o1" in text
    assert "def cn_n1 = cn_o1" in text
    assert "def delr_c1 = sqrt(" in text
    assert "def delr_n1 = sqrt(" in text


def test_exafs_mapping_pb_dmf_selected_paths_omit_unchecked_terminals(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_three_dmf_selection.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("O", 2.5, 0.0, 0.0),
            ("C", 3.0, 1.0, 0.0),
            ("N", 3.2, 2.0, 0.6),
            ("O", 0.0, 2.6, 0.0),
            ("C", -1.0, 3.1, 0.0),
            ("N", -2.0, 3.2, 0.6),
            ("O", 0.0, 0.0, 2.7),
            ("C", 0.8, 0.0, 3.2),
            ("N", 1.5, 0.5, 3.6),
        ],
    )

    document = build_gds_mapping_document(
        structure_path,
        mode="pb_dmf",
        absorber_atom_index=1,
        max_distance_angstrom=6.0,
        pair_cutoff_distances_angstrom={("Pb", "O"): 3.5},
        included_path_pairs=((1, 3),),
    )
    text = document.to_text().lower()

    assert "reff_pbo" not in text
    assert "reff_pbc" not in text
    assert "reff_pbn" not in text
    assert "cn_c" not in text
    assert "cn_n" not in text
    assert [hint.shell_label for hint in document.path_hints] == ["pb_o_1"]


def test_exafs_mapping_plot_annotation_toggles_filter_label_families(
    tmp_path: Path,
) -> None:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from saxshell.exafs.ui.main_window import (
        EXAFSGDSMappingMainWindow,
        _file_manager_reveal_command,
        _template_mode_from_context_values,
    )

    structure_path = tmp_path / "pb_dmf_plot_labels.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("Pb1", "PBI", 1, 0.50, 0.50, 0.00, "Pb"),
            ("O1", "DMF", 1, 1.00, 0.00, 0.00, "O"),
            ("C1", "DMF", 1, 2.00, 0.00, 0.00, "C"),
            ("N1", "DMF", 1, 2.00, 1.00, 0.00, "N"),
            ("C2", "DMF", 1, 2.00, 2.00, 1.00, "C"),
            ("C3", "DMF", 1, 2.00, 2.00, -1.00, "C"),
        ],
    )
    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=2.4,
    )
    app = QApplication.instance() or QApplication([])
    window = EXAFSGDSMappingMainWindow()
    try:
        assert (
            _template_mode_from_context_values("079_PbI2_DMF_0p8M_RT")
            == "pb_dmf"
        )
        assert (
            _template_mode_from_context_values("079_PbI2_DMSO_0p8M_RT")
            == "pb_dmso"
        )
        assert _template_mode_from_context_values("PbI2_notdmf") is None
        window._apply_template_mode_from_context(
            tmp_path / "079_PbI2_DMF_0p8M_RT"
        )
        assert window.mode_combo.currentData() == "pb_dmf"
        window._apply_template_mode_from_context(
            tmp_path / "079_PbI2_DMSO_0p8M_RT"
        )
        assert window.mode_combo.currentData() == "pb_dmso"
        assert window.reveal_structure_button.text() == "Reveal"
        assert window.generate_cif_button.text() == "Generate CIF"
        assert window.cif_padding_spin.value() == 20.0
        assert _file_manager_reveal_command(
            structure_path,
            platform="darwin",
        ) == ["open", "-R", str(structure_path.resolve())]
        windows_command = _file_manager_reveal_command(
            structure_path,
            platform="win32",
        )
        assert windows_command[0] == "explorer"
        assert windows_command[1].startswith("/select,")
        assert _file_manager_reveal_command(
            structure_path,
            platform="linux",
        ) == ["xdg-open", str(structure_path.resolve().parent)]
        assert isinstance(window.path3d_toolbar, NavigationToolbar2QT)
        assert isinstance(window.path2d_toolbar, NavigationToolbar2QT)
        assert {"Pan", "Zoom"} <= _toolbar_action_names(window.path3d_toolbar)
        assert {"Pan", "Zoom"} <= _toolbar_action_names(window.path2d_toolbar)
        assert window._pair_cutoff_distances() == {
            ("Pb", "I"): 3.36,
            ("Pb", "O"): 3.36,
        }
        window._refresh_scattering_path_table(preview)
        assert window.scattering_path_table.rowCount() > len(preview.paths)
        path_table_text = _table_text(window.scattering_path_table)
        assert "Solvent molecule DMF X1" in path_table_text
        assert "Pb1-O11-C11-N11" in path_table_text
        assert "Pb1-O11-C11" in path_table_text
        assert "O11-C11-N11" in path_table_text

        window.path_filter_column_combo.setCurrentIndex(
            window.path_filter_column_combo.findData("scatterer")
        )
        window.path_filter_edit.setText("N11")
        filtered_path_keys = _path_table_path_keys(
            window.scattering_path_table
        )
        assert len(filtered_path_keys) == 1
        assert "Pb1-O11-C11-N11" in _table_text(window.scattering_path_table)
        assert "Showing 1 of" in window.path_filter_status_label.text()
        window.path_filter_edit.clear()

        window.path_group_combo.setCurrentIndex(
            window.path_group_combo.findData("absorber")
        )
        assert window.path_group_combo.currentData() == "absorber"
        window.reset_path_grouping_button.click()
        assert window.path_group_combo.currentData() == "solvent_molecule"

        first_path_row = _first_path_table_row(window.scattering_path_table)
        first_use_item = window.scattering_path_table.item(first_path_row, 0)
        assert first_use_item is not None
        first_key = first_use_item.data(Qt.ItemDataRole.UserRole)
        first_use_item.setCheckState(Qt.CheckState.Unchecked)
        selected_pairs = window._selected_scattering_path_pairs()
        assert selected_pairs is not None
        assert first_key not in selected_pairs

        window._active_preview = preview
        window._draw_path_plots(preview)
        default_text = _exafs_plot_text(window)
        assert "R_{Pb1-O11}" in default_text
        assert " deg" not in default_text
        assert "phi" not in default_text

        window.show_bond_angles_box.setChecked(True)
        window.show_dihedral_angles_box.setChecked(True)
        window._draw_path_plots(preview)
        expanded_text = _exafs_plot_text(window)

        assert " deg" in expanded_text
        assert "phi" in expanded_text

        window.show_bond_distances_box.setChecked(False)
        window.show_bond_angles_box.setChecked(False)
        window.show_dihedral_angles_box.setChecked(False)
        window._draw_path_plots(preview)
        hidden_text = _exafs_plot_text(window)

        assert "R_{" not in hidden_text
        assert "b_{" not in hidden_text
        assert " deg" not in hidden_text
        assert "phi" not in hidden_text
        assert "Pb1" in hidden_text
    finally:
        window.close()
        app.processEvents()


def _toolbar_action_names(toolbar: object) -> set[str]:
    return {
        action.text()
        for action in getattr(toolbar, "actions")()
        if action.text()
    }


def _exafs_plot_text(window: object) -> str:
    figures = (
        getattr(window, "path3d_figure"),
        getattr(window, "path2d_figure"),
    )
    return "\n".join(
        text.get_text()
        for figure in figures
        for axis in figure.axes
        for text in axis.texts
    )


def _table_text(table: object) -> str:
    return "\n".join(
        item.text()
        for row in range(table.rowCount())
        for column in range(table.columnCount())
        for item in (table.item(row, column),)
        if item is not None
    )


def _path_table_path_keys(table: object) -> list[tuple[int, int]]:
    from PySide6.QtCore import Qt

    keys = []
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is None:
            continue
        key = item.data(Qt.ItemDataRole.UserRole)
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and all(isinstance(value, int) for value in key)
        ):
            keys.append((int(key[0]), int(key[1])))
    return keys


def _first_path_table_row(table: object) -> int:
    from PySide6.QtCore import Qt

    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is None:
            continue
        key = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(key, tuple) and len(key) == 2:
            return row
    raise AssertionError("No path row found in scattering path table.")


def test_exafs_mapping_labels_solvent_atoms_by_molecule_in_pdb(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_dmf2_i2.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("N1", "DMF", 1, 4.70, 0.00, 0.00, "N"),
            ("C1", "DMF", 1, 3.50, 0.00, 0.00, "C"),
            ("O1", "DMF", 1, 2.30, 0.00, 0.00, "O"),
            ("C2", "DMF", 1, 5.70, 0.80, 0.00, "C"),
            ("C3", "DMF", 1, 5.70, -0.80, 0.00, "C"),
            ("N1", "DMF", 2, 0.00, 4.70, 0.00, "N"),
            ("C1", "DMF", 2, 0.00, 3.50, 0.00, "C"),
            ("O1", "DMF", 2, 0.00, 2.30, 0.00, "O"),
            ("C2", "DMF", 2, 0.80, 5.70, 0.00, "C"),
            ("C3", "DMF", 2, -0.80, 5.70, 0.00, "C"),
            ("PB2", "PBI", 3, 0.00, 0.00, 0.00, "Pb"),
            ("I3", "PBI", 4, 3.00, 0.00, 0.00, "I"),
            ("I4", "PBI", 5, -3.00, 0.00, 0.00, "I"),
        ],
    )

    preview = load_structure_preview(
        structure_path,
        absorber_element="Pb",
        max_distance_angstrom=3.1,
    )

    assert {
        "C11",
        "C12",
        "C13",
        "C21",
        "C22",
        "C23",
        "N11",
        "N21",
        "O11",
        "O21",
        "Pb1",
        "I1",
        "I2",
    } == set(preview.atom_labels)
    assert {path.label for path in preview.paths} == {
        "Pb1-O11",
        "Pb1-O21",
        "Pb1-I1",
        "Pb1-I2",
    }
    assert {
        "b_{C11-O11}",
        "b_{N11-C11}",
        "b_{N11-C12}",
        "b_{N11-C13}",
        "b_{C21-O21}",
        "b_{N21-C21}",
        "b_{N21-C22}",
        "b_{N21-C23}",
    } <= {bond.label for bond in preview.static_bonds}
    assert "Pb1-O11-C11" in {
        angle.atom_triplet_label for angle in preview.angles
    }


def test_exafs_mapping_defaults_to_pb_and_allows_absorber_override(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "sn_pb_i.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Sn", 0.0, 0.0, 0.0),
            ("Pb", 3.0, 0.0, 0.0),
            ("I", 3.0, 3.2, 0.0),
        ],
    )

    default_preview = load_structure_preview(
        structure_path,
        max_distance_angstrom=3.5,
    )
    override_preview = load_structure_preview(
        structure_path,
        absorber_element="Sn",
        max_distance_angstrom=4.5,
    )
    override_document = build_gds_mapping_document(
        structure_path,
        absorber_element="Sn",
        max_distance_angstrom=4.5,
    )

    assert default_absorber_element(structure_path) == "Pb"
    assert default_preview.absorber_indices == (2,)
    assert override_preview.absorber_indices == (1,)
    assert {path.label for path in default_preview.paths} == {
        "Pb1-Sn1",
        "Pb1-I1",
    }
    assert {path.label for path in override_preview.paths} == {
        "Sn1-Pb1",
        "Sn1-I1",
    }
    assert all(
        shell.absorber_element == "Sn" for shell in override_document.shells
    )
