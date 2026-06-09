from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisWorkflow,
    BondAnalyzer,
    BondPairDefinition,
    CoordinationNumberDefinition,
    DihedralQuartetDefinition,
)
from saxshell.bondanalysis.cli import main as bondanalysis_main
from saxshell.saxshell import main as saxshell_main


def _write_xyz_cluster(
    path: Path,
    *,
    atoms: list[tuple[str, float, float, float]],
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n")


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


def _write_pdb_cluster(
    path: Path,
    *,
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
    path.write_text("\n".join(lines + ["END"]) + "\n")


def _build_sample_clusters_dir(base_dir: Path) -> Path:
    clusters_dir = base_dir / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        atoms=[
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("C", 1.0, 1.0, 0.0),
            ("N", 1.0, 1.0, 1.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        atoms=[
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )
    return clusters_dir


def test_bondanalysis_skips_single_atom_structure_files(tmp_path):
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    single_atom_dir = clusters_dir / "Cs"
    pbi2_dir.mkdir(parents=True)
    single_atom_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "single_pb.xyz",
        atoms=[("Pb", 0.0, 0.0, 0.0)],
    )
    _write_xyz_cluster(
        pbi2_dir / "pair_xyz.xyz",
        atoms=[
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    _write_pdb_cluster(
        pbi2_dir / "single_i.pdb",
        atoms=[("I1", "ION", 1, 0.0, 0.0, 0.0, "I")],
    )
    _write_pdb_cluster(
        pbi2_dir / "pair_pdb.pdb",
        atoms=[
            ("PB1", "PBI", 1, 0.0, 0.0, 0.0, "Pb"),
            ("I1", "PBI", 1, 2.0, 0.0, 0.0, "I"),
        ],
    )
    _write_xyz_cluster(
        single_atom_dir / "single_cs.xyz",
        atoms=[("Cs", 0.0, 0.0, 0.0)],
    )

    analyzer = BondAnalyzer()

    assert [path.name for path in analyzer.structure_files(pbi2_dir)] == [
        "pair_pdb.pdb",
        "pair_xyz.xyz",
    ]
    assert analyzer.structure_files(single_atom_dir) == []

    workflow = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 3.0)],
        angle_triplets=[
            AngleTripletDefinition("Pb", "I", "I", 3.0, 3.0),
        ],
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        output_dir=tmp_path / "bondanalysis_single_atom_skip",
    )

    summary = workflow.inspect()
    result = workflow.run()

    assert summary["cluster_type_count"] == 1
    assert summary["total_structure_files"] == 2
    assert result.selected_cluster_types == ("PbI2",)
    assert result.total_structure_files == 2
    assert result.cluster_results[0].structure_count == 2


def _read_histogram_csv(
    path: Path,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    metadata: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header: list[str] | None = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                if len(row) > 1:
                    metadata[row[0].removeprefix("# ").strip()] = row[1]
                continue
            header = row
            break
        if header is None:
            return metadata, rows
        for row in reader:
            rows.append(dict(zip(header, row)))
    return metadata, rows


def _strip_gds_metadata_from_histogram_csv(path: Path) -> None:
    stripped_lines = [
        line
        for line in path.read_text().splitlines()
        if not line.startswith("# gds_") and not line.startswith("# circular_")
    ]
    path.write_text("\n".join(stripped_lines) + "\n")


def test_bondanalysis_workflow_supports_notebook_style_usage(tmp_path):
    clusters_dir = _build_sample_clusters_dir(tmp_path)
    workflow = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[
            BondPairDefinition("Pb", "I", 2.5),
            BondPairDefinition("Pb", "O", 2.5),
        ],
        angle_triplets=[
            AngleTripletDefinition("Pb", "I", "I", 2.5, 2.5),
        ],
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        coordination_numbers=[
            CoordinationNumberDefinition("Pb", "I", 2.5),
        ],
    )

    summary = workflow.inspect()
    progress_messages: list[str] = []
    result = workflow.run(
        progress_callback=(
            lambda _processed, _total, message: progress_messages.append(
                message
            )
        ),
    )

    assert summary["cluster_type_count"] == 2
    assert summary["total_structure_files"] == 2
    assert result.output_dir == tmp_path / "bondanalysis_clusters_splitxyz0001"
    assert result.total_structure_files == 2
    assert result.results_index_path.exists()
    assert (
        result.output_dir / "cluster_types" / "PbI2" / "Pb_I_distribution.csv"
    ).exists()
    assert (
        result.output_dir / "cluster_types" / "PbI2" / "Pb_I_distribution.npy"
    ).exists()
    assert (
        result.output_dir / "cluster_types" / "PbI2" / "Pb_I_histogram.png"
    ).exists()
    assert (
        result.output_dir / "cluster_types" / "PbI2" / "Pb_I_histogram.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_I_distribution.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_I_distribution.npy"
    ).exists()
    all_histogram_csv = (
        result.output_dir / "all_clusters" / "Pb_I_histogram.csv"
    )
    assert all_histogram_csv.exists()
    angle_histogram_csv = (
        result.output_dir / "all_clusters" / "Pb_I_I_histogram.csv"
    )
    assert angle_histogram_csv.exists()
    dihedral_histogram_csv = (
        result.output_dir / "all_clusters" / "Pb_O_C_N_histogram.csv"
    )
    assert dihedral_histogram_csv.exists()
    coordination_histogram_csv = (
        result.output_dir / "all_clusters" / "CN_Pb_I_histogram.csv"
    )
    assert coordination_histogram_csv.exists()
    metadata, histogram_rows = _read_histogram_csv(all_histogram_csv)
    assert metadata["distribution_type"] == "bond"
    assert metadata["distribution_label"] == "Pb-I"
    assert metadata["scope"] == "All selected clusters"
    assert metadata["value_label"] == "Distance (A)"
    assert int(metadata["point_count"]) == 2
    assert float(metadata["mean"]) == 2.0
    assert float(metadata["median"]) == 2.0
    assert float(metadata["sigma"]) == 0.0
    assert float(metadata["sample_sigma"]) == 0.0
    assert float(metadata["gds_center_angstrom"]) == 2.0
    assert float(metadata["gds_sigma_angstrom"]) == 0.0
    assert float(metadata["gds_sigma2_angstrom_squared"]) == 0.0
    assert metadata["gds_center_variable"] == "ba_bond_all_pb_i_center"
    assert metadata["gds_sigma_variable"] == "ba_bond_all_pb_i_sigma"
    assert metadata["gds_variance_variable"] == "ba_bond_all_pb_i_sigma2"
    assert metadata["gds_center_variable_unit"] == "angstrom"
    assert metadata["gds_center_set"] == "set ba_bond_all_pb_i_center = 2"
    angle_metadata, _angle_rows = _read_histogram_csv(angle_histogram_csv)
    assert angle_metadata["distribution_type"] == "angle"
    assert "sigma" in angle_metadata
    assert float(angle_metadata["gds_center_degrees"]) == pytest.approx(90.0)
    assert float(angle_metadata["gds_center_radians"]) == pytest.approx(
        math.pi / 2
    )
    assert float(angle_metadata["gds_variance_radians_squared"]) == 0.0
    assert (
        angle_metadata["gds_center_variable"] == "ba_angle_all_i_pb_i_center"
    )
    assert angle_metadata["gds_center_variable_unit"] == "radians"
    assert (
        angle_metadata["gds_center_degrees_variable"]
        == "ba_angle_all_i_pb_i_center_degrees"
    )
    dihedral_metadata, dihedral_rows = _read_histogram_csv(
        dihedral_histogram_csv
    )
    assert dihedral_metadata["distribution_type"] == "dihedral"
    assert dihedral_metadata["distribution_label"] == "Pb-O-C-N"
    assert int(dihedral_metadata["point_count"]) == 1
    assert "gds_center_degrees" in dihedral_metadata
    assert "gds_sigma_degrees" in dihedral_metadata
    assert "gds_center_radians" in dihedral_metadata
    assert "gds_sigma_radians" in dihedral_metadata
    assert "gds_variance_radians_squared" in dihedral_metadata
    assert (
        dihedral_metadata["gds_center_variable"]
        == "ba_dihedral_all_pb_o_c_n_center"
    )
    results_index_payload = json.loads(result.results_index_path.read_text())
    gds_registry = results_index_payload["gds_variable_registry"]
    assert any(
        entry["distribution_type"] == "bond"
        and entry["distribution_label"] == "Pb-I"
        and entry["scope"] == "All selected clusters"
        and entry["gds_variable_prefix"] == "ba_bond_all_pb_i"
        for entry in gds_registry
    )
    angle_registry = next(
        entry
        for entry in gds_registry
        if entry["distribution_type"] == "angle"
        and entry["distribution_label"] == "I-Pb-I"
        and entry["scope"] == "All selected clusters"
    )
    assert "set ba_angle_all_i_pb_i_center =" in angle_registry["set_rows"]
    coordination_metadata, coordination_rows = _read_histogram_csv(
        coordination_histogram_csv
    )
    assert coordination_metadata["distribution_type"] == "coordination"
    assert coordination_metadata["distribution_label"] == "CN Pb-I"
    assert coordination_metadata["scope"] == "All selected clusters"
    assert coordination_metadata["value_label"] == "Coordination Number"
    assert coordination_metadata["center_atom"] == "Pb"
    assert coordination_metadata["atom_of_interest"] == "I"
    assert float(coordination_metadata["cutoff_angstrom"]) == 2.5
    assert int(coordination_metadata["point_count"]) == 2
    assert float(coordination_metadata["mean"]) == 1.0
    assert float(coordination_metadata["median"]) == 1.0
    assert float(coordination_metadata["mode"]) == 0.0
    assert float(coordination_metadata["sigma"]) == 1.0
    assert sum(int(row["count"]) for row in coordination_rows) == 2
    assert sum(int(row["count"]) for row in histogram_rows) == 2
    assert {"bin_left", "bin_right", "bin_center", "count", "density"} <= set(
        histogram_rows[0]
    )
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.png"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.csv"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.npy"
    ).exists()
    assert (
        result.output_dir
        / "cluster_types"
        / "PbI2"
        / "CN_Pb_I_coordination.csv"
    ).exists()
    assert (
        result.output_dir
        / "cluster_types"
        / "PbI2"
        / "CN_Pb_I_coordination.npy"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "CN_Pb_I_coordination.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "CN_Pb_I_coordination.npy"
    ).exists()
    assert (
        result.output_dir / "cluster_types" / "PbI2" / "Pb_O_C_N_dihedrals.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_O_C_N_dihedrals.npy"
    ).exists()
    assert sum(int(row["count"]) for row in dihedral_rows) == 1
    assert (
        result.output_dir / "comparisons" / "CN_Pb_I_cluster_type_overlay.csv"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "CN_Pb_I_cluster_type_overlay.npy"
    ).exists()
    assert "Checking for matching stored results." in progress_messages
    assert any(
        message.startswith("Processing PbI2: ")
        and "structures" in message
        and "cached" in message
        and "measured" in message
        for message in progress_messages
    )
    assert not any(
        "frame_0000_AAA.xyz" in message for message in progress_messages
    )
    assert any(
        message == "Writing all-cluster distributions."
        for message in progress_messages
    )


def test_bondanalysis_workflow_reuses_exact_matching_results(tmp_path):
    clusters_dir = _build_sample_clusters_dir(tmp_path)
    output_dir = tmp_path / "bondanalysis_cached"
    workflow = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 2.5)],
        angle_triplets=[
            AngleTripletDefinition("Pb", "I", "I", 2.5, 2.5),
        ],
        output_dir=output_dir,
    )

    first_result = workflow.run()
    payload = json.loads(first_result.results_index_path.read_text())
    payload["test_sentinel"] = "preserve-if-reused"
    first_result.results_index_path.write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    second_result = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 2.5)],
        angle_triplets=[
            AngleTripletDefinition("Pb", "I", "I", 2.5, 2.5),
        ],
        output_dir=output_dir,
    ).run()
    reused_payload = json.loads(second_result.results_index_path.read_text())

    assert first_result.analysis_signature
    assert second_result.reused_existing_result
    assert second_result.analysis_signature == first_result.analysis_signature
    assert second_result.output_dir == output_dir
    assert reused_payload["test_sentinel"] == "preserve-if-reused"


def test_bondanalysis_workflow_can_cancel_at_safe_checkpoint(tmp_path):
    clusters_dir = _build_sample_clusters_dir(tmp_path)
    workflow = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 2.5)],
        output_dir=tmp_path / "bondanalysis_cancel",
    )
    progress_messages: list[str] = []

    with pytest.raises(InterruptedError, match="canceled by user"):
        workflow.run(
            progress_callback=(
                lambda _processed, _total, message: progress_messages.append(
                    message
                )
            ),
            cancel_callback=lambda: True,
        )

    assert progress_messages == []


def test_bondanalysis_workflow_backfills_gds_registry_on_cached_run(tmp_path):
    clusters_dir = _build_sample_clusters_dir(tmp_path)
    output_dir = tmp_path / "bondanalysis_gds_backfill"
    first_result = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 2.5)],
        angle_triplets=[
            AngleTripletDefinition("Pb", "I", "I", 2.5, 2.5),
        ],
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        output_dir=output_dir,
    ).run()
    histogram_path = output_dir / "all_clusters" / "Pb_I_histogram.csv"
    _strip_gds_metadata_from_histogram_csv(histogram_path)
    payload = json.loads(first_result.results_index_path.read_text())
    payload["gds_variable_registry"] = []
    first_result.results_index_path.write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    second_result = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 2.5)],
        angle_triplets=[
            AngleTripletDefinition("Pb", "I", "I", 2.5, 2.5),
        ],
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        output_dir=output_dir,
    ).run()
    metadata, _rows = _read_histogram_csv(histogram_path)
    updated_payload = json.loads(second_result.results_index_path.read_text())

    assert second_result.reused_existing_result
    assert metadata["gds_center_variable"] == "ba_bond_all_pb_i_center"
    assert any(
        entry["distribution_type"] == "dihedral"
        and entry["distribution_label"] == "Pb-O-C-N"
        and entry["scope"] == "All selected clusters"
        for entry in updated_payload["gds_variable_registry"]
    )


def test_bondanalysis_workflow_backfills_legacy_index_with_no_signature(
    tmp_path,
):
    clusters_dir = _build_sample_clusters_dir(tmp_path)
    output_dir = tmp_path / "bondanalysis_legacy_backfill"
    first_result = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 2.5)],
        output_dir=output_dir,
    ).run()
    histogram_path = output_dir / "all_clusters" / "Pb_I_histogram.csv"
    _strip_gds_metadata_from_histogram_csv(histogram_path)
    payload = json.loads(first_result.results_index_path.read_text())
    payload.pop("analysis_signature_version", None)
    payload.pop("analysis_signature", None)
    payload.pop("analysis_signature_payload", None)
    payload["gds_variable_registry"] = []
    first_result.results_index_path.write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    second_result = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 2.5)],
        output_dir=output_dir,
    ).run()
    metadata, _rows = _read_histogram_csv(histogram_path)
    updated_payload = json.loads(second_result.results_index_path.read_text())

    assert second_result.reused_existing_result
    assert second_result.analysis_signature
    assert updated_payload["analysis_signature"] == (
        second_result.analysis_signature
    )
    assert metadata["gds_sigma_variable"] == "ba_bond_all_pb_i_sigma"
    registry = updated_payload["gds_variable_registry"]
    assert registry[0]["distribution_type"] == "bond"


def test_bondanalysis_workflow_dihedral_only_run_does_not_reuse_bond_run(
    tmp_path,
):
    clusters_dir = _build_sample_clusters_dir(tmp_path)
    output_dir = tmp_path / "bondanalysis_switch_modes"
    BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 2.5)],
        output_dir=output_dir,
    ).run()

    dihedral_result = BondAnalysisWorkflow(
        clusters_dir,
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        output_dir=output_dir,
    ).run()
    payload = json.loads(dihedral_result.results_index_path.read_text())

    assert not dihedral_result.reused_existing_result
    assert payload["bond_pairs"] == []
    assert len(payload["dihedral_quartets"]) == 1
    assert (
        dihedral_result.output_dir / "all_clusters" / "Pb_O_C_N_histogram.csv"
    ).exists()
    assert (
        sum(
            sum(result.dihedral_value_counts.values())
            for result in dihedral_result.cluster_results
        )
        == 1
    )


def test_bondanalysis_workflow_writes_empty_dihedral_histogram(tmp_path):
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbo_dir = clusters_dir / "PbO"
    pbo_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbo_dir / "frame_0000_AAA.xyz",
        atoms=[
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("C", 1.0, 1.0, 0.0),
        ],
    )
    result = BondAnalysisWorkflow(
        clusters_dir,
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        output_dir=tmp_path / "bondanalysis_empty_dihedral",
        generate_preview_plots=False,
    ).run()
    all_histogram_path = (
        result.output_dir / "all_clusters" / "Pb_O_C_N_histogram.csv"
    )
    cluster_histogram_path = (
        result.output_dir / "cluster_types" / "PbO" / "Pb_O_C_N_histogram.csv"
    )
    metadata, histogram_rows = _read_histogram_csv(all_histogram_path)
    payload = json.loads(result.results_index_path.read_text())

    assert all_histogram_path.exists()
    assert cluster_histogram_path.exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_O_C_N_dihedrals.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_O_C_N_dihedrals.npy"
    ).exists()
    assert result.cluster_results[0].dihedral_value_counts == {"Pb-O-C-N": 0}
    assert metadata["distribution_type"] == "dihedral"
    assert int(metadata["point_count"]) == 0
    assert metadata["gds_center_variable"] == (
        "ba_dihedral_all_pb_o_c_n_center"
    )
    assert sum(int(row["count"]) for row in histogram_rows) == 0
    assert any(
        entry["distribution_type"] == "dihedral"
        and entry["distribution_label"] == "Pb-O-C-N"
        and entry["scope"] == "All selected clusters"
        for entry in payload["gds_variable_registry"]
    )


def test_bondanalysis_workflow_backfills_empty_dihedral_histogram_on_cached_run(
    tmp_path,
):
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbo_dir = clusters_dir / "PbO"
    pbo_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbo_dir / "frame_0000_AAA.xyz",
        atoms=[
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("C", 1.0, 1.0, 0.0),
        ],
    )
    output_dir = tmp_path / "bondanalysis_cached_empty_dihedral"
    first_result = BondAnalysisWorkflow(
        clusters_dir,
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        output_dir=output_dir,
        generate_preview_plots=False,
    ).run()
    all_histogram_path = output_dir / "all_clusters" / "Pb_O_C_N_histogram.csv"
    cluster_histogram_path = (
        output_dir / "cluster_types" / "PbO" / "Pb_O_C_N_histogram.csv"
    )
    all_histogram_path.unlink()
    cluster_histogram_path.unlink()
    payload = json.loads(first_result.results_index_path.read_text())
    payload["gds_variable_registry"] = []
    first_result.results_index_path.write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    second_result = BondAnalysisWorkflow(
        clusters_dir,
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        output_dir=output_dir,
        generate_preview_plots=False,
    ).run()
    metadata, histogram_rows = _read_histogram_csv(all_histogram_path)
    updated_payload = json.loads(second_result.results_index_path.read_text())

    assert second_result.reused_existing_result
    assert all_histogram_path.exists()
    assert cluster_histogram_path.exists()
    assert int(metadata["point_count"]) == 0
    assert sum(int(row["count"]) for row in histogram_rows) == 0
    assert any(
        entry["distribution_type"] == "dihedral"
        and entry["distribution_label"] == "Pb-O-C-N"
        and entry["scope"] == "All selected clusters"
        for entry in updated_payload["gds_variable_registry"]
    )


def test_bondanalysis_cli_run_writes_expected_outputs(tmp_path, capsys):
    clusters_dir = _build_sample_clusters_dir(tmp_path)

    exit_code = bondanalysis_main(
        [
            "run",
            str(clusters_dir),
            "--cluster-type",
            "PbI2",
            "--bond-pair",
            "Pb:I:2.5",
            "--angle-triplet",
            "Pb:I:I:2.5:2.5",
            "--dihedral",
            "Pb:O:C:N:1.5:1.5:1.5",
            "--coordination-number",
            "Pb:I:2.5",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "bondanalysis_clusters_splitxyz0001"

    assert exit_code == 0
    assert f"Output directory: {output_dir}" in captured.out
    assert "Selected cluster types: PbI2" in captured.out
    assert "Structure files processed: 1" in captured.out
    assert "Results index file:" in captured.out
    assert "1 dihedral values" in captured.out
    assert "1 coordination values" in captured.out
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_distribution.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_distribution.npy"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_I_angles.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_I_angles.npy"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_O_C_N_dihedrals.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_O_C_N_dihedrals.npy"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_histogram.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_I_histogram.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_O_C_N_histogram.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "CN_Pb_I_coordination.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "CN_Pb_I_histogram.csv"
    ).exists()


def test_histogram_csv_metadata_includes_nonzero_sigma(tmp_path):
    histogram_path = tmp_path / "distribution_histogram.csv"

    BondAnalysisWorkflow._write_histogram_csv(
        histogram_path,
        [1.0, 3.0],
        distribution_type="bond",
        distribution_label="Pb-I",
        scope_label="All selected clusters",
        value_label="Distance (A)",
        bins=2,
    )

    metadata, _histogram_rows = _read_histogram_csv(histogram_path)

    assert float(metadata["sigma"]) == 1.0
    assert float(metadata["standard_deviation"]) == 1.0
    assert float(metadata["sample_sigma"]) == pytest.approx(2**0.5)
    assert float(metadata["gds_center_angstrom"]) == 2.0
    assert float(metadata["gds_sigma_angstrom"]) == 1.0
    assert float(metadata["gds_sigma2_angstrom_squared"]) == 1.0
    assert metadata["gds_variance_variable"] == "ba_bond_all_pb_i_sigma2"


def test_bondanalysis_cli_batch_ui_delegates_to_launcher(
    tmp_path,
    monkeypatch,
):
    project_dir = tmp_path / "project"
    clusters_dir = tmp_path / "clusters"
    launched: dict[str, object] = {}

    def fake_launch_batch_ui(
        initial_project_dir=None,
        *,
        initial_clusters_dir=None,
    ):
        launched["initial_project_dir"] = initial_project_dir
        launched["initial_clusters_dir"] = initial_clusters_dir
        return 7

    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.batch_queue_window."
        "launch_bondanalysis_batch_queue_ui",
        fake_launch_batch_ui,
    )

    exit_code = bondanalysis_main(
        [
            "batch-ui",
            str(project_dir),
            "--clusters-dir",
            str(clusters_dir),
        ]
    )

    assert exit_code == 7
    assert launched["initial_project_dir"] == project_dir
    assert launched["initial_clusters_dir"] == clusters_dir


def test_saxshell_cli_forwards_to_bondanalysis_subcommand(
    tmp_path,
    capsys,
):
    clusters_dir = _build_sample_clusters_dir(tmp_path)

    exit_code = saxshell_main(
        [
            "bondanalysis",
            "inspect",
            str(clusters_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Clusters directory: {clusters_dir}" in captured.out
    assert "Cluster types detected: 2" in captured.out
    assert "Cluster types: PbI2, PbO" in captured.out
