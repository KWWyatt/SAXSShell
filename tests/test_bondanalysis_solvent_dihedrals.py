from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from saxshell.bondanalysis import (
    BondAnalysisWorkflow,
    BondAnalyzer,
    DihedralQuartetDefinition,
    expanded_solvent_dihedral_quartets,
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def test_dmf_dihedrals_are_residue_aware_and_split_terminal_branches(
    tmp_path,
):
    structure_path = tmp_path / "dmf.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("Pb1", "PBI", 1, 0.5, 0.5, 0.0, "Pb"),
            ("O1", "DMF", 1, 1.0, 0.0, 0.0, "O"),
            ("C1", "DMF", 1, 2.0, 0.0, 0.0, "C"),
            ("N1", "DMF", 1, 2.0, 1.0, 0.0, "N"),
            ("C2", "DMF", 1, 2.0, 2.0, 1.0, "C"),
            ("C3", "DMF", 1, 2.0, 2.0, -1.0, "C"),
            ("O1", "DMF", 2, 0.0, 1.0, 0.0, "O"),
            ("C1", "DMF", 2, 0.0, 2.0, 0.0, "C"),
            ("N1", "DMF", 2, 0.0, 2.0, 1.0, "N"),
            ("C2", "DMF", 2, 0.5, 2.0, 2.0, "C"),
            ("C3", "DMF", 2, -0.5, 2.0, 2.0, "C"),
        ],
    )
    pb_o_c_n = DihedralQuartetDefinition(
        "Pb",
        "O",
        "C",
        "N",
        1.5,
        1.5,
        1.5,
    )
    o_c_n_c = DihedralQuartetDefinition(
        "O",
        "C",
        "N",
        "C",
        1.5,
        1.5,
        1.5,
    )
    analyzer = BondAnalyzer(dihedral_quartets=(pb_o_c_n, o_c_n_c))

    _bond_values, _angle_values, dihedral_values, _coordination_values = (
        analyzer.measure_structure_with_coordination_and_dihedrals(
            structure_path
        )
    )
    branch_c1 = o_c_n_c.with_branch_label("C1")
    branch_c2 = o_c_n_c.with_branch_label("C2")

    assert analyzer.dihedral_quartets == expanded_solvent_dihedral_quartets(
        (pb_o_c_n, o_c_n_c)
    )
    assert len(dihedral_values[pb_o_c_n]) == 2
    assert len(dihedral_values[o_c_n_c]) == 4
    assert len(dihedral_values[branch_c1]) == 2
    assert len(dihedral_values[branch_c2]) == 2
    assert sorted(dihedral_values[o_c_n_c]) == pytest.approx(
        sorted(dihedral_values[branch_c1] + dihedral_values[branch_c2])
    )


def test_dmso_workflow_writes_combined_and_terminal_dihedral_outputs(tmp_path):
    clusters_dir = tmp_path / "clusters"
    structure_path = clusters_dir / "PbDMSO" / "frame_0001.pdb"
    _write_pdb_structure(
        structure_path,
        [
            ("Pb1", "PBI", 1, 0.5, 0.5, 0.0, "Pb"),
            ("O1", "DMS", 1, 1.0, 0.0, 0.0, "O"),
            ("S1", "DMS", 1, 2.0, 0.0, 0.0, "S"),
            ("C1", "DMS", 1, 2.0, 1.0, 0.8, "C"),
            ("C2", "DMS", 1, 2.0, 1.0, -0.8, "C"),
        ],
    )
    pb_o_s_c = DihedralQuartetDefinition(
        "Pb",
        "O",
        "S",
        "C",
        1.5,
        1.5,
        1.5,
    )

    result = BondAnalysisWorkflow(
        clusters_dir,
        dihedral_quartets=(pb_o_s_c,),
        output_dir=tmp_path / "bondanalysis",
        generate_preview_plots=False,
    ).run()
    payload = json.loads(result.results_index_path.read_text())
    labels = [
        entry["branch_label"] for entry in payload["dihedral_quartets"][1:]
    ]
    combined_values = np.load(
        result.output_dir / "all_clusters" / "Pb_O_S_C_dihedrals.npy",
        allow_pickle=False,
    )
    branch_c1_values = np.load(
        result.output_dir / "all_clusters" / "Pb_O_S_C_C1_dihedrals.npy",
        allow_pickle=False,
    )
    branch_c2_values = np.load(
        result.output_dir / "all_clusters" / "Pb_O_S_C_C2_dihedrals.npy",
        allow_pickle=False,
    )
    for output_scope in (
        result.output_dir / "all_clusters",
        result.output_dir / "cluster_types" / "PbDMSO",
    ):
        assert (output_scope / "Pb_O_S_C_histogram.csv").exists()
        assert (output_scope / "Pb_O_S_C_C1_histogram.csv").exists()
        assert (output_scope / "Pb_O_S_C_C2_histogram.csv").exists()

    assert labels == ["C1", "C2"]
    assert combined_values["value"].tolist() == pytest.approx(
        branch_c1_values["value"].tolist() + branch_c2_values["value"].tolist()
    )
    assert result.cluster_results[0].dihedral_value_counts == {
        "Pb-O-S-C": 2,
        "Pb-O-S-C (terminal C1)": 1,
        "Pb-O-S-C (terminal C2)": 1,
    }
