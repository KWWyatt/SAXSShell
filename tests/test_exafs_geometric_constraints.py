from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from saxshell.exafs import (
    CoordinationGroupSpec,
    FourAtomDihedralConstraintSpec,
    GeometricGDSBuildSettings,
    IndependentPathSpec,
    ThreeAtomConstraintSpec,
    angle_variance_from_degrees,
    build_geometric_constraint_gds,
    dihedral_chord_distance,
    three_atom_chord_distance,
    validate_artemis_gds_text,
)
from saxshell.exafs.cli import main as exafsgds_cli_main


def _write_xyz_structure(
    path: Path,
    atoms: list[tuple[str, float, float, float]],
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.6f} {y_coord:.6f} {z_coord:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_three_atom_chord_distance_matches_law_of_cosines() -> None:
    distance = three_atom_chord_distance(
        anchor_distance=2.55,
        fixed_bridge_terminal_distance=1.52,
        angle_radians=math.radians(120.0),
    )

    assert distance == pytest.approx(
        math.sqrt(
            2.55 * 2.55
            + 1.52 * 1.52
            - 2.0 * 2.55 * 1.52 * math.cos(math.radians(120.0))
        )
    )


def test_dihedral_chord_distance_matches_four_atom_derivation() -> None:
    distance = dihedral_chord_distance(
        anchor_distance=2.55,
        fixed_bridge_hinge_distance=1.23,
        fixed_hinge_terminal_distance=1.37,
        angle_abc_radians=math.radians(118.0),
        angle_bcd_radians=math.radians(124.0),
        dihedral_radians=math.radians(35.0),
    )
    expected_squared = (
        2.55**2
        + 1.23**2
        + 1.37**2
        - 2.0 * 2.55 * 1.23 * math.cos(math.radians(118.0))
        - 2.0 * 1.23 * 1.37 * math.cos(math.radians(124.0))
        + 2.0
        * 2.55
        * 1.37
        * (
            math.cos(math.radians(118.0)) * math.cos(math.radians(124.0))
            + math.sin(math.radians(118.0))
            * math.sin(math.radians(124.0))
            * math.cos(math.radians(35.0))
        )
    )

    assert distance == pytest.approx(math.sqrt(expected_squared))


def test_angle_variance_from_degrees_returns_radians_squared() -> None:
    assert angle_variance_from_degrees(10.0) == pytest.approx(
        math.radians(10.0) ** 2
    )


def test_pb_i_and_pb_dmso_geometry_constraints_generate_valid_gds(
    tmp_path: Path,
) -> None:
    pb_o = 2.55
    o_s = 1.52
    angle_degrees = 120.0
    s_x = pb_o + o_s * math.cos(math.radians(60.0))
    s_y = o_s * math.sin(math.radians(60.0))
    structure_path = tmp_path / "pb_i_dmso.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.20, 0.0, 0.0),
            ("O", pb_o, 0.0, 0.0),
            ("S", s_x, s_y, 0.0),
        ],
    )

    document = build_geometric_constraint_gds(
        structure_path,
        independent_paths=(
            IndependentPathSpec(
                label="pb_i_01",
                absorber_atom_index=1,
                scatterer_atom_index=2,
                scatterer_element="I",
                initial_sigma2=0.004,
            ),
        ),
        three_atom_constraints=(
            ThreeAtomConstraintSpec(
                label="dmso_01",
                absorber_atom_index=1,
                bridge_atom_index=3,
                terminal_atom_index=4,
                fixed_bridge_terminal_distance_angstrom=o_s,
                angle_mean_degrees=angle_degrees,
                angle_sigma_degrees=8.0,
            ),
        ),
        settings=GeometricGDSBuildSettings(default_initial_sigma2=0.0035),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert "guess dr_pb_i_01 = 0" in text
    assert "guess ss_pb_i_01 = 0.004" in text
    assert "set b_o_s_dmso_01 = 1.52" in text
    assert "set mu_pb_o_dmso_01_s = 2.0944" in text
    assert "set sig2_theta_pb_o_dmso_01_s = 0.0194955" in text
    assert "def dr_pb_s_dmso_01 = sqrt(" in text
    assert "cos(mu_pb_o_dmso_01_s)" in text
    assert "sin(mu_pb_o_dmso_01_s)" in text
    assert "def ss_pb_s_dmso_01 =" in text
    assert "def dr_ms_pb_o_dmso_01_s =" in text
    assert all(
        line and not line.startswith(("#", "!", "%"))
        for line in text.splitlines()
    )
    hints_by_label = {hint.shell_label: hint for hint in document.path_hints}
    assert hints_by_label["pb_s_dmso_01"].delr == "dr_pb_s_dmso_01"
    assert hints_by_label["pb_s_dmso_01"].sigma2 == "ss_pb_s_dmso_01"
    assert hints_by_label["ms_pb_o_dmso_01_s"].delr == ("dr_ms_pb_o_dmso_01_s")
    assert hints_by_label["ms_pb_o_dmso_01_s"].sigma2 == (
        "ss_ms_pb_o_dmso_01_s"
    )


def test_four_atom_dihedral_constraint_generates_valid_gds(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_dmf_fragment.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.5, 0.5, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("N", 2.0, 1.0, 0.5),
        ],
    )

    document = build_geometric_constraint_gds(
        structure_path,
        four_atom_dihedral_constraints=(
            FourAtomDihedralConstraintSpec(
                label="dmf_01",
                absorber_atom_index=1,
                bridge_atom_index=2,
                hinge_atom_index=3,
                terminal_atom_index=4,
                fixed_bridge_hinge_distance_angstrom=1.23,
                fixed_hinge_terminal_distance_angstrom=1.37,
                angle_abc_mean_degrees=118.0,
                angle_bcd_mean_degrees=124.0,
                dihedral_mean_degrees=35.0,
                angle_abc_sigma_degrees=7.0,
                angle_bcd_sigma_degrees=5.0,
                dihedral_sigma_degrees=11.0,
            ),
        ),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert "set b_o_c_dmf_01 = 1.23" in text
    assert "set b_c_n_dmf_01 = 1.37" in text
    assert "set mu_pb_o_dmf_01_c = 2.05949" in text
    assert "set alpha_o_c_dmf_01_n = 2.16421" in text
    assert "set phi_pb_o_c_n_dmf_01 = 0.610865" in text
    assert "def dr_pb_n_dmf_01 = sqrt(" in text
    assert "cos(phi_pb_o_c_n_dmf_01)" in text
    assert "def ss_pb_n_dmf_01 =" in text
    hints_by_label = {hint.shell_label: hint for hint in document.path_hints}
    assert hints_by_label["pb_n_dmf_01"].delr == "dr_pb_n_dmf_01"
    assert hints_by_label["pb_n_dmf_01"].sigma2 == "ss_pb_n_dmf_01"


def test_coordination_groups_link_path_s02_and_add_soft_bounds(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_i_o_cn.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("O", 2.55, 0.0, 0.0),
            ("S", 3.31, 1.316359, 0.0),
            ("C", 3.31, 2.30, 0.80),
        ],
    )

    document = build_geometric_constraint_gds(
        structure_path,
        coordination_groups=(
            CoordinationGroupSpec(
                label="pb_i",
                initial_value=6.0,
                lower_bound=4.0,
                upper_bound=8.0,
            ),
            CoordinationGroupSpec(
                label="pb_o",
                initial_value=2.0,
                lower_bound=0.0,
                upper_bound=6.0,
            ),
        ),
        independent_paths=(
            IndependentPathSpec(
                label="pb_i_01",
                absorber_atom_index=1,
                scatterer_atom_index=2,
                scatterer_element="I",
                multiplicity=6.0,
            ),
        ),
        three_atom_constraints=(
            ThreeAtomConstraintSpec(
                label="dmso_01",
                absorber_atom_index=1,
                bridge_atom_index=3,
                terminal_atom_index=4,
                fixed_bridge_terminal_distance_angstrom=1.52,
                angle_mean_degrees=120.0,
                angle_sigma_degrees=8.0,
                terminal_multiplicity=2.0,
                multiple_scattering_multiplicity=2.0,
                anchor_reference_multiplicity=2.0,
                terminal_reference_multiplicity=2.0,
                multiple_scattering_reference_multiplicity=2.0,
            ),
        ),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert "guess cn_pb_i = 6" in text
    assert "restrain res_cn_pb_i = 1000*penalty(cn_pb_i, 4, 8)" in text
    assert "guess cn_pb_o = 2" in text
    assert "restrain res_cn_pb_o = 1000*penalty(cn_pb_o, 0, 6)" in text
    assert "set cnref_pb_i_01 = 6" in text
    assert "def cn_pb_i_01 = cn_pb_i*frac_pb_i_01" in text
    assert "set cnref_pb_o_dmso_01 = 2" in text
    assert "def cn_pb_o_dmso_01 = cn_pb_o*frac_pb_o_dmso_01" in text
    assert "set cnref_pb_s_dmso_01 = 2" in text
    assert "def cn_pb_s_dmso_01 = cn_pb_o*frac_pb_s_dmso_01" in text
    assert "set cnref_ms_pb_o_dmso_01_s = 2" in text
    assert "def cn_ms_pb_o_dmso_01_s = cn_pb_o*frac_ms_pb_o_dmso_01_s" in text
    hint_s02 = {hint.shell_label: hint.s02 for hint in document.path_hints}
    assert hint_s02["pb_i_01"] == "amp*cn_pb_i_01/cnref_pb_i_01"
    assert hint_s02["pb_o_dmso_01"] == (
        "amp*cn_pb_o_dmso_01/cnref_pb_o_dmso_01"
    )
    assert hint_s02["pb_s_dmso_01"] == (
        "amp*cn_pb_s_dmso_01/cnref_pb_s_dmso_01"
    )
    assert hint_s02["ms_pb_o_dmso_01_s"] == (
        "amp*cn_ms_pb_o_dmso_01_s/cnref_ms_pb_o_dmso_01_s"
    )


def test_three_atom_constraint_can_reuse_existing_anchor_variables(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "linked_anchor.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 2.5, 0.0, 0.0),
            ("S", 3.25, 1.299038, 0.0),
        ],
    )

    document = build_geometric_constraint_gds(
        structure_path,
        independent_paths=(
            IndependentPathSpec(
                label="shared_pb_o",
                absorber_atom_index=1,
                scatterer_atom_index=2,
            ),
        ),
        three_atom_constraints=(
            ThreeAtomConstraintSpec(
                label="dmso_shared",
                absorber_atom_index=1,
                bridge_atom_index=2,
                terminal_atom_index=3,
                fixed_bridge_terminal_distance_angstrom=1.5,
                angle_mean_degrees=120.0,
                angle_sigma_degrees=5.0,
                anchor_delta_r_name="dr_shared_pb_o",
                anchor_sigma2_name="ss_shared_pb_o",
                include_multiple_scattering=False,
            ),
        ),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert "guess dr_pb_o_dmso_shared" not in text
    assert "dr_shared_pb_o" in text
    assert "ss_shared_pb_o" in text
    assert "ms_pb_o_dmso_shared_s" not in text


def test_cli_build_geometry_uses_json_constraint_spec(tmp_path: Path) -> None:
    structure_path = tmp_path / "pb_dmso_cli.xyz"
    output_path = tmp_path / "pb_dmso_cli.gds"
    spec_path = tmp_path / "constraints.json"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("O", 2.55, 0.0, 0.0),
            ("S", 3.31, 1.316359, 0.0),
            ("C", 3.31, 2.30, 0.80),
        ],
    )
    spec_path.write_text(
        json.dumps(
            {
                "settings": {"default_initial_sigma2": 0.0035},
                "coordination_groups": [
                    {
                        "label": "pb_i",
                        "initial_value": 6.0,
                        "lower_bound": 4.0,
                        "upper_bound": 8.0,
                    },
                    {
                        "label": "pb_o",
                        "initial_value": 2.0,
                        "lower_bound": 0.0,
                        "upper_bound": 6.0,
                    },
                ],
                "independent_paths": [
                    {
                        "label": "pb_i_01",
                        "absorber_atom_index": 1,
                        "scatterer_atom_index": 2,
                        "scatterer_element": "I",
                        "multiplicity": 6.0,
                    }
                ],
                "three_atom_constraints": [
                    {
                        "label": "dmso_01",
                        "absorber_atom_index": 1,
                        "bridge_atom_index": 3,
                        "terminal_atom_index": 4,
                        "fixed_bridge_terminal_distance_angstrom": 1.52,
                        "angle_mean_degrees": 120.0,
                        "angle_sigma_degrees": 8.0,
                    }
                ],
                "four_atom_dihedral_constraints": [
                    {
                        "label": "dmso_c_01",
                        "absorber_atom_index": 1,
                        "bridge_atom_index": 3,
                        "hinge_atom_index": 4,
                        "terminal_atom_index": 5,
                        "fixed_bridge_hinge_distance_angstrom": 1.52,
                        "fixed_hinge_terminal_distance_angstrom": 1.20,
                        "angle_abc_mean_degrees": 120.0,
                        "angle_bcd_mean_degrees": 105.0,
                        "dihedral_mean_degrees": 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = exafsgds_cli_main(
        [
            "build-geometry",
            str(structure_path),
            str(output_path),
            "--spec",
            str(spec_path),
        ]
    )

    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "def dr_pb_s_dmso_01 = sqrt(" in text
    assert "def dr_pb_c_dmso_c_01 = sqrt(" in text
    assert "guess dr_pb_i_01 = 0" in text
    assert "guess cn_pb_i = 6" in text
