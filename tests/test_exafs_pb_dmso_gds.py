from __future__ import annotations

import math
from pathlib import Path

from saxshell.exafs import (
    PbDMFGDSBuildSettings,
    PbDMSOGDSBuildSettings,
    build_pb_dmf_gds_from_structure,
    build_pb_dmso_gds_from_structure,
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


def _dmso_sulfur_for_120_degree_pb_o_s(
    oxygen: tuple[float, float, float],
    axis: tuple[float, float, float],
    *,
    os_distance: float = 1.5,
) -> tuple[float, float, float]:
    cos_60 = math.cos(math.radians(60.0))
    sin_60 = math.sin(math.radians(60.0))
    return (
        oxygen[0] + os_distance * (axis[0] * cos_60 + axis[1] * sin_60),
        oxygen[1] + os_distance * (axis[1] * cos_60 + axis[2] * sin_60),
        oxygen[2] + os_distance * (axis[2] * cos_60 + axis[0] * sin_60),
    )


def test_pb_dmso_builder_generates_hand_fit_style_variables(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_i_dmso.xyz"
    oxygen_1 = (2.5, 0.0, 0.0)
    oxygen_2 = (0.0, 2.6, 0.0)
    oxygen_3 = (0.0, 0.0, 2.7)
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.1, 0.0, 0.0),
            ("I", -3.2, 0.0, 0.0),
            ("O", *oxygen_1),
            ("S", *_dmso_sulfur_for_120_degree_pb_o_s(oxygen_1, (1, 0, 0))),
            ("O", *oxygen_2),
            ("S", *_dmso_sulfur_for_120_degree_pb_o_s(oxygen_2, (0, 1, 0))),
            ("O", *oxygen_3),
            ("S", *_dmso_sulfur_for_120_degree_pb_o_s(oxygen_3, (0, 0, 1))),
        ],
    )

    document = build_pb_dmso_gds_from_structure(
        structure_path,
        PbDMSOGDSBuildSettings(oxygen_count=3, theta_pbos_degrees=120.0),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert "set snot = 0.9" in text
    assert "guess enot = 0" in text
    assert "guess cn_o1 = 1" in text
    assert "guess cn_i1 = 1" in text
    assert "guess cn_i2 = 1" in text
    assert "def cn_s1 = cn_o1" in text
    assert "cn_o_tot" not in text
    assert "cn_i_tot" not in text
    assert "cn_s_tot" not in text
    assert "cn_tot" not in text
    assert "guess delr_o1 = 0" in text
    assert "guess delr_i2 = 0" in text
    assert "guess sig2_o1 = 0.003" in text
    assert "def sig2_o2 = sig2_o1" in text
    assert "guess sig2_i1 = 0.003" in text
    assert "set bl_os = 1.5" in text
    assert "guess theta_pbos = 120" in text
    assert "def theta_pbos_rad = theta_pbos*pi/180" in text
    assert "set reff_pbo_1 = 2.5" in text
    assert "set reff_pbs_1 =" in text
    assert "def delr_s1 = sqrt(" in text
    assert "cos(theta_pbos_rad)" in text
    assert "def sig2_s1 =" in text
    assert "mu_o_eff" not in text
    assert "sig2_s_eff" not in text
    hint_expressions = {hint.s02 for hint in document.path_hints}
    assert "snot*cn_o1" in hint_expressions
    assert "snot*cn_s1" in hint_expressions
    assert "snot*cn_i1" in hint_expressions
    assert all(
        line and not line.startswith(("#", "!", "%"))
        for line in text.splitlines()
    )
    assert text.index("set snot = 0.9") < text.index("guess cn_o1 = 1")
    assert text.index("guess cn_o1 = 1") < text.index("set bl_os = 1.5")
    assert text.index("set bl_os = 1.5") < text.index("set reff_pbo_1 = 2.5")
    assert text.index("set reff_pbo_1 = 2.5") < text.index("guess delr_o1 = 0")
    assert text.index("guess delr_o1 = 0") < text.index(
        "guess sig2_o1 = 0.003"
    )


def test_pb_dmso_builder_uses_real_example_representative() -> None:
    structure_path = Path(
        "examples/saxs_runtime/projects/079_PbI2_DMSO_0p8M_RT_new_fit/"
        "exafs_gds_example/PbI2__representative__frame_30589_AAB.pdb"
    )

    document = build_pb_dmso_gds_from_structure(
        structure_path,
        PbDMSOGDSBuildSettings(oxygen_count=4),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert "guess cn_o4 = 1" in text
    assert "guess cn_i2 = 1" in text
    assert "def cn_s4 = cn_o4" in text
    assert "set reff_pbo_1 = 2.40071" in text


def test_cli_build_pbi2_dmso_writes_gds_file(tmp_path: Path) -> None:
    structure_path = tmp_path / "pb_dmso_cli.xyz"
    output_path = tmp_path / "pb_dmso_cli.gds"
    oxygen = (2.55, 0.0, 0.0)
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("O", *oxygen),
            ("S", *_dmso_sulfur_for_120_degree_pb_o_s(oxygen, (1, 0, 0))),
        ],
    )

    exit_code = exafsgds_cli_main(
        [
            "build-pbi2-dmso",
            str(structure_path),
            str(output_path),
            "--oxygen-count",
            "1",
            "--bl-os",
            "1.52",
            "--no-restraints",
        ]
    )

    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "set bl_os = 1.52" in text
    assert "cn_o_tot" not in text
    assert "def delr_s1 = sqrt(" in text


def test_pb_dmf_builder_generates_dihedral_constrained_variables(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_i_dmf.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("O", 2.5, 0.0, 0.0),
            ("C", 3.0, 1.0, 0.0),
            ("N", 3.2, 2.0, 0.6),
        ],
    )

    document = build_pb_dmf_gds_from_structure(
        structure_path,
        PbDMFGDSBuildSettings(
            oxygen_count=1,
            bl_oc_angstrom=1.12,
            bl_cn_angstrom=1.18,
            theta_pboc_degrees=120.0,
            theta_ocn_degrees=125.0,
            phi_pbocn_degrees=35.0,
        ),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert "guess cn_o1 = 1" in text
    assert "def cn_c1 = cn_o1" in text
    assert "def cn_n1 = cn_o1" in text
    assert "set bl_oc = 1.12" in text
    assert "set bl_cn = 1.18" in text
    assert "guess theta_pboc = 120" in text
    assert "guess theta_ocn = 125" in text
    assert "guess phi_pbocn = 35" in text
    assert "def theta_pboc_rad = theta_pboc*pi/180" in text
    assert "def phi_pbocn_rad = phi_pbocn*pi/180" in text
    assert "set reff_pbc_1 =" in text
    assert "set reff_pbn_1 =" in text
    assert "def delr_c1 = sqrt(" in text
    assert "def delr_n1 = sqrt(" in text
    assert "cos(phi_pbocn_rad)" in text
    assert "def sig2_n1 =" in text
    assert "sig2_n_eff" not in text
    hint_expressions = {hint.s02 for hint in document.path_hints}
    assert {"snot*cn_o1", "snot*cn_c1", "snot*cn_n1"} <= hint_expressions


def test_pb_dmf_builder_groups_multiple_absorber_paths_by_distance(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "two_pb_i_dmf.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.10, 0.0, 0.0),
            ("I", 0.0, 3.30, 0.0),
            ("I", 0.0, 0.0, 3.32),
            ("O", 2.5, 0.0, 0.0),
            ("C", 3.0, 1.0, 0.0),
            ("N", 3.2, 2.0, 0.6),
            ("Pb", 10.0, 0.0, 0.0),
            ("I", 13.50, 0.0, 0.0),
            ("I", 10.0, 3.31, 0.0),
            ("O", 12.5, 0.0, 0.0),
            ("C", 13.0, 1.0, 0.0),
            ("N", 13.2, 2.0, 0.6),
        ],
    )

    document = build_pb_dmf_gds_from_structure(
        structure_path,
        PbDMFGDSBuildSettings(
            oxygen_count=1,
            shell_tolerance_angstrom=0.05,
            include_restraints=False,
        ),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert document.settings.absorber_atom_index is None
    assert "guess cn_i1 = 0.5" in text
    assert "guess cn_i2 = 1.5" in text
    assert "guess cn_i3 = 0.5" in text
    assert "cn_i_tot" not in text
    assert "guess cn_o1 = 1" in text

    iodide_shells = [
        shell for shell in document.shells if shell.scatterer_element == "I"
    ]
    assert len(iodide_shells) == 3
    assert [shell.multiplicity for shell in iodide_shells] == [0.5, 1.5, 0.5]
    assert math.isclose(iodide_shells[1].mean_distance_angstrom, 3.31)
    assert math.isclose(document.path_hints[1].multiplicity, 1.5)
    overview_text = document.to_overview_text()
    assert "Template: Pb-I / DMF constrained GDS" in overview_text
    assert "Requested nearest iodides per absorber: all within cutoff" in (
        overview_text
    )
    assert "pb_i_2 | Pb | I | 1.5 | 3.31" in overview_text


def test_cli_build_pbi2_dmf_writes_gds_file(tmp_path: Path) -> None:
    structure_path = tmp_path / "pb_dmf_cli.xyz"
    output_path = tmp_path / "pb_dmf_cli.gds"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
            ("O", 2.5, 0.0, 0.0),
            ("C", 3.0, 1.0, 0.0),
            ("N", 3.2, 2.0, 0.6),
        ],
    )

    exit_code = exafsgds_cli_main(
        [
            "build-pbi2-dmf",
            str(structure_path),
            str(output_path),
            "--oxygen-count",
            "1",
            "--bl-oc",
            "1.12",
            "--bl-cn",
            "1.18",
            "--no-restraints",
        ]
    )

    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "set bl_oc = 1.12" in text
    assert "cn_o_tot" not in text
    assert "def delr_n1 = sqrt(" in text
