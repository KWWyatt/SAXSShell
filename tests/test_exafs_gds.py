from __future__ import annotations

from pathlib import Path

from saxshell.exafs import (
    ArtemisGDSBuildSettings,
    ArtemisGDSParameter,
    artemis_gds_overview_path,
    build_artemis_gds_for_structure,
    parse_artemis_gds_text,
    validate_artemis_gds_text,
    write_artemis_gds_file,
)
from saxshell.exafs.cli import main as exafsgds_cli_main


def _write_xyz_structure(
    path: Path,
    atoms: list[tuple[str, float, float, float]],
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.4f} {y_coord:.4f} {z_coord:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_artemis_gds_accepts_export_rows_and_comments() -> None:
    text = """
# Artemis export comments are ignored.
guess amp = 0.9 # global amplitude
set reff_pb_i_01, 3.2
def r_pb_i_01 = reff_pb_i_01 + dr_pb_i_01
% another comment
! and another
"""

    parameters = parse_artemis_gds_text(text)

    assert [parameter.kind for parameter in parameters] == [
        "guess",
        "set",
        "def",
    ]
    assert parameters[0].name == "amp"
    assert parameters[1].expression == "3.2"
    assert parameters[2].expression == "reff_pb_i_01 + dr_pb_i_01"


def test_artemis_gds_parameter_export_omits_inline_comments() -> None:
    parameter = ArtemisGDSParameter(
        "guess",
        "amp",
        "0.9",
        "global amplitude",
    )

    assert parameter.to_artemis_line() == "guess amp = 0.9"


def test_generated_gds_for_representative_structure_validates(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "pb_i_br.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.20, 0.0, 0.0),
            ("I", -3.24, 0.0, 0.0),
            ("Br", 0.0, 4.10, 0.0),
        ],
    )

    document = build_artemis_gds_for_structure(
        structure_path,
        ArtemisGDSBuildSettings(
            absorber_element="Pb",
            max_distance_angstrom=4.5,
            shell_tolerance_angstrom=0.08,
        ),
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)

    assert report.is_valid, report.summary_text()
    assert [shell.scatterer_element for shell in document.shells] == [
        "I",
        "Br",
    ]
    assert document.shells[0].multiplicity == 2.0
    assert "guess dr_pb_i_01 = 0" in text
    assert "restrain res_dr_pb_i_01" in text
    assert "set n_pb_i_01" not in text
    assert "set reff_pb_i_01" not in text
    assert "def r_pb_i_01" not in text
    assert document.path_hints[0].s02 == "amp"
    assert document.path_hints[0].delr == "dr_pb_i_01"
    assert document.path_hints[0].reff_angstrom == 3.22
    assert all(
        line and not line.startswith(("#", "!", "%"))
        for line in text.splitlines()
    )
    assert text.index("guess amp = 0.9") < text.index("guess dr_pb_i_01 = 0")
    assert text.index("guess dr_pb_i_01 = 0") < text.index(
        "guess ss_pb_i_01 = 0.003"
    )
    assert text.index("guess ss_pb_i_01 = 0.003") < text.index(
        "restrain res_dr_pb_i_01"
    )


def test_generated_gds_excludes_hydrogen_scatterers(tmp_path: Path) -> None:
    structure_path = tmp_path / "pb_i_h.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("H", 1.1, 0.0, 0.0),
            ("I", 3.2, 0.0, 0.0),
        ],
    )

    document = build_artemis_gds_for_structure(
        structure_path,
        ArtemisGDSBuildSettings(
            absorber_element="Pb",
            max_distance_angstrom=3.5,
        ),
    )
    text = document.to_text()

    assert [shell.scatterer_element for shell in document.shells] == ["I"]
    assert "pb_h" not in text.lower()
    assert "pb_i" in text.lower()


def test_validate_artemis_gds_reports_generation_hazards() -> None:
    report = validate_artemis_gds_text(
        """
guess amp = 0.9
set amp = 1.0
guess cos = 0
def bad = amp + missing_symbol
def self_ref = self_ref + 1
"""
    )

    messages = "\n".join(issue.message for issue in report.errors)

    assert not report.is_valid
    assert "duplicates a parameter" in messages
    assert "reserved" in messages
    assert "undefined symbol 'missing_symbol'" in messages
    assert "references itself" in messages


def test_write_and_cli_validate_generated_artemis_gds_file(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "sn_i.xyz"
    _write_xyz_structure(
        structure_path,
        [
            ("Sn", 0.0, 0.0, 0.0),
            ("I", 3.0, 0.0, 0.0),
        ],
    )
    output_path = tmp_path / "sn_i.gds"
    document = build_artemis_gds_for_structure(
        structure_path,
        ArtemisGDSBuildSettings(absorber_atom_index=1),
    )

    write_artemis_gds_file(output_path, document)
    overview_path = artemis_gds_overview_path(output_path)

    assert output_path.is_file()
    assert overview_path.is_file()
    overview_text = overview_path.read_text(encoding="utf-8")
    assert "EXAFS GDS File Overview" in overview_text
    assert "Source structure:" in overview_text
    assert "sn_i_01" in overview_text
    assert "Path Parameter Assignments" in overview_text
    assert exafsgds_cli_main(["validate", str(output_path)]) == 0


def test_cli_build_writes_generated_artemis_gds_file(tmp_path: Path) -> None:
    structure_path = tmp_path / "pb_cl.xyz"
    output_path = tmp_path / "pb_cl.gds"
    _write_xyz_structure(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("Cl", 2.9, 0.0, 0.0),
        ],
    )

    exit_code = exafsgds_cli_main(
        [
            "build",
            str(structure_path),
            str(output_path),
            "--absorber-element",
            "Pb",
            "--no-restraints",
        ]
    )

    assert exit_code == 0
    assert "guess dr_pb_cl_01 = 0" in output_path.read_text(encoding="utf-8")
    assert artemis_gds_overview_path(output_path).is_file()
