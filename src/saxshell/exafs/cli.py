from __future__ import annotations

import argparse
import json
from pathlib import Path

from saxshell.version import __version__

from .gds import (
    ArtemisGDSBuildSettings,
    artemis_gds_overview_path,
    build_artemis_gds_for_structure,
    validate_artemis_gds_file,
    write_artemis_gds_file,
)
from .geometry import (
    CoordinationGroupSpec,
    FourAtomDihedralConstraintSpec,
    GeometricGDSBuildSettings,
    IndependentPathSpec,
    ThreeAtomConstraintSpec,
    build_geometric_constraint_gds,
)
from .pb_dmf import PbDMFGDSBuildSettings, build_pb_dmf_gds_from_structure
from .pb_dmso import PbDMSOGDSBuildSettings, build_pb_dmso_gds_from_structure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exafsgds",
        description=(
            "Build and validate Artemis GDS files for representative "
            "structure based EXAFS refinement setup."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the EXAFS GDS helper version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser(
        "build",
        help="Build an Artemis GDS file from a PDB or XYZ representative.",
    )
    build_parser.add_argument("structure", type=Path)
    build_parser.add_argument("output", type=Path)
    build_parser.add_argument("--absorber-element", default=None)
    build_parser.add_argument(
        "--absorber-atom-index",
        type=int,
        default=None,
        help="One-based atom index to use as the absorber.",
    )
    build_parser.add_argument(
        "--min-distance",
        type=float,
        default=0.5,
        help="Minimum absorber-scatterer distance in angstrom.",
    )
    build_parser.add_argument(
        "--max-distance",
        type=float,
        default=6.0,
        help="Maximum absorber-scatterer distance in angstrom.",
    )
    build_parser.add_argument(
        "--shell-tolerance",
        type=float,
        default=0.12,
        help="Distance tolerance for grouping shells in angstrom.",
    )
    build_parser.add_argument(
        "--no-restraints",
        action="store_true",
        help="Do not emit restrain rows for delta-R and sigma2 bounds.",
    )
    build_parser.set_defaults(handler=_handle_build)

    geometry_parser = subparsers.add_parser(
        "build-geometry",
        help=(
            "Build linked geometric GDS constraints from a JSON constraint "
            "specification."
        ),
    )
    geometry_parser.add_argument("structure", type=Path)
    geometry_parser.add_argument("output", type=Path)
    geometry_parser.add_argument(
        "--spec",
        required=True,
        type=Path,
        help="JSON file describing independent paths and geometric links.",
    )
    geometry_parser.set_defaults(handler=_handle_build_geometry)

    pb_dmso_parser = subparsers.add_parser(
        "build-pbi2-dmso",
        help=(
            "Build a compact hand-fit-style Pb-I/DMSO GDS file directly "
            "from a representative structure."
        ),
    )
    pb_dmso_parser.add_argument("structure", type=Path)
    pb_dmso_parser.add_argument("output", type=Path)
    pb_dmso_parser.add_argument(
        "--absorber-atom-index",
        type=int,
        default=None,
        help="One-based Pb atom index to use as the absorber.",
    )
    pb_dmso_parser.add_argument(
        "--oxygen-count",
        type=int,
        default=3,
        help="Number of nearest DMSO oxygens to include.",
    )
    pb_dmso_parser.add_argument(
        "--iodide-count",
        type=int,
        default=None,
        help=(
            "Number of nearest iodides to include. Defaults to all within "
            "cutoff."
        ),
    )
    pb_dmso_parser.add_argument(
        "--max-oxygen-distance",
        type=float,
        default=4.0,
        help="Maximum Pb-O distance in angstrom.",
    )
    pb_dmso_parser.add_argument(
        "--max-iodide-distance",
        type=float,
        default=4.0,
        help="Maximum Pb-I distance in angstrom.",
    )
    pb_dmso_parser.add_argument(
        "--max-os-distance",
        type=float,
        default=2.2,
        help="Maximum O-S pairing distance in angstrom.",
    )
    pb_dmso_parser.add_argument(
        "--shell-tolerance",
        type=float,
        default=0.12,
        help="Distance tolerance for grouping near-degenerate paths in angstrom.",
    )
    pb_dmso_parser.add_argument(
        "--bl-os",
        type=float,
        default=1.5,
        help="Fixed O-S bond length used in the Pb-O-S geometry constraint.",
    )
    pb_dmso_parser.add_argument(
        "--theta-pbos",
        type=float,
        default=None,
        help=(
            "Initial mean Pb-O-S angle in degrees. Defaults to the mean "
            "angle in the selected structure."
        ),
    )
    pb_dmso_parser.add_argument(
        "--angle-width",
        type=float,
        default=8.0,
        help="Initial Pb-O-S angular width in degrees.",
    )
    pb_dmso_parser.add_argument(
        "--vary-snot",
        action="store_true",
        help="Emit snot as a guess row instead of a fixed set row.",
    )
    pb_dmso_parser.add_argument(
        "--no-link-oxygen-sigma2",
        action="store_true",
        help="Fit separate sigma2 values for each Pb-O path.",
    )
    pb_dmso_parser.add_argument(
        "--no-restraints",
        action="store_true",
        help="Do not emit soft restrain rows.",
    )
    pb_dmso_parser.set_defaults(handler=_handle_build_pb_dmso)

    pb_dmf_parser = subparsers.add_parser(
        "build-pbi2-dmf",
        help=(
            "Build a compact hand-fit-style Pb-I/DMF GDS file directly "
            "from a representative structure."
        ),
    )
    pb_dmf_parser.add_argument("structure", type=Path)
    pb_dmf_parser.add_argument("output", type=Path)
    pb_dmf_parser.add_argument(
        "--absorber-atom-index",
        type=int,
        default=None,
        help="One-based Pb atom index to use as the absorber.",
    )
    pb_dmf_parser.add_argument(
        "--oxygen-count",
        type=int,
        default=3,
        help="Number of nearest DMF oxygens to include.",
    )
    pb_dmf_parser.add_argument(
        "--iodide-count",
        type=int,
        default=None,
        help=(
            "Number of nearest iodides to include. Defaults to all within "
            "cutoff."
        ),
    )
    pb_dmf_parser.add_argument(
        "--max-oxygen-distance",
        type=float,
        default=4.0,
        help="Maximum Pb-O distance in angstrom.",
    )
    pb_dmf_parser.add_argument(
        "--max-iodide-distance",
        type=float,
        default=4.0,
        help="Maximum Pb-I distance in angstrom.",
    )
    pb_dmf_parser.add_argument(
        "--max-oc-distance",
        type=float,
        default=1.8,
        help="Maximum DMF O-C pairing distance in angstrom.",
    )
    pb_dmf_parser.add_argument(
        "--max-cn-distance",
        type=float,
        default=1.8,
        help="Maximum DMF C-N pairing distance in angstrom.",
    )
    pb_dmf_parser.add_argument(
        "--shell-tolerance",
        type=float,
        default=0.12,
        help="Distance tolerance for grouping near-degenerate paths in angstrom.",
    )
    pb_dmf_parser.add_argument(
        "--bl-oc",
        type=float,
        default=1.25,
        help="Fixed O-C bond length used in the DMF geometry constraint.",
    )
    pb_dmf_parser.add_argument(
        "--bl-cn",
        type=float,
        default=1.35,
        help="Fixed C-N bond length used in the DMF geometry constraint.",
    )
    pb_dmf_parser.add_argument(
        "--theta-pboc",
        type=float,
        default=None,
        help=(
            "Initial mean Pb-O-C angle in degrees. Defaults to the mean "
            "angle in the selected structure."
        ),
    )
    pb_dmf_parser.add_argument(
        "--theta-ocn",
        type=float,
        default=None,
        help=(
            "Initial mean O-C-N angle in degrees. Defaults to the mean "
            "angle in the selected structure."
        ),
    )
    pb_dmf_parser.add_argument(
        "--phi-pbocn",
        type=float,
        default=None,
        help=(
            "Initial mean Pb-O-C-N dihedral in degrees. Defaults to the "
            "circular mean in the selected structure."
        ),
    )
    pb_dmf_parser.add_argument(
        "--theta-width",
        type=float,
        default=8.0,
        help="Initial Pb-O-C angular width in degrees.",
    )
    pb_dmf_parser.add_argument(
        "--internal-angle-width",
        type=float,
        default=6.0,
        help="Initial O-C-N angular width in degrees.",
    )
    pb_dmf_parser.add_argument(
        "--dihedral-width",
        type=float,
        default=12.0,
        help="Initial Pb-O-C-N dihedral width in degrees.",
    )
    pb_dmf_parser.add_argument(
        "--vary-snot",
        action="store_true",
        help="Emit snot as a guess row instead of a fixed set row.",
    )
    pb_dmf_parser.add_argument(
        "--no-link-oxygen-sigma2",
        action="store_true",
        help="Fit separate sigma2 values for each Pb-O path.",
    )
    pb_dmf_parser.add_argument(
        "--no-restraints",
        action="store_true",
        help="Do not emit soft restrain rows.",
    )
    pb_dmf_parser.set_defaults(handler=_handle_build_pb_dmf)

    validate_parser = subparsers.add_parser(
        "validate",
        help=(
            "Validate a generated Artemis GDS file without launching "
            "Artemis."
        ),
    )
    validate_parser.add_argument("gds_file", type=Path)
    validate_parser.set_defaults(handler=_handle_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"exafsgds {__version__}")
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 2


def _handle_build(args: argparse.Namespace) -> int:
    settings = ArtemisGDSBuildSettings(
        absorber_element=args.absorber_element,
        absorber_atom_index=args.absorber_atom_index,
        min_distance_angstrom=args.min_distance,
        max_distance_angstrom=args.max_distance,
        shell_tolerance_angstrom=args.shell_tolerance,
        include_restraints=not bool(args.no_restraints),
    )
    document = build_artemis_gds_for_structure(args.structure, settings)
    output_path = write_artemis_gds_file(args.output, document)
    report = validate_artemis_gds_file(output_path)
    print(f"Wrote: {output_path}")
    print(f"Overview: {artemis_gds_overview_path(output_path)}")
    print(f"Shells: {len(document.shells)}")
    print(report.summary_text())
    return 0 if report.is_valid else 1


def _handle_build_geometry(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Geometry constraint spec must be a JSON object.")
    document = build_geometric_constraint_gds(
        args.structure,
        independent_paths=tuple(
            _independent_path_spec_from_dict(entry)
            for entry in payload.get("independent_paths", [])
        ),
        coordination_groups=tuple(
            _coordination_group_spec_from_dict(entry)
            for entry in payload.get("coordination_groups", [])
        ),
        three_atom_constraints=tuple(
            _three_atom_constraint_spec_from_dict(entry)
            for entry in payload.get("three_atom_constraints", [])
        ),
        four_atom_dihedral_constraints=tuple(
            _four_atom_dihedral_constraint_spec_from_dict(entry)
            for entry in payload.get("four_atom_dihedral_constraints", [])
        ),
        settings=_geometric_settings_from_dict(payload.get("settings", {})),
    )
    output_path = write_artemis_gds_file(args.output, document)
    report = validate_artemis_gds_file(output_path)
    print(f"Wrote: {output_path}")
    print(f"Overview: {artemis_gds_overview_path(output_path)}")
    print(f"Path hints: {len(document.path_hints)}")
    print(report.summary_text())
    return 0 if report.is_valid else 1


def _handle_build_pb_dmso(args: argparse.Namespace) -> int:
    settings = PbDMSOGDSBuildSettings(
        absorber_atom_index=args.absorber_atom_index,
        oxygen_count=args.oxygen_count,
        iodide_count=args.iodide_count,
        max_oxygen_distance_angstrom=args.max_oxygen_distance,
        max_iodide_distance_angstrom=args.max_iodide_distance,
        max_os_distance_angstrom=args.max_os_distance,
        shell_tolerance_angstrom=args.shell_tolerance,
        bl_os_angstrom=args.bl_os,
        theta_pbos_degrees=args.theta_pbos,
        angle_width_degrees=args.angle_width,
        vary_s02=bool(args.vary_snot),
        link_oxygen_sigma2=not bool(args.no_link_oxygen_sigma2),
        include_restraints=not bool(args.no_restraints),
    )
    document = build_pb_dmso_gds_from_structure(args.structure, settings)
    output_path = write_artemis_gds_file(args.output, document)
    report = validate_artemis_gds_file(output_path)
    print(f"Wrote: {output_path}")
    print(f"Overview: {artemis_gds_overview_path(output_path)}")
    print(f"Path hints: {len(document.path_hints)}")
    print(report.summary_text())
    return 0 if report.is_valid else 1


def _handle_build_pb_dmf(args: argparse.Namespace) -> int:
    settings = PbDMFGDSBuildSettings(
        absorber_atom_index=args.absorber_atom_index,
        oxygen_count=args.oxygen_count,
        iodide_count=args.iodide_count,
        max_oxygen_distance_angstrom=args.max_oxygen_distance,
        max_iodide_distance_angstrom=args.max_iodide_distance,
        max_oc_distance_angstrom=args.max_oc_distance,
        max_cn_distance_angstrom=args.max_cn_distance,
        shell_tolerance_angstrom=args.shell_tolerance,
        bl_oc_angstrom=args.bl_oc,
        bl_cn_angstrom=args.bl_cn,
        theta_pboc_degrees=args.theta_pboc,
        theta_ocn_degrees=args.theta_ocn,
        phi_pbocn_degrees=args.phi_pbocn,
        theta_width_degrees=args.theta_width,
        internal_angle_width_degrees=args.internal_angle_width,
        dihedral_width_degrees=args.dihedral_width,
        vary_s02=bool(args.vary_snot),
        link_oxygen_sigma2=not bool(args.no_link_oxygen_sigma2),
        include_restraints=not bool(args.no_restraints),
    )
    document = build_pb_dmf_gds_from_structure(args.structure, settings)
    output_path = write_artemis_gds_file(args.output, document)
    report = validate_artemis_gds_file(output_path)
    print(f"Wrote: {output_path}")
    print(f"Overview: {artemis_gds_overview_path(output_path)}")
    print(f"Path hints: {len(document.path_hints)}")
    print(report.summary_text())
    return 0 if report.is_valid else 1


def _handle_validate(args: argparse.Namespace) -> int:
    report = validate_artemis_gds_file(args.gds_file)
    print(report.summary_text())
    return 0 if report.is_valid else 1


def _geometric_settings_from_dict(
    payload: object,
) -> GeometricGDSBuildSettings:
    source = dict(payload) if isinstance(payload, dict) else {}
    return GeometricGDSBuildSettings(
        initial_s02=float(source.get("initial_s02", 0.9)),
        initial_e0=float(source.get("initial_e0", 0.0)),
        default_initial_delta_r=float(
            source.get("default_initial_delta_r", 0.0)
        ),
        default_initial_sigma2=float(
            source.get("default_initial_sigma2", 0.003)
        ),
    )


def _coordination_group_spec_from_dict(
    payload: object,
) -> CoordinationGroupSpec:
    source = dict(payload) if isinstance(payload, dict) else {}
    return CoordinationGroupSpec(
        label=str(source["label"]),
        initial_value=float(source["initial_value"]),
        lower_bound=(
            float(source["lower_bound"])
            if source.get("lower_bound") is not None
            else None
        ),
        upper_bound=(
            float(source["upper_bound"])
            if source.get("upper_bound") is not None
            else None
        ),
        restraint_scale=float(source.get("restraint_scale", 1000.0)),
        vary=bool(source.get("vary", True)),
    )


def _independent_path_spec_from_dict(payload: object) -> IndependentPathSpec:
    source = dict(payload) if isinstance(payload, dict) else {}
    return IndependentPathSpec(
        label=str(source["label"]),
        absorber_atom_index=int(source["absorber_atom_index"]),
        scatterer_atom_index=int(source["scatterer_atom_index"]),
        scatterer_element=(
            str(source["scatterer_element"])
            if source.get("scatterer_element") is not None
            else None
        ),
        initial_delta_r=float(source.get("initial_delta_r", 0.0)),
        initial_sigma2=float(source.get("initial_sigma2", 0.003)),
        multiplicity=float(source.get("multiplicity", 1.0)),
        coordination_group=(
            str(source["coordination_group"])
            if source.get("coordination_group") is not None
            else None
        ),
        coordination_fraction=(
            float(source["coordination_fraction"])
            if source.get("coordination_fraction") is not None
            else None
        ),
        reference_multiplicity=(
            float(source["reference_multiplicity"])
            if source.get("reference_multiplicity") is not None
            else None
        ),
    )


def _three_atom_constraint_spec_from_dict(
    payload: object,
) -> ThreeAtomConstraintSpec:
    source = dict(payload) if isinstance(payload, dict) else {}
    return ThreeAtomConstraintSpec(
        label=str(source["label"]),
        absorber_atom_index=int(source["absorber_atom_index"]),
        bridge_atom_index=int(source["bridge_atom_index"]),
        terminal_atom_index=int(source["terminal_atom_index"]),
        fixed_bridge_terminal_distance_angstrom=float(
            source["fixed_bridge_terminal_distance_angstrom"]
        ),
        angle_mean_degrees=float(source["angle_mean_degrees"]),
        angle_sigma_degrees=float(source.get("angle_sigma_degrees", 0.0)),
        anchor_delta_r_name=(
            str(source["anchor_delta_r_name"])
            if source.get("anchor_delta_r_name") is not None
            else None
        ),
        anchor_sigma2_name=(
            str(source["anchor_sigma2_name"])
            if source.get("anchor_sigma2_name") is not None
            else None
        ),
        include_multiple_scattering=bool(
            source.get("include_multiple_scattering", True)
        ),
        terminal_multiplicity=float(source.get("terminal_multiplicity", 1.0)),
        multiple_scattering_multiplicity=float(
            source.get("multiple_scattering_multiplicity", 1.0)
        ),
        coordination_group=(
            str(source["coordination_group"])
            if source.get("coordination_group") is not None
            else None
        ),
        coordination_fraction=(
            float(source["coordination_fraction"])
            if source.get("coordination_fraction") is not None
            else None
        ),
        anchor_reference_multiplicity=(
            float(source["anchor_reference_multiplicity"])
            if source.get("anchor_reference_multiplicity") is not None
            else None
        ),
        terminal_reference_multiplicity=(
            float(source["terminal_reference_multiplicity"])
            if source.get("terminal_reference_multiplicity") is not None
            else None
        ),
        multiple_scattering_reference_multiplicity=(
            float(source["multiple_scattering_reference_multiplicity"])
            if source.get("multiple_scattering_reference_multiplicity")
            is not None
            else None
        ),
    )


def _four_atom_dihedral_constraint_spec_from_dict(
    payload: object,
) -> FourAtomDihedralConstraintSpec:
    source = dict(payload) if isinstance(payload, dict) else {}
    return FourAtomDihedralConstraintSpec(
        label=str(source["label"]),
        absorber_atom_index=int(source["absorber_atom_index"]),
        bridge_atom_index=int(source["bridge_atom_index"]),
        hinge_atom_index=int(source["hinge_atom_index"]),
        terminal_atom_index=int(source["terminal_atom_index"]),
        fixed_bridge_hinge_distance_angstrom=float(
            source["fixed_bridge_hinge_distance_angstrom"]
        ),
        fixed_hinge_terminal_distance_angstrom=float(
            source["fixed_hinge_terminal_distance_angstrom"]
        ),
        angle_abc_mean_degrees=float(source["angle_abc_mean_degrees"]),
        angle_bcd_mean_degrees=float(source["angle_bcd_mean_degrees"]),
        dihedral_mean_degrees=float(source["dihedral_mean_degrees"]),
        angle_abc_sigma_degrees=float(
            source.get("angle_abc_sigma_degrees", 0.0)
        ),
        angle_bcd_sigma_degrees=float(
            source.get("angle_bcd_sigma_degrees", 0.0)
        ),
        dihedral_sigma_degrees=float(
            source.get("dihedral_sigma_degrees", 0.0)
        ),
        anchor_delta_r_name=(
            str(source["anchor_delta_r_name"])
            if source.get("anchor_delta_r_name") is not None
            else None
        ),
        anchor_sigma2_name=(
            str(source["anchor_sigma2_name"])
            if source.get("anchor_sigma2_name") is not None
            else None
        ),
        terminal_multiplicity=float(source.get("terminal_multiplicity", 1.0)),
        coordination_group=(
            str(source["coordination_group"])
            if source.get("coordination_group") is not None
            else None
        ),
        coordination_fraction=(
            float(source["coordination_fraction"])
            if source.get("coordination_fraction") is not None
            else None
        ),
        anchor_reference_multiplicity=(
            float(source["anchor_reference_multiplicity"])
            if source.get("anchor_reference_multiplicity") is not None
            else None
        ),
        terminal_reference_multiplicity=(
            float(source["terminal_reference_multiplicity"])
            if source.get("terminal_reference_multiplicity") is not None
            else None
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
