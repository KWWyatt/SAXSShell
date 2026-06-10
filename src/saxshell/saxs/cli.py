from __future__ import annotations

import argparse
from pathlib import Path

from saxshell.version import __version__

from ._model_templates import list_template_specs


def format_validation_report(result: object) -> str:
    from .template_installation import (
        format_validation_report as format_report,
    )

    return format_report(result)


def install_template_candidate(*args: object, **kwargs: object) -> object:
    from .template_installation import install_template_candidate as install

    return install(*args, **kwargs)


def validate_template_candidate(*args: object, **kwargs: object) -> object:
    from .template_installation import validate_template_candidate as validate

    return validate(*args, **kwargs)


def run_dream_batch_manifest(manifest_path: Path) -> object:
    from .dream.batch import run_dream_batch_manifest as run_manifest

    return run_manifest(manifest_path)


def compute_direct_frame_saxs(*args: object, **kwargs: object) -> object:
    from .direct_frames import compute_direct_frame_saxs as compute

    return compute(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saxs",
        description=(
            "Set up SAXS projects, launch the SAXS Qt UI, inspect bundled "
            "model templates, and manage prefit/DREAM workflows."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the SAXS application version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the SAXS Qt UI.")
    ui_parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        help="Optional project directory to open on launch.",
    )
    ui_parser.set_defaults(handler=_handle_ui)

    templates_parser = subparsers.add_parser(
        "templates",
        help="List, validate, or install SAXS model templates.",
    )
    templates_parser.set_defaults(handler=_handle_templates)
    template_subparsers = templates_parser.add_subparsers(
        dest="templates_command"
    )

    validate_parser = template_subparsers.add_parser(
        "validate",
        help="Validate a candidate SAXS template for prefit and DREAM use.",
    )
    validate_parser.add_argument(
        "template_path",
        type=Path,
        help="Path to the candidate template Python file.",
    )
    validate_parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional paired metadata JSON file for display text/tooltips.",
    )
    validate_parser.set_defaults(handler=_handle_templates_validate)

    install_parser = template_subparsers.add_parser(
        "install",
        help="Validate and install a SAXS template into a template directory.",
    )
    install_parser.add_argument(
        "template_path",
        type=Path,
        help="Path to the candidate template Python file.",
    )
    install_parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional paired metadata JSON file for display text/tooltips.",
    )
    install_parser.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help=(
            "Destination template directory. Defaults to the bundled "
            "template folder."
        ),
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing installed template with the same name.",
    )
    install_parser.set_defaults(handler=_handle_templates_install)

    dream_batch_parser = subparsers.add_parser(
        "dream-batch",
        help="Run or inspect generated SAXS DREAM backend batch run sets.",
    )
    dream_batch_parser.set_defaults(handler=_handle_dream_batch)
    dream_batch_subparsers = dream_batch_parser.add_subparsers(
        dest="dream_batch_command"
    )
    dream_batch_run_parser = dream_batch_subparsers.add_parser(
        "run",
        help="Execute a generated DREAM backend batch manifest.",
    )
    dream_batch_run_parser.add_argument(
        "manifest_path",
        type=Path,
        help="Path to dream_backend_run_set.json.",
    )
    dream_batch_run_parser.set_defaults(handler=_handle_dream_batch_run)
    dream_batch_setup_parser = dream_batch_subparsers.add_parser(
        "setup-ui",
        help="Launch the DREAM backend batch CLI setup window.",
    )
    dream_batch_setup_parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        help="Optional SAXSShell project directory to open.",
    )
    dream_batch_setup_parser.set_defaults(handler=_handle_dream_batch_setup_ui)

    direct_frame_parser = subparsers.add_parser(
        "direct-frame-saxs",
        help=(
            "Beta: compute direct Debye SAXS from PDB/XYZ frame(s), average "
            "the traces, and write finite-box diagnostics."
        ),
    )
    direct_frame_parser.add_argument(
        "input_path",
        type=Path,
        help="PDB/XYZ structure file or directory of PDB/XYZ frames.",
    )
    direct_frame_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV, JSON, notes, and figure outputs.",
    )
    direct_frame_parser.add_argument(
        "--q-min",
        type=float,
        default=0.01,
        help="Minimum q value in A^-1. Default: 0.01.",
    )
    direct_frame_parser.add_argument(
        "--q-max",
        type=float,
        default=2.0,
        help="Maximum q value in A^-1. Default: 2.0.",
    )
    direct_frame_parser.add_argument(
        "--q-step",
        type=float,
        default=0.01,
        help="q spacing in A^-1. Default: 0.01.",
    )
    direct_frame_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for testing a prefix of naturally sorted frames.",
    )
    direct_frame_parser.add_argument(
        "--box-length",
        type=float,
        default=None,
        help=(
            "Optional simulation box length in A for low-q diagnostics. "
            "Defaults to the first frame's maximum coordinate span."
        ),
    )
    direct_frame_parser.add_argument(
        "--box-dimensions",
        type=float,
        nargs=3,
        metavar=("LX", "LY", "LZ"),
        default=None,
        help=(
            "Optional periodic box dimensions in A. Required for "
            "--subtract-average-box-density."
        ),
    )
    direct_frame_parser.add_argument(
        "--subtract-average-box-density",
        action="store_true",
        help=(
            "Subtract a uniform medium amplitude using the average electron "
            "density of the supplied box before spherical averaging."
        ),
    )
    direct_frame_parser.add_argument(
        "--direction-count",
        type=int,
        default=512,
        help=(
            "Number of spherical directions for average-box-density contrast "
            "mode. Default: 512."
        ),
    )
    direct_frame_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    direct_frame_parser.add_argument(
        "--experimental-project",
        type=Path,
        default=None,
        help=(
            "Optional SAXSShell project folder whose active experimental "
            "data will be overlaid with log-log scaling."
        ),
    )
    direct_frame_parser.add_argument(
        "--experimental-data",
        type=Path,
        default=None,
        help=(
            "Optional experimental q/I text file to overlay. If both this "
            "and --experimental-project are provided, this file is used."
        ),
    )
    direct_frame_parser.add_argument(
        "--scale-fit-q-min",
        type=float,
        default=None,
        help=(
            "Optional lower q bound for log-space scale fitting. For "
            "average-box-density contrast overlays, the default is 2*pi/L."
        ),
    )
    direct_frame_parser.add_argument(
        "--scale-fit-q-max",
        type=float,
        default=None,
        help="Optional upper q bound for log-space scale fitting.",
    )
    direct_frame_parser.set_defaults(handler=_handle_direct_frame_saxs)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"saxs {__version__}")
        return 0

    if args.command is None:
        return _handle_ui(args)

    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


def _handle_ui(args: argparse.Namespace) -> int:
    from .ui.main_window import launch_saxs_ui

    return launch_saxs_ui(getattr(args, "project_dir", None))


def _handle_templates(_args: argparse.Namespace) -> int:
    for template in list_template_specs():
        print(f"{template.display_name} ({template.name})")
    return 0


def _handle_templates_validate(args: argparse.Namespace) -> int:
    result = validate_template_candidate(
        args.template_path,
        metadata_path=args.metadata,
    )
    print(format_validation_report(result))
    return 0 if result.passed else 1


def _handle_templates_install(args: argparse.Namespace) -> int:
    installed = install_template_candidate(
        args.template_path,
        metadata_path=args.metadata,
        destination_dir=args.template_dir,
        overwrite=bool(args.force),
    )
    print(format_validation_report(installed.validation_result))
    print(f"Installed template: {installed.installed_template_path}")
    if installed.installed_metadata_path is not None:
        print(f"Installed metadata: {installed.installed_metadata_path}")
    return 0


def _handle_dream_batch(args: argparse.Namespace) -> int:
    if getattr(args, "dream_batch_command", None) is None:
        raise ValueError("Choose a dream-batch subcommand.")
    return int(args.handler(args))


def _handle_dream_batch_run(args: argparse.Namespace) -> int:
    run_dream_batch_manifest(args.manifest_path)
    return 0


def _handle_dream_batch_setup_ui(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.dream_batch_window import launch_dream_batch_run_file_ui

    owns_app = QApplication.instance() is None
    launch_dream_batch_run_file_ui(
        initial_project_dir=getattr(args, "project_dir", None)
    )
    app = QApplication.instance()
    if owns_app and app is not None:
        return int(app.exec())
    return 0


def _handle_direct_frame_saxs(args: argparse.Namespace) -> int:
    result = compute_direct_frame_saxs(
        args.input_path,
        output_dir=args.output_dir,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_step=float(args.q_step),
        max_frames=args.max_frames,
        box_length_a=args.box_length,
        box_lengths_a=(
            None
            if args.box_dimensions is None
            else tuple(float(value) for value in args.box_dimensions)
        ),
        subtract_average_box_density=bool(args.subtract_average_box_density),
        direction_count=int(args.direction_count),
        write_plots=not bool(args.no_plots),
        experimental_project_dir=args.experimental_project,
        experimental_data_path=args.experimental_data,
        scale_fit_q_min=args.scale_fit_q_min,
        scale_fit_q_max=args.scale_fit_q_max,
    )
    print(f"Computed {len(result.frame_paths)} frame(s).")
    print(f"Profile: {result.profile_csv_path}")
    print(f"Profile text: {result.profile_txt_path}")
    print(f"Frame traces: {result.frame_trace_csv_path}")
    print(f"Metadata: {result.metadata_json_path}")
    print(f"Method notes: {result.method_notes_path}")
    print(f"Calculation mode: {result.calculation_mode}")
    if result.medium_density_e_per_a3 is not None:
        print(
            "Average medium density: "
            f"{result.medium_density_e_per_a3:.8g} e/A^3"
        )
    for figure_path in result.figures:
        print(f"Figure: {figure_path}")
    overlay = getattr(result, "experimental_overlay", None)
    if overlay is not None:
        print(f"Experimental overlay: {overlay.plot_path}")
        print(f"Overlay CSV: {overlay.scaled_profile_csv_path}")
        print(
            "Overlay log-scale factor: "
            f"{overlay.scale_factor:.8g} over "
            f"{overlay.fit_q_min_a_inverse:.6g}-"
            f"{overlay.fit_q_max_a_inverse:.6g} A^-1 "
            f"({overlay.fit_point_count} points)."
        )
    print(
        "Finite-box diagnostic: 2*pi/L = "
        f"{result.diagnostics.q_fundamental_a_inverse:.6g} A^-1; "
        f"{result.diagnostics.q_points_below_fundamental} q point(s) below."
    )
    return 0


__all__ = ["build_parser", "main"]
