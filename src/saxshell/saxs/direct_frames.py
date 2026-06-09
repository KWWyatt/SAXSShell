from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from saxshell.saxs.born_refinement.backend import build_shared_q_grid
from saxshell.saxs.debye.profiles import (
    build_f0_dictionary,
    compute_debye_intensity,
    load_structure_file,
)

DIRECT_FRAME_SAXS_OUTPUT_DIRNAME = "direct_frame_saxs_beta"
DIRECT_FRAME_STRUCTURE_SUFFIXES = frozenset({".pdb", ".xyz"})


@dataclass(slots=True, frozen=True)
class FiniteBoxDiagnostics:
    coordinate_span_a: tuple[float, float, float]
    inferred_box_length_a: float
    q_fundamental_a_inverse: float
    q_caution_a_inverse: float
    q_min_a_inverse: float
    q_max_a_inverse: float
    q_points_below_fundamental: int
    q_points_below_caution: int
    finite_box_note: str


@dataclass(slots=True, frozen=True)
class DirectFrameSAXSSettings:
    input_path: Path
    output_dir: Path
    q_min: float
    q_max: float
    q_step: float
    max_frames: int | None = None
    box_length_a: float | None = None
    box_lengths_a: tuple[float, float, float] | None = None
    subtract_average_box_density: bool = False
    direction_count: int = 512


@dataclass(slots=True, frozen=True)
class ExperimentalOverlayResult:
    experimental_data_path: Path
    plot_path: Path
    scaled_profile_csv_path: Path
    scale_factor: float
    fit_q_min_a_inverse: float
    fit_q_max_a_inverse: float
    fit_point_count: int
    log_rms_residual: float


@dataclass(slots=True, frozen=True)
class DirectFrameSAXSResult:
    settings: DirectFrameSAXSSettings
    frame_paths: tuple[Path, ...]
    q_values: np.ndarray
    frame_intensities: np.ndarray
    mean_intensity: np.ndarray
    variance_intensity: np.ndarray
    std_intensity: np.ndarray
    se_intensity: np.ndarray
    element_counts: dict[str, int]
    diagnostics: FiniteBoxDiagnostics
    calculation_mode: str
    medium_density_e_per_a3: float | None
    profile_csv_path: Path
    profile_txt_path: Path
    frame_trace_csv_path: Path
    metadata_json_path: Path
    method_notes_path: Path
    figures: tuple[Path, ...]
    experimental_overlay: ExperimentalOverlayResult | None = None


def discover_xyz_frame_paths(
    input_path: str | Path,
    *,
    max_frames: int | None = None,
) -> tuple[Path, ...]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in DIRECT_FRAME_STRUCTURE_SUFFIXES:
            raise ValueError(
                "Input structure must be a PDB or XYZ file: " f"{path.suffix}"
            )
        frames = (path,)
    elif path.is_dir():
        frames = tuple(
            sorted(
                (
                    candidate.resolve()
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                    and candidate.suffix.lower()
                    in DIRECT_FRAME_STRUCTURE_SUFFIXES
                ),
                key=lambda candidate: _structure_sort_key(path, candidate),
            )
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if not frames:
        raise ValueError(f"No PDB or XYZ frames were found in {path}")
    if max_frames is not None:
        if int(max_frames) < 1:
            raise ValueError("max_frames must be at least 1 when provided.")
        frames = frames[: int(max_frames)]
    return tuple(frames)


def build_direct_frame_q_grid(
    q_min: float,
    q_max: float,
    *,
    q_step: float,
) -> np.ndarray:
    return build_shared_q_grid(
        float(q_min),
        float(q_max),
        q_step=float(q_step),
    )


def estimate_finite_box_diagnostics(
    coordinates: np.ndarray,
    q_values: np.ndarray,
    *,
    box_length_a: float | None = None,
) -> FiniteBoxDiagnostics:
    coords = np.asarray(coordinates, dtype=float)
    q_grid = np.asarray(q_values, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError("coordinates must be a non-empty Nx3 array.")
    if q_grid.ndim != 1 or q_grid.size == 0:
        raise ValueError("q_values must be a non-empty one-dimensional array.")

    span = np.ptp(coords, axis=0)
    inferred_length = (
        float(box_length_a)
        if box_length_a is not None
        else float(np.max(span))
    )
    if inferred_length <= 0.0:
        raise ValueError("A positive box length is required for diagnostics.")
    q_fundamental = float(2.0 * np.pi / inferred_length)
    q_caution = float(2.0 * q_fundamental)
    q_min = float(np.min(q_grid))
    q_max = float(np.max(q_grid))
    below_fundamental = int(np.count_nonzero(q_grid < q_fundamental))
    below_caution = int(np.count_nonzero(q_grid < q_caution))
    if box_length_a is None:
        note = (
            "No explicit box length was supplied; diagnostics use the maximum "
            "coordinate span of the first parsed frame as a conservative "
            "finite sample length."
        )
    else:
        note = (
            "Diagnostics use the explicit box length supplied by the user. "
            "Low-q points near and below 2*pi/L should be treated as beta "
            "screening output until periodic/RDF corrections are added."
        )
    return FiniteBoxDiagnostics(
        coordinate_span_a=tuple(float(value) for value in span),
        inferred_box_length_a=float(inferred_length),
        q_fundamental_a_inverse=q_fundamental,
        q_caution_a_inverse=q_caution,
        q_min_a_inverse=q_min,
        q_max_a_inverse=q_max,
        q_points_below_fundamental=below_fundamental,
        q_points_below_caution=below_caution,
        finite_box_note=note,
    )


def compute_average_box_contrast_intensity(
    coordinates: np.ndarray,
    elements: list[str],
    q_values: np.ndarray,
    *,
    box_lengths_a: tuple[float, float, float],
    f0_dictionary: dict[str, np.ndarray],
    medium_density_e_per_a3: float | None = None,
    direction_count: int = 512,
    q_chunk_size: int = 8,
) -> tuple[np.ndarray, float]:
    q_grid = np.asarray(q_values, dtype=float)
    coords = _wrap_coordinates_into_centered_box(coordinates, box_lengths_a)
    box_lengths = np.asarray(box_lengths_a, dtype=float)
    directions = _fibonacci_sphere_directions(direction_count)
    projections = np.asarray(coords @ directions.T, dtype=float)
    density = (
        _average_electron_density_e_per_a3(elements, box_lengths_a)
        if medium_density_e_per_a3 is None
        else float(medium_density_e_per_a3)
    )
    f0_matrix = np.vstack(
        [
            f0_dictionary.get(
                element,
                np.zeros_like(q_grid, dtype=float),
            )
            for element in elements
        ]
    )
    intensity = np.zeros_like(q_grid, dtype=float)
    chunk_size = max(int(q_chunk_size), 1)
    for q_start in range(0, q_grid.size, chunk_size):
        q_stop = min(q_start + chunk_size, q_grid.size)
        q_chunk = q_grid[q_start:q_stop]
        phase = (
            q_chunk[:, np.newaxis, np.newaxis]
            * projections[
                np.newaxis,
                :,
                :,
            ]
        )
        atomic_amplitude = np.sum(
            f0_matrix[:, q_start:q_stop].T[:, :, np.newaxis]
            * np.exp(1j * phase),
            axis=1,
        )
        box_amplitude = density * _centered_box_amplitude(
            q_chunk,
            directions,
            box_lengths,
        )
        contrast_amplitude = atomic_amplitude - box_amplitude
        intensity[q_start:q_stop] = np.mean(
            np.square(np.abs(contrast_amplitude)),
            axis=1,
            dtype=float,
        )
    return np.asarray(intensity, dtype=float), float(density)


def compute_direct_frame_saxs(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    q_min: float = 0.01,
    q_max: float = 2.0,
    q_step: float = 0.01,
    max_frames: int | None = None,
    box_length_a: float | None = None,
    box_lengths_a: tuple[float, float, float] | None = None,
    subtract_average_box_density: bool = False,
    direction_count: int = 512,
    write_plots: bool = True,
    experimental_project_dir: str | Path | None = None,
    experimental_data_path: str | Path | None = None,
    scale_fit_q_min: float | None = None,
    scale_fit_q_max: float | None = None,
) -> DirectFrameSAXSResult:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    frame_paths = discover_xyz_frame_paths(input_path, max_frames=max_frames)
    q_values = build_direct_frame_q_grid(q_min, q_max, q_step=q_step)
    normalized_box_lengths = _normalized_box_lengths(box_lengths_a)
    if subtract_average_box_density and normalized_box_lengths is None:
        raise ValueError(
            "box_lengths_a is required when subtract_average_box_density is "
            "enabled."
        )
    diagnostic_box_length = (
        box_length_a
        if box_length_a is not None
        else (
            max(normalized_box_lengths)
            if normalized_box_lengths is not None
            else None
        )
    )

    loaded_frames: list[tuple[np.ndarray, list[str]]] = []
    all_elements: list[str] = []
    element_counts: dict[str, int] = {}
    for frame_path in frame_paths:
        coordinates, elements = load_structure_file(frame_path)
        loaded_frames.append((coordinates, elements))
        all_elements.extend(elements)
        for element in elements:
            element_counts[element] = element_counts.get(element, 0) + 1

    first_coordinates, _first_elements = loaded_frames[0]
    diagnostics = estimate_finite_box_diagnostics(
        first_coordinates,
        q_values,
        box_length_a=diagnostic_box_length,
    )
    f0_dictionary = build_f0_dictionary(all_elements, q_values)
    traces: list[np.ndarray] = []
    medium_densities: list[float] = []
    if subtract_average_box_density:
        assert normalized_box_lengths is not None
        for coordinates, elements in loaded_frames:
            trace, density = compute_average_box_contrast_intensity(
                coordinates,
                elements,
                q_values,
                box_lengths_a=normalized_box_lengths,
                f0_dictionary=f0_dictionary,
                direction_count=direction_count,
            )
            traces.append(np.asarray(trace, dtype=float))
            medium_densities.append(float(density))
        calculation_mode = "average_box_density_contrast"
        medium_density = float(np.mean(np.asarray(medium_densities)))
    else:
        traces = [
            np.asarray(
                compute_debye_intensity(
                    coordinates,
                    elements,
                    q_values,
                    f0_dictionary=f0_dictionary,
                ),
                dtype=float,
            )
            for coordinates, elements in loaded_frames
        ]
        calculation_mode = "finite_frame_atomic_debye_vacuum"
        medium_density = None
    stacked = np.asarray(traces, dtype=float)
    mean_intensity = np.mean(stacked, axis=0)
    variance_intensity = np.var(stacked, axis=0)
    std_intensity = np.sqrt(variance_intensity)
    se_intensity = std_intensity / np.sqrt(float(stacked.shape[0]))

    settings = DirectFrameSAXSSettings(
        input_path=Path(input_path).expanduser().resolve(),
        output_dir=output_path,
        q_min=float(q_min),
        q_max=float(q_max),
        q_step=float(q_step),
        max_frames=max_frames,
        box_length_a=box_length_a,
        box_lengths_a=normalized_box_lengths,
        subtract_average_box_density=bool(subtract_average_box_density),
        direction_count=max(int(direction_count), 1),
    )
    profile_csv_path = _write_profile_csv(
        output_path,
        q_values=q_values,
        mean_intensity=mean_intensity,
        variance_intensity=variance_intensity,
        std_intensity=std_intensity,
        se_intensity=se_intensity,
        diagnostics=diagnostics,
    )
    profile_txt_path = _write_profile_txt(
        output_path,
        q_values=q_values,
        mean_intensity=mean_intensity,
        variance_intensity=variance_intensity,
        std_intensity=std_intensity,
        se_intensity=se_intensity,
        diagnostics=diagnostics,
    )
    frame_trace_csv_path = _write_frame_trace_csv(
        output_path,
        frame_paths=frame_paths,
        q_values=q_values,
        frame_intensities=stacked,
    )
    method_notes_path = _write_method_notes(
        output_path,
        settings=settings,
        diagnostics=diagnostics,
    )
    figures: tuple[Path, ...] = ()
    if write_plots:
        figures = _write_figures(
            output_path,
            frame_paths=frame_paths,
            q_values=q_values,
            frame_intensities=stacked,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            se_intensity=se_intensity,
            diagnostics=diagnostics,
        )

    experimental_overlay = None
    if (
        experimental_project_dir is not None
        or experimental_data_path is not None
    ):
        if not write_plots:
            raise ValueError(
                "Experimental overlays require plot generation. Remove "
                "--no-plots or omit the experimental overlay input."
            )
        effective_scale_fit_q_min = (
            diagnostics.q_fundamental_a_inverse
            if scale_fit_q_min is None and subtract_average_box_density
            else scale_fit_q_min
        )
        experimental_overlay = _write_experimental_overlay(
            output_path,
            q_values=q_values,
            mean_intensity=mean_intensity,
            diagnostics=diagnostics,
            experimental_project_dir=experimental_project_dir,
            experimental_data_path=experimental_data_path,
            scale_fit_q_min=effective_scale_fit_q_min,
            scale_fit_q_max=scale_fit_q_max,
        )

    metadata_json_path = _write_metadata_json(
        output_path,
        settings=settings,
        frame_paths=frame_paths,
        element_counts=element_counts,
        diagnostics=diagnostics,
        calculation_mode=calculation_mode,
        medium_density_e_per_a3=medium_density,
        experimental_overlay=experimental_overlay,
    )

    return DirectFrameSAXSResult(
        settings=settings,
        frame_paths=frame_paths,
        q_values=q_values,
        frame_intensities=stacked,
        mean_intensity=mean_intensity,
        variance_intensity=variance_intensity,
        std_intensity=std_intensity,
        se_intensity=se_intensity,
        element_counts=element_counts,
        diagnostics=diagnostics,
        calculation_mode=calculation_mode,
        medium_density_e_per_a3=medium_density,
        profile_csv_path=profile_csv_path,
        profile_txt_path=profile_txt_path,
        frame_trace_csv_path=frame_trace_csv_path,
        metadata_json_path=metadata_json_path,
        method_notes_path=method_notes_path,
        figures=figures,
        experimental_overlay=experimental_overlay,
    )


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
        if token
    ]


def _structure_sort_key(
    root: Path,
    path: Path,
) -> tuple[tuple[object, ...], ...]:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts
    return tuple(tuple(_natural_sort_key(part)) for part in relative_parts)


def _normalized_box_lengths(
    box_lengths_a: tuple[float, float, float] | list[float] | None,
) -> tuple[float, float, float] | None:
    if box_lengths_a is None:
        return None
    if len(box_lengths_a) != 3:
        raise ValueError("box_lengths_a must contain exactly three values.")
    normalized = tuple(float(value) for value in box_lengths_a)
    if any(value <= 0.0 for value in normalized):
        raise ValueError("All box lengths must be positive.")
    return normalized


def _wrap_coordinates_into_centered_box(
    coordinates: np.ndarray,
    box_lengths_a: tuple[float, float, float],
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    lengths = np.asarray(box_lengths_a, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must be an Nx3 array.")
    wrapped = np.mod(coords, lengths)
    return np.asarray(wrapped - 0.5 * lengths[np.newaxis, :], dtype=float)


def _average_electron_density_e_per_a3(
    elements: list[str],
    box_lengths_a: tuple[float, float, float],
) -> float:
    q_zero = np.asarray([0.0], dtype=float)
    f0_zero = build_f0_dictionary(elements, q_zero)
    total_electrons = float(
        sum(float(f0_zero[element][0]) for element in elements)
    )
    volume = float(np.prod(np.asarray(box_lengths_a, dtype=float)))
    if volume <= 0.0:
        raise ValueError("Box volume must be positive.")
    return float(total_electrons / volume)


def _fibonacci_sphere_directions(direction_count: int) -> np.ndarray:
    count = max(int(direction_count), 1)
    indices = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y_values = 1.0 - (2.0 * indices + 1.0) / float(count)
    radial_values = np.sqrt(np.clip(1.0 - y_values * y_values, 0.0, None))
    theta_values = golden_angle * indices
    x_values = np.cos(theta_values) * radial_values
    z_values = np.sin(theta_values) * radial_values
    return np.asarray(
        np.column_stack((x_values, y_values, z_values)),
        dtype=float,
    )


def _centered_box_amplitude(
    q_values: np.ndarray,
    directions: np.ndarray,
    box_lengths_a: np.ndarray,
) -> np.ndarray:
    q_grid = np.asarray(q_values, dtype=float)
    directions_array = np.asarray(directions, dtype=float)
    lengths = np.asarray(box_lengths_a, dtype=float)
    arguments = (
        q_grid[:, np.newaxis, np.newaxis]
        * directions_array[np.newaxis, :, :]
        * lengths[np.newaxis, np.newaxis, :]
        / (2.0 * np.pi)
    )
    axis_amplitudes = lengths[np.newaxis, np.newaxis, :] * np.sinc(arguments)
    return np.prod(axis_amplitudes, axis=2)


def _write_profile_csv(
    output_dir: Path,
    *,
    q_values: np.ndarray,
    mean_intensity: np.ndarray,
    variance_intensity: np.ndarray,
    std_intensity: np.ndarray,
    se_intensity: np.ndarray,
    diagnostics: FiniteBoxDiagnostics,
) -> Path:
    path = output_dir / "direct_frame_saxs_profile.csv"
    _write_profile_table(
        path,
        delimiter=",",
        q_values=q_values,
        mean_intensity=mean_intensity,
        variance_intensity=variance_intensity,
        std_intensity=std_intensity,
        se_intensity=se_intensity,
        diagnostics=diagnostics,
    )
    return path


def _write_profile_txt(
    output_dir: Path,
    *,
    q_values: np.ndarray,
    mean_intensity: np.ndarray,
    variance_intensity: np.ndarray,
    std_intensity: np.ndarray,
    se_intensity: np.ndarray,
    diagnostics: FiniteBoxDiagnostics,
) -> Path:
    path = output_dir / "direct_frame_saxs_profile.txt"
    _write_profile_table(
        path,
        delimiter="\t",
        q_values=q_values,
        mean_intensity=mean_intensity,
        variance_intensity=variance_intensity,
        std_intensity=std_intensity,
        se_intensity=se_intensity,
        diagnostics=diagnostics,
    )
    return path


def _write_profile_table(
    path: Path,
    *,
    delimiter: str,
    q_values: np.ndarray,
    mean_intensity: np.ndarray,
    variance_intensity: np.ndarray,
    std_intensity: np.ndarray,
    se_intensity: np.ndarray,
    diagnostics: FiniteBoxDiagnostics,
) -> None:
    std_array = np.asarray(std_intensity, dtype=float)
    mean_array = np.asarray(mean_intensity, dtype=float)
    low_q_flag = np.asarray(q_values < diagnostics.q_fundamental_a_inverse)
    caution_flag = np.asarray(q_values < diagnostics.q_caution_a_inverse)
    data = np.column_stack(
        (
            q_values,
            mean_array,
            variance_intensity,
            std_array,
            se_intensity,
            mean_array - std_array,
            mean_array + std_array,
            low_q_flag.astype(int),
            caution_flag.astype(int),
        )
    )
    header = delimiter.join(
        (
            "q_a_inverse",
            "mean_intensity",
            "variance_intensity",
            "std_intensity",
            "se_intensity",
            "variance_band_lower_intensity",
            "variance_band_upper_intensity",
            "below_2pi_over_l",
            "below_4pi_over_l",
        )
    )
    np.savetxt(
        path,
        data,
        delimiter=delimiter,
        header=header,
        comments="",
        fmt="%.10g",
    )


def _write_frame_trace_csv(
    output_dir: Path,
    *,
    frame_paths: tuple[Path, ...],
    q_values: np.ndarray,
    frame_intensities: np.ndarray,
) -> Path:
    path = output_dir / "direct_frame_saxs_frame_traces.csv"
    columns = [q_values]
    columns.extend(
        frame_intensities[index] for index in range(len(frame_paths))
    )
    header = ["q_a_inverse"]
    header.extend(
        _csv_column_name(frame_path.stem) for frame_path in frame_paths
    )
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.10g",
    )
    return path


def _write_metadata_json(
    output_dir: Path,
    *,
    settings: DirectFrameSAXSSettings,
    frame_paths: tuple[Path, ...],
    element_counts: dict[str, int],
    diagnostics: FiniteBoxDiagnostics,
    calculation_mode: str,
    medium_density_e_per_a3: float | None,
    experimental_overlay: ExperimentalOverlayResult | None = None,
) -> Path:
    path = output_dir / "direct_frame_saxs_metadata.json"
    payload = {
        "created_at": (
            datetime.now().astimezone().isoformat(timespec="seconds")
        ),
        "mode": "direct_frame_saxs_beta",
        "settings": _settings_payload(settings),
        "calculation_mode": calculation_mode,
        "medium_density_e_per_a3": medium_density_e_per_a3,
        "frame_count": len(frame_paths),
        "frames": [str(frame_path) for frame_path in frame_paths],
        "element_counts_total": dict(sorted(element_counts.items())),
        "finite_box_diagnostics": _diagnostics_payload(diagnostics),
        "experimental_overlay": (
            None
            if experimental_overlay is None
            else _experimental_overlay_payload(experimental_overlay)
        ),
        "literature_context": {
            "main_reference": (
                "Dohn et al., Journal of Chemical Physics 159, 124115 "
                "(2023), doi:10.1063/5.0164365"
            ),
            "implementation_status": (
                "This beta computes direct finite-frame Debye scattering. "
                "The optional average-box-density contrast mode subtracts a "
                "uniform medium amplitude from the finite box, but does not "
                "yet apply RDF finite-size renormalization, periodic-image "
                "corrections, solvent displaced-volume terms, or windowed "
                "generalized-Debye RDF transforms."
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_method_notes(
    output_dir: Path,
    *,
    settings: DirectFrameSAXSSettings,
    diagnostics: FiniteBoxDiagnostics,
) -> Path:
    path = output_dir / "method_notes.md"
    lines = [
        "# Direct Frame SAXS Beta",
        "",
        "This beta computes SAXS from each PDB or XYZ structure frame using "
        "the discrete Debye equation with q-dependent atomic form factors, "
        "then averages the frame traces.",
        "",
        "## Literature Context",
        "",
        "Dohn et al. (J. Chem. Phys. 159, 124115, 2023; "
        "doi:10.1063/5.0164365) show that MD-derived RDF scattering is "
        "sensitive to finite simulation-cell sampling, especially at low q. "
        "Their correction strategy renormalizes finite-sampled RDFs to "
        "recover an infinite-system low-q limit and separates solvent-shell "
        "and displaced-volume effects. GRSQ exposes the same family of RDF "
        "finite-size corrections and damping/window tools.",
        "",
        "## Current Prototype Scope",
        "",
        "- Direct Debye scattering from finite PDB/XYZ coordinates.",
        (
            "- Per-frame traces plus mean, standard deviation, and standard "
            "error."
        ),
        (
            "- The profile CSV/TXT files include variance at each q-value "
            "and the mean +/- standard-deviation band used for plot shading."
        ),
        (
            "- Experimental overlays use only a scalar log-space intensity "
            "scale; concentration/effective-density differences can change "
            "the SAXS shape and are not corrected here."
        ),
        (
            "- Low-q diagnostics based on 2*pi/L from the supplied or "
            "inferred box length."
        ),
        "",
        "## Not Yet Included",
        "",
        "- RDF finite-size renormalization.",
        "- Periodic image summation or minimum-image pair distances.",
        "- Explicit solvent displaced-volume correction beyond atomic form "
        "factors.",
        "- Windowed RDF/generalized-Debye transforms.",
        "",
        "## This Run",
        "",
        f"- Input: `{settings.input_path}`",
        f"- q range: {settings.q_min:.6g} to {settings.q_max:.6g} A^-1",
        f"- q step: {settings.q_step:.6g} A^-1",
        f"- diagnostic L: {diagnostics.inferred_box_length_a:.6g} A",
        (
            "- diagnostic 2*pi/L: "
            f"{diagnostics.q_fundamental_a_inverse:.6g} A^-1"
        ),
        (
            "- q points below 2*pi/L: "
            f"{diagnostics.q_points_below_fundamental}"
        ),
        (
            "- calculation mode: average box-density contrast"
            if settings.subtract_average_box_density
            else "- calculation mode: finite frame in vacuum"
        ),
        "",
        diagnostics.finite_box_note,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_experimental_overlay(
    output_dir: Path,
    *,
    q_values: np.ndarray,
    mean_intensity: np.ndarray,
    diagnostics: FiniteBoxDiagnostics,
    experimental_project_dir: str | Path | None,
    experimental_data_path: str | Path | None,
    scale_fit_q_min: float | None,
    scale_fit_q_max: float | None,
) -> ExperimentalOverlayResult:
    experimental_path, experimental_q, experimental_i = (
        _load_experimental_overlay_data(
            experimental_project_dir=experimental_project_dir,
            experimental_data_path=experimental_data_path,
        )
    )
    (
        fit_q,
        fit_experimental,
        fit_model,
        scale_factor,
        log_rms_residual,
    ) = _fit_log_scale_to_experimental(
        q_values,
        mean_intensity,
        experimental_q,
        experimental_i,
        scale_fit_q_min=scale_fit_q_min,
        scale_fit_q_max=scale_fit_q_max,
    )
    scaled_profile_csv_path = _write_experimental_overlay_csv(
        output_dir,
        fit_q=fit_q,
        fit_experimental=fit_experimental,
        fit_model=fit_model,
        scale_factor=scale_factor,
    )
    plot_path = _plot_experimental_overlay(
        output_dir,
        q_values=q_values,
        mean_intensity=mean_intensity,
        experimental_q=experimental_q,
        experimental_i=experimental_i,
        scale_factor=scale_factor,
        diagnostics=diagnostics,
    )
    return ExperimentalOverlayResult(
        experimental_data_path=experimental_path,
        plot_path=plot_path,
        scaled_profile_csv_path=scaled_profile_csv_path,
        scale_factor=scale_factor,
        fit_q_min_a_inverse=float(np.min(fit_q)),
        fit_q_max_a_inverse=float(np.max(fit_q)),
        fit_point_count=int(fit_q.size),
        log_rms_residual=log_rms_residual,
    )


def _load_experimental_overlay_data(
    *,
    experimental_project_dir: str | Path | None,
    experimental_data_path: str | Path | None,
) -> tuple[Path, np.ndarray, np.ndarray]:
    if experimental_data_path is not None:
        from saxshell.saxs.project_manager.project import (
            load_experimental_data_file,
        )

        summary = load_experimental_data_file(experimental_data_path)
        return (
            summary.path,
            np.asarray(summary.q_values, dtype=float),
            np.asarray(summary.intensities, dtype=float),
        )

    if experimental_project_dir is None:
        raise ValueError(
            "Either experimental_project_dir or experimental_data_path is "
            "required for an experimental overlay."
        )

    from saxshell.saxs.project_manager.project import (
        SAXSProjectManager,
        build_project_paths,
        load_experimental_data_file,
    )

    manager = SAXSProjectManager()
    settings = manager.load_project(experimental_project_dir)
    project_paths = build_project_paths(settings.project_dir)
    active_path = settings.resolved_experimental_data_path
    if active_path is None:
        raise ValueError(
            "The provided SAXS project does not define experimental data."
        )
    if not active_path.is_file():
        staged_path = project_paths.experimental_data_dir / active_path.name
        if staged_path.is_file():
            active_path = staged_path
    summary = load_experimental_data_file(
        active_path,
        skiprows=settings.experimental_header_rows,
        q_column=settings.experimental_q_column,
        intensity_column=settings.experimental_intensity_column,
        error_column=settings.experimental_error_column,
    )
    return (
        summary.path,
        np.asarray(summary.q_values, dtype=float),
        np.asarray(summary.intensities, dtype=float),
    )


def _fit_log_scale_to_experimental(
    model_q: np.ndarray,
    model_i: np.ndarray,
    experimental_q: np.ndarray,
    experimental_i: np.ndarray,
    *,
    scale_fit_q_min: float | None,
    scale_fit_q_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    model_q = np.asarray(model_q, dtype=float)
    model_i = np.asarray(model_i, dtype=float)
    experimental_q = np.asarray(experimental_q, dtype=float)
    experimental_i = np.asarray(experimental_i, dtype=float)
    positive_model = (
        np.isfinite(model_q)
        & np.isfinite(model_i)
        & (model_q > 0.0)
        & (model_i > 0.0)
    )
    positive_experimental = (
        np.isfinite(experimental_q)
        & np.isfinite(experimental_i)
        & (experimental_q > 0.0)
        & (experimental_i > 0.0)
    )
    if np.count_nonzero(positive_model) < 2:
        raise ValueError("At least two positive model points are required.")
    fit_q = experimental_q[positive_experimental]
    fit_experimental = experimental_i[positive_experimental]
    overlap_min = float(np.min(model_q[positive_model]))
    overlap_max = float(np.max(model_q[positive_model]))
    if scale_fit_q_min is not None:
        overlap_min = max(overlap_min, float(scale_fit_q_min))
    if scale_fit_q_max is not None:
        overlap_max = min(overlap_max, float(scale_fit_q_max))
    overlap_mask = (fit_q >= overlap_min) & (fit_q <= overlap_max)
    fit_q = fit_q[overlap_mask]
    fit_experimental = fit_experimental[overlap_mask]
    if fit_q.size < 2:
        raise ValueError(
            "At least two positive experimental points are required in the "
            "overlap q-range for scale fitting."
        )
    fit_model = np.exp(
        np.interp(
            np.log(fit_q),
            np.log(model_q[positive_model]),
            np.log(model_i[positive_model]),
        )
    )
    log_ratio = np.log(fit_experimental) - np.log(fit_model)
    scale_factor = float(np.exp(np.mean(log_ratio, dtype=float)))
    residual = np.log(fit_experimental) - np.log(scale_factor * fit_model)
    log_rms_residual = float(np.sqrt(np.mean(np.square(residual))))
    return (
        np.asarray(fit_q, dtype=float),
        np.asarray(fit_experimental, dtype=float),
        np.asarray(fit_model, dtype=float),
        scale_factor,
        log_rms_residual,
    )


def _write_experimental_overlay_csv(
    output_dir: Path,
    *,
    fit_q: np.ndarray,
    fit_experimental: np.ndarray,
    fit_model: np.ndarray,
    scale_factor: float,
) -> Path:
    path = output_dir / "direct_frame_saxs_experimental_overlay.csv"
    scaled_model = float(scale_factor) * np.asarray(fit_model, dtype=float)
    log_residual = np.log(fit_experimental) - np.log(scaled_model)
    data = np.column_stack(
        (fit_q, fit_experimental, fit_model, scaled_model, log_residual)
    )
    header = (
        "q_a_inverse,experimental_intensity,unscaled_direct_frame_intensity,"
        "scaled_direct_frame_intensity,log_residual"
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )
    return path


def _plot_experimental_overlay(
    output_dir: Path,
    *,
    q_values: np.ndarray,
    mean_intensity: np.ndarray,
    experimental_q: np.ndarray,
    experimental_i: np.ndarray,
    scale_factor: float,
    diagnostics: FiniteBoxDiagnostics,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    model_mask = (
        np.isfinite(q_values)
        & np.isfinite(mean_intensity)
        & (q_values > 0.0)
        & (mean_intensity > 0.0)
    )
    experimental_mask = (
        np.isfinite(experimental_q)
        & np.isfinite(experimental_i)
        & (experimental_q > 0.0)
        & (experimental_i > 0.0)
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.loglog(
        experimental_q[experimental_mask],
        experimental_i[experimental_mask],
        color="#111827",
        linewidth=1.3,
        label="experiment",
    )
    ax.loglog(
        q_values[model_mask],
        float(scale_factor) * mean_intensity[model_mask],
        color="#2563EB",
        linewidth=1.8,
        label=f"direct frame SAXS x {scale_factor:.3g}",
    )
    if np.any(model_mask):
        ax.set_xlim(
            float(np.min(q_values[model_mask])),
            float(np.max(q_values[model_mask])),
        )
    _shade_low_q(ax, diagnostics)
    ax.axvline(
        diagnostics.q_fundamental_a_inverse,
        color="#C2410C",
        linestyle="--",
        linewidth=1.0,
        label="2*pi/L",
    )
    ax.set_xlabel("q (A^-1)")
    ax.set_ylabel("I(q) (arb. units)")
    ax.set_title("Direct frame SAXS vs experimental data")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "direct_frame_saxs_experimental_overlay_loglog.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_scaled_variance_band(
    ax: object,
    *,
    q_values: np.ndarray,
    mean_intensity: np.ndarray,
    std_intensity: np.ndarray,
    scale_factor: float,
    mask: np.ndarray,
) -> None:
    q_array = np.asarray(q_values, dtype=float)
    mean_array = np.asarray(mean_intensity, dtype=float)
    std_array = np.asarray(std_intensity, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if (
        q_array.shape != mean_array.shape
        or q_array.shape != std_array.shape
        or q_array.shape != mask_array.shape
    ):
        return
    lower = float(scale_factor) * (mean_array - std_array)
    upper = float(scale_factor) * (mean_array + std_array)
    band_mask = (
        mask_array & np.isfinite(lower) & np.isfinite(upper) & (upper > 0.0)
    )
    if not np.any(band_mask):
        return
    ax.fill_between(
        q_array[band_mask],
        np.clip(lower[band_mask], 1.0e-30, None),
        upper[band_mask],
        color="#60A5FA",
        alpha=0.18,
        label="frame variance (mean +/- SD)",
    )


def _write_figures(
    output_dir: Path,
    *,
    frame_paths: tuple[Path, ...],
    q_values: np.ndarray,
    frame_intensities: np.ndarray,
    mean_intensity: np.ndarray,
    std_intensity: np.ndarray,
    se_intensity: np.ndarray,
    diagnostics: FiniteBoxDiagnostics,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    paths.append(
        _plot_overlay(
            output_dir,
            plt,
            frame_paths=frame_paths,
            q_values=q_values,
            frame_intensities=frame_intensities,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            se_intensity=se_intensity,
            diagnostics=diagnostics,
        )
    )
    paths.append(
        _plot_normalized(
            output_dir,
            plt,
            q_values=q_values,
            mean_intensity=mean_intensity,
            diagnostics=diagnostics,
        )
    )
    paths.append(
        _plot_box_diagnostics(
            output_dir,
            plt,
            q_values=q_values,
            mean_intensity=mean_intensity,
            diagnostics=diagnostics,
        )
    )
    plt.close("all")
    return tuple(paths)


def _plot_overlay(
    output_dir: Path,
    plt: object,
    *,
    frame_paths: tuple[Path, ...],
    q_values: np.ndarray,
    frame_intensities: np.ndarray,
    mean_intensity: np.ndarray,
    std_intensity: np.ndarray,
    se_intensity: np.ndarray,
    diagnostics: FiniteBoxDiagnostics,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for index, frame_path in enumerate(frame_paths):
        if index >= 12:
            break
        trace = np.asarray(frame_intensities[index], dtype=float)
        trace_mask = (
            np.isfinite(q_values)
            & np.isfinite(trace)
            & (q_values > 0.0)
            & (trace > 0.0)
        )
        ax.loglog(
            q_values[trace_mask],
            trace[trace_mask],
            linewidth=1.0,
            alpha=0.28,
            label=frame_path.stem if len(frame_paths) <= 8 else None,
        )
    mean_mask = (
        np.isfinite(q_values)
        & np.isfinite(mean_intensity)
        & (q_values > 0.0)
        & (mean_intensity > 0.0)
    )
    ax.loglog(
        q_values[mean_mask],
        mean_intensity[mean_mask],
        color="black",
        linewidth=1.8,
        label="mean",
    )
    if len(frame_paths) > 1:
        lower = mean_intensity - std_intensity
        upper = mean_intensity + std_intensity
        band_mask = (
            np.isfinite(q_values)
            & np.isfinite(lower)
            & np.isfinite(upper)
            & (q_values > 0.0)
            & (upper > 0.0)
        )
        ax.fill_between(
            q_values[band_mask],
            np.clip(lower[band_mask], 1.0e-30, None),
            upper[band_mask],
            color="#4C78A8",
            alpha=0.18,
            label="frame variance (mean +/- SD)",
        )
    _shade_low_q(ax, diagnostics)
    ax.set_xlabel("q (A^-1)")
    ax.set_ylabel("I(q) (electron units^2)")
    ax.set_title("Direct frame Debye SAXS")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "direct_frame_saxs_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_normalized(
    output_dir: Path,
    plt: object,
    *,
    q_values: np.ndarray,
    mean_intensity: np.ndarray,
    diagnostics: FiniteBoxDiagnostics,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    finite = np.isfinite(mean_intensity)
    scale = float(mean_intensity[finite][0]) if np.any(finite) else 1.0
    if abs(scale) <= 1.0e-30:
        scale = 1.0
    normalized = mean_intensity / scale
    positive = (
        np.isfinite(q_values)
        & np.isfinite(normalized)
        & (q_values > 0.0)
        & (normalized > 0.0)
    )
    ax.loglog(q_values[positive], normalized[positive], color="#2F855A")
    _shade_low_q(ax, diagnostics)
    ax.set_xlabel("q (A^-1)")
    ax.set_ylabel("I(q) / I(q_min)")
    ax.set_title("Normalized direct frame SAXS")
    fig.tight_layout()
    path = output_dir / "direct_frame_saxs_normalized.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_box_diagnostics(
    output_dir: Path,
    plt: object,
    *,
    q_values: np.ndarray,
    mean_intensity: np.ndarray,
    diagnostics: FiniteBoxDiagnostics,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    finite = np.isfinite(mean_intensity)
    scale = float(np.nanmax(mean_intensity[finite])) if np.any(finite) else 1.0
    if abs(scale) <= 1.0e-30:
        scale = 1.0
    ax.plot(q_values, mean_intensity / scale, color="#3B5BDB", linewidth=1.6)
    ax.axvline(
        diagnostics.q_fundamental_a_inverse,
        color="#C2410C",
        linestyle="--",
        linewidth=1.4,
        label="2*pi/L",
    )
    ax.axvline(
        diagnostics.q_caution_a_inverse,
        color="#9A3412",
        linestyle=":",
        linewidth=1.4,
        label="4*pi/L",
    )
    _shade_low_q(ax, diagnostics)
    ax.set_xlabel("q (A^-1)")
    ax.set_ylabel("I(q) / max(I)")
    ax.set_title("Finite box q-window diagnostic")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "finite_box_q_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _shade_low_q(ax: object, diagnostics: FiniteBoxDiagnostics) -> None:
    lower_bound = 0.0
    get_xscale = getattr(ax, "get_xscale", None)
    if callable(get_xscale) and get_xscale() == "log":
        x_min, _x_max = ax.get_xlim()
        lower_bound = x_min if x_min > 0.0 else 1.0e-12
    ax.axvspan(
        lower_bound,
        diagnostics.q_fundamental_a_inverse,
        color="#F97316",
        alpha=0.10,
        linewidth=0,
    )
    ax.axvspan(
        diagnostics.q_fundamental_a_inverse,
        diagnostics.q_caution_a_inverse,
        color="#F59E0B",
        alpha=0.08,
        linewidth=0,
    )


def _csv_column_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "frame"


def _settings_payload(settings: DirectFrameSAXSSettings) -> dict[str, object]:
    return {
        "input_path": str(settings.input_path),
        "output_dir": str(settings.output_dir),
        "q_min": settings.q_min,
        "q_max": settings.q_max,
        "q_step": settings.q_step,
        "max_frames": settings.max_frames,
        "box_length_a": settings.box_length_a,
        "box_lengths_a": (
            None
            if settings.box_lengths_a is None
            else list(settings.box_lengths_a)
        ),
        "subtract_average_box_density": (
            settings.subtract_average_box_density
        ),
        "direction_count": settings.direction_count,
    }


def _diagnostics_payload(
    diagnostics: FiniteBoxDiagnostics,
) -> dict[str, object]:
    return {
        "coordinate_span_a": list(diagnostics.coordinate_span_a),
        "inferred_box_length_a": diagnostics.inferred_box_length_a,
        "q_fundamental_a_inverse": diagnostics.q_fundamental_a_inverse,
        "q_caution_a_inverse": diagnostics.q_caution_a_inverse,
        "q_min_a_inverse": diagnostics.q_min_a_inverse,
        "q_max_a_inverse": diagnostics.q_max_a_inverse,
        "q_points_below_fundamental": (diagnostics.q_points_below_fundamental),
        "q_points_below_caution": diagnostics.q_points_below_caution,
        "finite_box_note": diagnostics.finite_box_note,
    }


def _experimental_overlay_payload(
    overlay: ExperimentalOverlayResult,
) -> dict[str, object]:
    return {
        "experimental_data_path": str(overlay.experimental_data_path),
        "plot_path": str(overlay.plot_path),
        "scaled_profile_csv_path": str(overlay.scaled_profile_csv_path),
        "scale_factor": overlay.scale_factor,
        "fit_q_min_a_inverse": overlay.fit_q_min_a_inverse,
        "fit_q_max_a_inverse": overlay.fit_q_max_a_inverse,
        "fit_point_count": overlay.fit_point_count,
        "log_rms_residual": overlay.log_rms_residual,
    }
