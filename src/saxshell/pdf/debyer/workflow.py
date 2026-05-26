from __future__ import annotations

import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from saxshell.cluster.clusternetwork import (
    detect_frame_folder_mode,
    detect_source_box_dimensions,
    estimate_box_dimensions_from_coordinates,
)
from saxshell.saxs.debye.profiles import load_structure_file
from saxshell.saxs.project_manager import build_project_paths

DEBYER_DOCS_URL = "https://debyer.readthedocs.io/en/latest/"
DEBYER_GITHUB_URL = "https://github.com/wojdyr/debyer"
TOTAL_SCATTERING_PAPER_URL = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7941302/"
)
SUPPORTED_DEBYER_MODES = ("PDF", "RDF", "rPDF")
SUPPORTED_PLOT_REPRESENTATIONS = ("g(r)", "G(r)", "R(r)")
DEFAULT_COLOR_SCHEMES = ("tab20", "tab10", "viridis", "plasma", "summer")
GROUPED_PARTIAL_COLUMN_LABELS = (
    "solute-solute",
    "solute-solvent",
    "solvent-solvent",
)
_COLUMN_PREFIX = "# columns:"
_DEFAULT_AVERAGE_CHECKPOINT_INTERVAL_FRAMES = 1000
_MIN_AVERAGE_CHECKPOINT_INTERVAL_FRAMES = 100
_RUNNING_AVERAGE_MEMORY_TARGET_BYTES = 256 * 1024 * 1024
_MAX_PARALLEL_DEBYER_JOBS = 64


def default_parallel_debyer_jobs(cpu_count: int | None = None) -> int:
    available_cpus = os.cpu_count() if cpu_count is None else cpu_count
    return max(1, min(4, int(available_cpus or 1)))


def _coerce_parallel_debyer_jobs(value: object) -> int:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 1
    if requested <= 0:
        requested = default_parallel_debyer_jobs()
    return max(1, min(requested, _MAX_PARALLEL_DEBYER_JOBS))


def _resolve_parallel_debyer_jobs(
    value: object,
    *,
    total_frames: int,
) -> int:
    return min(
        _coerce_parallel_debyer_jobs(value),
        max(int(total_frames), 1),
    )


@dataclass(slots=True, frozen=True)
class DebyerRuntimeStatus:
    executable_path: Path | None
    available: bool
    runnable: bool
    permission_granted: bool
    message: str


@dataclass(slots=True, frozen=True)
class DebyerFrameInspection:
    frames_dir: Path
    frame_format: str
    frame_paths: tuple[Path, ...]
    detected_box_dimensions: tuple[float, float, float] | None
    detected_box_source: str | None
    detected_box_source_kind: str | None
    estimated_box_dimensions: tuple[float, float, float] | None
    atom_count: int
    element_counts: dict[str, int]


@dataclass(slots=True, frozen=True)
class DebyerPDFSettings:
    project_dir: Path
    frames_dir: Path
    filename_prefix: str
    mode: str = "PDF"
    from_value: float = 0.5
    to_value: float = 15.0
    step_value: float = 0.01
    box_dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    atom_count: int = 0
    store_frame_outputs: bool = False
    solute_elements: tuple[str, ...] = ()
    max_parallel_jobs: int = 1


@dataclass(slots=True, frozen=True)
class DebyerPeakFinderSettings:
    min_relative_height: float = 0.12
    min_spacing_angstrom: float = 0.35
    max_peak_count: int = 6


@dataclass(slots=True, frozen=True)
class DebyerPeakMarker:
    r_value: float
    label: str
    enabled: bool = True
    text_dx: float = 0.0
    text_dy: float = 0.0
    source: str = "auto"


@dataclass(slots=True, frozen=True)
class DebyerFitMetrics:
    r_squared: float
    rmse: float
    mae: float
    point_count: int
    r_min: float
    r_max: float


@dataclass(slots=True, frozen=True)
class DebyerCoordinationFitResult:
    r_min: float
    r_max: float
    center: float
    sigma: float
    coordination_number: float
    amplitude: float
    baseline_intercept: float
    baseline_slope: float
    rmse: float
    r_squared: float
    point_count: int
    fitted_values: np.ndarray


@dataclass(slots=True, frozen=True)
class DebyerPDFCalculationSummary:
    calculation_id: str
    calculation_dir: Path
    created_at: str
    filename_prefix: str
    mode: str
    frame_count: int
    frames_dir: Path


@dataclass(slots=True, frozen=True)
class DebyerPDFCalculation:
    calculation_id: str
    calculation_dir: Path
    created_at: str
    project_dir: Path
    frames_dir: Path
    frame_format: str
    frame_count: int
    filename_prefix: str
    mode: str
    from_value: float
    to_value: float
    step_value: float
    box_dimensions: tuple[float, float, float]
    box_source: str | None
    box_source_kind: str | None
    atom_count: int
    rho0: float
    store_frame_outputs: bool
    frame_output_dir: Path | None
    averaged_output_file: Path
    solute_elements: tuple[str, ...]
    parallel_jobs: int
    r_values: np.ndarray
    total_values: np.ndarray
    partial_values: dict[str, np.ndarray]
    processed_frame_count: int | None = None
    is_partial_average: bool = False
    elapsed_seconds: float | None = None
    estimated_remaining_seconds: float | None = None
    expected_total_seconds: float | None = None
    partial_peak_markers: dict[str, tuple[DebyerPeakMarker, ...]] = field(
        default_factory=dict
    )
    target_peak_markers: dict[str, dict[str, tuple[DebyerPeakMarker, ...]]] = (
        field(default_factory=dict)
    )
    peak_finder_settings: DebyerPeakFinderSettings = field(
        default_factory=DebyerPeakFinderSettings
    )


def _normalized_element(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def _normalize_solute_elements(
    values: list[str] | tuple[str, ...] | set[str] | None,
) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        element = _normalized_element(value)
        if not element or element in seen:
            continue
        normalized.append(element)
        seen.add(element)
    return tuple(normalized)


def infer_default_solute_elements(
    element_counts: dict[str, int] | list[str] | tuple[str, ...] | set[str],
) -> tuple[str, ...]:
    available = {
        _normalized_element(element)
        for element in element_counts
        if _normalized_element(element)
    }
    if {"Cs", "Pb", "I"}.issubset(available):
        return ("Cs", "Pb", "I")
    if {"Pb", "I"}.issubset(available):
        return ("Pb", "I")
    return ()


def _sanitize_prefix(value: str) -> str:
    text = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    ).strip("_")
    return text or "debyer_pdf"


def _coerce_peak_finder_settings(
    payload: dict[str, Any] | None,
) -> DebyerPeakFinderSettings:
    if not payload:
        return DebyerPeakFinderSettings()
    return DebyerPeakFinderSettings(
        min_relative_height=max(
            float(payload.get("min_relative_height", 0.12)),
            0.0,
        ),
        min_spacing_angstrom=max(
            float(payload.get("min_spacing_angstrom", 0.35)),
            0.0,
        ),
        max_peak_count=max(int(payload.get("max_peak_count", 6)), 0),
    )


def _serialize_peak_finder_settings(
    settings: DebyerPeakFinderSettings,
) -> dict[str, Any]:
    return {
        "min_relative_height": float(settings.min_relative_height),
        "min_spacing_angstrom": float(settings.min_spacing_angstrom),
        "max_peak_count": int(settings.max_peak_count),
    }


def _default_peak_label(pair_label: str, r_value: float) -> str:
    return f"{pair_label}: {float(r_value):.2f} A"


def _serialize_peak_markers(
    markers: dict[str, tuple[DebyerPeakMarker, ...]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        pair_label: [
            {
                "r_value": float(marker.r_value),
                "label": str(marker.label),
                "enabled": bool(marker.enabled),
                "text_dx": float(marker.text_dx),
                "text_dy": float(marker.text_dy),
                "source": str(marker.source),
            }
            for marker in pair_markers
        ]
        for pair_label, pair_markers in sorted(markers.items())
    }


def _deserialize_peak_markers(
    payload: dict[str, Any] | None,
) -> dict[str, tuple[DebyerPeakMarker, ...]]:
    if not payload:
        return {}
    resolved: dict[str, tuple[DebyerPeakMarker, ...]] = {}
    for pair_label, entries in payload.items():
        if not isinstance(entries, list):
            continue
        markers: list[DebyerPeakMarker] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                r_value = float(entry.get("r_value"))
            except (TypeError, ValueError):
                continue
            markers.append(
                DebyerPeakMarker(
                    r_value=r_value,
                    label=str(
                        entry.get(
                            "label",
                            _default_peak_label(str(pair_label), r_value),
                        )
                    ),
                    enabled=bool(entry.get("enabled", True)),
                    text_dx=float(entry.get("text_dx", 0.0)),
                    text_dy=float(entry.get("text_dy", 0.0)),
                    source=str(entry.get("source", "manual")),
                )
            )
        resolved[str(pair_label)] = tuple(
            sorted(markers, key=lambda marker: marker.r_value)
        )
    return resolved


def _serialize_target_peak_markers(
    payload: dict[str, dict[str, tuple[DebyerPeakMarker, ...]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        str(target_trace_key): _serialize_peak_markers(pair_markers)
        for target_trace_key, pair_markers in sorted(payload.items())
    }


def _deserialize_target_peak_markers(
    payload: dict[str, Any] | None,
) -> dict[str, dict[str, tuple[DebyerPeakMarker, ...]]]:
    if not payload:
        return {}
    resolved: dict[str, dict[str, tuple[DebyerPeakMarker, ...]]] = {}
    for target_trace_key, pair_payload in payload.items():
        if not isinstance(pair_payload, dict):
            continue
        resolved[str(target_trace_key)] = _deserialize_peak_markers(
            pair_payload
        )
    return resolved


def build_debyer_project_dir(project_dir: str | Path) -> Path:
    return (
        build_project_paths(project_dir).exported_data_dir
        / "debyer"
        / "saved_calculations"
    )


def _build_calculation_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_sanitize_prefix(prefix)}"


def calculate_number_density(
    atom_count: int,
    box_dimensions: tuple[float, float, float],
) -> float:
    volume = float(np.prod(np.asarray(box_dimensions, dtype=float)))
    if volume <= 0.0:
        raise ValueError(
            "The bounding-box volume must be positive to calculate rho0."
        )
    if int(atom_count) <= 0:
        raise ValueError("The atom count must be positive to calculate rho0.")
    return float(atom_count) / volume


def check_debyer_runtime(
    executable: str | Path | None = None,
    *,
    timeout_seconds: float = 3.0,
) -> DebyerRuntimeStatus:
    resolved = None if executable is None else Path(executable).expanduser()
    if resolved is None:
        discovered = shutil.which("debyer")
        if discovered:
            resolved = Path(discovered)

    if resolved is None:
        return DebyerRuntimeStatus(
            executable_path=None,
            available=False,
            runnable=False,
            permission_granted=False,
            message=(
                "Debyer was not found on PATH. Install Debyer and make sure "
                "the 'debyer' executable is available before running PDF "
                "calculations."
            ),
        )

    try:
        completed = subprocess.run(
            [str(resolved), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(float(timeout_seconds), 0.5),
        )
    except PermissionError:
        return DebyerRuntimeStatus(
            executable_path=resolved,
            available=True,
            runnable=False,
            permission_granted=False,
            message=(
                "Debyer was found but SAXSShell could not execute it. Check "
                "the executable permissions and any OS-level subprocess "
                "approval settings."
            ),
        )
    except OSError as exc:
        return DebyerRuntimeStatus(
            executable_path=resolved,
            available=True,
            runnable=False,
            permission_granted=False,
            message=f"Debyer was found but could not be started: {exc}",
        )
    except subprocess.TimeoutExpired:
        return DebyerRuntimeStatus(
            executable_path=resolved,
            available=True,
            runnable=False,
            permission_granted=False,
            message=(
                "Debyer was found, but the quick startup check timed out. "
                "Verify that Debyer can be launched manually from a terminal."
            ),
        )

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return DebyerRuntimeStatus(
            executable_path=resolved,
            available=True,
            runnable=False,
            permission_granted=False,
            message=(
                "Debyer was found, but the quick '--help' check failed"
                + (f": {detail}" if detail else ".")
            ),
        )

    return DebyerRuntimeStatus(
        executable_path=resolved,
        available=True,
        runnable=True,
        permission_granted=True,
        message=f"Debyer is available at {resolved}",
    )


def inspect_frames_dir(frames_dir: str | Path) -> DebyerFrameInspection:
    resolved_frames_dir = Path(frames_dir).expanduser().resolve()
    frame_format, frame_paths = detect_frame_folder_mode(resolved_frames_dir)
    if frame_format != "xyz":
        raise ValueError(
            "Debyer PDF calculations require XYZ frame files. Convert PDB "
            "frames to XYZ before using pdfsetup."
        )
    first_frame = frame_paths[0]

    detected_box_dimensions: tuple[float, float, float] | None = None
    detected_box_source: str | None = None
    detected_box_source_kind: str | None = None
    detected = detect_source_box_dimensions(resolved_frames_dir)
    if detected is not None:
        detected_box_dimensions, source_path = detected
        detected_box_source = source_path.name
        detected_box_source_kind = "source_filename"

    coordinates, elements = load_structure_file(first_frame)
    estimated_box_dimensions = estimate_box_dimensions_from_coordinates(
        coordinates
    )
    element_counts: dict[str, int] = {}
    for element in elements:
        normalized = _normalized_element(element)
        element_counts[normalized] = element_counts.get(normalized, 0) + 1

    return DebyerFrameInspection(
        frames_dir=resolved_frames_dir,
        frame_format=str(frame_format),
        frame_paths=tuple(frame_paths),
        detected_box_dimensions=detected_box_dimensions,
        detected_box_source=detected_box_source,
        detected_box_source_kind=detected_box_source_kind,
        estimated_box_dimensions=estimated_box_dimensions,
        atom_count=len(elements),
        element_counts=element_counts,
    )


def _parse_columns_from_comments(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(_COLUMN_PREFIX):
                payload = stripped[len(_COLUMN_PREFIX) :].strip()
                return [token for token in payload.split() if token]
            if stripped.startswith("# sum"):
                return stripped[1:].split()
    return ["sum"]


def parse_debyer_output_file(
    path: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    resolved = Path(path).expanduser().resolve()
    columns = _parse_columns_from_comments(resolved)
    raw = np.loadtxt(resolved, comments="#")
    if raw.size == 0:
        raise ValueError(f"Debyer output file is empty: {resolved}")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 2:
        raise ValueError(
            "Debyer output must contain the radial grid and at least one "
            f"distribution column: {resolved}"
        )
    r_values = np.asarray(raw[:, 0], dtype=float)
    value_columns = raw[:, 1:]
    if len(columns) != value_columns.shape[1]:
        columns = ["sum"] + [
            f"partial_{index:02d}"
            for index in range(1, value_columns.shape[1])
        ]
    values = {
        str(column): np.asarray(value_columns[:, index], dtype=float)
        for index, column in enumerate(columns)
    }
    if "sum" not in values and values:
        first_key = next(iter(values))
        values["sum"] = np.asarray(values[first_key], dtype=float)
    return r_values, values


def save_averaged_debyer_output(
    output_path: str | Path,
    *,
    r_values: np.ndarray,
    column_order: list[str],
    values: dict[str, np.ndarray],
    metadata: dict[str, object] | None = None,
) -> None:
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [name for name in column_order if name in values]
    data = np.column_stack(
        [np.asarray(r_values, dtype=float)]
        + [np.asarray(values[name], dtype=float) for name in ordered_columns]
    )
    lines: list[str] = []
    if metadata:
        for key, value in metadata.items():
            lines.append(f"# {key}: {value}")
    lines.append(f"{_COLUMN_PREFIX} {' '.join(ordered_columns)}")
    np.savetxt(
        resolved,
        data,
        header="\n".join(lines),
        comments="",
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)):
        return "unknown"
    rounded = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _average_checkpoint_interval(
    *,
    total_frames: int,
    average_state_bytes: int = 0,
) -> int:
    resolved_total = max(int(total_frames), 1)
    if average_state_bytes <= _RUNNING_AVERAGE_MEMORY_TARGET_BYTES:
        return min(
            _DEFAULT_AVERAGE_CHECKPOINT_INTERVAL_FRAMES,
            resolved_total,
        )
    pressure = int(
        math.ceil(
            average_state_bytes / float(_RUNNING_AVERAGE_MEMORY_TARGET_BYTES)
        )
    )
    memory_interval = max(
        _MIN_AVERAGE_CHECKPOINT_INTERVAL_FRAMES,
        _DEFAULT_AVERAGE_CHECKPOINT_INTERVAL_FRAMES // max(pressure, 1),
    )
    return min(memory_interval, resolved_total)


def _estimate_runtime(
    *,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float,
) -> tuple[float | None, float | None]:
    if processed_frames <= 0:
        return None, None
    mean_seconds_per_frame = float(elapsed_seconds) / float(processed_frames)
    remaining_frames = max(int(total_frames) - int(processed_frames), 0)
    remaining_seconds = mean_seconds_per_frame * float(remaining_frames)
    expected_total_seconds = float(elapsed_seconds) + remaining_seconds
    return remaining_seconds, expected_total_seconds


def _build_averaged_output_metadata(
    *,
    calculation_id: str,
    created_at: str,
    settings: DebyerPDFSettings,
    inspection: DebyerFrameInspection,
    rho0: float,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float | None,
    estimated_remaining_seconds: float | None,
    expected_total_seconds: float | None,
    parallel_jobs: int | None = None,
) -> dict[str, object]:
    return {
        "calculation_id": calculation_id,
        "created_at": created_at,
        "filename_prefix": settings.filename_prefix,
        "frames_dir": str(settings.frames_dir),
        "frame_format": inspection.frame_format,
        "processed_frames": int(processed_frames),
        "total_frames": int(total_frames),
        "mode": settings.mode,
        "from_value": settings.from_value,
        "to_value": settings.to_value,
        "step_value": settings.step_value,
        "box_dimensions": ", ".join(
            f"{component:.6g}" for component in settings.box_dimensions
        ),
        "box_source": inspection.detected_box_source or "estimated/manual",
        "box_source_kind": inspection.detected_box_source_kind or "estimate",
        "atom_count": settings.atom_count,
        "rho0": f"{rho0:.8g}",
        "store_frame_outputs": settings.store_frame_outputs,
        "solute_elements": ", ".join(settings.solute_elements) or "None",
        "parallel_jobs": int(
            settings.max_parallel_jobs
            if parallel_jobs is None
            else parallel_jobs
        ),
        "elapsed_seconds": (
            None
            if elapsed_seconds is None
            else f"{float(elapsed_seconds):.6f}"
        ),
        "estimated_remaining_seconds": (
            None
            if estimated_remaining_seconds is None
            else f"{float(estimated_remaining_seconds):.6f}"
        ),
        "expected_total_seconds": (
            None
            if expected_total_seconds is None
            else f"{float(expected_total_seconds):.6f}"
        ),
        "elapsed_hms": _format_duration(elapsed_seconds),
        "remaining_hms": _format_duration(estimated_remaining_seconds),
        "expected_total_hms": _format_duration(expected_total_seconds),
    }


def _average_frame_outputs(
    outputs: list[tuple[np.ndarray, dict[str, np.ndarray]]],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if not outputs:
        raise ValueError(
            "No Debyer frame outputs were provided for averaging."
        )

    reference_r = np.asarray(outputs[0][0], dtype=float)
    union_columns: list[str] = ["sum"]
    seen = {"sum"}
    for _r_values, columns in outputs:
        for key in columns:
            if key not in seen:
                seen.add(key)
                union_columns.append(key)

    stacked: dict[str, list[np.ndarray]] = {key: [] for key in union_columns}
    for r_values, columns in outputs:
        if not np.allclose(reference_r, np.asarray(r_values, dtype=float)):
            raise ValueError(
                "Debyer frame outputs do not share the same radial grid."
            )
        for key in union_columns:
            if key in columns:
                stacked[key].append(np.asarray(columns[key], dtype=float))
            else:
                stacked[key].append(np.zeros_like(reference_r, dtype=float))

    averaged = {
        key: np.mean(np.vstack(series), axis=0)
        for key, series in stacked.items()
    }
    return reference_r, union_columns, averaged


@dataclass(slots=True)
class _RunningDebyerAverage:
    reference_r: np.ndarray | None = None
    column_order: list[str] = field(default_factory=lambda: ["sum"])
    sums: dict[str, np.ndarray] = field(default_factory=dict)
    processed_count: int = 0

    def add_frame(
        self,
        r_values: np.ndarray,
        columns: dict[str, np.ndarray],
    ) -> None:
        radial = np.asarray(r_values, dtype=float)
        if self.reference_r is None:
            self.reference_r = radial.copy()
            self.sums = {
                "sum": np.zeros_like(self.reference_r, dtype=float),
            }
        elif not np.allclose(self.reference_r, radial):
            raise ValueError(
                "Debyer frame outputs do not share the same radial grid."
            )

        for key in columns:
            if key not in self.sums:
                self.column_order.append(key)
                self.sums[key] = np.zeros_like(self.reference_r, dtype=float)

        for key, values in columns.items():
            self.sums[key] += np.asarray(values, dtype=float)
        self.processed_count += 1

    @property
    def memory_bytes(self) -> int:
        total = 0
        if self.reference_r is not None:
            total += int(self.reference_r.nbytes)
        total += sum(int(values.nbytes) for values in self.sums.values())
        return total

    def average(
        self,
    ) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
        if self.reference_r is None or self.processed_count <= 0:
            raise ValueError(
                "No Debyer frame outputs were provided for averaging."
            )
        averaged = {
            key: np.asarray(self.sums[key], dtype=float)
            / float(self.processed_count)
            for key in self.column_order
        }
        return self.reference_r.copy(), list(self.column_order), averaged


@dataclass(slots=True, frozen=True)
class _DebyerFrameRunResult:
    frame_index: int
    frame_path: Path
    output_path: Path
    r_values: np.ndarray
    values: dict[str, np.ndarray]


def _candidate_peak_indices(values: np.ndarray) -> list[int]:
    array = np.asarray(values, dtype=float)
    count = int(array.size)
    if count == 0:
        return []
    if count == 1:
        return [0]
    candidate_indices: list[int] = []
    for index in range(count):
        center = float(array[index])
        if not np.isfinite(center):
            continue
        left = float(array[index - 1]) if index > 0 else -math.inf
        right = float(array[index + 1]) if index < (count - 1) else -math.inf
        if index == 0:
            is_peak = center > right
        elif index == (count - 1):
            is_peak = center > left
        else:
            is_peak = center >= left and center > right
        if is_peak:
            candidate_indices.append(index)
    return candidate_indices


def find_partial_peak_markers(
    *,
    pair_label: str,
    r_values: np.ndarray,
    values: np.ndarray,
    settings: DebyerPeakFinderSettings,
) -> tuple[DebyerPeakMarker, ...]:
    radial = np.asarray(r_values, dtype=float)
    signal = np.asarray(values, dtype=float)
    if radial.size == 0 or signal.size == 0:
        return ()
    max_value = float(np.nanmax(signal))
    if not np.isfinite(max_value) or max_value <= 0.0:
        return ()
    min_height = float(settings.min_relative_height) * max_value
    candidate_indices = [
        index
        for index in _candidate_peak_indices(signal)
        if float(signal[index]) >= min_height
    ]
    if not candidate_indices:
        return ()

    min_spacing = max(float(settings.min_spacing_angstrom), 0.0)
    max_peak_count = max(int(settings.max_peak_count), 0)
    selected: list[int] = []
    for index in sorted(
        candidate_indices,
        key=lambda candidate: float(signal[candidate]),
        reverse=True,
    ):
        peak_position = float(radial[index])
        if any(
            abs(peak_position - float(radial[chosen])) < min_spacing
            for chosen in selected
        ):
            continue
        selected.append(index)
        if max_peak_count and len(selected) >= max_peak_count:
            break

    selected.sort(key=lambda index: float(radial[index]))
    radial_span = (
        max(float(radial[-1]) - float(radial[0]), 0.0)
        if radial.size > 1
        else 0.0
    )
    default_dx = max(radial_span * 0.02, min_spacing * 0.5, 0.05)
    return tuple(
        DebyerPeakMarker(
            r_value=float(radial[index]),
            label=_default_peak_label(pair_label, float(radial[index])),
            enabled=True,
            text_dx=default_dx,
            text_dy=0.0,
            source="auto",
        )
        for index in selected
    )


def estimate_partial_peak_markers(
    *,
    r_values: np.ndarray,
    partial_values: dict[str, np.ndarray],
    settings: DebyerPeakFinderSettings,
) -> dict[str, tuple[DebyerPeakMarker, ...]]:
    return {
        pair_label: find_partial_peak_markers(
            pair_label=pair_label,
            r_values=r_values,
            values=np.asarray(values, dtype=float),
            settings=settings,
        )
        for pair_label, values in sorted(partial_values.items())
    }


def build_debyer_calculation_metadata(
    calculation: DebyerPDFCalculation,
) -> dict[str, Any]:
    return {
        "calculation_id": calculation.calculation_id,
        "created_at": calculation.created_at,
        "project_dir": str(calculation.project_dir),
        "frames_dir": str(calculation.frames_dir),
        "frame_format": calculation.frame_format,
        "frame_count": int(calculation.frame_count),
        "processed_frame_count": int(
            calculation.frame_count
            if calculation.processed_frame_count is None
            else calculation.processed_frame_count
        ),
        "is_partial_average": bool(calculation.is_partial_average),
        "filename_prefix": calculation.filename_prefix,
        "mode": calculation.mode,
        "from_value": float(calculation.from_value),
        "to_value": float(calculation.to_value),
        "step_value": float(calculation.step_value),
        "box_dimensions": [
            float(component) for component in calculation.box_dimensions
        ],
        "box_source": calculation.box_source,
        "box_source_kind": calculation.box_source_kind,
        "atom_count": int(calculation.atom_count),
        "rho0": float(calculation.rho0),
        "store_frame_outputs": bool(calculation.store_frame_outputs),
        "frame_output_dir": (
            None
            if calculation.frame_output_dir is None
            else str(calculation.frame_output_dir)
        ),
        "averaged_output_file": str(calculation.averaged_output_file),
        "solute_elements": list(calculation.solute_elements),
        "parallel_jobs": int(calculation.parallel_jobs),
        "elapsed_seconds": calculation.elapsed_seconds,
        "estimated_remaining_seconds": calculation.estimated_remaining_seconds,
        "expected_total_seconds": calculation.expected_total_seconds,
        "peak_finder_settings": _serialize_peak_finder_settings(
            calculation.peak_finder_settings
        ),
        "partial_peak_markers": _serialize_peak_markers(
            calculation.partial_peak_markers
        ),
        "target_peak_markers": _serialize_target_peak_markers(
            calculation.target_peak_markers
        ),
    }


def write_debyer_calculation_metadata(
    calculation: DebyerPDFCalculation,
) -> None:
    (calculation.calculation_dir / "calculation.json").write_text(
        json.dumps(build_debyer_calculation_metadata(calculation), indent=2)
        + "\n",
        encoding="utf-8",
    )


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    result = np.full_like(np.asarray(numerator, dtype=float), fill_value)
    valid = np.abs(np.asarray(denominator, dtype=float)) > 1.0e-12
    result[valid] = (
        np.asarray(numerator, dtype=float)[valid]
        / np.asarray(denominator, dtype=float)[valid]
    )
    return result


def convert_distribution_values(
    values: np.ndarray,
    *,
    r_values: np.ndarray,
    rho0: float,
    source_mode: str,
    target_representation: str,
    is_component: bool = False,
) -> np.ndarray:
    normalized_source = str(source_mode).strip()
    normalized_target = str(target_representation).strip()
    if normalized_source not in SUPPORTED_DEBYER_MODES:
        raise ValueError(f"Unsupported Debyer source mode: {source_mode}")
    if normalized_target not in SUPPORTED_PLOT_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported PDF target representation: {target_representation}"
        )

    radial = np.asarray(r_values, dtype=float)
    values_array = np.asarray(values, dtype=float)
    prefactor_r = 4.0 * math.pi * float(rho0) * radial
    prefactor_r2 = prefactor_r * radial

    if normalized_source == "PDF":
        canonical_g = values_array
    elif normalized_source == "RDF":
        canonical_g = _safe_divide(values_array, prefactor_r2)
    else:
        canonical_g = _safe_divide(values_array, prefactor_r)
        if not is_component:
            canonical_g = canonical_g + 1.0

    if normalized_target == "g(r)":
        return canonical_g
    if normalized_target == "R(r)":
        return prefactor_r2 * canonical_g

    if is_component:
        return prefactor_r * canonical_g
    return prefactor_r * (canonical_g - 1.0)


def compute_experimental_fit_metrics(
    *,
    model_r_values: np.ndarray,
    model_g_values: np.ndarray,
    experimental_r_values: np.ndarray,
    experimental_g_values: np.ndarray,
) -> DebyerFitMetrics | None:
    model_r = np.asarray(model_r_values, dtype=float)
    model_g = np.asarray(model_g_values, dtype=float)
    experimental_r = np.asarray(experimental_r_values, dtype=float)
    experimental_g = np.asarray(experimental_g_values, dtype=float)
    model_mask = np.isfinite(model_r) & np.isfinite(model_g)
    experimental_mask = np.isfinite(experimental_r) & np.isfinite(
        experimental_g
    )
    if model_mask.sum() < 2 or experimental_mask.sum() < 2:
        return None

    model_r = model_r[model_mask]
    model_g = model_g[model_mask]
    order = np.argsort(model_r)
    model_r = model_r[order]
    model_g = model_g[order]
    unique_r, unique_indices = np.unique(model_r, return_index=True)
    model_r = unique_r
    model_g = model_g[unique_indices]
    if model_r.size < 2:
        return None

    experimental_r = experimental_r[experimental_mask]
    experimental_g = experimental_g[experimental_mask]
    overlap_mask = (experimental_r >= model_r[0]) & (
        experimental_r <= model_r[-1]
    )
    if overlap_mask.sum() < 2:
        return None

    overlap_r = experimental_r[overlap_mask]
    overlap_g = experimental_g[overlap_mask]
    interpolated_model = np.interp(overlap_r, model_r, model_g)
    residuals = interpolated_model - overlap_g
    sse = float(np.sum(residuals**2))
    centered = overlap_g - float(np.mean(overlap_g))
    sst = float(np.sum(centered**2))
    r_squared = float("nan") if sst <= 0.0 else 1.0 - (sse / sst)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    return DebyerFitMetrics(
        r_squared=r_squared,
        rmse=rmse,
        mae=mae,
        point_count=int(overlap_r.size),
        r_min=float(np.min(overlap_r)),
        r_max=float(np.max(overlap_r)),
    )


def _coordination_gaussian_model(
    radial: np.ndarray,
    area: float,
    center: float,
    sigma: float,
    baseline_intercept: float,
    baseline_slope: float,
    *,
    baseline_pivot: float,
) -> np.ndarray:
    radial_values = np.asarray(radial, dtype=float)
    bounded_sigma = max(float(sigma), 1.0e-12)
    gaussian = (
        float(area)
        / (bounded_sigma * math.sqrt(2.0 * math.pi))
        * np.exp(-0.5 * ((radial_values - center) / bounded_sigma) ** 2)
    )
    baseline = float(baseline_intercept) + float(baseline_slope) * (
        radial_values - float(baseline_pivot)
    )
    return baseline + gaussian


def fit_coordination_peak_from_r(
    *,
    r_values: np.ndarray,
    r_distribution_values: np.ndarray,
    r_min: float,
    r_max: float,
    initial_center: float | None = None,
    initial_sigma: float | None = None,
) -> DebyerCoordinationFitResult:
    radial = np.asarray(r_values, dtype=float)
    values = np.asarray(r_distribution_values, dtype=float)
    if radial.shape != values.shape:
        raise ValueError("R(r) fit inputs must share the same shape.")
    if float(r_min) >= float(r_max):
        raise ValueError("The R(r) fit minimum must be below maximum.")

    mask = (
        np.isfinite(radial)
        & np.isfinite(values)
        & (radial >= float(r_min))
        & (radial <= float(r_max))
    )
    if mask.sum() < 5:
        raise ValueError(
            "At least five finite R(r) points are required inside the fit window."
        )
    fit_r = radial[mask]
    fit_values = values[mask]
    order = np.argsort(fit_r)
    fit_r = fit_r[order]
    fit_values = fit_values[order]

    window_width = float(fit_r[-1] - fit_r[0])
    if window_width <= 0.0:
        raise ValueError("The R(r) fit window has zero radial width.")
    edge_count = max(1, min(3, fit_r.size // 4))
    left_r = float(np.mean(fit_r[:edge_count]))
    right_r = float(np.mean(fit_r[-edge_count:]))
    left_y = float(np.mean(fit_values[:edge_count]))
    right_y = float(np.mean(fit_values[-edge_count:]))
    baseline_slope = (
        0.0
        if abs(right_r - left_r) < 1.0e-12
        else (right_y - left_y) / (right_r - left_r)
    )
    baseline_guess = left_y + baseline_slope * (fit_r - left_r)
    residual_guess = fit_values - baseline_guess
    center_guess = (
        float(initial_center)
        if initial_center is not None
        else float(fit_r[int(np.nanargmax(residual_guess))])
    )
    center_guess = min(max(center_guess, float(fit_r[0])), float(fit_r[-1]))
    sigma_guess = (
        float(initial_sigma)
        if initial_sigma is not None and float(initial_sigma) > 0.0
        else max(window_width / 6.0, 1.0e-4)
    )
    sigma_guess = min(max(sigma_guess, 1.0e-4), window_width)
    intercept_guess = float(left_y + baseline_slope * (center_guess - left_r))
    positive_peak = np.maximum(residual_guess, 0.0)
    if hasattr(np, "trapezoid"):
        area_guess = float(np.trapezoid(positive_peak, fit_r))
    else:
        area_guess = float(
            np.sum(
                0.5 * (positive_peak[1:] + positive_peak[:-1]) * np.diff(fit_r)
            )
        )
    if not np.isfinite(area_guess) or area_guess <= 0.0:
        area_guess = (
            max(float(np.nanmax(fit_values) - np.nanmin(fit_values)), 1.0e-6)
            * window_width
            / 3.0
        )
    baseline_pivot = center_guess

    def model(
        radial_values: np.ndarray,
        area: float,
        center: float,
        sigma: float,
        intercept: float,
        slope: float,
    ) -> np.ndarray:
        return _coordination_gaussian_model(
            radial_values,
            area,
            center,
            sigma,
            intercept,
            slope,
            baseline_pivot=baseline_pivot,
        )

    try:
        params, _covariance = curve_fit(
            model,
            fit_r,
            fit_values,
            p0=[
                area_guess,
                center_guess,
                sigma_guess,
                intercept_guess,
                baseline_slope,
            ],
            bounds=(
                [0.0, float(fit_r[0]), 1.0e-6, -np.inf, -np.inf],
                [np.inf, float(fit_r[-1]), window_width * 2.0, np.inf, np.inf],
            ),
            maxfev=20000,
        )
    except Exception as exc:
        raise ValueError(f"R(r) coordination fit failed: {exc}") from exc

    fitted_values = model(fit_r, *params)
    residual = fit_values - fitted_values
    rmse = float(np.sqrt(np.mean(residual**2)))
    total_variance = float(np.sum((fit_values - np.mean(fit_values)) ** 2))
    r_squared = (
        float("nan")
        if total_variance <= 1.0e-20
        else 1.0 - float(np.sum(residual**2)) / total_variance
    )
    area, center, sigma, intercept, slope = [float(value) for value in params]
    amplitude = area / (sigma * math.sqrt(2.0 * math.pi))
    return DebyerCoordinationFitResult(
        r_min=float(fit_r[0]),
        r_max=float(fit_r[-1]),
        center=center,
        sigma=sigma,
        coordination_number=area,
        amplitude=float(amplitude),
        baseline_intercept=intercept,
        baseline_slope=slope,
        rmse=rmse,
        r_squared=r_squared,
        point_count=int(fit_r.size),
        fitted_values=np.asarray(fitted_values, dtype=float),
    )


def classify_partial_pair(
    pair_label: str,
    *,
    solute_elements: set[str] | None = None,
) -> str | None:
    if not solute_elements or "-" not in pair_label:
        return None
    left, right = pair_label.split("-", 1)
    first = _normalized_element(left)
    second = _normalized_element(right)
    first_is_solute = first in solute_elements
    second_is_solute = second in solute_elements
    if first_is_solute and second_is_solute:
        return "solute-solute"
    if not first_is_solute and not second_is_solute:
        return "solvent-solvent"
    return "solute-solvent"


def _is_grouped_partial_column(column_name: str) -> bool:
    return str(column_name) in GROUPED_PARTIAL_COLUMN_LABELS


def _raw_partial_values_from_output_values(
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value, dtype=float)
        for key, value in values.items()
        if not _is_grouped_partial_column(key)
    }


def build_grouped_partial_values(
    partial_values: dict[str, np.ndarray],
    *,
    solute_elements: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    if not solute_elements:
        return {}
    normalized_solutes = set(_normalize_solute_elements(solute_elements))
    grouped: dict[str, np.ndarray] = {}
    for pair_label, values in partial_values.items():
        if _is_grouped_partial_column(pair_label):
            continue
        family = classify_partial_pair(
            pair_label,
            solute_elements=normalized_solutes,
        )
        if family is None:
            continue
        current = grouped.get(family)
        if current is None:
            grouped[family] = np.asarray(values, dtype=float).copy()
        else:
            grouped[family] = current + np.asarray(values, dtype=float)
    return grouped


def _output_values_with_grouped_partials(
    *,
    column_order: list[str],
    values: dict[str, np.ndarray],
    solute_elements: tuple[str, ...],
) -> tuple[list[str], dict[str, np.ndarray]]:
    cleaned_order = [
        column
        for column in column_order
        if not _is_grouped_partial_column(column)
    ]
    cleaned_values = {
        key: np.asarray(value, dtype=float)
        for key, value in values.items()
        if not _is_grouped_partial_column(key)
    }
    if not solute_elements:
        return cleaned_order, cleaned_values

    raw_partials = {
        key: value for key, value in cleaned_values.items() if key != "sum"
    }
    grouped = build_grouped_partial_values(
        raw_partials,
        solute_elements=solute_elements,
    )
    output_values = dict(cleaned_values)
    output_order = list(cleaned_order)
    for label in GROUPED_PARTIAL_COLUMN_LABELS:
        if label not in grouped:
            continue
        output_values[label] = np.asarray(grouped[label], dtype=float)
        output_order.append(label)
    return output_order, output_values


def _build_averaged_output_metadata_from_calculation(
    calculation: DebyerPDFCalculation,
) -> dict[str, object]:
    processed_frames = (
        calculation.frame_count
        if calculation.processed_frame_count is None
        else int(calculation.processed_frame_count)
    )
    return {
        "calculation_id": calculation.calculation_id,
        "created_at": calculation.created_at,
        "filename_prefix": calculation.filename_prefix,
        "frames_dir": str(calculation.frames_dir),
        "frame_format": calculation.frame_format,
        "processed_frames": int(processed_frames),
        "total_frames": int(calculation.frame_count),
        "mode": calculation.mode,
        "from_value": calculation.from_value,
        "to_value": calculation.to_value,
        "step_value": calculation.step_value,
        "box_dimensions": ", ".join(
            f"{component:.6g}" for component in calculation.box_dimensions
        ),
        "box_source": calculation.box_source or "estimated/manual",
        "box_source_kind": calculation.box_source_kind or "estimate",
        "atom_count": calculation.atom_count,
        "rho0": f"{calculation.rho0:.8g}",
        "store_frame_outputs": calculation.store_frame_outputs,
        "solute_elements": (", ".join(calculation.solute_elements) or "None"),
        "parallel_jobs": int(calculation.parallel_jobs),
        "elapsed_seconds": (
            None
            if calculation.elapsed_seconds is None
            else f"{float(calculation.elapsed_seconds):.6f}"
        ),
        "estimated_remaining_seconds": (
            None
            if calculation.estimated_remaining_seconds is None
            else f"{float(calculation.estimated_remaining_seconds):.6f}"
        ),
        "expected_total_seconds": (
            None
            if calculation.expected_total_seconds is None
            else f"{float(calculation.expected_total_seconds):.6f}"
        ),
        "elapsed_hms": _format_duration(calculation.elapsed_seconds),
        "remaining_hms": _format_duration(
            calculation.estimated_remaining_seconds
        ),
        "expected_total_hms": _format_duration(
            calculation.expected_total_seconds
        ),
    }


def rewrite_debyer_calculation_output(
    calculation: DebyerPDFCalculation,
) -> None:
    column_order = ["sum"]
    if calculation.averaged_output_file.is_file():
        try:
            parsed_order = _parse_columns_from_comments(
                calculation.averaged_output_file
            )
        except Exception:
            parsed_order = []
        column_order = [
            column
            for column in parsed_order
            if column == "sum" or column in calculation.partial_values
        ] or ["sum"]
    for pair_label in sorted(calculation.partial_values):
        if pair_label not in column_order:
            column_order.append(pair_label)
    values = {
        "sum": np.asarray(calculation.total_values, dtype=float),
        **{
            key: np.asarray(value, dtype=float)
            for key, value in calculation.partial_values.items()
        },
    }
    output_order, output_values = _output_values_with_grouped_partials(
        column_order=column_order,
        values=values,
        solute_elements=calculation.solute_elements,
    )
    save_averaged_debyer_output(
        calculation.averaged_output_file,
        r_values=calculation.r_values,
        column_order=output_order,
        values=output_values,
        metadata=_build_averaged_output_metadata_from_calculation(calculation),
    )


def build_display_traces(
    calculation: DebyerPDFCalculation,
    *,
    representation: str = "g(r)",
    include_grouped_partials: bool = True,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = [
        {
            "key": "average",
            "label": "Average",
            "kind": "average",
            "values": convert_distribution_values(
                calculation.total_values,
                r_values=calculation.r_values,
                rho0=calculation.rho0,
                source_mode=calculation.mode,
                target_representation=representation,
                is_component=False,
            ),
        }
    ]
    for pair_label, values in sorted(calculation.partial_values.items()):
        traces.append(
            {
                "key": f"partial:{pair_label}",
                "label": pair_label,
                "kind": "partial",
                "values": convert_distribution_values(
                    values,
                    r_values=calculation.r_values,
                    rho0=calculation.rho0,
                    source_mode=calculation.mode,
                    target_representation=representation,
                    is_component=True,
                ),
            }
        )
    if include_grouped_partials:
        grouped = build_grouped_partial_values(
            calculation.partial_values,
            solute_elements=calculation.solute_elements,
        )
        for family, values in sorted(grouped.items()):
            traces.append(
                {
                    "key": f"group:{family}",
                    "label": family,
                    "kind": "group",
                    "values": convert_distribution_values(
                        values,
                        r_values=calculation.r_values,
                        rho0=calculation.rho0,
                        source_mode=calculation.mode,
                        target_representation=representation,
                        is_component=True,
                    ),
                }
            )
    return traces


def load_debyer_calculation(
    calculation_dir: str | Path,
) -> DebyerPDFCalculation:
    resolved_dir = Path(calculation_dir).expanduser().resolve()
    metadata_file = resolved_dir / "calculation.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"The Debyer calculation metadata file is missing: {metadata_file}"
        )
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    averaged_output_file = Path(payload["averaged_output_file"]).resolve()
    r_values, raw_values = parse_debyer_output_file(averaged_output_file)
    total_values = np.asarray(raw_values.pop("sum"), dtype=float)
    partial_values = _raw_partial_values_from_output_values(raw_values)
    peak_finder_settings = _coerce_peak_finder_settings(
        payload.get("peak_finder_settings")
    )
    stored_peak_markers = _deserialize_peak_markers(
        payload.get("partial_peak_markers")
    )
    stored_target_peak_markers = _deserialize_target_peak_markers(
        payload.get("target_peak_markers")
    )
    estimated_peak_markers = estimate_partial_peak_markers(
        r_values=r_values,
        partial_values=partial_values,
        settings=peak_finder_settings,
    )
    resolved_peak_markers: dict[str, tuple[DebyerPeakMarker, ...]] = {}
    needs_metadata_refresh = False
    for pair_label in sorted(partial_values):
        pair_markers = stored_peak_markers.get(pair_label)
        if pair_markers is None:
            pair_markers = estimated_peak_markers.get(pair_label, ())
            needs_metadata_refresh = True
        resolved_peak_markers[pair_label] = tuple(pair_markers)
    if "peak_finder_settings" not in payload:
        needs_metadata_refresh = True
    frame_output_dir_value = payload.get("frame_output_dir")
    calculation = DebyerPDFCalculation(
        calculation_id=str(payload["calculation_id"]),
        calculation_dir=resolved_dir,
        created_at=str(payload["created_at"]),
        project_dir=Path(payload["project_dir"]).resolve(),
        frames_dir=Path(payload["frames_dir"]).resolve(),
        frame_format=str(payload["frame_format"]),
        frame_count=int(payload["frame_count"]),
        filename_prefix=str(payload["filename_prefix"]),
        mode=str(payload["mode"]),
        from_value=float(payload["from_value"]),
        to_value=float(payload["to_value"]),
        step_value=float(payload["step_value"]),
        box_dimensions=tuple(
            float(component)
            for component in payload.get("box_dimensions", (0.0, 0.0, 0.0))
        ),
        box_source=payload.get("box_source"),
        box_source_kind=payload.get("box_source_kind"),
        atom_count=int(payload["atom_count"]),
        rho0=float(payload["rho0"]),
        store_frame_outputs=bool(payload.get("store_frame_outputs", False)),
        frame_output_dir=(
            None
            if not frame_output_dir_value
            else Path(frame_output_dir_value).resolve()
        ),
        averaged_output_file=averaged_output_file,
        solute_elements=_normalize_solute_elements(
            payload.get("solute_elements", [])
        ),
        parallel_jobs=_coerce_parallel_debyer_jobs(
            payload.get("parallel_jobs", 1)
        ),
        r_values=r_values,
        total_values=total_values,
        partial_values=partial_values,
        processed_frame_count=int(
            payload.get("processed_frame_count", payload["frame_count"])
        ),
        is_partial_average=bool(payload.get("is_partial_average", False)),
        elapsed_seconds=(
            None
            if payload.get("elapsed_seconds") is None
            else float(payload["elapsed_seconds"])
        ),
        estimated_remaining_seconds=(
            None
            if payload.get("estimated_remaining_seconds") is None
            else float(payload["estimated_remaining_seconds"])
        ),
        expected_total_seconds=(
            None
            if payload.get("expected_total_seconds") is None
            else float(payload["expected_total_seconds"])
        ),
        partial_peak_markers=resolved_peak_markers,
        target_peak_markers=stored_target_peak_markers,
        peak_finder_settings=peak_finder_settings,
    )
    if needs_metadata_refresh:
        write_debyer_calculation_metadata(calculation)
    return calculation


def list_saved_debyer_calculations(
    project_dir: str | Path,
) -> list[DebyerPDFCalculationSummary]:
    root_dir = build_debyer_project_dir(project_dir)
    if not root_dir.is_dir():
        return []

    summaries: list[DebyerPDFCalculationSummary] = []
    for candidate in sorted(root_dir.iterdir()):
        metadata_file = candidate / "calculation.json"
        if not metadata_file.is_file():
            continue
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append(
            DebyerPDFCalculationSummary(
                calculation_id=str(payload["calculation_id"]),
                calculation_dir=candidate.resolve(),
                created_at=str(payload["created_at"]),
                filename_prefix=str(payload["filename_prefix"]),
                mode=str(payload["mode"]),
                frame_count=int(payload["frame_count"]),
                frames_dir=Path(payload["frames_dir"]).resolve(),
            )
        )
    summaries.sort(key=lambda entry: entry.created_at, reverse=True)
    return summaries


class DebyerPDFWorkflow:
    """Run Debyer PDF or partial-PDF calculations across a
    trajectory."""

    def __init__(
        self,
        settings: DebyerPDFSettings,
        *,
        debyer_executable: str | Path | None = None,
    ) -> None:
        if settings.mode not in SUPPORTED_DEBYER_MODES:
            raise ValueError(
                "Debyer mode must be one of "
                + ", ".join(SUPPORTED_DEBYER_MODES)
            )
        self.settings = DebyerPDFSettings(
            project_dir=Path(settings.project_dir).expanduser().resolve(),
            frames_dir=Path(settings.frames_dir).expanduser().resolve(),
            filename_prefix=_sanitize_prefix(settings.filename_prefix),
            mode=str(settings.mode),
            from_value=float(settings.from_value),
            to_value=float(settings.to_value),
            step_value=float(settings.step_value),
            box_dimensions=tuple(
                float(component) for component in settings.box_dimensions
            ),
            atom_count=int(settings.atom_count),
            store_frame_outputs=bool(settings.store_frame_outputs),
            solute_elements=_normalize_solute_elements(
                settings.solute_elements
            ),
            max_parallel_jobs=_coerce_parallel_debyer_jobs(
                settings.max_parallel_jobs
            ),
        )
        self.debyer_executable = (
            None
            if debyer_executable is None
            else Path(debyer_executable).expanduser().resolve()
        )
        self._cached_runtime_status: DebyerRuntimeStatus | None = None
        self._cached_inspection: DebyerFrameInspection | None = None

    def check_runtime(self) -> DebyerRuntimeStatus:
        if self._cached_runtime_status is None:
            self._cached_runtime_status = check_debyer_runtime(
                self.debyer_executable
            )
        return self._cached_runtime_status

    def inspect_frames(self) -> DebyerFrameInspection:
        if self._cached_inspection is None:
            self._cached_inspection = inspect_frames_dir(
                self.settings.frames_dir
            )
        return self._cached_inspection

    def _run_debyer_frame(
        self,
        *,
        frame_index: int,
        frame_path: Path,
        output_path: Path,
        rho0: float,
        executable_path: Path | None,
    ) -> _DebyerFrameRunResult:
        command = self._build_command(
            input_file=frame_path,
            output_file=output_path,
            rho0=rho0,
            executable_path=executable_path,
        )
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Debyer failed on {frame_path.name}: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        frame_r_values, frame_values = parse_debyer_output_file(output_path)
        return _DebyerFrameRunResult(
            frame_index=frame_index,
            frame_path=frame_path,
            output_path=output_path,
            r_values=frame_r_values,
            values=frame_values,
        )

    def run(
        self,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        preview_callback: Callable[[DebyerPDFCalculation], None] | None = None,
        preview_decision_callback: (
            Callable[[int, int, bool], bool] | None
        ) = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> DebyerPDFCalculation:
        runtime_status = self.check_runtime()
        if not runtime_status.runnable:
            raise RuntimeError(runtime_status.message)

        inspection = self.inspect_frames()
        if not self.settings.solute_elements:
            inferred_solutes = infer_default_solute_elements(
                inspection.element_counts
            )
            if inferred_solutes:
                self.settings = replace(
                    self.settings,
                    solute_elements=inferred_solutes,
                )
        calculation_id = _build_calculation_id(self.settings.filename_prefix)
        created_at = (
            datetime.now(timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        calculation_dir = (
            build_debyer_project_dir(self.settings.project_dir)
            / calculation_id
        )
        calculation_dir.mkdir(parents=True, exist_ok=True)
        frame_output_dir = calculation_dir / "frame_outputs"
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        averaged_output_file = calculation_dir / "averaged_raw.txt"

        rho0 = calculate_number_density(
            self.settings.atom_count,
            self.settings.box_dimensions,
        )
        peak_finder_settings = DebyerPeakFinderSettings()
        total_frames = len(inspection.frame_paths)
        parallel_jobs = _resolve_parallel_debyer_jobs(
            self.settings.max_parallel_jobs,
            total_frames=total_frames,
        )
        if status_callback is not None:
            status_callback("Running Debyer over trajectory frames")
        if log_callback is not None:
            log_callback(
                "Starting Debyer "
                f"{self.settings.mode} calculation on {total_frames} frames"
            )
            log_callback(
                f"Running up to {parallel_jobs} Debyer "
                f"{'job' if parallel_jobs == 1 else 'jobs'} in parallel"
            )
            log_callback(
                "Bounding box: "
                + " x ".join(
                    f"{component:.4g}"
                    for component in self.settings.box_dimensions
                )
                + f" A; rho0={rho0:.6g} atoms/A^3"
            )

        running_average = _RunningDebyerAverage()
        start_time = time.monotonic()
        last_verbose_log = time.monotonic()
        checkpoint_interval = _average_checkpoint_interval(
            total_frames=total_frames,
        )
        latest_preview: DebyerPDFCalculation | None = None
        cancelled = False
        frame_iterator = iter(enumerate(inspection.frame_paths, start=1))
        active_futures: dict[
            concurrent.futures.Future[_DebyerFrameRunResult],
            int,
        ] = {}

        def submit_next_frame(
            executor: concurrent.futures.ThreadPoolExecutor,
        ) -> bool:
            try:
                frame_index, frame_path = next(frame_iterator)
            except StopIteration:
                return False
            output_path = frame_output_dir / f"{frame_path.stem}.txt"
            future = executor.submit(
                self._run_debyer_frame,
                frame_index=frame_index,
                frame_path=frame_path,
                output_path=output_path,
                rho0=rho0,
                executable_path=runtime_status.executable_path,
            )
            active_futures[future] = frame_index
            return True

        def process_frame_result(
            frame_result: _DebyerFrameRunResult,
        ) -> None:
            nonlocal checkpoint_interval, latest_preview, last_verbose_log
            running_average.add_frame(
                frame_result.r_values,
                frame_result.values,
            )
            processed_frames = running_average.processed_count
            checkpoint_interval = _average_checkpoint_interval(
                total_frames=total_frames,
                average_state_bytes=running_average.memory_bytes,
            )
            elapsed_seconds = time.monotonic() - start_time
            (
                estimated_remaining_seconds,
                expected_total_seconds,
            ) = _estimate_runtime(
                processed_frames=processed_frames,
                total_frames=total_frames,
                elapsed_seconds=elapsed_seconds,
            )

            if (
                not self.settings.store_frame_outputs
                and frame_result.output_path.exists()
            ):
                frame_result.output_path.unlink()
            if progress_callback is not None:
                progress_message = (
                    f"Processed {processed_frames}/{total_frames} frames | "
                    f"elapsed {_format_duration(elapsed_seconds)} | "
                    f"remaining {_format_duration(estimated_remaining_seconds)}"
                )
                progress_callback(
                    processed_frames,
                    total_frames,
                    progress_message,
                )
            checkpoint_due = (
                processed_frames == total_frames
                or processed_frames % checkpoint_interval == 0
            )
            should_refresh_average = checkpoint_due
            if (
                preview_callback is not None
                and preview_decision_callback is not None
            ):
                should_refresh_average = bool(
                    preview_decision_callback(
                        processed_frames,
                        total_frames,
                        checkpoint_due,
                    )
                )
            if should_refresh_average:
                (
                    preview_r_values,
                    preview_column_order,
                    preview_values,
                ) = running_average.average()
                (
                    preview_output_column_order,
                    preview_output_values,
                ) = _output_values_with_grouped_partials(
                    column_order=preview_column_order,
                    values=preview_values,
                    solute_elements=self.settings.solute_elements,
                )
                save_averaged_debyer_output(
                    averaged_output_file,
                    r_values=preview_r_values,
                    column_order=preview_output_column_order,
                    values=preview_output_values,
                    metadata=_build_averaged_output_metadata(
                        calculation_id=calculation_id,
                        created_at=created_at,
                        settings=self.settings,
                        inspection=inspection,
                        rho0=rho0,
                        processed_frames=processed_frames,
                        total_frames=total_frames,
                        elapsed_seconds=elapsed_seconds,
                        estimated_remaining_seconds=(
                            estimated_remaining_seconds
                        ),
                        expected_total_seconds=expected_total_seconds,
                        parallel_jobs=parallel_jobs,
                    ),
                )
                latest_preview = DebyerPDFCalculation(
                    calculation_id=calculation_id,
                    calculation_dir=calculation_dir,
                    created_at=created_at,
                    project_dir=self.settings.project_dir,
                    frames_dir=self.settings.frames_dir,
                    frame_format=inspection.frame_format,
                    frame_count=total_frames,
                    filename_prefix=self.settings.filename_prefix,
                    mode=self.settings.mode,
                    from_value=self.settings.from_value,
                    to_value=self.settings.to_value,
                    step_value=self.settings.step_value,
                    box_dimensions=self.settings.box_dimensions,
                    box_source=inspection.detected_box_source,
                    box_source_kind=inspection.detected_box_source_kind,
                    atom_count=self.settings.atom_count,
                    rho0=rho0,
                    store_frame_outputs=self.settings.store_frame_outputs,
                    frame_output_dir=frame_output_dir,
                    averaged_output_file=averaged_output_file,
                    solute_elements=self.settings.solute_elements,
                    parallel_jobs=parallel_jobs,
                    r_values=preview_r_values,
                    total_values=np.asarray(
                        preview_values["sum"],
                        dtype=float,
                    ),
                    partial_values={
                        key: np.asarray(value, dtype=float)
                        for key, value in preview_values.items()
                        if key != "sum" and not _is_grouped_partial_column(key)
                    },
                    processed_frame_count=processed_frames,
                    is_partial_average=processed_frames < total_frames,
                    elapsed_seconds=elapsed_seconds,
                    estimated_remaining_seconds=estimated_remaining_seconds,
                    expected_total_seconds=expected_total_seconds,
                    partial_peak_markers={},
                    target_peak_markers={},
                    peak_finder_settings=peak_finder_settings,
                )
                if preview_callback is not None:
                    preview_callback(latest_preview)
            if log_callback is not None:
                should_log = (
                    processed_frames == 1
                    or processed_frames == total_frames
                    or (time.monotonic() - last_verbose_log) >= 5.0
                )
                if should_log:
                    checkpoint_text = (
                        f"; checkpoint every {checkpoint_interval} frames"
                        if processed_frames == 1
                        else ""
                    )
                    log_callback(
                        f"Processed {processed_frames}/{total_frames} frames "
                        f"({frame_result.frame_path.name}) | elapsed "
                        f"{_format_duration(elapsed_seconds)} | remaining "
                        f"{_format_duration(estimated_remaining_seconds)}"
                        f"{checkpoint_text}"
                    )
                    last_verbose_log = time.monotonic()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=parallel_jobs,
            thread_name_prefix="debyer-pdf",
        ) as executor:
            for _worker_index in range(parallel_jobs):
                if not submit_next_frame(executor):
                    break
            while active_futures:
                done_futures, _pending_futures = concurrent.futures.wait(
                    active_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done_futures:
                    active_futures.pop(future, None)
                    process_frame_result(future.result())
                    if (
                        cancel_callback is not None
                        and cancel_callback()
                        and running_average.processed_count < total_frames
                    ):
                        if not cancelled:
                            cancelled = True
                            if log_callback is not None:
                                active_count = len(active_futures)
                                suffix = (
                                    f" Waiting for {active_count} active "
                                    "Debyer job(s) to finish."
                                    if active_count
                                    else ""
                                )
                                log_callback(
                                    "Debyer calculation stop requested; saving "
                                    "the current average after "
                                    f"{running_average.processed_count}/"
                                    f"{total_frames} frames.{suffix}"
                                )
                    if not cancelled:
                        submit_next_frame(executor)

        if (
            not self.settings.store_frame_outputs
            and frame_output_dir.is_dir()
            and not any(frame_output_dir.iterdir())
        ):
            frame_output_dir.rmdir()
            stored_frame_output_dir: Path | None = None
        else:
            stored_frame_output_dir = frame_output_dir

        processed_frame_count = running_average.processed_count
        if (
            latest_preview is None
            or latest_preview.processed_frame_count != processed_frame_count
        ):
            elapsed_seconds = time.monotonic() - start_time
            (
                estimated_remaining_seconds,
                expected_total_seconds,
            ) = _estimate_runtime(
                processed_frames=processed_frame_count,
                total_frames=total_frames,
                elapsed_seconds=elapsed_seconds,
            )
            r_values, column_order, averaged_values = running_average.average()
            (
                output_column_order,
                output_values,
            ) = _output_values_with_grouped_partials(
                column_order=column_order,
                values=averaged_values,
                solute_elements=self.settings.solute_elements,
            )
            save_averaged_debyer_output(
                averaged_output_file,
                r_values=r_values,
                column_order=output_column_order,
                values=output_values,
                metadata=_build_averaged_output_metadata(
                    calculation_id=calculation_id,
                    created_at=created_at,
                    settings=self.settings,
                    inspection=inspection,
                    rho0=rho0,
                    processed_frames=processed_frame_count,
                    total_frames=total_frames,
                    elapsed_seconds=elapsed_seconds,
                    estimated_remaining_seconds=(estimated_remaining_seconds),
                    expected_total_seconds=expected_total_seconds,
                    parallel_jobs=parallel_jobs,
                ),
            )
        else:
            elapsed_seconds = latest_preview.elapsed_seconds
            estimated_remaining_seconds = (
                latest_preview.estimated_remaining_seconds
            )
            expected_total_seconds = latest_preview.expected_total_seconds

        final_r_values, final_raw_values = parse_debyer_output_file(
            averaged_output_file
        )
        final_total_values = np.asarray(
            final_raw_values.pop("sum"), dtype=float
        )
        final_partial_values = _raw_partial_values_from_output_values(
            final_raw_values
        )
        final_calculation = DebyerPDFCalculation(
            calculation_id=calculation_id,
            calculation_dir=calculation_dir,
            created_at=created_at,
            project_dir=self.settings.project_dir,
            frames_dir=self.settings.frames_dir,
            frame_format=inspection.frame_format,
            frame_count=total_frames,
            filename_prefix=self.settings.filename_prefix,
            mode=self.settings.mode,
            from_value=self.settings.from_value,
            to_value=self.settings.to_value,
            step_value=self.settings.step_value,
            box_dimensions=self.settings.box_dimensions,
            box_source=inspection.detected_box_source,
            box_source_kind=inspection.detected_box_source_kind,
            atom_count=self.settings.atom_count,
            rho0=rho0,
            store_frame_outputs=self.settings.store_frame_outputs,
            frame_output_dir=stored_frame_output_dir,
            averaged_output_file=averaged_output_file,
            solute_elements=self.settings.solute_elements,
            parallel_jobs=parallel_jobs,
            r_values=final_r_values,
            total_values=final_total_values,
            partial_values=final_partial_values,
            processed_frame_count=processed_frame_count,
            is_partial_average=processed_frame_count < total_frames,
            elapsed_seconds=elapsed_seconds,
            estimated_remaining_seconds=estimated_remaining_seconds,
            expected_total_seconds=expected_total_seconds,
            partial_peak_markers=estimate_partial_peak_markers(
                r_values=final_r_values,
                partial_values=final_partial_values,
                settings=peak_finder_settings,
            ),
            target_peak_markers={},
            peak_finder_settings=peak_finder_settings,
        )
        write_debyer_calculation_metadata(final_calculation)
        if log_callback is not None:
            log_callback(
                f"Saved averaged Debyer output to {averaged_output_file}"
            )
        if status_callback is not None:
            status_callback(
                "Debyer calculation stopped early"
                if cancelled
                else "Debyer calculation complete"
            )
        return final_calculation

    def _build_command(
        self,
        *,
        input_file: Path,
        output_file: Path,
        rho0: float,
        executable_path: Path | None,
    ) -> list[str]:
        executable = (
            str(executable_path)
            if executable_path is not None
            else str(self.debyer_executable)
        )
        box_a, box_b, box_c = self.settings.box_dimensions
        return [
            executable,
            f"--{self.settings.mode}",
            f"--pbc-a={box_a}",
            f"--pbc-b={box_b}",
            f"--pbc-c={box_c}",
            f"--from={self.settings.from_value}",
            f"--to={self.settings.to_value}",
            f"--step={self.settings.step_value}",
            "--weight=x",
            "--partials",
            f"--ro={rho0}",
            f"--output={output_file}",
            str(input_file),
        ]


__all__ = [
    "DEBYER_DOCS_URL",
    "DEBYER_GITHUB_URL",
    "DEFAULT_COLOR_SCHEMES",
    "GROUPED_PARTIAL_COLUMN_LABELS",
    "TOTAL_SCATTERING_PAPER_URL",
    "DebyerCoordinationFitResult",
    "DebyerFitMetrics",
    "DebyerFrameInspection",
    "DebyerPeakFinderSettings",
    "DebyerPeakMarker",
    "DebyerPDFCalculation",
    "DebyerPDFCalculationSummary",
    "DebyerPDFSettings",
    "DebyerPDFWorkflow",
    "DebyerRuntimeStatus",
    "SUPPORTED_DEBYER_MODES",
    "SUPPORTED_PLOT_REPRESENTATIONS",
    "build_debyer_project_dir",
    "build_display_traces",
    "build_grouped_partial_values",
    "calculate_number_density",
    "check_debyer_runtime",
    "classify_partial_pair",
    "compute_experimental_fit_metrics",
    "convert_distribution_values",
    "default_parallel_debyer_jobs",
    "estimate_partial_peak_markers",
    "find_partial_peak_markers",
    "fit_coordination_peak_from_r",
    "infer_default_solute_elements",
    "inspect_frames_dir",
    "list_saved_debyer_calculations",
    "load_debyer_calculation",
    "parse_debyer_output_file",
    "rewrite_debyer_calculation_output",
    "save_averaged_debyer_output",
    "write_debyer_calculation_metadata",
]
