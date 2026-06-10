from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from re import sub
from typing import Callable, Iterable, Sequence

import numpy as np
from matplotlib.figure import Figure

from .bondanalyzer import (
    AngleTripletDefinition,
    BondAnalyzer,
    BondPairDefinition,
    CoordinationNumberDefinition,
    DihedralQuartetDefinition,
    expanded_solvent_dihedral_quartets,
)
from .results import LEGACY_RESULTS_INDEX_FILENAME, RESULTS_INDEX_FILENAME

ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]
ANALYSIS_SIGNATURE_VERSION = 1


@dataclass(frozen=True, slots=True)
class ClusterTypeSummary:
    """One discovered stoichiometry-level cluster folder."""

    name: str
    path: Path
    structure_files: tuple[Path, ...]

    @property
    def structure_count(self) -> int:
        return len(self.structure_files)


@dataclass(slots=True)
class BondAnalysisClusterResult:
    """Per-cluster-type output summary."""

    cluster_type: str
    structure_count: int
    output_dir: Path
    bond_value_counts: dict[str, int]
    angle_value_counts: dict[str, int]
    dihedral_value_counts: dict[str, int]
    coordination_value_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_type": self.cluster_type,
            "structure_count": self.structure_count,
            "output_dir": str(self.output_dir),
            "bond_value_counts": dict(self.bond_value_counts),
            "angle_value_counts": dict(self.angle_value_counts),
            "dihedral_value_counts": dict(self.dihedral_value_counts),
            "coordination_value_counts": dict(self.coordination_value_counts),
        }


@dataclass(slots=True)
class BondAnalysisBatchResult:
    """Top-level output summary for one run."""

    clusters_dir: Path
    output_dir: Path
    selected_cluster_types: tuple[str, ...]
    total_structure_files: int
    cluster_results: list[BondAnalysisClusterResult]
    results_index_path: Path
    analysis_signature: str | None = None
    reused_existing_result: bool = False

    @property
    def manifest_path(self) -> Path:
        """Backward-compatible alias for older callers."""
        return self.results_index_path

    def to_dict(self) -> dict[str, object]:
        return {
            "clusters_dir": str(self.clusters_dir),
            "output_dir": str(self.output_dir),
            "selected_cluster_types": list(self.selected_cluster_types),
            "total_structure_files": self.total_structure_files,
            "results_index_path": str(self.results_index_path),
            "analysis_signature": self.analysis_signature,
            "reused_existing_result": self.reused_existing_result,
            "cluster_results": [
                result.to_dict() for result in self.cluster_results
            ],
        }


def next_available_output_dir(parent_dir: Path, folder_name: str) -> Path:
    """Return the next available output directory beside the source."""
    candidate = parent_dir / folder_name
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = parent_dir / f"{folder_name}{index:04d}"
        if not candidate.exists():
            return candidate
        index += 1


def suggest_bondanalysis_output_dir(clusters_dir: str | Path) -> Path:
    """Suggest a sibling directory for bond-analysis output."""
    source_path = Path(clusters_dir)
    folder_name = _base_output_dir_name(source_path)
    return next_available_output_dir(source_path.parent, folder_name)


def _base_output_dir_name(clusters_dir: Path) -> str:
    folder_label = sub(r"[^0-9A-Za-z]+", "_", clusters_dir.name).strip("_")
    if not folder_label:
        folder_label = "clusters"
    return f"bondanalysis_{folder_label}"


def _gds_token(value: str) -> str:
    token = sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")
    return token or "value"


def _resolved_path_text(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _canonical_definition_payloads(
    definitions: Iterable[object],
) -> list[dict[str, object]]:
    payloads = [
        dict(definition.to_dict())  # type: ignore[attr-defined]
        for definition in definitions
    ]
    return sorted(
        payloads,
        key=lambda payload: json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _canonical_payload_values(values: object) -> list[dict[str, object]]:
    if not isinstance(values, IterableABC) or isinstance(values, (str, bytes)):
        return []
    payloads = [dict(value) for value in values if isinstance(value, Mapping)]
    return sorted(
        payloads,
        key=lambda payload: json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _structure_file_signature_payload(
    structure_file: Path,
    *,
    clusters_root: Path,
) -> dict[str, object]:
    resolved_file = structure_file.expanduser().resolve()
    try:
        relative_path = resolved_file.relative_to(clusters_root)
    except ValueError:
        relative_path = Path(resolved_file.name)
    stat = resolved_file.stat()
    return {
        "path": relative_path.as_posix(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _analysis_signature_from_payload(
    payload: Mapping[str, object],
) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _discover_result_index_paths(
    search_roots: Iterable[Path],
) -> tuple[Path, ...]:
    index_paths: list[Path] = []
    seen: set[Path] = set()
    for raw_root in search_roots:
        root = Path(raw_root).expanduser()
        candidates: list[Path] = []
        if root.is_file() and root.name in {
            RESULTS_INDEX_FILENAME,
            LEGACY_RESULTS_INDEX_FILENAME,
        }:
            candidates.append(root)
        elif root.is_dir():
            for filename in (
                RESULTS_INDEX_FILENAME,
                LEGACY_RESULTS_INDEX_FILENAME,
            ):
                exact_path = root / filename
                if exact_path.is_file():
                    candidates.append(exact_path)
                candidates.extend(root.rglob(filename))
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            index_paths.append(candidate)
    index_paths.sort(
        key=lambda path: (
            path.stat().st_mtime if path.exists() else 0.0,
            str(path),
        ),
        reverse=True,
    )
    return tuple(index_paths)


def _load_result_index_payload(
    index_path: Path,
) -> dict[str, object] | None:
    try:
        payload = json.loads(index_path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _count_mapping(payload: object) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, value in payload.items():
        try:
            counts[str(key)] = int(value)
        except (TypeError, ValueError):
            counts[str(key)] = 0
    return counts


def _progress_definition_labels(
    definitions: Sequence[object],
    *,
    max_labels: int = 3,
) -> str:
    labels = [
        str(getattr(definition, "display_label", definition))
        for definition in definitions
    ]
    if len(labels) <= max_labels:
        return ", ".join(labels)
    shown = ", ".join(labels[:max_labels])
    return f"{shown}, +{len(labels) - max_labels} more"


def _progress_count_label(
    count: int,
    singular: str,
    plural: str,
) -> str:
    if count <= 0:
        return ""
    label = singular if count == 1 else plural
    return f"{count} {label}"


def _should_emit_structure_progress(index: int, total: int) -> bool:
    if index <= 1 or index >= total:
        return True
    interval = max(1, total // 20)
    return index % interval == 0


def _raise_if_cancel_requested(cancel_callback: CancelCallback | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise InterruptedError("Bond analysis canceled by user.")


def _batch_result_from_index_payload(
    index_path: Path,
    payload: Mapping[str, object],
    *,
    reused_existing_result: bool,
) -> BondAnalysisBatchResult:
    output_dir = Path(str(payload.get("output_dir") or index_path.parent))
    clusters_dir = Path(str(payload.get("clusters_dir") or ""))
    cluster_results: list[BondAnalysisClusterResult] = []
    for entry in payload.get("cluster_results", []):
        if not isinstance(entry, Mapping):
            continue
        cluster_type = str(entry.get("cluster_type", "")).strip()
        if not cluster_type:
            continue
        cluster_results.append(
            BondAnalysisClusterResult(
                cluster_type=cluster_type,
                structure_count=int(entry.get("structure_count", 0) or 0),
                output_dir=Path(
                    str(
                        entry.get("output_dir")
                        or output_dir / "cluster_types" / cluster_type
                    )
                ),
                bond_value_counts=_count_mapping(
                    entry.get("bond_value_counts")
                ),
                angle_value_counts=_count_mapping(
                    entry.get("angle_value_counts")
                ),
                dihedral_value_counts=_count_mapping(
                    entry.get("dihedral_value_counts")
                ),
                coordination_value_counts=_count_mapping(
                    entry.get("coordination_value_counts")
                ),
            )
        )
    selected_cluster_types = tuple(
        str(value)
        for value in payload.get("selected_cluster_types", [])
        if str(value).strip()
    )
    if not selected_cluster_types:
        selected_cluster_types = tuple(
            result.cluster_type for result in cluster_results
        )
    total_structure_files = int(
        payload.get("total_structure_files")
        or sum(result.structure_count for result in cluster_results)
    )
    analysis_signature = payload.get("analysis_signature")
    return BondAnalysisBatchResult(
        clusters_dir=clusters_dir,
        output_dir=output_dir,
        selected_cluster_types=selected_cluster_types,
        total_structure_files=total_structure_files,
        cluster_results=cluster_results,
        results_index_path=index_path,
        analysis_signature=(
            str(analysis_signature) if analysis_signature else None
        ),
        reused_existing_result=reused_existing_result,
    )


def discover_cluster_types(
    clusters_dir: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[ClusterTypeSummary]:
    """Discover stoichiometry-level cluster folders.

    The expected layout is a root directory containing one folder per
    cluster type, with structure files directly inside each of those
    folders. If the selected directory itself contains structure files
    directly, it is treated as one single cluster type.
    """

    source_path = Path(clusters_dir)
    if not source_path.is_dir():
        raise ValueError(f"Clusters directory does not exist: {source_path}")

    analyzer = BondAnalyzer()
    if progress_callback is not None:
        progress_callback(
            0,
            1,
            f"Scanning selected clusters directory: {source_path}",
        )
    direct_files = tuple(analyzer.structure_files(source_path))
    if direct_files:
        if progress_callback is not None:
            progress_callback(
                1,
                1,
                f"Detected {len(direct_files)} structure file(s) directly.",
            )
        return [
            ClusterTypeSummary(
                name=source_path.name,
                path=source_path,
                structure_files=direct_files,
            )
        ]

    summaries: list[ClusterTypeSummary] = []
    child_dirs = [
        child for child in sorted(source_path.iterdir()) if child.is_dir()
    ]
    total_children = max(len(child_dirs), 1)
    if progress_callback is not None:
        progress_callback(
            0,
            total_children,
            f"Inspecting {len(child_dirs)} cluster folder(s).",
        )
    for index, child in enumerate(child_dirs, start=1):
        if progress_callback is not None:
            progress_callback(
                index,
                total_children,
                (
                    f"Inspecting cluster folder {child.name} "
                    f"({index}/{total_children})."
                ),
            )
        structure_files = tuple(analyzer.structure_files(child))
        if not structure_files:
            continue
        summaries.append(
            ClusterTypeSummary(
                name=child.name,
                path=child,
                structure_files=structure_files,
            )
        )
    if progress_callback is not None:
        progress_callback(
            total_children,
            total_children,
            f"Discovered {len(summaries)} cluster type(s).",
        )
    return summaries


class BondAnalysisWorkflow:
    """Shared workflow used by the UI, CLI, and notebook entry
    points."""

    def __init__(
        self,
        clusters_dir: str | Path,
        *,
        bond_pairs: Iterable[BondPairDefinition] | None = None,
        angle_triplets: Iterable[AngleTripletDefinition] | None = None,
        dihedral_quartets: Iterable[DihedralQuartetDefinition] | None = None,
        coordination_numbers: (
            Iterable[CoordinationNumberDefinition] | None
        ) = None,
        output_dir: str | Path | None = None,
        selected_cluster_types: Sequence[str] | None = None,
        structure_distribution_store_dir: str | Path | None = None,
        generate_preview_plots: bool = True,
    ) -> None:
        self.clusters_dir = Path(clusters_dir)
        self.bond_pairs = tuple(dict.fromkeys(bond_pairs or ()))
        self.angle_triplets = tuple(dict.fromkeys(angle_triplets or ()))
        self.dihedral_quartets = expanded_solvent_dihedral_quartets(
            dihedral_quartets or ()
        )
        self.coordination_numbers = tuple(
            dict.fromkeys(coordination_numbers or ())
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.structure_distribution_store_dir = (
            None
            if structure_distribution_store_dir is None
            else Path(structure_distribution_store_dir)
        )
        self.generate_preview_plots = bool(generate_preview_plots)
        self.selected_cluster_types = (
            tuple(selected_cluster_types)
            if selected_cluster_types is not None
            else None
        )

    def inspect(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        cluster_types = discover_cluster_types(
            self.clusters_dir,
            progress_callback=progress_callback,
        )
        return {
            "clusters_dir": str(self.clusters_dir),
            "cluster_types": [summary.name for summary in cluster_types],
            "cluster_type_count": len(cluster_types),
            "total_structure_files": sum(
                summary.structure_count for summary in cluster_types
            ),
            "suggested_output_dir": str(
                self.output_dir
                if self.output_dir is not None
                else suggest_bondanalysis_output_dir(self.clusters_dir)
            ),
        }

    def analysis_signature_payload(self) -> dict[str, object]:
        """Return the canonical run-settings payload used for cache
        keys."""
        self._validate_requested_distributions()
        cluster_summaries = self._selected_cluster_summaries()
        if not cluster_summaries:
            raise ValueError("No cluster types were selected for analysis.")
        return self._build_analysis_signature_payload(cluster_summaries)

    def analysis_signature(self) -> str:
        """Return the stable fingerprint for this workflow's inputs."""
        return _analysis_signature_from_payload(
            self.analysis_signature_payload()
        )

    def find_matching_existing_result(self) -> BondAnalysisBatchResult | None:
        """Find an already-written result index for the same run
        inputs."""
        self._validate_requested_distributions()
        cluster_summaries = self._selected_cluster_summaries()
        if not cluster_summaries:
            raise ValueError("No cluster types were selected for analysis.")
        output_dir = (
            self.output_dir
            if self.output_dir is not None
            else suggest_bondanalysis_output_dir(self.clusters_dir)
        )
        signature_payload = self._build_analysis_signature_payload(
            cluster_summaries
        )
        signature = _analysis_signature_from_payload(signature_payload)
        return self._find_matching_existing_result(
            output_dir=output_dir,
            analysis_signature=signature,
            signature_payload=signature_payload,
        )

    def _active_distribution_progress_text(self) -> str:
        parts: list[str] = []
        if self.bond_pairs:
            parts.append(
                _progress_count_label(
                    len(self.bond_pairs),
                    "bond distribution",
                    "bond distributions",
                )
            )
        if self.angle_triplets:
            parts.append(
                _progress_count_label(
                    len(self.angle_triplets),
                    "angle distribution",
                    "angle distributions",
                )
            )
        if self.dihedral_quartets:
            parts.append(
                _progress_count_label(
                    len(self.dihedral_quartets),
                    "dihedral distribution",
                    "dihedral distributions",
                )
            )
        if self.coordination_numbers:
            parts.append(
                _progress_count_label(
                    len(self.coordination_numbers),
                    "coordination distribution",
                    "coordination distributions",
                )
            )
        parts = [part for part in parts if part]
        return "; ".join(parts) or "requested distributions"

    def run(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> BondAnalysisBatchResult:
        self._validate_requested_distributions()

        _raise_if_cancel_requested(cancel_callback)
        if progress_callback is not None:
            progress_callback(0, 1, "Inspecting selected clusters.")
        cluster_summaries = self._selected_cluster_summaries()
        if not cluster_summaries:
            raise ValueError("No cluster types were selected for analysis.")

        _raise_if_cancel_requested(cancel_callback)
        output_dir = (
            self.output_dir
            if self.output_dir is not None
            else suggest_bondanalysis_output_dir(self.clusters_dir)
        )
        total_files = sum(
            summary.structure_count for summary in cluster_summaries
        )
        if progress_callback is not None:
            progress_callback(
                0,
                max(total_files, 1),
                "Checking for matching stored results.",
            )
        _raise_if_cancel_requested(cancel_callback)
        signature_payload = self._build_analysis_signature_payload(
            cluster_summaries
        )
        analysis_signature = _analysis_signature_from_payload(
            signature_payload
        )
        _raise_if_cancel_requested(cancel_callback)
        existing_result = self._find_matching_existing_result(
            output_dir=output_dir,
            analysis_signature=analysis_signature,
            signature_payload=signature_payload,
        )
        if existing_result is not None:
            if progress_callback is not None:
                progress_callback(
                    1,
                    1,
                    "Loaded matching stored bond-analysis results.",
                )
            if log_callback is not None:
                log_callback(
                    "Reusing matching stored bond-analysis results from "
                    f"{existing_result.output_dir}."
                )
                log_callback(
                    "Results index file: "
                    f"{existing_result.results_index_path}."
                )
            return existing_result

        from saxshell.structure_distributions import StructureDistributionStore

        store_dir = (
            self.structure_distribution_store_dir
            if self.structure_distribution_store_dir is not None
            else output_dir / "structure_distribution_store"
        )
        distribution_store = StructureDistributionStore(store_dir)
        cluster_root = output_dir / "cluster_types"
        aggregate_root = output_dir / "all_clusters"
        comparison_root = output_dir / "comparisons"
        cluster_root.mkdir(parents=True, exist_ok=True)
        aggregate_root.mkdir(parents=True, exist_ok=True)
        comparison_root.mkdir(parents=True, exist_ok=True)

        post_processing_steps = len(cluster_summaries) + 4
        total_work_units = max(total_files + post_processing_steps, 1)
        gds_variable_registry: list[dict[str, object]] = []
        completed_units = 0
        if progress_callback is not None:
            progress_callback(
                0,
                total_work_units,
                "Preparing bond analysis.",
            )

        aggregate_bond_rows = {
            definition: [] for definition in self.bond_pairs
        }
        aggregate_angle_rows = {
            definition: [] for definition in self.angle_triplets
        }
        aggregate_dihedral_rows = {
            definition: [] for definition in self.dihedral_quartets
        }
        aggregate_coordination_rows = {
            definition: [] for definition in self.coordination_numbers
        }
        comparison_bonds = {definition: {} for definition in self.bond_pairs}
        comparison_angles = {
            definition: {} for definition in self.angle_triplets
        }
        comparison_dihedrals = {
            definition: {} for definition in self.dihedral_quartets
        }
        comparison_coordination = {
            definition: {} for definition in self.coordination_numbers
        }
        cluster_results: list[BondAnalysisClusterResult] = []
        cache_hit_count = 0
        cache_miss_count = 0
        active_distribution_text = self._active_distribution_progress_text()

        for summary in cluster_summaries:
            if log_callback is not None:
                log_callback(
                    "Analyzing cluster type "
                    f"{summary.name} ({summary.structure_count} files)."
                )
            if progress_callback is not None:
                progress_callback(
                    completed_units,
                    total_work_units,
                    (
                        f"Processing {summary.name}: "
                        f"{summary.structure_count} structures; "
                        f"{active_distribution_text}."
                    ),
                )
            cluster_output_dir = cluster_root / summary.name
            cluster_output_dir.mkdir(parents=True, exist_ok=True)

            cluster_bond_rows = {
                definition: [] for definition in self.bond_pairs
            }
            cluster_angle_rows = {
                definition: [] for definition in self.angle_triplets
            }
            cluster_dihedral_rows = {
                definition: [] for definition in self.dihedral_quartets
            }
            cluster_coordination_rows = {
                definition: [] for definition in self.coordination_numbers
            }
            cluster_cache_hit_count = 0
            cluster_cache_miss_count = 0

            for structure_index, structure_file in enumerate(
                summary.structure_files,
                start=1,
            ):
                _raise_if_cancel_requested(cancel_callback)
                measurement = distribution_store.measure_structure_file(
                    structure_file,
                    bond_pairs=self.bond_pairs,
                    angle_triplets=self.angle_triplets,
                    dihedral_quartets=self.dihedral_quartets,
                    coordination_numbers=self.coordination_numbers,
                    cluster_label=summary.name,
                    relative_label=structure_file.name,
                    autosave=False,
                )
                if measurement.from_cache:
                    cache_hit_count += 1
                    cluster_cache_hit_count += 1
                else:
                    cache_miss_count += 1
                    cluster_cache_miss_count += 1
                bond_values = measurement.bond_values
                angle_values = measurement.angle_values
                dihedral_values = measurement.dihedral_values
                coordination_values = measurement.coordination_values
                for definition, values in bond_values.items():
                    cluster_bond_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                    aggregate_bond_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                for definition, values in coordination_values.items():
                    cluster_coordination_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                    aggregate_coordination_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                for definition, values in angle_values.items():
                    cluster_angle_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                    aggregate_angle_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )

                for definition, values in dihedral_values.items():
                    cluster_dihedral_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                    aggregate_dihedral_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )

                completed_units += 1
                if (
                    progress_callback is not None
                    and _should_emit_structure_progress(
                        structure_index,
                        summary.structure_count,
                    )
                ):
                    progress_callback(
                        completed_units,
                        total_work_units,
                        (
                            f"Processing {summary.name}: "
                            f"{structure_index}/{summary.structure_count} "
                            "structures "
                            f"({cluster_cache_hit_count} cached, "
                            f"{cluster_cache_miss_count} measured)."
                        ),
                    )

            _raise_if_cancel_requested(cancel_callback)
            if progress_callback is not None:
                progress_callback(
                    completed_units,
                    total_work_units,
                    f"Writing {summary.name} distributions.",
                )
            _raise_if_cancel_requested(cancel_callback)
            cluster_bond_counts = self._write_bond_outputs(
                cluster_output_dir,
                cluster_bond_rows,
                title_prefix=summary.name,
                gds_variable_registry=gds_variable_registry,
            )
            cluster_angle_counts = self._write_angle_outputs(
                cluster_output_dir,
                cluster_angle_rows,
                title_prefix=summary.name,
                gds_variable_registry=gds_variable_registry,
            )
            cluster_dihedral_counts = self._write_dihedral_outputs(
                cluster_output_dir,
                cluster_dihedral_rows,
                title_prefix=summary.name,
                gds_variable_registry=gds_variable_registry,
            )
            cluster_coordination_counts = self._write_coordination_outputs(
                cluster_output_dir,
                cluster_coordination_rows,
                title_prefix=summary.name,
            )
            completed_units += 1
            if progress_callback is not None:
                progress_callback(
                    completed_units,
                    total_work_units,
                    f"Finished {summary.name} distributions.",
                )

            _raise_if_cancel_requested(cancel_callback)
            for definition, rows in cluster_bond_rows.items():
                comparison_bonds[definition][summary.name] = [
                    row[2] for row in rows
                ]
            for definition, rows in cluster_angle_rows.items():
                comparison_angles[definition][summary.name] = [
                    row[2] for row in rows
                ]
            for definition, rows in cluster_dihedral_rows.items():
                comparison_dihedrals[definition][summary.name] = [
                    row[2] for row in rows
                ]
            for definition, rows in cluster_coordination_rows.items():
                comparison_coordination[definition][summary.name] = [
                    row[2] for row in rows
                ]

            cluster_results.append(
                BondAnalysisClusterResult(
                    cluster_type=summary.name,
                    structure_count=summary.structure_count,
                    output_dir=cluster_output_dir,
                    bond_value_counts=cluster_bond_counts,
                    angle_value_counts=cluster_angle_counts,
                    dihedral_value_counts=cluster_dihedral_counts,
                    coordination_value_counts=cluster_coordination_counts,
                )
            )

        _raise_if_cancel_requested(cancel_callback)
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Saving cached structure measurements.",
            )
        _raise_if_cancel_requested(cancel_callback)
        distribution_store.flush()
        completed_units += 1
        _raise_if_cancel_requested(cancel_callback)
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Writing all-cluster distributions.",
            )
        _raise_if_cancel_requested(cancel_callback)

        self._write_bond_outputs(
            aggregate_root,
            aggregate_bond_rows,
            title_prefix="All selected clusters",
            gds_variable_registry=gds_variable_registry,
        )
        self._write_angle_outputs(
            aggregate_root,
            aggregate_angle_rows,
            title_prefix="All selected clusters",
            gds_variable_registry=gds_variable_registry,
        )
        self._write_dihedral_outputs(
            aggregate_root,
            aggregate_dihedral_rows,
            title_prefix="All selected clusters",
            gds_variable_registry=gds_variable_registry,
        )
        self._write_coordination_outputs(
            aggregate_root,
            aggregate_coordination_rows,
            title_prefix="All selected clusters",
        )
        completed_units += 1
        _raise_if_cancel_requested(cancel_callback)
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Writing cluster comparison overlays.",
            )
        _raise_if_cancel_requested(cancel_callback)
        self._write_comparison_bond_outputs(comparison_root, comparison_bonds)
        self._write_comparison_angle_outputs(
            comparison_root,
            comparison_angles,
        )
        self._write_comparison_dihedral_outputs(
            comparison_root,
            comparison_dihedrals,
        )
        self._write_comparison_coordination_outputs(
            comparison_root,
            comparison_coordination,
        )
        completed_units += 1
        _raise_if_cancel_requested(cancel_callback)
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Writing bond-analysis results index.",
            )
        _raise_if_cancel_requested(cancel_callback)

        results_index_path = output_dir / RESULTS_INDEX_FILENAME
        results_index_path.write_text(
            json.dumps(
                {
                    "clusters_dir": str(self.clusters_dir),
                    "output_dir": str(output_dir),
                    "analysis_signature_version": (ANALYSIS_SIGNATURE_VERSION),
                    "analysis_signature": analysis_signature,
                    "analysis_signature_payload": signature_payload,
                    "selected_cluster_types": [
                        summary.name for summary in cluster_summaries
                    ],
                    "total_structure_files": total_files,
                    "bond_pairs": [
                        definition.to_dict() for definition in self.bond_pairs
                    ],
                    "angle_triplets": [
                        definition.to_dict()
                        for definition in self.angle_triplets
                    ],
                    "dihedral_quartets": [
                        definition.to_dict()
                        for definition in self.dihedral_quartets
                    ],
                    "coordination_numbers": [
                        definition.to_dict()
                        for definition in self.coordination_numbers
                    ],
                    "cluster_results": [
                        result.to_dict() for result in cluster_results
                    ],
                    "gds_variable_registry": gds_variable_registry,
                    "aggregate_output_dir": str(aggregate_root),
                    "comparison_output_dir": str(comparison_root),
                    "structure_distribution_store_dir": str(
                        distribution_store.root_dir
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        completed_units += 1
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Bond analysis complete.",
            )

        if log_callback is not None:
            log_callback(
                "Wrote bond-analysis results index to "
                f"{results_index_path}."
            )
            log_callback(
                "Structure distribution cache: "
                f"{cache_hit_count} hit(s), {cache_miss_count} miss(es) at "
                f"{distribution_store.root_dir}."
            )

        return BondAnalysisBatchResult(
            clusters_dir=self.clusters_dir,
            output_dir=output_dir,
            selected_cluster_types=tuple(
                summary.name for summary in cluster_summaries
            ),
            total_structure_files=total_files,
            cluster_results=cluster_results,
            results_index_path=results_index_path,
            analysis_signature=analysis_signature,
            reused_existing_result=False,
        )

    def _validate_requested_distributions(self) -> None:
        if (
            not self.bond_pairs
            and not self.angle_triplets
            and not self.dihedral_quartets
            and not self.coordination_numbers
        ):
            raise ValueError(
                "Provide at least one bond pair, angle triplet, dihedral "
                "quartet, or coordination-number definition."
            )

    def _build_analysis_signature_payload(
        self,
        cluster_summaries: Sequence[ClusterTypeSummary],
    ) -> dict[str, object]:
        clusters_root = self.clusters_dir.expanduser().resolve()
        return {
            "schema": "saxshell.bondanalysis.analysis_signature",
            "version": ANALYSIS_SIGNATURE_VERSION,
            "clusters_dir": _resolved_path_text(self.clusters_dir),
            "selected_cluster_types": [
                summary.name for summary in cluster_summaries
            ],
            "cluster_distribution": [
                {
                    "cluster_type": summary.name,
                    "cluster_dir": _resolved_path_text(summary.path),
                    "structure_count": summary.structure_count,
                    "structure_files": [
                        _structure_file_signature_payload(
                            structure_file,
                            clusters_root=clusters_root,
                        )
                        for structure_file in summary.structure_files
                    ],
                }
                for summary in cluster_summaries
            ],
            "definitions": {
                "bond_pairs": _canonical_definition_payloads(self.bond_pairs),
                "angle_triplets": _canonical_definition_payloads(
                    self.angle_triplets
                ),
                "dihedral_quartets": _canonical_definition_payloads(
                    self.dihedral_quartets
                ),
                "coordination_numbers": _canonical_definition_payloads(
                    self.coordination_numbers
                ),
            },
        }

    def _find_matching_existing_result(
        self,
        *,
        output_dir: Path,
        analysis_signature: str,
        signature_payload: Mapping[str, object],
    ) -> BondAnalysisBatchResult | None:
        for index_path in _discover_result_index_paths(
            self._existing_result_search_roots(output_dir)
        ):
            payload = _load_result_index_payload(index_path)
            if payload is None:
                continue
            stored_signature = str(payload.get("analysis_signature", ""))
            signature_match = stored_signature == analysis_signature
            legacy_match = (
                not stored_signature
                and self._legacy_result_payload_matches(
                    payload,
                    signature_payload=signature_payload,
                )
            )
            if not signature_match and not legacy_match:
                continue
            payload = self._prepare_existing_result_for_reuse(
                index_path,
                payload,
                analysis_signature=analysis_signature,
                signature_payload=signature_payload,
            )
            return _batch_result_from_index_payload(
                index_path,
                payload,
                reused_existing_result=True,
            )
        return None

    def _legacy_result_payload_matches(
        self,
        payload: Mapping[str, object],
        *,
        signature_payload: Mapping[str, object],
    ) -> bool:
        payload_clusters_dir = _resolved_path_text(
            str(payload.get("clusters_dir", ""))
        )
        signature_clusters_dir = str(signature_payload.get("clusters_dir", ""))
        if payload_clusters_dir != signature_clusters_dir:
            return False

        selected_cluster_types = [
            str(value)
            for value in payload.get("selected_cluster_types", [])
            if str(value).strip()
        ]
        if not selected_cluster_types:
            selected_cluster_types = [
                cluster_type
                for cluster_type, _count in self._legacy_cluster_counts(
                    payload
                )
            ]
        signature_cluster_types = [
            str(value)
            for value in signature_payload.get("selected_cluster_types", [])
            if str(value).strip()
        ]
        if selected_cluster_types != signature_cluster_types:
            return False

        if self._legacy_cluster_counts(payload) != (
            self._signature_cluster_counts(signature_payload)
        ):
            return False

        signature_definitions = signature_payload.get("definitions", {})
        if not isinstance(signature_definitions, Mapping):
            return False
        for key in (
            "bond_pairs",
            "angle_triplets",
            "dihedral_quartets",
            "coordination_numbers",
        ):
            if _canonical_payload_values(payload.get(key, [])) != (
                _canonical_payload_values(signature_definitions.get(key, []))
            ):
                return False
        return True

    @staticmethod
    def _legacy_cluster_counts(
        payload: Mapping[str, object],
    ) -> tuple[tuple[str, int], ...]:
        rows = []
        for entry in payload.get("cluster_results", []):
            if not isinstance(entry, Mapping):
                continue
            cluster_type = str(entry.get("cluster_type", "")).strip()
            if not cluster_type:
                continue
            rows.append(
                (
                    cluster_type,
                    int(entry.get("structure_count", 0) or 0),
                )
            )
        return tuple(rows)

    @staticmethod
    def _signature_cluster_counts(
        signature_payload: Mapping[str, object],
    ) -> tuple[tuple[str, int], ...]:
        rows = []
        for entry in signature_payload.get("cluster_distribution", []):
            if not isinstance(entry, Mapping):
                continue
            cluster_type = str(entry.get("cluster_type", "")).strip()
            if not cluster_type:
                continue
            rows.append(
                (
                    cluster_type,
                    int(entry.get("structure_count", 0) or 0),
                )
            )
        return tuple(rows)

    def _prepare_existing_result_for_reuse(
        self,
        index_path: Path,
        payload: Mapping[str, object],
        *,
        analysis_signature: str,
        signature_payload: Mapping[str, object],
    ) -> dict[str, object]:
        updated_payload: dict[str, object] = dict(payload)
        updated_payload["analysis_signature_version"] = (
            ANALYSIS_SIGNATURE_VERSION
        )
        updated_payload["analysis_signature"] = analysis_signature
        updated_payload["analysis_signature_payload"] = dict(signature_payload)
        registry = self._backfill_gds_registry_from_existing_outputs(
            updated_payload
        )
        if registry is not None:
            updated_payload["gds_variable_registry"] = registry
        index_path.write_text(json.dumps(updated_payload, indent=2) + "\n")
        return updated_payload

    def _backfill_gds_registry_from_existing_outputs(
        self,
        payload: Mapping[str, object],
    ) -> list[dict[str, object]] | None:
        registry: list[dict[str, object]] = []
        output_dir = Path(str(payload.get("output_dir") or ""))
        if not str(output_dir):
            return None
        aggregate_root = Path(
            str(
                payload.get("aggregate_output_dir")
                or output_dir / "all_clusters"
            )
        )
        cluster_results = [
            entry
            for entry in payload.get("cluster_results", [])
            if isinstance(entry, Mapping)
        ]
        specs: tuple[
            tuple[str, str, str, str, tuple[object, ...]],
            ...,
        ] = (
            (
                "bond",
                "distribution",
                "Distance (A)",
                "_distribution",
                self.bond_pairs,
            ),
            (
                "angle",
                "angles",
                "Angle (deg)",
                "_angles",
                self.angle_triplets,
            ),
            (
                "dihedral",
                "dihedrals",
                "Dihedral (deg)",
                "_dihedrals",
                self.dihedral_quartets,
            ),
        )
        wrote_any = False
        for (
            distribution_type,
            _noun,
            value_label,
            data_suffix,
            definitions,
        ) in specs:
            for definition in definitions:
                cluster_values = []
                for cluster_result in cluster_results:
                    scope_label = str(cluster_result.get("cluster_type", ""))
                    cluster_output_dir = Path(
                        str(
                            cluster_result.get("output_dir")
                            or output_dir / "cluster_types" / scope_label
                        )
                    )
                    has_distribution_file = (
                        self._distribution_data_file_exists(
                            cluster_output_dir,
                            definition=definition,
                            data_suffix=data_suffix,
                        )
                    )
                    values = self._load_existing_distribution_values(
                        cluster_output_dir,
                        definition=definition,
                        data_suffix=data_suffix,
                    )
                    if values.size == 0 and (
                        distribution_type != "dihedral"
                        or not has_distribution_file
                    ):
                        continue
                    cluster_values.append(values)
                    metadata = self._write_histogram_csv(
                        cluster_output_dir
                        / f"{definition.filename_stem}_histogram.csv",
                        values.astype(float).tolist(),
                        distribution_type=distribution_type,
                        distribution_label=definition.display_label,
                        scope_label=scope_label,
                        value_label=value_label,
                    )
                    self._register_gds_variables(
                        registry,
                        histogram_csv_path=(
                            cluster_output_dir
                            / f"{definition.filename_stem}_histogram.csv"
                        ),
                        distribution_type=distribution_type,
                        distribution_label=definition.display_label,
                        scope_label=scope_label,
                        value_label=value_label,
                        metadata=metadata,
                    )
                    wrote_any = True
                has_aggregate_distribution_file = (
                    self._distribution_data_file_exists(
                        aggregate_root,
                        definition=definition,
                        data_suffix=data_suffix,
                    )
                )
                aggregate_values = self._load_existing_distribution_values(
                    aggregate_root,
                    definition=definition,
                    data_suffix=data_suffix,
                )
                if aggregate_values.size == 0 and cluster_values:
                    aggregate_values = np.concatenate(cluster_values)
                if aggregate_values.size == 0 and (
                    distribution_type != "dihedral"
                    or not has_aggregate_distribution_file
                ):
                    continue
                histogram_csv_path = (
                    aggregate_root
                    / f"{definition.filename_stem}_histogram.csv"
                )
                metadata = self._write_histogram_csv(
                    histogram_csv_path,
                    aggregate_values.astype(float).tolist(),
                    distribution_type=distribution_type,
                    distribution_label=definition.display_label,
                    scope_label="All selected clusters",
                    value_label=value_label,
                )
                self._register_gds_variables(
                    registry,
                    histogram_csv_path=histogram_csv_path,
                    distribution_type=distribution_type,
                    distribution_label=definition.display_label,
                    scope_label="All selected clusters",
                    value_label=value_label,
                    metadata=metadata,
                )
                wrote_any = True
        if wrote_any or registry:
            return registry
        existing_registry = payload.get("gds_variable_registry")
        if isinstance(existing_registry, list):
            return [dict(entry) for entry in existing_registry]
        return None

    @staticmethod
    def _distribution_data_file_exists(
        output_dir: Path,
        *,
        definition: object,
        data_suffix: str,
    ) -> bool:
        filename_stem = str(getattr(definition, "filename_stem"))
        return (output_dir / f"{filename_stem}{data_suffix}.npy").exists() or (
            output_dir / f"{filename_stem}{data_suffix}.csv"
        ).exists()

    @staticmethod
    def _load_existing_distribution_values(
        output_dir: Path,
        *,
        definition: object,
        data_suffix: str,
    ) -> np.ndarray:
        filename_stem = str(getattr(definition, "filename_stem"))
        npy_path = output_dir / f"{filename_stem}{data_suffix}.npy"
        if npy_path.exists():
            payload = np.load(npy_path, allow_pickle=False)
            if (
                getattr(payload.dtype, "names", None)
                and "value" in payload.dtype.names
            ):
                return np.asarray(payload["value"], dtype=float)
            return np.asarray(payload, dtype=float)

        csv_path = output_dir / f"{filename_stem}{data_suffix}.csv"
        if not csv_path.exists():
            return np.array([], dtype=float)
        values: list[float] = []
        with csv_path.open(newline="") as stream:
            reader = csv.reader(stream)
            header: list[str] | None = None
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                header = row
                break
            if header is None:
                return np.array([], dtype=float)
            value_index = len(header) - 1
            for row in reader:
                if len(row) <= value_index:
                    continue
                try:
                    values.append(float(row[value_index]))
                except ValueError:
                    continue
        return np.asarray(values, dtype=float)

    def _existing_result_search_roots(
        self,
        output_dir: Path,
    ) -> tuple[Path, ...]:
        roots: list[Path] = [output_dir]
        if output_dir.parent not in roots:
            roots.append(output_dir.parent)
        clusters_parent = self.clusters_dir.expanduser().parent
        if clusters_parent not in roots:
            roots.append(clusters_parent)
        return tuple(roots)

    def _selected_cluster_summaries(self) -> list[ClusterTypeSummary]:
        summaries = discover_cluster_types(self.clusters_dir)
        if self.selected_cluster_types is None:
            return summaries

        selected_names = set(self.selected_cluster_types)
        selected = [
            summary for summary in summaries if summary.name in selected_names
        ]
        missing = selected_names.difference(
            summary.name for summary in selected
        )
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "Unknown cluster type selection: " f"{missing_text}"
            )
        return selected

    def _write_bond_outputs(
        self,
        output_dir: Path,
        rows_by_definition: dict[
            BondPairDefinition,
            list[tuple[str, str, float]],
        ],
        *,
        title_prefix: str,
        gds_variable_registry: list[dict[str, object]] | None = None,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for definition, rows in rows_by_definition.items():
            csv_path = (
                output_dir / f"{definition.filename_stem}_distribution.csv"
            )
            npy_path = (
                output_dir / f"{definition.filename_stem}_distribution.npy"
            )
            self._write_distribution_csv(
                csv_path,
                rows,
                header=("Cluster Type", "Structure File", "Distance (A)"),
            )
            self._write_distribution_npy(npy_path, rows)
            values = [row[2] for row in rows]
            counts[definition.display_label] = len(values)
            if values:
                histogram_csv_path = (
                    output_dir / f"{definition.filename_stem}_histogram.csv"
                )
                histogram_metadata = self._write_histogram_csv(
                    histogram_csv_path,
                    values,
                    distribution_type="bond",
                    distribution_label=definition.display_label,
                    scope_label=title_prefix,
                    value_label="Distance (A)",
                )
                self._register_gds_variables(
                    gds_variable_registry,
                    histogram_csv_path=histogram_csv_path,
                    distribution_type="bond",
                    distribution_label=definition.display_label,
                    scope_label=title_prefix,
                    value_label="Distance (A)",
                    metadata=histogram_metadata,
                )
                if self.generate_preview_plots:
                    png_path = (
                        output_dir
                        / f"{definition.filename_stem}_histogram.png"
                    )
                    self._save_histogram(
                        values,
                        title=(
                            f"{title_prefix} • {definition.display_label} "
                            "bond distribution"
                        ),
                        xlabel="Distance (A)",
                        png_path=png_path,
                    )
        return counts

    def _write_angle_outputs(
        self,
        output_dir: Path,
        rows_by_definition: dict[
            AngleTripletDefinition,
            list[tuple[str, str, float]],
        ],
        *,
        title_prefix: str,
        gds_variable_registry: list[dict[str, object]] | None = None,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for definition, rows in rows_by_definition.items():
            csv_path = output_dir / f"{definition.filename_stem}_angles.csv"
            npy_path = output_dir / f"{definition.filename_stem}_angles.npy"
            self._write_distribution_csv(
                csv_path,
                rows,
                header=("Cluster Type", "Structure File", "Angle (deg)"),
            )
            self._write_distribution_npy(npy_path, rows)
            values = [row[2] for row in rows]
            counts[definition.display_label] = len(values)
            if values:
                histogram_csv_path = (
                    output_dir / f"{definition.filename_stem}_histogram.csv"
                )
                histogram_metadata = self._write_histogram_csv(
                    histogram_csv_path,
                    values,
                    distribution_type="angle",
                    distribution_label=definition.display_label,
                    scope_label=title_prefix,
                    value_label="Angle (deg)",
                )
                self._register_gds_variables(
                    gds_variable_registry,
                    histogram_csv_path=histogram_csv_path,
                    distribution_type="angle",
                    distribution_label=definition.display_label,
                    scope_label=title_prefix,
                    value_label="Angle (deg)",
                    metadata=histogram_metadata,
                )
                if self.generate_preview_plots:
                    png_path = (
                        output_dir
                        / f"{definition.filename_stem}_histogram.png"
                    )
                    self._save_histogram(
                        values,
                        title=(
                            f"{title_prefix} • {definition.display_label} "
                            "angle distribution"
                        ),
                        xlabel="Angle (deg)",
                        png_path=png_path,
                    )
        return counts

    def _write_dihedral_outputs(
        self,
        output_dir: Path,
        rows_by_definition: dict[
            DihedralQuartetDefinition,
            list[tuple[str, str, float]],
        ],
        *,
        title_prefix: str,
        gds_variable_registry: list[dict[str, object]] | None = None,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for definition, rows in rows_by_definition.items():
            csv_path = output_dir / f"{definition.filename_stem}_dihedrals.csv"
            npy_path = output_dir / f"{definition.filename_stem}_dihedrals.npy"
            self._write_distribution_csv(
                csv_path,
                rows,
                header=("Cluster Type", "Structure File", "Dihedral (deg)"),
            )
            self._write_distribution_npy(npy_path, rows)
            values = [row[2] for row in rows]
            counts[definition.display_label] = len(values)
            histogram_csv_path = (
                output_dir / f"{definition.filename_stem}_histogram.csv"
            )
            histogram_metadata = self._write_histogram_csv(
                histogram_csv_path,
                values,
                distribution_type="dihedral",
                distribution_label=definition.display_label,
                scope_label=title_prefix,
                value_label="Dihedral (deg)",
            )
            self._register_gds_variables(
                gds_variable_registry,
                histogram_csv_path=histogram_csv_path,
                distribution_type="dihedral",
                distribution_label=definition.display_label,
                scope_label=title_prefix,
                value_label="Dihedral (deg)",
                metadata=histogram_metadata,
            )
            if values:
                if self.generate_preview_plots:
                    png_path = (
                        output_dir
                        / f"{definition.filename_stem}_histogram.png"
                    )
                    self._save_histogram(
                        values,
                        title=(
                            f"{title_prefix} • {definition.display_label} "
                            "dihedral distribution"
                        ),
                        xlabel="Dihedral (deg)",
                        png_path=png_path,
                    )
        return counts

    def _write_coordination_outputs(
        self,
        output_dir: Path,
        rows_by_definition: dict[
            CoordinationNumberDefinition,
            list[tuple[str, str, float]],
        ],
        *,
        title_prefix: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for definition, rows in rows_by_definition.items():
            csv_path = (
                output_dir / f"{definition.filename_stem}_coordination.csv"
            )
            npy_path = (
                output_dir / f"{definition.filename_stem}_coordination.npy"
            )
            self._write_distribution_csv(
                csv_path,
                rows,
                header=(
                    "Cluster Type",
                    "Structure File",
                    "Coordination Number",
                ),
            )
            self._write_distribution_npy(npy_path, rows)
            values = [row[2] for row in rows]
            counts[definition.display_label] = len(values)
            if values:
                histogram_csv_path = (
                    output_dir / f"{definition.filename_stem}_histogram.csv"
                )
                self._write_histogram_csv(
                    histogram_csv_path,
                    values,
                    distribution_type="coordination",
                    distribution_label=definition.display_label,
                    scope_label=title_prefix,
                    value_label="Coordination Number",
                    metadata={
                        "center_atom": definition.center_atom,
                        "atom_of_interest": definition.neighbor_atom,
                        "cutoff_angstrom": definition.cutoff_angstrom,
                    },
                    bins=self._integer_histogram_edges(values),
                )
                if self.generate_preview_plots:
                    png_path = (
                        output_dir
                        / f"{definition.filename_stem}_histogram.png"
                    )
                    self._save_histogram(
                        values,
                        title=(
                            f"{title_prefix} • {definition.display_label} "
                            "coordination-number distribution"
                        ),
                        xlabel="Coordination Number",
                        png_path=png_path,
                    )
        return counts

    def _write_comparison_bond_outputs(
        self,
        output_dir: Path,
        values_by_definition: dict[
            BondPairDefinition,
            dict[str, list[float]],
        ],
    ) -> None:
        for definition, values_by_cluster in values_by_definition.items():
            if not any(values_by_cluster.values()):
                continue
            csv_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.csv"
            )
            npy_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.npy"
            )
            png_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.png"
            )
            self._write_overlay_csv(csv_path, values_by_cluster)
            self._write_overlay_npy(npy_path, values_by_cluster)
            if self.generate_preview_plots:
                self._save_overlay_histogram(
                    values_by_cluster,
                    title=(
                        "Cluster-type comparison • "
                        f"{definition.display_label} bond distribution"
                    ),
                    xlabel="Distance (A)",
                    png_path=png_path,
                )

    def _write_comparison_coordination_outputs(
        self,
        output_dir: Path,
        values_by_definition: dict[
            CoordinationNumberDefinition,
            dict[str, list[float]],
        ],
    ) -> None:
        for definition, values_by_cluster in values_by_definition.items():
            if not any(values_by_cluster.values()):
                continue
            csv_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.csv"
            )
            npy_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.npy"
            )
            png_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.png"
            )
            self._write_overlay_csv(csv_path, values_by_cluster)
            self._write_overlay_npy(npy_path, values_by_cluster)
            if self.generate_preview_plots:
                self._save_overlay_histogram(
                    values_by_cluster,
                    title=(
                        "Cluster-type comparison • "
                        f"{definition.display_label} "
                        "coordination-number distribution"
                    ),
                    xlabel="Coordination Number",
                    png_path=png_path,
                )

    def _write_comparison_angle_outputs(
        self,
        output_dir: Path,
        values_by_definition: dict[
            AngleTripletDefinition,
            dict[str, list[float]],
        ],
    ) -> None:
        for definition, values_by_cluster in values_by_definition.items():
            if not any(values_by_cluster.values()):
                continue
            csv_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.csv"
            )
            npy_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.npy"
            )
            png_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.png"
            )
            self._write_overlay_csv(csv_path, values_by_cluster)
            self._write_overlay_npy(npy_path, values_by_cluster)
            if self.generate_preview_plots:
                self._save_overlay_histogram(
                    values_by_cluster,
                    title=(
                        "Cluster-type comparison • "
                        f"{definition.display_label} angle distribution"
                    ),
                    xlabel="Angle (deg)",
                    png_path=png_path,
                )

    def _write_comparison_dihedral_outputs(
        self,
        output_dir: Path,
        values_by_definition: dict[
            DihedralQuartetDefinition,
            dict[str, list[float]],
        ],
    ) -> None:
        for definition, values_by_cluster in values_by_definition.items():
            if not any(values_by_cluster.values()):
                continue
            csv_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.csv"
            )
            npy_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.npy"
            )
            png_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.png"
            )
            self._write_overlay_csv(csv_path, values_by_cluster)
            self._write_overlay_npy(npy_path, values_by_cluster)
            if self.generate_preview_plots:
                self._save_overlay_histogram(
                    values_by_cluster,
                    title=(
                        "Cluster-type comparison • "
                        f"{definition.display_label} dihedral distribution"
                    ),
                    xlabel="Dihedral (deg)",
                    png_path=png_path,
                )

    @staticmethod
    def _write_distribution_csv(
        csv_path: Path,
        rows: list[tuple[str, str, float]],
        *,
        header: tuple[str, str, str],
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for cluster_type, structure_file, value in rows:
                writer.writerow([cluster_type, structure_file, f"{value:.6f}"])

    @staticmethod
    def _write_distribution_npy(
        npy_path: Path,
        rows: list[tuple[str, str, float]],
    ) -> None:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_width = max((len(row[0]) for row in rows), default=1)
        structure_width = max((len(row[1]) for row in rows), default=1)
        payload = np.empty(
            len(rows),
            dtype=[
                ("cluster_type", f"U{cluster_width}"),
                ("structure_file", f"U{structure_width}"),
                ("value", np.float64),
            ],
        )
        for index, (cluster_type, structure_file, value) in enumerate(rows):
            payload[index] = (cluster_type, structure_file, float(value))
        np.save(npy_path, payload)

    @classmethod
    def _write_histogram_csv(
        cls,
        csv_path: Path,
        values: list[float],
        *,
        distribution_type: str,
        distribution_label: str,
        scope_label: str,
        value_label: str,
        metadata: Mapping[str, object] | None = None,
        bins: int | Sequence[float] = 60,
    ) -> dict[str, object]:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        numeric_values = np.asarray(values, dtype=float)
        counts, edges = np.histogram(numeric_values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        densities = np.divide(
            counts,
            widths * max(float(numeric_values.size), 1.0),
            out=np.zeros_like(centers, dtype=float),
            where=widths > 0,
        )
        stats = cls._distribution_statistics(
            numeric_values,
            counts=counts,
            centers=centers,
        )
        extra_metadata = dict(metadata or {})
        extra_metadata.update(
            cls._gds_distribution_metadata(
                distribution_type=distribution_type,
                distribution_label=distribution_label,
                scope_label=scope_label,
                values=numeric_values,
                stats=stats,
            )
        )
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["# SAXSShell bondanalysis histogram distribution"]
            )
            for key, value in (
                ("distribution_type", distribution_type),
                ("distribution_label", distribution_label),
                ("scope", scope_label),
                ("value_label", value_label),
                *extra_metadata.items(),
                ("histogram_bins", int(counts.size)),
                ("point_count", int(numeric_values.size)),
                ("mean", stats["mean"]),
                ("median", stats["median"]),
                ("mode", stats["mode"]),
                ("sigma", stats["sigma"]),
                ("standard_deviation", stats["standard_deviation"]),
                ("sample_sigma", stats["sample_sigma"]),
                ("variance", stats["variance"]),
                ("minimum", stats["minimum"]),
                ("maximum", stats["maximum"]),
                ("q1", stats["q1"]),
                ("q3", stats["q3"]),
            ):
                writer.writerow([f"# {key}", value])
            writer.writerow(
                (
                    "bin_left",
                    "bin_right",
                    "bin_center",
                    "count",
                    "density",
                )
            )
            for left, right, center, count, density in zip(
                edges[:-1],
                edges[1:],
                centers,
                counts,
                densities,
            ):
                writer.writerow(
                    (
                        f"{float(left):.8g}",
                        f"{float(right):.8g}",
                        f"{float(center):.8g}",
                        int(count),
                        f"{float(density):.8g}",
                    )
                )
        return extra_metadata

    @staticmethod
    def _distribution_statistics(
        values: np.ndarray,
        *,
        counts: np.ndarray,
        centers: np.ndarray,
    ) -> dict[str, float]:
        if values.size == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "mode": 0.0,
                "sigma": 0.0,
                "standard_deviation": 0.0,
                "sample_sigma": 0.0,
                "variance": 0.0,
                "minimum": 0.0,
                "maximum": 0.0,
                "q1": 0.0,
                "q3": 0.0,
            }
        if np.all(np.isclose(values, np.round(values))):
            unique_values, unique_counts = np.unique(
                values, return_counts=True
            )
            peak_count = int(np.max(unique_counts))
            mode_value = float(unique_values[unique_counts == peak_count][0])
        else:
            peak_index = int(np.argmax(counts)) if counts.size else 0
            mode_value = (
                float(centers[peak_index])
                if centers.size
                else float(np.median(values))
            )
        sigma = float(np.std(values, ddof=0))
        sample_sigma = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "mode": mode_value,
            "sigma": sigma,
            "standard_deviation": sigma,
            "sample_sigma": sample_sigma,
            "variance": float(np.var(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
        }

    @staticmethod
    def _dihedral_gds_statistics(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {
                "circular_mean_degrees": 0.0,
                "circular_sigma_degrees": 0.0,
                "circular_resultant_length": 0.0,
                "gds_center_degrees": 0.0,
                "gds_sigma_degrees": 0.0,
                "gds_center_radians": 0.0,
                "gds_sigma_radians": 0.0,
                "gds_variance_radians_squared": 0.0,
            }
        radians = np.radians(np.asarray(values, dtype=float))
        sine_mean = float(np.mean(np.sin(radians)))
        cosine_mean = float(np.mean(np.cos(radians)))
        resultant_length = float(math.hypot(sine_mean, cosine_mean))
        mean_degrees = math.degrees(math.atan2(sine_mean, cosine_mean))
        if mean_degrees > 180.0:
            mean_degrees -= 360.0
        elif mean_degrees <= -180.0:
            mean_degrees += 360.0
        if resultant_length <= 1.0e-12:
            sigma_degrees = 180.0
        else:
            safe_length = min(max(resultant_length, 1.0e-12), 1.0)
            sigma_degrees = math.degrees(
                math.sqrt(max(-2.0 * math.log(safe_length), 0.0))
            )
        return {
            "circular_mean_degrees": float(mean_degrees),
            "circular_sigma_degrees": float(sigma_degrees),
            "circular_resultant_length": resultant_length,
            "gds_center_degrees": float(mean_degrees),
            "gds_sigma_degrees": float(sigma_degrees),
            "gds_center_radians": math.radians(mean_degrees),
            "gds_sigma_radians": math.radians(sigma_degrees),
            "gds_variance_radians_squared": math.radians(sigma_degrees) ** 2,
        }

    @classmethod
    def _gds_distribution_metadata(
        cls,
        *,
        distribution_type: str,
        distribution_label: str,
        scope_label: str,
        values: np.ndarray,
        stats: Mapping[str, float],
    ) -> dict[str, object]:
        if distribution_type == "bond":
            sigma = float(stats["sigma"])
            gds_metadata: dict[str, object] = {
                "gds_center_angstrom": float(stats["mean"]),
                "gds_sigma_angstrom": sigma,
                "gds_sigma2_angstrom_squared": sigma**2,
                "gds_variance_angstrom_squared": sigma**2,
            }
            return cls._with_gds_variable_rows(
                gds_metadata,
                distribution_type=distribution_type,
                distribution_label=distribution_label,
                scope_label=scope_label,
                center_value=float(gds_metadata["gds_center_angstrom"]),
                center_unit="angstrom",
                sigma_value=float(gds_metadata["gds_sigma_angstrom"]),
                sigma_unit="angstrom",
                variance_value=float(
                    gds_metadata["gds_sigma2_angstrom_squared"]
                ),
                variance_unit="angstrom_squared",
            )
        if distribution_type == "angle":
            center_degrees = float(stats["mean"])
            sigma_degrees = float(stats["sigma"])
            center_radians = math.radians(center_degrees)
            sigma_radians = math.radians(sigma_degrees)
            gds_metadata = {
                "gds_center_degrees": center_degrees,
                "gds_sigma_degrees": sigma_degrees,
                "gds_center_radians": center_radians,
                "gds_sigma_radians": sigma_radians,
                "gds_variance_radians_squared": sigma_radians**2,
            }
            return cls._with_gds_variable_rows(
                gds_metadata,
                distribution_type=distribution_type,
                distribution_label=distribution_label,
                scope_label=scope_label,
                center_value=center_radians,
                center_unit="radians",
                sigma_value=sigma_radians,
                sigma_unit="radians",
                variance_value=sigma_radians**2,
                variance_unit="radians_squared",
                extra_rows={
                    "gds_center_degrees_variable": (
                        "center_degrees",
                        center_degrees,
                        "degrees",
                    ),
                    "gds_sigma_degrees_variable": (
                        "sigma_degrees",
                        sigma_degrees,
                        "degrees",
                    ),
                },
            )
        if distribution_type == "dihedral":
            gds_metadata = cls._dihedral_gds_statistics(values)
            return cls._with_gds_variable_rows(
                gds_metadata,
                distribution_type=distribution_type,
                distribution_label=distribution_label,
                scope_label=scope_label,
                center_value=float(gds_metadata["gds_center_radians"]),
                center_unit="radians",
                sigma_value=float(gds_metadata["gds_sigma_radians"]),
                sigma_unit="radians",
                variance_value=float(
                    gds_metadata["gds_variance_radians_squared"]
                ),
                variance_unit="radians_squared",
                extra_rows={
                    "gds_center_degrees_variable": (
                        "center_degrees",
                        float(gds_metadata["gds_center_degrees"]),
                        "degrees",
                    ),
                    "gds_sigma_degrees_variable": (
                        "sigma_degrees",
                        float(gds_metadata["gds_sigma_degrees"]),
                        "degrees",
                    ),
                },
            )
        return {}

    @classmethod
    def _with_gds_variable_rows(
        cls,
        metadata: Mapping[str, object],
        *,
        distribution_type: str,
        distribution_label: str,
        scope_label: str,
        center_value: float,
        center_unit: str,
        sigma_value: float,
        sigma_unit: str,
        variance_value: float,
        variance_unit: str,
        extra_rows: Mapping[str, tuple[str, float, str]] | None = None,
    ) -> dict[str, object]:
        prefix = cls._gds_variable_prefix(
            distribution_type,
            distribution_label,
            scope_label,
        )
        rows: dict[str, object] = dict(metadata)
        variables = {
            "gds_center_variable": ("center", center_value, center_unit),
            "gds_sigma_variable": ("sigma", sigma_value, sigma_unit),
            "gds_variance_variable": (
                "sigma2",
                variance_value,
                variance_unit,
            ),
        }
        if extra_rows:
            variables.update(extra_rows)
        rows["gds_variable_prefix"] = prefix
        set_lines: list[str] = []
        for metadata_key, (suffix, value, unit) in variables.items():
            variable_name = f"{prefix}_{suffix}"
            rows[metadata_key] = variable_name
            rows[f"{metadata_key}_unit"] = unit
            set_line_key = metadata_key.replace("_variable", "_set")
            set_line = f"set {variable_name} = {cls._format_gds_float(value)}"
            rows[set_line_key] = set_line
            set_lines.append(set_line)
        rows["gds_set_rows"] = " ; ".join(set_lines)
        return rows

    @staticmethod
    def _gds_variable_prefix(
        distribution_type: str,
        distribution_label: str,
        scope_label: str,
    ) -> str:
        scope_token = _gds_token(scope_label)
        if scope_token in {"all_selected_clusters", "all_clusters"}:
            scope_token = "all"
        return "_".join(
            token
            for token in (
                "ba",
                _gds_token(distribution_type),
                scope_token,
                _gds_token(distribution_label),
            )
            if token
        )

    @staticmethod
    def _format_gds_float(value: float) -> str:
        return f"{float(value):.8g}"

    @staticmethod
    def _register_gds_variables(
        registry: list[dict[str, object]] | None,
        *,
        histogram_csv_path: Path,
        distribution_type: str,
        distribution_label: str,
        scope_label: str,
        value_label: str,
        metadata: Mapping[str, object],
    ) -> None:
        if registry is None or "gds_variable_prefix" not in metadata:
            return
        variable_keys = [
            "gds_center_variable",
            "gds_sigma_variable",
            "gds_variance_variable",
            "gds_center_degrees_variable",
            "gds_sigma_degrees_variable",
        ]
        variables: list[dict[str, object]] = []
        for key in variable_keys:
            variable_name = metadata.get(key)
            if not variable_name:
                continue
            set_key = key.replace("_variable", "_set")
            unit_key = f"{key}_unit"
            variables.append(
                {
                    "role": key.removeprefix("gds_").removesuffix("_variable"),
                    "name": variable_name,
                    "unit": metadata.get(unit_key, ""),
                    "set_row": metadata.get(set_key, ""),
                }
            )
        registry.append(
            {
                "distribution_type": distribution_type,
                "distribution_label": distribution_label,
                "scope": scope_label,
                "value_label": value_label,
                "histogram_csv": str(histogram_csv_path),
                "gds_variable_prefix": metadata["gds_variable_prefix"],
                "variables": variables,
                "set_rows": metadata.get("gds_set_rows", ""),
            }
        )

    @staticmethod
    def _integer_histogram_edges(values: list[float]) -> np.ndarray:
        numeric_values = np.asarray(values, dtype=float)
        if numeric_values.size == 0:
            return np.asarray([0.0, 1.0], dtype=float)
        left = math.floor(float(np.min(numeric_values))) - 0.5
        right = math.ceil(float(np.max(numeric_values))) + 0.5
        edges = np.arange(left, right + 1.0, 1.0, dtype=float)
        if edges.size < 2:
            return np.asarray([left, right], dtype=float)
        return edges

    @staticmethod
    def _write_overlay_csv(
        csv_path: Path,
        values_by_cluster: dict[str, list[float]],
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Cluster Type", "Value"))
            for cluster_type, values in sorted(values_by_cluster.items()):
                for value in values:
                    writer.writerow([cluster_type, f"{value:.6f}"])

    @staticmethod
    def _write_overlay_npy(
        npy_path: Path,
        values_by_cluster: dict[str, list[float]],
    ) -> None:
        rows = [
            (cluster_type, float(value))
            for cluster_type, values in sorted(values_by_cluster.items())
            for value in values
        ]
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_width = max((len(row[0]) for row in rows), default=1)
        payload = np.empty(
            len(rows),
            dtype=[
                ("cluster_type", f"U{cluster_width}"),
                ("value", np.float64),
            ],
        )
        for index, (cluster_type, value) in enumerate(rows):
            payload[index] = (cluster_type, value)
        np.save(npy_path, payload)

    @staticmethod
    def _save_histogram(
        values: list[float],
        *,
        title: str,
        xlabel: str,
        png_path: Path,
    ) -> None:
        figure = Figure(figsize=(8, 5))
        axis = figure.subplots()
        axis.hist(values, bins=60, color="#355070", edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        figure.tight_layout()
        figure.savefig(png_path, dpi=200)
        figure.clear()

    @staticmethod
    def _save_overlay_histogram(
        values_by_cluster: dict[str, list[float]],
        *,
        title: str,
        xlabel: str,
        png_path: Path,
    ) -> None:
        figure = Figure(figsize=(8, 5))
        axis = figure.subplots()
        for cluster_type, values in sorted(values_by_cluster.items()):
            if not values:
                continue
            axis.hist(
                values,
                bins=60,
                histtype="step",
                linewidth=1.5,
                label=cluster_type,
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(png_path, dpi=200)
        figure.clear()


__all__ = [
    "BondAnalysisBatchResult",
    "BondAnalysisClusterResult",
    "BondAnalysisWorkflow",
    "ClusterTypeSummary",
    "discover_cluster_types",
    "next_available_output_dir",
    "suggest_bondanalysis_output_dir",
]
