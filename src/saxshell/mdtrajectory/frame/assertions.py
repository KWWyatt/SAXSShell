from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

FRAME_FILENAME_PATTERN = re.compile(r"^frame_(\d+)\.xyz$")
FRAME_INDEX_PATTERN = re.compile(
    r"(?:^|[\s,;])i\s*=\s*(\d+)(?:\b|[\s,;])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CoordinateLineSignature:
    label: str
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class XYZFrameSignature:
    frame_index: int | None
    atom_count: int
    coordinates: tuple[CoordinateLineSignature, ...]


@dataclass(frozen=True, slots=True)
class MDTrajectoryAssertionIssue:
    kind: str
    message: str
    path: str | None = None
    frame_index: int | None = None


@dataclass(slots=True)
class MDTrajectoryAssertionResult:
    trajectory_file: Path
    frame_dir: Path
    coordinate_lines: int
    coordinate_tolerance: float
    source_raw_frames: int
    source_unique_indices: int
    source_missing_indices: int
    source_duplicate_indices: int
    source_duplicate_conflicts: int
    exported_files: int
    validated_files: int
    filename_index_min: int | None
    filename_index_max: int | None
    header_index_min: int | None
    header_index_max: int | None
    filename_header_offsets: dict[int, int]
    issue_counts: dict[str, int]
    issues: list[MDTrajectoryAssertionIssue]
    strict_source_duplicates: bool

    @property
    def passed(self) -> bool:
        if any(count > 0 for count in self.issue_counts.values()):
            return False
        if self.strict_source_duplicates and self.source_duplicate_indices:
            return False
        return True

    @property
    def failure_count(self) -> int:
        failures = sum(self.issue_counts.values())
        if self.strict_source_duplicates:
            failures += self.source_duplicate_indices
        return failures

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failure_count": self.failure_count,
            "trajectory_file": str(self.trajectory_file),
            "frame_dir": str(self.frame_dir),
            "coordinate_lines": self.coordinate_lines,
            "coordinate_tolerance": self.coordinate_tolerance,
            "source_raw_frames": self.source_raw_frames,
            "source_unique_indices": self.source_unique_indices,
            "source_missing_indices": self.source_missing_indices,
            "source_duplicate_indices": self.source_duplicate_indices,
            "source_duplicate_conflicts": self.source_duplicate_conflicts,
            "exported_files": self.exported_files,
            "validated_files": self.validated_files,
            "filename_index_min": self.filename_index_min,
            "filename_index_max": self.filename_index_max,
            "header_index_min": self.header_index_min,
            "header_index_max": self.header_index_max,
            "filename_header_offsets": dict(self.filename_header_offsets),
            "issue_counts": dict(self.issue_counts),
            "issues": [
                {
                    "kind": issue.kind,
                    "message": issue.message,
                    "path": issue.path,
                    "frame_index": issue.frame_index,
                }
                for issue in self.issues
            ],
            "strict_source_duplicates": self.strict_source_duplicates,
        }


def validate_xyz_export_against_source(
    trajectory_file: str | Path,
    frame_dir: str | Path,
    *,
    coordinate_lines: int = 3,
    coordinate_tolerance: float = 1.0e-9,
    expect_contiguous: bool = False,
    strict_source_duplicates: bool = False,
    max_issues: int = 20,
) -> MDTrajectoryAssertionResult:
    """Validate exported XYZ frames against their source trajectory.

    The export filename index must match the CP2K ``i =`` header index,
    output header indices must be unique, and the first coordinate lines
    must match the source trajectory frame with the same index.
    """
    if coordinate_lines <= 0:
        raise ValueError("coordinate_lines must be a positive integer.")
    if coordinate_tolerance < 0:
        raise ValueError("coordinate_tolerance must be non-negative.")
    if max_issues < 0:
        raise ValueError("max_issues must be non-negative.")

    trajectory_path = Path(trajectory_file)
    export_path = Path(frame_dir)
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {trajectory_path}"
        )
    if not export_path.is_dir():
        raise NotADirectoryError(f"Frame directory not found: {export_path}")

    source_by_index: dict[int, XYZFrameSignature] = {}
    source_raw_frames = 0
    source_missing_indices = 0
    source_duplicate_indices = 0
    source_duplicate_conflicts = 0
    issue_counts: Counter[str] = Counter()
    issues: list[MDTrajectoryAssertionIssue] = []

    def add_issue(
        kind: str,
        message: str,
        *,
        path: Path | None = None,
        frame_index: int | None = None,
        count: int = 1,
    ) -> None:
        issue_counts[kind] += count
        if len(issues) >= max_issues:
            return
        issues.append(
            MDTrajectoryAssertionIssue(
                kind=kind,
                message=message,
                path=None if path is None else str(path),
                frame_index=frame_index,
            )
        )

    for source_frame in _iter_xyz_frame_signatures(
        trajectory_path,
        coordinate_lines=coordinate_lines,
    ):
        source_raw_frames += 1
        if source_frame.frame_index is None:
            source_missing_indices += 1
            continue
        existing = source_by_index.get(source_frame.frame_index)
        if existing is not None:
            source_duplicate_indices += 1
            if not _signatures_match(
                existing,
                source_frame,
                coordinate_tolerance=coordinate_tolerance,
            ):
                source_duplicate_conflicts += 1
            source_by_index[source_frame.frame_index] = source_frame
            continue
        source_by_index[source_frame.frame_index] = source_frame

    exported_files = 0
    validated_files = 0
    filename_indices: list[int] = []
    header_indices: list[int] = []
    filename_index_counts: Counter[int] = Counter()
    header_index_counts: Counter[int] = Counter()
    offset_counts: Counter[int] = Counter()

    frame_paths = sorted(
        export_path.glob("*.xyz"),
        key=lambda path: _frame_file_sort_key(path),
    )
    if not frame_paths:
        add_issue(
            "no_exported_xyz_files",
            f"No exported XYZ files were found in {export_path}.",
            path=export_path,
        )
    for frame_path in frame_paths:
        exported_files += 1
        name_match = FRAME_FILENAME_PATTERN.match(frame_path.name)
        if name_match is None:
            add_issue(
                "invalid_export_filename",
                f"Expected frame_<index>.xyz filename, got {frame_path.name}.",
                path=frame_path,
            )
            continue

        filename_index = int(name_match.group(1))
        filename_indices.append(filename_index)
        filename_index_counts[filename_index] += 1

        try:
            export_frame = _read_single_xyz_frame_signature(
                frame_path,
                coordinate_lines=coordinate_lines,
            )
        except ValueError as exc:
            add_issue(
                "invalid_export_xyz",
                str(exc),
                path=frame_path,
                frame_index=filename_index,
            )
            continue

        validated_files += 1
        if export_frame.frame_index is None:
            add_issue(
                "missing_export_header_index",
                f"{frame_path.name} does not include a CP2K i = index.",
                path=frame_path,
                frame_index=filename_index,
            )
        else:
            header_index = export_frame.frame_index
            header_indices.append(header_index)
            header_index_counts[header_index] += 1
            offset = filename_index - header_index
            offset_counts[offset] += 1
            if offset != 0:
                add_issue(
                    "filename_header_offset",
                    (
                        f"{frame_path.name} filename index is "
                        f"{filename_index}, but header reports i = "
                        f"{header_index}."
                    ),
                    path=frame_path,
                    frame_index=filename_index,
                )

        source_frame = source_by_index.get(filename_index)
        if source_frame is None:
            add_issue(
                "missing_source_index",
                (
                    f"{frame_path.name} references index {filename_index}, "
                    "which is not present in the source trajectory."
                ),
                path=frame_path,
                frame_index=filename_index,
            )
            continue

        if export_frame.atom_count != source_frame.atom_count:
            add_issue(
                "atom_count_mismatch",
                (
                    f"{frame_path.name} atom count is "
                    f"{export_frame.atom_count}, but source frame "
                    f"{filename_index} has {source_frame.atom_count}."
                ),
                path=frame_path,
                frame_index=filename_index,
            )

        if not _coordinates_match(
            source_frame.coordinates,
            export_frame.coordinates,
            coordinate_tolerance=coordinate_tolerance,
        ):
            add_issue(
                "coordinate_mismatch",
                (
                    f"{frame_path.name} first {coordinate_lines} coordinate "
                    f"line(s) do not match source frame {filename_index}."
                ),
                path=frame_path,
                frame_index=filename_index,
            )

    for duplicate_index, count in sorted(filename_index_counts.items()):
        if count <= 1:
            continue
        add_issue(
            "duplicate_export_filename_index",
            (
                f"Export contains {count} filenames resolving to frame "
                f"index {duplicate_index}."
            ),
            frame_index=duplicate_index,
            count=count - 1,
        )

    for duplicate_index, count in sorted(header_index_counts.items()):
        if count <= 1:
            continue
        add_issue(
            "duplicate_export_header_index",
            (
                f"Export contains {count} frames whose headers report "
                f"i = {duplicate_index}."
            ),
            frame_index=duplicate_index,
            count=count - 1,
        )

    if expect_contiguous and filename_indices:
        filename_index_set = set(filename_indices)
        start = min(filename_index_set)
        stop = max(filename_index_set)
        missing_indices = [
            index
            for index in range(start, stop + 1)
            if index not in filename_index_set
        ]
        if missing_indices:
            add_issue(
                "missing_contiguous_export_index",
                (
                    f"Export filename range {start}-{stop} is missing "
                    f"{len(missing_indices)} index value(s); first missing "
                    f"index is {missing_indices[0]}."
                ),
                frame_index=missing_indices[0],
                count=len(missing_indices),
            )

    return MDTrajectoryAssertionResult(
        trajectory_file=trajectory_path,
        frame_dir=export_path,
        coordinate_lines=coordinate_lines,
        coordinate_tolerance=coordinate_tolerance,
        source_raw_frames=source_raw_frames,
        source_unique_indices=len(source_by_index),
        source_missing_indices=source_missing_indices,
        source_duplicate_indices=source_duplicate_indices,
        source_duplicate_conflicts=source_duplicate_conflicts,
        exported_files=exported_files,
        validated_files=validated_files,
        filename_index_min=min(filename_indices) if filename_indices else None,
        filename_index_max=max(filename_indices) if filename_indices else None,
        header_index_min=min(header_indices) if header_indices else None,
        header_index_max=max(header_indices) if header_indices else None,
        filename_header_offsets=dict(sorted(offset_counts.items())),
        issue_counts=dict(sorted(issue_counts.items())),
        issues=issues,
        strict_source_duplicates=strict_source_duplicates,
    )


def _iter_xyz_frame_signatures(
    path: Path,
    *,
    coordinate_lines: int,
):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("frame") or stripped.startswith("NSTEP="):
                header = line
                atom_count_line = handle.readline()
                if not atom_count_line:
                    break
                atom_count_text = atom_count_line.strip()
                if not atom_count_text.isdigit():
                    continue
                atom_count = int(atom_count_text)
                atom_lines = _read_atom_lines(handle, atom_count)
                if atom_lines is None:
                    break
                yield _frame_signature(
                    header,
                    atom_count,
                    atom_lines,
                    coordinate_lines=coordinate_lines,
                )
                continue

            if not stripped.isdigit():
                continue

            atom_count = int(stripped)
            header = handle.readline()
            if not header:
                break
            atom_lines = _read_atom_lines(handle, atom_count)
            if atom_lines is None:
                break
            yield _frame_signature(
                header,
                atom_count,
                atom_lines,
                coordinate_lines=coordinate_lines,
            )


def _read_single_xyz_frame_signature(
    path: Path,
    *,
    coordinate_lines: int,
) -> XYZFrameSignature:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        atom_count_line = handle.readline()
        if not atom_count_line:
            raise ValueError(f"{path.name} is empty.")
        atom_count_text = atom_count_line.strip()
        if not atom_count_text.isdigit():
            raise ValueError(
                f"{path.name} does not start with an XYZ atom count."
            )
        atom_count = int(atom_count_text)
        header = handle.readline()
        if not header:
            raise ValueError(f"{path.name} is missing its XYZ comment line.")
        atom_lines = _read_atom_lines(handle, atom_count)
        if atom_lines is None:
            raise ValueError(
                f"{path.name} ended before {atom_count} atom line(s)."
            )
    return _frame_signature(
        header,
        atom_count,
        atom_lines,
        coordinate_lines=coordinate_lines,
    )


def _read_atom_lines(handle, atom_count: int) -> list[str] | None:
    atom_lines: list[str] = []
    for _ in range(atom_count):
        atom_line = handle.readline()
        if not atom_line:
            return None
        atom_lines.append(atom_line)
    return atom_lines


def _frame_signature(
    header: str,
    atom_count: int,
    atom_lines: list[str],
    *,
    coordinate_lines: int,
) -> XYZFrameSignature:
    return XYZFrameSignature(
        frame_index=_parse_frame_index(header),
        atom_count=atom_count,
        coordinates=tuple(
            coordinate
            for coordinate in (
                _coordinate_signature(line)
                for line in atom_lines[:coordinate_lines]
            )
            if coordinate is not None
        ),
    )


def _parse_frame_index(header: str) -> int | None:
    match = FRAME_INDEX_PATTERN.search(header.strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _coordinate_signature(line: str) -> CoordinateLineSignature | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        return CoordinateLineSignature(
            label=_normalize_atom_label(parts[0]),
            x=float(parts[1]),
            y=float(parts[2]),
            z=float(parts[3]),
        )
    except ValueError:
        return None


def _normalize_atom_label(label: str) -> str:
    return "".join(char for char in label if not char.isdigit()).capitalize()


def _signatures_match(
    left: XYZFrameSignature,
    right: XYZFrameSignature,
    *,
    coordinate_tolerance: float,
) -> bool:
    return left.atom_count == right.atom_count and _coordinates_match(
        left.coordinates,
        right.coordinates,
        coordinate_tolerance=coordinate_tolerance,
    )


def _coordinates_match(
    left: tuple[CoordinateLineSignature, ...],
    right: tuple[CoordinateLineSignature, ...],
    *,
    coordinate_tolerance: float,
) -> bool:
    if len(left) != len(right):
        return False
    for left_line, right_line in zip(left, right, strict=True):
        if left_line.label != right_line.label:
            return False
        if not (
            math.isclose(
                left_line.x,
                right_line.x,
                rel_tol=0.0,
                abs_tol=coordinate_tolerance,
            )
            and math.isclose(
                left_line.y,
                right_line.y,
                rel_tol=0.0,
                abs_tol=coordinate_tolerance,
            )
            and math.isclose(
                left_line.z,
                right_line.z,
                rel_tol=0.0,
                abs_tol=coordinate_tolerance,
            )
        ):
            return False
    return True


def _frame_file_sort_key(path: Path) -> tuple[int, int | str]:
    match = FRAME_FILENAME_PATTERN.match(path.name)
    if match is None:
        return (1, path.name)
    return (0, int(match.group(1)))
