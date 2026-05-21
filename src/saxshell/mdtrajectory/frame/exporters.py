from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable

from .base import FrameRecord

ExportProgressCallback = Callable[[int, int, str], None]
XYZ_FRAME_INDEX_PATTERN = re.compile(
    r"(?:^|[\s,;])i\s*=\s*(\d+)(?:\b|[\s,;])",
    re.IGNORECASE,
)


def export_xyz_frames(
    frames: list[FrameRecord],
    output_dir: str | Path,
    *,
    allow_duplicate_frame_indices: bool = False,
    progress_callback: ExportProgressCallback | None = None,
) -> list[Path]:
    """Write frame records as XYZ files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    xyz_frames = [frame for frame in frames if frame.file_type == "xyz"]
    total_frames = len(xyz_frames)
    output_paths: set[Path] = set()
    frame_index_counts = Counter(frame.frame_index for frame in xyz_frames)
    seen_frame_indices: Counter[int] = Counter()

    for index, frame in enumerate(xyz_frames, start=1):
        _validate_xyz_frame_identity(frame)
        seen_frame_indices[frame.frame_index] += 1
        file_name = _xyz_output_filename(
            frame.frame_index,
            occurrence=seen_frame_indices[frame.frame_index],
            total=frame_index_counts[frame.frame_index],
            allow_duplicate_frame_indices=allow_duplicate_frame_indices,
        )
        file_path = output_path / file_name
        if file_path in output_paths:
            raise ValueError(
                "Multiple XYZ frames resolve to the same output file: "
                f"{file_path.name}"
            )
        output_paths.add(file_path)
        with file_path.open("w") as handle:
            handle.write(f"{frame.atom_count}\n")
            handle.write(frame.lines[0])
            for line in frame.lines[1:]:
                parts = line.split()
                if len(parts) != 4:
                    continue
                label = "".join(c for c in parts[0] if not c.isdigit())
                label = label.capitalize()
                x, y, z = parts[1], parts[2], parts[3]
                handle.write(f"{label:>4} {x:>10} {y:>10} {z:>10}\n")
        written_files.append(file_path)
        if progress_callback is not None:
            progress_callback(
                index,
                total_frames,
                f"Exporting frame {index} of {total_frames}: {file_path.name}",
            )

    return written_files


def _xyz_output_filename(
    frame_index: int,
    *,
    occurrence: int,
    total: int,
    allow_duplicate_frame_indices: bool,
) -> str:
    if total <= 1:
        return f"frame_{frame_index:04d}.xyz"
    if not allow_duplicate_frame_indices:
        return f"frame_{frame_index:04d}.xyz"
    if occurrence == total:
        return f"frame_{frame_index:04d}.xyz"
    return f"frame_{frame_index:04d}_duplicate{occurrence:04d}.xyz"


def export_pdb_frames(
    frames: list[FrameRecord],
    output_dir: str | Path,
    *,
    progress_callback: ExportProgressCallback | None = None,
) -> list[Path]:
    """Write frame records as PDB files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    pdb_frames = [frame for frame in frames if frame.file_type == "pdb"]
    total_frames = len(pdb_frames)
    output_paths: set[Path] = set()

    for index, frame in enumerate(pdb_frames, start=1):
        file_path = output_path / f"frame_{frame.frame_index:04d}.pdb"
        if file_path in output_paths:
            raise ValueError(
                "Multiple PDB frames resolve to the same output file: "
                f"{file_path.name}"
            )
        output_paths.add(file_path)
        with file_path.open("w") as handle:
            handle.writelines(frame.lines)
        written_files.append(file_path)
        if progress_callback is not None:
            progress_callback(
                index,
                total_frames,
                f"Exporting frame {index} of {total_frames}: {file_path.name}",
            )

    return written_files


def _validate_xyz_frame_identity(frame: FrameRecord) -> None:
    """Reject CP2K XYZ records whose header index and output index
    differ."""
    if not frame.lines:
        return
    source_index = _parse_xyz_source_frame_index(frame.lines[0])
    if source_index is None or source_index == frame.frame_index:
        return
    raise ValueError(
        "XYZ frame identity mismatch: header reports "
        f"i = {source_index}, but the export frame index is "
        f"{frame.frame_index}."
    )


def _parse_xyz_source_frame_index(header: str) -> int | None:
    match = XYZ_FRAME_INDEX_PATTERN.search(header.strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
