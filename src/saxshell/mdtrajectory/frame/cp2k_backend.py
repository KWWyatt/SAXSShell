from __future__ import annotations

import re
from pathlib import Path

from .base import FrameMetadata, FrameRecord, TrajectoryBackend

TIME_PATTERNS = (
    re.compile(
        r"time(?:\s*\[\s*fs\s*\])?\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"time_fs\s*=\s*" r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
)
FRAME_INDEX_PATTERN = re.compile(
    r"(?:^|[\s,;])i\s*=\s*(\d+)(?:\b|[\s,;])",
    re.IGNORECASE,
)


class CP2KTrajectoryBackend(TrajectoryBackend):
    """Backend for CP2K-style XYZ and PDB trajectory files.

    Supports:
    - standard XYZ frames
    - metadata-style XYZ frames
    - prefix-style XYZ frames using 'frame' or 'NSTEP='
    - PDB trajectories split by MODEL/ENDMDL
    """

    def __init__(
        self,
        input_file: str | Path,
        topology_file: str | Path | None = None,
        *,
        include_restart_duplicates: bool = False,
    ) -> None:
        super().__init__(input_file=input_file, topology_file=topology_file)
        suffix = self.input_file.suffix.lower()
        if suffix not in {".xyz", ".pdb"}:
            raise ValueError(
                "CP2KTrajectoryBackend supports only .xyz and .pdb files."
            )
        self.file_type = suffix.lstrip(".")
        self.include_restart_duplicates = bool(include_restart_duplicates)
        self._raw_frame_count: int | None = None
        self._duplicate_source_frame_count: int = 0

    def inspect(self) -> dict[str, object]:
        frame_metadata = self.load_frame_metadata()
        summary: dict[str, object] = {
            "input_file": str(self.input_file),
            "file_type": self.file_type,
            "n_frames": len(frame_metadata),
            "include_restart_duplicates": self.include_restart_duplicates,
        }
        if self._raw_frame_count is not None and (
            self._raw_frame_count != len(frame_metadata)
            or self._duplicate_source_frame_count
        ):
            summary["raw_frames"] = self._raw_frame_count
            summary["duplicate_source_frames"] = (
                self._duplicate_source_frame_count
            )
        return summary

    def iter_frame_metadata(self) -> list[FrameMetadata]:
        if self.file_type == "xyz":
            return self._parse_xyz_frame_metadata()
        if self.file_type == "pdb":
            return self._parse_pdb_frame_metadata()
        raise ValueError("Unsupported CP2K trajectory file type.")

    def iter_frames(self) -> list[FrameRecord]:
        if self.file_type == "xyz":
            return self._parse_xyz_frames()
        if self.file_type == "pdb":
            return self._parse_pdb_frames()
        raise ValueError("Unsupported CP2K trajectory file type.")

    def _estimate_frame_count(self) -> int:
        if self.file_type == "xyz":
            return self._estimate_frame_count_xyz()
        if self.file_type == "pdb":
            return self._estimate_frame_count_pdb()
        return 0

    def _estimate_frame_count_xyz(self) -> int:
        count = 0
        with self.input_file.open("r") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break

                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith("frame") or stripped.startswith(
                    "NSTEP="
                ):
                    count += 1
                    atom_count_line = handle.readline()
                    if not atom_count_line:
                        break
                    atom_count = atom_count_line.strip()
                    if atom_count.isdigit():
                        for _ in range(int(atom_count)):
                            if not handle.readline():
                                break
                    continue

                if stripped.isdigit():
                    count += 1
                    atom_count = int(stripped)
                    if not handle.readline():
                        break
                    for _ in range(atom_count):
                        if not handle.readline():
                            break
        return count

    def _parse_xyz_frame_metadata(self) -> list[FrameMetadata]:
        frames_by_index: dict[int, FrameMetadata] = {}
        frames_with_duplicates: list[FrameMetadata] = []
        seen_source_indices: set[int] = set()
        raw_frame_count = 0
        duplicate_source_frame_count = 0

        with self.input_file.open("r") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break

                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith("frame") or stripped.startswith(
                    "NSTEP="
                ):
                    atom_count_line = handle.readline()
                    if not atom_count_line:
                        break
                    atom_count_text = atom_count_line.strip()
                    if not atom_count_text.isdigit():
                        continue
                    source_index = self._parse_frame_index_from_metadata(line)
                    fallback_index = (
                        raw_frame_count
                        if self.include_restart_duplicates
                        else len(frames_by_index)
                    )
                    frame_index = self._resolve_frame_index(
                        line,
                        fallback_index=fallback_index,
                    )
                    raw_frame_count += 1
                    if (
                        source_index is not None
                        and source_index in seen_source_indices
                    ):
                        duplicate_source_frame_count += 1
                    if source_index is not None:
                        seen_source_indices.add(source_index)
                    frame = FrameMetadata(
                        frame_index=frame_index,
                        time_fs=self._parse_time_from_header(line),
                    )
                    if self.include_restart_duplicates:
                        frames_with_duplicates.append(frame)
                    else:
                        frames_by_index[frame_index] = frame
                    for _ in range(int(atom_count_text)):
                        if not handle.readline():
                            break
                    continue

                if not stripped.isdigit():
                    continue

                atom_count = int(stripped)
                comment = handle.readline()
                if not comment:
                    break
                time_val = (
                    None
                    if not comment
                    else self._parse_time_from_metadata(comment)
                )
                source_index = self._parse_frame_index_from_metadata(comment)
                fallback_index = (
                    raw_frame_count
                    if self.include_restart_duplicates
                    else len(frames_by_index)
                )
                frame_index = self._resolve_frame_index(
                    comment,
                    fallback_index=fallback_index,
                )
                raw_frame_count += 1
                if (
                    source_index is not None
                    and source_index in seen_source_indices
                ):
                    duplicate_source_frame_count += 1
                if source_index is not None:
                    seen_source_indices.add(source_index)
                frame = FrameMetadata(
                    frame_index=frame_index,
                    time_fs=time_val,
                )
                if self.include_restart_duplicates:
                    frames_with_duplicates.append(frame)
                else:
                    frames_by_index[frame_index] = frame
                for _ in range(atom_count):
                    if not handle.readline():
                        break

        self._raw_frame_count = raw_frame_count
        self._duplicate_source_frame_count = duplicate_source_frame_count
        if self.include_restart_duplicates:
            return frames_with_duplicates
        return [
            frames_by_index[frame_index]
            for frame_index in sorted(frames_by_index)
        ]

    def _parse_xyz_frames(self) -> list[FrameRecord]:
        frames_by_index: dict[int, FrameRecord] = {}
        frames_with_duplicates: list[FrameRecord] = []
        seen_source_indices: set[int] = set()
        raw_frame_count = 0
        duplicate_source_frame_count = 0

        with self.input_file.open("r") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break

                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith("frame") or stripped.startswith(
                    "NSTEP="
                ):
                    atom_count_line = handle.readline()
                    if not atom_count_line:
                        break
                    atom_count_text = atom_count_line.strip()
                    if not atom_count_text.isdigit():
                        continue
                    atom_count = int(atom_count_text)
                    atom_lines = self._read_xyz_atom_lines(
                        handle,
                        atom_count,
                    )
                    if atom_lines is None:
                        break
                    source_index = self._parse_frame_index_from_metadata(line)
                    fallback_index = (
                        raw_frame_count
                        if self.include_restart_duplicates
                        else len(frames_by_index)
                    )
                    frame_index = self._resolve_frame_index(
                        line,
                        fallback_index=fallback_index,
                    )
                    raw_frame_count += 1
                    if (
                        source_index is not None
                        and source_index in seen_source_indices
                    ):
                        duplicate_source_frame_count += 1
                    if source_index is not None:
                        seen_source_indices.add(source_index)
                    frame = FrameRecord(
                        frame_index=frame_index,
                        file_type="xyz",
                        atom_count=atom_count,
                        lines=[line, *atom_lines],
                        time_fs=self._parse_time_from_header(line),
                    )
                    if self.include_restart_duplicates:
                        frames_with_duplicates.append(frame)
                    else:
                        frames_by_index[frame_index] = frame
                    continue

                if not stripped.isdigit():
                    continue

                atom_count = int(stripped)
                comment = handle.readline()
                if not comment:
                    break
                atom_lines = self._read_xyz_atom_lines(handle, atom_count)
                if atom_lines is None:
                    break
                source_index = self._parse_frame_index_from_metadata(comment)
                fallback_index = (
                    raw_frame_count
                    if self.include_restart_duplicates
                    else len(frames_by_index)
                )
                frame_index = self._resolve_frame_index(
                    comment,
                    fallback_index=fallback_index,
                )
                raw_frame_count += 1
                if (
                    source_index is not None
                    and source_index in seen_source_indices
                ):
                    duplicate_source_frame_count += 1
                if source_index is not None:
                    seen_source_indices.add(source_index)
                frame = FrameRecord(
                    frame_index=frame_index,
                    file_type="xyz",
                    atom_count=atom_count,
                    lines=[comment, *atom_lines],
                    time_fs=self._parse_time_from_metadata(comment),
                )
                if self.include_restart_duplicates:
                    frames_with_duplicates.append(frame)
                else:
                    frames_by_index[frame_index] = frame

        self._raw_frame_count = raw_frame_count
        self._duplicate_source_frame_count = duplicate_source_frame_count
        if self.include_restart_duplicates:
            return frames_with_duplicates
        return [
            frames_by_index[frame_index]
            for frame_index in sorted(frames_by_index)
        ]

    def _read_xyz_atom_lines(
        self,
        handle,
        atom_count: int,
    ) -> list[str] | None:
        atom_lines: list[str] = []
        for _ in range(atom_count):
            atom_line = handle.readline()
            if not atom_line:
                return None
            atom_lines.append(atom_line)
        return atom_lines

    def _resolve_frame_index(
        self,
        header: str,
        *,
        fallback_index: int,
    ) -> int:
        source_index = self._parse_frame_index_from_metadata(header)
        if source_index is None:
            return fallback_index
        return source_index

    def _parse_frame_index_from_metadata(self, line: str) -> int | None:
        match = FRAME_INDEX_PATTERN.search(line.strip())
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _parse_time_from_metadata(self, line: str) -> float | None:
        return self._parse_time_from_header(line)

    def _parse_time_from_header(self, line: str) -> float | None:
        text = line.strip()
        if "time" not in text.lower():
            return None
        try:
            for pattern in TIME_PATTERNS:
                match = pattern.search(text)
                if match is not None:
                    return float(match.group(1))
        except ValueError:
            pass
        return None

    def _estimate_frame_count_pdb(self) -> int:
        count = 0
        with self.input_file.open("r") as handle:
            for line in handle:
                if line.startswith("MODEL"):
                    count += 1
        return count

    def _parse_pdb_frame_metadata(self) -> list[FrameMetadata]:
        frames: list[FrameMetadata] = []
        frame_idx = 0
        with self.input_file.open("r") as handle:
            for line in handle:
                if not line.startswith("MODEL"):
                    continue
                frames.append(
                    FrameMetadata(frame_index=frame_idx, time_fs=None)
                )
                frame_idx += 1
        return frames

    def _parse_pdb_frames(self) -> list[FrameRecord]:
        frames: list[FrameRecord] = []
        frame_idx = 0
        buffer: list[str] = []

        with self.input_file.open("r") as handle:
            for line in handle:
                if line.startswith("MODEL"):
                    buffer = [line]
                    continue

                if line.startswith("ENDMDL") or line.startswith("END"):
                    if buffer:
                        buffer.append(line)
                        frames.append(
                            FrameRecord(
                                frame_index=frame_idx,
                                file_type="pdb",
                                atom_count=None,
                                lines=buffer.copy(),
                                time_fs=None,
                            )
                        )
                        frame_idx += 1
                        buffer = []
                    continue

                if buffer:
                    buffer.append(line)

        return frames
