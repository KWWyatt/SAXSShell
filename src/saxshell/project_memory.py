from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from saxshell.cluster import (
    SEARCH_MODE_KDTREE,
    PairCutoffDefinitions,
    normalize_search_mode,
)
from saxshell.clusterdynamics.run_config import (
    coerce_atom_type_definitions,
    coerce_box_dimensions,
    coerce_pair_cutoff_definitions,
    serialize_atom_type_definitions,
    serialize_pair_cutoff_definitions,
)
from saxshell.structure import AtomTypeDefinitions

CLUSTER_EXTRACTION_SETTINGS_VERSION = 1
MDTRAJECTORY_TIME_AXIS_SETTINGS_VERSION = 1


def build_cluster_extraction_settings_payload(
    *,
    atom_type_definitions: AtomTypeDefinitions,
    pair_cutoff_definitions: PairCutoffDefinitions,
    box_dimensions: tuple[float, float, float] | None,
    use_pbc: bool,
    default_cutoff: float | None,
    shell_levels: Iterable[int],
    shared_shells: bool,
    include_shell_atoms_in_stoichiometry: bool,
    search_mode: str,
    include_shell_levels: Iterable[int] = (0,),
    smart_solvation_shells: bool = False,
    save_state_frequency: int | None = None,
    frames_dir: str | Path | None = None,
    clusters_dir: str | Path | None = None,
    box_dimensions_source_kind: str | None = None,
    box_dimensions_source: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": CLUSTER_EXTRACTION_SETTINGS_VERSION,
        "frames_dir": _optional_path_text(frames_dir),
        "clusters_dir": _optional_path_text(clusters_dir),
        "atom_type_definitions": serialize_atom_type_definitions(
            atom_type_definitions
        ),
        "pair_cutoff_definitions": serialize_pair_cutoff_definitions(
            pair_cutoff_definitions
        ),
        "box_dimensions": (
            None
            if box_dimensions is None
            else [float(component) for component in box_dimensions]
        ),
        "use_pbc": bool(use_pbc),
        "default_cutoff": _optional_positive_float(default_cutoff),
        "shell_levels": _int_list(shell_levels),
        "include_shell_levels": _int_list(include_shell_levels),
        "shared_shells": bool(shared_shells),
        "smart_solvation_shells": bool(smart_solvation_shells),
        "include_shell_atoms_in_stoichiometry": bool(
            include_shell_atoms_in_stoichiometry
        ),
        "search_mode": normalize_search_mode(search_mode),
    }
    if save_state_frequency is not None:
        payload["save_state_frequency"] = max(int(save_state_frequency), 1)
    if box_dimensions_source_kind:
        payload["box_dimensions_source_kind"] = str(
            box_dimensions_source_kind
        ).strip()
    if box_dimensions_source:
        payload["box_dimensions_source"] = str(box_dimensions_source).strip()
    return payload


def coerce_cluster_extraction_settings(
    payload: object,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}

    settings: dict[str, object] = {}
    atom_type_payload = payload.get("atom_type_definitions")
    if isinstance(atom_type_payload, dict):
        settings["atom_type_definitions"] = coerce_atom_type_definitions(
            atom_type_payload
        )

    pair_cutoff_payload = payload.get("pair_cutoff_definitions")
    if isinstance(pair_cutoff_payload, list):
        settings["pair_cutoff_definitions"] = coerce_pair_cutoff_definitions(
            pair_cutoff_payload
        )

    box_payload = payload.get("box_dimensions")
    if box_payload is not None:
        try:
            settings["box_dimensions"] = coerce_box_dimensions(box_payload)
        except (TypeError, ValueError):
            pass

    if "use_pbc" in payload:
        settings["use_pbc"] = _bool(payload.get("use_pbc"))
    if "default_cutoff" in payload:
        settings["default_cutoff"] = _optional_positive_float(
            payload.get("default_cutoff")
        )
    if "shell_levels" in payload:
        settings["shell_levels"] = tuple(
            _int_list(payload.get("shell_levels"))
        )
    elif "shell_growth_levels" in payload:
        settings["shell_levels"] = tuple(
            _int_list(payload.get("shell_growth_levels"))
        )
    if "include_shell_levels" in payload:
        settings["include_shell_levels"] = tuple(
            _int_list(payload.get("include_shell_levels"))
        )
    if "shared_shells" in payload:
        settings["shared_shells"] = _bool(payload.get("shared_shells"))
    if "smart_solvation_shells" in payload:
        settings["smart_solvation_shells"] = _bool(
            payload.get("smart_solvation_shells")
        )
    if "include_shell_atoms_in_stoichiometry" in payload:
        settings["include_shell_atoms_in_stoichiometry"] = _bool(
            payload.get("include_shell_atoms_in_stoichiometry")
        )
    if "search_mode" in payload:
        settings["search_mode"] = normalize_search_mode(
            str(payload.get("search_mode", SEARCH_MODE_KDTREE))
        )
    if "save_state_frequency" in payload:
        try:
            settings["save_state_frequency"] = max(
                int(payload.get("save_state_frequency")), 1
            )
        except (TypeError, ValueError):
            pass
    return settings


def build_mdtrajectory_time_axis_settings_payload(
    *,
    trajectory_file: str | Path | None,
    topology_file: str | Path | None,
    energy_file: str | Path | None,
    start: int | None,
    stop: int | None,
    stride: int,
    frame_timestep_fs: float,
    use_manual_frame_timestep: bool,
    use_cutoff_for_export: bool,
    selected_cutoff_fs: float | None,
    suggested_cutoff_fs: float | None,
    use_post_cutoff_stride: bool,
    post_cutoff_stride: int,
    include_restart_duplicates: bool,
    output_dir: str | Path | None = None,
    applied_cutoff_fs: float | None = None,
) -> dict[str, object]:
    return {
        "version": MDTRAJECTORY_TIME_AXIS_SETTINGS_VERSION,
        "trajectory_file": _optional_path_text(trajectory_file),
        "topology_file": _optional_path_text(topology_file),
        "energy_file": _optional_path_text(energy_file),
        "start": _optional_int(start),
        "stop": _optional_int(stop),
        "stride": max(int(stride), 1),
        "frame_timestep_fs": float(frame_timestep_fs),
        "use_manual_frame_timestep": bool(use_manual_frame_timestep),
        "use_cutoff_for_export": bool(use_cutoff_for_export),
        "selected_cutoff_fs": _optional_float(selected_cutoff_fs),
        "suggested_cutoff_fs": _optional_float(suggested_cutoff_fs),
        "use_post_cutoff_stride": bool(use_post_cutoff_stride),
        "post_cutoff_stride": max(int(post_cutoff_stride), 1),
        "include_restart_duplicates": bool(include_restart_duplicates),
        "output_dir": _optional_path_text(output_dir),
        "applied_cutoff_fs": _optional_float(applied_cutoff_fs),
    }


def coerce_mdtrajectory_time_axis_settings(
    payload: object,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    settings: dict[str, object] = {}
    for key in (
        "trajectory_file",
        "topology_file",
        "energy_file",
        "output_dir",
    ):
        text = _optional_text(payload.get(key))
        if text is not None:
            settings[key] = text
    for key in ("start", "stop"):
        try:
            settings[key] = _optional_int(payload.get(key))
        except (TypeError, ValueError):
            settings[key] = None
    if "stride" in payload:
        try:
            settings["stride"] = max(int(payload.get("stride")), 1)
        except (TypeError, ValueError):
            pass
    if "frame_timestep_fs" in payload:
        timestep = _optional_positive_float(payload.get("frame_timestep_fs"))
        if timestep is not None:
            settings["frame_timestep_fs"] = timestep
    for key in (
        "use_manual_frame_timestep",
        "use_cutoff_for_export",
        "use_post_cutoff_stride",
        "include_restart_duplicates",
    ):
        if key in payload:
            settings[key] = _bool(payload.get(key))
    for key in (
        "selected_cutoff_fs",
        "suggested_cutoff_fs",
        "applied_cutoff_fs",
    ):
        settings[key] = _optional_float(payload.get(key))
    if "post_cutoff_stride" in payload:
        try:
            settings["post_cutoff_stride"] = max(
                int(payload.get("post_cutoff_stride")), 1
            )
        except (TypeError, ValueError):
            pass
    return settings


def cluster_dynamics_time_axis_defaults_from_mdtrajectory(
    payload: object,
) -> dict[str, object]:
    settings = coerce_mdtrajectory_time_axis_settings(payload)
    defaults: dict[str, object] = {}
    if "frame_timestep_fs" in settings:
        defaults["frame_timestep_fs"] = settings["frame_timestep_fs"]
    cutoff = settings.get("applied_cutoff_fs")
    if cutoff is None and settings.get("use_cutoff_for_export", True):
        cutoff = settings.get("selected_cutoff_fs")
    if cutoff is not None:
        defaults["folder_start_time_fs"] = cutoff
    return defaults


def _optional_path_text(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _optional_positive_float(value: object) -> float | None:
    result = _optional_float(value)
    if result is None or result <= 0.0:
        return None
    return result


def _int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        candidates: Iterable[object] = [value]
    elif isinstance(value, Iterable):
        candidates = value
    else:
        candidates = [value]
    parsed: list[int] = []
    for candidate in candidates:
        try:
            parsed.append(int(candidate))
        except (TypeError, ValueError):
            continue
    return sorted(set(parsed))


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)
