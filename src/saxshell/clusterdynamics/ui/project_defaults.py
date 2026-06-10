from __future__ import annotations

from saxshell.project_memory import (
    cluster_dynamics_time_axis_defaults_from_mdtrajectory,
    coerce_cluster_extraction_settings,
)


def apply_cluster_extraction_project_defaults(
    definitions_panel,
    payload: object,
) -> bool:
    settings = coerce_cluster_extraction_settings(payload)
    if not settings:
        return False

    changed = False
    atom_type_definitions = settings.get("atom_type_definitions")
    if atom_type_definitions is not None:
        definitions_panel.load_atom_type_definitions(
            atom_type_definitions,
            emit_signal=False,
        )
        changed = True

    pair_cutoff_definitions = settings.get("pair_cutoff_definitions")
    if pair_cutoff_definitions is not None:
        definitions_panel.load_pair_cutoff_definitions(
            pair_cutoff_definitions,
            emit_signal=False,
        )
        changed = True

    if "box_dimensions" in settings:
        definitions_panel.set_box_dimensions(
            settings.get("box_dimensions"),
            emit_signal=False,
        )
        changed = True
    if "use_pbc" in settings:
        definitions_panel.set_use_pbc(
            bool(settings["use_pbc"]),
            emit_signal=False,
        )
        changed = True
    if "default_cutoff" in settings:
        definitions_panel.set_default_cutoff(
            settings.get("default_cutoff"),
            emit_signal=False,
        )
        changed = True
    if "shell_levels" in settings:
        definitions_panel.set_shell_growth_levels(
            tuple(int(level) for level in settings["shell_levels"]),
            emit_signal=False,
        )
        changed = True
    elif "include_shell_levels" in settings:
        definitions_panel.set_shell_growth_levels(
            tuple(
                int(level)
                for level in settings["include_shell_levels"]
                if int(level) > 0
            ),
            emit_signal=False,
        )
        changed = True
    if "shared_shells" in settings:
        definitions_panel.set_shared_shells(
            bool(settings["shared_shells"]),
            emit_signal=False,
        )
        changed = True
    if "smart_solvation_shells" in settings:
        definitions_panel.set_smart_solvation_shells(
            bool(settings["smart_solvation_shells"]),
            emit_signal=False,
        )
        changed = True
    if "include_shell_atoms_in_stoichiometry" in settings:
        definitions_panel.set_include_shell_atoms_in_stoichiometry(
            bool(settings["include_shell_atoms_in_stoichiometry"]),
            emit_signal=False,
        )
        changed = True
    if "search_mode" in settings:
        definitions_panel.set_search_mode(
            str(settings["search_mode"]),
            emit_signal=False,
        )
        changed = True
    if "save_state_frequency" in settings and hasattr(
        definitions_panel,
        "set_save_state_frequency",
    ):
        definitions_panel.set_save_state_frequency(
            int(settings["save_state_frequency"]),
            emit_signal=False,
        )
        changed = True
    return changed


def apply_mdtrajectory_time_axis_project_defaults(
    time_panel,
    payload: object,
) -> bool:
    defaults = cluster_dynamics_time_axis_defaults_from_mdtrajectory(payload)
    if not defaults:
        return False

    changed = False
    if defaults.get("folder_start_time_fs") is not None:
        time_panel.set_folder_start_time_fs(
            float(defaults["folder_start_time_fs"]),
            emit_signal=False,
        )
        changed = True
    if defaults.get("frame_timestep_fs") is not None:
        time_panel.set_frame_timestep_fs(
            float(defaults["frame_timestep_fs"]),
            emit_signal=False,
        )
        changed = True
    return changed
