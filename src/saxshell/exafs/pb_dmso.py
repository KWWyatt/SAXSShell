from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from saxshell.saxs.debye import load_structure_file

from .gds import (
    ArtemisGDSBuildSettings,
    ArtemisGDSDocument,
    ArtemisGDSParameter,
    ArtemisGDSShell,
    ArtemisPathParameterHint,
    _format_float,
)

_T = TypeVar("_T")


@dataclass(slots=True, frozen=True)
class PbDMSOGDSBuildSettings:
    """Build settings for compact Pb-I/DMSO Artemis GDS files."""

    absorber_element: str = "Pb"
    absorber_atom_index: int | None = None
    iodide_element: str = "I"
    oxygen_element: str = "O"
    sulfur_element: str = "S"
    oxygen_count: int = 3
    iodide_count: int | None = None
    min_distance_angstrom: float = 0.5
    max_iodide_distance_angstrom: float = 4.0
    max_oxygen_distance_angstrom: float = 4.0
    max_os_distance_angstrom: float = 2.2
    shell_tolerance_angstrom: float = 0.12
    initial_s02: float = 0.9
    vary_s02: bool = False
    initial_e0: float = 0.0
    initial_oxygen_cn: float = 1.0
    initial_iodide_cn: float = 1.0
    initial_delta_r: float = 0.0
    initial_oxygen_sigma2: float = 0.003
    initial_iodide_sigma2: float = 0.003
    link_oxygen_sigma2: bool = True
    bl_os_angstrom: float = 1.5
    theta_pbos_degrees: float | None = None
    angle_width_degrees: float = 8.0
    include_restraints: bool = True
    restraint_scale: float = 1000.0
    e0_lower_bound: float = -10.0
    e0_upper_bound: float = 10.0
    oxygen_cn_lower_bound: float = 0.0
    oxygen_cn_upper_bound: float = 4.0
    iodide_cn_lower_bound: float = 0.0
    iodide_cn_upper_bound: float = 4.0
    sigma2_lower_bound: float = 0.0001
    sigma2_upper_bound: float = 0.03
    theta_lower_bound_degrees: float = 90.0
    theta_upper_bound_degrees: float = 180.0
    width_lower_bound_degrees: float = 0.0
    width_upper_bound_degrees: float = 40.0
    included_path_pairs: tuple[tuple[int, int], ...] | None = None


@dataclass(slots=True, frozen=True)
class _DMSOPair:
    absorber_index: int
    oxygen_index: int
    sulfur_index: int
    pb_o_distance: float
    pb_s_distance: float
    os_distance: float
    angle_degrees: float


@dataclass(slots=True, frozen=True)
class _IodidePath:
    absorber_index: int
    iodide_index: int
    distance: float


@dataclass(slots=True, frozen=True)
class _IodideGroup:
    paths: tuple[_IodidePath, ...]
    absorber_count: int

    @property
    def multiplicity(self) -> float:
        return float(len(self.paths)) / float(self.absorber_count)

    @property
    def mean_distance(self) -> float:
        return _mean(path.distance for path in self.paths)

    @property
    def std_distance(self) -> float:
        return _std(path.distance for path in self.paths)

    @property
    def min_distance(self) -> float:
        return min(path.distance for path in self.paths)

    @property
    def max_distance(self) -> float:
        return max(path.distance for path in self.paths)


@dataclass(slots=True, frozen=True)
class _DMSOGroup:
    pairs: tuple[_DMSOPair, ...]
    absorber_count: int

    @property
    def multiplicity(self) -> float:
        return float(len(self.pairs)) / float(self.absorber_count)

    @property
    def pb_o_distance(self) -> float:
        return _mean(pair.pb_o_distance for pair in self.pairs)

    @property
    def pb_s_distance(self) -> float:
        return _mean(pair.pb_s_distance for pair in self.pairs)

    @property
    def pb_o_std_distance(self) -> float:
        return _std(pair.pb_o_distance for pair in self.pairs)

    @property
    def pb_s_std_distance(self) -> float:
        return _std(pair.pb_s_distance for pair in self.pairs)

    @property
    def pb_o_min_distance(self) -> float:
        return min(pair.pb_o_distance for pair in self.pairs)

    @property
    def pb_o_max_distance(self) -> float:
        return max(pair.pb_o_distance for pair in self.pairs)

    @property
    def pb_s_min_distance(self) -> float:
        return min(pair.pb_s_distance for pair in self.pairs)

    @property
    def pb_s_max_distance(self) -> float:
        return max(pair.pb_s_distance for pair in self.pairs)


@dataclass(slots=True, frozen=True)
class _DMSOPathSelection:
    group: _DMSOGroup
    include_oxygen_path: bool
    include_sulfur_path: bool

    @property
    def is_active(self) -> bool:
        return self.include_oxygen_path or self.include_sulfur_path


def build_pb_dmso_gds_from_structure(
    structure_path: str | Path,
    settings: PbDMSOGDSBuildSettings | None = None,
) -> ArtemisGDSDocument:
    """Build a compact Pb-I/DMSO GDS file from a structure file.

    The emitted variable names intentionally mirror the hand-fit GDS style:
    ``snot``, ``enot``, ``cn_o1``, ``delr_i1``, and Pb-O/Pb-S geometry
    constraints for selected terminal paths.
    """

    active_settings = settings or PbDMSOGDSBuildSettings()
    path = Path(structure_path).expanduser().resolve()
    positions, elements = load_structure_file(path)
    normalized_elements = tuple(
        _normalize_element(element) for element in elements
    )
    absorber_indices = _resolve_absorbers(
        normalized_elements,
        active_settings,
    )
    iodide_paths = _collect_iodide_paths(
        positions=positions,
        elements=normalized_elements,
        absorber_indices=absorber_indices,
        settings=active_settings,
    )
    dmso_pairs = _collect_dmso_pairs(
        positions=positions,
        elements=normalized_elements,
        absorber_indices=absorber_indices,
        settings=active_settings,
    )
    tolerance = max(float(active_settings.shell_tolerance_angstrom), 0.0)
    iodide_groups = _group_iodide_paths(
        iodide_paths,
        absorber_count=len(absorber_indices),
        tolerance=tolerance,
    )
    dmso_groups = _group_dmso_pairs(
        dmso_pairs,
        absorber_count=len(absorber_indices),
        tolerance=tolerance,
    )
    dmso_selections = _dmso_group_path_selections(
        dmso_groups,
        included_path_pairs=active_settings.included_path_pairs,
    )
    sulfur_path_indices = tuple(
        index
        for index, selection in enumerate(dmso_selections, start=1)
        if selection.include_sulfur_path
    )
    theta_degrees = (
        float(active_settings.theta_pbos_degrees)
        if active_settings.theta_pbos_degrees is not None
        else _mean(pair.angle_degrees for pair in dmso_pairs)
    )

    parameters: list[ArtemisGDSParameter] = []
    shells: list[ArtemisGDSShell] = []
    hints: list[ArtemisPathParameterHint] = []

    _append_globals(parameters, active_settings)
    _append_coordination_parameters(
        parameters,
        oxygen_multiplicities=tuple(
            selection.group.multiplicity for selection in dmso_selections
        ),
        iodide_multiplicities=tuple(
            group.multiplicity for group in iodide_groups
        ),
        sulfur_path_indices=sulfur_path_indices,
        settings=active_settings,
    )
    _append_delta_r_parameters(
        parameters,
        oxygen_count=len(dmso_selections),
        iodide_count=len(iodide_groups),
        settings=active_settings,
    )
    _append_sigma2_parameters(
        parameters,
        oxygen_count=len(dmso_selections),
        iodide_count=len(iodide_groups),
        settings=active_settings,
    )
    _append_geometry_parameters(
        parameters,
        dmso_selections=dmso_selections,
        sulfur_path_indices=sulfur_path_indices,
        theta_degrees=theta_degrees,
        settings=active_settings,
    )
    if active_settings.include_restraints:
        _append_restraints(
            parameters,
            oxygen_count=len(dmso_selections),
            iodide_count=len(iodide_groups),
            sulfur_path_indices=sulfur_path_indices,
            settings=active_settings,
        )

    absorber_element = _normalize_element(active_settings.absorber_element)
    _append_iodide_shells_and_hints(
        shells=shells,
        hints=hints,
        absorber_element=absorber_element,
        iodide_element=_normalize_element(active_settings.iodide_element),
        iodide_groups=iodide_groups,
    )
    _append_dmso_shells_and_hints(
        shells=shells,
        hints=hints,
        absorber_element=absorber_element,
        oxygen_element=_normalize_element(active_settings.oxygen_element),
        sulfur_element=_normalize_element(active_settings.sulfur_element),
        dmso_selections=dmso_selections,
    )

    return ArtemisGDSDocument(
        source_structure=str(path),
        settings=ArtemisGDSBuildSettings(
            absorber_element=absorber_element,
            absorber_atom_index=(
                absorber_indices[0] + 1 if len(absorber_indices) == 1 else None
            ),
            min_distance_angstrom=active_settings.min_distance_angstrom,
            max_distance_angstrom=max(
                active_settings.max_iodide_distance_angstrom,
                active_settings.max_oxygen_distance_angstrom,
            ),
            shell_tolerance_angstrom=active_settings.shell_tolerance_angstrom,
            initial_s02=active_settings.initial_s02,
            initial_e0=active_settings.initial_e0,
            initial_delta_r=active_settings.initial_delta_r,
            initial_sigma2=active_settings.initial_oxygen_sigma2,
            include_restraints=active_settings.include_restraints,
        ),
        shells=tuple(shells),
        parameters=tuple(parameters),
        path_hints=tuple(hints),
        overview_notes=(
            "Template: Pb-I / DMSO constrained GDS",
            (
                "Absorber paths were generated independently around each Pb "
                "absorber and then grouped when multi-absorber distances were "
                "within tolerance."
            ),
            (
                "Requested nearest DMSO oxygens per absorber: "
                f"{active_settings.oxygen_count}"
            ),
            (
                "Requested nearest iodides per absorber: "
                f"{_overview_count(active_settings.iodide_count)}"
            ),
            (
                "Pb-I cutoff: "
                f"{_format_float(active_settings.max_iodide_distance_angstrom)} A"
            ),
            (
                "Pb-O cutoff: "
                f"{_format_float(active_settings.max_oxygen_distance_angstrom)} A"
            ),
            (
                "O-S pairing cutoff: "
                f"{_format_float(active_settings.max_os_distance_angstrom)} A"
            ),
            (
                "DMSO Pb-S paths use a Pb-O-S three-atom geometry constraint "
                "linked to the corresponding Pb-O path."
            ),
        ),
    )


def _append_globals(
    parameters: list[ArtemisGDSParameter],
    settings: PbDMSOGDSBuildSettings,
) -> None:
    parameters.extend(
        (
            ArtemisGDSParameter(
                "guess" if settings.vary_s02 else "set",
                "snot",
                _format_float(settings.initial_s02),
                "global S02 amplitude factor",
            ),
            ArtemisGDSParameter(
                "guess",
                "enot",
                _format_float(settings.initial_e0),
                "global edge-energy shift",
            ),
        )
    )


def _append_coordination_parameters(
    parameters: list[ArtemisGDSParameter],
    *,
    oxygen_multiplicities: tuple[float, ...],
    iodide_multiplicities: tuple[float, ...],
    sulfur_path_indices: tuple[int, ...],
    settings: PbDMSOGDSBuildSettings,
) -> None:
    for index, multiplicity in enumerate(oxygen_multiplicities, start=1):
        parameters.append(
            ArtemisGDSParameter(
                "guess",
                f"cn_o{index}",
                _format_float(settings.initial_oxygen_cn * multiplicity),
                "Pb-O coordination number",
            )
        )

    for index, multiplicity in enumerate(iodide_multiplicities, start=1):
        parameters.append(
            ArtemisGDSParameter(
                "guess",
                f"cn_i{index}",
                _format_float(settings.initial_iodide_cn * multiplicity),
                "Pb-I coordination number",
            )
        )

    for index in sulfur_path_indices:
        parameters.append(
            ArtemisGDSParameter(
                "def",
                f"cn_s{index}",
                f"cn_o{index}",
                "Pb-S DMSO coordination linked to corresponding oxygen",
            )
        )


def _append_delta_r_parameters(
    parameters: list[ArtemisGDSParameter],
    *,
    oxygen_count: int,
    iodide_count: int,
    settings: PbDMSOGDSBuildSettings,
) -> None:
    for index in range(1, oxygen_count + 1):
        parameters.append(
            ArtemisGDSParameter(
                "guess",
                f"delr_o{index}",
                _format_float(settings.initial_delta_r),
                "Pb-O delta-R",
            )
        )
    for index in range(1, iodide_count + 1):
        parameters.append(
            ArtemisGDSParameter(
                "guess",
                f"delr_i{index}",
                _format_float(settings.initial_delta_r),
                "Pb-I delta-R",
            )
        )


def _append_sigma2_parameters(
    parameters: list[ArtemisGDSParameter],
    *,
    oxygen_count: int,
    iodide_count: int,
    settings: PbDMSOGDSBuildSettings,
) -> None:
    for index in range(1, oxygen_count + 1):
        if index == 1 or not settings.link_oxygen_sigma2:
            parameters.append(
                ArtemisGDSParameter(
                    "guess",
                    f"sig2_o{index}",
                    _format_float(settings.initial_oxygen_sigma2),
                    "Pb-O sigma2",
                )
            )
        else:
            parameters.append(
                ArtemisGDSParameter(
                    "def",
                    f"sig2_o{index}",
                    "sig2_o1",
                    "Pb-O sigma2 linked to first oxygen",
                )
            )
    for index in range(1, iodide_count + 1):
        parameters.append(
            ArtemisGDSParameter(
                "guess",
                f"sig2_i{index}",
                _format_float(settings.initial_iodide_sigma2),
                "Pb-I sigma2",
            )
        )


def _append_geometry_parameters(
    parameters: list[ArtemisGDSParameter],
    *,
    dmso_selections: tuple[_DMSOPathSelection, ...],
    sulfur_path_indices: tuple[int, ...],
    theta_degrees: float,
    settings: PbDMSOGDSBuildSettings,
) -> None:
    if sulfur_path_indices:
        parameters.extend(
            (
                ArtemisGDSParameter(
                    "set",
                    "bl_os",
                    _format_float(settings.bl_os_angstrom),
                    "fixed O-S bond length",
                ),
                ArtemisGDSParameter(
                    "guess",
                    "theta_pbos",
                    _format_float(theta_degrees),
                    "mean Pb-O-S angle in degrees",
                ),
                ArtemisGDSParameter(
                    "def",
                    "theta_pbos_rad",
                    "theta_pbos*pi/180",
                    "mean Pb-O-S angle in radians",
                ),
                ArtemisGDSParameter(
                    "guess",
                    "width",
                    _format_float(settings.angle_width_degrees),
                    "Pb-O-S angular width in degrees",
                ),
                ArtemisGDSParameter(
                    "def",
                    "sig2_theta_pbos",
                    "((width/2)*pi/180)*((width/2)*pi/180)",
                    "Pb-O-S angle variance in radians squared",
                ),
            )
        )
    for index, selection in enumerate(dmso_selections, start=1):
        group = selection.group
        if selection.include_sulfur_path:
            parameters.extend(
                (
                    ArtemisGDSParameter(
                        "set",
                        f"reff_pbo_{index}",
                        _format_float(group.pb_o_distance),
                        "fixed model Pb-O distance used by Pb-S geometry",
                    ),
                    ArtemisGDSParameter(
                        "set",
                        f"reff_pbs_{index}",
                        _format_float(group.pb_s_distance),
                        "fixed model Pb-S distance used by Pb-S geometry",
                    ),
                )
            )
    for index in sulfur_path_indices:
        parameters.extend(
            (
                ArtemisGDSParameter(
                    "def",
                    f"delr_s{index}",
                    _terminal_delta_r_expression(index),
                    "geometry-linked Pb-S delta-R",
                ),
                ArtemisGDSParameter(
                    "def",
                    f"sig2_s{index}",
                    _terminal_sigma2_expression(index),
                    "geometry-linked Pb-S sigma2",
                ),
            )
        )


def _append_restraints(
    parameters: list[ArtemisGDSParameter],
    *,
    oxygen_count: int,
    iodide_count: int,
    sulfur_path_indices: tuple[int, ...],
    settings: PbDMSOGDSBuildSettings,
) -> None:
    scale = _format_float(settings.restraint_scale)
    parameters.append(
        ArtemisGDSParameter(
            "restrain",
            "res_enot",
            (
                f"{scale}*penalty(enot, "
                f"{_format_float(settings.e0_lower_bound)}, "
                f"{_format_float(settings.e0_upper_bound)})"
            ),
            "soft E0 bound",
        )
    )
    for index in range(1, oxygen_count + 1):
        parameters.append(
            ArtemisGDSParameter(
                "restrain",
                f"res_cn_o{index}",
                (
                    f"{scale}*penalty(cn_o{index}, "
                    f"{_format_float(settings.oxygen_cn_lower_bound)}, "
                    f"{_format_float(settings.oxygen_cn_upper_bound)})"
                ),
                "soft Pb-O coordination bound",
            )
        )
    for index in range(1, iodide_count + 1):
        parameters.append(
            ArtemisGDSParameter(
                "restrain",
                f"res_cn_i{index}",
                (
                    f"{scale}*penalty(cn_i{index}, "
                    f"{_format_float(settings.iodide_cn_lower_bound)}, "
                    f"{_format_float(settings.iodide_cn_upper_bound)})"
                ),
                "soft Pb-I coordination bound",
            )
        )
    if sulfur_path_indices:
        parameters.extend(
            (
                ArtemisGDSParameter(
                    "restrain",
                    "res_theta_pbos",
                    (
                        f"{scale}*penalty(theta_pbos, "
                        f"{_format_float(settings.theta_lower_bound_degrees)}, "
                        f"{_format_float(settings.theta_upper_bound_degrees)})"
                    ),
                    "soft Pb-O-S angle bound",
                ),
                ArtemisGDSParameter(
                    "restrain",
                    "res_width",
                    (
                        f"{scale}*penalty(width, "
                        f"{_format_float(settings.width_lower_bound_degrees)}, "
                        f"{_format_float(settings.width_upper_bound_degrees)})"
                    ),
                    "soft Pb-O-S angular-width bound",
                ),
            )
        )
    for index in range(1, iodide_count + 1):
        parameters.append(
            ArtemisGDSParameter(
                "restrain",
                f"res_sig2_i{index}",
                (
                    f"{scale}*penalty(sig2_i{index}, "
                    f"{_format_float(settings.sigma2_lower_bound)}, "
                    f"{_format_float(settings.sigma2_upper_bound)})"
                ),
                "soft Pb-I sigma2 bound",
            )
        )


def _append_iodide_shells_and_hints(
    *,
    shells: list[ArtemisGDSShell],
    hints: list[ArtemisPathParameterHint],
    absorber_element: str,
    iodide_element: str,
    iodide_groups: tuple[_IodideGroup, ...],
) -> None:
    for index, group in enumerate(iodide_groups, start=1):
        distance = group.mean_distance
        label = f"pb_i_{index}"
        shells.append(
            ArtemisGDSShell(
                label=label,
                absorber_element=absorber_element,
                scatterer_element=iodide_element,
                multiplicity=group.multiplicity,
                mean_distance_angstrom=distance,
                std_distance_angstrom=group.std_distance,
                min_distance_angstrom=group.min_distance,
                max_distance_angstrom=group.max_distance,
            )
        )
        hints.append(
            ArtemisPathParameterHint(
                shell_label=label,
                s02=f"snot*cn_i{index}",
                e0="enot",
                delr=f"delr_i{index}",
                sigma2=f"sig2_i{index}",
                reff_angstrom=distance,
                multiplicity=group.multiplicity,
            )
        )


def _append_dmso_shells_and_hints(
    *,
    shells: list[ArtemisGDSShell],
    hints: list[ArtemisPathParameterHint],
    absorber_element: str,
    oxygen_element: str,
    sulfur_element: str,
    dmso_selections: tuple[_DMSOPathSelection, ...],
) -> None:
    for index, selection in enumerate(dmso_selections, start=1):
        group = selection.group
        oxygen_label = f"pb_o_{index}"
        sulfur_label = f"pb_s_{index}"
        if selection.include_oxygen_path:
            shells.append(
                ArtemisGDSShell(
                    label=oxygen_label,
                    absorber_element=absorber_element,
                    scatterer_element=oxygen_element,
                    multiplicity=group.multiplicity,
                    mean_distance_angstrom=group.pb_o_distance,
                    std_distance_angstrom=group.pb_o_std_distance,
                    min_distance_angstrom=group.pb_o_min_distance,
                    max_distance_angstrom=group.pb_o_max_distance,
                )
            )
            hints.append(
                ArtemisPathParameterHint(
                    shell_label=oxygen_label,
                    s02=f"snot*cn_o{index}",
                    e0="enot",
                    delr=f"delr_o{index}",
                    sigma2=f"sig2_o{index}",
                    reff_angstrom=group.pb_o_distance,
                    multiplicity=group.multiplicity,
                )
            )
        if selection.include_sulfur_path:
            shells.append(
                ArtemisGDSShell(
                    label=sulfur_label,
                    absorber_element=absorber_element,
                    scatterer_element=sulfur_element,
                    multiplicity=group.multiplicity,
                    mean_distance_angstrom=group.pb_s_distance,
                    std_distance_angstrom=group.pb_s_std_distance,
                    min_distance_angstrom=group.pb_s_min_distance,
                    max_distance_angstrom=group.pb_s_max_distance,
                )
            )
            hints.append(
                ArtemisPathParameterHint(
                    shell_label=sulfur_label,
                    s02=f"snot*cn_s{index}",
                    e0="enot",
                    delr=f"delr_s{index}",
                    sigma2=f"sig2_s{index}",
                    reff_angstrom=group.pb_s_distance,
                    multiplicity=group.multiplicity,
                ),
            )


def _resolve_absorbers(
    elements: tuple[str, ...],
    settings: PbDMSOGDSBuildSettings,
) -> tuple[int, ...]:
    absorber_element = _normalize_element(settings.absorber_element)
    if settings.absorber_atom_index is not None:
        atom_index = int(settings.absorber_atom_index)
        if atom_index < 1 or atom_index > len(elements):
            raise ValueError(
                "Absorber atom index is one-based and must refer to an "
                "atom in the structure."
            )
        zero_based = atom_index - 1
        if elements[zero_based] != absorber_element:
            raise ValueError(
                "Absorber atom index element "
                f"{elements[zero_based]!r} does not match requested "
                f"absorber element {absorber_element!r}."
            )
        return (zero_based,)

    absorber_indices = tuple(
        index
        for index, element in enumerate(elements)
        if element == absorber_element
    )
    if not absorber_indices:
        raise ValueError(
            f"No absorber atoms with element {absorber_element!r} were "
            "found in the structure."
        )
    return absorber_indices


def _collect_iodide_paths(
    *,
    positions: np.ndarray,
    elements: tuple[str, ...],
    absorber_indices: tuple[int, ...],
    settings: PbDMSOGDSBuildSettings,
) -> tuple[_IodidePath, ...]:
    iodide_paths: list[_IodidePath] = []
    require_count = len(absorber_indices) == 1
    iodide_element = _normalize_element(settings.iodide_element)
    for absorber_index in absorber_indices:
        iodide_indices = _select_nearest_indices(
            positions=positions,
            elements=elements,
            absorber_index=absorber_index,
            element=iodide_element,
            min_distance=settings.min_distance_angstrom,
            max_distance=settings.max_iodide_distance_angstrom,
            count=settings.iodide_count,
            role="iodide",
            require_count=require_count,
        )
        iodide_paths.extend(
            _IodidePath(
                absorber_index=absorber_index,
                iodide_index=iodide_index,
                distance=_distance(positions, absorber_index, iodide_index),
            )
            for iodide_index in iodide_indices
        )
    iodide_paths = [
        path
        for path in iodide_paths
        if _path_pair_is_included(
            active_pairs=settings.included_path_pairs,
            absorber_index=path.absorber_index,
            scatterer_index=path.iodide_index,
        )
    ]
    if not iodide_paths:
        if settings.included_path_pairs is not None:
            return ()
        raise ValueError(
            f"No iodide atoms with element {iodide_element!r} were found "
            "within the Pb-I distance cutoff for any absorber."
        )
    return tuple(iodide_paths)


def _collect_dmso_pairs(
    *,
    positions: np.ndarray,
    elements: tuple[str, ...],
    absorber_indices: tuple[int, ...],
    settings: PbDMSOGDSBuildSettings,
) -> tuple[_DMSOPair, ...]:
    dmso_pairs: list[_DMSOPair] = []
    require_count = len(absorber_indices) == 1
    for absorber_index in absorber_indices:
        oxygen_indices = _select_nearest_indices(
            positions=positions,
            elements=elements,
            absorber_index=absorber_index,
            element=_normalize_element(settings.oxygen_element),
            min_distance=settings.min_distance_angstrom,
            max_distance=settings.max_oxygen_distance_angstrom,
            count=settings.oxygen_count,
            role="oxygen",
            require_count=require_count,
        )
        if not oxygen_indices:
            continue
        dmso_pairs.extend(
            _pair_oxygens_to_sulfurs(
                positions=positions,
                elements=elements,
                absorber_index=absorber_index,
                oxygen_indices=oxygen_indices,
                sulfur_element=_normalize_element(settings.sulfur_element),
                max_os_distance=settings.max_os_distance_angstrom,
            )
        )
    if not dmso_pairs:
        raise ValueError(
            "No coordinated DMSO oxygens were found within the Pb-O distance "
            "cutoff for any absorber."
        )
    return tuple(dmso_pairs)


def _group_iodide_paths(
    paths: tuple[_IodidePath, ...],
    *,
    absorber_count: int,
    tolerance: float,
) -> tuple[_IodideGroup, ...]:
    if absorber_count == 1:
        return tuple(
            _IodideGroup(paths=(path,), absorber_count=absorber_count)
            for path in sorted(paths, key=lambda path: path.distance)
        )
    return tuple(
        _IodideGroup(paths=tuple(cluster), absorber_count=absorber_count)
        for cluster in _cluster_records(
            paths,
            feature_getter=lambda path: (path.distance,),
            tolerance=tolerance,
        )
    )


def _group_dmso_pairs(
    pairs: tuple[_DMSOPair, ...],
    *,
    absorber_count: int,
    tolerance: float,
) -> tuple[_DMSOGroup, ...]:
    if absorber_count == 1:
        return tuple(
            _DMSOGroup(pairs=(pair,), absorber_count=absorber_count)
            for pair in sorted(
                pairs,
                key=lambda pair: (pair.pb_o_distance, pair.pb_s_distance),
            )
        )
    return tuple(
        _DMSOGroup(pairs=tuple(cluster), absorber_count=absorber_count)
        for cluster in _cluster_records(
            pairs,
            feature_getter=lambda pair: (
                pair.pb_o_distance,
                pair.pb_s_distance,
            ),
            tolerance=tolerance,
        )
    )


def _dmso_group_path_selections(
    groups: tuple[_DMSOGroup, ...],
    *,
    included_path_pairs: tuple[tuple[int, int], ...] | None,
) -> tuple[_DMSOPathSelection, ...]:
    selections: list[_DMSOPathSelection] = []
    for group in groups:
        include_oxygen = any(
            _path_pair_is_included(
                active_pairs=included_path_pairs,
                absorber_index=pair.absorber_index,
                scatterer_index=pair.oxygen_index,
            )
            for pair in group.pairs
        )
        include_sulfur = any(
            _path_pair_is_included(
                active_pairs=included_path_pairs,
                absorber_index=pair.absorber_index,
                scatterer_index=pair.sulfur_index,
            )
            for pair in group.pairs
        )
        selection = _DMSOPathSelection(
            group=group,
            include_oxygen_path=include_oxygen,
            include_sulfur_path=include_sulfur,
        )
        if selection.is_active:
            selections.append(selection)
    return tuple(selections)


def _path_pair_is_included(
    *,
    active_pairs: tuple[tuple[int, int], ...] | None,
    absorber_index: int,
    scatterer_index: int,
) -> bool:
    if active_pairs is None:
        return True
    return (absorber_index + 1, scatterer_index + 1) in active_pairs


def _select_nearest_indices(
    *,
    positions: np.ndarray,
    elements: tuple[str, ...],
    absorber_index: int,
    element: str,
    min_distance: float,
    max_distance: float,
    count: int | None,
    role: str,
    require_count: bool = True,
) -> tuple[int, ...]:
    candidates = []
    for atom_index, atom_element in enumerate(elements):
        if atom_index == absorber_index or atom_element != element:
            continue
        distance = _distance(positions, absorber_index, atom_index)
        if min_distance <= distance <= max_distance:
            candidates.append((distance, atom_index))
    candidates.sort()
    if count is not None:
        if count < 1:
            raise ValueError(f"{role.capitalize()} count must be at least 1.")
        candidates = candidates[: int(count)]
    if not candidates:
        if not require_count:
            return ()
        raise ValueError(
            f"No {role} atoms with element {element!r} were found within "
            f"{_format_float(max_distance)} angstrom of the absorber."
        )
    if count is not None and len(candidates) < int(count) and require_count:
        raise ValueError(
            f"Requested {count} {role} atoms, but only {len(candidates)} "
            "were found within the distance cutoff."
        )
    return tuple(atom_index for _distance_value, atom_index in candidates)


def _pair_oxygens_to_sulfurs(
    *,
    positions: np.ndarray,
    elements: tuple[str, ...],
    absorber_index: int,
    oxygen_indices: tuple[int, ...],
    sulfur_element: str,
    max_os_distance: float,
) -> list[_DMSOPair]:
    sulfur_indices = [
        atom_index
        for atom_index, element in enumerate(elements)
        if element == sulfur_element
    ]
    if not sulfur_indices:
        raise ValueError(
            f"No sulfur atoms with element {sulfur_element!r} were found."
        )

    used_sulfurs: set[int] = set()
    pairs: list[_DMSOPair] = []
    for oxygen_index in oxygen_indices:
        candidates = [
            (
                _distance(positions, oxygen_index, sulfur_index),
                sulfur_index,
            )
            for sulfur_index in sulfur_indices
            if sulfur_index not in used_sulfurs
        ]
        candidates.sort()
        if not candidates or candidates[0][0] > max_os_distance:
            raise ValueError(
                "Could not find an unused DMSO sulfur within "
                f"{_format_float(max_os_distance)} angstrom of oxygen atom "
                f"{oxygen_index + 1}."
            )
        os_distance, sulfur_index = candidates[0]
        used_sulfurs.add(sulfur_index)
        pairs.append(
            _DMSOPair(
                absorber_index=absorber_index,
                oxygen_index=oxygen_index,
                sulfur_index=sulfur_index,
                pb_o_distance=_distance(
                    positions, absorber_index, oxygen_index
                ),
                pb_s_distance=_distance(
                    positions, absorber_index, sulfur_index
                ),
                os_distance=os_distance,
                angle_degrees=_angle_degrees(
                    positions[absorber_index],
                    positions[oxygen_index],
                    positions[sulfur_index],
                ),
            )
        )
    return pairs


def _cluster_records(
    records: tuple[_T, ...],
    *,
    feature_getter: Callable[[_T], tuple[float, ...]],
    tolerance: float,
) -> list[list[_T]]:
    clusters: list[list[_T]] = []
    for record in sorted(records, key=feature_getter):
        features = tuple(float(value) for value in feature_getter(record))
        if not clusters:
            clusters.append([record])
            continue
        current_features = [
            tuple(float(value) for value in feature_getter(item))
            for item in clusters[-1]
        ]
        current_mean = tuple(
            sum(values) / len(values) for values in zip(*current_features)
        )
        if (
            max(
                abs(feature - mean)
                for feature, mean in zip(features, current_mean)
            )
            <= tolerance
        ):
            clusters[-1].append(record)
        else:
            clusters.append([record])
    return clusters


def _terminal_delta_r_expression(index: int) -> str:
    anchor_distance = f"(reff_pbo_{index} + delr_o{index})"
    return (
        f"sqrt({anchor_distance}*{anchor_distance} + bl_os*bl_os - "
        f"2*{anchor_distance}*bl_os*cos(theta_pbos_rad)) - "
        f"reff_pbs_{index}"
    )


def _terminal_sigma2_expression(index: int) -> str:
    anchor_distance = f"(reff_pbo_{index} + delr_o{index})"
    terminal_distance = f"(reff_pbs_{index} + delr_s{index})"
    radial = (
        f"(({anchor_distance} - bl_os*cos(theta_pbos_rad))/"
        f"{terminal_distance})"
    )
    angular = (
        f"(({anchor_distance}*bl_os*sin(theta_pbos_rad))/"
        f"{terminal_distance})"
    )
    return (
        f"{radial}*{radial}*sig2_o{index} + "
        f"{angular}*{angular}*sig2_theta_pbos"
    )


def _sum_expression(terms: object) -> str:
    values = tuple(str(term) for term in terms)
    if not values:
        return "0"
    return " + ".join(values)


def _distance(
    positions: np.ndarray,
    first_index: int,
    second_index: int,
) -> float:
    return float(
        np.linalg.norm(positions[first_index] - positions[second_index])
    )


def _angle_degrees(
    first: np.ndarray,
    vertex: np.ndarray,
    third: np.ndarray,
) -> float:
    first_vector = first - vertex
    third_vector = third - vertex
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(third_vector)
    if denominator == 0.0:
        raise ValueError("Cannot compute an angle with a zero-length vector.")
    cosine = float(np.dot(first_vector, third_vector) / denominator)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _mean(values: object) -> float:
    sequence = tuple(float(value) for value in values)
    if not sequence:
        raise ValueError("Cannot compute a mean from an empty sequence.")
    return sum(sequence) / len(sequence)


def _std(values: object) -> float:
    sequence = tuple(float(value) for value in values)
    if not sequence:
        raise ValueError(
            "Cannot compute a standard deviation from an empty sequence."
        )
    return float(np.asarray(sequence, dtype=float).std(ddof=0))


def _overview_count(value: int | None) -> str:
    if value is None:
        return "all within cutoff"
    return str(value)


def _normalize_element(element: str) -> str:
    text = str(element).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()
