from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

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
from .geometry import (
    dihedral_chord_delta_r_expression,
    dihedral_chord_sigma2_expression,
    three_atom_chord_delta_r_expression,
    three_atom_chord_sigma2_expression,
)
from .pb_dmso import (
    _append_globals,
    _append_iodide_shells_and_hints,
    _cluster_records,
    _collect_iodide_paths,
    _distance,
    _group_iodide_paths,
    _mean,
    _overview_count,
    _path_pair_is_included,
    _select_nearest_indices,
    _std,
)


@dataclass(slots=True, frozen=True)
class PbDMFGDSBuildSettings:
    """Build settings for compact Pb-I/DMF Artemis GDS files."""

    absorber_element: str = "Pb"
    absorber_atom_index: int | None = None
    iodide_element: str = "I"
    oxygen_element: str = "O"
    carbon_element: str = "C"
    nitrogen_element: str = "N"
    oxygen_count: int = 3
    iodide_count: int | None = None
    min_distance_angstrom: float = 0.5
    max_iodide_distance_angstrom: float = 4.0
    max_oxygen_distance_angstrom: float = 4.0
    max_oc_distance_angstrom: float = 1.8
    max_cn_distance_angstrom: float = 1.8
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
    bl_oc_angstrom: float = 1.25
    bl_cn_angstrom: float = 1.35
    theta_pboc_degrees: float | None = None
    theta_ocn_degrees: float | None = None
    phi_pbocn_degrees: float | None = None
    theta_width_degrees: float = 8.0
    internal_angle_width_degrees: float = 6.0
    dihedral_width_degrees: float = 12.0
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
    theta_lower_bound_degrees: float = 80.0
    theta_upper_bound_degrees: float = 180.0
    width_lower_bound_degrees: float = 0.0
    width_upper_bound_degrees: float = 60.0
    included_path_pairs: tuple[tuple[int, int], ...] | None = None


@dataclass(slots=True, frozen=True)
class _DMFTriplet:
    absorber_index: int
    oxygen_index: int
    carbon_index: int
    nitrogen_index: int
    pb_o_distance: float
    pb_c_distance: float
    pb_n_distance: float
    oc_distance: float
    cn_distance: float
    pboc_angle_degrees: float
    ocn_angle_degrees: float
    pbocn_dihedral_degrees: float


@dataclass(slots=True, frozen=True)
class _DMFGroup:
    triplets: tuple[_DMFTriplet, ...]
    absorber_count: int

    @property
    def multiplicity(self) -> float:
        return float(len(self.triplets)) / float(self.absorber_count)

    @property
    def pb_o_distance(self) -> float:
        return _mean(triplet.pb_o_distance for triplet in self.triplets)

    @property
    def pb_c_distance(self) -> float:
        return _mean(triplet.pb_c_distance for triplet in self.triplets)

    @property
    def pb_n_distance(self) -> float:
        return _mean(triplet.pb_n_distance for triplet in self.triplets)

    @property
    def pb_o_std_distance(self) -> float:
        return _std(triplet.pb_o_distance for triplet in self.triplets)

    @property
    def pb_c_std_distance(self) -> float:
        return _std(triplet.pb_c_distance for triplet in self.triplets)

    @property
    def pb_n_std_distance(self) -> float:
        return _std(triplet.pb_n_distance for triplet in self.triplets)

    @property
    def pb_o_min_distance(self) -> float:
        return min(triplet.pb_o_distance for triplet in self.triplets)

    @property
    def pb_o_max_distance(self) -> float:
        return max(triplet.pb_o_distance for triplet in self.triplets)

    @property
    def pb_c_min_distance(self) -> float:
        return min(triplet.pb_c_distance for triplet in self.triplets)

    @property
    def pb_c_max_distance(self) -> float:
        return max(triplet.pb_c_distance for triplet in self.triplets)

    @property
    def pb_n_min_distance(self) -> float:
        return min(triplet.pb_n_distance for triplet in self.triplets)

    @property
    def pb_n_max_distance(self) -> float:
        return max(triplet.pb_n_distance for triplet in self.triplets)


@dataclass(slots=True, frozen=True)
class _DMFPathSelection:
    group: _DMFGroup
    include_oxygen_path: bool
    include_carbon_path: bool
    include_nitrogen_path: bool

    @property
    def is_active(self) -> bool:
        return (
            self.include_oxygen_path
            or self.include_carbon_path
            or self.include_nitrogen_path
        )


def build_pb_dmf_gds_from_structure(
    structure_path: str | Path,
    settings: PbDMFGDSBuildSettings | None = None,
) -> ArtemisGDSDocument:
    """Build a compact hand-fit-style Pb-I/DMF GDS file."""

    active_settings = settings or PbDMFGDSBuildSettings()
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
    dmf_triplets = _collect_dmf_triplets(
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
    dmf_groups = _group_dmf_triplets(
        dmf_triplets,
        absorber_count=len(absorber_indices),
        tolerance=tolerance,
    )
    dmf_selections = _dmf_group_path_selections(
        dmf_groups,
        included_path_pairs=active_settings.included_path_pairs,
    )
    carbon_path_indices = tuple(
        index
        for index, selection in enumerate(dmf_selections, start=1)
        if selection.include_carbon_path
    )
    nitrogen_path_indices = tuple(
        index
        for index, selection in enumerate(dmf_selections, start=1)
        if selection.include_nitrogen_path
    )
    theta_pboc_degrees = (
        float(active_settings.theta_pboc_degrees)
        if active_settings.theta_pboc_degrees is not None
        else _mean(pair.pboc_angle_degrees for pair in dmf_triplets)
    )
    theta_ocn_degrees = (
        float(active_settings.theta_ocn_degrees)
        if active_settings.theta_ocn_degrees is not None
        else _mean(pair.ocn_angle_degrees for pair in dmf_triplets)
    )
    phi_pbocn_degrees = (
        float(active_settings.phi_pbocn_degrees)
        if active_settings.phi_pbocn_degrees is not None
        else _circular_mean_degrees(
            pair.pbocn_dihedral_degrees for pair in dmf_triplets
        )
    )

    parameters: list[ArtemisGDSParameter] = []
    shells: list[ArtemisGDSShell] = []
    hints: list[ArtemisPathParameterHint] = []

    _append_globals(parameters, active_settings)
    _append_coordination_parameters(
        parameters,
        oxygen_multiplicities=tuple(
            selection.group.multiplicity for selection in dmf_selections
        ),
        iodide_multiplicities=tuple(
            group.multiplicity for group in iodide_groups
        ),
        carbon_path_indices=carbon_path_indices,
        nitrogen_path_indices=nitrogen_path_indices,
        settings=active_settings,
    )
    _append_delta_r_parameters(
        parameters,
        oxygen_count=len(dmf_selections),
        iodide_count=len(iodide_groups),
        carbon_path_indices=carbon_path_indices,
        nitrogen_path_indices=nitrogen_path_indices,
        settings=active_settings,
    )
    _append_sigma2_parameters(
        parameters,
        oxygen_count=len(dmf_selections),
        iodide_count=len(iodide_groups),
        carbon_path_indices=carbon_path_indices,
        nitrogen_path_indices=nitrogen_path_indices,
        settings=active_settings,
    )
    _append_geometry_parameters(
        parameters,
        dmf_selections=dmf_selections,
        carbon_path_indices=carbon_path_indices,
        nitrogen_path_indices=nitrogen_path_indices,
        theta_pboc_degrees=theta_pboc_degrees,
        theta_ocn_degrees=theta_ocn_degrees,
        phi_pbocn_degrees=phi_pbocn_degrees,
        settings=active_settings,
    )
    if active_settings.include_restraints:
        _append_restraints(
            parameters,
            oxygen_count=len(dmf_selections),
            iodide_count=len(iodide_groups),
            carbon_path_indices=carbon_path_indices,
            nitrogen_path_indices=nitrogen_path_indices,
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
    _append_dmf_shells_and_hints(
        shells=shells,
        hints=hints,
        absorber_element=absorber_element,
        oxygen_element=_normalize_element(active_settings.oxygen_element),
        carbon_element=_normalize_element(active_settings.carbon_element),
        nitrogen_element=_normalize_element(active_settings.nitrogen_element),
        dmf_selections=dmf_selections,
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
            "Template: Pb-I / DMF constrained GDS",
            (
                "Absorber paths were generated independently around each Pb "
                "absorber and then grouped when multi-absorber distances were "
                "within tolerance."
            ),
            (
                "Requested nearest DMF oxygens per absorber: "
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
                "O-C pairing cutoff: "
                f"{_format_float(active_settings.max_oc_distance_angstrom)} A"
            ),
            (
                "C-N pairing cutoff: "
                f"{_format_float(active_settings.max_cn_distance_angstrom)} A"
            ),
            (
                "DMF Pb-C and Pb-N paths use linked three-atom and four-atom "
                "geometry constraints from the corresponding Pb-O path."
            ),
        ),
    )


def _append_coordination_parameters(
    parameters: list[ArtemisGDSParameter],
    *,
    oxygen_multiplicities: tuple[float, ...],
    iodide_multiplicities: tuple[float, ...],
    carbon_path_indices: tuple[int, ...],
    nitrogen_path_indices: tuple[int, ...],
    settings: PbDMFGDSBuildSettings,
) -> None:
    for index, multiplicity in enumerate(oxygen_multiplicities, start=1):
        parameters.append(
            ArtemisGDSParameter(
                "guess",
                f"cn_o{index}",
                _format_float(settings.initial_oxygen_cn * multiplicity),
                "Pb-O DMF coordination number",
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

    for prefix, element_name, path_indices in (
        ("c", "carbonyl carbon", carbon_path_indices),
        ("n", "nitrogen", nitrogen_path_indices),
    ):
        for index in path_indices:
            parameters.append(
                ArtemisGDSParameter(
                    "def",
                    f"cn_{prefix}{index}",
                    f"cn_o{index}",
                    f"Pb-{element_name} DMF coordination linked to oxygen",
                )
            )


def _append_delta_r_parameters(
    parameters: list[ArtemisGDSParameter],
    *,
    oxygen_count: int,
    iodide_count: int,
    carbon_path_indices: tuple[int, ...],
    nitrogen_path_indices: tuple[int, ...],
    settings: PbDMFGDSBuildSettings,
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
    for index in carbon_path_indices:
        parameters.append(
            ArtemisGDSParameter(
                "def",
                f"delr_c{index}",
                _carbon_delta_r_expression(index),
                "geometry-linked Pb-C delta-R",
            )
        )
    for index in nitrogen_path_indices:
        parameters.append(
            ArtemisGDSParameter(
                "def",
                f"delr_n{index}",
                _nitrogen_delta_r_expression(index),
                "dihedral-linked Pb-N delta-R",
            )
        )


def _append_sigma2_parameters(
    parameters: list[ArtemisGDSParameter],
    *,
    oxygen_count: int,
    iodide_count: int,
    carbon_path_indices: tuple[int, ...],
    nitrogen_path_indices: tuple[int, ...],
    settings: PbDMFGDSBuildSettings,
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
    for index in carbon_path_indices:
        parameters.append(
            ArtemisGDSParameter(
                "def",
                f"sig2_c{index}",
                _carbon_sigma2_expression(index),
                "geometry-linked Pb-C sigma2",
            )
        )
    for index in nitrogen_path_indices:
        parameters.append(
            ArtemisGDSParameter(
                "def",
                f"sig2_n{index}",
                _nitrogen_sigma2_expression(index),
                "dihedral-linked Pb-N sigma2",
            )
        )


def _append_geometry_parameters(
    parameters: list[ArtemisGDSParameter],
    *,
    dmf_selections: tuple[_DMFPathSelection, ...],
    carbon_path_indices: tuple[int, ...],
    nitrogen_path_indices: tuple[int, ...],
    theta_pboc_degrees: float,
    theta_ocn_degrees: float,
    phi_pbocn_degrees: float,
    settings: PbDMFGDSBuildSettings,
) -> None:
    include_carbon_geometry = bool(
        carbon_path_indices or nitrogen_path_indices
    )
    include_nitrogen_geometry = bool(nitrogen_path_indices)
    if not include_carbon_geometry:
        return

    parameters.extend(
        (
            ArtemisGDSParameter(
                "set",
                "bl_oc",
                _format_float(settings.bl_oc_angstrom),
                "fixed DMF O-C bond length",
            ),
            ArtemisGDSParameter(
                "guess",
                "theta_pboc",
                _format_float(theta_pboc_degrees),
                "mean Pb-O-C angle in degrees",
            ),
            ArtemisGDSParameter(
                "def",
                "theta_pboc_rad",
                "theta_pboc*pi/180",
                "mean Pb-O-C angle in radians",
            ),
            ArtemisGDSParameter(
                "guess",
                "width_pboc",
                _format_float(settings.theta_width_degrees),
                "Pb-O-C angular width in degrees",
            ),
            ArtemisGDSParameter(
                "def",
                "sig2_theta_pboc",
                "((width_pboc/2)*pi/180)*((width_pboc/2)*pi/180)",
                "Pb-O-C angle variance in radians squared",
            ),
        )
    )
    if include_nitrogen_geometry:
        parameters.extend(
            (
                ArtemisGDSParameter(
                    "set",
                    "bl_cn",
                    _format_float(settings.bl_cn_angstrom),
                    "fixed DMF C-N bond length",
                ),
                ArtemisGDSParameter(
                    "guess",
                    "theta_ocn",
                    _format_float(theta_ocn_degrees),
                    "mean O-C-N angle in degrees",
                ),
                ArtemisGDSParameter(
                    "def",
                    "theta_ocn_rad",
                    "theta_ocn*pi/180",
                    "mean O-C-N angle in radians",
                ),
                ArtemisGDSParameter(
                    "guess",
                    "phi_pbocn",
                    _format_float(phi_pbocn_degrees),
                    "mean Pb-O-C-N dihedral in degrees",
                ),
                ArtemisGDSParameter(
                    "def",
                    "phi_pbocn_rad",
                    "phi_pbocn*pi/180",
                    "mean Pb-O-C-N dihedral in radians",
                ),
                ArtemisGDSParameter(
                    "guess",
                    "width_ocn",
                    _format_float(settings.internal_angle_width_degrees),
                    "O-C-N angular width in degrees",
                ),
                ArtemisGDSParameter(
                    "def",
                    "sig2_theta_ocn",
                    "((width_ocn/2)*pi/180)*((width_ocn/2)*pi/180)",
                    "O-C-N angle variance in radians squared",
                ),
                ArtemisGDSParameter(
                    "guess",
                    "width_phi_pbocn",
                    _format_float(settings.dihedral_width_degrees),
                    "Pb-O-C-N dihedral width in degrees",
                ),
                ArtemisGDSParameter(
                    "def",
                    "sig2_phi_pbocn",
                    (
                        "((width_phi_pbocn/2)*pi/180)*"
                        "((width_phi_pbocn/2)*pi/180)"
                    ),
                    "Pb-O-C-N dihedral variance in radians squared",
                ),
            )
        )
    for index, selection in enumerate(dmf_selections, start=1):
        group = selection.group
        if selection.include_carbon_path or selection.include_nitrogen_path:
            parameters.append(
                ArtemisGDSParameter(
                    "set",
                    f"reff_pbo_{index}",
                    _format_float(group.pb_o_distance),
                    "fixed model Pb-O distance used by terminal geometry",
                )
            )
        if selection.include_carbon_path:
            parameters.append(
                ArtemisGDSParameter(
                    "set",
                    f"reff_pbc_{index}",
                    _format_float(group.pb_c_distance),
                    "fixed model Pb-C distance used by Pb-C geometry",
                )
            )
        if selection.include_nitrogen_path:
            parameters.append(
                ArtemisGDSParameter(
                    "set",
                    f"reff_pbn_{index}",
                    _format_float(group.pb_n_distance),
                    "fixed model Pb-N distance used by Pb-N geometry",
                )
            )


def _append_restraints(
    parameters: list[ArtemisGDSParameter],
    *,
    oxygen_count: int,
    iodide_count: int,
    carbon_path_indices: tuple[int, ...],
    nitrogen_path_indices: tuple[int, ...],
    settings: PbDMFGDSBuildSettings,
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
    geometry_restraints: list[tuple[str, float, float, str]] = []
    if carbon_path_indices or nitrogen_path_indices:
        geometry_restraints.extend(
            (
                (
                    "theta_pboc",
                    settings.theta_lower_bound_degrees,
                    settings.theta_upper_bound_degrees,
                    "soft Pb-O-C angle bound",
                ),
                (
                    "width_pboc",
                    settings.width_lower_bound_degrees,
                    settings.width_upper_bound_degrees,
                    "soft Pb-O-C angular-width bound",
                ),
            )
        )
    if nitrogen_path_indices:
        geometry_restraints.extend(
            (
                (
                    "theta_ocn",
                    settings.theta_lower_bound_degrees,
                    settings.theta_upper_bound_degrees,
                    "soft O-C-N angle bound",
                ),
                (
                    "width_ocn",
                    settings.width_lower_bound_degrees,
                    settings.width_upper_bound_degrees,
                    "soft O-C-N angular-width bound",
                ),
                (
                    "width_phi_pbocn",
                    settings.width_lower_bound_degrees,
                    settings.width_upper_bound_degrees,
                    "soft Pb-O-C-N dihedral-width bound",
                ),
            )
        )
    for name, lower, upper, comment in geometry_restraints:
        parameters.append(
            ArtemisGDSParameter(
                "restrain",
                f"res_{name}",
                (
                    f"{scale}*penalty({name}, "
                    f"{_format_float(lower)}, {_format_float(upper)})"
                ),
                comment,
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


def _append_dmf_shells_and_hints(
    *,
    shells: list[ArtemisGDSShell],
    hints: list[ArtemisPathParameterHint],
    absorber_element: str,
    oxygen_element: str,
    carbon_element: str,
    nitrogen_element: str,
    dmf_selections: tuple[_DMFPathSelection, ...],
) -> None:
    for index, selection in enumerate(dmf_selections, start=1):
        group = selection.group
        oxygen_label = f"pb_o_{index}"
        carbon_label = f"pb_c_{index}"
        nitrogen_label = f"pb_n_{index}"
        if selection.include_oxygen_path:
            shells.append(
                _shell(
                    label=oxygen_label,
                    absorber_element=absorber_element,
                    scatterer_element=oxygen_element,
                    distance=group.pb_o_distance,
                    multiplicity=group.multiplicity,
                    std_distance=group.pb_o_std_distance,
                    min_distance=group.pb_o_min_distance,
                    max_distance=group.pb_o_max_distance,
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
        if selection.include_carbon_path:
            shells.append(
                _shell(
                    label=carbon_label,
                    absorber_element=absorber_element,
                    scatterer_element=carbon_element,
                    distance=group.pb_c_distance,
                    multiplicity=group.multiplicity,
                    std_distance=group.pb_c_std_distance,
                    min_distance=group.pb_c_min_distance,
                    max_distance=group.pb_c_max_distance,
                )
            )
            hints.append(
                ArtemisPathParameterHint(
                    shell_label=carbon_label,
                    s02=f"snot*cn_c{index}",
                    e0="enot",
                    delr=f"delr_c{index}",
                    sigma2=f"sig2_c{index}",
                    reff_angstrom=group.pb_c_distance,
                    multiplicity=group.multiplicity,
                )
            )
        if selection.include_nitrogen_path:
            shells.append(
                _shell(
                    label=nitrogen_label,
                    absorber_element=absorber_element,
                    scatterer_element=nitrogen_element,
                    distance=group.pb_n_distance,
                    multiplicity=group.multiplicity,
                    std_distance=group.pb_n_std_distance,
                    min_distance=group.pb_n_min_distance,
                    max_distance=group.pb_n_max_distance,
                )
            )
            hints.append(
                ArtemisPathParameterHint(
                    shell_label=nitrogen_label,
                    s02=f"snot*cn_n{index}",
                    e0="enot",
                    delr=f"delr_n{index}",
                    sigma2=f"sig2_n{index}",
                    reff_angstrom=group.pb_n_distance,
                    multiplicity=group.multiplicity,
                ),
            )


def _shell(
    *,
    label: str,
    absorber_element: str,
    scatterer_element: str,
    distance: float,
    multiplicity: float = 1.0,
    std_distance: float = 0.0,
    min_distance: float | None = None,
    max_distance: float | None = None,
) -> ArtemisGDSShell:
    return ArtemisGDSShell(
        label=label,
        absorber_element=absorber_element,
        scatterer_element=scatterer_element,
        multiplicity=float(multiplicity),
        mean_distance_angstrom=distance,
        std_distance_angstrom=std_distance,
        min_distance_angstrom=(
            distance if min_distance is None else min_distance
        ),
        max_distance_angstrom=(
            distance if max_distance is None else max_distance
        ),
    )


def _collect_dmf_triplets(
    *,
    positions: np.ndarray,
    elements: tuple[str, ...],
    absorber_indices: tuple[int, ...],
    settings: PbDMFGDSBuildSettings,
) -> tuple[_DMFTriplet, ...]:
    dmf_triplets: list[_DMFTriplet] = []
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
        dmf_triplets.extend(
            _pair_oxygens_to_carbonyl_carbons_and_nitrogens(
                positions=positions,
                elements=elements,
                absorber_index=absorber_index,
                oxygen_indices=oxygen_indices,
                carbon_element=_normalize_element(settings.carbon_element),
                nitrogen_element=_normalize_element(settings.nitrogen_element),
                max_oc_distance=settings.max_oc_distance_angstrom,
                max_cn_distance=settings.max_cn_distance_angstrom,
            )
        )
    if not dmf_triplets:
        raise ValueError(
            "No coordinated DMF oxygens were found within the Pb-O distance "
            "cutoff for any absorber."
        )
    return tuple(dmf_triplets)


def _group_dmf_triplets(
    triplets: tuple[_DMFTriplet, ...],
    *,
    absorber_count: int,
    tolerance: float,
) -> tuple[_DMFGroup, ...]:
    if absorber_count == 1:
        return tuple(
            _DMFGroup(triplets=(triplet,), absorber_count=absorber_count)
            for triplet in sorted(
                triplets,
                key=lambda triplet: (
                    triplet.pb_o_distance,
                    triplet.pb_c_distance,
                    triplet.pb_n_distance,
                ),
            )
        )
    return tuple(
        _DMFGroup(triplets=tuple(cluster), absorber_count=absorber_count)
        for cluster in _cluster_records(
            triplets,
            feature_getter=lambda triplet: (
                triplet.pb_o_distance,
                triplet.pb_c_distance,
                triplet.pb_n_distance,
            ),
            tolerance=tolerance,
        )
    )


def _dmf_group_path_selections(
    groups: tuple[_DMFGroup, ...],
    *,
    included_path_pairs: tuple[tuple[int, int], ...] | None,
) -> tuple[_DMFPathSelection, ...]:
    selections: list[_DMFPathSelection] = []
    for group in groups:
        include_oxygen = any(
            _path_pair_is_included(
                active_pairs=included_path_pairs,
                absorber_index=triplet.absorber_index,
                scatterer_index=triplet.oxygen_index,
            )
            for triplet in group.triplets
        )
        include_carbon = any(
            _path_pair_is_included(
                active_pairs=included_path_pairs,
                absorber_index=triplet.absorber_index,
                scatterer_index=triplet.carbon_index,
            )
            for triplet in group.triplets
        )
        include_nitrogen = any(
            _path_pair_is_included(
                active_pairs=included_path_pairs,
                absorber_index=triplet.absorber_index,
                scatterer_index=triplet.nitrogen_index,
            )
            for triplet in group.triplets
        )
        selection = _DMFPathSelection(
            group=group,
            include_oxygen_path=include_oxygen,
            include_carbon_path=include_carbon,
            include_nitrogen_path=include_nitrogen,
        )
        if selection.is_active:
            selections.append(selection)
    return tuple(selections)


def _pair_oxygens_to_carbonyl_carbons_and_nitrogens(
    *,
    positions: np.ndarray,
    elements: tuple[str, ...],
    absorber_index: int,
    oxygen_indices: tuple[int, ...],
    carbon_element: str,
    nitrogen_element: str,
    max_oc_distance: float,
    max_cn_distance: float,
) -> list[_DMFTriplet]:
    carbon_indices = [
        atom_index
        for atom_index, element in enumerate(elements)
        if element == carbon_element
    ]
    nitrogen_indices = [
        atom_index
        for atom_index, element in enumerate(elements)
        if element == nitrogen_element
    ]
    if not carbon_indices:
        raise ValueError(
            f"No carbon atoms with element {carbon_element!r} were found."
        )
    if not nitrogen_indices:
        raise ValueError(
            f"No nitrogen atoms with element {nitrogen_element!r} were found."
        )

    used_carbons: set[int] = set()
    used_nitrogens: set[int] = set()
    triplets: list[_DMFTriplet] = []
    for oxygen_index in oxygen_indices:
        carbon_candidates = [
            (
                _distance(positions, oxygen_index, carbon_index),
                carbon_index,
            )
            for carbon_index in carbon_indices
            if carbon_index not in used_carbons
        ]
        carbon_candidates.sort()
        if not carbon_candidates or carbon_candidates[0][0] > max_oc_distance:
            raise ValueError(
                "Could not find an unused DMF carbonyl carbon within "
                f"{_format_float(max_oc_distance)} angstrom of oxygen atom "
                f"{oxygen_index + 1}."
            )
        oc_distance, carbon_index = carbon_candidates[0]
        nitrogen_candidates = [
            (
                _distance(positions, carbon_index, nitrogen_index),
                nitrogen_index,
            )
            for nitrogen_index in nitrogen_indices
            if nitrogen_index not in used_nitrogens
        ]
        nitrogen_candidates.sort()
        if (
            not nitrogen_candidates
            or nitrogen_candidates[0][0] > max_cn_distance
        ):
            raise ValueError(
                "Could not find an unused DMF nitrogen within "
                f"{_format_float(max_cn_distance)} angstrom of carbon atom "
                f"{carbon_index + 1}."
            )
        cn_distance, nitrogen_index = nitrogen_candidates[0]
        used_carbons.add(carbon_index)
        used_nitrogens.add(nitrogen_index)
        triplets.append(
            _DMFTriplet(
                absorber_index=absorber_index,
                oxygen_index=oxygen_index,
                carbon_index=carbon_index,
                nitrogen_index=nitrogen_index,
                pb_o_distance=_distance(
                    positions,
                    absorber_index,
                    oxygen_index,
                ),
                pb_c_distance=_distance(
                    positions,
                    absorber_index,
                    carbon_index,
                ),
                pb_n_distance=_distance(
                    positions,
                    absorber_index,
                    nitrogen_index,
                ),
                oc_distance=oc_distance,
                cn_distance=cn_distance,
                pboc_angle_degrees=_angle_degrees(
                    positions[absorber_index],
                    positions[oxygen_index],
                    positions[carbon_index],
                ),
                ocn_angle_degrees=_angle_degrees(
                    positions[oxygen_index],
                    positions[carbon_index],
                    positions[nitrogen_index],
                ),
                pbocn_dihedral_degrees=_dihedral_degrees(
                    positions[absorber_index],
                    positions[oxygen_index],
                    positions[carbon_index],
                    positions[nitrogen_index],
                ),
            )
        )
    return triplets


def _carbon_delta_r_expression(index: int) -> str:
    return three_atom_chord_delta_r_expression(
        anchor_reff_name=f"reff_pbo_{index}",
        anchor_delta_r_name=f"delr_o{index}",
        fixed_distance_name="bl_oc",
        angle_mean_name="theta_pboc_rad",
        terminal_reff_name=f"reff_pbc_{index}",
    )


def _carbon_sigma2_expression(index: int) -> str:
    return three_atom_chord_sigma2_expression(
        anchor_reff_name=f"reff_pbo_{index}",
        anchor_delta_r_name=f"delr_o{index}",
        anchor_sigma2_name=f"sig2_o{index}",
        fixed_distance_name="bl_oc",
        angle_mean_name="theta_pboc_rad",
        angle_variance_name="sig2_theta_pboc",
        terminal_reff_name=f"reff_pbc_{index}",
        terminal_delta_r_name=f"delr_c{index}",
    )


def _nitrogen_delta_r_expression(index: int) -> str:
    return dihedral_chord_delta_r_expression(
        anchor_reff_name=f"reff_pbo_{index}",
        anchor_delta_r_name=f"delr_o{index}",
        fixed_bridge_hinge_distance_name="bl_oc",
        fixed_hinge_terminal_distance_name="bl_cn",
        angle_abc_mean_name="theta_pboc_rad",
        angle_bcd_mean_name="theta_ocn_rad",
        dihedral_mean_name="phi_pbocn_rad",
        terminal_reff_name=f"reff_pbn_{index}",
    )


def _nitrogen_sigma2_expression(index: int) -> str:
    return dihedral_chord_sigma2_expression(
        anchor_reff_name=f"reff_pbo_{index}",
        anchor_delta_r_name=f"delr_o{index}",
        anchor_sigma2_name=f"sig2_o{index}",
        fixed_bridge_hinge_distance_name="bl_oc",
        fixed_hinge_terminal_distance_name="bl_cn",
        angle_abc_mean_name="theta_pboc_rad",
        angle_abc_variance_name="sig2_theta_pboc",
        angle_bcd_mean_name="theta_ocn_rad",
        angle_bcd_variance_name="sig2_theta_ocn",
        dihedral_mean_name="phi_pbocn_rad",
        dihedral_variance_name="sig2_phi_pbocn",
        terminal_reff_name=f"reff_pbn_{index}",
        terminal_delta_r_name=f"delr_n{index}",
    )


def _resolve_absorbers(
    elements: tuple[str, ...],
    settings: PbDMFGDSBuildSettings,
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


def _dihedral_degrees(
    atom1: np.ndarray,
    atom2: np.ndarray,
    atom3: np.ndarray,
    atom4: np.ndarray,
) -> float:
    point1 = np.asarray(atom1, dtype=float)
    point2 = np.asarray(atom2, dtype=float)
    point3 = np.asarray(atom3, dtype=float)
    point4 = np.asarray(atom4, dtype=float)
    bond1 = point1 - point2
    bond2 = point3 - point2
    bond3 = point4 - point3
    bond2_norm = float(np.linalg.norm(bond2))
    if bond2_norm == 0.0:
        raise ValueError("Cannot compute a dihedral with a zero-length bond.")
    bond2_unit = bond2 / bond2_norm
    normal1 = bond1 - np.dot(bond1, bond2_unit) * bond2_unit
    normal2 = bond3 - np.dot(bond3, bond2_unit) * bond2_unit
    normal1_norm = float(np.linalg.norm(normal1))
    normal2_norm = float(np.linalg.norm(normal2))
    if normal1_norm == 0.0 or normal2_norm == 0.0:
        raise ValueError("Cannot compute a degenerate dihedral angle.")
    x_value = float(np.dot(normal1, normal2))
    y_value = float(np.dot(np.cross(bond2_unit, normal1), normal2))
    return float(np.degrees(np.arctan2(y_value, x_value)))


def _circular_mean_degrees(values: object) -> float:
    radians = tuple(math.radians(float(value)) for value in values)
    if not radians:
        raise ValueError(
            "Cannot compute a circular mean from an empty sequence."
        )
    sine_mean = sum(math.sin(value) for value in radians) / len(radians)
    cosine_mean = sum(math.cos(value) for value in radians) / len(radians)
    return math.degrees(math.atan2(sine_mean, cosine_mean))


def _normalize_element(element: object) -> str:
    text = str(element).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


__all__ = ["PbDMFGDSBuildSettings", "build_pb_dmf_gds_from_structure"]
