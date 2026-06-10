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


@dataclass(slots=True, frozen=True)
class CoordinationGroupSpec:
    """Total coordination number variable shared by related paths."""

    label: str
    initial_value: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    restraint_scale: float = 1000.0
    vary: bool = True


@dataclass(slots=True, frozen=True)
class IndependentPathSpec:
    """A path refined independently, such as a Pb-I first shell."""

    label: str
    absorber_atom_index: int
    scatterer_atom_index: int
    scatterer_element: str | None = None
    initial_delta_r: float = 0.0
    initial_sigma2: float = 0.003
    multiplicity: float = 1.0
    coordination_group: str | None = None
    coordination_fraction: float | None = None
    reference_multiplicity: float | None = None


@dataclass(slots=True, frozen=True)
class ThreeAtomConstraintSpec:
    """A-B-C geometry where A-B anchors terminal A-C paths."""

    label: str
    absorber_atom_index: int
    bridge_atom_index: int
    terminal_atom_index: int
    fixed_bridge_terminal_distance_angstrom: float
    angle_mean_degrees: float
    angle_sigma_degrees: float = 0.0
    anchor_delta_r_name: str | None = None
    anchor_sigma2_name: str | None = None
    include_multiple_scattering: bool = True
    terminal_multiplicity: float = 1.0
    multiple_scattering_multiplicity: float = 1.0
    coordination_group: str | None = None
    coordination_fraction: float | None = None
    anchor_reference_multiplicity: float | None = None
    terminal_reference_multiplicity: float | None = None
    multiple_scattering_reference_multiplicity: float | None = None


@dataclass(slots=True, frozen=True)
class FourAtomDihedralConstraintSpec:
    """A-B-C-D geometry where A-B anchors terminal A-D paths."""

    label: str
    absorber_atom_index: int
    bridge_atom_index: int
    hinge_atom_index: int
    terminal_atom_index: int
    fixed_bridge_hinge_distance_angstrom: float
    fixed_hinge_terminal_distance_angstrom: float
    angle_abc_mean_degrees: float
    angle_bcd_mean_degrees: float
    dihedral_mean_degrees: float
    angle_abc_sigma_degrees: float = 0.0
    angle_bcd_sigma_degrees: float = 0.0
    dihedral_sigma_degrees: float = 0.0
    anchor_delta_r_name: str | None = None
    anchor_sigma2_name: str | None = None
    terminal_multiplicity: float = 1.0
    coordination_group: str | None = None
    coordination_fraction: float | None = None
    anchor_reference_multiplicity: float | None = None
    terminal_reference_multiplicity: float | None = None


@dataclass(slots=True, frozen=True)
class GeometricGDSBuildSettings:
    initial_s02: float = 0.9
    initial_e0: float = 0.0
    default_initial_delta_r: float = 0.0
    default_initial_sigma2: float = 0.003


def build_geometric_constraint_gds(
    structure_path: str | Path,
    *,
    coordination_groups: tuple[CoordinationGroupSpec, ...] = (),
    independent_paths: tuple[IndependentPathSpec, ...] = (),
    three_atom_constraints: tuple[ThreeAtomConstraintSpec, ...] = (),
    four_atom_dihedral_constraints: tuple[
        FourAtomDihedralConstraintSpec,
        ...,
    ] = (),
    settings: GeometricGDSBuildSettings | None = None,
) -> ArtemisGDSDocument:
    active_settings = settings or GeometricGDSBuildSettings()
    path = Path(structure_path).expanduser().resolve()
    positions, elements = load_structure_file(path)
    normalized_elements = tuple(
        _normalize_element(element) for element in elements
    )

    parameters: list[ArtemisGDSParameter] = [
        ArtemisGDSParameter(
            "guess",
            "amp",
            _format_float(active_settings.initial_s02),
            "global S02 amplitude factor",
        ),
        ArtemisGDSParameter(
            "guess",
            "enot",
            _format_float(active_settings.initial_e0),
            "global edge-energy shift",
        ),
    ]
    shells: list[ArtemisGDSShell] = []
    hints: list[ArtemisPathParameterHint] = []
    defined_names = {"amp", "enot"}
    coordination_group_map = _append_coordination_groups(
        parameters,
        defined_names,
        coordination_groups,
    )

    for spec in independent_paths:
        _append_independent_path(
            parameters=parameters,
            shells=shells,
            hints=hints,
            defined_names=defined_names,
            positions=positions,
            elements=normalized_elements,
            spec=spec,
            coordination_groups=coordination_group_map,
        )

    for spec in three_atom_constraints:
        _append_three_atom_constraint(
            parameters=parameters,
            shells=shells,
            hints=hints,
            defined_names=defined_names,
            positions=positions,
            elements=normalized_elements,
            spec=spec,
            settings=active_settings,
            coordination_groups=coordination_group_map,
        )

    for spec in four_atom_dihedral_constraints:
        _append_four_atom_dihedral_constraint(
            parameters=parameters,
            shells=shells,
            hints=hints,
            defined_names=defined_names,
            positions=positions,
            elements=normalized_elements,
            spec=spec,
            settings=active_settings,
            coordination_groups=coordination_group_map,
        )

    return ArtemisGDSDocument(
        source_structure=str(path),
        settings=ArtemisGDSBuildSettings(
            initial_s02=active_settings.initial_s02,
            initial_e0=active_settings.initial_e0,
            initial_delta_r=active_settings.default_initial_delta_r,
            initial_sigma2=active_settings.default_initial_sigma2,
        ),
        shells=tuple(shells),
        parameters=tuple(parameters),
        path_hints=tuple(hints),
        overview_notes=(
            "Template: explicit geometric-constraint GDS",
            (
                "Independent paths and linked three-atom or four-atom "
                "constraints were emitted from the supplied specification."
            ),
        ),
    )


def three_atom_chord_distance(
    anchor_distance: float,
    fixed_bridge_terminal_distance: float,
    angle_radians: float,
) -> float:
    return math.sqrt(
        anchor_distance * anchor_distance
        + fixed_bridge_terminal_distance * fixed_bridge_terminal_distance
        - 2.0
        * anchor_distance
        * fixed_bridge_terminal_distance
        * math.cos(angle_radians)
    )


def dihedral_chord_distance(
    *,
    anchor_distance: float,
    fixed_bridge_hinge_distance: float,
    fixed_hinge_terminal_distance: float,
    angle_abc_radians: float,
    angle_bcd_radians: float,
    dihedral_radians: float,
) -> float:
    a = float(anchor_distance)
    b = float(fixed_bridge_hinge_distance)
    c = float(fixed_hinge_terminal_distance)
    mu = float(angle_abc_radians)
    alpha = float(angle_bcd_radians)
    phi = float(dihedral_radians)
    squared_distance = (
        a * a
        + b * b
        + c * c
        - 2.0 * a * b * math.cos(mu)
        - 2.0 * b * c * math.cos(alpha)
        + 2.0
        * a
        * c
        * (
            math.cos(mu) * math.cos(alpha)
            + math.sin(mu) * math.sin(alpha) * math.cos(phi)
        )
    )
    return math.sqrt(max(squared_distance, 0.0))


def three_atom_chord_delta_r_expression(
    *,
    anchor_reff_name: str,
    anchor_delta_r_name: str,
    fixed_distance_name: str,
    angle_mean_name: str,
    terminal_reff_name: str,
) -> str:
    anchor_distance = f"({anchor_reff_name} + {anchor_delta_r_name})"
    return (
        f"sqrt({anchor_distance}*{anchor_distance} + "
        f"{fixed_distance_name}*{fixed_distance_name} - "
        f"2*{anchor_distance}*{fixed_distance_name}*"
        f"cos({angle_mean_name})) - {terminal_reff_name}"
    )


def dihedral_chord_delta_r_expression(
    *,
    anchor_reff_name: str,
    anchor_delta_r_name: str,
    fixed_bridge_hinge_distance_name: str,
    fixed_hinge_terminal_distance_name: str,
    angle_abc_mean_name: str,
    angle_bcd_mean_name: str,
    dihedral_mean_name: str,
    terminal_reff_name: str,
) -> str:
    anchor_distance = f"({anchor_reff_name} + {anchor_delta_r_name})"
    bridge_hinge = fixed_bridge_hinge_distance_name
    hinge_terminal = fixed_hinge_terminal_distance_name
    mu = angle_abc_mean_name
    alpha = angle_bcd_mean_name
    phi = dihedral_mean_name
    return (
        f"sqrt({anchor_distance}*{anchor_distance} + "
        f"{bridge_hinge}*{bridge_hinge} + "
        f"{hinge_terminal}*{hinge_terminal} - "
        f"2*{anchor_distance}*{bridge_hinge}*cos({mu}) - "
        f"2*{bridge_hinge}*{hinge_terminal}*cos({alpha}) + "
        f"2*{anchor_distance}*{hinge_terminal}*"
        f"(cos({mu})*cos({alpha}) + "
        f"sin({mu})*sin({alpha})*cos({phi}))) - "
        f"{terminal_reff_name}"
    )


def three_atom_chord_sigma2_expression(
    *,
    anchor_reff_name: str,
    anchor_delta_r_name: str,
    anchor_sigma2_name: str,
    fixed_distance_name: str,
    angle_mean_name: str,
    angle_variance_name: str,
    terminal_reff_name: str,
    terminal_delta_r_name: str,
) -> str:
    anchor_distance = f"({anchor_reff_name} + {anchor_delta_r_name})"
    terminal_distance = f"({terminal_reff_name} + {terminal_delta_r_name})"
    radial_jacobian = (
        f"(({anchor_distance} - {fixed_distance_name}*"
        f"cos({angle_mean_name}))/{terminal_distance})"
    )
    angle_jacobian = (
        f"(({anchor_distance}*{fixed_distance_name}*"
        f"sin({angle_mean_name}))/{terminal_distance})"
    )
    return (
        f"{radial_jacobian}*{radial_jacobian}*{anchor_sigma2_name} + "
        f"{angle_jacobian}*{angle_jacobian}*{angle_variance_name}"
    )


def dihedral_chord_sigma2_expression(
    *,
    anchor_reff_name: str,
    anchor_delta_r_name: str,
    anchor_sigma2_name: str,
    fixed_bridge_hinge_distance_name: str,
    fixed_hinge_terminal_distance_name: str,
    angle_abc_mean_name: str,
    angle_abc_variance_name: str,
    angle_bcd_mean_name: str,
    angle_bcd_variance_name: str,
    dihedral_mean_name: str,
    dihedral_variance_name: str,
    terminal_reff_name: str,
    terminal_delta_r_name: str,
) -> str:
    a = f"({anchor_reff_name} + {anchor_delta_r_name})"
    b = fixed_bridge_hinge_distance_name
    c = fixed_hinge_terminal_distance_name
    mu = angle_abc_mean_name
    alpha = angle_bcd_mean_name
    phi = dihedral_mean_name
    terminal_distance = f"({terminal_reff_name} + {terminal_delta_r_name})"
    coupling = (
        f"(cos({mu})*cos({alpha}) + " f"sin({mu})*sin({alpha})*cos({phi}))"
    )
    d_anchor = f"(({a} - {b}*cos({mu}) + {c}*{coupling})/{terminal_distance})"
    d_mu = (
        f"(({a}*{b}*sin({mu}) + {a}*{c}*"
        f"(-sin({mu})*cos({alpha}) + "
        f"cos({mu})*sin({alpha})*cos({phi})))/{terminal_distance})"
    )
    d_alpha = (
        f"(({b}*{c}*sin({alpha}) + {a}*{c}*"
        f"(-cos({mu})*sin({alpha}) + "
        f"sin({mu})*cos({alpha})*cos({phi})))/{terminal_distance})"
    )
    d_phi = (
        f"(-({a}*{c}*sin({mu})*sin({alpha})*sin({phi}))/"
        f"{terminal_distance})"
    )
    return (
        f"{d_anchor}*{d_anchor}*{anchor_sigma2_name} + "
        f"{d_mu}*{d_mu}*{angle_abc_variance_name} + "
        f"{d_alpha}*{d_alpha}*{angle_bcd_variance_name} + "
        f"{d_phi}*{d_phi}*{dihedral_variance_name}"
    )


def angle_variance_from_degrees(angle_sigma_degrees: float) -> float:
    return math.radians(float(angle_sigma_degrees)) ** 2


def _append_independent_path(
    *,
    parameters: list[ArtemisGDSParameter],
    shells: list[ArtemisGDSShell],
    hints: list[ArtemisPathParameterHint],
    defined_names: set[str],
    positions: np.ndarray,
    elements: tuple[str, ...],
    spec: IndependentPathSpec,
    coordination_groups: dict[str, CoordinationGroupSpec],
) -> None:
    label = _safe_label(spec.label)
    absorber_index = _zero_based_atom_index(
        spec.absorber_atom_index,
        len(elements),
    )
    scatterer_index = _zero_based_atom_index(
        spec.scatterer_atom_index,
        len(elements),
    )
    reff = _distance(positions, absorber_index, scatterer_index)
    absorber_element = elements[absorber_index]
    scatterer_element = (
        _normalize_element(spec.scatterer_element)
        if spec.scatterer_element
        else elements[scatterer_index]
    )
    reff_name = f"reff_{label}"
    dr_name = f"dr_{label}"
    sigma2_name = f"ss_{label}"
    r_name = f"r_{label}"
    coordination_group_label = _path_coordination_group_label(
        requested_group=spec.coordination_group,
        default_absorber_element=absorber_element,
        default_scatterer_element=scatterer_element,
        coordination_groups=coordination_groups,
    )
    s02_expression = _append_path_coordination_parameters(
        parameters=parameters,
        defined_names=defined_names,
        path_label=label,
        coordination_group_label=coordination_group_label,
        coordination_fraction=spec.coordination_fraction,
        reference_multiplicity=(
            spec.reference_multiplicity
            if spec.reference_multiplicity is not None
            else spec.multiplicity
        ),
    )
    _append_unique_parameters(
        parameters,
        defined_names,
        (
            ArtemisGDSParameter(
                "set",
                reff_name,
                _format_float(reff),
                "representative-path effective distance",
            ),
            ArtemisGDSParameter(
                "guess",
                dr_name,
                _format_float(spec.initial_delta_r),
                "independent delta-R",
            ),
            ArtemisGDSParameter(
                "guess",
                sigma2_name,
                _format_float(spec.initial_sigma2),
                "independent sigma2",
            ),
            ArtemisGDSParameter(
                "def",
                r_name,
                f"{reff_name} + {dr_name}",
                "reported refined distance",
            ),
        ),
    )
    shells.append(
        ArtemisGDSShell(
            label=label,
            absorber_element=absorber_element,
            scatterer_element=scatterer_element,
            multiplicity=float(spec.multiplicity),
            mean_distance_angstrom=reff,
            std_distance_angstrom=0.0,
            min_distance_angstrom=reff,
            max_distance_angstrom=reff,
        )
    )
    hints.append(
        ArtemisPathParameterHint(
            shell_label=label,
            s02=s02_expression,
            e0="enot",
            delr=dr_name,
            sigma2=sigma2_name,
            reff_angstrom=reff,
            multiplicity=float(spec.multiplicity),
        )
    )


def _append_three_atom_constraint(
    *,
    parameters: list[ArtemisGDSParameter],
    shells: list[ArtemisGDSShell],
    hints: list[ArtemisPathParameterHint],
    defined_names: set[str],
    positions: np.ndarray,
    elements: tuple[str, ...],
    spec: ThreeAtomConstraintSpec,
    settings: GeometricGDSBuildSettings,
    coordination_groups: dict[str, CoordinationGroupSpec],
) -> None:
    label = _safe_label(spec.label)
    absorber_index = _zero_based_atom_index(
        spec.absorber_atom_index,
        len(elements),
    )
    bridge_index = _zero_based_atom_index(
        spec.bridge_atom_index,
        len(elements),
    )
    terminal_index = _zero_based_atom_index(
        spec.terminal_atom_index,
        len(elements),
    )

    absorber_element = elements[absorber_index]
    bridge_element = elements[bridge_index]
    terminal_element = elements[terminal_index]
    absorber_token = _safe_label(absorber_element)
    bridge_token = _safe_label(bridge_element)
    terminal_token = _safe_label(terminal_element)
    anchor_label = f"{absorber_token}_{bridge_token}_{label}"
    terminal_label = f"{absorber_token}_{terminal_token}_{label}"
    ms_label = f"ms_{anchor_label}_{_safe_label(terminal_element)}"

    anchor_reff = _distance(positions, absorber_index, bridge_index)
    terminal_reff = _distance(positions, absorber_index, terminal_index)
    fixed_distance = float(spec.fixed_bridge_terminal_distance_angstrom)
    angle_mean_radians = math.radians(float(spec.angle_mean_degrees))
    angle_variance = angle_variance_from_degrees(spec.angle_sigma_degrees)
    ms_reff = 0.5 * (anchor_reff + fixed_distance + terminal_reff)

    anchor_reff_name = f"reff_{anchor_label}"
    anchor_dr_name = spec.anchor_delta_r_name or f"dr_{anchor_label}"
    anchor_sigma2_name = spec.anchor_sigma2_name or f"ss_{anchor_label}"
    anchor_r_name = f"r_{anchor_label}"
    fixed_distance_name = f"b_{bridge_token}_{terminal_token}_{label}"
    angle_mean_name = f"mu_{anchor_label}_{_safe_label(terminal_element)}"
    angle_variance_name = f"sig2_theta_{anchor_label}_{terminal_token}"
    terminal_reff_name = f"reff_{terminal_label}"
    terminal_dr_name = f"dr_{terminal_label}"
    terminal_sigma2_name = f"ss_{terminal_label}"
    terminal_r_name = f"r_{terminal_label}"
    coordination_group_label = _path_coordination_group_label(
        requested_group=spec.coordination_group,
        default_absorber_element=absorber_element,
        default_scatterer_element=bridge_element,
        coordination_groups=coordination_groups,
    )

    anchor_parameters = [
        ArtemisGDSParameter(
            "set",
            anchor_reff_name,
            _format_float(anchor_reff),
            "anchor path effective distance",
        ),
        ArtemisGDSParameter(
            "def",
            anchor_r_name,
            f"{anchor_reff_name} + {anchor_dr_name}",
            "reported anchor distance",
        ),
    ]
    if spec.anchor_delta_r_name is None:
        anchor_parameters.insert(
            1,
            ArtemisGDSParameter(
                "guess",
                anchor_dr_name,
                _format_float(settings.default_initial_delta_r),
                "anchor delta-R",
            ),
        )
    if spec.anchor_sigma2_name is None:
        anchor_parameters.append(
            ArtemisGDSParameter(
                "guess",
                anchor_sigma2_name,
                _format_float(settings.default_initial_sigma2),
                "anchor sigma2",
            )
        )
    _append_unique_parameters(
        parameters,
        defined_names,
        tuple(anchor_parameters),
    )

    terminal_delta_expression = three_atom_chord_delta_r_expression(
        anchor_reff_name=anchor_reff_name,
        anchor_delta_r_name=anchor_dr_name,
        fixed_distance_name=fixed_distance_name,
        angle_mean_name=angle_mean_name,
        terminal_reff_name=terminal_reff_name,
    )
    terminal_sigma2_expression = three_atom_chord_sigma2_expression(
        anchor_reff_name=anchor_reff_name,
        anchor_delta_r_name=anchor_dr_name,
        anchor_sigma2_name=anchor_sigma2_name,
        fixed_distance_name=fixed_distance_name,
        angle_mean_name=angle_mean_name,
        angle_variance_name=angle_variance_name,
        terminal_reff_name=terminal_reff_name,
        terminal_delta_r_name=terminal_dr_name,
    )

    _append_unique_parameters(
        parameters,
        defined_names,
        (
            ArtemisGDSParameter(
                "set",
                fixed_distance_name,
                _format_float(fixed_distance),
                "fixed bridge-terminal bond length",
            ),
            ArtemisGDSParameter(
                "set",
                angle_mean_name,
                _format_float(angle_mean_radians),
                "mean bridge angle in radians",
            ),
            ArtemisGDSParameter(
                "set",
                angle_variance_name,
                _format_float(angle_variance),
                "angle variance in radians squared",
            ),
            ArtemisGDSParameter(
                "set",
                terminal_reff_name,
                _format_float(terminal_reff),
                "terminal path effective distance",
            ),
            ArtemisGDSParameter(
                "def",
                terminal_dr_name,
                terminal_delta_expression,
                "geometry-linked terminal delta-R",
            ),
            ArtemisGDSParameter(
                "def",
                terminal_r_name,
                f"{terminal_reff_name} + {terminal_dr_name}",
                "reported terminal distance",
            ),
            ArtemisGDSParameter(
                "def",
                terminal_sigma2_name,
                terminal_sigma2_expression,
                "geometry-linked terminal sigma2",
            ),
        ),
    )
    _append_three_atom_ms_parameters(
        parameters=parameters,
        defined_names=defined_names,
        spec=spec,
        ms_label=ms_label,
        ms_reff=ms_reff,
        anchor_r_name=anchor_r_name,
        fixed_distance_name=fixed_distance_name,
        terminal_r_name=terminal_r_name,
        anchor_sigma2_name=anchor_sigma2_name,
        terminal_sigma2_name=terminal_sigma2_name,
        hints=hints,
        multiplicity=spec.multiple_scattering_multiplicity,
        coordination_group_label=coordination_group_label,
        coordination_fraction=spec.coordination_fraction,
        reference_multiplicity=(
            spec.multiple_scattering_reference_multiplicity
            if spec.multiple_scattering_reference_multiplicity is not None
            else spec.multiple_scattering_multiplicity
        ),
    )

    anchor_s02 = _append_path_coordination_parameters(
        parameters=parameters,
        defined_names=defined_names,
        path_label=anchor_label,
        coordination_group_label=coordination_group_label,
        coordination_fraction=spec.coordination_fraction,
        reference_multiplicity=(
            spec.anchor_reference_multiplicity
            if spec.anchor_reference_multiplicity is not None
            else 1.0
        ),
    )
    terminal_s02 = _append_path_coordination_parameters(
        parameters=parameters,
        defined_names=defined_names,
        path_label=terminal_label,
        coordination_group_label=coordination_group_label,
        coordination_fraction=spec.coordination_fraction,
        reference_multiplicity=(
            spec.terminal_reference_multiplicity
            if spec.terminal_reference_multiplicity is not None
            else spec.terminal_multiplicity
        ),
    )

    shells.extend(
        (
            ArtemisGDSShell(
                label=anchor_label,
                absorber_element=absorber_element,
                scatterer_element=bridge_element,
                multiplicity=1.0,
                mean_distance_angstrom=anchor_reff,
                std_distance_angstrom=0.0,
                min_distance_angstrom=anchor_reff,
                max_distance_angstrom=anchor_reff,
            ),
            ArtemisGDSShell(
                label=terminal_label,
                absorber_element=absorber_element,
                scatterer_element=terminal_element,
                multiplicity=float(spec.terminal_multiplicity),
                mean_distance_angstrom=terminal_reff,
                std_distance_angstrom=0.0,
                min_distance_angstrom=terminal_reff,
                max_distance_angstrom=terminal_reff,
            ),
        )
    )
    hints.extend(
        (
            ArtemisPathParameterHint(
                shell_label=anchor_label,
                s02=anchor_s02,
                e0="enot",
                delr=anchor_dr_name,
                sigma2=anchor_sigma2_name,
                reff_angstrom=anchor_reff,
                multiplicity=1.0,
            ),
            ArtemisPathParameterHint(
                shell_label=terminal_label,
                s02=terminal_s02,
                e0="enot",
                delr=terminal_dr_name,
                sigma2=terminal_sigma2_name,
                reff_angstrom=terminal_reff,
                multiplicity=float(spec.terminal_multiplicity),
            ),
        )
    )


def _append_four_atom_dihedral_constraint(
    *,
    parameters: list[ArtemisGDSParameter],
    shells: list[ArtemisGDSShell],
    hints: list[ArtemisPathParameterHint],
    defined_names: set[str],
    positions: np.ndarray,
    elements: tuple[str, ...],
    spec: FourAtomDihedralConstraintSpec,
    settings: GeometricGDSBuildSettings,
    coordination_groups: dict[str, CoordinationGroupSpec],
) -> None:
    label = _safe_label(spec.label)
    absorber_index = _zero_based_atom_index(
        spec.absorber_atom_index,
        len(elements),
    )
    bridge_index = _zero_based_atom_index(
        spec.bridge_atom_index,
        len(elements),
    )
    hinge_index = _zero_based_atom_index(
        spec.hinge_atom_index,
        len(elements),
    )
    terminal_index = _zero_based_atom_index(
        spec.terminal_atom_index,
        len(elements),
    )

    absorber_element = elements[absorber_index]
    bridge_element = elements[bridge_index]
    hinge_element = elements[hinge_index]
    terminal_element = elements[terminal_index]
    absorber_token = _safe_label(absorber_element)
    bridge_token = _safe_label(bridge_element)
    hinge_token = _safe_label(hinge_element)
    terminal_token = _safe_label(terminal_element)
    anchor_label = f"{absorber_token}_{bridge_token}_{label}"
    terminal_label = f"{absorber_token}_{terminal_token}_{label}"

    anchor_reff = _distance(positions, absorber_index, bridge_index)
    terminal_reff = _distance(positions, absorber_index, terminal_index)
    fixed_bridge_hinge = float(spec.fixed_bridge_hinge_distance_angstrom)
    fixed_hinge_terminal = float(spec.fixed_hinge_terminal_distance_angstrom)
    angle_abc_mean_radians = math.radians(float(spec.angle_abc_mean_degrees))
    angle_bcd_mean_radians = math.radians(float(spec.angle_bcd_mean_degrees))
    dihedral_mean_radians = math.radians(float(spec.dihedral_mean_degrees))
    angle_abc_variance = angle_variance_from_degrees(
        spec.angle_abc_sigma_degrees
    )
    angle_bcd_variance = angle_variance_from_degrees(
        spec.angle_bcd_sigma_degrees
    )
    dihedral_variance = angle_variance_from_degrees(
        spec.dihedral_sigma_degrees
    )

    anchor_reff_name = f"reff_{anchor_label}"
    anchor_dr_name = spec.anchor_delta_r_name or f"dr_{anchor_label}"
    anchor_sigma2_name = spec.anchor_sigma2_name or f"ss_{anchor_label}"
    anchor_r_name = f"r_{anchor_label}"
    bridge_hinge_name = f"b_{bridge_token}_{hinge_token}_{label}"
    hinge_terminal_name = f"b_{hinge_token}_{terminal_token}_{label}"
    angle_abc_name = f"mu_{anchor_label}_{hinge_token}"
    angle_abc_variance_name = f"sig2_mu_{anchor_label}_{hinge_token}"
    angle_bcd_name = (
        f"alpha_{bridge_token}_{hinge_token}_{label}_{terminal_token}"
    )
    angle_bcd_variance_name = (
        f"sig2_alpha_{bridge_token}_{hinge_token}_{label}_{terminal_token}"
    )
    dihedral_name = (
        f"phi_{absorber_token}_{bridge_token}_{hinge_token}_"
        f"{terminal_token}_{label}"
    )
    dihedral_variance_name = (
        f"sig2_phi_{absorber_token}_{bridge_token}_{hinge_token}_"
        f"{terminal_token}_{label}"
    )
    terminal_reff_name = f"reff_{terminal_label}"
    terminal_dr_name = f"dr_{terminal_label}"
    terminal_sigma2_name = f"ss_{terminal_label}"
    terminal_r_name = f"r_{terminal_label}"
    coordination_group_label = _path_coordination_group_label(
        requested_group=spec.coordination_group,
        default_absorber_element=absorber_element,
        default_scatterer_element=bridge_element,
        coordination_groups=coordination_groups,
    )

    anchor_parameters = [
        ArtemisGDSParameter(
            "set",
            anchor_reff_name,
            _format_float(anchor_reff),
            "anchor path effective distance",
        ),
        ArtemisGDSParameter(
            "def",
            anchor_r_name,
            f"{anchor_reff_name} + {anchor_dr_name}",
            "reported anchor distance",
        ),
    ]
    if spec.anchor_delta_r_name is None:
        anchor_parameters.insert(
            1,
            ArtemisGDSParameter(
                "guess",
                anchor_dr_name,
                _format_float(settings.default_initial_delta_r),
                "anchor delta-R",
            ),
        )
    if spec.anchor_sigma2_name is None:
        anchor_parameters.append(
            ArtemisGDSParameter(
                "guess",
                anchor_sigma2_name,
                _format_float(settings.default_initial_sigma2),
                "anchor sigma2",
            )
        )
    _append_unique_parameters(
        parameters,
        defined_names,
        tuple(anchor_parameters),
    )

    terminal_delta_expression = dihedral_chord_delta_r_expression(
        anchor_reff_name=anchor_reff_name,
        anchor_delta_r_name=anchor_dr_name,
        fixed_bridge_hinge_distance_name=bridge_hinge_name,
        fixed_hinge_terminal_distance_name=hinge_terminal_name,
        angle_abc_mean_name=angle_abc_name,
        angle_bcd_mean_name=angle_bcd_name,
        dihedral_mean_name=dihedral_name,
        terminal_reff_name=terminal_reff_name,
    )
    terminal_sigma2_expression = dihedral_chord_sigma2_expression(
        anchor_reff_name=anchor_reff_name,
        anchor_delta_r_name=anchor_dr_name,
        anchor_sigma2_name=anchor_sigma2_name,
        fixed_bridge_hinge_distance_name=bridge_hinge_name,
        fixed_hinge_terminal_distance_name=hinge_terminal_name,
        angle_abc_mean_name=angle_abc_name,
        angle_abc_variance_name=angle_abc_variance_name,
        angle_bcd_mean_name=angle_bcd_name,
        angle_bcd_variance_name=angle_bcd_variance_name,
        dihedral_mean_name=dihedral_name,
        dihedral_variance_name=dihedral_variance_name,
        terminal_reff_name=terminal_reff_name,
        terminal_delta_r_name=terminal_dr_name,
    )

    _append_unique_parameters(
        parameters,
        defined_names,
        (
            ArtemisGDSParameter(
                "set",
                bridge_hinge_name,
                _format_float(fixed_bridge_hinge),
                "fixed bridge-hinge bond length",
            ),
            ArtemisGDSParameter(
                "set",
                hinge_terminal_name,
                _format_float(fixed_hinge_terminal),
                "fixed hinge-terminal bond length",
            ),
            ArtemisGDSParameter(
                "set",
                angle_abc_name,
                _format_float(angle_abc_mean_radians),
                "mean A-B-C angle in radians",
            ),
            ArtemisGDSParameter(
                "set",
                angle_abc_variance_name,
                _format_float(angle_abc_variance),
                "A-B-C angle variance in radians squared",
            ),
            ArtemisGDSParameter(
                "set",
                angle_bcd_name,
                _format_float(angle_bcd_mean_radians),
                "mean B-C-D angle in radians",
            ),
            ArtemisGDSParameter(
                "set",
                angle_bcd_variance_name,
                _format_float(angle_bcd_variance),
                "B-C-D angle variance in radians squared",
            ),
            ArtemisGDSParameter(
                "set",
                dihedral_name,
                _format_float(dihedral_mean_radians),
                "mean A-B-C-D dihedral in radians",
            ),
            ArtemisGDSParameter(
                "set",
                dihedral_variance_name,
                _format_float(dihedral_variance),
                "dihedral variance in radians squared",
            ),
            ArtemisGDSParameter(
                "set",
                terminal_reff_name,
                _format_float(terminal_reff),
                "terminal path effective distance",
            ),
            ArtemisGDSParameter(
                "def",
                terminal_dr_name,
                terminal_delta_expression,
                "dihedral-linked terminal delta-R",
            ),
            ArtemisGDSParameter(
                "def",
                terminal_r_name,
                f"{terminal_reff_name} + {terminal_dr_name}",
                "reported terminal distance",
            ),
            ArtemisGDSParameter(
                "def",
                terminal_sigma2_name,
                terminal_sigma2_expression,
                "dihedral-linked terminal sigma2",
            ),
        ),
    )

    anchor_s02 = _append_path_coordination_parameters(
        parameters=parameters,
        defined_names=defined_names,
        path_label=anchor_label,
        coordination_group_label=coordination_group_label,
        coordination_fraction=spec.coordination_fraction,
        reference_multiplicity=(
            spec.anchor_reference_multiplicity
            if spec.anchor_reference_multiplicity is not None
            else 1.0
        ),
    )
    terminal_s02 = _append_path_coordination_parameters(
        parameters=parameters,
        defined_names=defined_names,
        path_label=terminal_label,
        coordination_group_label=coordination_group_label,
        coordination_fraction=spec.coordination_fraction,
        reference_multiplicity=(
            spec.terminal_reference_multiplicity
            if spec.terminal_reference_multiplicity is not None
            else spec.terminal_multiplicity
        ),
    )

    shells.extend(
        (
            ArtemisGDSShell(
                label=anchor_label,
                absorber_element=absorber_element,
                scatterer_element=bridge_element,
                multiplicity=1.0,
                mean_distance_angstrom=anchor_reff,
                std_distance_angstrom=0.0,
                min_distance_angstrom=anchor_reff,
                max_distance_angstrom=anchor_reff,
            ),
            ArtemisGDSShell(
                label=terminal_label,
                absorber_element=absorber_element,
                scatterer_element=terminal_element,
                multiplicity=float(spec.terminal_multiplicity),
                mean_distance_angstrom=terminal_reff,
                std_distance_angstrom=0.0,
                min_distance_angstrom=terminal_reff,
                max_distance_angstrom=terminal_reff,
            ),
        )
    )
    hints.extend(
        (
            ArtemisPathParameterHint(
                shell_label=anchor_label,
                s02=anchor_s02,
                e0="enot",
                delr=anchor_dr_name,
                sigma2=anchor_sigma2_name,
                reff_angstrom=anchor_reff,
                multiplicity=1.0,
            ),
            ArtemisPathParameterHint(
                shell_label=terminal_label,
                s02=terminal_s02,
                e0="enot",
                delr=terminal_dr_name,
                sigma2=terminal_sigma2_name,
                reff_angstrom=terminal_reff,
                multiplicity=float(spec.terminal_multiplicity),
            ),
        )
    )


def _append_three_atom_ms_parameters(
    *,
    parameters: list[ArtemisGDSParameter],
    defined_names: set[str],
    spec: ThreeAtomConstraintSpec,
    ms_label: str,
    ms_reff: float,
    anchor_r_name: str,
    fixed_distance_name: str,
    terminal_r_name: str,
    anchor_sigma2_name: str,
    terminal_sigma2_name: str,
    hints: list[ArtemisPathParameterHint],
    multiplicity: float,
    coordination_group_label: str | None,
    coordination_fraction: float | None,
    reference_multiplicity: float,
) -> None:
    if not spec.include_multiple_scattering:
        return
    reff_ms_name = f"reff_{ms_label}"
    dr_ms_name = f"dr_{ms_label}"
    sigma2_ms_name = f"ss_{ms_label}"
    _append_unique_parameters(
        parameters,
        defined_names,
        (
            ArtemisGDSParameter(
                "set",
                reff_ms_name,
                _format_float(ms_reff),
                "three-atom half-path effective distance",
            ),
            ArtemisGDSParameter(
                "def",
                dr_ms_name,
                (
                    f"0.5*({anchor_r_name} + {fixed_distance_name} + "
                    f"{terminal_r_name}) - {reff_ms_name}"
                ),
                "geometry-linked multiple-scattering delta-R",
            ),
            ArtemisGDSParameter(
                "def",
                sigma2_ms_name,
                f"0.25*({anchor_sigma2_name} + {terminal_sigma2_name})",
                "diagonal MSRD approximation",
            ),
        ),
    )
    s02_expression = _append_path_coordination_parameters(
        parameters=parameters,
        defined_names=defined_names,
        path_label=ms_label,
        coordination_group_label=coordination_group_label,
        coordination_fraction=coordination_fraction,
        reference_multiplicity=reference_multiplicity,
    )
    hints.append(
        ArtemisPathParameterHint(
            shell_label=ms_label,
            s02=s02_expression,
            e0="enot",
            delr=dr_ms_name,
            sigma2=sigma2_ms_name,
            reff_angstrom=ms_reff,
            multiplicity=float(multiplicity),
        )
    )


def _append_coordination_groups(
    parameters: list[ArtemisGDSParameter],
    defined_names: set[str],
    specs: tuple[CoordinationGroupSpec, ...],
) -> dict[str, CoordinationGroupSpec]:
    groups: dict[str, CoordinationGroupSpec] = {}
    for spec in specs:
        label = _safe_label(spec.label)
        normalized = CoordinationGroupSpec(
            label=label,
            initial_value=float(spec.initial_value),
            lower_bound=(
                float(spec.lower_bound)
                if spec.lower_bound is not None
                else None
            ),
            upper_bound=(
                float(spec.upper_bound)
                if spec.upper_bound is not None
                else None
            ),
            restraint_scale=float(spec.restraint_scale),
            vary=bool(spec.vary),
        )
        groups[label] = normalized
        cn_name = _coordination_total_name(label)
        _append_unique_parameters(
            parameters,
            defined_names,
            (
                ArtemisGDSParameter(
                    "guess" if normalized.vary else "set",
                    cn_name,
                    _format_float(normalized.initial_value),
                    "total coordination number",
                ),
            ),
        )
        if (
            normalized.vary
            and normalized.lower_bound is not None
            and normalized.upper_bound is not None
        ):
            _append_unique_parameters(
                parameters,
                defined_names,
                (
                    ArtemisGDSParameter(
                        "restrain",
                        f"res_{cn_name}",
                        (
                            f"{_format_float(normalized.restraint_scale)}"
                            f"*penalty({cn_name}, "
                            f"{_format_float(normalized.lower_bound)}, "
                            f"{_format_float(normalized.upper_bound)})"
                        ),
                        "soft coordination-number bound",
                    ),
                ),
            )
    return groups


def _path_coordination_group_label(
    *,
    requested_group: str | None,
    default_absorber_element: str,
    default_scatterer_element: str,
    coordination_groups: dict[str, CoordinationGroupSpec],
) -> str | None:
    if requested_group:
        label = _safe_label(requested_group)
        if label not in coordination_groups:
            raise ValueError(
                f"Path references coordination group {label!r}, but no "
                "matching CoordinationGroupSpec was provided."
            )
        return label

    pair_label = _coordination_pair_label(
        default_absorber_element,
        default_scatterer_element,
    )
    if pair_label in coordination_groups:
        return pair_label
    return None


def _append_path_coordination_parameters(
    *,
    parameters: list[ArtemisGDSParameter],
    defined_names: set[str],
    path_label: str,
    coordination_group_label: str | None,
    coordination_fraction: float | None,
    reference_multiplicity: float,
) -> str:
    if coordination_group_label is None:
        return "amp"

    label = _safe_label(path_label)
    fraction = 1.0 if coordination_fraction is None else coordination_fraction
    ref_value = float(reference_multiplicity)
    if ref_value <= 0.0:
        raise ValueError(
            f"Reference multiplicity for {path_label!r} must be positive."
        )

    total_name = _coordination_total_name(coordination_group_label)
    path_cn_name = f"cn_{label}"
    fraction_name = f"frac_{label}"
    reference_name = f"cnref_{label}"
    _append_unique_parameters(
        parameters,
        defined_names,
        (
            ArtemisGDSParameter(
                "set",
                fraction_name,
                _format_float(fraction),
                "fraction of total coordination assigned to this path",
            ),
            ArtemisGDSParameter(
                "set",
                reference_name,
                _format_float(ref_value),
                "reference path degeneracy for S02 scaling",
            ),
            ArtemisGDSParameter(
                "def",
                path_cn_name,
                f"{total_name}*{fraction_name}",
                "path coordination linked to total coordination",
            ),
        ),
    )
    return f"amp*{path_cn_name}/{reference_name}"


def _append_unique_parameters(
    parameters: list[ArtemisGDSParameter],
    defined_names: set[str],
    new_parameters: tuple[ArtemisGDSParameter, ...],
) -> None:
    for parameter in new_parameters:
        lower_name = parameter.name.lower()
        if lower_name in defined_names:
            continue
        parameters.append(parameter)
        defined_names.add(lower_name)


def _zero_based_atom_index(atom_index: int, atom_count: int) -> int:
    value = int(atom_index)
    if value < 1 or value > atom_count:
        raise ValueError(
            f"Atom index {atom_index} is outside the one-based range "
            f"1-{atom_count}."
        )
    return value - 1


def _coordination_total_name(group_label: str) -> str:
    return f"cn_{_safe_label(group_label)}"


def _coordination_pair_label(
    absorber_element: str,
    scatterer_element: str,
) -> str:
    return f"{_safe_label(absorber_element)}_{_safe_label(scatterer_element)}"


def _distance(
    positions: np.ndarray,
    first_index: int,
    second_index: int,
) -> float:
    return float(
        np.linalg.norm(positions[first_index] - positions[second_index])
    )


def _safe_label(value: str) -> str:
    label = "".join(
        char.lower() if char.isalnum() else "_" for char in str(value).strip()
    )
    label = "_".join(part for part in label.split("_") if part)
    if not label:
        label = "path"
    if label[0].isdigit():
        label = f"x_{label}"
    return label[:48]


def _normalize_element(element: str) -> str:
    text = str(element).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()
