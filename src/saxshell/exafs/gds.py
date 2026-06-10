from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from saxshell.saxs.debye import load_structure_file

ARTEMIS_GDS_TYPES = (
    "guess",
    "def",
    "set",
    "lguess",
    "restrain",
    "after",
    "skip",
    "penalty",
    "merge",
)

_GDS_TYPE_SET = set(ARTEMIS_GDS_TYPES)
_COMMENT_MARKERS = ("#", "!", "%")
_SEPARATOR_RE = re.compile(r"[ \t]*[ \t=,][ \t]*")
_GDS_ROW_PREFIX_RE = re.compile(
    rf"^({'|'.join(ARTEMIS_GDS_TYPES)})\b",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_?&:][A-Za-z0-9_?&:]*\b")
_ARTEMIS_NAME_RE = re.compile(r"^[A-Za-z_?&:][A-Za-z0-9_?&:]{0,63}$")
_UI_SAFE_NAME_RE = re.compile(r"^[A-Za-z_?][A-Za-z0-9_?]{0,63}$")

_IFEFFIT_FUNCTIONS = {
    "abs",
    "acos",
    "asin",
    "atan",
    "ceil",
    "cos",
    "coth",
    "debye",
    "deriv",
    "eins",
    "erf",
    "erfc",
    "exp",
    "floor",
    "gamma",
    "gauss",
    "interp",
    "ln",
    "log",
    "log10",
    "loggamma",
    "loren",
    "max",
    "min",
    "npts",
    "ones",
    "penalty",
    "pvoight",
    "qinterp",
    "range",
    "sign",
    "sin",
    "smooth",
    "splint",
    "sqrt",
    "tan",
    "tanh",
    "vprod",
    "vsum",
    "zeros",
}

_IFEFFIT_PROGRAM_VARIABLES = {
    "chi_reduced",
    "chi_square",
    "core_width",
    "correl_min",
    "cursor_x",
    "cursor_y",
    "data_set",
    "data_total",
    "dk",
    "dk1",
    "dk1_spl",
    "dk2",
    "dk2_spl",
    "dr",
    "dr1",
    "dr2",
    "e0",
    "edge_step",
    "epsilon_k",
    "epsilon_r",
    "etok",
    "kmax",
    "kmax_spl",
    "kmax_suggest",
    "kmin",
    "kmin_spl",
    "kweight",
    "kweight_spl",
    "kwindow",
    "n_idp",
    "n_varys",
    "ncolumn_label",
    "nknots",
    "norm1",
    "norm2",
    "norm_c0",
    "norm_c1",
    "norm_c2",
    "path_index",
    "pi",
    "pre1",
    "pre2",
    "pre_offset",
    "pre_slope",
    "qmax_out",
    "qsp",
    "r_factor",
    "rbkg",
    "rmax",
    "rmax_out",
    "rmin",
    "rsp",
    "rweight",
    "rwin",
    "rwindow",
    "toler",
}

_PATH_PARAMETER_NAMES = {
    "dphase",
    "delr",
    "e0",
    "ei",
    "fourth",
    "s02",
    "sigma2",
    "third",
}

_RESERVED_PARAMETER_NAMES = (
    _IFEFFIT_FUNCTIONS
    | _IFEFFIT_PROGRAM_VARIABLES
    | _PATH_PARAMETER_NAMES
    | {"cv", "reff"}
)
_EXPRESSION_SPECIALS = {"cv", "etok", "pi", "reff"}
_EXPRESSION_BUILTINS = (
    _IFEFFIT_FUNCTIONS | _IFEFFIT_PROGRAM_VARIABLES | _EXPRESSION_SPECIALS
)
_HYDROGEN_ELEMENTS = {"H", "D", "T"}
_GDS_SECTION_ORDER = (
    "bondanalysis",
    "global",
    "coordination",
    "geometry",
    "reff",
    "delr",
    "distance",
    "sigma2",
    "restraint",
    "other",
)


@dataclass(slots=True, frozen=True)
class ArtemisGDSParameter:
    """One import/export row from the Artemis GDS grid."""

    kind: str
    name: str
    expression: str
    comment: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", self.kind.strip().lower())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "expression", self.expression.strip())
        if self.comment is not None:
            object.__setattr__(self, "comment", self.comment.strip())

    def to_artemis_line(self) -> str:
        return f"{self.kind} {self.name} = {self.expression}"


@dataclass(slots=True, frozen=True)
class ArtemisGDSShell:
    label: str
    absorber_element: str
    scatterer_element: str
    multiplicity: float
    mean_distance_angstrom: float
    std_distance_angstrom: float
    min_distance_angstrom: float
    max_distance_angstrom: float


@dataclass(slots=True, frozen=True)
class ArtemisPathParameterHint:
    shell_label: str
    s02: str
    e0: str
    delr: str
    sigma2: str
    reff_angstrom: float
    multiplicity: float

    def to_comment(self) -> str:
        return (
            f"{self.shell_label}: s02={self.s02}, e0={self.e0}, "
            f"delr={self.delr}, sigma2={self.sigma2}, "
            f"reff={_format_float(self.reff_angstrom)}, "
            f"n={_format_float(self.multiplicity)}"
        )


@dataclass(slots=True, frozen=True)
class ArtemisGDSBuildSettings:
    absorber_element: str | None = None
    absorber_atom_index: int | None = None
    min_distance_angstrom: float = 0.5
    max_distance_angstrom: float = 6.0
    included_path_pairs: tuple[tuple[int, int], ...] | None = None
    shell_tolerance_angstrom: float = 0.12
    initial_s02: float = 0.9
    initial_e0: float = 0.0
    initial_delta_r: float = 0.0
    initial_sigma2: float = 0.003
    delta_r_bound_angstrom: float = 0.15
    sigma2_lower_bound: float = 0.0001
    sigma2_upper_bound: float = 0.02
    restraint_scale: float = 1000.0
    include_restraints: bool = True


@dataclass(slots=True, frozen=True)
class ArtemisGDSDocument:
    source_structure: str
    settings: ArtemisGDSBuildSettings
    shells: tuple[ArtemisGDSShell, ...]
    parameters: tuple[ArtemisGDSParameter, ...]
    path_hints: tuple[ArtemisPathParameterHint, ...]
    overview_notes: tuple[str, ...] = ()

    def to_text(self) -> str:
        lines = _ordered_gds_lines(self.parameters)
        return "\n".join(lines) + ("\n" if lines else "")

    def to_overview_text(self, gds_path: str | Path | None = None) -> str:
        return build_artemis_gds_overview_text(self, gds_path=gds_path)


def _ordered_gds_lines(
    parameters: Iterable[ArtemisGDSParameter],
) -> list[str]:
    buckets: dict[str, list[ArtemisGDSParameter]] = {
        section: [] for section in _GDS_SECTION_ORDER
    }
    for parameter in parameters:
        buckets[_gds_parameter_section(parameter)].append(parameter)

    lines: list[str] = []
    for section in _GDS_SECTION_ORDER:
        lines.extend(
            parameter.to_artemis_line() for parameter in buckets[section]
        )
    return lines


def _gds_parameter_section(parameter: ArtemisGDSParameter) -> str:
    name = parameter.name.lower()
    kind = parameter.kind.lower()
    if kind == "restrain" or name.startswith("res_"):
        return "restraint"
    if name.startswith("ba_"):
        return "bondanalysis"
    if name in {"amp", "snot", "enot", "s02"} or name.startswith(
        ("amp_", "s02_")
    ):
        return "global"
    if name.startswith(("cn", "n_", "frac_")):
        return "coordination"
    if (
        name.startswith("sig2_theta")
        or name in {"width"}
        or name.startswith(("b_", "bl_", "theta_", "alpha_", "phi_", "width_"))
    ):
        return "geometry"
    if name.startswith("reff_"):
        return "reff"
    if name.startswith("mu_"):
        return "distance" if name.endswith("_eff") else "geometry"
    if name.startswith(("delr_", "dr_")):
        return "delr"
    if name.startswith("r_"):
        return "distance"
    if name.startswith(("sig2_", "ss_", "sigma2_")):
        return "sigma2"
    return "other"


@dataclass(slots=True, frozen=True)
class GDSValidationIssue:
    line_number: int | None
    message: str

    def format(self) -> str:
        if self.line_number is None:
            return self.message
        return f"line {self.line_number}: {self.message}"


@dataclass(slots=True, frozen=True)
class GDSValidationReport:
    parameters: tuple[ArtemisGDSParameter, ...]
    errors: tuple[GDSValidationIssue, ...]
    warnings: tuple[GDSValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary_text(self) -> str:
        lines = [
            f"Parameters parsed: {len(self.parameters)}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        lines.extend(f"ERROR: {issue.format()}" for issue in self.errors)
        lines.extend(f"WARNING: {issue.format()}" for issue in self.warnings)
        return "\n".join(lines)


def build_artemis_gds_for_structure(
    structure_path: str | Path,
    settings: ArtemisGDSBuildSettings,
) -> ArtemisGDSDocument:
    path = Path(structure_path).expanduser().resolve()
    positions, elements = load_structure_file(path)
    normalized_elements = tuple(
        _normalize_element(element) for element in elements
    )
    absorber_indices, absorber_element = _resolve_absorbers(
        normalized_elements,
        settings,
    )
    shells = _build_shells(
        positions=positions,
        elements=normalized_elements,
        absorber_indices=absorber_indices,
        absorber_element=absorber_element,
        settings=settings,
    )
    parameters, hints = _build_parameters_for_shells(shells, settings)
    return ArtemisGDSDocument(
        source_structure=str(path),
        settings=settings,
        shells=tuple(shells),
        parameters=tuple(parameters),
        path_hints=tuple(hints),
        overview_notes=(
            "Template: generic absorber-scatterer shell GDS",
            (
                "Shells were grouped by scatterer element and absorber-scatterer "
                "distance."
            ),
        ),
    )


def artemis_gds_overview_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    return path.with_name(f"{path.stem}_overview.txt")


def write_artemis_gds_file(
    output_path: str | Path,
    document: ArtemisGDSDocument,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.to_text(), encoding="utf-8")
    overview_path = artemis_gds_overview_path(path)
    overview_path.write_text(
        document.to_overview_text(gds_path=path),
        encoding="utf-8",
    )
    return path


def build_artemis_gds_overview_text(
    document: ArtemisGDSDocument,
    *,
    gds_path: str | Path | None = None,
) -> str:
    settings = document.settings
    lines = [
        "EXAFS GDS File Overview",
        "========================",
        "",
    ]
    if gds_path is not None:
        lines.append(f"GDS file: {Path(gds_path).expanduser().resolve()}")
    lines.extend(
        [
            f"Source structure: {document.source_structure}",
            f"Parameter rows: {len(document.parameters)}",
            f"Scattering shells: {len(document.shells)}",
            f"Path parameter assignments: {len(document.path_hints)}",
            "",
            "Build Choices",
            "-------------",
        ]
    )
    lines.extend(f"- {note}" for note in document.overview_notes)
    lines.extend(
        [
            f"- Absorber element: {settings.absorber_element or 'not specified'}",
            f"- Absorber atom index: {_overview_absorber_index(settings)}",
            (
                "- Distance window: "
                f"{_format_float(settings.min_distance_angstrom)} to "
                f"{_format_float(settings.max_distance_angstrom)} A"
            ),
            (
                "- Shell grouping tolerance: "
                f"{_format_float(settings.shell_tolerance_angstrom)} A"
            ),
            (
                "- Selected absorber-scatterer pairs: "
                f"{_overview_pair_count(settings)}"
            ),
            (
                "- Delta-R initial value: "
                f"{_format_float(settings.initial_delta_r)} A"
            ),
            f"- Sigma2 initial value: {_format_float(settings.initial_sigma2)}",
            (
                "- Restraints: "
                + (
                    "included with soft bounds"
                    if settings.include_restraints
                    else "not included"
                )
            ),
            (
                "- Multiplicity values are average path counts per absorber. "
                "Grouped near-degenerate paths share one GDS path weight."
            ),
            "",
            "GDS Parameter Sections",
            "----------------------",
        ]
    )
    section_counts: dict[str, int] = {
        section: 0 for section in _GDS_SECTION_ORDER
    }
    for parameter in document.parameters:
        section_counts[_gds_parameter_section(parameter)] += 1
    lines.extend(
        f"- {section}: {count}"
        for section, count in section_counts.items()
        if count
    )
    lines.extend(["", "Scattering Shells", "------------------"])
    if document.shells:
        lines.append(
            "label | absorber | scatterer | n | mean_A | std_A | min_A | max_A"
        )
        lines.append(
            "----- | -------- | --------- | - | ------ | ----- | ----- | -----"
        )
        for shell in document.shells:
            lines.append(
                " | ".join(
                    (
                        shell.label,
                        shell.absorber_element,
                        shell.scatterer_element,
                        _format_float(shell.multiplicity),
                        _format_float(shell.mean_distance_angstrom),
                        _format_float(shell.std_distance_angstrom),
                        _format_float(shell.min_distance_angstrom),
                        _format_float(shell.max_distance_angstrom),
                    )
                )
            )
    else:
        lines.append("No scattering shells were emitted.")
    lines.extend(
        ["", "Path Parameter Assignments", "--------------------------"]
    )
    if document.path_hints:
        lines.extend(f"- {hint.to_comment()}" for hint in document.path_hints)
    else:
        lines.append("No path parameter assignments were emitted.")
    return "\n".join(lines) + "\n"


def _overview_absorber_index(settings: ArtemisGDSBuildSettings) -> str:
    if settings.absorber_atom_index is None:
        return "all matching absorber atoms"
    return str(settings.absorber_atom_index)


def _overview_pair_count(settings: ArtemisGDSBuildSettings) -> str:
    if settings.included_path_pairs is None:
        return "all paths passing the distance and element filters"
    return f"{len(settings.included_path_pairs)} explicit pair(s)"


def parse_artemis_gds_text(text: str) -> tuple[ArtemisGDSParameter, ...]:
    parameters: list[ArtemisGDSParameter] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parameter = _parse_gds_line(line, line_number, strict=True)
        if parameter is not None:
            parameters.append(parameter)
    return tuple(parameters)


def validate_artemis_gds_file(path: str | Path) -> GDSValidationReport:
    text = Path(path).expanduser().read_text(encoding="utf-8")
    return validate_artemis_gds_text(text)


def validate_artemis_gds_text(text: str) -> GDSValidationReport:
    parameters: list[ArtemisGDSParameter] = []
    errors: list[GDSValidationIssue] = []
    warnings: list[GDSValidationIssue] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            parameter = _parse_gds_line(line, line_number, strict=False)
        except ValueError as exc:
            errors.append(GDSValidationIssue(line_number, str(exc)))
            continue
        if parameter is None:
            continue
        parameters.append(parameter)

    names_by_lower: dict[str, int] = {}
    for parameter in parameters:
        line_number = _line_number_for_parameter(text, parameter)
        lower_name = parameter.name.lower()
        if parameter.kind not in _GDS_TYPE_SET:
            errors.append(
                GDSValidationIssue(
                    line_number,
                    f"{parameter.kind!r} is not an Artemis GDS type",
                )
            )
        if not _ARTEMIS_NAME_RE.match(parameter.name):
            errors.append(
                GDSValidationIssue(
                    line_number,
                    (
                        f"{parameter.name!r} is not a valid Artemis "
                        "parameter name"
                    ),
                )
            )
        elif not _UI_SAFE_NAME_RE.match(parameter.name):
            warnings.append(
                GDSValidationIssue(
                    line_number,
                    (
                        f"{parameter.name!r} may import but is outside "
                        "the safest Artemis UI naming subset"
                    ),
                )
            )
        if lower_name in _RESERVED_PARAMETER_NAMES:
            errors.append(
                GDSValidationIssue(
                    line_number,
                    (
                        f"{parameter.name!r} is reserved by Ifeffit, "
                        "Larch, or Artemis path parameters"
                    ),
                )
            )
        if lower_name in names_by_lower:
            errors.append(
                GDSValidationIssue(
                    line_number,
                    (
                        f"{parameter.name!r} duplicates a parameter "
                        f"defined on line {names_by_lower[lower_name]}"
                    ),
                )
            )
        else:
            names_by_lower[lower_name] = line_number or 0
        if not parameter.expression and parameter.kind not in {
            "skip",
            "merge",
        }:
            errors.append(
                GDSValidationIssue(
                    line_number,
                    f"{parameter.name!r} has an empty math expression",
                )
            )
        _validate_expression_shape(
            parameter.expression,
            line_number,
            errors,
        )

    defined_names = {parameter.name.lower() for parameter in parameters}
    seen_names: set[str] = set()
    for parameter in parameters:
        line_number = _line_number_for_parameter(text, parameter)
        refs = _expression_references(parameter.expression)
        for ref in sorted(refs):
            ref_lower = ref.lower()
            if ref_lower == parameter.name.lower():
                errors.append(
                    GDSValidationIssue(
                        line_number,
                        f"{parameter.name!r} references itself",
                    )
                )
            elif ref_lower in defined_names:
                if ref_lower not in seen_names:
                    warnings.append(
                        GDSValidationIssue(
                            line_number,
                            (
                                f"{parameter.name!r} references "
                                f"{ref!r} before it is defined"
                            ),
                        )
                    )
            elif ref_lower not in _EXPRESSION_BUILTINS:
                errors.append(
                    GDSValidationIssue(
                        line_number,
                        (
                            f"{parameter.name!r} references undefined "
                            f"symbol {ref!r}"
                        ),
                    )
                )
        seen_names.add(parameter.name.lower())

    return GDSValidationReport(
        parameters=tuple(parameters),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _build_parameters_for_shells(
    shells: Iterable[ArtemisGDSShell],
    settings: ArtemisGDSBuildSettings,
) -> tuple[list[ArtemisGDSParameter], list[ArtemisPathParameterHint]]:
    parameters = [
        ArtemisGDSParameter(
            "guess",
            "amp",
            _format_float(settings.initial_s02),
            "global S02 amplitude factor",
        ),
        ArtemisGDSParameter(
            "guess",
            "enot",
            _format_float(settings.initial_e0),
            "global edge-energy shift",
        ),
    ]
    hints: list[ArtemisPathParameterHint] = []

    for shell in shells:
        dr_name = f"dr_{shell.label}"
        sigma_name = f"ss_{shell.label}"
        parameters.extend(
            [
                ArtemisGDSParameter(
                    "guess",
                    dr_name,
                    _format_float(settings.initial_delta_r),
                    "fit delta-R for this shell",
                ),
                ArtemisGDSParameter(
                    "guess",
                    sigma_name,
                    _format_float(settings.initial_sigma2),
                    "fit sigma2 for this shell",
                ),
            ]
        )
        if settings.include_restraints:
            delta_r_low = _format_float(-settings.delta_r_bound_angstrom)
            delta_r_high = _format_float(settings.delta_r_bound_angstrom)
            sigma2_low = _format_float(settings.sigma2_lower_bound)
            sigma2_high = _format_float(settings.sigma2_upper_bound)
            parameters.extend(
                [
                    ArtemisGDSParameter(
                        "restrain",
                        f"res_{dr_name}",
                        (
                            f"{_format_float(settings.restraint_scale)}"
                            f"*penalty({dr_name}, "
                            f"{delta_r_low}, {delta_r_high})"
                        ),
                        "soft delta-R bound",
                    ),
                    ArtemisGDSParameter(
                        "restrain",
                        f"res_{sigma_name}",
                        (
                            f"{_format_float(settings.restraint_scale)}"
                            f"*penalty({sigma_name}, "
                            f"{sigma2_low}, {sigma2_high})"
                        ),
                        "soft sigma2 bound",
                    ),
                ]
            )
        hints.append(
            ArtemisPathParameterHint(
                shell_label=shell.label,
                s02="amp",
                e0="enot",
                delr=dr_name,
                sigma2=sigma_name,
                reff_angstrom=shell.mean_distance_angstrom,
                multiplicity=shell.multiplicity,
            )
        )
    return parameters, hints


def _build_shells(
    *,
    positions: np.ndarray,
    elements: tuple[str, ...],
    absorber_indices: tuple[int, ...],
    absorber_element: str,
    settings: ArtemisGDSBuildSettings,
) -> list[ArtemisGDSShell]:
    distances_by_scatterer: dict[str, list[float]] = {}
    min_distance = max(float(settings.min_distance_angstrom), 0.0)
    max_distance = max(float(settings.max_distance_angstrom), min_distance)
    included_path_pairs = (
        None
        if settings.included_path_pairs is None
        else {
            (int(absorber_index), int(scatterer_index))
            for absorber_index, scatterer_index in settings.included_path_pairs
        }
    )

    for absorber_index in absorber_indices:
        absorber_position = positions[absorber_index]
        for scatterer_index, scatterer_element in enumerate(elements):
            if scatterer_index == absorber_index:
                continue
            if _is_hydrogen_element(scatterer_element):
                continue
            if (
                included_path_pairs is not None
                and (
                    absorber_index + 1,
                    scatterer_index + 1,
                )
                not in included_path_pairs
            ):
                continue
            distance = float(
                np.linalg.norm(positions[scatterer_index] - absorber_position)
            )
            if min_distance <= distance <= max_distance:
                distances_by_scatterer.setdefault(
                    scatterer_element,
                    [],
                ).append(distance)

    shells: list[ArtemisGDSShell] = []
    absorber_label = _safe_token(absorber_element)
    for scatterer_element in sorted(distances_by_scatterer):
        clusters = _cluster_distances(
            distances_by_scatterer[scatterer_element],
            tolerance=max(float(settings.shell_tolerance_angstrom), 0.0),
        )
        scatterer_label = _safe_token(scatterer_element)
        for shell_index, cluster in enumerate(clusters, start=1):
            values = np.asarray(cluster, dtype=float)
            label = f"{absorber_label}_{scatterer_label}_{shell_index:02d}"
            shells.append(
                ArtemisGDSShell(
                    label=label,
                    absorber_element=absorber_element,
                    scatterer_element=scatterer_element,
                    multiplicity=float(len(values)) / len(absorber_indices),
                    mean_distance_angstrom=float(values.mean()),
                    std_distance_angstrom=float(values.std(ddof=0)),
                    min_distance_angstrom=float(values.min()),
                    max_distance_angstrom=float(values.max()),
                )
            )
    return sorted(
        shells,
        key=lambda shell: (
            shell.mean_distance_angstrom,
            shell.scatterer_element,
            shell.label,
        ),
    )


def _cluster_distances(
    distances: Iterable[float],
    *,
    tolerance: float,
) -> list[list[float]]:
    clusters: list[list[float]] = []
    for distance in sorted(float(value) for value in distances):
        if not clusters:
            clusters.append([distance])
            continue
        current = clusters[-1]
        current_mean = sum(current) / len(current)
        if abs(distance - current_mean) <= tolerance:
            current.append(distance)
        else:
            clusters.append([distance])
    return clusters


def _resolve_absorbers(
    elements: tuple[str, ...],
    settings: ArtemisGDSBuildSettings,
) -> tuple[tuple[int, ...], str]:
    if settings.absorber_atom_index is not None:
        atom_index = int(settings.absorber_atom_index)
        if atom_index < 1 or atom_index > len(elements):
            raise ValueError(
                "Absorber atom index is one-based and must refer to an "
                "atom in the structure."
            )
        zero_based = atom_index - 1
        absorber_element = elements[zero_based]
        if _is_hydrogen_element(absorber_element):
            raise ValueError(
                "Hydrogen atoms are excluded from EXAFS path generation."
            )
        if settings.absorber_element:
            requested = _normalize_element(settings.absorber_element)
            if requested != absorber_element:
                raise ValueError(
                    "Absorber atom index element "
                    f"{absorber_element!r} does not match requested "
                    f"absorber element {requested!r}."
                )
        return (zero_based,), absorber_element

    if not settings.absorber_element:
        raise ValueError(
            "Provide absorber_element or absorber_atom_index before "
            "building an Artemis GDS file."
        )
    absorber_element = _normalize_element(settings.absorber_element)
    if _is_hydrogen_element(absorber_element):
        raise ValueError(
            "Hydrogen atoms are excluded from EXAFS path generation."
        )
    indices = tuple(
        index
        for index, element in enumerate(elements)
        if element == absorber_element
    )
    if not indices:
        raise ValueError(
            f"No absorber atoms with element {absorber_element!r} were "
            "found in the structure."
        )
    return indices, absorber_element


def _parse_gds_line(
    line: str,
    line_number: int,
    *,
    strict: bool,
) -> ArtemisGDSParameter | None:
    body = _strip_comment(line).strip()
    if not body:
        return None
    if not _GDS_ROW_PREFIX_RE.match(body):
        if strict:
            return None
        raise ValueError("line is not an Artemis GDS parameter row")
    parts = _SEPARATOR_RE.split(body, maxsplit=2)
    if len(parts) < 3:
        raise ValueError(
            "could not parse GDS row into type, name, and expression"
        )
    kind, name, expression = parts[0].lower(), parts[1], parts[2]
    if kind not in _GDS_TYPE_SET:
        raise ValueError(f"{kind!r} is not an Artemis GDS type")
    if not name.strip():
        raise ValueError("GDS parameter name is empty")
    if not expression.strip() and kind not in {"skip", "merge"}:
        raise ValueError("GDS parameter expression is empty")
    return ArtemisGDSParameter(kind=kind, name=name, expression=expression)


def _strip_comment(line: str) -> str:
    comment_positions = [
        line.find(marker)
        for marker in _COMMENT_MARKERS
        if line.find(marker) >= 0
    ]
    if not comment_positions:
        return line
    return line[: min(comment_positions)]


def _validate_expression_shape(
    expression: str,
    line_number: int | None,
    errors: list[GDSValidationIssue],
) -> None:
    depth = 0
    for char in expression:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth < 0:
            errors.append(
                GDSValidationIssue(
                    line_number,
                    "math expression has unmatched closing parenthesis",
                )
            )
            return
    if depth:
        errors.append(
            GDSValidationIssue(
                line_number,
                "math expression has unmatched opening parenthesis",
            )
        )


def _expression_references(expression: str) -> set[str]:
    refs: set[str] = set()
    for match in _IDENTIFIER_RE.finditer(expression):
        token = match.group(0)
        end = match.end()
        next_non_space = ""
        for char in expression[end:]:
            if not char.isspace():
                next_non_space = char
                break
        if next_non_space == "(" and token.lower() in _IFEFFIT_FUNCTIONS:
            continue
        refs.add(token)
    return refs


def _line_number_for_parameter(
    text: str,
    parameter: ArtemisGDSParameter,
) -> int | None:
    target = parameter.to_artemis_line().split("#", 1)[0].strip()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _strip_comment(line).strip() == target:
            return line_number
    return None


def _normalize_element(element: str) -> str:
    text = str(element).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def _is_hydrogen_element(element: str) -> bool:
    return _normalize_element(element) in _HYDROGEN_ELEMENTS


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_?]+", "_", value.strip().lower())
    token = token.strip("_")
    if not token:
        token = "x"
    if token[0].isdigit():
        token = f"x_{token}"
    return token[:32]


def _format_float(value: float) -> str:
    return f"{float(value):.6g}"
