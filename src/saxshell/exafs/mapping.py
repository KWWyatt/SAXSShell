from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from saxshell.bondanalysis.results import (
    RESULTS_INDEX_FILENAME,
    BondAnalysisResultIndex,
    load_result_index,
)
from saxshell.fullrmc.project_model import build_rmcsetup_paths
from saxshell.fullrmc.representatives import (
    RepresentativeSelectionEntry,
    load_representative_selection_metadata,
    representative_structure_variant_path,
)
from saxshell.saxs.debye import load_structure_file

from .gds import (
    ArtemisGDSBuildSettings,
    ArtemisGDSDocument,
    ArtemisGDSParameter,
    parse_artemis_gds_text,
    validate_artemis_gds_text,
    write_artemis_gds_file,
)
from .pb_dmf import PbDMFGDSBuildSettings, build_pb_dmf_gds_from_structure
from .pb_dmso import PbDMSOGDSBuildSettings, build_pb_dmso_gds_from_structure

_STRUCTURE_SUFFIXES = {".pdb", ".xyz"}
_DEFAULT_CIF_PADDING_ANGSTROM = 20.0
_COMMON_ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "Sn": 50,
    "Pb": 82,
}
_HYDROGEN_ELEMENTS = {"H", "D", "T"}
_DEFAULT_ABSORBER_ELEMENT = "Pb"
_DEFAULT_PAIR_CUTOFF_DISTANCES_ANGSTROM: Mapping[
    tuple[str, str],
    float,
] = {
    ("Pb", "I"): 3.36,
    ("Pb", "O"): 3.36,
}
_SOLVENT_RESIDUE_NAMES = {"DMF", "DMS", "DMSO"}
_SOLVENT_DONOR_ELEMENTS = {"O"}
_COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "D": 0.31,
    "T": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "K": 2.03,
    "Ca": 1.76,
    "Br": 1.20,
    "I": 1.39,
    "Sn": 1.39,
    "Pb": 1.46,
}

PairCutoffDistanceDefinitions = Mapping[
    tuple[str, str],
    Mapping[int, float] | float | int,
]


@dataclass(frozen=True, slots=True)
class EXAFSRepresentativeOption:
    stoichiometry: str
    motif: str
    source_file: Path
    source_file_name: str
    source_solvent_mode: str
    param: str
    selected_weight: float
    cluster_count: int
    atom_count: int
    element_counts: Mapping[str, int]

    @property
    def display_label(self) -> str:
        motif_suffix = "" if self.motif == "no_motif" else f" / {self.motif}"
        return f"{self.stoichiometry}{motif_suffix} - {self.source_file_name}"

    def variant_path(self, variant: str) -> Path | None:
        return representative_structure_variant_path(self.source_file, variant)


@dataclass(frozen=True, slots=True)
class EXAFSBondAnalysisResult:
    output_dir: Path
    results_index_path: Path
    selected_cluster_types: tuple[str, ...]
    gds_variable_count: int

    @property
    def display_label(self) -> str:
        cluster_text = ", ".join(self.selected_cluster_types) or "all"
        return f"{self.output_dir.name} ({cluster_text})"


@dataclass(frozen=True, slots=True)
class EXAFSScatteringPath:
    label: str
    absorber_index: int
    scatterer_index: int
    absorber_element: str
    scatterer_element: str
    absorber_atom_label: str
    scatterer_atom_label: str
    distance_angstrom: float
    start: tuple[float, float, float]
    end: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class EXAFSScatteringPathEvent:
    label: str
    path_key: tuple[int, int]
    atom_indices: tuple[int, ...]
    atom_labels: tuple[str, ...]
    absorber_index: int
    scatterer_index: int
    absorber_element: str
    scatterer_element: str
    degeneracy: float
    total_path_length_angstrom: float
    effective_distance_angstrom: float
    bond_lengths: tuple[str, ...]
    angles: tuple[str, ...]
    dihedrals: tuple[str, ...]
    solvent_molecule_key: str
    solvent_molecule_label: str

    @property
    def absorber_atom_label(self) -> str:
        return self.atom_labels[0] if self.atom_labels else ""

    @property
    def scatterer_atom_label(self) -> str:
        return self.atom_labels[-1] if self.atom_labels else ""


@dataclass(frozen=True, slots=True)
class EXAFSBondAnnotation:
    label: str
    kind: str
    atom1_index: int
    atom2_index: int
    atom1_element: str
    atom2_element: str
    atom1_label_text: str
    atom2_label_text: str
    distance_angstrom: float
    start: tuple[float, float, float]
    end: tuple[float, float, float]

    @property
    def atom1_label(self) -> str:
        return self.atom1_label_text

    @property
    def atom2_label(self) -> str:
        return self.atom2_label_text


@dataclass(frozen=True, slots=True)
class EXAFSAngleAnnotation:
    label: str
    absorber_index: int
    bridge_index: int
    terminal_index: int
    absorber_element: str
    bridge_element: str
    terminal_element: str
    absorber_atom_label: str
    bridge_atom_label: str
    terminal_atom_label: str
    angle_degrees: float
    absorber: tuple[float, float, float]
    bridge: tuple[float, float, float]
    terminal: tuple[float, float, float]

    @property
    def atom_triplet_label(self) -> str:
        return (
            f"{self.absorber_atom_label}-"
            f"{self.bridge_atom_label}-"
            f"{self.terminal_atom_label}"
        )


@dataclass(frozen=True, slots=True)
class EXAFSDihedralAnnotation:
    label: str
    atom_indices: tuple[int, int, int, int]
    plane1_indices: tuple[int, int, int]
    plane2_indices: tuple[int, int, int]
    atom_labels: tuple[str, str, str, str]
    residue_key: str | None
    angle_degrees: float
    points: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


@dataclass(frozen=True, slots=True)
class EXAFSStructurePreview:
    structure_path: Path
    coordinates: np.ndarray
    elements: tuple[str, ...]
    atom_indices: tuple[int, ...]
    atom_labels: tuple[str, ...]
    atom_records: tuple[_StructureAtomRecord, ...]
    absorber_indices: tuple[int, ...]
    paths: tuple[EXAFSScatteringPath, ...]
    dynamic_bonds: tuple[EXAFSBondAnnotation, ...]
    static_bonds: tuple[EXAFSBondAnnotation, ...]
    angles: tuple[EXAFSAngleAnnotation, ...]
    dihedrals: tuple[EXAFSDihedralAnnotation, ...] = ()

    @property
    def element_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for element in self.elements:
            counts[element] = counts.get(element, 0) + 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True, slots=True)
class _StructureAtomRecord:
    atom_index: int
    element: str
    atom_name: str
    residue_name: str
    residue_key: str | None


def discover_representative_structures(
    project_dir: str | Path,
) -> tuple[EXAFSRepresentativeOption, ...]:
    paths = build_rmcsetup_paths(project_dir)
    metadata = load_representative_selection_metadata(
        paths.representative_selection_path
    )
    if metadata is None:
        return ()

    options: list[EXAFSRepresentativeOption] = []
    for entry in metadata.representative_entries:
        option = _representative_option_from_entry(entry)
        if option is not None:
            options.append(option)
    return tuple(
        sorted(
            options,
            key=lambda item: (
                _natural_sort_key(item.stoichiometry),
                _natural_sort_key(item.motif),
                _natural_sort_key(item.source_file_name),
            ),
        )
    )


def discover_bondanalysis_results(
    project_dir: str | Path,
) -> tuple[EXAFSBondAnalysisResult, ...]:
    root = Path(project_dir).expanduser().resolve()
    candidate_roots = (
        root / "analysis" / "bondanalysis",
        root / "saved_distributions",
        root,
    )
    seen: set[Path] = set()
    results: list[EXAFSBondAnalysisResult] = []
    for candidate_root in candidate_roots:
        if not candidate_root.is_dir():
            continue
        for index_path in _bondanalysis_result_index_paths(candidate_root):
            resolved_index = index_path.resolve()
            if resolved_index in seen:
                continue
            seen.add(resolved_index)
            try:
                index = load_result_index(resolved_index.parent)
            except Exception:
                continue
            if not index.gds_variable_registry:
                continue
            results.append(
                EXAFSBondAnalysisResult(
                    output_dir=index.output_dir,
                    results_index_path=index.results_index_path,
                    selected_cluster_types=tuple(index.selected_cluster_types),
                    gds_variable_count=len(index.gds_variable_registry),
                )
            )
    return tuple(
        sorted(
            results, key=lambda item: _natural_sort_key(str(item.output_dir))
        )
    )


def _bondanalysis_result_index_paths(root: Path) -> tuple[Path, ...]:
    candidate_dirs = [root]
    try:
        candidate_dirs.extend(
            child for child in root.iterdir() if child.is_dir()
        )
    except OSError:
        candidate_dirs = [root]
    paths = [
        candidate_dir / RESULTS_INDEX_FILENAME
        for candidate_dir in candidate_dirs
        if (candidate_dir / RESULTS_INDEX_FILENAME).is_file()
    ]
    paths.sort(
        key=lambda path: (_safe_result_index_mtime(path), str(path)),
        reverse=True,
    )
    return tuple(paths)


def _safe_result_index_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def load_bondanalysis_result(
    result: EXAFSBondAnalysisResult | str | Path,
) -> BondAnalysisResultIndex:
    if isinstance(result, EXAFSBondAnalysisResult):
        return load_result_index(result.output_dir)
    path = Path(result).expanduser().resolve()
    return load_result_index(path.parent if path.is_file() else path)


def gds_registry_entries_for_stoichiometry(
    result_index: BondAnalysisResultIndex,
    stoichiometry: str | None,
    *,
    include_aggregate: bool = True,
) -> tuple[dict[str, object], ...]:
    selected = []
    normalized_stoichiometry = str(stoichiometry or "").strip()
    for entry in result_index.gds_variable_registry:
        scope = str(entry.get("scope", "")).strip()
        if scope == normalized_stoichiometry:
            selected.append(dict(entry))
            continue
        if include_aggregate and scope.lower() in {
            "all selected clusters",
            "all clusters",
            "all",
        }:
            selected.append(dict(entry))
    return tuple(selected)


def load_structure_preview(
    structure_path: str | Path,
    *,
    absorber_element: str | None = None,
    absorber_atom_index: int | None = None,
    min_distance_angstrom: float = 0.5,
    max_distance_angstrom: float = 6.0,
    pair_cutoff_distances_angstrom: PairCutoffDistanceDefinitions | None = (
        None
    ),
) -> EXAFSStructurePreview:
    path = Path(structure_path).expanduser().resolve()
    coordinates, raw_elements = load_structure_file(path)
    coordinate_array = np.asarray(coordinates, dtype=float)
    elements = tuple(_normalize_element(element) for element in raw_elements)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
        raise ValueError(f"Expected Nx3 structure coordinates in {path}.")
    if len(elements) != len(coordinate_array):
        raise ValueError(
            f"Element and coordinate counts differ in {path}: "
            f"{len(elements)} elements for {len(coordinate_array)} positions."
        )
    included_atom_indices = tuple(
        index
        for index, element in enumerate(elements)
        if not _is_hydrogen_element(element)
    )
    if not included_atom_indices:
        raise ValueError(
            "No non-hydrogen atoms were found in the selected structure."
        )
    atom_records = _structure_atom_records(path, elements)
    atom_labels = _structure_atom_labels(atom_records, elements)
    absorber_indices = _absorber_indices(
        elements,
        absorber_element=absorber_element,
        absorber_atom_index=absorber_atom_index,
    )
    pair_cutoffs = _normalized_pair_cutoff_distances(
        pair_cutoff_distances_angstrom
    )
    min_distance = max(float(min_distance_angstrom), 0.0)
    max_distance = max(float(max_distance_angstrom), min_distance)
    admitted_solvent_atoms = _absorber_solvent_atom_indices(
        coordinate_array,
        elements,
        atom_records=atom_records,
        absorber_indices=absorber_indices,
        included_atom_indices=included_atom_indices,
        min_distance_angstrom=min_distance,
        max_distance_angstrom=max_distance,
        pair_cutoffs=pair_cutoffs,
    )
    solvent_atom_indices = _solvent_atom_indices(
        coordinate_array,
        elements,
        atom_records=atom_records,
        absorber_indices=absorber_indices,
        included_atom_indices=included_atom_indices,
    )
    paths: list[EXAFSScatteringPath] = []
    for absorber_index in absorber_indices:
        start = coordinate_array[absorber_index]
        absorber = elements[absorber_index]
        for scatterer_index in included_atom_indices:
            scatterer = elements[scatterer_index]
            if scatterer_index == absorber_index:
                continue
            if (
                scatterer_index in solvent_atom_indices
                and scatterer_index
                not in admitted_solvent_atoms.get(absorber_index, set())
            ):
                continue
            end = coordinate_array[scatterer_index]
            distance = float(np.linalg.norm(end - start))
            if not (min_distance <= distance <= max_distance):
                continue
            paths.append(
                EXAFSScatteringPath(
                    label=(
                        f"{atom_labels[absorber_index]}-"
                        f"{atom_labels[scatterer_index]}"
                    ),
                    absorber_index=absorber_index + 1,
                    scatterer_index=scatterer_index + 1,
                    absorber_element=absorber,
                    scatterer_element=scatterer,
                    absorber_atom_label=atom_labels[absorber_index],
                    scatterer_atom_label=atom_labels[scatterer_index],
                    distance_angstrom=distance,
                    start=tuple(float(value) for value in start),
                    end=tuple(float(value) for value in end),
                )
            )
    sorted_paths = tuple(
        sorted(paths, key=lambda item: item.distance_angstrom)
    )
    static_bonds = _static_bond_annotations(
        coordinate_array,
        elements,
        atom_labels=atom_labels,
        absorber_indices=absorber_indices,
        included_atom_indices=included_atom_indices,
        atom_records=atom_records,
        dynamic_paths=sorted_paths,
    )
    dynamic_bonds = _dynamic_bond_annotations(
        sorted_paths,
        static_bonds=static_bonds,
    )
    static_bonds = _static_bonds_for_dynamic_components(
        static_bonds,
        dynamic_bonds=dynamic_bonds,
    )
    angles = _angle_annotations(
        coordinate_array,
        elements,
        atom_labels=atom_labels,
        dynamic_bonds=dynamic_bonds,
        static_bonds=static_bonds,
    )
    dihedrals = _dihedral_annotations(
        coordinate_array,
        elements,
        atom_labels=atom_labels,
        atom_records=atom_records,
        dynamic_bonds=dynamic_bonds,
        static_bonds=static_bonds,
    )
    return EXAFSStructurePreview(
        structure_path=path,
        coordinates=coordinate_array[list(included_atom_indices)],
        elements=tuple(elements[index] for index in included_atom_indices),
        atom_indices=tuple(index + 1 for index in included_atom_indices),
        atom_labels=tuple(
            atom_labels[index] for index in included_atom_indices
        ),
        atom_records=atom_records,
        absorber_indices=tuple(index + 1 for index in absorber_indices),
        paths=sorted_paths,
        dynamic_bonds=dynamic_bonds,
        static_bonds=static_bonds,
        angles=angles,
        dihedrals=dihedrals,
    )


def write_padded_cif_from_structure(
    structure_path: str | Path,
    *,
    padding_angstrom: float = _DEFAULT_CIF_PADDING_ANGSTROM,
    output_path: str | Path | None = None,
) -> Path:
    source_path = Path(structure_path).expanduser().resolve()
    coordinates, raw_elements = load_structure_file(source_path)
    coordinate_array = np.asarray(coordinates, dtype=float)
    elements = tuple(_normalize_element(element) for element in raw_elements)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
        raise ValueError(
            f"Expected Nx3 structure coordinates in {source_path}."
        )
    if len(elements) != len(coordinate_array):
        raise ValueError(
            f"Element and coordinate counts differ in {source_path}: "
            f"{len(elements)} elements for {len(coordinate_array)} positions."
        )
    if len(elements) == 0:
        raise ValueError("Cannot write a CIF for an empty structure.")

    padding = max(float(padding_angstrom), 0.0)
    destination = (
        _default_padded_cif_output_path(source_path, padding)
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _padded_cif_text(
            source_path,
            coordinates=coordinate_array,
            elements=elements,
            padding_angstrom=padding,
        ),
        encoding="utf-8",
    )
    return destination


def default_absorber_element(
    structure_path: str | Path,
    *,
    preferred_element: str | None = None,
) -> str | None:
    _coordinates, raw_elements = load_structure_file(structure_path)
    return default_absorber_element_from_elements(
        raw_elements,
        preferred_element=preferred_element,
    )


def default_absorber_element_from_elements(
    elements: Iterable[object],
    *,
    preferred_element: str | None = None,
) -> str | None:
    normalized_elements = tuple(
        element
        for element in (_normalize_element(element) for element in elements)
        if not _is_hydrogen_element(element)
    )
    if not normalized_elements:
        return None
    unique_elements = sorted(set(normalized_elements), key=_natural_sort_key)
    for candidate in (
        _normalize_element(preferred_element) if preferred_element else "",
        _DEFAULT_ABSORBER_ELEMENT,
    ):
        if candidate and candidate in unique_elements:
            return candidate
    return max(
        unique_elements,
        key=lambda element: _COMMON_ATOMIC_NUMBERS.get(element, 0),
    )


def build_gds_mapping_document(
    structure_path: str | Path,
    *,
    mode: str = "generic",
    absorber_element: str | None = None,
    absorber_atom_index: int | None = None,
    min_distance_angstrom: float = 0.5,
    max_distance_angstrom: float = 6.0,
    pair_cutoff_distances_angstrom: PairCutoffDistanceDefinitions | None = (
        None
    ),
    included_path_pairs: Iterable[tuple[int, int]] | None = None,
    shell_tolerance_angstrom: float = 0.12,
    include_restraints: bool = True,
    gds_registry_entries: Iterable[Mapping[str, object]] = (),
) -> ArtemisGDSDocument:
    normalized_mode = str(mode or "generic").strip().lower()
    pair_cutoffs = _normalized_pair_cutoff_distances(
        pair_cutoff_distances_angstrom
    )
    if normalized_mode in {"pb_dmso", "pbi2_dmso", "pb-i/dmso"}:
        effective_absorber = absorber_element or default_absorber_element(
            structure_path
        )
        oxygen_cutoff = _pair_cutoff(
            pair_cutoffs,
            effective_absorber or _DEFAULT_ABSORBER_ELEMENT,
            "O",
        )
        document = build_pb_dmso_gds_from_structure(
            structure_path,
            PbDMSOGDSBuildSettings(
                absorber_atom_index=absorber_atom_index,
                min_distance_angstrom=min_distance_angstrom,
                max_iodide_distance_angstrom=max_distance_angstrom,
                max_oxygen_distance_angstrom=(
                    max_distance_angstrom
                    if oxygen_cutoff is None
                    else min(float(max_distance_angstrom), oxygen_cutoff)
                ),
                included_path_pairs=_normalized_included_path_pairs(
                    included_path_pairs
                ),
                shell_tolerance_angstrom=shell_tolerance_angstrom,
                include_restraints=include_restraints,
            ),
        )
    elif normalized_mode in {"pb_dmf", "pbi2_dmf", "pb-i/dmf"}:
        effective_absorber = absorber_element or default_absorber_element(
            structure_path
        )
        oxygen_cutoff = _pair_cutoff(
            pair_cutoffs,
            effective_absorber or _DEFAULT_ABSORBER_ELEMENT,
            "O",
        )
        document = build_pb_dmf_gds_from_structure(
            structure_path,
            PbDMFGDSBuildSettings(
                absorber_atom_index=absorber_atom_index,
                min_distance_angstrom=min_distance_angstrom,
                max_iodide_distance_angstrom=max_distance_angstrom,
                max_oxygen_distance_angstrom=(
                    max_distance_angstrom
                    if oxygen_cutoff is None
                    else min(float(max_distance_angstrom), oxygen_cutoff)
                ),
                included_path_pairs=_normalized_included_path_pairs(
                    included_path_pairs
                ),
                shell_tolerance_angstrom=shell_tolerance_angstrom,
                include_restraints=include_restraints,
            ),
        )
    else:
        effective_absorber = absorber_element or default_absorber_element(
            structure_path
        )
        document = _build_generic_gds_document(
            structure_path,
            absorber_element=effective_absorber,
            absorber_atom_index=absorber_atom_index,
            min_distance_angstrom=min_distance_angstrom,
            max_distance_angstrom=max_distance_angstrom,
            pair_cutoff_distances_angstrom=pair_cutoffs,
            included_path_pairs=_normalized_included_path_pairs(
                included_path_pairs
            ),
            shell_tolerance_angstrom=shell_tolerance_angstrom,
            include_restraints=include_restraints,
        )

    registry_parameters = gds_parameters_from_registry_entries(
        gds_registry_entries
    )
    if not registry_parameters:
        return document
    return replace(
        document,
        parameters=tuple(registry_parameters) + tuple(document.parameters),
        overview_notes=tuple(document.overview_notes)
        + (
            (
                "Prepended bond-analysis registry parameters: "
                f"{len(registry_parameters)}"
            ),
        ),
    )


def write_gds_mapping_file(
    output_path: str | Path,
    structure_path: str | Path,
    *,
    mode: str = "generic",
    absorber_element: str | None = None,
    absorber_atom_index: int | None = None,
    min_distance_angstrom: float = 0.5,
    max_distance_angstrom: float = 6.0,
    pair_cutoff_distances_angstrom: PairCutoffDistanceDefinitions | None = (
        None
    ),
    included_path_pairs: Iterable[tuple[int, int]] | None = None,
    shell_tolerance_angstrom: float = 0.12,
    include_restraints: bool = True,
    gds_registry_entries: Iterable[Mapping[str, object]] = (),
) -> Path:
    document = build_gds_mapping_document(
        structure_path,
        mode=mode,
        absorber_element=absorber_element,
        absorber_atom_index=absorber_atom_index,
        min_distance_angstrom=min_distance_angstrom,
        max_distance_angstrom=max_distance_angstrom,
        pair_cutoff_distances_angstrom=pair_cutoff_distances_angstrom,
        included_path_pairs=included_path_pairs,
        shell_tolerance_angstrom=shell_tolerance_angstrom,
        include_restraints=include_restraints,
        gds_registry_entries=gds_registry_entries,
    )
    text = document.to_text()
    report = validate_artemis_gds_text(text)
    if not report.is_valid:
        raise ValueError(report.summary_text())
    return write_artemis_gds_file(output_path, document)


def gds_parameters_from_registry_entries(
    entries: Iterable[Mapping[str, object]],
) -> tuple[ArtemisGDSParameter, ...]:
    set_rows: list[str] = []
    for entry in entries:
        rows = str(entry.get("set_rows", "")).strip()
        set_rows.extend(
            row.strip()
            for row in rows.replace("\n", ";").split(";")
            if row.strip().lower().startswith("set ")
        )
    if not set_rows:
        return ()
    parsed = parse_artemis_gds_text("\n".join(set_rows))
    deduped: list[ArtemisGDSParameter] = []
    seen: set[str] = set()
    for parameter in parsed:
        lower_name = parameter.name.lower()
        if lower_name in seen:
            continue
        seen.add(lower_name)
        deduped.append(parameter)
    return tuple(deduped)


def scattering_path_events_from_preview(
    preview: EXAFSStructurePreview,
) -> tuple[EXAFSScatteringPathEvent, ...]:
    """Summarize preview paths as selectable scattering events."""

    if not preview.paths:
        return ()

    bond_edges = _path_event_bond_edges(
        tuple(preview.dynamic_bonds) + tuple(preview.static_bonds)
    )
    angle_by_key = _path_event_angles(preview.angles)
    dihedral_by_key = _path_event_dihedrals(preview.dihedrals)
    labels_by_index = dict(
        zip(preview.atom_indices, preview.atom_labels, strict=False)
    )
    elements_by_index = dict(
        zip(preview.atom_indices, preview.elements, strict=False)
    )
    records_by_index = {
        record.atom_index: record for record in preview.atom_records
    }

    events: list[EXAFSScatteringPathEvent] = []
    for path in preview.paths:
        route = _best_path_event_route(
            path.absorber_index,
            path.scatterer_index,
            bond_edges,
            angle_by_key,
            dihedral_by_key,
        )
        if route is None:
            route = (path.absorber_index, path.scatterer_index)
        bond_lengths = _path_event_bond_length_labels(
            route,
            bond_edges=bond_edges,
            fallback_path=path,
        )
        angles = _path_event_angle_labels(route, angle_by_key)
        dihedrals = _path_event_dihedral_labels(route, dihedral_by_key)
        total_length = _path_event_total_length(
            route,
            bond_edges=bond_edges,
            fallback_distance=path.distance_angstrom,
        )
        atom_labels = tuple(
            labels_by_index.get(index, str(index)) for index in route
        )
        solvent_molecule_key, solvent_molecule_label = (
            _path_event_solvent_molecule_group(
                route,
                atom_labels=atom_labels,
                elements_by_index=elements_by_index,
                records_by_index=records_by_index,
            )
        )
        events.append(
            EXAFSScatteringPathEvent(
                label="-".join(atom_labels),
                path_key=(path.absorber_index, path.scatterer_index),
                atom_indices=route,
                atom_labels=atom_labels,
                absorber_index=path.absorber_index,
                scatterer_index=path.scatterer_index,
                absorber_element=path.absorber_element,
                scatterer_element=path.scatterer_element,
                degeneracy=1.0,
                total_path_length_angstrom=total_length,
                effective_distance_angstrom=path.distance_angstrom,
                bond_lengths=bond_lengths,
                angles=angles,
                dihedrals=dihedrals,
                solvent_molecule_key=solvent_molecule_key,
                solvent_molecule_label=solvent_molecule_label,
            )
        )
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.absorber_index,
                event.effective_distance_angstrom,
                event.scatterer_index,
            ),
        )
    )


def _build_generic_gds_document(
    structure_path: str | Path,
    *,
    absorber_element: str | None,
    absorber_atom_index: int | None,
    min_distance_angstrom: float,
    max_distance_angstrom: float,
    pair_cutoff_distances_angstrom: PairCutoffDistanceDefinitions | None,
    included_path_pairs: tuple[tuple[int, int], ...] | None,
    shell_tolerance_angstrom: float,
    include_restraints: bool,
) -> ArtemisGDSDocument:
    if absorber_atom_index is None and not absorber_element:
        raise ValueError(
            "Choose an absorber element or absorber atom index before "
            "building a generic EXAFS GDS file."
        )
    preview = load_structure_preview(
        structure_path,
        absorber_element=absorber_element,
        absorber_atom_index=absorber_atom_index,
        min_distance_angstrom=min_distance_angstrom,
        max_distance_angstrom=max_distance_angstrom,
        pair_cutoff_distances_angstrom=pair_cutoff_distances_angstrom,
    )
    return _build_generic_gds_for_structure(
        structure_path,
        ArtemisGDSBuildSettings(
            absorber_element=absorber_element,
            absorber_atom_index=absorber_atom_index,
            min_distance_angstrom=min_distance_angstrom,
            max_distance_angstrom=max_distance_angstrom,
            included_path_pairs=tuple(
                (path.absorber_index, path.scatterer_index)
                for path in preview.paths
                if included_path_pairs is None
                or (path.absorber_index, path.scatterer_index)
                in included_path_pairs
            ),
            shell_tolerance_angstrom=shell_tolerance_angstrom,
            include_restraints=include_restraints,
        ),
    )


def _build_generic_gds_for_structure(
    structure_path: str | Path,
    settings: ArtemisGDSBuildSettings,
) -> ArtemisGDSDocument:
    from .gds import build_artemis_gds_for_structure

    return build_artemis_gds_for_structure(structure_path, settings)


def _representative_option_from_entry(
    entry: RepresentativeSelectionEntry,
) -> EXAFSRepresentativeOption | None:
    source_file = str(entry.source_file or "").strip()
    if not source_file:
        return None
    source_path = Path(source_file).expanduser().resolve()
    if source_path.suffix.lower() not in _STRUCTURE_SUFFIXES:
        return None
    if not source_path.is_file():
        return None
    return EXAFSRepresentativeOption(
        stoichiometry=str(entry.structure).strip(),
        motif=str(entry.motif).strip() or "no_motif",
        source_file=source_path,
        source_file_name=str(entry.source_file_name).strip()
        or source_path.name,
        source_solvent_mode=str(entry.source_solvent_mode).strip()
        or "unknown",
        param=str(entry.param).strip(),
        selected_weight=float(entry.selected_weight),
        cluster_count=int(entry.cluster_count),
        atom_count=int(entry.atom_count),
        element_counts=dict(entry.element_counts),
    )


def _absorber_indices(
    elements: tuple[str, ...],
    *,
    absorber_element: str | None,
    absorber_atom_index: int | None,
) -> tuple[int, ...]:
    if absorber_atom_index is not None:
        index = int(absorber_atom_index)
        if index < 1 or index > len(elements):
            raise ValueError(
                f"Absorber atom index {index} is outside the structure atom "
                f"range 1-{len(elements)}."
            )
        if _is_hydrogen_element(elements[index - 1]):
            raise ValueError(
                "Hydrogen atoms are excluded from EXAFS path generation."
            )
        return (index - 1,)
    normalized_absorber = _normalize_element(
        absorber_element
        or default_absorber_element_from_elements(
            elements,
            preferred_element=_DEFAULT_ABSORBER_ELEMENT,
        )
    )
    if not normalized_absorber:
        raise ValueError(
            "No non-hydrogen absorber atoms were found in the selected "
            "structure."
        )
    if _is_hydrogen_element(normalized_absorber):
        raise ValueError(
            "Hydrogen atoms are excluded from EXAFS path generation."
        )
    indices = tuple(
        index
        for index, element in enumerate(elements)
        if (
            _normalize_element(element) == normalized_absorber
            and not _is_hydrogen_element(element)
        )
    )
    if not indices:
        raise ValueError(
            f"No absorber atoms with element {normalized_absorber!r} were "
            "found in the selected structure."
        )
    return indices


def _normalize_element(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def _is_hydrogen_element(value: object) -> bool:
    return _normalize_element(value) in _HYDROGEN_ELEMENTS


def _default_padded_cif_output_path(
    source_path: Path,
    padding_angstrom: float,
) -> Path:
    suffix = _padding_suffix(padding_angstrom)
    return source_path.with_name(f"{source_path.stem}_padded_{suffix}A.cif")


def _padding_suffix(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _padded_cif_text(
    source_path: Path,
    *,
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    padding_angstrom: float,
) -> str:
    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)
    lengths = maximum - minimum + 2.0 * float(padding_angstrom)
    lengths = np.maximum(lengths, 1.0)
    shifted = coordinates - minimum + float(padding_angstrom)
    fractional = shifted / lengths
    labels = _cif_atom_labels(elements)

    lines = [
        f"data_{_cif_data_block_name(source_path)}",
        "_audit_creation_method 'SAXShell EXAFS padded CIF export'",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_space_group_name_H-M_alt 'P 1'",
        "_space_group_IT_number 1",
        f"_cell_length_a {_cif_float(lengths[0])}",
        f"_cell_length_b {_cif_float(lengths[1])}",
        f"_cell_length_c {_cif_float(lengths[2])}",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 90",
        "_cell_angle_gamma 90",
        "loop_",
        "_space_group_symop_operation_xyz",
        "'x, y, z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    for label, element, values in zip(
        labels, elements, fractional, strict=False
    ):
        lines.append(
            f"{label} {element} "
            f"{_cif_float(values[0])} "
            f"{_cif_float(values[1])} "
            f"{_cif_float(values[2])} 1"
        )
    return "\n".join(lines) + "\n"


def _cif_atom_labels(elements: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for element in elements:
        normalized = _normalize_element(element)
        counts[normalized] = counts.get(normalized, 0) + 1
        labels.append(f"{normalized}{counts[normalized]}")
    return tuple(labels)


def _cif_data_block_name(source_path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", source_path.stem).strip("_")
    return name or "padded_structure"


def _cif_float(value: object) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def _normalized_pair_cutoff_distances(
    pair_cutoffs: PairCutoffDistanceDefinitions | None,
) -> dict[tuple[str, str], float]:
    normalized = {
        (_normalize_element(first), _normalize_element(second)): float(cutoff)
        for (
            first,
            second,
        ), cutoff in _DEFAULT_PAIR_CUTOFF_DISTANCES_ANGSTROM.items()
    }
    if pair_cutoffs is None:
        return normalized
    for pair, values in pair_cutoffs.items():
        if len(pair) != 2:
            continue
        first = _normalize_element(pair[0])
        second = _normalize_element(pair[1])
        if not first or not second:
            continue
        cutoff = _pair_cutoff_value(values)
        key = (first, second)
        if cutoff is None:
            normalized.pop(key, None)
            normalized.pop((second, first), None)
            continue
        normalized[key] = cutoff
    return normalized


def _pair_cutoff_value(
    values: Mapping[int, float] | float | int,
) -> float | None:
    if isinstance(values, Mapping):
        if 0 not in values:
            return None
        raw_value = values[0]
    else:
        raw_value = values
    try:
        cutoff = float(raw_value)
    except (TypeError, ValueError):
        return None
    if cutoff <= 0.0:
        return None
    return cutoff


def _pair_cutoff(
    pair_cutoffs: Mapping[tuple[str, str], float],
    first_element: str,
    second_element: str,
) -> float | None:
    pair = (
        _normalize_element(first_element),
        _normalize_element(second_element),
    )
    reverse_pair = (pair[1], pair[0])
    return pair_cutoffs.get(pair, pair_cutoffs.get(reverse_pair))


def _normalized_included_path_pairs(
    pairs: Iterable[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...] | None:
    if pairs is None:
        return None
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for absorber_index, scatterer_index in pairs:
        pair = (int(absorber_index), int(scatterer_index))
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append(pair)
    return tuple(normalized)


def _absorber_solvent_atom_indices(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    *,
    atom_records: tuple[_StructureAtomRecord, ...],
    absorber_indices: tuple[int, ...],
    included_atom_indices: tuple[int, ...],
    min_distance_angstrom: float,
    max_distance_angstrom: float,
    pair_cutoffs: Mapping[tuple[str, str], float],
) -> dict[int, set[int]]:
    admitted: dict[int, set[int]] = {
        absorber_index: set() for absorber_index in absorber_indices
    }
    solvent_groups = _solvent_atom_groups(
        coordinates,
        elements,
        atom_records=atom_records,
        absorber_indices=absorber_indices,
        included_atom_indices=included_atom_indices,
    )
    for group in solvent_groups:
        donor_indices = tuple(
            atom_index
            for atom_index in group
            if elements[atom_index] in _SOLVENT_DONOR_ELEMENTS
        )
        if not donor_indices:
            continue
        group_atoms = set(group)
        for absorber_index in absorber_indices:
            absorber_element = elements[absorber_index]
            for donor_index in donor_indices:
                cutoff = _pair_cutoff(
                    pair_cutoffs,
                    absorber_element,
                    elements[donor_index],
                )
                if cutoff is None:
                    continue
                donor_distance = float(
                    np.linalg.norm(
                        coordinates[donor_index] - coordinates[absorber_index]
                    )
                )
                if (
                    min_distance_angstrom
                    <= donor_distance
                    <= min(
                        max_distance_angstrom,
                        cutoff,
                    )
                ):
                    admitted[absorber_index].update(group_atoms)
                    break
    return admitted


def _solvent_atom_indices(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    *,
    atom_records: tuple[_StructureAtomRecord, ...],
    absorber_indices: tuple[int, ...],
    included_atom_indices: tuple[int, ...],
) -> set[int]:
    return {
        atom_index
        for group in _solvent_atom_groups(
            coordinates,
            elements,
            atom_records=atom_records,
            absorber_indices=absorber_indices,
            included_atom_indices=included_atom_indices,
        )
        for atom_index in group
    }


def _solvent_atom_groups(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    *,
    atom_records: tuple[_StructureAtomRecord, ...],
    absorber_indices: tuple[int, ...],
    included_atom_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    absorber_set = set(absorber_indices)
    included_set = set(included_atom_indices) - absorber_set
    residue_groups: dict[str, list[int]] = {}
    residue_solvent_indices: set[int] = set()
    for record in atom_records:
        atom_index = record.atom_index - 1
        if atom_index not in included_set:
            continue
        if not record.residue_key or not _is_solvent_record(record):
            continue
        residue_groups.setdefault(record.residue_key, []).append(atom_index)
        residue_solvent_indices.add(atom_index)

    groups = [
        tuple(sorted(indices))
        for indices in residue_groups.values()
        if indices
    ]
    records_by_index = {
        record.atom_index - 1: record for record in atom_records
    }
    fallback_candidates = []
    for index in included_atom_indices:
        if index in absorber_set or index in residue_solvent_indices:
            continue
        record = records_by_index.get(index)
        if record is not None and record.residue_key:
            continue
        fallback_candidates.append(index)
    for component in _covalent_components_for_indices(
        coordinates,
        elements,
        tuple(fallback_candidates),
    ):
        if _is_solvent_like_component(component, elements):
            groups.append(component)
    return tuple(groups)


def _covalent_components_for_indices(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    atom_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    if len(atom_indices) < 2:
        return ()
    adjacency: dict[int, set[int]] = {}
    for offset, atom1_index in enumerate(atom_indices):
        for atom2_index in atom_indices[offset + 1 :]:
            if not _likely_covalent_bond(
                elements[atom1_index],
                elements[atom2_index],
                coordinates[atom1_index],
                coordinates[atom2_index],
            ):
                continue
            adjacency.setdefault(atom1_index, set()).add(atom2_index)
            adjacency.setdefault(atom2_index, set()).add(atom1_index)
    if not adjacency:
        return ()

    components: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for atom_index in sorted(adjacency):
        if atom_index in visited:
            continue
        stack = [atom_index]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(sorted(adjacency.get(current, ()) - visited))
        if len(component) > 1:
            components.append(tuple(sorted(component)))
    return tuple(components)


def _is_solvent_like_component(
    component: tuple[int, ...],
    elements: tuple[str, ...],
) -> bool:
    component_elements = {elements[index] for index in component}
    return "O" in component_elements and bool(
        component_elements & {"C", "N", "S"}
    )


def _dynamic_bond_from_path(path: EXAFSScatteringPath) -> EXAFSBondAnnotation:
    return EXAFSBondAnnotation(
        label=f"R_{{{path.absorber_atom_label}-{path.scatterer_atom_label}}}",
        kind="dynamic",
        atom1_index=path.absorber_index,
        atom2_index=path.scatterer_index,
        atom1_element=path.absorber_element,
        atom2_element=path.scatterer_element,
        atom1_label_text=path.absorber_atom_label,
        atom2_label_text=path.scatterer_atom_label,
        distance_angstrom=path.distance_angstrom,
        start=path.start,
        end=path.end,
    )


def _dynamic_bond_annotations(
    paths: tuple[EXAFSScatteringPath, ...],
    *,
    static_bonds: tuple[EXAFSBondAnnotation, ...],
) -> tuple[EXAFSBondAnnotation, ...]:
    if not paths:
        return ()
    component_by_atom = _static_component_by_atom(static_bonds)
    if not component_by_atom:
        return tuple(_dynamic_bond_from_path(path) for path in paths)

    direct_paths: list[EXAFSScatteringPath] = []
    solvent_component_paths: dict[
        tuple[int, int],
        list[EXAFSScatteringPath],
    ] = {}
    for path in paths:
        component_id = component_by_atom.get(path.scatterer_index)
        if component_id is None:
            direct_paths.append(path)
        else:
            solvent_component_paths.setdefault(
                (path.absorber_index, component_id),
                [],
            ).append(path)

    refined_paths = list(direct_paths)
    for component_paths in solvent_component_paths.values():
        selected_component_paths = _coordinated_solvent_component_paths(
            component_paths,
        )
        refined_paths.extend(selected_component_paths)

    return tuple(
        _dynamic_bond_from_path(path)
        for path in sorted(
            refined_paths,
            key=lambda item: item.distance_angstrom,
        )
    )


def _coordinated_solvent_component_paths(
    component_paths: list[EXAFSScatteringPath],
) -> tuple[EXAFSScatteringPath, ...]:
    if not component_paths:
        return ()
    oxygen_paths = tuple(
        path for path in component_paths if path.scatterer_element == "O"
    )

    minimum_distance = min(path.distance_angstrom for path in component_paths)
    selected_paths = {
        path
        for path in component_paths
        if path.distance_angstrom <= minimum_distance + 0.15
    }
    selected_paths.update(oxygen_paths)
    return tuple(
        sorted(selected_paths, key=lambda item: item.distance_angstrom)
    )


def _static_bonds_for_dynamic_components(
    static_bonds: tuple[EXAFSBondAnnotation, ...],
    *,
    dynamic_bonds: tuple[EXAFSBondAnnotation, ...],
) -> tuple[EXAFSBondAnnotation, ...]:
    if not static_bonds or not dynamic_bonds:
        return ()
    component_by_atom = _static_component_by_atom(static_bonds)
    dynamic_components = {
        component_id
        for bond in dynamic_bonds
        for component_id in (component_by_atom.get(bond.atom2_index),)
        if component_id is not None
    }
    if not dynamic_components:
        return ()
    return tuple(
        bond
        for bond in static_bonds
        if component_by_atom.get(bond.atom1_index) in dynamic_components
        and component_by_atom.get(bond.atom2_index) in dynamic_components
    )


def _static_bond_annotations(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    *,
    atom_labels: tuple[str, ...],
    absorber_indices: tuple[int, ...],
    included_atom_indices: tuple[int, ...],
    atom_records: tuple[_StructureAtomRecord, ...],
    dynamic_paths: tuple[EXAFSScatteringPath, ...],
) -> tuple[EXAFSBondAnnotation, ...]:
    absorber_set = set(absorber_indices)
    candidate_indices = tuple(
        index for index in included_atom_indices if index not in absorber_set
    )
    if not candidate_indices or not dynamic_paths:
        return ()

    residue_keys = {
        record.atom_index - 1: record.residue_key
        for record in atom_records
        if record.residue_key
    }
    groups = _static_bond_candidate_groups(candidate_indices, residue_keys)
    candidate_pairs: list[tuple[int, int]] = []
    for group in groups:
        for offset, atom1_index in enumerate(group):
            for atom2_index in group[offset + 1 :]:
                if _likely_covalent_bond(
                    elements[atom1_index],
                    elements[atom2_index],
                    coordinates[atom1_index],
                    coordinates[atom2_index],
                ):
                    candidate_pairs.append((atom1_index, atom2_index))

    if not candidate_pairs:
        return ()

    dynamic_scatterers = {
        path.scatterer_index - 1 for path in dynamic_paths
    } & set(candidate_indices)
    retained_pairs = _pairs_in_dynamic_components(
        candidate_pairs,
        dynamic_scatterers=dynamic_scatterers,
    )
    annotations: list[EXAFSBondAnnotation] = []
    for atom1_index, atom2_index in retained_pairs:
        atom1 = elements[atom1_index]
        atom2 = elements[atom2_index]
        start = coordinates[atom1_index]
        end = coordinates[atom2_index]
        distance = float(np.linalg.norm(end - start))
        atom1_label = atom_labels[atom1_index]
        atom2_label = atom_labels[atom2_index]
        annotations.append(
            EXAFSBondAnnotation(
                label=f"b_{{{atom1_label}-{atom2_label}}}",
                kind="static",
                atom1_index=atom1_index + 1,
                atom2_index=atom2_index + 1,
                atom1_element=atom1,
                atom2_element=atom2,
                atom1_label_text=atom1_label,
                atom2_label_text=atom2_label,
                distance_angstrom=distance,
                start=tuple(float(value) for value in start),
                end=tuple(float(value) for value in end),
            )
        )
    return tuple(
        sorted(
            annotations,
            key=lambda item: (
                min(item.atom1_index, item.atom2_index),
                max(item.atom1_index, item.atom2_index),
            ),
        )
    )


def _static_bond_candidate_groups(
    candidate_indices: tuple[int, ...],
    residue_keys: Mapping[int, str],
) -> tuple[tuple[int, ...], ...]:
    if not residue_keys:
        return (candidate_indices,)

    grouped: dict[str, list[int]] = {}
    for index in candidate_indices:
        key = residue_keys.get(index)
        if not key:
            continue
        grouped.setdefault(key, []).append(index)
    return tuple(
        tuple(indices) for indices in grouped.values() if len(indices) > 1
    )


def _pairs_in_dynamic_components(
    candidate_pairs: Iterable[tuple[int, int]],
    *,
    dynamic_scatterers: set[int],
) -> tuple[tuple[int, int], ...]:
    pairs = tuple(candidate_pairs)
    if not pairs or not dynamic_scatterers:
        return ()
    adjacency: dict[int, set[int]] = {}
    for atom1_index, atom2_index in pairs:
        adjacency.setdefault(atom1_index, set()).add(atom2_index)
        adjacency.setdefault(atom2_index, set()).add(atom1_index)

    retained_atoms: set[int] = set()
    visited: set[int] = set()
    for start_index in adjacency:
        if start_index in visited:
            continue
        stack = [start_index]
        component: set[int] = set()
        while stack:
            atom_index = stack.pop()
            if atom_index in visited:
                continue
            visited.add(atom_index)
            component.add(atom_index)
            stack.extend(adjacency.get(atom_index, ()) - visited)
        if component & dynamic_scatterers:
            retained_atoms.update(component)
    return tuple(
        (atom1_index, atom2_index)
        for atom1_index, atom2_index in pairs
        if atom1_index in retained_atoms and atom2_index in retained_atoms
    )


def _angle_annotations(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    *,
    atom_labels: tuple[str, ...],
    dynamic_bonds: tuple[EXAFSBondAnnotation, ...],
    static_bonds: tuple[EXAFSBondAnnotation, ...],
) -> tuple[EXAFSAngleAnnotation, ...]:
    static_adjacency: dict[int, set[int]] = {}
    for bond in static_bonds:
        atom1_index = bond.atom1_index - 1
        atom2_index = bond.atom2_index - 1
        static_adjacency.setdefault(atom1_index, set()).add(atom2_index)
        static_adjacency.setdefault(atom2_index, set()).add(atom1_index)

    annotations: list[EXAFSAngleAnnotation] = []
    seen: set[tuple[int, int, int]] = set()
    for bond in dynamic_bonds:
        absorber_index = bond.atom1_index - 1
        bridge_index = bond.atom2_index - 1
        for terminal_index in sorted(static_adjacency.get(bridge_index, ())):
            key = (absorber_index, bridge_index, terminal_index)
            if key in seen:
                continue
            seen.add(key)
            angle_degrees = _angle_degrees(
                coordinates[absorber_index],
                coordinates[bridge_index],
                coordinates[terminal_index],
            )
            atom_triplet_label = (
                f"{atom_labels[absorber_index]}-"
                f"{atom_labels[bridge_index]}-"
                f"{atom_labels[terminal_index]}"
            )
            annotations.append(
                EXAFSAngleAnnotation(
                    label=atom_triplet_label,
                    absorber_index=absorber_index + 1,
                    bridge_index=bridge_index + 1,
                    terminal_index=terminal_index + 1,
                    absorber_element=elements[absorber_index],
                    bridge_element=elements[bridge_index],
                    terminal_element=elements[terminal_index],
                    absorber_atom_label=atom_labels[absorber_index],
                    bridge_atom_label=atom_labels[bridge_index],
                    terminal_atom_label=atom_labels[terminal_index],
                    angle_degrees=angle_degrees,
                    absorber=tuple(
                        float(value) for value in coordinates[absorber_index]
                    ),
                    bridge=tuple(
                        float(value) for value in coordinates[bridge_index]
                    ),
                    terminal=tuple(
                        float(value) for value in coordinates[terminal_index]
                    ),
                )
            )
    for bridge_index in sorted(static_adjacency):
        neighbor_indices = sorted(static_adjacency.get(bridge_index, ()))
        for offset, arm1_index in enumerate(neighbor_indices[:-1]):
            for arm2_index in neighbor_indices[offset + 1 :]:
                key = (arm1_index, bridge_index, arm2_index)
                if key in seen:
                    continue
                seen.add(key)
                angle_degrees = _angle_degrees(
                    coordinates[arm1_index],
                    coordinates[bridge_index],
                    coordinates[arm2_index],
                )
                atom_triplet_label = (
                    f"{atom_labels[arm1_index]}-"
                    f"{atom_labels[bridge_index]}-"
                    f"{atom_labels[arm2_index]}"
                )
                annotations.append(
                    EXAFSAngleAnnotation(
                        label=atom_triplet_label,
                        absorber_index=arm1_index + 1,
                        bridge_index=bridge_index + 1,
                        terminal_index=arm2_index + 1,
                        absorber_element=elements[arm1_index],
                        bridge_element=elements[bridge_index],
                        terminal_element=elements[arm2_index],
                        absorber_atom_label=atom_labels[arm1_index],
                        bridge_atom_label=atom_labels[bridge_index],
                        terminal_atom_label=atom_labels[arm2_index],
                        angle_degrees=angle_degrees,
                        absorber=tuple(
                            float(value) for value in coordinates[arm1_index]
                        ),
                        bridge=tuple(
                            float(value) for value in coordinates[bridge_index]
                        ),
                        terminal=tuple(
                            float(value) for value in coordinates[arm2_index]
                        ),
                    )
                )
    return tuple(
        sorted(
            annotations,
            key=lambda item: (
                item.bridge_index,
                item.absorber_index,
                item.terminal_index,
            ),
        )
    )


def _dihedral_annotations(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    *,
    atom_labels: tuple[str, ...],
    atom_records: tuple[_StructureAtomRecord, ...],
    dynamic_bonds: tuple[EXAFSBondAnnotation, ...],
    static_bonds: tuple[EXAFSBondAnnotation, ...],
) -> tuple[EXAFSDihedralAnnotation, ...]:
    if not dynamic_bonds or not static_bonds:
        return ()
    records_by_index = {
        record.atom_index - 1: record for record in atom_records
    }
    static_adjacency = _static_adjacency(static_bonds)
    annotations: list[EXAFSDihedralAnnotation] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bond in dynamic_bonds:
        absorber_index = bond.atom1_index - 1
        oxygen_index = bond.atom2_index - 1
        if elements[oxygen_index] != "O":
            continue
        oxygen_record = records_by_index.get(oxygen_index)
        if oxygen_record is None or not _is_solvent_record(oxygen_record):
            continue
        residue_key = oxygen_record.residue_key
        if not residue_key:
            continue
        residue_name = oxygen_record.residue_name.upper()
        if residue_name == "DMF":
            _append_dmf_dihedral_annotations(
                annotations,
                seen=seen,
                coordinates=coordinates,
                elements=elements,
                atom_labels=atom_labels,
                records_by_index=records_by_index,
                static_adjacency=static_adjacency,
                absorber_index=absorber_index,
                oxygen_index=oxygen_index,
                residue_key=residue_key,
            )
        elif residue_name in {"DMS", "DMSO"}:
            _append_dmso_dihedral_annotations(
                annotations,
                seen=seen,
                coordinates=coordinates,
                elements=elements,
                atom_labels=atom_labels,
                records_by_index=records_by_index,
                static_adjacency=static_adjacency,
                absorber_index=absorber_index,
                oxygen_index=oxygen_index,
                residue_key=residue_key,
            )
    return tuple(
        sorted(
            annotations,
            key=lambda item: (
                item.atom_indices[1],
                item.atom_indices[2],
                item.atom_indices[0],
                item.atom_indices[3],
            ),
        )
    )


def _append_dmf_dihedral_annotations(
    annotations: list[EXAFSDihedralAnnotation],
    *,
    seen: set[tuple[int, int, int, int]],
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    atom_labels: tuple[str, ...],
    records_by_index: dict[int, _StructureAtomRecord],
    static_adjacency: dict[int, set[int]],
    absorber_index: int,
    oxygen_index: int,
    residue_key: str,
) -> None:
    carbon_indices = _same_residue_static_neighbors(
        oxygen_index,
        elements=elements,
        records_by_index=records_by_index,
        static_adjacency=static_adjacency,
        residue_key=residue_key,
        element="C",
    )
    for carbon_index in carbon_indices:
        nitrogen_indices = _same_residue_static_neighbors(
            carbon_index,
            elements=elements,
            records_by_index=records_by_index,
            static_adjacency=static_adjacency,
            residue_key=residue_key,
            element="N",
        )
        for nitrogen_index in nitrogen_indices:
            _append_dihedral_annotation(
                annotations,
                seen=seen,
                coordinates=coordinates,
                atom_labels=atom_labels,
                atom_indices=(
                    absorber_index,
                    oxygen_index,
                    carbon_index,
                    nitrogen_index,
                ),
                residue_key=residue_key,
            )
            terminal_carbons = _same_residue_static_neighbors(
                nitrogen_index,
                elements=elements,
                records_by_index=records_by_index,
                static_adjacency=static_adjacency,
                residue_key=residue_key,
                element="C",
                excluded={carbon_index},
            )
            for terminal_index in terminal_carbons:
                _append_dihedral_annotation(
                    annotations,
                    seen=seen,
                    coordinates=coordinates,
                    atom_labels=atom_labels,
                    atom_indices=(
                        oxygen_index,
                        carbon_index,
                        nitrogen_index,
                        terminal_index,
                    ),
                    residue_key=residue_key,
                )


def _append_dmso_dihedral_annotations(
    annotations: list[EXAFSDihedralAnnotation],
    *,
    seen: set[tuple[int, int, int, int]],
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    atom_labels: tuple[str, ...],
    records_by_index: dict[int, _StructureAtomRecord],
    static_adjacency: dict[int, set[int]],
    absorber_index: int,
    oxygen_index: int,
    residue_key: str,
) -> None:
    sulfur_indices = _same_residue_static_neighbors(
        oxygen_index,
        elements=elements,
        records_by_index=records_by_index,
        static_adjacency=static_adjacency,
        residue_key=residue_key,
        element="S",
    )
    for sulfur_index in sulfur_indices:
        terminal_carbons = _same_residue_static_neighbors(
            sulfur_index,
            elements=elements,
            records_by_index=records_by_index,
            static_adjacency=static_adjacency,
            residue_key=residue_key,
            element="C",
        )
        for terminal_index in terminal_carbons:
            _append_dihedral_annotation(
                annotations,
                seen=seen,
                coordinates=coordinates,
                atom_labels=atom_labels,
                atom_indices=(
                    absorber_index,
                    oxygen_index,
                    sulfur_index,
                    terminal_index,
                ),
                residue_key=residue_key,
            )


def _append_dihedral_annotation(
    annotations: list[EXAFSDihedralAnnotation],
    *,
    seen: set[tuple[int, int, int, int]],
    coordinates: np.ndarray,
    atom_labels: tuple[str, ...],
    atom_indices: tuple[int, int, int, int],
    residue_key: str | None,
) -> None:
    if atom_indices in seen:
        return
    seen.add(atom_indices)
    angle_degrees = _dihedral_degrees(
        coordinates[atom_indices[0]],
        coordinates[atom_indices[1]],
        coordinates[atom_indices[2]],
        coordinates[atom_indices[3]],
    )
    if angle_degrees is None:
        return
    labels = tuple(atom_labels[index] for index in atom_indices)
    annotations.append(
        EXAFSDihedralAnnotation(
            label="-".join(labels),
            atom_indices=tuple(index + 1 for index in atom_indices),
            plane1_indices=tuple(index + 1 for index in atom_indices[:3]),
            plane2_indices=tuple(index + 1 for index in atom_indices[1:]),
            atom_labels=labels,
            residue_key=residue_key,
            angle_degrees=angle_degrees,
            points=tuple(
                tuple(float(value) for value in coordinates[index])
                for index in atom_indices
            ),
        )
    )


def _same_residue_static_neighbors(
    atom_index: int,
    *,
    elements: tuple[str, ...],
    records_by_index: dict[int, _StructureAtomRecord],
    static_adjacency: dict[int, set[int]],
    residue_key: str,
    element: str,
    excluded: set[int] | None = None,
) -> tuple[int, ...]:
    excluded_indices = excluded or set()
    target_element = _normalize_element(element)
    neighbors: list[int] = []
    for neighbor_index in sorted(static_adjacency.get(atom_index, ())):
        if neighbor_index in excluded_indices:
            continue
        if elements[neighbor_index] != target_element:
            continue
        record = records_by_index.get(neighbor_index)
        if record is None or record.residue_key != residue_key:
            continue
        neighbors.append(neighbor_index)
    return tuple(neighbors)


def _static_adjacency(
    static_bonds: tuple[EXAFSBondAnnotation, ...],
) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for bond in static_bonds:
        atom1_index = bond.atom1_index - 1
        atom2_index = bond.atom2_index - 1
        adjacency.setdefault(atom1_index, set()).add(atom2_index)
        adjacency.setdefault(atom2_index, set()).add(atom1_index)
    return adjacency


def _path_event_bond_edges(
    bonds: tuple[EXAFSBondAnnotation, ...],
) -> dict[tuple[int, int], EXAFSBondAnnotation]:
    edges: dict[tuple[int, int], EXAFSBondAnnotation] = {}
    for bond in bonds:
        key = _path_event_edge_key(bond.atom1_index, bond.atom2_index)
        edges[key] = bond
    return edges


def _path_event_edge_key(
    first_index: int, second_index: int
) -> tuple[int, int]:
    return tuple(sorted((int(first_index), int(second_index))))  # type: ignore[return-value]


def _best_path_event_route(
    start_index: int,
    end_index: int,
    bond_edges: Mapping[tuple[int, int], EXAFSBondAnnotation],
    angles_by_key: Mapping[tuple[int, int, int], EXAFSAngleAnnotation],
    dihedrals_by_key: Mapping[
        tuple[int, int, int, int],
        EXAFSDihedralAnnotation,
    ],
) -> tuple[int, ...] | None:
    routes = _path_event_routes(
        start_index,
        end_index,
        bond_edges,
        max_nodes=6,
    )
    if not routes:
        return None
    return max(
        routes,
        key=lambda route: (
            _path_event_dihedral_count(route, dihedrals_by_key),
            _path_event_angle_count(route, angles_by_key),
            -len(route),
            -_path_event_total_length(
                route,
                bond_edges=bond_edges,
                fallback_distance=0.0,
            ),
        ),
    )


def _path_event_routes(
    start_index: int,
    end_index: int,
    bond_edges: Mapping[tuple[int, int], EXAFSBondAnnotation],
    *,
    max_nodes: int,
) -> tuple[tuple[int, ...], ...]:
    if start_index == end_index:
        return ((start_index,),)
    adjacency: dict[int, set[int]] = {}
    for first_index, second_index in bond_edges:
        adjacency.setdefault(first_index, set()).add(second_index)
        adjacency.setdefault(second_index, set()).add(first_index)
    routes: list[tuple[int, ...]] = []
    queue: list[tuple[int, ...]] = [(start_index,)]
    while queue:
        route = queue.pop(0)
        current = route[-1]
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor in route:
                continue
            next_route = route + (neighbor,)
            if neighbor == end_index:
                routes.append(next_route)
                continue
            if len(next_route) < max_nodes:
                queue.append(next_route)
    return tuple(routes)


def _path_event_angle_count(
    route: tuple[int, ...],
    angles_by_key: Mapping[tuple[int, int, int], EXAFSAngleAnnotation],
) -> int:
    return sum(
        1
        for index in range(max(len(route) - 2, 0))
        if route[index : index + 3] in angles_by_key
    )


def _path_event_dihedral_count(
    route: tuple[int, ...],
    dihedrals_by_key: Mapping[
        tuple[int, int, int, int],
        EXAFSDihedralAnnotation,
    ],
) -> int:
    return sum(
        1
        for index in range(max(len(route) - 3, 0))
        if route[index : index + 4] in dihedrals_by_key
    )


def _path_event_angles(
    angles: tuple[EXAFSAngleAnnotation, ...],
) -> dict[tuple[int, int, int], EXAFSAngleAnnotation]:
    result: dict[tuple[int, int, int], EXAFSAngleAnnotation] = {}
    for angle in angles:
        key = (
            angle.absorber_index,
            angle.bridge_index,
            angle.terminal_index,
        )
        result[key] = angle
        result[(key[2], key[1], key[0])] = angle
    return result


def _path_event_dihedrals(
    dihedrals: tuple[EXAFSDihedralAnnotation, ...],
) -> dict[tuple[int, int, int, int], EXAFSDihedralAnnotation]:
    result: dict[tuple[int, int, int, int], EXAFSDihedralAnnotation] = {}
    for dihedral in dihedrals:
        key = tuple(dihedral.atom_indices)
        result[key] = dihedral
        result[tuple(reversed(key))] = dihedral
    return result


def _path_event_solvent_molecule_group(
    route: tuple[int, ...],
    *,
    atom_labels: tuple[str, ...],
    elements_by_index: Mapping[int, str],
    records_by_index: Mapping[int, _StructureAtomRecord],
) -> tuple[str, str]:
    solvent_records = tuple(
        record
        for atom_index in route
        for record in (records_by_index.get(atom_index),)
        if record is not None and _is_solvent_record(record)
    )
    if solvent_records:
        record = solvent_records[0]
        molecule_key = record.residue_key or f"solvent-{record.atom_index}"
        return (
            f"solvent:{molecule_key}",
            _solvent_molecule_group_label(record),
        )

    if len(route) >= 3:
        donor_index = route[1]
        donor_label = (
            atom_labels[1] if len(atom_labels) > 1 else str(donor_index)
        )
        return (
            f"solvent-like:{donor_index}",
            f"Solvent-like molecule {donor_label}",
        )

    if len(route) >= 2:
        terminal_index = route[-1]
        terminal_element = _normalize_element(
            elements_by_index.get(terminal_index, "")
        )
        if terminal_element in _SOLVENT_DONOR_ELEMENTS:
            terminal_label = (
                atom_labels[-1] if atom_labels else str(terminal_index)
            )
            return (
                f"solvent-like:{terminal_index}",
                f"Solvent-like molecule {terminal_label}",
            )

    return ("direct", "Direct / non-solvent paths")


def _solvent_molecule_group_label(record: _StructureAtomRecord) -> str:
    residue_name = str(record.residue_name or "solvent").strip().upper()
    residue_parts = str(record.residue_key or "").split("|")
    chain_id = ""
    residue_index = ""
    insertion_code = ""
    if len(residue_parts) >= 5:
        chain_id = residue_parts[1].strip()
        residue_index = residue_parts[2].strip()
        insertion_code = residue_parts[3].strip()
    residue_label = ""
    if residue_index:
        residue_label = f"{chain_id}{residue_index}{insertion_code}".strip()
    if not residue_label:
        residue_label = f"atom {record.atom_index}"
    return f"Solvent molecule {residue_name} {residue_label}"


def _path_event_bond_length_labels(
    route: tuple[int, ...],
    *,
    bond_edges: Mapping[tuple[int, int], EXAFSBondAnnotation],
    fallback_path: EXAFSScatteringPath,
) -> tuple[str, ...]:
    if len(route) < 2:
        return ()
    labels: list[str] = []
    for first_index, second_index in zip(route, route[1:], strict=False):
        bond = bond_edges.get(_path_event_edge_key(first_index, second_index))
        if bond is None:
            labels.append(
                f"R_{{{fallback_path.label}}}="
                f"{fallback_path.distance_angstrom:.3f} A"
            )
            continue
        labels.append(f"{bond.label}={bond.distance_angstrom:.3f} A")
    return tuple(labels)


def _path_event_angle_labels(
    route: tuple[int, ...],
    angles_by_key: Mapping[tuple[int, int, int], EXAFSAngleAnnotation],
) -> tuple[str, ...]:
    labels: list[str] = []
    for index in range(max(len(route) - 2, 0)):
        key = route[index : index + 3]
        angle = angles_by_key.get(key)
        if angle is None:
            continue
        labels.append(
            f"{angle.atom_triplet_label}={angle.angle_degrees:.2f} deg"
        )
    return tuple(labels)


def _path_event_dihedral_labels(
    route: tuple[int, ...],
    dihedrals_by_key: Mapping[
        tuple[int, int, int, int],
        EXAFSDihedralAnnotation,
    ],
) -> tuple[str, ...]:
    labels: list[str] = []
    for index in range(max(len(route) - 3, 0)):
        key = route[index : index + 4]
        dihedral = dihedrals_by_key.get(key)
        if dihedral is None:
            continue
        labels.append(f"{dihedral.label}={dihedral.angle_degrees:.2f} deg")
    return tuple(labels)


def _path_event_total_length(
    route: tuple[int, ...],
    *,
    bond_edges: Mapping[tuple[int, int], EXAFSBondAnnotation],
    fallback_distance: float,
) -> float:
    if len(route) < 2:
        return 0.0
    total = 0.0
    for first_index, second_index in zip(route, route[1:], strict=False):
        bond = bond_edges.get(_path_event_edge_key(first_index, second_index))
        if bond is None:
            return float(fallback_distance)
        total += bond.distance_angstrom
    return total


def _static_component_by_atom(
    static_bonds: tuple[EXAFSBondAnnotation, ...],
) -> dict[int, int]:
    adjacency: dict[int, set[int]] = {}
    for bond in static_bonds:
        adjacency.setdefault(bond.atom1_index, set()).add(bond.atom2_index)
        adjacency.setdefault(bond.atom2_index, set()).add(bond.atom1_index)

    component_by_atom: dict[int, int] = {}
    visited: set[int] = set()
    for atom_index in sorted(adjacency):
        if atom_index in visited:
            continue
        component_id = len(component_by_atom) + 1
        stack = [atom_index]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_by_atom[current] = component_id
            stack.extend(sorted(adjacency.get(current, ()) - visited))
    return component_by_atom


def _structure_atom_records(
    path: Path,
    elements: tuple[str, ...],
) -> tuple[_StructureAtomRecord, ...]:
    if path.suffix.lower() != ".pdb":
        return tuple(
            _StructureAtomRecord(
                atom_index=index + 1,
                element=element,
                atom_name=f"{element}{index + 1}",
                residue_name="",
                residue_key=None,
            )
            for index, element in enumerate(elements)
        )
    records = _pdb_atom_records(path)
    if len(records) == len(elements):
        return records
    return tuple(
        _StructureAtomRecord(
            atom_index=index + 1,
            element=element,
            atom_name=f"{element}{index + 1}",
            residue_name="",
            residue_key=None,
        )
        for index, element in enumerate(elements)
    )


def _structure_atom_labels(
    atom_records: tuple[_StructureAtomRecord, ...],
    elements: tuple[str, ...],
) -> tuple[str, ...]:
    if len(atom_records) != len(elements):
        return _element_serial_atom_labels(elements)

    labels: list[str] = []
    element_counts: dict[str, int] = {}
    solvent_molecule_indices: dict[str, int] = {}
    solvent_local_counts: dict[tuple[str, str], int] = {}
    for index, (record, raw_element) in enumerate(
        zip(atom_records, elements, strict=False)
    ):
        element = _normalize_element(record.element or raw_element)
        if not element:
            element = _normalize_element(raw_element)
        if _is_solvent_record(record):
            molecule_key = record.residue_key or f"solvent-{index + 1}"
            molecule_index = solvent_molecule_indices.get(molecule_key)
            if molecule_index is None:
                molecule_index = len(solvent_molecule_indices) + 1
                solvent_molecule_indices[molecule_key] = molecule_index
            local_suffix = _solvent_atom_local_suffix(
                record,
                element=element,
                local_counts=solvent_local_counts,
            )
            labels.append(f"{element}{molecule_index}{local_suffix}")
            continue
        element_counts[element] = element_counts.get(element, 0) + 1
        labels.append(f"{element}{element_counts[element]}")
    return tuple(labels)


def _element_serial_atom_labels(elements: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    labels = []
    for element in elements:
        normalized = _normalize_element(element)
        counts[normalized] = counts.get(normalized, 0) + 1
        labels.append(f"{normalized}{counts[normalized]}")
    return tuple(labels)


def _is_solvent_record(record: _StructureAtomRecord) -> bool:
    return str(record.residue_name).strip().upper() in _SOLVENT_RESIDUE_NAMES


def _solvent_atom_local_suffix(
    record: _StructureAtomRecord,
    *,
    element: str,
    local_counts: dict[tuple[str, str], int],
) -> str:
    atom_name = str(record.atom_name or "").strip()
    trailing_digits = re.search(r"(\d+)$", atom_name)
    if trailing_digits:
        return trailing_digits.group(1)
    residue_key = record.residue_key or f"atom-{record.atom_index}"
    local_key = (residue_key, element)
    local_counts[local_key] = local_counts.get(local_key, 0) + 1
    return str(local_counts[local_key])


def _pdb_atom_records(path: Path) -> tuple[_StructureAtomRecord, ...]:
    records: list[_StructureAtomRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                float(line[30:38].strip())
                float(line[38:46].strip())
                float(line[46:54].strip())
            except Exception:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip()
            if not element:
                element = "".join(
                    char for char in atom_name if char.isalpha()
                )[:2].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            residue_index = line[22:26].strip()
            insertion_code = line[26:27].strip()
            segment_id = line[72:76].strip()
            residue_key = "|".join(
                (
                    segment_id,
                    chain_id,
                    residue_index,
                    insertion_code,
                    residue_name,
                )
            )
            records.append(
                _StructureAtomRecord(
                    atom_index=len(records) + 1,
                    element=_normalize_element(element),
                    atom_name=atom_name,
                    residue_name=residue_name,
                    residue_key=residue_key,
                )
            )
    return tuple(records)


def _likely_covalent_bond(
    element1: str,
    element2: str,
    point1: np.ndarray,
    point2: np.ndarray,
) -> bool:
    distance = float(np.linalg.norm(point2 - point1))
    if distance <= 0.1:
        return False
    radius1 = _COVALENT_RADII_ANGSTROM.get(_normalize_element(element1), 0.8)
    radius2 = _COVALENT_RADII_ANGSTROM.get(_normalize_element(element2), 0.8)
    cutoff = 1.25 * (radius1 + radius2) + 0.15
    return distance <= cutoff


def _angle_degrees(
    atom1: np.ndarray,
    vertex: np.ndarray,
    atom2: np.ndarray,
) -> float:
    vector1 = np.asarray(atom1, dtype=float) - np.asarray(vertex, dtype=float)
    vector2 = np.asarray(atom2, dtype=float) - np.asarray(vertex, dtype=float)
    norm_product = float(np.linalg.norm(vector1) * np.linalg.norm(vector2))
    if norm_product == 0.0:
        return 0.0
    cosine = float(np.dot(vector1, vector2) / norm_product)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _dihedral_degrees(
    atom1: np.ndarray,
    atom2: np.ndarray,
    atom3: np.ndarray,
    atom4: np.ndarray,
) -> float | None:
    point1 = np.asarray(atom1, dtype=float)
    point2 = np.asarray(atom2, dtype=float)
    point3 = np.asarray(atom3, dtype=float)
    point4 = np.asarray(atom4, dtype=float)
    bond1 = point1 - point2
    bond2 = point3 - point2
    bond3 = point4 - point3
    bond2_norm = float(np.linalg.norm(bond2))
    if bond2_norm == 0.0:
        return None
    bond2_unit = bond2 / bond2_norm
    normal1 = bond1 - np.dot(bond1, bond2_unit) * bond2_unit
    normal2 = bond3 - np.dot(bond3, bond2_unit) * bond2_unit
    normal1_norm = float(np.linalg.norm(normal1))
    normal2_norm = float(np.linalg.norm(normal2))
    if normal1_norm == 0.0 or normal2_norm == 0.0:
        return None
    x_value = float(np.dot(normal1, normal2))
    y_value = float(np.dot(np.cross(bond2_unit, normal1), normal2))
    return float(np.degrees(np.arctan2(y_value, x_value)))


def _natural_sort_key(value: object) -> list[object]:
    import re

    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
        if token
    ]


__all__ = [
    "EXAFSAngleAnnotation",
    "EXAFSBondAnalysisResult",
    "EXAFSBondAnnotation",
    "EXAFSDihedralAnnotation",
    "EXAFSRepresentativeOption",
    "EXAFSScatteringPath",
    "EXAFSScatteringPathEvent",
    "EXAFSStructurePreview",
    "build_gds_mapping_document",
    "default_absorber_element",
    "default_absorber_element_from_elements",
    "discover_bondanalysis_results",
    "discover_representative_structures",
    "gds_parameters_from_registry_entries",
    "gds_registry_entries_for_stoichiometry",
    "load_bondanalysis_result",
    "load_structure_preview",
    "scattering_path_events_from_preview",
    "write_gds_mapping_file",
]
