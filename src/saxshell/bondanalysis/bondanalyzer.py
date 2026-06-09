from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

_SOLVENT_RESIDUE_NAMES = frozenset(("DMF", "DMS", "DMSO"))
_DEGENERATE_TERMINAL_BRANCH_LABELS = ("C1", "C2")


def _normalized_element_symbol(value: str) -> str:
    text = re.sub(r"[^A-Za-z]", "", str(value or "")).strip()
    if not text:
        raise ValueError("Element symbols must contain at least one letter.")
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:].lower()


@dataclass(frozen=True, slots=True)
class AtomRecord:
    """Simple atom container used for bond and angle measurements."""

    atom_id: int
    atom_name: str
    residue_name: str
    residue_number: int
    x: float
    y: float
    z: float
    element: str
    chain_id: str = ""
    insertion_code: str = ""


@dataclass(frozen=True, slots=True)
class BondPairDefinition:
    """One requested bond-pair distribution with its cutoff."""

    atom1: str
    atom2: str
    cutoff_angstrom: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom1",
            _normalized_element_symbol(self.atom1),
        )
        object.__setattr__(
            self,
            "atom2",
            _normalized_element_symbol(self.atom2),
        )
        cutoff = float(self.cutoff_angstrom)
        if cutoff <= 0.0:
            raise ValueError("Bond-pair cutoffs must be greater than zero.")
        object.__setattr__(self, "cutoff_angstrom", cutoff)

    @property
    def normalized_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.atom1, self.atom2)))

    @property
    def display_label(self) -> str:
        return f"{self.atom1}-{self.atom2}"

    @property
    def filename_stem(self) -> str:
        return f"{self.atom1}_{self.atom2}"

    def to_dict(self) -> dict[str, object]:
        return {
            "atom1": self.atom1,
            "atom2": self.atom2,
            "cutoff_angstrom": self.cutoff_angstrom,
        }


@dataclass(frozen=True, slots=True)
class AngleTripletDefinition:
    """One requested angle-triplet distribution with cutoffs."""

    vertex: str
    arm1: str
    arm2: str
    cutoff1_angstrom: float
    cutoff2_angstrom: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vertex",
            _normalized_element_symbol(self.vertex),
        )
        object.__setattr__(
            self,
            "arm1",
            _normalized_element_symbol(self.arm1),
        )
        object.__setattr__(
            self,
            "arm2",
            _normalized_element_symbol(self.arm2),
        )
        cutoff1 = float(self.cutoff1_angstrom)
        cutoff2 = float(self.cutoff2_angstrom)
        if cutoff1 <= 0.0 or cutoff2 <= 0.0:
            raise ValueError(
                "Angle-triplet cutoffs must be greater than zero."
            )
        object.__setattr__(self, "cutoff1_angstrom", cutoff1)
        object.__setattr__(self, "cutoff2_angstrom", cutoff2)

    @property
    def display_label(self) -> str:
        return f"{self.arm1}-{self.vertex}-{self.arm2}"

    @property
    def filename_stem(self) -> str:
        return f"{self.vertex}_{self.arm1}_{self.arm2}"

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex": self.vertex,
            "arm1": self.arm1,
            "arm2": self.arm2,
            "cutoff1_angstrom": self.cutoff1_angstrom,
            "cutoff2_angstrom": self.cutoff2_angstrom,
        }


@dataclass(frozen=True, slots=True)
class DihedralQuartetDefinition:
    """One requested signed dihedral distribution with bond cutoffs."""

    atom1: str
    atom2: str
    atom3: str
    atom4: str
    cutoff12_angstrom: float
    cutoff23_angstrom: float
    cutoff34_angstrom: float
    branch_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom1",
            _normalized_element_symbol(self.atom1),
        )
        object.__setattr__(
            self,
            "atom2",
            _normalized_element_symbol(self.atom2),
        )
        object.__setattr__(
            self,
            "atom3",
            _normalized_element_symbol(self.atom3),
        )
        object.__setattr__(
            self,
            "atom4",
            _normalized_element_symbol(self.atom4),
        )
        cutoff12 = float(self.cutoff12_angstrom)
        cutoff23 = float(self.cutoff23_angstrom)
        cutoff34 = float(self.cutoff34_angstrom)
        if cutoff12 <= 0.0 or cutoff23 <= 0.0 or cutoff34 <= 0.0:
            raise ValueError("Dihedral cutoffs must be greater than zero.")
        object.__setattr__(self, "cutoff12_angstrom", cutoff12)
        object.__setattr__(self, "cutoff23_angstrom", cutoff23)
        object.__setattr__(self, "cutoff34_angstrom", cutoff34)
        object.__setattr__(
            self,
            "branch_label",
            str(self.branch_label or "").strip(),
        )

    @property
    def display_label(self) -> str:
        base_label = f"{self.atom1}-{self.atom2}-{self.atom3}-{self.atom4}"
        if not self.branch_label:
            return base_label
        return f"{base_label} (terminal {self.branch_label})"

    @property
    def filename_stem(self) -> str:
        stem = f"{self.atom1}_{self.atom2}_{self.atom3}_{self.atom4}"
        if not self.branch_label:
            return stem
        branch = re.sub(
            r"[^0-9A-Za-z]+",
            "_",
            self.branch_label,
        ).strip("_")
        return f"{stem}_{branch or 'branch'}"

    def with_branch_label(
        self,
        branch_label: str,
    ) -> "DihedralQuartetDefinition":
        return DihedralQuartetDefinition(
            self.atom1,
            self.atom2,
            self.atom3,
            self.atom4,
            self.cutoff12_angstrom,
            self.cutoff23_angstrom,
            self.cutoff34_angstrom,
            branch_label=branch_label,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "atom1": self.atom1,
            "atom2": self.atom2,
            "atom3": self.atom3,
            "atom4": self.atom4,
            "cutoff12_angstrom": self.cutoff12_angstrom,
            "cutoff23_angstrom": self.cutoff23_angstrom,
            "cutoff34_angstrom": self.cutoff34_angstrom,
        }
        if self.branch_label:
            payload["branch_label"] = self.branch_label
        return payload


def expanded_solvent_dihedral_quartets(
    dihedral_quartets: Iterable[DihedralQuartetDefinition],
) -> tuple[DihedralQuartetDefinition, ...]:
    """Add branch-specific series for twofold solvent terminal
    carbons."""

    expanded: list[DihedralQuartetDefinition] = []
    for definition in dihedral_quartets:
        expanded.append(definition)
        if definition.branch_label:
            continue
        if not _is_degenerate_solvent_dihedral_definition(definition):
            continue
        expanded.extend(
            definition.with_branch_label(branch_label)
            for branch_label in _DEGENERATE_TERMINAL_BRANCH_LABELS
        )
    return tuple(dict.fromkeys(expanded))


def _is_dmf_absorber_dihedral_definition(
    definition: DihedralQuartetDefinition,
) -> bool:
    return (
        definition.atom2 == "O"
        and definition.atom3 == "C"
        and definition.atom4 == "N"
    )


def _is_dmf_terminal_dihedral_definition(
    definition: DihedralQuartetDefinition,
) -> bool:
    return (
        definition.atom1 == "O"
        and definition.atom2 == "C"
        and definition.atom3 == "N"
        and definition.atom4 == "C"
    )


def _is_dmso_absorber_dihedral_definition(
    definition: DihedralQuartetDefinition,
) -> bool:
    return (
        definition.atom2 == "O"
        and definition.atom3 == "S"
        and definition.atom4 == "C"
    )


def _is_degenerate_solvent_dihedral_definition(
    definition: DihedralQuartetDefinition,
) -> bool:
    return _is_dmf_terminal_dihedral_definition(
        definition
    ) or _is_dmso_absorber_dihedral_definition(definition)


def _is_solvent_dihedral_definition(
    definition: DihedralQuartetDefinition,
) -> bool:
    return _is_dmf_absorber_dihedral_definition(
        definition
    ) or _is_degenerate_solvent_dihedral_definition(definition)


def _terminal_branch_index(branch_label: str) -> int | None:
    if not branch_label:
        return None
    normalized = branch_label.strip().upper()
    try:
        return _DEGENERATE_TERMINAL_BRANCH_LABELS.index(normalized)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class CoordinationNumberDefinition:
    """One requested first-shell coordination-number distribution."""

    center_atom: str
    neighbor_atom: str
    cutoff_angstrom: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_atom",
            _normalized_element_symbol(self.center_atom),
        )
        object.__setattr__(
            self,
            "neighbor_atom",
            _normalized_element_symbol(self.neighbor_atom),
        )
        cutoff = float(self.cutoff_angstrom)
        if cutoff <= 0.0:
            raise ValueError(
                "Coordination-number cutoffs must be greater than zero."
            )
        object.__setattr__(self, "cutoff_angstrom", cutoff)

    @property
    def display_label(self) -> str:
        return f"CN {self.center_atom}-{self.neighbor_atom}"

    @property
    def filename_stem(self) -> str:
        return f"CN_{self.center_atom}_{self.neighbor_atom}"

    def to_dict(self) -> dict[str, object]:
        return {
            "center_atom": self.center_atom,
            "neighbor_atom": self.neighbor_atom,
            "cutoff_angstrom": self.cutoff_angstrom,
        }


class BondAnalyzer:
    """Measure bond-pair, angle, dihedral, and coordination
    distributions from flat cluster folders.

    The analyzer expects one cluster-type directory to contain single-frame
    ``.pdb`` or ``.xyz`` files directly inside the directory. The higher-level
    workflow is responsible for discovering multiple stoichiometry folders and
    collecting output files.
    """

    structure_suffixes = (".pdb", ".xyz")

    def __init__(
        self,
        bond_pairs: Iterable[BondPairDefinition] | None = None,
        angle_triplets: Iterable[AngleTripletDefinition] | None = None,
        dihedral_quartets: Iterable[DihedralQuartetDefinition] | None = None,
        coordination_numbers: (
            Iterable[CoordinationNumberDefinition] | None
        ) = None,
    ) -> None:
        self.bond_pairs = tuple(self._dedupe_bond_pairs(bond_pairs or ()))
        self.angle_triplets = tuple(dict.fromkeys(angle_triplets or ()))
        self.dihedral_quartets = expanded_solvent_dihedral_quartets(
            dihedral_quartets or ()
        )
        self.coordination_numbers = tuple(
            dict.fromkeys(coordination_numbers or ())
        )

    def structure_files(
        self,
        cluster_dir: str | Path,
        *,
        include_single_atom: bool = False,
    ) -> list[Path]:
        """Return informative structure files directly inside one
        folder.

        Single-atom structures cannot contribute bond, angle, dihedral,
        or coordination information, so they are skipped by default.
        """
        path = Path(cluster_dir)
        return sorted(
            file_path
            for file_path in path.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in self.structure_suffixes
            and (
                include_single_atom
                or self.is_informative_structure_file(file_path)
            )
        )

    def is_informative_structure_file(
        self, structure_file: str | Path
    ) -> bool:
        """Return whether a structure has enough atoms for analysis."""
        atom_count = self._structure_atom_count(Path(structure_file))
        if atom_count is None:
            return True
        return atom_count > 1

    def read_structure(self, structure_file: str | Path) -> list[AtomRecord]:
        path = Path(structure_file)
        if path.suffix.lower() == ".pdb":
            return self._read_pdb(path)
        if path.suffix.lower() == ".xyz":
            return self._read_xyz(path)
        raise ValueError(f"Unsupported structure format: {path.suffix}")

    def measure_structure(
        self,
        structure_file: str | Path,
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
    ]:
        return self.measure_atoms(self.read_structure(structure_file))

    def measure_structure_with_coordination(
        self,
        structure_file: str | Path,
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        return self.measure_atoms_with_coordination(
            self.read_structure(structure_file)
        )

    def measure_structure_with_coordination_and_dihedrals(
        self,
        structure_file: str | Path,
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[DihedralQuartetDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        return self.measure_atoms_with_coordination_and_dihedrals(
            self.read_structure(structure_file)
        )

    def measure_atoms(
        self,
        atoms: list[AtomRecord],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
    ]:
        if not atoms:
            return (
                {definition: [] for definition in self.bond_pairs},
                {definition: [] for definition in self.angle_triplets},
            )
        coords = np.asarray(
            [[atom.x, atom.y, atom.z] for atom in atoms], dtype=float
        )
        elements = [atom.element for atom in atoms]
        return self.measure_structure_data(coords, elements)

    def measure_atoms_with_coordination(
        self,
        atoms: list[AtomRecord],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        (
            bond_values,
            angle_values,
            _dihedral_values,
            coordination_values,
        ) = self.measure_atoms_with_coordination_and_dihedrals(atoms)
        return bond_values, angle_values, coordination_values

    def measure_atoms_with_coordination_and_dihedrals(
        self,
        atoms: list[AtomRecord],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[DihedralQuartetDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        if not atoms:
            return (
                {definition: [] for definition in self.bond_pairs},
                {definition: [] for definition in self.angle_triplets},
                {definition: [] for definition in self.dihedral_quartets},
                {definition: [] for definition in self.coordination_numbers},
            )
        coords = np.asarray(
            [[atom.x, atom.y, atom.z] for atom in atoms], dtype=float
        )
        elements = [atom.element for atom in atoms]
        (
            bond_values,
            angle_values,
            dihedral_values,
            coordination_values,
        ) = self.measure_structure_data_with_coordination_and_dihedrals(
            coords,
            elements,
        )
        solvent_dihedral_values = self._measure_solvent_dihedrals(
            atoms,
            coords,
        )
        for definition, values in solvent_dihedral_values.items():
            dihedral_values[definition] = values
        return (
            bond_values,
            angle_values,
            dihedral_values,
            coordination_values,
        )

    def measure_structure_data(
        self,
        coordinates: np.ndarray,
        elements: Iterable[str],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
    ]:
        bond_values, angle_values, _coordination_values = (
            self.measure_structure_data_with_coordination(
                coordinates,
                elements,
            )
        )
        return bond_values, angle_values

    def measure_structure_data_with_coordination(
        self,
        coordinates: np.ndarray,
        elements: Iterable[str],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        (
            bond_values,
            angle_values,
            _dihedral_values,
            coordination_values,
        ) = self.measure_structure_data_with_coordination_and_dihedrals(
            coordinates,
            elements,
        )
        return bond_values, angle_values, coordination_values

    def measure_structure_data_with_coordination_and_dihedrals(
        self,
        coordinates: np.ndarray,
        elements: Iterable[str],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[DihedralQuartetDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        bond_values = {definition: [] for definition in self.bond_pairs}
        angle_values = {definition: [] for definition in self.angle_triplets}
        dihedral_values = {
            definition: [] for definition in self.dihedral_quartets
        }
        coordination_values = {
            definition: [] for definition in self.coordination_numbers
        }

        coords = np.asarray(coordinates, dtype=float)
        normalized_elements = tuple(
            _normalized_element_symbol(element) for element in elements
        )
        if coords.size == 0 or not normalized_elements:
            return (
                bond_values,
                angle_values,
                dihedral_values,
                coordination_values,
            )
        if coords.ndim != 2 or coords.shape[0] != len(normalized_elements):
            raise ValueError(
                "Coordinates and element symbols must describe the same atoms."
            )

        tree = cKDTree(coords)
        element_array = np.asarray(normalized_elements, dtype=object)

        bond_groups: defaultdict[float, list[BondPairDefinition]] = (
            defaultdict(list)
        )
        for definition in self.bond_pairs:
            bond_groups[float(definition.cutoff_angstrom)].append(definition)
        for cutoff, definitions in bond_groups.items():
            raw_pairs = tree.query_pairs(cutoff)
            if not raw_pairs:
                continue
            pair_indices = np.asarray(list(raw_pairs), dtype=int)
            if pair_indices.size == 0:
                continue
            pair_indices = pair_indices.reshape(-1, 2)
            left_elements = element_array[pair_indices[:, 0]]
            right_elements = element_array[pair_indices[:, 1]]
            distances = np.linalg.norm(
                coords[pair_indices[:, 0]] - coords[pair_indices[:, 1]],
                axis=1,
            )
            for definition in definitions:
                pair_a, pair_b = definition.normalized_pair
                if pair_a == pair_b:
                    mask = (left_elements == pair_a) & (
                        right_elements == pair_b
                    )
                else:
                    mask = (
                        (left_elements == pair_a) & (right_elements == pair_b)
                    ) | (
                        (left_elements == pair_b) & (right_elements == pair_a)
                    )
                if np.any(mask):
                    bond_values[definition].extend(
                        distances[mask].astype(float).tolist()
                    )

        angle_groups: defaultdict[
            tuple[str, float], list[AngleTripletDefinition]
        ] = defaultdict(list)
        for definition in self.angle_triplets:
            angle_groups[
                (
                    definition.vertex,
                    max(
                        float(definition.cutoff1_angstrom),
                        float(definition.cutoff2_angstrom),
                    ),
                )
            ].append(definition)
        for (vertex, max_cutoff), definitions in angle_groups.items():
            center_indices = np.flatnonzero(element_array == vertex)
            if center_indices.size == 0:
                continue
            for center_index in center_indices.tolist():
                neighbor_indices = np.asarray(
                    tree.query_ball_point(coords[center_index], r=max_cutoff),
                    dtype=int,
                )
                if neighbor_indices.size == 0:
                    continue
                neighbor_indices = neighbor_indices[
                    neighbor_indices != center_index
                ]
                if neighbor_indices.size == 0:
                    continue
                neighbor_vectors = (
                    coords[neighbor_indices] - coords[center_index]
                )
                neighbor_distances = np.linalg.norm(neighbor_vectors, axis=1)
                valid_mask = neighbor_distances > 0.0
                if not np.any(valid_mask):
                    continue
                neighbor_elements = element_array[neighbor_indices]
                unit_vectors = np.zeros_like(neighbor_vectors)
                unit_vectors[valid_mask] = (
                    neighbor_vectors[valid_mask]
                    / neighbor_distances[valid_mask, np.newaxis]
                )
                for definition in definitions:
                    arm1_positions = np.flatnonzero(
                        (neighbor_elements == definition.arm1)
                        & (neighbor_distances <= definition.cutoff1_angstrom)
                        & valid_mask
                    )
                    arm2_positions = np.flatnonzero(
                        (neighbor_elements == definition.arm2)
                        & (neighbor_distances <= definition.cutoff2_angstrom)
                        & valid_mask
                    )
                    if arm1_positions.size == 0 or arm2_positions.size == 0:
                        continue
                    if definition.arm1 == definition.arm2:
                        for offset, arm1_position in enumerate(
                            arm1_positions[:-1]
                        ):
                            other_positions = arm1_positions[offset + 1 :]
                            if other_positions.size == 0:
                                continue
                            angles = self._angles_from_unit_vectors(
                                unit_vectors[arm1_position],
                                unit_vectors[other_positions],
                            )
                            angle_values[definition].extend(angles)
                        continue
                    for arm1_position in arm1_positions.tolist():
                        angles = self._angles_from_unit_vectors(
                            unit_vectors[arm1_position],
                            unit_vectors[arm2_positions],
                        )
                        angle_values[definition].extend(angles)

        for definition in self.dihedral_quartets:
            if definition.branch_label:
                continue
            atom2_indices = np.flatnonzero(element_array == definition.atom2)
            if atom2_indices.size == 0:
                continue
            for atom2_index in atom2_indices.tolist():
                atom1_indices = self._neighbor_indices_for_element(
                    tree=tree,
                    coords=coords,
                    element_array=element_array,
                    center_index=atom2_index,
                    target_element=definition.atom1,
                    cutoff=definition.cutoff12_angstrom,
                )
                atom3_indices = self._neighbor_indices_for_element(
                    tree=tree,
                    coords=coords,
                    element_array=element_array,
                    center_index=atom2_index,
                    target_element=definition.atom3,
                    cutoff=definition.cutoff23_angstrom,
                )
                if atom1_indices.size == 0 or atom3_indices.size == 0:
                    continue
                for atom3_index in atom3_indices.tolist():
                    if atom3_index == atom2_index:
                        continue
                    atom4_indices = self._neighbor_indices_for_element(
                        tree=tree,
                        coords=coords,
                        element_array=element_array,
                        center_index=atom3_index,
                        target_element=definition.atom4,
                        cutoff=definition.cutoff34_angstrom,
                    )
                    if atom4_indices.size == 0:
                        continue
                    for atom1_index in atom1_indices.tolist():
                        if atom1_index in (atom2_index, atom3_index):
                            continue
                        for atom4_index in atom4_indices.tolist():
                            if atom4_index in (
                                atom1_index,
                                atom2_index,
                                atom3_index,
                            ):
                                continue
                            angle = self._dihedral_angle(
                                coords[atom1_index],
                                coords[atom2_index],
                                coords[atom3_index],
                                coords[atom4_index],
                            )
                            if angle is not None:
                                dihedral_values[definition].append(angle)

        coordination_groups: defaultdict[
            tuple[str, float], list[CoordinationNumberDefinition]
        ] = defaultdict(list)
        for definition in self.coordination_numbers:
            coordination_groups[
                (definition.center_atom, float(definition.cutoff_angstrom))
            ].append(definition)
        for (center_atom, cutoff), definitions in coordination_groups.items():
            center_indices = np.flatnonzero(element_array == center_atom)
            if center_indices.size == 0:
                continue
            for center_index in center_indices.tolist():
                neighbor_indices = np.asarray(
                    tree.query_ball_point(coords[center_index], r=cutoff),
                    dtype=int,
                )
                if neighbor_indices.size == 0:
                    neighbor_elements = np.asarray((), dtype=object)
                else:
                    neighbor_indices = neighbor_indices[
                        neighbor_indices != center_index
                    ]
                    neighbor_elements = element_array[neighbor_indices]
                for definition in definitions:
                    count = int(
                        np.count_nonzero(
                            neighbor_elements == definition.neighbor_atom
                        )
                    )
                    coordination_values[definition].append(float(count))

        return (
            bond_values,
            angle_values,
            dihedral_values,
            coordination_values,
        )

    def _read_pdb(self, filepath: Path) -> list[AtomRecord]:
        atoms: list[AtomRecord] = []
        with filepath.open() as stream:
            for line in stream:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                atom_name = line[12:16].strip()
                element = (
                    line[76:78].strip()
                    or re.sub(r"[^A-Za-z]", "", atom_name)[:2]
                )
                atoms.append(
                    AtomRecord(
                        atom_id=int(line[6:11]),
                        atom_name=atom_name,
                        residue_name=line[17:20].strip(),
                        residue_number=int(line[22:26] or 0),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                        element=_normalized_element_symbol(element),
                        chain_id=line[21:22].strip(),
                        insertion_code=line[26:27].strip(),
                    )
                )
        return atoms

    def _structure_atom_count(self, filepath: Path) -> int | None:
        suffix = filepath.suffix.lower()
        if suffix == ".pdb":
            return self._pdb_atom_count(filepath)
        if suffix == ".xyz":
            return self._xyz_atom_count(filepath)
        return None

    def _pdb_atom_count(self, filepath: Path) -> int | None:
        count = 0
        try:
            with filepath.open() as stream:
                for line in stream:
                    if line.startswith(("ATOM", "HETATM")):
                        count += 1
                        if count > 1:
                            return count
        except OSError:
            return None
        return count

    def _xyz_atom_count(self, filepath: Path) -> int | None:
        try:
            with filepath.open() as stream:
                first_line = stream.readline().strip()
        except OSError:
            return None
        try:
            return int(first_line)
        except ValueError:
            return None

    def _read_xyz(self, filepath: Path) -> list[AtomRecord]:
        lines = filepath.read_text().splitlines()
        if not lines:
            return []
        atom_count = int(lines[0].strip())
        atoms: list[AtomRecord] = []
        for index, line in enumerate(lines[2 : 2 + atom_count], start=1):
            parts = line.split()
            if len(parts) < 4:
                continue
            atoms.append(
                AtomRecord(
                    atom_id=index,
                    atom_name=parts[0],
                    residue_name="",
                    residue_number=0,
                    x=float(parts[1]),
                    y=float(parts[2]),
                    z=float(parts[3]),
                    element=_normalized_element_symbol(parts[0]),
                )
            )
        return atoms

    def _measure_solvent_dihedrals(
        self,
        atoms: list[AtomRecord],
        coords: np.ndarray,
    ) -> dict[DihedralQuartetDefinition, list[float]]:
        special_definitions = tuple(
            definition
            for definition in self.dihedral_quartets
            if _is_solvent_dihedral_definition(definition)
        )
        if not special_definitions:
            return {}

        solvent_groups = self._solvent_residue_groups(atoms)
        if not solvent_groups:
            return {}

        atom_indices_by_element: defaultdict[str, list[int]] = defaultdict(
            list
        )
        for atom_index, atom in enumerate(atoms):
            atom_indices_by_element[atom.element].append(atom_index)

        results: dict[DihedralQuartetDefinition, list[float]] = {
            definition: [] for definition in special_definitions
        }
        for definition in special_definitions:
            if _is_dmf_absorber_dihedral_definition(definition):
                self._collect_dmf_absorber_dihedrals(
                    atoms=atoms,
                    coords=coords,
                    solvent_groups=solvent_groups,
                    atom_indices_by_element=atom_indices_by_element,
                    definition=definition,
                    values=results[definition],
                )
            elif _is_dmf_terminal_dihedral_definition(definition):
                self._collect_dmf_terminal_dihedrals(
                    atoms=atoms,
                    coords=coords,
                    solvent_groups=solvent_groups,
                    definition=definition,
                    values=results[definition],
                )
            elif _is_dmso_absorber_dihedral_definition(definition):
                self._collect_dmso_absorber_dihedrals(
                    atoms=atoms,
                    coords=coords,
                    solvent_groups=solvent_groups,
                    atom_indices_by_element=atom_indices_by_element,
                    definition=definition,
                    values=results[definition],
                )
        return results

    def _collect_dmf_absorber_dihedrals(
        self,
        *,
        atoms: list[AtomRecord],
        coords: np.ndarray,
        solvent_groups: dict[tuple[str, str, int, str], list[int]],
        atom_indices_by_element: defaultdict[str, list[int]],
        definition: DihedralQuartetDefinition,
        values: list[float],
    ) -> None:
        if definition.branch_label:
            return
        absorber_indices = atom_indices_by_element.get(definition.atom1, [])
        if not absorber_indices:
            return
        for residue_key, residue_indices in solvent_groups.items():
            residue_name = residue_key[0]
            if residue_name != "DMF":
                continue
            for (
                oxygen_index,
                carbon_index,
                nitrogen_index,
            ) in self._dmf_o_c_n_motifs(
                atoms,
                coords,
                residue_indices,
                cutoff_oc=definition.cutoff23_angstrom,
                cutoff_cn=definition.cutoff34_angstrom,
            ):
                for absorber_index in absorber_indices:
                    if absorber_index in (
                        oxygen_index,
                        carbon_index,
                        nitrogen_index,
                    ):
                        continue
                    if (
                        self._distance(coords, absorber_index, oxygen_index)
                        > definition.cutoff12_angstrom
                    ):
                        continue
                    angle = self._dihedral_angle(
                        coords[absorber_index],
                        coords[oxygen_index],
                        coords[carbon_index],
                        coords[nitrogen_index],
                    )
                    if angle is not None:
                        values.append(angle)

    def _collect_dmf_terminal_dihedrals(
        self,
        *,
        atoms: list[AtomRecord],
        coords: np.ndarray,
        solvent_groups: dict[tuple[str, str, int, str], list[int]],
        definition: DihedralQuartetDefinition,
        values: list[float],
    ) -> None:
        branch_index = _terminal_branch_index(definition.branch_label)
        for residue_key, residue_indices in solvent_groups.items():
            residue_name = residue_key[0]
            if residue_name != "DMF":
                continue
            for (
                oxygen_index,
                carbon_index,
                nitrogen_index,
            ) in self._dmf_o_c_n_motifs(
                atoms,
                coords,
                residue_indices,
                cutoff_oc=definition.cutoff12_angstrom,
                cutoff_cn=definition.cutoff23_angstrom,
            ):
                terminal_carbons = [
                    candidate_index
                    for candidate_index in residue_indices
                    if candidate_index != carbon_index
                    and atoms[candidate_index].element == definition.atom4
                    and self._distance(
                        coords,
                        nitrogen_index,
                        candidate_index,
                    )
                    <= definition.cutoff34_angstrom
                ]
                self._append_terminal_branch_dihedrals(
                    atoms=atoms,
                    coords=coords,
                    atom_indices=(
                        oxygen_index,
                        carbon_index,
                        nitrogen_index,
                    ),
                    terminal_indices=terminal_carbons,
                    branch_index=branch_index,
                    values=values,
                )

    def _collect_dmso_absorber_dihedrals(
        self,
        *,
        atoms: list[AtomRecord],
        coords: np.ndarray,
        solvent_groups: dict[tuple[str, str, int, str], list[int]],
        atom_indices_by_element: defaultdict[str, list[int]],
        definition: DihedralQuartetDefinition,
        values: list[float],
    ) -> None:
        absorber_indices = atom_indices_by_element.get(definition.atom1, [])
        if not absorber_indices:
            return
        branch_index = _terminal_branch_index(definition.branch_label)
        for residue_key, residue_indices in solvent_groups.items():
            residue_name = residue_key[0]
            if residue_name not in {"DMS", "DMSO"}:
                continue
            for oxygen_index in self._indices_with_element(
                atoms,
                residue_indices,
                definition.atom2,
            ):
                sulfur_indices = [
                    sulfur_index
                    for sulfur_index in self._indices_with_element(
                        atoms,
                        residue_indices,
                        definition.atom3,
                    )
                    if self._distance(coords, oxygen_index, sulfur_index)
                    <= definition.cutoff23_angstrom
                ]
                for sulfur_index in sulfur_indices:
                    terminal_carbons = [
                        candidate_index
                        for candidate_index in residue_indices
                        if atoms[candidate_index].element == definition.atom4
                        and self._distance(
                            coords,
                            sulfur_index,
                            candidate_index,
                        )
                        <= definition.cutoff34_angstrom
                    ]
                    for absorber_index in absorber_indices:
                        if absorber_index in (
                            oxygen_index,
                            sulfur_index,
                        ):
                            continue
                        if (
                            self._distance(
                                coords,
                                absorber_index,
                                oxygen_index,
                            )
                            > definition.cutoff12_angstrom
                        ):
                            continue
                        self._append_terminal_branch_dihedrals(
                            atoms=atoms,
                            coords=coords,
                            atom_indices=(
                                absorber_index,
                                oxygen_index,
                                sulfur_index,
                            ),
                            terminal_indices=terminal_carbons,
                            branch_index=branch_index,
                            values=values,
                        )

    def _append_terminal_branch_dihedrals(
        self,
        *,
        atoms: list[AtomRecord],
        coords: np.ndarray,
        atom_indices: tuple[int, int, int],
        terminal_indices: list[int],
        branch_index: int | None,
        values: list[float],
    ) -> None:
        sorted_terminal_indices = self._sorted_residue_indices(
            atoms,
            terminal_indices,
        )
        if branch_index is not None:
            if branch_index >= len(sorted_terminal_indices):
                return
            sorted_terminal_indices = [sorted_terminal_indices[branch_index]]
        for terminal_index in sorted_terminal_indices:
            if terminal_index in atom_indices:
                continue
            angle = self._dihedral_angle(
                coords[atom_indices[0]],
                coords[atom_indices[1]],
                coords[atom_indices[2]],
                coords[terminal_index],
            )
            if angle is not None:
                values.append(angle)

    def _dmf_o_c_n_motifs(
        self,
        atoms: list[AtomRecord],
        coords: np.ndarray,
        residue_indices: list[int],
        *,
        cutoff_oc: float,
        cutoff_cn: float,
    ) -> list[tuple[int, int, int]]:
        motifs: list[tuple[int, int, int]] = []
        oxygen_indices = self._indices_with_element(
            atoms, residue_indices, "O"
        )
        carbon_indices = self._indices_with_element(
            atoms, residue_indices, "C"
        )
        nitrogen_indices = self._indices_with_element(
            atoms,
            residue_indices,
            "N",
        )
        for oxygen_index in oxygen_indices:
            for carbon_index in carbon_indices:
                if (
                    self._distance(coords, oxygen_index, carbon_index)
                    > cutoff_oc
                ):
                    continue
                for nitrogen_index in nitrogen_indices:
                    if (
                        self._distance(coords, carbon_index, nitrogen_index)
                        > cutoff_cn
                    ):
                        continue
                    motifs.append((oxygen_index, carbon_index, nitrogen_index))
        return motifs

    @staticmethod
    def _indices_with_element(
        atoms: list[AtomRecord],
        indices: Iterable[int],
        element: str,
    ) -> list[int]:
        return [
            index
            for index in indices
            if atoms[index].element == _normalized_element_symbol(element)
        ]

    @staticmethod
    def _solvent_residue_groups(
        atoms: list[AtomRecord],
    ) -> dict[tuple[str, str, int, str], list[int]]:
        groups: defaultdict[
            tuple[str, str, int, str],
            list[int],
        ] = defaultdict(list)
        for atom_index, atom in enumerate(atoms):
            residue_name = atom.residue_name.upper()
            if residue_name not in _SOLVENT_RESIDUE_NAMES:
                continue
            groups[
                (
                    residue_name,
                    atom.chain_id,
                    atom.residue_number,
                    atom.insertion_code,
                )
            ].append(atom_index)
        return dict(groups)

    @staticmethod
    def _sorted_residue_indices(
        atoms: list[AtomRecord],
        indices: Iterable[int],
    ) -> list[int]:
        def sort_key(index: int) -> tuple[str, int, int]:
            atom = atoms[index]
            return (atom.atom_name, atom.atom_id, index)

        return sorted(indices, key=sort_key)

    @staticmethod
    def _distance(
        coords: np.ndarray,
        index1: int,
        index2: int,
    ) -> float:
        return float(np.linalg.norm(coords[index1] - coords[index2]))

    @staticmethod
    def _angle_between(
        vector1: np.ndarray, vector2: np.ndarray
    ) -> float | None:
        norm1 = float(np.linalg.norm(vector1))
        norm2 = float(np.linalg.norm(vector2))
        if norm1 == 0.0 or norm2 == 0.0:
            return None
        cosine = float(np.dot(vector1, vector2) / (norm1 * norm2))
        return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))

    @staticmethod
    def _angles_from_unit_vectors(
        vector: np.ndarray,
        other_vectors: np.ndarray,
    ) -> list[float]:
        vectors = np.asarray(other_vectors, dtype=float)
        if vectors.size == 0:
            return []
        dots = np.clip(vectors @ np.asarray(vector, dtype=float), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        return np.asarray(angles, dtype=float).tolist()

    @staticmethod
    def _neighbor_indices_for_element(
        *,
        tree: cKDTree,
        coords: np.ndarray,
        element_array: np.ndarray,
        center_index: int,
        target_element: str,
        cutoff: float,
    ) -> np.ndarray:
        neighbor_indices = np.asarray(
            tree.query_ball_point(coords[center_index], r=float(cutoff)),
            dtype=int,
        )
        if neighbor_indices.size == 0:
            return neighbor_indices
        return neighbor_indices[
            (neighbor_indices != center_index)
            & (element_array[neighbor_indices] == target_element)
        ]

    @staticmethod
    def _dihedral_angle(
        point1: np.ndarray,
        point2: np.ndarray,
        point3: np.ndarray,
        point4: np.ndarray,
    ) -> float | None:
        p1 = np.asarray(point1, dtype=float)
        p2 = np.asarray(point2, dtype=float)
        p3 = np.asarray(point3, dtype=float)
        p4 = np.asarray(point4, dtype=float)
        bond1 = p1 - p2
        bond2 = p3 - p2
        bond3 = p4 - p3
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
        return float(math.degrees(math.atan2(y_value, x_value)))

    @staticmethod
    def _dedupe_bond_pairs(
        bond_pairs: Iterable[BondPairDefinition],
    ) -> list[BondPairDefinition]:
        deduped: list[BondPairDefinition] = []
        seen: set[tuple[tuple[str, str], float]] = set()
        for definition in bond_pairs:
            key = (definition.normalized_pair, definition.cutoff_angstrom)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(definition)
        return deduped


__all__ = [
    "AngleTripletDefinition",
    "AtomRecord",
    "BondAnalyzer",
    "BondPairDefinition",
    "CoordinationNumberDefinition",
    "DihedralQuartetDefinition",
    "expanded_solvent_dihedral_quartets",
]
