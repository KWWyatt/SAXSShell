from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from saxshell.fullrmc.packmol_planning import PackmolPlanningMetadata
from saxshell.fullrmc.representatives import (
    RepresentativeSelectionEntry,
    RepresentativeSelectionMetadata,
    validate_representative_selection_covers_distribution,
)
from saxshell.fullrmc.solvent_handling import (
    SolventHandlingMetadata,
    representative_structure_mode_label,
    resolved_representative_structure_mode,
)
from saxshell.saxs.debye import load_structure_file
from saxshell.saxs.stoichiometry import parse_stoich_label
from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb import resolve_reference_path

if False:  # pragma: no cover
    from .project_loader import RMCDreamProjectSource


_PACKMOL_SOLUTE_MATCH_TOLERANCE_A = 0.05


@dataclass(slots=True)
class PackmolSetupSettings:
    tolerance_angstrom: float = 2.0
    output_filename: str = "packmol_combined.inp"
    packed_output_filename: str = "packed_combined.pdb"
    use_completed_representatives: bool = True
    include_free_solvent: bool = True
    free_solvent_reference: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "PackmolSetupSettings":
        source = dict(payload or {})
        return cls(
            tolerance_angstrom=max(
                0.1,
                float(source.get("tolerance_angstrom", 2.0)),
            ),
            output_filename=_safe_filename(
                str(source.get("output_filename", "packmol_combined.inp"))
            ),
            packed_output_filename=_safe_filename(
                str(
                    source.get(
                        "packed_output_filename",
                        "packed_combined.pdb",
                    )
                )
            ),
            use_completed_representatives=bool(
                source.get("use_completed_representatives", True)
            ),
            include_free_solvent=bool(
                source.get("include_free_solvent", True)
            ),
            free_solvent_reference=_optional_text(
                source.get("free_solvent_reference")
            ),
        )


@dataclass(slots=True)
class PackmolSetupEntry:
    structure: str
    motif: str
    param: str
    planned_count: int
    selected_weight: float
    planned_count_weight: float
    planned_atom_weight: float
    residue_name: str
    source_pdb: str
    packmol_pdb: str
    atom_count: int
    solute_atom_count: int = 0
    solvent_atom_count: int = 0
    solvent_residue_count: int = 0
    solvent_residue_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "PackmolSetupEntry":
        return cls(
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            param=str(payload.get("param", "")).strip(),
            planned_count=int(payload.get("planned_count", 0)),
            selected_weight=float(payload.get("selected_weight", 0.0)),
            planned_count_weight=float(
                payload.get("planned_count_weight", 0.0)
            ),
            planned_atom_weight=float(payload.get("planned_atom_weight", 0.0)),
            residue_name=str(payload.get("residue_name", "")).strip(),
            source_pdb=str(payload.get("source_pdb", "")).strip(),
            packmol_pdb=str(payload.get("packmol_pdb", "")).strip(),
            atom_count=int(payload.get("atom_count", 0)),
            solute_atom_count=int(payload.get("solute_atom_count", 0)),
            solvent_atom_count=int(payload.get("solvent_atom_count", 0)),
            solvent_residue_count=int(payload.get("solvent_residue_count", 0)),
            solvent_residue_names=tuple(
                str(value).strip()
                for value in payload.get("solvent_residue_names", [])
                if str(value).strip()
            ),
        )


@dataclass(slots=True)
class PackmolSetupSupplementalEntry:
    role: str
    name: str
    source_type: str
    reference_name: str | None
    reference_path: str | None
    residue_name: str
    planned_count: int
    atom_count: int
    element_counts: dict[str, int]
    packmol_pdb: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "PackmolSetupSupplementalEntry":
        return cls(
            role=str(payload.get("role", "solute") or "solute").strip(),
            name=str(payload.get("name", "") or "").strip(),
            source_type=str(payload.get("source_type", "") or "").strip(),
            reference_name=_optional_text(payload.get("reference_name")),
            reference_path=_optional_text(payload.get("reference_path")),
            residue_name=_normalized_residue_name(
                str(payload.get("residue_name", "") or "")
            ),
            planned_count=int(payload.get("planned_count", 0)),
            atom_count=int(payload.get("atom_count", 0)),
            element_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("element_counts", {})
                ).items()
            },
            packmol_pdb=str(payload.get("packmol_pdb", "") or "").strip(),
        )


@dataclass(slots=True)
class PackmolSetupMetadata:
    settings: PackmolSetupSettings
    updated_at: str
    planning_mode: str
    representative_selection_mode: str
    representative_structure_mode: str
    box_side_length_a: float
    packmol_input_path: str
    packed_output_filename: str
    solvent_pdb_path: str | None
    free_solvent_reference_name: str | None
    free_solvent_reference_path: str | None
    target_solvent_molecules: int
    solvent_molecules_in_clusters: int
    free_solvent_molecules: int
    audit_report_path: str
    build_report_path: str
    entries: list[PackmolSetupEntry]
    supplemental_entries: list[PackmolSetupSupplementalEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "settings": self.settings.to_dict(),
            "updated_at": self.updated_at,
            "planning_mode": self.planning_mode,
            "representative_selection_mode": (
                self.representative_selection_mode
            ),
            "representative_structure_mode": (
                self.representative_structure_mode
            ),
            "box_side_length_a": self.box_side_length_a,
            "packmol_input_path": self.packmol_input_path,
            "packed_output_filename": self.packed_output_filename,
            "solvent_pdb_path": self.solvent_pdb_path,
            "free_solvent_reference_name": self.free_solvent_reference_name,
            "free_solvent_reference_path": self.free_solvent_reference_path,
            "target_solvent_molecules": self.target_solvent_molecules,
            "solvent_molecules_in_clusters": (
                self.solvent_molecules_in_clusters
            ),
            "free_solvent_molecules": self.free_solvent_molecules,
            "audit_report_path": self.audit_report_path,
            "build_report_path": self.build_report_path,
            "entries": [entry.to_dict() for entry in self.entries],
            "supplemental_entries": [
                entry.to_dict() for entry in self.supplemental_entries
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "PackmolSetupMetadata | None":
        if not payload:
            return None
        return cls(
            settings=PackmolSetupSettings.from_dict(
                payload.get("settings")
                if isinstance(payload.get("settings"), dict)
                else None
            ),
            updated_at=str(payload.get("updated_at", "")).strip(),
            planning_mode=str(payload.get("planning_mode", "")).strip(),
            representative_selection_mode=str(
                payload.get("representative_selection_mode", "")
            ).strip(),
            representative_structure_mode=str(
                payload.get("representative_structure_mode", "")
            ).strip(),
            box_side_length_a=float(payload.get("box_side_length_a", 0.0)),
            packmol_input_path=str(
                payload.get("packmol_input_path", "")
            ).strip(),
            packed_output_filename=str(
                payload.get("packed_output_filename", "")
            ).strip(),
            solvent_pdb_path=_optional_text(payload.get("solvent_pdb_path")),
            free_solvent_reference_name=_optional_text(
                payload.get("free_solvent_reference_name")
            ),
            free_solvent_reference_path=_optional_text(
                payload.get("free_solvent_reference_path")
            ),
            target_solvent_molecules=int(
                payload.get("target_solvent_molecules", 0)
            ),
            solvent_molecules_in_clusters=int(
                payload.get("solvent_molecules_in_clusters", 0)
            ),
            free_solvent_molecules=int(
                payload.get("free_solvent_molecules", 0)
            ),
            audit_report_path=str(
                payload.get("audit_report_path", "")
            ).strip(),
            build_report_path=str(
                payload.get(
                    "build_report_path",
                    payload.get("audit_report_path", ""),
                )
            ).strip(),
            entries=[
                PackmolSetupEntry.from_dict(dict(entry))
                for entry in payload.get("entries", [])
                if isinstance(entry, dict)
            ],
            supplemental_entries=[
                PackmolSetupSupplementalEntry.from_dict(dict(entry))
                for entry in payload.get("supplemental_entries", [])
                if isinstance(entry, dict)
            ],
        )

    def summary_text(self) -> str:
        lines = [
            f"Planning mode: {self.planning_mode}",
            f"Representative mode: {self.representative_selection_mode}",
            (
                "Representative structure set: "
                f"{representative_structure_mode_label(self.representative_structure_mode)}"
            ),
            f"Saved at: {self.updated_at}",
            f"Box side: {self.box_side_length_a:.3f} A",
            f"Packmol tolerance: {self.settings.tolerance_angstrom:.3f} A",
            f"Packmol input: {Path(self.packmol_input_path).name}",
            f"Build report: {Path(self.build_report_path).name}",
            f"Representative PDBs copied: {len(self.entries)}",
        ]
        supplemental_count = sum(
            entry.planned_count for entry in self.supplemental_entries
        )
        if supplemental_count > 0:
            lines.append(
                f"Supplemental solute components: {supplemental_count}"
            )
        if self.free_solvent_reference_name:
            lines.append(
                "Free solvent structure: "
                f"{self.free_solvent_reference_name}"
            )
        lines.extend(
            [
                f"Total solvent molecules: {self.target_solvent_molecules}",
                (
                    "Cluster solvent molecules: "
                    f"{self.solvent_molecules_in_clusters}"
                ),
                f"Free solvent molecules: {self.free_solvent_molecules}",
            ]
        )
        if self.entries:
            first = self.entries[0]
            lines.extend(
                [
                    "",
                    "Example Packmol structure:",
                    f"  {first.structure}/{first.motif}",
                    f"  residue: {first.residue_name}",
                    f"  count: {first.planned_count}",
                    f"  file: {Path(first.packmol_pdb).name}",
                ]
            )
        if self.solvent_pdb_path:
            lines.append(f"Solvent input: {Path(self.solvent_pdb_path).name}")
        return "\n".join(lines)


@dataclass(slots=True)
class _PreparedPackmolStructure:
    structure: PDBStructure
    solute_atom_count: int
    solvent_atom_count: int
    solvent_residue_count: int
    solvent_residue_names: tuple[str, ...]


def build_packmol_setup(
    project_source: "RMCDreamProjectSource",
    settings: PackmolSetupSettings | None = None,
    *,
    plan_metadata: PackmolPlanningMetadata | None = None,
    representative_metadata: RepresentativeSelectionMetadata | None = None,
    solvent_metadata: SolventHandlingMetadata | None = None,
) -> PackmolSetupMetadata:
    active_settings = settings or PackmolSetupSettings()
    active_plan = plan_metadata or project_source.packmol_planning
    if active_plan is None or not active_plan.entries:
        raise ValueError(
            "Compute Packmol planning counts before building Packmol setup."
        )
    active_representatives = (
        representative_metadata or project_source.representative_selection
    )
    if (
        active_representatives is None
        or not active_representatives.representative_entries
    ):
        raise ValueError(
            "Save representative structures before building Packmol setup."
        )
    validate_representative_selection_covers_distribution(
        active_representatives
    )

    active_solvent = solvent_metadata or project_source.solvent_handling
    free_solvent_reference_name: str | None = None
    free_solvent_reference_path: str | None = None
    if active_settings.include_free_solvent:
        (
            free_solvent_reference_name,
            free_solvent_reference_path,
        ) = _resolve_free_solvent_reference(
            active_settings,
            active_plan,
            active_solvent,
        )
        if free_solvent_reference_path is None:
            raise ValueError(
                "Choose a free-solvent structure before generating Packmol inputs."
            )

    representative_lookup = {
        (entry.structure, entry.motif, entry.param): entry
        for entry in active_representatives.representative_entries
    }
    solvent_lookup = {}
    if active_solvent is not None:
        solvent_lookup = {
            (entry.structure, entry.motif, entry.param): entry
            for entry in active_solvent.entries
        }
    known_solvent_residue_names = _solvent_residue_names_for_packmol_source(
        active_solvent,
        free_solvent_reference_path=free_solvent_reference_path,
    )
    representative_structure_mode = resolved_representative_structure_mode(
        active_representatives,
        active_solvent,
    )

    entries: list[PackmolSetupEntry] = []
    box_side_length_a = active_plan.settings.box_side_length_a
    for index, plan_entry in enumerate(active_plan.entries):
        if plan_entry.planned_count <= 0:
            continue
        key = (plan_entry.structure, plan_entry.motif, plan_entry.param)
        representative_entry = representative_lookup.get(key)
        if representative_entry is None:
            raise ValueError(
                "Packmol planning referenced a cluster bin without a representative: "
                f"{plan_entry.structure}/{plan_entry.motif}"
            )
        solvent_entry = solvent_lookup.get(key)
        source_structure, source_pdb_path = _resolve_structure_for_packmol(
            representative_entry,
            solvent_entry,
            representative_structure_mode=representative_structure_mode,
            use_completed=active_settings.use_completed_representatives,
        )
        residue_name = _packmol_residue_code(index)
        packmol_filename = (
            f"{index + 1:03d}_"
            f"{_safe_name(plan_entry.structure)}_"
            f"{_safe_name(plan_entry.motif)}_"
            f"{residue_name}.pdb"
        )
        packmol_path = (
            project_source.rmcsetup_paths.packmol_inputs_dir / packmol_filename
        )
        prepared_structure = _prepare_packmol_structure(
            source_structure,
            residue_name=residue_name,
            solvent_residue_names=known_solvent_residue_names,
            solute_reference_structure=(
                _solute_reference_structure_for_packmol_source(solvent_entry)
            ),
            expected_solute_element_counts=parse_stoich_label(
                plan_entry.structure
            ),
            solute_atom_count=_solute_atom_count_for_packmol_source(
                solvent_entry
            ),
        )
        prepared_structure.structure.write_pdb_file(packmol_path)
        entries.append(
            PackmolSetupEntry(
                structure=plan_entry.structure,
                motif=plan_entry.motif,
                param=plan_entry.param,
                planned_count=plan_entry.planned_count,
                selected_weight=plan_entry.selected_weight,
                planned_count_weight=plan_entry.planned_count_weight,
                planned_atom_weight=plan_entry.planned_atom_weight,
                residue_name=residue_name,
                source_pdb=str(source_pdb_path),
                packmol_pdb=str(packmol_path),
                atom_count=len(prepared_structure.structure.atoms),
                solute_atom_count=prepared_structure.solute_atom_count,
                solvent_atom_count=prepared_structure.solvent_atom_count,
                solvent_residue_count=(
                    prepared_structure.solvent_residue_count
                ),
                solvent_residue_names=(
                    prepared_structure.solvent_residue_names
                ),
            )
        )

    if not entries:
        raise ValueError(
            "The current Packmol plan did not produce any cluster entries with positive counts."
        )

    solvent_pdb_path: str | None = None
    solvent_allocation = active_plan.solvent_allocation
    target_solvent_molecules = int(
        round(
            float(
                active_plan.target_box_composition.get("solvent_molecules", 0)
            )
        )
    )
    solvent_molecules_in_clusters = 0
    free_solvent_molecules = target_solvent_molecules
    if solvent_allocation is not None:
        target_solvent_molecules = int(
            solvent_allocation.target_solvent_molecules
        )
        solvent_molecules_in_clusters = int(
            solvent_allocation.solvent_molecules_in_clusters
        )
        free_solvent_molecules = int(solvent_allocation.free_solvent_molecules)
    if (
        active_settings.include_free_solvent
        and free_solvent_reference_path is not None
    ):
        source_solvent = (
            Path(free_solvent_reference_path).expanduser().resolve()
        )
        solvent_copy_name = f"{_safe_name(free_solvent_reference_name or source_solvent.stem)}_single.pdb"
        destination = (
            project_source.rmcsetup_paths.packmol_inputs_dir
            / solvent_copy_name
        )
        shutil.copy2(source_solvent, destination)
        solvent_pdb_path = str(destination)

    supplemental_entries = _write_supplemental_packmol_structures(
        project_source.rmcsetup_paths.packmol_inputs_dir,
        active_plan,
    )

    input_path = _write_packmol_input(
        project_source.rmcsetup_paths.packmol_inputs_dir,
        entries,
        supplemental_entries=supplemental_entries,
        solvent_pdb_path=solvent_pdb_path,
        free_solvent_molecules=free_solvent_molecules,
        box_side_length_a=box_side_length_a,
        settings=active_settings,
    )
    audit_path = _write_packmol_audit_report(
        project_source,
        active_plan,
        entries,
        input_path=input_path,
        solvent_pdb_path=solvent_pdb_path,
        free_solvent_reference_name=free_solvent_reference_name,
        free_solvent_reference_path=free_solvent_reference_path,
        target_solvent_molecules=target_solvent_molecules,
        solvent_molecules_in_clusters=solvent_molecules_in_clusters,
        free_solvent_molecules=free_solvent_molecules,
        supplemental_entries=supplemental_entries,
    )
    build_report_path = _write_packmol_build_report(
        project_source,
        active_plan,
        entries,
        input_path=input_path,
        solvent_pdb_path=solvent_pdb_path,
        free_solvent_reference_name=free_solvent_reference_name,
        free_solvent_reference_path=free_solvent_reference_path,
        target_solvent_molecules=target_solvent_molecules,
        solvent_molecules_in_clusters=solvent_molecules_in_clusters,
        free_solvent_molecules=free_solvent_molecules,
        representative_structure_mode=representative_structure_mode,
        representative_selection_mode=active_representatives.selection_mode,
        settings=active_settings,
        supplemental_entries=supplemental_entries,
    )
    metadata = PackmolSetupMetadata(
        settings=active_settings,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        planning_mode=active_plan.settings.planning_mode,
        representative_selection_mode=active_representatives.selection_mode,
        representative_structure_mode=representative_structure_mode,
        box_side_length_a=box_side_length_a,
        packmol_input_path=str(input_path),
        packed_output_filename=active_settings.packed_output_filename,
        solvent_pdb_path=solvent_pdb_path,
        free_solvent_reference_name=free_solvent_reference_name,
        free_solvent_reference_path=free_solvent_reference_path,
        target_solvent_molecules=target_solvent_molecules,
        solvent_molecules_in_clusters=solvent_molecules_in_clusters,
        free_solvent_molecules=free_solvent_molecules,
        audit_report_path=str(audit_path),
        build_report_path=str(build_report_path),
        entries=entries,
        supplemental_entries=supplemental_entries,
    )
    save_packmol_setup_metadata(
        project_source.rmcsetup_paths.packmol_setup_path,
        metadata,
    )
    return metadata


def save_packmol_setup_metadata(
    output_path: str | Path,
    metadata: PackmolSetupMetadata,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_packmol_setup_metadata(
    metadata_path: str | Path,
) -> PackmolSetupMetadata | None:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return PackmolSetupMetadata.from_dict(payload)


def _resolve_structure_for_packmol(
    representative_entry: RepresentativeSelectionEntry,
    solvent_entry: object | None,
    *,
    representative_structure_mode: str,
    use_completed: bool,
) -> tuple[PDBStructure, Path]:
    candidate_paths: list[Path] = []
    if solvent_entry is not None:
        completed_path = Path(
            getattr(solvent_entry, "completed_pdb", "")
        ).expanduser()
        no_solvent_path = Path(
            getattr(solvent_entry, "no_solvent_pdb", "")
        ).expanduser()
        if representative_structure_mode == "full_solvent":
            candidate_paths.extend([completed_path, no_solvent_path])
        elif representative_structure_mode == "no_solvent":
            candidate_paths.extend([no_solvent_path, completed_path])
        elif representative_structure_mode == "partial_solvent":
            candidate_paths.extend(
                [Path(representative_entry.source_file).expanduser()]
            )
        elif use_completed:
            candidate_paths.extend([completed_path, no_solvent_path])
        else:
            candidate_paths.extend([no_solvent_path, completed_path])
    candidate_paths.append(Path(representative_entry.source_file).expanduser())
    source_path = Path(representative_entry.source_file).expanduser().resolve()
    for candidate_path in candidate_paths:
        if str(candidate_path).strip() and candidate_path.is_file():
            resolved = candidate_path.resolve()
            if resolved == source_path:
                return (
                    _load_structure_as_pdb(
                        resolved,
                        structure_label=representative_entry.structure,
                    ),
                    resolved,
                )
            return PDBStructure.from_file(resolved), resolved
    return (
        _load_structure_as_pdb(
            source_path,
            structure_label=representative_entry.structure,
        ),
        source_path,
    )


def _prepare_packmol_structure(
    structure: PDBStructure,
    *,
    residue_name: str,
    solvent_residue_names: set[str] | frozenset[str] | None = None,
    solute_reference_structure: PDBStructure | None = None,
    expected_solute_element_counts: dict[str, int] | None = None,
    solute_atom_count: int | None = None,
) -> _PreparedPackmolStructure:
    copied_atoms = [atom.copy() for atom in structure.atoms]
    solute_indices = _packmol_solute_atom_indices(
        copied_atoms,
        solvent_residue_names=solvent_residue_names,
        solute_reference_structure=solute_reference_structure,
        expected_solute_element_counts=expected_solute_element_counts,
        solute_atom_count=solute_atom_count,
    )
    solute_counters: dict[str, int] = {}
    solvent_residue_numbers: dict[tuple[str, int], int] = {}
    next_solvent_residue_number = 2
    solvent_residue_names: set[str] = set()
    for atom_index, atom in enumerate(copied_atoms):
        index = atom_index + 1
        atom.atom_id = index
        atom.element = str(atom.element).title()
        if atom_index in solute_indices:
            atom.residue_number = 1
            atom.residue_name = residue_name
            element_index = solute_counters.get(atom.element, 0) + 1
            solute_counters[atom.element] = element_index
            atom.atom_name = f"{atom.element}{element_index}"
            continue

        original_residue_name = _normalized_residue_name(
            atom.residue_name or "SOL"
        )
        original_residue_number = int(atom.residue_number)
        residue_key = (original_residue_name, original_residue_number)
        residue_number = solvent_residue_numbers.get(residue_key)
        if residue_number is None:
            residue_number = next_solvent_residue_number
            solvent_residue_numbers[residue_key] = residue_number
            next_solvent_residue_number += 1
        atom.residue_name = original_residue_name
        atom.residue_number = residue_number
        if not str(atom.atom_name).strip():
            atom.atom_name = f"{atom.element}{index}"
        solvent_residue_names.add(original_residue_name)

    prepared_structure = PDBStructure(
        atoms=copied_atoms, source_name=structure.source_name
    )
    return _PreparedPackmolStructure(
        structure=prepared_structure,
        solute_atom_count=len(solute_indices),
        solvent_atom_count=len(copied_atoms) - len(solute_indices),
        solvent_residue_count=len(solvent_residue_numbers),
        solvent_residue_names=tuple(sorted(solvent_residue_names)),
    )


def _packmol_solute_atom_indices(
    atoms: list[PDBAtom],
    *,
    solvent_residue_names: set[str] | frozenset[str] | None = None,
    solute_reference_structure: PDBStructure | None = None,
    expected_solute_element_counts: dict[str, int] | None = None,
    solute_atom_count: int | None = None,
) -> set[int]:
    if not atoms:
        return set()
    known_solvent_residue_names = _normalized_residue_names(
        solvent_residue_names or ()
    )
    if known_solvent_residue_names:
        solute_indices = {
            index
            for index, atom in enumerate(atoms)
            if _normalized_residue_name(atom.residue_name)
            not in known_solvent_residue_names
        }
        if solute_indices:
            return solute_indices

    matched_solute_indices = _coordinate_matched_solute_atom_indices(
        atoms,
        solute_reference_structure,
    )
    if matched_solute_indices:
        return matched_solute_indices

    expected_solute_indices = _element_matched_solute_atom_indices(
        atoms,
        expected_solute_element_counts or {},
    )
    if expected_solute_indices:
        return expected_solute_indices

    if solute_atom_count is not None and solute_atom_count > 0:
        bounded_count = min(int(solute_atom_count), len(atoms))
        return set(range(bounded_count))
    first_atom = atoms[0]
    first_key = (
        str(first_atom.residue_name).strip().upper(),
        int(first_atom.residue_number),
    )
    return {
        index
        for index, atom in enumerate(atoms)
        if (
            str(atom.residue_name).strip().upper(),
            int(atom.residue_number),
        )
        == first_key
    }


def _solvent_residue_names_for_packmol_source(
    solvent_metadata: SolventHandlingMetadata | None,
    *,
    free_solvent_reference_path: str | None,
) -> frozenset[str]:
    residue_names: set[str] = set()
    if solvent_metadata is not None:
        _add_residue_name_candidate(
            residue_names,
            solvent_metadata.reference_residue_name,
        )
        _add_residue_name_candidate(
            residue_names,
            solvent_metadata.reference_name,
        )
        _add_reference_residue_name(
            residue_names,
            solvent_metadata.reference_path,
        )
    _add_reference_residue_name(residue_names, free_solvent_reference_path)
    return frozenset(residue_names)


def _add_residue_name_candidate(
    residue_names: set[str],
    value: object,
) -> None:
    text = str(value or "").strip()
    if text:
        residue_names.add(_normalized_residue_name(text))


def _add_reference_residue_name(
    residue_names: set[str],
    reference_path: str | None,
) -> None:
    path_text = _optional_text(reference_path)
    if path_text is None:
        return
    path = Path(path_text).expanduser()
    if not path.is_file() or path.suffix.lower() != ".pdb":
        return
    try:
        reference_structure = PDBStructure.from_file(path)
    except Exception:
        return
    for atom in reference_structure.atoms:
        _add_residue_name_candidate(residue_names, atom.residue_name)
        return


def _solute_reference_structure_for_packmol_source(
    solvent_entry: object | None,
) -> PDBStructure | None:
    if solvent_entry is None:
        return None
    path_text = _optional_text(getattr(solvent_entry, "no_solvent_pdb", None))
    if path_text is None:
        return None
    path = Path(path_text).expanduser()
    if not path.is_file():
        return None
    try:
        return PDBStructure.from_file(path)
    except Exception:
        return None


def _coordinate_matched_solute_atom_indices(
    atoms: list[PDBAtom],
    reference_structure: PDBStructure | None,
) -> set[int]:
    if reference_structure is None or not reference_structure.atoms:
        return set()
    matched_indices: set[int] = set()
    for reference_atom in reference_structure.atoms:
        reference_element = str(reference_atom.element).title()
        best_index: int | None = None
        best_distance: float | None = None
        for index, atom in enumerate(atoms):
            if index in matched_indices:
                continue
            if str(atom.element).title() != reference_element:
                continue
            distance = float(
                np.linalg.norm(atom.coordinates - reference_atom.coordinates)
            )
            if distance > _PACKMOL_SOLUTE_MATCH_TOLERANCE_A:
                continue
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            return set()
        matched_indices.add(best_index)
    return matched_indices


def _element_matched_solute_atom_indices(
    atoms: list[PDBAtom],
    expected_counts: dict[str, int],
) -> set[int]:
    expected = {
        str(element).title(): int(count)
        for element, count in expected_counts.items()
        if int(count) > 0
    }
    if not expected:
        return set()
    selected_indices: set[int] = set()
    for element, expected_count in expected.items():
        matching_indices = [
            index
            for index, atom in enumerate(atoms)
            if str(atom.element).title() == element
        ]
        if len(matching_indices) != expected_count:
            return set()
        selected_indices.update(matching_indices)
    return selected_indices


def _normalized_residue_names(
    residue_names: set[str] | frozenset[str] | tuple[str, ...] | list[str],
) -> frozenset[str]:
    normalized: set[str] = set()
    for residue_name in residue_names:
        text = str(residue_name or "").strip()
        if text:
            normalized.add(_normalized_residue_name(text))
    return frozenset(normalized)


def _solute_atom_count_for_packmol_source(
    solvent_entry: object | None,
) -> int | None:
    if solvent_entry is None:
        return None
    try:
        value = int(getattr(solvent_entry, "atom_count_no_solvent", 0))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _write_supplemental_packmol_structures(
    output_dir: Path,
    plan_metadata: PackmolPlanningMetadata,
) -> list[PackmolSetupSupplementalEntry]:
    allocation = plan_metadata.supplemental_allocation
    if allocation is None:
        return []
    entries: list[PackmolSetupSupplementalEntry] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, component in enumerate(allocation.entries, start=1):
        if component.planned_count <= 0:
            continue
        source_structure = _supplemental_source_structure(component)
        prepared_structure = _prepare_supplemental_packmol_structure(
            source_structure,
            residue_name=component.residue_name,
        )
        filename = (
            f"supplemental_{index:03d}_"
            f"{_safe_name(component.name)}_"
            f"{_safe_name(component.residue_name)}.pdb"
        )
        output_path = output_dir / filename
        prepared_structure.write_pdb_file(output_path)
        entries.append(
            PackmolSetupSupplementalEntry(
                role=component.role,
                name=component.name,
                source_type=component.source_type,
                reference_name=component.reference_name,
                reference_path=component.reference_path,
                residue_name=component.residue_name,
                planned_count=component.planned_count,
                atom_count=len(prepared_structure.atoms),
                element_counts=dict(component.element_counts),
                packmol_pdb=str(output_path),
            )
        )
    return entries


def _supplemental_source_structure(
    component: object,
) -> PDBStructure:
    source_type = str(getattr(component, "source_type", "")).strip()
    reference_path = _optional_text(getattr(component, "reference_path", None))
    if source_type == "reference" and reference_path is not None:
        return PDBStructure.from_file(Path(reference_path).expanduser())
    if source_type == "single_atom":
        element_counts = dict(getattr(component, "element_counts", {}) or {})
        element = next(
            (
                str(key)
                for key, value in element_counts.items()
                if int(value) > 0
            ),
            "X",
        )
        residue_name = _normalized_residue_name(
            str(getattr(component, "residue_name", "") or element)
        )
        return PDBStructure(
            atoms=[
                PDBAtom(
                    atom_id=1,
                    atom_name=f"{element}1",
                    residue_name=residue_name,
                    residue_number=1,
                    coordinates=np.zeros(3, dtype=float),
                    element=element,
                )
            ],
            source_name=str(getattr(component, "name", "") or element),
        )
    raise ValueError(
        "Unsupported supplemental Packmol component source: "
        f"{source_type or '(none)'}"
    )


def _prepare_supplemental_packmol_structure(
    structure: PDBStructure,
    *,
    residue_name: str,
) -> PDBStructure:
    copied_atoms = [atom.copy() for atom in structure.atoms]
    normalized_residue = _normalized_residue_name(residue_name)
    counters: dict[str, int] = {}
    for index, atom in enumerate(copied_atoms, start=1):
        atom.atom_id = index
        atom.element = str(atom.element).title()
        atom.residue_name = normalized_residue
        atom.residue_number = 1
        if not str(atom.atom_name).strip():
            counters[atom.element] = counters.get(atom.element, 0) + 1
            atom.atom_name = f"{atom.element}{counters[atom.element]}"
    return PDBStructure(atoms=copied_atoms, source_name=structure.source_name)


def _write_packmol_input(
    output_dir: Path,
    entries: list[PackmolSetupEntry],
    *,
    supplemental_entries: list[PackmolSetupSupplementalEntry],
    solvent_pdb_path: str | None,
    free_solvent_molecules: int,
    box_side_length_a: float,
    settings: PackmolSetupSettings,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / settings.output_filename
    with input_path.open("w", encoding="utf-8") as handle:
        handle.write(f"tolerance {settings.tolerance_angstrom:.3f}\n")
        handle.write("filetype pdb\n")
        handle.write(f"output {settings.packed_output_filename}\n\n")
        for entry in entries:
            handle.write(f"structure {Path(entry.packmol_pdb).name}\n")
            handle.write(f"  number {entry.planned_count}\n")
            handle.write(
                "  inside box 0.0 0.0 0.0 "
                f"{box_side_length_a:.3f} {box_side_length_a:.3f} {box_side_length_a:.3f}\n"
            )
            handle.write("end structure\n\n")
        for entry in supplemental_entries:
            handle.write(f"structure {Path(entry.packmol_pdb).name}\n")
            handle.write(f"  number {entry.planned_count}\n")
            handle.write(
                "  inside box 0.0 0.0 0.0 "
                f"{box_side_length_a:.3f} {box_side_length_a:.3f} {box_side_length_a:.3f}\n"
            )
            handle.write("end structure\n\n")
        if solvent_pdb_path and free_solvent_molecules > 0:
            handle.write(f"structure {Path(solvent_pdb_path).name}\n")
            handle.write(f"  number {free_solvent_molecules}\n")
            handle.write(
                "  inside box 0.0 0.0 0.0 "
                f"{box_side_length_a:.3f} {box_side_length_a:.3f} {box_side_length_a:.3f}\n"
            )
            handle.write("end structure\n")
    return input_path


def _write_packmol_audit_report(
    project_source: "RMCDreamProjectSource",
    plan_metadata: PackmolPlanningMetadata,
    entries: list[PackmolSetupEntry],
    *,
    input_path: Path,
    solvent_pdb_path: str | None,
    free_solvent_reference_name: str | None,
    free_solvent_reference_path: str | None,
    target_solvent_molecules: int,
    solvent_molecules_in_clusters: int,
    free_solvent_molecules: int,
    supplemental_entries: list[PackmolSetupSupplementalEntry],
) -> Path:
    lines = [
        "# Packmol Build Audit",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Project: {project_source.settings.project_dir}",
        "",
        "## Packmol Input",
        f"- Input file: {input_path}",
        f"- Referenced packed output: {input_path.parent / input_path.stem}.pdb",
        (
            "- Solvent input: "
            f"{solvent_pdb_path if solvent_pdb_path is not None else '(none)'}"
        ),
        (
            "- Free solvent structure: "
            f"{free_solvent_reference_name or '(none)'}"
        ),
        (
            "- Free solvent source path: "
            f"{free_solvent_reference_path if free_solvent_reference_path is not None else '(none)'}"
        ),
        f"- Target solvent molecules: {target_solvent_molecules}",
        ("- Cluster solvent molecules: " f"{solvent_molecules_in_clusters}"),
        f"- Free solvent molecules: {free_solvent_molecules}",
        "",
        "## Planned Clusters",
        f"- Planning mode: {plan_metadata.settings.planning_mode}",
        f"- Box side: {plan_metadata.settings.box_side_length_a:.3f} A",
        f"- Cluster entries: {len(entries)}",
        f"- Total cluster count: {sum(entry.planned_count for entry in entries)}",
        (
            "- Supplemental component count: "
            f"{sum(entry.planned_count for entry in supplemental_entries)}"
        ),
        "",
        "## Structure Table",
        "| Structure | Motif | Param | Count | Residue | File |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            "| "
            f"{entry.structure} | {entry.motif} | {entry.param} | "
            f"{entry.planned_count} | {entry.residue_name} | "
            f"{Path(entry.packmol_pdb).name} |"
        )
    if supplemental_entries:
        lines.extend(
            [
                "",
                "## Supplemental Solute Components",
                "| Name | Role | Residue | Count | Formula | File |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for entry in supplemental_entries:
            lines.append(
                "| "
                f"{entry.name} | {entry.role} | {entry.residue_name} | "
                f"{entry.planned_count} | "
                f"{_format_element_counts(entry.element_counts)} | "
                f"{Path(entry.packmol_pdb).name} |"
            )
    lines.extend(
        [
            "",
            "## Related Reports",
            f"- Count table: {project_source.rmcsetup_paths.cluster_counts_csv_path}",
            f"- Count-normalized weights: {project_source.rmcsetup_paths.planned_count_weights_csv_path}",
            f"- Atom-normalized weights: {project_source.rmcsetup_paths.planned_atom_weights_csv_path}",
            f"- Planning report: {project_source.rmcsetup_paths.packmol_plan_report_path}",
            (
                "- Reproducibility report: "
                f"{project_source.rmcsetup_paths.packmol_build_report_path}"
            ),
            "",
            "## Notes",
            (
                "- Cluster solute atoms were rewritten with unique residue "
                "names for Packmol use."
            ),
            (
                "- Embedded solvent residues were preserved and reindexed as "
                "separate solvent molecules."
            ),
            "- Free solvent counts subtract solvent molecules already present in the cluster files from the bulk-solvent target.",
            "- Supplemental solute components are placed as independent Packmol structures to complete solute stoichiometry not represented by the weighted cluster files.",
            "- If solvent-handling outputs are available, the completed full-solvent representative PDBs define the embedded cluster solvent counts.",
        ]
    )
    audit_path = project_source.rmcsetup_paths.packmol_audit_report_path
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit_path


def _write_packmol_build_report(
    project_source: "RMCDreamProjectSource",
    plan_metadata: PackmolPlanningMetadata,
    entries: list[PackmolSetupEntry],
    *,
    input_path: Path,
    solvent_pdb_path: str | None,
    free_solvent_reference_name: str | None,
    free_solvent_reference_path: str | None,
    target_solvent_molecules: int,
    solvent_molecules_in_clusters: int,
    free_solvent_molecules: int,
    representative_structure_mode: str,
    representative_selection_mode: str,
    settings: PackmolSetupSettings,
    supplemental_entries: list[PackmolSetupSupplementalEntry],
) -> Path:
    report_path = project_source.rmcsetup_paths.packmol_build_report_path
    lines = [
        "SAXSShell rmcsetup Packmol build report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Project: {project_source.settings.project_dir}",
        "",
        "Source input information",
        f"  Packmol input file: {input_path}",
        (
            "  Packed output path: "
            f"{input_path.parent / settings.packed_output_filename}"
        ),
        f"  Planning metadata: {project_source.rmcsetup_paths.packmol_plan_path}",
        f"  Setup metadata: {project_source.rmcsetup_paths.packmol_setup_path}",
        f"  Representative selection mode: {representative_selection_mode}",
        (
            "  Representative structure set: "
            f"{representative_structure_mode_label(representative_structure_mode)}"
        ),
        f"  Planning mode: {plan_metadata.settings.planning_mode}",
        f"  Packmol tolerance: {settings.tolerance_angstrom:.6g} A",
        "",
        "Box and number-density targets",
        (
            "  Box side length: "
            f"{plan_metadata.settings.box_side_length_a:.6g} A"
        ),
        (
            "  Target total number density: "
            f"{plan_metadata.target_total_number_density_a3:.8g} atoms/A^3"
        ),
        (
            "  Achieved cluster number density: "
            f"{plan_metadata.achieved_total_number_density_a3:.8g} atoms/A^3"
        ),
        "  Target element number densities:",
    ]
    if plan_metadata.target_element_number_density_a3:
        for element in sorted(plan_metadata.target_element_number_density_a3):
            lines.append(
                f"    {element}: "
                f"{plan_metadata.target_element_number_density_a3[element]:.8g} "
                "atoms/A^3"
            )
    else:
        lines.append("    none")
    lines.append("  Achieved element number densities:")
    if plan_metadata.achieved_element_number_density_a3:
        for element in sorted(
            plan_metadata.achieved_element_number_density_a3
        ):
            lines.append(
                f"    {element}: "
                f"{plan_metadata.achieved_element_number_density_a3[element]:.8g} "
                "atoms/A^3"
            )
    else:
        lines.append("    none")

    lines.extend(
        [
            "",
            "Solvent accounting",
            (
                "  Free solvent structure: "
                f"{free_solvent_reference_name or '(none)'}"
            ),
            (
                "  Free solvent source path: "
                f"{free_solvent_reference_path or '(none)'}"
            ),
            f"  Free solvent Packmol PDB: {solvent_pdb_path or '(none)'}",
            f"  Computed solvent molecules: {target_solvent_molecules}",
            f"  Cluster solvent molecules: {solvent_molecules_in_clusters}",
            f"  Free solvent molecules: {free_solvent_molecules}",
            "",
            "Representative cluster inputs",
            (
                "  Structure | Motif | Param | Source PDB | Packmol PDB | "
                "Cluster residue | Count | Selected weight | Planned count "
                "weight | Atom weight | Solute atoms | Solvent atoms | "
                "Solvent residues"
            ),
        ]
    )
    for entry in entries:
        solvent_residues = (
            ",".join(entry.solvent_residue_names)
            if entry.solvent_residue_names
            else "none"
        )
        lines.append(
            "  "
            f"{entry.structure} | {entry.motif} | {entry.param} | "
            f"{entry.source_pdb} | {entry.packmol_pdb} | "
            f"{entry.residue_name} | {entry.planned_count} | "
            f"{entry.selected_weight:.8g} | "
            f"{entry.planned_count_weight:.8g} | "
            f"{entry.planned_atom_weight:.8g} | "
            f"{entry.solute_atom_count} | {entry.solvent_atom_count} | "
            f"{solvent_residues}"
        )

    allocation = plan_metadata.solvent_allocation
    if allocation is not None and allocation.entries:
        lines.extend(["", "Cluster solvent allocation"])
        for allocation_entry in allocation.entries:
            lines.append(
                "  "
                f"{allocation_entry.structure}/{allocation_entry.motif} "
                f"({allocation_entry.param}): "
                f"{allocation_entry.planned_count} clusters x "
                f"{allocation_entry.solvent_molecules_per_cluster} solvent "
                "molecules per cluster = "
                f"{allocation_entry.solvent_molecules_total}"
            )

    supplemental_allocation = plan_metadata.supplemental_allocation
    if supplemental_allocation is not None:
        lines.extend(["", "Supplemental solute accounting"])
        lines.append(
            "  Formula units represented by weighted clusters: "
            f"{supplemental_allocation.target_solute_formula_units}"
        )
        lines.append(
            "  Formula-unit basis: "
            + _format_float_counts(supplemental_allocation.formula_unit_basis)
        )
        lines.append(
            "  Cluster solute element totals: "
            + _format_element_counts(
                supplemental_allocation.cluster_solute_element_totals
            )
        )
        lines.append(
            "  Target solute element totals: "
            + _format_element_counts(
                supplemental_allocation.target_solute_element_totals
            )
        )
        lines.append(
            "  Missing solute elements before supplementals: "
            + _format_element_counts(
                supplemental_allocation.missing_solute_element_totals
            )
        )
        lines.append(
            "  Added solute elements: "
            + _format_element_counts(
                supplemental_allocation.added_solute_element_totals
            )
        )
        lines.append(
            "  Unfilled solute elements: "
            + _format_element_counts(
                supplemental_allocation.unfilled_solute_element_totals
            )
        )
        for warning in supplemental_allocation.warnings:
            lines.append(f"  Warning: {warning}")
    if supplemental_entries:
        lines.extend(
            [
                "",
                "Supplemental Packmol components",
                (
                    "  Name | Role | Source | Residue | Count | Atom count | "
                    "Formula | Packmol PDB"
                ),
            ]
        )
        for entry in supplemental_entries:
            source = (
                entry.reference_path
                if entry.reference_path is not None
                else entry.source_type
            )
            lines.append(
                "  "
                f"{entry.name} | {entry.role} | {source} | "
                f"{entry.residue_name} | {entry.planned_count} | "
                f"{entry.atom_count} | "
                f"{_format_element_counts(entry.element_counts)} | "
                f"{entry.packmol_pdb}"
            )

    lines.extend(
        [
            "",
            "Generated files",
            f"  Packmol input: {input_path}",
            (
                "  Supplemental PDBs: "
                + (
                    ", ".join(
                        Path(entry.packmol_pdb).name
                        for entry in supplemental_entries
                    )
                    if supplemental_entries
                    else "none"
                )
            ),
            f"  Free solvent PDB: {solvent_pdb_path or '(none)'}",
            (
                "  Packmol inputs directory: "
                f"{project_source.rmcsetup_paths.packmol_inputs_dir}"
            ),
            (
                "  Count report: "
                f"{project_source.rmcsetup_paths.cluster_counts_csv_path}"
            ),
            (
                "  Count-normalized weights: "
                f"{project_source.rmcsetup_paths.planned_count_weights_csv_path}"
            ),
            (
                "  Atom-normalized weights: "
                f"{project_source.rmcsetup_paths.planned_atom_weights_csv_path}"
            ),
            (
                "  Planning report: "
                f"{project_source.rmcsetup_paths.packmol_plan_report_path}"
            ),
            (
                "  Audit report: "
                f"{project_source.rmcsetup_paths.packmol_audit_report_path}"
            ),
            "",
            "Residue and constraint notes",
            (
                "  Cluster solute atoms are assigned the cluster-specific "
                "residues listed above."
            ),
            (
                "  Embedded solvent atoms keep solvent residue names and are "
                "reindexed by solvent molecule."
            ),
            (
                "  Constraint generation filters to the cluster-specific "
                "residues, so embedded solvent residues are not used for "
                "solute cluster constraints."
            ),
            (
                "  Supplemental solute residues are independent Packmol "
                "components and are not used for cluster-specific constraint "
                "generation."
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _packmol_residue_code(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < 0:
        index = 0
    prefix = "C"
    middle = alphabet[(index // 26) % 26]
    suffix = alphabet[index % 26]
    return f"{prefix}{middle}{suffix}"


def _load_structure_as_pdb(
    source_file: str | Path,
    *,
    structure_label: str,
) -> PDBStructure:
    path = Path(source_file).expanduser().resolve()
    if path.suffix.lower() == ".pdb":
        source_structure = PDBStructure.from_file(path)
        copied_atoms = [atom.copy() for atom in source_structure.atoms]
        for index, atom in enumerate(copied_atoms, start=1):
            atom.atom_id = index
        return PDBStructure(atoms=copied_atoms, source_name=path.stem)

    positions, elements = load_structure_file(path)
    residue_name = _normalized_residue_name(structure_label)
    counters: dict[str, int] = {}
    atoms: list[PDBAtom] = []
    for index, (coordinates, element) in enumerate(
        zip(positions, elements, strict=True),
        start=1,
    ):
        counters[element] = counters.get(element, 0) + 1
        atoms.append(
            PDBAtom(
                atom_id=index,
                atom_name=f"{element}{counters[element]}",
                residue_name=residue_name,
                residue_number=1,
                coordinates=np.asarray(coordinates, dtype=float),
                element=str(element),
            )
        )
    return PDBStructure(atoms=atoms, source_name=path.stem)


def _normalized_residue_name(text: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9]+", "", text).upper()
    if not collapsed:
        collapsed = "CLU"
    return collapsed[:3]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_name(text: str) -> str:
    collapsed = re.sub(r"[^0-9A-Za-z]+", "_", str(text).strip())
    collapsed = re.sub(r"_+", "_", collapsed).strip("_")
    return collapsed or "item"


def _safe_filename(text: str) -> str:
    name = Path(text.strip() or "item").name
    return name or "item"


def _format_element_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return (
        ", ".join(
            f"{element} x{int(count)}"
            for element, count in sorted(counts.items())
            if int(count) != 0
        )
        or "none"
    )


def _format_float_counts(counts: dict[str, float]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{element}={float(value):.6g}"
        for element, value in sorted(counts.items())
    )


def _resolve_free_solvent_reference(
    settings: PackmolSetupSettings,
    plan_metadata: PackmolPlanningMetadata,
    solvent_metadata: SolventHandlingMetadata | None,
) -> tuple[str | None, str | None]:
    candidates = [
        settings.free_solvent_reference,
        plan_metadata.settings.free_solvent_reference,
        (
            None
            if plan_metadata.solvent_allocation is None
            else plan_metadata.solvent_allocation.reference_path
        ),
        (
            None
            if solvent_metadata is None
            else solvent_metadata.reference_path
        ),
    ]
    for candidate in candidates:
        reference_identifier = _optional_text(candidate)
        if reference_identifier is None:
            continue
        resolved_reference = resolve_reference_path(
            reference_identifier
        ).expanduser()
        return resolved_reference.stem, str(resolved_reference.resolve())
    return None, None


__all__ = [
    "PackmolSetupEntry",
    "PackmolSetupMetadata",
    "PackmolSetupSettings",
    "PackmolSetupSupplementalEntry",
    "build_packmol_setup",
    "load_packmol_setup_metadata",
    "save_packmol_setup_metadata",
]
