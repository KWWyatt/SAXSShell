from __future__ import annotations

import json
from pathlib import Path

import pytest

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondPairDefinition,
    CoordinationNumberDefinition,
    DihedralQuartetDefinition,
)
from saxshell.structure_distributions import (
    StructureDistributionStore,
    load_structure_distribution_index,
)


def _write_xyz(
    path: Path,
    atoms: list[tuple[str, float, float, float]],
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_structure_distribution_store_reuses_warm_bond_angle_cache(
    tmp_path,
    monkeypatch,
):
    structure_path = tmp_path / "candidate.xyz"
    _write_xyz(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("C", 1.0, 1.0, 0.0),
            ("N", 1.0, 1.0, 1.0),
        ],
    )
    store = StructureDistributionStore(tmp_path / "store")
    bond = BondPairDefinition("Pb", "I", 3.0)
    angle = AngleTripletDefinition("Pb", "I", "I", 3.0, 3.0)
    dihedral = DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5)
    coordination = CoordinationNumberDefinition("Pb", "I", 3.0)

    first = store.measure_structure_file(
        structure_path,
        bond_pairs=(bond,),
        angle_triplets=(angle,),
        dihedral_quartets=(dihedral,),
        coordination_numbers=(coordination,),
    )
    assert not first.from_cache
    assert first.bond_values[bond] == pytest.approx([2.0, 2.0])
    assert first.angle_values[angle] == pytest.approx([90.0])
    assert first.dihedral_values[dihedral] == pytest.approx([90.0])
    assert first.coordination_values[coordination] == pytest.approx([2.0])

    def fail_measurement(*_args, **_kwargs):
        raise AssertionError("warm cache should avoid measuring again")

    monkeypatch.setattr(
        "saxshell.bondanalysis.bondanalyzer."
        "BondAnalyzer.measure_structure_with_coordination_and_dihedrals",
        fail_measurement,
    )
    second = StructureDistributionStore(
        tmp_path / "store"
    ).measure_structure_file(
        structure_path,
        bond_pairs=(bond,),
        angle_triplets=(angle,),
        dihedral_quartets=(dihedral,),
        coordination_numbers=(coordination,),
    )
    assert second.from_cache
    assert second.bond_values[bond] == pytest.approx(first.bond_values[bond])
    assert second.angle_values[angle] == pytest.approx(
        first.angle_values[angle]
    )
    assert second.dihedral_values[dihedral] == pytest.approx(
        first.dihedral_values[dihedral]
    )
    assert second.coordination_values[coordination] == pytest.approx(
        first.coordination_values[coordination]
    )


def test_structure_distribution_store_invalidates_changed_structure(tmp_path):
    structure_path = tmp_path / "candidate.xyz"
    _write_xyz(
        structure_path,
        [("Pb", 0.0, 0.0, 0.0), ("I", 2.0, 0.0, 0.0)],
    )
    bond = BondPairDefinition("Pb", "I", 3.0)
    store = StructureDistributionStore(tmp_path / "store")
    first = store.measure_structure_file(
        structure_path,
        bond_pairs=(bond,),
        angle_triplets=(),
    )

    _write_xyz(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.5, 0.0, 0.0),
            ("I", 0.0, 2.5, 0.0),
        ],
    )
    second = StructureDistributionStore(
        tmp_path / "store"
    ).measure_structure_file(
        structure_path,
        bond_pairs=(bond,),
        angle_triplets=(),
    )
    assert not second.from_cache
    assert first.bond_values[bond] == pytest.approx([2.0])
    assert second.bond_values[bond] == pytest.approx([2.5, 2.5])


def test_structure_distribution_store_caches_cutoff_pair_distances(tmp_path):
    structure_path = tmp_path / "cluster.xyz"
    _write_xyz(
        structure_path,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("O", 0.0, 0.0, 8.0),
        ],
    )
    store = StructureDistributionStore(tmp_path / "store")
    first = store.measure_cutoff_pair_distances(
        structure_path,
        pair_cutoff_definitions={("Pb", "I"): {0: 3.0}},
        allowed_elements=("Pb", "I"),
    )
    second = StructureDistributionStore(
        tmp_path / "store"
    ).measure_cutoff_pair_distances(
        structure_path,
        pair_cutoff_definitions={("I", "Pb"): {0: 3.0}},
        allowed_elements=("I", "Pb"),
    )

    assert not first.from_cache
    assert second.from_cache
    assert set(second.pair_distances) == {("I", "Pb")}
    assert second.pair_distances[("I", "Pb")].tolist() == pytest.approx([2.0])

    manifest = json.loads((tmp_path / "store" / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["entries"]


def test_structure_distribution_index_aggregates_cluster_scopes(tmp_path):
    root_dir = tmp_path / "analysis" / "structure_distributions"
    store = StructureDistributionStore(root_dir / "bondanalysis")
    bond = BondPairDefinition("Pb", "I", 3.0)
    angle = AngleTripletDefinition("Pb", "I", "I", 3.0, 3.0)
    dihedral = DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5)
    coordination = CoordinationNumberDefinition("Pb", "I", 3.0)
    cluster_a = tmp_path / "a.xyz"
    cluster_b = tmp_path / "b.xyz"
    _write_xyz(
        cluster_a,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("C", 1.0, 1.0, 0.0),
            ("N", 1.0, 1.0, 1.0),
        ],
    )
    _write_xyz(
        cluster_b,
        [("Pb", 0.0, 0.0, 0.0), ("I", 2.5, 0.0, 0.0)],
    )
    store.measure_structure_file(
        cluster_a,
        bond_pairs=(bond,),
        angle_triplets=(angle,),
        dihedral_quartets=(dihedral,),
        coordination_numbers=(coordination,),
        cluster_label="PbI2",
    )
    store.measure_structure_file(
        cluster_b,
        bond_pairs=(bond,),
        angle_triplets=(angle,),
        dihedral_quartets=(dihedral,),
        coordination_numbers=(coordination,),
        cluster_label="PbI",
    )

    index = load_structure_distribution_index(root_dir)
    bond_group = next(
        group for group in index.groups if group.category == "bond"
    )
    angle_group = next(
        group for group in index.groups if group.category == "angle"
    )
    dihedral_group = next(
        group for group in index.groups if group.category == "dihedral"
    )
    coordination_group = next(
        group for group in index.groups if group.category == "coordination"
    )

    assert index.source_names == ("bondanalysis",)
    assert bond_group.display_label == "Pb-I <= 3 A"
    assert bond_group.all_leaf.point_count == 3
    assert bond_group.all_leaf.structure_count == 2
    assert {
        leaf.scope_name: leaf.values.tolist()
        for leaf in bond_group.cluster_leaves
    } == {
        "PbI": pytest.approx([2.5]),
        "PbI2": pytest.approx([2.0, 2.0]),
    }
    assert angle_group.display_label == "I-Pb-I <= 3/3 A"
    assert angle_group.all_leaf.values.tolist() == pytest.approx([90.0])
    assert dihedral_group.display_label == "Pb-O-C-N <= 1.5/1.5/1.5 A"
    assert dihedral_group.all_leaf.values.tolist() == pytest.approx([90.0])
    assert coordination_group.display_label == "CN Pb-I <= 3 A"
    assert coordination_group.all_leaf.values.tolist() == pytest.approx(
        [1.0, 2.0]
    )


def test_structure_distribution_index_skips_stale_entries(tmp_path):
    root_dir = tmp_path / "analysis" / "structure_distributions"
    store = StructureDistributionStore(root_dir / "representativefinder")
    structure_path = tmp_path / "candidate.xyz"
    _write_xyz(
        structure_path,
        [("Pb", 0.0, 0.0, 0.0), ("I", 2.0, 0.0, 0.0)],
    )
    bond = BondPairDefinition("Pb", "I", 3.0)
    store.measure_structure_file(
        structure_path,
        bond_pairs=(bond,),
        angle_triplets=(),
        cluster_label="PbI",
    )

    _write_xyz(
        structure_path,
        [("Pb", 0.0, 0.0, 0.0), ("I", 2.5, 0.0, 0.0)],
    )

    index = load_structure_distribution_index(root_dir)

    assert index.entry_count == 1
    assert index.stale_entry_count == 1
    assert index.groups == ()
