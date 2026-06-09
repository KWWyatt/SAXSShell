import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import to_hex
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTreeWidgetItem,
)

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisWorkflow,
    BondPairDefinition,
    CoordinationNumberDefinition,
    DihedralQuartetDefinition,
)
from saxshell.bondanalysis.results import (
    BondAnalysisDistributionSeries,
    BondAnalysisPlotRequest,
    BondAnalysisResultGroup,
    BondAnalysisResultIndex,
    BondAnalysisResultLeaf,
    build_plot_request,
    load_result_index,
)
from saxshell.bondanalysis.ui.main_window import (
    BOND_ANALYSIS_WINDOW_LOAD_TOTAL_STEPS,
    SELECTION_PREVIEW_DEBOUNCE_MS,
    SELECTION_PREVIEW_TOTAL_STEPS,
    BondAnalysisMainWindow,
    _result_index_paths_below,
)
from saxshell.bondanalysis.ui.plot_window import BondAnalysisPlotWindow
from saxshell.saxs.project_manager import SAXSProjectManager


def _write_xyz_cluster(path, atoms):
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n")


def _build_bondanalysis_output(tmp_path):
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbi3_dir = clusters_dir / "PbI3"
    pbi2_dir.mkdir(parents=True)
    pbi3_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbi3_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
            ("I", 0.0, 0.0, 2.0),
        ],
    )

    workflow = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 3.1)],
        angle_triplets=[AngleTripletDefinition("Pb", "I", "I", 3.1, 3.1)],
        coordination_numbers=[CoordinationNumberDefinition("Pb", "I", 3.1)],
        output_dir=tmp_path / "bondanalysis_results",
    )
    result = workflow.run()
    return clusters_dir, result.output_dir


def _find_results_leaf(window, category_label, distribution_label, leaf_label):
    for top_index in range(window.results_tree.topLevelItemCount()):
        category_item = window.results_tree.topLevelItem(top_index)
        if category_item.text(0) != category_label:
            continue
        for group_index in range(category_item.childCount()):
            distribution_item = category_item.child(group_index)
            if distribution_item.text(0) != distribution_label:
                continue
            for leaf_index in range(distribution_item.childCount()):
                leaf_item = distribution_item.child(leaf_index)
                if leaf_item.text(0) == leaf_label:
                    return leaf_item
    raise AssertionError(
        f"Unable to find results leaf {category_label}/{distribution_label}/{leaf_label}"
    )


def _find_results_group(window, category_label, distribution_label):
    for top_index in range(window.results_tree.topLevelItemCount()):
        category_item = window.results_tree.topLevelItem(top_index)
        if category_item.text(0) != category_label:
            continue
        for group_index in range(category_item.childCount()):
            distribution_item = category_item.child(group_index)
            if distribution_item.text(0) == distribution_label:
                return distribution_item
    raise AssertionError(
        f"Unable to find results group {category_label}/{distribution_label}"
    )


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_bondanalysis_main_window_prefills_cluster_types_and_output_dir(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)

    assert window.cluster_type_list.count() == 2
    assert window.use_checked_cluster_types_box.isChecked()
    assert window.use_checked_cluster_types_box.text() == (
        "Analyze all cluster types"
    )
    assert not window.cluster_type_list.isEnabled()
    assert (
        window.cluster_type_list.item(0).checkState() == Qt.CheckState.Checked
    )
    assert (
        window.cluster_type_list.item(1).checkState() == Qt.CheckState.Checked
    )
    assert window.output_dir_edit.text().endswith(
        "bondanalysis_clusters_splitxyz0001"
    )
    preset_names = [
        window.preset_combo.itemText(index)
        for index in range(window.preset_combo.count())
    ]
    assert "DMSO (Built-in)" in preset_names
    assert "DMF (Built-in)" in preset_names
    assert "Analyzing cluster types: all detected types" in (
        window.selection_box.toPlainText()
    )
    preview_text = window.selection_box.toPlainText()
    assert "Cluster types detected: 2" in preview_text
    assert "Checked cluster types: 2" in preview_text
    assert (
        "Stored computed runs for this clusters directory: none found"
        in preview_text
    )
    assert "Displacement analysis: deprecated" in preview_text

    window.load_preset("DMSO")

    assert window.bond_pair_table.rowCount() == 7
    assert window.bond_pair_table.item(0, 0).text() == "Pb"
    assert window.bond_pair_table.item(0, 1).text() == "I"
    assert window.bond_pair_table.item(0, 2).text() == "4"
    assert window.angle_triplet_table.rowCount() == 5
    assert window.angle_triplet_table.item(2, 0).text() == "O"
    assert window.angle_triplet_table.item(2, 1).text() == "Pb"
    assert window.angle_triplet_table.item(2, 2).text() == "S"
    assert window.dihedral_quartet_table.rowCount() == 1
    assert window.dihedral_quartet_table.item(0, 0).text() == "Pb"
    assert window.dihedral_quartet_table.item(0, 1).text() == "O"
    assert window.dihedral_quartet_table.item(0, 2).text() == "S"
    assert window.dihedral_quartet_table.item(0, 3).text() == "C"
    assert window.coordination_number_table.rowCount() == 3
    assert window.coordination_number_table.item(0, 0).text() == "Pb"
    assert window.coordination_number_table.item(0, 1).text() == "I"
    assert window.coordination_number_table.item(0, 2).text() == "4"
    window.close()


def test_bondanalysis_main_window_reports_startup_loader_progress(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)

    dialog = window._startup_progress_dialog
    assert dialog is not None
    assert not dialog.isVisible()
    assert dialog.windowTitle() == "Opening Bond Analysis"
    assert (
        dialog.progress_bar.maximum() == BOND_ANALYSIS_WINDOW_LOAD_TOTAL_STEPS
    )
    assert dialog.progress_bar.value() == BOND_ANALYSIS_WINDOW_LOAD_TOTAL_STEPS
    output = dialog.output_box.toPlainText()
    assert "Preparing Bond Analysis window." in output
    assert "Loading built-in and custom bond-analysis presets." in output
    assert "Inspecting 2 cluster folder(s)." in output
    assert "Discovered 2 cluster type(s)." in output
    assert "Bond Analysis window is ready." in output
    window.close()


def test_bondanalysis_preset_combo_selection_does_not_load_or_scan(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)

    preview_calls = []

    def preview_spy(**kwargs):
        preview_calls.append(kwargs)
        return []

    monkeypatch.setattr(
        window,
        "_stored_results_preview_lines",
        preview_spy,
    )
    index = window.preset_combo.findData("DMSO")

    window.preset_combo.setCurrentIndex(index)
    qapp.processEvents()

    assert preview_calls == []
    assert window.bond_pair_table.rowCount() == 1
    assert window.bond_pair_table.item(0, 0).text() == ""
    window.close()


def test_bondanalysis_preset_load_refreshes_preview_once(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    preview_calls = []

    def preview_spy(**kwargs):
        preview_calls.append(kwargs)
        return ["Stored preview reused from cache"]

    monkeypatch.setattr(
        window,
        "_stored_results_preview_lines",
        preview_spy,
    )

    window.load_preset("DMSO")

    assert len(preview_calls) == 1
    assert "Stored preview reused from cache" in (
        window.selection_box.toPlainText()
    )
    window.close()


def test_bondanalysis_preset_load_uses_selection_preview_progress(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    progress_seen = []

    def preview_spy(**kwargs):
        progress_callback = kwargs["progress_callback"]
        progress_callback(
            3,
            SELECTION_PREVIEW_TOTAL_STEPS,
            "Comparing preset to stored bond-analysis runs...",
        )
        progress_seen.append(
            window._selection_summary_progress_dialog is not None
            and window._selection_summary_progress_dialog.isVisible()
        )
        return ["Stored preview reused from cache"]

    monkeypatch.setattr(
        window,
        "_stored_results_preview_lines",
        preview_spy,
    )

    window.load_preset("DMSO")
    qapp.processEvents()

    dialog = window._selection_summary_progress_dialog
    assert dialog is not None
    assert dialog.windowTitle() == "Updating Selection Preview"
    assert not dialog.isVisible()
    assert progress_seen == [True]
    assert "Reading current bond-analysis selections" in (
        dialog.output_box.toPlainText()
    )
    assert "Comparing preset to stored bond-analysis runs" in (
        dialog.output_box.toPlainText()
    )
    assert "Stored preview reused from cache" in (
        window.selection_box.toPlainText()
    )
    window.close()


def test_bondanalysis_preset_load_keeps_populated_tables_paint_enabled(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.show()
    qapp.processEvents()

    window.load_preset("DMSO")
    qapp.processEvents()

    tables = (
        window.bond_pair_table,
        window.angle_triplet_table,
        window.dihedral_quartet_table,
        window.coordination_number_table,
    )
    assert window.updatesEnabled()
    assert all(table.updatesEnabled() for table in tables)
    assert window.bond_pair_table.rowCount() == 7
    assert window.bond_pair_table.item(0, 0).text() == "Pb"
    assert window.angle_triplet_table.item(2, 0).text() == "O"
    assert window.dihedral_quartet_table.item(0, 0).text() == "Pb"
    assert window.coordination_number_table.item(0, 0).text() == "Pb"
    window.close()


def test_bondanalysis_definition_edits_debounce_selection_preview(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    preview_calls = []

    def preview_spy(**kwargs):
        preview_calls.append(kwargs)
        return []

    monkeypatch.setattr(
        window,
        "_stored_results_preview_lines",
        preview_spy,
    )

    window._add_bond_pair_row()
    QTest.qWait(SELECTION_PREVIEW_DEBOUNCE_MS + 50)
    qapp.processEvents()

    assert preview_calls == []

    row = window.bond_pair_table.rowCount() - 1
    window.bond_pair_table.item(row, 0).setText("Pb")
    window.bond_pair_table.item(row, 1).setText("I")
    window.bond_pair_table.item(row, 2).setText("3.1")
    qapp.processEvents()

    assert preview_calls == []

    QTest.qWait(SELECTION_PREVIEW_DEBOUNCE_MS + 50)
    qapp.processEvents()

    assert len(preview_calls) == 1
    window.close()


def test_bondanalysis_result_index_discovery_is_shallow(tmp_path):
    root = tmp_path / "workspace"
    direct_result = root / "bondanalysis_clusters_splitxyz0001"
    nested_cluster_dir = root / "clusters_splitxyz0001" / "PbI2"
    direct_result.mkdir(parents=True)
    nested_cluster_dir.mkdir(parents=True)
    direct_index = direct_result / "bondanalysis_results_index.json"
    nested_index = nested_cluster_dir / "bondanalysis_results_index.json"
    direct_index.write_text("{}")
    nested_index.write_text("{}")

    paths = _result_index_paths_below(root)

    assert direct_index in paths
    assert nested_index not in paths


def test_bondanalysis_cluster_types_section_collapses(qapp, tmp_path):
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)

    assert window.cluster_types_section.is_collapsed()
    assert window.cluster_types_content.isHidden()
    assert window.cluster_type_list.count() == 1
    assert window.cluster_type_status_label.text() == (
        "1 cluster type(s) ready."
    )

    window.cluster_types_section.set_collapsed(False)
    qapp.processEvents()

    assert not window.cluster_types_section.is_collapsed()
    assert not window.cluster_types_content.isHidden()

    window.cluster_types_section.set_collapsed(True)
    qapp.processEvents()

    assert window.cluster_types_section.is_collapsed()
    assert window.cluster_types_content.isHidden()
    assert window.cluster_type_list.count() == 1
    window.close()


def test_bondanalysis_main_window_shows_compact_project_status_and_registers_clusters_dir(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    manager.create_project(project_dir)
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(
        initial_clusters_dir=clusters_dir,
        initial_project_dir=project_dir,
    )
    window.show()
    qapp.processEvents()
    saved_settings = manager.load_project(project_dir)
    header = window.findChild(QFrame, "BondAnalysisHeader")
    splitter = window.findChild(QSplitter, "BondAnalysisSplitter")

    assert window.project_banner is None
    assert header is not None
    assert splitter is not None
    assert header.height() < 100
    assert splitter.height() > header.height() * 5
    assert window.project_status_label is not None
    assert project_dir.name in window.project_status_label.toolTip()
    assert str(project_dir) in window.project_status_label.full_text()
    assert window.project_status_label.parent() is window.statusBar()
    assert saved_settings.resolved_clusters_dir == clusters_dir.resolve()
    window.close()


def test_bondanalysis_main_window_loads_project_batch_results(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    manager.create_project(project_dir)
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    output_dir = (
        project_dir / "analysis" / "bondanalysis" / "clusters_splitxyz0001"
    )
    result = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[BondPairDefinition("Pb", "I", 3.1)],
        angle_triplets=[AngleTripletDefinition("Pb", "I", "I", 3.1, 3.1)],
        coordination_numbers=[CoordinationNumberDefinition("Pb", "I", 3.1)],
        output_dir=output_dir,
    ).run()

    window = BondAnalysisMainWindow(
        initial_clusters_dir=clusters_dir,
        initial_project_dir=project_dir,
    )

    assert Path(window.output_dir_edit.text()) == result.output_dir
    assert window.results_tree.topLevelItemCount() == 3
    preview_text = window.selection_box.toPlainText()
    assert (
        "Stored computed runs for this clusters directory: 1" in preview_text
    )
    assert "Current settings match stored run:" in preview_text
    assert "Run will reuse without recalculation" in preview_text
    assert "1 bond pair, 1 angle, 1 coordination rule" in preview_text
    assert (
        _find_results_leaf(window, "Bond Pairs", "Pb-I", "all").text(2) == "2"
    )
    assert (
        _find_results_leaf(
            window,
            "Coordination Numbers",
            "CN Pb-I",
            "all",
        ).text(2)
        == "1"
    )
    window.close()


def test_bondanalysis_main_window_loads_latest_project_results_for_clusters(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    manager.create_project(project_dir)
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("C", 1.0, 1.0, 0.0),
            ("N", 1.0, 1.0, 1.0),
        ],
    )
    output_dir = (
        project_dir / "analysis" / "bondanalysis" / "custom_dihedral_only_run"
    )
    result = BondAnalysisWorkflow(
        clusters_dir,
        dihedral_quartets=[
            DihedralQuartetDefinition("Pb", "O", "C", "N", 1.5, 1.5, 1.5),
        ],
        output_dir=output_dir,
    ).run()

    window = BondAnalysisMainWindow(
        initial_clusters_dir=clusters_dir,
        initial_project_dir=project_dir,
    )

    assert Path(window.output_dir_edit.text()) == result.output_dir
    assert window.dihedral_quartet_table.rowCount() == 1
    assert window.dihedral_quartet_table.item(0, 0).text() == "Pb"
    assert window.dihedral_quartet_table.item(0, 1).text() == "O"
    assert window.dihedral_quartet_table.item(0, 2).text() == "C"
    assert window.dihedral_quartet_table.item(0, 3).text() == "N"
    assert window.results_tree.topLevelItemCount() == 1
    assert (
        _find_results_leaf(
            window,
            "Dihedral Angles",
            "Pb-O-C-N",
            "all",
        ).text(2)
        == "1"
    )
    assert window.results_stats_table.rowCount() == 1
    assert window.results_stats_table.item(0, 0).text() == "Dihedral"
    assert window.results_stats_table.item(0, 1).text() == "Pb-O-C-N"
    assert window.results_stats_table.item(0, 2).text() == "1"
    assert window.results_stats_table.item(0, 6).text() == "deg"
    assert (
        window.results_stats_table.item(0, 7).text()
        == "ba_dihedral_all_pb_o_c_n_center"
    )
    window.close()


def test_bondanalysis_main_window_saves_custom_presets_across_sessions(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    preset_file = tmp_path / "bondanalysis_presets.json"
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(preset_file),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.load_preset("DMF")
    window.bond_pair_table.item(0, 2).setText("4.5")
    window.save_current_preset("DMF Custom")

    assert preset_file.exists()

    reloaded_window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    preset_names = [
        reloaded_window.preset_combo.itemText(index)
        for index in range(reloaded_window.preset_combo.count())
    ]
    assert "DMF Custom" in preset_names

    reloaded_window.load_preset("DMF Custom")

    assert reloaded_window.bond_pair_table.item(0, 2).text() == "4.5"
    assert reloaded_window.dihedral_quartet_table.rowCount() == 2
    assert reloaded_window.dihedral_quartet_table.item(0, 0).text() == "Pb"
    assert reloaded_window.dihedral_quartet_table.item(0, 1).text() == "O"
    assert reloaded_window.dihedral_quartet_table.item(0, 2).text() == "C"
    assert reloaded_window.dihedral_quartet_table.item(0, 3).text() == "N"
    assert reloaded_window.dihedral_quartet_table.item(1, 0).text() == "O"
    assert reloaded_window.dihedral_quartet_table.item(1, 1).text() == "C"
    assert reloaded_window.dihedral_quartet_table.item(1, 2).text() == "N"
    assert reloaded_window.dihedral_quartet_table.item(1, 3).text() == "C"
    assert reloaded_window.coordination_number_table.rowCount() == 3


def test_bondanalysis_main_window_uses_checked_cluster_types_filter(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    assert window._selected_cluster_types() is None
    assert not window.cluster_type_list.isEnabled()

    window.use_checked_cluster_types_box.setChecked(False)
    assert window.cluster_type_list.isEnabled()
    window.cluster_type_list.item(1).setCheckState(Qt.CheckState.Unchecked)

    assert window._selected_cluster_types() == ["PbI2"]
    preview_text = window.selection_box.toPlainText()
    assert "Checked cluster types: 1" in preview_text
    assert "Analyzing checked cluster types: PbI2" in preview_text


def test_bondanalysis_results_tree_groups_distributions_by_type(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)

    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    assert window.results_tree.topLevelItemCount() == 3
    assert window.open_selected_window_button.text() == "Open Selected in Tab"
    assert (
        window.open_all_all_plots_button.text() == "Open All 'All' Plot Tabs"
    )
    for top_index in range(window.results_tree.topLevelItemCount()):
        category_item = window.results_tree.topLevelItem(top_index)
        assert category_item.isExpanded()
        assert category_item.childIndicatorPolicy() == (
            QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
        )
        for group_index in range(category_item.childCount()):
            group_item = category_item.child(group_index)
            assert not group_item.isExpanded()
            assert group_item.childIndicatorPolicy() == (
                QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
            )

    bond_group = _find_results_group(window, "Bond Pairs", "Pb-I")
    bond_group_payload = bond_group.data(0, Qt.ItemDataRole.UserRole)
    assert bond_group.flags() & Qt.ItemFlag.ItemIsSelectable
    assert isinstance(bond_group_payload, BondAnalysisResultLeaf)
    assert bond_group_payload.is_all
    assert bond_group_payload.display_label == "Pb-I"
    assert bond_group_payload.point_count == 5
    assert bond_group.text(1) == "all clusters"
    assert bond_group.text(2) == "5"
    assert "Cmd-click" in bond_group.toolTip(0)
    assert "distribution-name rows" in window.results_hint_label.text()

    all_bond_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "all")
    pb_i2_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI2")
    all_angle_leaf = _find_results_leaf(
        window,
        "Bond Angles",
        "I-Pb-I",
        "all",
    )
    all_coordination_leaf = _find_results_leaf(
        window,
        "Coordination Numbers",
        "CN Pb-I",
        "all",
    )

    assert all_bond_leaf.text(2) == "5"
    assert pb_i2_leaf.text(2) == "2"
    assert all_angle_leaf.text(2) == "4"
    assert all_coordination_leaf.text(2) == "2"
    assert window.results_stats_table.rowCount() == 3
    stats_rows = {
        window.results_stats_table.item(row, 1).text(): row
        for row in range(window.results_stats_table.rowCount())
    }
    bond_row = stats_rows["Pb-I"]
    assert window.results_stats_table.item(bond_row, 0).text() == "Bond Pair"
    assert window.results_stats_table.item(bond_row, 2).text() == "5"
    assert window.results_stats_table.item(bond_row, 3).text() == "2"
    assert window.results_stats_table.item(bond_row, 4).text() == "2"
    assert window.results_stats_table.item(bond_row, 5).text() == "0"
    assert window.results_stats_table.item(bond_row, 6).text() == "A"
    assert (
        window.results_stats_table.item(bond_row, 7).text()
        == "ba_bond_all_pb_i_center"
    )
    assert (
        window.results_stats_table.item(bond_row, 8).text()
        == "ba_bond_all_pb_i_sigma"
    )

    angle_row = stats_rows["I-Pb-I"]
    assert window.results_stats_table.item(angle_row, 0).text() == "Angle"
    assert window.results_stats_table.item(angle_row, 3).text() == "90"
    assert window.results_stats_table.item(angle_row, 5).text() == "0"
    assert window.results_stats_table.item(angle_row, 6).text() == "deg"

    coordination_row = stats_rows["CN Pb-I"]
    assert (
        window.results_stats_table.item(coordination_row, 0).text()
        == "Coordination"
    )
    assert window.results_stats_table.item(coordination_row, 3).text() == "2.5"
    assert window.results_stats_table.item(coordination_row, 4).text() == "2.5"
    assert window.results_stats_table.item(coordination_row, 5).text() == "0.5"
    assert (
        window.results_stats_table.item(coordination_row, 6).text() == "count"
    )
    assert window.results_stats_table.item(coordination_row, 7).text() == "-"


def test_bondanalysis_run_finishes_by_refreshing_results_tree(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.output_dir_edit.setText(str(tmp_path / "bondanalysis_async_run"))
    window._set_bond_pair_rows((BondPairDefinition("Pb", "I", 3.1),))
    window._set_angle_triplet_rows(
        (AngleTripletDefinition("Pb", "I", "I", 3.1, 3.1),)
    )

    window._start_run()
    for _attempt in range(200):
        qapp.processEvents()
        if window._run_thread is None:
            break
        QTest.qWait(25)

    assert window._run_thread is None, window._active_run_status
    assert window.run_button.isEnabled()
    assert window.progress_bar.value() == window.progress_bar.maximum()
    assert "complete" in window.progress_label.text().lower()
    assert "Bond analysis complete" in window.statusBar().currentMessage()
    assert window.results_tree.topLevelItemCount() == 2
    all_bond_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "all")
    all_angle_leaf = _find_results_leaf(
        window,
        "Bond Angles",
        "I-Pb-I",
        "all",
    )
    assert all_bond_leaf.text(2) == "2"
    assert all_angle_leaf.text(2) == "1"
    assert window.close()


def test_bondanalysis_can_reload_existing_results_folder_without_recomputing(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    shutil.rmtree(clusters_dir)

    window = BondAnalysisMainWindow()
    window.load_existing_results_dir(output_dir)

    assert window.output_dir_edit.text() == str(output_dir)
    assert window.clusters_dir_edit.text() == str(clusters_dir)
    assert window.bond_pair_table.rowCount() == 1
    assert window.bond_pair_table.item(0, 0).text() == "Pb"
    assert window.bond_pair_table.item(0, 1).text() == "I"
    assert window.angle_triplet_table.rowCount() == 1
    assert window.angle_triplet_table.item(0, 0).text() == "Pb"
    assert window.coordination_number_table.rowCount() == 1
    assert window.coordination_number_table.item(0, 0).text() == "Pb"
    assert window.results_tree.topLevelItemCount() == 3
    assert (
        "Loaded existing bondanalysis folder" in window.log_box.toPlainText()
    )


def test_bondanalysis_show_output_folder_button_opens_results_dir(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.load_existing_results_dir(output_dir)
    opened: dict[str, str] = {}

    def fake_open_url(url):
        opened["path"] = url.toLocalFile()
        return True

    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.main_window.QDesktopServices.openUrl",
        fake_open_url,
    )

    window.show_output_folder_button.click()

    assert opened["path"] == str(output_dir.resolve())
    window.close()


def test_bondanalysis_results_tree_selection_updates_ready_status(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.show()
    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    pb_i2_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI2")
    window.results_tree.clearSelection()
    pb_i2_leaf.setSelected(True)
    qapp.processEvents()

    assert (
        window.results_status_label.text()
        == "Ready to open Pb-I for PbI2 in a plot tab."
    )


def test_bondanalysis_open_selected_window_can_overlay_selected_clusters_and_all_leaf(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.show()
    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    pb_i2_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI2")
    pb_i3_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI3")
    window.results_tree.clearSelection()
    pb_i2_leaf.setSelected(True)
    pb_i3_leaf.setSelected(True)
    qapp.processEvents()
    opened_plot_windows = []
    window.plot_window_opened.connect(opened_plot_windows.append)
    window._open_selected_plot_window()

    assert len(window._plot_windows) == 1
    assert opened_plot_windows == [window._plot_windows[0]]
    overlay_window = window._plot_windows[0]
    legend = overlay_window.axis.get_legend()
    assert legend is not None
    assert {"PbI2", "PbI3", "Mean", "Median"} <= {
        text.get_text() for text in legend.get_texts()
    }
    assert overlay_window.series_color_container.isVisible()
    assert overlay_window.transparency_spin.isEnabled()

    all_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "all")
    window.results_tree.clearSelection()
    all_leaf.setSelected(True)
    qapp.processEvents()
    window._open_selected_plot_window()

    assert len(window._plot_windows) == 1
    assert opened_plot_windows == [window._plot_windows[0]]
    all_window = window._plot_windows[0]
    assert all_window.tab_widget.count() == 2
    assert all_window.tab_widget.currentIndex() == 1
    assert len(all_window.plot_request.series) == 1
    assert all_window.plot_request.series[0].label == "all"
    assert all_window.plot_request.series[0].values.size == 5
    all_legend = all_window.axis.get_legend()
    assert all_legend is not None
    assert {"all", "Mean", "Median"} <= {
        text.get_text() for text in all_legend.get_texts()
    }
    assert len(all_window.axis.patches) > 0
    assert any("Mean:" in text.get_text() for text in all_window.axis.texts)
    assert not all_window.series_color_container.isVisible()

    bond_group = _find_results_group(window, "Bond Pairs", "Pb-I")
    window.results_tree.clearSelection()
    bond_group.setSelected(True)
    qapp.processEvents()
    window._open_selected_plot_window()

    assert all_window.tab_widget.count() == 3
    assert all_window.plot_request.display_label == "Pb-I"
    assert all_window.plot_request.series[0].label == "all"

    angle_group = _find_results_group(window, "Bond Angles", "I-Pb-I")
    coordination_group = _find_results_group(
        window,
        "Coordination Numbers",
        "CN Pb-I",
    )
    window.results_tree.clearSelection()
    angle_group.setSelected(True)
    coordination_group.setSelected(True)
    qapp.processEvents()

    assert "Ready to open 2 selected all-cluster distributions" in (
        window.results_status_label.text()
    )

    window._open_selected_plot_window()

    assert all_window.tab_widget.count() == 5
    tab_titles = [
        all_window.tab_widget.widget(index).plot_request.title
        for index in range(all_window.tab_widget.count())
    ]
    assert "I-Pb-I across all cluster types" in tab_titles
    assert "CN Pb-I across all cluster types" in tab_titles


def test_bondanalysis_can_open_every_all_cluster_distribution(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.show()
    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    window.open_all_all_plots_button.click()
    qapp.processEvents()

    assert len(window._plot_windows) == 1
    plot_window = window._plot_windows[0]
    assert plot_window.tab_widget.count() == 3
    requests = [
        plot_window.tab_widget.widget(index).plot_request
        for index in range(plot_window.tab_widget.count())
    ]
    assert {request.display_label for request in requests} == {
        "Pb-I",
        "I-Pb-I",
        "CN Pb-I",
    }
    assert all(request.series[0].label == "all" for request in requests)
    assert all(request.series[0].values.size > 0 for request in requests)
    assert "Opened 3 all-cluster distribution plot(s)." in (
        window.results_status_label.text()
    )


def test_bondanalysis_right_pane_is_scrollable_with_tree_above_run_log(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    splitter = window.centralWidget().layout().itemAt(1).widget()

    assert isinstance(splitter, QSplitter)
    assert isinstance(splitter.widget(1), QScrollArea)

    right_panel = splitter.widget(1).widget()
    right_layout = right_panel.layout()
    browser_log_panel = right_layout.itemAt(2).widget()
    browser_log_layout = browser_log_panel.layout()

    assert isinstance(right_layout.itemAt(0).widget(), QGroupBox)
    assert isinstance(right_layout.itemAt(1).widget(), QGroupBox)
    assert isinstance(browser_log_layout.itemAt(0).widget(), QGroupBox)
    assert isinstance(browser_log_layout.itemAt(1).widget(), QGroupBox)
    assert (
        browser_log_layout.itemAt(0).widget().title()
        == "Computed Distributions"
    )
    assert browser_log_layout.itemAt(1).widget().title() == "Run Log"


def test_bondanalysis_close_prompts_for_safe_cancel_when_run_is_active(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    window = BondAnalysisMainWindow()
    warning_calls = []

    class RunningThread:
        def isRunning(self):
            return True

    class RunningWorker:
        def __init__(self):
            self.cancel_requested = False

        def request_cancel(self):
            self.cancel_requested = True

    class CloseEvent:
        def __init__(self):
            self.ignored = False

        def ignore(self):
            self.ignored = True

    def capture_warning(parent, title, message, *args):
        warning_calls.append((parent, title, message, args))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.main_window.QMessageBox.warning",
        capture_warning,
    )

    window._run_thread = RunningThread()
    window._run_worker = RunningWorker()
    window._update_progress(
        3,
        8,
        "Processing PbI: 10/100 structures (8 cached, 2 measured).",
    )
    event = CloseEvent()
    window.closeEvent(event)

    assert event.ignored
    assert window._run_cancel_requested
    assert window._close_after_run_cancel
    assert window._run_worker.cancel_requested
    assert warning_calls
    assert warning_calls[0][1] == "Cancel Bond Analysis?"
    assert "cancel it at the next safe checkpoint" in warning_calls[0][2]
    assert "Processing PbI: 10/100 structures" in (warning_calls[0][2])
    window._run_thread = None
    window._run_worker = None
    window.close()


def test_bondanalysis_can_open_selected_plot_in_tabbed_plot_workspace(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.show()
    window.output_dir_edit.setText(str(output_dir))
    window._refresh_results_tree()

    pb_i2_leaf = _find_results_leaf(window, "Bond Pairs", "Pb-I", "PbI2")
    window.results_tree.clearSelection()
    pb_i2_leaf.setSelected(True)
    qapp.processEvents()
    opened_plot_windows = []
    window.plot_window_opened.connect(opened_plot_windows.append)
    window._open_selected_plot_window()

    assert len(window._plot_windows) == 1
    plot_window = window._plot_windows[0]
    assert opened_plot_windows == [plot_window]
    assert isinstance(plot_window, BondAnalysisPlotWindow)
    assert plot_window.tab_widget.count() == 1
    assert "PbI2" in plot_window.windowTitle()
    assert plot_window.axis.get_title() == "PbI2 • Pb-I"


def test_bondanalysis_standalone_plot_window_saves_csv(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    _clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    result_index = load_result_index(output_dir)
    group = result_index.find_group("bond", "Pb-I")
    plot_request = build_plot_request(result_index, [group.all_leaf])
    plot_window = BondAnalysisPlotWindow(
        plot_request,
        default_output_dir=output_dir,
    )

    saved_csv_path = tmp_path / "saved_plot.csv"
    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.plot_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(saved_csv_path), "CSV Files (*.csv)"),
    )
    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.plot_window.QMessageBox.information",
        lambda *args, **kwargs: None,
    )

    plot_window.save_plot_data_as()

    assert saved_csv_path.exists()
    csv_lines = saved_csv_path.read_text().splitlines()
    assert csv_lines[0] == "Series,Value"
    assert all(line.startswith("all,") for line in csv_lines[1:])


def test_bondanalysis_plot_window_uses_horizontal_histogram_controls_and_stats(
    qapp,
    tmp_path,
    monkeypatch,
):
    del monkeypatch
    _clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    result_index = load_result_index(output_dir)
    group = result_index.find_group("bond", "Pb-I")
    plot_request = build_plot_request(
        result_index,
        [group.cluster_leaves[0], group.cluster_leaves[1]],
    )
    plot_window = BondAnalysisPlotWindow(
        plot_request,
        default_output_dir=output_dir,
    )
    plot_window.show()
    qapp.processEvents()

    assert isinstance(plot_window.controls_widget.layout(), QHBoxLayout)
    assert not hasattr(plot_window, "plot_mode_combo")
    assert plot_window.bin_size_spin.isEnabled()
    assert plot_window.transparency_spin.isEnabled()
    assert plot_window.series_color_container.isVisible()

    plot_window.bin_size_spin.setValue(0.2)
    plot_window.transparency_spin.setValue(0.35)
    plot_window.refresh_plot()

    assert len(plot_window.axis.patches) > 0
    assert to_hex(plot_window.axis.patches[0].get_facecolor()) == "#87ceeb"
    assert any(
        line.get_linestyle() == "--" and to_hex(line.get_color()) == "#000000"
        for line in plot_window.axis.lines
    )
    assert any(
        line.get_linestyle() == "--" and to_hex(line.get_color()) == "#ff0000"
        for line in plot_window.axis.lines
    )
    assert any(
        "Mean:" in text.get_text()
        and "Median:" in text.get_text()
        and "Mode:" in text.get_text()
        and "Width sigma:" in text.get_text()
        for text in plot_window.axis.texts
    )


def test_bondanalysis_dihedral_plot_uses_circular_and_bimodal_stats(
    qapp,
    tmp_path,
):
    values = np.array([-178.0, -174.0, -170.0, 170.0, 174.0, 178.0])
    plot_window = BondAnalysisPlotWindow(
        BondAnalysisPlotRequest(
            category="dihedral",
            display_label="Pb-O-C-N",
            title="Pb-O-C-N across all cluster types",
            xlabel="Dihedral (deg)",
            series=(
                BondAnalysisDistributionSeries(
                    label="all",
                    values=values,
                ),
            ),
        ),
        default_output_dir=tmp_path,
    )
    plot_window.show()
    qapp.processEvents()

    assert plot_window.dihedral_model_combo.isVisible()
    assert to_hex(plot_window.axis.patches[0].get_facecolor()) == "#2e8b57"
    legend_labels = [
        text.get_text() for text in plot_window.axis.get_legend().get_texts()
    ]
    assert "Circular center" in legend_labels
    stats_text = "\n".join(text.get_text() for text in plot_window.axis.texts)
    assert "Model: single circular" in stats_text
    assert "Circular center:" in stats_text
    assert "Width sigma:" in stats_text
    assert "Mean:" not in stats_text
    center_line = next(
        line
        for line in plot_window.axis.lines
        if line.get_label() == "Circular center"
    )
    assert float(center_line.get_xdata()[0]) == pytest.approx(180.0)
    assert plot_window.axis.get_xlim() == pytest.approx((-180.0, 180.0))
    assert plot_window.axis.get_xticks().tolist() == pytest.approx(
        [-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
    )
    bar_centers = [
        patch.get_x() + patch.get_width() / 2.0
        for patch in plot_window.axis.patches
        if patch.get_height() > 0
    ]
    assert min(bar_centers) < -169.0
    assert max(bar_centers) > 169.0
    tick_formatter = plot_window.axis.xaxis.get_major_formatter()
    assert tick_formatter(-180.0, None) == "-180"
    assert tick_formatter(180.0, None) == "180"
    assert tick_formatter(190.0, None) == "-170"

    plot_window.dihedral_model_combo.setCurrentIndex(1)
    qapp.processEvents()
    plot_window.refresh_plot()

    legend_labels = [
        text.get_text() for text in plot_window.axis.get_legend().get_texts()
    ]
    assert {"Center 1", "Center 2"} <= set(legend_labels)
    stats_text = "\n".join(text.get_text() for text in plot_window.axis.texts)
    assert "Model: bimodal circular" in stats_text
    assert "Pooled width sigma:" in stats_text


def test_bondanalysis_dihedral_plot_reload_wraps_legacy_0_to_360_values(
    qapp,
    tmp_path,
):
    legacy_npy_path = tmp_path / "O_C_N_C_dihedrals.npy"
    np.save(
        legacy_npy_path,
        np.array([0.0, 5.0, 180.0, 185.0, 190.0, 355.0, 359.0]),
    )
    leaf = BondAnalysisResultLeaf(
        category="dihedral",
        display_label="O-C-N-C",
        scope_name="legacy",
        npy_path=legacy_npy_path,
        point_count=7,
    )
    group = BondAnalysisResultGroup(
        category="dihedral",
        display_label="O-C-N-C",
        xlabel="Dihedral (deg)",
        cluster_leaves=(leaf,),
        all_leaf=BondAnalysisResultLeaf(
            category="dihedral",
            display_label="O-C-N-C",
            scope_name="all",
            npy_path=None,
            point_count=7,
            is_all=True,
        ),
    )
    result_index = BondAnalysisResultIndex(
        results_index_path=tmp_path / "bondanalysis_results_index.json",
        output_dir=tmp_path,
        clusters_dir=tmp_path,
        selected_cluster_types=(),
        cluster_type_names=("legacy",),
        bond_pairs=(),
        angle_triplets=(),
        dihedral_quartets=(),
        coordination_numbers=(),
        bond_groups=(),
        angle_groups=(),
        dihedral_groups=(group,),
        coordination_groups=(),
        gds_variable_registry=(),
    )

    plot_request = build_plot_request(result_index, [leaf])

    reloaded_values = plot_request.series[0].values
    assert np.min(reloaded_values) >= -180.0
    assert np.max(reloaded_values) <= 180.0
    assert sorted(np.round(reloaded_values, 6).tolist()) == [
        -175.0,
        -170.0,
        -5.0,
        -1.0,
        0.0,
        5.0,
        180.0,
    ]

    plot_window = BondAnalysisPlotWindow(
        plot_request,
        default_output_dir=tmp_path,
    )
    plot_window.show()
    qapp.processEvents()

    tick_formatter = plot_window.axis.xaxis.get_major_formatter()
    assert tick_formatter(185.0, None) == "-175"
    assert "185.000" not in "\n".join(
        text.get_text() for text in plot_window.axis.texts
    )


def test_bondanalysis_dihedral_plot_can_toggle_radial_histogram(
    qapp,
    tmp_path,
):
    plot_window = BondAnalysisPlotWindow(
        BondAnalysisPlotRequest(
            category="dihedral",
            display_label="O-C-N-C",
            title="O-C-N-C across all cluster types",
            xlabel="Dihedral (deg)",
            series=(
                BondAnalysisDistributionSeries(
                    label="terminal C2",
                    values=np.array([-178.0, -174.0, 170.0, 174.0, 178.0]),
                ),
                BondAnalysisDistributionSeries(
                    label="terminal C3",
                    values=np.array([-6.0, -2.0, 0.0, 2.0, 6.0]),
                ),
            ),
        ),
        default_output_dir=tmp_path,
    )
    plot_window.show()
    qapp.processEvents()

    assert plot_window.dihedral_plot_style_combo.isVisible()
    assert plot_window.axis.name == "rectilinear"

    plot_window.dihedral_plot_style_combo.setCurrentIndex(1)
    qapp.processEvents()

    assert plot_window.axis.name == "polar"
    assert len(plot_window.axis.patches) > 0
    legend_labels = [
        text.get_text() for text in plot_window.axis.get_legend().get_texts()
    ]
    assert {"terminal C2", "terminal C3"} <= set(legend_labels)
    assert {"Circular center", "Mode"} <= set(legend_labels)
    tick_labels = {
        label.get_text() for label in plot_window.axis.get_xticklabels()
    }
    assert {"0", "90", "180", "-90"} <= tick_labels
    stats_text = "\n".join(text.get_text() for text in plot_window.axis.texts)
    assert "Model: single circular" in stats_text

    plot_window.dihedral_plot_style_combo.setCurrentIndex(0)
    qapp.processEvents()

    assert plot_window.axis.name == "rectilinear"


def test_bondanalysis_non_coordination_default_bin_size_is_finer(
    qapp,
    tmp_path,
):
    plot_window = BondAnalysisPlotWindow(
        BondAnalysisPlotRequest(
            category="dihedral",
            display_label="Pb-O-C-N",
            title="Pb-O-C-N across all cluster types",
            xlabel="Dihedral (deg)",
            series=(
                BondAnalysisDistributionSeries(
                    label="all",
                    values=np.array([-180.0, 0.0, 180.0]),
                ),
            ),
        ),
        default_output_dir=tmp_path,
    )
    plot_window.show()
    qapp.processEvents()

    assert plot_window.bin_size_spin.value() == pytest.approx(5.0)


def test_bondanalysis_coordination_plot_uses_nonnegative_integer_bins(
    qapp,
    tmp_path,
):
    plot_window = BondAnalysisPlotWindow(
        BondAnalysisPlotRequest(
            category="coordination",
            display_label="CN Pb-I",
            title="CN Pb-I across all cluster types",
            xlabel="Coordination Number",
            series=(
                BondAnalysisDistributionSeries(
                    label="all",
                    values=np.array([0.0, 1.0, 1.0, 2.0]),
                ),
            ),
        ),
        default_output_dir=tmp_path,
    )
    plot_window.show()
    qapp.processEvents()

    assert plot_window.axis.get_xlim()[0] >= 0.0
    assert to_hex(plot_window.axis.patches[0].get_facecolor()) == "#8e44ad"
    assert list(plot_window.axis.get_xticks()) == [0, 1, 2]
    centers = sorted(
        round(patch.get_x() + patch.get_width() / 2)
        for patch in plot_window.axis.patches
        if patch.get_height() > 0
    )
    assert centers == [0, 1, 2]

    plot_window.bin_size_spin.setValue(3.0)
    plot_window.refresh_plot()

    assert plot_window.axis.get_xlim()[0] >= 0.0
    assert list(plot_window.axis.get_xticks()) == [0, 1, 2]
    centers = sorted(
        round(patch.get_x() + patch.get_width() / 2)
        for patch in plot_window.axis.patches
        if patch.get_height() > 0
    )
    assert centers == [0, 1, 2]


def test_bondanalysis_plot_window_tabs_switch_with_arrow_keys(
    qapp,
    tmp_path,
    monkeypatch,
):
    del monkeypatch
    _clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    result_index = load_result_index(output_dir)
    group = result_index.find_group("bond", "Pb-I")
    plot_window = BondAnalysisPlotWindow(
        build_plot_request(result_index, [group.cluster_leaves[0]]),
        default_output_dir=output_dir,
    )
    plot_window.add_plot_request(
        build_plot_request(result_index, [group.cluster_leaves[1]])
    )
    plot_window.show()
    plot_window.activateWindow()
    plot_window.setFocus()
    qapp.processEvents()

    assert plot_window.tab_widget.count() == 2
    assert plot_window.tab_widget.currentIndex() == 1

    QTest.keyClick(plot_window, Qt.Key.Key_Left)
    qapp.processEvents()
    assert plot_window.tab_widget.currentIndex() == 0
    assert plot_window.axis.get_title() == "PbI2 • Pb-I"

    QTest.keyClick(plot_window, Qt.Key.Key_Right)
    qapp.processEvents()
    assert plot_window.tab_widget.currentIndex() == 1
    assert plot_window.axis.get_title() == "PbI3 • Pb-I"


def test_bondanalysis_overlay_series_list_reorders_histogram_stacking(
    qapp,
    tmp_path,
    monkeypatch,
):
    del monkeypatch
    _clusters_dir, output_dir = _build_bondanalysis_output(tmp_path)
    result_index = load_result_index(output_dir)
    group = result_index.find_group("bond", "Pb-I")
    plot_window = BondAnalysisPlotWindow(
        build_plot_request(
            result_index,
            [group.cluster_leaves[0], group.cluster_leaves[1]],
        ),
        default_output_dir=output_dir,
    )
    plot_window.show()
    qapp.processEvents()

    assert [
        plot_window.series_color_list.item(i).text() for i in range(2)
    ] == [
        "PbI2",
        "PbI3",
    ]

    moved_item = plot_window.series_color_list.takeItem(0)
    plot_window.series_color_list.insertItem(1, moved_item)
    plot_window.current_plot_tab._on_series_order_changed()
    qapp.processEvents()

    assert [
        plot_window.series_color_list.item(i).text() for i in range(2)
    ] == [
        "PbI3",
        "PbI2",
    ]
    legend_labels = [
        text.get_text() for text in plot_window.axis.get_legend().get_texts()
    ]
    assert legend_labels[:2] == ["PbI3", "PbI2"]
