from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
from pathlib import Path
from typing import Callable, Mapping, Sequence

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisBatchResult,
    BondAnalysisPreset,
    BondAnalysisWorkflow,
    BondPairDefinition,
    CoordinationNumberDefinition,
    DihedralQuartetDefinition,
    load_presets,
    ordered_preset_names,
    save_custom_preset,
    suggest_bondanalysis_output_dir,
)
from saxshell.bondanalysis.results import (
    LEGACY_RESULTS_INDEX_FILENAME,
    RESULTS_INDEX_FILENAME,
    BondAnalysisPlotRequest,
    BondAnalysisResultIndex,
    BondAnalysisResultLeaf,
    build_plot_request,
    load_result_index,
)
from saxshell.bondanalysis.ui.plot_window import BondAnalysisPlotWindow
from saxshell.saxs.project_manager import (
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.ui.branding import (
    build_saxshell_stylesheet,
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
    track_saxshell_window,
)
from saxshell.saxs.ui.progress_dialog import SAXSProgressDialog
from saxshell.saxs.ui.project_status_label import CompactProjectStatusLabel
from saxshell.structure_distributions import (
    application_structure_distribution_store_dir,
)

_OPEN_WINDOWS: list["BondAnalysisMainWindow"] = []
BOND_ANALYSIS_WINDOW_LOAD_TOTAL_STEPS = 5
SELECTION_PREVIEW_TOTAL_STEPS = 4
SELECTION_PREVIEW_DEBOUNCE_MS = 350


def _safe_path_label(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() else "_" for char in str(value).strip()
    ).strip("_")
    return cleaned or "clusters"


def _has_results_index(output_dir: str | Path) -> bool:
    output_path = Path(output_dir)
    return any(
        (output_path / filename).is_file()
        for filename in (
            RESULTS_INDEX_FILENAME,
            LEGACY_RESULTS_INDEX_FILENAME,
        )
    )


def _result_index_paths_below(root: str | Path) -> tuple[Path, ...]:
    root_path = Path(root).expanduser()
    if root_path.is_file() and root_path.name in {
        RESULTS_INDEX_FILENAME,
        LEGACY_RESULTS_INDEX_FILENAME,
    }:
        return (root_path,)
    if not root_path.is_dir():
        return ()

    paths: list[Path] = []
    seen: set[Path] = set()
    candidate_dirs = [root_path]
    try:
        candidate_dirs.extend(
            child for child in root_path.iterdir() if child.is_dir()
        )
    except OSError:
        candidate_dirs = [root_path]

    for directory in candidate_dirs:
        for filename in (
            RESULTS_INDEX_FILENAME,
            LEGACY_RESULTS_INDEX_FILENAME,
        ):
            candidate = directory / filename
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(candidate)
    paths.sort(
        key=lambda path: (_safe_mtime(path), str(path)),
        reverse=True,
    )
    return tuple(paths)


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


class CollapsibleSection(QWidget):
    """Compact section with a disclosure-style title bar."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.toggle_button = QToolButton()
        self.content_widget = QFrame()
        self._build_ui(title)

    def _build_ui(self, title: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.toggle_button.setObjectName("CollapsibleSectionHeader")
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_button.toggled.connect(self._set_expanded)
        root.addWidget(self.toggle_button)

        self.content_widget.setObjectName("CollapsibleSectionContent")
        root.addWidget(self.content_widget)

    def set_content_layout(self, layout: QVBoxLayout) -> None:
        self.content_widget.setLayout(layout)

    def set_collapsed(self, collapsed: bool) -> None:
        self.toggle_button.setChecked(not collapsed)

    def is_collapsed(self) -> bool:
        return not self.toggle_button.isChecked()

    @Slot(bool)
    def _set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content_widget.setVisible(expanded)


class BondAnalysisWorker(QObject):
    """Background worker that runs one bond-analysis workflow."""

    log = Signal(str)
    progress = Signal(int, int, str)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    canceled = Signal(str)

    def __init__(self, workflow: BondAnalysisWorkflow) -> None:
        super().__init__()
        self.workflow = workflow
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        try:
            result = self.workflow.run(
                progress_callback=self._emit_progress,
                log_callback=self.log.emit,
                cancel_callback=self._cancel_requested.is_set,
            )
            self.finished.emit(result)
        except InterruptedError as exc:
            self.canceled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.progress.emit(processed, total, message)
        self.status.emit(message)


class BondAnalysisMainWindow(QMainWindow):
    """Main Qt window for bond-pair and angle-distribution analysis."""

    plot_window_opened = Signal(object)

    def __init__(
        self,
        initial_clusters_dir: str | Path | None = None,
        initial_project_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project_manager = SAXSProjectManager()
        self._project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._run_thread: QThread | None = None
        self._run_worker: BondAnalysisWorker | None = None
        self._active_run_status = ""
        self._run_is_saving_distribution_outputs = False
        self._run_cancel_requested = False
        self._close_after_run_cancel = False
        self._presets: dict[str, BondAnalysisPreset] = {}
        self._results_index: BondAnalysisResultIndex | None = None
        self._stored_result_indices_cache: dict[
            tuple[str, ...],
            tuple[BondAnalysisResultIndex, ...],
        ] = {}
        self._plot_windows: list[BondAnalysisPlotWindow] = []
        self._startup_progress_dialog: SAXSProgressDialog | None = None
        self._last_startup_load_message = ""
        self._selection_summary_progress_dialog: SAXSProgressDialog | None = (
            None
        )
        self._last_selection_summary_progress_message = ""
        self._selection_summary_pending_show_progress = False
        self._selection_summary_pending_reason = (
            "Refreshing selection preview..."
        )
        self._selection_summary_timer = QTimer(self)
        self._selection_summary_timer.setSingleShot(True)
        self._selection_summary_timer.setInterval(
            SELECTION_PREVIEW_DEBOUNCE_MS
        )
        self._selection_summary_timer.timeout.connect(
            self._run_scheduled_selection_summary_update
        )
        self._begin_startup_load_progress("Preparing Bond Analysis window...")
        try:
            self._update_startup_load_progress(
                1,
                "Preparing Bond Analysis window...",
                log_message="Preparing Bond Analysis window.",
            )
            self._build_ui()
            self._update_startup_load_progress(
                2,
                "Loading bond-analysis presets...",
                log_message=(
                    "Loading built-in and custom bond-analysis presets."
                ),
            )
            self._reload_presets()
            if initial_clusters_dir is not None:
                self._update_startup_load_progress(
                    3,
                    "Inspecting initial clusters directory...",
                    log_message=(
                        "Inspecting initial clusters directory: "
                        f"{Path(initial_clusters_dir).expanduser()}"
                    ),
                )
                self.set_clusters_dir(
                    initial_clusters_dir,
                    progress_callback=(
                        self._update_startup_cluster_scan_progress
                    ),
                )
                self._load_default_existing_results_if_available()
                self._update_startup_load_progress(
                    4,
                    (
                        "Loaded "
                        f"{self.cluster_type_list.count()} cluster type(s)."
                    ),
                    log_message=(
                        "Loaded "
                        f"{self.cluster_type_list.count()} cluster type(s) "
                        "from the initial clusters directory."
                    ),
                )
            else:
                self._update_startup_load_progress(
                    3,
                    "Preparing empty bond-analysis workspace...",
                    log_message=(
                        "No initial clusters directory was supplied."
                    ),
                )
                self._update_selection_summary()
                self._update_startup_load_progress(
                    4,
                    "Waiting for a clusters directory.",
                    log_message=(
                        "Bond Analysis is waiting for a clusters directory."
                    ),
                )
            self._update_startup_load_progress(
                BOND_ANALYSIS_WINDOW_LOAD_TOTAL_STEPS,
                "Bond Analysis window ready.",
                log_message="Bond Analysis window is ready.",
            )
        finally:
            self._close_startup_load_progress_dialog()

    def closeEvent(self, event) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            if self._close_after_run_cancel:
                event.ignore()
                return
            response = QMessageBox.warning(
                self,
                "Cancel Bond Analysis?",
                "A bond-analysis run is still active. You can cancel it at "
                "the next safe checkpoint and close this window, or keep the "
                "run going.\n\nCurrent step: "
                f"{self._active_run_status or 'running'}",
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                ),
                QMessageBox.StandardButton.No,
            )
            if response == QMessageBox.StandardButton.Yes:
                self._request_run_cancel(close_when_finished=True)
                self.hide()
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell - Bond Analysis")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1320, 840)

        central = QWidget()
        central.setObjectName("BondAnalysisCentral")
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.project_banner = None

        root.addWidget(self._build_header(), stretch=0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setObjectName("BondAnalysisSplitter")
        splitter.setSizes([560, 760])

        root.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)
        self._apply_presentation_style()
        self.project_status_label = self._build_project_status_label()
        if self.project_status_label is not None:
            self.statusBar().addPermanentWidget(self.project_status_label)
        self.statusBar().showMessage("Ready")

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("BondAnalysisHeader")
        header.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(12)

        title_block = QWidget()
        title_block.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        title_label = QLabel("Bond Analysis")
        title_label.setObjectName("BondAnalysisTitle")
        title_layout.addWidget(title_label)

        subtitle_label = QLabel(
            "Bond, angle, dihedral, and coordination-distribution workspace"
        )
        subtitle_label.setObjectName("BondAnalysisSubtitle")
        title_layout.addWidget(subtitle_label)

        header_layout.addWidget(title_block, stretch=1)

        context_text = (
            "Linked project" if self._project_dir is not None else "Standalone"
        )
        self.window_context_label = QLabel(context_text)
        self.window_context_label.setObjectName("BondAnalysisContextPill")
        self.window_context_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.window_context_label.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        header_layout.addWidget(
            self.window_context_label,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )
        return header

    def _apply_presentation_style(self) -> None:
        app = QApplication.instance()
        if app is not None and app.styleSheet() == build_saxshell_stylesheet():
            return
        self.setStyleSheet(build_saxshell_stylesheet())

    def _begin_startup_load_progress(self, message: str) -> None:
        dialog = SAXSProgressDialog(self)
        self._startup_progress_dialog = dialog
        self._last_startup_load_message = ""
        dialog.begin(
            BOND_ANALYSIS_WINDOW_LOAD_TOTAL_STEPS,
            message,
            unit_label="steps",
            title="Opening Bond Analysis",
        )
        QApplication.processEvents()

    def _update_startup_load_progress(
        self,
        processed: int,
        message: str,
        *,
        log_message: str | None = None,
    ) -> None:
        if self._startup_progress_dialog is None:
            return
        self._startup_progress_dialog.update_progress(
            processed,
            BOND_ANALYSIS_WINDOW_LOAD_TOTAL_STEPS,
            message,
            unit_label="steps",
        )
        self._append_startup_load_output(log_message or message)
        if self.centralWidget() is not None:
            self.statusBar().showMessage(message)
        QApplication.processEvents()

    def _update_startup_cluster_scan_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        if self._startup_progress_dialog is None:
            return
        self._startup_progress_dialog.update_progress(
            processed,
            total,
            message,
            unit_label="folders",
        )
        self._append_startup_load_output(message)
        if self.centralWidget() is not None:
            self.statusBar().showMessage(message)
        QApplication.processEvents()

    def _append_startup_load_output(self, message: str) -> None:
        stripped = str(message).strip()
        if (
            not stripped
            or self._startup_progress_dialog is None
            or stripped == self._last_startup_load_message
        ):
            return
        self._last_startup_load_message = stripped
        self._startup_progress_dialog.append_output(stripped)

    def _close_startup_load_progress_dialog(self) -> None:
        self._last_startup_load_message = ""
        if self._startup_progress_dialog is not None:
            self._startup_progress_dialog.close()

    def _begin_selection_summary_progress(self, message: str) -> None:
        dialog = self._selection_summary_progress_dialog
        if dialog is None:
            dialog = SAXSProgressDialog(self)
            self._selection_summary_progress_dialog = dialog
        self._last_selection_summary_progress_message = ""
        dialog.begin(
            SELECTION_PREVIEW_TOTAL_STEPS,
            message,
            unit_label="steps",
            title="Updating Selection Preview",
        )
        self._append_selection_summary_progress_output(message)
        QApplication.processEvents()

    def _update_selection_summary_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        if self._selection_summary_progress_dialog is None:
            return
        self._selection_summary_progress_dialog.update_progress(
            processed,
            total,
            message,
            unit_label="steps",
        )
        self._append_selection_summary_progress_output(message)
        self.statusBar().showMessage(message)
        QApplication.processEvents()

    def _append_selection_summary_progress_output(self, message: str) -> None:
        stripped = str(message).strip()
        if (
            not stripped
            or self._selection_summary_progress_dialog is None
            or stripped == self._last_selection_summary_progress_message
        ):
            return
        self._last_selection_summary_progress_message = stripped
        self._selection_summary_progress_dialog.append_output(stripped)

    def _close_selection_summary_progress_dialog(self) -> None:
        self._last_selection_summary_progress_message = ""
        if self._selection_summary_progress_dialog is not None:
            self._selection_summary_progress_dialog.close()

    def _schedule_selection_summary_update(
        self,
        *_args,
        reason: str = "Selection changed; refreshing preview...",
        show_progress: bool = False,
    ) -> None:
        if not hasattr(self, "_selection_summary_timer"):
            return
        self._selection_summary_pending_reason = reason
        self._selection_summary_pending_show_progress = (
            self._selection_summary_pending_show_progress or show_progress
        )
        self._selection_summary_timer.start(SELECTION_PREVIEW_DEBOUNCE_MS)

    def _run_scheduled_selection_summary_update(self) -> None:
        show_progress = self._selection_summary_pending_show_progress
        reason = self._selection_summary_pending_reason
        self._selection_summary_pending_show_progress = False
        self._selection_summary_pending_reason = (
            "Refreshing selection preview..."
        )
        self._update_selection_summary(
            show_progress=show_progress,
            progress_message=reason,
        )

    def _load_project_settings(self) -> ProjectSettings | None:
        if self._project_dir is None:
            return None
        project_file = build_project_paths(self._project_dir).project_file
        if not project_file.is_file():
            return None
        try:
            return self._project_manager.load_project(self._project_dir)
        except Exception:
            return None

    def _project_status_text(self) -> str | None:
        if self._project_dir is None:
            return None
        return f"Active project: {self._project_dir}"

    def _project_status_tooltip(self) -> str | None:
        if self._project_dir is None:
            return None
        settings = self._load_project_settings()
        project_name = (
            self._project_dir.name
            if settings is None
            else settings.project_name.strip() or self._project_dir.name
        )
        return (
            f"Active project: {project_name}\n"
            f"{self._project_dir}\n\n"
            "This window is linked to the active SAXS project, so a "
            "selected clusters folder is saved back to that project."
        )

    def _build_project_status_label(
        self,
    ) -> CompactProjectStatusLabel | None:
        status_text = self._project_status_text()
        if status_text is None:
            return None
        label = CompactProjectStatusLabel(self.statusBar())
        label.setToolTip(self._project_status_tooltip() or "")
        label.set_full_text(status_text)
        return label

    def _register_project_clusters_dir(self) -> None:
        settings = self._load_project_settings()
        if settings is None:
            return
        try:
            clusters_dir = self._clusters_dir_path()
            settings.clusters_dir = (
                None
                if clusters_dir is None
                else str(clusters_dir.expanduser().resolve())
            )
            self._project_manager.save_project(settings)
        except Exception:
            return

    def _project_bondanalysis_output_dir(
        self,
        clusters_dir: str | Path,
    ) -> Path | None:
        if self._project_dir is None:
            return None
        return (
            build_project_paths(self._project_dir).analysis_dir
            / "bondanalysis"
            / _safe_path_label(Path(clusters_dir).name)
        )

    def _suggest_output_dir_for_clusters(
        self,
        clusters_dir: str | Path,
    ) -> Path:
        project_output_dir = self._project_bondanalysis_output_dir(
            clusters_dir
        )
        if project_output_dir is not None:
            return project_output_dir
        return suggest_bondanalysis_output_dir(clusters_dir)

    def _load_default_existing_results_if_available(self) -> None:
        output_dir = self._output_dir_path()
        candidate_dirs: list[Path] = []
        if output_dir is not None and _has_results_index(output_dir):
            candidate_dirs.append(output_dir)
        clusters_dir = self._clusters_dir_path()
        candidate_dirs.extend(
            self._discover_existing_result_dirs_for_clusters(clusters_dir)
        )
        seen: set[Path] = set()
        for candidate_dir in candidate_dirs:
            resolved = candidate_dir.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                self.load_existing_results_dir(candidate_dir)
                return
            except Exception as exc:
                self._append_log(
                    "Unable to load existing bondanalysis results from "
                    f"{candidate_dir}: {exc}"
                )

    def _discover_existing_result_dirs_for_clusters(
        self,
        clusters_dir: Path | None,
    ) -> list[Path]:
        return [
            result_index.output_dir
            for result_index in (
                self._discover_existing_result_indices_for_clusters(
                    clusters_dir
                )
            )
        ]

    def _discover_existing_result_indices_for_clusters(
        self,
        clusters_dir: Path | None,
    ) -> list[BondAnalysisResultIndex]:
        if clusters_dir is None:
            return []
        roots = self._result_index_search_roots(clusters_dir)
        cache_key = self._stored_result_indices_cache_key(
            clusters_dir,
            roots,
        )
        cached_indices = self._stored_result_indices_cache.get(cache_key)
        if cached_indices is not None:
            return list(cached_indices)

        expected_clusters_dir = clusters_dir.expanduser().resolve()
        result_indices: list[BondAnalysisResultIndex] = []
        seen_paths: set[Path] = set()
        for root in roots:
            for index_path in _result_index_paths_below(root):
                resolved_index = index_path.resolve()
                if resolved_index in seen_paths:
                    continue
                seen_paths.add(resolved_index)
                try:
                    result_index = load_result_index(index_path.parent)
                except Exception:
                    continue
                if (
                    result_index.clusters_dir.expanduser().resolve()
                    != expected_clusters_dir
                ):
                    continue
                result_indices.append(result_index)
        result_indices.sort(
            key=lambda result_index: (
                result_index.results_index_path.stat().st_mtime,
                str(result_index.results_index_path),
            ),
            reverse=True,
        )
        self._stored_result_indices_cache[cache_key] = tuple(result_indices)
        return result_indices

    def _result_index_search_roots(self, clusters_dir: Path) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()

        def append_root(root: Path) -> None:
            root = root.expanduser()
            try:
                resolved = root.resolve()
            except OSError:
                resolved = root
            if resolved in seen:
                return
            seen.add(resolved)
            roots.append(root)

        output_dir = self._output_dir_path()
        if output_dir is not None:
            append_root(output_dir.parent)
        if self._project_dir is not None:
            append_root(
                build_project_paths(self._project_dir).analysis_dir
                / "bondanalysis"
            )
        append_root(clusters_dir.expanduser().parent)
        return roots

    def _stored_result_indices_cache_key(
        self,
        clusters_dir: Path,
        roots: Sequence[Path],
    ) -> tuple[str, ...]:
        try:
            clusters_token = str(clusters_dir.expanduser().resolve())
        except OSError:
            clusters_token = str(clusters_dir.expanduser())
        return (
            clusters_token,
            *(self._path_cache_token(root) for root in roots),
        )

    @staticmethod
    def _path_cache_token(path: Path) -> str:
        expanded = path.expanduser()
        try:
            resolved = expanded.resolve()
        except OSError:
            resolved = expanded
        try:
            mtime_ns = resolved.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        return f"{resolved}:{mtime_ns}"

    def _invalidate_stored_result_indices_cache(self) -> None:
        self._stored_result_indices_cache.clear()

    def _stored_results_preview_lines(
        self,
        *,
        clusters_dir: Path,
        selected_cluster_types: list[str] | None,
        bond_pairs: Sequence[BondPairDefinition] | None,
        angle_triplets: Sequence[AngleTripletDefinition] | None,
        dihedral_quartets: Sequence[DihedralQuartetDefinition] | None,
        coordination_numbers: Sequence[CoordinationNumberDefinition] | None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[str]:
        if progress_callback is not None:
            progress_callback(
                2,
                SELECTION_PREVIEW_TOTAL_STEPS,
                "Searching for stored bond-analysis results...",
            )
        result_indices = self._discover_existing_result_indices_for_clusters(
            clusters_dir
        )
        if not result_indices:
            if progress_callback is not None:
                progress_callback(
                    SELECTION_PREVIEW_TOTAL_STEPS,
                    SELECTION_PREVIEW_TOTAL_STEPS,
                    "Selection preview ready.",
                )
            return [
                "Stored computed runs for this clusters directory: none found"
            ]

        if progress_callback is not None:
            progress_callback(
                3,
                SELECTION_PREVIEW_TOTAL_STEPS,
                "Comparing current selections to stored runs...",
            )
        lines = [
            "Stored computed runs for this clusters directory: "
            f"{len(result_indices)}"
        ]
        current_signature = self._current_selection_analysis_signature(
            clusters_dir=clusters_dir,
            selected_cluster_types=selected_cluster_types,
            bond_pairs=bond_pairs,
            angle_triplets=angle_triplets,
            dihedral_quartets=dihedral_quartets,
            coordination_numbers=coordination_numbers,
        )
        exact_match = None
        if current_signature is not None:
            exact_match = next(
                (
                    result_index
                    for result_index in result_indices
                    if result_index.analysis_signature == current_signature
                ),
                None,
            )
        if exact_match is not None:
            lines.append(
                "Current settings match stored run: "
                f"{exact_match.output_dir} (Run will reuse without "
                "recalculation)."
            )

        for result_index in result_indices[:3]:
            summary = self._stored_result_distribution_summary(result_index)
            cluster_count = len(result_index.cluster_type_names)
            gds_count = len(result_index.gds_variable_registry)
            exact_suffix = (
                "; exact current settings"
                if result_index is exact_match
                else ""
            )
            lines.append(
                f"- {result_index.output_dir}: "
                f"{cluster_count} cluster type(s); {summary}; "
                f"{gds_count} GDS variable(s){exact_suffix}"
            )
        if len(result_indices) > 3:
            lines.append(
                f"- ... {len(result_indices) - 3} older stored run(s) also "
                "available"
            )
        if progress_callback is not None:
            progress_callback(
                SELECTION_PREVIEW_TOTAL_STEPS,
                SELECTION_PREVIEW_TOTAL_STEPS,
                "Selection preview ready.",
            )
        return lines

    def _current_selection_analysis_signature(
        self,
        *,
        clusters_dir: Path,
        selected_cluster_types: list[str] | None,
        bond_pairs: Sequence[BondPairDefinition] | None,
        angle_triplets: Sequence[AngleTripletDefinition] | None,
        dihedral_quartets: Sequence[DihedralQuartetDefinition] | None,
        coordination_numbers: Sequence[CoordinationNumberDefinition] | None,
    ) -> str | None:
        if (
            bond_pairs is None
            or angle_triplets is None
            or dihedral_quartets is None
            or coordination_numbers is None
        ):
            return None
        if not (
            bond_pairs
            or angle_triplets
            or dihedral_quartets
            or coordination_numbers
        ):
            return None
        try:
            workflow = BondAnalysisWorkflow(
                clusters_dir,
                bond_pairs=bond_pairs,
                angle_triplets=angle_triplets,
                dihedral_quartets=dihedral_quartets,
                coordination_numbers=coordination_numbers,
                output_dir=self._output_dir_path(),
                selected_cluster_types=selected_cluster_types,
            )
            return workflow.analysis_signature()
        except Exception:
            return None

    @staticmethod
    def _stored_result_distribution_summary(
        result_index: BondAnalysisResultIndex,
    ) -> str:
        parts = [
            BondAnalysisMainWindow._count_label(
                len(result_index.bond_groups),
                "bond pair",
                "bond pairs",
            ),
            BondAnalysisMainWindow._count_label(
                len(result_index.angle_groups),
                "angle",
                "angles",
            ),
            BondAnalysisMainWindow._count_label(
                len(result_index.dihedral_groups),
                "dihedral",
                "dihedrals",
            ),
            BondAnalysisMainWindow._count_label(
                len(result_index.coordination_groups),
                "coordination rule",
                "coordination rules",
            ),
        ]
        nonzero_parts = [part for part in parts if part]
        return ", ".join(nonzero_parts) or "no browsable distributions"

    @staticmethod
    def _count_label(count: int, singular: str, plural: str) -> str:
        if count <= 0:
            return ""
        label = singular if count == 1 else plural
        return f"{count} {label}"

    def _build_left_panel(self) -> QWidget:
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_presets_group())
        layout.addWidget(self._build_cluster_types_group())
        layout.addWidget(self._build_bond_pairs_group())
        layout.addWidget(self._build_angle_triplets_group())
        layout.addWidget(self._build_dihedral_quartets_group())
        layout.addWidget(self._build_coordination_numbers_group())
        layout.addStretch(1)

        return self._wrap_scroll_area(left)

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        preview_group = QGroupBox("Selection Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.selection_box = QTextEdit()
        self.selection_box.setReadOnly(True)
        self.selection_box.setMinimumHeight(150)
        preview_layout.addWidget(self.selection_box)
        layout.addWidget(preview_group)

        run_group = QGroupBox("Run")
        run_layout = QVBoxLayout(run_group)
        self.run_button = QPushButton(
            "Analyze Bond, Angle, Dihedral, and Coordination Distributions"
        )
        self.run_button.setObjectName("PrimaryActionButton")
        self.run_button.clicked.connect(self._start_run)
        run_layout.addWidget(self.run_button)

        self.progress_label = QLabel("Progress: idle")
        self.progress_label.setWordWrap(True)
        run_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m files")
        run_layout.addWidget(self.progress_bar)

        self.legacy_label = QLabel(
            "Legacy note: displacement analysis is deprecated and is not "
            "part of this interface until that workflow is updated."
        )
        self.legacy_label.setWordWrap(True)
        self.legacy_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.legacy_label.setObjectName("MutedPanelLabel")
        run_layout.addWidget(self.legacy_label)
        layout.addWidget(run_group)

        browser_log_panel = QWidget()
        browser_log_layout = QVBoxLayout(browser_log_panel)
        browser_log_layout.setContentsMargins(0, 0, 0, 0)
        browser_log_layout.setSpacing(12)
        browser_log_layout.addWidget(self._build_results_browser_group())

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        log_layout.addWidget(self.log_box)
        browser_log_layout.addWidget(log_group)

        layout.addWidget(browser_log_panel, stretch=1)

        layout.addStretch(1)
        return self._wrap_scroll_area(right)

    def _build_results_browser_group(self) -> QGroupBox:
        group = QGroupBox("Computed Distributions")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.refresh_results_button = QPushButton("Refresh Results")
        self.refresh_results_button.clicked.connect(self._refresh_results_tree)
        controls.addWidget(self.refresh_results_button)

        self.open_selected_window_button = QPushButton("Open Selected in Tab")
        self.open_selected_window_button.clicked.connect(
            self._open_selected_plot_window
        )
        controls.addWidget(self.open_selected_window_button)

        self.open_all_all_plots_button = QPushButton(
            "Open All 'All' Plot Tabs"
        )
        self.open_all_all_plots_button.setToolTip(
            "Open every non-empty all-cluster distribution as tabs in the "
            "plot workspace."
        )
        self.open_all_all_plots_button.clicked.connect(
            self._open_all_all_cluster_plot_windows
        )
        controls.addWidget(self.open_all_all_plots_button)

        self.show_output_folder_button = QPushButton("Show Output Folder")
        self.show_output_folder_button.setToolTip(
            "Open the computed bond, angle, dihedral, and coordination "
            "output folder in Finder."
        )
        self.show_output_folder_button.clicked.connect(
            self._show_output_folder
        )
        controls.addWidget(self.show_output_folder_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.results_tree = QTreeWidget()
        self.results_tree.setColumnCount(3)
        self.results_tree.setHeaderLabels(["Distribution", "Scope", "Values"])
        self.results_tree.setAlternatingRowColors(True)
        self.results_tree.setAnimated(True)
        self.results_tree.setItemsExpandable(True)
        self.results_tree.setRootIsDecorated(True)
        self.results_tree.setExpandsOnDoubleClick(True)
        self.results_tree.setMinimumHeight(220)
        self.results_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.results_tree.itemSelectionChanged.connect(
            self._on_results_tree_selection_changed
        )
        self.results_tree.header().setStretchLastSection(False)
        self.results_tree.header().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Stretch,
        )
        self.results_tree.header().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.results_tree.header().setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        layout.addWidget(self.results_tree, stretch=1)

        self.results_stats_table = QTableWidget(0, 9)
        self.results_stats_table.setHorizontalHeaderLabels(
            [
                "Type",
                "Distribution",
                "N",
                "Average",
                "Median",
                "Sigma",
                "Unit",
                "GDS Center Var",
                "GDS Sigma Var",
            ]
        )
        self.results_stats_table.setAlternatingRowColors(True)
        self.results_stats_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.results_stats_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.results_stats_table.verticalHeader().setVisible(False)
        self.results_stats_table.setMinimumHeight(150)
        self.results_stats_table.horizontalHeader().setStretchLastSection(True)
        self.results_stats_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )
        for column in (0, 2, 3, 4, 5, 6):
            self.results_stats_table.horizontalHeader().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        layout.addWidget(self.results_stats_table)

        self.results_hint_label = QLabel(
            "Select one computed bond, angle, dihedral, or coordination "
            "distribution and use 'Open Selected in Tab' to view it. "
            "Select a distribution-name row to open its all-cluster plot "
            "without expanding that distribution; Cmd-click or Ctrl-click "
            "distribution-name rows to open multiple all-cluster plots. "
            "Select multiple leaves of the same type across different "
            "cluster types to overlay them together in a plot tab. "
            "The 'all' entry opens that distribution across all cluster "
            "types, and \"Open All 'All' Plot Tabs\" opens every non-empty "
            "all-cluster distribution in the plot workspace."
        )
        self.results_hint_label.setWordWrap(True)
        layout.addWidget(self.results_hint_label)

        self.results_status_label = QLabel(
            "Run bondanalysis or refresh an existing output directory to "
            "browse computed distributions."
        )
        self.results_status_label.setWordWrap(True)
        self.results_status_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.results_status_label.setObjectName("MutedPanelLabel")
        layout.addWidget(self.results_status_label)
        return group

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Workspace")
        layout = QFormLayout(group)

        self.clusters_dir_edit = QLineEdit()
        self.clusters_dir_edit.textChanged.connect(
            self._on_clusters_dir_changed
        )
        self.clusters_dir_edit.editingFinished.connect(
            self._register_project_clusters_dir
        )
        layout.addRow(
            "Clusters directory",
            self._make_dir_row(
                self.clusters_dir_edit,
                "Select clusters directory",
            ),
        )

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.textChanged.connect(
            lambda _text: self._update_selection_summary()
        )
        layout.addRow(
            "Output directory",
            self._make_dir_row(
                self.output_dir_edit,
                "Select bondanalysis output directory",
            ),
        )

        refresh_button = QPushButton("Refresh Cluster Types")
        refresh_button.clicked.connect(self._refresh_cluster_types)
        layout.addRow("", refresh_button)

        load_existing_button = QPushButton("Load Existing Bondanalysis Folder")
        load_existing_button.clicked.connect(self._choose_existing_results_dir)
        layout.addRow("", load_existing_button)
        return group

    def _build_presets_group(self) -> QGroupBox:
        group = QGroupBox("Presets")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip(
            "Load a built-in bondanalysis preset or a custom preset saved "
            "from a previous session."
        )
        row.addWidget(self.preset_combo, stretch=1)

        load_button = QPushButton("Load")
        load_button.clicked.connect(self._load_selected_preset)
        row.addWidget(load_button)

        save_button = QPushButton("Save Current As...")
        save_button.clicked.connect(self._save_current_as_preset)
        row.addWidget(save_button)
        layout.addLayout(row)

        self.preset_hint_label = QLabel(
            "Built-in presets: DMSO and DMF. Custom presets are saved for "
            "later sessions."
        )
        self.preset_hint_label.setWordWrap(True)
        layout.addWidget(self.preset_hint_label)
        return group

    def _build_cluster_types_group(self) -> CollapsibleSection:
        self.cluster_types_section = CollapsibleSection("Cluster Types")
        self.cluster_types_content = self.cluster_types_section.content_widget
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.use_checked_cluster_types_box = QCheckBox(
            "Analyze all cluster types"
        )
        self.use_checked_cluster_types_box.setChecked(True)
        layout.addWidget(self.use_checked_cluster_types_box)

        self.cluster_type_list = QListWidget()
        self.cluster_type_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.cluster_type_list.setMinimumHeight(120)
        self.cluster_type_list.setMaximumHeight(220)
        self.cluster_type_list.itemChanged.connect(
            lambda _item: self._update_selection_summary()
        )
        layout.addWidget(self.cluster_type_list)
        self.use_checked_cluster_types_box.toggled.connect(
            self._on_analyze_all_cluster_types_toggled
        )

        self.cluster_type_status_label = QLabel("No clusters loaded.")
        self.cluster_type_status_label.setWordWrap(True)
        self.cluster_type_status_label.setObjectName("MutedPanelLabel")
        layout.addWidget(self.cluster_type_status_label)

        self.cluster_types_section.set_content_layout(layout)
        self._update_cluster_type_interactivity()
        self.cluster_types_section.set_collapsed(True)
        return self.cluster_types_section

    def _build_bond_pairs_group(self) -> QGroupBox:
        group = QGroupBox("Bond Pairs")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        add_button = QPushButton("Add Bond Pair")
        add_button.clicked.connect(self._add_bond_pair_row)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_bond_pair_rows)
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.bond_pair_table = QTableWidget(0, 3)
        self.bond_pair_table.setHorizontalHeaderLabels(
            ["Atom 1", "Atom 2", "Cutoff (A)"]
        )
        self.bond_pair_table.setAlternatingRowColors(True)
        self.bond_pair_table.verticalHeader().setVisible(False)
        self.bond_pair_table.setMinimumHeight(140)
        self.bond_pair_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.bond_pair_table)
        self._add_bond_pair_row()
        self.bond_pair_table.itemChanged.connect(
            lambda _item: self._schedule_selection_summary_update(
                reason="Bond pair settings changed; refreshing preview..."
            )
        )
        return group

    def _build_angle_triplets_group(self) -> QGroupBox:
        group = QGroupBox("Angle Triplets")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        add_button = QPushButton("Add Angle Triplet")
        add_button.clicked.connect(self._add_angle_triplet_row)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_angle_triplet_rows)
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.angle_triplet_table = QTableWidget(0, 5)
        self.angle_triplet_table.setHorizontalHeaderLabels(
            [
                "Vertex",
                "Arm 1",
                "Arm 2",
                "Vertex-Arm 1 Cutoff (A)",
                "Vertex-Arm 2 Cutoff (A)",
            ]
        )
        self.angle_triplet_table.setAlternatingRowColors(True)
        self.angle_triplet_table.verticalHeader().setVisible(False)
        self.angle_triplet_table.setMinimumHeight(150)
        self.angle_triplet_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.angle_triplet_table)
        self._add_angle_triplet_row()
        self.angle_triplet_table.itemChanged.connect(
            lambda _item: self._schedule_selection_summary_update(
                reason="Angle triplet settings changed; refreshing preview..."
            )
        )
        return group

    def _build_dihedral_quartets_group(self) -> QGroupBox:
        group = QGroupBox("Dihedral Quartets")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        add_button = QPushButton("Add Dihedral Quartet")
        add_button.clicked.connect(self._add_dihedral_quartet_row)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(
            self._remove_selected_dihedral_quartet_rows
        )
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.dihedral_quartet_table = QTableWidget(0, 7)
        self.dihedral_quartet_table.setHorizontalHeaderLabels(
            [
                "Atom 1",
                "Atom 2",
                "Atom 3",
                "Atom 4",
                "1-2 Cutoff (A)",
                "2-3 Cutoff (A)",
                "3-4 Cutoff (A)",
            ]
        )
        self.dihedral_quartet_table.setAlternatingRowColors(True)
        self.dihedral_quartet_table.verticalHeader().setVisible(False)
        self.dihedral_quartet_table.setMinimumHeight(150)
        self.dihedral_quartet_table.horizontalHeader().setStretchLastSection(
            True
        )
        layout.addWidget(self.dihedral_quartet_table)
        self._add_dihedral_quartet_row()
        self.dihedral_quartet_table.itemChanged.connect(
            lambda _item: self._schedule_selection_summary_update(
                reason=(
                    "Dihedral quartet settings changed; refreshing preview..."
                )
            )
        )
        return group

    def _build_coordination_numbers_group(self) -> QGroupBox:
        group = QGroupBox("Coordination Numbers")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        add_button = QPushButton("Add Coordination Rule")
        add_button.clicked.connect(self._add_coordination_number_row)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(
            self._remove_selected_coordination_number_rows
        )
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.coordination_number_table = QTableWidget(0, 3)
        self.coordination_number_table.setHorizontalHeaderLabels(
            ["Center Atom", "Atom of Interest", "Cutoff (A)"]
        )
        self.coordination_number_table.setAlternatingRowColors(True)
        self.coordination_number_table.verticalHeader().setVisible(False)
        self.coordination_number_table.setMinimumHeight(130)
        self.coordination_number_table.horizontalHeader().setStretchLastSection(
            True
        )
        layout.addWidget(self.coordination_number_table)
        self._add_coordination_number_row()
        self.coordination_number_table.itemChanged.connect(
            lambda _item: self._schedule_selection_summary_update(
                reason=(
                    "Coordination-number settings changed; refreshing "
                    "preview..."
                )
            )
        )
        return group

    def _make_dir_row(
        self,
        line_edit: QLineEdit,
        title: str,
    ) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(
            lambda _checked=False: self._choose_dir(line_edit, title)
        )

        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        return row_widget

    @staticmethod
    def _wrap_scroll_area(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _choose_dir(self, line_edit: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line_edit.setText(path)
            if line_edit is self.clusters_dir_edit:
                self._register_project_clusters_dir()

    def _choose_existing_results_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select existing bondanalysis output directory",
        )
        if not path:
            return
        try:
            self.load_existing_results_dir(path)
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis", str(exc))

    def set_clusters_dir(
        self,
        clusters_dir: str | Path,
        *,
        progress_callback=None,
    ) -> None:
        if progress_callback is None:
            self.clusters_dir_edit.setText(str(clusters_dir))
        else:
            self.clusters_dir_edit.blockSignals(True)
            self.clusters_dir_edit.setText(str(clusters_dir))
            self.clusters_dir_edit.blockSignals(False)
            self._refresh_cluster_types(progress_callback=progress_callback)
        self._register_project_clusters_dir()

    def load_existing_results_dir(self, output_dir: str | Path) -> None:
        result_index = load_result_index(output_dir)
        self._invalidate_stored_result_indices_cache()
        self._results_index = result_index
        self.output_dir_edit.blockSignals(True)
        self.output_dir_edit.setText(str(result_index.output_dir))
        self.output_dir_edit.blockSignals(False)
        self.clusters_dir_edit.blockSignals(True)
        self.clusters_dir_edit.setText(str(result_index.clusters_dir))
        self.clusters_dir_edit.blockSignals(False)
        self._register_project_clusters_dir()
        self._set_bond_pair_rows(result_index.bond_pairs)
        self._set_angle_triplet_rows(result_index.angle_triplets)
        self._set_dihedral_quartet_rows(result_index.dihedral_quartets)
        self._set_coordination_number_rows(result_index.coordination_numbers)
        self._restore_cluster_type_list(result_index)
        self._refresh_results_tree()
        self._append_log(
            "Loaded existing bondanalysis folder: "
            f"{result_index.output_dir}"
        )
        self._append_log(
            f"Results index file: {result_index.results_index_path}"
        )
        self.statusBar().showMessage(
            f"Loaded existing bondanalysis results: {result_index.output_dir}"
        )
        self._update_selection_summary()

    def _restore_cluster_type_list(
        self,
        result_index: BondAnalysisResultIndex,
    ) -> None:
        cluster_type_names = list(result_index.cluster_type_names)
        selected_names = set(result_index.selected_cluster_types)
        use_subset_filter = bool(selected_names) and (
            len(selected_names) != len(cluster_type_names)
        )

        self.cluster_type_list.blockSignals(True)
        self.cluster_type_list.clear()
        for cluster_type in cluster_type_names:
            item = QListWidgetItem(cluster_type)
            item.setFlags(
                (item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsSelectable
            )
            check_state = (
                Qt.CheckState.Checked
                if not use_subset_filter or cluster_type in selected_names
                else Qt.CheckState.Unchecked
            )
            item.setCheckState(check_state)
            self.cluster_type_list.addItem(item)
        self.cluster_type_list.blockSignals(False)
        self.use_checked_cluster_types_box.blockSignals(True)
        self.use_checked_cluster_types_box.setChecked(not use_subset_filter)
        self.use_checked_cluster_types_box.blockSignals(False)
        self._update_cluster_type_interactivity()
        self._set_cluster_type_status(
            f"{len(cluster_type_names)} cluster type(s) loaded from results."
        )

    def _set_cluster_type_status(self, text: str) -> None:
        if hasattr(self, "cluster_type_status_label"):
            self.cluster_type_status_label.setText(text)

    def _on_clusters_dir_changed(self, _text: str) -> None:
        self._invalidate_stored_result_indices_cache()
        self._refresh_cluster_types()

    def _refresh_cluster_types(
        self,
        *,
        progress_callback=None,
    ) -> None:
        clusters_dir = self._clusters_dir_path()
        previous_states = self._cluster_type_check_states()
        self.cluster_type_list.blockSignals(True)
        self.cluster_type_list.clear()
        if clusters_dir is None:
            self.cluster_type_list.blockSignals(False)
            self._set_cluster_type_status("No clusters directory selected.")
            self._update_cluster_type_interactivity()
            self._update_selection_summary()
            return

        try:
            workflow = BondAnalysisWorkflow(clusters_dir)
            summary = workflow.inspect(progress_callback=progress_callback)
        except Exception as exc:
            self.cluster_type_list.blockSignals(False)
            self._set_cluster_type_status("Cluster scan unavailable.")
            self._update_cluster_type_interactivity()
            self._append_log(f"Unable to inspect clusters directory: {exc}")
            self._update_selection_summary(error_text=str(exc))
            return

        for cluster_type in summary["cluster_types"]:
            item = QListWidgetItem(cluster_type)
            item.setFlags(
                (item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                previous_states.get(cluster_type, Qt.CheckState.Checked)
            )
            self.cluster_type_list.addItem(item)
        self.cluster_type_list.blockSignals(False)
        self._update_cluster_type_interactivity()
        self._set_cluster_type_status(
            f"{self.cluster_type_list.count()} cluster type(s) ready."
        )

        current_output_dir = self.output_dir_edit.text().strip()
        suggested_output_dir = str(
            self._suggest_output_dir_for_clusters(clusters_dir)
        )
        if not current_output_dir:
            self.output_dir_edit.blockSignals(True)
            self.output_dir_edit.setText(suggested_output_dir)
            self.output_dir_edit.blockSignals(False)

        self._update_selection_summary()

    def _clusters_dir_path(self) -> Path | None:
        text = self.clusters_dir_edit.text().strip()
        return Path(text) if text else None

    def _output_dir_path(self) -> Path | None:
        text = self.output_dir_edit.text().strip()
        return Path(text) if text else None

    def _selected_cluster_types(self) -> list[str] | None:
        if self.use_checked_cluster_types_box.isChecked():
            return None
        return self._checked_cluster_types()

    def _checked_cluster_types(self) -> list[str]:
        return [
            self.cluster_type_list.item(index).text()
            for index in range(self.cluster_type_list.count())
            if self.cluster_type_list.item(index).checkState()
            == Qt.CheckState.Checked
        ]

    def _cluster_type_check_states(self) -> dict[str, Qt.CheckState]:
        return {
            self.cluster_type_list.item(index)
            .text(): self.cluster_type_list.item(index)
            .checkState()
            for index in range(self.cluster_type_list.count())
        }

    @Slot(bool)
    def _on_analyze_all_cluster_types_toggled(self, _checked: bool) -> None:
        self._update_cluster_type_interactivity()
        self._update_selection_summary()

    def _update_cluster_type_interactivity(self) -> None:
        if not hasattr(self, "cluster_type_list"):
            return
        analyze_all = self.use_checked_cluster_types_box.isChecked()
        self.cluster_type_list.setEnabled(not analyze_all)
        if not analyze_all:
            return
        self.cluster_type_list.blockSignals(True)
        try:
            for index in range(self.cluster_type_list.count()):
                self.cluster_type_list.item(index).setCheckState(
                    Qt.CheckState.Checked
                )
        finally:
            self.cluster_type_list.blockSignals(False)

    def _selected_preset_name(self) -> str | None:
        return self.preset_combo.currentData()

    def _reload_presets(self, *, selected_name: str | None = None) -> None:
        previous_name = selected_name or self._selected_preset_name()
        self._presets = load_presets()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select preset...", None)
        selected_index = 0
        for index, name in enumerate(
            ordered_preset_names(self._presets),
            start=1,
        ):
            preset = self._presets[name]
            label = name
            if preset.builtin:
                label = f"{name} (Built-in)"
            self.preset_combo.addItem(label, name)
            if name == previous_name:
                selected_index = index
        self.preset_combo.setCurrentIndex(selected_index)
        self.preset_combo.blockSignals(False)
        self._update_selection_summary()

    def load_preset(self, preset_name: str) -> None:
        preset = self._presets.get(preset_name)
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name}")
        self._selection_summary_timer.stop()
        self._apply_preset(preset, update_summary=False)
        self._select_preset_name(preset_name, block_signals=True)
        self._update_selection_summary(
            show_progress=True,
            progress_message=f"Loading preset: {preset_name}",
        )
        self._append_log(f"Loaded preset: {preset_name}")

    def save_current_preset(self, preset_name: str) -> None:
        name = preset_name.strip()
        if not name:
            raise ValueError("Preset names cannot be empty.")
        preset = BondAnalysisPreset(
            name=name,
            bond_pairs=tuple(self._read_bond_pairs()),
            angle_triplets=tuple(self._read_angle_triplets()),
            dihedral_quartets=tuple(self._read_dihedral_quartets()),
            coordination_numbers=tuple(self._read_coordination_numbers()),
        )
        save_custom_preset(preset)
        self._reload_presets(selected_name=name)
        self._append_log(f"Saved preset: {name}")

    def _select_preset_name(
        self,
        preset_name: str,
        *,
        block_signals: bool = False,
    ) -> None:
        previous_blocked = None
        if block_signals:
            previous_blocked = self.preset_combo.blockSignals(True)
        try:
            for index in range(self.preset_combo.count()):
                if self.preset_combo.itemData(index) == preset_name:
                    self.preset_combo.setCurrentIndex(index)
                    return
        finally:
            if previous_blocked is not None:
                self.preset_combo.blockSignals(previous_blocked)

    def _load_selected_preset(self) -> None:
        preset_name = self._selected_preset_name()
        if preset_name is None:
            QMessageBox.information(
                self,
                "Bond Analysis Presets",
                "Select a preset to load.",
            )
            return
        try:
            self.load_preset(preset_name)
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis Presets", str(exc))

    def _save_current_as_preset(self) -> None:
        try:
            self._read_bond_pairs()
            self._read_angle_triplets()
            self._read_dihedral_quartets()
            self._read_coordination_numbers()
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis Presets", str(exc))
            return

        suggested_name = self._selected_preset_name() or ""
        name, accepted = QInputDialog.getText(
            self,
            "Save Bondanalysis Preset",
            "Preset name:",
            text=suggested_name,
        )
        if not accepted:
            return
        name = name.strip()
        if not name:
            return

        if name in self._presets:
            response = QMessageBox.question(
                self,
                "Overwrite Preset?",
                f"A preset named '{name}' already exists. Overwrite it?",
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        try:
            self.save_current_preset(name)
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis Presets", str(exc))

    def _apply_preset(
        self,
        preset: BondAnalysisPreset,
        *,
        update_summary: bool = True,
    ) -> None:
        self._set_bond_pair_rows(preset.bond_pairs)
        self._set_angle_triplet_rows(preset.angle_triplets)
        self._set_dihedral_quartet_rows(preset.dihedral_quartets)
        self._set_coordination_number_rows(preset.coordination_numbers)
        if update_summary:
            self._update_selection_summary()

    def _set_bond_pair_rows(
        self,
        definitions: tuple[BondPairDefinition, ...],
    ) -> None:
        previous_blocked = self.bond_pair_table.blockSignals(True)
        previous_updates_enabled = self.bond_pair_table.updatesEnabled()
        self.bond_pair_table.setUpdatesEnabled(False)
        try:
            self.bond_pair_table.setRowCount(0)
            if not definitions:
                self._add_empty_bond_pair_row(blocked=True)
            else:
                self.bond_pair_table.setRowCount(len(definitions))
                for row, definition in enumerate(definitions):
                    self.bond_pair_table.setItem(
                        row,
                        0,
                        QTableWidgetItem(definition.atom1),
                    )
                    self.bond_pair_table.setItem(
                        row,
                        1,
                        QTableWidgetItem(definition.atom2),
                    )
                    self.bond_pair_table.setItem(
                        row,
                        2,
                        QTableWidgetItem(f"{definition.cutoff_angstrom:g}"),
                    )
        finally:
            self.bond_pair_table.setUpdatesEnabled(previous_updates_enabled)
            self.bond_pair_table.blockSignals(previous_blocked)

    def _set_angle_triplet_rows(
        self,
        definitions: tuple[AngleTripletDefinition, ...],
    ) -> None:
        previous_blocked = self.angle_triplet_table.blockSignals(True)
        previous_updates_enabled = self.angle_triplet_table.updatesEnabled()
        self.angle_triplet_table.setUpdatesEnabled(False)
        try:
            self.angle_triplet_table.setRowCount(0)
            if not definitions:
                self._add_empty_angle_triplet_row(blocked=True)
            else:
                self.angle_triplet_table.setRowCount(len(definitions))
                for row, definition in enumerate(definitions):
                    self.angle_triplet_table.setItem(
                        row,
                        0,
                        QTableWidgetItem(definition.vertex),
                    )
                    self.angle_triplet_table.setItem(
                        row,
                        1,
                        QTableWidgetItem(definition.arm1),
                    )
                    self.angle_triplet_table.setItem(
                        row,
                        2,
                        QTableWidgetItem(definition.arm2),
                    )
                    self.angle_triplet_table.setItem(
                        row,
                        3,
                        QTableWidgetItem(f"{definition.cutoff1_angstrom:g}"),
                    )
                    self.angle_triplet_table.setItem(
                        row,
                        4,
                        QTableWidgetItem(f"{definition.cutoff2_angstrom:g}"),
                    )
        finally:
            self.angle_triplet_table.setUpdatesEnabled(
                previous_updates_enabled
            )
            self.angle_triplet_table.blockSignals(previous_blocked)

    def _set_dihedral_quartet_rows(
        self,
        definitions: tuple[DihedralQuartetDefinition, ...],
    ) -> None:
        editable_definitions = tuple(
            definition
            for definition in definitions
            if not definition.branch_label
        )
        previous_blocked = self.dihedral_quartet_table.blockSignals(True)
        previous_updates_enabled = self.dihedral_quartet_table.updatesEnabled()
        self.dihedral_quartet_table.setUpdatesEnabled(False)
        try:
            self.dihedral_quartet_table.setRowCount(0)
            if not editable_definitions:
                self._add_empty_dihedral_quartet_row(blocked=True)
            else:
                self.dihedral_quartet_table.setRowCount(
                    len(editable_definitions)
                )
                for row, definition in enumerate(editable_definitions):
                    values = (
                        definition.atom1,
                        definition.atom2,
                        definition.atom3,
                        definition.atom4,
                        f"{definition.cutoff12_angstrom:g}",
                        f"{definition.cutoff23_angstrom:g}",
                        f"{definition.cutoff34_angstrom:g}",
                    )
                    for column, value in enumerate(values):
                        self.dihedral_quartet_table.setItem(
                            row,
                            column,
                            QTableWidgetItem(value),
                        )
        finally:
            self.dihedral_quartet_table.setUpdatesEnabled(
                previous_updates_enabled
            )
            self.dihedral_quartet_table.blockSignals(previous_blocked)

    def _set_coordination_number_rows(
        self,
        definitions: tuple[CoordinationNumberDefinition, ...],
    ) -> None:
        previous_blocked = self.coordination_number_table.blockSignals(True)
        previous_updates_enabled = (
            self.coordination_number_table.updatesEnabled()
        )
        self.coordination_number_table.setUpdatesEnabled(False)
        try:
            self.coordination_number_table.setRowCount(0)
            if not definitions:
                self._add_empty_coordination_number_row(blocked=True)
            else:
                self.coordination_number_table.setRowCount(len(definitions))
                for row, definition in enumerate(definitions):
                    self.coordination_number_table.setItem(
                        row,
                        0,
                        QTableWidgetItem(definition.center_atom),
                    )
                    self.coordination_number_table.setItem(
                        row,
                        1,
                        QTableWidgetItem(definition.neighbor_atom),
                    )
                    self.coordination_number_table.setItem(
                        row,
                        2,
                        QTableWidgetItem(f"{definition.cutoff_angstrom:g}"),
                    )
        finally:
            self.coordination_number_table.setUpdatesEnabled(
                previous_updates_enabled
            )
            self.coordination_number_table.blockSignals(previous_blocked)

    def _add_empty_angle_triplet_row(self, *, blocked: bool = False) -> None:
        previous_blocked = self.angle_triplet_table.blockSignals(blocked)
        row = self.angle_triplet_table.rowCount()
        self.angle_triplet_table.insertRow(row)
        for column in range(self.angle_triplet_table.columnCount()):
            self.angle_triplet_table.setItem(
                row,
                column,
                QTableWidgetItem(""),
            )
        self.angle_triplet_table.blockSignals(previous_blocked)

    def _add_empty_dihedral_quartet_row(
        self,
        *,
        blocked: bool = False,
    ) -> None:
        previous_blocked = self.dihedral_quartet_table.blockSignals(blocked)
        row = self.dihedral_quartet_table.rowCount()
        self.dihedral_quartet_table.insertRow(row)
        for column in range(self.dihedral_quartet_table.columnCount()):
            self.dihedral_quartet_table.setItem(
                row,
                column,
                QTableWidgetItem(""),
            )
        self.dihedral_quartet_table.blockSignals(previous_blocked)

    def _add_empty_bond_pair_row(self, *, blocked: bool = False) -> None:
        previous_blocked = self.bond_pair_table.blockSignals(blocked)
        row = self.bond_pair_table.rowCount()
        self.bond_pair_table.insertRow(row)
        for column in range(self.bond_pair_table.columnCount()):
            self.bond_pair_table.setItem(row, column, QTableWidgetItem(""))
        self.bond_pair_table.blockSignals(previous_blocked)

    def _add_empty_coordination_number_row(
        self,
        *,
        blocked: bool = False,
    ) -> None:
        previous_blocked = self.coordination_number_table.blockSignals(blocked)
        row = self.coordination_number_table.rowCount()
        self.coordination_number_table.insertRow(row)
        for column in range(self.coordination_number_table.columnCount()):
            self.coordination_number_table.setItem(
                row,
                column,
                QTableWidgetItem(""),
            )
        self.coordination_number_table.blockSignals(previous_blocked)

    def _add_bond_pair_row(self) -> None:
        self._add_empty_bond_pair_row(blocked=True)

    def _remove_selected_bond_pair_rows(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.bond_pair_table.selectedIndexes()},
            reverse=True,
        )
        for row in selected_rows:
            self.bond_pair_table.removeRow(row)
        self._schedule_selection_summary_update(
            reason="Bond pair rows removed; refreshing preview..."
        )

    def _add_angle_triplet_row(self) -> None:
        self._add_empty_angle_triplet_row(blocked=True)

    def _remove_selected_angle_triplet_rows(self) -> None:
        selected_rows = sorted(
            {
                index.row()
                for index in self.angle_triplet_table.selectedIndexes()
            },
            reverse=True,
        )
        for row in selected_rows:
            self.angle_triplet_table.removeRow(row)
        self._schedule_selection_summary_update(
            reason="Angle triplet rows removed; refreshing preview..."
        )

    def _add_dihedral_quartet_row(self) -> None:
        self._add_empty_dihedral_quartet_row(blocked=True)

    def _remove_selected_dihedral_quartet_rows(self) -> None:
        selected_rows = sorted(
            {
                index.row()
                for index in self.dihedral_quartet_table.selectedIndexes()
            },
            reverse=True,
        )
        for row in selected_rows:
            self.dihedral_quartet_table.removeRow(row)
        self._schedule_selection_summary_update(
            reason="Dihedral quartet rows removed; refreshing preview..."
        )

    def _add_coordination_number_row(self) -> None:
        self._add_empty_coordination_number_row(blocked=True)

    def _remove_selected_coordination_number_rows(self) -> None:
        selected_rows = sorted(
            {
                index.row()
                for index in self.coordination_number_table.selectedIndexes()
            },
            reverse=True,
        )
        for row in selected_rows:
            self.coordination_number_table.removeRow(row)
        self._schedule_selection_summary_update(
            reason="Coordination-number rows removed; refreshing preview..."
        )

    def _read_bond_pairs(self) -> list[BondPairDefinition]:
        definitions: list[BondPairDefinition] = []
        for row in range(self.bond_pair_table.rowCount()):
            atom1 = self._table_text(self.bond_pair_table, row, 0)
            atom2 = self._table_text(self.bond_pair_table, row, 1)
            cutoff_text = self._table_text(self.bond_pair_table, row, 2)
            if not atom1 and not atom2 and not cutoff_text:
                continue
            if not atom1 or not atom2 or not cutoff_text:
                raise ValueError(
                    "Every populated bond-pair row needs atom 1, atom 2, "
                    "and a cutoff."
                )
            definitions.append(
                BondPairDefinition(atom1, atom2, float(cutoff_text))
            )
        return definitions

    def _read_angle_triplets(self) -> list[AngleTripletDefinition]:
        definitions: list[AngleTripletDefinition] = []
        for row in range(self.angle_triplet_table.rowCount()):
            vertex = self._table_text(self.angle_triplet_table, row, 0)
            arm1 = self._table_text(self.angle_triplet_table, row, 1)
            arm2 = self._table_text(self.angle_triplet_table, row, 2)
            cutoff1_text = self._table_text(self.angle_triplet_table, row, 3)
            cutoff2_text = self._table_text(self.angle_triplet_table, row, 4)
            if (
                not vertex
                and not arm1
                and not arm2
                and not cutoff1_text
                and not cutoff2_text
            ):
                continue
            if not all((vertex, arm1, arm2, cutoff1_text, cutoff2_text)):
                raise ValueError(
                    "Every populated angle-triplet row needs the vertex, "
                    "both arms, and both cutoffs."
                )
            definitions.append(
                AngleTripletDefinition(
                    vertex,
                    arm1,
                    arm2,
                    float(cutoff1_text),
                    float(cutoff2_text),
                )
            )
        return definitions

    def _read_dihedral_quartets(self) -> list[DihedralQuartetDefinition]:
        definitions: list[DihedralQuartetDefinition] = []
        for row in range(self.dihedral_quartet_table.rowCount()):
            atom1 = self._table_text(self.dihedral_quartet_table, row, 0)
            atom2 = self._table_text(self.dihedral_quartet_table, row, 1)
            atom3 = self._table_text(self.dihedral_quartet_table, row, 2)
            atom4 = self._table_text(self.dihedral_quartet_table, row, 3)
            cutoff12_text = self._table_text(
                self.dihedral_quartet_table,
                row,
                4,
            )
            cutoff23_text = self._table_text(
                self.dihedral_quartet_table,
                row,
                5,
            )
            cutoff34_text = self._table_text(
                self.dihedral_quartet_table,
                row,
                6,
            )
            if not any(
                (
                    atom1,
                    atom2,
                    atom3,
                    atom4,
                    cutoff12_text,
                    cutoff23_text,
                    cutoff34_text,
                )
            ):
                continue
            if not all(
                (
                    atom1,
                    atom2,
                    atom3,
                    atom4,
                    cutoff12_text,
                    cutoff23_text,
                    cutoff34_text,
                )
            ):
                raise ValueError(
                    "Every populated dihedral-quartet row needs four atoms "
                    "and three adjacent-pair cutoffs."
                )
            definitions.append(
                DihedralQuartetDefinition(
                    atom1,
                    atom2,
                    atom3,
                    atom4,
                    float(cutoff12_text),
                    float(cutoff23_text),
                    float(cutoff34_text),
                )
            )
        return definitions

    def _read_coordination_numbers(
        self,
    ) -> list[CoordinationNumberDefinition]:
        definitions: list[CoordinationNumberDefinition] = []
        for row in range(self.coordination_number_table.rowCount()):
            center_atom = self._table_text(
                self.coordination_number_table,
                row,
                0,
            )
            neighbor_atom = self._table_text(
                self.coordination_number_table,
                row,
                1,
            )
            cutoff_text = self._table_text(
                self.coordination_number_table,
                row,
                2,
            )
            if not center_atom and not neighbor_atom and not cutoff_text:
                continue
            if not center_atom or not neighbor_atom or not cutoff_text:
                raise ValueError(
                    "Every populated coordination-number row needs a center "
                    "atom, atom of interest, and cutoff."
                )
            definitions.append(
                CoordinationNumberDefinition(
                    center_atom,
                    neighbor_atom,
                    float(cutoff_text),
                )
            )
        return definitions

    @staticmethod
    def _table_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _update_selection_summary(
        self,
        *,
        error_text: str | None = None,
        show_progress: bool = False,
        progress_message: str = "Refreshing selection preview...",
    ) -> None:
        if show_progress:
            self._begin_selection_summary_progress(progress_message)
        try:
            progress_callback = (
                self._update_selection_summary_progress
                if show_progress
                else None
            )
            if progress_callback is not None:
                progress_callback(
                    1,
                    SELECTION_PREVIEW_TOTAL_STEPS,
                    "Reading current bond-analysis selections...",
                )
            clusters_dir = self._clusters_dir_path()
            output_dir = self._output_dir_path()
            selected_cluster_types = self._selected_cluster_types()

            lines: list[str] = []
            if error_text is not None:
                lines.append(f"Inspection error: {error_text}")
            if clusters_dir is None:
                lines.append(
                    "Select a stoichiometry-level clusters directory to "
                    "preview the bond-analysis run."
                )
                self.selection_box.setPlainText("\n".join(lines))
                return

            lines.append(f"Clusters directory: {clusters_dir}")
            if output_dir is not None:
                lines.append(f"Output directory: {output_dir}")

            cluster_types = [
                self.cluster_type_list.item(index).text()
                for index in range(self.cluster_type_list.count())
            ]
            lines.append(f"Cluster types detected: {len(cluster_types)}")
            checked_cluster_types = self._checked_cluster_types()
            lines.append(
                f"Checked cluster types: {len(checked_cluster_types)}"
            )
            if self.use_checked_cluster_types_box.isChecked():
                if cluster_types:
                    lines.append("Analyzing cluster types: all detected types")
                else:
                    lines.append("Analyzing cluster types: none detected yet")
            elif selected_cluster_types:
                lines.append(
                    "Analyzing checked cluster types: "
                    + ", ".join(selected_cluster_types)
                )
            else:
                lines.append(
                    "Analyzing checked cluster types: none checked yet"
                )

            preset_name = self._selected_preset_name()
            if preset_name is not None:
                lines.append(f"Selected preset: {preset_name}")

            bond_pairs: list[BondPairDefinition] | None = None
            try:
                bond_pairs = self._read_bond_pairs()
                lines.append(f"Bond pairs configured: {len(bond_pairs)}")
            except Exception as exc:
                lines.append(f"Bond pairs configured: invalid ({exc})")

            angle_triplets: list[AngleTripletDefinition] | None = None
            try:
                angle_triplets = self._read_angle_triplets()
                lines.append(
                    f"Angle triplets configured: {len(angle_triplets)}"
                )
            except Exception as exc:
                lines.append(f"Angle triplets configured: invalid ({exc})")

            dihedral_quartets: list[DihedralQuartetDefinition] | None = None
            try:
                dihedral_quartets = self._read_dihedral_quartets()
                lines.append(
                    "Dihedral quartets configured: "
                    f"{len(dihedral_quartets)}"
                )
            except Exception as exc:
                lines.append(f"Dihedral quartets configured: invalid ({exc})")

            coordination_numbers: list[CoordinationNumberDefinition] | None = (
                None
            )
            try:
                coordination_numbers = self._read_coordination_numbers()
                lines.append(
                    "Coordination rules configured: "
                    f"{len(coordination_numbers)}"
                )
            except Exception as exc:
                lines.append(f"Coordination rules configured: invalid ({exc})")

            lines.append(
                "Displacement analysis: deprecated and not part of this window"
            )
            lines.extend(
                self._stored_results_preview_lines(
                    clusters_dir=clusters_dir,
                    selected_cluster_types=selected_cluster_types,
                    bond_pairs=bond_pairs,
                    angle_triplets=angle_triplets,
                    dihedral_quartets=dihedral_quartets,
                    coordination_numbers=coordination_numbers,
                    progress_callback=progress_callback,
                )
            )
            self.selection_box.setPlainText("\n".join(lines))
        finally:
            if show_progress:
                self._close_selection_summary_progress_dialog()

    def _refresh_results_tree(self) -> None:
        output_dir = self._output_dir_path()
        if output_dir is None:
            self._clear_results_tree(
                "Choose a bondanalysis output directory first."
            )
            return

        try:
            self._results_index = load_result_index(output_dir)
        except Exception as exc:
            self._results_index = None
            self._clear_results_tree(str(exc))
            return

        previous_tree_blocked = self.results_tree.blockSignals(True)
        previous_tree_updates = self.results_tree.updatesEnabled()
        self.results_tree.setUpdatesEnabled(False)
        try:
            self.results_tree.clear()
            self._populate_results_category(
                "Bond Pairs",
                self._results_index.bond_groups,
            )
            self._populate_results_category(
                "Bond Angles",
                self._results_index.angle_groups,
            )
            self._populate_results_category(
                "Dihedral Angles",
                self._results_index.dihedral_groups,
            )
            self._populate_results_category(
                "Coordination Numbers",
                self._results_index.coordination_groups,
            )
            self._expand_results_categories_only()
        finally:
            self.results_tree.setUpdatesEnabled(previous_tree_updates)
            self.results_tree.blockSignals(previous_tree_blocked)
        self._populate_all_cluster_stats_table(self._results_index)
        self.results_status_label.setText(
            "Browse computed bond, angle, dihedral, and coordination "
            "distributions from the current output directory: "
            f"{self._results_index.output_dir}"
        )

    def _populate_all_cluster_stats_table(
        self,
        result_index: BondAnalysisResultIndex,
    ) -> None:
        rows = self._all_cluster_stats_rows(result_index)
        previous_blocked = self.results_stats_table.blockSignals(True)
        previous_updates_enabled = self.results_stats_table.updatesEnabled()
        self.results_stats_table.setUpdatesEnabled(False)
        try:
            self.results_stats_table.setRowCount(len(rows))
            for row_index, row_values in enumerate(rows):
                for column_index, value in enumerate(row_values):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.results_stats_table.setItem(
                        row_index,
                        column_index,
                        item,
                    )
        finally:
            self.results_stats_table.setUpdatesEnabled(
                previous_updates_enabled
            )
            self.results_stats_table.blockSignals(previous_blocked)

    def _all_cluster_stats_rows(
        self,
        result_index: BondAnalysisResultIndex,
    ) -> list[tuple[str, str, str, str, str, str, str, str, str]]:
        rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        rows.extend(
            self._stats_rows_for_groups(
                result_index,
                category="bond",
                type_label="Bond Pair",
                definitions=result_index.bond_pairs,
                groups=result_index.bond_groups,
            )
        )
        rows.extend(
            self._stats_rows_for_groups(
                result_index,
                category="angle",
                type_label="Angle",
                definitions=result_index.angle_triplets,
                groups=result_index.angle_groups,
            )
        )
        rows.extend(
            self._stats_rows_for_groups(
                result_index,
                category="dihedral",
                type_label="Dihedral",
                definitions=result_index.dihedral_quartets,
                groups=result_index.dihedral_groups,
            )
        )
        rows.extend(
            self._stats_rows_for_groups(
                result_index,
                category="coordination",
                type_label="Coordination",
                definitions=result_index.coordination_numbers,
                groups=result_index.coordination_groups,
            )
        )
        return rows

    def _stats_rows_for_groups(
        self,
        result_index: BondAnalysisResultIndex,
        *,
        category: str,
        type_label: str,
        definitions: Sequence[object],
        groups: Sequence[object],
    ) -> list[tuple[str, str, str, str, str, str, str, str, str]]:
        rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        all_clusters_dir = result_index.output_dir / "all_clusters"
        for definition, group in zip(definitions, groups):
            histogram_path = (
                all_clusters_dir / f"{definition.filename_stem}_histogram.csv"
            )
            metadata = self._read_histogram_metadata(histogram_path)
            rows.append(
                self._stats_row_from_metadata(
                    category=category,
                    type_label=type_label,
                    display_label=group.display_label,
                    point_count=group.all_leaf.point_count,
                    metadata=metadata,
                )
            )
        return rows

    def _stats_row_from_metadata(
        self,
        *,
        category: str,
        type_label: str,
        display_label: str,
        point_count: int,
        metadata: Mapping[str, str],
    ) -> tuple[str, str, str, str, str, str, str, str, str]:
        if category == "bond":
            average_keys = ("gds_center_angstrom", "mean")
            sigma_keys = ("gds_sigma_angstrom", "sigma")
            unit = "A"
        elif category == "angle":
            average_keys = ("gds_center_degrees", "mean")
            sigma_keys = ("gds_sigma_degrees", "sigma")
            unit = "deg"
        elif category == "dihedral":
            average_keys = (
                "gds_center_degrees",
                "circular_mean_degrees",
                "mean",
            )
            sigma_keys = (
                "gds_sigma_degrees",
                "circular_sigma_degrees",
                "sigma",
            )
            unit = "deg"
        else:
            average_keys = ("mean",)
            sigma_keys = ("sigma",)
            unit = "count"
        average = self._metadata_float(metadata, *average_keys)
        median = self._metadata_float(metadata, "median")
        sigma = self._metadata_float(metadata, *sigma_keys)
        metadata_count = self._metadata_int(metadata, "point_count")
        return (
            type_label,
            display_label,
            str(metadata_count if metadata_count is not None else point_count),
            self._format_stat_value(average),
            self._format_stat_value(median),
            self._format_stat_value(sigma),
            unit,
            metadata.get("gds_center_variable", "-") or "-",
            metadata.get("gds_sigma_variable", "-") or "-",
        )

    @staticmethod
    def _read_histogram_metadata(histogram_path: Path) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if not histogram_path.exists():
            return metadata
        try:
            with histogram_path.open(newline="") as stream:
                for row in csv.reader(stream):
                    if not row:
                        continue
                    if not row[0].startswith("#"):
                        break
                    if len(row) < 2:
                        continue
                    key = row[0].removeprefix("# ").strip()
                    if key:
                        metadata[key] = row[1]
        except OSError:
            return {}
        return metadata

    @staticmethod
    def _metadata_float(
        metadata: Mapping[str, str],
        *keys: str,
    ) -> float | None:
        for key in keys:
            value = metadata.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
        return None

    @staticmethod
    def _metadata_int(
        metadata: Mapping[str, str],
        key: str,
    ) -> int | None:
        value = metadata.get(key)
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_stat_value(value: float | None) -> str:
        if value is None:
            return "-"
        text = f"{value:.6g}"
        return "0" if text == "-0" else text

    def _populate_results_category(
        self,
        title: str,
        groups: tuple[object, ...],
    ) -> None:
        if not groups:
            return
        category_item = QTreeWidgetItem([title, "", ""])
        category_item.setChildIndicatorPolicy(
            QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
        )
        category_item.setFlags(
            category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
        )
        category_item.setToolTip(
            0,
            "Use the arrow to expand or collapse this distribution section.",
        )
        self.results_tree.addTopLevelItem(category_item)
        for group in groups:
            group_item = QTreeWidgetItem(
                [
                    group.display_label,
                    "all clusters",
                    str(group.all_leaf.point_count),
                ]
            )
            group_item.setChildIndicatorPolicy(
                QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
            )
            group_item.setData(0, Qt.ItemDataRole.UserRole, group.all_leaf)
            group_item.setToolTip(
                0,
                "Select this distribution-name row to open the all-cluster "
                "plot. Cmd-click or Ctrl-click distribution-name rows to "
                "select multiple all-cluster plots. Use the arrow to show "
                "or hide cluster-level entries.",
            )
            category_item.addChild(group_item)
            group_item.addChild(self._make_results_leaf_item(group.all_leaf))
            for leaf in group.cluster_leaves:
                group_item.addChild(self._make_results_leaf_item(leaf))
            group_item.setExpanded(False)

    def _expand_results_categories_only(self) -> None:
        for category_index in range(self.results_tree.topLevelItemCount()):
            category_item = self.results_tree.topLevelItem(category_index)
            category_item.setExpanded(True)
            for group_index in range(category_item.childCount()):
                category_item.child(group_index).setExpanded(False)

    def _make_results_leaf_item(
        self,
        leaf: BondAnalysisResultLeaf,
    ) -> QTreeWidgetItem:
        scope_label = "all clusters" if leaf.is_all else "cluster type"
        item = QTreeWidgetItem(
            [leaf.scope_name, scope_label, str(leaf.point_count)]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, leaf)
        item.setToolTip(
            0,
            f"{leaf.display_label} • {leaf.scope_name} "
            f"({leaf.point_count} values)",
        )
        return item

    def _selected_result_leaves(self) -> list[BondAnalysisResultLeaf]:
        leaves: list[BondAnalysisResultLeaf] = []
        seen_keys: set[tuple[str, str, str, bool]] = set()
        for item in self.results_tree.selectedItems():
            payload = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(payload, BondAnalysisResultLeaf):
                key = (
                    payload.category,
                    payload.display_label,
                    payload.scope_name,
                    payload.is_all,
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                leaves.append(payload)
        return leaves

    def _on_results_tree_selection_changed(self) -> None:
        leaves = self._selected_result_leaves()
        if not leaves:
            self.results_status_label.setText(
                "Select one computed distribution and use 'Open Selected in "
                "Tab' to view it, or select multiple leaves of the same "
                "type and open them together as an overlay tab."
            )
            return

        if len(leaves) == 1:
            leaf = leaves[0]
            self.results_status_label.setText(
                f"Ready to open {leaf.display_label} for {leaf.scope_name} "
                "in a plot tab."
            )
            return

        if all(leaf.is_all for leaf in leaves):
            self.results_status_label.setText(
                "Ready to open "
                f"{len(leaves)} selected all-cluster distributions as plot "
                "tabs."
            )
            return

        try:
            self._validate_multi_leaf_selection(leaves)
        except ValueError as exc:
            self.results_status_label.setText(str(exc))
            return

        cluster_names = ", ".join(leaf.scope_name for leaf in leaves)
        self.results_status_label.setText(
            "Ready to overlay "
            f"{leaves[0].display_label} across: {cluster_names}"
        )

    def _validate_multi_leaf_selection(
        self,
        leaves: list[BondAnalysisResultLeaf],
    ) -> None:
        first_leaf = leaves[0]
        if any(leaf.is_all for leaf in leaves):
            raise ValueError(
                "Select either the 'all' entry or multiple individual "
                "cluster leaves, not both together."
            )
        if any(
            leaf.category != first_leaf.category
            or leaf.display_label != first_leaf.display_label
            for leaf in leaves[1:]
        ):
            raise ValueError(
                "To overlay distributions, select distributions of "
                "the same type across different cluster types."
            )

    def _open_selected_plot_window(self) -> None:
        leaves = self._selected_result_leaves()
        if not leaves:
            QMessageBox.information(
                self,
                "Computed Distributions",
                "Select one or more computed bond, angle, dihedral, or "
                "coordination distributions first.",
            )
            return
        if len(leaves) > 1 and all(leaf.is_all for leaf in leaves):
            self._open_selected_all_cluster_plot_windows(leaves)
            return
        self._open_plot_window_for_leaves(leaves)

    def _open_selected_all_cluster_plot_windows(
        self,
        leaves: list[BondAnalysisResultLeaf],
    ) -> None:
        if self._results_index is None:
            self._refresh_results_tree()
        if self._results_index is None:
            QMessageBox.warning(
                self,
                "Computed Distributions",
                "Run bondanalysis or refresh an existing output directory "
                "before opening plot tabs.",
            )
            return
        opened_count = 0
        for leaf in leaves:
            try:
                plot_request = build_plot_request(self._results_index, [leaf])
            except Exception as exc:
                QMessageBox.warning(self, "Computed Distributions", str(exc))
                return
            self._open_plot_window_for_request(plot_request)
            opened_count += 1
        self.results_status_label.setText(
            f"Opened {opened_count} selected all-cluster distribution "
            "plot(s)."
        )

    def _open_all_all_cluster_plot_windows(self) -> None:
        if self._results_index is None:
            self._refresh_results_tree()
        if self._results_index is None:
            QMessageBox.information(
                self,
                "Computed Distributions",
                "Run bondanalysis or load an existing output directory first.",
            )
            return

        leaves = self._all_cluster_result_leaves()
        if not leaves:
            QMessageBox.information(
                self,
                "Computed Distributions",
                "No non-empty all-cluster distributions were found.",
            )
            return

        opened_count = 0
        for leaf in leaves:
            try:
                plot_request = build_plot_request(self._results_index, [leaf])
            except Exception as exc:
                QMessageBox.warning(self, "Computed Distributions", str(exc))
                return
            self._open_plot_window_for_request(plot_request)
            opened_count += 1
        self.results_status_label.setText(
            f"Opened {opened_count} all-cluster distribution plot(s)."
        )

    def _all_cluster_result_leaves(self) -> list[BondAnalysisResultLeaf]:
        if self._results_index is None:
            return []
        groups = (
            *self._results_index.bond_groups,
            *self._results_index.angle_groups,
            *self._results_index.dihedral_groups,
            *self._results_index.coordination_groups,
        )
        return [
            group.all_leaf
            for group in groups
            if group.all_leaf.point_count > 0
        ]

    def _current_results_output_dir(self) -> Path | None:
        if self._results_index is not None:
            return self._results_index.output_dir
        output_dir = self._output_dir_path()
        if output_dir is not None and output_dir.exists():
            return output_dir
        return None

    def _show_output_folder(self) -> None:
        output_dir = self._current_results_output_dir()
        if output_dir is None:
            QMessageBox.information(
                self,
                "Computed Distributions",
                "Run bondanalysis or load an existing output directory first.",
            )
            return
        if not QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(output_dir.resolve()))
        ):
            QMessageBox.warning(
                self,
                "Computed Distributions",
                f"Unable to open output folder: {output_dir}",
            )

    def _open_plot_window_for_leaves(
        self,
        leaves: list[BondAnalysisResultLeaf],
    ) -> None:
        if self._results_index is None:
            QMessageBox.warning(
                self,
                "Computed Distributions",
                "Run bondanalysis or refresh an existing output directory "
                "before opening plot tabs.",
            )
            return
        try:
            if len(leaves) > 1:
                self._validate_multi_leaf_selection(leaves)
            plot_request = build_plot_request(self._results_index, leaves)
        except Exception as exc:
            QMessageBox.warning(self, "Computed Distributions", str(exc))
            return
        self._open_plot_window_for_request(plot_request)

    def _open_plot_window_for_request(
        self,
        plot_request: BondAnalysisPlotRequest,
    ) -> None:
        default_output_dir = (
            self._results_index.output_dir
            if self._results_index is not None
            else (self._output_dir_path() or Path.cwd())
        )
        if self._plot_windows:
            window = self._plot_windows[0]
            window.add_plot_request(plot_request)
        else:
            window = BondAnalysisPlotWindow(
                plot_request,
                default_output_dir=default_output_dir,
                parent=self,
            )
            track_saxshell_window(window, self._plot_windows)
            self.plot_window_opened.emit(window)
        window.show()
        window.raise_()
        window.activateWindow()

    def _remove_plot_window(self, window: BondAnalysisPlotWindow) -> None:
        self._plot_windows = [
            existing
            for existing in self._plot_windows
            if existing is not window
        ]

    def _clear_results_tree(self, message: str) -> None:
        self.results_tree.clear()
        if hasattr(self, "results_stats_table"):
            self.results_stats_table.setRowCount(0)
        self.results_status_label.setText(message)

    def _append_log(self, text: str) -> None:
        current = self.log_box.toPlainText().strip()
        if current:
            self.log_box.append(text)
        else:
            self.log_box.setPlainText(text)

    def _start_run(self) -> None:
        if self._run_thread is not None:
            return

        try:
            clusters_dir = self._clusters_dir_path()
            if clusters_dir is None or not clusters_dir.is_dir():
                raise ValueError(
                    "Choose a valid clusters directory before running."
                )
            output_dir = self._output_dir_path()
            selected_cluster_types = self._selected_cluster_types()
            if (
                not self.use_checked_cluster_types_box.isChecked()
                and not selected_cluster_types
            ):
                raise ValueError(
                    "Check at least one cluster type, or turn on "
                    "'Analyze all cluster types'."
                )

            workflow = BondAnalysisWorkflow(
                clusters_dir,
                bond_pairs=self._read_bond_pairs(),
                angle_triplets=self._read_angle_triplets(),
                dihedral_quartets=self._read_dihedral_quartets(),
                coordination_numbers=self._read_coordination_numbers(),
                output_dir=output_dir,
                selected_cluster_types=selected_cluster_types,
                structure_distribution_store_dir=(
                    None
                    if self._project_dir is None
                    else application_structure_distribution_store_dir(
                        project_dir=self._project_dir,
                        application="bondanalysis",
                    )
                ),
                generate_preview_plots=False,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis", str(exc))
            return

        self.log_box.clear()
        self._append_log("Bond-analysis run started.")
        self.run_button.setEnabled(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: preparing")
        self._active_run_status = "Preparing bond analysis..."
        self._run_is_saving_distribution_outputs = False
        self.statusBar().showMessage("Preparing bond analysis...")

        self._run_thread = QThread(self)
        self._run_worker = BondAnalysisWorker(workflow)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.log.connect(self._append_log)
        self._run_worker.progress.connect(self._update_progress)
        self._run_worker.status.connect(self._update_run_status)
        self._run_worker.finished.connect(self._finish_run)
        self._run_worker.failed.connect(self._fail_run)
        self._run_worker.canceled.connect(self._cancel_run_finished)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_worker.canceled.connect(self._run_thread.quit)
        self._run_worker.finished.connect(self._run_worker.deleteLater)
        self._run_worker.failed.connect(self._run_worker.deleteLater)
        self._run_worker.canceled.connect(self._run_worker.deleteLater)
        self._run_thread.finished.connect(self._cleanup_run_thread)
        self._run_thread.finished.connect(self._run_thread.deleteLater)
        self._run_thread.start()

    def _request_run_cancel(
        self, *, close_when_finished: bool = False
    ) -> None:
        if self._run_cancel_requested:
            if close_when_finished:
                self._close_after_run_cancel = True
            return
        self._run_cancel_requested = True
        self._close_after_run_cancel = close_when_finished
        self.run_button.setEnabled(False)
        self._active_run_status = "Canceling bond analysis..."
        self._run_is_saving_distribution_outputs = False
        self.progress_label.setText(
            "Progress: canceling at next safe checkpoint"
        )
        self.statusBar().showMessage(
            "Canceling bond analysis at the next safe checkpoint..."
        )
        self._append_log(
            "Cancel requested; stopping at the next safe checkpoint."
        )
        if self._run_worker is not None:
            self._run_worker.request_cancel()

    def _update_progress(
        self,
        processed: int,
        total: int,
        message: str = "",
    ) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat("%v / %m steps")
        if message:
            self._active_run_status = message
            self._run_is_saving_distribution_outputs = (
                self._is_distribution_save_progress_message(message)
            )
        if message:
            self.progress_label.setText(
                f"Progress: {message} ({processed}/{total})"
            )
            self.progress_label.setToolTip(message)
        else:
            self.progress_label.setText(
                f"Progress: {processed} processed, "
                f"{total - processed} remaining"
            )
            self.progress_label.setToolTip("")

    def _update_run_status(self, message: str) -> None:
        self._active_run_status = message
        self._run_is_saving_distribution_outputs = (
            self._is_distribution_save_progress_message(message)
        )
        self.statusBar().showMessage(message)

    @staticmethod
    def _is_distribution_save_progress_message(message: str) -> bool:
        normalized = str(message).strip().lower()
        return (
            normalized.startswith("saving cached structure measurements")
            or (
                normalized.startswith("writing ")
                and " distributions" in normalized
            )
            or normalized.startswith(
                (
                    "writing cluster comparison overlays",
                    "writing bond-analysis results index",
                )
            )
        )

    def _finish_run(self, result: BondAnalysisBatchResult) -> None:
        self._invalidate_stored_result_indices_cache()
        self.run_button.setEnabled(True)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_label.setText(
            f"Progress: complete ({result.total_structure_files} files)"
        )
        self.statusBar().showMessage(
            f"Bond analysis complete: {result.output_dir}"
        )
        self._active_run_status = "Bond analysis complete."
        self._run_is_saving_distribution_outputs = False
        self._run_cancel_requested = False
        self.output_dir_edit.setText(str(result.output_dir))
        self._append_log(f"Output directory: {result.output_dir}")
        self._append_log(f"Results index file: {result.results_index_path}")
        for cluster_result in result.cluster_results:
            bond_total = sum(cluster_result.bond_value_counts.values())
            angle_total = sum(cluster_result.angle_value_counts.values())
            dihedral_total = sum(cluster_result.dihedral_value_counts.values())
            coordination_total = sum(
                cluster_result.coordination_value_counts.values()
            )
            self._append_log(
                f"{cluster_result.cluster_type}: "
                f"{cluster_result.structure_count} file(s), "
                f"{bond_total} bond values, "
                f"{angle_total} angle values, "
                f"{dihedral_total} dihedral values, "
                f"{coordination_total} coordination values"
            )
        self._refresh_results_tree()
        self._update_selection_summary()

    def _fail_run(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.progress_label.setText("Progress: failed")
        self.statusBar().showMessage("Bond analysis failed")
        self._active_run_status = "Bond analysis failed."
        self._run_is_saving_distribution_outputs = False
        self._run_cancel_requested = False
        self._close_after_run_cancel = False
        self._append_log(f"Run failed: {message}")
        QMessageBox.critical(self, "Bond Analysis", message)

    def _cancel_run_finished(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.progress_label.setText("Progress: canceled")
        self.progress_label.setToolTip("")
        self.statusBar().showMessage("Bond analysis canceled")
        self._active_run_status = "Bond analysis canceled."
        self._run_is_saving_distribution_outputs = False
        self._append_log(message or "Bond analysis canceled.")

    def _cleanup_run_thread(self) -> None:
        self._run_worker = None
        self._run_thread = None
        self._run_cancel_requested = False
        if self._close_after_run_cancel:
            self._close_after_run_cancel = False
            QTimer.singleShot(0, self.close)


def launch_bondanalysis_ui(
    clusters_dir: str | Path | None = None,
) -> int:
    """Launch the Qt6 bond-analysis UI."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    track_saxshell_window(window, _OPEN_WINDOWS)
    window.show()
    if owns_app:
        return app.exec()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for launching the Qt6 bond-analysis UI."""
    parser = argparse.ArgumentParser(
        prog="bondanalysis-ui",
        description=(
            "Launch the SAXSShell bondanalysis UI for stoichiometry-level "
            "cluster folders."
        ),
    )
    parser.add_argument(
        "clusters_dir",
        nargs="?",
        help="Optional clusters directory to prefill in the UI.",
    )
    args = parser.parse_args(argv)
    return launch_bondanalysis_ui(args.clusters_dir)


__all__ = ["BondAnalysisMainWindow", "launch_bondanalysis_ui", "main"]
