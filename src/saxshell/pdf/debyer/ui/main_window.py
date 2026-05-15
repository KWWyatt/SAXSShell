from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.pdf.debyer.workflow import (
    DEBYER_DOCS_URL,
    DEBYER_GITHUB_URL,
    DEFAULT_COLOR_SCHEMES,
    SUPPORTED_DEBYER_MODES,
    SUPPORTED_PLOT_REPRESENTATIONS,
    TOTAL_SCATTERING_PAPER_URL,
    DebyerPDFCalculation,
    DebyerPDFCalculationSummary,
    DebyerPDFSettings,
    DebyerPDFWorkflow,
    DebyerPeakFinderSettings,
    DebyerPeakMarker,
    build_display_traces,
    check_debyer_runtime,
    classify_partial_pair,
    compute_experimental_fit_metrics,
    convert_distribution_values,
    default_parallel_debyer_jobs,
    estimate_partial_peak_markers,
    find_partial_peak_markers,
    fit_coordination_peak_from_r,
    infer_default_solute_elements,
    inspect_frames_dir,
    list_saved_debyer_calculations,
    load_debyer_calculation,
    rewrite_debyer_calculation_output,
    write_debyer_calculation_metadata,
)
from saxshell.saxs.project_manager import (
    ExperimentalDataSummary,
    load_experimental_data_file,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.saxs.ui.experimental_data_loader import (
    ExperimentalDataHeaderDialog,
)

_OPEN_WINDOWS: list["DebyerPDFMainWindow"] = []

_GROUP_TRACE_DEFAULT_COLORS = {
    "group:solute-solute": "#cc79a7",
    "group:solute-solvent": "#e69f00",
    "group:solvent-solvent": "#009e73",
}

_SPLITTER_HANDLE_STYLE = """
QSplitter::handle {
    background-color: #c8d1de;
    border: 1px solid #9aa8ba;
    border-radius: 2px;
}
QSplitter::handle:hover {
    background-color: #9fb2ca;
    border-color: #6f83a0;
}
QSplitter::handle:pressed {
    background-color: #8299b8;
    border-color: #536b8c;
}
"""


def _configure_resize_splitter(
    splitter: QSplitter,
    *,
    handle_width: int,
    tooltip: str,
) -> None:
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(handle_width)
    splitter.setOpaqueResize(True)
    splitter.setStyleSheet(_SPLITTER_HANDLE_STYLE)
    for index in range(1, splitter.count()):
        splitter.handle(index).setToolTip(tooltip)


class DebyerPDFWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    status = Signal(str)
    preview = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        settings: DebyerPDFSettings,
        *,
        preview_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.settings = settings
        self._cancel_requested = threading.Event()
        self._preview_enabled = threading.Event()
        self._preview_update_requested = threading.Event()
        if preview_enabled:
            self._preview_enabled.set()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def set_preview_enabled(self, enabled: bool) -> None:
        if enabled:
            self._preview_enabled.set()
            self._preview_update_requested.set()
        else:
            self._preview_enabled.clear()
            self._preview_update_requested.clear()

    @Slot()
    def run(self) -> None:
        workflow = DebyerPDFWorkflow(self.settings)
        try:
            result = workflow.run(
                progress_callback=self._emit_progress,
                log_callback=self.log.emit,
                status_callback=self.status.emit,
                preview_callback=self._emit_preview,
                preview_decision_callback=self._should_emit_preview,
                cancel_callback=self._cancel_requested.is_set,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    def _emit_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.progress.emit(processed, total, message)

    def _should_emit_preview(
        self,
        _processed: int,
        _total: int,
        checkpoint_due: bool,
    ) -> bool:
        if self._preview_update_requested.is_set():
            return True
        return bool(checkpoint_due and self._preview_enabled.is_set())

    def _emit_preview(self, calculation: DebyerPDFCalculation) -> None:
        self._preview_update_requested.clear()
        self.preview.emit(calculation)


class DebyerPeakEditorDialog(QDialog):
    def __init__(
        self,
        *,
        pair_label: str,
        markers: tuple[DebyerPeakMarker, ...],
        r_values: np.ndarray,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pair_label = pair_label
        self._r_values = np.asarray(r_values, dtype=float)
        self._result_markers: tuple[DebyerPeakMarker, ...] = tuple(markers)
        self.setWindowTitle(f"Edit Peak Markers: {pair_label}")
        self.resize(760, 360)

        layout = QVBoxLayout(self)
        summary = QLabel(
            "Edit the detected peak markers for this partial trace. "
            "You can add, remove, rename, or reposition peaks here."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Use", "r (A)", "Tag", "dx", "dy", "Source"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self.table, stretch=1)

        button_row = QHBoxLayout()
        add_button = QPushButton("Add Peak")
        add_button.clicked.connect(self._add_peak_row)
        button_row.addWidget(add_button)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_rows)
        button_row.addWidget(remove_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if markers:
            for marker in markers:
                self._append_marker_row(marker)
        else:
            self._add_peak_row()

    @property
    def result_markers(self) -> tuple[DebyerPeakMarker, ...]:
        return self._result_markers

    def _append_marker_row(self, marker: DebyerPeakMarker) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        enabled_item = QTableWidgetItem("")
        enabled_item.setFlags(
            enabled_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
        )
        enabled_item.setCheckState(
            Qt.CheckState.Checked
            if marker.enabled
            else Qt.CheckState.Unchecked
        )
        self.table.setItem(row, 0, enabled_item)
        self.table.setItem(row, 1, QTableWidgetItem(f"{marker.r_value:.6g}"))
        self.table.setItem(row, 2, QTableWidgetItem(marker.label))
        self.table.setItem(row, 3, QTableWidgetItem(f"{marker.text_dx:.6g}"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{marker.text_dy:.6g}"))
        self.table.setItem(row, 5, QTableWidgetItem(marker.source))

    def _add_peak_row(self) -> None:
        if self._r_values.size:
            default_r = float(self._r_values[len(self._r_values) // 2])
        else:
            default_r = 0.0
        self._append_marker_row(
            DebyerPeakMarker(
                r_value=default_r,
                label=f"{self._pair_label}: {default_r:.2f} A",
                enabled=True,
                text_dx=0.1,
                text_dy=0.0,
                source="manual",
            )
        )

    def _remove_selected_rows(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.table.selectedIndexes()},
            reverse=True,
        )
        for row in selected_rows:
            self.table.removeRow(row)

    def accept(self) -> None:
        try:
            markers: list[DebyerPeakMarker] = []
            for row in range(self.table.rowCount()):
                enabled_item = self.table.item(row, 0)
                r_item = self.table.item(row, 1)
                label_item = self.table.item(row, 2)
                dx_item = self.table.item(row, 3)
                dy_item = self.table.item(row, 4)
                source_item = self.table.item(row, 5)
                r_value = float(r_item.text().strip()) if r_item else 0.0
                label = (
                    label_item.text().strip()
                    if label_item and label_item.text().strip()
                    else f"{self._pair_label}: {r_value:.2f} A"
                )
                text_dx = float(dx_item.text().strip()) if dx_item else 0.0
                text_dy = float(dy_item.text().strip()) if dy_item else 0.0
                source = (
                    source_item.text().strip()
                    if source_item and source_item.text().strip()
                    else "manual"
                )
                markers.append(
                    DebyerPeakMarker(
                        r_value=r_value,
                        label=label,
                        enabled=(
                            enabled_item.checkState() == Qt.CheckState.Checked
                            if enabled_item is not None
                            else True
                        ),
                        text_dx=text_dx,
                        text_dy=text_dy,
                        source=source,
                    )
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid peak marker", str(exc))
            return
        self._result_markers = tuple(
            sorted(markers, key=lambda marker: marker.r_value)
        )
        super().accept()


class DebyerPDFMainWindow(QMainWindow):
    """Qt window for Debyer-backed PDF and partial-PDF averaging."""

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
        *,
        initial_frames_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_thread: QThread | None = None
        self._run_worker: DebyerPDFWorker | None = None
        self._latest_run_preview: DebyerPDFCalculation | None = None
        self._loaded_summaries: list[DebyerPDFCalculationSummary] = []
        self._current_calculation: DebyerPDFCalculation | None = None
        self._current_traces: list[dict[str, object]] = []
        self._trace_visibility: dict[str, bool] = {}
        self._trace_tag_visibility: dict[str, bool] = {}
        self._trace_colors: dict[str, str] = {}
        self._close_requested_during_run = False
        self._tag_artist_records: list[dict[str, object]] = []
        self._drag_state: dict[str, object] | None = None
        self._selected_tag: dict[str, object] | None = None
        self._experimental_summary: ExperimentalDataSummary | None = None
        self._build_ui()
        self._refresh_experimental_controls()
        self._delete_tag_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Delete),
            self,
        )
        self._delete_tag_shortcut.activated.connect(self._delete_selected_tag)
        self._backspace_tag_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Backspace),
            self,
        )
        self._backspace_tag_shortcut.activated.connect(
            self._delete_selected_tag
        )
        self._refresh_runtime_status()
        if initial_project_dir is not None:
            self.set_project_dir(initial_project_dir)
        if initial_frames_dir is not None:
            self.frames_dir_edit.setText(
                str(Path(initial_frames_dir).expanduser().resolve())
            )
            self._inspect_frames_dir()
        else:
            self._refresh_saved_calculations()
        self._refresh_plot()

    def closeEvent(self, event) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            self._request_run_cancel(
                "Closing window; stopping Debyer after active frame jobs "
                "finish and saving the partial average."
            )
            self.hide()
            while (
                self._run_thread is not None and self._run_thread.isRunning()
            ):
                QApplication.processEvents()
                if self._run_thread is not None:
                    self._run_thread.wait(50)
            event.accept()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (pdfsetup)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1460, 920)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.addWidget(self._build_left_panel())
        self._main_splitter.addWidget(self._build_right_panel())
        _configure_resize_splitter(
            self._main_splitter,
            handle_width=14,
            tooltip="Drag to resize the setup and results panes.",
        )
        self._main_splitter.setSizes([460, 980])
        root.addWidget(self._main_splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_left_panel(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_runtime_group())
        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_experimental_group())
        layout.addWidget(self._build_saved_calculations_group())
        layout.addWidget(self._build_settings_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_console_group(), stretch=1)

        wrapper = QScrollArea()
        wrapper.setWidgetResizable(True)
        wrapper.setFrameShape(QFrame.Shape.NoFrame)
        wrapper.setWidget(content)
        return wrapper

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self._build_results_tab(), "Results")
        self.result_tabs.addTab(
            self._build_shape_function_tab(),
            "Shape Function Analysis",
        )
        self.result_tabs.addTab(self._build_fit_tab(), "Fit")
        self.result_tabs.addTab(self._build_plot_settings_tab(), "Settings")
        layout.addWidget(self.result_tabs, stretch=1)
        return panel

    def _build_results_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_plot_controls())

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)
        self.figure = Figure(figsize=(10.2, 7.4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        plot_layout.addWidget(NavigationToolbar(self.canvas, plot_container))
        plot_layout.addWidget(self.canvas, stretch=1)
        right_splitter.addWidget(plot_container)

        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(6)
        self.calculation_info_label = QLabel(
            "Load or calculate a Debyer result to inspect the averaged PDF."
        )
        self.calculation_info_label.setWordWrap(True)
        self.calculation_info_label.setFrameShape(QFrame.Shape.StyledPanel)
        table_layout.addWidget(self.calculation_info_label)
        self.trace_table = QTableWidget(0, 8)
        self.trace_table.setHorizontalHeaderLabels(
            [
                "Visible",
                "Tag",
                "Trace",
                "Kind",
                "Peaks",
                "Edit",
                "Reset",
                "Color",
            ]
        )
        self.trace_table.verticalHeader().setVisible(False)
        self.trace_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.trace_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.trace_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.trace_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.trace_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        self.trace_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.ResizeToContents
        )
        self.trace_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.ResizeToContents
        )
        self.trace_table.horizontalHeader().setSectionResizeMode(
            7, QHeaderView.ResizeMode.ResizeToContents
        )
        table_layout.addWidget(self.trace_table, stretch=1)
        right_splitter.addWidget(table_container)
        _configure_resize_splitter(
            right_splitter,
            handle_width=12,
            tooltip="Drag to resize the plot and trace table.",
        )
        right_splitter.setSizes([620, 260])
        layout.addWidget(right_splitter, stretch=1)
        return tab

    def _build_shape_function_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        self.shape_function_status_label = QLabel(
            "Shape-function analysis setup will be added here."
        )
        self.shape_function_status_label.setWordWrap(True)
        self.shape_function_status_label.setFrameShape(
            QFrame.Shape.StyledPanel
        )
        layout.addWidget(self.shape_function_status_label)
        layout.addStretch(1)
        return tab

    def _build_fit_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QFormLayout()
        self.coordination_fit_trace_combo = QComboBox()
        self.coordination_fit_trace_combo.currentIndexChanged.connect(
            self._suggest_coordination_fit_window
        )
        form.addRow("R(r) trace", self.coordination_fit_trace_combo)

        window_widget = QWidget()
        window_layout = QHBoxLayout(window_widget)
        window_layout.setContentsMargins(0, 0, 0, 0)
        self.coordination_fit_r_min_spin = QDoubleSpinBox()
        self.coordination_fit_r_min_spin.setRange(0.0, 100000.0)
        self.coordination_fit_r_min_spin.setDecimals(4)
        self.coordination_fit_r_min_spin.setSingleStep(0.05)
        self.coordination_fit_r_min_spin.setValue(1.0)
        self.coordination_fit_r_max_spin = QDoubleSpinBox()
        self.coordination_fit_r_max_spin.setRange(0.0, 100000.0)
        self.coordination_fit_r_max_spin.setDecimals(4)
        self.coordination_fit_r_max_spin.setSingleStep(0.05)
        self.coordination_fit_r_max_spin.setValue(4.0)
        window_layout.addWidget(QLabel("from"))
        window_layout.addWidget(self.coordination_fit_r_min_spin)
        window_layout.addWidget(QLabel("to"))
        window_layout.addWidget(self.coordination_fit_r_max_spin)
        window_layout.addStretch(1)
        form.addRow("Fit window (A)", window_widget)

        seed_widget = QWidget()
        seed_layout = QHBoxLayout(seed_widget)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        self.coordination_fit_center_spin = QDoubleSpinBox()
        self.coordination_fit_center_spin.setRange(0.0, 100000.0)
        self.coordination_fit_center_spin.setDecimals(4)
        self.coordination_fit_center_spin.setSingleStep(0.05)
        self.coordination_fit_center_spin.setValue(2.5)
        self.coordination_fit_sigma_spin = QDoubleSpinBox()
        self.coordination_fit_sigma_spin.setRange(0.0001, 100000.0)
        self.coordination_fit_sigma_spin.setDecimals(4)
        self.coordination_fit_sigma_spin.setSingleStep(0.01)
        self.coordination_fit_sigma_spin.setValue(0.2)
        seed_layout.addWidget(QLabel("center"))
        seed_layout.addWidget(self.coordination_fit_center_spin)
        seed_layout.addWidget(QLabel("sigma"))
        seed_layout.addWidget(self.coordination_fit_sigma_spin)
        seed_layout.addStretch(1)
        form.addRow("Initial peak", seed_widget)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.coordination_fit_button = QPushButton("Fit R(r) Peak")
        self.coordination_fit_button.clicked.connect(
            self._fit_coordination_number
        )
        button_row.addWidget(self.coordination_fit_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.coordination_fit_status_label = QLabel(
            "Load or calculate a Debyer result before fitting R(r)."
        )
        self.coordination_fit_status_label.setWordWrap(True)
        self.coordination_fit_status_label.setFrameShape(
            QFrame.Shape.StyledPanel
        )
        layout.addWidget(self.coordination_fit_status_label)

        self.coordination_fit_results_table = QTableWidget(0, 9)
        self.coordination_fit_results_table.setHorizontalHeaderLabels(
            [
                "Trace",
                "r min",
                "r max",
                "Center",
                "Sigma",
                "CN",
                "Amplitude",
                "R^2",
                "RMSE",
            ]
        )
        self.coordination_fit_results_table.verticalHeader().setVisible(False)
        self.coordination_fit_results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in range(1, 9):
            self.coordination_fit_results_table.horizontalHeader().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        layout.addWidget(self.coordination_fit_results_table, stretch=1)
        return tab

    def _build_plot_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        form = QFormLayout(content)

        self.tag_font_size_spin = QDoubleSpinBox()
        self.tag_font_size_spin.setRange(4.0, 48.0)
        self.tag_font_size_spin.setDecimals(1)
        self.tag_font_size_spin.setValue(9.0)
        self.tag_font_size_spin.valueChanged.connect(self._refresh_plot)
        form.addRow("Tag font size", self.tag_font_size_spin)

        self.tag_font_family_combo = QFontComboBox()
        self.tag_font_family_combo.setCurrentFont(QFont("DejaVu Sans"))
        self.tag_font_family_combo.currentFontChanged.connect(
            self._refresh_plot
        )
        form.addRow("Tag font family", self.tag_font_family_combo)

        self.tag_bold_checkbox = QCheckBox("Use bold tag text")
        self.tag_bold_checkbox.toggled.connect(self._refresh_plot)
        form.addRow("", self.tag_bold_checkbox)

        self.tag_line_width_spin = QDoubleSpinBox()
        self.tag_line_width_spin.setRange(0.1, 8.0)
        self.tag_line_width_spin.setDecimals(2)
        self.tag_line_width_spin.setValue(0.8)
        self.tag_line_width_spin.valueChanged.connect(self._refresh_plot)
        form.addRow("Tag line width", self.tag_line_width_spin)

        self.tag_line_style_combo = QComboBox()
        self.tag_line_style_combo.addItems(["solid", "dashed", "dotted"])
        self.tag_line_style_combo.currentIndexChanged.connect(
            self._refresh_plot
        )
        form.addRow("Tag line style", self.tag_line_style_combo)

        self.axis_label_size_spin = QDoubleSpinBox()
        self.axis_label_size_spin.setRange(6.0, 40.0)
        self.axis_label_size_spin.setDecimals(1)
        self.axis_label_size_spin.setValue(11.0)
        self.axis_label_size_spin.valueChanged.connect(self._refresh_plot)
        form.addRow("Axis label size", self.axis_label_size_spin)

        self.axis_font_family_combo = QFontComboBox()
        self.axis_font_family_combo.setCurrentFont(QFont("DejaVu Sans"))
        self.axis_font_family_combo.currentFontChanged.connect(
            self._refresh_plot
        )
        form.addRow("Axis label font", self.axis_font_family_combo)

        self.axis_label_bold_checkbox = QCheckBox("Use bold axis labels")
        self.axis_label_bold_checkbox.toggled.connect(self._refresh_plot)
        form.addRow("", self.axis_label_bold_checkbox)

        peak_header = QLabel(
            "Peak maxima finder settings for raw partial PDFs. "
            "Grouped solvent/solute traces are intentionally excluded."
        )
        peak_header.setWordWrap(True)
        peak_header.setFrameShape(QFrame.Shape.StyledPanel)
        form.addRow("", peak_header)

        self.peak_min_height_spin = QDoubleSpinBox()
        self.peak_min_height_spin.setRange(0.0, 1.0)
        self.peak_min_height_spin.setDecimals(3)
        self.peak_min_height_spin.setSingleStep(0.01)
        self.peak_min_height_spin.setValue(0.12)
        form.addRow("Min relative height", self.peak_min_height_spin)

        self.peak_min_spacing_spin = QDoubleSpinBox()
        self.peak_min_spacing_spin.setRange(0.0, 20.0)
        self.peak_min_spacing_spin.setDecimals(3)
        self.peak_min_spacing_spin.setSingleStep(0.05)
        self.peak_min_spacing_spin.setValue(0.35)
        form.addRow("Min spacing (A)", self.peak_min_spacing_spin)

        self.peak_max_count_spin = QSpinBox()
        self.peak_max_count_spin.setRange(0, 50)
        self.peak_max_count_spin.setValue(6)
        form.addRow("Max peak count", self.peak_max_count_spin)

        button_row = QHBoxLayout()
        apply_plot_button = QPushButton("Apply Plot Settings")
        apply_plot_button.clicked.connect(self._refresh_plot)
        button_row.addWidget(apply_plot_button)
        recompute_button = QPushButton("Recompute Peak Maxima")
        recompute_button.clicked.connect(self._recompute_peak_markers)
        button_row.addWidget(recompute_button)
        button_row.addStretch(1)
        form.addRow("", button_row)

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)
        return tab

    def _build_runtime_group(self) -> QGroupBox:
        group = QGroupBox("Debyer Runtime")
        layout = QVBoxLayout(group)
        self.runtime_status_label = QLabel("Checking Debyer...")
        self.runtime_status_label.setWordWrap(True)
        self.runtime_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.runtime_status_label.setToolTip(
            "Debyer docs: "
            f"{DEBYER_DOCS_URL}\n"
            f"Debyer GitHub: {DEBYER_GITHUB_URL}"
        )
        layout.addWidget(self.runtime_status_label)
        return group

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Project and Frames")
        layout = QFormLayout(group)

        project_row = QWidget()
        project_layout = QHBoxLayout(project_row)
        project_layout.setContentsMargins(0, 0, 0, 0)
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.editingFinished.connect(
            self._refresh_saved_calculations
        )
        project_layout.addWidget(self.project_dir_edit, stretch=1)
        project_button = QPushButton("Browse…")
        project_button.clicked.connect(self._choose_project_dir)
        project_layout.addWidget(project_button)
        layout.addRow("Project folder", project_row)

        frames_row = QWidget()
        frames_layout = QHBoxLayout(frames_row)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        self.frames_dir_edit = QLineEdit()
        self.frames_dir_edit.editingFinished.connect(self._inspect_frames_dir)
        frames_layout.addWidget(self.frames_dir_edit, stretch=1)
        frames_button = QPushButton("Browse…")
        frames_button.clicked.connect(self._choose_frames_dir)
        frames_layout.addWidget(frames_button)
        layout.addRow("Frames folder", frames_row)

        self.frames_summary_label = QLabel(
            "Select a trajectory frame folder containing .xyz files."
        )
        self.frames_summary_label.setWordWrap(True)
        self.frames_summary_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addRow("", self.frames_summary_label)
        return group

    def _build_experimental_group(self) -> QGroupBox:
        group = QGroupBox("Experimental g(r)")
        layout = QFormLayout(group)

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.experimental_file_edit = QLineEdit()
        self.experimental_file_edit.editingFinished.connect(
            self._load_experimental_path_from_edit
        )
        file_layout.addWidget(self.experimental_file_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._choose_experimental_file)
        file_layout.addWidget(browse_button)
        layout.addRow("Data file", file_row)

        button_row = QHBoxLayout()
        self.experimental_columns_button = QPushButton("Columns...")
        self.experimental_columns_button.clicked.connect(
            self._configure_experimental_columns
        )
        button_row.addWidget(self.experimental_columns_button)
        self.clear_experimental_button = QPushButton("Clear")
        self.clear_experimental_button.clicked.connect(
            self._clear_experimental_file
        )
        button_row.addWidget(self.clear_experimental_button)
        button_row.addStretch(1)
        layout.addRow("", button_row)

        self.experimental_status_label = QLabel(
            "Optional: load an experimental file with r(A) and g(r) columns."
        )
        self.experimental_status_label.setWordWrap(True)
        self.experimental_status_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addRow("", self.experimental_status_label)
        self._refresh_experimental_controls()
        return group

    def _build_saved_calculations_group(self) -> QGroupBox:
        group = QGroupBox("Saved Calculations")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.saved_calculations_combo = QComboBox()
        controls.addWidget(self.saved_calculations_combo, stretch=1)
        self.load_saved_button = QPushButton("Load")
        self.load_saved_button.clicked.connect(self._load_selected_calculation)
        controls.addWidget(self.load_saved_button)
        self.refresh_saved_button = QPushButton("Refresh")
        self.refresh_saved_button.clicked.connect(
            self._refresh_saved_calculations
        )
        controls.addWidget(self.refresh_saved_button)
        layout.addLayout(controls)
        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("Debyer Settings")
        layout = QFormLayout(group)

        self.filename_prefix_edit = QLineEdit("debyer_pdf")
        layout.addRow("Output prefix", self.filename_prefix_edit)

        self.mode_combo = QComboBox()
        for mode in SUPPORTED_DEBYER_MODES:
            self.mode_combo.addItem(mode)
        self.mode_combo.setCurrentText("PDF")
        layout.addRow("Mode", self.mode_combo)

        range_widget = QWidget()
        range_layout = QGridLayout(range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(QLabel("from"), 0, 0)
        self.from_edit = QLineEdit("0.5")
        range_layout.addWidget(self.from_edit, 0, 1)
        range_layout.addWidget(QLabel("to"), 0, 2)
        self.to_edit = QLineEdit("15")
        range_layout.addWidget(self.to_edit, 0, 3)
        range_layout.addWidget(QLabel("step"), 0, 4)
        self.step_edit = QLineEdit("0.01")
        range_layout.addWidget(self.step_edit, 0, 5)
        layout.addRow("r-range (A)", range_widget)

        box_widget = QWidget()
        box_layout = QGridLayout(box_widget)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.addWidget(QLabel("a"), 0, 0)
        self.box_a_edit = QLineEdit()
        box_layout.addWidget(self.box_a_edit, 0, 1)
        box_layout.addWidget(QLabel("b"), 0, 2)
        self.box_b_edit = QLineEdit()
        box_layout.addWidget(self.box_b_edit, 0, 3)
        box_layout.addWidget(QLabel("c"), 0, 4)
        self.box_c_edit = QLineEdit()
        box_layout.addWidget(self.box_c_edit, 0, 5)
        layout.addRow("Bounding box (A)", box_widget)

        self.atom_count_edit = QLineEdit()
        layout.addRow("Atom count", self.atom_count_edit)
        self.rho0_label = QLabel(
            "rho0 will be computed from the atom count and box."
        )
        self.rho0_label.setWordWrap(True)
        layout.addRow("", self.rho0_label)

        self.solute_elements_edit = QLineEdit()
        self.solute_elements_edit.setPlaceholderText("Optional, e.g. Pb, I")
        self.solute_elements_edit.setToolTip(
            "Optional element list used to group partials into "
            "solute-solute, solute-solvent, and solvent-solvent traces."
        )
        layout.addRow("Solute elements", self.solute_elements_edit)

        self.apply_solute_groups_button = QPushButton("Apply Solute Groups")
        self.apply_solute_groups_button.clicked.connect(
            self._apply_solute_groups_from_ui
        )
        self.apply_solute_groups_button.setToolTip(
            "Update the loaded Debyer result with these solute elements and "
            "rebuild grouped partial traces without rerunning Debyer."
        )
        layout.addRow("", self.apply_solute_groups_button)

        self.store_frame_outputs_checkbox = QCheckBox(
            "Store per-frame Debyer output files"
        )
        self.store_frame_outputs_checkbox.setChecked(False)
        layout.addRow("", self.store_frame_outputs_checkbox)

        self.parallel_jobs_spin = QSpinBox()
        self.parallel_jobs_spin.setRange(1, 64)
        self.parallel_jobs_spin.setValue(default_parallel_debyer_jobs())
        self.parallel_jobs_spin.setToolTip(
            "Run multiple independent Debyer frame calculations at the same "
            "time. Use 1 for the old serial behavior."
        )
        layout.addRow("Parallel Debyer jobs", self.parallel_jobs_spin)

        self.update_plot_during_run_checkbox = QCheckBox(
            "Update plot while averaging"
        )
        self.update_plot_during_run_checkbox.setChecked(True)
        self.update_plot_during_run_checkbox.setToolTip(
            "If enabled, the average PDF plot refreshes during the Debyer "
            "run as more frame outputs are included. You can toggle this "
            "while averaging; turning it back on requests the next average."
        )
        self.update_plot_during_run_checkbox.toggled.connect(
            self._on_update_plot_during_run_toggled
        )
        layout.addRow("", self.update_plot_during_run_checkbox)

        for widget in (
            self.box_a_edit,
            self.box_b_edit,
            self.box_c_edit,
            self.atom_count_edit,
        ):
            widget.editingFinished.connect(self._update_rho0_label)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Calculate")
        layout = QVBoxLayout(group)

        self.calculate_button = QPushButton("Calculate Average PDF")
        self.calculate_button.clicked.connect(self._start_calculation)
        layout.addWidget(self.calculate_button)

        self.progress_label = QLabel("Progress: idle")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m frames")
        layout.addWidget(self.progress_bar)

        self.time_estimate_label = QLabel(
            "Estimated time remaining: waiting for the first timing sample..."
        )
        self.time_estimate_label.setWordWrap(True)
        layout.addWidget(self.time_estimate_label)
        return group

    def _build_console_group(self) -> QGroupBox:
        group = QGroupBox("Output Console")
        layout = QVBoxLayout(group)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)
        return group

    def _build_plot_controls(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("pdfPlotControls")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(8)
        layout.addLayout(selector_row)

        selector_row.addWidget(QLabel("Plot"))
        self.representation_combo = QComboBox()
        for label in SUPPORTED_PLOT_REPRESENTATIONS:
            self.representation_combo.addItem(label)
        self.representation_combo.setCurrentText("g(r)")
        self.representation_combo.currentIndexChanged.connect(
            self._rebuild_traces_and_plot
        )
        selector_row.addWidget(self.representation_combo)

        selector_row.addWidget(QLabel("Partial colors"))
        self.color_scheme_combo = QComboBox()
        for scheme in DEFAULT_COLOR_SCHEMES:
            self.color_scheme_combo.addItem(scheme)
        self.color_scheme_combo.setCurrentText("tab20")
        self.color_scheme_combo.currentIndexChanged.connect(
            self._apply_color_scheme
        )
        selector_row.addWidget(self.color_scheme_combo)

        self.legend_checkbox = QCheckBox("Legend")
        self.legend_checkbox.setChecked(True)
        self.legend_checkbox.toggled.connect(self._refresh_plot)
        selector_row.addWidget(self.legend_checkbox)

        self.fit_box_checkbox = QCheckBox("Fit Coefficient")
        self.fit_box_checkbox.setChecked(True)
        self.fit_box_checkbox.toggled.connect(self._refresh_plot)
        selector_row.addWidget(self.fit_box_checkbox)
        selector_row.addStretch(1)

        trace_row = QHBoxLayout()
        trace_row.setContentsMargins(0, 0, 0, 0)
        trace_row.setSpacing(8)
        layout.addLayout(trace_row)

        self.average_toggle_button = QPushButton("Hide Average")
        self.average_toggle_button.clicked.connect(self._toggle_average_trace)
        trace_row.addWidget(self.average_toggle_button)

        self.partials_toggle_button = QPushButton("Show Partial PDFs")
        self.partials_toggle_button.clicked.connect(
            self._toggle_partial_traces
        )
        trace_row.addWidget(self.partials_toggle_button)

        self.groups_toggle_button = QPushButton("Show Grouped Partials")
        self.groups_toggle_button.clicked.connect(self._toggle_group_traces)
        trace_row.addWidget(self.groups_toggle_button)

        self.experimental_toggle_button = QPushButton("Hide Experimental")
        self.experimental_toggle_button.clicked.connect(
            self._toggle_experimental_trace
        )
        trace_row.addWidget(self.experimental_toggle_button)

        self.export_active_traces_button = QPushButton(
            "Export Active Traces..."
        )
        self.export_active_traces_button.clicked.connect(
            self._export_active_traces
        )
        trace_row.addWidget(self.export_active_traces_button)
        trace_row.addStretch(1)
        return widget

    def set_project_dir(self, project_dir: str | Path | None) -> None:
        if project_dir is None:
            self.project_dir_edit.clear()
        else:
            self.project_dir_edit.setText(
                str(Path(project_dir).expanduser().resolve())
            )
        self._refresh_saved_calculations()

    def _refresh_runtime_status(self) -> None:
        status = check_debyer_runtime()
        self.runtime_status_label.setText(
            status.message
            + "\n\nDebyer docs: "
            + DEBYER_DOCS_URL
            + "\nDebyer GitHub: "
            + DEBYER_GITHUB_URL
            + "\nTotal scattering formalism reference: "
            + TOTAL_SCATTERING_PAPER_URL
        )

    def _choose_project_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select SAXSShell project folder",
            self.project_dir_edit.text().strip() or str(Path.home()),
        )
        if not selected:
            return
        self.set_project_dir(selected)

    def _choose_frames_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Debyer frames folder",
            self.frames_dir_edit.text().strip() or str(Path.home()),
        )
        if not selected:
            return
        self.frames_dir_edit.setText(selected)
        self._inspect_frames_dir()

    def _choose_experimental_file(self) -> None:
        start_dir = (
            self.experimental_file_edit.text().strip()
            or self.project_dir_edit.text().strip()
            or str(Path.home())
        )
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select experimental PDF g(r) data file",
            start_dir,
            "Data files (*.txt *.dat *.iq);;All files (*)",
        )
        if not selected_path:
            return
        self._load_experimental_file(Path(selected_path).expanduser())

    def _load_experimental_path_from_edit(self) -> None:
        text = self.experimental_file_edit.text().strip()
        if not text:
            if self._experimental_summary is not None:
                self._clear_experimental_file()
            return
        path = Path(text).expanduser()
        if (
            self._experimental_summary is not None
            and path.resolve() == self._experimental_summary.path
        ):
            return
        self._load_experimental_file(path)

    def _load_experimental_file(self, file_path: Path) -> None:
        resolved = file_path.expanduser().resolve()
        if not resolved.is_file():
            QMessageBox.warning(
                self,
                "Experimental g(r)",
                "The selected experimental data file does not exist: "
                f"{resolved}",
            )
            return
        try:
            summary = load_experimental_data_file(resolved, skiprows=0)
        except Exception:
            dialog = self._build_experimental_header_dialog(resolved)
            if dialog.exec() != int(QDialog.DialogCode.Accepted):
                return
            summary = dialog.accepted_summary
            if summary is None:
                return
        self._apply_experimental_file(summary)

    def _build_experimental_header_dialog(
        self,
        file_path: Path,
    ) -> ExperimentalDataHeaderDialog:
        initial_summary = (
            self._experimental_summary
            if self._experimental_summary is not None
            and self._experimental_summary.path == file_path
            else None
        )
        return ExperimentalDataHeaderDialog(
            file_path,
            self,
            title="Check Experimental g(r) Data File",
            independent_column_label="r(A) column",
            dependent_column_label="g(r) column",
            error_column_label="Error column (optional)",
            intro_text=(
                "Adjust the number of header rows to skip, confirm which "
                "columns correspond to r(A) and g(r), and then load the file."
            ),
            initial_header_rows=(
                initial_summary.header_rows
                if initial_summary is not None
                else None
            ),
            initial_q_column=(
                initial_summary.q_column
                if initial_summary is not None
                else None
            ),
            initial_intensity_column=(
                initial_summary.intensity_column
                if initial_summary is not None
                else None
            ),
            initial_error_column=(
                initial_summary.error_column
                if initial_summary is not None
                else None
            ),
        )

    def _configure_experimental_columns(self) -> None:
        if self._experimental_summary is None:
            text = self.experimental_file_edit.text().strip()
            if not text:
                self.experimental_status_label.setText(
                    "Select an experimental g(r) file before configuring "
                    "columns."
                )
                return
            file_path = Path(text).expanduser().resolve()
        else:
            file_path = self._experimental_summary.path
        dialog = self._build_experimental_header_dialog(file_path)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        summary = dialog.accepted_summary
        if summary is None:
            return
        self._apply_experimental_file(summary)

    def _apply_experimental_file(
        self,
        summary: ExperimentalDataSummary,
    ) -> None:
        self._experimental_summary = summary
        self.experimental_file_edit.setText(str(summary.path))
        self._trace_visibility.setdefault("experimental", True)
        self._trace_colors.setdefault("experimental", "#d62728")
        self.experimental_status_label.setText(
            self._experimental_summary_text(summary)
        )
        self._append_log(
            "Loaded experimental g(r) data from " f"{summary.path}"
        )
        self._refresh_experimental_controls()
        self._rebuild_traces_and_plot()

    def _clear_experimental_file(self) -> None:
        self._experimental_summary = None
        self.experimental_file_edit.clear()
        self._trace_visibility.pop("experimental", None)
        self._trace_tag_visibility.pop("experimental", None)
        self.experimental_status_label.setText(
            "Optional: load an experimental file with r(A) and g(r) columns."
        )
        self._refresh_experimental_controls()
        self._rebuild_traces_and_plot()

    def _refresh_experimental_controls(self) -> None:
        has_experimental = self._experimental_summary is not None
        if hasattr(self, "experimental_columns_button"):
            self.experimental_columns_button.setEnabled(has_experimental)
        if hasattr(self, "clear_experimental_button"):
            self.clear_experimental_button.setEnabled(has_experimental)
        if hasattr(self, "experimental_toggle_button"):
            self.experimental_toggle_button.setEnabled(has_experimental)
        if hasattr(self, "fit_box_checkbox"):
            self.fit_box_checkbox.setEnabled(has_experimental)

    def _experimental_summary_text(
        self,
        summary: ExperimentalDataSummary,
    ) -> str:
        r_values = np.asarray(summary.q_values, dtype=float)
        if r_values.size:
            r_range = (
                f"{float(np.nanmin(r_values)):.6g} to "
                f"{float(np.nanmax(r_values)):.6g} A"
            )
        else:
            r_range = "unknown"
        return (
            f"Loaded {summary.path.name}: {len(r_values)} points\n"
            f"r range: {r_range}\n"
            f"Columns: {self._experimental_column_text(summary)}"
        )

    @staticmethod
    def _experimental_column_text(
        summary: ExperimentalDataSummary,
    ) -> str:
        def _column_label(index: int | None, fallback: str) -> str:
            if index is None:
                return "None"
            if 0 <= index < len(summary.column_names):
                return summary.column_names[index]
            return fallback

        return (
            f"r(A)={_column_label(summary.q_column, 'Column 1')}; "
            f"g(r)={_column_label(summary.intensity_column, 'Column 2')}"
        )

    def _inspect_frames_dir(self) -> None:
        text = self.frames_dir_edit.text().strip()
        if not text:
            self.frames_summary_label.setText(
                "Select a trajectory frame folder containing .xyz files."
            )
            return
        try:
            inspection = inspect_frames_dir(text)
        except Exception as exc:
            self.frames_summary_label.setText(str(exc))
            return

        if not self.filename_prefix_edit.text().strip():
            self.filename_prefix_edit.setText(inspection.frames_dir.name)
        if not self.project_dir_edit.text().strip():
            self.project_dir_edit.setText(str(self._suggest_project_dir()))
            self._refresh_saved_calculations()
        if not self.atom_count_edit.text().strip():
            self.atom_count_edit.setText(str(inspection.atom_count))
        detected_box = (
            inspection.detected_box_dimensions
            if inspection.detected_box_dimensions is not None
            else inspection.estimated_box_dimensions
        )
        if detected_box is not None:
            if not self.box_a_edit.text().strip():
                self.box_a_edit.setText(f"{detected_box[0]:g}")
            if not self.box_b_edit.text().strip():
                self.box_b_edit.setText(f"{detected_box[1]:g}")
            if not self.box_c_edit.text().strip():
                self.box_c_edit.setText(f"{detected_box[2]:g}")
        inferred_solutes = infer_default_solute_elements(
            inspection.element_counts
        )
        if inferred_solutes and not self.solute_elements_edit.text().strip():
            self.solute_elements_edit.setText(", ".join(inferred_solutes))
        element_summary = ", ".join(
            f"{element}{count if count != 1 else ''}"
            for element, count in sorted(inspection.element_counts.items())
        )
        solute_summary = (
            ", ".join(inferred_solutes) if inferred_solutes else "not inferred"
        )
        box_summary = "unknown"
        if inspection.detected_box_dimensions is not None:
            box_summary = (
                " x ".join(
                    f"{value:.3f}"
                    for value in inspection.detected_box_dimensions
                )
                + " A"
            )
            if inspection.detected_box_source is not None:
                box_summary += f" (from {inspection.detected_box_source})"
        elif inspection.estimated_box_dimensions is not None:
            box_summary = (
                " x ".join(
                    f"{value:.3f}"
                    for value in inspection.estimated_box_dimensions
                )
                + " A (estimated from first frame)"
            )
        self.frames_summary_label.setText(
            f"Detected {inspection.frame_format.upper()} frames: "
            f"{len(inspection.frame_paths)} files\n"
            f"Elements in first frame: {element_summary or 'unknown'}\n"
            f"Default solutes: {solute_summary}\n"
            f"Bounding box: {box_summary}"
        )
        self._update_rho0_label()

    def _refresh_saved_calculations(self) -> None:
        project_dir = self.project_dir_edit.text().strip()
        self.saved_calculations_combo.blockSignals(True)
        self.saved_calculations_combo.clear()
        self._loaded_summaries = (
            []
            if not project_dir
            else list_saved_debyer_calculations(project_dir)
        )
        for summary in self._loaded_summaries:
            label = (
                f"{summary.created_at} | {summary.filename_prefix} | "
                f"{summary.mode} | {summary.frame_count} frames"
            )
            self.saved_calculations_combo.addItem(
                label,
                str(summary.calculation_dir),
            )
        self.saved_calculations_combo.blockSignals(False)
        has_saved = bool(self._loaded_summaries)
        self.load_saved_button.setEnabled(has_saved)
        if has_saved and self._current_calculation is None:
            self.saved_calculations_combo.setCurrentIndex(0)
            self._load_selected_calculation()

    def _load_selected_calculation(self) -> None:
        calculation_dir = self.saved_calculations_combo.currentData()
        if not calculation_dir:
            return
        try:
            calculation = load_debyer_calculation(calculation_dir)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to load Debyer calculation",
                str(exc),
            )
            return
        self._apply_loaded_calculation(calculation)

    def _apply_loaded_calculation(
        self,
        calculation: DebyerPDFCalculation,
    ) -> None:
        self._selected_tag = None
        self._drag_state = None
        self._current_calculation = calculation
        self.project_dir_edit.setText(str(calculation.project_dir))
        self.frames_dir_edit.setText(str(calculation.frames_dir))
        self.filename_prefix_edit.setText(calculation.filename_prefix)
        self.mode_combo.setCurrentText(calculation.mode)
        self.from_edit.setText(f"{calculation.from_value:g}")
        self.to_edit.setText(f"{calculation.to_value:g}")
        self.step_edit.setText(f"{calculation.step_value:g}")
        self.box_a_edit.setText(f"{calculation.box_dimensions[0]:g}")
        self.box_b_edit.setText(f"{calculation.box_dimensions[1]:g}")
        self.box_c_edit.setText(f"{calculation.box_dimensions[2]:g}")
        self.atom_count_edit.setText(str(calculation.atom_count))
        self.solute_elements_edit.setText(
            ", ".join(calculation.solute_elements)
        )
        self.store_frame_outputs_checkbox.setChecked(
            calculation.store_frame_outputs
        )
        self._load_peak_finder_settings_into_ui(
            calculation.peak_finder_settings
        )
        self._update_rho0_label()
        self.calculation_info_label.setText(
            self._calculation_summary_text(calculation)
        )
        self._rebuild_traces_and_plot()

    def _calculation_summary_text(
        self,
        calculation: DebyerPDFCalculation,
    ) -> str:
        processed_frames = (
            calculation.frame_count
            if calculation.processed_frame_count is None
            else int(calculation.processed_frame_count)
        )
        heading = (
            "Live running average"
            if calculation.is_partial_average
            else "Saved calculation"
        )
        lines = [
            f"{heading}: {calculation.created_at}",
            (
                f"Frames averaged: {processed_frames}/{calculation.frame_count}"
                if calculation.is_partial_average
                else f"Frames averaged: {calculation.frame_count}"
            ),
            f"Raw Debyer mode: {calculation.mode}",
            (
                f"Range: {calculation.from_value:g} to "
                f"{calculation.to_value:g} A (step {calculation.step_value:g})"
            ),
            f"rho0: {calculation.rho0:.6g} atoms/A^3",
            "Solute elements: "
            + (
                ", ".join(calculation.solute_elements)
                if calculation.solute_elements
                else "None"
            ),
            f"Frames folder: {calculation.frames_dir}",
        ]
        if calculation.elapsed_seconds is not None:
            lines.append(
                f"Elapsed: {self._format_duration(calculation.elapsed_seconds)}"
            )
        if calculation.estimated_remaining_seconds is not None:
            lines.append(
                "Estimated remaining: "
                + self._format_duration(
                    calculation.estimated_remaining_seconds
                )
            )
        if calculation.expected_total_seconds is not None:
            lines.append(
                "Estimated total: "
                + self._format_duration(calculation.expected_total_seconds)
            )
        return "\n".join(lines)

    def _parse_box_dimensions(self) -> tuple[float, float, float]:
        return (
            float(self.box_a_edit.text().strip()),
            float(self.box_b_edit.text().strip()),
            float(self.box_c_edit.text().strip()),
        )

    def _coerce_r_range_maximum_for_box(
        self,
        r_max: float,
        box_dimensions: tuple[float, float, float],
    ) -> float:
        box_values = np.asarray(box_dimensions, dtype=float)
        if (
            box_values.size != 3
            or not np.all(np.isfinite(box_values))
            or np.any(box_values <= 0.0)
        ):
            return r_max
        allowed_r_max = float(np.min(box_values) * 0.5)
        if r_max <= allowed_r_max:
            return r_max
        self.to_edit.setText(f"{allowed_r_max:g}")
        self.statusBar().showMessage(
            "Adjusted r-range maximum to half of the minimum box dimension.",
            5000,
        )
        return allowed_r_max

    def _parse_solute_elements(self) -> tuple[str, ...]:
        raw = self.solute_elements_edit.text().strip()
        if not raw:
            return ()
        values = [token.strip() for token in raw.replace(";", ",").split(",")]
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            element = value[:1].upper() + value[1:].lower()
            if element in seen:
                continue
            cleaned.append(element)
            seen.add(element)
        return tuple(cleaned)

    def _apply_solute_groups_from_ui(self) -> None:
        solute_elements = self._parse_solute_elements()
        if self._current_calculation is None:
            self.statusBar().showMessage(
                "Solute elements will be used for the next Debyer run.",
                4000,
            )
            return
        self._current_calculation = replace(
            self._current_calculation,
            solute_elements=solute_elements,
            target_peak_markers={},
        )
        rewrite_debyer_calculation_output(self._current_calculation)
        self._persist_current_calculation()
        self.calculation_info_label.setText(
            self._calculation_summary_text(self._current_calculation)
        )
        self._rebuild_traces_and_plot()
        self.statusBar().showMessage(
            "Updated grouped partial traces from the solute elements.",
            4000,
        )

    def _suggest_project_dir(self) -> Path:
        frames_dir = self.frames_dir_edit.text().strip()
        if frames_dir:
            resolved = Path(frames_dir).expanduser().resolve()
            return resolved.parent / f"{resolved.name}_pdfsetup"
        return Path.home() / "saxshell_pdf_project"

    def _build_settings(self) -> DebyerPDFSettings:
        project_text = self.project_dir_edit.text().strip()
        if not project_text:
            suggested = self._suggest_project_dir()
            self.project_dir_edit.setText(str(suggested))
            project_text = str(suggested)
        frames_text = self.frames_dir_edit.text().strip()
        if not frames_text:
            raise ValueError("Select a frames folder before running Debyer.")

        box_dimensions = self._parse_box_dimensions()
        to_value = self._coerce_r_range_maximum_for_box(
            float(self.to_edit.text().strip()),
            box_dimensions,
        )

        return DebyerPDFSettings(
            project_dir=Path(project_text).expanduser().resolve(),
            frames_dir=Path(frames_text).expanduser().resolve(),
            filename_prefix=self.filename_prefix_edit.text().strip()
            or "debyer_pdf",
            mode=self.mode_combo.currentText(),
            from_value=float(self.from_edit.text().strip()),
            to_value=to_value,
            step_value=float(self.step_edit.text().strip()),
            box_dimensions=box_dimensions,
            atom_count=int(float(self.atom_count_edit.text().strip())),
            store_frame_outputs=bool(
                self.store_frame_outputs_checkbox.isChecked()
            ),
            solute_elements=self._parse_solute_elements(),
            max_parallel_jobs=int(self.parallel_jobs_spin.value()),
        )

    def _current_peak_finder_settings_from_ui(
        self,
    ) -> DebyerPeakFinderSettings:
        return DebyerPeakFinderSettings(
            min_relative_height=float(self.peak_min_height_spin.value()),
            min_spacing_angstrom=float(self.peak_min_spacing_spin.value()),
            max_peak_count=int(self.peak_max_count_spin.value()),
        )

    def _load_peak_finder_settings_into_ui(
        self,
        settings: DebyerPeakFinderSettings,
    ) -> None:
        self.peak_min_height_spin.blockSignals(True)
        self.peak_min_spacing_spin.blockSignals(True)
        self.peak_max_count_spin.blockSignals(True)
        self.peak_min_height_spin.setValue(settings.min_relative_height)
        self.peak_min_spacing_spin.setValue(settings.min_spacing_angstrom)
        self.peak_max_count_spin.setValue(settings.max_peak_count)
        self.peak_min_height_spin.blockSignals(False)
        self.peak_min_spacing_spin.blockSignals(False)
        self.peak_max_count_spin.blockSignals(False)

    def _peak_markers_for_trace(
        self, trace_key: str
    ) -> tuple[DebyerPeakMarker, ...]:
        if self._current_calculation is None or not trace_key.startswith(
            "partial:"
        ):
            return ()
        pair_label = trace_key.split(":", 1)[1]
        return tuple(
            self._current_calculation.partial_peak_markers.get(pair_label, ())
        )

    def _peak_summary_text(self, trace_key: str) -> str:
        markers = [
            marker
            for marker in self._peak_markers_for_trace(trace_key)
            if marker.enabled
        ]
        if not markers:
            return "—"
        return ", ".join(
            f"{marker.r_value:.2f}"
            for marker in sorted(
                markers,
                key=lambda peak_marker: peak_marker.r_value,
            )
        )

    def _persist_current_calculation(self) -> None:
        if self._current_calculation is None:
            return
        write_debyer_calculation_metadata(self._current_calculation)

    def _update_partial_peak_markers(
        self,
        pair_label: str,
        markers: tuple[DebyerPeakMarker, ...],
        *,
        persist: bool,
        refresh_table: bool = True,
        sort_markers: bool = True,
    ) -> None:
        if self._current_calculation is None:
            return
        updated_markers = dict(self._current_calculation.partial_peak_markers)
        updated_markers[pair_label] = (
            tuple(sorted(markers, key=lambda marker: marker.r_value))
            if sort_markers
            else tuple(markers)
        )
        self._current_calculation = replace(
            self._current_calculation,
            partial_peak_markers=updated_markers,
            peak_finder_settings=self._current_peak_finder_settings_from_ui(),
        )
        if persist:
            self._persist_current_calculation()
        if refresh_table:
            self._refresh_trace_table()
        self._refresh_plot()

    def _recompute_peak_markers(self) -> None:
        if self._current_calculation is None:
            return
        settings = self._current_peak_finder_settings_from_ui()
        recalculated = estimate_partial_peak_markers(
            r_values=self._current_calculation.r_values,
            partial_values=self._current_calculation.partial_values,
            settings=settings,
        )
        self._current_calculation = replace(
            self._current_calculation,
            partial_peak_markers=recalculated,
            target_peak_markers={},
            peak_finder_settings=settings,
        )
        self._persist_current_calculation()
        self._refresh_trace_table()
        self._refresh_plot()

    def _edit_peak_markers(self, trace_key: str) -> None:
        if self._current_calculation is None or not trace_key.startswith(
            "partial:"
        ):
            return
        pair_label = trace_key.split(":", 1)[1]
        dialog = DebyerPeakEditorDialog(
            pair_label=pair_label,
            markers=self._peak_markers_for_trace(trace_key),
            r_values=self._current_calculation.r_values,
            parent=self,
        )
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        self._update_partial_peak_markers(
            pair_label,
            dialog.result_markers,
            persist=True,
            refresh_table=True,
            sort_markers=True,
        )

    def _set_trace_tag_visible(self, trace_key: str, visible: bool) -> None:
        self._trace_tag_visibility[trace_key] = bool(visible)
        self._refresh_plot()

    def _set_selected_tag(
        self,
        pair_label: str | None,
        marker_index: int | None = None,
        target_trace_key: str | None = None,
    ) -> None:
        previous = self._selected_tag
        if pair_label is None or marker_index is None:
            self._selected_tag = None
        else:
            self._selected_tag = {
                "pair_label": str(pair_label),
                "marker_index": int(marker_index),
                "target_trace_key": (
                    str(target_trace_key)
                    if target_trace_key is not None
                    else "average"
                ),
            }
        if previous != self._selected_tag:
            self._refresh_plot()

    def _enabled_target_trace_keys_for_pair(
        self,
        pair_label: str,
        *,
        require_visible: bool,
    ) -> list[str]:
        if self._current_calculation is None:
            return []
        target_trace_keys: list[str] = []
        if self._trace_tag_visibility.get("average", False) and (
            not require_visible or self._trace_visibility.get("average", False)
        ):
            target_trace_keys.append("average")
        family = classify_partial_pair(
            pair_label,
            solute_elements=set(self._current_calculation.solute_elements),
        )
        if family is None:
            return target_trace_keys
        group_trace_key = f"group:{family}"
        if self._trace_tag_visibility.get(group_trace_key, False) and (
            not require_visible
            or self._trace_visibility.get(group_trace_key, False)
        ):
            target_trace_keys.append(group_trace_key)
        return target_trace_keys

    def _target_peak_markers_for_trace(
        self,
        *,
        pair_label: str,
        target_trace_key: str,
    ) -> tuple[DebyerPeakMarker, ...]:
        if self._current_calculation is None:
            return ()
        if target_trace_key != "average":
            target_payload = self._current_calculation.target_peak_markers.get(
                target_trace_key,
                {},
            )
            target_markers = target_payload.get(pair_label)
            if target_markers is not None:
                return tuple(target_markers)
        return tuple(
            self._current_calculation.partial_peak_markers.get(pair_label, ())
        )

    def _update_target_peak_markers(
        self,
        *,
        target_trace_key: str,
        pair_label: str,
        markers: tuple[DebyerPeakMarker, ...],
        persist: bool,
        refresh_table: bool = True,
        sort_markers: bool = True,
    ) -> None:
        if self._current_calculation is None:
            return
        if target_trace_key == "average":
            self._update_partial_peak_markers(
                pair_label,
                markers,
                persist=persist,
                refresh_table=refresh_table,
                sort_markers=sort_markers,
            )
            return
        updated_target_markers = {
            trace_key: dict(pair_map)
            for trace_key, pair_map in self._current_calculation.target_peak_markers.items()
        }
        target_map = dict(updated_target_markers.get(target_trace_key, {}))
        target_map[pair_label] = (
            tuple(sorted(markers, key=lambda marker: marker.r_value))
            if sort_markers
            else tuple(markers)
        )
        updated_target_markers[target_trace_key] = target_map
        self._current_calculation = replace(
            self._current_calculation,
            target_peak_markers=updated_target_markers,
        )
        if persist:
            self._persist_current_calculation()
        if refresh_table:
            self._refresh_trace_table()
        self._refresh_plot()

    def _snapshot_other_target_tags(
        self,
        *,
        pair_label: str,
        exclude_target_trace_key: str,
    ) -> None:
        if self._current_calculation is None:
            return
        updated_target_markers = {
            trace_key: dict(pair_map)
            for trace_key, pair_map in self._current_calculation.target_peak_markers.items()
        }
        changed = False
        for target_trace_key in self._enabled_target_trace_keys_for_pair(
            pair_label,
            require_visible=False,
        ):
            if target_trace_key == exclude_target_trace_key:
                continue
            if target_trace_key == "average":
                continue
            target_map = dict(updated_target_markers.get(target_trace_key, {}))
            if pair_label in target_map:
                continue
            target_map[pair_label] = tuple(
                self._target_peak_markers_for_trace(
                    pair_label=pair_label,
                    target_trace_key=target_trace_key,
                )
            )
            updated_target_markers[target_trace_key] = target_map
            changed = True
        if changed:
            self._current_calculation = replace(
                self._current_calculation,
                target_peak_markers=updated_target_markers,
            )

    def _clear_target_peak_overrides_for_pair(self, pair_label: str) -> None:
        if self._current_calculation is None:
            return
        updated_target_markers = {
            trace_key: dict(pair_map)
            for trace_key, pair_map in self._current_calculation.target_peak_markers.items()
        }
        changed = False
        for trace_key, pair_map in list(updated_target_markers.items()):
            if pair_label in pair_map:
                del pair_map[pair_label]
                changed = True
            if pair_map:
                updated_target_markers[trace_key] = pair_map
            else:
                updated_target_markers.pop(trace_key, None)
        if changed:
            self._current_calculation = replace(
                self._current_calculation,
                target_peak_markers=updated_target_markers,
            )

    def _reset_partial_peak_markers(self, trace_key: str) -> None:
        if self._current_calculation is None or not trace_key.startswith(
            "partial:"
        ):
            return
        pair_label = trace_key.split(":", 1)[1]
        default_markers = find_partial_peak_markers(
            pair_label=pair_label,
            r_values=self._current_calculation.r_values,
            values=np.asarray(
                self._current_calculation.partial_values.get(pair_label, []),
                dtype=float,
            ),
            settings=self._current_peak_finder_settings_from_ui(),
        )
        self._set_selected_tag(None)
        self._clear_target_peak_overrides_for_pair(pair_label)
        self._update_partial_peak_markers(
            pair_label,
            default_markers,
            persist=True,
            refresh_table=True,
            sort_markers=True,
        )

    def _delete_selected_tag(self) -> None:
        if self._current_calculation is None or self._selected_tag is None:
            return
        pair_label = str(self._selected_tag["pair_label"])
        marker_index = int(self._selected_tag["marker_index"])
        target_trace_key = str(
            self._selected_tag.get("target_trace_key", "average")
        )
        markers = list(
            self._target_peak_markers_for_trace(
                pair_label=pair_label,
                target_trace_key=target_trace_key,
            )
        )
        if not (0 <= marker_index < len(markers)):
            self._set_selected_tag(None)
            return
        del markers[marker_index]
        self._set_selected_tag(None)
        if target_trace_key == "average":
            self._snapshot_other_target_tags(
                pair_label=pair_label,
                exclude_target_trace_key=target_trace_key,
            )
            self._update_partial_peak_markers(
                pair_label,
                tuple(markers),
                persist=True,
                refresh_table=True,
                sort_markers=True,
            )
            return
        self._update_target_peak_markers(
            target_trace_key=target_trace_key,
            pair_label=pair_label,
            markers=tuple(markers),
            persist=True,
            refresh_table=True,
            sort_markers=True,
        )

    def _average_trace_values(self) -> np.ndarray | None:
        for trace in self._current_traces:
            if str(trace["kind"]) == "average":
                return np.asarray(trace["values"], dtype=float)
        return None

    def _average_value_at_r(self, r_value: float) -> float | None:
        if self._current_calculation is None:
            return None
        average_values = self._average_trace_values()
        if average_values is None:
            return None
        return float(
            np.interp(
                float(r_value),
                np.asarray(self._current_calculation.r_values, dtype=float),
                average_values,
            )
        )

    def _tag_arrow_linestyle(self) -> str:
        mapping = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
        }
        return mapping.get(self.tag_line_style_combo.currentText(), "-")

    @staticmethod
    def _sanitize_trace_column_name(value: str) -> str:
        text = "".join(
            (
                character
                if character.isalnum() or character in {"-", "_", ":"}
                else "_"
            )
            for character in str(value).strip()
        ).strip("_")
        return text or "trace"

    def _trace_x_values(self, trace: dict[str, object]) -> np.ndarray:
        if "x_values" in trace:
            return np.asarray(trace["x_values"], dtype=float)
        if self._current_calculation is None:
            return np.asarray([], dtype=float)
        return np.asarray(self._current_calculation.r_values, dtype=float)

    def _trace_values_on_export_grid(
        self,
        trace: dict[str, object],
    ) -> np.ndarray:
        if self._current_calculation is None:
            return np.asarray([], dtype=float)
        target_r = np.asarray(self._current_calculation.r_values, dtype=float)
        source_r = self._trace_x_values(trace)
        source_values = np.asarray(trace["values"], dtype=float)
        if source_r.size == target_r.size and np.allclose(
            source_r,
            target_r,
        ):
            return source_values
        valid = np.isfinite(source_r) & np.isfinite(source_values)
        if valid.sum() < 2:
            return np.full_like(target_r, np.nan, dtype=float)
        source_r = source_r[valid]
        source_values = source_values[valid]
        order = np.argsort(source_r)
        source_r = source_r[order]
        source_values = source_values[order]
        unique_r, unique_indices = np.unique(source_r, return_index=True)
        source_r = unique_r
        source_values = source_values[unique_indices]
        if source_r.size < 2:
            return np.full_like(target_r, np.nan, dtype=float)
        interpolated = np.full_like(target_r, np.nan, dtype=float)
        inside = (target_r >= source_r[0]) & (target_r <= source_r[-1])
        interpolated[inside] = np.interp(
            target_r[inside],
            source_r,
            source_values,
        )
        return interpolated

    def _active_trace_export_columns(
        self,
    ) -> tuple[list[str], list[np.ndarray]] | None:
        if self._current_calculation is None or not self._current_traces:
            return None
        column_names = ["r"]
        column_arrays = [
            np.asarray(self._current_calculation.r_values, dtype=float)
        ]
        for trace in self._current_traces:
            trace_key = str(trace["key"])
            if not self._trace_visibility.get(trace_key, False):
                continue
            column_names.append(
                self._sanitize_trace_column_name(str(trace["label"]))
            )
            column_arrays.append(self._trace_values_on_export_grid(trace))
        if len(column_names) <= 1:
            return None
        return column_names, column_arrays

    def _default_active_trace_export_path(self) -> Path:
        representation = self._sanitize_trace_column_name(
            self.representation_combo.currentText()
        )
        if self._current_calculation is not None:
            filename = (
                f"{self._current_calculation.filename_prefix}_"
                f"{representation}_active_traces.txt"
            )
            return self._current_calculation.calculation_dir / filename
        project_text = self.project_dir_edit.text().strip()
        root = (
            Path(project_text).expanduser().resolve()
            if project_text
            else Path.home()
        )
        return root / f"debyer_{representation}_active_traces.txt"

    def _export_active_traces(self) -> None:
        payload = self._active_trace_export_columns()
        if payload is None:
            QMessageBox.information(
                self,
                "No active traces",
                "Show at least one trace before exporting active traces.",
            )
            return
        default_path = self._default_active_trace_export_path()
        selected_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export active Debyer traces",
            str(default_path),
            "Text files (*.txt);;Data files (*.dat);;All files (*)",
        )
        if not selected_path:
            return
        column_names, column_arrays = payload
        output_path = Path(selected_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header_lines = [
            "# Debyer active trace export",
            f"# representation: {self.representation_combo.currentText()}",
        ]
        if self._current_calculation is not None:
            header_lines.extend(
                [
                    f"# calculation_id: {self._current_calculation.calculation_id}",
                    f"# created_at: {self._current_calculation.created_at}",
                    f"# filename_prefix: {self._current_calculation.filename_prefix}",
                    f"# mode: {self._current_calculation.mode}",
                ]
            )
        header_lines.append("# columns: " + " ".join(column_names))
        np.savetxt(
            output_path,
            np.column_stack(column_arrays),
            header="\n".join(header_lines),
            comments="",
        )
        self._append_log(f"Exported active traces to {output_path}")
        self.statusBar().showMessage(f"Saved active traces to {output_path}")

    @staticmethod
    def _default_marker_label(pair_label: str, r_value: float) -> str:
        return f"{pair_label}: {float(r_value):.2f} A"

    def _marker_display_text(
        self,
        pair_label: str,
        marker: DebyerPeakMarker,
    ) -> str:
        current_value_text = f"{float(marker.r_value):.2f} A"
        label_text = str(marker.label).strip()
        if not label_text:
            return current_value_text
        if label_text == self._default_marker_label(
            pair_label, marker.r_value
        ):
            return label_text
        if current_value_text in label_text:
            return label_text
        return f"{label_text}\n({current_value_text})"

    def _marker_after_reposition(
        self,
        *,
        pair_label: str,
        marker: DebyerPeakMarker,
        new_r_value: float,
        text_dx: float,
        text_dy: float,
        source: str = "manual",
    ) -> DebyerPeakMarker:
        previous_default_label = self._default_marker_label(
            pair_label,
            marker.r_value,
        )
        label_text = marker.label
        if marker.source == "auto" or label_text == previous_default_label:
            label_text = self._default_marker_label(pair_label, new_r_value)
        return DebyerPeakMarker(
            r_value=float(new_r_value),
            label=str(label_text),
            enabled=marker.enabled,
            text_dx=float(text_dx),
            text_dy=float(text_dy),
            source=str(source),
        )

    def _start_calculation(self) -> None:
        try:
            settings = self._build_settings()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid Debyer settings",
                str(exc),
            )
            return
        settings.project_dir.mkdir(parents=True, exist_ok=True)
        self.calculate_button.setEnabled(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: starting")
        self.time_estimate_label.setText(
            "Estimated time remaining: collecting initial timing samples..."
        )
        self.console.clear()
        self._latest_run_preview = None

        self._run_thread = QThread(self)
        self._run_worker = DebyerPDFWorker(
            settings,
            preview_enabled=self.update_plot_during_run_checkbox.isChecked(),
        )
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.log.connect(self._append_log)
        self._run_worker.progress.connect(self._on_progress)
        self._run_worker.status.connect(self._on_status)
        self._run_worker.preview.connect(self._on_preview_update)
        self._run_worker.finished.connect(self._on_finished)
        self._run_worker.failed.connect(self._on_failed)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_thread.finished.connect(self._cleanup_run_thread)
        self._run_thread.finished.connect(self._run_thread.deleteLater)
        self._run_thread.start()

    def _request_run_cancel(self, message: str) -> None:
        self._close_requested_during_run = True
        self.calculate_button.setEnabled(False)
        self.progress_label.setText("Progress: stopping active Debyer jobs")
        self.time_estimate_label.setText(
            "Estimated time remaining: stopping active Debyer jobs..."
        )
        self.statusBar().showMessage(message, 5000)
        self._append_log(message)
        if self._run_worker is not None:
            self._run_worker.request_cancel()

    def _append_log(self, message: str) -> None:
        self.console.append(message)

    def _on_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(processed)
        self.progress_label.setText(f"Progress: {processed}/{total} frames")
        self.time_estimate_label.setText(message)

    def _on_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _on_update_plot_during_run_toggled(self, checked: bool) -> None:
        if self._run_worker is not None:
            self._run_worker.set_preview_enabled(checked)
        if checked and self._latest_run_preview is not None:
            self._apply_loaded_calculation(self._latest_run_preview)
            self.statusBar().showMessage(
                "Plot updates resumed; showing the latest available average.",
                3000,
            )
        elif (
            checked
            and self._run_thread is not None
            and self._run_thread.isRunning()
        ):
            self.statusBar().showMessage(
                "Plot updates resumed; the next completed frame will refresh "
                "the average.",
                3000,
            )
        elif self._run_thread is not None and self._run_thread.isRunning():
            self.statusBar().showMessage(
                "Plot updates paused; averaging will continue.",
                3000,
            )

    def _on_preview_update(self, result: object) -> None:
        calculation = result
        if not isinstance(calculation, DebyerPDFCalculation):
            return
        self._latest_run_preview = calculation
        if not self.update_plot_during_run_checkbox.isChecked():
            return
        self._apply_loaded_calculation(calculation)

    def _on_finished(self, result: object) -> None:
        self.calculate_button.setEnabled(True)
        calculation = result
        if not isinstance(calculation, DebyerPDFCalculation):
            QMessageBox.warning(
                self,
                "Unexpected Debyer result",
                "The Debyer worker finished without returning a valid calculation.",
            )
            return
        if calculation.is_partial_average:
            processed_frames = (
                calculation.frame_count
                if calculation.processed_frame_count is None
                else int(calculation.processed_frame_count)
            )
            self._append_log(
                "Debyer calculation stopped early; saved running average "
                f"after {processed_frames}/{calculation.frame_count} frames."
            )
            self.time_estimate_label.setText(
                "Estimated time remaining: stopped early"
            )
        else:
            self._append_log("Debyer calculation completed successfully.")
            self.time_estimate_label.setText("Estimated time remaining: 00:00")
        self._refresh_saved_calculations()
        self._apply_loaded_calculation(calculation)
        self._latest_run_preview = None

    def _on_failed(self, message: str) -> None:
        self.calculate_button.setEnabled(True)
        self._append_log(f"Debyer failed: {message}")
        self.time_estimate_label.setText(
            "Estimated time remaining: unavailable"
        )
        self._latest_run_preview = None
        if self._close_requested_during_run:
            return
        QMessageBox.warning(self, "Debyer calculation failed", message)

    def _cleanup_run_thread(self) -> None:
        if self._run_worker is not None:
            self._run_worker.deleteLater()
            self._run_worker = None
        self._run_thread = None
        self._close_requested_during_run = False

    def _coordination_fit_trace_candidates(self) -> list[dict[str, object]]:
        if self._current_calculation is None:
            return []
        return [
            trace
            for trace in build_display_traces(
                self._current_calculation,
                representation="R(r)",
                include_grouped_partials=True,
            )
            if str(trace.get("kind", "")) in {"average", "partial", "group"}
        ]

    def _refresh_coordination_fit_trace_combo(self) -> None:
        if not hasattr(self, "coordination_fit_trace_combo"):
            return
        previous_key = self.coordination_fit_trace_combo.currentData()
        traces = self._coordination_fit_trace_candidates()
        self.coordination_fit_trace_combo.blockSignals(True)
        self.coordination_fit_trace_combo.clear()
        selected_index = 0
        for index, trace in enumerate(traces):
            key = str(trace["key"])
            if previous_key == key:
                selected_index = index
            self.coordination_fit_trace_combo.addItem(
                str(trace["label"]),
                key,
            )
        if traces:
            self.coordination_fit_trace_combo.setCurrentIndex(selected_index)
            self.coordination_fit_button.setEnabled(True)
        else:
            self.coordination_fit_button.setEnabled(False)
            self.coordination_fit_status_label.setText(
                "Load or calculate a Debyer result before fitting R(r)."
            )
        self.coordination_fit_trace_combo.blockSignals(False)
        if traces:
            self._suggest_coordination_fit_window()

    def _selected_coordination_fit_trace(
        self,
    ) -> dict[str, object] | None:
        selected_key = self.coordination_fit_trace_combo.currentData()
        if not selected_key:
            return None
        for trace in self._coordination_fit_trace_candidates():
            if str(trace["key"]) == str(selected_key):
                return trace
        return None

    def _suggest_coordination_fit_window(self, *_args) -> None:
        trace = self._selected_coordination_fit_trace()
        if trace is None:
            return
        r_values = self._trace_x_values(trace)
        values = np.asarray(trace["values"], dtype=float)
        valid = np.isfinite(r_values) & np.isfinite(values)
        if valid.sum() < 5:
            return
        fit_r = r_values[valid]
        fit_values = values[valid]
        order = np.argsort(fit_r)
        fit_r = fit_r[order]
        fit_values = fit_values[order]
        peak_index = int(np.nanargmax(fit_values))
        peak_r = float(fit_r[peak_index])
        span = max(float(fit_r[-1] - fit_r[0]), 1.0e-6)
        half_width = max(0.35, span * 0.08)
        r_min = max(float(fit_r[0]), peak_r - half_width)
        r_max = min(float(fit_r[-1]), peak_r + half_width)
        if r_max <= r_min:
            r_min = float(fit_r[0])
            r_max = float(fit_r[-1])
        self.coordination_fit_r_min_spin.setValue(r_min)
        self.coordination_fit_r_max_spin.setValue(r_max)
        self.coordination_fit_center_spin.setValue(peak_r)
        self.coordination_fit_sigma_spin.setValue(
            max((r_max - r_min) / 6.0, 0.0001)
        )

    def _fit_coordination_number(self) -> None:
        trace = self._selected_coordination_fit_trace()
        if self._current_calculation is None or trace is None:
            self.coordination_fit_status_label.setText(
                "Load or calculate a Debyer result before fitting R(r)."
            )
            return
        try:
            result = fit_coordination_peak_from_r(
                r_values=self._trace_x_values(trace),
                r_distribution_values=np.asarray(trace["values"], dtype=float),
                r_min=float(self.coordination_fit_r_min_spin.value()),
                r_max=float(self.coordination_fit_r_max_spin.value()),
                initial_center=float(
                    self.coordination_fit_center_spin.value()
                ),
                initial_sigma=float(self.coordination_fit_sigma_spin.value()),
            )
        except Exception as exc:
            self.coordination_fit_status_label.setText(str(exc))
            return

        trace_key = str(trace["key"])
        trace_label = str(trace["label"])
        self.representation_combo.setCurrentText("R(r)")
        self._trace_visibility[trace_key] = True
        self._refresh_trace_table()
        self._refresh_plot()
        self._append_coordination_fit_result(trace_label, result)
        self.coordination_fit_status_label.setText(
            f"{trace_label}: CN = {result.coordination_number:.4g}; "
            f"center = {result.center:.4g} A; sigma = {result.sigma:.4g} A"
        )

    def _append_coordination_fit_result(
        self,
        trace_label: str,
        result,
    ) -> None:
        row = self.coordination_fit_results_table.rowCount()
        self.coordination_fit_results_table.insertRow(row)
        values = [
            trace_label,
            f"{result.r_min:.5g}",
            f"{result.r_max:.5g}",
            f"{result.center:.5g}",
            f"{result.sigma:.5g}",
            f"{result.coordination_number:.6g}",
            f"{result.amplitude:.6g}",
            (
                "nan"
                if not np.isfinite(result.r_squared)
                else f"{result.r_squared:.6g}"
            ),
            f"{result.rmse:.6g}",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.coordination_fit_results_table.setItem(row, column, item)

    def _rebuild_traces_and_plot(self) -> None:
        if self._current_calculation is None:
            self._current_traces = []
            self._refresh_trace_table()
            self._refresh_coordination_fit_trace_combo()
            self._refresh_plot()
            return
        previous_visibility = dict(self._trace_visibility)
        previous_tag_visibility = dict(self._trace_tag_visibility)
        previous_colors = dict(self._trace_colors)
        self._current_traces = build_display_traces(
            self._current_calculation,
            representation=self.representation_combo.currentText(),
            include_grouped_partials=True,
        )
        if self._experimental_summary is not None:
            experimental_r = np.asarray(
                self._experimental_summary.q_values,
                dtype=float,
            )
            experimental_values = convert_distribution_values(
                self._experimental_summary.intensities,
                r_values=experimental_r,
                rho0=self._current_calculation.rho0,
                source_mode="PDF",
                target_representation=self.representation_combo.currentText(),
                is_component=False,
            )
            self._current_traces.append(
                {
                    "key": "experimental",
                    "label": (
                        "Experimental g(r) "
                        f"({self._experimental_summary.path.name})"
                    ),
                    "kind": "experimental",
                    "x_values": experimental_r,
                    "values": experimental_values,
                }
            )
        for trace in self._current_traces:
            key = str(trace["key"])
            kind = str(trace["kind"])
            if key in previous_visibility:
                self._trace_visibility[key] = bool(previous_visibility[key])
            elif kind == "average":
                self._trace_visibility[key] = True
            elif kind == "experimental":
                self._trace_visibility[key] = True
            else:
                self._trace_visibility[key] = False
            if key in previous_tag_visibility:
                self._trace_tag_visibility[key] = bool(
                    previous_tag_visibility[key]
                )
            elif kind == "average":
                self._trace_tag_visibility[key] = True
            elif kind == "partial":
                self._trace_tag_visibility[key] = False
            else:
                self._trace_tag_visibility[key] = False
            if key in previous_colors:
                self._trace_colors[key] = str(previous_colors[key])
        self._apply_color_scheme(preserve_existing=True)
        self._refresh_coordination_fit_trace_combo()

    def _apply_color_scheme(
        self, *_args, preserve_existing: bool = False
    ) -> None:
        if not self._current_traces:
            self._refresh_trace_table()
            self._refresh_plot()
            return
        scheme_name = self.color_scheme_combo.currentText() or "tab20"
        scheme = colormaps[scheme_name]
        colored_traces = [
            trace
            for trace in self._current_traces
            if str(trace["kind"]) not in {"average", "experimental"}
        ]
        count = max(len(colored_traces), 1)
        for index, trace in enumerate(colored_traces):
            key = str(trace["key"])
            if preserve_existing and key in self._trace_colors:
                continue
            if key in _GROUP_TRACE_DEFAULT_COLORS:
                self._trace_colors[key] = _GROUP_TRACE_DEFAULT_COLORS[key]
                continue
            rgba = scheme(index / max(count - 1, 1))
            color = QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3]).name()
            self._trace_colors[key] = color
        self._trace_colors["average"] = "#000000"
        self._trace_colors.setdefault("experimental", "#d62728")
        self._refresh_trace_table()
        self._refresh_plot()

    def _toggle_average_trace(self) -> None:
        keys = [
            str(trace["key"])
            for trace in self._current_traces
            if str(trace["kind"]) == "average"
        ]
        self._toggle_trace_keys(keys)

    def _toggle_partial_traces(self) -> None:
        keys = [
            str(trace["key"])
            for trace in self._current_traces
            if str(trace["kind"]) == "partial"
        ]
        self._toggle_trace_keys(keys)

    def _toggle_group_traces(self) -> None:
        keys = [
            str(trace["key"])
            for trace in self._current_traces
            if str(trace["kind"]) == "group"
        ]
        self._toggle_trace_keys(keys)

    def _toggle_experimental_trace(self) -> None:
        keys = [
            str(trace["key"])
            for trace in self._current_traces
            if str(trace["kind"]) == "experimental"
        ]
        self._toggle_trace_keys(keys)

    def _toggle_trace_keys(self, keys: list[str]) -> None:
        if not keys:
            return
        any_visible = any(
            self._trace_visibility.get(key, False) for key in keys
        )
        target = not any_visible
        for key in keys:
            self._trace_visibility[key] = target
        self._refresh_trace_table()
        self._refresh_plot()

    def _refresh_plot(self, *_args) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        self._tag_artist_records = []
        if self._current_calculation is None or not self._current_traces:
            axis.text(
                0.5,
                0.5,
                "Load or calculate a Debyer PDF result to plot it here.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            return

        plotted = []
        for trace in self._current_traces:
            key = str(trace["key"])
            if not self._trace_visibility.get(key, False):
                continue
            kind = str(trace["kind"])
            line = axis.plot(
                self._trace_x_values(trace),
                np.asarray(trace["values"], dtype=float),
                color=self._trace_colors.get(key, "#000000"),
                linewidth=2.0 if kind == "average" else 1.4,
                alpha=1.0 if kind == "average" else 0.9,
                linestyle="--" if kind == "experimental" else "-",
                label=str(trace["label"]),
            )[0]
            plotted.append(line)

        axis_label_kwargs = {
            "fontsize": float(self.axis_label_size_spin.value()),
            "fontweight": (
                "bold"
                if self.axis_label_bold_checkbox.isChecked()
                else "normal"
            ),
            "fontfamily": self.axis_font_family_combo.currentFont().family()
            or "DejaVu Sans",
        }
        axis.set_xlabel("r (A)", **axis_label_kwargs)
        axis.set_ylabel(
            self.representation_combo.currentText(), **axis_label_kwargs
        )
        axis.set_title("Debyer PDF / partial-PDF average")
        axis.tick_params(
            labelsize=max(float(self.axis_label_size_spin.value()) - 1.0, 1.0)
        )
        self._draw_peak_tags(axis)
        self._draw_fit_metrics_box(axis)
        if plotted and self.legend_checkbox.isChecked():
            axis.legend(loc="best", fontsize="small")
        elif not plotted:
            axis.text(
                0.5,
                0.5,
                "All traces are currently hidden.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.grid(True, alpha=0.25)
        self.figure.tight_layout()
        self._update_toggle_button_labels()
        self.canvas.draw_idle()

    def _experimental_fit_metrics_text(self) -> str | None:
        if self._current_calculation is None:
            return None
        if self._experimental_summary is None:
            return None
        model_g = convert_distribution_values(
            self._current_calculation.total_values,
            r_values=self._current_calculation.r_values,
            rho0=self._current_calculation.rho0,
            source_mode=self._current_calculation.mode,
            target_representation="g(r)",
            is_component=False,
        )
        metrics = compute_experimental_fit_metrics(
            model_r_values=self._current_calculation.r_values,
            model_g_values=model_g,
            experimental_r_values=self._experimental_summary.q_values,
            experimental_g_values=self._experimental_summary.intensities,
        )
        if metrics is None:
            return "Fit unavailable\nNo overlapping r range"
        r_squared_text = (
            "nan"
            if not np.isfinite(metrics.r_squared)
            else f"{metrics.r_squared:.4f}"
        )
        return (
            "AIMD g(r) vs experimental\n"
            f"R^2 = {r_squared_text}\n"
            f"RMSE = {metrics.rmse:.4g}\n"
            f"MAE = {metrics.mae:.4g}\n"
            f"n = {metrics.point_count}"
        )

    def _draw_fit_metrics_box(self, axis) -> None:
        if not hasattr(self, "fit_box_checkbox"):
            return
        if not self.fit_box_checkbox.isChecked():
            return
        metrics_text = self._experimental_fit_metrics_text()
        if metrics_text is None:
            return
        axis.text(
            0.02,
            0.04,
            metrics_text,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize="small",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#ffffff",
                "edgecolor": "#666666",
                "alpha": 0.88,
            },
            zorder=10,
        )

    def _draw_peak_tags(self, axis) -> None:
        if self._current_calculation is None:
            return
        radial = np.asarray(self._current_calculation.r_values, dtype=float)
        if radial.size == 0:
            return
        x_span = max(float(radial[-1] - radial[0]), 1.0e-6)
        trace_value_map = {
            str(trace["key"]): np.asarray(trace["values"], dtype=float)
            for trace in self._current_traces
        }
        average_values = trace_value_map.get("average")
        if average_values is None or average_values.size == 0:
            return
        y_span = max(
            float(np.nanmax(average_values) - np.nanmin(average_values)),
            1.0e-6,
        )
        for trace in self._current_traces:
            source_trace_key = str(trace["key"])
            if not self._trace_tag_visibility.get(source_trace_key, False):
                continue
            if not source_trace_key.startswith("partial:"):
                continue
            pair_label = source_trace_key.split(":", 1)[1]
            target_trace_keys = self._enabled_target_trace_keys_for_pair(
                pair_label,
                require_visible=True,
            )
            if not target_trace_keys:
                continue
            color = self._trace_colors.get(source_trace_key, "#000000")
            for target_trace_key in target_trace_keys:
                target_values = trace_value_map.get(target_trace_key)
                if target_values is None or target_values.size == 0:
                    continue
                for marker_index, marker in enumerate(
                    self._target_peak_markers_for_trace(
                        pair_label=pair_label,
                        target_trace_key=target_trace_key,
                    )
                ):
                    if not marker.enabled:
                        continue
                    peak_x = float(marker.r_value)
                    peak_y = float(np.interp(peak_x, radial, target_values))
                    default_dx = max(x_span * 0.015, 0.05)
                    default_dy = max(y_span * 0.04, 0.02)
                    text_x = peak_x + (
                        float(marker.text_dx)
                        if abs(float(marker.text_dx)) > 1.0e-12
                        else default_dx
                    )
                    text_y = peak_y + (
                        float(marker.text_dy)
                        if abs(float(marker.text_dy)) > 1.0e-12
                        else default_dy
                    )
                    is_selected = (
                        self._selected_tag is not None
                        and str(self._selected_tag.get("pair_label", ""))
                        == pair_label
                        and int(self._selected_tag.get("marker_index", -1))
                        == marker_index
                        and str(
                            self._selected_tag.get(
                                "target_trace_key",
                                "average",
                            )
                        )
                        == target_trace_key
                    )
                    marker_artist = axis.plot(
                        [peak_x],
                        [peak_y],
                        marker="o",
                        markersize=6.5 if is_selected else 4.5,
                        color=color,
                        linestyle="None",
                        zorder=5,
                    )[0]
                    annotation = axis.annotate(
                        self._marker_display_text(pair_label, marker),
                        xy=(peak_x, peak_y),
                        xytext=(text_x, text_y),
                        textcoords="data",
                        xycoords="data",
                        fontsize=float(self.tag_font_size_spin.value()),
                        fontweight=(
                            "bold"
                            if self.tag_bold_checkbox.isChecked()
                            else "normal"
                        ),
                        fontfamily=self.tag_font_family_combo.currentFont().family()
                        or "DejaVu Sans",
                        color=color,
                        arrowprops={
                            "arrowstyle": "-",
                            "linewidth": float(
                                self.tag_line_width_spin.value()
                            ),
                            "linestyle": self._tag_arrow_linestyle(),
                            "color": color,
                            "shrinkA": 0.0,
                            "shrinkB": 0.0,
                        },
                        zorder=6,
                        bbox=(
                            {
                                "boxstyle": "round,pad=0.18",
                                "fc": "#ffffff",
                                "ec": color,
                                "lw": 0.9,
                                "alpha": 0.85,
                            }
                            if is_selected
                            else None
                        ),
                    )
                    self._tag_artist_records.append(
                        {
                            "trace_key": source_trace_key,
                            "target_trace_key": target_trace_key,
                            "pair_label": pair_label,
                            "marker_index": marker_index,
                            "marker_artist": marker_artist,
                            "annotation": annotation,
                        }
                    )

    def _on_plot_press(self, event) -> None:
        if event.button != 1 or event.inaxes is None:
            return
        for record in reversed(self._tag_artist_records):
            marker_artist = record["marker_artist"]
            annotation = record["annotation"]
            marker_hit, _marker_data = marker_artist.contains(event)
            if marker_hit:
                self._set_selected_tag(
                    str(record["pair_label"]),
                    int(record["marker_index"]),
                    target_trace_key=str(record["target_trace_key"]),
                )
                self._drag_state = {
                    "mode": "marker",
                    "pair_label": record["pair_label"],
                    "marker_index": record["marker_index"],
                    "target_trace_key": record["target_trace_key"],
                }
                return
            annotation_hit, _annotation_data = annotation.contains(event)
            if annotation_hit:
                self._drag_state = None
                self._set_selected_tag(
                    str(record["pair_label"]),
                    int(record["marker_index"]),
                    target_trace_key=str(record["target_trace_key"]),
                )
                return
        self._drag_state = None
        self._set_selected_tag(None)

    def _on_plot_motion(self, event) -> None:
        if self._drag_state is None or self._current_calculation is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        pair_label = str(self._drag_state["pair_label"])
        marker_index = int(self._drag_state["marker_index"])
        target_trace_key = str(
            self._drag_state.get("target_trace_key", "average")
        )
        markers = list(
            self._target_peak_markers_for_trace(
                pair_label=pair_label,
                target_trace_key=target_trace_key,
            )
        )
        if not (0 <= marker_index < len(markers)):
            return
        marker = markers[marker_index]
        radial = np.asarray(self._current_calculation.r_values, dtype=float)
        nearest_index = int(np.argmin(np.abs(radial - float(event.xdata))))
        new_r = float(radial[nearest_index])
        markers[marker_index] = self._marker_after_reposition(
            pair_label=pair_label,
            marker=marker,
            new_r_value=new_r,
            text_dx=marker.text_dx,
            text_dy=marker.text_dy,
            source="manual",
        )
        if target_trace_key == "average":
            self._snapshot_other_target_tags(
                pair_label=pair_label,
                exclude_target_trace_key=target_trace_key,
            )
            self._update_partial_peak_markers(
                pair_label,
                tuple(markers),
                persist=False,
                refresh_table=False,
                sort_markers=False,
            )
            return
        self._update_target_peak_markers(
            target_trace_key=target_trace_key,
            pair_label=pair_label,
            markers=tuple(markers),
            persist=False,
            refresh_table=False,
            sort_markers=False,
        )

    def _on_plot_release(self, _event) -> None:
        if self._drag_state is None:
            return
        pair_label = str(self._drag_state["pair_label"])
        target_trace_key = str(
            self._drag_state.get("target_trace_key", "average")
        )
        self._drag_state = None
        if self._current_calculation is None:
            return
        markers = self._target_peak_markers_for_trace(
            pair_label=pair_label,
            target_trace_key=target_trace_key,
        )
        if target_trace_key == "average":
            self._update_partial_peak_markers(
                pair_label,
                markers,
                persist=True,
                refresh_table=True,
                sort_markers=True,
            )
            return
        self._update_target_peak_markers(
            target_trace_key=target_trace_key,
            pair_label=pair_label,
            markers=markers,
            persist=True,
            refresh_table=True,
            sort_markers=True,
        )

    def _refresh_trace_table(self) -> None:
        self.trace_table.setRowCount(0)
        for row_index, trace in enumerate(self._current_traces):
            self.trace_table.insertRow(row_index)
            key = str(trace["key"])
            kind = str(trace["kind"])

            visible_box = QCheckBox()
            visible_box.setChecked(
                bool(self._trace_visibility.get(key, False))
            )
            visible_box.toggled.connect(
                lambda checked, trace_key=key: self._set_trace_visible(
                    trace_key,
                    checked,
                )
            )
            self.trace_table.setCellWidget(row_index, 0, visible_box)

            tag_box = QCheckBox()
            is_partial = kind == "partial"
            is_tag_target = kind in {"average", "group"}
            tag_box.setEnabled(is_partial or is_tag_target)
            tag_box.setChecked(
                bool(self._trace_tag_visibility.get(key, False))
                if (is_partial or is_tag_target)
                else False
            )
            tag_box.toggled.connect(
                lambda checked, trace_key=key: self._set_trace_tag_visible(
                    trace_key,
                    checked,
                )
            )
            self.trace_table.setCellWidget(row_index, 1, tag_box)

            label_item = QTableWidgetItem(str(trace["label"]))
            label_item.setFlags(
                label_item.flags() ^ Qt.ItemFlag.ItemIsEditable
            )
            self.trace_table.setItem(row_index, 2, label_item)

            kind_item = QTableWidgetItem(kind.title())
            kind_item.setFlags(kind_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.trace_table.setItem(row_index, 3, kind_item)

            peaks_item = QTableWidgetItem(self._peak_summary_text(key))
            peaks_item.setFlags(
                peaks_item.flags() ^ Qt.ItemFlag.ItemIsEditable
            )
            self.trace_table.setItem(row_index, 4, peaks_item)

            edit_button = QPushButton("Edit")
            edit_button.setEnabled(is_partial)
            if is_partial:
                edit_button.clicked.connect(
                    lambda _checked=False, trace_key=key: self._edit_peak_markers(
                        trace_key
                    )
                )
            self.trace_table.setCellWidget(row_index, 5, edit_button)

            reset_button = QPushButton("Reset")
            reset_button.setEnabled(is_partial)
            if is_partial:
                reset_button.clicked.connect(
                    lambda _checked=False, trace_key=key: self._reset_partial_peak_markers(
                        trace_key
                    )
                )
            self.trace_table.setCellWidget(row_index, 6, reset_button)

            color_button = QPushButton()
            color_button.clicked.connect(
                lambda _checked=False, trace_key=key: self._choose_trace_color(
                    trace_key
                )
            )
            self._configure_color_button(
                color_button,
                self._trace_colors.get(key, "#000000"),
            )
            self.trace_table.setCellWidget(row_index, 7, color_button)

        self._update_toggle_button_labels()

    def _configure_color_button(
        self,
        button: QPushButton,
        color: str,
    ) -> None:
        qcolor = QColor(color)
        text_color = "#000000" if qcolor.lightnessF() > 0.55 else "#ffffff"
        button.setText(color)
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: {color}; color: {text_color};"
            "padding: 3px 8px;"
            "}"
        )

    def _choose_trace_color(self, trace_key: str) -> None:
        current = self._trace_colors.get(trace_key, "#000000")
        chosen = QColorDialog.getColor(
            QColor(current),
            self,
            "Choose trace color",
        )
        if not chosen.isValid():
            return
        self._trace_colors[trace_key] = chosen.name()
        self._refresh_trace_table()
        self._refresh_plot()

    def _set_trace_visible(self, trace_key: str, visible: bool) -> None:
        self._trace_visibility[trace_key] = bool(visible)
        self._update_toggle_button_labels()
        self._refresh_plot()

    def _update_toggle_button_labels(self) -> None:
        average_visible = any(
            self._trace_visibility.get(str(trace["key"]), False)
            for trace in self._current_traces
            if str(trace["kind"]) == "average"
        )
        partial_visible = any(
            self._trace_visibility.get(str(trace["key"]), False)
            for trace in self._current_traces
            if str(trace["kind"]) == "partial"
        )
        group_keys = [
            str(trace["key"])
            for trace in self._current_traces
            if str(trace["kind"]) == "group"
        ]
        group_visible = any(
            self._trace_visibility.get(key, False) for key in group_keys
        )
        experimental_keys = [
            str(trace["key"])
            for trace in self._current_traces
            if str(trace["kind"]) == "experimental"
        ]
        experimental_visible = any(
            self._trace_visibility.get(key, False) for key in experimental_keys
        )
        self.average_toggle_button.setText(
            "Hide Average" if average_visible else "Show Average"
        )
        self.partials_toggle_button.setText(
            "Hide Partial PDFs" if partial_visible else "Show Partial PDFs"
        )
        self.groups_toggle_button.setEnabled(bool(group_keys))
        self.groups_toggle_button.setText(
            "Hide Grouped Partials"
            if group_visible
            else "Show Grouped Partials"
        )
        if hasattr(self, "experimental_toggle_button"):
            self.experimental_toggle_button.setEnabled(bool(experimental_keys))
            self.experimental_toggle_button.setText(
                "Hide Experimental"
                if experimental_visible
                else "Show Experimental"
            )

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "unknown"
        rounded = max(int(round(float(seconds))), 0)
        hours, remainder = divmod(rounded, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _update_rho0_label(self) -> None:
        try:
            atom_count = int(float(self.atom_count_edit.text().strip()))
            box = self._parse_box_dimensions()
            volume = float(np.prod(np.asarray(box, dtype=float)))
            rho0 = atom_count / volume
        except Exception:
            self.rho0_label.setText(
                "rho0 will be computed from the atom count and box."
            )
            return
        self.rho0_label.setText(
            f"rho0 = {rho0:.6g} atoms/A^3 (volume {volume:.6g} A^3)"
        )


def launch_debyer_pdf_ui(
    project_dir: str | Path | None = None,
    *,
    frames_dir: str | Path | None = None,
) -> int:
    prepare_saxshell_application_identity()
    app = QApplication.instance()
    should_exec = app is None
    if app is None:
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = DebyerPDFMainWindow(
        initial_project_dir=project_dir,
        initial_frames_dir=frames_dir,
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda _obj=None: _OPEN_WINDOWS.remove(window))
    if not should_exec:
        return 0
    return app.exec()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfsetup",
        description=(
            "Launch the Debyer-backed PDF / partial-PDF averaging UI."
        ),
    )
    parser.add_argument(
        "project_dir",
        nargs="?",
        help="Optional SAXSShell project directory to prefill in the UI.",
    )
    parser.add_argument(
        "--frames-dir",
        help="Optional extracted trajectory frames directory to prefill.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return launch_debyer_pdf_ui(
        getattr(args, "project_dir", None),
        frames_dir=getattr(args, "frames_dir", None),
    )


__all__ = [
    "DebyerPDFMainWindow",
    "launch_debyer_pdf_ui",
    "main",
]
