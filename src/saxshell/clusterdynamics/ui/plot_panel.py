from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import replace
from html import escape as html_escape

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette, QTextDocument
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from saxshell.clusterdynamics.workflow import (
    ClusterDynamicsResult,
    _summarize_series_lifetimes,
)
from saxshell.plotting.igor_inline import (
    apply_igor_inline_text_artist,
    igor_inline_to_mathtext,
    prepare_igor_inline_segments,
)
from saxshell.plotting.plot_editor import (
    HeatmapPlotDefaults,
    HeatmapPlotEditorControls,
    HeatmapPlotSettings,
    LifetimeDistributionPlotDefaults,
    LifetimeDistributionPlotSettings,
    LifetimeHistogramPlotDefaults,
    LifetimeHistogramPlotEditorControls,
    LifetimeHistogramPlotSettings,
    PlotEditorWindow,
)
from saxshell.saxs.stoichiometry import format_stoich_for_axis

PLOT_COLORMAPS = ("viridis", "magma", "cividis", "inferno", "turbo")
DISPLAY_MODE_LABELS = {
    "count": "Counts / bin",
    "fraction": "Fraction / bin",
    "mean_count": "Mean count / frame",
}
DISPLAY_MODE_COLORBAR_LABELS = {
    "count": "Clusters in bin",
    "fraction": "Cluster fraction",
    "mean_count": "Mean clusters per frame",
}
OVERLAY_SERIES = (
    ("None", None),
    ("Temperature", "temperature"),
    ("Potential Energy", "potential"),
    ("Kinetic Energy", "kinetic"),
)
OVERLAY_COLORS = {
    "temperature": "#1f77b4",
    "potential": "#2e8b57",
    "kinetic": "#c0392b",
}
LIFETIME_SAMPLE_MODES = (
    ("Completed lifetimes", "completed"),
    ("All observed lifetimes", "all"),
)
LIFETIME_UNITS = (
    ("ps", "ps", 1.0 / 1000.0),
    ("fs", "fs", 1.0),
)
_STOICHIOMETRY_SCRIPT_PATTERN = re.compile(
    r"\$([_^])\{([^{}]+)\}\$|\$([_^])([^$])\$"
)


def _stoichiometry_mathtext_to_html(label: str) -> str:
    html_parts: list[str] = []
    position = 0
    for match in _STOICHIOMETRY_SCRIPT_PATTERN.finditer(label):
        html_parts.append(html_escape(label[position : match.start()]))
        marker = match.group(1) or match.group(3)
        script = (
            match.group(2) if match.group(2) is not None else match.group(4)
        )
        tag = "sub" if marker == "_" else "sup"
        html_parts.append(f"<{tag}>{html_escape(script)}</{tag}>")
        position = match.end()
    html_parts.append(html_escape(label[position:]))
    return "".join(html_parts)


class _RichStoichiometryDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        if "$" not in text:
            super().paint(painter, option, index)
            return

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        opt.text = ""
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget
        )

        color_role = (
            QPalette.ColorRole.HighlightedText
            if opt.state & QStyle.StateFlag.State_Selected
            else QPalette.ColorRole.Text
        )
        text_color = opt.palette.color(color_role).name()
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(opt.font)
        doc.setHtml(
            f'<span style="color:{text_color}">'
            f"{_stoichiometry_mathtext_to_html(text)}</span>"
        )
        rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            opt,
            widget,
        )
        doc.setTextWidth(rect.width())
        vertical_offset = max(0.0, (rect.height() - doc.size().height()) / 2.0)
        painter.save()
        try:
            painter.translate(rect.left(), rect.top() + vertical_offset)
            doc.drawContents(painter)
        finally:
            painter.restore()


class _XAxisOrderDialog(QDialog):
    def __init__(
        self,
        entries: list[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Custom X-Axis Order")
        self.resize(520, 400)
        self._build_ui(entries)

    def _build_ui(self, entries: list[tuple[str, str]]) -> None:
        layout = QVBoxLayout(self)
        note = QLabel(
            "Rearrange rows to set the x-axis order. Edit Display Text to "
            "customise axis labels. Use $_{n}$ for subscript and $^{n}$ for "
            "superscript."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        content = QHBoxLayout()
        self._table = QTableWidget(len(entries), 2)
        headers = ["Stoichiometry", "Display Text"]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        for row, (raw, display) in enumerate(entries):
            raw_item = QTableWidgetItem(raw)
            raw_item.setFlags(raw_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, raw_item)
            self._table.setItem(row, 1, QTableWidgetItem(display))
        self._table.resizeColumnToContents(0)
        content.addWidget(self._table, stretch=1)

        buttons = QVBoxLayout()
        up_button = QPushButton("Up")
        up_button.clicked.connect(self._move_up)
        down_button = QPushButton("Down")
        down_button.clicked.connect(self._move_down)
        buttons.addWidget(up_button)
        buttons.addWidget(down_button)
        buttons.addStretch(1)
        content.addLayout(buttons)
        layout.addLayout(content)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _move_up(self) -> None:
        row = self._table.currentRow()
        if row <= 0:
            return
        self._swap_rows(row, row - 1)
        self._table.selectRow(row - 1)

    def _move_down(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= self._table.rowCount() - 1:
            return
        self._swap_rows(row, row + 1)
        self._table.selectRow(row + 1)

    def _swap_rows(self, a: int, b: int) -> None:
        for column in range(self._table.columnCount()):
            item_a = self._table.takeItem(a, column)
            item_b = self._table.takeItem(b, column)
            if item_a is not None:
                self._table.setItem(b, column, item_a)
            if item_b is not None:
                self._table.setItem(a, column, item_b)

    def result_entries(self) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for row in range(self._table.rowCount()):
            raw_item = self._table.item(row, 0)
            display_item = self._table.item(row, 1)
            raw = "" if raw_item is None else raw_item.text()
            display = "" if display_item is None else display_item.text()
            if raw:
                entries.append((raw, display or raw))
        return entries


class ClusterDynamicsPlotPanel(QWidget):
    """Interactive time-binned cluster heatmap panel."""

    save_colormap_requested = Signal()
    save_lifetime_requested = Signal()

    _MIN_PANEL_HEIGHT = 420
    _MIN_CANVAS_HEIGHT = 300

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        enable_plot_editor: bool = False,
    ) -> None:
        super().__init__(parent)
        self._enable_plot_editor = bool(enable_plot_editor)
        self._result: ClusterDynamicsResult | None = None
        self._plot_settings = HeatmapPlotSettings()
        self._plot_editor_window: PlotEditorWindow | None = None
        self._plot_editor_controls: HeatmapPlotEditorControls | None = None
        self._lifetime_distribution_plot_settings = (
            LifetimeDistributionPlotSettings()
        )
        self._lifetime_distribution_plot_editor_window: (
            PlotEditorWindow | None
        ) = None
        self._lifetime_distribution_plot_editor_controls: (
            LifetimeHistogramPlotEditorControls | None
        ) = None
        self._lifetime_histogram_plot_settings = (
            LifetimeHistogramPlotSettings()
        )
        self._lifetime_histogram_plot_editor_window: (
            PlotEditorWindow | None
        ) = None
        self._lifetime_histogram_plot_editor_controls: (
            LifetimeHistogramPlotEditorControls | None
        ) = None
        self._lifetime_x_axis_custom_order: list[tuple[str, str]] = []
        self._selected_lifetime_histogram_label: str | None = None
        self.plot_editor_button: QPushButton | None = None
        self.lifetime_plot_editor_button: QPushButton | None = None
        self.lifetime_histogram_plot_editor_button: QPushButton | None = None
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        self.setMinimumHeight(self._MIN_PANEL_HEIGHT)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        self.heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(self.heatmap_tab)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_layout.setSpacing(8)
        self.lifetime_distribution_tab = QWidget()
        lifetime_layout = QVBoxLayout(self.lifetime_distribution_tab)
        lifetime_layout.setContentsMargins(0, 0, 0, 0)
        lifetime_layout.setSpacing(8)
        self.lifetime_histogram_tab = QWidget()
        lifetime_histogram_layout = QVBoxLayout(self.lifetime_histogram_tab)
        lifetime_histogram_layout.setContentsMargins(0, 0, 0, 0)
        lifetime_histogram_layout.setSpacing(8)

        heatmap_action_row = QHBoxLayout()
        heatmap_action_row.setContentsMargins(0, 0, 0, 0)
        heatmap_action_row.setSpacing(8)
        if self._enable_plot_editor:
            self.plot_editor_button = QPushButton("Open Plot Editor")
            self.plot_editor_button.clicked.connect(self.open_plot_editor)
            heatmap_action_row.addWidget(self.plot_editor_button)
        self.save_colormap_button = QPushButton("Save Colormap Data")
        self.save_colormap_button.setToolTip(
            "Write the currently plotted heatmap data to a CSV file using "
            "the active display mode and time-unit selections."
        )
        self.save_colormap_button.clicked.connect(
            lambda _checked=False: self.save_colormap_requested.emit()
        )
        heatmap_action_row.addWidget(self.save_colormap_button)
        heatmap_action_row.addStretch(1)
        heatmap_layout.addLayout(heatmap_action_row)

        controls_widget = QWidget()
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        controls.addWidget(QLabel("Heatmap"))
        self.display_mode_combo = QComboBox()
        for mode, label in DISPLAY_MODE_LABELS.items():
            self.display_mode_combo.addItem(label, mode)
        self.display_mode_combo.setCurrentIndex(1)
        self.display_mode_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.display_mode_combo)

        controls.addWidget(QLabel("Units"))
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItem("fs", "fs")
        self.time_unit_combo.addItem("ps", "ps")
        self.time_unit_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.time_unit_combo)

        controls.addWidget(QLabel("Colormap"))
        self.colormap_combo = QComboBox()
        for cmap_name in PLOT_COLORMAPS:
            self.colormap_combo.addItem(cmap_name, cmap_name)
        self.colormap_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.colormap_combo)

        controls.addWidget(QLabel("Lower q"))
        self.lower_quantile_spin = QDoubleSpinBox()
        self.lower_quantile_spin.setDecimals(2)
        self.lower_quantile_spin.setRange(0.0, 0.95)
        self.lower_quantile_spin.setSingleStep(0.05)
        self.lower_quantile_spin.setValue(0.05)
        self.lower_quantile_spin.valueChanged.connect(
            self._on_quantile_changed
        )
        controls.addWidget(self.lower_quantile_spin)

        controls.addWidget(QLabel("Upper q"))
        self.upper_quantile_spin = QDoubleSpinBox()
        self.upper_quantile_spin.setDecimals(2)
        self.upper_quantile_spin.setRange(0.05, 1.0)
        self.upper_quantile_spin.setSingleStep(0.05)
        self.upper_quantile_spin.setValue(0.95)
        self.upper_quantile_spin.valueChanged.connect(
            self._on_quantile_changed
        )
        controls.addWidget(self.upper_quantile_spin)

        controls.addWidget(QLabel("Overlay"))
        self.overlay_combo = QComboBox()
        for label, data in OVERLAY_SERIES:
            self.overlay_combo.addItem(label, data)
        self.overlay_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.overlay_combo)
        controls.addStretch(1)

        heatmap_layout.addWidget(controls_widget)

        self.figure = Figure(figsize=(9.2, 7.2))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(self._MIN_CANVAS_HEIGHT)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        heatmap_layout.addWidget(NavigationToolbar(self.canvas, self))
        heatmap_layout.addWidget(self.canvas, stretch=1)

        lifetime_action_row = QHBoxLayout()
        lifetime_action_row.setContentsMargins(0, 0, 0, 0)
        lifetime_action_row.setSpacing(8)
        if self._enable_plot_editor:
            self.lifetime_plot_editor_button = QPushButton("Open Plot Editor")
            self.lifetime_plot_editor_button.clicked.connect(
                self.open_lifetime_distribution_plot_editor
            )
            lifetime_action_row.addWidget(self.lifetime_plot_editor_button)
        self.save_lifetime_button = QPushButton("Save Lifetime Table")
        self.save_lifetime_button.setToolTip(
            "Write the observed lifetime summary table to a CSV file."
        )
        self.save_lifetime_button.clicked.connect(
            lambda _checked=False: self.save_lifetime_requested.emit()
        )
        lifetime_action_row.addWidget(self.save_lifetime_button)
        lifetime_action_row.addStretch(1)
        lifetime_layout.addLayout(lifetime_action_row)

        lifetime_controls_widget = QWidget()
        lifetime_controls = QHBoxLayout(lifetime_controls_widget)
        lifetime_controls.setContentsMargins(0, 0, 0, 0)
        lifetime_controls.setSpacing(8)

        lifetime_controls.addWidget(QLabel("Samples"))
        self.lifetime_sample_combo = QComboBox()
        for label, value in LIFETIME_SAMPLE_MODES:
            self.lifetime_sample_combo.addItem(label, value)
        self.lifetime_sample_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_lifetime_plots()
        )
        lifetime_controls.addWidget(self.lifetime_sample_combo)

        lifetime_controls.addWidget(QLabel("Units"))
        self.lifetime_unit_combo = QComboBox()
        for label, value, _scale in LIFETIME_UNITS:
            self.lifetime_unit_combo.addItem(label, value)
        self.lifetime_unit_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_lifetime_plots()
        )
        lifetime_controls.addWidget(self.lifetime_unit_combo)

        lifetime_controls.addWidget(QLabel("X-axis"))
        self.lifetime_x_axis_order_combo = QComboBox()
        self.lifetime_x_axis_order_combo.addItem("Auto", "auto")
        self.lifetime_x_axis_order_combo.addItem("Custom", "custom")
        self.lifetime_x_axis_order_combo.currentIndexChanged.connect(
            self._on_lifetime_x_axis_order_changed
        )
        lifetime_controls.addWidget(self.lifetime_x_axis_order_combo)
        self.edit_lifetime_x_axis_button = QPushButton("Edit Custom")
        self.edit_lifetime_x_axis_button.setEnabled(False)
        self.edit_lifetime_x_axis_button.clicked.connect(
            self._on_edit_lifetime_x_axis_order
        )
        lifetime_controls.addWidget(self.edit_lifetime_x_axis_button)
        lifetime_controls.addStretch(1)
        lifetime_layout.addWidget(lifetime_controls_widget)

        self.histogram_figure = Figure(figsize=(9.2, 7.2))
        self.histogram_canvas = FigureCanvas(self.histogram_figure)
        self.histogram_canvas.setMinimumHeight(self._MIN_CANVAS_HEIGHT)
        self.histogram_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        lifetime_layout.addWidget(
            NavigationToolbar(self.histogram_canvas, self)
        )
        lifetime_layout.addWidget(self.histogram_canvas, stretch=1)

        lifetime_histogram_controls_widget = QWidget()
        lifetime_histogram_controls = QHBoxLayout(
            lifetime_histogram_controls_widget
        )
        lifetime_histogram_controls.setContentsMargins(0, 0, 0, 0)
        lifetime_histogram_controls.setSpacing(8)
        if self._enable_plot_editor:
            self.lifetime_histogram_plot_editor_button = QPushButton(
                "Open Plot Editor"
            )
            self.lifetime_histogram_plot_editor_button.clicked.connect(
                self.open_lifetime_histogram_plot_editor
            )
            lifetime_histogram_controls.addWidget(
                self.lifetime_histogram_plot_editor_button
            )
        lifetime_histogram_controls.addWidget(QLabel("Bins"))
        self.lifetime_histogram_bins_spin = QSpinBox()
        self.lifetime_histogram_bins_spin.setRange(1, 200)
        self.lifetime_histogram_bins_spin.setValue(20)
        self.lifetime_histogram_bins_spin.valueChanged.connect(
            lambda _value: self.refresh_lifetime_histogram_plot()
        )
        lifetime_histogram_controls.addWidget(
            self.lifetime_histogram_bins_spin
        )
        lifetime_histogram_controls.addStretch(1)
        lifetime_histogram_layout.addWidget(lifetime_histogram_controls_widget)

        lifetime_histogram_body = QWidget()
        lifetime_histogram_body_layout = QHBoxLayout(lifetime_histogram_body)
        lifetime_histogram_body_layout.setContentsMargins(0, 0, 0, 0)
        lifetime_histogram_body_layout.setSpacing(8)

        lifetime_histogram_plot_widget = QWidget()
        lifetime_histogram_plot_layout = QVBoxLayout(
            lifetime_histogram_plot_widget
        )
        lifetime_histogram_plot_layout.setContentsMargins(0, 0, 0, 0)
        lifetime_histogram_plot_layout.setSpacing(8)
        self.lifetime_histogram_figure = Figure(figsize=(9.2, 7.2))
        self.lifetime_histogram_canvas = FigureCanvas(
            self.lifetime_histogram_figure
        )
        self.lifetime_histogram_canvas.setMinimumHeight(
            self._MIN_CANVAS_HEIGHT
        )
        self.lifetime_histogram_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        lifetime_histogram_plot_layout.addWidget(
            NavigationToolbar(self.lifetime_histogram_canvas, self)
        )
        lifetime_histogram_plot_layout.addWidget(
            self.lifetime_histogram_canvas,
            stretch=1,
        )
        lifetime_histogram_body_layout.addWidget(
            lifetime_histogram_plot_widget,
            stretch=3,
        )

        lifetime_histogram_table_widget = QWidget()
        lifetime_histogram_table_layout = QVBoxLayout(
            lifetime_histogram_table_widget
        )
        lifetime_histogram_table_layout.setContentsMargins(0, 0, 0, 0)
        lifetime_histogram_table_layout.setSpacing(6)
        lifetime_histogram_table_layout.addWidget(QLabel("Stoichiometries"))
        self.lifetime_histogram_table = QTableWidget(0, 3)
        self.lifetime_histogram_table.setHorizontalHeaderLabels(
            ["Stoichiometry", "Samples", "Mean"]
        )
        self._lifetime_histogram_stoichiometry_delegate = (
            _RichStoichiometryDelegate(self.lifetime_histogram_table)
        )
        self.lifetime_histogram_table.setItemDelegateForColumn(
            0,
            self._lifetime_histogram_stoichiometry_delegate,
        )
        self.lifetime_histogram_table.horizontalHeader().setStretchLastSection(
            True
        )
        self.lifetime_histogram_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.lifetime_histogram_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.lifetime_histogram_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.lifetime_histogram_table.itemSelectionChanged.connect(
            self._on_lifetime_histogram_selection_changed
        )
        lifetime_histogram_table_layout.addWidget(
            self.lifetime_histogram_table,
            stretch=1,
        )
        lifetime_histogram_body_layout.addWidget(
            lifetime_histogram_table_widget,
            stretch=1,
        )
        lifetime_histogram_layout.addWidget(
            lifetime_histogram_body,
            stretch=1,
        )

        self.tabs.addTab(self.heatmap_tab, "Cluster Heatmap")
        self.tabs.addTab(
            self.lifetime_distribution_tab,
            "Lifetime Distribution",
        )
        self.tabs.addTab(
            self.lifetime_histogram_tab,
            "Lifetime Histogram",
        )
        if self._enable_plot_editor:
            self._build_embedded_plot_editor_tab()
            self.tabs.addTab(self.plot_editor_tab, "Plot Editor")
        root.addWidget(self.tabs, stretch=1)

    def _build_embedded_plot_editor_tab(self) -> None:
        self.plot_editor_tab = QWidget()
        plot_editor_layout = QVBoxLayout(self.plot_editor_tab)
        plot_editor_layout.setContentsMargins(0, 0, 0, 0)
        plot_editor_layout.setSpacing(8)
        self.plot_editor_tabs = QTabWidget()

        defaults = self._current_plot_defaults()
        self._plot_editor_controls = HeatmapPlotEditorControls(
            settings=self._plot_settings,
            defaults=defaults,
            parent=self,
        )
        self._plot_editor_controls.settings_changed.connect(self.refresh_plot)
        self._plot_editor_controls.x_axis_unit_changed.connect(
            self._on_plot_editor_x_axis_unit_changed
        )
        self._plot_editor_controls.colormap_changed.connect(
            self._on_plot_editor_colormap_changed
        )
        self._plot_editor_window = PlotEditorWindow(
            window_title="Cluster Dynamics Colormap Editor",
            controls_widget=self._plot_editor_controls,
            render_preview=self._render_plot_figure,
            pickle_state_provider=self._plot_editor_pickle_state,
            apply_loaded_pickle_state=self._apply_loaded_plot_editor_pickle_state,
            embedded=True,
            parent=self.plot_editor_tabs,
        )
        self.plot_editor_tabs.addTab(
            self._plot_editor_window, "Cluster Heatmap"
        )

        lifetime_distribution_defaults = (
            self._current_lifetime_distribution_plot_defaults()
        )
        self._lifetime_distribution_plot_editor_controls = (
            LifetimeHistogramPlotEditorControls(
                settings=self._lifetime_distribution_plot_settings,
                defaults=lifetime_distribution_defaults,
                plot_description="lifetime distribution",
                fill_color_label="Box Color",
                edge_color_label="Box Edge Color",
                fill_alpha_label="Box Opacity",
                appearance_description="box and point appearance",
                parent=self,
            )
        )
        self._lifetime_distribution_plot_editor_controls.settings_changed.connect(
            self.refresh_lifetime_distribution_plot
        )
        self._lifetime_distribution_plot_editor_window = PlotEditorWindow(
            window_title="Cluster Dynamics Lifetime Distribution Editor",
            controls_widget=self._lifetime_distribution_plot_editor_controls,
            render_preview=self._render_lifetime_distribution_figure,
            pickle_state_provider=(
                self._lifetime_distribution_plot_editor_pickle_state
            ),
            apply_loaded_pickle_state=(
                self._apply_loaded_lifetime_distribution_plot_editor_pickle_state
            ),
            embedded=True,
            parent=self.plot_editor_tabs,
        )
        self.plot_editor_tabs.addTab(
            self._lifetime_distribution_plot_editor_window,
            "Lifetime Distribution",
        )

        lifetime_histogram_defaults = (
            self._current_lifetime_histogram_plot_defaults()
        )
        self._lifetime_histogram_plot_editor_controls = (
            LifetimeHistogramPlotEditorControls(
                settings=self._lifetime_histogram_plot_settings,
                defaults=lifetime_histogram_defaults,
                parent=self,
            )
        )
        self._lifetime_histogram_plot_editor_controls.settings_changed.connect(
            self.refresh_lifetime_histogram_plot
        )
        self._lifetime_histogram_plot_editor_window = PlotEditorWindow(
            window_title="Cluster Dynamics Lifetime Histogram Editor",
            controls_widget=self._lifetime_histogram_plot_editor_controls,
            render_preview=self._render_lifetime_histogram_figure,
            pickle_state_provider=(
                self._lifetime_histogram_plot_editor_pickle_state
            ),
            apply_loaded_pickle_state=(
                self._apply_loaded_lifetime_histogram_plot_editor_pickle_state
            ),
            embedded=True,
            parent=self.plot_editor_tabs,
        )
        self.plot_editor_tabs.addTab(
            self._lifetime_histogram_plot_editor_window,
            "Lifetime Histogram",
        )

        plot_editor_layout.addWidget(self.plot_editor_tabs, stretch=1)

    def set_result(self, result: ClusterDynamicsResult | None) -> None:
        self._result = result
        has_energy = bool(
            result is not None and result.energy_data is not None
        )
        self.overlay_combo.setEnabled(has_energy)
        if not has_energy:
            self.overlay_combo.setCurrentIndex(0)
        self.refresh_plot()

    def open_plot_editor(self) -> None:
        if not self._enable_plot_editor:
            return
        self._show_embedded_plot_editor(self._plot_editor_window)

    def open_lifetime_distribution_plot_editor(self) -> None:
        if not self._enable_plot_editor:
            return
        self._show_embedded_plot_editor(
            self._lifetime_distribution_plot_editor_window
        )

    def open_lifetime_histogram_plot_editor(self) -> None:
        if not self._enable_plot_editor:
            return
        self._show_embedded_plot_editor(
            self._lifetime_histogram_plot_editor_window
        )

    def _show_embedded_plot_editor(
        self,
        editor: PlotEditorWindow | None,
    ) -> None:
        if editor is None:
            return
        self.tabs.setCurrentWidget(self.plot_editor_tab)
        self.plot_editor_tabs.setCurrentWidget(editor)
        editor.refresh_preview()

    def refresh_plot(self) -> None:
        self._render_plot_figure(self.figure)
        self.canvas.draw_idle()
        self.refresh_lifetime_plots()
        if self._plot_editor_window is not None:
            self._plot_editor_window.refresh_preview()

    def refresh_lifetime_plots(self) -> None:
        self.refresh_lifetime_distribution_plot()
        self.refresh_lifetime_histogram_plot()

    def refresh_lifetime_distribution_plot(self) -> None:
        self._render_lifetime_distribution_figure(self.histogram_figure)
        self.histogram_canvas.draw_idle()
        if self._lifetime_distribution_plot_editor_window is not None:
            self._lifetime_distribution_plot_editor_window.refresh_preview()

    def refresh_lifetime_histogram_plot(self) -> None:
        self._sync_lifetime_histogram_table()
        self._render_lifetime_histogram_figure(self.lifetime_histogram_figure)
        self.lifetime_histogram_canvas.draw_idle()
        if self._lifetime_histogram_plot_editor_window is not None:
            self._lifetime_histogram_plot_editor_window.refresh_preview()

    def _on_plot_editor_closed(self) -> None:
        self._plot_editor_window = None
        self._plot_editor_controls = None

    def _on_lifetime_distribution_plot_editor_closed(self) -> None:
        self._lifetime_distribution_plot_editor_window = None
        self._lifetime_distribution_plot_editor_controls = None

    def _on_lifetime_histogram_plot_editor_closed(self) -> None:
        self._lifetime_histogram_plot_editor_window = None
        self._lifetime_histogram_plot_editor_controls = None

    def _on_plot_editor_colormap_changed(self, colormap_name: str) -> None:
        index = self.colormap_combo.findData(colormap_name)
        if index < 0 or index == self.colormap_combo.currentIndex():
            return
        self.colormap_combo.setCurrentIndex(index)

    def _on_plot_editor_x_axis_unit_changed(self, unit_name: str) -> None:
        index = self.time_unit_combo.findData(unit_name)
        if index < 0 or index == self.time_unit_combo.currentIndex():
            return
        self.time_unit_combo.setCurrentIndex(index)

    def _sync_plot_editor_defaults(
        self, defaults: HeatmapPlotDefaults
    ) -> None:
        if (
            self._plot_editor_controls is not None
            and self._plot_editor_controls.needs_default_sync(defaults)
        ):
            self._plot_editor_controls.sync_defaults(defaults)

    def _plot_editor_pickle_state(self) -> dict[str, object]:
        return {
            "plot_editor_state": {
                "kind": "heatmap_plot_editor_state",
                "version": 1,
                "heatmap_settings": self._plot_settings.to_dict(),
                "panel_state": {
                    "display_mode": self._display_mode(),
                    "time_unit": str(self.time_unit_combo.currentData() or ""),
                    "colormap_name": str(
                        self.colormap_combo.currentData() or ""
                    ),
                    "lower_quantile": float(self.lower_quantile_spin.value()),
                    "upper_quantile": float(self.upper_quantile_spin.value()),
                    "overlay_name": self.overlay_combo.currentData(),
                },
            }
        }

    def _apply_loaded_plot_editor_pickle_state(
        self,
        payload: Mapping[str, object],
    ) -> bool:
        editor_state = payload.get("plot_editor_state")
        if not isinstance(editor_state, Mapping):
            return False
        if str(editor_state.get("kind")) != "heatmap_plot_editor_state":
            return False

        heatmap_settings = editor_state.get("heatmap_settings")
        if isinstance(heatmap_settings, Mapping):
            self._plot_settings.update_from_dict(heatmap_settings)

        panel_state = editor_state.get("panel_state")
        if isinstance(panel_state, Mapping):
            self._apply_panel_state_from_pickle(panel_state)

        defaults = self._current_plot_defaults()
        self._plot_settings.sync_labels(
            defaults.raw_cluster_labels,
            default_label_entries=defaults.default_label_entries,
        )
        if self._plot_editor_controls is not None:
            self._plot_editor_controls.sync_defaults(defaults)
        self.refresh_plot()
        return True

    def _lifetime_distribution_plot_editor_pickle_state(
        self,
    ) -> dict[str, object]:
        return {
            "plot_editor_state": {
                "kind": "lifetime_distribution_plot_editor_state",
                "version": 1,
                "lifetime_distribution_settings": (
                    self._lifetime_distribution_plot_settings.to_dict()
                ),
                "panel_state": {
                    "lifetime_sample_mode": (
                        self.lifetime_sample_combo.currentData()
                    ),
                    "lifetime_unit": self.lifetime_unit_combo.currentData(),
                },
            }
        }

    def _apply_loaded_lifetime_distribution_plot_editor_pickle_state(
        self,
        payload: Mapping[str, object],
    ) -> bool:
        editor_state = payload.get("plot_editor_state")
        if not isinstance(editor_state, Mapping):
            return False
        if (
            str(editor_state.get("kind"))
            != "lifetime_distribution_plot_editor_state"
        ):
            return False

        settings = editor_state.get("lifetime_distribution_settings")
        if isinstance(settings, Mapping):
            self._lifetime_distribution_plot_settings.update_from_dict(
                settings
            )

        panel_state = editor_state.get("panel_state")
        if isinstance(panel_state, Mapping):
            self._apply_lifetime_distribution_panel_state_from_pickle(
                panel_state
            )

        defaults = self._current_lifetime_distribution_plot_defaults()
        controls = self._lifetime_distribution_plot_editor_controls
        if controls is not None:
            controls.sync_defaults(defaults)
        self.refresh_lifetime_distribution_plot()
        return True

    def _lifetime_histogram_plot_editor_pickle_state(
        self,
    ) -> dict[str, object]:
        return {
            "plot_editor_state": {
                "kind": "lifetime_histogram_plot_editor_state",
                "version": 1,
                "lifetime_histogram_settings": (
                    self._lifetime_histogram_plot_settings.to_dict()
                ),
                "panel_state": {
                    "lifetime_sample_mode": (
                        self.lifetime_sample_combo.currentData()
                    ),
                    "lifetime_unit": self.lifetime_unit_combo.currentData(),
                    "histogram_bins": int(
                        self.lifetime_histogram_bins_spin.value()
                    ),
                    "selected_label": (
                        self._selected_lifetime_histogram_label
                    ),
                },
            }
        }

    def _apply_loaded_lifetime_histogram_plot_editor_pickle_state(
        self,
        payload: Mapping[str, object],
    ) -> bool:
        editor_state = payload.get("plot_editor_state")
        if not isinstance(editor_state, Mapping):
            return False
        if (
            str(editor_state.get("kind"))
            != "lifetime_histogram_plot_editor_state"
        ):
            return False

        settings = editor_state.get("lifetime_histogram_settings")
        if isinstance(settings, Mapping):
            self._lifetime_histogram_plot_settings.update_from_dict(settings)

        panel_state = editor_state.get("panel_state")
        if isinstance(panel_state, Mapping):
            self._apply_lifetime_histogram_panel_state_from_pickle(panel_state)

        defaults = self._current_lifetime_histogram_plot_defaults()
        controls = self._lifetime_histogram_plot_editor_controls
        if controls is not None:
            controls.sync_defaults(defaults)
        self.refresh_lifetime_histogram_plot()
        return True

    def _apply_lifetime_distribution_panel_state_from_pickle(
        self,
        panel_state: Mapping[str, object],
    ) -> None:
        self.lifetime_sample_combo.blockSignals(True)
        self.lifetime_unit_combo.blockSignals(True)
        try:
            self._set_combo_data_if_present(
                self.lifetime_sample_combo,
                panel_state.get("lifetime_sample_mode"),
            )
            self._set_combo_data_if_present(
                self.lifetime_unit_combo,
                panel_state.get("lifetime_unit"),
            )
        finally:
            self.lifetime_sample_combo.blockSignals(False)
            self.lifetime_unit_combo.blockSignals(False)

    def _apply_lifetime_histogram_panel_state_from_pickle(
        self,
        panel_state: Mapping[str, object],
    ) -> None:
        self.lifetime_sample_combo.blockSignals(True)
        self.lifetime_unit_combo.blockSignals(True)
        self.lifetime_histogram_bins_spin.blockSignals(True)
        try:
            self._set_combo_data_if_present(
                self.lifetime_sample_combo,
                panel_state.get("lifetime_sample_mode"),
            )
            self._set_combo_data_if_present(
                self.lifetime_unit_combo,
                panel_state.get("lifetime_unit"),
            )
            if "histogram_bins" in panel_state:
                self.lifetime_histogram_bins_spin.setValue(
                    int(panel_state["histogram_bins"])
                )
            selected_label = panel_state.get("selected_label")
            self._selected_lifetime_histogram_label = (
                None if selected_label is None else str(selected_label)
            )
        finally:
            self.lifetime_sample_combo.blockSignals(False)
            self.lifetime_unit_combo.blockSignals(False)
            self.lifetime_histogram_bins_spin.blockSignals(False)

    def _apply_panel_state_from_pickle(
        self,
        panel_state: Mapping[str, object],
    ) -> None:
        self.display_mode_combo.blockSignals(True)
        self.time_unit_combo.blockSignals(True)
        self.colormap_combo.blockSignals(True)
        self.lower_quantile_spin.blockSignals(True)
        self.upper_quantile_spin.blockSignals(True)
        self.overlay_combo.blockSignals(True)
        try:
            self._set_combo_data_if_present(
                self.display_mode_combo,
                panel_state.get("display_mode"),
            )
            self._set_combo_data_if_present(
                self.time_unit_combo,
                panel_state.get("time_unit"),
            )
            self._set_combo_data_if_present(
                self.colormap_combo,
                panel_state.get("colormap_name"),
            )
            if "lower_quantile" in panel_state:
                self.lower_quantile_spin.setValue(
                    float(panel_state["lower_quantile"])
                )
            if "upper_quantile" in panel_state:
                self.upper_quantile_spin.setValue(
                    float(panel_state["upper_quantile"])
                )
            self._ensure_valid_quantiles()
            self._set_combo_data_if_present(
                self.overlay_combo,
                panel_state.get("overlay_name"),
            )
        finally:
            self.display_mode_combo.blockSignals(False)
            self.time_unit_combo.blockSignals(False)
            self.colormap_combo.blockSignals(False)
            self.lower_quantile_spin.blockSignals(False)
            self.upper_quantile_spin.blockSignals(False)
            self.overlay_combo.blockSignals(False)

    @staticmethod
    def _set_combo_data_if_present(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _render_plot_figure(self, figure: Figure) -> None:
        defaults = self._current_plot_defaults()
        self._plot_settings.sync_labels(
            defaults.raw_cluster_labels,
            default_label_entries=defaults.default_label_entries,
        )

        figure.clear()
        if self._result is None:
            self._sync_plot_editor_defaults(defaults)
            axis = figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "Run the analysis to render the cluster-distribution heatmap.",
            )
            return

        if self._result.bin_count == 0:
            self._sync_plot_editor_defaults(defaults)
            axis = figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "No time bins are available for the current selection.",
            )
            return

        matrix = self._result.matrix(self._display_mode())
        if matrix.size == 0 or len(self._result.cluster_labels) == 0:
            self._sync_plot_editor_defaults(defaults)
            axis = figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "No clusters were detected in the selected time window.",
            )
            return

        overlay_name = self.overlay_combo.currentData()
        show_overlay = bool(
            overlay_name is not None and self._result.energy_data is not None
        )

        if show_overlay:
            grid = figure.add_gridspec(
                2,
                1,
                height_ratios=[4.0, 1.2],
                hspace=0.08,
            )
            heatmap_axis = figure.add_subplot(grid[0, 0])
            overlay_axis = figure.add_subplot(
                grid[1, 0],
                sharex=heatmap_axis,
            )
        else:
            heatmap_axis = figure.add_subplot(111)
            overlay_axis = None

        time_unit = self.time_unit_combo.currentData()
        time_edges = self._result.time_edges(time_unit)
        cmap = colormaps[self.colormap_combo.currentData()]
        ordered_labels = self._plot_settings.ordered_raw_labels(defaults)
        ordered_index_lookup = {
            str(label): index
            for index, label in enumerate(self._result.cluster_labels)
        }
        ordered_indices = [
            ordered_index_lookup[label]
            for label in ordered_labels
            if label in ordered_index_lookup
        ]
        display_matrix = (
            matrix
            if not ordered_indices
            else np.asarray(matrix, dtype=float)[ordered_indices, :]
        )
        auto_vmin, auto_vmax = self._auto_color_limits(display_matrix)
        defaults = replace(
            defaults,
            auto_color_limit_min=auto_vmin,
            auto_color_limit_max=auto_vmax,
        )
        self._sync_plot_editor_defaults(defaults)
        norm = self._heatmap_norm(defaults)

        image = heatmap_axis.imshow(
            display_matrix,
            aspect=self._resolved_aspect(),
            origin="lower",
            interpolation="nearest",
            extent=(
                float(time_edges[0]),
                float(time_edges[-1]),
                -0.5,
                len(ordered_labels) - 0.5,
            ),
            cmap=cmap,
            norm=norm,
        )
        colorbar = figure.colorbar(image, ax=heatmap_axis, pad=0.02)
        colorbar.set_label(
            self._plot_settings.resolve_colorbar_label(defaults),
            fontsize=self._plot_settings.axis_label_font_size,
            **self._font_kwargs(),
        )
        apply_igor_inline_text_artist(
            colorbar.ax.yaxis.label,
            self._plot_settings.resolve_colorbar_label(defaults),
            default_font_size=self._plot_settings.axis_label_font_size,
            gid_prefix="heatmap-colorbar-label",
            target_axes=colorbar.ax,
        )
        colorbar.ax.tick_params(
            labelsize=self._plot_settings.tick_label_font_size
        )
        for tick_label in colorbar.ax.get_yticklabels():
            self._apply_font_to_text(tick_label)

        label_count = len(ordered_labels)
        label_step = max(
            1,
            int(
                math.ceil(
                    label_count / max(self._plot_settings.max_y_ticks, 1)
                )
            ),
        )
        tick_positions = np.arange(0, len(ordered_labels), label_step)
        y_tick_labels = [
            self._plot_settings.display_label(ordered_labels[index])
            for index in tick_positions
        ]
        rendered_y_tick_labels: list[str] = []
        composite_y_tick_labels: dict[int, str] = {}
        for tick_index, tick_label in enumerate(y_tick_labels):
            segments, has_markup = prepare_igor_inline_segments(
                tick_label,
                default_font_size=self._plot_settings.cluster_label_font_size,
            )
            if not has_markup:
                rendered_y_tick_labels.append(tick_label)
                continue
            if any(
                not math.isclose(
                    segment.font_size,
                    self._plot_settings.cluster_label_font_size,
                )
                for segment in segments
            ):
                rendered_y_tick_labels.append(" ")
                composite_y_tick_labels[tick_index] = tick_label
                continue
            rendered_y_tick_labels.append(
                igor_inline_to_mathtext(
                    tick_label,
                    default_font_size=self._plot_settings.cluster_label_font_size,
                )
            )
        heatmap_axis.set_yticks(tick_positions)
        heatmap_axis.set_yticklabels(rendered_y_tick_labels)
        heatmap_axis.set_ylabel(
            self._plot_settings.resolve_y_label(defaults),
            fontsize=self._plot_settings.axis_label_font_size,
            **self._font_kwargs(),
        )
        apply_igor_inline_text_artist(
            heatmap_axis.yaxis.label,
            self._plot_settings.resolve_y_label(defaults),
            default_font_size=self._plot_settings.axis_label_font_size,
            gid_prefix="heatmap-y-label",
            target_axes=heatmap_axis,
        )
        heatmap_axis.set_xlim(float(time_edges[0]), float(time_edges[-1]))
        heatmap_axis.set_title(
            self._plot_settings.resolve_title(defaults),
            y=self._plot_settings.resolve_title_position_y(defaults),
            fontsize=self._plot_settings.title_font_size,
            **self._font_kwargs(),
        )
        heatmap_axis.title.set_x(
            self._plot_settings.resolve_title_position_x(defaults)
        )
        apply_igor_inline_text_artist(
            heatmap_axis.title,
            self._plot_settings.resolve_title(defaults),
            default_font_size=self._plot_settings.title_font_size,
            gid_prefix="heatmap-title",
            target_axes=heatmap_axis,
        )
        heatmap_axis.xaxis.set_major_locator(
            MaxNLocator(nbins=max(self._plot_settings.max_x_ticks, 2))
        )

        if overlay_axis is None:
            heatmap_axis.set_xlabel(
                self._plot_settings.resolve_x_label(defaults),
                fontsize=self._plot_settings.axis_label_font_size,
                **self._font_kwargs(),
            )
            apply_igor_inline_text_artist(
                heatmap_axis.xaxis.label,
                self._plot_settings.resolve_x_label(defaults),
                default_font_size=self._plot_settings.axis_label_font_size,
                gid_prefix="heatmap-x-label",
                target_axes=heatmap_axis,
            )
        else:
            heatmap_axis.tick_params(labelbottom=False)

        self._style_heatmap_ticks(heatmap_axis)
        for tick_index, tick_label in enumerate(
            heatmap_axis.get_yticklabels()
        ):
            if tick_index not in composite_y_tick_labels:
                continue
            apply_igor_inline_text_artist(
                tick_label,
                composite_y_tick_labels[tick_index],
                default_font_size=self._plot_settings.cluster_label_font_size,
                gid_prefix=f"heatmap-y-tick-{tick_index}",
                target_axes=heatmap_axis,
            )

        if overlay_axis is not None and overlay_name is not None:
            x_values, y_values, y_label = self._result.energy_series(
                overlay_name,
                unit=time_unit,
            )
            overlay_axis.plot(
                x_values,
                y_values,
                color=OVERLAY_COLORS.get(overlay_name, "#333333"),
                linewidth=1.5,
            )
            overlay_axis.set_ylabel(
                y_label,
                fontsize=self._plot_settings.axis_label_font_size,
                **self._font_kwargs(),
            )
            overlay_axis.set_xlabel(
                self._plot_settings.resolve_x_label(defaults),
                fontsize=self._plot_settings.axis_label_font_size,
                **self._font_kwargs(),
            )
            apply_igor_inline_text_artist(
                overlay_axis.xaxis.label,
                self._plot_settings.resolve_x_label(defaults),
                default_font_size=self._plot_settings.axis_label_font_size,
                gid_prefix="overlay-x-label",
                target_axes=overlay_axis,
            )
            overlay_axis.grid(alpha=0.25, linestyle=":")
            overlay_axis.xaxis.set_major_locator(
                MaxNLocator(nbins=max(self._plot_settings.max_x_ticks, 2))
            )
            self._style_overlay_ticks(overlay_axis)

        figure.tight_layout()

    def _render_lifetime_distribution_figure(self, figure: Figure) -> None:
        figure.clear()
        axis = figure.add_subplot(111)
        defaults = self._current_lifetime_distribution_plot_defaults()
        settings = self._lifetime_distribution_plot_settings
        self._sync_lifetime_distribution_plot_editor_defaults(defaults)

        def _apply_axis_style() -> None:
            axis.set_xlabel(
                settings.resolve_x_label(defaults),
                fontsize=settings.axis_label_font_size,
                **self._lifetime_distribution_font_kwargs(),
            )
            apply_igor_inline_text_artist(
                axis.xaxis.label,
                settings.resolve_x_label(defaults),
                default_font_size=settings.axis_label_font_size,
                gid_prefix="lifetime-distribution-x-label",
                target_axes=axis,
            )
            axis.set_ylabel(
                settings.resolve_y_label(defaults),
                fontsize=settings.axis_label_font_size,
                **self._lifetime_distribution_font_kwargs(),
            )
            apply_igor_inline_text_artist(
                axis.yaxis.label,
                settings.resolve_y_label(defaults),
                default_font_size=settings.axis_label_font_size,
                gid_prefix="lifetime-distribution-y-label",
                target_axes=axis,
            )
            axis.set_title(
                settings.resolve_title(defaults),
                y=settings.resolve_title_position_y(defaults),
                fontsize=settings.title_font_size,
                **self._lifetime_distribution_font_kwargs(),
            )
            axis.title.set_x(settings.resolve_title_position_x(defaults))
            apply_igor_inline_text_artist(
                axis.title,
                settings.resolve_title(defaults),
                default_font_size=settings.title_font_size,
                gid_prefix="lifetime-distribution-title",
                target_axes=axis,
            )
            if settings.show_grid:
                axis.grid(axis="y", alpha=0.28)
            else:
                axis.grid(False)
            self._style_lifetime_distribution_ticks(axis)

        def _tick_labels() -> list[str]:
            raw_tick_labels = [
                display_labels.get(label, label) for label in labels
            ]
            max_ticks = max(int(settings.max_x_ticks), 2)
            if len(raw_tick_labels) <= max_ticks:
                return raw_tick_labels
            step = max(1, math.ceil(len(raw_tick_labels) / max_ticks))
            return [
                (
                    tick_label
                    if index % step == 0 or index == len(raw_tick_labels) - 1
                    else ""
                )
                for index, tick_label in enumerate(raw_tick_labels)
            ]

        if self._result is None:
            self._draw_placeholder(
                axis,
                "Run the analysis to render lifetime distributions.",
            )
            return

        labels = self._ordered_lifetime_labels()
        if not labels:
            self._draw_placeholder(
                axis,
                "No stoichiometry labels are available "
                "for the current result.",
            )
            return

        samples_by_label = self._lifetime_samples_by_label(labels)
        positive_sample_count = sum(
            len(samples) for samples in samples_by_label
        )
        display_labels = dict(self._resolved_lifetime_x_axis_entries(labels))
        positions = np.arange(1, len(labels) + 1, dtype=float)
        if positive_sample_count == 0:
            axis.set_xticks(positions)
            axis.set_xticklabels(
                _tick_labels(),
                ha=(
                    "right"
                    if settings.x_tick_rotation not in (0, 360, -360)
                    else "center"
                ),
            )
            _apply_axis_style()
            axis.text(
                0.5,
                0.5,
                "No positive lifetime samples are available "
                "for the selected mode.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            figure.tight_layout()
            return

        box_positions: list[float] = []
        box_samples: list[np.ndarray] = []
        for position, samples in zip(
            positions,
            samples_by_label,
            strict=False,
        ):
            if samples.size == 0:
                continue
            box_positions.append(float(position))
            box_samples.append(samples)

        if box_samples:
            axis.boxplot(
                box_samples,
                positions=box_positions,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 1.6},
                boxprops={
                    "facecolor": self._resolved_plot_color(
                        settings.bar_color,
                        "#8ecae6",
                    ),
                    "edgecolor": self._resolved_plot_color(
                        settings.edge_color,
                        "#2b6f8a",
                    ),
                    "alpha": max(0.0, min(float(settings.bar_alpha), 1.0)),
                },
                whiskerprops={
                    "color": self._resolved_plot_color(
                        settings.edge_color,
                        "#111111",
                    ),
                    "linewidth": 1.0,
                },
                capprops={
                    "color": self._resolved_plot_color(
                        settings.edge_color,
                        "#111111",
                    ),
                    "linewidth": 1.0,
                },
            )

        point_alpha = min(
            max(0.0, min(float(settings.bar_alpha), 1.0)),
            0.48,
        )
        for position, samples in zip(
            positions,
            samples_by_label,
            strict=False,
        ):
            if samples.size == 0:
                continue
            offsets = (
                np.zeros(1, dtype=float)
                if samples.size == 1
                else np.linspace(-0.16, 0.16, samples.size)
            )
            axis.scatter(
                np.full(samples.size, float(position)) + offsets,
                samples,
                s=20.0,
                color=self._resolved_plot_color(
                    settings.bar_color,
                    "#219ebc",
                ),
                alpha=point_alpha,
                edgecolors="none",
                zorder=3,
            )

        axis.set_yscale("log")
        axis.set_xlim(0.5, len(labels) + 0.5)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            _tick_labels(),
            ha=(
                "right"
                if settings.x_tick_rotation not in (0, 360, -360)
                else "center"
            ),
        )
        _apply_axis_style()
        self._draw_lifetime_group_separators(axis, labels)
        figure.tight_layout()

    def _render_lifetime_histogram_figure(self, figure: Figure) -> None:
        figure.clear()
        axis = figure.add_subplot(111)
        defaults = self._current_lifetime_histogram_plot_defaults()
        settings = self._lifetime_histogram_plot_settings
        if self._result is None:
            self._sync_lifetime_histogram_plot_editor_defaults(defaults)
            self._draw_placeholder(
                axis,
                "Run the analysis to render lifetime histograms.",
            )
            return

        labels = self._ordered_lifetime_labels()
        if not labels:
            self._sync_lifetime_histogram_plot_editor_defaults(defaults)
            self._draw_placeholder(
                axis,
                "No stoichiometry labels are available "
                "for the current result.",
            )
            return

        selected_label = self._selected_lifetime_histogram_label
        if selected_label not in labels:
            selected_label = labels[0]
            self._selected_lifetime_histogram_label = selected_label

        defaults = self._current_lifetime_histogram_plot_defaults()
        self._sync_lifetime_histogram_plot_editor_defaults(defaults)
        samples = self._lifetime_samples_by_label((selected_label,))[0]
        axis.set_xlabel(
            settings.resolve_x_label(defaults),
            fontsize=settings.axis_label_font_size,
            **self._lifetime_histogram_font_kwargs(),
        )
        apply_igor_inline_text_artist(
            axis.xaxis.label,
            settings.resolve_x_label(defaults),
            default_font_size=settings.axis_label_font_size,
            gid_prefix="lifetime-histogram-x-label",
            target_axes=axis,
        )
        axis.set_ylabel(
            settings.resolve_y_label(defaults),
            fontsize=settings.axis_label_font_size,
            **self._lifetime_histogram_font_kwargs(),
        )
        apply_igor_inline_text_artist(
            axis.yaxis.label,
            settings.resolve_y_label(defaults),
            default_font_size=settings.axis_label_font_size,
            gid_prefix="lifetime-histogram-y-label",
            target_axes=axis,
        )
        axis.set_title(
            settings.resolve_title(defaults),
            y=settings.resolve_title_position_y(defaults),
            fontsize=settings.title_font_size,
            **self._lifetime_histogram_font_kwargs(),
        )
        axis.title.set_x(settings.resolve_title_position_x(defaults))
        apply_igor_inline_text_artist(
            axis.title,
            settings.resolve_title(defaults),
            default_font_size=settings.title_font_size,
            gid_prefix="lifetime-histogram-title",
            target_axes=axis,
        )
        if settings.show_grid:
            axis.grid(axis="y", alpha=0.28)
        else:
            axis.grid(False)
        self._style_lifetime_histogram_ticks(axis)

        if samples.size == 0:
            axis.text(
                0.5,
                0.5,
                "No positive lifetime samples are available "
                "for the selected stoichiometry.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            figure.tight_layout()
            return

        axis.hist(
            samples,
            bins=int(self.lifetime_histogram_bins_spin.value()),
            color=self._resolved_plot_color(settings.bar_color, "#219ebc"),
            edgecolor=self._resolved_plot_color(
                settings.edge_color,
                "#184e63",
            ),
            alpha=max(0.0, min(float(settings.bar_alpha), 1.0)),
        )
        figure.tight_layout()

    def _sync_lifetime_histogram_table(self) -> None:
        labels = self._ordered_lifetime_labels()
        samples_by_label = self._lifetime_samples_by_label(labels)
        display_labels = dict(self._resolved_lifetime_x_axis_entries(labels))
        selected_label = self._resolved_lifetime_histogram_label(
            labels,
            samples_by_label,
        )

        table = self.lifetime_histogram_table
        table.blockSignals(True)
        try:
            table.setRowCount(len(labels))
            table.setHorizontalHeaderLabels(
                [
                    "Stoichiometry",
                    "Samples",
                    f"Mean ({self._lifetime_unit_name()})",
                ]
            )
            selected_row: int | None = None
            for row, (label, samples) in enumerate(
                zip(labels, samples_by_label, strict=False)
            ):
                if label == selected_label:
                    selected_row = row
                label_item = QTableWidgetItem(display_labels.get(label, label))
                label_item.setData(Qt.ItemDataRole.UserRole, label)
                table.setItem(row, 0, label_item)
                table.setItem(row, 1, QTableWidgetItem(str(samples.size)))
                table.setItem(
                    row,
                    2,
                    QTableWidgetItem(self._format_sample_mean(samples)),
                )

            table.resizeColumnToContents(0)
            table.resizeColumnToContents(1)
            table.clearSelection()
            if selected_row is not None:
                table.selectRow(selected_row)
        finally:
            table.blockSignals(False)

        self._selected_lifetime_histogram_label = selected_label

    def _resolved_lifetime_histogram_label(
        self,
        labels: tuple[str, ...],
        samples_by_label: list[np.ndarray],
    ) -> str | None:
        selected_label = self._selected_lifetime_histogram_label
        if selected_label in labels:
            return selected_label
        for label, samples in zip(labels, samples_by_label, strict=False):
            if samples.size:
                return label
        return labels[0] if labels else None

    def _on_lifetime_histogram_selection_changed(self) -> None:
        selected_label = self._selected_histogram_table_label()
        if selected_label is None:
            return
        self._selected_lifetime_histogram_label = selected_label
        self._render_lifetime_histogram_figure(self.lifetime_histogram_figure)
        self.lifetime_histogram_canvas.draw_idle()

    def _selected_histogram_table_label(self) -> str | None:
        row = self.lifetime_histogram_table.currentRow()
        if row < 0:
            return None
        item = self.lifetime_histogram_table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return None if value is None else str(value)

    @staticmethod
    def _format_sample_mean(samples: np.ndarray) -> str:
        if samples.size == 0:
            return ""
        return f"{float(np.mean(samples)):.4g}"

    def _on_lifetime_x_axis_order_changed(self, _index: int) -> None:
        is_custom = self.lifetime_x_axis_order_combo.currentData() == "custom"
        self.edit_lifetime_x_axis_button.setEnabled(is_custom)
        self.refresh_lifetime_plots()

    def _on_edit_lifetime_x_axis_order(self) -> None:
        labels = self._auto_lifetime_labels()
        entries = (
            self._resolved_lifetime_x_axis_entries(labels)
            if self._lifetime_x_axis_custom_order
            else self._default_lifetime_x_axis_entries(labels)
        )
        dialog = _XAxisOrderDialog(list(entries), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._lifetime_x_axis_custom_order = dialog.result_entries()
        custom_index = self.lifetime_x_axis_order_combo.findData("custom")
        if custom_index >= 0:
            self.lifetime_x_axis_order_combo.blockSignals(True)
            self.lifetime_x_axis_order_combo.setCurrentIndex(custom_index)
            self.lifetime_x_axis_order_combo.blockSignals(False)
        self.edit_lifetime_x_axis_button.setEnabled(True)
        self.refresh_lifetime_plots()

    def _auto_lifetime_labels(self) -> tuple[str, ...]:
        if self._result is None:
            return ()
        return tuple(str(label) for label in self._result.cluster_labels)

    def _ordered_lifetime_labels(self) -> tuple[str, ...]:
        labels = self._auto_lifetime_labels()
        if (
            self.lifetime_x_axis_order_combo.currentData() != "custom"
            or not self._lifetime_x_axis_custom_order
        ):
            return labels
        available = set(labels)
        ordered = [
            raw
            for raw, _display in self._lifetime_x_axis_custom_order
            if raw in available
        ]
        ordered_set = set(ordered)
        ordered.extend(label for label in labels if label not in ordered_set)
        return tuple(ordered)

    def _default_lifetime_x_axis_entries(
        self,
        labels: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (label, self._format_cluster_axis_label(label)) for label in labels
        )

    def _resolved_lifetime_x_axis_entries(
        self,
        labels: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        default_map = dict(self._default_lifetime_x_axis_entries(labels))
        if (
            self.lifetime_x_axis_order_combo.currentData() != "custom"
            or not self._lifetime_x_axis_custom_order
        ):
            return tuple(
                (label, default_map.get(label, label)) for label in labels
            )
        custom_map = {
            raw: display
            for raw, display in self._lifetime_x_axis_custom_order
            if raw in default_map
        }
        ordered_labels = self._ordered_lifetime_labels()
        return tuple(
            (
                label,
                custom_map.get(label, default_map.get(label, label)),
            )
            for label in ordered_labels
        )

    def _lifetime_samples_by_label(
        self,
        labels: tuple[str, ...],
    ) -> list[np.ndarray]:
        if self._result is None:
            return []
        label_index = {
            str(label): index
            for index, label in enumerate(self._result.cluster_labels)
        }
        frame_times_fs = self._result.frame_times_fs
        mode = str(self.lifetime_sample_combo.currentData() or "completed")
        scale = self._lifetime_unit_scale()
        samples_by_label: list[np.ndarray] = []
        for label in labels:
            index = label_index.get(label)
            if index is None:
                samples_by_label.append(np.zeros(0, dtype=float))
                continue
            metrics = _summarize_series_lifetimes(
                self._result.frame_count_matrix[index, :],
                frame_times_fs=frame_times_fs,
                observation_start_fs=self._result.preview.analysis_start_fs,
                observation_stop_fs=self._result.preview.analysis_stop_fs,
            )
            lifetime_values = list(metrics.completed_lifetimes_fs)
            if mode == "all":
                lifetime_values.extend(metrics.window_truncated_lifetimes_fs)
            samples = np.asarray(lifetime_values, dtype=float) * scale
            samples = samples[np.isfinite(samples) & (samples > 0.0)]
            samples_by_label.append(samples)
        return samples_by_label

    def _lifetime_unit_name(self) -> str:
        value = self.lifetime_unit_combo.currentData()
        return "ps" if value is None else str(value)

    def _lifetime_unit_scale(self) -> float:
        unit = self._lifetime_unit_name()
        for _label, value, scale in LIFETIME_UNITS:
            if value == unit:
                return float(scale)
        return 1.0 / 1000.0

    def _draw_lifetime_group_separators(
        self,
        axis,
        labels: tuple[str, ...],
    ) -> None:
        if self._result is None or len(labels) < 2:
            return
        previous_size = self._result.cluster_sizes.get(labels[0])
        for index, label in enumerate(labels[1:], start=1):
            current_size = self._result.cluster_sizes.get(label)
            if current_size == previous_size:
                continue
            axis.axvline(
                index + 0.5,
                color="#8f8f8f",
                linestyle="--",
                linewidth=0.9,
                alpha=0.75,
                zorder=0,
            )
            previous_size = current_size

    def _current_plot_defaults(self) -> HeatmapPlotDefaults:
        time_unit = self.time_unit_combo.currentData()
        raw_labels = (
            ()
            if self._result is None
            else tuple(str(label) for label in self._result.cluster_labels)
        )
        current_colormap = self.colormap_combo.currentData()
        default_label_entries = tuple(
            (raw_label, self._format_cluster_axis_label(raw_label))
            for raw_label in raw_labels
        )
        return HeatmapPlotDefaults(
            title=(
                "Time-Binned Cluster Distribution "
                f"({DISPLAY_MODE_LABELS[self._display_mode()]})"
            ),
            x_label=f"Time ({time_unit})",
            y_label="Cluster label",
            colorbar_label=DISPLAY_MODE_COLORBAR_LABELS[self._display_mode()],
            default_x_axis_unit_name=(
                "" if time_unit is None else str(time_unit)
            ),
            available_x_axis_unit_names=("fs", "ps"),
            default_colormap_name=(
                "" if current_colormap is None else str(current_colormap)
            ),
            available_colormap_names=tuple(PLOT_COLORMAPS),
            raw_cluster_labels=raw_labels,
            default_label_entries=default_label_entries,
        )

    def _current_lifetime_histogram_plot_defaults(
        self,
    ) -> LifetimeHistogramPlotDefaults:
        labels = self._ordered_lifetime_labels()
        selected_label = self._selected_lifetime_histogram_label
        if selected_label not in labels:
            selected_label = labels[0] if labels else None
        display_label = ""
        if selected_label is not None:
            display_labels = dict(
                self._resolved_lifetime_x_axis_entries(labels)
            )
            display_label = display_labels.get(selected_label, selected_label)
        return LifetimeHistogramPlotDefaults(
            title=(
                "Lifetime Histogram"
                if not display_label
                else f"Lifetime Histogram: {display_label}"
            ),
            x_label=f"Lifetime ({self._lifetime_unit_name()})",
            y_label="Count",
        )

    def _current_lifetime_distribution_plot_defaults(
        self,
    ) -> LifetimeDistributionPlotDefaults:
        return LifetimeDistributionPlotDefaults(
            title="Lifetime Distribution by Stoichiometry",
            x_label="Stoichiometry",
            y_label=f"Lifetime ({self._lifetime_unit_name()})",
        )

    def _sync_lifetime_distribution_plot_editor_defaults(
        self,
        defaults: LifetimeDistributionPlotDefaults,
    ) -> None:
        controls = self._lifetime_distribution_plot_editor_controls
        if controls is not None and controls.needs_default_sync(defaults):
            controls.sync_defaults(defaults)

    def _sync_lifetime_histogram_plot_editor_defaults(
        self,
        defaults: LifetimeHistogramPlotDefaults,
    ) -> None:
        controls = self._lifetime_histogram_plot_editor_controls
        if controls is not None and controls.needs_default_sync(defaults):
            controls.sync_defaults(defaults)

    def _resolved_aspect(self) -> str | float:
        if self._plot_settings.aspect_mode == "equal":
            return "equal"
        if self._plot_settings.aspect_mode == "custom":
            return float(self._plot_settings.custom_aspect)
        return "auto"

    def _font_kwargs(self) -> dict[str, str]:
        if not self._plot_settings.font_family:
            return {}
        return {"fontfamily": self._plot_settings.font_family}

    @staticmethod
    def _format_cluster_axis_label(label: str) -> str:
        return format_stoich_for_axis(label)

    def _apply_font_to_text(self, text_artist) -> None:
        if self._plot_settings.font_family:
            text_artist.set_fontfamily(self._plot_settings.font_family)

    def _style_heatmap_ticks(self, axis) -> None:
        axis.tick_params(
            axis="x",
            labelsize=self._plot_settings.tick_label_font_size,
            labelrotation=self._plot_settings.x_tick_rotation,
        )
        axis.tick_params(
            axis="y",
            labelsize=self._plot_settings.cluster_label_font_size,
            labelrotation=self._plot_settings.y_tick_rotation,
        )
        if (
            self._plot_settings.show_minor_x_ticks
            or self._plot_settings.show_minor_y_ticks
        ):
            axis.minorticks_on()
        else:
            axis.minorticks_off()
        axis.tick_params(
            axis="x",
            which="minor",
            bottom=self._plot_settings.show_minor_x_ticks,
            top=False,
        )
        axis.tick_params(
            axis="y",
            which="minor",
            left=self._plot_settings.show_minor_y_ticks,
            right=False,
        )
        for tick_label in axis.get_xticklabels():
            self._apply_font_to_text(tick_label)
        for tick_label in axis.get_yticklabels():
            self._apply_font_to_text(tick_label)

    def _style_overlay_ticks(self, axis) -> None:
        axis.tick_params(
            axis="both",
            labelsize=self._plot_settings.tick_label_font_size,
        )
        axis.tick_params(
            axis="x",
            labelrotation=self._plot_settings.x_tick_rotation,
        )
        if self._plot_settings.show_minor_x_ticks:
            axis.minorticks_on()
        else:
            axis.minorticks_off()
        axis.tick_params(
            axis="x",
            which="minor",
            bottom=self._plot_settings.show_minor_x_ticks,
            top=False,
        )
        for tick_label in axis.get_xticklabels():
            self._apply_font_to_text(tick_label)
        for tick_label in axis.get_yticklabels():
            self._apply_font_to_text(tick_label)

    def _style_lifetime_histogram_ticks(self, axis) -> None:
        settings = self._lifetime_histogram_plot_settings
        axis.xaxis.set_major_locator(
            MaxNLocator(nbins=max(settings.max_x_ticks, 2))
        )
        axis.yaxis.set_major_locator(
            MaxNLocator(nbins=max(settings.max_y_ticks, 2), integer=True)
        )
        axis.tick_params(
            axis="x",
            labelsize=settings.tick_label_font_size,
            labelrotation=settings.x_tick_rotation,
        )
        axis.tick_params(
            axis="y",
            labelsize=settings.tick_label_font_size,
            labelrotation=settings.y_tick_rotation,
        )
        if settings.show_minor_x_ticks or settings.show_minor_y_ticks:
            axis.minorticks_on()
        else:
            axis.minorticks_off()
        axis.tick_params(
            axis="x",
            which="minor",
            bottom=settings.show_minor_x_ticks,
            top=False,
        )
        axis.tick_params(
            axis="y",
            which="minor",
            left=settings.show_minor_y_ticks,
            right=False,
        )
        for tick_label in axis.get_xticklabels():
            self._apply_lifetime_histogram_font_to_text(tick_label)
        for tick_label in axis.get_yticklabels():
            self._apply_lifetime_histogram_font_to_text(tick_label)

    def _style_lifetime_distribution_ticks(self, axis) -> None:
        settings = self._lifetime_distribution_plot_settings
        axis.tick_params(
            axis="x",
            labelsize=settings.tick_label_font_size,
            labelrotation=settings.x_tick_rotation,
        )
        axis.tick_params(
            axis="y",
            labelsize=settings.tick_label_font_size,
            labelrotation=settings.y_tick_rotation,
        )
        if settings.show_minor_x_ticks or settings.show_minor_y_ticks:
            axis.minorticks_on()
        else:
            axis.minorticks_off()
        axis.tick_params(
            axis="x",
            which="minor",
            bottom=settings.show_minor_x_ticks,
            top=False,
        )
        axis.tick_params(
            axis="y",
            which="minor",
            left=settings.show_minor_y_ticks,
            right=False,
        )
        for tick_label in axis.get_xticklabels():
            self._apply_lifetime_distribution_font_to_text(tick_label)
        for tick_label in axis.get_yticklabels():
            self._apply_lifetime_distribution_font_to_text(tick_label)

    def _lifetime_distribution_font_kwargs(self) -> dict[str, str]:
        font_family = self._lifetime_distribution_plot_settings.font_family
        if not font_family:
            return {}
        return {"fontfamily": font_family}

    def _apply_lifetime_distribution_font_to_text(self, text_artist) -> None:
        font_family = self._lifetime_distribution_plot_settings.font_family
        if font_family:
            text_artist.set_fontfamily(font_family)

    def _lifetime_histogram_font_kwargs(self) -> dict[str, str]:
        font_family = self._lifetime_histogram_plot_settings.font_family
        if not font_family:
            return {}
        return {"fontfamily": font_family}

    def _apply_lifetime_histogram_font_to_text(self, text_artist) -> None:
        font_family = self._lifetime_histogram_plot_settings.font_family
        if font_family:
            text_artist.set_fontfamily(font_family)

    @staticmethod
    def _resolved_plot_color(value: str, fallback: str) -> str:
        return value if mcolors.is_color_like(value) else fallback

    def _display_mode(self) -> str:
        value = self.display_mode_combo.currentData()
        return "fraction" if value is None else str(value)

    def _ensure_valid_quantiles(self) -> None:
        lower = self.lower_quantile_spin.value()
        upper = self.upper_quantile_spin.value()
        if lower >= upper:
            self.upper_quantile_spin.blockSignals(True)
            self.upper_quantile_spin.setValue(min(lower + 0.05, 1.0))
            self.upper_quantile_spin.blockSignals(False)
            lower = self.lower_quantile_spin.value()
            upper = self.upper_quantile_spin.value()
        if lower >= upper:
            self.lower_quantile_spin.blockSignals(True)
            self.lower_quantile_spin.setValue(max(upper - 0.05, 0.0))
            self.lower_quantile_spin.blockSignals(False)

    def _on_quantile_changed(self) -> None:
        self._ensure_valid_quantiles()
        self.refresh_plot()

    def _heatmap_norm(
        self,
        defaults: HeatmapPlotDefaults,
    ) -> mcolors.Normalize:
        vmin = float(self._plot_settings.resolve_color_limit_min(defaults))
        vmax = float(self._plot_settings.resolve_color_limit_max(defaults))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _auto_color_limits(self, matrix: np.ndarray) -> tuple[float, float]:
        values = np.asarray(matrix, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 1.0)

        positive = finite[finite > 0.0]
        if positive.size:
            finite = positive

        lower_q = float(self.lower_quantile_spin.value())
        upper_q = float(self.upper_quantile_spin.value())
        vmin = float(np.quantile(finite, lower_q))
        vmax = float(np.quantile(finite, upper_q))
        if vmax <= vmin:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return (vmin, vmax)

    @staticmethod
    def _draw_placeholder(axis, message: str) -> None:
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)


__all__ = ["ClusterDynamicsPlotPanel"]
