from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis.results import (
    BondAnalysisPlotRequest,
    export_plot_request_csv,
    recommended_plot_request_filename,
)

CATEGORY_COLORS = {
    "bond": "#87ceeb",
    "angle": "#f28e2b",
    "dihedral": "#2e8b57",
    "coordination": "#8e44ad",
}
CATEGORY_OVERLAY_COLORS = {
    "bond": ("#87ceeb", "#4fa3d1", "#b7e4f9", "#2f80aa"),
    "angle": ("#f28e2b", "#ffbe7d", "#d37222", "#f5a14a"),
    "dihedral": ("#2e8b57", "#56b870", "#1f6f43", "#8fd19e"),
    "coordination": ("#8e44ad", "#b07cc6", "#6f2f8f", "#c39bd3"),
}


class BondAnalysisPlotTab(QWidget):
    """One bondanalysis plot tab inside the shared plotting window."""

    def __init__(
        self,
        plot_request: BondAnalysisPlotRequest,
        default_output_dir: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plot_request = plot_request
        self.default_output_dir = Path(default_output_dir)
        self._series_colors = self._default_series_colors()
        self._series_states = self._initial_series_states()
        self._updating_series_list = False
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.controls_widget = QWidget()
        controls = QHBoxLayout(self.controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        self.save_button = QPushButton("Save Plot Data As...")
        self.save_button.clicked.connect(self.save_plot_data_as)
        controls.addWidget(self.save_button)

        controls.addWidget(QLabel("Bin size"))
        self.bin_size_spin = QDoubleSpinBox()
        self.bin_size_spin.setDecimals(3)
        self.bin_size_spin.setRange(0.001, 1000000.0)
        self.bin_size_spin.setSingleStep(0.05)
        self.bin_size_spin.setValue(self._default_bin_size())
        self.bin_size_spin.valueChanged.connect(
            lambda _value: self.refresh_plot()
        )
        controls.addWidget(self.bin_size_spin)

        controls.addWidget(QLabel("Transparency"))
        self.transparency_spin = QDoubleSpinBox()
        self.transparency_spin.setDecimals(2)
        self.transparency_spin.setRange(0.05, 1.0)
        self.transparency_spin.setSingleStep(0.05)
        self.transparency_spin.setValue(0.45)
        self.transparency_spin.valueChanged.connect(
            lambda _value: self.refresh_plot()
        )
        controls.addWidget(self.transparency_spin)

        self.dihedral_model_label = QLabel("Dihedral model")
        controls.addWidget(self.dihedral_model_label)
        self.dihedral_model_combo = QComboBox()
        self.dihedral_model_combo.addItem("Single circular", "single")
        self.dihedral_model_combo.addItem("Bimodal circular", "bimodal")
        self.dihedral_model_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.dihedral_model_combo)
        show_dihedral_model = self.plot_request.category == "dihedral"
        self.dihedral_model_label.setVisible(show_dihedral_model)
        self.dihedral_model_combo.setVisible(show_dihedral_model)

        self.dihedral_plot_style_label = QLabel("Dihedral plot")
        controls.addWidget(self.dihedral_plot_style_label)
        self.dihedral_plot_style_combo = QComboBox()
        self.dihedral_plot_style_combo.addItem("Signed histogram", "normal")
        self.dihedral_plot_style_combo.addItem("Radial histogram", "radial")
        self.dihedral_plot_style_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.dihedral_plot_style_combo)
        self.dihedral_plot_style_label.setVisible(show_dihedral_model)
        self.dihedral_plot_style_combo.setVisible(show_dihedral_model)

        self.series_color_label = QLabel("Overlay colors")
        controls.addWidget(self.series_color_label)
        self.series_color_container = QWidget()
        self.series_color_layout = QHBoxLayout(self.series_color_container)
        self.series_color_layout.setContentsMargins(0, 0, 0, 0)
        self.series_color_layout.setSpacing(6)
        self.series_color_list = QListWidget()
        self.series_color_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.series_color_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.series_color_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.series_color_list.setDragEnabled(True)
        self.series_color_list.setAcceptDrops(True)
        self.series_color_list.setDropIndicatorShown(True)
        self.series_color_list.setMaximumHeight(90)
        self.series_color_list.setMinimumWidth(240)
        self.series_color_list.setToolTip(
            "Drag items to change histogram stacking order. "
            "Double-click an item to change its color."
        )
        self.series_color_list.itemDoubleClicked.connect(
            self._choose_series_color_for_item
        )
        self.series_color_list.model().rowsMoved.connect(
            self._on_series_order_changed
        )
        self.series_color_layout.addWidget(self.series_color_list)
        controls.addWidget(self.series_color_container)
        controls.addStretch(1)

        root.addWidget(self.controls_widget)

        self.figure = Figure(figsize=(8.4, 6.4))
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        root.addWidget(NavigationToolbar(self.canvas, self))
        root.addWidget(self.canvas, stretch=1)

        self._refresh_series_color_list()

    def save_plot_data_as(self) -> None:
        default_filename = recommended_plot_request_filename(self.plot_request)
        default_path = self.default_output_dir / default_filename
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Bondanalysis Plot Data",
            str(default_path),
            "CSV Files (*.csv)",
        )
        if not selected_path:
            return
        csv_path = export_plot_request_csv(self.plot_request, selected_path)
        QMessageBox.information(
            self,
            "Bondanalysis Plot Data",
            f"Saved plot data to:\n{csv_path}",
        )

    def _ensure_axis_projection(self, projection: str | None) -> None:
        target_name = "polar" if projection == "polar" else "rectilinear"
        if getattr(self.axis, "name", "rectilinear") == target_name:
            return
        self.figure.delaxes(self.axis)
        kwargs = {"projection": projection} if projection is not None else {}
        self.axis = self.figure.add_subplot(111, **kwargs)

    def _plot_projection(self) -> str | None:
        if self._dihedral_plot_style() == "radial":
            return "polar"
        return None

    def refresh_plot(self) -> None:
        self._ensure_axis_projection(self._plot_projection())
        self.axis.clear()
        non_empty_series = self._ordered_series_states()
        self._update_overlay_controls(non_empty_series)

        if not non_empty_series:
            self.axis.text(
                0.5,
                0.5,
                "No computed values were found for this selection.",
                ha="center",
                va="center",
                transform=self.axis.transAxes,
            )
            self.axis.set_title(self.plot_request.title)
            self.axis.set_xlabel(self.plot_request.xlabel)
            self.axis.set_ylabel("Count")
            self.canvas.draw_idle()
            return

        combined_values = np.concatenate(
            [series["values"] for series in non_empty_series]
        )
        radial_dihedral_plot = self._dihedral_plot_style() == "radial"
        if radial_dihedral_plot:
            plot_series = self._canonical_dihedral_series_states(
                non_empty_series
            )
            plot_values = np.concatenate(
                [series["values"] for series in plot_series]
            )
            histogram_edges = self._dihedral_histogram_edges()
        else:
            plot_series = self._canonical_dihedral_series_states(
                non_empty_series
            )
            plot_values = np.concatenate(
                [series["values"] for series in plot_series]
            )
            if self.plot_request.category == "dihedral":
                histogram_edges = self._dihedral_histogram_edges()
            else:
                histogram_edges = self._histogram_edges(plot_values)
        stats = self._distribution_stats(
            combined_values,
            histogram_edges,
            display_values=plot_values,
        )

        if radial_dihedral_plot:
            self._draw_radial_dihedral_histogram(
                plot_series,
                histogram_edges,
                stats,
            )
            self.figure.tight_layout()
            self.canvas.draw_idle()
            return

        if len(plot_series) == 1:
            series = plot_series[0]
            self.axis.hist(
                series["values"],
                bins=histogram_edges,
                color=series["color"],
                edgecolor="black",
                linewidth=0.8,
                alpha=1.0,
                label=series["label"],
            )
        else:
            for index, series in enumerate(plot_series):
                self.axis.hist(
                    series["values"],
                    bins=histogram_edges,
                    color=series["color"],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=self.transparency_spin.value(),
                    label=series["label"],
                )

        for line in self._stat_reference_lines(stats):
            self.axis.axvline(
                line["value"],
                color=line["color"],
                linestyle="--",
                linewidth=1.4,
                label=line["label"],
            )

        self.axis.set_title(self.plot_request.title)
        self.axis.set_xlabel(self.plot_request.xlabel)
        self.axis.set_ylabel("Count")
        self._configure_category_axis(plot_values)
        self._configure_dihedral_cartesian_axis()

        legend = self.axis.legend(loc="upper right", frameon=True)
        stats_y = self._stats_box_y(legend)
        self.axis.text(
            0.98,
            stats_y,
            "\n".join(self._stats_box_lines(stats)),
            ha="right",
            va="top",
            transform=self.axis.transAxes,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.92,
            },
        )

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_radial_dihedral_histogram(
        self,
        plot_series: list[dict],
        histogram_edges: np.ndarray,
        stats: dict[str, object],
    ) -> None:
        theta_edges = np.radians(histogram_edges)
        theta_widths = np.diff(theta_edges)
        theta_centers = theta_edges[:-1] + theta_widths / 2.0
        alpha = (
            1.0
            if len(plot_series) == 1
            else float(self.transparency_spin.value())
        )
        max_count = 0.0
        for series in plot_series:
            counts, _edges = np.histogram(
                np.asarray(series["values"], dtype=float),
                bins=histogram_edges,
            )
            if counts.size == 0:
                continue
            max_count = max(max_count, float(np.max(counts)))
            self.axis.bar(
                theta_centers,
                counts,
                width=theta_widths,
                bottom=0.0,
                align="center",
                color=series["color"],
                edgecolor="black",
                linewidth=0.6,
                alpha=alpha,
                label=series["label"],
            )

        radial_limit = max(1.0, max_count) * 1.08
        self._configure_radial_dihedral_axis(radial_limit)

        for line in self._stat_reference_lines(stats):
            theta = math.radians(self._wrap_degree(float(line["value"])))
            self.axis.plot(
                [theta, theta],
                [0.0, radial_limit],
                color=line["color"],
                linestyle="--",
                linewidth=1.4,
                label=line["label"],
            )

        self.axis.set_title(self.plot_request.title, pad=18)
        legend = self.axis.legend(loc="upper right", frameon=True)
        stats_y = self._stats_box_y(legend)
        self.axis.text(
            0.98,
            stats_y,
            "\n".join(self._stats_box_lines(stats)),
            ha="right",
            va="top",
            transform=self.axis.transAxes,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.92,
            },
        )

    def _configure_radial_dihedral_axis(self, radial_limit: float) -> None:
        self.axis.set_theta_zero_location("E")
        self.axis.set_theta_direction(1)
        self.axis.set_thetagrids(
            np.arange(0, 360, 45),
            labels=("0", "45", "90", "135", "180", "-135", "-90", "-45"),
        )
        self.axis.set_rlabel_position(135)
        self.axis.set_ylim(0.0, radial_limit)
        self.axis.grid(True, alpha=0.35)

    def _update_overlay_controls(self, non_empty_series: list[dict]) -> None:
        show_overlay_controls = len(non_empty_series) > 1
        self.transparency_spin.setEnabled(show_overlay_controls)
        self.series_color_label.setVisible(show_overlay_controls)
        self.series_color_container.setVisible(show_overlay_controls)
        self._refresh_series_color_list(non_empty_series)

    def _refresh_series_color_list(
        self, non_empty_series: list[dict] | None = None
    ) -> None:
        series = (
            non_empty_series
            if non_empty_series is not None
            else self._ordered_series_states()
        )
        self._updating_series_list = True
        self.series_color_list.clear()
        if len(series) <= 1:
            self._updating_series_list = False
            return

        for entry in series:
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.ItemDataRole.UserRole, entry["key"])
            self._style_series_color_item(item, entry["color"])
            self.series_color_list.addItem(item)
        self._updating_series_list = False

    def _choose_series_color_for_item(self, item: QListWidgetItem) -> None:
        state = self._series_state_from_key(
            str(item.data(Qt.ItemDataRole.UserRole))
        )
        if state is None:
            return
        initial = QColor(state["color"])
        selected = QColorDialog.getColor(
            initial,
            self,
            f"Select color for {state['label']}",
        )
        if not selected.isValid():
            return
        state["color"] = selected.name()
        self._style_series_color_item(item, selected.name())
        self.refresh_plot()

    @staticmethod
    def _style_series_color_item(item: QListWidgetItem, color: str) -> None:
        qcolor = QColor(color)
        item.setBackground(qcolor)
        item.setForeground(
            QColor("black") if qcolor.lightnessF() > 0.62 else QColor("white")
        )

    def _default_bin_size(self) -> float:
        combined_values = np.concatenate(
            [
                series.values
                for series in self.plot_request.series
                if series.values.size > 0
            ]
            or [np.array([0.0, 1.0], dtype=float)]
        )
        if self.plot_request.category == "coordination":
            return 1.0
        if self.plot_request.category == "dihedral":
            return 5.0
        value_range = float(np.max(combined_values) - np.min(combined_values))
        if value_range <= 0:
            return 0.1
        recommended = value_range / 72.0
        return max(0.01, round(recommended, 3))

    def _default_series_colors(self) -> list[str]:
        category_color = CATEGORY_COLORS.get(self.plot_request.category)
        if category_color is not None and len(self.plot_request.series) <= 1:
            return [category_color]
        category_palette = CATEGORY_OVERLAY_COLORS.get(
            self.plot_request.category,
            (),
        )
        if category_palette:
            return [
                category_palette[index % len(category_palette)]
                for index in range(max(1, len(self.plot_request.series)))
            ]
        color_map = colormaps["tab10"]
        return [
            mcolors.to_hex(color_map(index % 10))
            for index in range(max(1, len(self.plot_request.series)))
        ]

    @staticmethod
    def _fallback_series_color(index: int) -> str:
        return mcolors.to_hex(colormaps["tab10"](index % 10))

    def _initial_series_states(self) -> list[dict[str, object]]:
        states: list[dict[str, object]] = []
        non_empty_series = [
            entry
            for entry in self.plot_request.series
            if entry.values.size > 0
        ]
        while len(self._series_colors) < len(non_empty_series):
            self._series_colors.append(
                self._fallback_series_color(len(self._series_colors))
            )
        for index, series in enumerate(non_empty_series):
            states.append(
                {
                    "key": f"series-{index}",
                    "label": series.label,
                    "values": series.values,
                    "color": self._series_colors[index],
                }
            )
        return states

    def _ordered_series_states(self) -> list[dict[str, object]]:
        if self.series_color_list.count() <= 1:
            return list(self._series_states)
        ordered_keys = [
            str(
                self.series_color_list.item(index).data(
                    Qt.ItemDataRole.UserRole
                )
            )
            for index in range(self.series_color_list.count())
        ]
        lookup = {str(entry["key"]): entry for entry in self._series_states}
        ordered = [lookup[key] for key in ordered_keys if key in lookup]
        if len(ordered) == len(self._series_states):
            return ordered
        return list(self._series_states)

    def _canonical_dihedral_series_states(
        self,
        series_states: list[dict],
    ) -> list[dict]:
        if self.plot_request.category != "dihedral":
            return list(series_states)
        display_series = []
        for series in series_states:
            display_entry = dict(series)
            display_entry["values"] = self._wrap_degrees(
                np.asarray(series["values"], dtype=float)
            )
            display_series.append(display_entry)
        return display_series

    def _series_state_from_key(self, key: str) -> dict[str, object] | None:
        for entry in self._series_states:
            if str(entry["key"]) == key:
                return entry
        return None

    def _on_series_order_changed(self, *_args) -> None:
        if self._updating_series_list:
            return
        self._series_states = self._ordered_series_states()
        self.refresh_plot()

    def _histogram_edges(self, values: np.ndarray) -> np.ndarray:
        bin_size = max(self.bin_size_spin.value(), 1e-6)
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if self.plot_request.category == "coordination":
            first_integer = max(0, math.floor(value_min))
            last_integer = max(first_integer, math.ceil(value_max))
            edges = np.arange(
                first_integer - 0.5,
                last_integer + 1.5,
                1.0,
                dtype=float,
            )
            if edges.size < 2:
                return np.array(
                    [float(first_integer), float(first_integer + 1)]
                )
            return edges
        if np.isclose(value_min, value_max):
            half_width = bin_size / 2.0
            return np.array([value_min - half_width, value_min + half_width])

        bin_count = max(1, int(np.ceil((value_max - value_min) / bin_size)))
        edges = value_min + np.arange(bin_count + 1) * bin_size
        if edges[-1] < value_max:
            edges = np.append(edges, edges[-1] + bin_size)
        return edges

    def _dihedral_histogram_edges(self) -> np.ndarray:
        bin_size = max(self.bin_size_spin.value(), 1e-6)
        bin_count = max(1, int(math.ceil(360.0 / bin_size)))
        return np.linspace(-180.0, 180.0, bin_count + 1, dtype=float)

    def _configure_category_axis(self, values: np.ndarray) -> None:
        if self.plot_request.category != "coordination" or values.size == 0:
            return
        first_integer = max(0, math.floor(float(np.min(values))))
        last_integer = max(first_integer, math.ceil(float(np.max(values))))
        left = max(0.0, first_integer - 0.5)
        right = max(left + 0.5, last_integer + 0.5)
        self.axis.set_xlim(left=left, right=right)
        self.axis.set_xticks(
            np.arange(first_integer, last_integer + 1, dtype=int)
        )

    def _configure_dihedral_cartesian_axis(self) -> None:
        if self.plot_request.category != "dihedral":
            return
        self.axis.set_xlim(-180.0, 180.0)
        self.axis.set_xticks(np.arange(-180.0, 181.0, 45.0))
        self.axis.xaxis.set_major_formatter(
            FuncFormatter(self._signed_degree_tick_label)
        )

    @classmethod
    def _signed_degree_tick_label(cls, value: float, _position: int) -> str:
        raw_value = float(value)
        if np.isclose(raw_value, -180.0):
            signed_value = -180.0
        elif np.isclose(raw_value, 180.0):
            signed_value = 180.0
        else:
            signed_value = cls._wrap_degree(raw_value)
        if np.isclose(signed_value, round(signed_value)):
            return str(int(round(signed_value)))
        return f"{signed_value:.3g}"

    def _distribution_stats(
        self,
        values: np.ndarray,
        edges: np.ndarray,
        *,
        display_values: np.ndarray | None = None,
    ) -> dict[str, object]:
        if self.plot_request.category == "dihedral":
            plot_values = values if display_values is None else display_values
            if self._dihedral_model() == "bimodal":
                return self._bimodal_dihedral_stats(
                    values,
                    edges,
                    display_values=plot_values,
                )
            return self._single_dihedral_stats(
                values,
                edges,
                display_values=plot_values,
            )
        return self._linear_distribution_stats(values, edges)

    @staticmethod
    def _linear_distribution_stats(
        values: np.ndarray, edges: np.ndarray
    ) -> dict[str, object]:
        if values.size == 0:
            return {
                "kind": "linear",
                "mean": 0.0,
                "median": 0.0,
                "mode": 0.0,
                "sigma": 0.0,
                "fwhm": 0.0,
            }
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        mode_value = BondAnalysisPlotTab._histogram_mode(values, edges)
        sigma = float(np.std(values, ddof=0))
        return {
            "kind": "linear",
            "mean": mean_value,
            "median": median_value,
            "mode": mode_value,
            "sigma": sigma,
            "fwhm": 2.355 * sigma,
        }

    def _single_dihedral_stats(
        self,
        values: np.ndarray,
        edges: np.ndarray,
        *,
        display_values: np.ndarray | None = None,
    ) -> dict[str, object]:
        if values.size == 0:
            return {
                "kind": "dihedral-single",
                "center": 0.0,
                "mode": 0.0,
                "sigma": 0.0,
                "fwhm": 0.0,
            }
        plot_values = values if display_values is None else display_values
        center = self._wrap_degree(self._circular_mean_degrees(values))
        sigma = self._circular_sigma_degrees(values)
        return {
            "kind": "dihedral-single",
            "center": center,
            "mode": self._histogram_mode(plot_values, edges),
            "sigma": sigma,
            "fwhm": 2.355 * sigma,
        }

    def _bimodal_dihedral_stats(
        self,
        values: np.ndarray,
        edges: np.ndarray,
        *,
        display_values: np.ndarray | None = None,
    ) -> dict[str, object]:
        plot_values = values if display_values is None else display_values
        components = self._bimodal_dihedral_components(
            values,
            edges,
            display_values=plot_values,
        )
        if not components:
            return self._single_dihedral_stats(
                values,
                edges,
                display_values=plot_values,
            )
        pooled_sigma = math.sqrt(
            sum(
                float(component["weight"]) * float(component["sigma"]) ** 2
                for component in components
            )
        )
        return {
            "kind": "dihedral-bimodal",
            "components": components,
            "sigma": pooled_sigma,
            "fwhm": 2.355 * pooled_sigma,
        }

    def _bimodal_dihedral_components(
        self,
        values: np.ndarray,
        edges: np.ndarray,
        *,
        display_values: np.ndarray | None = None,
    ) -> list[dict[str, float]]:
        plot_values = values if display_values is None else display_values
        if values.size == 0:
            return []
        if values.size == 1:
            center = float(plot_values[0])
            return [
                {
                    "center": center,
                    "mode": center,
                    "sigma": 0.0,
                    "count": 1.0,
                    "weight": 1.0,
                }
            ]

        first_center = self._histogram_mode(plot_values, edges)
        distances = self._circular_distance_degrees(values, first_center)
        second_center = float(values[int(np.argmax(distances))])
        if float(np.max(distances)) <= 1.0e-9:
            return [
                {
                    "center": self._wrap_degree(
                        self._circular_mean_degrees(values)
                    ),
                    "mode": first_center,
                    "sigma": self._circular_sigma_degrees(values),
                    "count": float(values.size),
                    "weight": 1.0,
                }
            ]

        centers = np.array([first_center, second_center], dtype=float)
        labels = np.zeros(values.size, dtype=int)
        for _iteration in range(40):
            distance_matrix = np.vstack(
                [
                    self._circular_distance_degrees(values, centers[0]),
                    self._circular_distance_degrees(values, centers[1]),
                ]
            )
            new_labels = np.argmin(distance_matrix, axis=0)
            if np.any(new_labels == 0) and np.any(new_labels == 1):
                labels = new_labels
            else:
                sorted_indices = np.argsort(distances)
                labels = np.zeros(values.size, dtype=int)
                labels[sorted_indices[values.size // 2 :]] = 1
            new_centers = centers.copy()
            for component_index in (0, 1):
                component_values = values[labels == component_index]
                if component_values.size:
                    new_centers[component_index] = self._circular_mean_degrees(
                        component_values
                    )
            if np.allclose(
                self._wrap_degrees(new_centers - centers),
                0.0,
                atol=1.0e-6,
            ):
                centers = new_centers
                break
            centers = new_centers

        components: list[dict[str, float]] = []
        for component_index in (0, 1):
            component_values = values[labels == component_index]
            component_display_values = plot_values[labels == component_index]
            if component_values.size == 0:
                continue
            components.append(
                {
                    "center": self._wrap_degree(
                        self._circular_mean_degrees(component_values)
                    ),
                    "mode": self._histogram_mode(
                        component_display_values,
                        edges,
                    ),
                    "sigma": self._circular_sigma_degrees(component_values),
                    "count": float(component_values.size),
                    "weight": float(component_values.size / values.size),
                }
            )
        components.sort(key=lambda component: component["center"])
        return components

    @staticmethod
    def _histogram_mode(values: np.ndarray, edges: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        if np.all(np.isclose(values, np.round(values))):
            unique_values, unique_counts = np.unique(
                values, return_counts=True
            )
            peak_count = int(np.max(unique_counts))
            return float(unique_values[unique_counts == peak_count][0])
        counts, histogram_edges = np.histogram(values, bins=edges)
        if counts.size == 0:
            return float(np.mean(values))
        peak_index = int(np.argmax(counts))
        return float(
            0.5
            * (histogram_edges[peak_index] + histogram_edges[peak_index + 1])
        )

    def _dihedral_model(self) -> str:
        if self.plot_request.category != "dihedral":
            return "linear"
        return str(self.dihedral_model_combo.currentData() or "single")

    def _dihedral_plot_style(self) -> str:
        if self.plot_request.category != "dihedral":
            return "normal"
        return str(self.dihedral_plot_style_combo.currentData() or "normal")

    @staticmethod
    def _wrap_degrees(values):
        wrapped = (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0
        return np.where(np.isclose(wrapped, -180.0), 180.0, wrapped)

    @classmethod
    def _wrap_degree(cls, value: float) -> float:
        return float(cls._wrap_degrees(np.asarray([value], dtype=float))[0])

    @classmethod
    def _circular_distance_degrees(
        cls,
        values: np.ndarray,
        center: float,
    ) -> np.ndarray:
        return np.abs(cls._wrap_degrees(values - float(center)))

    @classmethod
    def _circular_mean_degrees(cls, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        radians = np.radians(np.asarray(values, dtype=float))
        sine_mean = float(np.mean(np.sin(radians)))
        cosine_mean = float(np.mean(np.cos(radians)))
        return cls._wrap_degree(
            math.degrees(math.atan2(sine_mean, cosine_mean))
        )

    @staticmethod
    def _circular_sigma_degrees(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        radians = np.radians(np.asarray(values, dtype=float))
        sine_mean = float(np.mean(np.sin(radians)))
        cosine_mean = float(np.mean(np.cos(radians)))
        resultant_length = min(
            max(float(math.hypot(sine_mean, cosine_mean)), 1.0e-12),
            1.0,
        )
        return math.degrees(
            math.sqrt(max(-2.0 * math.log(resultant_length), 0.0))
        )

    @staticmethod
    def _stat_reference_lines(
        stats: dict[str, object]
    ) -> list[dict[str, object]]:
        kind = stats.get("kind")
        if kind == "linear":
            return [
                {
                    "value": float(stats["mean"]),
                    "color": "black",
                    "label": "Mean",
                },
                {
                    "value": float(stats["median"]),
                    "color": "red",
                    "label": "Median",
                },
            ]
        if kind == "dihedral-single":
            return [
                {
                    "value": float(stats["center"]),
                    "color": "black",
                    "label": "Circular center",
                },
                {
                    "value": float(stats["mode"]),
                    "color": "red",
                    "label": "Mode",
                },
            ]
        if kind == "dihedral-bimodal":
            colors = ("black", "red")
            lines = []
            for index, component in enumerate(
                stats.get("components", []),
                start=1,
            ):
                lines.append(
                    {
                        "value": float(component["center"]),
                        "color": colors[(index - 1) % len(colors)],
                        "label": f"Center {index}",
                    }
                )
            return lines
        return []

    @classmethod
    def _stats_box_lines(cls, stats: dict[str, object]) -> list[str]:
        kind = stats.get("kind")
        if kind == "linear":
            return [
                f"Mean: {float(stats['mean']):.3f}",
                f"Median: {float(stats['median']):.3f}",
                f"Mode: {float(stats['mode']):.3f}",
                f"Width sigma: {float(stats['sigma']):.3f}",
                f"FWHM: {float(stats['fwhm']):.3f}",
            ]
        if kind == "dihedral-single":
            return [
                "Model: single circular",
                "Circular center: "
                f"{cls._wrap_degree(float(stats['center'])):.3f}",
                f"Mode: {cls._wrap_degree(float(stats['mode'])):.3f}",
                f"Width sigma: {float(stats['sigma']):.3f}",
                f"FWHM: {float(stats['fwhm']):.3f}",
            ]
        if kind == "dihedral-bimodal":
            lines = ["Model: bimodal circular"]
            for index, component in enumerate(
                stats.get("components", []),
                start=1,
            ):
                lines.append(
                    "C"
                    f"{index}: "
                    f"{cls._wrap_degree(float(component['center'])):.3f}, "
                    f"sigma {float(component['sigma']):.3f}, "
                    f"n {int(component['count'])}"
                )
            lines.append(f"Pooled width sigma: {float(stats['sigma']):.3f}")
            lines.append(f"Pooled FWHM: {float(stats['fwhm']):.3f}")
            return lines
        return []

    @staticmethod
    def _stats_box_y(legend) -> float:
        if legend is None:
            return 0.98
        legend_entries = max(1, len(legend.get_texts()))
        return max(0.18, 0.98 - 0.075 * legend_entries)


class BondAnalysisPlotWindow(QMainWindow):
    """Shared tabbed bondanalysis plotting workspace."""

    def __init__(
        self,
        plot_request: BondAnalysisPlotRequest,
        default_output_dir: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.default_output_dir = Path(default_output_dir)
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.tab_widget.currentChanged.connect(self._sync_active_tab)
        self.setCentralWidget(self.tab_widget)
        self.resize(1040, 820)

        self._next_tab_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Right), self
        )
        self._next_tab_shortcut.activated.connect(self._select_next_tab)
        self._previous_tab_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Left), self
        )
        self._previous_tab_shortcut.activated.connect(
            self._select_previous_tab
        )

        self.add_plot_request(plot_request)

    def add_plot_request(self, plot_request: BondAnalysisPlotRequest) -> None:
        plot_tab = BondAnalysisPlotTab(
            plot_request,
            default_output_dir=self.default_output_dir,
            parent=self,
        )
        tab_index = self.tab_widget.addTab(
            plot_tab, self._unique_tab_label(plot_request.title)
        )
        self.tab_widget.setCurrentIndex(tab_index)
        self._sync_active_tab(tab_index)

    @property
    def current_plot_tab(self) -> BondAnalysisPlotTab | None:
        widget = self.tab_widget.currentWidget()
        if isinstance(widget, BondAnalysisPlotTab):
            return widget
        return None

    @property
    def plot_request(self) -> BondAnalysisPlotRequest:
        current = self.current_plot_tab
        if current is None:
            raise RuntimeError("No active bondanalysis plot tab is available.")
        return current.plot_request

    @property
    def controls_widget(self) -> QWidget:
        return self.current_plot_tab.controls_widget

    @property
    def bin_size_spin(self) -> QDoubleSpinBox:
        return self.current_plot_tab.bin_size_spin

    @property
    def transparency_spin(self) -> QDoubleSpinBox:
        return self.current_plot_tab.transparency_spin

    @property
    def series_color_container(self) -> QWidget:
        return self.current_plot_tab.series_color_container

    @property
    def series_color_list(self) -> QListWidget:
        return self.current_plot_tab.series_color_list

    @property
    def dihedral_model_combo(self) -> QComboBox:
        return self.current_plot_tab.dihedral_model_combo

    @property
    def dihedral_plot_style_combo(self) -> QComboBox:
        return self.current_plot_tab.dihedral_plot_style_combo

    @property
    def axis(self):
        return self.current_plot_tab.axis

    def save_plot_data_as(self) -> None:
        current = self.current_plot_tab
        if current is not None:
            current.save_plot_data_as()

    def refresh_plot(self) -> None:
        current = self.current_plot_tab
        if current is not None:
            current.refresh_plot()

    def _close_tab(self, index: int) -> None:
        widget = self.tab_widget.widget(index)
        self.tab_widget.removeTab(index)
        if widget is not None:
            widget.deleteLater()
        if self.tab_widget.count() == 0:
            self.close()
            return
        self._sync_active_tab(self.tab_widget.currentIndex())

    def _select_next_tab(self) -> None:
        count = self.tab_widget.count()
        if count <= 1:
            return
        next_index = (self.tab_widget.currentIndex() + 1) % count
        self.tab_widget.setCurrentIndex(next_index)

    def _select_previous_tab(self) -> None:
        count = self.tab_widget.count()
        if count <= 1:
            return
        next_index = (self.tab_widget.currentIndex() - 1) % count
        self.tab_widget.setCurrentIndex(next_index)

    def _sync_active_tab(self, _index: int) -> None:
        current = self.current_plot_tab
        if current is None:
            self.setWindowTitle("Bond Analysis Plots")
            return
        self.setWindowTitle(
            f"Bond Analysis Plots - {current.plot_request.title}"
        )

    def _unique_tab_label(self, title: str) -> str:
        base = self._tab_label(title)
        existing = {
            self.tab_widget.tabText(index)
            for index in range(self.tab_widget.count())
        }
        if base not in existing:
            return base
        suffix = 2
        while f"{base} ({suffix})" in existing:
            suffix += 1
        return f"{base} ({suffix})"

    @staticmethod
    def _tab_label(title: str) -> str:
        if len(title) <= 36:
            return title
        return title[:33] + "..."
