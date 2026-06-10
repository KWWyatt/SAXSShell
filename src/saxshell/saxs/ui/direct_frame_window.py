from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.direct_frames import (
    DirectFrameSAXSResult,
    compute_direct_frame_saxs,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
    track_saxshell_window,
)

_OPEN_WINDOWS: list["DirectFrameSAXSWindow"] = []
_SCALE_FIT_2PI = "2*pi/L"
_SCALE_FIT_4PI = "4*pi/L"
_SCALE_FIT_MANUAL = "Manual q min"


@dataclass(frozen=True, slots=True)
class DirectFrameSAXSRunRequest:
    input_path: Path
    output_dir: Path
    q_min: float
    q_max: float
    q_step: float
    max_frames: int | None
    box_lengths_a: tuple[float, float, float]
    subtract_average_box_density: bool
    direction_count: int
    write_plots: bool
    experimental_project_dir: Path | None
    experimental_data_path: Path | None
    scale_fit_q_min: float | None
    scale_fit_q_max: float | None


class DirectFrameSAXSWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, request: DirectFrameSAXSRunRequest) -> None:
        super().__init__()
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            result = compute_direct_frame_saxs(
                self.request.input_path,
                output_dir=self.request.output_dir,
                q_min=self.request.q_min,
                q_max=self.request.q_max,
                q_step=self.request.q_step,
                max_frames=self.request.max_frames,
                box_lengths_a=self.request.box_lengths_a,
                subtract_average_box_density=(
                    self.request.subtract_average_box_density
                ),
                direction_count=self.request.direction_count,
                write_plots=self.request.write_plots,
                experimental_project_dir=(
                    self.request.experimental_project_dir
                ),
                experimental_data_path=self.request.experimental_data_path,
                scale_fit_q_min=self.request.scale_fit_q_min,
                scale_fit_q_max=self.request.scale_fit_q_max,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class DirectFrameSAXSWindow(QMainWindow):
    """Standalone UI for direct finite-frame SAXS screening runs."""

    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_input_path: str | Path | None = None,
        initial_output_dir: str | Path | None = None,
        initial_experimental_data_file: str | Path | None = None,
        initial_q_min: float | None = None,
        initial_q_max: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_thread: QThread | None = None
        self._run_worker: DirectFrameSAXSWorker | None = None
        self._last_result: DirectFrameSAXSResult | None = None
        self._build_ui()
        self._apply_initial_values(
            initial_project_dir=initial_project_dir,
            initial_input_path=initial_input_path,
            initial_output_dir=initial_output_dir,
            initial_experimental_data_file=initial_experimental_data_file,
            initial_q_min=initial_q_min,
            initial_q_max=initial_q_max,
        )

    def closeEvent(self, event) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            QMessageBox.warning(
                self,
                "Direct Frame SAXS",
                "Please wait for the active direct-frame SAXS run to finish "
                "before closing this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (Direct Frame SAXS Beta)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(980, 760)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        input_group = QGroupBox("Inputs")
        input_layout = QFormLayout(input_group)
        self.input_path_edit = QLineEdit()
        input_layout.addRow(
            "PDB/XYZ frame file or folder",
            self._path_row(self.input_path_edit, self._choose_input_path),
        )
        self.output_dir_edit = QLineEdit()
        input_layout.addRow(
            "Output folder",
            self._path_row(self.output_dir_edit, self._choose_output_dir),
        )
        self.experimental_project_edit = QLineEdit()
        input_layout.addRow(
            "Experimental project",
            self._path_row(
                self.experimental_project_edit,
                self._choose_experimental_project,
            ),
        )
        self.experimental_data_edit = QLineEdit()
        input_layout.addRow(
            "Experimental data file",
            self._path_row(
                self.experimental_data_edit,
                self._choose_experimental_data_file,
                file_dialog=True,
            ),
        )
        root.addWidget(input_group)

        q_group = QGroupBox("q Grid and Frames")
        q_layout = QFormLayout(q_group)
        self.q_min_spin = self._float_spin(0.0, 100.0, 0.049952293, 9)
        self.q_max_spin = self._float_spin(0.001, 100.0, 2.0015939, 9)
        self.q_step_spin = self._float_spin(1.0e-6, 10.0, 0.01, 9)
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 10_000_000)
        self.max_frames_spin.setSpecialValueText("All")
        self.max_frames_spin.setValue(20)
        q_layout.addRow("q min (A^-1)", self.q_min_spin)
        q_layout.addRow("q max (A^-1)", self.q_max_spin)
        q_layout.addRow("q step (A^-1)", self.q_step_spin)
        q_layout.addRow("Max frames", self.max_frames_spin)
        root.addWidget(q_group)

        contrast_group = QGroupBox("Average Box-Density Contrast")
        contrast_layout = QFormLayout(contrast_group)
        self.subtract_average_box_density_checkbox = QCheckBox(
            "Subtract average box electron density"
        )
        self.subtract_average_box_density_checkbox.setChecked(True)
        contrast_layout.addRow(
            "Contrast mode",
            self.subtract_average_box_density_checkbox,
        )
        box_row = QHBoxLayout()
        self.box_lx_edit = self._box_dimension_edit("Lx")
        self.box_ly_edit = self._box_dimension_edit("Ly")
        self.box_lz_edit = self._box_dimension_edit("Lz")
        box_row.addWidget(QLabel("Lx"))
        box_row.addWidget(self.box_lx_edit)
        box_row.addWidget(QLabel("Ly"))
        box_row.addWidget(self.box_ly_edit)
        box_row.addWidget(QLabel("Lz"))
        box_row.addWidget(self.box_lz_edit)
        contrast_layout.addRow("Box dimensions (A)", box_row)
        self.direction_count_spin = QSpinBox()
        self.direction_count_spin.setRange(1, 1_000_000)
        self.direction_count_spin.setValue(256)
        contrast_layout.addRow(
            "Spherical directions",
            self.direction_count_spin,
        )
        root.addWidget(contrast_group)

        scale_group = QGroupBox("Experimental Overlay Scaling")
        scale_layout = QFormLayout(scale_group)
        self.write_plots_checkbox = QCheckBox("Write plots and overlay CSV")
        self.write_plots_checkbox.setChecked(True)
        scale_layout.addRow("Outputs", self.write_plots_checkbox)
        self.scale_fit_mode_combo = QComboBox()
        self.scale_fit_mode_combo.addItems(
            [_SCALE_FIT_4PI, _SCALE_FIT_2PI, _SCALE_FIT_MANUAL]
        )
        self.scale_fit_mode_combo.setCurrentText(_SCALE_FIT_4PI)
        self.scale_fit_mode_combo.currentTextChanged.connect(
            self._update_manual_scale_controls
        )
        scale_layout.addRow("Scale-fit q min", self.scale_fit_mode_combo)
        self.manual_scale_q_min_spin = self._float_spin(0.0, 100.0, 0.0, 9)
        scale_layout.addRow(
            "Manual q min (A^-1)",
            self.manual_scale_q_min_spin,
        )
        self.scale_q_max_spin = self._float_spin(0.0, 100.0, 0.0, 9)
        self.scale_q_max_spin.setSpecialValueText("No upper bound")
        scale_layout.addRow("Scale-fit q max (A^-1)", self.scale_q_max_spin)
        root.addWidget(scale_group)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Run Direct Frame SAXS")
        self.run_button.clicked.connect(self._run_calculation)
        button_row.addWidget(self.run_button)
        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.clicked.connect(self._open_output_dir)
        button_row.addWidget(self.open_output_button)
        self.clear_button = QPushButton("Clear Log")
        button_row.addWidget(self.clear_button)
        button_row.addStretch(1)
        root.addLayout(button_row)

        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(220)
        self.clear_button.clicked.connect(self.output_box.clear)
        root.addWidget(self.output_box, stretch=1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self._update_manual_scale_controls(
            self.scale_fit_mode_combo.currentText()
        )

    def _apply_initial_values(
        self,
        *,
        initial_project_dir: str | Path | None,
        initial_input_path: str | Path | None,
        initial_output_dir: str | Path | None,
        initial_experimental_data_file: str | Path | None,
        initial_q_min: float | None,
        initial_q_max: float | None,
    ) -> None:
        project_path = self._optional_path(initial_project_dir)
        input_path = self._optional_path(initial_input_path)
        output_dir = self._optional_path(initial_output_dir)
        experimental_data_file = self._optional_path(
            initial_experimental_data_file
        )
        if project_path is not None:
            self.experimental_project_edit.setText(str(project_path))
        if input_path is not None:
            self.input_path_edit.setText(str(input_path))
        if output_dir is None and project_path is not None:
            output_dir = project_path / "direct_frame_saxs_runs"
        if output_dir is not None:
            self.output_dir_edit.setText(str(output_dir))
        if experimental_data_file is not None:
            self.experimental_data_edit.setText(str(experimental_data_file))
        if initial_q_min is not None:
            self.q_min_spin.setValue(float(initial_q_min))
        if initial_q_max is not None:
            self.q_max_spin.setValue(float(initial_q_max))

    @staticmethod
    def _optional_path(value: str | Path | None) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _path_row(
        self,
        line_edit: QLineEdit,
        callback,
        *,
        file_dialog: bool = False,
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, stretch=1)
        button = QPushButton("File..." if file_dialog else "Browse...")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    @staticmethod
    def _float_spin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setValue(float(value))
        spin.setSingleStep(0.01)
        return spin

    @staticmethod
    def _box_dimension_edit(label: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(f"{label} in A")
        edit.setMaximumWidth(120)
        return edit

    def _choose_input_path(self) -> None:
        start = self.input_path_edit.text().strip()
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select PDB/XYZ frame folder",
            start,
        )
        if not selected:
            file_path, _selected_filter = QFileDialog.getOpenFileName(
                self,
                "Select PDB/XYZ structure file",
                start,
                "Structure files (*.xyz *.pdb);;All files (*)",
            )
            selected = file_path
        if selected:
            self.input_path_edit.setText(selected)

    def _choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_dir_edit.text().strip(),
        )
        if selected:
            self.output_dir_edit.setText(selected)

    def _choose_experimental_project(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select SAXSShell experimental project",
            self.experimental_project_edit.text().strip(),
        )
        if selected:
            self.experimental_project_edit.setText(selected)

    def _choose_experimental_data_file(self) -> None:
        selected, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select experimental SAXS data",
            self.experimental_data_edit.text().strip(),
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if selected:
            self.experimental_data_edit.setText(selected)

    def _update_manual_scale_controls(self, mode: str) -> None:
        self.manual_scale_q_min_spin.setEnabled(mode == _SCALE_FIT_MANUAL)

    def _collect_run_request(self) -> DirectFrameSAXSRunRequest:
        input_path = self._required_path(
            self.input_path_edit,
            "Select a PDB/XYZ frame file or folder.",
        )
        output_dir = self._required_path(
            self.output_dir_edit,
            "Select an output folder.",
        )
        q_min = float(self.q_min_spin.value())
        q_max = float(self.q_max_spin.value())
        q_step = float(self.q_step_spin.value())
        if q_max <= q_min:
            raise ValueError("q max must be greater than q min.")
        if q_step <= 0.0:
            raise ValueError("q step must be positive.")
        box_lengths = self._box_lengths_value()
        return DirectFrameSAXSRunRequest(
            input_path=input_path,
            output_dir=output_dir,
            q_min=q_min,
            q_max=q_max,
            q_step=q_step,
            max_frames=(
                None
                if self.max_frames_spin.value() == 0
                else int(self.max_frames_spin.value())
            ),
            box_lengths_a=box_lengths,
            subtract_average_box_density=bool(
                self.subtract_average_box_density_checkbox.isChecked()
            ),
            direction_count=int(self.direction_count_spin.value()),
            write_plots=bool(self.write_plots_checkbox.isChecked()),
            experimental_project_dir=self._optional_line_edit_path(
                self.experimental_project_edit
            ),
            experimental_data_path=self._optional_line_edit_path(
                self.experimental_data_edit
            ),
            scale_fit_q_min=self._scale_fit_q_min_value(box_lengths),
            scale_fit_q_max=(
                None
                if float(self.scale_q_max_spin.value()) <= 0.0
                else float(self.scale_q_max_spin.value())
            ),
        )

    @staticmethod
    def _required_path(line_edit: QLineEdit, message: str) -> Path:
        text = line_edit.text().strip()
        if not text:
            raise ValueError(message)
        return Path(text).expanduser().resolve()

    @staticmethod
    def _optional_line_edit_path(line_edit: QLineEdit) -> Path | None:
        text = line_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _box_lengths_value(self) -> tuple[float, float, float]:
        values: list[float] = []
        for label, edit in (
            ("Lx", self.box_lx_edit),
            ("Ly", self.box_ly_edit),
            ("Lz", self.box_lz_edit),
        ):
            text = edit.text().strip()
            if not text:
                raise ValueError(
                    "Enter all three box dimensions before running "
                    "average-box-density contrast."
                )
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(f"{label} must be a number.") from exc
            if value <= 0.0:
                raise ValueError(f"{label} must be positive.")
            values.append(value)
        return (values[0], values[1], values[2])

    def _scale_fit_q_min_value(
        self,
        box_lengths: tuple[float, float, float],
    ) -> float | None:
        mode = self.scale_fit_mode_combo.currentText()
        if mode == _SCALE_FIT_MANUAL:
            value = float(self.manual_scale_q_min_spin.value())
            return None if value <= 0.0 else value
        diagnostic_length = max(float(value) for value in box_lengths)
        q_2pi = 2.0 * math.pi / diagnostic_length
        if mode == _SCALE_FIT_4PI:
            return 2.0 * q_2pi
        return q_2pi

    def _run_calculation(self) -> None:
        try:
            request = self._collect_run_request()
        except Exception as exc:
            QMessageBox.warning(self, "Direct Frame SAXS", str(exc))
            return
        self._set_running(True)
        self._last_result = None
        self.output_box.setPlainText(
            "Running direct-frame SAXS...\n"
            f"Input: {request.input_path}\n"
            f"Output: {request.output_dir}\n"
            f"Box dimensions: {request.box_lengths_a}\n"
            f"Scale-fit q min: {request.scale_fit_q_min}\n"
        )
        self._run_thread = QThread(self)
        self._run_worker = DirectFrameSAXSWorker(request)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.failed.connect(self._on_run_failed)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_thread.finished.connect(self._cleanup_run_thread)
        self._run_thread.start()

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.statusBar().showMessage(
            "Running direct-frame SAXS..." if running else "Ready"
        )

    @Slot(object)
    def _on_run_finished(self, result: DirectFrameSAXSResult) -> None:
        self._last_result = result
        lines = [
            f"Computed {len(result.frame_paths)} frame(s).",
            f"Calculation mode: {result.calculation_mode}",
            f"Profile: {result.profile_csv_path}",
            f"Profile text: {result.profile_txt_path}",
            f"Frame traces: {result.frame_trace_csv_path}",
            f"Metadata: {result.metadata_json_path}",
            f"Method notes: {result.method_notes_path}",
            (
                "Finite-box diagnostic: "
                f"2*pi/L = {result.diagnostics.q_fundamental_a_inverse:.6g} "
                "A^-1"
            ),
            (
                "Finite-box caution: "
                f"4*pi/L = {result.diagnostics.q_caution_a_inverse:.6g} "
                "A^-1"
            ),
        ]
        if result.medium_density_e_per_a3 is not None:
            lines.append(
                "Average medium density: "
                f"{result.medium_density_e_per_a3:.8g} e/A^3"
            )
        if result.experimental_overlay is not None:
            overlay = result.experimental_overlay
            lines.extend(
                [
                    f"Experimental overlay: {overlay.plot_path}",
                    f"Overlay CSV: {overlay.scaled_profile_csv_path}",
                    (
                        "Overlay log-scale factor: "
                        f"{overlay.scale_factor:.8g} over "
                        f"{overlay.fit_q_min_a_inverse:.6g}-"
                        f"{overlay.fit_q_max_a_inverse:.6g} A^-1 "
                        f"({overlay.fit_point_count} points)."
                    ),
                    (
                        "Overlay log RMS residual: "
                        f"{overlay.log_rms_residual:.8g}"
                    ),
                ]
            )
        for figure_path in result.figures:
            lines.append(f"Figure: {figure_path}")
        self.output_box.setPlainText("\n".join(lines))
        self.statusBar().showMessage("Direct-frame SAXS complete")
        self._set_running(False)

    @Slot(str)
    def _on_run_failed(self, message: str) -> None:
        self.output_box.setPlainText(
            "Direct-frame SAXS failed:\n" f"{message}"
        )
        self.statusBar().showMessage("Direct-frame SAXS failed")
        self._set_running(False)

    @Slot()
    def _cleanup_run_thread(self) -> None:
        if self._run_worker is not None:
            self._run_worker.deleteLater()
        if self._run_thread is not None:
            self._run_thread.deleteLater()
        self._run_worker = None
        self._run_thread = None

    def _open_output_dir(self) -> None:
        text = self.output_dir_edit.text().strip()
        if not text:
            return
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(text).expanduser()))
        )


def launch_direct_frame_saxs_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
    initial_output_dir: str | Path | None = None,
    initial_experimental_data_file: str | Path | None = None,
    initial_q_min: float | None = None,
    initial_q_max: float | None = None,
) -> DirectFrameSAXSWindow:
    window = DirectFrameSAXSWindow(
        initial_project_dir=initial_project_dir,
        initial_input_path=initial_input_path,
        initial_output_dir=initial_output_dir,
        initial_experimental_data_file=initial_experimental_data_file,
        initial_q_min=initial_q_min,
        initial_q_max=initial_q_max,
    )
    track_saxshell_window(window, _OPEN_WINDOWS)
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def main() -> int:
    prepare_saxshell_application_identity()
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    configure_saxshell_application(app)
    launch_direct_frame_saxs_ui()
    return int(app.exec()) if owns_app else 0


if __name__ == "__main__":
    raise SystemExit(main())
