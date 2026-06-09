from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
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

from saxshell.exafs.gds import artemis_gds_overview_path
from saxshell.exafs.mapping import (
    EXAFSAngleAnnotation,
    EXAFSBondAnalysisResult,
    EXAFSRepresentativeOption,
    EXAFSScatteringPathEvent,
    EXAFSStructurePreview,
    build_gds_mapping_document,
    default_absorber_element,
    discover_bondanalysis_results,
    discover_representative_structures,
    gds_registry_entries_for_stoichiometry,
    load_bondanalysis_result,
    load_structure_preview,
    scattering_path_events_from_preview,
    write_gds_mapping_file,
    write_padded_cif_from_structure,
)
from saxshell.saxs.electron_density_mapping.workflow import (
    load_electron_density_structure,
)
from saxshell.saxs.structure_viewer.ui.widget import StructureViewerWidget
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
    track_saxshell_window,
)

_OPEN_WINDOWS: list["EXAFSGDSMappingMainWindow"] = []
_PATH_TABLE_DEFAULT_GROUP_KEY = "solvent_molecule"
_PATH_TABLE_NO_GROUP_KEY = "__none__"
_PATH_TABLE_FILTER_ALL_KEY = "__all__"
_PATH_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("solvent_molecule", "Molecule"),
    ("path", "Path"),
    ("absorber", "Absorber"),
    ("scatterer", "Scatterer"),
    ("degeneracy", "Deg."),
    ("path_length", "Path Length (A)"),
    ("reff", "Reff (A)"),
    ("bonds", "Bonds"),
    ("angles", "Angles"),
    ("dihedrals", "Dihedrals"),
)
_PATH_TABLE_GROUP_OPTIONS: tuple[tuple[str, str], ...] = (
    (_PATH_TABLE_DEFAULT_GROUP_KEY, "Solvent Molecule"),
    ("absorber", "Absorber"),
    ("scatterer", "Scatterer"),
    ("scatterer_element", "Scatterer Element"),
    (_PATH_TABLE_NO_GROUP_KEY, "No Grouping"),
)


@dataclass(frozen=True, slots=True)
class _Deferred2DLabel:
    text: str
    anchor: np.ndarray
    color: str
    fontsize: float
    zorder: int
    priority: int
    bbox: dict[str, object] | None = None
    connector: bool = True
    offset_scale: float = 1.0
    fontweight: str = "normal"
    required: bool = False


@dataclass(frozen=True, slots=True)
class _PlotAnnotationVisibility:
    bond_distances: bool = True
    bond_angles: bool = False
    dihedral_angles: bool = False


class EXAFSGDSMappingMainWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_absorber_element: str | None = None,
    ) -> None:
        super().__init__()
        self._project_dir: Path | None = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._representatives: tuple[EXAFSRepresentativeOption, ...] = ()
        self._bondanalysis_results: tuple[EXAFSBondAnalysisResult, ...] = ()
        self._active_preview: EXAFSStructurePreview | None = None
        self._active_scattering_path_events: tuple[
            EXAFSScatteringPathEvent,
            ...,
        ] = ()
        self._scattering_path_check_states_by_key: dict[
            tuple[int, int],
            Qt.CheckState,
        ] = {}
        self._active_registry_entries: tuple[dict[str, object], ...] = ()
        self._suppress_selection_refresh = False
        self._preferred_absorber_element = _normalize_element_text(
            initial_absorber_element
        )
        self._auto_absorber_element: str | None = None

        self.setWindowTitle("EXAFS GDS Mapping")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1420, 920)
        self._build_menu_bar()
        self._build_ui()
        if self._preferred_absorber_element:
            self._auto_absorber_element = self._preferred_absorber_element
            self.absorber_element_edit.setText(
                self._preferred_absorber_element
            )

        if self._project_dir is not None:
            self.project_dir_edit.setText(str(self._project_dir))
            self._load_project_context()

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_project_action = QAction("Open Project...", self)
        open_project_action.triggered.connect(self._choose_project_dir)
        file_menu.addAction(open_project_action)
        file_menu.addSeparator()
        close_action = QAction("Close", self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        root_layout.addWidget(splitter, stretch=1)

        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        left_layout.addWidget(self._build_project_group())
        left_layout.addWidget(self._build_representative_group())
        left_layout.addWidget(self._build_plot_annotation_group())
        left_layout.addWidget(self._build_bondanalysis_group())
        left_layout.addWidget(self._build_build_group())
        left_layout.addStretch(1)
        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        self.preview_tabs = QTabWidget(self)
        self.structure_viewer = StructureViewerWidget(self.preview_tabs)
        self.preview_tabs.addTab(self.structure_viewer, "Structure")

        self.path3d_figure = Figure(figsize=(7.5, 5.6))
        self.path3d_canvas = FigureCanvas(self.path3d_figure)
        path3d_tab, self.path3d_toolbar = self._build_plot_canvas_tab(
            self.path3d_canvas
        )
        self.preview_tabs.addTab(path3d_tab, "3D Paths")

        self.path2d_figure = Figure(figsize=(7.5, 5.6))
        self.path2d_canvas = FigureCanvas(self.path2d_figure)
        path2d_tab, self.path2d_toolbar = self._build_plot_canvas_tab(
            self.path2d_canvas
        )
        self.preview_tabs.addTab(path2d_tab, "2D Projection")

        scattering_paths_tab = QWidget()
        scattering_paths_layout = QVBoxLayout(scattering_paths_tab)
        scattering_paths_layout.setContentsMargins(8, 8, 8, 8)
        scattering_paths_controls = QHBoxLayout()
        self.path_group_combo = QComboBox(scattering_paths_tab)
        for key, label in _PATH_TABLE_GROUP_OPTIONS:
            self.path_group_combo.addItem(label, key)
        self.path_group_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_scattering_path_table()
        )
        scattering_paths_controls.addWidget(QLabel("Group by"))
        scattering_paths_controls.addWidget(self.path_group_combo)
        self.reset_path_grouping_button = QPushButton(
            "Reset Molecule Grouping"
        )
        self.reset_path_grouping_button.setToolTip(
            "Restore the default grouping so paths from the same solvent "
            "molecule appear together."
        )
        self.reset_path_grouping_button.clicked.connect(
            self._reset_scattering_path_grouping
        )
        scattering_paths_controls.addWidget(self.reset_path_grouping_button)
        scattering_paths_controls.addSpacing(12)
        self.path_filter_column_combo = QComboBox(scattering_paths_tab)
        self.path_filter_column_combo.addItem(
            "All columns",
            _PATH_TABLE_FILTER_ALL_KEY,
        )
        for key, label in _PATH_TABLE_COLUMNS:
            self.path_filter_column_combo.addItem(label, key)
        self.path_filter_column_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_scattering_path_table()
        )
        scattering_paths_controls.addWidget(QLabel("Filter"))
        scattering_paths_controls.addWidget(self.path_filter_column_combo)
        self.path_filter_edit = QLineEdit(scattering_paths_tab)
        self.path_filter_edit.setPlaceholderText("Type to filter paths")
        self.path_filter_edit.textChanged.connect(
            lambda _text: self._refresh_scattering_path_table()
        )
        scattering_paths_controls.addWidget(self.path_filter_edit, stretch=1)
        self.clear_path_filter_button = QPushButton("Clear")
        self.clear_path_filter_button.clicked.connect(
            self.path_filter_edit.clear
        )
        scattering_paths_controls.addWidget(self.clear_path_filter_button)
        scattering_paths_layout.addLayout(scattering_paths_controls)
        self.path_filter_status_label = QLabel("No paths loaded.")
        self.path_filter_status_label.setWordWrap(True)
        scattering_paths_layout.addWidget(self.path_filter_status_label)
        self.scattering_path_table = QTableWidget(
            0,
            1 + len(_PATH_TABLE_COLUMNS),
            scattering_paths_tab,
        )
        self.scattering_path_table.setHorizontalHeaderLabels(
            ("Use",) + tuple(label for _key, label in _PATH_TABLE_COLUMNS)
        )
        self.scattering_path_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.scattering_path_table.setWordWrap(False)
        scattering_paths_layout.addWidget(self.scattering_path_table)
        self.preview_tabs.addTab(scattering_paths_tab, "Scattering Paths")

        variables_tab = QWidget()
        variables_layout = QVBoxLayout(variables_tab)
        variables_layout.setContentsMargins(8, 8, 8, 8)
        self.variable_table = QTableWidget(0, 6, variables_tab)
        self.variable_table.setHorizontalHeaderLabels(
            ("Use", "Scope", "Type", "Distribution", "Variables", "Set Rows")
        )
        self.variable_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.variable_table.setWordWrap(False)
        variables_layout.addWidget(self.variable_table)
        self.preview_tabs.addTab(variables_tab, "GDS Variables")

        gds_tab = QWidget()
        gds_layout = QVBoxLayout(gds_tab)
        gds_layout.setContentsMargins(8, 8, 8, 8)
        self.validation_label = QLabel("Build or preview a GDS file.")
        self.validation_label.setWordWrap(True)
        gds_layout.addWidget(self.validation_label)
        self.gds_preview_text = QTextEdit()
        self.gds_preview_text.setReadOnly(True)
        self.gds_preview_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        gds_layout.addWidget(self.gds_preview_text, stretch=1)
        self.preview_tabs.addTab(gds_tab, "GDS Preview")

        splitter.addWidget(self.preview_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 960])

        self.statusBar().showMessage("Ready")
        self._draw_empty_path_plots("Load a representative structure.")

    def _build_plot_canvas_tab(
        self,
        canvas: FigureCanvas,
    ) -> tuple[QWidget, NavigationToolbar2QT]:
        tab = QWidget(self.preview_tabs)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = NavigationToolbar2QT(canvas, tab)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, stretch=1)
        return tab, toolbar

    def _build_project_group(self) -> QWidget:
        group = QGroupBox("Project")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.setPlaceholderText("SAXS project directory")
        self.project_dir_edit.returnPressed.connect(self._load_project_context)
        row.addWidget(self.project_dir_edit, stretch=1)
        browse_button = QPushButton("Choose")
        browse_button.clicked.connect(self._choose_project_dir)
        row.addWidget(browse_button)
        layout.addLayout(row)
        load_button = QPushButton("Load Project Context")
        load_button.clicked.connect(self._load_project_context)
        layout.addWidget(load_button)
        self.project_status_label = QLabel(
            "Load a project with representative structures and bondanalysis "
            "results."
        )
        self.project_status_label.setWordWrap(True)
        layout.addWidget(self.project_status_label)
        return group

    def _build_representative_group(self) -> QWidget:
        group = QGroupBox("Representative Structure")
        layout = QFormLayout(group)
        self.representative_combo = QComboBox()
        self.representative_combo.currentIndexChanged.connect(
            self._handle_representative_changed
        )
        layout.addRow("Stoichiometry", self.representative_combo)

        self.variant_combo = QComboBox()
        for label, mode in (
            ("Saved source", "source"),
            ("No solvent", "nosolv"),
            ("Partial solvent", "partialsolv"),
            ("Full solvent", "fullsolv"),
        ):
            self.variant_combo.addItem(label, mode)
        self.variant_combo.currentIndexChanged.connect(
            self._handle_representative_changed
        )
        layout.addRow("Variant", self.variant_combo)

        path_row = QHBoxLayout()
        self.structure_path_edit = QLineEdit()
        self.structure_path_edit.setPlaceholderText(
            "Representative PDB or XYZ file"
        )
        self.structure_path_edit.returnPressed.connect(self._refresh_preview)
        path_row.addWidget(self.structure_path_edit, stretch=1)
        choose_button = QPushButton("Choose")
        choose_button.clicked.connect(self._choose_structure_file)
        path_row.addWidget(choose_button)
        self.reveal_structure_button = QPushButton("Reveal")
        self.reveal_structure_button.setToolTip(
            "Reveal the active representative structure file in the system "
            "file manager."
        )
        self.reveal_structure_button.clicked.connect(
            self._reveal_structure_file
        )
        path_row.addWidget(self.reveal_structure_button)
        self.generate_cif_button = QPushButton("Generate CIF")
        self.generate_cif_button.setToolTip(
            "Write a padded P1 CIF beside the active representative "
            "structure for FEFF input preparation."
        )
        self.generate_cif_button.clicked.connect(
            self._generate_padded_structure_cif
        )
        path_row.addWidget(self.generate_cif_button)
        layout.addRow("File", path_row)

        self.cif_padding_spin = QDoubleSpinBox()
        self.cif_padding_spin.setRange(0.0, 1000.0)
        self.cif_padding_spin.setDecimals(3)
        self.cif_padding_spin.setSingleStep(1.0)
        self.cif_padding_spin.setValue(20.0)
        self.cif_padding_spin.setToolTip(
            "Void-space padding added on each side of the representative "
            "structure when generating a CIF."
        )
        layout.addRow("CIF Padding (A)", self.cif_padding_spin)

        self.absorber_element_edit = QLineEdit()
        self.absorber_element_edit.setPlaceholderText("Auto (Pb if present)")
        self.absorber_element_edit.editingFinished.connect(
            self._refresh_preview
        )
        layout.addRow("Absorber Element", self.absorber_element_edit)

        self.absorber_index_spin = QSpinBox()
        self.absorber_index_spin.setRange(0, 100000)
        self.absorber_index_spin.setSpecialValueText("All matching atoms")
        self.absorber_index_spin.valueChanged.connect(self._refresh_preview)
        layout.addRow("Absorber Atom", self.absorber_index_spin)

        self.min_distance_spin = QDoubleSpinBox()
        self.min_distance_spin.setRange(0.0, 1000.0)
        self.min_distance_spin.setDecimals(3)
        self.min_distance_spin.setSingleStep(0.1)
        self.min_distance_spin.setValue(0.5)
        self.min_distance_spin.valueChanged.connect(self._refresh_preview)
        layout.addRow("Min R (A)", self.min_distance_spin)

        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(0.0, 1000.0)
        self.max_distance_spin.setDecimals(3)
        self.max_distance_spin.setSingleStep(0.1)
        self.max_distance_spin.setValue(6.0)
        self.max_distance_spin.valueChanged.connect(self._refresh_preview)
        layout.addRow("Max R (A)", self.max_distance_spin)

        self.absorber_iodide_cutoff_spin = QDoubleSpinBox()
        self.absorber_iodide_cutoff_spin.setRange(0.0, 1000.0)
        self.absorber_iodide_cutoff_spin.setDecimals(3)
        self.absorber_iodide_cutoff_spin.setSingleStep(0.1)
        self.absorber_iodide_cutoff_spin.setValue(3.36)
        self.absorber_iodide_cutoff_spin.setToolTip(
            "Cluster-style absorber-I cutoff reference. Max R still controls "
            "how far non-solvent paths are displayed."
        )
        self.absorber_iodide_cutoff_spin.valueChanged.connect(
            self._refresh_preview
        )
        layout.addRow("Abs-I cutoff (A)", self.absorber_iodide_cutoff_spin)

        self.absorber_oxygen_cutoff_spin = QDoubleSpinBox()
        self.absorber_oxygen_cutoff_spin.setRange(0.0, 1000.0)
        self.absorber_oxygen_cutoff_spin.setDecimals(3)
        self.absorber_oxygen_cutoff_spin.setSingleStep(0.1)
        self.absorber_oxygen_cutoff_spin.setValue(3.36)
        self.absorber_oxygen_cutoff_spin.setToolTip(
            "Absorber-O donor cutoff used to admit complete coordinated "
            "solvent residues into the GDS path model."
        )
        self.absorber_oxygen_cutoff_spin.valueChanged.connect(
            self._refresh_preview
        )
        layout.addRow("Abs-O cutoff (A)", self.absorber_oxygen_cutoff_spin)
        return group

    def _build_plot_annotation_group(self) -> QWidget:
        group = QGroupBox("Plot Annotations")
        layout = QVBoxLayout(group)
        self.show_bond_distances_box = QCheckBox("Bond distances")
        self.show_bond_distances_box.setChecked(True)
        self.show_bond_angles_box = QCheckBox("Bond angles")
        self.show_bond_angles_box.setChecked(False)
        self.show_dihedral_angles_box = QCheckBox("Dihedral angles")
        self.show_dihedral_angles_box.setChecked(False)
        for checkbox in (
            self.show_bond_distances_box,
            self.show_bond_angles_box,
            self.show_dihedral_angles_box,
        ):
            checkbox.toggled.connect(
                self._handle_plot_annotation_visibility_changed
            )
            layout.addWidget(checkbox)
        return group

    def _build_bondanalysis_group(self) -> QWidget:
        group = QGroupBox("Bondanalysis Variables")
        layout = QVBoxLayout(group)
        self.bondanalysis_combo = QComboBox()
        self.bondanalysis_combo.currentIndexChanged.connect(
            self._handle_bondanalysis_changed
        )
        layout.addWidget(self.bondanalysis_combo)
        row = QHBoxLayout()
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_bondanalysis_results)
        row.addWidget(refresh_button)
        choose_button = QPushButton("Choose Folder")
        choose_button.clicked.connect(self._choose_bondanalysis_output_dir)
        row.addWidget(choose_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.include_aggregate_variables_box = QCheckBox(
            "Include all-cluster variables"
        )
        self.include_aggregate_variables_box.setChecked(True)
        self.include_aggregate_variables_box.toggled.connect(
            self._refresh_registry_table
        )
        layout.addWidget(self.include_aggregate_variables_box)
        self.bondanalysis_status_label = QLabel(
            "Run bondanalysis before building a GDS file with MD-derived "
            "centers and sigmas."
        )
        self.bondanalysis_status_label.setWordWrap(True)
        layout.addWidget(self.bondanalysis_status_label)
        return group

    def _build_build_group(self) -> QWidget:
        group = QGroupBox("GDS Build")
        layout = QFormLayout(group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Generic shell GDS", "generic")
        self.mode_combo.addItem("Pb-I / DMF constrained GDS", "pb_dmf")
        self.mode_combo.addItem("Pb-I / DMSO constrained GDS", "pb_dmso")
        self.mode_combo.currentIndexChanged.connect(self._refresh_preview)
        layout.addRow("Template", self.mode_combo)

        self.shell_tolerance_spin = QDoubleSpinBox()
        self.shell_tolerance_spin.setRange(0.0, 10.0)
        self.shell_tolerance_spin.setDecimals(3)
        self.shell_tolerance_spin.setSingleStep(0.01)
        self.shell_tolerance_spin.setValue(0.12)
        layout.addRow("Shell Tolerance (A)", self.shell_tolerance_spin)

        self.include_restraints_box = QCheckBox("Include restraints")
        self.include_restraints_box.setChecked(True)
        layout.addRow("", self.include_restraints_box)

        output_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Output .gds file")
        output_row.addWidget(self.output_path_edit, stretch=1)
        choose_button = QPushButton("Choose")
        choose_button.clicked.connect(self._choose_output_file)
        output_row.addWidget(choose_button)
        layout.addRow("Output", output_row)

        button_row = QHBoxLayout()
        preview_button = QPushButton("Preview GDS")
        preview_button.clicked.connect(self._preview_gds)
        button_row.addWidget(preview_button)
        build_button = QPushButton("Build GDS File")
        build_button.clicked.connect(self._build_gds_file)
        button_row.addWidget(build_button)
        button_row.addStretch(1)
        layout.addRow("", button_row)
        return group

    def _choose_project_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select SAXS Project Directory",
            self.project_dir_edit.text().strip() or str(Path.home()),
        )
        if not selected:
            return
        self.project_dir_edit.setText(selected)
        self._load_project_context()

    def _load_project_context(self) -> None:
        text = self.project_dir_edit.text().strip()
        if not text:
            return
        self._project_dir = Path(text).expanduser().resolve()
        self._apply_template_mode_from_context(self._project_dir)
        self._refresh_representatives()
        self._refresh_bondanalysis_results()
        self.project_status_label.setText(
            f"Loaded {len(self._representatives)} representative structure"
            f"{'' if len(self._representatives) == 1 else 's'} and "
            f"{len(self._bondanalysis_results)} bondanalysis result folder"
            f"{'' if len(self._bondanalysis_results) == 1 else 's'}."
        )
        self.statusBar().showMessage(
            f"Loaded EXAFS context for {self._project_dir}"
        )

    def _refresh_representatives(self) -> None:
        self._representatives = ()
        self._suppress_selection_refresh = True
        self.representative_combo.clear()
        self._suppress_selection_refresh = False
        if self._project_dir is None:
            return
        try:
            self._representatives = discover_representative_structures(
                self._project_dir
            )
        except Exception as exc:
            self.project_status_label.setText(
                f"Unable to load representatives: {exc}"
            )
            return
        self._suppress_selection_refresh = True
        for option in self._representatives:
            self.representative_combo.addItem(option.display_label, option)
        self._suppress_selection_refresh = False
        if self._representatives:
            self.representative_combo.setCurrentIndex(0)
            self._handle_representative_changed()

    def _refresh_bondanalysis_results(self) -> None:
        self._bondanalysis_results = ()
        self._suppress_selection_refresh = True
        self.bondanalysis_combo.clear()
        self._suppress_selection_refresh = False
        if self._project_dir is None:
            self._refresh_registry_table()
            return
        try:
            self._bondanalysis_results = discover_bondanalysis_results(
                self._project_dir
            )
        except Exception as exc:
            self.bondanalysis_status_label.setText(
                f"Unable to load bondanalysis results: {exc}"
            )
            self._refresh_registry_table()
            return
        self._suppress_selection_refresh = True
        for result in self._bondanalysis_results:
            self.bondanalysis_combo.addItem(result.display_label, result)
        self._suppress_selection_refresh = False
        if self._bondanalysis_results:
            self.bondanalysis_combo.setCurrentIndex(0)
            self.bondanalysis_status_label.setText(
                f"Loaded {len(self._bondanalysis_results)} result folder"
                f"{'' if len(self._bondanalysis_results) == 1 else 's'}."
            )
        else:
            self.bondanalysis_status_label.setText(
                "No completed bondanalysis result with GDS variables was "
                "found for this project."
            )
        self._refresh_registry_table()

    def _choose_structure_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select Representative Structure",
            self.structure_path_edit.text().strip() or str(Path.home()),
            "Structures (*.pdb *.xyz);;All Files (*)",
        )
        if not selected:
            return
        self.structure_path_edit.setText(selected)
        self._apply_template_mode_from_context(self._project_dir, selected)
        self._set_default_output_path()
        self._refresh_preview()

    def _reveal_structure_file(self) -> None:
        source = self.structure_path_edit.text().strip()
        if not source:
            QMessageBox.warning(
                self,
                "EXAFS GDS Mapping",
                "Choose a representative structure file first.",
            )
            return
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            QMessageBox.warning(
                self,
                "EXAFS GDS Mapping",
                f"Representative structure does not exist: {path}",
            )
            return
        try:
            _reveal_file_in_file_manager(path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "EXAFS GDS Mapping",
                f"Unable to reveal representative structure: {exc}",
            )

    def _generate_padded_structure_cif(self) -> None:
        source = self.structure_path_edit.text().strip()
        if not source:
            QMessageBox.warning(
                self,
                "EXAFS GDS Mapping",
                "Choose a representative structure file first.",
            )
            return
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            QMessageBox.warning(
                self,
                "EXAFS GDS Mapping",
                f"Representative structure does not exist: {path}",
            )
            return
        try:
            output_path = write_padded_cif_from_structure(
                path,
                padding_angstrom=self.cif_padding_spin.value(),
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "EXAFS GDS Mapping",
                f"Unable to generate padded CIF: {exc}",
            )
            return
        self.statusBar().showMessage(f"Wrote padded CIF: {output_path}")
        QMessageBox.information(
            self,
            "EXAFS GDS Mapping",
            f"Wrote padded CIF:\n{output_path}",
        )

    def _choose_bondanalysis_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Bondanalysis Output Directory",
            str(self._project_dir or Path.home()),
        )
        if not selected:
            return
        try:
            result_index = load_bondanalysis_result(selected)
        except Exception as exc:
            QMessageBox.warning(self, "EXAFS GDS Mapping", str(exc))
            return
        result = EXAFSBondAnalysisResult(
            output_dir=result_index.output_dir,
            results_index_path=result_index.results_index_path,
            selected_cluster_types=result_index.selected_cluster_types,
            gds_variable_count=len(result_index.gds_variable_registry),
        )
        self._bondanalysis_results = (result,) + tuple(
            existing
            for existing in self._bondanalysis_results
            if existing.output_dir != result.output_dir
        )
        self.bondanalysis_combo.clear()
        for item in self._bondanalysis_results:
            self.bondanalysis_combo.addItem(item.display_label, item)
        self.bondanalysis_combo.setCurrentIndex(0)
        self._refresh_registry_table()

    def _choose_output_file(self) -> None:
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Artemis GDS File",
            self.output_path_edit.text().strip()
            or str((self._project_dir or Path.home()) / "exafs_mapping.gds"),
            "Artemis GDS (*.gds);;All Files (*)",
        )
        if not selected:
            return
        path = Path(selected).expanduser()
        if path.suffix.lower() != ".gds":
            path = path.with_suffix(".gds")
        self.output_path_edit.setText(str(path))

    def _handle_representative_changed(self) -> None:
        if self._suppress_selection_refresh:
            return
        option = self._selected_representative()
        if option is not None:
            path = self._selected_representative_path(option)
            if path is not None:
                self.structure_path_edit.setText(str(path))
            self._apply_template_mode_from_context(
                self._project_dir,
                option.source_file_name,
                option.source_file,
                path,
            )
            self._set_default_output_path()
            self._set_default_absorber_for_structure()
        self._refresh_registry_table()
        self._refresh_preview()

    def _handle_bondanalysis_changed(self) -> None:
        if self._suppress_selection_refresh:
            return
        self._refresh_registry_table()

    def _handle_plot_annotation_visibility_changed(self) -> None:
        if self._active_preview is not None:
            self._draw_path_plots(self._active_preview)

    def _plot_annotation_visibility(self) -> _PlotAnnotationVisibility:
        return _PlotAnnotationVisibility(
            bond_distances=self.show_bond_distances_box.isChecked(),
            bond_angles=self.show_bond_angles_box.isChecked(),
            dihedral_angles=self.show_dihedral_angles_box.isChecked(),
        )

    def _selected_representative(self) -> EXAFSRepresentativeOption | None:
        data = self.representative_combo.currentData()
        return data if isinstance(data, EXAFSRepresentativeOption) else None

    def _selected_representative_path(
        self,
        option: EXAFSRepresentativeOption,
    ) -> Path | None:
        mode = self.variant_combo.currentData()
        if mode == "source":
            return option.source_file
        return option.variant_path(str(mode))

    def _selected_bondanalysis_result(self) -> EXAFSBondAnalysisResult | None:
        data = self.bondanalysis_combo.currentData()
        return data if isinstance(data, EXAFSBondAnalysisResult) else None

    def _set_default_output_path(self) -> None:
        if self.output_path_edit.text().strip():
            return
        source = self.structure_path_edit.text().strip()
        if not source:
            return
        source_path = Path(source).expanduser()
        self.output_path_edit.setText(
            str(source_path.with_name(f"{source_path.stem}_exafs_mapping.gds"))
        )

    def _set_default_absorber_for_structure(self) -> None:
        current = self.absorber_element_edit.text().strip()
        previous_auto = self._auto_absorber_element or ""
        if current and current != previous_auto:
            return
        try:
            absorber = default_absorber_element(
                self.structure_path_edit.text(),
                preferred_element=self._preferred_absorber_element,
            )
        except Exception:
            absorber = None
        if absorber:
            self._auto_absorber_element = absorber
            self.absorber_element_edit.setText(absorber)

    def _refresh_preview(self) -> None:
        source_text = self.structure_path_edit.text().strip()
        if not source_text:
            return
        absorber_element = self.absorber_element_edit.text().strip() or None
        absorber_index = self.absorber_index_spin.value() or None
        try:
            preview = load_structure_preview(
                source_text,
                absorber_element=absorber_element,
                absorber_atom_index=absorber_index,
                min_distance_angstrom=self.min_distance_spin.value(),
                max_distance_angstrom=self.max_distance_spin.value(),
                pair_cutoff_distances_angstrom=self._pair_cutoff_distances(),
            )
        except Exception as exc:
            self._active_preview = None
            self._active_scattering_path_events = ()
            self._scattering_path_check_states_by_key.clear()
            self.scattering_path_table.setRowCount(0)
            self._draw_empty_path_plots(str(exc))
            self.statusBar().showMessage(f"Unable to load structure: {exc}")
            return
        self._active_preview = preview
        self._refresh_structure_viewer(preview.structure_path)
        self._draw_path_plots(preview)
        self._refresh_scattering_path_table(preview)
        counts = ", ".join(
            f"{element}{count}"
            for element, count in preview.element_counts.items()
        )
        self.statusBar().showMessage(
            f"Loaded {preview.structure_path.name}: {len(preview.paths)} "
            f"path(s), {len(preview.dynamic_bonds)} refined bond(s), "
            f"{len(preview.static_bonds)} static bond(s), "
            f"{len(preview.angles)} angle(s), {counts}"
        )

    def _refresh_structure_viewer(self, structure_path: Path) -> None:
        try:
            structure = load_electron_density_structure(
                structure_path,
                center_mode="center_of_mass",
                include_bonds=True,
            )
        except Exception:
            return
        self.structure_viewer.set_structure(structure)

    def _refresh_scattering_path_table(
        self,
        preview: EXAFSStructurePreview | None = None,
    ) -> None:
        self._sync_visible_scattering_path_check_states()
        if preview is not None:
            events = scattering_path_events_from_preview(preview)
            self._active_scattering_path_events = events
            active_keys = {event.path_key for event in events}
            self._scattering_path_check_states_by_key = {
                key: state
                for key, state in self._scattering_path_check_states_by_key.items()
                if key in active_keys
            }
            for event in events:
                self._scattering_path_check_states_by_key.setdefault(
                    event.path_key,
                    Qt.CheckState.Checked,
                )
        events = tuple(self._active_scattering_path_events)
        visible_events = self._filtered_scattering_path_events(events)
        grouped_events = self._grouped_scattering_path_events(visible_events)
        row_count = sum(
            len(group_events) + (1 if group_label is not None else 0)
            for group_label, group_events in grouped_events
        )
        self.scattering_path_table.setRowCount(0)
        self.scattering_path_table.setRowCount(row_count)
        row = 0
        for group_label, group_events in grouped_events:
            if group_label is not None:
                self._set_scattering_path_group_row(
                    row,
                    group_label,
                    len(group_events),
                )
                row += 1
            for event in group_events:
                self._set_scattering_path_event_row(row, event)
                row += 1
        self.scattering_path_table.resizeColumnsToContents()
        self._update_scattering_path_filter_status(
            total_count=len(events),
            visible_count=len(visible_events),
        )

    def _set_scattering_path_group_row(
        self,
        row: int,
        group_label: str,
        group_count: int,
    ) -> None:
        item = QTableWidgetItem(f"{group_label} ({group_count} path(s))")
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.ItemDataRole.UserRole, ("group", group_label))
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.scattering_path_table.setItem(row, 0, item)
        self.scattering_path_table.setSpan(
            row,
            0,
            1,
            self.scattering_path_table.columnCount(),
        )

    def _set_scattering_path_event_row(
        self,
        row: int,
        event: EXAFSScatteringPathEvent,
    ) -> None:
        path_key = event.path_key
        use_item = QTableWidgetItem()
        use_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsSelectable
        )
        use_item.setCheckState(
            self._scattering_path_check_states_by_key.get(
                path_key,
                Qt.CheckState.Checked,
            )
        )
        use_item.setData(Qt.ItemDataRole.UserRole, path_key)
        self.scattering_path_table.setItem(row, 0, use_item)
        values = self._scattering_path_event_fields(event)
        for column, (key, _label) in enumerate(_PATH_TABLE_COLUMNS, start=1):
            item = QTableWidgetItem(values.get(key, ""))
            item.setData(Qt.ItemDataRole.UserRole, path_key)
            if key in {"degeneracy", "path_length", "reff"}:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            self.scattering_path_table.setItem(row, column, item)

    def _scattering_path_event_fields(
        self,
        event: EXAFSScatteringPathEvent,
    ) -> dict[str, str]:
        return {
            "solvent_molecule": str(event.solvent_molecule_label),
            "path": str(event.label),
            "absorber": str(event.absorber_atom_label),
            "scatterer": str(event.scatterer_atom_label),
            "scatterer_element": str(event.scatterer_element),
            "degeneracy": f"{event.degeneracy:.3g}",
            "path_length": f"{event.total_path_length_angstrom:.3f}",
            "reff": f"{event.effective_distance_angstrom:.3f}",
            "bonds": "; ".join(event.bond_lengths),
            "angles": "; ".join(event.angles),
            "dihedrals": "; ".join(event.dihedrals),
        }

    def _filtered_scattering_path_events(
        self,
        events: tuple[EXAFSScatteringPathEvent, ...],
    ) -> tuple[EXAFSScatteringPathEvent, ...]:
        filter_text = self.path_filter_edit.text().strip().casefold()
        if not filter_text:
            return events
        column_key = str(self.path_filter_column_combo.currentData())
        filtered = []
        for event in events:
            fields = self._scattering_path_event_fields(event)
            if column_key == _PATH_TABLE_FILTER_ALL_KEY:
                haystack = "\n".join(fields.values()).casefold()
            else:
                haystack = fields.get(column_key, "").casefold()
            if filter_text in haystack:
                filtered.append(event)
        return tuple(filtered)

    def _grouped_scattering_path_events(
        self,
        events: tuple[EXAFSScatteringPathEvent, ...],
    ) -> tuple[tuple[str | None, tuple[EXAFSScatteringPathEvent, ...]], ...]:
        group_key = str(self.path_group_combo.currentData())
        if group_key == _PATH_TABLE_NO_GROUP_KEY:
            return ((None, events),) if events else ()
        groups: dict[str, list[EXAFSScatteringPathEvent]] = {}
        group_order: list[str] = []
        for event in events:
            label = self._scattering_path_group_label(event, group_key)
            if label not in groups:
                groups[label] = []
                group_order.append(label)
            groups[label].append(event)
        group_order.sort(
            key=lambda label: (
                1 if label == "Direct / non-solvent paths" else 0,
                _natural_label_sort_key(label),
            )
        )
        return tuple((label, tuple(groups[label])) for label in group_order)

    def _scattering_path_group_label(
        self,
        event: EXAFSScatteringPathEvent,
        group_key: str,
    ) -> str:
        if group_key == _PATH_TABLE_DEFAULT_GROUP_KEY:
            return str(event.solvent_molecule_label)
        return (
            self._scattering_path_event_fields(event).get(
                group_key,
                "Other",
            )
            or "Other"
        )

    def _update_scattering_path_filter_status(
        self,
        *,
        total_count: int,
        visible_count: int,
    ) -> None:
        if total_count == 0:
            self.path_filter_status_label.setText("No paths loaded.")
            return
        group_label = self.path_group_combo.currentText()
        if visible_count == total_count:
            self.path_filter_status_label.setText(
                f"Showing {visible_count} path(s), grouped by {group_label}."
            )
            return
        self.path_filter_status_label.setText(
            f"Showing {visible_count} of {total_count} path(s), grouped by "
            f"{group_label}."
        )

    def _reset_scattering_path_grouping(self) -> None:
        index = self.path_group_combo.findData(_PATH_TABLE_DEFAULT_GROUP_KEY)
        if index >= 0:
            if self.path_group_combo.currentIndex() == index:
                self._refresh_scattering_path_table()
            else:
                self.path_group_combo.setCurrentIndex(index)
            return
        self._refresh_scattering_path_table()

    def _sync_visible_scattering_path_check_states(self) -> None:
        if not hasattr(self, "scattering_path_table"):
            return
        for row in range(self.scattering_path_table.rowCount()):
            item = self.scattering_path_table.item(row, 0)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and all(isinstance(value, int) for value in key)
            ):
                self._scattering_path_check_states_by_key[
                    (int(key[0]), int(key[1]))
                ] = item.checkState()

    def _scattering_path_check_states(
        self,
    ) -> dict[tuple[int, int], Qt.CheckState]:
        self._sync_visible_scattering_path_check_states()
        states: dict[tuple[int, int], Qt.CheckState] = {}
        if not hasattr(self, "scattering_path_table"):
            return dict(self._scattering_path_check_states_by_key)
        states.update(self._scattering_path_check_states_by_key)
        return states

    def _refresh_registry_table(self) -> None:
        self.variable_table.setRowCount(0)
        self._active_registry_entries = ()
        result = self._selected_bondanalysis_result()
        if result is None:
            return
        try:
            result_index = load_bondanalysis_result(result)
        except Exception as exc:
            self.bondanalysis_status_label.setText(
                f"Unable to load selected bondanalysis result: {exc}"
            )
            return
        representative = self._selected_representative()
        stoichiometry = (
            representative.stoichiometry
            if representative is not None
            else None
        )
        entries = gds_registry_entries_for_stoichiometry(
            result_index,
            stoichiometry,
            include_aggregate=self.include_aggregate_variables_box.isChecked(),
        )
        self._active_registry_entries = entries
        self.variable_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            use_item = QTableWidgetItem()
            use_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
            )
            use_item.setCheckState(Qt.CheckState.Checked)
            use_item.setData(Qt.ItemDataRole.UserRole, entry)
            self.variable_table.setItem(row, 0, use_item)
            self.variable_table.setItem(
                row,
                1,
                QTableWidgetItem(str(entry.get("scope", ""))),
            )
            self.variable_table.setItem(
                row,
                2,
                QTableWidgetItem(str(entry.get("distribution_type", ""))),
            )
            self.variable_table.setItem(
                row,
                3,
                QTableWidgetItem(str(entry.get("distribution_label", ""))),
            )
            variables = entry.get("variables", [])
            variable_text = ", ".join(
                str(item.get("name", ""))
                for item in variables
                if isinstance(item, Mapping)
            )
            self.variable_table.setItem(
                row, 4, QTableWidgetItem(variable_text)
            )
            self.variable_table.setItem(
                row,
                5,
                QTableWidgetItem(str(entry.get("set_rows", ""))),
            )
        self.variable_table.resizeColumnsToContents()
        self.bondanalysis_status_label.setText(
            f"{len(entries)} GDS-ready distribution entr"
            f"{'y' if len(entries) == 1 else 'ies'} available for the "
            "selected stoichiometry."
        )

    def _selected_registry_entries(self) -> tuple[dict[str, object], ...]:
        selected: list[dict[str, object]] = []
        for row in range(self.variable_table.rowCount()):
            item = self.variable_table.item(row, 0)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            entry = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(entry, dict):
                selected.append(dict(entry))
        return tuple(selected)

    def _selected_scattering_path_pairs(
        self,
    ) -> tuple[tuple[int, int], ...] | None:
        self._sync_visible_scattering_path_check_states()
        if not self._active_scattering_path_events:
            return None
        selected: list[tuple[int, int]] = []
        for event in self._active_scattering_path_events:
            if (
                self._scattering_path_check_states_by_key.get(
                    event.path_key,
                    Qt.CheckState.Checked,
                )
                == Qt.CheckState.Checked
            ):
                selected.append(event.path_key)
        return tuple(selected)

    def _preview_gds(self) -> None:
        try:
            document = self._build_document()
            text = document.to_text()
            from saxshell.exafs import validate_artemis_gds_text

            report = validate_artemis_gds_text(text)
        except Exception as exc:
            QMessageBox.warning(self, "EXAFS GDS Mapping", str(exc))
            return
        self.gds_preview_text.setPlainText(text)
        self.validation_label.setText(report.summary_text())
        self.preview_tabs.setCurrentWidget(
            self.gds_preview_text.parentWidget()
        )

    def _build_gds_file(self) -> None:
        output_text = self.output_path_edit.text().strip()
        if not output_text:
            self._choose_output_file()
            output_text = self.output_path_edit.text().strip()
            if not output_text:
                return
        try:
            output_path = write_gds_mapping_file(
                output_text,
                self._structure_path(),
                mode=str(self.mode_combo.currentData()),
                absorber_element=self.absorber_element_edit.text().strip()
                or None,
                absorber_atom_index=self.absorber_index_spin.value() or None,
                min_distance_angstrom=self.min_distance_spin.value(),
                max_distance_angstrom=self.max_distance_spin.value(),
                pair_cutoff_distances_angstrom=self._pair_cutoff_distances(),
                included_path_pairs=self._selected_scattering_path_pairs(),
                shell_tolerance_angstrom=self.shell_tolerance_spin.value(),
                include_restraints=self.include_restraints_box.isChecked(),
                gds_registry_entries=self._selected_registry_entries(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "EXAFS GDS Mapping", str(exc))
            return
        self.gds_preview_text.setPlainText(Path(output_path).read_text())
        overview_path = artemis_gds_overview_path(output_path)
        self.validation_label.setText(
            f"Wrote valid GDS file: {output_path}\nOverview: {overview_path}"
        )
        self.statusBar().showMessage(f"Wrote {output_path}")

    def _build_document(self):
        return build_gds_mapping_document(
            self._structure_path(),
            mode=str(self.mode_combo.currentData()),
            absorber_element=self.absorber_element_edit.text().strip() or None,
            absorber_atom_index=self.absorber_index_spin.value() or None,
            min_distance_angstrom=self.min_distance_spin.value(),
            max_distance_angstrom=self.max_distance_spin.value(),
            pair_cutoff_distances_angstrom=self._pair_cutoff_distances(),
            included_path_pairs=self._selected_scattering_path_pairs(),
            shell_tolerance_angstrom=self.shell_tolerance_spin.value(),
            include_restraints=self.include_restraints_box.isChecked(),
            gds_registry_entries=self._selected_registry_entries(),
        )

    def _pair_cutoff_distances(self) -> dict[tuple[str, str], float]:
        absorber_element = (
            _normalize_element_text(self.absorber_element_edit.text()) or "Pb"
        )
        return {
            (absorber_element, "I"): self.absorber_iodide_cutoff_spin.value(),
            (absorber_element, "O"): self.absorber_oxygen_cutoff_spin.value(),
        }

    def _apply_template_mode_from_context(self, *values: object) -> None:
        mode = _template_mode_from_context_values(*values)
        if mode is None:
            return
        mode_index = self.mode_combo.findData(mode)
        if mode_index >= 0 and self.mode_combo.currentIndex() != mode_index:
            self.mode_combo.setCurrentIndex(mode_index)

    def _structure_path(self) -> Path:
        text = self.structure_path_edit.text().strip()
        if not text:
            raise ValueError("Choose a representative structure first.")
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"Representative structure does not exist: {path}"
            )
        return path

    def _draw_empty_path_plots(self, message: str) -> None:
        for figure, canvas in (
            (self.path3d_figure, self.path3d_canvas),
            (self.path2d_figure, self.path2d_canvas),
        ):
            figure.clear()
            axis = figure.add_subplot(111)
            axis.text(0.5, 0.5, message, ha="center", va="center")
            axis.axis("off")
            canvas.draw_idle()

    def _draw_path_plots(self, preview: EXAFSStructurePreview) -> None:
        self._draw_3d_paths(preview)
        self._draw_2d_paths(preview)

    def _draw_3d_paths(self, preview: EXAFSStructurePreview) -> None:
        self.path3d_figure.clear()
        axis = self.path3d_figure.add_subplot(111, projection="3d")
        visibility = self._plot_annotation_visibility()
        coords = np.asarray(preview.coordinates, dtype=float)
        centered = coords - np.mean(coords, axis=0)
        plot_center = np.mean(centered, axis=0)
        atom_points = {
            atom_index: np.asarray(point, dtype=float)
            for atom_index, point in zip(
                preview.atom_indices,
                centered,
                strict=False,
            )
        }
        if visibility.dihedral_angles:
            dihedrals = _preview_dihedrals(preview)[:24]
            for label_index, dihedral in enumerate(dihedrals):
                _draw_3d_dihedral_annotation(
                    axis,
                    atom_points,
                    dihedral,
                    plot_center=plot_center,
                    label_index=label_index,
                )
        for label_index, bond in enumerate(preview.static_bonds[:80]):
            start = atom_points.get(bond.atom1_index)
            end = atom_points.get(bond.atom2_index)
            if start is None or end is None:
                continue
            midpoint = (start + end) * 0.5
            axis.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color="#626a73",
                linestyle="-",
                linewidth=1.0,
                alpha=0.75,
            )
            if visibility.bond_distances:
                label_point = _offset_3d_label_point(
                    midpoint,
                    plot_center,
                    label_index=label_index,
                    scale=0.18,
                )
                axis.text(
                    label_point[0],
                    label_point[1],
                    label_point[2],
                    f"{bond.label} {bond.distance_angstrom:.3f} A",
                    fontsize=6.2,
                    color="#626a73",
                    bbox=_plot_label_bbox(alpha=0.58),
                )
        for label_index, bond in enumerate(preview.dynamic_bonds[:60]):
            start = atom_points.get(bond.atom1_index)
            end = atom_points.get(bond.atom2_index)
            if start is None or end is None:
                continue
            midpoint = (start + end) * 0.5
            axis.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color="#1769aa",
                linestyle="--",
                linewidth=1.7,
                alpha=0.9,
            )
            if visibility.bond_distances:
                label_point = _offset_3d_label_point(
                    midpoint,
                    plot_center,
                    label_index=label_index + len(preview.static_bonds),
                    scale=0.22,
                )
                axis.text(
                    label_point[0],
                    label_point[1],
                    label_point[2],
                    f"{bond.label} {bond.distance_angstrom:.3f} A",
                    fontsize=6.5,
                    color="#1769aa",
                    bbox=_plot_label_bbox(alpha=0.62),
                )
        if visibility.bond_angles:
            for label_index, angle in enumerate(preview.angles[:48]):
                _draw_3d_angle_annotation(
                    axis,
                    atom_points,
                    angle,
                    plot_center=plot_center,
                    label_index=label_index,
                )
        absorber_indices = set(preview.absorber_indices)
        for _element, atom_index, atom_label, point in zip(
            preview.elements,
            preview.atom_indices,
            preview.atom_labels,
            centered,
            strict=False,
        ):
            is_absorber = atom_index in absorber_indices
            axis.scatter(
                [point[0]],
                [point[1]],
                [point[2]],
                s=90 if is_absorber else 30,
                color="#c72535" if is_absorber else "#2f3438",
                edgecolors="black" if is_absorber else "none",
                linewidths=0.6 if is_absorber else 0.0,
            )
            axis.text(
                point[0],
                point[1],
                point[2],
                atom_label,
                fontsize=7,
                fontweight="bold" if is_absorber else "normal",
            )
        self._autoscale_3d_axis(axis, centered)
        axis.set_title(f"{preview.structure_path.name} GDS geometry")
        axis.set_xlabel("x (A)")
        axis.set_ylabel("y (A)")
        axis.set_zlabel("z (A)")
        self.path3d_figure.tight_layout()
        self.path3d_canvas.draw_idle()

    def _draw_2d_paths(self, preview: EXAFSStructurePreview) -> None:
        self.path2d_figure.clear()
        axis = self.path2d_figure.add_subplot(111)
        visibility = self._plot_annotation_visibility()
        coords = np.asarray(preview.coordinates, dtype=float)
        centered = coords - np.mean(coords, axis=0)
        label_requests: list[_Deferred2DLabel] = []
        atom_points = {
            atom_index: np.asarray(point[:2], dtype=float)
            for atom_index, point in zip(
                preview.atom_indices,
                centered,
                strict=False,
            )
        }
        atom_points3d = {
            atom_index: np.asarray(point, dtype=float)
            for atom_index, point in zip(
                preview.atom_indices,
                centered,
                strict=False,
            )
        }
        if visibility.dihedral_angles:
            for dihedral in _preview_dihedrals(preview)[:32]:
                _draw_2d_dihedral_annotation(
                    axis,
                    atom_points,
                    atom_points3d,
                    dihedral,
                    label_requests,
                )
        for bond in preview.static_bonds[:90]:
            _draw_2d_bond_annotation(
                axis,
                atom_points,
                bond.atom1_index,
                bond.atom2_index,
                label=f"{bond.label}\n{bond.distance_angstrom:.3f} A",
                color="#626a73",
                linestyle="-",
                linewidth=1.1,
                zorder=2,
                label_requests=label_requests,
                show_label=visibility.bond_distances,
            )
        for bond in preview.dynamic_bonds[:70]:
            _draw_2d_bond_annotation(
                axis,
                atom_points,
                bond.atom1_index,
                bond.atom2_index,
                label=f"{bond.label}\n{bond.distance_angstrom:.3f} A",
                color="#1769aa",
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                label_requests=label_requests,
                show_label=visibility.bond_distances,
            )
        if visibility.bond_angles:
            for angle in preview.angles[:60]:
                _draw_2d_angle_annotation(
                    axis,
                    atom_points,
                    angle,
                    label_requests,
                )

        absorber_indices = set(preview.absorber_indices)
        for _element, atom_index, atom_label, point in zip(
            preview.elements,
            preview.atom_indices,
            preview.atom_labels,
            centered,
            strict=False,
        ):
            is_absorber = atom_index in absorber_indices
            if is_absorber:
                axis.scatter(
                    [point[0]],
                    [point[1]],
                    s=150,
                    marker="o",
                    facecolors="none",
                    edgecolors="#c72535",
                    linewidths=2.1,
                    zorder=7,
                )
                axis.scatter(
                    [point[0]],
                    [point[1]],
                    s=46,
                    marker="o",
                    color="#c72535",
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=8,
                )
                fontweight = "bold"
            else:
                axis.scatter(
                    [point[0]],
                    [point[1]],
                    s=30,
                    color="#2f3438",
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=6,
                )
                fontweight = "normal"
            label_requests.append(
                _Deferred2DLabel(
                    text=atom_label,
                    anchor=np.asarray(point[:2], dtype=float),
                    color="#2f3438",
                    fontsize=8,
                    zorder=9,
                    priority=90 if is_absorber else 80,
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.68,
                    },
                    connector=False,
                    offset_scale=0.65 if is_absorber else 0.5,
                    fontweight=fontweight,
                    required=True,
                )
            )
        axis.set_aspect("equal", adjustable="datalim")
        axis.margins(0.24)
        _draw_deferred_2d_labels(axis, label_requests)
        axis.set_title("XY projection with EXAFS GDS geometry labels")
        axis.set_xlabel("x (A)")
        axis.set_ylabel("y (A)")
        axis.grid(True, alpha=0.2)
        self.path2d_figure.tight_layout()
        self.path2d_canvas.draw_idle()

    @staticmethod
    def _autoscale_3d_axis(axis, coordinates: np.ndarray) -> None:
        if coordinates.size == 0:
            return
        mins = np.min(coordinates, axis=0)
        maxs = np.max(coordinates, axis=0)
        center = (mins + maxs) * 0.5
        radius = max(float(np.max(maxs - mins)) * 0.55, 1.0)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)


def _preview_dihedrals(preview: EXAFSStructurePreview) -> tuple[object, ...]:
    raw_dihedrals = getattr(preview, "dihedrals", ())
    if raw_dihedrals is None:
        return ()
    try:
        return tuple(raw_dihedrals)
    except TypeError:
        return ()


def _plot_label_bbox(*, alpha: float) -> dict[str, object]:
    return {
        "boxstyle": "round,pad=0.10",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": alpha,
    }


def _offset_3d_label_point(
    anchor: np.ndarray,
    plot_center: np.ndarray,
    *,
    label_index: int,
    scale: float,
) -> np.ndarray:
    anchor_point = np.asarray(anchor, dtype=float)
    center = np.asarray(plot_center, dtype=float)
    if anchor_point.size < 3 or center.size < 3:
        return anchor_point
    outward = anchor_point[:3] - center[:3]
    outward_length = float(np.linalg.norm(outward))
    if outward_length < 1e-8:
        outward = np.asarray((1.0, 0.0, 0.0), dtype=float)
        outward_length = 1.0
    outward_unit = outward / outward_length
    z_unit = np.asarray((0.0, 0.0, 1.0), dtype=float)
    lateral = np.cross(outward_unit, z_unit)
    lateral_length = float(np.linalg.norm(lateral))
    if lateral_length < 1e-8:
        lateral = np.asarray((0.0, 1.0, 0.0), dtype=float)
        lateral_length = 1.0
    lateral_unit = lateral / lateral_length
    lateral_phase = (label_index % 5) - 2
    vertical_phase = (label_index % 3) - 1
    return (
        anchor_point[:3]
        + outward_unit * scale
        + lateral_unit * lateral_phase * scale * 0.24
        + z_unit * vertical_phase * scale * 0.38
    )


def _draw_3d_angle_annotation(
    axis,
    atom_points: Mapping[int, np.ndarray],
    angle: EXAFSAngleAnnotation,
    *,
    plot_center: np.ndarray,
    label_index: int,
) -> None:
    absorber = atom_points.get(angle.absorber_index)
    bridge = atom_points.get(angle.bridge_index)
    terminal = atom_points.get(angle.terminal_index)
    if absorber is None or bridge is None or terminal is None:
        return
    absorber_vector = absorber - bridge
    terminal_vector = terminal - bridge
    absorber_length = float(np.linalg.norm(absorber_vector))
    terminal_length = float(np.linalg.norm(terminal_vector))
    if absorber_length < 1e-6 or terminal_length < 1e-6:
        return
    absorber_unit = absorber_vector / absorber_length
    terminal_unit = terminal_vector / terminal_length
    terminal_perp = (
        terminal_unit - np.dot(terminal_unit, absorber_unit) * absorber_unit
    )
    terminal_perp_length = float(np.linalg.norm(terminal_perp))
    if terminal_perp_length < 1e-6:
        return
    plane_unit = terminal_perp / terminal_perp_length
    signed_angle = float(
        np.arctan2(
            np.dot(terminal_unit, plane_unit),
            np.dot(terminal_unit, absorber_unit),
        )
    )
    radius = min(max(min(absorber_length, terminal_length) * 0.24, 0.22), 0.72)
    samples = np.linspace(0.0, signed_angle, 28)
    arc_points = np.asarray(
        [
            bridge
            + (np.cos(theta) * absorber_unit + np.sin(theta) * plane_unit)
            * radius
            for theta in samples
        ],
        dtype=float,
    )
    axis.plot(
        arc_points[:, 0],
        arc_points[:, 1],
        arc_points[:, 2],
        color="#2f7d32",
        linewidth=1.25,
        alpha=0.88,
    )
    mid_theta = signed_angle * 0.5
    label_anchor = bridge + (
        np.cos(mid_theta) * absorber_unit + np.sin(mid_theta) * plane_unit
    ) * (radius + 0.18)
    label_point = _offset_3d_label_point(
        label_anchor,
        plot_center,
        label_index=label_index,
        scale=0.16,
    )
    axis.text(
        label_point[0],
        label_point[1],
        label_point[2],
        f"{angle.atom_triplet_label} {angle.angle_degrees:.1f} deg",
        fontsize=6.2,
        color="#2f7d32",
        bbox=_plot_label_bbox(alpha=0.56),
    )


def _draw_3d_dihedral_annotation(
    axis,
    atom_points: Mapping[int, np.ndarray],
    dihedral: object,
    *,
    plot_center: np.ndarray,
    label_index: int,
) -> None:
    atom_indices = _dihedral_atom_indices(dihedral)
    if atom_indices is None:
        return
    plane1_indices = _dihedral_plane_indices(
        dihedral,
        "plane1_indices",
        atom_indices[:3],
    )
    plane2_indices = _dihedral_plane_indices(
        dihedral,
        "plane2_indices",
        atom_indices[1:],
    )
    _draw_3d_plane_fill(
        axis,
        atom_points,
        plane1_indices,
        color="#f59e0b",
        label="p1",
    )
    _draw_3d_plane_fill(
        axis,
        atom_points,
        plane2_indices,
        color="#7c3aed",
        label="p2",
    )
    points = [atom_points.get(atom_index) for atom_index in atom_indices]
    if any(point is None for point in points):
        return
    centroid = np.mean(np.asarray(points, dtype=float), axis=0)
    label_point = _offset_3d_label_point(
        centroid,
        plot_center,
        label_index=label_index,
        scale=0.20,
    )
    axis.text(
        label_point[0],
        label_point[1],
        label_point[2],
        _dihedral_label_text(dihedral),
        fontsize=6.2,
        color="#7c3aed",
        bbox=_plot_label_bbox(alpha=0.58),
    )


def _draw_3d_plane_fill(
    axis,
    atom_points: Mapping[int, np.ndarray],
    atom_indices: tuple[int, int, int],
    *,
    color: str,
    label: str,
) -> None:
    points = [atom_points.get(atom_index) for atom_index in atom_indices]
    if any(point is None for point in points):
        return
    coordinates = np.asarray(points, dtype=float)
    collection = Poly3DCollection(
        [coordinates],
        facecolors=color,
        edgecolors=color,
        linewidths=0.75,
        alpha=0.18,
    )
    axis.add_collection3d(collection)
    centroid = np.mean(coordinates, axis=0)
    axis.text(
        centroid[0],
        centroid[1],
        centroid[2],
        label,
        fontsize=6,
        color=color,
    )


def _draw_2d_dihedral_annotation(
    axis,
    atom_points: Mapping[int, np.ndarray],
    atom_points3d: Mapping[int, np.ndarray],
    dihedral: object,
    label_requests: list[_Deferred2DLabel],
) -> None:
    atom_indices = _dihedral_atom_indices(dihedral)
    if atom_indices is None:
        return
    plane1_indices = _dihedral_plane_indices(
        dihedral,
        "plane1_indices",
        atom_indices[:3],
    )
    plane2_indices = _dihedral_plane_indices(
        dihedral,
        "plane2_indices",
        atom_indices[1:],
    )
    atom_label_values = _dihedral_atom_label_tuple(dihedral)
    _draw_2d_plane_fill(
        axis,
        atom_points,
        plane1_indices,
        color="#f59e0b",
        label=(
            "in-plane "
            f"{atom_label_values[0]}-{atom_label_values[1]}-"
            f"{atom_label_values[2]}"
        ),
        zorder=1,
        label_requests=label_requests,
    )
    _draw_2d_plane_fill(
        axis,
        atom_points,
        plane2_indices,
        color="#7c3aed",
        label=(
            "rotated plane "
            f"{atom_label_values[1]}-{atom_label_values[2]}-"
            f"{atom_label_values[3]}"
        ),
        zorder=1,
        label_requests=label_requests,
    )
    _draw_2d_dihedral_components(
        axis,
        atom_points,
        atom_points3d,
        atom_indices,
        atom_label_values,
        label_requests,
    )
    points = [atom_points.get(atom_index) for atom_index in atom_indices]
    if any(point is None for point in points):
        return
    centroid = np.mean(np.asarray(points, dtype=float), axis=0)
    label_requests.append(
        _Deferred2DLabel(
            text=_dihedral_label_text(dihedral),
            anchor=centroid,
            fontsize=6.5,
            color="#7c3aed",
            zorder=8,
            priority=70,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            offset_scale=1.15,
        )
    )


def _draw_2d_plane_fill(
    axis,
    atom_points: Mapping[int, np.ndarray],
    atom_indices: tuple[int, int, int],
    *,
    color: str,
    label: str,
    zorder: int,
    label_requests: list[_Deferred2DLabel],
) -> None:
    points = [atom_points.get(atom_index) for atom_index in atom_indices]
    if any(point is None for point in points):
        return
    coordinates = np.asarray(points, dtype=float)
    axis.add_patch(
        Polygon(
            coordinates,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.18,
            zorder=zorder,
        )
    )
    centroid = np.mean(coordinates, axis=0)
    label_requests.append(
        _Deferred2DLabel(
            text=label,
            anchor=centroid,
            fontsize=6,
            color=color,
            zorder=zorder + 4,
            priority=45,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.52,
            },
            offset_scale=0.8,
        )
    )


def _draw_2d_dihedral_components(
    axis,
    atom_points: Mapping[int, np.ndarray],
    atom_points3d: Mapping[int, np.ndarray],
    atom_indices: tuple[int, int, int, int],
    atom_label_values: tuple[str, str, str, str],
    label_requests: list[_Deferred2DLabel],
) -> None:
    component = _dihedral_component_geometry(atom_points3d, atom_indices)
    if component is None:
        return
    _d0_3d, in_plane_length, out_of_plane_length = component
    point_c = atom_points.get(atom_indices[2])
    point_d = atom_points.get(atom_indices[3])
    point_d0 = np.asarray(_d0_3d[:2], dtype=float)
    if point_c is None or point_d is None:
        return
    out_of_plane_start = point_d0
    out_of_plane_end = point_d
    projected_out_length = float(np.linalg.norm(point_d - point_d0))
    plot_span = _projected_point_span(atom_points)
    if projected_out_length < plot_span * 0.025:
        reference = point_d0 - point_c
        reference_length = float(np.linalg.norm(reference))
        if reference_length < 1e-8:
            display_direction = np.asarray((0.0, 1.0), dtype=float)
        else:
            display_direction = (
                np.asarray((-reference[1], reference[0]), dtype=float)
                / reference_length
            )
        out_of_plane_end = point_d0 + display_direction * plot_span * 0.075
    axis.plot(
        (point_c[0], point_d0[0]),
        (point_c[1], point_d0[1]),
        color="#b7791f",
        linestyle=":",
        linewidth=1.15,
        alpha=0.88,
        zorder=4,
    )
    axis.plot(
        (out_of_plane_start[0], out_of_plane_end[0]),
        (out_of_plane_start[1], out_of_plane_end[1]),
        color="#7c3aed",
        linestyle="--",
        linewidth=1.15,
        alpha=0.88,
        zorder=4,
    )
    axis.scatter(
        [point_d0[0]],
        [point_d0[1]],
        s=34,
        marker="s",
        facecolors="white",
        edgecolors="#7c3aed",
        linewidths=0.8,
        zorder=6,
    )
    label_requests.append(
        _Deferred2DLabel(
            text=(
                f"in-plane {atom_label_values[2]}-D0\n"
                f"{in_plane_length:.2f} A"
            ),
            anchor=(point_c + point_d0) * 0.5,
            fontsize=6,
            color="#b7791f",
            zorder=7,
            priority=55,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
            },
            offset_scale=0.9,
        )
    )
    label_requests.append(
        _Deferred2DLabel(
            text=(
                f"out-of-plane D0-{atom_label_values[3]}\n"
                f"{out_of_plane_length:.2f} A"
            ),
            anchor=(out_of_plane_start + out_of_plane_end) * 0.5,
            fontsize=6,
            color="#7c3aed",
            zorder=7,
            priority=60,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
            },
            offset_scale=1.0,
        )
    )


def _projected_point_span(atom_points: Mapping[int, np.ndarray]) -> float:
    points = [
        np.asarray(point, dtype=float)[:2] for point in atom_points.values()
    ]
    if not points:
        return 1.0
    coordinates = np.asarray(points, dtype=float)
    span = np.ptp(coordinates, axis=0)
    return max(float(np.max(span)), 1.0)


def _dihedral_atom_indices(
    dihedral: object,
) -> tuple[int, int, int, int] | None:
    return _index_tuple(getattr(dihedral, "atom_indices", None), length=4)


def _dihedral_plane_indices(
    dihedral: object,
    attribute_name: str,
    fallback: tuple[int, int, int],
) -> tuple[int, int, int]:
    explicit_indices = _index_tuple(
        getattr(dihedral, attribute_name, None),
        length=3,
    )
    return explicit_indices if explicit_indices is not None else fallback


def _dihedral_atom_label_tuple(dihedral: object) -> tuple[str, str, str, str]:
    raw_labels = getattr(dihedral, "atom_labels", None)
    if raw_labels is None or isinstance(raw_labels, (str, bytes)):
        return ("A", "B", "C", "D")
    try:
        labels = tuple(str(value) for value in raw_labels)
    except TypeError:
        return ("A", "B", "C", "D")
    if len(labels) != 4:
        return ("A", "B", "C", "D")
    return labels


def _index_tuple(value: object, *, length: int) -> tuple[int, ...] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        indices = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(indices) != length:
        return None
    return indices


def _dihedral_label_text(dihedral: object) -> str:
    label = str(getattr(dihedral, "label", "dihedral") or "dihedral").strip()
    try:
        angle_degrees = float(getattr(dihedral, "angle_degrees"))
    except (TypeError, ValueError):
        return label
    if not np.isfinite(angle_degrees):
        return label
    return f"{label}\nphi {angle_degrees:.1f} deg"


def _dihedral_component_geometry(
    atom_points3d: Mapping[int, np.ndarray],
    atom_indices: tuple[int, int, int, int],
) -> tuple[np.ndarray, float, float] | None:
    points = [atom_points3d.get(atom_index) for atom_index in atom_indices]
    if any(point is None for point in points):
        return None
    point_a, point_b, point_c, point_d = (
        np.asarray(point, dtype=float) for point in points
    )
    normal = np.cross(point_b - point_a, point_c - point_a)
    normal_length = float(np.linalg.norm(normal))
    if normal_length < 1e-8:
        return None
    unit_normal = normal / normal_length
    out_of_plane = float(np.dot(point_d - point_c, unit_normal))
    projected_d = point_d - out_of_plane * unit_normal
    in_plane_length = float(np.linalg.norm(projected_d - point_c))
    return projected_d, in_plane_length, abs(out_of_plane)


def _draw_deferred_2d_labels(
    axis,
    labels: list[_Deferred2DLabel],
) -> None:
    if not labels:
        return
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    xspan = max(abs(float(xlim[1] - xlim[0])), 1.0)
    yspan = max(abs(float(ylim[1] - ylim[0])), 1.0)
    base_dx = xspan * 0.036
    base_dy = yspan * 0.052
    occupied_boxes: list[tuple[float, float, float, float]] = []
    for label in sorted(labels, key=lambda item: item.priority, reverse=True):
        anchor = np.asarray(label.anchor, dtype=float)
        if anchor.size < 2:
            continue
        anchor = anchor[:2]
        width, height = _estimated_label_extent(axis, label, xspan, yspan)
        center, overlap = _placed_label_center(
            anchor=anchor,
            label=label,
            width=width,
            height=height,
            base_dx=base_dx,
            base_dy=base_dy,
            xlim=xlim,
            ylim=ylim,
            occupied_boxes=occupied_boxes,
        )
        box = _label_box(center, width, height)
        label_area = max(width * height, 1e-9)
        if not label.required and overlap / label_area > 0.30:
            continue
        occupied_boxes.append(box)
        offset_distance = float(np.linalg.norm(center - anchor))
        if label.connector and offset_distance > min(xspan, yspan) * 0.018:
            axis.plot(
                (anchor[0], center[0]),
                (anchor[1], center[1]),
                color=label.color,
                linewidth=0.55,
                alpha=0.36,
                zorder=max(label.zorder - 1, 1),
            )
        axis.text(
            center[0],
            center[1],
            label.text,
            fontsize=label.fontsize,
            fontweight=label.fontweight,
            color=label.color,
            ha="center",
            va="center",
            zorder=label.zorder,
            bbox=label.bbox,
            clip_on=False,
        )


def _placed_label_center(
    *,
    anchor: np.ndarray,
    label: _Deferred2DLabel,
    width: float,
    height: float,
    base_dx: float,
    base_dy: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    occupied_boxes: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, float]:
    fallback_center = _clamp_label_center(anchor, width, height, xlim, ylim)
    fallback_overlap = float("inf")
    for offset in _label_candidate_offsets(
        base_dx,
        base_dy,
        scale=label.offset_scale,
    ):
        center = _clamp_label_center(
            anchor + offset,
            width,
            height,
            xlim,
            ylim,
        )
        box = _label_box(center, width, height)
        overlap = sum(
            _label_overlap_area(box, occupied) for occupied in occupied_boxes
        )
        if overlap <= 0.0:
            return center, overlap
        if overlap < fallback_overlap:
            fallback_center = center
            fallback_overlap = overlap
    return fallback_center, fallback_overlap


def _label_candidate_offsets(
    base_dx: float,
    base_dy: float,
    *,
    scale: float,
) -> tuple[np.ndarray, ...]:
    unit_offsets = (
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (-1.0, 0.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (0.0, 1.8),
        (1.8, 0.0),
        (0.0, -1.8),
        (-1.8, 0.0),
        (1.8, 1.8),
        (-1.8, 1.8),
        (1.8, -1.8),
        (-1.8, -1.8),
        (0.0, 2.8),
        (2.8, 0.0),
        (0.0, -2.8),
        (-2.8, 0.0),
        (2.8, 2.8),
        (-2.8, 2.8),
        (2.8, -2.8),
        (-2.8, -2.8),
        (0.0, 4.0),
        (4.0, 0.0),
        (0.0, -4.0),
        (-4.0, 0.0),
        (4.0, 2.2),
        (-4.0, 2.2),
        (4.0, -2.2),
        (-4.0, -2.2),
    )
    return tuple(
        np.asarray((x_value * base_dx * scale, y_value * base_dy * scale))
        for x_value, y_value in unit_offsets
    )


def _estimated_label_extent(
    axis,
    label: _Deferred2DLabel,
    xspan: float,
    yspan: float,
) -> tuple[float, float]:
    lines = str(label.text).splitlines() or [""]
    figure = axis.figure
    position = axis.get_position()
    axis_width_px = max(position.width * figure.get_figwidth() * figure.dpi, 1)
    axis_height_px = max(
        position.height * figure.get_figheight() * figure.dpi,
        1,
    )
    max_chars = max(len(line) for line in lines)
    char_px = label.fontsize * figure.dpi / 72.0
    width_px = max_chars * char_px * 0.62 + 16.0
    height_px = len(lines) * char_px * 1.34 + 11.0
    return (
        width_px / axis_width_px * xspan,
        height_px / axis_height_px * yspan,
    )


def _clamp_label_center(
    center: np.ndarray,
    width: float,
    height: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> np.ndarray:
    x_min, x_max = min(xlim), max(xlim)
    y_min, y_max = min(ylim), max(ylim)
    half_width = width * 0.5
    half_height = height * 0.5
    if x_min + half_width >= x_max - half_width:
        x_value = (x_min + x_max) * 0.5
    else:
        x_value = float(
            np.clip(center[0], x_min + half_width, x_max - half_width)
        )
    if y_min + half_height >= y_max - half_height:
        y_value = (y_min + y_max) * 0.5
    else:
        y_value = float(
            np.clip(center[1], y_min + half_height, y_max - half_height)
        )
    return np.asarray(
        (
            x_value,
            y_value,
        ),
        dtype=float,
    )


def _label_box(
    center: np.ndarray,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    return (
        float(center[0] - width * 0.5),
        float(center[1] - height * 0.5),
        float(center[0] + width * 0.5),
        float(center[1] + height * 0.5),
    )


def _label_overlap_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    y_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return x_overlap * y_overlap


def _draw_2d_bond_annotation(
    axis,
    atom_points: Mapping[int, np.ndarray],
    atom1_index: int,
    atom2_index: int,
    *,
    label: str,
    color: str,
    linestyle: str,
    linewidth: float,
    zorder: int,
    label_requests: list[_Deferred2DLabel],
    show_label: bool = True,
) -> None:
    start = atom_points.get(atom1_index)
    end = atom_points.get(atom2_index)
    if start is None or end is None:
        return
    midpoint = (start + end) * 0.5
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length > 0.0:
        normal = np.array([-delta[1], delta[0]], dtype=float) / length
        midpoint = midpoint + normal * 0.055
    axis.plot(
        (start[0], end[0]),
        (start[1], end[1]),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.9,
        zorder=zorder,
    )
    if not show_label:
        return
    label_requests.append(
        _Deferred2DLabel(
            text=label,
            anchor=midpoint,
            fontsize=7,
            color=color,
            zorder=zorder + 3,
            priority=50 if linestyle == "--" else 40,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            offset_scale=0.8,
        )
    )


def _draw_2d_angle_annotation(
    axis,
    atom_points: Mapping[int, np.ndarray],
    angle: EXAFSAngleAnnotation,
    label_requests: list[_Deferred2DLabel],
) -> None:
    absorber = atom_points.get(angle.absorber_index)
    bridge = atom_points.get(angle.bridge_index)
    terminal = atom_points.get(angle.terminal_index)
    if absorber is None or bridge is None or terminal is None:
        return
    absorber_vector = absorber - bridge
    terminal_vector = terminal - bridge
    absorber_length = float(np.linalg.norm(absorber_vector))
    terminal_length = float(np.linalg.norm(terminal_vector))
    if absorber_length < 1e-6 or terminal_length < 1e-6:
        return
    theta1 = float(
        np.degrees(np.arctan2(absorber_vector[1], absorber_vector[0]))
    )
    theta2 = float(
        np.degrees(np.arctan2(terminal_vector[1], terminal_vector[0]))
    )
    arc_start, arc_end = _minor_arc_span(theta1, theta2)
    radius = min(max(min(absorber_length, terminal_length) * 0.26, 0.22), 0.75)
    axis.add_patch(
        Arc(
            (bridge[0], bridge[1]),
            width=radius * 2.0,
            height=radius * 2.0,
            angle=0.0,
            theta1=arc_start,
            theta2=arc_end,
            color="#2f7d32",
            linewidth=1.15,
            alpha=0.86,
            zorder=5,
        )
    )
    mid_angle = np.radians((arc_start + arc_end) * 0.5)
    label_point = bridge + np.array(
        [np.cos(mid_angle), np.sin(mid_angle)],
        dtype=float,
    ) * (radius + 0.16)
    label_requests.append(
        _Deferred2DLabel(
            text=f"{angle.atom_triplet_label}\n{angle.angle_degrees:.1f} deg",
            anchor=label_point,
            fontsize=6.5,
            color="#2f7d32",
            zorder=7,
            priority=65,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.66,
            },
            offset_scale=0.95,
        )
    )


def _minor_arc_span(theta1: float, theta2: float) -> tuple[float, float]:
    normalized1 = theta1 % 360.0
    normalized2 = theta2 % 360.0
    delta = (normalized2 - normalized1) % 360.0
    if delta <= 180.0:
        return normalized1, normalized1 + delta
    return normalized2, normalized2 + (360.0 - delta)


def launch_exafs_gds_mapping_ui(
    initial_project_dir: str | Path | None = None,
    initial_absorber_element: str | None = None,
) -> EXAFSGDSMappingMainWindow | int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = EXAFSGDSMappingMainWindow(
        initial_project_dir=initial_project_dir,
        initial_absorber_element=initial_absorber_element,
    )
    track_saxshell_window(window, _OPEN_WINDOWS)
    window.show()
    window.raise_()
    if owns_app:
        return app.exec()
    return window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="exafs-gds-mapping-ui",
        description="Launch the SAXSShell EXAFS GDS Mapping UI.",
    )
    parser.add_argument("project_dir", nargs="?")
    parser.add_argument(
        "--absorber-element",
        default=None,
        help="Preferred absorber/coordination-center element.",
    )
    args = parser.parse_args(argv)
    result = launch_exafs_gds_mapping_ui(
        args.project_dir,
        initial_absorber_element=args.absorber_element,
    )
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())


def _normalize_element_text(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:1].upper() + text[1:].lower()


def _natural_label_sort_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def _template_mode_from_context_values(*values: object) -> str | None:
    text = " ".join(str(value or "") for value in values).upper()
    if _has_solvent_token(text, "DMF"):
        return "pb_dmf"
    if _has_solvent_token(text, "DMSO"):
        return "pb_dmso"
    return None


def _has_solvent_token(text: str, token: str) -> bool:
    return (
        re.search(
            rf"(^|[^A-Z0-9]){re.escape(token)}([^A-Z0-9]|$)",
            text,
        )
        is not None
    )


def _reveal_file_in_file_manager(path: Path) -> None:
    subprocess.Popen(_file_manager_reveal_command(path))


def _file_manager_reveal_command(
    path: Path,
    *,
    platform: str | None = None,
) -> list[str]:
    resolved = Path(path).expanduser().resolve()
    platform_name = platform or sys.platform
    if platform_name == "darwin":
        return ["open", "-R", str(resolved)]
    if platform_name.startswith("win"):
        return ["explorer", f"/select,{resolved}"]
    return ["xdg-open", str(resolved.parent)]


__all__ = ["EXAFSGDSMappingMainWindow", "launch_exafs_gds_mapping_ui", "main"]
