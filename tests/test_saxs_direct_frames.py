from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import saxshell.saxs.debye.profiles as profiles
from saxshell.saxs.cli import main as saxs_main
from saxshell.saxs.direct_frames import (
    compute_direct_frame_saxs,
    discover_xyz_frame_paths,
    estimate_finite_box_diagnostics,
)
from saxshell.saxs.ui.direct_frame_window import DirectFrameSAXSWindow


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class FakeXrayDB:
    @staticmethod
    def f0(element: str, sin_theta_over_lambda: float) -> np.ndarray:
        base = {
            "H": 1.0,
            "C": 6.0,
            "O": 8.0,
        }[element]
        return np.asarray([base + float(sin_theta_over_lambda)], dtype=float)


def _write_xyz(
    path: Path,
    rows: list[tuple[str, float, float, float]],
) -> None:
    path.write_text(
        "\n".join(
            [
                str(len(rows)),
                path.stem,
                *(
                    f"{element} {x:.6f} {y:.6f} {z:.6f}"
                    for element, x, y, z in rows
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_discover_xyz_frame_paths_uses_natural_order(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(frames_dir / "frame_10.xyz", [("C", 0.0, 0.0, 0.0)])
    _write_xyz(frames_dir / "frame_2.xyz", [("C", 0.0, 0.0, 0.0)])
    _write_xyz(frames_dir / "frame_1.xyz", [("C", 0.0, 0.0, 0.0)])

    discovered = discover_xyz_frame_paths(frames_dir, max_frames=2)

    assert [path.name for path in discovered] == [
        "frame_1.xyz",
        "frame_2.xyz",
    ]


def test_discover_xyz_frame_paths_recurses_pdb_and_xyz(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    nested_dir = frames_dir / "PbI2"
    nested_dir.mkdir(parents=True)
    _write_xyz(nested_dir / "frame_10.xyz", [("C", 0.0, 0.0, 0.0)])
    (nested_dir / "frame_2.pdb").write_text(
        "ATOM      1 C1   MOL A   1       0.000   0.000   0.000  1.00"
        "  0.00           C\n"
        "END\n",
        encoding="utf-8",
    )

    discovered = discover_xyz_frame_paths(frames_dir)

    assert [path.name for path in discovered] == [
        "frame_2.pdb",
        "frame_10.xyz",
    ]


def test_estimate_finite_box_diagnostics_flags_low_q_points() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [10.0, 1.0, 1.0]],
        dtype=float,
    )
    q_values = np.asarray([0.05, 0.07, 0.20], dtype=float)

    diagnostics = estimate_finite_box_diagnostics(
        coordinates,
        q_values,
        box_length_a=100.0,
    )

    assert diagnostics.inferred_box_length_a == 100.0
    assert diagnostics.q_points_below_fundamental == 1
    assert diagnostics.q_points_below_caution == 2


def test_compute_direct_frame_saxs_writes_profiles_and_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(profiles, "xraydb", FakeXrayDB())
    frames_dir = tmp_path / "frames"
    output_dir = tmp_path / "direct_frame_saxs_beta"
    frames_dir.mkdir()
    _write_xyz(
        frames_dir / "frame_0001.xyz",
        [
            ("C", 0.0, 0.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
        ],
    )
    _write_xyz(
        frames_dir / "frame_0002.xyz",
        [
            ("C", 0.0, 0.0, 0.0),
            ("O", 2.0, 0.0, 0.0),
        ],
    )

    result = compute_direct_frame_saxs(
        frames_dir,
        output_dir=output_dir,
        q_min=0.10,
        q_max=0.30,
        q_step=0.10,
        box_length_a=20.0,
        write_plots=False,
    )

    assert result.frame_intensities.shape == (2, 3)
    np.testing.assert_allclose(
        result.mean_intensity,
        np.mean(result.frame_intensities, axis=0),
    )
    np.testing.assert_allclose(
        result.variance_intensity,
        np.var(result.frame_intensities, axis=0),
    )
    assert result.profile_csv_path.is_file()
    assert result.profile_txt_path.is_file()
    assert result.frame_trace_csv_path.is_file()
    assert result.metadata_json_path.is_file()
    assert result.method_notes_path.is_file()
    assert result.figures == ()
    metadata = json.loads(result.metadata_json_path.read_text("utf-8"))
    assert metadata["frame_count"] == 2
    assert metadata["element_counts_total"] == {"C": 2, "O": 2}
    assert metadata["mode"] == "direct_frame_saxs_beta"
    profile_text = result.profile_txt_path.read_text("utf-8")
    assert "variance_intensity" in profile_text
    assert "variance_band_lower_intensity" in profile_text
    assert "Dohn et al." in result.method_notes_path.read_text("utf-8")


def test_compute_direct_frame_saxs_writes_loglog_experimental_overlay(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(profiles, "xraydb", FakeXrayDB())
    frame_path = tmp_path / "frame_0001.xyz"
    output_dir = tmp_path / "direct_frame_saxs_beta"
    experimental_path = tmp_path / "experiment.txt"
    _write_xyz(frame_path, [("C", 0.0, 0.0, 0.0)])
    q_values = np.asarray([0.1, 0.2, 0.3], dtype=float)
    f_c = 6.0 + q_values / (4.0 * np.pi)
    experimental = 3.0 * np.square(f_c)
    np.savetxt(
        experimental_path,
        np.column_stack((q_values, experimental)),
        fmt="%.10f",
    )

    result = compute_direct_frame_saxs(
        frame_path,
        output_dir=output_dir,
        q_min=0.10,
        q_max=0.30,
        q_step=0.10,
        box_length_a=20.0,
        experimental_data_path=experimental_path,
    )

    assert result.experimental_overlay is not None
    assert result.experimental_overlay.plot_path.is_file()
    assert result.experimental_overlay.scaled_profile_csv_path.is_file()
    assert abs(result.experimental_overlay.scale_factor - 3.0) < 1.0e-6
    metadata = json.loads(result.metadata_json_path.read_text("utf-8"))
    assert metadata["experimental_overlay"]["fit_point_count"] == 3


def test_average_box_density_overlay_scales_above_finite_box_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(profiles, "xraydb", FakeXrayDB())
    frame_path = tmp_path / "frame_0001.xyz"
    _write_xyz(
        frame_path,
        [
            ("C", 0.0, 0.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
        ],
    )

    baseline = compute_direct_frame_saxs(
        frame_path,
        output_dir=tmp_path / "baseline",
        q_min=0.05,
        q_max=0.20,
        q_step=0.05,
        box_lengths_a=(100.0, 100.0, 100.0),
        subtract_average_box_density=True,
        direction_count=16,
        write_plots=False,
    )
    experimental_path = tmp_path / "experiment.txt"
    experimental = baseline.mean_intensity.copy()
    experimental[0] *= 20.0
    experimental[1:] *= 3.0
    np.savetxt(
        experimental_path,
        np.column_stack((baseline.q_values, experimental)),
        fmt="%.10f",
    )

    result = compute_direct_frame_saxs(
        frame_path,
        output_dir=tmp_path / "overlay",
        q_min=0.05,
        q_max=0.20,
        q_step=0.05,
        box_lengths_a=(100.0, 100.0, 100.0),
        subtract_average_box_density=True,
        direction_count=16,
        experimental_data_path=experimental_path,
    )

    assert result.experimental_overlay is not None
    assert abs(result.experimental_overlay.scale_factor - 3.0) < 1.0e-6
    assert result.experimental_overlay.fit_point_count == 3
    assert (
        result.experimental_overlay.fit_q_min_a_inverse
        >= result.diagnostics.q_fundamental_a_inverse
    )


def test_compute_direct_frame_saxs_can_subtract_average_box_density(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(profiles, "xraydb", FakeXrayDB())
    frame_path = tmp_path / "frame_0001.xyz"
    output_dir = tmp_path / "direct_frame_saxs_beta"
    _write_xyz(
        frame_path,
        [
            ("C", 0.0, 0.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
        ],
    )

    result = compute_direct_frame_saxs(
        frame_path,
        output_dir=output_dir,
        q_min=0.10,
        q_max=0.20,
        q_step=0.10,
        box_lengths_a=(10.0, 10.0, 10.0),
        subtract_average_box_density=True,
        direction_count=24,
        write_plots=False,
    )

    assert result.calculation_mode == "average_box_density_contrast"
    assert result.medium_density_e_per_a3 is not None
    assert result.medium_density_e_per_a3 > 0.0
    metadata = json.loads(result.metadata_json_path.read_text("utf-8"))
    assert metadata["settings"]["box_lengths_a"] == [10.0, 10.0, 10.0]
    assert metadata["settings"]["subtract_average_box_density"] is True
    assert metadata["calculation_mode"] == "average_box_density_contrast"


def test_direct_frame_ui_collects_example_box_contrast_settings(
    qapp,
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    output_dir = tmp_path / "direct_frame"
    frames_dir.mkdir()
    window = DirectFrameSAXSWindow(
        initial_input_path=frames_dir,
        initial_output_dir=output_dir,
        initial_q_min=0.049934,
        initial_q_max=4.4578,
    )
    try:
        window.box_lx_edit.setText("14.06")
        window.box_ly_edit.setText("14.06")
        window.box_lz_edit.setText("14.06")

        request = window._collect_run_request()
    finally:
        window.close()

    assert request.input_path == frames_dir.resolve()
    assert request.output_dir == output_dir.resolve()
    assert request.q_min == 0.049934
    assert request.q_max == 4.4578
    assert request.q_step == 0.01
    assert request.max_frames == 20
    assert request.box_lengths_a == (14.06, 14.06, 14.06)
    assert request.subtract_average_box_density is True
    assert request.direction_count == 256
    assert abs(request.scale_fit_q_min - (4.0 * math.pi / 14.06)) < 1.0e-12


def test_saxs_direct_frame_cli_reports_outputs(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    fake_result = SimpleNamespace(
        frame_paths=(tmp_path / "frame_0001.xyz",),
        profile_csv_path=output_dir / "profile.csv",
        profile_txt_path=output_dir / "profile.txt",
        frame_trace_csv_path=output_dir / "traces.csv",
        metadata_json_path=output_dir / "metadata.json",
        method_notes_path=output_dir / "method_notes.md",
        figures=(output_dir / "overlay.png",),
        diagnostics=SimpleNamespace(
            q_fundamental_a_inverse=0.314159,
            q_points_below_fundamental=3,
        ),
        calculation_mode="finite_frame_atomic_debye_vacuum",
        medium_density_e_per_a3=None,
    )
    called = []

    def fake_compute(*args, **kwargs):
        called.append((args, kwargs))
        return fake_result

    monkeypatch.setattr(
        "saxshell.saxs.cli.compute_direct_frame_saxs",
        fake_compute,
    )

    assert (
        saxs_main(
            [
                "direct-frame-saxs",
                str(tmp_path / "frames"),
                "--output-dir",
                str(output_dir),
                "--q-min",
                "0.1",
                "--q-max",
                "0.3",
                "--q-step",
                "0.1",
                "--max-frames",
                "1",
                "--no-plots",
            ]
        )
        == 0
    )

    assert called
    assert called[0][1]["write_plots"] is False
    assert called[0][1]["max_frames"] == 1
    assert called[0][1]["subtract_average_box_density"] is False
    output = capsys.readouterr().out
    assert "Computed 1 frame(s)." in output
    assert "Finite-box diagnostic: 2*pi/L" in output
