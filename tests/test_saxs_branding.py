from __future__ import annotations

import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from saxshell.saxs.ui.branding import (
    SAXSHELL_THEME_COLORS,
    _reset_qt_message_filter_state,
    _should_suppress_qt_message,
    configure_saxshell_application,
    load_saxshell_brand_pixmap,
    load_saxshell_icon,
    track_saxshell_window,
)


def _contrast_ratio(color_a: QColor, color_b: QColor) -> float:
    def _relative_luminance(color: QColor) -> float:
        channels = []
        for channel in (color.redF(), color.greenF(), color.blueF()):
            if channel <= 0.03928:
                channels.append(channel / 12.92)
            else:
                channels.append(((channel + 0.055) / 1.055) ** 2.4)
        return (
            0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
        )

    luminance_a = _relative_luminance(color_a)
    luminance_b = _relative_luminance(color_b)
    lighter = max(luminance_a, luminance_b)
    darker = min(luminance_a, luminance_b)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_load_saxshell_icon_handles_embedded_raster_svg_without_qt_warning(
    qapp,
    capfd,
):
    del qapp
    load_saxshell_icon.cache_clear()

    icon = load_saxshell_icon()
    pixmap = icon.pixmap(64, 64)
    captured = capfd.readouterr()

    assert not icon.isNull()
    assert not pixmap.isNull()
    assert "Image filename is empty" not in captured.err


def test_load_saxshell_brand_pixmap_fills_requested_height(qapp):
    del qapp

    pixmap = load_saxshell_brand_pixmap(34)
    logical_width = pixmap.width() / pixmap.devicePixelRatio()
    logical_height = pixmap.height() / pixmap.devicePixelRatio()

    assert not pixmap.isNull()
    assert logical_height == pytest.approx(34.0)
    assert logical_width > logical_height


def test_configure_saxshell_application_applies_shared_theme(qapp):
    configure_saxshell_application(qapp)

    stylesheet = qapp.styleSheet()
    palette = qapp.palette()

    assert "QGroupBox" in stylesheet
    assert "QTabWidget::pane" in stylesheet
    assert SAXSHELL_THEME_COLORS["primary_soft"] in stylesheet
    assert (
        palette.color(QPalette.ColorRole.Window).name()
        == SAXSHELL_THEME_COLORS["window"]
    )


def test_saxshell_theme_keeps_fields_and_tables_readable(qapp):
    configure_saxshell_application(qapp)

    stylesheet = qapp.styleSheet()
    palette = qapp.palette()
    line_edit = QLineEdit("readable field")
    read_only_edit = QPlainTextEdit("readable JSON preview")
    read_only_edit.setReadOnly(True)
    table = QTableWidget(1, 1)
    table.setItem(0, 0, QTableWidgetItem("readable table value"))

    field_text = line_edit.palette().color(QPalette.ColorRole.Text)
    table_text = table.palette().color(QPalette.ColorRole.Text)
    disabled_text = palette.color(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
    )
    disabled_base = palette.color(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Base,
    )

    assert field_text.name() == SAXSHELL_THEME_COLORS["field_text"]
    assert table_text.name() == SAXSHELL_THEME_COLORS["field_text"]
    assert read_only_edit.isReadOnly()
    assert (
        _contrast_ratio(
            field_text,
            QColor(SAXSHELL_THEME_COLORS["field_surface"]),
        )
        >= 7
    )
    assert (
        _contrast_ratio(
            table_text,
            QColor(SAXSHELL_THEME_COLORS["field_surface"]),
        )
        >= 7
    )
    assert (
        _contrast_ratio(
            QColor(SAXSHELL_THEME_COLORS["editor_text"]),
            QColor(SAXSHELL_THEME_COLORS["editor_surface"]),
        )
        >= 7
    )
    assert _contrast_ratio(disabled_text, disabled_base) >= 4.5
    assert 'QLineEdit[readOnly="true"]' in stylesheet
    assert "QTextEdit, QPlainTextEdit" in stylesheet
    assert 'font-family: "Menlo"' in stylesheet
    assert SAXSHELL_THEME_COLORS["field_surface"] in stylesheet
    assert SAXSHELL_THEME_COLORS["editor_surface"] in stylesheet
    assert SAXSHELL_THEME_COLORS["editor_text"] in stylesheet
    assert "QTableView::item:disabled" in stylesheet
    assert "QTableView:disabled" in stylesheet
    assert SAXSHELL_THEME_COLORS["table_text"] in stylesheet


def test_dream_batch_text_previews_use_readable_editor_theme(qapp):
    configure_saxshell_application(qapp)

    from saxshell.saxs.ui.dream_batch_window import (
        DREAM_BATCH_JSON_EDITOR_OBJECT_NAME,
        DreamBatchRunFileWindow,
    )

    window = DreamBatchRunFileWindow()

    try:
        editors = (
            window.settings_json_edit,
            window.filter_json_edit,
            window.command_box,
        )
        stylesheet = qapp.styleSheet()

        assert all(isinstance(editor, QPlainTextEdit) for editor in editors)
        assert window.command_box.isReadOnly()
        assert "QTextEdit, QPlainTextEdit" in stylesheet
        assert SAXSHELL_THEME_COLORS["editor_text"] in stylesheet
        assert SAXSHELL_THEME_COLORS["editor_surface"] in stylesheet
        assert 'font-family: "Menlo"' in stylesheet

        for editor in (
            window.settings_json_edit,
            window.filter_json_edit,
        ):
            assert editor.objectName() == DREAM_BATCH_JSON_EDITOR_OBJECT_NAME
            assert SAXSHELL_THEME_COLORS["editor_text"] in editor.styleSheet()
            assert "font-weight: 650" in editor.styleSheet()
            assert (
                editor.palette().color(QPalette.ColorRole.Text).name()
                == SAXSHELL_THEME_COLORS["editor_text"]
            )
            assert (
                editor.palette()
                .color(
                    QPalette.ColorGroup.Disabled,
                    QPalette.ColorRole.Text,
                )
                .name()
                == SAXSHELL_THEME_COLORS["editor_text"]
            )
    finally:
        window.close()


def test_qt_message_filter_only_suppresses_noisy_painter_warnings(
    monkeypatch,
):
    _reset_qt_message_filter_state()
    state_message = "QPainter::end: Painter ended with 2 saved states"
    pixmap_begin_message = (
        "QPainter::begin: Paint device returned engine == 0, type: 2"
    )

    monkeypatch.delenv("SAXSHELL_SHOW_QPAINTER_STATE_WARNINGS", raising=False)

    assert _should_suppress_qt_message(state_message)
    assert _should_suppress_qt_message(pixmap_begin_message)
    assert _should_suppress_qt_message("QPainter::setPen: Painter not active")
    assert _should_suppress_qt_message(
        "QPainter::setBrush: Painter not active"
    )
    assert _should_suppress_qt_message(
        "QPainter::setRenderHint: Painter must be active to set rendering hints"
    )
    assert not _should_suppress_qt_message(
        "QPainter::setPen: Painter not active"
    )
    assert not _should_suppress_qt_message(
        "QPainter::begin: Paint device returned engine == 0"
    )
    assert not _should_suppress_qt_message("unrelated Qt warning")

    assert _should_suppress_qt_message(pixmap_begin_message)
    assert not _should_suppress_qt_message("unrelated Qt warning")
    assert not _should_suppress_qt_message(
        "QPainter::setPen: Painter not active"
    )

    monkeypatch.setenv("SAXSHELL_SHOW_QPAINTER_STATE_WARNINGS", "1")

    assert not _should_suppress_qt_message(state_message)
    assert not _should_suppress_qt_message(pixmap_begin_message)


def test_track_saxshell_window_releases_registry_on_close(qapp):
    registry: list[QWidget] = []
    window = QWidget()

    track_saxshell_window(window, registry)
    window.show()
    qapp.processEvents()

    assert registry == [window]
    assert window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    window.close()
    qapp.processEvents()
    qapp.sendPostedEvents(None, 0)
    qapp.processEvents()

    assert registry == []
