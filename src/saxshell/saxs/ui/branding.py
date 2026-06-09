from __future__ import annotations

import base64
import os
import sys
import weakref
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote_to_bytes
from xml.etree import ElementTree

from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QSize,
    Qt,
    qInstallMessageHandler,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QImage,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplashScreen,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

SAXSHELL_APPLICATION_NAME = "SAXSShell"
BRAND_PRIMARY_HEX = "#0f5c73"
BRAND_SECONDARY_HEX = "#52666b"
BRAND_ACCENT_HEX = "#b4652a"
BRAND_ICON_MIN_SIZE = 34
BRAND_ICON_MAX_SIZE = 34
BRAND_TITLE_MAX_POINT_SIZE = 13.5
BRAND_ICON_MAX_ASPECT_RATIO = 3.0
SAXSHELL_THEME_COLORS = {
    "window": "#edf3f1",
    "surface": "#ffffff",
    "surface_alt": "#f7faf9",
    "surface_muted": "#eef5f3",
    "field_surface": "#f8fbfa",
    "field_surface_readonly": "#f1f7f5",
    "editor_surface": "#f4f8f6",
    "text": "#1d2d2f",
    "muted": "#5f7071",
    "border": "#d4dfdc",
    "border_strong": "#b9c8c4",
    "primary": BRAND_PRIMARY_HEX,
    "primary_hover": "#0b7189",
    "primary_pressed": "#09485b",
    "primary_soft": "#e2f2f3",
    "accent": BRAND_ACCENT_HEX,
    "accent_soft": "#fff1e5",
    "success": "#177245",
    "warning": "#a15c07",
    "danger": "#b42318",
    "focus": "#2b8c7e",
    "selection": "#0b7189",
    "selection_text": "#ffffff",
    "disabled": "#5f7071",
    "placeholder": "#687b7c",
    "field_text": "#0d2529",
    "editor_text": "#0a1f24",
    "table_text": "#0d2529",
}

_SUPPRESSED_QT_PAINTER_STATE_WARNING_PREFIX = (
    "QPainter::end: Painter ended with "
)
_SUPPRESSED_QT_PIXMAP_PAINTER_BEGIN_PREFIX = (
    "QPainter::begin: Paint device returned engine == 0, type: 2"
)
_SUPPRESSED_QT_INACTIVE_PAINTER_FOLLOWUP_PREFIXES = (
    "QPainter::setPen: Painter not active",
    "QPainter::setBrush: Painter not active",
    "QPainter::setRenderHint: Painter must be active",
)
_QT_INACTIVE_PAINTER_FOLLOWUP_BUDGET = len(
    _SUPPRESSED_QT_INACTIVE_PAINTER_FOLLOWUP_PREFIXES
)
_pending_qt_inactive_painter_followups = 0
_QT_MESSAGE_FILTER_INSTALLED = False
_PREVIOUS_QT_MESSAGE_HANDLER = None
_SAXSHELL_THEME_APPLIED_PROPERTY = "_saxshell_theme_applied"
_SAXSHELL_THEME_VERSION = "saxshell-theme-2026-06-01"
_SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_NON_DRAWING_SVG_TAGS = {
    f"{_SVG_NAMESPACE}defs",
    f"{_SVG_NAMESPACE}desc",
    f"{_SVG_NAMESPACE}metadata",
    f"{_SVG_NAMESPACE}style",
    f"{_SVG_NAMESPACE}title",
}


def saxshell_icon_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "_ui_assets"
        / ("saxshell_icon.svg")
    )


def _decode_data_uri_image(data_uri: str) -> bytes | None:
    metadata, separator, data = data_uri.partition(",")
    if separator == "":
        return None
    if ";base64" in metadata:
        try:
            return base64.b64decode("".join(data.split()))
        except ValueError:
            return None
    return unquote_to_bytes(data)


def _trim_transparent_padding(pixmap: QPixmap) -> QPixmap:
    if pixmap.isNull():
        return pixmap

    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    if width <= 0 or height <= 0:
        return pixmap

    bytes_per_line = image.bytesPerLine()
    image_bytes = bytes(image.constBits())

    # QImage.pixelColor() is too slow for the large embedded icon; scan alpha
    # bytes directly so startup does not stall on transparent padding removal.
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1

    for y in range(height):
        row_start = y * bytes_per_line
        row = image_bytes[row_start : row_start + width * 4]
        alpha_values = row[3::4]
        if not any(alpha_values):
            continue

        first_opaque_x = next(
            x for x, alpha in enumerate(alpha_values) if alpha
        )
        last_opaque_x = (
            width
            - 1
            - next(
                x for x, alpha in enumerate(reversed(alpha_values)) if alpha
            )
        )
        min_x = min(min_x, first_opaque_x)
        min_y = min(min_y, y)
        max_x = max(max_x, last_opaque_x)
        max_y = max(max_y, y)

    if max_x < min_x or max_y < min_y:
        return pixmap
    if (
        min_x == 0
        and min_y == 0
        and max_x == width - 1
        and max_y == height - 1
    ):
        return pixmap

    return pixmap.copy(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


@lru_cache(maxsize=4)
def _load_embedded_svg_raster_pixmap(path: Path) -> QPixmap | None:
    try:
        root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ElementTree.ParseError):
        return None

    image_element: ElementTree.Element | None = None
    for element in root.iter():
        if element is root:
            continue
        if element.tag == f"{_SVG_NAMESPACE}image":
            if image_element is not None:
                return None
            image_element = element
            continue
        if element.tag not in _NON_DRAWING_SVG_TAGS:
            return None

    if image_element is None:
        return None

    href = image_element.get("href") or image_element.get(_XLINK_HREF)
    if not href or not href.startswith("data:image/"):
        return None

    image_bytes = _decode_data_uri_image(href)
    if image_bytes is None:
        return None

    pixmap = QPixmap()
    if not pixmap.loadFromData(image_bytes):
        return None
    return _trim_transparent_padding(pixmap)


def _load_embedded_svg_raster_icon(path: Path) -> QIcon | None:
    pixmap = _load_embedded_svg_raster_pixmap(path)
    if pixmap is None:
        return None
    return QIcon(pixmap)


@lru_cache(maxsize=1)
def load_saxshell_icon() -> QIcon:
    path = saxshell_icon_path()
    if path.suffix.lower() == ".svg":
        # Qt's SVG handler warns on embedded raster data URIs even though it
        # still renders them, so decode that case ourselves first.
        embedded_icon = _load_embedded_svg_raster_icon(path)
        if embedded_icon is not None:
            return embedded_icon
    return QIcon(str(path))


@lru_cache(maxsize=16)
def load_saxshell_brand_pixmap(target_height: int) -> QPixmap:
    target_height = max(1, int(target_height))
    path = saxshell_icon_path()
    if path.suffix.lower() == ".svg":
        embedded_pixmap = _load_embedded_svg_raster_pixmap(path)
        if embedded_pixmap is not None:
            return embedded_pixmap.scaledToHeight(
                target_height,
                Qt.TransformationMode.SmoothTransformation,
            )

    max_width = max(
        target_height,
        round(target_height * BRAND_ICON_MAX_ASPECT_RATIO),
    )
    return load_saxshell_icon().pixmap(QSize(max_width, target_height))


class SAXShellBrandWidget(QWidget):
    """Top-left application branding that tracks UI font scaling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("saxshellBrandWidget")
        self.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Fixed,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(14, 2, 16, 2)
        self._layout.setSpacing(10)

        self._icon_label = QLabel(self)
        self._icon_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self._layout.addWidget(
            self._icon_label,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(0)

        self._title_label = QLabel("SAXSShell", self)
        self._title_label.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Fixed,
        )
        self._title_label.setStyleSheet(f"color: {BRAND_PRIMARY_HEX};")
        text_column.addWidget(self._title_label)

        self._layout.addLayout(text_column)
        self._sync_brand_metrics()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        ):
            self._sync_brand_metrics()

    def _sync_brand_metrics(self) -> None:
        base_font = QFont(self.font())
        base_point_size = base_font.pointSizeF()
        if base_point_size <= 0:
            app_font = QApplication.font(self)
            base_point_size = app_font.pointSizeF()
            base_font = QFont(app_font)
        if base_point_size <= 0:
            base_point_size = 10.0

        title_font = QFont(base_font)
        title_font.setBold(True)
        title_font.setPointSizeF(
            min(
                max(base_point_size * 1.18, base_point_size + 1.5),
                BRAND_TITLE_MAX_POINT_SIZE,
            )
        )
        self._title_label.setFont(title_font)

        icon_size = round(title_font.pointSizeF() * 2.35)
        icon_size = max(
            BRAND_ICON_MIN_SIZE, min(BRAND_ICON_MAX_SIZE, icon_size)
        )
        icon_pixmap = load_saxshell_brand_pixmap(icon_size)
        icon_width = round(
            icon_pixmap.width() / icon_pixmap.devicePixelRatio()
        )
        icon_height = round(
            icon_pixmap.height() / icon_pixmap.devicePixelRatio()
        )
        self._icon_label.setPixmap(icon_pixmap)
        self._icon_label.setFixedSize(icon_width, icon_height)

        layout_size = self._layout.sizeHint()
        self.setMinimumWidth(layout_size.width())
        self.setFixedHeight(layout_size.height())
        self.updateGeometry()


def build_saxshell_brand_widget(parent: QWidget | None = None) -> QWidget:
    return SAXShellBrandWidget(parent)


def _remove_tracked_window(
    registry: list[QWidget],
    window_ref: weakref.ReferenceType[QWidget],
) -> None:
    window = window_ref()
    if window is None:
        return
    registry[:] = [existing for existing in registry if existing is not window]


def track_saxshell_window(
    window: QWidget,
    registry: list[QWidget],
    *,
    delete_on_close: bool = True,
) -> None:
    """Keep a top-level window alive without retaining it forever."""
    if window not in registry:
        registry.append(window)
    if delete_on_close:
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    window_ref = weakref.ref(window)
    window.destroyed.connect(
        lambda _obj=None, ref=window_ref: _remove_tracked_window(
            registry,
            ref,
        )
    )


def _should_suppress_qt_message(message: str) -> bool:
    global _pending_qt_inactive_painter_followups

    if os.environ.get("SAXSHELL_SHOW_QPAINTER_STATE_WARNINGS"):
        _pending_qt_inactive_painter_followups = 0
        return False

    if message.startswith(_SUPPRESSED_QT_PAINTER_STATE_WARNING_PREFIX):
        return True

    if message.startswith(_SUPPRESSED_QT_PIXMAP_PAINTER_BEGIN_PREFIX):
        _pending_qt_inactive_painter_followups = (
            _QT_INACTIVE_PAINTER_FOLLOWUP_BUDGET
        )
        return True

    if _pending_qt_inactive_painter_followups > 0:
        if message.startswith(
            _SUPPRESSED_QT_INACTIVE_PAINTER_FOLLOWUP_PREFIXES
        ):
            _pending_qt_inactive_painter_followups -= 1
            return True
        _pending_qt_inactive_painter_followups = 0

    return False


def _reset_qt_message_filter_state() -> None:
    global _pending_qt_inactive_painter_followups
    _pending_qt_inactive_painter_followups = 0


def _saxshell_qt_message_handler(mode, context, message: str) -> None:
    if _should_suppress_qt_message(str(message)):
        return

    if _PREVIOUS_QT_MESSAGE_HANDLER is not None:
        _PREVIOUS_QT_MESSAGE_HANDLER(mode, context, message)
        return

    print(message, file=sys.stderr)


def install_saxshell_qt_message_filter() -> None:
    """Suppress noisy benign Qt painter diagnostics.

    Qt 6 on macOS can repeatedly print
    ``QPainter::end: Painter ended with N saved states`` while repainting
    native/Qt-managed widgets.  It can also emit a short inactive-painter
    sequence after a platform or icon helper is asked to paint a null
    ``QPixmap``.  These messages are emitted by Qt itself, not by SAXSShell's
    data workflows, and can flood terminal launches.  Keep other Qt messages
    visible and allow these diagnostics to be re-enabled with
    ``SAXSHELL_SHOW_QPAINTER_STATE_WARNINGS=1`` for painter debugging.
    """

    global _PREVIOUS_QT_MESSAGE_HANDLER
    global _QT_MESSAGE_FILTER_INSTALLED

    if _QT_MESSAGE_FILTER_INSTALLED:
        return
    _PREVIOUS_QT_MESSAGE_HANDLER = qInstallMessageHandler(
        _saxshell_qt_message_handler
    )
    _QT_MESSAGE_FILTER_INSTALLED = True


@lru_cache(maxsize=1)
def _build_saxshell_palette() -> QPalette:
    colors = SAXSHELL_THEME_COLORS
    palette = QPalette()
    field_text = QColor(colors["field_text"])
    disabled_text = QColor(colors["disabled"])
    placeholder_text = QColor(colors["placeholder"])
    palette.setColor(
        QPalette.ColorRole.Window,
        QColor(colors["window"]),
    )
    palette.setColor(
        QPalette.ColorRole.WindowText,
        QColor(colors["text"]),
    )
    palette.setColor(
        QPalette.ColorRole.Base,
        QColor(colors["field_surface"]),
    )
    palette.setColor(
        QPalette.ColorRole.AlternateBase,
        QColor(colors["surface_alt"]),
    )
    palette.setColor(
        QPalette.ColorRole.ToolTipBase,
        QColor(colors["surface"]),
    )
    palette.setColor(
        QPalette.ColorRole.ToolTipText,
        QColor(colors["text"]),
    )
    palette.setColor(QPalette.ColorRole.Text, field_text)
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["surface"]))
    palette.setColor(
        QPalette.ColorRole.ButtonText,
        QColor(colors["text"]),
    )
    palette.setColor(
        QPalette.ColorRole.Highlight,
        QColor(colors["selection"]),
    )
    palette.setColor(
        QPalette.ColorRole.HighlightedText,
        QColor(colors["selection_text"]),
    )
    palette.setColor(QPalette.ColorRole.Link, QColor(colors["primary"]))
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
    ):
        palette.setColor(group, QPalette.ColorRole.Text, field_text)
        palette.setColor(
            group,
            QPalette.ColorRole.PlaceholderText,
            placeholder_text,
        )
        palette.setColor(
            group,
            QPalette.ColorRole.Highlight,
            QColor(colors["selection"]),
        )
        palette.setColor(
            group,
            QPalette.ColorRole.HighlightedText,
            QColor(colors["selection_text"]),
        )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Base,
        QColor(colors["surface_muted"]),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Button,
        QColor(colors["surface_muted"]),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        disabled_text,
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        disabled_text,
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        disabled_text,
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.PlaceholderText,
        disabled_text,
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Highlight,
        QColor(colors["primary_soft"]),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.HighlightedText,
        QColor(colors["primary_pressed"]),
    )
    return palette


def _resolve_saxshell_application_font(app: QApplication) -> QFont | None:
    current_font = QFont(app.font())
    current_family = current_font.family()
    families = set(QFontDatabase.families())
    if current_family in families:
        return None

    for family in (
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans",
    ):
        if family in families:
            resolved_font = QFont(current_font)
            resolved_font.setFamily(family)
            return resolved_font
    if families:
        resolved_font = QFont(current_font)
        resolved_font.setFamily(sorted(families)[0])
        return resolved_font
    return None


@lru_cache(maxsize=1)
def build_saxshell_stylesheet() -> str:
    colors = SAXSHELL_THEME_COLORS
    return f"""
    QWidget {{
        color: {colors["text"]};
        selection-background-color: {colors["selection"]};
        selection-color: {colors["selection_text"]};
    }}
    QMainWindow, QDialog {{
        background: {colors["window"]};
    }}
    QToolTip {{
        background: {colors["surface"]};
        border: 1px solid {colors["border_strong"]};
        border-radius: 4px;
        color: {colors["text"]};
        padding: 5px 7px;
    }}
    QMenuBar {{
        background: {colors["surface"]};
        border-bottom: 1px solid {colors["border"]};
        spacing: 2px;
    }}
    QMenuBar::item {{
        background: transparent;
        border-radius: 4px;
        padding: 5px 9px;
    }}
    QMenuBar::item:selected {{
        background: {colors["surface_muted"]};
        color: {colors["primary"]};
    }}
    QMenu {{
        background: {colors["surface"]};
        border: 1px solid {colors["border_strong"]};
        border-radius: 6px;
        padding: 5px;
    }}
    QMenu::item {{
        border-radius: 4px;
        padding: 5px 24px 5px 20px;
    }}
    QMenu::item:selected {{
        background: {colors["primary_soft"]};
        color: {colors["primary_pressed"]};
    }}
    QStatusBar {{
        background: {colors["surface"]};
        border-top: 1px solid {colors["border"]};
        color: {colors["muted"]};
    }}
    QToolBar {{
        background: {colors["surface"]};
        border: 0;
        spacing: 5px;
        padding: 5px;
    }}
    QTabWidget::pane {{
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-radius: 8px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: {colors["surface_alt"]};
        border: 1px solid {colors["border"]};
        border-bottom-color: {colors["border_strong"]};
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        color: {colors["muted"]};
        font-weight: 600;
        margin-right: 3px;
        min-height: 24px;
        padding: 7px 14px;
    }}
    QTabBar::tab:selected {{
        background: {colors["surface"]};
        border-bottom-color: {colors["surface"]};
        color: {colors["primary_pressed"]};
    }}
    QTabBar::tab:hover:!selected {{
        background: {colors["surface_muted"]};
        color: {colors["primary"]};
    }}
    QGroupBox {{
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-radius: 8px;
        color: {colors["text"]};
        font-weight: 650;
        margin-top: 18px;
        padding: 12px 10px 10px 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 5px;
        color: {colors["primary_pressed"]};
    }}
    QFrame[frameShape="4"], QFrame[frameShape="5"], QFrame[frameShape="6"] {{
        color: {colors["border"]};
    }}
    QScrollArea, QAbstractScrollArea {{
        background: transparent;
        border: 1px solid {colors["border"]};
        border-radius: 7px;
    }}
    QScrollArea > QWidget > QWidget {{
        background: {colors["surface"]};
    }}
    QLabel {{
        background: transparent;
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit,
    QDateTimeEdit, QComboBox {{
        background: {colors["field_surface"]};
        border: 1px solid {colors["border_strong"]};
        border-radius: 6px;
        color: {colors["field_text"]};
        font-size: 13px;
        font-weight: 500;
        min-height: 24px;
        padding: 4px 7px;
        selection-background-color: {colors["selection"]};
        selection-color: {colors["selection_text"]};
    }}
    QTextEdit, QPlainTextEdit {{
        background: {colors["editor_surface"]};
        border: 1px solid {colors["border_strong"]};
        border-radius: 6px;
        color: {colors["editor_text"]};
        font-family: "Menlo", "Consolas", "DejaVu Sans Mono",
            "Courier New", monospace;
        font-size: 13px;
        font-weight: 500;
        padding: 6px 8px;
        selection-background-color: {colors["selection"]};
        selection-color: {colors["selection_text"]};
    }}
    QLineEdit[readOnly="true"] {{
        background: {colors["field_surface_readonly"]};
        color: {colors["field_text"]};
    }}
    QTextEdit[readOnly="true"], QPlainTextEdit[readOnly="true"] {{
        background: {colors["editor_surface"]};
        color: {colors["editor_text"]};
    }}
    QLineEdit::placeholder, QTextEdit::placeholder,
    QPlainTextEdit::placeholder {{
        color: {colors["placeholder"]};
    }}
    QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover,
    QSpinBox:hover, QDoubleSpinBox:hover, QDateEdit:hover, QTimeEdit:hover,
    QDateTimeEdit:hover, QComboBox:hover {{
        border-color: {colors["focus"]};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
    QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QTimeEdit:focus,
    QDateTimeEdit:focus, QComboBox:focus {{
        border: 1px solid {colors["focus"]};
        background: {colors["field_surface"]};
    }}
    QTextEdit:focus, QPlainTextEdit:focus {{
        background: {colors["editor_surface"]};
    }}
    QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
    QSpinBox:disabled, QDoubleSpinBox:disabled, QDateEdit:disabled,
    QTimeEdit:disabled, QDateTimeEdit:disabled, QComboBox:disabled {{
        background: {colors["surface_muted"]};
        color: {colors["disabled"]};
    }}
    QComboBox::drop-down {{
        border: 0;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background: {colors["surface"]};
        border: 1px solid {colors["border_strong"]};
        color: {colors["table_text"]};
        selection-background-color: {colors["primary_soft"]};
        selection-color: {colors["primary_pressed"]};
    }}
    QPushButton, QToolButton {{
        background: {colors["surface_alt"]};
        border: 1px solid {colors["border_strong"]};
        border-radius: 6px;
        color: {colors["text"]};
        min-height: 24px;
        padding: 5px 10px;
    }}
    QPushButton:hover, QToolButton:hover {{
        background: {colors["primary_soft"]};
        border-color: {colors["focus"]};
        color: {colors["primary_pressed"]};
    }}
    QPushButton:pressed, QToolButton:pressed {{
        background: {colors["surface_muted"]};
        border-color: {colors["primary"]};
    }}
    QPushButton:default, QPushButton#PrimaryActionButton {{
        background: {colors["primary"]};
        border-color: {colors["primary_pressed"]};
        color: {colors["selection_text"]};
        font-weight: 700;
    }}
    QPushButton:default:hover, QPushButton#PrimaryActionButton:hover {{
        background: {colors["primary_hover"]};
        color: {colors["selection_text"]};
    }}
    QPushButton:checked, QToolButton:checked {{
        background: {colors["primary_soft"]};
        border-color: {colors["primary"]};
        color: {colors["primary_pressed"]};
    }}
    QPushButton:disabled, QToolButton:disabled {{
        background: {colors["surface_muted"]};
        border-color: {colors["border"]};
        color: {colors["disabled"]};
    }}
    QCheckBox, QRadioButton {{
        spacing: 7px;
    }}
    QTableView, QTableWidget, QListView, QListWidget, QTreeView, QTreeWidget {{
        background: {colors["surface"]};
        alternate-background-color: {colors["surface_alt"]};
        border: 1px solid {colors["border"]};
        border-radius: 7px;
        color: {colors["table_text"]};
        gridline-color: {colors["border"]};
        selection-background-color: {colors["primary_soft"]};
        selection-color: {colors["primary_pressed"]};
    }}
    QTableView::item, QTableWidget::item, QListView::item, QListWidget::item,
    QTreeView::item, QTreeWidget::item {{
        color: {colors["table_text"]};
        padding: 4px;
    }}
    QTableView::item:selected, QTableWidget::item:selected,
    QListView::item:selected, QListWidget::item:selected,
    QTreeView::item:selected, QTreeWidget::item:selected {{
        background: {colors["primary_soft"]};
        color: {colors["primary_pressed"]};
    }}
    QTableView::item:selected:!active, QTableWidget::item:selected:!active,
    QListView::item:selected:!active, QListWidget::item:selected:!active,
    QTreeView::item:selected:!active, QTreeWidget::item:selected:!active {{
        background: {colors["primary_soft"]};
        color: {colors["primary_pressed"]};
    }}
    QTableView::item:disabled, QTableWidget::item:disabled,
    QListView::item:disabled, QListWidget::item:disabled,
    QTreeView::item:disabled, QTreeWidget::item:disabled {{
        color: {colors["disabled"]};
    }}
    QTableView:disabled, QTableWidget:disabled,
    QListView:disabled, QListWidget:disabled,
    QTreeView:disabled, QTreeWidget:disabled {{
        background: {colors["surface_muted"]};
        color: {colors["disabled"]};
        selection-background-color: {colors["primary_soft"]};
        selection-color: {colors["primary_pressed"]};
    }}
    QHeaderView::section {{
        background: {colors["surface_muted"]};
        border: 0;
        border-right: 1px solid {colors["border"]};
        border-bottom: 1px solid {colors["border"]};
        color: {colors["primary_pressed"]};
        font-weight: 700;
        padding: 6px 7px;
    }}
    QProgressBar {{
        background: {colors["surface_muted"]};
        border: 1px solid {colors["border"]};
        border-radius: 6px;
        color: {colors["text"]};
        min-height: 16px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {colors["primary_hover"]};
        border-radius: 5px;
    }}
    QSplitter::handle {{
        background: {colors["window"]};
        border: 1px solid {colors["border"]};
        border-radius: 4px;
        margin: 2px;
    }}
    QSplitter::handle:hover {{
        background: {colors["primary_soft"]};
        border-color: {colors["focus"]};
    }}
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: {colors["surface_alt"]};
        border: 0;
        margin: 0;
    }}
    QScrollBar:vertical {{
        width: 12px;
    }}
    QScrollBar:horizontal {{
        height: 12px;
    }}
    QScrollBar::handle {{
        background: {colors["border_strong"]};
        border-radius: 5px;
        min-height: 28px;
        min-width: 28px;
        margin: 2px;
    }}
    QScrollBar::handle:hover {{
        background: {colors["muted"]};
    }}
    QScrollBar::add-line, QScrollBar::sub-line,
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: transparent;
        border: 0;
        height: 0;
        width: 0;
    }}
    QWidget#BondAnalysisCentral {{
        background: {colors["window"]};
    }}
    QFrame#BondAnalysisHeader {{
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-radius: 8px;
    }}
    QLabel#BondAnalysisTitle {{
        color: {colors["primary_pressed"]};
        font-size: 20px;
        font-weight: 750;
    }}
    QLabel#BondAnalysisSubtitle {{
        color: {colors["muted"]};
        font-size: 12px;
    }}
    QLabel#BondAnalysisContextPill {{
        background: {colors["accent_soft"]};
        border: 1px solid {colors["accent"]};
        border-radius: 10px;
        color: {colors["accent"]};
        font-weight: 700;
        padding: 4px 10px;
    }}
    QToolButton#CollapsibleSectionHeader {{
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        color: {colors["primary_pressed"]};
        font-weight: 700;
        padding: 8px;
        text-align: left;
    }}
    QFrame#CollapsibleSectionContent {{
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-top: 0;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
    }}
    QLabel#MutedPanelLabel {{
        background: {colors["surface_alt"]};
        border: 1px solid {colors["border"]};
        border-radius: 6px;
        color: {colors["muted"]};
        padding: 8px;
    }}
    """


def apply_saxshell_theme(app: QApplication) -> None:
    stylesheet = build_saxshell_stylesheet()
    if (
        app.property(_SAXSHELL_THEME_APPLIED_PROPERTY)
        == _SAXSHELL_THEME_VERSION
        and app.styleSheet() == stylesheet
    ):
        return

    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    resolved_font = _resolve_saxshell_application_font(app)
    if resolved_font is not None:
        app.setFont(resolved_font)
    app.setPalette(_build_saxshell_palette())
    app.setStyleSheet(stylesheet)
    app.setProperty(
        _SAXSHELL_THEME_APPLIED_PROPERTY,
        _SAXSHELL_THEME_VERSION,
    )


def _configure_macos_application_identity() -> None:
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle, NSProcessInfo
    except Exception:
        return

    NSProcessInfo.processInfo().setProcessName_(SAXSHELL_APPLICATION_NAME)
    info = NSBundle.mainBundle().infoDictionary()
    if info is not None:
        info["CFBundleName"] = SAXSHELL_APPLICATION_NAME
        info["CFBundleDisplayName"] = SAXSHELL_APPLICATION_NAME


def prepare_saxshell_application_identity() -> None:
    QCoreApplication.setApplicationName(SAXSHELL_APPLICATION_NAME)
    QApplication.setApplicationDisplayName(SAXSHELL_APPLICATION_NAME)
    QApplication.setDesktopFileName(SAXSHELL_APPLICATION_NAME)
    _configure_macos_application_identity()


def configure_saxshell_application(app: QApplication) -> None:
    prepare_saxshell_application_identity()
    install_saxshell_qt_message_filter()
    app.setApplicationName(SAXSHELL_APPLICATION_NAME)
    app.setApplicationDisplayName(SAXSHELL_APPLICATION_NAME)
    app.setDesktopFileName(SAXSHELL_APPLICATION_NAME)
    app.setWindowIcon(load_saxshell_icon())
    apply_saxshell_theme(app)


def create_saxshell_startup_splash() -> QSplashScreen:
    pixmap = QPixmap(420, 210)
    if pixmap.isNull():
        return QSplashScreen()

    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SAXSHELL_THEME_COLORS["surface"]))
        painter.drawRoundedRect(10, 10, 400, 190, 18, 18)

        border_pen = QPen(QColor(SAXSHELL_THEME_COLORS["border"]))
        border_pen.setWidth(2)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(10, 10, 400, 190, 18, 18)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SAXSHELL_THEME_COLORS["primary_soft"]))
        painter.drawRoundedRect(24, 38, 94, 94, 16, 16)
        icon_pixmap = load_saxshell_icon().pixmap(92, 92)
        painter.drawPixmap(25, 39, icon_pixmap)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(19)
        painter.setFont(title_font)
        painter.setPen(QColor(BRAND_PRIMARY_HEX))
        painter.drawText(136, 84, "SAXSShell")

        subtitle_font = QFont()
        subtitle_font.setPointSize(10)
        painter.setFont(subtitle_font)
        painter.setPen(QColor(BRAND_SECONDARY_HEX))
        painter.drawText(138, 113, "Loading SAXS workflow...")
        painter.drawText(138, 136, "Initializing interface and project state")

        accent_pen = QPen(QColor(BRAND_ACCENT_HEX))
        accent_pen.setWidth(4)
        painter.setPen(accent_pen)
        painter.drawLine(138, 154, 286, 154)
    finally:
        painter.end()

    splash = QSplashScreen(
        pixmap,
        Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint,
    )
    splash.setWindowIcon(load_saxshell_icon())
    splash.showMessage(
        "Starting SAXSShell",
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        QColor(BRAND_PRIMARY_HEX),
    )
    return splash
