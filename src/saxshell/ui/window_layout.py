from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QApplication, QWidget

DEFAULT_WINDOW_PRESET_KEY = "laptop_14"
SCREEN_MARGIN_WIDTH = 48
SCREEN_MARGIN_HEIGHT = 64
MIN_WINDOW_WIDTH = 640
MIN_WINDOW_HEIGHT = 520


@dataclass(frozen=True, slots=True)
class WindowLayoutPreset:
    key: str
    label: str
    width: int
    height: int
    ui_scale: float


WINDOW_LAYOUT_PRESETS: tuple[WindowLayoutPreset, ...] = (
    WindowLayoutPreset(
        key="laptop_13",
        label="13-inch Laptop (Compact)",
        width=1180,
        height=760,
        ui_scale=0.95,
    ),
    WindowLayoutPreset(
        key="laptop_14",
        label="14-inch Laptop / MacBook Pro",
        width=1280,
        height=820,
        ui_scale=1.0,
    ),
    WindowLayoutPreset(
        key="laptop_16",
        label="15-inch / 16-inch Laptop",
        width=1440,
        height=900,
        ui_scale=1.05,
    ),
    WindowLayoutPreset(
        key="display_1080p",
        label="External Display (1080p)",
        width=1500,
        height=880,
        ui_scale=1.0,
    ),
    WindowLayoutPreset(
        key="display_1440p",
        label="External Display (1440p / QHD)",
        width=1680,
        height=980,
        ui_scale=1.1,
    ),
)
WINDOW_LAYOUT_PRESET_MAP = {
    preset.key: preset for preset in WINDOW_LAYOUT_PRESETS
}


def primary_screen() -> QScreen | None:
    app = QApplication.instance()
    if app is None:
        return None
    return app.primaryScreen()


def available_geometry(screen: QScreen | None = None) -> QRect | None:
    resolved = screen if screen is not None else primary_screen()
    if resolved is None:
        return None
    return resolved.availableGeometry()


def fit_window_size_to_screen(
    size: QSize,
    *,
    screen: QScreen | None = None,
) -> QSize:
    available = available_geometry(screen)
    if available is None:
        return size
    usable_width = max(MIN_WINDOW_WIDTH, available.width() - SCREEN_MARGIN_WIDTH)
    usable_height = max(
        MIN_WINDOW_HEIGHT,
        available.height() - SCREEN_MARGIN_HEIGHT,
    )
    return QSize(
        min(size.width(), usable_width),
        min(size.height(), usable_height),
    )


def preset_size(
    preset_key: str,
    *,
    screen: QScreen | None = None,
) -> QSize:
    preset = WINDOW_LAYOUT_PRESET_MAP.get(str(preset_key).strip())
    if preset is None:
        raise ValueError(f"Unknown window preset: {preset_key}")
    return fit_window_size_to_screen(
        QSize(preset.width, preset.height),
        screen=screen,
    )


def recommended_window_layout_preset(
    *,
    screen: QScreen | None = None,
) -> WindowLayoutPreset:
    available = available_geometry(screen)
    if available is None:
        return WINDOW_LAYOUT_PRESET_MAP[DEFAULT_WINDOW_PRESET_KEY]
    if available.width() <= 1366 or available.height() <= 820:
        return WINDOW_LAYOUT_PRESET_MAP["laptop_13"]
    if available.width() <= 1600 or available.height() <= 940:
        return WINDOW_LAYOUT_PRESET_MAP["laptop_14"]
    if available.width() <= 1920 or available.height() <= 1100:
        return WINDOW_LAYOUT_PRESET_MAP["display_1080p"]
    return WINDOW_LAYOUT_PRESET_MAP["display_1440p"]


def default_main_window_size(*, screen: QScreen | None = None) -> QSize:
    return preset_size(DEFAULT_WINDOW_PRESET_KEY, screen=screen)


def tool_window_size(
    *,
    preferred_width: int = 1160,
    preferred_height: int = 820,
    min_width: int = 760,
    min_height: int = 560,
    screen: QScreen | None = None,
) -> QSize:
    """Size helper for auxiliary tool windows/dialogs."""
    available = available_geometry(screen)
    if available is None:
        return QSize(preferred_width, preferred_height)
    width = min(
        preferred_width,
        max(min_width, available.width() - 180),
    )
    height = min(
        preferred_height,
        max(min_height, available.height() - 160),
    )
    return fit_window_size_to_screen(QSize(width, height), screen=screen)


def apply_window_size(
    window: QWidget,
    width: int,
    height: int,
    *,
    screen: QScreen | None = None,
) -> QSize:
    target = fit_window_size_to_screen(QSize(width, height), screen=screen)
    window.resize(target)
    return target


def apply_preset_window_size(
    window: QWidget,
    preset_key: str,
    *,
    screen: QScreen | None = None,
) -> QSize:
    target = preset_size(preset_key, screen=screen)
    window.resize(target)
    return target


def apply_recommended_window_size(
    window: QWidget,
    *,
    screen: QScreen | None = None,
) -> QSize:
    preset = recommended_window_layout_preset(screen=screen)
    return apply_preset_window_size(window, preset.key, screen=screen)
