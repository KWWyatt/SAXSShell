"""Shared Qt widgets and display helpers for SAXSShell applications."""

from .scroll import create_scroll_area
from .window_layout import (
    DEFAULT_WINDOW_PRESET_KEY,
    WINDOW_LAYOUT_PRESET_MAP,
    WINDOW_LAYOUT_PRESETS,
    WindowLayoutPreset,
    apply_preset_window_size,
    apply_recommended_window_size,
    apply_window_size,
    default_main_window_size,
    fit_window_size_to_screen,
    preset_size,
    recommended_window_layout_preset,
    tool_window_size,
)
from .wheel_guard import (
    ValueWheelGuard,
    install_global_value_wheel_guard,
    should_suppress_value_wheel_event,
)
from .periodic_table import (
    PERIODIC_TABLE_ELEMENTS,
    PeriodicElement,
    PeriodicTableElementDialog,
    PeriodicTableWidget,
    element_by_symbol,
)

__all__ = [
    "PERIODIC_TABLE_ELEMENTS",
    "PeriodicElement",
    "PeriodicTableElementDialog",
    "PeriodicTableWidget",
    "element_by_symbol",
]
