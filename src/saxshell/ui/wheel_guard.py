from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QWidget,
)

_GLOBAL_VALUE_WHEEL_GUARD: ValueWheelGuard | None = None


class ValueWheelGuard(QObject):
    """Prevent mouse-wheel events from changing spin boxes and combo boxes."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if should_suppress_value_wheel_event(watched, event):
            event.ignore()
            return True
        return super().eventFilter(watched, event)


def should_suppress_value_wheel_event(
    watched: QObject,
    event: QEvent,
) -> bool:
    if event.type() != QEvent.Type.Wheel or not isinstance(watched, QWidget):
        return False
    control = wheel_guarded_value_control(watched)
    return control is not None and control.isEnabled()


def wheel_guarded_value_control(widget: QWidget) -> QWidget | None:
    current: QWidget | None = widget
    while current is not None:
        if isinstance(current, QAbstractSpinBox):
            return current
        if isinstance(current, QComboBox):
            view = current.view()
            if view is not None and view.isVisible():
                return None
            return current
        current = current.parentWidget()
    return None


def install_global_value_wheel_guard(
    app: QApplication | None = None,
) -> ValueWheelGuard:
    global _GLOBAL_VALUE_WHEEL_GUARD
    resolved_app = app if app is not None else QApplication.instance()
    if resolved_app is None:
        raise RuntimeError("No QApplication instance is available")
    if _GLOBAL_VALUE_WHEEL_GUARD is None:
        _GLOBAL_VALUE_WHEEL_GUARD = ValueWheelGuard(resolved_app)
        resolved_app.installEventFilter(_GLOBAL_VALUE_WHEEL_GUARD)
    return _GLOBAL_VALUE_WHEEL_GUARD


def global_value_wheel_guard() -> ValueWheelGuard | None:
    return _GLOBAL_VALUE_WHEEL_GUARD
