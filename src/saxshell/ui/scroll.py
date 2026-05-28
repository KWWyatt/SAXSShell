from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QWidget


def create_scroll_area(
    widget: QWidget,
    *,
    horizontal: Qt.ScrollBarPolicy = Qt.ScrollBarPolicy.ScrollBarAsNeeded,
    vertical: Qt.ScrollBarPolicy = Qt.ScrollBarPolicy.ScrollBarAsNeeded,
    frame_shape: QFrame.Shape = QFrame.Shape.NoFrame,
    widget_resizable: bool = True,
) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(widget_resizable)
    area.setFrameShape(frame_shape)
    area.setHorizontalScrollBarPolicy(horizontal)
    area.setVerticalScrollBarPolicy(vertical)
    area.setWidget(widget)
    return area
