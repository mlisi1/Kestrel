"""Qt UI helper factories: slider+spinbox pair and combo box."""

from __future__ import annotations

from PyQt5.QtCore  import Qt, QLocale
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QSlider, QSpinBox, QWidget,
)


class NoScrollCombo(QComboBox):
    """QComboBox that ignores scroll-wheel events to prevent accidental value changes."""
    def wheelEvent(self, e):
        e.ignore()


def slider_spin(mn: float, mx: float, val: float, on_change,
                is_float: bool = False,
                decimals: int = 2,
                step: float | None = None) -> QWidget:
    """Horizontal slider paired with a spinbox, kept bidirectionally in sync."""
    row = QWidget()
    hl  = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
    updating = False

    if is_float:
        if step is None:
            step = (mx - mn) / 100.0
        n_steps = max(1, round((mx - mn) / step))
        init_i  = min(n_steps, max(0, round((val - mn) / step)))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, n_steps)
        slider.setValue(init_i)

        spin = QDoubleSpinBox()
        spin.setLocale(QLocale(QLocale.C))
        spin.setRange(mn, mx); spin.setDecimals(decimals)
        spin.setSingleStep(step); spin.setFixedWidth(72)
        spin.setValue(mn + init_i * step)

        def on_s(i):
            nonlocal updating
            if updating: return
            v = mn + i * step
            updating = True; spin.setValue(v); updating = False
            on_change(v)

        def on_b(v):
            nonlocal updating
            if updating: return
            i = min(n_steps, max(0, round((v - mn) / step)))
            updating = True; slider.setValue(i); updating = False
            on_change(v)

        slider.valueChanged.connect(on_s)
        spin.valueChanged.connect(on_b)
    else:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(mn), int(mx))
        slider.setValue(int(val))

        spin = QSpinBox()
        spin.setRange(int(mn), int(mx))
        spin.setFixedWidth(55); spin.setValue(int(val))

        def on_s(i):
            nonlocal updating
            if updating: return
            updating = True; spin.setValue(i); updating = False
            on_change(i)

        def on_b(v):
            nonlocal updating
            if updating: return
            updating = True; slider.setValue(v); updating = False
            on_change(v)

        slider.valueChanged.connect(on_s)
        spin.valueChanged.connect(on_b)

    hl.addWidget(slider)
    hl.addWidget(spin)
    return row


def combo(items, current, on_change) -> NoScrollCombo:
    c = NoScrollCombo()
    c.addItems(items)
    c.setCurrentText(current)
    c.currentTextChanged.connect(on_change)
    return c
