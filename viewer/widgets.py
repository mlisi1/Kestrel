"""RenderWidget: QLabel subclass that displays rendered frames and draws overlays."""

from __future__ import annotations

import collections
import math

import numpy as np
from PyQt5.QtCore  import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui   import (
    QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap,
)
from PyQt5.QtWidgets import QLabel, QSizePolicy

from viewer.config import fmt_splats


class RenderWidget(QLabel):
    """
    Displays the rendered Kestrel frame and handles all mouse/keyboard events
    for camera navigation.

    Signals
    -------
    camera_changed  — emitted when the camera pose has been modified
    key_down(int)   — key press (non-repeat)
    key_up(int)     — key release (non-repeat); -1 is the sentinel "clear all"
    """
    camera_changed = pyqtSignal()
    key_down       = pyqtSignal(int)
    key_up         = pyqtSignal(int)

    def __init__(self, camera):
        super().__init__()
        self.cam               = camera
        self.mouse_inv_x       = True    # default: inverted (drag-scene feel)
        self.mouse_inv_y       = True
        self._last             = None
        self._btns             = set()
        self._fps_history      = collections.deque(maxlen=90)
        self._splat_history    = collections.deque(maxlen=90)
        self._show_fps_overlay   = True
        self._show_splat_overlay = True
        self._no_culling_warning = False
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #1a1a1a;")
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame(self, img_np: np.ndarray):
        H, W = img_np.shape[:2]
        qimg = QImage(img_np.tobytes(), W, H, W * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    # ── mouse / keyboard event forwarding ─────────────────────────────────────

    def mousePressEvent(self, e):
        self._last = e.pos()
        self._btns.add(e.button())
        self.setFocus()

    def mouseReleaseEvent(self, e):
        self._btns.discard(e.button())

    def mouseMoveEvent(self, e):
        if self._last is None:
            return
        dx = e.x() - self._last.x()
        dy = e.y() - self._last.y()
        self._last = e.pos()
        if Qt.LeftButton in self._btns:
            sx = -1.0 if self.mouse_inv_x else 1.0
            sy = -1.0 if self.mouse_inv_y else 1.0
            self.cam.orbit(sx * dx * 0.005, sy * dy * 0.005)
            self.camera_changed.emit()
        elif Qt.RightButton in self._btns:
            scale = self.cam.distance * 0.001
            self.cam.pan(-dx * scale, dy * scale)
            self.camera_changed.emit()

    def wheelEvent(self, e):
        self.cam.zoom(e.angleDelta().y() / 120.0)
        self.camera_changed.emit()

    def keyPressEvent(self, e):
        if not e.isAutoRepeat():
            self.key_down.emit(e.key())

    def keyReleaseEvent(self, e):
        if not e.isAutoRepeat():
            self.key_up.emit(e.key())

    def focusOutEvent(self, e):
        self.key_up.emit(-1)   # sentinel: clear all held keys
        super().focusOutEvent(e)

    # ── overlay data ingestion ─────────────────────────────────────────────────

    def update_fps(self, fps: float):
        self._fps_history.append(fps)
        self.update()

    def update_splat_count(self, n: int):
        self._splat_history.append(n)
        self.update()

    # ── paint overlays ─────────────────────────────────────────────────────────

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._show_fps_overlay and self._fps_history:
            p = QPainter(self)
            self._draw_fps_overlay(p)
            p.end()
        if self._show_splat_overlay and self._splat_history:
            p = QPainter(self)
            h_fps = 90 + 4 if (self._show_fps_overlay and self._fps_history) else 0
            self._draw_splat_overlay(p, y_offset=h_fps)
            p.end()
        if self._no_culling_warning:
            p = QPainter(self)
            self._draw_culling_warning(p)
            p.end()

    def _draw_fps_overlay(self, p: QPainter):
        W_OV, H_OV = 150, 90
        MARGIN      = 8
        PAD_L       = 30    # width of Y-axis label column
        PAD_T       = 18    # height of fps-text row at top
        PAD_R       = 4
        PAD_B       = 4

        ox = self.width() - W_OV - MARGIN
        oy = MARGIN
        if ox < 0:
            return

        p.setRenderHint(QPainter.Antialiasing)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRoundedRect(ox, oy, W_OV, H_OV, 6, 6)

        vals = list(self._fps_history)
        current_fps = vals[-1] if vals else 0.0

        peak = max(vals) if vals else 30.0
        y_max = float(next((n for n in (30, 60, 90, 120, 144, 240)
                            if n >= peak * 0.8), 240))

        gx = float(ox + PAD_L)
        gy = float(oy + PAD_T)
        gw = float(W_OV - PAD_L - PAD_R)
        gh = float(H_OV - PAD_T - PAD_B)

        font = QFont(); font.setPointSize(7)
        p.setFont(font)
        for y_val in (0.0, y_max / 2, y_max):
            py = gy + gh - (y_val / y_max) * gh
            p.setPen(QPen(QColor(255, 255, 255, 45), 0.8))
            p.drawLine(QPointF(gx, py), QPointF(gx + gw, py))
            p.setPen(QColor(200, 200, 200, 200))
            p.drawText(QRectF(ox, py - 6.0, float(PAD_L - 3), 12.0),
                       Qt.AlignRight | Qt.AlignVCenter, str(int(y_val)))

        if len(vals) >= 2:
            path = QPainterPath()
            n = len(vals)
            for i, v in enumerate(vals):
                x = gx + (i / (n - 1)) * gw
                y = gy + gh - min(v / y_max, 1.0) * gh
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            col = (QColor(80, 220, 80)  if current_fps >= 25 else
                   QColor(255, 200, 50) if current_fps >= 12 else
                   QColor(255, 80,  80))
            p.setPen(QPen(col, 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        font2 = QFont(); font2.setPointSize(8); font2.setBold(True)
        p.setFont(font2)
        p.setPen(QColor(255, 255, 255, 230))
        p.drawText(QRectF(float(ox), float(oy) + 1.0, float(W_OV), float(PAD_T) - 2.0),
                   Qt.AlignCenter, f"{current_fps:.1f} fps")

    def _draw_splat_overlay(self, p: QPainter, y_offset: int = 0):
        W_OV, H_OV = 150, 90
        MARGIN      = 8
        PAD_L       = 36    # wider — splat labels like "2.5M" need space
        PAD_T       = 18
        PAD_R       = 4
        PAD_B       = 4

        ox = self.width() - W_OV - MARGIN
        oy = MARGIN + y_offset
        if ox < 0:
            return

        p.setRenderHint(QPainter.Antialiasing)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRoundedRect(ox, oy, W_OV, H_OV, 6, 6)

        vals = list(self._splat_history)
        current_n = vals[-1] if vals else 0

        peak = max(vals) if vals else 1
        magnitude = 10 ** math.floor(math.log10(max(peak, 1)))
        y_max = float(math.ceil(peak / magnitude) * magnitude)
        if y_max == 0:
            y_max = 1.0

        gx = float(ox + PAD_L)
        gy = float(oy + PAD_T)
        gw = float(W_OV - PAD_L - PAD_R)
        gh = float(H_OV - PAD_T - PAD_B)

        font = QFont(); font.setPointSize(7)
        p.setFont(font)
        for y_val in (0.0, y_max / 2, y_max):
            py = gy + gh - (y_val / y_max) * gh
            p.setPen(QPen(QColor(255, 255, 255, 45), 0.8))
            p.drawLine(QPointF(gx, py), QPointF(gx + gw, py))
            p.setPen(QColor(200, 200, 200, 200))
            p.drawText(QRectF(float(ox), py - 6.0, float(PAD_L - 3), 12.0),
                       Qt.AlignRight | Qt.AlignVCenter, fmt_splats(int(y_val)))

        if len(vals) >= 2:
            path = QPainterPath()
            n = len(vals)
            for i, v in enumerate(vals):
                x = gx + (i / (n - 1)) * gw
                y = gy + gh - min(v / y_max, 1.0) * gh
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.setPen(QPen(QColor(100, 180, 255), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        font2 = QFont(); font2.setPointSize(8); font2.setBold(True)
        p.setFont(font2)
        p.setPen(QColor(255, 255, 255, 230))
        p.drawText(QRectF(float(ox), float(oy) + 1.0, float(W_OV), float(PAD_T) - 2.0),
                   Qt.AlignCenter, f"{fmt_splats(current_n)} splats")

    def _draw_culling_warning(self, p: QPainter):
        MARGIN = 8
        PAD_X, PAD_Y = 8, 4
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)
        text = "No culling index  —  frustum culling disabled"
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        bw = tw + PAD_X * 2
        bh = th + PAD_Y * 2
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(180, 100, 0, 210))
        p.drawRoundedRect(MARGIN, MARGIN, bw, bh, 4, 4)
        p.setPen(QColor(255, 255, 255, 230))
        p.drawText(QRectF(float(MARGIN + PAD_X), float(MARGIN),
                          float(tw), float(bh)),
                   Qt.AlignVCenter, text)
