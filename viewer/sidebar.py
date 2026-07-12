"""Sidebar panel: builds all control groups and owns their UI callbacks."""

from __future__ import annotations

import os
import threading

from PyQt5.QtCore    import Qt, QLocale
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

import gsplat2d_rendering as gs2d

from viewer.config     import (
    UP_AXIS_OPTIONS,
    AR_RATIO_OPTIONS, AR_RATIO_VALUES,
    RESOLUTION_PRESETS,
    RENDER_TYPES,
    COMPRESSION_LABELS,
    VERBOSITY_LABELS,
)
from viewer.ui_helpers import slider_spin, combo, NoScrollCombo
from viewer.ply_loader import _compressed_ply_path


class Sidebar:
    """Builds and owns all sidebar control panels for LocalViewer."""

    def __init__(self, viewer):
        self.v = viewer
        self.widget = self._build_controls()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        g = QGroupBox(title)
        f = QFormLayout(); f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        g.setLayout(f)
        return g, f

    def _progress_bar(self) -> QProgressBar:
        pb = QProgressBar()
        pb.setRange(0, 0)
        pb.setTextVisible(False)
        pb.setFixedHeight(8)
        pb.hide()
        return pb

    def _build_controls(self) -> QWidget:
        w   = QWidget()
        vbl = QVBoxLayout(w); vbl.setContentsMargins(4, 4, 4, 4)
        self._build_culling_group(vbl)
        self._build_compression_group(vbl)
        self._build_status_group(vbl)
        self._build_camera_group(vbl)
        self._build_resolution_group(vbl)
        self._build_render_group(vbl)
        self._build_gaussian_group(vbl)
        self._build_crop_group(vbl)
        vbl.addStretch()
        return w

    # ── panel builders ─────────────────────────────────────────────────────────

    def _build_culling_group(self, vbl: QVBoxLayout):
        g, f = self._group("Frustum Culling")
        self._culling_status_label = QLabel()
        self._build_idx_btn = QPushButton()
        self._build_idx_btn.clicked.connect(self._on_build_index_clicked)
        self._leaf_max_spin = QSpinBox()
        self._leaf_max_spin.setRange(100, 1_000_000)
        self._leaf_max_spin.setSingleStep(1000)
        self._leaf_max_spin.setValue(self.v._leaf_max)
        self._leaf_max_spin.valueChanged.connect(lambda v: setattr(self.v, '_leaf_max', v))
        self._culling_progress = self._progress_bar()

        if self.v._no_culling_flag:
            self._culling_status_label.setText("Disabled (--no-culling)")
            self._culling_status_label.setStyleSheet("color: #888888;")
            f.addRow("Status:", self._culling_status_label)
        elif self.v.renderer.octree is not None:
            n = len(self.v.renderer.octree.node_aabbs)
            self._culling_status_label.setText(f"Active — {n:,} leaves")
            self._culling_status_label.setStyleSheet("color: #66cc66;")
            self._build_idx_btn.setText("Rebuild Index")
            f.addRow("Status:", self._culling_status_label)
            f.addRow("Leaf size:", self._leaf_max_spin)
            f.addRow(self._build_idx_btn)
        else:
            self._culling_status_label.setText("No index — culling disabled")
            self._culling_status_label.setStyleSheet("color: #ff9900;")
            self._build_idx_btn.setText("Build Index")
            f.addRow("Status:", self._culling_status_label)
            f.addRow("Leaf size:", self._leaf_max_spin)
            f.addRow(self._build_idx_btn)
        f.addRow(self._culling_progress)
        vbl.addWidget(g)

    def _build_compression_group(self, vbl: QVBoxLayout):
        g, f = self._group("Compression")
        self._compression_combo = NoScrollCombo()
        self._compression_combo.addItems(COMPRESSION_LABELS)
        self._compression_combo.blockSignals(True)
        self._compression_combo.setCurrentIndex(self.v._current_compression)
        self._compression_combo.blockSignals(False)
        self._compression_combo.currentIndexChanged.connect(self._on_compression_changed)
        self._compression_status_label = QLabel(f"L{self.v._current_compression} active")
        self._compression_status_label.setStyleSheet("color: #66cc66;")
        self._compress_progress = self._progress_bar()
        f.addRow("Level:", self._compression_combo)
        f.addRow("Status:", self._compression_status_label)
        f.addRow(self._compress_progress)
        vbl.addWidget(g)

    def _build_status_group(self, vbl: QVBoxLayout):
        g, f = self._group("Status")
        self._fps_label   = QLabel("--")
        self._gpu_label   = QLabel("--")
        self._splat_label = QLabel("--")
        f.addRow("FPS:",    self._fps_label)
        f.addRow("Splats:", self._splat_label)
        f.addRow("VRAM:",   self._gpu_label)

        overlay_row = QWidget(); ol = QHBoxLayout(overlay_row); ol.setContentsMargins(0, 0, 0, 0)
        fps_cb   = QCheckBox("FPS graph");   fps_cb.setChecked(self.v._show_fps_overlay)
        splat_cb = QCheckBox("Splat graph"); splat_cb.setChecked(self.v._show_splat_overlay)
        fps_cb.toggled.connect(  lambda v: setattr(self.v.render_widget, '_show_fps_overlay',   v))
        splat_cb.toggled.connect(lambda v: setattr(self.v.render_widget, '_show_splat_overlay', v))
        ol.addWidget(fps_cb); ol.addWidget(splat_cb)
        f.addRow("Overlays:", overlay_row)

        profiling_cb = QCheckBox("print GPU stage breakdown to console")
        profiling_cb.setChecked(self.v.renderer.profiling_enabled)
        profiling_cb.setToolTip(
            "Periodically prints a per-stage (SH eval / rasterize / post-process) "
            "GPU timing breakdown to the console — see renderer/renderer.py."
        )
        profiling_cb.toggled.connect(self.v.renderer.set_profiling_enabled)
        f.addRow("Profiling:", profiling_cb)

        def _on_verbosity_changed(text: str):
            gs2d.set_verbosity(int(text.split(" ", 1)[0]))

        verbosity_combo = combo(VERBOSITY_LABELS, VERBOSITY_LABELS[gs2d.get_verbosity()],
                                _on_verbosity_changed)
        verbosity_combo.setToolTip(
            "gsplat2d_rendering console log level — errors always print; "
            "warnings from Normal up; extra diagnostic detail at Verbose."
        )
        f.addRow("Verbosity:", verbosity_combo)
        vbl.addWidget(g)

    def _build_camera_group(self, vbl: QVBoxLayout):
        g, f = self._group("Camera")
        f.addRow("World Up:", combo(UP_AXIS_OPTIONS, self.v._cfg["world_up"],
                                    self.v._on_world_up_changed))
        f.addRow("FOV:", slider_spin(10, 120, self.v.fov_deg,
                                     lambda v: self.v._set("fov_deg", float(v))))
        f.addRow("Move Speed:",
            slider_spin(0.1, 5.0, self.v._move_speed,
                        lambda v: setattr(self.v, '_move_speed', v),
                        is_float=True, decimals=2, step=0.05))
        f.addRow("Orbit Speed:",
            slider_spin(0.1, 5.0, self.v._orbit_speed,
                        lambda v: setattr(self.v, '_orbit_speed', v),
                        is_float=True, decimals=2, step=0.05))

        mouse_row = QWidget(); ml = QHBoxLayout(mouse_row); ml.setContentsMargins(0, 0, 0, 0)
        minv_x = QCheckBox("Inv X"); minv_x.setChecked(self.v._cfg["mouse_inv_x"])
        minv_y = QCheckBox("Inv Y"); minv_y.setChecked(self.v._cfg["mouse_inv_y"])
        minv_x.toggled.connect(lambda v: setattr(self.v.render_widget, 'mouse_inv_x', v))
        minv_y.toggled.connect(lambda v: setattr(self.v.render_widget, 'mouse_inv_y', v))
        ml.addWidget(minv_x); ml.addWidget(minv_y)
        f.addRow("Mouse Inv:", mouse_row)

        kb_row = QWidget(); kl = QHBoxLayout(kb_row); kl.setContentsMargins(0, 0, 0, 0)
        kinv_x = QCheckBox("Inv X"); kinv_x.setChecked(self.v._cfg["kb_inv_x"])
        kinv_y = QCheckBox("Inv Y"); kinv_y.setChecked(self.v._cfg["kb_inv_y"])
        kinv_x.toggled.connect(lambda v: setattr(self.v, '_kb_inv_x', v))
        kinv_y.toggled.connect(lambda v: setattr(self.v, '_kb_inv_y', v))
        kl.addWidget(kinv_x); kl.addWidget(kinv_y)
        f.addRow("KB Rot Inv:", kb_row)

        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self.v._reset_camera)
        f.addRow(btn_reset)
        hint = QLabel("LMB: orbit  |  RMB: pan  |  Scroll: zoom\n"
                      "WASD: translate  |  Arrows: FPS look  |  R: reset")
        hint.setWordWrap(True)
        f.addRow(hint)
        vbl.addWidget(g)

    def _build_resolution_group(self, vbl: QVBoxLayout):
        g, f = self._group("Resolution")
        self._res_width_spin = QSpinBox()
        self._res_width_spin.setRange(2, 7680); self._res_width_spin.setSingleStep(2)

        self._res_height_spin = QSpinBox()
        self._res_height_spin.setRange(2, 4320); self._res_height_spin.setSingleStep(2)

        self._ar_combo = NoScrollCombo()
        self._ar_combo.addItems(AR_RATIO_OPTIONS)

        self._ar_custom_spin = QDoubleSpinBox()
        self._ar_custom_spin.setLocale(QLocale(QLocale.C))
        self._ar_custom_spin.setRange(0.1, 10.0); self._ar_custom_spin.setDecimals(3)
        self._ar_custom_spin.setSingleStep(0.001); self._ar_custom_spin.setFixedWidth(72)
        self._ar_custom_spin.setValue(self.v._aspect_ratio)
        self._ar_custom_spin.setEnabled(False)

        self._lock_ar_cb = QCheckBox("Lock AR")
        self._lock_ar_cb.setChecked(self.v._lock_ar)
        self._lock_ar_cb.toggled.connect(lambda v: setattr(self.v, '_lock_ar', v))

        preset_combo = NoScrollCombo()
        preset_combo.addItems(list(RESOLUTION_PRESETS.keys()))
        preset_combo.currentTextChanged.connect(self._on_preset_changed)
        _preset = next(
            (name for name, wh in RESOLUTION_PRESETS.items()
             if wh is not None and wh == (self.v._render_w, self.v._render_h)),
            "Custom"
        )
        preset_combo.blockSignals(True)
        preset_combo.setCurrentText(_preset)
        preset_combo.blockSignals(False)

        for spin, val in [(self._res_width_spin,  self.v._render_w),
                          (self._res_height_spin, self.v._render_h)]:
            spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)

        self._res_width_spin.valueChanged.connect(self._on_width_changed)
        self._res_height_spin.valueChanged.connect(self._on_height_changed)
        self._ar_combo.currentTextChanged.connect(self._on_ar_combo_changed)
        self._ar_custom_spin.valueChanged.connect(self._on_ar_custom_changed)
        self._ar_combo.setCurrentText(self.v._cfg["ar_preset"])

        wh_row = QWidget(); wl = QHBoxLayout(wh_row); wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(self._res_width_spin)
        wl.addWidget(QLabel("×"))
        wl.addWidget(self._res_height_spin)
        f.addRow("Preset:", preset_combo)
        f.addRow("Size:", wh_row)
        ar_row = QWidget(); al = QHBoxLayout(ar_row); al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(self._ar_combo, 2)
        al.addWidget(self._ar_custom_spin, 1)
        al.addWidget(self._lock_ar_cb, 1)
        f.addRow("Aspect:", ar_row)
        vbl.addWidget(g)

    def _build_render_group(self, vbl: QVBoxLayout):
        g, f = self._group("Render Options")
        f.addRow("Type:", combo(RENDER_TYPES, self.v.render_type,
                                lambda v: self.v._set("render_type", v)))
        f.addRow("Depth Ratio:",
            slider_spin(0.0, 1.0, self.v.depth_ratio,
                        lambda v: self.v._set("depth_ratio", v),
                        is_float=True, decimals=2, step=0.05))
        split_cb = QCheckBox()
        split_cb.setChecked(self.v.split_enabled)
        split_cb.toggled.connect(lambda v: self.v._set("split_enabled", v))
        f.addRow("Split View:", split_cb)
        f.addRow("Split Pos:",
            slider_spin(0.0, 1.0, self.v.split_pos,
                        lambda v: self.v._set("split_pos", v),
                        is_float=True, decimals=2, step=0.01))
        f.addRow("Left:",  combo(RENDER_TYPES, self.v.render_type1,
                                 lambda v: self.v._set("render_type1", v)))
        f.addRow("Right:", combo(RENDER_TYPES, self.v.render_type2,
                                 lambda v: self.v._set("render_type2", v)))
        vbl.addWidget(g)

    def _build_gaussian_group(self, vbl: QVBoxLayout):
        g, f = self._group("Gaussian Model")
        self._sh_spin = QSpinBox()
        self._sh_spin.setRange(0, self.v._ply_sh_degree)
        self._sh_spin.setValue(self.v.sh_degree)
        self._sh_spin.valueChanged.connect(lambda v: self.v._set("sh_degree", v))
        f.addRow("SH Degree:", self._sh_spin)
        f.addRow("Opacity Thr:",
            slider_spin(0.0, 0.5, self.v.opacity_thresh,
                        lambda v: self.v._set("opacity_thresh", v),
                        is_float=True, decimals=3, step=0.005))
        f.addRow("Sparsity:",
            slider_spin(1, 10, self.v.sparsity,
                        lambda v: self.v._set("sparsity", int(v))))
        f.addRow("Scale:",
            slider_spin(0.1, 2.0, self.v.scaling_mod,
                        lambda v: self.v._set("scaling_mod", v),
                        is_float=True, decimals=2, step=0.05))
        ptc_cb = QCheckBox()
        ptc_cb.setChecked(self.v.show_ptc)
        ptc_cb.toggled.connect(lambda v: self.v._set("show_ptc", v))
        f.addRow("Pointcloud:", ptc_cb)
        disk_cb = QCheckBox("disk mode")
        disk_cb.setChecked(self.v.surfel_disk)
        disk_cb.toggled.connect(lambda v: self.v._set("surfel_disk", v))
        f.addRow("", disk_cb)
        f.addRow("Point Size:",
            slider_spin(0.001, 0.1, self.v.point_size,
                        lambda v: self.v._set("point_size", v),
                        is_float=True, decimals=3, step=0.001))
        vbl.addWidget(g)

    def _build_crop_group(self, vbl: QVBoxLayout):
        g, f = self._group("Crop Box")
        crop_cb = QCheckBox()
        crop_cb.setChecked(self.v.crop_enabled)
        crop_cb.toggled.connect(lambda v: self.v._set("crop_enabled", v))
        f.addRow("Enable:", crop_cb)
        for ax, attr in [("X", "crop_x"), ("Y", "crop_y"), ("Z", "crop_z")]:
            vals = getattr(self.v, attr)
            f.addRow(f"{ax} min:",
                slider_spin(-16.0, 16.0, vals[0],
                            lambda v, a=attr: self.v._set_crop(a, 0, v),
                            is_float=True, decimals=1, step=0.1))
            f.addRow(f"{ax} max:",
                slider_spin(-16.0, 16.0, vals[1],
                            lambda v, a=attr: self.v._set_crop(a, 1, v),
                            is_float=True, decimals=1, step=0.1))
        vbl.addWidget(g)

    # ── culling callback ───────────────────────────────────────────────────────

    def _on_build_index_clicked(self):
        self._build_idx_btn.setEnabled(False)
        self._build_idx_btn.setText("Building...")
        self._culling_status_label.setText("Building index...")
        self._culling_status_label.setStyleSheet("color: #aaaaaa;")
        self._culling_progress.show()
        threading.Thread(target=self.v._build_index_worker, daemon=True).start()

    # ── compression callback ───────────────────────────────────────────────────

    def _on_compression_changed(self, index: int):
        if index == self.v._current_compression:
            return
        self._compression_combo.setEnabled(False)
        label = COMPRESSION_LABELS[index]
        action = ("Loading" if index == 0 or os.path.exists(
            _compressed_ply_path(self.v.ply_path, index)) else "Compressing")
        self._compression_status_label.setText(f"{action} {label.split(' — ')[0]}...")
        self._compression_status_label.setStyleSheet("color: #aaaaaa;")
        self._compress_progress.show()
        self.v._start_compression(index)

    # ── resolution callbacks ───────────────────────────────────────────────────

    def _on_width_changed(self, val: int):
        self.v._render_w = val
        if self.v._lock_ar:
            new_h = max(2, round(val / self.v._aspect_ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self.v._render_h = new_h
        else:
            self.v._aspect_ratio = val / max(self.v._render_h, 1)
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self.v._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            self._ar_combo.blockSignals(True)
            self._ar_combo.setCurrentText("Custom")
            self._ar_combo.blockSignals(False)
            self._ar_custom_spin.setEnabled(True)
        self.v._render_trig.set()

    def _on_height_changed(self, val: int):
        self.v._render_h = val
        if self.v._lock_ar:
            new_w = max(2, round(val * self.v._aspect_ratio))
            self._res_width_spin.blockSignals(True)
            self._res_width_spin.setValue(new_w)
            self._res_width_spin.blockSignals(False)
            self.v._render_w = new_w
        else:
            self.v._aspect_ratio = max(self.v._render_w, 1) / val
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self.v._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            self._ar_combo.blockSignals(True)
            self._ar_combo.setCurrentText("Custom")
            self._ar_combo.blockSignals(False)
            self._ar_custom_spin.setEnabled(True)
        self.v._render_trig.set()

    def _on_preset_changed(self, text: str):
        wh = RESOLUTION_PRESETS.get(text)
        if wh is None:
            return
        w, h = wh
        self.v._render_w = w; self.v._render_h = h
        for spin, val in [(self._res_width_spin, w), (self._res_height_spin, h)]:
            spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)
        self.v._render_trig.set()

    def _on_ar_combo_changed(self, text: str):
        ratio = AR_RATIO_VALUES.get(text)
        if ratio is None:
            self._ar_custom_spin.setEnabled(True)
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self.v._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            return
        self._ar_custom_spin.setEnabled(False)
        self._ar_custom_spin.blockSignals(True)
        self._ar_custom_spin.setValue(ratio)
        self._ar_custom_spin.blockSignals(False)
        self.v._aspect_ratio = ratio
        if self.v._lock_ar:
            new_h = max(2, round(self.v._render_w / ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self.v._render_h = new_h
        self.v._render_trig.set()

    def _on_ar_custom_changed(self, val: float):
        self.v._aspect_ratio = max(0.1, val)
        if self.v._lock_ar:
            new_h = max(2, round(self.v._render_w / self.v._aspect_ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self.v._render_h = new_h
        self.v._render_trig.set()

    # ── notification API (called by LocalViewer._poll_frame) ───────────────────

    def on_culling_ready(self, n_leaves: int):
        self._culling_status_label.setText(f"Active — {n_leaves:,} leaves")
        self._culling_status_label.setStyleSheet("color: #66cc66;")
        self._build_idx_btn.setEnabled(True)
        self._build_idx_btn.setText("Rebuild Index")
        self._culling_progress.hide()

    def on_build_error(self):
        self._culling_status_label.setText("Build failed — see console")
        self._culling_status_label.setStyleSheet("color: #ff4444;")
        self._build_idx_btn.setEnabled(True)
        self._build_idx_btn.setText("Retry Build")
        self._culling_progress.hide()

    def on_ply_loaded(self, level: int, sh_degree: int):
        self.v._current_compression = level
        self._sh_spin.setMaximum(sh_degree)
        self._sh_spin.setValue(sh_degree)
        self._compression_status_label.setText(f"L{level} active")
        self._compression_status_label.setStyleSheet("color: #66cc66;")
        self._compression_combo.setEnabled(True)
        self._compression_combo.blockSignals(True)
        self._compression_combo.setCurrentIndex(level)
        self._compression_combo.blockSignals(False)
        self._compress_progress.hide()

    def on_compress_error(self, current_level: int):
        self._compression_status_label.setText("Failed — see console")
        self._compression_status_label.setStyleSheet("color: #ff4444;")
        self._compression_combo.blockSignals(True)
        self._compression_combo.setCurrentIndex(current_level)
        self._compression_combo.blockSignals(False)
        self._compression_combo.setEnabled(True)
        self._compress_progress.hide()

    def update_stats(self, fps_str: str, splat_str: str, gpu_str: str):
        self._fps_label.setText(fps_str)
        self._splat_label.setText(splat_str)
        self._gpu_label.setText(gpu_str)

    # ── config state accessor ─────────────────────────────────────────────────

    def ui_state_for_config(self) -> dict:
        return {
            "ar_preset":          self._ar_combo.currentText(),
            "show_fps_overlay":   self.v.render_widget._show_fps_overlay,
            "show_splat_overlay": self.v.render_widget._show_splat_overlay,
        }
