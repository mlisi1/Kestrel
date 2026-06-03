"""LocalViewer: native single-window 2DGS viewer using PyQt5."""

from __future__ import annotations

import math
import os
import sys
import threading
import time

import numpy as np
import torch

from PyQt5.QtCore    import Qt, QLocale, QTimer
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QDoubleSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "2d_gaussian_splatting"))

from cameras.cameras          import Cameras
from utils.graphics_utils      import fov2focal
from renderer                  import ViewerRenderer
from model                     import GaussianModelforViewer as GaussianModel
from viewer.config             import (
    UP_AXIS_OPTIONS, UP_AXIS_VECTORS,
    AR_RATIO_OPTIONS, AR_RATIO_VALUES,
    RESOLUTION_PRESETS,
    RENDER_TYPES, RENDER_TYPE_MAP,
    CONFIG_DEFAULTS, load_config, save_config, fmt_splats,
)
from viewer.camera             import OrbitCamera
from viewer.widgets            import RenderWidget
from viewer.ui_helpers         import slider_spin, combo

from PyQt5.QtWidgets import QComboBox


def _detect_sh_degree(path: str) -> int:
    from plyfile import PlyData
    el = PlyData.read(path).elements[0]
    n_rest = sum(1 for p in el.properties if p.name.startswith("f_rest_"))
    if n_rest == 0:
        return 0
    for deg in range(1, 5):
        if 3 * ((deg + 1) ** 2 - 1) == n_rest:
            return deg
    raise ValueError(f"Cannot determine SH degree: {n_rest} f_rest_ properties in {path}")


def _load_octree(ply_path: str) -> dict | None:
    import numpy as _np
    idx_path = os.path.splitext(ply_path)[0] + ".idx"
    if not os.path.exists(idx_path):
        print(f"[viewer] No octree index found at {idx_path} — frustum culling disabled")
        return None
    data = _np.load(idx_path)
    L = len(data["node_aabbs"])
    print(f"[viewer] Loaded octree index: {L:,} leaf nodes from {idx_path}")
    return {"node_aabbs":   data["node_aabbs"],
            "node_offsets": data["node_offsets"],
            "flat_indices": data["flat_indices"]}


def _load_ply_into_model(model, path: str) -> None:
    """Load a 2DGS .ply file into a GaussianModel."""
    from plyfile import PlyData
    import numpy as _np

    FP16_MAX = 65504.0
    el = PlyData.read(path).elements[0]

    def _sorted(prefix):
        names = [p.name for p in el.properties if p.name.startswith(prefix)]
        return sorted(names, key=lambda x: int(x.split("_")[-1]))

    def _stack(names):
        return _np.stack([_np.asarray(el[n]) for n in names], axis=1).astype(_np.float32)

    def _safe(arr, label=""):
        n_bad = int(_np.sum(~_np.isfinite(arr)))
        if n_bad > 0:
            print(f"[viewer]   {label}: zeroing {n_bad:,} NaN/Inf values")
            arr = _np.where(_np.isfinite(arr), arr, 0.0)
        if _np.abs(arr).max() > FP16_MAX:
            thr = float(min(_np.percentile(_np.abs(arr), 99.9), FP16_MAX))
            n_c = int(_np.sum(_np.abs(arr) > thr))
            print(f"[viewer]   {label}: clipping {n_c:,} overflow values to ±{thr:.1f}")
            arr = _np.clip(arr, -thr, thr)
        return arr

    xyz       = _safe(_stack(["x", "y", "z"]),                                  "xyz")
    opacities = _safe(_np.asarray(el["opacity"], dtype=_np.float32)[..., None],  "opacity")
    scales    = _safe(_stack(_sorted("scale_")),                                 "scale")
    rotations = _safe(_stack(_sorted("rot_")),                                   "rotation")
    features_dc = _safe(_stack(_sorted("f_dc_")), "f_dc")[:, _np.newaxis, :]

    f_rest_names = _sorted("f_rest_")
    if f_rest_names:
        f_rest_flat = _safe(_stack(f_rest_names), "f_rest")   # [N, 3*K]
        n_coeffs    = f_rest_flat.shape[1]
        file_sh     = next((d for d in range(1, 5)
                            if 3 * ((d + 1) ** 2 - 1) == n_coeffs), 0)
        K           = (file_sh + 1) ** 2 - 1
        features_rest = f_rest_flat.reshape((-1, 3, K)).transpose(0, 2, 1)   # [N, K, 3]
        print(f"[viewer]   SH degree: {file_sh} ({n_coeffs} f_rest_ props)")
    else:
        features_rest = _np.zeros((xyz.shape[0], 0, 3), dtype=_np.float32)
        file_sh = 0
        print("[viewer]   No f_rest_ — degree-0 (view-independent colour)")

    def _cuda(a):
        return torch.from_numpy(a.astype(_np.float32)).cuda()

    model._xyz           = _cuda(xyz)
    model._opacity       = _cuda(opacities)
    model._features_dc   = _cuda(features_dc)
    model._features_rest = _cuda(features_rest)
    model._scaling       = _cuda(scales)
    model._rotation      = _cuda(rotations)

    if hasattr(model, "active_sh_degree"):
        model.active_sh_degree = file_sh
    if hasattr(model, "max_sh_degree"):
        model.max_sh_degree = min(model.max_sh_degree, file_sh)

    vram_mb = sum(
        t.element_size() * t.nelement()
        for t in [model._xyz, model._opacity, model._features_dc,
                  model._features_rest, model._scaling, model._rotation]
    ) / 1024 / 1024
    print(f"[viewer]   Loaded {xyz.shape[0]:,} splats — {vram_mb:.0f} MB VRAM")


class LocalViewer(QMainWindow):

    def __init__(self, ply_path: str, args):
        super().__init__()
        self.device   = torch.device("cuda")
        self.ply_path = ply_path
        self.cam_tf   = torch.eye(4, dtype=torch.float64)

        # Config (loaded before model so defaults are available everywhere)
        self._cfg = load_config()
        cfg = self._cfg

        # Load model
        sh = _detect_sh_degree(ply_path) if args.sh_degree < 0 else args.sh_degree
        self._max_sh = sh
        model = GaussianModel(sh_degree=sh)
        _load_ply_into_model(model, ply_path)

        if getattr(args, 'build_index', False):
            from build_index import build_octree, read_xyz
            import numpy as _np
            idx_path = os.path.splitext(ply_path)[0] + ".idx"
            leaf_max = getattr(args, 'leaf_max', 5000)
            print(f"[viewer] Building octree index (leaf_max={leaf_max:,}) ...")
            t0 = time.perf_counter()
            xyz = read_xyz(ply_path)
            node_aabbs, node_offsets, flat_indices = build_octree(xyz, leaf_max=leaf_max)
            print(f"[viewer]   Done in {time.perf_counter()-t0:.1f}s — saving to {idx_path}")
            with open(idx_path, "wb") as fh:
                _np.savez_compressed(fh, node_aabbs=node_aabbs,
                                      node_offsets=node_offsets,
                                      flat_indices=flat_indices)
            octree = {"node_aabbs": node_aabbs, "node_offsets": node_offsets,
                      "flat_indices": flat_indices}
        else:
            octree = _load_octree(ply_path)

        self._total_vram = (torch.cuda.get_device_properties(self.device)
                            .total_memory / 1024**2)
        bg = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=self.device)
        self.renderer = ViewerRenderer(
            model, bg,
            do_initialize=True,
            octree=octree,
            culling_enabled=not getattr(args, 'no_culling', False),
            profiling_enabled=not getattr(args, 'no_profiling', False),
        )

        # Camera
        self.camera = OrbitCamera(
            world_up=UP_AXIS_VECTORS[cfg["world_up"]].copy()
        )

        # Render resolution
        self._render_w     = cfg["render_w"]
        self._render_h     = cfg["render_h"]
        self._aspect_ratio = cfg["render_w"] / max(cfg["render_h"], 1)
        self._lock_ar      = cfg["lock_ar"]

        # Render options
        self.fov_deg        = cfg["fov_deg"]
        self.render_type    = cfg["render_type"] if cfg["render_type"] in RENDER_TYPES else "RGB"
        self.render_type1   = "RGB"
        self.render_type2   = "RGB"
        self.split_enabled  = False
        self.split_pos      = 0.5
        self.depth_ratio    = cfg["depth_ratio"]
        _sh_cfg             = cfg["active_sh_degree"]
        self.sh_degree      = sh if _sh_cfg < 0 else min(_sh_cfg, sh)
        self.opacity_thresh = cfg["opacity_thresh"]
        self.sparsity       = cfg["sparsity"]
        self.scaling_mod    = cfg["scaling_mod"]
        self.point_size     = cfg["point_size"]
        self.show_ptc       = False
        self.surfel_disk    = False
        self.crop_enabled   = False
        self.crop_x         = [-4.0, 4.0]
        self.crop_y         = [-4.0, 4.0]
        self.crop_z         = [-4.0, 4.0]

        # Navigation speeds (multipliers)
        self._move_speed  = cfg["move_speed"]
        self._orbit_speed = cfg["orbit_speed"]

        # KB rotation inversion (arrow keys only, not WASD)
        self._kb_inv_x  = cfg["kb_inv_x"]
        self._kb_inv_y  = cfg["kb_inv_y"]
        self._keys_held = set()

        # Overlay visibility
        self._show_fps_overlay   = cfg["show_fps_overlay"]
        self._show_splat_overlay = cfg["show_splat_overlay"]

        # Frame slot
        self._frame_slot  = None
        self._frame_lock  = threading.Lock()
        self._render_trig = threading.Event()
        self._running     = True

        self._build_ui()
        self._start_render_thread()

        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_frame)
        self._poll_timer.start(16)

        self._kb_timer = QTimer()
        self._kb_timer.timeout.connect(self._kb_tick)
        self._kb_timer.start(16)

        self._render_trig.set()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("2DGS Viewer")
        self.resize(1640, 900)

        self.render_widget = RenderWidget(self.camera)
        self.render_widget.mouse_inv_x         = self._cfg["mouse_inv_x"]
        self.render_widget.mouse_inv_y         = self._cfg["mouse_inv_y"]
        self.render_widget._show_fps_overlay   = self._show_fps_overlay
        self.render_widget._show_splat_overlay = self._show_splat_overlay
        self.render_widget.camera_changed.connect(self._on_camera_change)
        self.render_widget.key_down.connect(self._on_key_down)
        self.render_widget.key_up.connect(self._on_key_up)

        controls = self._build_controls()
        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(370)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.render_widget)
        splitter.setSizes([370, 1270])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        self.setCentralWidget(splitter)

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        g = QGroupBox(title)
        f = QFormLayout(); f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        g.setLayout(f)
        return g, f

    def _build_controls(self) -> QWidget:
        w   = QWidget()
        vbl = QVBoxLayout(w); vbl.setContentsMargins(4, 4, 4, 4)

        # ── Status ─────────────────────────────────────────────────────────────
        g, f = self._group("Status")
        self._fps_label   = QLabel("--")
        self._gpu_label   = QLabel("--")
        self._splat_label = QLabel("--")
        f.addRow("FPS:",    self._fps_label)
        f.addRow("Splats:", self._splat_label)
        f.addRow("VRAM:",   self._gpu_label)

        overlay_row = QWidget(); ol = QHBoxLayout(overlay_row); ol.setContentsMargins(0, 0, 0, 0)
        fps_cb   = QCheckBox("FPS graph");   fps_cb.setChecked(self._show_fps_overlay)
        splat_cb = QCheckBox("Splat graph"); splat_cb.setChecked(self._show_splat_overlay)
        fps_cb.toggled.connect(lambda v: setattr(self.render_widget, '_show_fps_overlay',   v))
        splat_cb.toggled.connect(lambda v: setattr(self.render_widget, '_show_splat_overlay', v))
        ol.addWidget(fps_cb); ol.addWidget(splat_cb)
        f.addRow("Overlays:", overlay_row)
        vbl.addWidget(g)

        # ── Camera ─────────────────────────────────────────────────────────────
        g, f = self._group("Camera")

        f.addRow("World Up:", combo(UP_AXIS_OPTIONS, self._cfg["world_up"], self._on_world_up_changed))

        f.addRow("FOV:",
            slider_spin(10, 120, self.fov_deg, self._set_fov))

        f.addRow("Move Speed:",
            slider_spin(0.1, 5.0, self._move_speed,
                         lambda v: setattr(self, '_move_speed', v),
                         is_float=True, decimals=2, step=0.05))
        f.addRow("Orbit Speed:",
            slider_spin(0.1, 5.0, self._orbit_speed,
                         lambda v: setattr(self, '_orbit_speed', v),
                         is_float=True, decimals=2, step=0.05))

        mouse_row = QWidget(); ml = QHBoxLayout(mouse_row); ml.setContentsMargins(0,0,0,0)
        minv_x = QCheckBox("Inv X"); minv_x.setChecked(self._cfg["mouse_inv_x"])
        minv_y = QCheckBox("Inv Y"); minv_y.setChecked(self._cfg["mouse_inv_y"])
        minv_x.toggled.connect(lambda v: setattr(self.render_widget, 'mouse_inv_x', v))
        minv_y.toggled.connect(lambda v: setattr(self.render_widget, 'mouse_inv_y', v))
        ml.addWidget(minv_x); ml.addWidget(minv_y)
        f.addRow("Mouse Inv:", mouse_row)

        kb_row = QWidget(); kl = QHBoxLayout(kb_row); kl.setContentsMargins(0,0,0,0)
        kinv_x = QCheckBox("Inv X"); kinv_x.setChecked(self._cfg["kb_inv_x"])
        kinv_y = QCheckBox("Inv Y"); kinv_y.setChecked(self._cfg["kb_inv_y"])
        kinv_x.toggled.connect(lambda v: setattr(self, '_kb_inv_x', v))
        kinv_y.toggled.connect(lambda v: setattr(self, '_kb_inv_y', v))
        kl.addWidget(kinv_x); kl.addWidget(kinv_y)
        f.addRow("KB Rot Inv:", kb_row)

        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self._reset_camera)
        f.addRow(btn_reset)
        hint = QLabel("LMB: orbit  |  RMB: pan  |  Scroll: zoom\n"
                       "WASD: translate  |  Arrows: FPS look  |  R: reset")
        hint.setWordWrap(True)
        f.addRow(hint)
        vbl.addWidget(g)

        # ── Resolution ─────────────────────────────────────────────────────────
        g, f = self._group("Resolution")

        self._res_width_spin = QSpinBox()
        self._res_width_spin.setRange(2, 7680); self._res_width_spin.setSingleStep(2)
        self._res_width_spin.setValue(self._render_w)

        self._res_height_spin = QSpinBox()
        self._res_height_spin.setRange(2, 4320); self._res_height_spin.setSingleStep(2)
        self._res_height_spin.setValue(self._render_h)

        self._ar_combo = QComboBox()
        self._ar_combo.addItems(AR_RATIO_OPTIONS)

        self._ar_custom_spin = QDoubleSpinBox()
        self._ar_custom_spin.setLocale(QLocale(QLocale.C))
        self._ar_custom_spin.setRange(0.1, 10.0); self._ar_custom_spin.setDecimals(3)
        self._ar_custom_spin.setSingleStep(0.001); self._ar_custom_spin.setFixedWidth(72)
        self._ar_custom_spin.setValue(self._aspect_ratio)
        self._ar_custom_spin.setEnabled(False)

        self._lock_ar_cb = QCheckBox("Lock AR")
        self._lock_ar_cb.setChecked(self._lock_ar)
        self._lock_ar_cb.toggled.connect(lambda v: setattr(self, '_lock_ar', v))

        preset_combo = QComboBox()
        preset_combo.addItems(list(RESOLUTION_PRESETS.keys()))
        preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_combo.setCurrentText("720p (HD)")

        for spin, val in [(self._res_width_spin, self._render_w),
                          (self._res_height_spin, self._render_h)]:
            spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)

        self._res_width_spin.valueChanged.connect(self._on_width_changed)
        self._res_height_spin.valueChanged.connect(self._on_height_changed)
        self._ar_combo.currentTextChanged.connect(self._on_ar_combo_changed)
        self._ar_custom_spin.valueChanged.connect(self._on_ar_custom_changed)
        self._ar_combo.setCurrentText(self._cfg["ar_preset"])

        f.addRow("Preset:", preset_combo)
        f.addRow("Width:",  self._res_width_spin)
        f.addRow("Height:", self._res_height_spin)
        ar_row = QWidget(); al = QHBoxLayout(ar_row); al.setContentsMargins(0,0,0,0)
        al.addWidget(self._ar_combo, 2)
        al.addWidget(self._ar_custom_spin, 1)
        al.addWidget(self._lock_ar_cb, 1)
        f.addRow("Aspect:", ar_row)
        vbl.addWidget(g)

        # ── Render Options ──────────────────────────────────────────────────────
        g, f = self._group("Render Options")
        f.addRow("Type:", combo(RENDER_TYPES, self.render_type,
            lambda v: self._set("render_type", v)))
        f.addRow("Depth Ratio:",
            slider_spin(0.0, 1.0, self.depth_ratio,
                lambda v: self._set("depth_ratio", v),
                is_float=True, decimals=2, step=0.05))
        split_cb = QCheckBox()
        split_cb.setChecked(self.split_enabled)
        split_cb.toggled.connect(lambda v: self._set("split_enabled", v))
        f.addRow("Split View:", split_cb)
        f.addRow("Split Pos:",
            slider_spin(0.0, 1.0, self.split_pos,
                lambda v: self._set("split_pos", v),
                is_float=True, decimals=2, step=0.01))
        f.addRow("Left:",  combo(RENDER_TYPES, self.render_type1,
            lambda v: self._set("render_type1", v)))
        f.addRow("Right:", combo(RENDER_TYPES, self.render_type2,
            lambda v: self._set("render_type2", v)))
        vbl.addWidget(g)

        # ── Gaussian Model ──────────────────────────────────────────────────────
        g, f = self._group("Gaussian Model")
        if self._max_sh > 0:
            f.addRow("SH Degree:",
                slider_spin(0, self._max_sh, self.sh_degree,
                    lambda v: self._set("sh_degree", int(v))))
        f.addRow("Opacity Thr:",
            slider_spin(0.0, 0.5, self.opacity_thresh,
                lambda v: self._set("opacity_thresh", v),
                is_float=True, decimals=3, step=0.005))
        f.addRow("Sparsity:",
            slider_spin(1, 10, self.sparsity,
                lambda v: self._set("sparsity", int(v))))
        f.addRow("Scale:",
            slider_spin(0.1, 2.0, self.scaling_mod,
                lambda v: self._set("scaling_mod", v),
                is_float=True, decimals=2, step=0.05))
        ptc_cb = QCheckBox()
        ptc_cb.setChecked(self.show_ptc)
        ptc_cb.toggled.connect(lambda v: self._set("show_ptc", v))
        f.addRow("Pointcloud:", ptc_cb)
        disk_cb = QCheckBox("disk mode")
        disk_cb.setChecked(self.surfel_disk)
        disk_cb.toggled.connect(lambda v: self._set("surfel_disk", v))
        f.addRow("", disk_cb)
        f.addRow("Point Size:",
            slider_spin(0.001, 0.1, self.point_size,
                lambda v: self._set("point_size", v),
                is_float=True, decimals=3, step=0.001))
        vbl.addWidget(g)

        # ── Crop Box ────────────────────────────────────────────────────────────
        g, f = self._group("Crop Box")
        crop_cb = QCheckBox()
        crop_cb.setChecked(self.crop_enabled)
        crop_cb.toggled.connect(lambda v: self._set("crop_enabled", v))
        f.addRow("Enable:", crop_cb)
        for ax, attr in [("X", "crop_x"), ("Y", "crop_y"), ("Z", "crop_z")]:
            vals = getattr(self, attr)
            f.addRow(f"{ax} min:",
                slider_spin(-16.0, 16.0, vals[0],
                    lambda v, a=attr: self._set_crop(a, 0, v),
                    is_float=True, decimals=1, step=0.1))
            f.addRow(f"{ax} max:",
                slider_spin(-16.0, 16.0, vals[1],
                    lambda v, a=attr: self._set_crop(a, 1, v),
                    is_float=True, decimals=1, step=0.1))
        vbl.addWidget(g)

        vbl.addStretch()
        return w

    # ── Settings helpers ───────────────────────────────────────────────────────

    def _set(self, attr: str, val):
        setattr(self, attr, val)
        self._render_trig.set()

    def _set_fov(self, val: float):
        self.fov_deg = float(val)
        self._render_trig.set()

    def _set_crop(self, attr: str, idx: int, val: float):
        getattr(self, attr)[idx] = val
        self._render_trig.set()

    def _on_camera_change(self):
        self._render_trig.set()

    def _reset_camera(self):
        self.camera.look_at  = np.array([0., 0., 0.])
        self.camera.distance = 5.0
        self.camera.yaw      = 0.0
        self.camera.pitch    = 0.3
        self._render_trig.set()

    def _on_world_up_changed(self, text: str):
        self.camera.world_up = UP_AXIS_VECTORS[text].copy()
        self._render_trig.set()

    # ── Resolution callbacks ───────────────────────────────────────────────────

    def _on_width_changed(self, val: int):
        self._render_w = val
        if self._lock_ar:
            new_h = max(2, round(val / self._aspect_ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self._render_h = new_h
        else:
            self._aspect_ratio = val / max(self._render_h, 1)
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            self._ar_combo.blockSignals(True)
            self._ar_combo.setCurrentText("Custom")
            self._ar_combo.blockSignals(False)
            self._ar_custom_spin.setEnabled(True)
        self._render_trig.set()

    def _on_height_changed(self, val: int):
        self._render_h = val
        if self._lock_ar:
            new_w = max(2, round(val * self._aspect_ratio))
            self._res_width_spin.blockSignals(True)
            self._res_width_spin.setValue(new_w)
            self._res_width_spin.blockSignals(False)
            self._render_w = new_w
        else:
            self._aspect_ratio = max(self._render_w, 1) / val
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            self._ar_combo.blockSignals(True)
            self._ar_combo.setCurrentText("Custom")
            self._ar_combo.blockSignals(False)
            self._ar_custom_spin.setEnabled(True)
        self._render_trig.set()

    def _on_preset_changed(self, text: str):
        wh = RESOLUTION_PRESETS.get(text)
        if wh is None:
            return
        w, h = wh
        self._render_w = w; self._render_h = h
        for spin, val in [(self._res_width_spin, w), (self._res_height_spin, h)]:
            spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)
        self._render_trig.set()

    def _on_ar_combo_changed(self, text: str):
        ratio = AR_RATIO_VALUES.get(text)
        if ratio is None:
            self._ar_custom_spin.setEnabled(True)
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            return
        self._ar_custom_spin.setEnabled(False)
        self._ar_custom_spin.blockSignals(True)
        self._ar_custom_spin.setValue(ratio)
        self._ar_custom_spin.blockSignals(False)
        self._aspect_ratio = ratio
        if self._lock_ar:
            new_h = max(2, round(self._render_w / ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self._render_h = new_h
        self._render_trig.set()

    def _on_ar_custom_changed(self, val: float):
        self._aspect_ratio = max(0.1, val)
        if self._lock_ar:
            new_h = max(2, round(self._render_w / self._aspect_ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self._render_h = new_h
        self._render_trig.set()

    # ── Keyboard navigation ────────────────────────────────────────────────────

    def _on_key_down(self, key: int):
        if key == -1:
            self._keys_held.clear(); return
        if key == Qt.Key_R:
            self._reset_camera(); return
        self._keys_held.add(key)

    def _on_key_up(self, key: int):
        if key == -1:
            self._keys_held.clear(); return
        self._keys_held.discard(key)

    def _kb_tick(self):
        k = self._keys_held
        if not k:
            return

        move_speed  = self.camera.distance * 0.02 * self._move_speed
        orbit_speed = 0.02 * self._orbit_speed

        fwd_d = right_d = up_d = dyaw = dpitch = 0.0

        if Qt.Key_W in k: fwd_d   += move_speed
        if Qt.Key_S in k: fwd_d   -= move_speed
        if Qt.Key_A in k: right_d -= move_speed
        if Qt.Key_D in k: right_d += move_speed

        if Qt.Key_E in k: up_d += move_speed
        if Qt.Key_Q in k: up_d -= move_speed

        if Qt.Key_Left  in k: dyaw   -= orbit_speed
        if Qt.Key_Right in k: dyaw   += orbit_speed
        if Qt.Key_Up    in k: dpitch += orbit_speed
        if Qt.Key_Down  in k: dpitch -= orbit_speed

        if self._kb_inv_x: dyaw   = -dyaw
        if self._kb_inv_y: dpitch = -dpitch

        changed = False
        if fwd_d or right_d:
            self.camera.move(fwd_d, right_d); changed = True
        if up_d:
            self.camera.translate_up(up_d); changed = True
        if dyaw or dpitch:
            self.camera.fps_look(dyaw, dpitch); changed = True
        if changed:
            self._render_trig.set()

    # ── Camera construction ────────────────────────────────────────────────────

    def _build_camera(self, W: int, H: int) -> "Camera":
        fov_rad = math.radians(self.fov_deg)
        fx = torch.tensor([fov2focal(fov_rad, H)], dtype=torch.float)
        R, T = self.camera.build_RT(self.cam_tf)
        return Cameras(
            R=R.unsqueeze(0), T=T.unsqueeze(0), fx=fx, fy=fx,
            cx=torch.tensor([W // 2], dtype=torch.int),
            cy=torch.tensor([H // 2], dtype=torch.int),
            width=torch.tensor([W],   dtype=torch.int),
            height=torch.tensor([H],  dtype=torch.int),
            appearance_id=torch.tensor([0], dtype=torch.int),
            normalized_appearance_id=torch.tensor([0.], dtype=torch.float),
            time=torch.tensor([0.], dtype=torch.float),
            distortion_params=None,
            camera_type=torch.tensor([0], dtype=torch.int),
        )[0].to_device(self.device)

    # ── Render thread ──────────────────────────────────────────────────────────

    def _start_render_thread(self):
        threading.Thread(target=self._render_loop, daemon=True).start()

    def _render_loop(self):
        while self._running:
            self._render_trig.wait(timeout=0.2)
            self._render_trig.clear()
            if not self._running:
                break

            W = max(self._render_w, 2)
            H = max(self._render_h, 2)
            valid_range = (self.crop_x, self.crop_y, self.crop_z) if self.crop_enabled else None

            t0 = time.perf_counter()
            try:
                cam = self._build_camera(W, H)
                with torch.no_grad():
                    image = self.renderer.get_outputs(
                        cam,
                        valid_range       = valid_range,
                        split             = self.split_enabled,
                        slider            = self.split_pos,
                        active_sh_degree  = self.sh_degree,
                        scaling_modifier  = self.scaling_mod,
                        sparsity          = self.sparsity,
                        opacity_threshold = self.opacity_thresh,
                        depth_ratio       = self.depth_ratio,
                        render_type       = RENDER_TYPE_MAP[self.render_type],
                        render_type1      = RENDER_TYPE_MAP[self.render_type1],
                        render_type2      = RENDER_TYPE_MAP[self.render_type2],
                        show_ptc          = self.show_ptc and not self.surfel_disk,
                        show_disk         = self.show_ptc and self.surfel_disk,
                        point_size        = self.point_size,
                    )
            except Exception:
                import traceback; traceback.print_exc()
                continue

            img_np   = (image.clamp(0., 1.)
                        .permute(1, 2, 0).mul(255).byte().cpu().numpy())
            dt       = time.perf_counter() - t0
            fps_val  = 1.0 / dt if dt > 0 else 0.0
            fps_str  = f"{fps_val:.1f} fps" if dt > 0 else "--"
            n_splats = self.renderer.last_visible_count
            used     = torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
            gpu_str  = f"{used / 1024**2:.0f} / {self._total_vram:.0f} MB"

            with self._frame_lock:
                self._frame_slot = (img_np, fps_str, gpu_str, fps_val, n_splats,
                                    fmt_splats(n_splats))

    # ── Frame polling ──────────────────────────────────────────────────────────

    def _poll_frame(self):
        with self._frame_lock:
            slot = self._frame_slot
            self._frame_slot = None
        if slot is None:
            return
        img_np, fps_str, gpu_str, fps_val, n_splats, splat_str = slot
        self.render_widget.set_frame(img_np)
        if fps_val > 0:
            self.render_widget.update_fps(fps_val)
        if n_splats > 0:
            self.render_widget.update_splat_count(n_splats)
        self._fps_label.setText(fps_str)
        self._splat_label.setText(splat_str)
        self._gpu_label.setText(gpu_str)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def closeEvent(self, e):
        self._running = False
        self._render_trig.set()
        self._cfg.update({
            "fov_deg":           self.fov_deg,
            "move_speed":        self._move_speed,
            "orbit_speed":       self._orbit_speed,
            "mouse_inv_x":       self.render_widget.mouse_inv_x,
            "mouse_inv_y":       self.render_widget.mouse_inv_y,
            "kb_inv_x":          self._kb_inv_x,
            "kb_inv_y":          self._kb_inv_y,
            "world_up":          next((k for k, v in UP_AXIS_VECTORS.items()
                                       if np.allclose(v, self.camera.world_up)), "+Z"),
            "render_w":          self._render_w,
            "render_h":          self._render_h,
            "ar_preset":         self._ar_combo.currentText(),
            "lock_ar":           self._lock_ar,
            "depth_ratio":       self.depth_ratio,
            "active_sh_degree":  self.sh_degree,
            "opacity_thresh":    self.opacity_thresh,
            "sparsity":          self.sparsity,
            "scaling_mod":       self.scaling_mod,
            "point_size":        self.point_size,
            "render_type":       self.render_type,
            "show_fps_overlay":  self.render_widget._show_fps_overlay,
            "show_splat_overlay": self.render_widget._show_splat_overlay,
        })
        save_config(self._cfg)
        super().closeEvent(e)


# ── Entry points ───────────────────────────────────────────────────────────────

def run_local_viewer(ply_path: str, args) -> None:
    """Launch the Qt GUI. Called from main.py or directly."""
    app = QApplication.instance() or QApplication(sys.argv)
    QLocale.setDefault(QLocale(QLocale.C))   # force dot as decimal separator
    app.setStyle("Fusion")
    viewer = LocalViewer(ply_path, args)
    viewer.show()
    sys.exit(app.exec_())
