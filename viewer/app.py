"""Kestrel: native single-window 2DGS viewer using PyQt5."""

from __future__ import annotations

import math
import os
import sys
import threading
import time

import numpy as np
import torch

from PyQt5.QtCore    import Qt, QLocale, QTimer
from PyQt5.QtGui     import QIcon
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QMainWindow, QScrollArea, QWidget,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mirrors gsplat2d_rendering.camera.ZNEAR_DEFAULT (not re-exported from the
# package). Only used for the dual-camera-debug frustum wireframe's near
# plane -- see _render_loop for the (scene-scale-relative) far plane.
_DEBUG_FRUSTUM_ZNEAR = 0.01

import gsplat2d_rendering as gs2d

from renderer             import ViewerRenderer
from viewer.config        import (
    UP_AXIS_VECTORS,
    RENDER_TYPES, RENDER_TYPE_MAP,
    load_config, save_config, fmt_splats,
)
from viewer.camera        import OrbitCamera, frustum_corners_world, project_world_points
from viewer.widgets       import RenderWidget
from viewer.sidebar       import Sidebar
from viewer.dialogs       import HelpMenu
from viewer.ply_loader    import (
    _idx_path, _compressed_ply_path, _load_octree,
    _load_gaussian_model, _model_to_cuda,
    load_model_config, save_model_config,
)


class LocalViewer(QMainWindow):

    def __init__(self, ply_path: str, args):
        super().__init__()
        self.device           = torch.device("cuda")
        self.ply_path         = ply_path
        self._leaf_max        = getattr(args, 'leaf_max', 5000)
        self._no_culling_flag = getattr(args, 'no_culling', False)
        self._dual_camera_debug = getattr(args, 'debug_dual_camera', False)
        self.cam_tf           = torch.eye(4, dtype=torch.float64)

        # Config: global defaults merged with per-model saved state
        self._cfg = load_config()
        cfg = self._cfg
        _mcfg = load_model_config(ply_path)
        cfg.update({k: _mcfg[k] for k in _mcfg if k in cfg})

        # gsplat2d_rendering's own log verbosity — set before any library
        # call so model/octree loading below is covered too. --verbosity
        # overrides the saved config for this session; omit it (or leave it
        # at its argparse default of None) to keep using whatever the
        # sidebar's Verbosity combo was last set to. Not `or`: 0 (SILENT) is
        # falsy and would otherwise be silently discarded in favor of cfg.
        _verbosity_arg = getattr(args, 'verbosity', None)
        if _verbosity_arg is not None:
            cfg["verbosity"] = _verbosity_arg
        gs2d.set_verbosity(cfg["verbosity"])

        # Determine which PLY to load (restore last-used compression level)
        _start_compression = int(_mcfg.get("compression", 0))
        if _start_compression > 0:
            if not os.path.exists(_compressed_ply_path(ply_path, _start_compression)):
                _start_compression = 0
        _load_path = (_compressed_ply_path(ply_path, _start_compression)
                      if _start_compression > 0 else ply_path)

        self._current_compression = _start_compression
        model = _load_gaussian_model(_load_path, sh_degree=args.sh_degree, device="cuda")
        self._ply_sh_degree = model.active_sh_degree

        if getattr(args, 'build_index', False):
            from utils.build_index import read_xyz
            idx_path = _idx_path(ply_path)
            os.makedirs(os.path.dirname(idx_path), exist_ok=True)
            xyz = read_xyz(ply_path)
            octree = gs2d.build_octree(xyz, leaf_max=getattr(args, 'leaf_max', 5000))
            # Not gs2d.save_octree(idx_path, ...): a plain string/Path lets
            # np.savez_compressed silently append ".npz" to it — writing
            # through an open handle (save_octree passes file-like objects
            # through untouched) keeps the literal ".idx" filename.
            with open(idx_path, "wb") as fh:
                gs2d.save_octree(fh, octree)
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
            # --no-profiling always forces it off for the session; otherwise
            # falls back to the last state of the sidebar's Profiling checkbox.
            profiling_enabled=cfg["profiling_enabled"] and not getattr(args, 'no_profiling', False),
        )

        self.camera = OrbitCamera(world_up=UP_AXIS_VECTORS[cfg["world_up"]].copy())
        if "camera_look_at" in _mcfg:
            self.camera.look_at  = np.array(_mcfg["camera_look_at"], dtype=np.float64)
            self.camera.distance = float(_mcfg.get("camera_distance", 5.0))
            self.camera.yaw      = float(_mcfg.get("camera_yaw",     0.0))
            self.camera.pitch    = float(_mcfg.get("camera_pitch",   0.3))

        # Dual-camera debug mode (--debug-dual-camera): camera_b is a
        # free-fly observer, entirely separate from camera A (the real
        # render camera, always keyboard-controlled -- see _kb_tick). Mouse
        # orbit follows whichever of the two is currently displayed (see
        # the Tab handler in _on_key_down), so it drives camera_b only once
        # you've toggled the view onto it. Ephemeral by design -- never
        # read from or written to _cfg/_mcfg, see closeEvent.
        self.camera_b = (OrbitCamera(world_up=self.camera.world_up.copy())
                          if self._dual_camera_debug else None)
        self._viewing_camera_b = False

        self._render_w     = cfg["render_w"]
        self._render_h     = cfg["render_h"]
        self._aspect_ratio = cfg["render_w"] / max(cfg["render_h"], 1)
        self._lock_ar      = cfg["lock_ar"]

        self.fov_deg        = cfg["fov_deg"]
        self.render_type    = cfg["render_type"] if cfg["render_type"] in RENDER_TYPES else "RGB"
        self.render_type1   = _mcfg.get("render_type1",  "RGB")
        self.render_type2   = _mcfg.get("render_type2",  "RGB")
        self.split_enabled  = _mcfg.get("split_enabled", False)
        self.split_pos      = _mcfg.get("split_pos",     0.5)
        self.depth_ratio    = cfg["depth_ratio"]
        self.sh_degree      = self._ply_sh_degree
        self.opacity_thresh = cfg["opacity_thresh"]
        self.sparsity       = cfg["sparsity"]
        self.scaling_mod    = cfg["scaling_mod"]
        self.point_size     = cfg["point_size"]
        self.show_ptc       = _mcfg.get("show_ptc",     False)
        self.surfel_disk    = _mcfg.get("surfel_disk",  False)
        self.crop_enabled   = _mcfg.get("crop_enabled", False)
        self.crop_x         = _mcfg.get("crop_x",       [-4.0, 4.0])
        self.crop_y         = _mcfg.get("crop_y",       [-4.0, 4.0])
        self.crop_z         = _mcfg.get("crop_z",       [-4.0, 4.0])

        self._move_speed  = cfg["move_speed"]
        self._orbit_speed = cfg["orbit_speed"]

        self._kb_inv_x  = cfg["kb_inv_x"]
        self._kb_inv_y  = cfg["kb_inv_y"]
        self._keys_held = set()

        self._show_fps_overlay   = cfg["show_fps_overlay"]
        self._show_splat_overlay = cfg["show_splat_overlay"]

        self._frame_slot  = None
        self._frame_lock  = threading.Lock()
        self._render_trig = threading.Event()
        self._running     = True

        # Background index build state
        self._octree_pending       = None
        self._culling_enabled_flag = False
        self._build_error_flag     = False

        # Background PLY compression/load state
        self._ply_pending         = None
        self._ply_loaded_level    = None
        self._compress_error_flag = False

        self._build_ui()

        if not self._no_culling_flag and self.renderer.octree is None:
            self.render_widget._no_culling_warning = True
        threading.Thread(target=self._render_loop, daemon=True).start()

        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_frame)
        self._poll_timer.start(16)

        self._kb_timer = QTimer()
        self._kb_timer.timeout.connect(self._kb_tick)
        self._kb_timer.start(16)

        self._render_trig.set()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Kestrel")
        self.resize(1640, 900)
        self.setWindowIcon(QIcon(os.path.join(_ROOT, "res", "kestrel_icon.png")))
        HelpMenu(self)

        # Mouse follows whichever camera is currently displayed (camera A
        # to start, in both normal and dual-camera-debug mode); the Tab
        # handler in _on_key_down retargets it to camera_b when the view
        # toggles. Keyboard always drives camera A regardless (_kb_tick).
        self.render_widget = RenderWidget(self.camera)
        self.render_widget.mouse_inv_x         = self._cfg["mouse_inv_x"]
        self.render_widget.mouse_inv_y         = self._cfg["mouse_inv_y"]
        self.render_widget._show_fps_overlay   = self._show_fps_overlay
        self.render_widget._show_splat_overlay = self._show_splat_overlay
        self.render_widget.camera_changed.connect(lambda: self._render_trig.set())
        self.render_widget.key_down.connect(self._on_key_down)
        self.render_widget.key_up.connect(self._on_key_up)
        if self._dual_camera_debug:
            self._update_debug_banner()

        self._sidebar = Sidebar(self)
        scroll = QScrollArea()
        scroll.setWidget(self._sidebar.widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(370)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        hl.addWidget(scroll)
        hl.addWidget(self.render_widget, 1)
        self.setCentralWidget(container)


    # ── Settings helpers ───────────────────────────────────────────────────────

    def _set(self, attr: str, val):
        setattr(self, attr, val)
        self._render_trig.set()

    def _set_crop(self, attr: str, idx: int, val: float):
        getattr(self, attr)[idx] = val
        self._render_trig.set()

    def _reset_camera(self):
        self.camera.look_at  = np.array([0., 0., 0.])
        self.camera.distance = 5.0
        self.camera.yaw      = 0.0
        self.camera.pitch    = 0.3
        self._render_trig.set()

    def _on_world_up_changed(self, text: str):
        self.camera.world_up = UP_AXIS_VECTORS[text].copy()
        if self._dual_camera_debug:
            # Keep camera_b's orbit reference axis from going stale relative
            # to a mid-session world-up change.
            self.camera_b.world_up = UP_AXIS_VECTORS[text].copy()
        self._render_trig.set()

    def _update_debug_banner(self):
        if self._viewing_camera_b:
            text = ("DEBUG dual-camera — viewing camera B (mouse orbit)  |  "
                     "camera A still moves in background (WASD)  |  Tab → back to A")
        else:
            text = "DEBUG dual-camera — viewing camera A (WASD + mouse)  |  Tab → camera B (orbit)"
        self.render_widget.set_debug_banner(text)

    # ── Index build ────────────────────────────────────────────────────────────

    def _build_index_worker(self):
        from utils.build_index import read_xyz
        try:
            idx_path = _idx_path(self.ply_path)
            os.makedirs(os.path.dirname(idx_path), exist_ok=True)
            xyz = read_xyz(self.ply_path)
            octree = gs2d.build_octree(xyz, leaf_max=self._leaf_max)
            with open(idx_path, "wb") as fh:
                gs2d.save_octree(fh, octree)
            self._octree_pending = octree
            self._render_trig.set()
        except Exception:
            import traceback; traceback.print_exc()
            self._build_error_flag = True

    # ── Compression ────────────────────────────────────────────────────────────

    def _start_compression(self, level: int):
        threading.Thread(target=self._compression_worker,
                         args=(level,), daemon=True).start()

    def _compression_worker(self, level: int):
        try:
            if level == 0:
                src_path = self.ply_path
            else:
                src_path = _compressed_ply_path(self.ply_path, level)
                if not os.path.exists(src_path):
                    from utils.compress import compress_level1, compress_level2, compress_level3
                    from plyfile import PlyData as _PlyData
                    t0 = time.perf_counter()
                    print(f"[viewer] Compressing to L{level}: {src_path}")
                    original = _PlyData.read(self.ply_path)
                    if   level == 1: compressed = compress_level1(original)
                    elif level == 2: compressed = compress_level2(original)
                    elif level == 3: compressed = compress_level3(original)
                    os.makedirs(os.path.dirname(src_path), exist_ok=True)
                    compressed.write(src_path)
                    print(f"[viewer]   Saved in {time.perf_counter()-t0:.1f}s: {src_path}")

            # Parsed on CPU here (background thread) — moved to CUDA on the
            # render thread by _render_loop, same split the old code used.
            model_cpu = _load_gaussian_model(src_path, sh_degree=-1, device="cpu")
            self._ply_pending = (model_cpu, level)
            self._render_trig.set()
        except Exception:
            import traceback; traceback.print_exc()
            self._compress_error_flag = True

    # ── Keyboard navigation ────────────────────────────────────────────────────

    def _on_key_down(self, key: int):
        if key == -1:
            self._keys_held.clear(); return
        if key == Qt.Key_R:
            self._reset_camera(); return
        if key == Qt.Key_Tab and self._dual_camera_debug:
            self._viewing_camera_b = not self._viewing_camera_b
            # Mouse follows whichever camera is now displayed; keyboard
            # (_kb_tick) always drives camera A regardless.
            self.render_widget.set_camera(self.camera_b if self._viewing_camera_b else self.camera)
            self._update_debug_banner()
            self._render_trig.set()
            return
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
        if Qt.Key_E in k: up_d    += move_speed
        if Qt.Key_Q in k: up_d    -= move_speed

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

    def _build_camera(self, W: int, H: int, orbit_cam: OrbitCamera | None = None) -> gs2d.Camera:
        # fx=fy (square pixels), fx derived from vertical FOV — matches the
        # exact convention the old cameras.cameras.Cameras dataclass used.
        # Not gs2d.Intrinsics.from_fov(): that derives fov_y via a linear
        # angle scaling (fov_y = fov_x * H/W) rather than the exact
        # tan()-based relation this fx=fy scheme implies, which would subtly
        # skew non-square-aspect renders.
        orbit_cam = orbit_cam if orbit_cam is not None else self.camera
        fov_rad = math.radians(self.fov_deg)
        fx = H / (2.0 * math.tan(fov_rad * 0.5))
        intrinsics = gs2d.Intrinsics(width=W, height=H, fx=fx, fy=fx)
        R, T = orbit_cam.build_RT(self.cam_tf)
        return gs2d.Camera.from_w2c(R.numpy(), T.numpy(), intrinsics, device=str(self.device))

    # ── Render thread ──────────────────────────────────────────────────────────

    def _render_loop(self):
        while self._running:
            self._render_trig.wait(timeout=0.2)
            self._render_trig.clear()
            if not self._running:
                break

            pending = self._octree_pending
            if pending is not None:
                self._octree_pending = None
                self.renderer.octree = pending
                self.renderer._spatially_ordered = False
                self.renderer.culling_enabled = True
                self.renderer.update_pc_features()
                self._culling_enabled_flag = True

            ply_pending = self._ply_pending
            if ply_pending is not None:
                self._ply_pending = None
                model_cpu, level = ply_pending
                self.renderer.gaussian_model = _model_to_cuda(model_cpu)
                self.renderer._spatially_ordered = False
                self.renderer.culling_enabled = (
                    self.renderer.octree is not None and not self._no_culling_flag
                )
                self.renderer.update_pc_features()
                self._ply_sh_degree    = self.renderer.gaussian_model.active_sh_degree
                self._ply_loaded_level = level

            W = max(self._render_w, 2)
            H = max(self._render_h, 2)
            valid_range = (self.crop_x, self.crop_y, self.crop_z) if self.crop_enabled else None

            t0 = time.perf_counter()
            try:
                cam = self._build_camera(W, H, self.camera)
                viewing_b = self._dual_camera_debug and self._viewing_camera_b
                cam_b = self._build_camera(W, H, self.camera_b) if self._dual_camera_debug else None
                frustum_overlay = None
                if viewing_b:
                    # The library's octree cull has no true far-plane clip
                    # (see CLAUDE.md: "Far plane deliberately omitted"), so
                    # there's no single "correct" far extent to draw here --
                    # it's a pure visualization choice. Scale it to camera
                    # A's current orbit distance (a proxy for the scale of
                    # whatever's near its look_at point) instead of a fixed
                    # far-away constant, so the wireframe stays proportionate
                    # to the scene rather than dwarfing it.
                    debug_zfar = max(self.camera.distance * 3.0, _DEBUG_FRUSTUM_ZNEAR * 10)
                    corners_world = frustum_corners_world(
                        self.camera, self.fov_deg, W / H,
                        _DEBUG_FRUSTUM_ZNEAR, debug_zfar,
                    )
                    frustum_overlay = project_world_points(corners_world, cam_b)
                with torch.no_grad():
                    image = self.renderer.get_outputs(
                        cam,
                        render_camera      = cam_b if viewing_b else None,
                        valid_range       = valid_range,
                        split             = self.split_enabled,
                        slider            = self.split_pos,
                        active_sh_degree  = min(self.sh_degree, self._ply_sh_degree),
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
                                    fmt_splats(n_splats), frustum_overlay)

    # ── Frame polling ──────────────────────────────────────────────────────────

    def _poll_frame(self):
        if self._culling_enabled_flag:
            self._culling_enabled_flag = False
            self._sidebar.on_culling_ready(len(self.renderer.octree.node_aabbs))
            self.render_widget._no_culling_warning = False
            self.render_widget.update()
        if self._build_error_flag:
            self._build_error_flag = False
            self._sidebar.on_build_error()
        lvl = self._ply_loaded_level
        if lvl is not None:
            self._ply_loaded_level = None
            self.sh_degree = self._ply_sh_degree
            self._sidebar.on_ply_loaded(lvl, self._ply_sh_degree)
        if self._compress_error_flag:
            self._compress_error_flag = False
            self._sidebar.on_compress_error(self._current_compression)

        with self._frame_lock:
            slot, self._frame_slot = self._frame_slot, None
        if slot is None:
            return
        img_np, fps_str, gpu_str, fps_val, n_splats, splat_str, frustum_overlay = slot
        self.render_widget.set_frame(img_np)
        if self._dual_camera_debug:
            self.render_widget.set_frustum_overlay(frustum_overlay)
        if fps_val > 0:
            self.render_widget.update_fps(fps_val)
        if n_splats > 0:
            self.render_widget.update_splat_count(n_splats)
        self._sidebar.update_stats(fps_str, splat_str, gpu_str)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def closeEvent(self, e):
        self._running = False
        self._render_trig.set()
        _up_key = next((k for k, v in UP_AXIS_VECTORS.items()
                        if np.allclose(v, self.camera.world_up)), "+Z")
        ui = self._sidebar.ui_state_for_config()
        self._cfg.update({
            "fov_deg":            self.fov_deg,
            "move_speed":         self._move_speed,
            "orbit_speed":        self._orbit_speed,
            "mouse_inv_x":        self.render_widget.mouse_inv_x,
            "mouse_inv_y":        self.render_widget.mouse_inv_y,
            "kb_inv_x":           self._kb_inv_x,
            "kb_inv_y":           self._kb_inv_y,
            "world_up":           _up_key,
            "render_w":           self._render_w,
            "render_h":           self._render_h,
            "ar_preset":          ui["ar_preset"],
            "lock_ar":            self._lock_ar,
            "depth_ratio":        self.depth_ratio,
            "active_sh_degree":   self.sh_degree,
            "opacity_thresh":     self.opacity_thresh,
            "sparsity":           self.sparsity,
            "scaling_mod":        self.scaling_mod,
            "point_size":         self.point_size,
            "render_type":        self.render_type,
            "show_fps_overlay":   ui["show_fps_overlay"],
            "show_splat_overlay": ui["show_splat_overlay"],
            "profiling_enabled":  self.renderer.profiling_enabled,
            "verbosity":          gs2d.get_verbosity(),
        })
        save_config(self._cfg)
        save_model_config(self.ply_path, {
            **self._cfg,
            "camera_look_at":   self.camera.look_at.tolist(),
            "camera_distance":  float(self.camera.distance),
            "camera_yaw":       float(self.camera.yaw),
            "camera_pitch":     float(self.camera.pitch),
            "compression":      self._current_compression,
            "render_type1":     self.render_type1,
            "render_type2":     self.render_type2,
            "split_enabled":    self.split_enabled,
            "split_pos":        float(self.split_pos),
            "show_ptc":         self.show_ptc,
            "surfel_disk":      self.surfel_disk,
            "crop_enabled":     self.crop_enabled,
            "crop_x":           list(self.crop_x),
            "crop_y":           list(self.crop_y),
            "crop_z":           list(self.crop_z),
        })
        super().closeEvent(e)


# ── Entry point ────────────────────────────────────────────────────────────────

def run_local_viewer(ply_path: str, args) -> None:
    """Launch the Qt GUI. Called from main.py or directly."""
    app = QApplication.instance() or QApplication(sys.argv)
    QLocale.setDefault(QLocale(QLocale.C))
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(os.path.join(_ROOT, "res", "kestrel_icon.png")))
    viewer = LocalViewer(ply_path, args)
    viewer.show()
    sys.exit(app.exec_())
