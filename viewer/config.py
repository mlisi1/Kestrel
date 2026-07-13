"""Constants, persistent config, and small display helpers."""

import json
import math
import pathlib

import numpy as np


# ── World-up axis ──────────────────────────────────────────────────────────────

UP_AXIS_OPTIONS = ["+Z", "+Y", "+X", "-Z", "-Y", "-X"]
UP_AXIS_VECTORS: dict[str, np.ndarray] = {
    "+Z": np.array([ 0.,  0.,  1.], dtype=np.float64),
    "+Y": np.array([ 0.,  1.,  0.], dtype=np.float64),
    "+X": np.array([ 1.,  0.,  0.], dtype=np.float64),
    "-Z": np.array([ 0.,  0., -1.], dtype=np.float64),
    "-Y": np.array([ 0., -1.,  0.], dtype=np.float64),
    "-X": np.array([-1.,  0.,  0.], dtype=np.float64),
}

# ── Aspect ratio presets ───────────────────────────────────────────────────────

AR_RATIO_OPTIONS = ["16:9", "4:3", "16:10", "21:9", "3:2", "1:1", "9:16", "3:4", "Custom"]
AR_RATIO_VALUES: dict[str, float | None] = {
    "16:9":  16/9, "4:3":  4/3,  "16:10": 16/10, "21:9": 21/9,
    "3:2":   3/2,  "1:1":  1.0,  "9:16":  9/16,  "3:4":  3/4,
    "Custom": None,
}

# ── Resolution presets ────────────────────────────────────────────────────────

RESOLUTION_PRESETS: dict[str, tuple[int, int] | None] = {
    "Custom":       None,
    "360p":         (640,  360),
    "480p":         (854,  480),
    "720p (HD)":    (1280, 720),
    "1080p (FHD)":  (1920, 1080),
    "1440p (QHD)":  (2560, 1440),
    "4K (UHD)":     (3840, 2160),
}

# ── Render type names ─────────────────────────────────────────────────────────

RENDER_TYPES = [
    "RGB", "Edge", "Alpha", "Normal", "View-Normal",
    "Depth", "Depth-Distort", "Depth-to-Normal", "Depth-to-Curvature",
]
COMPRESSION_LABELS = [
    "L0 — original",
    "L1 — fp16",
    "L2 — fp16 + SH1",
    "L3 — fp16 + int8",
]

# gsplat2d_rendering log verbosity — see gsplat2d_rendering.set_verbosity.
VERBOSITY_LABELS = ["0 — Silent", "1 — Normal", "2 — Verbose"]

RENDER_TYPE_MAP = {
    "RGB":                "render",
    "Edge":               "edge",
    "Alpha":              "rend_alpha",
    "Normal":             "rend_normal",
    "View-Normal":        "view_normal",
    "Depth":              "surf_depth",
    "Depth-Distort":      "rend_dist",
    "Depth-to-Normal":    "surf_normal",
    "Depth-to-Curvature": "curvature",
}

# ── Default render resolution ─────────────────────────────────────────────────

_DEFAULT_W, _DEFAULT_H = 1280, 720

# ── Persistent config ─────────────────────────────────────────────────────────

CONFIG_PATH = pathlib.Path.home() / ".config" / "kestrel" / "config.json"

CONFIG_DEFAULTS: dict = {
    "fov_deg":            60.0,
    "move_speed":         1.0,
    "orbit_speed":        1.0,
    "mouse_inv_x":        True,
    "mouse_inv_y":        False,
    "kb_inv_x":           True,
    "kb_inv_y":           True,
    "world_up":           "+Z",
    "render_w":           _DEFAULT_W,
    "render_h":           _DEFAULT_H,
    "ar_preset":          "16:9",
    "lock_ar":            True,
    "depth_ratio":        0.0,
    "active_sh_degree":   -1,      # -1 = use PLY file's max
    "opacity_thresh":     0.05,
    "sparsity":           1,
    "scaling_mod":        1.0,
    "point_size":         0.01,
    "render_type":        "RGB",
    "show_fps_overlay":   True,
    "show_splat_overlay": True,
    "profiling_enabled":  True,
    "verbosity":          1,       # gsplat2d_rendering log level: 0=silent, 1=normal, 2=verbose
    "chunk_streaming_enabled": False,   # off by default -- see renderer/chunk_manager.py
    "chunk_target_size":       500_000, # target splats/chunk, same UX as octree "Leaf size"
    # Adjacency-hop-count residency (renderer/chunk_manager.py's K-NN chunk
    # graph + BFS), replacing an earlier world-space-distance design. All
    # three are integers; 0 means "off" for either margin, no separate
    # enable/disable flag needed (same idiom as chunk_size=0 elsewhere).
    "chunk_vram_margin_hops": 0, # adjacency hops beyond the strict frustum that are ALSO
                                 # promoted to actual VRAM residency (cheap per-frame GPU
                                 # culling then hides/shows them, no rebuild needed)
    "chunk_ram_margin_hops":  0, # adjacency hops beyond the VRAM tier that are CPU-only
                                 # prefetched (was "chunk_margin", a world-space float)
    "chunk_max_load_hops":    3, # overall residency reach bound: no chunk more than this
                                 # many adjacency-hops from the camera's own nearest chunk
                                 # is ever considered, regardless of the frustum test result
                                 # (was "chunk_max_load_distance_mult", a camera.distance
                                 # multiplier) -- sidebar-tunable (Chunk Streaming panel)
    "chunk_prune_opacity_threshold": 0.0, # permanently drops splats at/under this opacity at the
                                          # *next* chunk manifest rebuild (utils/build_chunks.py's
                                          # opacity_threshold) -- separate from the live, per-frame
                                          # "Opacity threshold" render slider (opacity_thresh)
                                          # above, which never touches the chunk manifest at all
}


def load_config() -> dict:
    cfg = CONFIG_DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open() as fh:
                stored = json.load(fh)
            cfg.update({k: v for k, v in stored.items() if k in CONFIG_DEFAULTS})
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w") as fh:
            json.dump(cfg, fh, indent=2)
    except Exception:
        pass


def fmt_splats(n: int) -> str:
    """Format a splat count as e.g. '1.2M' or '500K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)
