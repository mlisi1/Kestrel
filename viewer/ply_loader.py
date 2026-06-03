"""PLY and index file I/O helpers for the Kestrel viewer.

All functions are pure Python/numpy/torch — no Qt dependency.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


# ── SH detection ──────────────────────────────────────────────────────────────

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


# ── Path helpers ──────────────────────────────────────────────────────────────

def _idx_path(ply_path: str) -> str:
    """<iteration_dir>/.kestrel/<ply_stem>.idx"""
    d = os.path.dirname(os.path.abspath(ply_path))
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    return os.path.join(d, ".kestrel", stem + ".idx")


def _compressed_ply_path(ply_path: str, level: int) -> str:
    """<iteration_dir>/.kestrel/<ply_stem>_L{level}.ply"""
    d = os.path.dirname(os.path.abspath(ply_path))
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    return os.path.join(d, ".kestrel", f"{stem}_L{level}.ply")


# ── Octree index ──────────────────────────────────────────────────────────────

def _load_octree(ply_path: str) -> dict | None:
    idx = _idx_path(ply_path)
    if not os.path.exists(idx):
        print(f"[viewer] No octree index found at {idx} — frustum culling disabled")
        return None
    data = np.load(idx)
    L = len(data["node_aabbs"])
    print(f"[viewer] Loaded octree index: {L:,} leaf nodes from {idx}")
    return {"node_aabbs":   data["node_aabbs"],
            "node_offsets": data["node_offsets"],
            "flat_indices": data["flat_indices"]}


# ── PLY loading ───────────────────────────────────────────────────────────────

def _ply_safety_helpers(el):
    """Return (_sorted, _stack, _safe) closures over a PlyElement."""
    FP16_MAX = 65504.0

    def _sorted(prefix):
        names = [p.name for p in el.properties if p.name.startswith(prefix)]
        return sorted(names, key=lambda x: int(x.split("_")[-1]))

    def _stack(names):
        return np.stack([np.asarray(el[n]) for n in names], axis=1).astype(np.float32)

    def _safe(arr, label=""):
        n_bad = int(np.sum(~np.isfinite(arr)))
        if n_bad > 0:
            print(f"[viewer]   {label}: zeroing {n_bad:,} NaN/Inf values")
            arr = np.where(np.isfinite(arr), arr, 0.0)
        if np.abs(arr).max() > FP16_MAX:
            thr = float(min(np.percentile(np.abs(arr), 99.9), FP16_MAX))
            n_c = int(np.sum(np.abs(arr) > thr))
            print(f"[viewer]   {label}: clipping {n_c:,} overflow values to ±{thr:.1f}")
            arr = np.clip(arr, -thr, thr)
        return arr

    return _sorted, _stack, _safe


def _load_ply_into_model(model, path: str) -> None:
    """Read a 2DGS PLY and install tensors directly as CUDA tensors (startup path)."""
    from plyfile import PlyData
    el = PlyData.read(path).elements[0]
    _sorted, _stack, _safe = _ply_safety_helpers(el)

    xyz         = _safe(_stack(["x", "y", "z"]), "xyz")
    opacities   = _safe(np.asarray(el["opacity"], dtype=np.float32)[..., None], "opacity")
    scales      = _safe(_stack(_sorted("scale_")), "scale")
    rotations   = _safe(_stack(_sorted("rot_")), "rotation")
    features_dc = _safe(_stack(_sorted("f_dc_")), "f_dc")[:, np.newaxis, :]

    f_rest_names = _sorted("f_rest_")
    if f_rest_names:
        f_rest_flat = _safe(_stack(f_rest_names), "f_rest")
        n_coeffs = f_rest_flat.shape[1]
        file_sh  = next((d for d in range(1, 5)
                         if 3 * ((d + 1) ** 2 - 1) == n_coeffs), 0)
        K = (file_sh + 1) ** 2 - 1
        features_rest = f_rest_flat.reshape((-1, 3, K)).transpose(0, 2, 1)
        print(f"[viewer]   SH degree: {file_sh} ({n_coeffs} f_rest_ props)")
    else:
        features_rest = np.zeros((xyz.shape[0], 0, 3), dtype=np.float32)
        file_sh = 0
        print("[viewer]   No f_rest_ — degree-0 (view-independent colour)")

    def _cuda(a):
        return torch.from_numpy(a.astype(np.float32)).cuda()

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


def _read_ply_numpy(path: str) -> dict:
    """Read a 2DGS PLY into numpy float32 arrays (no CUDA — background-thread safe)."""
    from plyfile import PlyData
    el = PlyData.read(path).elements[0]
    _sorted, _stack, _safe = _ply_safety_helpers(el)

    xyz         = _safe(_stack(["x", "y", "z"]), "xyz")
    opacities   = _safe(np.asarray(el["opacity"], dtype=np.float32)[..., None], "opacity")
    scales      = _safe(_stack(_sorted("scale_")), "scale")
    rotations   = _safe(_stack(_sorted("rot_")), "rotation")
    features_dc = _safe(_stack(_sorted("f_dc_")), "f_dc")[:, np.newaxis, :]

    f_rest_names = _sorted("f_rest_")
    if f_rest_names:
        f_rest_flat = _safe(_stack(f_rest_names), "f_rest")
        n_coeffs = f_rest_flat.shape[1]
        file_sh  = next((d for d in range(1, 5)
                         if 3 * ((d + 1) ** 2 - 1) == n_coeffs), 0)
        K = (file_sh + 1) ** 2 - 1
        features_rest = f_rest_flat.reshape((-1, 3, K)).transpose(0, 2, 1)
    else:
        features_rest = np.zeros((xyz.shape[0], 0, 3), dtype=np.float32)
        file_sh = 0

    print(f"[viewer]   {xyz.shape[0]:,} splats  SH={file_sh}  ← {os.path.basename(path)}")
    return {
        "xyz": xyz, "opacity": opacities, "features_dc": features_dc,
        "features_rest": features_rest, "scaling": scales, "rotation": rotations,
        "active_sh_degree": file_sh,
    }


# ── Per-model persistent config ───────────────────────────────────────────────

def _model_config_path(ply_path: str) -> str:
    """<iteration_dir>/.kestrel/<ply_stem>_view.json"""
    d = os.path.dirname(os.path.abspath(ply_path))
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    return os.path.join(d, ".kestrel", f"{stem}_view.json")


def load_model_config(ply_path: str) -> dict:
    path = _model_config_path(ply_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_model_config(ply_path: str, data: dict) -> None:
    path = _model_config_path(ply_path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass


def _install_numpy_into_model(model, arrays: dict) -> None:
    """Install pre-read numpy arrays as CUDA tensors (render-thread path)."""
    def _cuda(a):
        return torch.from_numpy(a.astype(np.float32)).cuda()

    model._xyz           = _cuda(arrays["xyz"])
    model._opacity       = _cuda(arrays["opacity"])
    model._features_dc   = _cuda(arrays["features_dc"])
    model._features_rest = _cuda(arrays["features_rest"])
    model._scaling       = _cuda(arrays["scaling"])
    model._rotation      = _cuda(arrays["rotation"])

    sh = arrays["active_sh_degree"]
    if hasattr(model, "active_sh_degree"):
        model.active_sh_degree = sh
    if hasattr(model, "max_sh_degree"):
        model.max_sh_degree = min(model.max_sh_degree, sh)

    vram_mb = sum(
        t.element_size() * t.nelement()
        for t in [model._xyz, model._opacity, model._features_dc,
                  model._features_rest, model._scaling, model._rotation]
    ) / 1024 / 1024
    print(f"[viewer]   Installed {arrays['xyz'].shape[0]:,} splats — {vram_mb:.0f} MB VRAM")
