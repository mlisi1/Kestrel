"""PLY and index file I/O helpers for the Kestrel viewer.

PLY parsing, SH-reshape, fp16-safety clipping, and octree building all now
live in gsplat2d_rendering — this module only keeps Kestrel-specific
concerns: the .kestrel/ artifact path convention, per-model persistent
config, and the CPU-parse / CUDA-install thread split the background
compression worker relies on (see viewer/app.py).
"""

from __future__ import annotations

import dataclasses
import json
import os

import gsplat2d_rendering as gs2d


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

def _load_octree(ply_path: str):
    """Returns a gsplat2d_rendering.culling.Octree, or None if no cached index.
    load_octree/SplatRenderer report the leaf count and culling status
    themselves (gsplat2d_rendering's own logging), so this only applies
    Kestrel's .kestrel/ path convention."""
    idx = _idx_path(ply_path)
    if not os.path.exists(idx):
        return None
    return gs2d.load_octree(idx)


# ── PLY / GaussianModel loading ────────────────────────────────────────────────

def _load_gaussian_model(path: str, sh_degree: int = -1, device: str = "cuda"):
    """Thin wrapper over gsplat2d_rendering.load_gaussian_model — Kestrel's own
    compression flow bakes fp16/SH-truncation into cached .kestrel/*_L{n}.ply
    files up front (see utils/compress.py), so this is always called with
    compression_level=0 (whatever the source file already contains)."""
    return gs2d.load_gaussian_model(path, sh_degree=sh_degree, device=device)


def _model_to_cuda(model):
    """Moves a CPU-loaded GaussianModel's tensors to CUDA, returning a new
    instance. Used to keep PLY parsing (background thread) separate from CUDA
    tensor creation (render thread) — see viewer/app.py's _compression_worker
    / _render_loop split."""
    return dataclasses.replace(
        model,
        xyz=model.xyz.cuda(),
        raw_opacity=model.raw_opacity.cuda(),
        raw_scaling=model.raw_scaling.cuda(),
        raw_rotation=model.raw_rotation.cuda(),
        features_dc=model.features_dc.cuda(),
        features_rest=model.features_rest.cuda(),
    )


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
