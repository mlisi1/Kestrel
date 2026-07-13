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


def _chunk_manifest_path(ply_path: str) -> str:
    """<iteration_dir>/.kestrel/<ply_stem>_chunks.idx -- the coarse,
    disk-chunk-granularity Octree (see utils/build_chunks.py), named
    distinctly from _idx_path's fine-grained per-frame culling index so the
    two don't collide."""
    d = os.path.dirname(os.path.abspath(ply_path))
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    return os.path.join(d, ".kestrel", stem + "_chunks.idx")


def _chunked_ply_path(ply_path: str) -> str:
    """<iteration_dir>/.kestrel/<ply_stem>_chunked.ply -- the whole model,
    physically reordered into chunk-contiguous order, so a chunk's rows are
    one contiguous byte range for _load_gaussian_model_range."""
    d = os.path.dirname(os.path.abspath(ply_path))
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    return os.path.join(d, ".kestrel", f"{stem}_chunked.ply")


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


def _ply_vertex_count(path: str) -> int:
    """Header-ish vertex count peek: plyfile memory-maps the vertex element
    rather than reading it eagerly (see io.chunked_ply's own docstring), so
    this is cheap even against a huge PLY. Used for chunk-manifest staleness
    checks and OOM-dialog size estimates."""
    from plyfile import PlyData
    return int(PlyData.read(path).elements[0].count)


def _load_chunk_manifest(ply_path: str):
    """Returns the coarse chunk-manifest Octree (see utils/build_chunks.py),
    or None if missing or stale. Stale means the manifest's total point
    count no longer matches the source PLY's -- mirrors
    gsplat2d_rendering.culling.cache.load_or_build_octree's own stale-cache
    guard, just against Kestrel's own .kestrel/ path convention instead of
    that helper's .gsplat2d_rendering/ cache dir."""
    manifest_path = _chunk_manifest_path(ply_path)
    if not os.path.exists(manifest_path):
        return None
    manifest = gs2d.load_octree(manifest_path)
    if int(manifest.node_offsets[-1]) != _ply_vertex_count(ply_path):
        return None
    return manifest


# ── PLY / GaussianModel loading ────────────────────────────────────────────────

def _load_gaussian_model(path: str, sh_degree: int = -1, device: str = "cuda"):
    """Thin wrapper over gsplat2d_rendering.load_gaussian_model — Kestrel's own
    compression flow bakes fp16/SH-truncation into cached .kestrel/*_L{n}.ply
    files up front (see utils/compress.py), so this is always called with
    compression_level=0 (whatever the source file already contains)."""
    return gs2d.load_gaussian_model(path, sh_degree=sh_degree, device=device)


def _load_gaussian_model_range(path: str, row_start: int, row_count: int, device: str = "cpu"):
    """Thin wrapper over gsplat2d_rendering.load_gaussian_model_range, same
    compression_level=0 convention as _load_gaussian_model (a chunked PLY
    already has whatever compression the source PLY had baked in)."""
    return gs2d.load_gaussian_model_range(path, row_start, row_count, device=device)


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
