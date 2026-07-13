#!/usr/bin/env python3
"""
build_chunks.py — Offline chunk-manifest + chunk-reordered PLY builder for
load-time chunk streaming.

Usage:
    python utils/build_chunks.py <path_to_ply> [--chunk-size 500000] [--max-depth 6] [--verbosity 2]

Produces, alongside the .ply, under .kestrel/:
    <stem>_chunks.idx    (npz, via gsplat2d_rendering.save_octree — same Octree
                          shape as the fine-grained culling index, just built
                          at a coarser leaf_max meant for disk chunks rather
                          than per-frame GPU culling)
    <stem>_chunked.ply   (the whole model, physically reordered into
                          chunk-contiguous order via gsplat2d_rendering.
                          write_gaussian_model — so a chunk's rows are one
                          contiguous byte range for viewer/ply_loader.py's
                          runtime ranged reads)

Unlike utils/build_index.py (which writes its .idx beside the source .ply),
this always writes under .kestrel/ — chunk streaming's runtime path
(gsplat2d_rendering.streaming.ChunkManager, via viewer/ply_loader.py's path
helpers) looks for the manifest there, the same place the app's own
--build-index flag already writes the fine-grained index.

Chunk partitioning itself is gsplat2d_rendering.build_octree, called with a
coarse leaf_max (target splats/chunk); this script only owns the Kestrel
.kestrel/ path convention and the write-back-reordered-PLY step.
"""

import argparse
import json
import os

import torch

import gsplat2d_rendering as gs2d
from viewer.ply_loader import _chunk_manifest_path, _chunk_meta_path, _chunked_ply_path, _ply_vertex_count

CHUNK_MAX_DEPTH  = 6
CHUNK_TARGET_SIZE = 500_000   # target splats per chunk


def build_chunks(ply_path: str, chunk_size: int = CHUNK_TARGET_SIZE,
                  max_depth: int = CHUNK_MAX_DEPTH,
                  opacity_threshold: float = 0.0) -> None:
    """opacity_threshold (default 0.0, off) permanently drops splats at/under
    this activated opacity before chunking -- same semantics as
    load_gaussian_model's own parameter (see its docstring: real floater
    artifacts in trained models can sit at mean opacity ~0.003-0.004,
    correctly isolated by the adaptive split into their own chunk since they
    don't spatially cluster with real geometry -- a chunk that's nonempty by
    point count but renders as visually nothing). Must happen here, before
    build_octree ever runs, not as a per-read parameter at runtime: pruning
    changes which rows exist at all, so the chunk octree's indices would
    desync from what's actually written if it ran any later -- same
    reasoning as ChunkedPlyReader/load_gaussian_model_range's own opacity
    parameter being build-time-only, not exposed per range read."""
    manifest_path = _chunk_manifest_path(ply_path)
    chunked_path  = _chunked_ply_path(ply_path)
    meta_path     = _chunk_meta_path(ply_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    # Recorded before pruning: the *source* PLY's own row count, independent
    # of whatever opacity_threshold does to what's actually written below --
    # see _load_chunk_manifest's own docstring for why viewer/ply_loader.py's
    # staleness check needs this instead of comparing the (possibly now much
    # smaller, pruned) manifest's point count directly against the source.
    source_point_count = _ply_vertex_count(ply_path)

    # xyz is derived from the (possibly pruned) loaded model rather than a
    # separate read of the raw file, so the octree's point count/indices
    # always match what write_gaussian_model actually writes below.
    model = gs2d.load_gaussian_model(ply_path, device="cpu", opacity_threshold=opacity_threshold)
    xyz = model.xyz.float().cpu().numpy()
    chunk_octree = gs2d.build_octree(xyz, leaf_max=chunk_size, max_depth=max_depth)

    model.reorder_(torch.from_numpy(chunk_octree.flat_indices.astype("int64")))
    gs2d.write_gaussian_model(chunked_path, model)

    # Not gs2d.save_octree(manifest_path, ...): see utils/build_index.py's
    # identical comment -- a plain string path lets np.savez_compressed
    # silently append ".npz" to it.
    with open(manifest_path, "wb") as fh:
        gs2d.save_octree(fh, chunk_octree)

    with open(meta_path, "w") as fh:
        json.dump({"source_point_count": source_point_count, "opacity_threshold": opacity_threshold}, fh)


def main():
    parser = argparse.ArgumentParser(
        description="Build a chunk manifest + chunk-reordered PLY for load-time chunk streaming."
    )
    parser.add_argument("ply_path", help="Path to .ply file")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_TARGET_SIZE,
                        help=f"Target splats per chunk (default {CHUNK_TARGET_SIZE:,})")
    parser.add_argument("--max-depth", type=int, default=CHUNK_MAX_DEPTH,
                        help=f"Max octree depth when partitioning into chunks (default {CHUNK_MAX_DEPTH})")
    parser.add_argument("--opacity-threshold", type=float, default=0.0,
                        help="Permanently drop splats at/under this activated opacity before "
                             "chunking (default 0.0, off). Eliminates near-invisible floater "
                             "artifacts that the adaptive split would otherwise isolate into "
                             "their own chunk (nonempty by point count, empty-looking on screen).")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=2,
                        help="gsplat2d_rendering log verbosity: 0=silent (errors only), "
                             "1=normal, 2=verbose (default, since this is an offline tool "
                             "you're watching run)")
    args = parser.parse_args()

    gs2d.set_verbosity(args.verbosity)
    build_chunks(args.ply_path, chunk_size=args.chunk_size, max_depth=args.max_depth,
                 opacity_threshold=args.opacity_threshold)


if __name__ == "__main__":
    main()
