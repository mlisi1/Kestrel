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
(renderer/chunk_manager.py, via viewer/ply_loader.py's path helpers) looks
for the manifest there, the same place the app's own --build-index flag
already writes the fine-grained index.

Chunk partitioning itself is gsplat2d_rendering.build_octree, called with a
coarse leaf_max (target splats/chunk); this script only owns the Kestrel
.kestrel/ path convention and the write-back-reordered-PLY step.
"""

import argparse
import os

import torch

import gsplat2d_rendering as gs2d
from utils.build_index import read_xyz
from viewer.ply_loader import _chunk_manifest_path, _chunked_ply_path

CHUNK_MAX_DEPTH  = 6
CHUNK_TARGET_SIZE = 500_000   # target splats per chunk


def build_chunks(ply_path: str, chunk_size: int = CHUNK_TARGET_SIZE,
                  max_depth: int = CHUNK_MAX_DEPTH) -> None:
    manifest_path = _chunk_manifest_path(ply_path)
    chunked_path  = _chunked_ply_path(ply_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    xyz = read_xyz(ply_path)
    chunk_octree = gs2d.build_octree(xyz, leaf_max=chunk_size, max_depth=max_depth)

    model = gs2d.load_gaussian_model(ply_path, device="cpu")
    model.reorder_(torch.from_numpy(chunk_octree.flat_indices.astype("int64")))
    gs2d.write_gaussian_model(chunked_path, model)

    # Not gs2d.save_octree(manifest_path, ...): see utils/build_index.py's
    # identical comment -- a plain string path lets np.savez_compressed
    # silently append ".npz" to it.
    with open(manifest_path, "wb") as fh:
        gs2d.save_octree(fh, chunk_octree)


def main():
    parser = argparse.ArgumentParser(
        description="Build a chunk manifest + chunk-reordered PLY for load-time chunk streaming."
    )
    parser.add_argument("ply_path", help="Path to .ply file")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_TARGET_SIZE,
                        help=f"Target splats per chunk (default {CHUNK_TARGET_SIZE:,})")
    parser.add_argument("--max-depth", type=int, default=CHUNK_MAX_DEPTH,
                        help=f"Max octree depth when partitioning into chunks (default {CHUNK_MAX_DEPTH})")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=2,
                        help="gsplat2d_rendering log verbosity: 0=silent (errors only), "
                             "1=normal, 2=verbose (default, since this is an offline tool "
                             "you're watching run)")
    args = parser.parse_args()

    gs2d.set_verbosity(args.verbosity)
    build_chunks(args.ply_path, chunk_size=args.chunk_size, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
