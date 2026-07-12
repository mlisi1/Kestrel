#!/usr/bin/env python3
"""
build_index.py — Offline octree builder for frustum culling.

Usage:
    python build_index.py <path_to_ply>

Produces a .idx file (NumPy .npz, via gsplat2d_rendering.save_octree)
alongside the .ply:
    node_aabbs   float32 [num_nodes, 6]      min_xyz + max_xyz per leaf
    node_offsets int64   [num_nodes + 1]     CSR row pointers into flat_indices
    flat_indices int64   [N]                 splat indices grouped by leaf node

Octree building itself is delegated to gsplat2d_rendering.build_octree —
this script only owns the Kestrel .idx CLI/path convention.
"""

import argparse
import os
import time

import numpy as np
from plyfile import PlyData

import gsplat2d_rendering as gs2d

MAX_DEPTH       = 7
LEAF_MAX_SPLATS = 170_000   # stop subdividing when node has <= this many splats


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def read_xyz(ply_path: str) -> np.ndarray:
    """Return float32 [N, 3] XYZ array from a 2DGS PLY."""
    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]
    x = np.asarray(el["x"], dtype=np.float32)
    y = np.asarray(el["y"], dtype=np.float32)
    z = np.asarray(el["z"], dtype=np.float32)
    return np.stack([x, y, z], axis=1)


def build_octree(xyz: np.ndarray, max_depth: int = MAX_DEPTH, leaf_max: int = LEAF_MAX_SPLATS):
    """Thin wrapper over gsplat2d_rendering.build_octree, kept for backward
    compatibility with callers (viewer/app.py) that unpack the three arrays
    directly rather than an Octree instance."""
    octree = gs2d.build_octree(xyz, leaf_max=leaf_max, max_depth=max_depth)
    return octree.node_aabbs, octree.node_offsets, octree.flat_indices


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build an octree frustum-culling index alongside a 2DGS .ply file."
    )
    parser.add_argument("ply_path", help="Path to .ply file")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                        help=f"Max octree depth (default {MAX_DEPTH})")
    parser.add_argument("--leaf-max", type=int, default=LEAF_MAX_SPLATS,
                        help=f"Stop splitting when node has <= this many splats "
                             f"(default {LEAF_MAX_SPLATS:,})")
    args = parser.parse_args()

    ply_path = args.ply_path
    idx_path = os.path.splitext(ply_path)[0] + ".idx"

    print(f"[build_index] Input:     {ply_path}")
    print(f"[build_index] Output:    {idx_path}")
    print(f"[build_index] max_depth= {args.max_depth}  leaf_max= {args.leaf_max:,}")

    t0 = time.perf_counter()
    print("[build_index] Reading PLY…")
    xyz = read_xyz(ply_path)
    N   = xyz.shape[0]
    print(f"[build_index]   {N:,} splats   ({time.perf_counter()-t0:.2f}s)")

    t1 = time.perf_counter()
    print("[build_index] Building octree…")
    octree = gs2d.build_octree(xyz, leaf_max=args.leaf_max, max_depth=args.max_depth)
    print(f"[build_index]   Done in {time.perf_counter()-t1:.2f}s")

    assert octree.flat_indices.shape[0] == N, (
        f"BUG: index covers {octree.flat_indices.shape[0]:,} splats but PLY has {N:,}"
    )
    sorted_check = np.sort(octree.flat_indices)
    assert (sorted_check == np.arange(N, dtype=octree.flat_indices.dtype)).all(), \
        "BUG: flat_indices is not a permutation of [0, N)"

    L      = len(octree.node_aabbs)
    counts = np.diff(octree.node_offsets)
    print(f"[build_index]   Leaf nodes : {L:,}")
    print(f"[build_index]   Splats/leaf: avg={counts.mean():.0f}  "
          f"min={counts.min():,}  max={counts.max():,}")

    edge      = (octree.node_aabbs[:, 3:] - octree.node_aabbs[:, :3]).max(axis=1)
    root_edge = float(edge.max())
    if root_edge > 0:
        depths = np.round(np.log2(root_edge / np.maximum(edge, 1e-12))).astype(int)
        for d in sorted(set(depths.tolist())):
            cnt = int((depths == d).sum())
            print(f"[build_index]     depth {d}: {cnt:,} leaf nodes")

    t2 = time.perf_counter()
    # Not gs2d.save_octree(idx_path, ...): it always str()s its path arg
    # before handing it to np.savez_compressed, which silently appends
    # ".npz" to any string path — writing through an open file handle keeps
    # the literal ".idx" filename the rest of Kestrel (and existing caches)
    # expect. See docs/gsplat2d-rendering-gap.md.
    with open(idx_path, "wb") as fh:
        np.savez_compressed(fh, node_aabbs=octree.node_aabbs,
                            node_offsets=octree.node_offsets,
                            flat_indices=octree.flat_indices)
    sz_mb = os.path.getsize(idx_path) / 1024 / 1024
    print(f"[build_index]   Saved {idx_path}  ({sz_mb:.1f} MB, "
          f"{time.perf_counter()-t2:.2f}s)")
    print(f"[build_index] Total: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
