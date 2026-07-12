#!/usr/bin/env python3
"""
build_index.py — Offline octree builder for frustum culling.

Usage:
    python utils/build_index.py <path_to_ply> [--verbosity 0|1|2]

Produces a .idx file (NumPy .npz, via gsplat2d_rendering.save_octree)
alongside the .ply. Octree construction, progress logging, and leaf-count/
depth-histogram reporting are entirely gsplat2d_rendering's own (see
gsplat2d_rendering.set_verbosity) — this script only owns the Kestrel .idx
CLI/path convention.
"""

import argparse
import os

import numpy as np
from plyfile import PlyData

import gsplat2d_rendering as gs2d

MAX_DEPTH       = 7
LEAF_MAX_SPLATS = 170_000   # stop subdividing when node has <= this many splats


def read_xyz(ply_path: str) -> np.ndarray:
    """Return float32 [N, 3] XYZ array from a 2DGS PLY."""
    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]
    x = np.asarray(el["x"], dtype=np.float32)
    y = np.asarray(el["y"], dtype=np.float32)
    z = np.asarray(el["z"], dtype=np.float32)
    return np.stack([x, y, z], axis=1)


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
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=2,
                        help="gsplat2d_rendering log verbosity: 0=silent (errors only), "
                             "1=normal, 2=verbose (default, since this is an offline tool "
                             "you're watching run)")
    args = parser.parse_args()

    gs2d.set_verbosity(args.verbosity)

    idx_path = os.path.splitext(args.ply_path)[0] + ".idx"
    xyz = read_xyz(args.ply_path)
    octree = gs2d.build_octree(xyz, leaf_max=args.leaf_max, max_depth=args.max_depth)

    # Not gs2d.save_octree(idx_path, ...): passing a plain string/Path lets
    # np.savez_compressed silently append ".npz" to it. save_octree passes
    # file-like objects through untouched instead (see its own docstring),
    # so writing through an open handle keeps the literal ".idx" filename.
    with open(idx_path, "wb") as fh:
        gs2d.save_octree(fh, octree)


if __name__ == "__main__":
    main()
