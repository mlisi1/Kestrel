"""
Kestrel — 2DGS Native Viewer
Usage: python main.py <path/to/scene.ply> [options]
       python main.py <path/to/model_dir/> [--iterations N] [options]
"""

import argparse
import os
import sys


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kestrel — 2DGS Native Viewer")
    p.add_argument("ply_path", type=str,
                   help="Path to .ply file, or model directory "
                        "(resolves point_cloud/iteration_N/point_cloud.ply)")
    p.add_argument("--sh_degree", "--sh-degree", type=int, default=-1,
                   help="SH degree (-1 = auto-detect from PLY header)")
    p.add_argument("--no-culling",   action="store_true", default=False,
                   help="Disable frustum culling even when an octree index exists")
    p.add_argument("--no-profiling", action="store_true", default=False,
                   help="Disable per-frame GPU timing output at startup "
                        "(also toggleable live from the sidebar's Status panel)")
    p.add_argument("--verbosity",    type=int, choices=[0, 1, 2], default=None,
                   help="gsplat2d_rendering log level: 0=silent (errors only), "
                        "1=normal, 2=verbose. Omit to use the saved default "
                        "(also toggleable live from the sidebar's Status panel)")
    p.add_argument("--build-index",  action="store_true", default=False,
                   help="Build (or rebuild) the octree frustum-culling index")
    p.add_argument("--leaf-max",     type=int, default=5000,
                   help="Max splats per octree leaf node (default 5000)")
    p.add_argument("--fp16-load",    action="store_true", default=False,
                   help="Transfer tensors via fp16 during PLY load")
    p.add_argument("--iterations",   type=int, default=30000,
                   help="Iteration number when resolving model directory path")
    p.add_argument("--debug-dual-camera", action="store_true", default=False,
                   help="Debug mode: a second, mouse-orbit-controlled camera "
                        "you can Tab into, for auditing what the keyboard-"
                        "controlled render camera's frustum selects")
    return p


def _resolve_ply(ply_path: str, iterations: int) -> str:
    if ply_path.lower().endswith(".ply"):
        if not os.path.exists(ply_path):
            sys.exit(f"[ERROR] PLY not found: {ply_path}")
        return ply_path
    candidate = os.path.join(ply_path, "point_cloud",
                             f"iteration_{iterations}", "point_cloud.ply")
    if not os.path.exists(candidate):
        sys.exit(f"[ERROR] PLY not found: {candidate}\n"
                 f"       (use --iterations N to specify iteration, default is {iterations})")
    return candidate


if __name__ == "__main__":
    args = _make_parser().parse_args()
    ply_path = _resolve_ply(args.ply_path, args.iterations)

    from viewer import run_local_viewer
    run_local_viewer(ply_path, args)
