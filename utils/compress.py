#!/usr/bin/env python3
"""
compress_ply.py — Offline quantization for 2D Gaussian Splatting .ply files.

Three compression levels:
  Level 1 (conservative): fp16 round-trip, full SH degree kept.
  Level 2 (balanced):     fp16 + SH degree clamped to 1.   ~2.4x smaller.
  Level 3 (aggressive):   fp16 + drop all SH rest + int8 rot/normals. ~3.6x smaller.

Usage:
    python compress_ply.py input.ply --scan
    python compress_ply.py input.ply output.ply --level 2 --clean --stats
"""

import argparse
import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement

from gsplat2d_rendering.compression import to_fp16_safe

FP16_MAX = np.float32(65504.0)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_ply(path: str) -> PlyData:
    print(f"  Loading: {path}")
    data = PlyData.read(path)
    print(f"  Splat count: {data.elements[0].count:,}")
    return data

def _file_mb(path: str) -> float:
    return os.path.getsize(path) / 1024 / 1024

def _get_prop_names(el, prefix: str):
    names = [p.name for p in el.properties if p.name.startswith(prefix)]
    return sorted(names, key=lambda x: int(x.split("_")[-1]))

def _stack_props(el, names) -> np.ndarray:
    return np.stack([np.asarray(el[n]) for n in names], axis=1)

def _get_sh_degree(el) -> int:
    n_rest = len(_get_prop_names(el, "f_rest_"))
    if n_rest == 0:
        return 0
    for deg in range(1, 5):
        if 3 * ((deg + 1) ** 2 - 1) == n_rest:
            return deg
    raise ValueError(f"Cannot determine SH degree from {n_rest} f_rest_ properties.")

def _to_f32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32)

# fp16 cast with NaN zeroing + 99.9th-percentile overflow clipping —
# gsplat2d_rendering.compression.to_fp16_safe (same algorithm this file used
# to duplicate; kept as a local alias since every call site here passes a
# `label` positionally).
def _to_f16(arr, label=""): return to_fp16_safe(arr, label)

def _to_i8_norm(arr, lo=-1.0, hi=1.0):
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 254 - 127, -127, 127).astype(np.int8)

def _from_i8_norm(arr, lo=-1.0, hi=1.0):
    return ((arr.astype(np.float32) + 127) / 254) * (hi - lo) + lo


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------

def _build_element(xyz, normals, f_dc, f_rest, opacity, scale, rotation,
                   int8_normals=False, int8_rotation=False):
    attrs, dtypes = [], []
    for i, ax in enumerate(["x","y","z"]):
        attrs.append(xyz[:,i]); dtypes.append((ax,"f4"))
    for i, ax in enumerate(["nx","ny","nz"]):
        a = normals[:,i]
        if int8_normals: a=a.astype(np.int8); dtypes.append((ax,"i1"))
        else:            a=a.astype(np.float32); dtypes.append((ax,"f4"))
        attrs.append(a)
    for i in range(f_dc.shape[1]):
        attrs.append(f_dc[:,i].astype(np.float32)); dtypes.append((f"f_dc_{i}","f4"))
    for i in range(f_rest.shape[1]):
        attrs.append(f_rest[:,i].astype(np.float32)); dtypes.append((f"f_rest_{i}","f4"))
    attrs.append(opacity[:,0].astype(np.float32)); dtypes.append(("opacity","f4"))
    for i in range(scale.shape[1]):
        attrs.append(scale[:,i].astype(np.float32)); dtypes.append((f"scale_{i}","f4"))
    for i in range(rotation.shape[1]):
        a = rotation[:,i]
        if int8_rotation: a=a.astype(np.int8); dtypes.append((f"rot_{i}","i1"))
        else:             a=a.astype(np.float32); dtypes.append((f"rot_{i}","f4"))
        attrs.append(a)
    N = xyz.shape[0]
    el = np.empty(N, dtype=dtypes)
    for (name,_), arr in zip(dtypes, attrs):
        el[name] = arr
    return PlyElement.describe(el, "vertex")


# ---------------------------------------------------------------------------
# Compression levels
# ---------------------------------------------------------------------------

def compress_level1(plydata):
    el = plydata.elements[0]
    sh = _get_sh_degree(el)
    print(f"  Detected SH degree: {sh}  (kept)")
    xyz      = _to_f32(_to_f16(_stack_props(el,["x","y","z"]),      "xyz"))
    normals  = _to_f32(_to_f16(_stack_props(el,["nx","ny","nz"]),   "normals"))
    f_dc     = _to_f32(_to_f16(_stack_props(el,_get_prop_names(el,"f_dc_")), "f_dc"))
    opacity  = _to_f32(_to_f16(_stack_props(el,["opacity"]),        "opacity"))
    scale    = _to_f32(_to_f16(_stack_props(el,_get_prop_names(el,"scale_")), "scale"))
    rotation = _to_f32(_to_f16(_stack_props(el,_get_prop_names(el,"rot_")),   "rotation"))
    rn = _get_prop_names(el,"f_rest_")
    f_rest = _to_f32(_to_f16(_stack_props(el,rn),"f_rest")) if rn else np.zeros((xyz.shape[0],0),np.float32)
    return PlyData([_build_element(xyz,normals,f_dc,f_rest,opacity,scale,rotation)], text=False)


def compress_level2(plydata, target_sh=1):
    el = plydata.elements[0]
    sh = _get_sh_degree(el)
    print(f"  Detected SH degree: {sh}  →  clamped to {target_sh}")
    xyz      = _to_f32(_to_f16(_stack_props(el,["x","y","z"]),      "xyz"))
    normals  = _to_f32(_to_f16(_stack_props(el,["nx","ny","nz"]),   "normals"))
    f_dc     = _to_f32(_to_f16(_stack_props(el,_get_prop_names(el,"f_dc_")), "f_dc"))
    opacity  = _to_f32(_to_f16(_stack_props(el,["opacity"]),        "opacity"))
    scale    = _to_f32(_to_f16(_stack_props(el,_get_prop_names(el,"scale_")), "scale"))
    rotation = _to_f32(_to_f16(_stack_props(el,_get_prop_names(el,"rot_")),   "rotation"))
    rn = _get_prop_names(el,"f_rest_")
    if rn and sh > target_sh:
        keep = (target_sh + 1) ** 2 - 1
        full = _stack_props(el, rn)
        K = full.shape[1] // 3
        trimmed = np.concatenate([full[:,:K][:,:keep], full[:,K:2*K][:,:keep], full[:,2*K:][:,:keep]], axis=1)
        f_rest = _to_f32(_to_f16(trimmed, "f_rest"))
        print(f"  SH f_rest: {full.shape[1]} coeffs → {f_rest.shape[1]} coeffs per splat")
    elif rn:
        f_rest = _to_f32(_to_f16(_stack_props(el,rn),"f_rest"))
    else:
        f_rest = np.zeros((xyz.shape[0],0),np.float32)
    return PlyData([_build_element(xyz,normals,f_dc,f_rest,opacity,scale,rotation)], text=False)


def compress_level3(plydata):
    el = plydata.elements[0]
    sh = _get_sh_degree(el)
    print(f"  Detected SH degree: {sh}  →  dropped to 0 (no f_rest)")
    xyz     = _to_f32(_to_f16(_stack_props(el,["x","y","z"]),      "xyz"))
    f_dc    = _to_f32(_to_f16(_stack_props(el,_get_prop_names(el,"f_dc_")), "f_dc"))
    opacity = _to_f32(_to_f16(_stack_props(el,["opacity"]),        "opacity"))
    scale   = _to_f32(_to_f16(_stack_props(el,_get_prop_names(el,"scale_")), "scale"))
    raw_n   = _stack_props(el,["nx","ny","nz"])
    normals = _from_i8_norm(_to_i8_norm(raw_n))
    raw_r   = _stack_props(el,_get_prop_names(el,"rot_"))
    rotation = _from_i8_norm(_to_i8_norm(raw_r))
    norms = np.linalg.norm(rotation,axis=1,keepdims=True)
    rotation = rotation / np.where(norms>0,norms,1.0)
    f_rest = np.zeros((xyz.shape[0],0),np.float32)
    return PlyData([_build_element(xyz,normals,f_dc,f_rest,opacity,scale,rotation)], text=False)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan_ply(plydata):
    el = plydata.elements[0]
    N  = el.count
    total_bad = None

    def _report(label, names):
        nonlocal total_bad
        if not names:
            print(f"  {label:<20}  (not present)")
            return
        arr = _stack_props(el, names).astype(np.float32)
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        if label == "xyz":
            total_bad = int((~np.isfinite(arr).all(axis=1)).sum())
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            print(f"  {label:<22} *** ALL VALUES ARE NaN/Inf ***  ({n_nan:,} NaN, {n_inf:,} Inf)")
            return
        p99  = np.percentile(np.abs(finite), 99.0)
        p999 = np.percentile(np.abs(finite), 99.9)
        n_ov = int(np.sum(np.abs(finite) > FP16_MAX))
        flags = []
        if n_nan > 0: flags.append(f"⚠ {n_nan:,} NaN")
        if n_inf > 0: flags.append(f"⚠ {n_inf:,} Inf")
        if n_ov  > 0: flags.append(f"⚠ {n_ov:,} fp16-overflow")
        flag_str = "  " + ", ".join(flags) if flags else ""
        print(f"  {label:<22} min={finite.min():>12.4f}  max={finite.max():>12.4f}"
              f"  |p99|={p99:>10.4f}  |p99.9|={p999:>10.4f}{flag_str}")

    print(f"\n  {'Field':<22} {'min':>14}  {'max':>14}  {'|p99|':>12}  {'|p99.9|':>12}")
    print("  " + "-"*90)
    _report("xyz",      ["x","y","z"])
    _report("normals",  ["nx","ny","nz"])
    _report("f_dc",     _get_prop_names(el,"f_dc_"))
    _report("f_rest",   _get_prop_names(el,"f_rest_"))
    _report("opacity",  ["opacity"])
    _report("scale",    _get_prop_names(el,"scale_"))
    _report("rotation", _get_prop_names(el,"rot_"))
    print()
    if total_bad and total_bad > 0:
        pct = 100.0*total_bad/N
        print(f"  ⚠  DEGENERATE SPLATS: {total_bad:,} / {N:,} ({pct:.2f}%) have NaN/Inf xyz.")
        print(f"     These cause rendering artifacts even in the uncompressed PLY.")
        print(f"     Fix: add --clean when compressing to strip them out.")
    else:
        print(f"  ✓  No degenerate splats found.")
    print(f"  fp16 safe range: ±65504. Overflow values are clipped at 99.9th percentile.")
    print()


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean_ply(plydata):
    """Remove all splats that have NaN or Inf in any field. Returns (plydata, n_removed)."""
    el = plydata.elements[0]
    N  = el.count
    valid = np.ones(N, dtype=bool)
    for prop in el.properties:
        col = np.asarray(el[prop.name]).astype(np.float32)
        valid &= np.isfinite(col)
    n_bad = int((~valid).sum())
    if n_bad == 0:
        print(f"  [clean] All {N:,} splats are finite — nothing removed.")
        return plydata, 0
    pct = 100.0*n_bad/N
    print(f"  [clean] Removing {n_bad:,} degenerate splats ({pct:.2f}% of {N:,})")
    dtype = [(p.name, el[p.name].dtype) for p in el.properties]
    new_el = np.empty(valid.sum(), dtype=dtype)
    for prop in el.properties:
        new_el[prop.name] = np.asarray(el[prop.name])[valid]
    return PlyData([PlyElement.describe(new_el, "vertex")], text=False), n_bad


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(before: PlyData, after: PlyData, in_path: str, out_path: str):
    eb = before.elements[0]
    ea = after.elements[0]
    orig_sh = _get_sh_degree(eb)
    try: comp_sh = _get_sh_degree(ea)
    except ValueError: comp_sh = 0
    in_mb  = _file_mb(in_path)
    out_mb = _file_mb(out_path)
    vb = eb.count * len(eb.properties) * 4 / 1024 / 1024
    va = ea.count * len(ea.properties) * 4 / 1024 / 1024
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                  Compression Stats                   │")
    print("  ├───────────────────────┬──────────────┬──────────────┤")
    print(f"  │                       │   Original   │  Compressed  │")
    print("  ├───────────────────────┼──────────────┼──────────────┤")
    print(f"  │ Splat count           │ {eb.count:>12,} │ {ea.count:>12,} │")
    print(f"  │ Properties per splat  │ {len(eb.properties):>12} │ {len(ea.properties):>12} │")
    print(f"  │ SH degree             │ {orig_sh:>12} │ {comp_sh:>12} │")
    print(f"  │ File size (MB)        │ {in_mb:>12.1f} │ {out_mb:>12.1f} │")
    print(f"  │ Ratio                 │ {'1.00x':>12} │ {in_mb/out_mb:>11.2f}x │")
    print(f"  │ VRAM est. fp32 (MB)   │ {vb:>12.0f} │ {va:>12.0f} │")
    print("  └───────────────────────┴──────────────┴──────────────┘")
    print()
    print("  NOTE: VRAM estimate assumes all fields loaded as fp32.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compress a 2DGS point_cloud.ply to reduce VRAM footprint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Levels:
  1  fp16 round-trip, full SH              — quality ≈ lossless, same disk size
  2  fp16 + SH degree 3→1                  — ~2.4x smaller, minor view-dep. loss
  3  fp16 + drop all SH rest + int8 rot    — ~3.6x smaller, no view-dep. colour

Recommended workflow:
  python compress_ply.py model.ply --scan
  python compress_ply.py model.ply model_L2.ply --level 2 --clean --stats
        """,
    )
    parser.add_argument("input",  help="Input .ply file")
    parser.add_argument("output", nargs="?", default=None, help="Output .ply file (not needed for --scan)")
    parser.add_argument("--level",     "-l", type=int, choices=[1,2,3], default=2)
    parser.add_argument("--sh-degree", type=int, default=1,
                        help="Target SH degree for level 2 (default: 1)")
    parser.add_argument("--clean",  action="store_true",
                        help="Strip NaN/Inf splats before compressing")
    parser.add_argument("--scan",   action="store_true",
                        help="Print field-level health stats. No output written.")
    parser.add_argument("--stats",  action="store_true",
                        help="Print before/after size stats after writing.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: file not found: {args.input}", file=sys.stderr); sys.exit(1)

    if args.scan:
        print(f"\n[compress_ply] Scanning: {args.input}  ({_file_mb(args.input):.1f} MB)")
        scan_ply(_load_ply(args.input))
        return

    if args.output is None:
        print("ERROR: output path required unless --scan is used.", file=sys.stderr); sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"\n[compress_ply] Level {args.level}  |  {args.input}  ({_file_mb(args.input):.1f} MB)")

    original = _load_ply(args.input)
    to_compress = original

    if args.clean:
        to_compress, _ = clean_ply(original)
    else:
        el  = original.elements[0]
        xyz = _stack_props(el, ["x","y","z"]).astype(np.float32)
        n_bad = int((~np.isfinite(xyz).all(axis=1)).sum())
        if n_bad > 0:
            print(f"  ⚠ WARNING: {n_bad:,} splats have NaN/Inf xyz. Add --clean to remove them.")

    if   args.level == 1: compressed = compress_level1(to_compress)
    elif args.level == 2: compressed = compress_level2(to_compress, target_sh=args.sh_degree)
    elif args.level == 3: compressed = compress_level3(to_compress)

    compressed.write(args.output)
    print(f"  Output: {args.output}  ({_file_mb(args.output):.1f} MB)")

    if args.stats:
        print_stats(to_compress, compressed, args.input, args.output)

    print("[compress_ply] Done.\n")


if __name__ == "__main__":
    main()
