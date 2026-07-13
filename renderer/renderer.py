"""ViewerRenderer: Kestrel's thin GUI-facing wrapper around
gsplat2d_rendering.SplatRenderer.

All actual rendering work — culling, LOD, candidate filters (sparsity/crop
box/opacity threshold), render mode (gaussian/point/disk), depth-ratio
blending, and per-frame profiling — is delegated to the library. This class
only owns:

  - rebuilding SplatRenderer when the model/octree/culling_enabled change
    (those are constructor-only on SplatRenderer, not mutable afterward —
    see gsplat2d_rendering/render/rasterizer.py)
  - applying Kestrel's per-frame SH-degree cap by mutating
    `gaussian_model.active_sh_degree` in place before each render() call —
    there's no per-call override on SplatRenderer.render() for this
  - turning RenderOutput's raw GPU tensors into the specific display image
    Kestrel's render_type dropdown asks for (turbo colormap for
    alpha/depth/distortion, world-space normal rotation, depth-to-normal/
    curvature via gsplat2d_rendering.depth_to_normal, split-view
    compositing) — all "how do I look at this" concerns the library
    deliberately leaves to callers.

Known gap: Kestrel's "Scale" slider (global splat-size multiplier) has no
effect right now. `SplatRenderer.render()` hardcodes the rasterizer's
`scale_modifier` to 1.0 instead of exposing it as a parameter — that field
is a genuine, already-wired CUDA kernel parameter (see
diff_surfel_rasterization.GaussianRasterizationSettings), so this is a
trivial library-side pass-through fix, not something worth working around
here with an O(N) per-frame scale-tensor rewrite.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import gsplat2d_rendering as gs2d
from gsplat2d_rendering.render import SplatRenderer

from renderer.helpers import gradient_map


class ViewerRenderer:
    def __init__(self,
                 gaussian_model,
                 background_color,
                 do_initialize: bool = True,
                 # Frustum culling
                 octree=None,
                 culling_enabled: bool = True,
                 # Per-frame profiling
                 profiling_enabled: bool = True,
                 profiling_warmup: int = 5,
                 profiling_print_every: int = 30):
        """
        Parameters
        ----------
        gaussian_model
            gsplat2d_rendering.GaussianModel
        octree
            gsplat2d_rendering.culling.Octree, or None to disable frustum culling.
        culling_enabled
            Set False to bypass culling even when an octree is present.
        profiling_enabled
            Print a per-stage GPU timing breakdown every
            *profiling_print_every* frames after *profiling_warmup* warmup frames.
        """
        self.gaussian_model     = gaussian_model
        self.background_color   = background_color
        self.octree              = octree
        self.culling_enabled     = culling_enabled
        self._spatially_ordered  = False
        # Captured once per model (before any per-frame active_sh_degree
        # mutation below) — the ceiling the sidebar's SH Degree spinbox caps
        # against.
        self._max_sh_degree      = gaussian_model.active_sh_degree
        self.clm_colors          = torch.tensor(plt.cm.get_cmap("turbo").colors, device="cuda")

        self._prof_warmup        = profiling_warmup
        self._prof_print_every   = profiling_print_every
        self._prof_frame_count   = 0
        self._profiling_wanted   = profiling_enabled

        # Exposed after each get_outputs() call
        self.last_visible_count  = 0

        self._splat_renderer: SplatRenderer | None = None
        if do_initialize:
            self.update_pc_features()
        else:
            self._rebuild_splat_renderer()

        self._log_startup()

    # ── model / octree wiring ─────────────────────────────────────────────────

    def update_pc_features(self, verbose_log: bool = False):
        """Reorders the model into the octree's leaf-contiguous order
        (SplatRenderer's own precondition — see its class docstring) and
        (re)builds the underlying SplatRenderer. Called whenever the model
        or octree changes (new PLY installed, compression level switched,
        background index build finished). GaussianModel.reorder_ logs the
        reorder itself (gsplat2d_rendering's own logging).

        verbose_log forwards to reorder_/_rebuild_splat_renderer, routing
        their summary lines through VERBOSE instead of NORMAL — pass True
        for a caller that invokes this repeatedly (chunk streaming's
        per-rebuild swap install), leave False for a one-off call (model
        load, compression switch) that's worth NORMAL-level visibility.

        Load-time chunk streaming (viewer/app.py's chunk-streaming install
        sites) deliberately sets `_spatially_ordered = True` *before* calling
        this, bypassing the reorder branch entirely — ChunkManager already
        bakes each resident chunk into its own octree's leaf-contiguous
        order once, at read time (renderer/chunk_manager.py::ChunkManager.
        _ensure_fine_octree), so the composited model it hands back is
        already in the order this method would otherwise spend an
        O(total composited points) reorder deriving from scratch on every
        single rebuild. This is intentional, not a bug to "fix" back to
        False — see that method's docstring for the full reasoning."""
        if self.octree is not None and not self._spatially_ordered:
            # flat_indices is already int64 by construction (build_octree
            # always produces it via .astype(np.int64), and _stitch_fine_octree
            # preserves that dtype through concatenation/offsetting) -- the
            # .astype("int64") that used to be here was a needless full-array
            # copy (numpy's astype() copies unconditionally unless told not
            # to, even when the dtype already matches): measured ~23ms vs
            # ~3ms on a 7.19M-point octree, a real chunk of the periodic
            # rebuild-install cost during continuous camera motion.
            perm = torch.from_numpy(
                self.octree.flat_indices
            ).to(self.gaussian_model.xyz.device)
            self.gaussian_model.reorder_(perm, verbose_log=verbose_log)
            del perm
            self._spatially_ordered = True
        self._max_sh_degree = self.gaussian_model.active_sh_degree
        self._rebuild_splat_renderer(verbose_log=verbose_log)

    def _rebuild_splat_renderer(self, verbose_log: bool = False):
        """SplatRenderer's octree/culling_enabled/with_extras are
        constructor-only, so swapping the octree (background index build) or
        toggling culling means building a fresh instance — profiler state
        resets with it, same as switching models. `with_extras=True`
        unconditionally: the kernel computes alpha/normal/middepth/
        distortion every frame regardless (see rasterizer.py's module
        docstring) — the only real cost is a few extra small GPU tensors
        staying alive between frames, negligible next to the splat model
        itself, and it's what lets render_type switch live without a rebuild."""
        was_enabled = (self._splat_renderer.profiler.enabled
                       if self._splat_renderer is not None else self._profiling_wanted)
        old_splat_renderer = self._splat_renderer
        self._splat_renderer = SplatRenderer(
            self.gaussian_model,
            octree=self.octree,
            culling_enabled=self.culling_enabled,
            with_extras=True,
            verbose_log=verbose_log,
        )
        # Explicit del + empty_cache. SplatRenderer used to hold a reference
        # cycle (its Profiler's sync_fn was a bound method pointing back to
        # the SplatRenderer instance itself), which meant plain `del` could
        # never free it via refcounting alone -- only Python's cyclic GC
        # could, and only whenever it happened to run. That's now fixed at
        # the source in gsplat2d_rendering (rasterizer.py's sync_fn closes
        # over the device string instead of `self`), so `del` here is
        # sufficient on its own -- verified directly with the automatic
        # cyclic GC fully disabled. (Two earlier attempts at working around
        # this from the Kestrel side instead -- an explicit full gc.collect(),
        # then a cheaper gc.collect(0) -- are gone: the full collect() walked
        # every tracked object across all 3 generations and measured
        # ~21-24ms of a ~35-38ms total rebuild-install cost on this scene's
        # full 7.19M-point model; gc.collect(0) was faster but unreliable,
        # since a cycle can get promoted out of generation 0 by an unrelated
        # automatic collection -- e.g. triggered by chunk streaming's own
        # background transition-worker allocations -- before this explicit
        # call ever runs, and gc.collect(0) then can't see it. Both were
        # real, measured causes of "chunk streaming is slower than no
        # streaming, even at a steady swipe" once the VRAM-correctness fix
        # was in place; fixing the cycle at its root removes the need for
        # either.)
        del old_splat_renderer
        torch.cuda.empty_cache()
        # Not exposed as a constructor/render() param on SplatRenderer —
        # background is a plain mutable instance attribute there, so this is
        # a supported way to override its (otherwise hardcoded black) default.
        self._splat_renderer.background = self.background_color
        self.profiler = self._splat_renderer.profiler
        self._prof_frame_count = 0
        if was_enabled:
            self._splat_renderer.enable_profiling()

    def _log_startup(self):
        # Frustum-culling status is logged by SplatRenderer itself on every
        # (re)build (gsplat2d_rendering's own logging, see _rebuild_splat_renderer
        # above) — only Kestrel's own profiling-cadence choice is reported here.
        if self.profiling_enabled:
            print(f"[viewer] Profiling       : enabled "
                  f"(warmup {self._prof_warmup} frames, "
                  f"printing a stage breakdown every {self._prof_print_every} frames — "
                  f"toggle live from the sidebar's Status panel)")
        else:
            print("[viewer] Profiling       : disabled via --no-profiling "
                  "(toggle live from the sidebar's Status panel)")

    # ── profiling ────────────────────────────────────────────────────────────

    @property
    def profiling_enabled(self) -> bool:
        return self.profiler.enabled

    def set_profiling_enabled(self, enabled: bool):
        """Runtime toggle (sidebar 'Profiling' checkbox) — cheaper than
        restarting with/without --no-profiling. Resets accumulated stats and
        the warmup counter on every toggle, so turning it back on always
        starts a clean warmup window rather than mixing in stale timings
        from before it was disabled."""
        if enabled == self.profiling_enabled:
            return
        self._prof_frame_count = 0
        if enabled:
            self._splat_renderer.enable_profiling()
        else:
            self._splat_renderer.disable_profiling()
        self._splat_renderer.reset_profiling()
        if enabled:
            print(f"[viewer] Profiling       : enabled "
                  f"(warmup {self._prof_warmup} frames, "
                  f"printing every {self._prof_print_every} frames)")
        else:
            print("[viewer] Profiling       : disabled")

    # Known library stages (in the pipeline order SplatRenderer.render() laps
    # them), given nicer print labels; anything not listed here (future
    # library stages, or Kestrel's own "kestrel_post" lap below) falls back
    # to its raw key with underscores turned to spaces — see
    # _print_profile_block, which iterates whatever's actually in
    # Profiler.stats() rather than assuming a fixed stage set.
    _STAGE_LABELS = {
        "cull":             "octree cull",
        "lod_select":       "LOD select",
        "sparsity":         "sparsity",
        "narrow_cull":      "narrow cull",
        "screen_size_cull": "screen-size cull",
        "bounds_cull":      "crop box",
        "opacity_cull":     "opacity thresh",
        "gather":           "gather",
        "sh_eval":          "SH eval",
        "render_mode":      "render mode",
        "rasterize":        "rasterize",
        "depth_extract":    "depth extract",
        "kestrel_post":     "Kestrel post-proc",
    }

    def _record_profile(self, n_vis: int):
        """Pulls mean/min/max/count straight from gsplat2d_rendering's own
        Profiler.stats() (accumulated since the last reset) rather than
        keeping a second, Kestrel-side rolling-window history — reset()
        every profiling_print_every frames turns that cumulative window into
        a fresh one each time a block prints, so figures always reflect only
        the frames since the last printout, not an all-time average."""
        self._prof_frame_count += 1

        if self._prof_frame_count == self._prof_warmup + 1:
            # Warmup just ended — drop whatever accumulated during it (first
            # frames after (re)enabling profiling include CUDA/JIT warmup
            # cost that isn't representative of steady-state performance).
            self.profiler.reset()
        if self._prof_frame_count <= self._prof_warmup:
            return
        if (self._prof_frame_count - self._prof_warmup) % self._prof_print_every != 0:
            return

        stats = self.profiler.stats()
        self._print_profile_block(n_vis, stats)
        self.profiler.reset()

    def _print_profile_block(self, n_vis: int, stats: dict):
        if not stats:
            return
        order      = list(stats.keys())  # Profiler.stats() preserves lap() call order
        means      = {k: stats[k]["mean_ms"] for k in order}
        total_ms   = sum(means.values())
        fps        = 1000.0 / total_ms if total_ms > 0 else 0.0
        n_frames   = stats[order[0]]["count"]
        bottleneck = max(means, key=means.get)
        labels     = {k: self._STAGE_LABELS.get(k, k.replace("_", " ")) for k in order}
        label_w    = max(max(len(l) for l in labels.values()), len("stage"))

        def _stage_row(label: str, mean_ms: float, min_ms: float, max_ms: float,
                       share: float, marker: str) -> str:
            # Marker is a fixed-width field (not appended free-form) so every
            # stage row is exactly as long as the column header, regardless
            # of which stage happens to be the bottleneck this window.
            return (f" {label:<{label_w}}  {mean_ms:>7.2f}ms{min_ms:>7.2f}ms"
                    f"{max_ms:>7.2f}ms{share:>7.0f}% {marker:<12}")

        header  = f" PROFILE — mean over last {n_frames} frames, {n_vis:,} splats visible "
        col_row = f" {'stage':<{label_w}}  {'mean':>9}{'min':>9}{'max':>9}{'share':>8} {'':<12}"
        W = max(len(header), len(col_row))

        def _line(content: str) -> str:
            return f"[viewer] │{content:<{W}}│"

        rule = "─" * W
        print(f"[viewer] ╭{rule}╮")
        print(f"[viewer] │{header:^{W}}│")
        print(f"[viewer] ├{rule}┤")
        print(_line(col_row))
        for key in order:
            s      = stats[key]
            share  = 100.0 * s["mean_ms"] / total_ms if total_ms > 0 else 0.0
            marker = "◀ bottleneck" if key == bottleneck else ""
            print(_line(_stage_row(labels[key], s["mean_ms"], s["min_ms"], s["max_ms"], share, marker)))
        print(f"[viewer] ├{rule}┤")
        prefix = f" total: {total_ms:.2f}ms"
        suffix = f"{fps:.1f} fps "
        print(_line(prefix + suffix.rjust(W - len(prefix))))
        print(f"[viewer] ╰{rule}╯")

    # ── display helpers ─────────────────────────────────────────────────────

    def color_map(self, map: torch.Tensor) -> torch.Tensor:
        """Turbo colormap for scalar fields (alpha/depth/distortion) — map
        is [H, W] (or broadcastable to it), returns [3, H, W]."""
        if map.min() == map.max():
            idx = torch.zeros_like(map, device=map.device).round().long().squeeze()
        else:
            map = (map - map.min()) / (map.max() - map.min())
            idx = (map * 255).round().long().squeeze()
        return self.clm_colors[idx].permute(2, 0, 1)

    def _compute_result(self, rtype: str, output, proj_camera) -> torch.Tensor:
        """proj_camera must be whichever camera actually produced output's
        pixels (render_camera if one was passed to get_outputs(), else the
        cull camera) — rend_normal/surf_normal/curvature all reproject
        relative to it."""
        if rtype == "render":
            return output.rgb
        if rtype == "edge":
            return self.color_map(gradient_map(output.rgb))
        if rtype == "rend_alpha":
            return self.color_map(output.alpha)
        if rtype == "surf_depth":
            return self.color_map(output.depth)
        if rtype == "rend_dist":
            return self.color_map(output.distortion)
        if rtype == "rend_normal":
            # output.normal is raw camera-space kernel output (see
            # gsplat2d_rendering/render/extras.py) — rotate into world space.
            world_normal = (output.normal.permute(1, 2, 0)
                             @ proj_camera.world_view_transform[:3, :3].T).permute(2, 0, 1)
            return F.normalize(world_normal, dim=0) * 0.5 + 0.5
        if rtype == "view_normal":
            return -F.normalize(output.normal, dim=0) * 0.5 + 0.5
        if rtype in ("surf_normal", "curvature"):
            surf_normal = gs2d.depth_to_normal(proj_camera, output.depth).permute(2, 0, 1)
            surf_normal = surf_normal * output.alpha.unsqueeze(0).detach()
            surf_normal = surf_normal * 0.5 + 0.5
            if rtype == "curvature":
                return self.color_map(gradient_map(surf_normal))
            return surf_normal
        return output.rgb

    # ── main render path ──────────────────────────────────────────────────────

    def get_outputs(self,
                    camera,
                    render_camera=None,
                    valid_range: tuple = None,
                    split: bool = False,
                    slider: float = 0.5,
                    show_ptc: bool = False,
                    show_disk: bool = False,
                    point_size: float = 0.01,
                    active_sh_degree: int = 3,
                    scaling_modifier: float = 1.,
                    sparsity: int = 1,
                    opacity_threshold: float = 0.0,
                    depth_ratio: float = 0.,
                    render_type: str = "render",
                    render_type1: str = "render",
                    render_type2: str = "render"):
        # scaling_modifier: currently a no-op — see module docstring.
        del scaling_modifier

        self.gaussian_model.active_sh_degree = min(active_sh_degree, self._max_sh_degree)
        render_mode = "disk" if (show_ptc and show_disk) else "point" if show_ptc else "gaussian"
        bounds = tuple(tuple(axis) for axis in valid_range) if valid_range is not None else None

        output = self._splat_renderer.render(
            camera, render_camera=render_camera,
            render_mode=render_mode, point_size=point_size,
            sparsity=sparsity, bounds=bounds, min_opacity=opacity_threshold,
            depth_ratio=depth_ratio,
        )
        self.last_visible_count = output.num_rendered
        proj_camera = render_camera if render_camera is not None else camera

        cache: dict[str, torch.Tensor] = {}

        def result(rtype: str) -> torch.Tensor:
            if rtype not in cache:
                cache[rtype] = self._compute_result(rtype, output, proj_camera)
            return cache[rtype]

        if not split:
            final = result(render_type)
        else:
            final = torch.zeros_like(output.rgb)
            _, _, H = final.shape
            sp = int(H * slider)
            final[:, :, :sp] = result(render_type1)[:, :, :sp]
            final[:, :, sp:] = result(render_type2)[:, :, sp:]
            final[:, :, sp]  = torch.ones_like(final[:, :, sp])

        # Kestrel-side visualization work (color_map/depth_to_normal/
        # gradient_map above) isn't inside SplatRenderer's own profiler laps —
        # lap it here, on the same shared Profiler instance, so the printed
        # breakdown accounts for 100% of frame time, not just the library's
        # portion of it.
        self.profiler.lap("kestrel_post")
        if self.profiling_enabled:
            self._record_profile(self.last_visible_count)

        return final
