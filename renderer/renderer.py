"""ViewerRenderer: GPU render pipeline for 2D Gaussian Splatting scenes.

Kept as Kestrel's own render path (rather than gsplat2d_rendering's
SplatRenderer/Renderer facade) because it supports render types and display
modes the library doesn't expose yet (normal/alpha/distortion maps,
point-cloud/disk display, crop box, sparsity, a live opacity-threshold
slider) — see docs/gsplat2d-rendering-gap.md for the plan to fold this back
into the library. Model loading, camera construction, octree building,
per-frame profiling, SH evaluation, and depth-to-normal all go through
gsplat2d_rendering; only the rasterization call itself stays local. Kestrel
no longer depends on the 2d_gaussian_splatting submodule at all —
diff_surfel_rasterization resolves via the globally pip-installed package
built from gsplat2d-rendering's own vendored copy.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from gsplat2d_rendering.render.profiling import Profiler
from gsplat2d_rendering.render.depth_normal import depth_to_normal
from gsplat2d_rendering.sh import C0 as SH_C0, eval_sh

from renderer.helpers import gradient_map
from renderer.culling import frustum_cull_mask


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
            Print GPU timing (A:sh / B:raster / C:post) every
            *profiling_print_every* frames after *profiling_warmup* warmup frames.
        """
        super().__init__()
        self.gaussian_model   = gaussian_model
        self.background_color = background_color
        self.clm_colors       = torch.tensor(plt.cm.get_cmap("turbo").colors, device="cuda")

        # Frustum-culling state
        self.octree             = octree
        self.culling_enabled    = culling_enabled
        self._spatially_ordered = False
        self._node_aabbs_gpu    = None
        self._update_node_aabbs_gpu()

        # Profiling state — gsplat2d_rendering's own stage-timing collector
        # (see render/profiling.py). Stats accumulate in the Profiler itself
        # (mean/min/max/count per stage) rather than a Kestrel-side history
        # list; _record_profile below prints + resets it every
        # profiling_print_every frames, so each printed block is a fresh
        # window rather than an all-time running average.
        self.profiler               = Profiler(sync_fn=torch.cuda.synchronize)
        self.profiling_enabled      = profiling_enabled
        if profiling_enabled:
            self.profiler.enable()
        self._prof_warmup           = profiling_warmup
        self._prof_print_every      = profiling_print_every
        self._prof_frame_count      = 0

        # Exposed after each render_viewer() call
        self.last_visible_count    = 0

        if do_initialize:
            self.update_pc_features()

        self._log_startup()

    def _update_node_aabbs_gpu(self):
        if self.octree is not None:
            self._node_aabbs_gpu = torch.from_numpy(self.octree.node_aabbs).cuda()
        else:
            self._node_aabbs_gpu = None

    def _log_startup(self):
        if self.culling_enabled and self.octree is not None:
            L = len(self.octree.node_aabbs)
            print(f"[viewer] Frustum culling : enabled — {L:,} leaf nodes")
        elif not self.culling_enabled:
            print("[viewer] Frustum culling : disabled via --no-culling")
        else:
            print("[viewer] Frustum culling : no index found — run with --build-index to enable")

        if self.profiling_enabled:
            print(f"[viewer] Profiling       : enabled "
                  f"(warmup {self._prof_warmup} frames, "
                  f"printing a stage breakdown every {self._prof_print_every} frames — "
                  f"toggle live from the sidebar's Status panel)")
        else:
            print("[viewer] Profiling       : disabled via --no-profiling "
                  "(toggle live from the sidebar's Status panel)")

    def set_profiling_enabled(self, enabled: bool):
        """Runtime toggle (sidebar 'Profiling' checkbox) — cheaper than
        restarting with/without --no-profiling. Resets accumulated stats and
        the warmup counter on every toggle, so turning it back on always
        starts a clean warmup window rather than mixing in stale timings
        from before it was disabled."""
        if enabled == self.profiling_enabled:
            return
        self.profiling_enabled = enabled
        self._prof_frame_count = 0
        self.profiler.reset()
        if enabled:
            self.profiler.enable()
            print(f"[viewer] Profiling       : enabled "
                  f"(warmup {self._prof_warmup} frames, "
                  f"printing every {self._prof_print_every} frames)")
        else:
            self.profiler.disable()
            print("[viewer] Profiling       : disabled")

    # ── public helpers ────────────────────────────────────────────────────────

    def update_pc_features(self):
        self.means3D   = self.gaussian_model.get_xyz
        self.all_ids   = torch.ones(self.means3D.shape[0], dtype=torch.bool,
                                    device=self.means3D.device)
        self.means2D   = torch.zeros_like(self.means3D)
        self.opacity   = self.gaussian_model.get_opacity
        self.scales    = self.gaussian_model.get_scaling
        self.rotations = self.gaussian_model.get_rotation
        self.shs       = self.gaussian_model.get_features

        if self.octree is not None and not self._spatially_ordered:
            # Reorder all splat tensors to match octree flat_indices leaf order
            # (GaussianModel.reorder_ — see gsplat2d_rendering/model.py). After
            # this, leaf j maps to contiguous indices [node_offsets[j], node_offsets[j+1]),
            # enabling direct slice gather instead of scattered bool-mask gather.
            perm = torch.from_numpy(
                self.octree.flat_indices.astype(np.int64)
            ).to(self.means3D.device)
            self.gaussian_model.reorder_(perm)
            del perm
            self._spatially_ordered = True
            self.means3D   = self.gaussian_model.get_xyz
            self.means2D   = torch.zeros_like(self.means3D)
            self.opacity   = self.gaussian_model.get_opacity
            self.scales    = self.gaussian_model.get_scaling
            self.rotations = self.gaussian_model.get_rotation
            self.shs       = self.gaussian_model.get_features
            print(f"[viewer] Spatial reorder : {self.means3D.shape[0]:,} splats sorted by octree leaf")

    def disk_kernel(self, opacity):
        return torch.exp(-0.5 * 100 * torch.clamp(opacity - 0.5, min=0) ** 2)

    def color_map(self, map):
        if map.min() == map.max():
            idx = torch.zeros_like(map, device=map.device).round().long().squeeze()
        else:
            map = (map - map.min()) / (map.max() - map.min())
            idx = (map * 255).round().long().squeeze()
        return self.clm_colors[idx].permute(2, 0, 1)

    # ── frustum culling ───────────────────────────────────────────────────────

    def _apply_frustum_cull(self,
                             is_in_box: torch.Tensor,
                             viewpoint_camera):
        return frustum_cull_mask(
            self.octree,
            is_in_box,
            viewpoint_camera,
            self._spatially_ordered,
            self.all_ids,
            self._node_aabbs_gpu,
        )

    # ── profiling ─────────────────────────────────────────────────────────────

    # (internal stage key → printed label, display order)
    _PROF_STAGES = (("sh", "A: SH eval"), ("raster", "B: Rasterize"), ("post", "C: Post-proc"))

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
        means = {key: stats[key]["mean_ms"] for key, _ in self._PROF_STAGES if key in stats}
        if not means:
            return
        total_ms   = sum(means.values())
        fps        = 1000.0 / total_ms if total_ms > 0 else 0.0
        n_frames   = next(iter(stats.values()))["count"]
        bottleneck = max(means, key=means.get)

        def _stage_row(label: str, mean_ms: float, min_ms: float, max_ms: float,
                       share: float, marker: str) -> str:
            # Marker is a fixed-width field (not appended free-form) so every
            # stage row is exactly as long as the column header, regardless
            # of which stage happens to be the bottleneck this window.
            return (f" {label:<14}{mean_ms:>7.2f}ms{min_ms:>7.2f}ms"
                    f"{max_ms:>7.2f}ms{share:>7.0f}% {marker:<12}")

        header  = f" PROFILE — mean over last {n_frames} frames, {n_vis:,} splats visible "
        col_row = f" {'stage':<14}{'mean':>9}{'min':>9}{'max':>9}{'share':>8} {'':<12}"
        W = max(len(header), len(col_row))

        def _line(content: str) -> str:
            return f"[viewer] │{content:<{W}}│"

        rule = "─" * W
        print(f"[viewer] ╭{rule}╮")
        print(f"[viewer] │{header:^{W}}│")
        print(f"[viewer] ├{rule}┤")
        print(_line(col_row))
        for key, label in self._PROF_STAGES:
            if key not in stats:
                continue
            s      = stats[key]
            share  = 100.0 * s["mean_ms"] / total_ms if total_ms > 0 else 0.0
            marker = "◀ bottleneck" if key == bottleneck else ""
            print(_line(_stage_row(label, s["mean_ms"], s["min_ms"], s["max_ms"], share, marker)))
        print(f"[viewer] ├{rule}┤")
        prefix = f" total: {total_ms:.2f}ms"
        suffix = f"{fps:.1f} fps "
        print(_line(prefix + suffix.rjust(W - len(prefix))))
        print(f"[viewer] ╰{rule}╯")

    # ── main render path ──────────────────────────────────────────────────────

    def render_viewer(self,
                      viewpoint_camera,
                      active_sh_degree,
                      scaling_modifier,
                      depth_ratio,
                      bg_color: torch.Tensor,
                      sparsity: int = 1,
                      opacity_threshold: float = 0.0,
                      show_ptc: bool = False,
                      show_disk: bool = False,
                      point_size: float = 0.001,
                      valid_range=None,
                      compute_post: bool = True):
        """
        Render the scene.  bg_color must be on GPU.

        Profiling breakdown (printed when profiling_enabled=True):
          A:sh     — SH evaluation (pre-computed here for accurate timing)
          B:raster — CUDA sort + alpha-composite kernel
          C:post   — normal / depth / distortion maps
        """
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # ── spatial crop box ──────────────────────────────────────────────────
        if valid_range is not None:
            is_in_box = (
                (valid_range[0][0] <= self.means3D[:, 0]) & (self.means3D[:, 0] <= valid_range[0][1]) &
                (valid_range[1][0] <= self.means3D[:, 1]) & (self.means3D[:, 1] <= valid_range[1][1]) &
                (valid_range[2][0] <= self.means3D[:, 2]) & (self.means3D[:, 2] <= valid_range[2][1])
            )
        else:
            is_in_box = self.all_ids

        # ── frustum culling (CPU octree walk + GPU mask) ──────────────────────
        vis_ranges = None
        if self.culling_enabled and self.octree is not None:
            is_in_box, vis_ranges = self._apply_frustum_cull(is_in_box, viewpoint_camera)

        # ── opacity threshold (GPU, removes near-transparent splats) ──────────
        if opacity_threshold > 0.0:
            is_in_box = is_in_box & (self.opacity[:, 0] > opacity_threshold)
            vis_ranges = None  # ranges no longer safe with per-splat opacity filter

        # ── gather visible splats ─────────────────────────────────────────────
        # Fast path (spatially ordered, no crop box, no opacity threshold):
        # contiguous slice-cat per leaf — avoids scattered bool-mask gather.
        if vis_ranges:
            _cat = lambda t: torch.cat([t[s:e:sparsity] for s, e in vis_ranges])
        else:
            _cat = lambda t: t[is_in_box][::sparsity]

        means3D_f = _cat(self.means3D)
        means2D_f = _cat(self.means2D)
        opacity_f = _cat(self.opacity)
        if show_disk:
            opacity_f = self.disk_kernel(opacity_f)
        scales_f = _cat(self.scales)
        if show_ptc:
            scales_f = torch.full(scales_f.shape, point_size * 0.1, device=scales_f.device)
        else:
            scales_f = scaling_modifier * scales_f
        rot_f = _cat(self.rotations)
        shs_f = _cat(self.shs)

        # ── [A] SH evaluation ─────────────────────────────────────────────────
        # SH is always pre-computed here (not inside the rasterizer) so we can
        # time it accurately with profiling_enabled=True.
        self.profiler.start()

        if active_sh_degree > 0:
            dir_vecs = means3D_f - viewpoint_camera.camera_center
            dir_vecs = dir_vecs / (dir_vecs.norm(dim=1, keepdim=True) + 1e-8)
            sh_dim   = (active_sh_degree + 1) ** 2
            colors   = eval_sh(active_sh_degree,
                               shs_f.transpose(1, 2)[:, :, :sh_dim],
                               dir_vecs)
            colors   = torch.clamp_min(colors + 0.5, 0.0)
        else:
            colors = torch.clamp_min(SH_C0 * shs_f[:, 0, :] + 0.5, 0.0)

        self.profiler.lap("sh")

        # ── [B] Rasterizer ────────────────────────────────────────────────────
        rendered_image, radii, allmap = rasterizer(
            means3D        = means3D_f,
            means2D        = means2D_f,
            shs            = None,
            colors_precomp = colors,
            opacities      = opacity_f,
            scales         = scales_f,
            rotations      = rot_f,
            cov3D_precomp  = None,
        )

        self.profiler.lap("raster")

        # ── [C] Post-processing ───────────────────────────────────────────────
        if compute_post:
            render_alpha          = allmap[1:2]
            render_normal         = allmap[2:5]
            render_normal         = (
                render_normal.permute(1, 2, 0) @
                viewpoint_camera.world_view_transform[:3, :3].T
            ).permute(2, 0, 1)
            render_depth_median   = torch.nan_to_num(allmap[5:6], 0, 0)
            render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
            render_dist           = allmap[6:7]
            surf_depth  = render_depth_expected * (1 - depth_ratio) + depth_ratio * render_depth_median
            surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
            surf_normal = surf_normal.permute(2, 0, 1) * render_alpha.detach()
            render_normal = F.normalize(render_normal, dim=0) * 0.5 + 0.5
            surf_normal   = surf_normal * 0.5 + 0.5
            view_normal   = -F.normalize(allmap[2:5], dim=0) * 0.5 + 0.5

        self.profiler.lap("post")

        n_vis = (sum(e - s for s, e in vis_ranges) if vis_ranges
                 else int(is_in_box.sum())) // sparsity
        self.last_visible_count = n_vis

        if self.profiling_enabled:
            self._record_profile(n_vis)

        if not compute_post:
            return {"render": rendered_image}

        return {
            "render":          rendered_image,
            "rend_alpha":      self.color_map(render_alpha.unsqueeze(-1)),
            "rend_normal":     render_normal,
            "view_normal":     view_normal,
            "surf_depth":      self.color_map(surf_depth.unsqueeze(-1)),
            "surf_depth_raw":  surf_depth.squeeze(0),   # float32 [H, W] in scene units
            "surf_normal":     surf_normal,
            "rend_dist":       self.color_map(render_dist.unsqueeze(-1)),
        }

    # ── output routing ────────────────────────────────────────────────────────

    # Render types that only need the raw rendered image (no allmap post-processing).
    _POST_FREE = frozenset({"render", "edge"})

    def get_outputs(self,
                    camera,
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

        def get_result(results, rtype):
            if rtype in results:
                return results[rtype]
            if rtype == "curvature":
                return self.color_map(gradient_map(results["surf_normal"]))
            if rtype == "edge":
                return self.color_map(gradient_map(results["render"]))
            return results["render"]

        if split:
            compute_post = not (render_type1 in self._POST_FREE and render_type2 in self._POST_FREE)
        else:
            compute_post = render_type not in self._POST_FREE

        results = self.render_viewer(
            camera, active_sh_degree, scaling_modifier, depth_ratio,
            self.background_color,
            sparsity=sparsity, opacity_threshold=opacity_threshold,
            valid_range=valid_range,
            show_ptc=show_ptc, show_disk=show_disk, point_size=point_size,
            compute_post=compute_post,
        )

        if not split:
            return get_result(results, render_type)

        out = torch.zeros_like(results["render"])
        _, _, H = out.shape
        sp = int(H * slider)
        out[:, :, :sp]  = get_result(results, render_type1)[:, :, :sp]
        out[:, :, sp:]  = get_result(results, render_type2)[:, :, sp:]
        out[:, :, sp]   = torch.ones_like(out[:, :, sp])
        return out
