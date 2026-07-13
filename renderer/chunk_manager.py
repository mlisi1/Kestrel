"""ChunkManager: per-frame VRAM/RAM/disk residency for load-time chunk
streaming.

Not GUI code (no Qt) -- lives alongside ViewerRenderer in renderer/, per
Kestrel's GUI-separation rule (only sidebar/dialog/canvas code is required
to live under viewer/*).

Chunks are gsplat2d_rendering.Octree leaves built at disk-chunk granularity
(see utils/build_chunks.py) -- an AABB + a contiguous row range per chunk
into the chunk-reordered PLY. Every frame, `update(camera)` runs the same
GPU frustum test the library already uses for per-frame splat culling
(`visible_leaf_mask_torch`), just against this coarser chunk manifest, to
decide which chunks should be "VRAM tier" (in the composition actually
rendered right now, strictly inside the frustum) and -- if hybrid prefetch
is enabled -- which should be "RAM tier" (inside a world-space margin
around the frustum, prefetched but not yet composited).

**Chunk source data is always CPU-resident, in both tiers.** SplatRenderer
needs one single composited model (its own constructor-only precondition --
see gsplat2d_rendering's own docstring), so per-chunk data has to be
concatenated into one tensor set before it's useful for rendering regardless
of tier. Keeping each chunk as its *own* persistent GPU tensor on top of
that composited copy would mean two full copies of the resident working set
live in VRAM at once at high residency -- measured via a real stress test as
the actual cause of an otherwise-unexplained "VRAM keeps climbing -> OOM on
a model that fits fine when loaded directly" report. Now the only chunk data
that's ever GPU-resident is the one merged tensor `_stitch_fine_octree`
produces and installs into the renderer -- steady-state usage matches a
plain whole-file load instead of ~2x it. A useful side effect: since both
tiers are CPU-resident, VRAM<->RAM *tier reclassification* (a chunk crossing
from "prefetched" to "actually needed" or back) is a synchronous, zero-cost
dict move in `update()` -- no disk read, no GPU copy, no background thread
needed -- making the promised "instant CPU->GPU promotion" of the hybrid
margin tier even more literally true than it was when this still meant an
actual `.to('cuda')` copy.

Background I/O (disk reads only, now -- see above) is capped at
`_MAX_CONCURRENT_TRANSITIONS` in-flight transitions at once, not one: a
single newly-exposed frustum edge routinely uncovers several chunks in the
same frame (measured ~0.3s per chunk read against a real scene before the
ChunkedPlyReader cache; ~0.03-0.05s warm), and serializing them one-at-a-time
is exactly what shows up on screen as a chunk sitting gray for multiple
seconds while its neighbors queue up behind it. Landed transitions are
handed back via a lock-protected list (`_pending_transitions`) rather than
the single-slot "background thread writes one plain attribute" idiom used
elsewhere in viewer/app.py (_octree_pending, _ply_pending) -- that idiom
assumes one producer, and this module has several worker threads finishing
concurrently.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import threading
import time
import traceback

import numpy as np
import torch

import gsplat2d_rendering as gs2d
from gsplat2d_rendering.culling.frustum import visible_leaf_mask_torch
from gsplat2d_rendering.model import GaussianModel

_OOM_RETRY_COOLDOWN_S = 2.0
_MAX_CONCURRENT_TRANSITIONS = 3


def _move_model(model: GaussianModel, device: str) -> GaussianModel:
    """dataclasses.replace onto a new device -- mirrors viewer/ply_loader.py's
    _model_to_cuda, generalized to work in either direction (this module
    doesn't import from viewer/, which sits above renderer/ in Kestrel's
    package layering)."""
    return dataclasses.replace(
        model,
        xyz=model.xyz.to(device),
        raw_opacity=model.raw_opacity.to(device),
        raw_scaling=model.raw_scaling.to(device),
        raw_rotation=model.raw_rotation.to(device),
        features_dc=model.features_dc.to(device),
        features_rest=model.features_rest.to(device),
    )


class ChunkManager:
    def __init__(self, chunked_ply_path: str, manifest, device: str,
                 hybrid_enabled: bool = False, margin: float = 0.0,
                 fine_leaf_max: int = 5000, rebuild_throttle_s: float = 0.4):
        self.chunked_ply_path  = chunked_ply_path
        self.manifest          = manifest
        self.device            = device
        self.hybrid_enabled    = hybrid_enabled
        self.margin            = margin
        self.fine_leaf_max     = fine_leaf_max
        self.rebuild_throttle_s = rebuild_throttle_s

        self._chunk_offsets = manifest.node_offsets
        self._aabbs_gpu = torch.from_numpy(manifest.node_aabbs).to(device)
        self._aabbs_expanded_gpu = self._build_expanded_aabbs(margin, device)

        # One open mmap for the whole session instead of re-parsing the PLY
        # header and re-establishing the mmap on every single chunk read --
        # see ChunkedPlyReader's own docstring for why this matters once a
        # session issues dozens of range reads against the same file.
        self._reader = gs2d.ChunkedPlyReader(chunked_ply_path)

        # Tier classification only -- both dicts hold CPU-resident
        # GaussianModels (see module docstring for why). Mutated only from
        # the render thread (inside update()'s drain/sync-unload/
        # reclassify steps) -- background workers below never touch these
        # dicts directly, only self._pending_transitions.
        self._vram: dict[int, GaussianModel] = {}
        self._ram: dict[int, GaussianModel] = {}

        # Up to _MAX_CONCURRENT_TRANSITIONS worker threads run at once, each
        # doing a disk read (the only I/O left that needs backgrounding now
        # that tier reclassification is a synchronous dict move); _inflight
        # and _pending_transitions are the only state they touch, both
        # guarded by _pending_lock (short critical sections only -- never
        # held across a disk read).
        self._pending_lock = threading.Lock()
        self._inflight: set[int] = set()
        self._pending_transitions: list[tuple[int, str, GaussianModel | None]] = []
        self._oom_cooldown_until: dict[int, float] = {}
        self.last_oom_chunk_id: int | None = None

        self._rebuild_busy = False
        self._pending_rebuild = None      # (model, octree, frozenset(ids))
        self._last_rebuild_kick = 0.0
        self._last_rebuilt_ids: frozenset[int] = frozenset()
        self._rebuild_oom_cooldown_until = 0.0

        # Each chunk's own local fine-culling octree, built once the first
        # time that chunk's data is read from disk and cached for the rest
        # of the session -- a chunk's point set never changes, so its split
        # never needs to be recomputed just because *other* chunks entered
        # or left the composited set. See _ensure_fine_octree/_stitch_fine_octree.
        self._fine_octrees: dict[int, gs2d.Octree] = {}

    def _build_expanded_aabbs(self, margin: float, device: str):
        if not self.hybrid_enabled or margin <= 0.0:
            return None
        expanded = self.manifest.node_aabbs.copy()
        expanded[:, :3] -= margin
        expanded[:, 3:] += margin
        return torch.from_numpy(expanded).to(device)

    def set_hybrid(self, enabled: bool, margin: float) -> None:
        """Live reconfiguration from the sidebar -- margin/enable changes
        only affect which chunks *update()* asks for next frame; already
        resident chunks are left alone rather than eagerly evicted."""
        self.hybrid_enabled = enabled
        self.margin = margin
        self._aabbs_expanded_gpu = self._build_expanded_aabbs(margin, self.device)

    # ── chunk row-range I/O ──────────────────────────────────────────────────

    def _read_chunk(self, chunk_id: int) -> GaussianModel:
        """Always reads to CPU -- see module docstring; the only GPU upload
        in this whole module happens once, in _stitch_fine_octree, for the
        final composited model."""
        start = int(self._chunk_offsets[chunk_id])
        end = int(self._chunk_offsets[chunk_id + 1])
        return self._reader.read_range(start, end - start, device="cpu")

    def _distance_mask(self, camera, aabbs_gpu: torch.Tensor, max_dist: float) -> torch.Tensor:
        """Additional bound beyond visible_leaf_mask_torch's frustum test,
        which is deliberately far-clip-less (correct for rendering -- see
        CLAUDE.md's "Far plane deliberately omitted": aerial scenes have
        objects far away that must still render). Residency is a different
        question than rendering, though: a chunk manifold whose direction
        from the camera happens to align with the scene's long axis passes
        the infinite frustum test regardless of actual distance, and can
        span most of the scene -- measured on a real scene, the same fixed
        9-unit-radius orbit saw between 3 and 56 (of 66 total) chunks purely
        as a function of view direction, with no far-plane bound to stop it.
        That's what actually explains "VRAM keeps growing during a one-
        directional continuous orbit": periodically, as rotation sweeps
        through that aligned direction, the desired VRAM/RAM sets balloon
        toward the whole model. This bounds chunk *residency* to a sane
        radius around the camera regardless of view direction, independent
        of the frustum test above."""
        centers = 0.5 * (aabbs_gpu[:, :3] + aabbs_gpu[:, 3:])
        cam_pos = camera.camera_center.to(aabbs_gpu.device)
        dists = (centers - cam_pos).norm(dim=-1)
        return dists <= max_dist

    # ── startup ──────────────────────────────────────────────────────────────

    def initial_sync_load(self, camera, max_load_distance: float | None = None):
        """Blocking -- used once at startup (or when chunk streaming is
        toggled on mid-session), same category as today's synchronous
        --build-index. Falls back to the single nearest chunk if the
        starting camera pose doesn't intersect any (so the scene is never
        blank at launch). Reads the initial chunk set in parallel (bounded
        by _MAX_CONCURRENT_TRANSITIONS) -- the same reasoning as the
        per-frame transition pool below applies here too: a wide starting
        FOV can cover several chunks at once, and reading them one at a
        time would directly extend startup latency."""
        mask = visible_leaf_mask_torch(self._aabbs_gpu, camera.full_proj_transform)
        if max_load_distance is not None:
            mask = mask & self._distance_mask(camera, self._aabbs_gpu, max_load_distance)
        ids = torch.nonzero(mask, as_tuple=True)[0].tolist()
        if not ids:
            centers = 0.5 * (self.manifest.node_aabbs[:, :3] + self.manifest.node_aabbs[:, 3:])
            cam_pos = camera.camera_center.detach().cpu().numpy()
            dists = ((centers - cam_pos) ** 2).sum(axis=1)
            ids = [int(dists.argmin())]

        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_TRANSITIONS) as pool:
            models = list(pool.map(lambda cid: self._read_chunk(cid), ids))
        for cid, model in zip(ids, models):
            self._vram[cid] = model
            self._ensure_fine_octree(cid, model)

        return self._rebuild_fine_octree_sync()

    def _ensure_fine_octree(self, chunk_id: int, model: GaussianModel) -> None:
        """Builds and caches chunk_id's own local fine-culling octree the
        first time its data is read from disk. A chunk's point set never
        changes for the life of a ChunkManager session, so this only ever
        runs once per chunk_id no matter how many times it later cycles
        between tiers -- the expensive recursive split that used to re-run
        over the *entire* resident set on every composition change (an
        O(total resident points) cost that grew as more chunks loaded) now
        happens once, sized to a single chunk (~1e5-1e6 points), and gets
        parallelized for free across the transition worker pool instead of
        being serialized into one big merge."""
        if chunk_id in self._fine_octrees:
            return
        xyz_np = model.xyz.float().cpu().numpy()
        # verbose_log=True: this fires once per chunk (up to hundreds per
        # session), not once per model load -- see build_octree's own
        # docstring for why that's the dividing line for NORMAL vs VERBOSE.
        self._fine_octrees[chunk_id] = gs2d.build_octree(
            xyz_np, leaf_max=self.fine_leaf_max, verbose_log=True,
        )

    def _stitch_fine_octree(self, snapshot: dict[int, GaussianModel]):
        """Assembles a combined fine octree + composited model over
        `snapshot` (CPU-resident chunk models) purely by concatenating each
        chunk's already-built local octree (see _ensure_fine_octree) --
        small per-chunk node_aabbs/flat_indices arrays, shifting
        flat_indices by a running point offset -- instead of re-running the
        O(total points) recursive split every time the composition changes.
        Octree-stitch cost scales with total leaf-node count across
        resident chunks (hundreds), not total point count (millions).

        The concatenation itself (gs2d.concat_gaussian_models) runs on the
        CPU-resident inputs, and the *single* resulting composited model is
        moved to GPU as one transfer at the end -- the only GPU upload
        chunk data ever goes through, replacing per-chunk GPU tensors that
        would otherwise double steady-state VRAM at high residency (see
        module docstring)."""
        ids = list(snapshot.keys())
        models = [snapshot[cid] for cid in ids]
        merged_cpu = gs2d.concat_gaussian_models(models)

        aabb_parts, idx_parts, size_parts = [], [], []
        point_offset = 0
        for cid, model in zip(ids, models):
            octree = self._fine_octrees[cid]
            aabb_parts.append(octree.node_aabbs)
            idx_parts.append(octree.flat_indices + point_offset)
            size_parts.append(np.diff(octree.node_offsets))
            point_offset += model.xyz.shape[0]

        node_aabbs = np.concatenate(aabb_parts, axis=0) if aabb_parts else np.zeros((0, 6), dtype=np.float32)
        flat_indices = np.concatenate(idx_parts, axis=0) if idx_parts else np.zeros(0, dtype=np.int64)
        node_sizes = np.concatenate(size_parts, axis=0) if size_parts else np.zeros(0, dtype=np.int64)
        node_offsets = np.zeros(len(node_sizes) + 1, dtype=np.int64)
        np.cumsum(node_sizes, out=node_offsets[1:])

        stitched = gs2d.Octree(node_aabbs=node_aabbs, node_offsets=node_offsets, flat_indices=flat_indices)

        merged = _move_model(merged_cpu, self.device)
        del merged_cpu
        return merged, stitched

    def _rebuild_fine_octree_sync(self):
        merged, fine_octree = self._stitch_fine_octree(dict(self._vram))
        self._last_rebuilt_ids = frozenset(self._vram.keys())
        return merged, fine_octree

    # ── per-frame tier diff ─────────────────────────────────────────────────

    def update(self, camera, max_load_distance: float | None = None) -> None:
        """Non-blocking. Call once per frame (render thread) with the
        current render camera. Recomputes desired VRAM/RAM sets, applies
        free synchronous tier reclassification and full-unloads, and tops
        up the disk-read transition pool up to _MAX_CONCURRENT_TRANSITIONS
        in-flight workers. Also kicks a time-throttled fine-octree rebuild
        (stitched from cached per-chunk octrees, see _stitch_fine_octree)
        whenever the VRAM-tier set has changed since the last landed
        rebuild -- purely time-gated, not gated on the transition pool being
        idle, since under continuous camera motion the pool may never
        actually go idle.

        max_load_distance additionally bounds both tiers to a sane radius
        around the camera -- see _distance_mask for why the frustum test
        alone (deliberately far-clip-less) isn't enough for a residency
        decision. None (default) disables the bound, matching the frustum
        test's own behavior."""
        self._drain_transitions()

        proj = camera.full_proj_transform
        vram_mask = visible_leaf_mask_torch(self._aabbs_gpu, proj)
        if max_load_distance is not None:
            vram_mask = vram_mask & self._distance_mask(camera, self._aabbs_gpu, max_load_distance)
        desired_vram = set(torch.nonzero(vram_mask, as_tuple=True)[0].tolist())

        desired_ram: set[int] = set()
        if self.hybrid_enabled and self._aabbs_expanded_gpu is not None:
            margin_mask = visible_leaf_mask_torch(self._aabbs_expanded_gpu, proj)
            if max_load_distance is not None:
                margin_mask = margin_mask & self._distance_mask(
                    camera, self._aabbs_expanded_gpu, max_load_distance,
                )
            desired_ram = set(torch.nonzero(margin_mask, as_tuple=True)[0].tolist()) - desired_vram

        # Instant, synchronous tier reclassification for chunks whose data
        # is already CPU-resident in the other tier -- no disk I/O, no GPU
        # copy (chunk source data lives on CPU in both tiers, see module
        # docstring), so there's no reason to route this through the
        # background worker pool at all.
        for cid in list(desired_vram & self._ram.keys()):
            self._vram[cid] = self._ram.pop(cid)
        for cid in list((set(self._vram.keys()) - desired_vram) & desired_ram):
            self._ram[cid] = self._vram.pop(cid)

        current_vram = set(self._vram.keys())
        current_ram = set(self._ram.keys())

        # Free, synchronous full-unload: a chunk that fell out of both the
        # strict frustum and the margin ring needs no thread, just a dict
        # drop -- GC reclaims its (CPU) tensors. Chunks currently in flight
        # are excluded since a disk-read worker may be about to land a
        # result for them.
        for cid in current_vram - desired_vram - desired_ram - self._inflight:
            del self._vram[cid]
        for cid in current_ram - desired_ram - desired_vram - self._inflight:
            del self._ram[cid]

        self._maybe_spawn_transitions(desired_vram, desired_ram)

        # Purely time-throttled, not gated on the transition pool being idle:
        # under continuous camera motion a freed worker slot is immediately
        # refilled by the next newly-visible chunk, so _inflight can stay
        # non-empty indefinitely -- gating on "settled" here would mean
        # chunks finish loading into VRAM-tier in the background but never
        # get folded into what's actually rendered until motion stops.
        if set(self._vram.keys()) != self._last_rebuilt_ids:
            self._maybe_kick_rebuild()

    def _maybe_spawn_transitions(self, desired_vram: set[int], desired_ram: set[int]) -> None:
        """Only genuine disk reads reach here now -- tier reclassification
        between already-CPU-resident chunks is handled synchronously in
        update() above, before this is called."""
        now = time.monotonic()
        current_vram = set(self._vram.keys())
        current_ram = set(self._ram.keys())

        def cooled_down(cid: int) -> bool:
            return now >= self._oom_cooldown_until.get(cid, 0.0)

        def slot_available() -> bool:
            return len(self._inflight) < _MAX_CONCURRENT_TRANSITIONS

        ram_candidates = [
            cid for cid in desired_ram - current_ram - current_vram - self._inflight
            if cooled_down(cid)
        ]

        # Priority order: promotions to VRAM tier (what's actually rendered)
        # before RAM-tier prefetch -- matches "closer to the camera's actual
        # view wins" without needing a distance sort. At least one worker
        # slot is reserved for RAM prefetch whenever there's RAM demand --
        # otherwise, under continuous rotation, a steady stream of
        # newly-visible VRAM-tier chunks claims every slot every frame and
        # the margin buffer never gets populated ahead of need, silently
        # turning hybrid prefetch into a no-op under exactly the
        # sustained-motion case it exists for.
        vram_candidates = [
            cid for cid in desired_vram - current_vram - self._inflight
            if cooled_down(cid)
        ]
        vram_slot_cap = (_MAX_CONCURRENT_TRANSITIONS - 1) if ram_candidates else _MAX_CONCURRENT_TRANSITIONS
        for i, cid in enumerate(vram_candidates):
            if i >= vram_slot_cap or not slot_available():
                break
            self._spawn_transition(cid, "to_vram_from_disk")

        for cid in ram_candidates:
            if not slot_available():
                return
            self._spawn_transition(cid, "to_ram_from_disk")

    def _spawn_transition(self, chunk_id: int, kind: str) -> None:
        self._inflight.add(chunk_id)
        threading.Thread(target=self._transition_worker, args=(chunk_id, kind), daemon=True).start()

    def _transition_worker(self, chunk_id: int, kind: str) -> None:
        try:
            if kind in ("to_vram_from_disk", "to_ram_from_disk"):
                model = self._read_chunk(chunk_id)
                self._ensure_fine_octree(chunk_id, model)
            else:
                raise ValueError(f"unknown transition kind: {kind}")
            result = (chunk_id, kind, model)
        except MemoryError:
            # Host-RAM OOM reading this chunk's rows -- same cooldown/skip
            # treatment as a CUDA OOM used to get, just no empty_cache() to
            # call (nothing here ever touched the GPU).
            self._oom_cooldown_until[chunk_id] = time.monotonic() + _OOM_RETRY_COOLDOWN_S
            self.last_oom_chunk_id = chunk_id
            result = (chunk_id, kind, None)
        except Exception:
            traceback.print_exc()
            result = (chunk_id, kind, None)
        with self._pending_lock:
            self._pending_transitions.append(result)
            self._inflight.discard(chunk_id)

    def _drain_transitions(self) -> None:
        with self._pending_lock:
            pending = self._pending_transitions
            self._pending_transitions = []
        for chunk_id, kind, model in pending:
            if model is None:
                continue  # failed (OOM or otherwise) -- leave prior state as-is
            if kind == "to_vram_from_disk":
                self._vram[chunk_id] = model
            elif kind == "to_ram_from_disk":
                self._ram[chunk_id] = model

    # ── fine-grained GPU-culling octree rebuild ─────────────────────────────

    def _maybe_kick_rebuild(self) -> None:
        if self._rebuild_busy or not self._vram:
            return
        now = time.monotonic()
        if now - self._last_rebuild_kick < self.rebuild_throttle_s:
            return
        if now < self._rebuild_oom_cooldown_until:
            return
        self._rebuild_busy = True
        self._last_rebuild_kick = now
        snapshot = dict(self._vram)
        threading.Thread(target=self._rebuild_worker, args=(snapshot,), daemon=True).start()

    def _rebuild_worker(self, snapshot: dict[int, GaussianModel]) -> None:
        try:
            merged, fine_octree = self._stitch_fine_octree(snapshot)
            self._pending_rebuild = (merged, fine_octree, frozenset(snapshot.keys()))
        except torch.cuda.OutOfMemoryError:
            # The one place chunk data ever touches the GPU now -- the final
            # composited model's H2D upload. Not tied to one specific chunk
            # (the whole composition didn't fit), so there's no single chunk
            # to cool down; back off retrying the rebuild itself instead,
            # and leave the renderer showing its last successfully landed
            # (smaller) composition rather than crashing.
            torch.cuda.empty_cache()
            self._rebuild_oom_cooldown_until = time.monotonic() + _OOM_RETRY_COOLDOWN_S
            self.last_oom_chunk_id = -1
            traceback.print_exc()
        except Exception:
            traceback.print_exc()
        finally:
            self._rebuild_busy = False

    def drain_pending_swap(self):
        """Drained once per frame from _render_loop, exactly where
        _octree_pending/_ply_pending are drained today. Returns
        (gaussian_model, octree) for ViewerRenderer, or None."""
        pending = self._pending_rebuild
        if pending is None:
            return None
        self._pending_rebuild = None
        model, octree, ids = pending
        self._last_rebuilt_ids = ids
        return model, octree

    # ── debug / sidebar introspection ───────────────────────────────────────

    def visible_chunk_ids(self, camera) -> set[int]:
        """Which chunks intersect `camera`'s frustum -- independent of
        residency. Used by the dual-camera-debug overlay to bound which
        disk-tier chunk boxes are even worth drawing (see viewer/app.py's
        _render_loop), keeping overlay cost independent of total chunk count."""
        mask = visible_leaf_mask_torch(self._aabbs_gpu, camera.full_proj_transform)
        return set(torch.nonzero(mask, as_tuple=True)[0].tolist())

    def chunk_states(self) -> dict[int, str]:
        """chunk_id -> 'vram' | 'ram'. Absence means disk-only. Names refer
        to tier ("should be in the rendered composition" vs "prefetched"),
        not literal current GPU/CPU residency of that chunk's own tensor --
        see module docstring."""
        states = {cid: "vram" for cid in self._vram}
        for cid in self._ram:
            states.setdefault(cid, "ram")
        return states

    def stats(self) -> tuple[int, int, int]:
        return len(self._vram), len(self._ram), len(self.manifest.node_aabbs)
