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


_ADJACENCY_K = 6  # neighbors per chunk in the K-NN adjacency graph -- see
                  # _build_adjacency's own docstring for why this, not a
                  # true "shares a boundary" adjacency, is what's used


class ChunkManager:
    def __init__(self, chunked_ply_path: str, manifest, device: str,
                 vram_margin_hops: int = 0, ram_margin_hops: int = 0,
                 fine_leaf_max: int = 5000, rebuild_throttle_s: float = 0.4,
                 compression_level: int = 0):
        self.chunked_ply_path  = chunked_ply_path
        self.manifest          = manifest
        self.device            = device
        # Adjacency-hop-count residency, replacing an earlier world-space
        # AABB-expansion-margin design -- see _build_adjacency/_bfs_expand
        # and module docstring for the full reasoning. 0 hops means "off"
        # for either tier (no separate enable/disable flag needed, same
        # "0 is a well-defined way to turn a feature off" idiom already
        # used elsewhere, e.g. chunk_size=0/huge in the sidebar).
        self.vram_margin_hops  = vram_margin_hops
        self.ram_margin_hops   = ram_margin_hops
        self.fine_leaf_max     = fine_leaf_max
        self.rebuild_throttle_s = rebuild_throttle_s

        # K-nearest-neighbor adjacency graph over chunk centroids, built
        # once (depends only on the coarse manifest, fixed for the
        # session). NOT true "shares a boundary" adjacency: build_octree's
        # leaves are *tight* AABBs around each chunk's actual points, not
        # padded to the octant they were split from, so there are real gaps
        # between neighboring chunks (measured directly: a naive "AABBs
        # touch within an epsilon" test left 35% of chunks with zero
        # neighbors on a real scene). Fixed-K-nearest-by-centroid instead
        # guarantees every chunk has exactly K neighbors regardless of its
        # own size or local density -- not "true" adjacency (a chunk's
        # K-nearest aren't necessarily chunks that actually border it), but
        # avoids both the orphan problem above and the opposite failure
        # mode (a size-relative distance threshold let a single large
        # sparse chunk become "adjacent" to dozens of others).
        self._adjacency = self._build_adjacency(manifest.node_aabbs, _ADJACENCY_K)

        # Populated by update()/initial_sync_load() each time they run --
        # chunk_states() reads these instead of recomputing frustum+BFS
        # itself, since the debug overlay's per-frame call already happens
        # right after update() in the same frame (see viewer/app.py).
        self._last_desired_vram: frozenset[int] = frozenset()
        self._last_desired_vram_margin: frozenset[int] = frozenset()
        # gsplat2d_rendering's own in-memory compression_level (0=none,
        # 1=fp16, 2=fp16+SH-degree-1, 3+=fp16+SH-degree-0) applied per chunk
        # read via ChunkedPlyReader.read_range -- exactly the same semantics
        # load_gaussian_model already uses for non-chunked mode's L0/L1/L2.
        # Kestrel's own offline L3 (utils/compress.py's compress_level3)
        # additionally int8-packs rotation/normals, which this in-memory
        # path has no equivalent for -- chunk streaming's "L3" is therefore
        # fp16+SH0 (identical to library level 3+), not bit-identical to
        # non-chunked L3. See set_compression for why a level change forces
        # a full re-read of every resident chunk.
        self.compression_level = compression_level

        self._chunk_offsets = manifest.node_offsets
        self._aabbs_gpu = torch.from_numpy(manifest.node_aabbs).to(device)

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
        self._pending_transitions: list[tuple[int, str, GaussianModel | None, int]] = []
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
        # The permutation each chunk's octree implies, cached separately from
        # the octree itself (whose own flat_indices gets rewritten to
        # identity once the permutation has been applied) -- re-applied
        # every time this chunk_id is read from disk again after an eviction,
        # since a fresh read is a fresh, still-unordered GaussianModel each
        # time, even though the split computation itself never needs redoing.
        self._chunk_reorder_perm: dict[int, torch.Tensor] = {}

    @staticmethod
    def _build_adjacency(aabbs: np.ndarray, k: int) -> dict[int, list[int]]:
        """K-nearest-neighbor graph over chunk AABB centroids -- see class
        docstring/__init__ for why this, not a "shares a boundary" test, is
        used. O(N^2) pairwise distances: fine for the tens-to-low-hundreds
        of chunks a sane chunk_size produces, not meant to scale to the
        thousands of chunks an extremely fine chunk_size produces (already
        documented elsewhere as outside this feature's intended operating
        range)."""
        centers = 0.5 * (aabbs[:, :3] + aabbs[:, 3:])
        n = len(centers)
        k = min(k, n - 1)
        if k <= 0:
            return {i: [] for i in range(n)}
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        nearest = np.argpartition(dists, k - 1, axis=1)[:, :k]
        return {i: nearest[i].tolist() for i in range(n)}

    def _bfs_expand(self, seed: set[int], hops: int) -> set[int]:
        """All chunk ids reachable from any chunk in `seed` within `hops`
        adjacency steps, including `seed` itself. hops<=0 returns `seed`
        unchanged (0 hops = the margin/bound is effectively off)."""
        visited = set(seed)
        frontier = set(seed)
        for _ in range(max(hops, 0)):
            next_frontier: set[int] = set()
            for cid in frontier:
                next_frontier.update(self._adjacency.get(cid, ()))
            next_frontier -= visited
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier
        return visited

    def _nearest_chunk_id(self, camera, anchor: np.ndarray | None = None) -> int:
        """The single chunk whose centroid is closest to `anchor` (a world-
        space xyz point) -- used as initial_sync_load's/update()'s fallback
        so the scene is never blank when the starting/current pose doesn't
        intersect any chunk's frustum test.

        `anchor` should be the scene point the camera is actually looking
        at (viewer/app.py passes OrbitCamera.look_at), NOT `camera.
        camera_center` (the camera's raw position) -- an orbit camera sits
        `distance` away from what it's actually pointed at, often well
        outside the model entirely, so "nearest chunk to camera_center" can
        land in a totally different, frustum-disconnected part of the scene
        from what's actually in view. Verified directly: on a real scene,
        the chunk nearest to camera_center had zero adjacency-graph overlap
        within 3 hops with the actual frustum-visible set, even though
        their centroids were only 3-9 world units apart -- an unrelated
        KNN-graph-connectivity artifact, not a real distance problem, that
        only surfaced because the anchor itself was wrong. Falls back to
        `camera.camera_center` only if no better anchor is available (kept
        for callers that can't supply one; strictly worse for an orbiting
        viewer, see above)."""
        centers = 0.5 * (self.manifest.node_aabbs[:, :3] + self.manifest.node_aabbs[:, 3:])
        if anchor is None:
            anchor = camera.camera_center.detach().cpu().numpy()
        dists = ((centers - np.asarray(anchor)) ** 2).sum(axis=1)
        return int(dists.argmin())

    def set_margins(self, vram_margin_hops: int, ram_margin_hops: int) -> None:
        """Live reconfiguration from the sidebar -- changes only affect
        which chunks *update()* asks for next frame; already-resident
        chunks are left alone rather than eagerly evicted."""
        self.vram_margin_hops = vram_margin_hops
        self.ram_margin_hops = ram_margin_hops

    def set_compression(self, level: int) -> None:
        """Live compression-level change from the sidebar. Unlike
        set_margins, this can't leave already-resident chunks alone: their
        cached tensors are at the *old* level's precision/SH-degree, and
        gsplat2d_rendering.concat_gaussian_models deliberately raises on a
        mismatched SH degree across inputs -- its own guard against exactly
        this "chunks from different compression levels" scenario. So this
        evicts every resident chunk outright (both tiers) and clears the
        per-chunk octree/permutation caches (conservative but correct: fp16
        rounding could in principle shift a borderline point across an
        octant boundary, so a stale split isn't safe to keep either) --
        update()'s normal transition machinery then re-reads everything
        fresh at the new level, exactly like a live-panned-in chunk that was
        never seen before. last_rebuilt_ids is reset so the resulting
        composition change is detected as one once new data lands."""
        if level == self.compression_level:
            return
        self.compression_level = level
        self._vram.clear()
        self._ram.clear()
        self._fine_octrees.clear()
        self._chunk_reorder_perm.clear()
        self._last_rebuilt_ids = frozenset()

    # ── chunk row-range I/O ──────────────────────────────────────────────────

    def _read_chunk(self, chunk_id: int) -> GaussianModel:
        """Always reads to CPU -- see module docstring; the only GPU upload
        in this whole module happens once, in _stitch_fine_octree, for the
        final composited model. compression_level/target_sh_degree=1 apply
        gsplat2d_rendering's own in-memory compression per read, same as
        non-chunked mode's load_gaussian_model(compression_level=N) -- see
        set_compression for how a level change is propagated to already-
        resident chunks."""
        start = int(self._chunk_offsets[chunk_id])
        end = int(self._chunk_offsets[chunk_id + 1])
        return self._reader.read_range(
            start, end - start, device="cpu",
            compression_level=self.compression_level, target_sh_degree=1,
        )

    # ── startup ──────────────────────────────────────────────────────────────

    def initial_sync_load(self, camera, anchor: np.ndarray | None = None):
        """Blocking -- used once at startup (or when chunk streaming is
        toggled on mid-session), same category as today's synchronous
        --build-index. Falls back to the single nearest-to-`anchor` chunk if
        the starting camera pose doesn't intersect any (so the scene is
        never blank at launch) -- see _nearest_chunk_id's docstring for why
        `anchor` should be the look-at point, not the raw camera position.
        Reads the initial chunk set in parallel (bounded by
        _MAX_CONCURRENT_TRANSITIONS) -- the same reasoning as the per-frame
        transition pool below applies here too: a wide starting FOV can
        cover several chunks at once, and reading them one at a time would
        directly extend startup latency.

        No `max_load_hops` parameter here (unlike update()): this only ever
        loads the strict frustum-visible set, with no margin expansion --
        bounding a set by hops from itself is a no-op (BFS always contains
        its own seed), so there's nothing for max_load_hops to actually cap
        until update()'s margin expansion exists to bound."""
        mask = visible_leaf_mask_torch(self._aabbs_gpu, camera.full_proj_transform)
        ids = torch.nonzero(mask, as_tuple=True)[0].tolist()
        if not ids:
            ids = [self._nearest_chunk_id(camera, anchor)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_TRANSITIONS) as pool:
            models = list(pool.map(lambda cid: self._read_chunk(cid), ids))
        for cid, model in zip(ids, models):
            self._vram[cid] = model
            self._ensure_fine_octree(cid, model)

        self._last_desired_vram = frozenset(ids)
        self._last_desired_vram_margin = frozenset()
        return self._rebuild_fine_octree_sync()

    def _ensure_fine_octree(self, chunk_id: int, model: GaussianModel) -> None:
        """Builds and caches chunk_id's own local fine-culling octree the
        first time its data is read from disk, and permutes `model` (the
        same object about to be stored in _vram/_ram) into that octree's own
        leaf-contiguous order.

        Two *different* things are cached vs. repeated here, and conflating
        them is a real correctness bug, not just a missed optimization: the
        octree *structure* (self._fine_octrees[chunk_id]) is genuinely safe
        to build only once, since a chunk's point *set* never changes for
        the life of a session -- the expensive recursive split that used to
        re-run over the *entire* resident set on every composition change
        now happens once, sized to a single chunk, and gets parallelized for
        free across the transition worker pool. But the *reorder* must be
        (re-)applied on every call, not just the first: `model` is a fresh,
        still-disk-order `GaussianModel` every single time this chunk is
        actually read from disk, and a chunk can be read from disk more than
        once per session (evicted from both tiers, then later re-entering
        residency) -- gating the reorder behind the same "already cached"
        check as the octree build left re-read data silently un-reordered
        while the cached octree's flat_indices (already rewritten to
        identity below) claimed it wasn't. Caught directly: instrumenting
        _ensure_fine_octree during a hybrid-margin tier-cycling stress test
        and asserting a chunk's stored tensor never changes across a cache
        hit -- it did, on the first re-read of an evicted chunk.

        The permutation itself (`self._chunk_reorder_perm[chunk_id]`) is
        cached once alongside the octree and reapplied on every read, since
        recomputing the split via `build_octree` is the expensive part, not
        re-applying an already-known permutation to fresh tensors.

        Baking this reorder in per-chunk, at read time, instead of on the
        whole composited model on every rebuild (the old design) is why
        _stitch_fine_octree's concatenation of already-locally-ordered
        chunks is *itself* already globally leaf-contiguous (each chunk
        contributes a contiguous run of leaves, in the same order its own
        rows now sit in) -- see viewer/app.py's chunk-streaming install
        sites, which set `_spatially_ordered = True` instead of re-deriving
        this via a full O(total composited points) reorder every rebuild.
        This also moves the cost off the render thread entirely: it runs
        here, inside the same background transition-worker call that
        already pays for the chunk's disk read (tens to hundreds of ms), so
        it's effectively free in context, not merely cheaper in isolation."""
        if chunk_id not in self._fine_octrees:
            xyz_np = model.xyz.float().cpu().numpy()
            # verbose_log=True: this fires once per chunk (up to hundreds
            # per session), not once per model load -- see build_octree's
            # own docstring for why that's the NORMAL/VERBOSE dividing line.
            local_octree = gs2d.build_octree(
                xyz_np, leaf_max=self.fine_leaf_max, verbose_log=True,
            )
            self._chunk_reorder_perm[chunk_id] = torch.from_numpy(
                local_octree.flat_indices.copy()
            ).to(model.xyz.device)
            # node_aabbs/node_offsets are unaffected by row order (per-leaf
            # bounds/counts, not row indices) -- only flat_indices needs
            # rewriting, to identity, since every future read of this
            # chunk_id will already be reordered by the line below before
            # _stitch_fine_octree ever sees it.
            local_octree.flat_indices = np.arange(model.xyz.shape[0], dtype=np.int64)
            self._fine_octrees[chunk_id] = local_octree

        model.reorder_(self._chunk_reorder_perm[chunk_id], verbose_log=True)

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

    def update(self, camera, anchor: np.ndarray | None = None,
               max_load_hops: int | None = None) -> None:
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

        Three concepts, all adjacency-hop-based (see _bfs_expand/module
        docstring), each answering a different question:
        - max_load_hops bounds *margin reach*: chunks more than this many
          adjacency-hops from the strictly-visible set are never considered
          for VRAM-margin/RAM-tier residency, regardless of vram_margin_hops/
          ram_margin_hops -- replaces an earlier world-distance bound that
          existed for a similar reason (the frustum test is deliberately
          far-clip-less, correct for rendering but not for bounding
          residency growth). Seeded from the strictly-visible set itself, so
          it can only ever cap *additional* reach -- it never excludes a
          chunk the frustum test already found, unlike gating from a single
          "nearest to camera" anchor (tried and reverted: a chunk's K-NN-hop
          neighborhood doesn't reliably reach chunks that are genuinely
          nearby in world space, so filtering the strict set itself this way
          could -- and on a real scene, did -- silently evict everything
          actually on screen). None (default) disables the bound.
        - vram_margin_hops expands *outward from the strictly-visible set*:
          chunks this many hops from a frustum-visible chunk are also
          promoted to actual VRAM residency (composited into the rendered
          model) even though they're outside the frustum right now -- so
          minor camera rotation across this margin needs zero rebuild
          latency, since the fine per-frame GPU cull (already free) is what
          then hides/shows them, not a chunk-composition rebuild.
        - ram_margin_hops expands *outward from the VRAM tier* (strict +
          margin) into CPU-only prefetch, same role the old world-space
          margin played, just hop-based instead of distance-based now.
        Both margins default to 0 (off) -- no separate enable/disable flag
        needed, matching chunk_size=0's own "0 is a well-defined off state"
        idiom elsewhere in this codebase.

        `anchor` (world-space xyz, e.g. OrbitCamera.look_at) is only used as
        a fallback seed when the frustum test finds nothing at all (so
        residency never goes fully empty on a degenerate camera pose) -- see
        _nearest_chunk_id's docstring for why this must be the look-at point,
        not `camera.camera_center`."""
        self._drain_transitions()

        proj = camera.full_proj_transform
        frustum_mask = visible_leaf_mask_torch(self._aabbs_gpu, proj)
        frustum_visible = set(torch.nonzero(frustum_mask, as_tuple=True)[0].tolist())

        if not frustum_visible:
            frustum_visible = {self._nearest_chunk_id(camera, anchor)}

        if max_load_hops is not None:
            # Seeded from the visible set itself -- BFS always contains its
            # own seed, so this can only ever ADD reach for the margins
            # below, never exclude a chunk the frustum test already found.
            reachable = self._bfs_expand(frustum_visible, max_load_hops)
        else:
            reachable = None

        desired_vram_strict = frustum_visible
        vram_expanded = self._bfs_expand(desired_vram_strict, self.vram_margin_hops)
        if reachable is not None:
            vram_expanded &= reachable
        desired_vram_margin = vram_expanded - desired_vram_strict
        desired_vram = desired_vram_strict | desired_vram_margin

        ram_expanded = self._bfs_expand(desired_vram, self.ram_margin_hops)
        if reachable is not None:
            ram_expanded &= reachable
        desired_ram = ram_expanded - desired_vram

        # Read by chunk_states() for the debug overlay -- computed here,
        # once per frame, rather than recomputed redundantly there.
        self._last_desired_vram = frozenset(desired_vram_strict)
        self._last_desired_vram_margin = frozenset(desired_vram_margin)

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
        # Captured at spawn time, not completion: if set_compression() runs
        # while this read is in flight, the read was for the *old* level --
        # _drain_transitions compares this against the (possibly by-then-
        # different) current compression_level and discards a stale result,
        # rather than letting an old-precision chunk land back into a
        # freshly-evicted, new-level composition.
        level_at_spawn = self.compression_level
        try:
            if kind in ("to_vram_from_disk", "to_ram_from_disk"):
                model = self._read_chunk(chunk_id)
                self._ensure_fine_octree(chunk_id, model)
            else:
                raise ValueError(f"unknown transition kind: {kind}")
            result = (chunk_id, kind, model, level_at_spawn)
        except MemoryError:
            # Host-RAM OOM reading this chunk's rows -- same cooldown/skip
            # treatment as a CUDA OOM used to get, just no empty_cache() to
            # call (nothing here ever touched the GPU).
            self._oom_cooldown_until[chunk_id] = time.monotonic() + _OOM_RETRY_COOLDOWN_S
            self.last_oom_chunk_id = chunk_id
            result = (chunk_id, kind, None, level_at_spawn)
        except Exception:
            traceback.print_exc()
            result = (chunk_id, kind, None, level_at_spawn)
        with self._pending_lock:
            self._pending_transitions.append(result)
            self._inflight.discard(chunk_id)

    def _drain_transitions(self) -> None:
        with self._pending_lock:
            pending = self._pending_transitions
            self._pending_transitions = []
        for chunk_id, kind, model, level in pending:
            if model is None:
                continue  # failed (OOM or otherwise) -- leave prior state as-is
            if level != self.compression_level:
                continue  # stale: set_compression() ran while this read was in flight
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
        """chunk_id -> 'vram' | 'vram_margin' | 'ram'. Absence means
        disk-only. 'vram' = strictly frustum-visible (the composition
        actually rendered); 'vram_margin' = adjacency-promoted into that
        same GPU-resident composition despite being outside the frustum
        right now (see update()'s vram_margin_hops) -- fine per-frame GPU
        culling hides it until the camera turns enough to bring it back
        into frame, with zero rebuild needed; 'ram' = CPU-only prefetch.
        Reflects the most recent update() call's classification (self.
        _last_desired_vram/_last_desired_vram_margin), not literal current
        GPU/CPU residency of that chunk's own tensor -- see module
        docstring. All three are drawn differently by the dual-camera-debug
        chunk-border overlay."""
        states = {}
        for cid in self._vram:
            states[cid] = "vram_margin" if cid in self._last_desired_vram_margin else "vram"
        for cid in self._ram:
            states.setdefault(cid, "ram")
        return states

    def stats(self) -> tuple[int, int, int]:
        return len(self._vram), len(self._ram), len(self.manifest.node_aabbs)
