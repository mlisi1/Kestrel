"""CPU-side octree frustum culling, decoupled from ViewerRenderer."""

import numpy as np
import torch


def frustum_cull_mask(
    octree: dict,
    is_in_box: torch.Tensor,
    viewpoint_camera,
    spatially_ordered: bool,
    all_ids: torch.Tensor,
) -> tuple[torch.Tensor, list | None]:
    """
    Narrow *is_in_box* to splats whose octree leaf intersects the view frustum.

    Returns
    -------
    bool_mask   : torch.Tensor [N, bool]  — visible splat mask
    vis_ranges  : list[(start, end)] | None
        Contiguous slice pairs when spatially ordered and no crop box is active.
        None whenever a per-splat boolean gather is required instead.

    Notes
    -----
    Uses 5 planes (left / right / top / bottom / near). The far plane is
    deliberately omitted: the GS rasterizer does not hard-clip at zfar, so
    scenes with objects far away would be incorrectly culled.

    Matrix convention: p_clip = p_world @ full_proj_transform  (row-vector).
    Planes are extracted from the columns of M (Gribb-Hartmann method).
    """
    M      = viewpoint_camera.full_proj_transform.detach().cpu().numpy()  # [4,4]
    planes = np.stack([
        M[:, 0] + M[:, 3],   # left
        M[:, 3] - M[:, 0],   # right
        M[:, 1] + M[:, 3],   # bottom
        M[:, 3] - M[:, 1],   # top
        M[:, 2],              # near (camera-Z >= znear)
    ], axis=0)                # [5, 4]

    normals = planes[:, :3]   # [5, 3]
    d_vals  = planes[:, 3]    # [5]

    node_aabbs   = octree["node_aabbs"]    # float32 [L, 6]
    node_offsets = octree["node_offsets"]  # int64   [L+1]

    aabb_min = node_aabbs[:, :3]   # [L, 3]
    aabb_max = node_aabbs[:, 3:]   # [L, 3]

    # Gribb-Hartmann p-vertex test, vectorised over all L nodes.
    pos_mask = normals[:, np.newaxis, :] >= 0
    p_vert   = np.where(pos_mask, aabb_max[np.newaxis], aabb_min[np.newaxis])  # [5,L,3]
    dots     = (p_vert * normals[:, np.newaxis, :]).sum(axis=2) + d_vals[:, np.newaxis]
    node_vis = (dots >= 0).all(axis=0)   # [L]

    N_total       = is_in_box.shape[0]
    vis_cpu       = np.zeros(N_total, dtype=np.bool_)
    visible_nodes = np.where(node_vis)[0]
    vis_ranges    = None

    if len(visible_nodes) > 0:
        starts = node_offsets[visible_nodes]
        ends   = node_offsets[visible_nodes + 1]

        if spatially_ordered:
            # Leaf j → contiguous [node_offsets[j], node_offsets[j+1]):
            # use direct slice assignment (numpy memset).
            for s, e in zip(starts, ends):
                vis_cpu[s:e] = True
            # Expose ranges for contiguous GPU gather only when no crop box is active.
            if is_in_box is all_ids:
                vis_ranges = list(zip(starts.tolist(), ends.tolist()))
        else:
            flat_indices = octree["flat_indices"]
            all_idx = np.concatenate([flat_indices[s:e] for s, e in zip(starts, ends)])
            vis_cpu[all_idx] = True

    bool_mask = is_in_box & torch.from_numpy(vis_cpu).to(is_in_box.device)
    return bool_mask, vis_ranges
