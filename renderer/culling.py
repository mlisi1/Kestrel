"""CPU-side octree frustum culling, decoupled from ViewerRenderer.

The frustum-plane test itself (Gribb-Hartmann, matrix convention, far-plane
omission) is delegated to gsplat2d_rendering.culling.visible_leaf_mask_torch
so Kestrel doesn't carry its own copy of that math. What's left here is
Kestrel-specific: turning the per-leaf visibility mask into either a bool
mask over all N splats, or (fast path) a list of contiguous (start, end)
ranges when the model has been spatially reordered to match the octree.
"""

import numpy as np
import torch

from gsplat2d_rendering.culling import visible_leaf_mask_torch


def frustum_cull_mask(
    octree,
    is_in_box: torch.Tensor,
    viewpoint_camera,
    spatially_ordered: bool,
    all_ids: torch.Tensor,
    node_aabbs_gpu: torch.Tensor,
) -> tuple[torch.Tensor, list | None]:
    """
    Narrow *is_in_box* to splats whose octree leaf intersects the view frustum.

    Parameters
    ----------
    octree
        gsplat2d_rendering.culling.Octree
    node_aabbs_gpu
        octree.node_aabbs already resident on viewpoint_camera's device —
        cached by the caller (ViewerRenderer) rather than re-uploaded every frame.

    Returns
    -------
    bool_mask   : torch.Tensor [N, bool]  — visible splat mask
    vis_ranges  : list[(start, end)] | None
        Contiguous slice pairs when spatially ordered and no crop box is active.
        None whenever a per-splat boolean gather is required instead.
    """
    node_vis = visible_leaf_mask_torch(
        node_aabbs_gpu, viewpoint_camera.full_proj_transform
    ).cpu().numpy()

    node_offsets = octree.node_offsets  # int64 [L+1]

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
            flat_indices = octree.flat_indices
            all_idx = np.concatenate([flat_indices[s:e] for s, e in zip(starts, ends)])
            vis_cpu[all_idx] = True

    bool_mask = is_in_box & torch.from_numpy(vis_cpu).to(is_in_box.device)
    return bool_mask, vis_ranges
