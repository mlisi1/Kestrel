"""Standalone helpers: edge-detection kernel, Sobel filters.

GPU-timing is delegated to gsplat2d_rendering.render.profiling.Profiler, and
the zeroth-order SH constant to gsplat2d_rendering.sh.C0 (see
renderer/renderer.py) rather than duplicating either here.
"""

import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0) / 4
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0) / 4


def gradient_map(image: torch.Tensor) -> torch.Tensor:
    sobel_x = _SOBEL_X.to(image.device)
    sobel_y = _SOBEL_Y.to(image.device)
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    return torch.sqrt(grad_x ** 2 + grad_y ** 2).norm(dim=0, keepdim=True)
