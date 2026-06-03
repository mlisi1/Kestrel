"""Standalone helpers: GPU timer, edge-detection kernel, Sobel filters."""

import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814  # zeroth-order SH basis constant

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0) / 4
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0) / 4


def gradient_map(image: torch.Tensor) -> torch.Tensor:
    sobel_x = _SOBEL_X.to(image.device)
    sobel_y = _SOBEL_Y.to(image.device)
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    return torch.sqrt(grad_x ** 2 + grad_y ** 2).norm(dim=0, keepdim=True)


class _CudaTimer:
    """Synchronised GPU wall-clock timer using CUDA Events."""
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end   = torch.cuda.Event(enable_timing=True)
        self.ms    = 0.0

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *_):
        self.end.record()
        torch.cuda.synchronize()
        self.ms = self.start.elapsed_time(self.end)
