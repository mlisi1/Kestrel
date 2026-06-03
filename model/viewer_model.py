import os
import sys
import types

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_2DGS = os.path.join(_ROOT, "2d_gaussian_splatting")

if _2DGS not in sys.path:
    sys.path.insert(0, _2DGS)

# Stub out the scene package to avoid executing scene/__init__.py,
# which pulls in the full 2dgs training stack (dataset readers, arguments, etc.).
# We only need GaussianModel from scene/gaussian_model.py.
if "scene" not in sys.modules:
    _scene_pkg = types.ModuleType("scene")
    _scene_pkg.__path__ = [os.path.join(_2DGS, "scene")]
    _scene_pkg.__package__ = "scene"
    sys.modules["scene"] = _scene_pkg

from scene.gaussian_model import GaussianModel


class GaussianModelforViewer(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.scaling_modifier = 1.
        self.depth_ratio = 0.
