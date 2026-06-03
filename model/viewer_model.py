import os
import sys
import types

import torch
import numpy as np

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

import utils.gaussian_utils as gaussian_utils


class GaussianModelforViewer(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self._opacity_origin = None
        self.scaling_modifier = 1.
        self.depth_ratio = 0.

    def select(self, mask: torch.Tensor):
        if self._opacity_origin is None:
            self._opacity_origin = torch.clone(self._opacity)
        else:
            self._opacity = torch.clone(self._opacity_origin)

        new_opacity = self._opacity.clone()
        new_opacity[mask] = 0.
        self._opacity = new_opacity

    def delete_gaussians(self, mask: torch.Tensor):
        gaussians_to_be_preserved = torch.bitwise_not(mask).to(self._xyz.device)
        self._xyz = self._xyz[gaussians_to_be_preserved]
        self._scaling = self._scaling[gaussians_to_be_preserved]
        self._rotation = self._rotation[gaussians_to_be_preserved]

        if self._opacity_origin is not None:
            self._opacity = self._opacity_origin
            self._opacity_origin = None
        self._opacity = self._opacity[gaussians_to_be_preserved]

        self._features_dc = self._features_dc[gaussians_to_be_preserved]
        self._features_rest = self._features_rest[gaussians_to_be_preserved]
        self.backup()

    def backup(self):
        self.org_xyz = self._xyz
        self.org_scaling = self._scaling
        self.org_rotation = self._rotation
        self.org_features_dc = self._features_dc
        self.org_features_rest = self._features_rest

    def transform_with_vectors(self,
                                idx: int,
                                scale: float,
                                r_wxyz: np.ndarray,
                                t_xyz: np.ndarray):
        xyz = self.org_xyz
        scaling = self.org_scaling
        rotation = self.org_rotation
        features = torch.cat((self.org_features_dc, self.org_features_rest), dim=1)

        xyz, scaling = gaussian_utils.GaussianTransformUtils.rescale(xyz, scaling, scale)
        xyz, rotation, new_features = gaussian_utils.GaussianTransformUtils.rotate_by_wxyz_quaternions(
            xyz=xyz,
            rotations=rotation,
            features=features,
            quaternions=torch.tensor(r_wxyz).to(xyz),
        )
        xyz = gaussian_utils.GaussianTransformUtils.translation(xyz, *t_xyz.tolist())

        self._xyz = xyz
        self._scaling = scaling
        self._rotation = rotation
        self._features_dc = new_features[:, 0, None, :]
        self._features_rest = new_features[:, 1:]
