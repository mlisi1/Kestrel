"""Spherical-coordinate orbit camera with configurable world-up axis."""

from __future__ import annotations

import math

import numpy as np
import torch


class OrbitCamera:
    """
    Spherical-coordinate orbit camera.

    All navigation methods update (yaw, pitch, look_at, distance) in place.
    Call build_RT() to get the (R, T) tensors needed by the renderer.
    """

    def __init__(self, look_at=(0., 0., 0.), distance=5.0,
                 yaw=0.0, pitch=0.3, fov_deg=60.0,
                 world_up: np.ndarray | None = None):
        self.look_at  = np.array(look_at, dtype=np.float64)
        self.distance = float(distance)
        self.yaw      = float(yaw)
        self.pitch    = float(pitch)
        self.fov_deg  = float(fov_deg)
        self.world_up = (world_up.copy() if world_up is not None
                         else np.array([0., 0., 1.], dtype=np.float64))

    def _basis(self) -> tuple[np.ndarray, np.ndarray]:
        """right_ref, fwd_ref — two unit vectors perpendicular to world_up."""
        up  = self.world_up / np.linalg.norm(self.world_up)
        ref = np.array([0., 0., 1.]) if abs(up[2]) < 0.9 else np.array([1., 0., 0.])
        right = np.cross(up, ref);   right /= np.linalg.norm(right)
        fwd   = np.cross(right, up); fwd   /= np.linalg.norm(fwd)
        return right, fwd

    @property
    def position(self) -> np.ndarray:
        up = self.world_up / np.linalg.norm(self.world_up)
        right_ref, fwd_ref = self._basis()
        e_yaw = math.cos(self.yaw) * fwd_ref + math.sin(self.yaw) * right_ref
        return self.look_at + self.distance * (
            math.cos(self.pitch) * e_yaw + math.sin(self.pitch) * up)

    def orbit(self, dyaw: float, dpitch: float):
        """Orbit camera around look_at — position moves, look_at fixed."""
        self.yaw  += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch,
                                   -math.pi / 2 + 0.01, math.pi / 2 - 0.01))

    def fps_look(self, dyaw: float, dpitch: float):
        """FPS-style look — camera position stays fixed, look_at moves."""
        pos = self.position.copy()
        self.yaw  += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch,
                                   -math.pi / 2 + 0.01, math.pi / 2 - 0.01))
        up = self.world_up / np.linalg.norm(self.world_up)
        right_ref, fwd_ref = self._basis()
        e_yaw = math.cos(self.yaw) * fwd_ref + math.sin(self.yaw) * right_ref
        offset = self.distance * (math.cos(self.pitch) * e_yaw
                                  + math.sin(self.pitch) * up)
        self.look_at = pos - offset

    def pan(self, dx: float, dy: float):
        up  = self.world_up / np.linalg.norm(self.world_up)
        pos = self.position
        fwd = self.look_at - pos;  fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right, _ = self._basis()
        else:
            right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        self.look_at += dx * right + dy * cam_up

    def move(self, fwd_d: float, right_d: float):
        """Translate look_at horizontally relative to current view direction."""
        up   = self.world_up / np.linalg.norm(self.world_up)
        pos  = self.position
        look = self.look_at - pos;  look /= np.linalg.norm(look)
        horiz = look - np.dot(look, up) * up
        if np.linalg.norm(horiz) < 1e-6:
            _, horiz = self._basis()
        else:
            horiz /= np.linalg.norm(horiz)
        right_vec = np.cross(horiz, up)
        if np.linalg.norm(right_vec) < 1e-6:
            right_vec, _ = self._basis()
        else:
            right_vec /= np.linalg.norm(right_vec)
        self.look_at = self.look_at + fwd_d * horiz + right_d * right_vec

    def translate_up(self, amount: float):
        """Move both camera and look_at along world_up (Q/E vertical translation)."""
        up = self.world_up / np.linalg.norm(self.world_up)
        self.look_at += amount * up

    def zoom(self, delta: float):
        self.distance = max(0.05, self.distance * (1.0 - delta * 0.15))

    def build_RT(self, cam_tf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        up  = self.world_up / np.linalg.norm(self.world_up)
        pos = self.position
        fwd = self.look_at - pos;  fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            _, fwd_ref = self._basis()
            right = np.cross(fwd, fwd_ref)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        # c2w in OpenGL convention (Y-up, Z-backward)
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, 0] = torch.tensor(right,  dtype=torch.float64)
        c2w[:3, 1] = torch.tensor(cam_up, dtype=torch.float64)
        c2w[:3, 2] = torch.tensor(-fwd,   dtype=torch.float64)
        c2w[:3, 3] = torch.tensor(pos,    dtype=torch.float64)
        c2w = torch.matmul(cam_tf, c2w)
        c2w[:3, 1:3] *= -1          # flip Y, Z — matches client.py:get_RT
        w2c = torch.linalg.inv(c2w)
        return w2c[:3, :3].float(), w2c[:3, 3].float()
