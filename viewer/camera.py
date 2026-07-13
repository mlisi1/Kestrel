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

    def camera_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """pos, right, cam_up, fwd -- the actual view-direction basis (not
        to be confused with _basis()'s spherical-parameterization reference
        axes). Factored out of build_RT() so anything that needs to match
        the exact camera this instance renders with -- e.g.
        frustum_corners_world() -- shares identical basis math rather than
        risking a second, subtly different derivation."""
        up  = self.world_up / np.linalg.norm(self.world_up)
        pos = self.position
        fwd = self.look_at - pos;  fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            _, fwd_ref = self._basis()
            right = np.cross(fwd, fwd_ref)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        return pos, right, cam_up, fwd

    def build_RT(self, cam_tf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos, right, cam_up, fwd = self.camera_frame()
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


def frustum_corners_world(orbit_cam: OrbitCamera, fov_deg: float, aspect: float,
                           znear: float, zfar: float) -> np.ndarray:
    """8 world-space frustum corners for orbit_cam's current pose -- 4 at
    znear then 4 at zfar, each ordered top-left/top-right/bottom-right/
    bottom-left. Built from camera_frame(), the exact basis build_RT()
    feeds into Camera.from_w2c, so this always matches what that camera
    actually renders (the OpenGL-vs-optical-frame Y/Z flip inside
    build_RT() is purely an axis-labeling convention for talking to
    Camera.from_w2c -- it doesn't change which world-space region is
    physically visible, so it plays no part here). `fov_deg` is vertical
    FOV (matches LocalViewer.fov_deg / _build_camera's fx=fy convention);
    `aspect` = width / height."""
    pos, right, cam_up, fwd = orbit_cam.camera_frame()
    tan_half_y = math.tan(math.radians(fov_deg) * 0.5)
    tan_half_x = tan_half_y * aspect
    corners = np.empty((8, 3), dtype=np.float64)
    for i, depth in enumerate((znear, zfar)):
        half_w, half_h = depth * tan_half_x, depth * tan_half_y
        center = pos + fwd * depth
        corners[4 * i + 0] = center + cam_up * half_h - right * half_w
        corners[4 * i + 1] = center + cam_up * half_h + right * half_w
        corners[4 * i + 2] = center - cam_up * half_h + right * half_w
        corners[4 * i + 3] = center - cam_up * half_h - right * half_w
    return corners


def chunk_aabb_corners_world(aabb: np.ndarray) -> np.ndarray:
    """8 world-space corners of an axis-aligned box (aabb =
    [xmin,ymin,zmin,xmax,ymax,zmax], the exact row format of
    gsplat2d_rendering.Octree.node_aabbs), in the same 4-near/4-far/
    4-connector edge topology frustum_corners_world uses (0-3 a quad loop,
    4-7 a second quad loop, i<->i+4 the connectors) -- letting
    RenderWidget's box-edge drawing be shared between an asymmetric view
    frustum and a symmetric chunk AABB with no special-casing."""
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    return np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=np.float64)


def project_world_points(points_world: np.ndarray, cam) -> np.ndarray:
    """Projects [N, 3] world points through cam.full_proj_transform to
    [N, 2] pixel coordinates in cam.width x cam.height space, using the
    exact row-vector/transposed convention documented in
    gsplat2d_rendering.camera's module docstring (`clip = [x,y,z,1] @
    full_proj_transform`). Pixel mapping is (ndc+1)/2*dim on BOTH axes --
    no Y-flip -- matching gsplat2d_rendering/render/depth_normal.py's own
    ndc2pix: the optical y-down convention already agrees with raster
    y-down, unlike the textbook OpenGL NDC->raster formula. Points behind
    the camera (clip.w too small) come back as NaN so callers can skip
    frustum edges touching them."""
    n = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((n, 1))], axis=1)
    proj = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
    clip = homo @ proj
    w = clip[:, 3]
    px = np.full((n, 2), np.nan, dtype=np.float64)
    valid = w > 1e-4
    ndc = clip[valid, :3] / w[valid, None]
    px[valid, 0] = (ndc[:, 0] + 1.0) * 0.5 * cam.width
    px[valid, 1] = (ndc[:, 1] + 1.0) * 0.5 * cam.height
    return px
