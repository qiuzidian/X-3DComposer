"""Input preparation and the condition encoders (intrinsics / camera pose / depth).

Owns casting and validating a model input dictionary (keys: ``x3dcomposer.constants``), the grouping of views
by resolution, and ``ConditionEncoder``, whose outputs are added to the backbone tokens:

* camera token += MLP(11-d camera summary): world-to-camera translation (3) and quaternion (4), vertical and
  horizontal field of view (2), an "is a render" flag and a "camera given" flag;
* patch tokens += Linear(GELU(Conv(6-channel Pluecker rays))) for views with intrinsics. With intrinsics but no
  pose the rays are expressed in the view's own camera frame and two learned "pose unknown" vectors are added;
* patch tokens += Conv([log depth, validity]) for views with depth (bias-free, so it adds nothing elsewhere).

Pixel centres are at integer coordinates. Depth and camera translations are expected in normalized units
(``x3dcomposer.data.conditions.normalize_conditions``).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..constants import (BOOL_KEYS, FLOAT_KEYS, IDENTITY_SLOTS, INPUT_KEYS, INT_KEYS, ROLE_CROP, ROLE_RENDER,
                         ROLE_SCENE, SPATIAL_KEYS)
from .backbone import PATCH, WIDTH, rotation_to_quaternion

HIDDEN = 128
DEPTH_INIT_STD = 0.0125       # uniform init of the depth convolution (small next to the patch-token norm)
DEPTH_INIT_SEED = 0


def prepare_inputs(inputs: dict, device=None, validate: bool = True) -> dict:
    """Cast to float32 / int64 / bool tensors on ``device``, add a batch axis if missing, optionally validate."""
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError(f"Model input keys mismatch: {sorted(set(inputs) ^ set(INPUT_KEYS))}")
    dtypes = {**{k: torch.float32 for k in FLOAT_KEYS}, **{k: torch.int64 for k in INT_KEYS},
              **{k: torch.bool for k in BOOL_KEYS}}
    x = {k: torch.as_tensor(v).to(device=device, dtype=dtypes[k], non_blocking=True) for k, v in inputs.items()}
    if x["images"].ndim == 4:
        x = {k: v.unsqueeze(0) for k, v in x.items()}
    if validate:
        validate_inputs(x)
    return x


def validate_inputs(x: dict) -> None:
    """Raise ``ValueError`` unless the batched inputs obey the contract of ``x3dcomposer.constants``."""
    b, n, c, h, w = x["images"].shape
    o = x["owner_slots"].shape[1]
    if c != 3 or h % PATCH or w % PATCH or not 0 < o <= IDENTITY_SLOTS:
        raise ValueError("Images must be RGB with sides divisible by 16, and a sample holds 1..64 objects")
    shapes = {"images": (b, n, 3, h, w), "instance_masks": (b, n, o, h, w), "K": (b, n, 3, 3),
              "camera_c2w": (b, n, 4, 4), "crop_A": (b, n, 3, 3), "depth": (b, n, 1, h, w),
              "depth_valid": (b, n, 1, h, w), "image_valid": (b, n, 1, h, w), "owner_slots": (b, o),
              "view_size": (b, n, 2)}
    shapes.update({k: (b, n) for k in ("role", "owner", "parent", "camera_available", "pose_available",
                                       "depth_available")})
    for key, shape in shapes.items():
        if tuple(x[key].shape) != shape:
            raise ValueError(f"{key}: expected {shape}, got {tuple(x[key].shape)}")
    for key in FLOAT_KEYS:
        if not torch.isfinite(x[key]).all():
            raise ValueError("Non-finite " + key)
    role, owner, sizes = x["role"], x["owner"], x["view_size"]
    scene, crop, render = role == ROLE_SCENE, role == ROLE_CROP, role == ROLE_RENDER
    if (role[:, 0] != ROLE_SCENE).any() or (role[:, 1:] < role[:, :-1]).any() or (role > ROLE_RENDER).any():
        raise ValueError("Views must be ordered scene, crop, render, starting with a scene view")
    if (x["instance_masks"].sum(2) > 1.00001).any():
        raise ValueError("Object masks overlap")
    slots = x["owner_slots"]
    if any(len(set(row.tolist())) != o for row in slots) or ((slots < 0) | (slots >= IDENTITY_SLOTS)).any():
        raise ValueError("owner_slots must be distinct indices below 64")
    if (owner[scene] != -1).any() or (owner[~scene] < 0).any() or (owner[~scene] >= o).any():
        raise ValueError("owner must be -1 for scene views and an object index otherwise")
    if not (sizes == sizes[:1]).all() or (sizes % PATCH != 0).any() or (sizes <= 0).any():
        raise ValueError("view_size must be identical across the batch and divisible by 16")
    cam, pose, depth = x["camera_available"], x["pose_available"], x["depth_available"]
    if (pose & ~cam).any() or (depth & scene & ~cam).any():
        raise ValueError("Camera poses and depth require intrinsics")
    for flag in (cam, pose, depth):
        if ((flag != flag[:, :1]) & scene).any() or (~flag & render).any():
            raise ValueError("Conditions are given for all scene views or none; renders always carry theirs")
        parent = flag.gather(1, x["parent"].clamp(0, n - 1))
        if ((flag != parent) & crop).any():
            raise ValueError("A crop inherits the conditions of its parent scene view")
    if x["K"][~cam].any() or x["camera_c2w"][~cam].any():
        raise ValueError("Cameras that are not given must be zero")
    if (x["camera_c2w"][cam & ~pose] != torch.eye(4, device=cam.device)).any():
        raise ValueError("A view with intrinsics but no pose must carry the identity pose")
    valid = x["depth_valid"]
    if (valid & ~depth[..., None, None, None]).any() or (x["depth"][valid] <= 0).any() or x["depth"][~valid].any():
        raise ValueError("Depth must be positive where valid and zero elsewhere")
    for index in range(o):
        if not ((owner == index) & render).any(1).all():
            raise ValueError("Every object needs at least one render")


def view_groups(x: dict) -> list[tuple[int, int, tuple[int, int]]]:
    """Runs ``(start, stop, (h, w))`` of consecutive views with the same size."""
    sizes = [tuple(int(v) for v in row) for row in x["view_size"][0].tolist()]
    groups, start = [], 0
    for j in range(1, len(sizes) + 1):
        if j == len(sizes) or sizes[j] != sizes[start]:
            groups.append((start, j, sizes[start]))
            start = j
    return groups


def group_inputs(x: dict, start: int, stop: int, hw: tuple[int, int]) -> dict:
    """Views ``start:stop`` of ``x`` with their dense arrays cut from the canvas to ``hw``."""
    out = {}
    for key, value in x.items():
        if key == "owner_slots":
            out[key] = value
        elif key in SPATIAL_KEYS:
            out[key] = value[:, start:stop, ..., :hw[0], :hw[1]]
        else:
            out[key] = value[:, start:stop]
    return out


def camera_features(x: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The 11-d camera summary (B, N, 11) and the Pluecker rays (B, N, 6, h, w) of every view."""
    b, n, _, h, w = x["images"].shape
    present = x["camera_available"]
    posed = present & x["pose_available"]
    K = torch.where(present[..., None, None], x["K"], torch.eye(3, device=x["K"].device))
    C = torch.where(posed[..., None, None], x["camera_c2w"], torch.eye(4, device=K.device))
    R, origin = C[..., :3, :3], C[..., :3, 3]
    Rt = R.transpose(-1, -2)
    t = -(Rt @ origin[..., None]).squeeze(-1)
    q = F.normalize(rotation_to_quaternion(Rt), dim=-1)
    q = torch.where(q[..., -1:] < 0, -q, q)
    fov = torch.stack([2 * torch.atan(h / (2 * K[..., 1, 1])), 2 * torch.atan(w / (2 * K[..., 0, 0]))], -1)
    observed = torch.cat([torch.cat([t, q], -1) * posed[..., None], fov * present[..., None]], -1)
    summary = torch.cat([observed, (x["role"] == ROLE_RENDER)[..., None], present[..., None]], -1).float()
    yy, xx = torch.meshgrid(torch.arange(h, device=K.device, dtype=torch.float32),
                            torch.arange(w, device=K.device, dtype=torch.float32), indexing="ij")
    pixels = torch.stack([xx, yy, torch.ones_like(xx)])[None].expand(b, 3, h, w)
    direction = F.normalize(torch.einsum("bnij,bjhw->bnihw", R @ torch.linalg.inv(K), pixels), dim=2)
    moment = torch.linalg.cross(origin[..., None, None].expand_as(direction), direction, dim=2)
    return summary, torch.cat([direction, moment], 2) * present[..., None, None, None]


def depth_features(x: dict) -> torch.Tensor:
    """(B * N, 2, h, w) input of the depth convolution: ``log(depth)`` and a hole indicator, zero without depth."""
    valid = x["depth_valid"].float()
    z = torch.where(x["depth_valid"], x["depth"], torch.ones_like(x["depth"]))
    carries = x["depth_valid"].flatten(-3).any(-1)[..., None, None, None].float()
    return torch.cat([torch.log(z) * valid, (valid - 1) * carries], 2).flatten(0, 1)


class ConditionEncoder(nn.Module):
    """Camera-token delta (B, N, 1, 1024) and patch-token delta (B, N, P, 1024) of the given conditions."""

    def __init__(self):
        super().__init__()
        self.camera_mlp = nn.Sequential(nn.Linear(11, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, WIDTH))
        self.ray_conv = nn.Conv2d(6, HIDDEN, PATCH, PATCH)
        self.ray_proj = nn.Linear(HIDDEN, WIDTH)
        self.depth_conv = nn.Conv2d(2, WIDTH, PATCH, PATCH, bias=False)
        self.pose_unknown = nn.Parameter(torch.zeros(2, WIDTH))   # row 0: camera token, row 1: patch tokens
        for layer in (self.camera_mlp[-1], self.ray_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        bound = DEPTH_INIT_STD * math.sqrt(3)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(DEPTH_INIT_SEED)
            self.depth_conv.weight.copy_(torch.empty(self.depth_conv.weight.shape).uniform_(-bound, bound,
                                                                                         generator=generator))

    def forward(self, x: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(patch delta, camera-token delta)``; runs in float32 regardless of autocast."""
        with torch.autocast(device_type=x["images"].device.type, enabled=False):
            b, n = x["images"].shape[:2]
            summary, rays = camera_features(x)
            ray = F.gelu(self.ray_conv(rays.flatten(0, 1))).flatten(2).transpose(1, 2)
            ray = self.ray_proj(ray).reshape(b, n, -1, WIDTH) * x["camera_available"][..., None, None]
            depth = self.depth_conv(depth_features(x)).flatten(2).transpose(1, 2).reshape(b, n, -1, WIDTH)
            depth = depth * x["depth_available"][..., None, None]
            token = self.camera_mlp(summary)[:, :, None]
            unposed = (x["camera_available"] & ~x["pose_available"])[..., None, None].float()
            token = token + unposed * self.pose_unknown[0]
            ray = ray + unposed * self.pose_unknown[1]
            return ray + depth, token
