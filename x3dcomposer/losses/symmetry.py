"""Symmetry-aware labels: for an object that is symmetric about its vertical axis, use the equivalent
ground-truth pose that is closest to the current prediction.

An object's symmetry group about the asset's +Y axis is a code (``data.assets.symmetry_code``): 1 = none,
n >= 2 = n-fold, 0 = continuous. If ``S = Ry(phi)`` is in the group, ``[R S | t]`` places the same shape as
``[R | t]``. With the label camera of a render ``C = [R | t] V`` (``V`` the render camera in the scaled asset
frame), the camera label of the equivalent pose is ``C' = [R S | t] V``, i.e. with ``W = R S R^T``:

    R_C' = W R_C,     o_C' = W (o_C - t) + t.

``phi`` is chosen per object, without gradient, to maximize ``sum_m tr(E_m Ry(phi))`` over its renders, where
``E_m = R^T R_C R_P^T R`` compares the predicted camera rotation ``R_P`` of render m with its label.
"""

from __future__ import annotations

import math

import torch

from .placement import camera_from_pose_encoding, per_view

MAX_ORDER = 12
CONTINUOUS, NONE = 0, 1


def rot_y(phi: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation by ``phi`` about +Y."""
    c, s = phi.cos(), phi.sin()
    one, zero = torch.ones_like(phi), torch.zeros_like(phi)
    return torch.stack([c, zero, s, zero, one, zero, -s, zero, c], -1).reshape(phi.shape + (3, 3))


def best_angle(a: torch.Tensor, b: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """The group element ``phi`` that maximizes ``a cos(phi) + b sin(phi)`` (0 for objects without symmetry)."""
    code = code.round().long()
    k = torch.arange(MAX_ORDER, device=a.device, dtype=a.dtype)
    n = code.clamp_min(1).to(a.dtype)[..., None]
    phi_k = 2 * math.pi * k / n
    score = a[..., None] * phi_k.cos() + b[..., None] * phi_k.sin()
    score = torch.where(k < n, score, torch.full_like(score, -torch.inf))
    discrete = phi_k.gather(-1, score.argmax(-1, keepdim=True))[..., 0]
    phi = torch.where(code == CONTINUOUS, torch.atan2(b, a), discrete)
    return torch.where(code == NONE, torch.zeros_like(phi), phi)


def align(rotation: torch.Tensor, reference: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """``Ry(phi)`` in the group ``code`` (no gradient) for which ``reference @ Ry(phi)`` is closest to ``rotation``."""
    E = (rotation.transpose(-1, -2) @ reference).detach()
    return rot_y(best_angle(E[..., 0, 0] + E[..., 2, 2], E[..., 2, 0] - E[..., 0, 2], code))


@torch.no_grad()
def symmetric_camera_labels(camera_c2w: torch.Tensor, pose_enc: torch.Tensor, owner: torch.Tensor,
                            select: torch.Tensor, object_R: torch.Tensor, object_t: torch.Tensor,
                            symmetry: torch.Tensor) -> torch.Tensor:
    """Camera labels (B, N, 4, 4) with the ``select`` rows (valid renders) of symmetric objects replaced by the
    labels of the equivalent pose closest to the prediction."""
    objects = object_R.shape[1]
    C = camera_c2w.float()
    rows = select & (owner >= 0) & (per_view(symmetry.float(), owner).round() != NONE)
    Ro, to = per_view(object_R.float(), owner), per_view(object_t.float(), owner)
    RP, _ = camera_from_pose_encoding(pose_enc.detach().float())
    RC = C[..., :3, :3]
    E = Ro.transpose(-1, -2) @ RC @ RP.transpose(-1, -2) @ Ro
    masks = (owner[..., None] == torch.arange(objects, device=owner.device)) & rows[..., None]
    a = (torch.where(rows, E[..., 0, 0] + E[..., 2, 2], 0.0)[..., None] * masks).sum(1)
    b = (torch.where(rows, E[..., 2, 0] - E[..., 0, 2], 0.0)[..., None] * masks).sum(1)
    phi = torch.where(rows, per_view(best_angle(a, b, symmetry.float()), owner), 0.0)
    W = Ro @ rot_y(phi) @ Ro.transpose(-1, -2)
    out = C.clone()
    centre = C[..., :3, 3]
    out[..., :3, :3] = torch.where(rows[..., None, None], W @ RC, RC)
    out[..., :3, 3] = torch.where(rows[..., None], (W @ (centre - to)[..., None])[..., 0] + to, centre)
    return out
