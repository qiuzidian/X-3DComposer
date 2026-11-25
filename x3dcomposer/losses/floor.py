"""Floor loss: objects that stand on the floor in the ground truth should do so in the predicted layout.

Computed on the fused poses of the differentiable readout (``placement.fuse``), with heights in metres. For a
floor-supported object i with surface points p, the predicted bottom is the soft minimum (temperature 5 mm) of
``up . (R diag(s) p + t)``. Its offset e_i from the floor is

    floor height known:   (bottom_i - bottom*_i) + (min-bottom*_i - floor height)
    otherwise:            (bottom_i - bottom*_i) - the midpoint of that difference over the supported objects

and the loss is the mean of ``relu(|e_i| - 1 cm)``. Targets: ``data.assemble.floor_targets``.
"""

from __future__ import annotations

import torch

SOFTMIN_M = 0.005
TOLERANCE_M = 0.01


def floor_loss(fusion: dict, object_log_scale: torch.Tensor, t: dict) -> torch.Tensor:
    """``fusion``: result of ``placement.fuse``; ``t``: batched targets."""
    scale = object_log_scale.float().exp()
    points = t["object_points"].float() * scale[..., None, :]                              # (B, O, P, 3)
    placed = points @ fusion["R_mean"].transpose(-1, -2) + fusion["t_mean"][..., None, :]
    height = (placed * t["floor_up"].float()[:, None, None]).sum(-1) / t["metre"].float().reshape(-1, 1, 1)
    offset = -SOFTMIN_M * torch.logsumexp(-height / SOFTMIN_M, -1) - t["floor_bottom_soft"].float()
    use = (fusion["count"] > 0) & t["floor_support"].bool()
    known = t["floor_height_known"].bool().reshape(-1, 1).expand_as(use)
    absolute = offset + t["floor_bottom_min"].float() - t["floor_height"].float().reshape(-1, 1)
    detached = torch.where(use, offset.detach(), torch.full_like(offset, float("nan")))
    middle = torch.nan_to_num((detached.nanmedian(1).values - (-detached).nanmedian(1).values) / 2)
    relative = offset - middle[:, None]
    error = torch.where(known, absolute, relative)
    ok = torch.where(known, use, use & (use.sum(1, keepdim=True) >= 2))
    loss = torch.relu(error.abs() - TOLERANCE_M)
    return torch.where(ok, loss, torch.zeros_like(loss)).sum() / ok.sum().clamp_min(1)
