"""Grounding loss: the camera token of a crop / render must point at its object's patches in the scene views.

For a crop or render r of object i, the target distribution over all scene-view patches u is the object's mask
coverage ``q_r(u) = alpha_{i,u} / sum_u' alpha_{i,u'}`` and the loss is ``KL(q_r || p_r)`` with ``p_r`` the softmax of
the grounding head's logits, averaged over the crops and renders whose object is visible in a scene view.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ..constants import PATCH, ROLE_SCENE


def grounding_loss(logits: torch.Tensor, x: dict) -> torch.Tensor:
    """``logits`` (B, N, S * P) from ``models.grounding.GroundingHead``; ``x``: the (prepared) model inputs."""
    rows = torch.stack([torch.where(r == ROLE_SCENE)[0] for r in x["role"]])               # (B, S)
    batch = torch.arange(len(rows), device=rows.device)[:, None]
    h, w = (int(v) for v in x["view_size"][0, 0])
    b, s = rows.shape
    objects = x["instance_masks"].shape[2]
    masks = x["instance_masks"][batch, rows][..., :h, :w].float().reshape(b * s, objects, h, w)
    coverage = F.avg_pool2d(masks, PATCH, PATCH).flatten(2).transpose(1, 2).reshape(b, -1, objects)   # (B, S*P, O)
    support = x["image_valid"][batch, rows][..., :h, :w].float().reshape(b * s, 1, h, w)
    coverage = coverage * (F.avg_pool2d(support, PATCH, PATCH).reshape(b, -1, 1) > 0)
    owner = x["owner"].clamp_min(0)
    mine = torch.gather(coverage, 2, owner[:, None, :].expand(-1, coverage.shape[1], -1)).transpose(1, 2)
    total = mine.sum(-1)
    use = (x["role"] != ROLE_SCENE) & (total > 0)
    target = mine / total.clamp_min(1e-6)[..., None]
    divergence = (target * (target.clamp_min(1e-12).log() - logits.float().log_softmax(-1))).sum(-1)
    return (divergence * use).sum() / use.sum().clamp_min(1)
