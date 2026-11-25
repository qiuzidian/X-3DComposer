"""The training objective.

    L = L_scene + L_render-camera
      + w_crop (L_crop-camera + L_crop-depth)
      + w_scale (L_s + w_shape L_shape) + w_cons L_cons
      + w_cand L_cand + w_fuse L_fuse + w_agree L_agree
      + w_gr L_grounding + w_floor L_floor

* ``L_scene``: the backbone's camera and depth losses and a point loss on the scene views (``losses.scene``);
* camera loss of a render / crop, on 9-d encodings against the label camera (``data.assemble``):
  ``2 mean|t - t*| + mean|q - q*| + 0.1 (1 - |<q, q*>|) + mean|fov - fov*|``;
* ``L_crop-depth``: per pixel ``e + 0.2 e / z*`` with ``e = |z - z*|``, averaged per crop;
* ``L_s``, ``L_shape``, ``L_cons``: per-render log-scale against ``log s*``, its axis-ratio part, and its
  deviation from the object's mean; ``L_s`` and ``L_shape`` weight small objects up to three times more;
* ``L_cand``, ``L_fuse``, ``L_agree``: pose terms of the differentiable readout (``losses.placement``);
* symmetric objects use the equivalent label closest to the prediction (``losses.symmetry``).

Every per-view term is averaged over the views of an object first and then over objects. Default weights:
``x3dcomposer.config.LossConfig``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import LossConfig
from ..constants import ROLE_CROP, ROLE_RENDER
from .floor import floor_loss
from .grounding import grounding_loss
from .placement import (candidate_errors, fused_terms, owner_mean, per_view, scale_errors, shape_errors,
                        small_object_weights)
from .scene import SceneLoss
from .symmetry import symmetric_camera_labels


def crop_depth_loss(predicted: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, C) mean over the valid pixels of each crop of ``e + 0.2 e / z*`` and (B, C) whether a crop has any."""
    valid = valid.bool() & torch.isfinite(truth) & (truth > 0)
    truth = torch.where(valid, truth, torch.ones_like(truth))
    error = (predicted - truth).abs()
    pixel = error + 0.2 * error / truth.clamp_min(1e-3)
    count = valid.sum((-1, -2))
    return torch.where(valid, pixel, 0.0).sum((-1, -2)) / count.clamp_min(1), count > 0


class Objective(nn.Module):
    """Module docstring; ``forward`` returns the total ``loss`` and every term (detached copies are for logging)."""

    def __init__(self, vggt_omega_root: str, config: LossConfig | None = None):
        super().__init__()
        self.config = config or LossConfig()
        self.scene = SceneLoss(vggt_omega_root)

    def encode_labels(self, camera_c2w: torch.Tensor, K: torch.Tensor, view_size: torch.Tensor) -> torch.Tensor:
        """9-d encodings of the label cameras; the field of view refers to each view's own raster."""
        w2c = torch.linalg.inv(camera_c2w)[..., :3, :]
        encoded = None
        for h, w in torch.unique(view_size.reshape(-1, 2), dim=0).tolist():
            current = self.scene.encode_pose(w2c, K, (h, w))
            select = (view_size == view_size.new_tensor([h, w])).all(-1)[..., None]
            encoded = current if encoded is None else torch.where(select, current, encoded)
        return encoded

    def forward(self, p: dict, x: dict, t: dict, progress: float = 0.0) -> dict[str, torch.Tensor]:
        """``p``: model outputs; ``x``: prepared model inputs (``models.conditions.prepare_inputs``); ``t``: batched
        targets on the same device; ``progress``: fraction of training done (the upstream losses' schedule)."""
        cfg = self.config
        role, owner = x["role"], x["owner"]
        objects = p["object_log_scale"].shape[1]
        render, crop = role == ROLE_RENDER, role == ROLE_CROP
        pose = p["pose_enc"].float()
        valid = t["camera_valid"].bool()
        terms = {f"scene_{k}": v for k, v in self.scene(p, x, t, progress).items()}

        labels = t["camera_c2w"].float()
        if cfg.symmetry:
            labels = symmetric_camera_labels(labels, pose, owner, valid & render, t["object_R"], t["object_t"],
                                             t["object_symmetry"])
            t = dict(t, camera_c2w=labels)
        eye4, eye3 = torch.eye(4, device=pose.device), torch.eye(3, device=pose.device)
        truth = self.encode_labels(torch.where(valid[..., None, None], labels, eye4),
                                   torch.where(valid[..., None, None], t["camera_K"].float(), eye3), x["view_size"])
        q, q_true = pose[..., 3:7], truth[..., 3:7]
        q_true = torch.where(((q - q_true).abs().sum(-1) < (q + q_true).abs().sum(-1))[..., None], q_true, -q_true)
        rotation = (q - q_true).abs().mean(-1) + 0.1 * (
            1 - (F.normalize(q, dim=-1) * F.normalize(q_true, dim=-1)).sum(-1).abs().clamp(max=1))
        camera = (2 * (pose[..., :3] - truth[..., :3]).abs().clamp(max=128).mean(-1) + rotation
                  + (pose[..., 7:9] - truth[..., 7:9]).abs().mean(-1)) * t["camera_weight"].float()
        terms["render_camera"] = owner_mean(camera, valid & render, owner, objects)
        terms["crop_camera"] = owner_mean(camera, valid & crop, owner, objects)

        terms["crop_depth"] = pose.sum() * 0
        if "crop_depth" in p and p["crop_depth"].shape[1]:
            h, w = t["crop_depth"].shape[-2:]
            per_crop, has = crop_depth_loss(p["crop_depth"].float()[..., 0][..., :h, :w], t["crop_depth"].float(),
                                            t["crop_depth_valid"])
            batch = torch.arange(len(owner), device=owner.device)[:, None]
            terms["crop_depth"] = owner_mean(per_crop, has, owner[batch, p["crop_indices"]], objects)

        small = small_object_weights(x, cfg.small_object)
        scale_error = scale_errors(p["view_log_scale"], t["object_scale"], owner)
        shape_error = shape_errors(p["view_log_scale"], t["object_scale"], owner)
        terms["scale"] = owner_mean(torch.where(render, scale_error, 0.0), render, owner, objects, small)
        terms["shape"] = owner_mean(torch.where(render, shape_error, 0.0), render, owner, objects, small)
        deviation = (p["view_log_scale"] - per_view(p["object_log_scale"], owner)).square().mean(-1)
        terms["consistency"] = owner_mean(deviation, render, owner, objects)

        select = render & valid
        errors = candidate_errors(p["pose_enc"], p["object_log_scale"], owner, select, t)
        terms["candidate"] = owner_mean(errors * t["camera_weight"].float(), select, owner, objects)
        terms["fused"], terms["agreement"], fusion = fused_terms(p, x, t, cfg.symmetry)
        terms["grounding"] = grounding_loss(p["grounding_logits"], x)
        terms["floor"] = floor_loss(fusion, p["object_log_scale"], t)

        loss = (terms["scene_loss"] + terms["render_camera"] + cfg.crop * (terms["crop_camera"] + terms["crop_depth"])
                + cfg.scale * (terms["scale"] + cfg.shape * terms["shape"]) + cfg.consistency * terms["consistency"]
                + cfg.candidate * terms["candidate"] + cfg.fused * terms["fused"] + cfg.agreement * terms["agreement"]
                + cfg.grounding * terms["grounding"] + cfg.floor * terms["floor"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training objective")
        return {"loss": loss, **{k: v.detach() for k, v in terms.items()}}
