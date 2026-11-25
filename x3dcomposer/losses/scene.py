"""Losses of the scene views: the backbone's own camera and depth losses plus a point loss.

The camera and depth losses are VGGT-Omega's training losses, imported unchanged from a clone of its repository
(``<vggt_omega_root>/training/losses``; they are not part of the pip package). Targets are already normalized
(``data.assemble``), so the upstream normalization stays off. The point loss follows the upstream one, except
that points are unprojected with the intrinsics the model was given, or else the predicted ones, instead of the
ground-truth intrinsics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from ..constants import ROLE_SCENE

CAMERA = dict(weight=5.0, weight_trans=2.0, weight_rot=1.0, weight_focal=1.0, normalize_trans_by_gt_scale=True,
              weight_pairwise=1.0, pairwise_weight_rotation=1.0, pairwise_weight_translation=1.0)
DEPTH = dict(weight=5.0, weight_normal=1.0, min_valid_pts=100, abs_l1_loss_ratio=1.0, rel_l1_loss_ratio=0.2,
             max_pixel_loss=100.0, use_conf_loss=True, conf_alpha=0.2, add_separate_reg_loss=True,
             separate_reg_loss_weight=1.0, apply_aux_loss_to_synthetic_only=True,
             gradient_loss_config=[dict(type="normal", scales=1, min_valid_samples=100, weight=0.1),
                                   dict(type="grad", scales=4, weight=1.0)])
POINT = dict(weight=5.0, min_valid_pts=100, abs_l1_loss_ratio=1.0, rel_l1_loss_ratio=0.2, max_pixel_loss=100.0)


def import_upstream_losses(vggt_omega_root: str | Path):
    """``(MultitaskLoss, extri_intri_to_pose_encoding, encoding_to_camera, weighted_mean)`` of the backbone's
    training code. Adds ``<root>/training`` (and ``<root>`` if ``vggt_omega`` is not installed) to ``sys.path``."""
    root = Path(vggt_omega_root).resolve()
    if not (root / "training" / "losses" / "loss.py").is_file():
        raise FileNotFoundError(f"{root} is not a clone of the VGGT-Omega repository (training/losses is missing)")
    for path in (root / "training", root):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    from losses.base_loss.depth import weighted_mean
    from losses.loss import MultitaskLoss
    from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding

    return MultitaskLoss, extri_intri_to_pose_encoding, encoding_to_camera, weighted_mean


def scene_rows(role: torch.Tensor) -> torch.Tensor:
    """(B, S) indices of the scene views."""
    return torch.stack([torch.where(r == ROLE_SCENE)[0] for r in role])


def gather_views(tensor: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``tensor[b, rows[b]]`` for every sample b."""
    return tensor[torch.arange(tensor.shape[0], device=tensor.device)[:, None], rows]


def unproject(depth: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
    """World points (B, S, H, W, 3) of depth maps (B, S, H, W, 1) with full intrinsics (integer pixel centres)."""
    B, S, H, W, _ = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                          torch.arange(W, device=depth.device, dtype=depth.dtype), indexing="ij")
    pixels = torch.stack((u, v, torch.ones_like(u)), -1).expand(B, S, H, W, 3)
    rays = torch.einsum("bvij,bvhwj->bvhwi", torch.linalg.inv(K.float()), pixels)
    points = rays / rays[..., 2:3] * depth
    R, t = w2c[..., :3, :3].float(), w2c[..., :3, 3].float()
    return torch.einsum("bvji,bvhwj->bvhwi", R, points - t[..., None, None, :])


class SceneLoss(nn.Module):
    """``5 * camera + 5 * depth + 5 * point`` on the scene views of a batch."""

    def __init__(self, vggt_omega_root: str | Path):
        super().__init__()
        multitask, self.encode_pose, self.decode_pose, self.weighted_mean = import_upstream_losses(vggt_omega_root)
        self.upstream = multitask(camera=dict(CAMERA), depth=dict(DEPTH), point=None, track=None,
                                  normalize_predictions=False, normalize_gt=None)

    def forward(self, p: dict, x: dict, t: dict, progress: float) -> dict[str, torch.Tensor]:
        """``p``: model outputs, ``x``: prepared inputs, ``t``: targets; returns ``loss`` and its parts."""
        rows = scene_rows(x["role"])
        h, w = (int(v) for v in x["view_size"][0, 0])
        pose = gather_views(p["pose_enc"], rows).float()
        depth = p["depth"].float()[:, :, :h, :w]
        C = gather_views(t["camera_c2w"], rows).float()
        K = t["scene_K"].float()
        truth = t["scene_depth"].float()[:, :, :h, :w, None]
        mask = t["scene_depth_valid"][:, :, :h, :w].bool() & torch.isfinite(truth[..., 0]) & (truth[..., 0] > 0)
        truth = torch.where(mask[..., None], truth, torch.zeros_like(truth))
        w2c = torch.linalg.inv(C)[..., :3, :]
        shift = torch.eye(3, device=K.device)
        shift[0, 2] = shift[1, 2] = 0.5                 # the upstream losses put pixel centres at half integers
        batch = dict(images=gather_views(x["images"], rows)[..., :h, :w], extrinsics=w2c, intrinsics=shift @ K,
                     depths=truth, point_masks=mask,
                     valid_seq_mask=torch.ones(len(rows), dtype=torch.bool, device=K.device),
                     depth_train_mask=mask.flatten(1).sum(1) > 100, is_synthetic=t["is_synthetic"].bool().reshape(-1))
        prediction = {"pose_enc_list": [pose], "depth": depth, "depth_conf": p["depth_conf"].float()[:, :, :h, :w]}
        upstream = self.upstream(prediction, batch, float(progress))
        point = self.point_loss(pose, depth, batch, K, gather_views(x["K"], rows).float(),
                                gather_views(x["camera_available"], rows))
        return {"loss": upstream["loss_objective"] + POINT["weight"] * point,
                "camera": upstream["loss_camera"], "depth": upstream["loss_depth"], "point": point}

    def point_loss(self, pose: torch.Tensor, depth: torch.Tensor, batch: dict, K_true: torch.Tensor,
                   K_given: torch.Tensor, given: torch.Tensor) -> torch.Tensor:
        """L1 between world points of the predicted depth (predicted camera; given or predicted intrinsics) and
        of the ground truth, with the upstream absolute + relative weighting."""
        w2c, K_predicted = self.decode_pose(pose, depth.shape[-3:-1])
        K = torch.where(given[..., None, None], K_given, K_predicted)
        sequence = batch["depth_train_mask"]
        mask = batch["point_masks"][sequence]
        zero = depth.sum() * 0
        if not sequence.any() or mask.sum() <= POINT["min_valid_pts"]:
            return zero
        predicted = unproject(depth, K, w2c)[sequence]
        truth = unproject(batch["depths"], K_true, batch["extrinsics"])[sequence]
        truth_depth = batch["depths"][sequence].squeeze(-1)
        difference = (predicted - truth).abs()
        mean = self.weighted_mean(truth_depth, mask, dim=(-2, -1), keepdim=True)
        weights = (1 / (truth_depth.clamp_min(0.1 * mean) + 1e-6)).clamp(0.1, 10)
        weighted = (difference * POINT["abs_l1_loss_ratio"]
                    + difference * weights[..., None] * POINT["rel_l1_loss_ratio"])
        return weighted[mask].clamp(max=POINT["max_pixel_loss"]).mean()
