"""Pose and scale terms on the asset renders, computed with a differentiable pose readout.

Notation: render m of object i has the known camera ``L = [L_R | L_o]`` in the asset frame and the predicted
camera ``P = [P_R | P_o]`` in the scene frame. ``d(R, t; R', t') = |t - t'| / |s*| + |R - R'|_F / sqrt(2)`` compares
two poses of an object with ground-truth scale ``s*`` (translation in object diagonals, rotation ~ radians).

* ``scale_errors`` / ``shape_errors``: L1 error of the per-render log-scale and of its axis-ratio part;
* ``candidate_errors``: d(candidate, ground truth), with candidates built from the ground-truth axis ratios and
  the predicted mean log-scale;
* ``fused_terms``: d(fused pose, ground truth) and d(candidate, fused pose), with the predicted scale; the fusion
  is the one of the inference readout (mean translation, chordal mean rotation).

``owner_mean`` averages per object first and then over objects, so objects with many views do not dominate.
"""

from __future__ import annotations

import math

import torch

from ..constants import ROLE_RENDER

EPS = 1e-12
POLAR_EPS = 1e-6
SMALL_AREA = 0.01             # objects covering less than 1 % of a scene view get a larger scale-loss weight


def per_view(values: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    """Per-object values (B, O, ...) gathered at each view's object -> (B, N, ...) (scene views read object 0)."""
    index = owner.clamp_min(0)
    return values[torch.arange(len(index), device=index.device)[:, None], index]


def owner_sum(values: torch.Tensor, valid: torch.Tensor, owner: torch.Tensor, objects: int,
              weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean of ``values`` (B, N) over the valid views of each object, then summed over objects (times ``weight``
    (B, O)). Returns ``(sum, sum of weights, number of objects with a valid view)``."""
    masks = (owner[..., None] == torch.arange(objects, device=values.device)) & valid[..., None]
    counts = masks.sum(1)
    ok = counts > 0
    each = (values[..., None] * masks).sum(1) / counts.clamp_min(1)
    if weight is None:
        weight = torch.ones_like(each)
    return (each * weight * ok).sum(), (weight * ok).sum(), ok.sum()


def owner_mean(values: torch.Tensor, valid: torch.Tensor, owner: torch.Tensor, objects: int,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """``owner_sum`` normalized by the sum of the object weights (plain mean over objects without ``weight``)."""
    total, weights, _ = owner_sum(values, valid, owner, objects, weight)
    return total / weights.clamp_min(1)


def small_object_weights(x: dict, strength: float) -> torch.Tensor:
    """(B, O) weights ``1 + strength * clip(1 - a / 0.01, 0, 1)``, a = largest fraction of a scene view covered by
    the object's mask."""
    pixels = x["instance_masks"].sum((-1, -2), dtype=torch.float32)                  # (B, N, O)
    support = x["image_valid"].sum((-1, -2, -3), dtype=torch.float32).clamp_min(1)   # (B, N)
    fraction = torch.where((x["role"] == 0)[..., None], pixels / support[..., None], 0.0).amax(1)
    return 1 + strength * ((SMALL_AREA - fraction) / SMALL_AREA).clamp(0, 1)


def camera_from_pose_encoding(pose_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Camera-to-world rotation (..., 3, 3) and centre (..., 3) of 9-d camera encodings (differentiable)."""
    T = pose_enc[..., :3]
    q = pose_enc[..., 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    -1).reshape(q.shape[:-1] + (3, 3))
    Rc = R.transpose(-1, -2)
    return Rc, -(Rc @ T[..., None])[..., 0]


def scale_errors(view_log_scale: torch.Tensor, object_scale: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    """(B, N) mean over axes of ``|log-scale_view - log s*|``."""
    truth = per_view(object_scale.clamp_min(1e-8).log(), owner)
    return (view_log_scale.float() - truth).abs().mean(-1)


def shape_errors(view_log_scale: torch.Tensor, object_scale: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    """(B, N) the same error after removing the mean over axes (the axis ratios, independent of the unit)."""
    truth = per_view(object_scale.clamp_min(1e-8).log(), owner)
    predicted = view_log_scale.float()
    return ((predicted - predicted.mean(-1, keepdim=True)) - (truth - truth.mean(-1, keepdim=True))).abs().mean(-1)


def _norm(v: torch.Tensor, dims) -> torch.Tensor:
    return (v.square().sum(dims) + EPS).sqrt()


def candidate_errors(pose_enc: torch.Tensor, object_log_scale: torch.Tensor, owner: torch.Tensor,
                     select: torch.Tensor, targets: dict) -> torch.Tensor:
    """(B, N) ``d(candidate, ground truth)`` on the ``select`` rows (valid renders), 0 elsewhere.

    The label camera ``C`` and pose ``G`` of a render give ``V^-1 = C^-1 G`` for the ground-truth scale. The
    candidate uses the scale ``gamma s*`` with ``gamma = exp(mean(log s - log s*))``, which multiplies the
    translation of ``V^-1`` by gamma: ``t = P_o + gamma P_R (V^-1)_t``, ``R = P_R (V^-1)_R``.
    """
    eye = torch.eye(4, device=pose_enc.device)
    s3 = select[..., None]
    C = torch.where(select[..., None, None], targets["camera_c2w"].float(), eye)
    Rg = torch.where(s3[..., None], per_view(targets["object_R"].float(), owner), eye[:3, :3])
    tg = torch.where(s3, per_view(targets["object_t"].float(), owner), 0.0)
    sg = torch.where(s3, per_view(targets["object_scale"].float().clamp_min(1e-8), owner), 1.0)
    ls = per_view(object_log_scale.float(), owner)
    ls = torch.where(s3 & torch.isfinite(ls), ls, sg.log())
    CR, Co = C[..., :3, :3], C[..., :3, 3]
    inverse_R = CR.transpose(-1, -2) @ Rg
    inverse_t = (CR.transpose(-1, -2) @ (tg - Co)[..., None])[..., 0]
    gamma = (ls - sg.log()).mean(-1).exp()
    RP, oP = camera_from_pose_encoding(pose_enc.float())
    translation = _norm(oP + gamma[..., None] * (RP @ inverse_t[..., None])[..., 0] - tg, -1) / sg.norm(dim=-1)
    rotation = _norm(RP @ inverse_R - Rg, (-1, -2)) / math.sqrt(2)
    return torch.where(select, translation + rotation, torch.zeros_like(translation))


class _ProjectSO3(torch.autograd.Function):
    """Nearest rotation ``U diag(1, 1, det(U V^T)) V^T`` of M (SVD) with the polar-decomposition derivative,
    which stays finite when the singular values coincide (all candidates agree)."""

    @staticmethod
    def forward(ctx, M: torch.Tensor) -> torch.Tensor:
        U, S, Vh = torch.linalg.svd(M.double())
        d = torch.linalg.det(U @ Vh)
        D = torch.cat([torch.ones_like(S[..., :2]), d[..., None]], -1)
        R = U @ (D[..., :, None] * Vh)
        ctx.save_for_backward(R, Vh, S * D)
        return R.to(M.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        R, Vh, sigma = ctx.saved_tensors
        V = Vh.transpose(-1, -2)
        G = Vh @ R.transpose(-1, -2) @ grad.double() @ V
        F = sigma[..., :, None] + sigma[..., None, :]
        F = torch.where(F.abs() < POLAR_EPS, torch.where(F < 0, -POLAR_EPS, POLAR_EPS), F)
        return (R @ V @ ((G - G.transpose(-1, -2)) / F) @ Vh).to(grad.dtype)


def project_so3(M: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) chordal projection onto SO(3), differentiable."""
    return _ProjectSO3.apply(M)


def fuse(p: dict, x: dict) -> dict[str, torch.Tensor]:
    """Pose candidates of every render with the predicted scale, and their fusion per object.

    Predicted cameras are rebased on the (detached) predicted camera of scene view 0, as in the inference
    readout. Returns ``render`` (B, N), candidate ``R`` / ``t`` (identity / 0 off the renders), membership ``m``
    (B, N, O), ``count`` (B, O), and the fused ``R_mean`` (B, O, 3, 3), ``t_mean`` (B, O, 3).
    """
    render = (x["role"] == ROLE_RENDER) & (x["owner"] >= 0)
    R, o = camera_from_pose_encoding(p["pose_enc"][..., :9].float())
    R0, o0 = R[:, :1].detach(), o[:, :1].detach()
    R = R0.transpose(-1, -2) @ R
    o = (R0.transpose(-1, -2) @ (o - o0)[..., None])[..., 0]
    L = x["camera_c2w"].float()
    scale = per_view(p["object_log_scale"].float(), x["owner"]).exp()
    GR = R @ L[..., :3, :3].transpose(-1, -2)
    Gt = o - (GR @ (scale * L[..., :3, 3])[..., None])[..., 0]
    objects = p["object_log_scale"].shape[1]
    eye = torch.eye(3, device=GR.device, dtype=GR.dtype)
    GR = torch.where(render[..., None, None], GR, eye)
    Gt = torch.where(render[..., None], Gt, torch.zeros_like(Gt))
    m = ((x["owner"][..., None] == torch.arange(objects, device=GR.device)) & render[..., None]).to(GR.dtype)
    count = m.sum(1)
    safe = count.clamp_min(1)
    t_mean = torch.einsum("bno,bnk->bok", m, Gt) / safe[..., None]
    M = torch.where((count > 0)[..., None, None], torch.einsum("bno,bnij->boij", m, GR) / safe[..., None, None], eye)
    return dict(render=render, R=GR, t=Gt, m=m, count=count, R_mean=project_so3(M), t_mean=t_mean)


def fused_terms(p: dict, x: dict, t: dict, symmetry: bool) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """``(L_fuse, L_agree, fusion)``: d(fused pose, ground truth) averaged over objects, and the mean over an
    object's renders of d(candidate, fused pose), averaged over objects with at least two renders.

    Candidates are aligned within the object's symmetry group before their spread is measured, so candidates
    that differ by a symmetry of the shape do not count as disagreement. With ``symmetry`` the ground-truth
    rotation is also replaced by its symmetric equivalent closest to the fused pose.
    """
    from .symmetry import align

    f = fuse(p, x)
    render, GR, Gt, m, count = f["render"], f["R"], f["t"], f["m"], f["count"]
    owner = x["owner"]
    has, multi, safe = count > 0, count > 1, count.clamp_min(1)
    Rg, tg = t["object_R"].float(), t["object_t"].float()
    diagonal = t["object_scale"].float().clamp_min(1e-8).norm(dim=-1)
    code = t["object_symmetry"].float()
    if symmetry:
        Rg = Rg @ align(f["R_mean"], Rg, code)
    fused = _norm(f["t_mean"] - tg, -1) / diagonal + _norm(f["R_mean"] - Rg, (-1, -2)) / math.sqrt(2)
    spread_t = torch.where(render, _norm(Gt - per_view(f["t_mean"], owner), -1) / per_view(diagonal, owner), 0.0)
    aligned = GR @ align(per_view(Rg, owner), GR, per_view(code, owner))
    aligned = torch.where(render[..., None, None], aligned, torch.eye(3, device=GR.device, dtype=GR.dtype))
    mean = project_so3(torch.where(has[..., None, None],
                                   torch.einsum("bno,bnij->boij", m, aligned) / safe[..., None, None],
                                   torch.eye(3, device=GR.device, dtype=GR.dtype)))
    spread_R = torch.where(render, _norm(aligned - per_view(mean, owner), (-1, -2)) / math.sqrt(2), 0.0)
    agree = torch.einsum("bno,bn->bo", m, spread_t + spread_R) / safe
    loss_fused = (fused * has).sum() / has.sum().clamp_min(1)
    loss_agree = (agree * multi).sum() / multi.sum().clamp_min(1)
    return loss_fused, loss_agree, f
