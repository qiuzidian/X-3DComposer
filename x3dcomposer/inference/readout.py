"""Pose readout: predicted cameras of the asset renders -> object poses, in closed form (NumPy).

For render m of object i, ``L = [L_R | L_o]`` is its known camera in the asset frame and ``P = [P_R | P_o]`` the
predicted camera in the scene frame (the frame of the first scene camera; all predicted cameras are rebased on
the predicted camera of scene view 0). With the object scale ``s`` the camera in the scaled asset frame is
``V = [L_R | diag(s) L_o]``, and each render gives one rigid pose candidate

    [R_m | t_m] = P V^-1,   i.e.   R_m = P_R L_R^T,   t_m = P_o - R_m diag(s) L_o.

The scale is ``s = exp(mean_m log-scale_m)`` (``object_log_scale`` of the model). Candidates are fused by the
mean translation and the chordal mean rotation. Predictions are in normalized units (mean scene-point distance
to the first camera = 1); ``metric_factor`` recovers one global factor to the unit of the input depth or
camera poses when they are given. The readout uses no ground truth, no optimization and no crop.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants import ROLE_CROP, ROLE_RENDER, ROLE_SCENE
from ..geometry.camera import decode_pose_encoding, mean_rotation

CANDIDATE_POLICIES = ("mean", "first")
SAFE_ENCODING = (0, 0, 0, 0, 0, 0, 1, 1, 1)      # identity camera, used for the rows of auxiliary crops


def _numpy(value) -> np.ndarray:
    """Tensor or array -> NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _unbatch(value, rank: int) -> np.ndarray:
    value = _numpy(value)
    if value.ndim == rank + 1:
        if value.shape[0] != 1:
            raise ValueError("The readout takes one sample at a time")
        value = value[0]
    return value


def decode_cameras(pose_enc, inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Predicted cameras of the scene views and renders, rebased on scene view 0: ``c2w`` (N, 4, 4) and centred
    ``K`` (N, 3, 3). Crops are auxiliary views; their rows are returned as identity cameras."""
    pose_enc = _unbatch(pose_enc, 2).astype(np.float64).copy()
    sizes = _unbatch(inputs["view_size"], 2).astype(int)
    pose_enc[_unbatch(inputs["role"], 1).astype(int) == ROLE_CROP] = SAFE_ENCODING
    c2w, K = np.zeros((len(pose_enc), 4, 4)), np.zeros((len(pose_enc), 3, 3))
    for size in {tuple(row) for row in sizes.tolist()}:
        rows = (sizes == size).all(1)
        c2w[rows], K[rows] = decode_pose_encoding(pose_enc[rows], size)
    return np.linalg.inv(c2w[0])[None] @ c2w, K


def metric_factor(prediction: dict, inputs: dict, conditions: dict) -> tuple[float | None, str]:
    """Factor from normalized units to the unit of the inputs, and where it comes from.

    With depth given: the median of (input depth / predicted depth) over the valid pixels of the scene views.
    Otherwise, with camera poses of several views: the least-squares ratio of the given to the predicted
    distances of the other cameras to the first one. Otherwise ``None`` (the output stays normalized).
    """
    scenes = np.flatnonzero(_unbatch(inputs["role"], 1).astype(int) == ROLE_SCENE)
    if conditions.get("depth_scale") is not None and "depth" in prediction:
        predicted = _unbatch(prediction["depth"], 4).astype(np.float64)[..., 0]
        given = _unbatch(inputs["depth"], 4).astype(np.float64)[scenes, 0]
        valid = _unbatch(inputs["depth_valid"], 4).astype(bool)[scenes, 0]
        valid = valid & np.isfinite(predicted) & (predicted > 0) & (given > 0)
        if valid.any():
            return float(np.median(given[valid] * conditions["depth_scale"] / predicted[valid])), "depth"
    if conditions.get("has_baseline") and len(scenes) >= 2:
        cameras, _ = decode_cameras(prediction["pose_enc"], inputs)
        b = np.linalg.norm(cameras[scenes[1:], :3, 3] - cameras[scenes[0], :3, 3], axis=1)
        centres = _unbatch(inputs["camera_c2w"], 3).astype(np.float64)[scenes, :3, 3]
        a = np.linalg.norm(centres[1:] - centres[0], axis=1) * conditions["camera_scale"]
        if float(b @ b) > 1e-12 and float(a @ a) > 1e-12:
            return float(a @ b) / float(b @ b), "camera_poses"
    return None, "none"


def resolve_scales(predicted: np.ndarray, known: Sequence | None, factor: float | None) -> np.ndarray:
    """Per-object output scales in normalized units, with the user's known sizes applied.

    ``known[i]`` is None, a number or a 3-vector: the scale of object i in the unit of the inputs. If that unit
    can be recovered (``factor``), the known scale replaces the prediction. Otherwise only its axis ratios are
    used and the predicted magnitude (geometric mean) is kept. A known size changes the returned scale only:
    rotation and translation are read out with the predicted scale, with which the predicted cameras agree.
    """
    scales = predicted.copy()
    for i, value in enumerate(known or ()):
        if value is None:
            continue
        given = np.broadcast_to(np.asarray(value, np.float64), (3,))
        if not np.isfinite(given).all() or (given <= 0).any():
            raise ValueError("Known object sizes must be positive")
        if factor is not None:
            scales[i] = given / factor
        else:
            scales[i] = given * np.exp(np.log(predicted[i]).mean() - np.log(given).mean())
    return scales


def read_out(prediction: dict, inputs: dict, conditions: dict, *, candidate_policy: str = "mean",
             known_sizes: Sequence | None = None) -> dict:
    """Object poses and scene cameras of one sample from a model ``prediction``.

    ``inputs`` are the model inputs of the forward pass, ``conditions`` the record of
    ``data.conditions.apply_conditions``. ``candidate_policy``: ``mean`` (fuse all renders) or ``first`` (the
    first render only). Returns a dictionary with, per object, ``R`` (3, 3), ``t`` (3,), ``scale`` (3,) and
    ``pose`` (4, 4) = [R diag(scale) | t] (asset frame -> scene frame); ``candidates`` (the per-render poses);
    the scene cameras ``camera_c2w`` / ``camera_K``; and ``unit`` (``input`` or ``normalized``).
    """
    if candidate_policy not in CANDIDATE_POLICIES:
        raise ValueError(f"candidate_policy must be one of {CANDIDATE_POLICIES}")
    role = _unbatch(inputs["role"], 1).astype(int)
    owner = _unbatch(inputs["owner"], 1).astype(int)
    render_cameras = _unbatch(inputs["camera_c2w"], 3).astype(np.float64)
    cameras, K_predicted = decode_cameras(prediction["pose_enc"], inputs)
    factor, source = metric_factor(prediction, inputs, conditions)
    predicted = np.exp(_unbatch(prediction["object_log_scale"], 2).astype(np.float64))
    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite predicted scale")
    scales = resolve_scales(predicted, known_sizes, factor)
    unit_factor = 1.0 if factor is None else factor
    objects = []
    for index in range(len(scales)):
        candidates = []
        for view in np.flatnonzero((role == ROLE_RENDER) & (owner == index)):
            virtual = render_cameras[view].copy()
            virtual[:3, 3] *= predicted[index]                # V = [L_R | diag(s) L_o], s = predicted scale
            candidates.append(cameras[view] @ np.linalg.inv(virtual))
        chosen = candidates if candidate_policy == "mean" else candidates[:1]
        R = mean_rotation(np.stack([c[:3, :3] for c in chosen]))
        t = np.stack([c[:3, 3] for c in chosen]).mean(0) * unit_factor
        scale = scales[index] * unit_factor
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R * scale, t
        spread = max(np.degrees(np.arccos(np.clip((np.trace(R.T @ c[:3, :3]) - 1) / 2, -1, 1))) for c in candidates)
        objects.append({"R": R, "t": t, "scale": scale, "pose": pose, "candidates": np.stack(candidates),
                        "rotation_spread_deg": float(spread)})
    scenes = np.flatnonzero(role == ROLE_SCENE)
    scene_cameras = cameras[scenes].copy()
    scene_cameras[:, :3, 3] *= unit_factor
    given = _unbatch(inputs["camera_available"], 1).astype(bool)[scenes]
    K_input = _unbatch(inputs["K"], 3).astype(np.float64)[scenes]
    result = {"objects": objects, "camera_c2w": scene_cameras,
              "camera_K": np.where(given[:, None, None], K_input, K_predicted[scenes]),
              "unit": "input" if factor is not None else "normalized", "unit_source": source,
              "unit_factor": factor, "mode": conditions["mode"]}
    if "depth" in prediction:
        result["depth"] = _unbatch(prediction["depth"], 4)[..., 0] * unit_factor
    return result
