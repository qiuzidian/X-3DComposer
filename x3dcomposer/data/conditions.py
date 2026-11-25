"""Condition modes: which geometric conditions a sample carries, and their normalization.

A sample is assembled with all conditions present (``views.SampleBuilder.stack``). ``apply_conditions`` keeps
those of the requested mode (``K+P+D``, ``K+P``, ``K+D``, ``K``, ``none``), blanks the others and normalizes
what is kept: input depth is divided by its mean over the scene views, and camera centres by the mean distance
of the other scene cameras to the first one (1 for a single view). Asset renders always keep their known
camera and depth, which are in the unit-free asset frame.
"""

from __future__ import annotations

import numpy as np

from ..constants import CONDITION_MODES, IDENTITY_SLOTS, ROLE_RENDER, ROLE_SCENE, mode_flags, mode_name

MIN_DEPTH_PIXELS = 16
BASELINE_EPSILON = 1e-6
# Training: P(K) = 0.8, then poses and depth independently with probability 0.5, i.e. 0.2 for each of the five modes.
P_INTRINSICS, P_POSE, P_DEPTH = 0.8, 0.5, 0.5


def draw_mode(rng: np.random.Generator) -> str:
    """Random training mode (module constants)."""
    u = rng.random(3)
    k = bool(u[0] < P_INTRINSICS)
    return mode_name(k, k and bool(u[1] < P_POSE), k and bool(u[2] < P_DEPTH))


def richest_mode(has_intrinsics: bool, has_poses: bool, has_depth: bool, views: int) -> str:
    """The mode that uses everything the inputs provide (poses only matter with several views)."""
    return mode_name(has_intrinsics, has_intrinsics and has_poses and views >= 2, has_intrinsics and has_depth)


def mean_baseline(x: dict) -> float | None:
    """Mean distance of the other scene cameras to the first one; None for one view or coincident cameras."""
    centres = x["camera_c2w"][x["role"] == ROLE_SCENE][:, :3, 3].astype(np.float64)
    if len(centres) < 2:
        return None
    value = float(np.linalg.norm(centres[1:] - centres[0], axis=1).mean())
    return value if value > BASELINE_EPSILON else None


def mean_depth(x: dict) -> float:
    """Mean of the valid input depth of the scene views."""
    scene = x["role"] == ROLE_SCENE
    valid = x["depth_valid"][scene]
    if valid.sum() < MIN_DEPTH_PIXELS:
        raise ValueError("Depth was requested as a condition, but the scene views have (almost) no valid depth")
    return float(np.mean(x["depth"][scene][valid], dtype=np.float64))


def apply_conditions(x: dict, mode: str) -> tuple[dict, dict]:
    """A copy of ``x`` under ``mode`` and the normalization record.

    The record holds ``depth_scale`` and ``camera_scale`` (the divisors, None when the condition is not given)
    and ``has_baseline``; ``inference.readout.metric_factor`` uses them to restore the input unit.
    """
    if mode not in CONDITION_MODES:
        raise ValueError(f"Unknown condition mode {mode!r}")
    intrinsics, poses, depth = mode_flags(mode)
    x = {key: value.copy() for key, value in x.items()}
    common = x["role"] != ROLE_RENDER                       # scene views and crops
    record = {"mode": mode, "depth_scale": None, "camera_scale": None, "has_baseline": False}
    if poses:
        baseline = mean_baseline(x)
        record.update(camera_scale=1.0 if baseline is None else baseline, has_baseline=baseline is not None)
        x["camera_c2w"][common, :3, 3] /= record["camera_scale"]
    else:
        x["pose_available"][common] = False
        if intrinsics:
            x["camera_c2w"][common] = np.eye(4, dtype=x["camera_c2w"].dtype)
        else:
            x["camera_available"][common] = False
            x["K"][common] = 0
            x["camera_c2w"][common] = 0
    if depth:
        record["depth_scale"] = mean_depth(x)
        x["depth"][common] /= record["depth_scale"]
    else:
        x["depth_available"][common] = False
        x["depth"][common] = 0
        x["depth_valid"][common] = False
    return x, record


def draw_owner_slots(objects: int, rng: np.random.Generator) -> np.ndarray:
    """Random distinct identity-embedding indices for the objects of a sample."""
    return rng.choice(IDENTITY_SLOTS, objects, replace=False).astype(np.int64)
