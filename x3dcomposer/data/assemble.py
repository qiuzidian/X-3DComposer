"""Assemble one training sample (model inputs and targets) from a ``Selection`` (NumPy).

Views are prepared as at inference (``x3dcomposer.data.views``). Targets, all in the scene frame (the camera of
the first chosen scene view) and divided by the normalization length g = mean distance of the valid scene
points to the first camera:

    camera_c2w (N, 4, 4), camera_K (N, 3, 3)   label camera of every view with a centred principal point:
        scene view: its true camera; crop: the virtual camera fitted to the object's visible surface
        (``geometry.camera_fit``); render: [R | t] @ [L_R | diag(s) L_o] for the object pose (R, t, s)
    camera_valid, camera_weight (N,)            crops with a poor fit are down-weighted or dropped
    scene_K (S, 3, 3), scene_depth, scene_depth_valid (S, H, W)   true intrinsics / depth of the scene views
    crop_depth, crop_depth_valid (C, H, W)      depth of the crops (depth discontinuities removed)
    object_R (O, 3, 3), object_t (O, 3), object_scale (O, 3), object_symmetry (O,)
    object_points (O, P, 3), floor_up (3,), floor_support (O,), floor_bottom_min / floor_bottom_soft (O,),
    floor_height, floor_height_known, metre     inputs of the floor loss (``losses.floor``)
    is_synthetic                                 the depth-gradient losses apply to synthetic scenes only
"""

from __future__ import annotations

import numpy as np

from ..constants import ROLE_CROP, ROLE_SCENE
from ..geometry.camera import centered_K, pixel_map, scene_window, transform, unproject
from ..geometry.camera_fit import fit_camera
from . import augment
from .assets import SYMMETRY_CONTINUOUS, SYMMETRY_NONE, Asset
from .conditions import apply_conditions, draw_mode, draw_owner_slots
from .sampling import Selection
from .scene import Scene
from .views import SampleBuilder, order_views, resample_view

CROP_FIT_POINTS = 256
MIN_SCENE_POINTS = 100
SYMMETRY_SCALE_TOLERANCE = 0.02      # |log(s_x / s_z)| above this breaks every rotation about +Y but the half turn
FLOOR_POINTS = 512
FLOOR_SUPPORT_M = 0.02
FLOOR_SOFTMIN_M = 0.005
LENGTH_TARGETS = ("scene_depth", "crop_depth", "object_t", "object_scale")


def trusted_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Valid depth pixels that are not next to a depth discontinuity (2 cm or 2 %)."""
    valid = valid.astype(bool) & np.isfinite(depth) & (depth > 0)
    trust = valid.copy()
    for axis in (0, 1):
        for shift in (-1, 1):
            neighbour, neighbour_valid = np.roll(depth, shift, axis), np.roll(valid, shift, axis)
            trust &= ~neighbour_valid | (abs(depth - neighbour) <= np.maximum(0.02, 0.02 * depth))
    return trust


def crop_camera_label(box, mask: np.ndarray, depth: np.ndarray, trusted: np.ndarray, K: np.ndarray,
                      c2w: np.ndarray, size: int) -> dict:
    """Virtual centred camera of a crop, fitted on up to 256 visible surface points of the object."""
    A = pixel_map(box, (size, size))
    region = np.zeros_like(mask)
    height, width = mask.shape
    region[max(0, box[1]):min(height, box[3]), max(0, box[0]):min(width, box[2])] = True
    yy, xx = np.nonzero(mask & trusted & region)
    pick = np.linspace(0, len(xx) - 1, min(CROP_FIT_POINTS, len(xx))).astype(int) if len(xx) else []
    xx, yy = xx[pick], yy[pick]
    pixels = np.stack([xx, yy, np.ones(len(xx))])
    world = transform((np.linalg.inv(K) @ pixels).T * depth[yy, xx, None], c2w)
    K_virtual = centered_K(A @ K, (size, size))
    return {"K": K_virtual, **fit_camera(world, (A @ pixels)[:2].T, K_virtual, c2w, size)}


def scaled_symmetry(code: int, scale: np.ndarray) -> int:
    """Symmetry of the scaled asset: unequal x / z scales leave only the half turn (even orders) or nothing."""
    if code == SYMMETRY_NONE or abs(np.log(scale[0] / scale[2])) <= SYMMETRY_SCALE_TOLERANCE:
        return code
    return 2 if code == SYMMETRY_CONTINUOUS or code % 2 == 0 else SYMMETRY_NONE


def _softmin(h: np.ndarray, temperature: float) -> np.ndarray:
    m = h.min(-1, keepdims=True)
    return (m - temperature * np.log(np.exp(-(h - m) / temperature).sum(-1, keepdims=True)))[..., 0]


def floor_targets(scene: Scene, to_scene: np.ndarray, poses: list, scales: list, points: list) -> dict:
    """Targets of the floor loss: which objects stand on the floor, and the height of their bottoms.

    The floor is ``scene.json`` ``floor_height`` (along ``up``) if given; otherwise the reference is the 10th
    percentile (the minimum below ten objects) of the bottoms of all labelled objects of the sample. An object
    is floor-supported if its bottom lies within 2 cm of the reference. Objects without ``points.npy`` are not.
    """
    count = len(poses)
    P = np.zeros((count, FLOOR_POINTS, 3), np.float32)
    known = np.zeros(count, bool)
    for index, p in enumerate(points):
        if p is not None and len(p):
            P[index] = np.resize(p[:FLOOR_POINTS], (FLOOR_POINTS, 3))
            known[index] = True
    up = to_scene[:3, :3] @ scene.up / np.linalg.norm(scene.up)
    shift = float(up @ to_scene[:3, 3])                       # height in the scene frame = world height + shift
    world = np.stack([transform(P[i].astype(np.float64) * scales[i], poses[i]) for i in range(count)])
    h = world @ up
    bottom_min, bottom_soft = h.min(-1), _softmin(h, FLOOR_SOFTMIN_M)
    floor = scene.meta.get("floor_height")
    reference = None
    if floor is not None:
        reference = float(floor) + shift
    elif known.any():
        bottoms = bottom_min[known]
        reference = float(np.percentile(bottoms, 10)) if len(bottoms) >= 10 else float(bottoms.min())
    support = np.zeros(count, bool)
    if reference is not None:
        support = known & (np.abs(bottom_min - reference) <= FLOOR_SUPPORT_M)
    return {"object_points": P, "floor_up": up.astype(np.float32), "floor_support": support,
            "floor_bottom_min": bottom_min.astype(np.float32), "floor_bottom_soft": bottom_soft.astype(np.float32),
            "floor_height": np.float32(reference if floor is not None else 0.0),
            "floor_height_known": np.bool_(floor is not None), "metre": np.float32(1.0)}


def assemble(selection: Selection, assets_root: str, augmentation: bool = True,
             mode: str | None = None) -> tuple[dict, dict]:
    """Model inputs ``x`` and targets ``t`` of one sample (single sample, no batch axis).

    ``mode``: a fixed condition mode, or None to draw one at random. Raises ``ValueError`` for a sample without
    enough valid scene depth to define the normalization length.
    """
    rng = np.random.default_rng(selection.seed)
    scene = Scene(selection.scene_path)
    objects, hw = list(selection.objects), tuple(selection.scene_hw)
    full = [scene.view(v) for v in selection.views]
    if augmentation:
        for view in full:
            view["rgb"] = augment.degrade_source(view["rgb"], rng)
    to_scene = np.linalg.inv(full[0]["c2w"])                  # world -> scene frame (first camera)
    relative = [to_scene @ view["c2w"] for view in full]
    builder = SampleBuilder(len(objects))
    cameras, intrinsics, valid, weight = [], [], [], []
    scene_depth, scene_K, maps, cloud = [], [], [], []

    def label(camera: np.ndarray, K: np.ndarray, ok: bool = True, w: float = 1.0) -> None:
        cameras.append(camera)
        intrinsics.append(K)
        valid.append(ok)
        weight.append(w)

    for j, view in enumerate(full):
        window = scene_window(view["K"], view["ids"].shape, hw)
        raster = resample_view(view["rgb"], view["ids"], view["depth"], window, hw, "edge")
        builder.add_view(ROLE_SCENE, -1, j, raster, objects, view["K"], relative[j])
        K = raster["A"] @ view["K"]
        label(relative[j], centered_K(K, hw))
        scene_depth.append((raster["depth"], raster["depth_valid"]))
        scene_K.append(K)
        maps.append(raster["A"])
        cloud.append(unproject(raster["depth"], raster["depth_valid"], K, relative[j]))
    poses, scales = [], []
    for instance in objects:
        pose, scale = scene.label(instance)
        poses.append(to_scene @ pose)
        scales.append(scale)
    crops, renders = order_views(list(selection.crops), list(selection.renders))
    crop_depth, trusted = [], {}
    for index, j, box, size in crops:
        view = full[j]
        crop = resample_view(view["rgb"], view["ids"], view["depth"], box, (size, size))
        builder.add_view(ROLE_CROP, index, j, crop, objects, view["K"], relative[j], crop["A"] @ np.linalg.inv(maps[j]))
        if j not in trusted:
            trusted[j] = trusted_depth(view["depth"], view["depth"] > 0)
        fit = crop_camera_label(box, view["ids"] == objects[index], view["depth"], trusted[j], view["K"],
                                view["c2w"], size)
        label(np.eye(4) if fit["c2w"] is None else to_scene @ fit["c2w"], fit["K"], fit["valid"],
              1 / (1 + (fit["rmse"] / max(1, size * 0.02)) ** 2) if fit["valid"] else 0.0)
        crop_depth.append((crop["depth"], trusted_depth(crop["depth"], crop["depth_valid"])))
    loaded = {asset_id: Asset(f"{assets_root}/{asset_id}") for asset_id in set(selection.assets)}
    for index, k, size in renders:
        render = loaded[selection.assets[index]].view(k)
        row = builder.add_render(index, render, size)
        virtual = render["c2w"].copy()
        virtual[:3, 3] *= scales[index]                        # V = [L_R | diag(s) L_o]
        label(poses[index] @ virtual, centered_K(builder.rows[row]["K"], (size, size)))
    x = builder.stack()
    H, W = x["images"].shape[-2:]

    def canvas(arrays: list, dtype) -> np.ndarray:
        out = np.zeros((len(arrays), H, W), dtype)
        for i, a in enumerate(arrays):
            out[i, :a.shape[0], :a.shape[1]] = a
        return out

    assets = [loaded[a] for a in selection.assets]
    t = {"camera_c2w": np.stack(cameras), "camera_K": np.stack(intrinsics), "camera_valid": np.asarray(valid, bool),
         "camera_weight": np.asarray(weight), "scene_K": np.stack(scene_K),
         "scene_depth": canvas([np.where(v, z, 0) for z, v in scene_depth], np.float32),
         "scene_depth_valid": canvas([v for _, v in scene_depth], bool),
         "crop_depth": canvas([np.where(v, z, 0) for z, v in crop_depth], np.float32),
         "crop_depth_valid": canvas([v for _, v in crop_depth], bool),
         "object_R": np.stack([p[:3, :3] for p in poses]), "object_t": np.stack([p[:3, 3] for p in poses]),
         "object_scale": np.stack(scales),
         "object_symmetry": np.asarray([scaled_symmetry(a.symmetry, s) for a, s in zip(assets, scales)]),
         "is_synthetic": np.bool_(scene.meta.get("synthetic", True)),
         **floor_targets(scene, to_scene, poses, scales, [a.points() for a in assets])}
    points = np.concatenate(cloud)
    if len(points) <= MIN_SCENE_POINTS:
        raise ValueError("Too little valid scene depth: " + selection.scene_path)
    g = float(np.linalg.norm(points, axis=1).mean())
    x, _ = apply_conditions(x, mode or draw_mode(rng))
    t["camera_c2w"][:, :3, 3] /= g
    for key in LENGTH_TARGETS:
        t[key] = t[key] / g
    t["metre"] = np.float32(1.0 / g)                           # normalized units per metre
    x["owner_slots"] = draw_owner_slots(len(objects), rng)
    if augmentation:
        augment.scene_exposure(x, rng)
        augment.render_color_gain(x, rng)
        augment.render_jitter(x, rng)
    t = {k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in t.items()}
    return x, t
