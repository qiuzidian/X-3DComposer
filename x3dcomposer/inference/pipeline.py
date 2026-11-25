"""End-to-end inference: images + object masks + assets (+ optional K / poses / depth) -> layout.

``compose`` prepares the views exactly as in training (scene views, object crops chosen from the masks, asset
renders at fixed viewpoints), runs one forward pass and reads out the object poses. Nothing here reads ground
truth. The output poses are in the frame of the first scene camera (OpenCV axes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..constants import CONDITION_MODES, ROLE_CROP, ROLE_SCENE, mode_flags
from ..data.assets import Asset
from ..data.conditions import apply_conditions, draw_owner_slots, richest_mode
from ..data.views import (SampleBuilder, centered_crop_box, crop_candidates, inference_render_views, mask_box,
                          order_views, resample_view)
from ..geometry.camera import nearest_aspect, scene_shape, scene_window
from .readout import read_out

SOURCE_SCALE = 2.0            # crops are cut from images with (at most) twice the resolution of the scene views


@dataclass
class SceneObservation:
    """The observed scene: S images of equal size with per-pixel object ids and optional geometric conditions.

    ``masks[j]`` is an (H, W) integer map: 0 = background, i + 1 = object i (the i-th asset). ``intrinsics`` is
    (3, 3) or (S, 3, 3) for the given image size, ``poses`` (S, 4, 4) camera-to-world in any world frame,
    ``depths[j]`` (H, W) depth along the optical axis with 0 = unknown. Poses and depth require intrinsics.
    """

    images: Sequence[np.ndarray]
    masks: Sequence[np.ndarray]
    intrinsics: np.ndarray | None = None
    poses: np.ndarray | None = None
    depths: Sequence[np.ndarray] | None = None

    def __post_init__(self) -> None:
        shape = self.images[0].shape[:2]
        if len(self.masks) != len(self.images) or any(a.shape[:2] != shape for a in (*self.images, *self.masks)):
            raise ValueError("Every image needs a mask, and all images of a scene must have the same size")
        if (self.poses is not None or self.depths is not None) and self.intrinsics is None:
            raise ValueError("Camera poses and depth require intrinsics")
        if self.intrinsics is not None:
            K = np.asarray(self.intrinsics, np.float64)
            self.intrinsics = np.broadcast_to(K, (len(self.images), 3, 3)).copy()


@dataclass
class Layout:
    """Result of ``compose``: per object the pose of its asset frame and of its original mesh in the scene frame."""

    objects: list[dict]                       # R, t, scale, pose (asset -> scene), mesh_pose (mesh -> scene), asset_id
    camera_c2w: np.ndarray                    # (S, 4, 4) scene cameras, the first one is the identity
    camera_K: np.ndarray                      # (S, 3, 3) intrinsics of the model's scene views
    unit: str                                 # "input" (unit of the given depth / poses) or "normalized"
    mode: str
    depth: np.ndarray | None = None           # (S, h, w) predicted depth of the model's scene views
    assets: list[Asset] = field(default_factory=list, repr=False)


def _working_views(scene: SceneObservation, hw: tuple[int, int]) -> tuple[list[dict], np.ndarray, tuple[int, int]]:
    """The source images at the working resolution (at most ``SOURCE_SCALE`` source pixels per scene-view pixel),
    with their intrinsics (principal point at the image centre and unit focal length when none are given)."""
    H, W = scene.images[0].shape[:2]
    shrink = max(1.0, min(W / hw[1], H / hw[0]) / SOURCE_SCALE)
    size = (int(round(H / shrink)), int(round(W / shrink)))
    views, intrinsics = [], []
    for j, image in enumerate(scene.images):
        depth = np.zeros((H, W), np.float32) if scene.depths is None else np.asarray(scene.depths[j], np.float32)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
        view = resample_view(np.ascontiguousarray(image[..., :3], dtype=np.uint8), np.asarray(scene.masks[j]),
                             depth, [0, 0, W, H], size, "edge")
        K = (np.array([[1.0, 0, (W - 1) / 2], [0, 1.0, (H - 1) / 2], [0, 0, 1]]) if scene.intrinsics is None
             else scene.intrinsics[j])
        views.append({"rgb": np.rint(view["rgb"].transpose(1, 2, 0) * 255).astype(np.uint8), "ids": view["ids"],
                      "depth": view["depth"]})
        intrinsics.append(view["A"] @ K)
    return views, np.stack(intrinsics), size


def _crops(views: list[dict], objects: int, image_hw: tuple[int, int]) -> list[tuple]:
    """(object, view position, box) of every crop: ``views.crop_candidates`` per object, or the view with its
    largest mask if no view qualifies."""
    crops = []
    for index in range(objects):
        masks = [view["ids"] == index + 1 for view in views]
        boxes = {j: mask_box(m) for j, m in enumerate(masks)}
        boxes = {j: b for j, b in boxes.items() if b is not None}
        pixels = {j: int(masks[j].sum()) for j in boxes}
        chosen = crop_candidates(boxes, pixels, image_hw) or sorted(pixels, key=lambda j: -pixels[j])[:1]
        crops += [(index, j, centered_crop_box(boxes[j])) for j in chosen]
    return crops


def prepare(scene: SceneObservation, assets: Sequence[Asset], mode: str = "auto", renders_per_object: int = 4,
            seed: int = 0, conditions_estimated: bool = False) -> tuple[dict, dict]:
    """Model inputs (NumPy, one sample) and the condition record for ``scene`` and its ``assets``.

    ``mode``: one of ``CONDITION_MODES`` or ``auto`` = everything the scene provides. With
    ``conditions_estimated`` (intrinsics / depth from a monocular estimator) and a single image, ``auto`` selects
    mode ``none``.
    """
    count = len(assets)
    if max(int(np.max(m)) for m in scene.masks) > count:
        raise ValueError("Mask ids must be 0 (background) or 1..number of assets")
    views_n = len(scene.images)
    hw = scene_shape(nearest_aspect(*scene.images[0].shape[:2]))
    if mode == "auto":
        mode = richest_mode(scene.intrinsics is not None, scene.poses is not None, scene.depths is not None, views_n)
        if conditions_estimated and views_n == 1:
            mode = "none"
    intrinsics, poses, depth = mode_flags(mode)
    if (intrinsics and scene.intrinsics is None) or (poses and scene.poses is None) or (depth and scene.depths is None):
        raise ValueError(f"Mode {mode} needs conditions that the scene does not provide")
    views, K, size = _working_views(scene, hw)
    relative = [np.eye(4)] * views_n
    if scene.poses is not None:
        first = np.linalg.inv(np.asarray(scene.poses[0], np.float64))
        relative = [first @ np.asarray(p, np.float64) for p in scene.poses]
    builder = SampleBuilder(count)
    ids = list(range(1, count + 1))
    maps = []
    for j, view in enumerate(views):
        raster = resample_view(view["rgb"], view["ids"], view["depth"], scene_window(K[j], size, hw), hw, "edge")
        builder.add_view(ROLE_SCENE, -1, j, raster, ids, K[j], relative[j])
        maps.append(raster["A"])
    renders = [(index, k) for index in range(count) for k in inference_render_views(renders_per_object)]
    crops, renders = order_views(_crops(views, count, size), renders)
    for index, j, box, pixels in crops:
        crop = resample_view(views[j]["rgb"], views[j]["ids"], views[j]["depth"], box, (pixels, pixels))
        builder.add_view(ROLE_CROP, index, j, crop, ids, K[j], relative[j], crop["A"] @ np.linalg.inv(maps[j]))
    for index, k, pixels in renders:
        builder.add_render(index, assets[index].view(k), pixels)
    x, record = apply_conditions(builder.stack(), mode)
    x["owner_slots"] = draw_owner_slots(count, np.random.default_rng(seed))
    return x, record


@torch.no_grad()
def compose(model: torch.nn.Module, scene: SceneObservation, assets: Sequence[Asset], *, mode: str = "auto",
            renders_per_object: int = 4, candidate_policy: str = "mean", known_sizes: Sequence | None = None,
            conditions_estimated: bool = False, seed: int = 0) -> Layout:
    """Predict the layout of ``assets`` in ``scene`` with one forward pass.

    ``mode``: ``auto`` or one of ``K+P+D``, ``K+P``, ``K+D``, ``K``, ``none``. ``renders_per_object``: 1 to 8.
    ``known_sizes``: per object None or its known scale (number or 3-vector, see ``readout.resolve_scales``).
    ``seed`` fixes the random assignment of identity embeddings to objects.
    """
    if mode != "auto" and mode not in CONDITION_MODES:
        raise ValueError(f"mode must be 'auto' or one of {CONDITION_MODES}")
    if not 1 <= renders_per_object <= 8:
        raise ValueError("renders_per_object must be between 1 and 8")
    x, record = prepare(scene, assets, mode, renders_per_object, seed, conditions_estimated)
    device = next(model.parameters()).device
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model(x)
    result = read_out(prediction, x, record, candidate_policy=candidate_policy, known_sizes=known_sizes)
    objects = []
    for asset, item in zip(assets, result["objects"]):
        objects.append({"asset_id": asset.meta.get("asset_id", asset.path.name), "R": item["R"], "t": item["t"],
                        "scale": item["scale"], "pose": item["pose"], "mesh_pose": item["pose"] @ asset.N,
                        "rotation_spread_deg": item["rotation_spread_deg"]})
    h, w = (int(v) for v in x["view_size"][0])
    depth = result["depth"][:, :h, :w] if "depth" in result else None
    return Layout(objects, result["camera_c2w"], result["camera_K"], result["unit"], result["mode"], depth,
                  list(assets))
