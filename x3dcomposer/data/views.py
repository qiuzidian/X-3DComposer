"""Resampling of images into model views and the row-by-row builder of a model input sample (NumPy + Pillow).

Used by both inference (``x3dcomposer.inference.pipeline``) and training (``x3dcomposer.data.assemble``), so the
two see identically prepared views:

* a *scene view* is the window of the source image given by ``geometry.camera.scene_window``, resized to the
  scene raster of the image's aspect ratio (about 416^2 pixels);
* an *object crop* is a square box around the object's mask, cut from the full-resolution source image and
  resized to 336 or 256 pixels; the box may leave the image (zero padding, ``image_valid`` False);
* an *asset render* is a pre-rendered view of the asset resized to 336 or 256 pixels.

RGB is resampled with an antialiased bilinear filter; instance ids and depth with nearest neighbour at the
raster pixel centres (consistent with ``K_view = A @ K_source``).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image

from ..constants import ROLE_RENDER, ROLE_SCENE
from ..geometry.camera import pixel_map

CROP_SIZES = (336, 256)       # crops whose box side is at least 336 source pixels use 336, the others 256
CROP_CONTEXT = 1.3            # inference crop box side / long side of the object's mask box
MIN_CROP_PIXELS = 64          # crop candidates: mask pixels, mask long side
MIN_CROP_AREA = 0.0025        # ... mask area / image area
MAX_CROP_BOX = 0.8            # ... mask box side / image side
MAX_CROPS_PER_OBJECT = 3
RING_AZIMUTHS = 8             # asset views 0..7: elevation 10 deg, 8..15: 25 deg (see docs/DATA.md)


def nearest(array: np.ndarray, A: np.ndarray, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour resampling of an (H, W) array through the axis-aligned pixel map ``A``; returns the
    raster and its support (False where the raster pixel centre falls outside the source)."""
    h, w = array.shape[:2]
    inverse = np.linalg.inv(A)
    x = inverse[0, 0] * np.arange(hw[1], dtype=np.float64) + inverse[0, 2]
    y = inverse[1, 1] * np.arange(hw[0], dtype=np.float64) + inverse[1, 2]
    valid = ((y >= -0.5) & (y < h - 0.5))[:, None] & ((x >= -0.5) & (x < w - 0.5))[None, :]
    ix = np.clip(np.floor(x + 0.5).astype(int), 0, w - 1)
    iy = np.clip(np.floor(y + 0.5).astype(int), 0, h - 1)
    value = array[iy[:, None], ix[None, :]]
    value[~valid] = 0
    return value, valid


def resize_rgb(rgb: np.ndarray, box: Sequence[float], hw: tuple[int, int], pad_mode: str) -> np.ndarray:
    """uint8 (H, W, 3) region ``box`` (pixel-edge coordinates, may leave the image) -> (3, h, w) float32 in [0, 1].

    ``pad_mode``: how the image is continued outside its border, ``"edge"`` or ``"constant"`` (zeros).
    """
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    pad = int(math.ceil(max(1.0 if pad_mode == "edge" else 0.0, -x0, -y0, x1 - width, y1 - height)))
    if pad:
        rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)
    out = Image.fromarray(rgb).resize((hw[1], hw[0]), Image.BILINEAR, box=(x0 + pad, y0 + pad, x1 + pad, y1 + pad))
    return (np.asarray(out, np.float32) / 255).transpose(2, 0, 1)


def resample_view(rgb: np.ndarray, ids: np.ndarray, depth: np.ndarray, box: Sequence[float],
                  hw: tuple[int, int], pad_mode: str = "constant") -> dict:
    """Cut ``box`` from a source view and resize it to ``hw``.

    ``rgb`` uint8 (H, W, 3), ``ids`` integer instance ids (H, W), ``depth`` float (H, W) with 0 = invalid.
    Returns ``rgb`` (3, h, w), ``ids``, ``depth``, ``depth_valid``, ``support`` (h, w) and the pixel map ``A``.
    """
    A = pixel_map(box, hw)
    support = nearest(np.ones(ids.shape[:2], np.uint8), A, hw)[0].astype(bool)
    labels = nearest(ids, A, hw)[0]
    z = nearest(depth, A, hw)[0]
    return {"rgb": resize_rgb(rgb, box, hw, pad_mode) * support, "ids": labels, "depth": z,
            "depth_valid": (z > 0) & support, "support": support, "A": A}


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """[x0, y0, x1, y1) of a boolean mask (None if empty)."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def centered_crop_box(box: Sequence[int], context: float = CROP_CONTEXT) -> list[int]:
    """Square crop box of side ``ceil(context * L)`` (L = long side of the mask box), centred on the mask box
    and not clipped to the image."""
    x0, y0, x1, y1 = (int(b) for b in box)
    side = int(math.ceil(context * max(x1 - x0, y1 - y0)))
    left = int(math.floor((x0 + x1 - side) / 2))
    top = int(math.floor((y0 + y1 - side) / 2))
    return [left, top, left + side, top + side]


def crop_candidates(boxes: dict, pixels: dict, image_hw: tuple[int, int]) -> list[int]:
    """The views an object gets a crop from, chosen from its masks only.

    ``boxes`` / ``pixels`` map a view position to the object's mask box and pixel count in that view. A view
    qualifies with at least 64 mask pixels, 0.25 % of the image area and a mask box of at most 0.8 of the image
    per side. The qualifying views are ranked by the long side L of the mask box; those with L >= 64 are kept
    (the best one otherwise), at most three. Returns view positions in ascending order.
    """
    H, W = image_hw
    ranked = []
    for view, box in boxes.items():
        width, height = box[2] - box[0], box[3] - box[1]
        if (pixels[view] >= MIN_CROP_PIXELS and pixels[view] >= MIN_CROP_AREA * H * W
                and width <= MAX_CROP_BOX * W and height <= MAX_CROP_BOX * H):
            ranked.append((max(width, height), pixels[view], view))
    ranked.sort(key=lambda c: (-c[0], -c[1], c[2]))
    kept = [c for c in ranked if c[0] >= MIN_CROP_PIXELS] or ranked[:1]
    return sorted(c[2] for c in kept[:MAX_CROPS_PER_OBJECT])


def crop_size(box: Sequence[int]) -> int:
    """Raster size of a crop: 336 if its box side is at least 336 source pixels, else 256."""
    return CROP_SIZES[0] if box[2] - box[0] >= CROP_SIZES[0] else CROP_SIZES[1]


def order_views(crops: list[tuple], renders: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    """Order crops and renders so that views of equal size are consecutive, and attach their sizes.

    ``crops``: (object, view position, box); ``renders``: (object, asset view). Crops are sorted by box side
    (largest first). The renders of an object that has a 336-pixel crop are 336 pixels and come first; the other
    renders are 256 pixels. Returns ``[(object, view position, box, size)]`` and ``[(object, asset view, size)]``.
    """
    crops = sorted(crops, key=lambda c: -(c[2][2] - c[2][0]))
    sized = [(o, j, box, crop_size(box)) for o, j, box in crops]
    large = {o for o, _, _, size in sized if size == CROP_SIZES[0]}
    owners = sorted({o for o, _ in renders}, key=lambda o: (o not in large, o))
    ordered = [(o, k, CROP_SIZES[0] if o in large else CROP_SIZES[1]) for o in owners for owner, k in renders
               if owner == o]
    return sized, ordered


def inference_render_views(count: int) -> list[int]:
    """Asset view indices used at inference: ``count`` viewpoints spread evenly in azimuth, at elevations
    alternating between 10 and 25 degrees."""
    return [(i % 2) * RING_AZIMUTHS + round(i * RING_AZIMUTHS / count) % RING_AZIMUTHS for i in range(count)]


class SampleBuilder:
    """Collects the views of one sample row by row and stacks them into the model input arrays.

    Every dense array is zero-padded to the canvas (the largest view); ``view_size`` records each view's size.
    """

    def __init__(self, objects: int):
        self.objects = objects
        self.rows: list[dict] = []

    def add(self, role: int, owner: int, parent: int, rgb: np.ndarray, masks: np.ndarray, depth: np.ndarray,
            depth_valid: np.ndarray, K: np.ndarray, c2w: np.ndarray, crop_A: np.ndarray | None = None,
            image_valid: np.ndarray | None = None) -> int:
        """Append a view; ``masks`` is (O, h, w). Returns the row index."""
        hw = rgb.shape[-2:]
        self.rows.append(dict(
            images=rgb.astype(np.float32), instance_masks=masks.astype(np.uint8), role=role, owner=owner,
            parent=len(self.rows) if role == ROLE_SCENE else parent, K=K, camera_c2w=c2w,
            crop_A=np.eye(3) if crop_A is None else crop_A,
            depth=np.where(depth_valid, depth, 0)[None].astype(np.float32), depth_valid=depth_valid[None],
            image_valid=np.ones((1,) + tuple(hw), bool) if image_valid is None else image_valid[None],
            view_size=list(hw)))
        return len(self.rows) - 1

    def add_view(self, role: int, owner: int, parent: int, view: dict, object_ids: Sequence[int], K: np.ndarray,
                 c2w: np.ndarray, crop_A: np.ndarray | None = None) -> int:
        """Append a scene view or crop from a ``resample_view`` result (``K`` = intrinsics of the source)."""
        masks = np.stack([view["ids"] == pid for pid in object_ids])
        return self.add(role, owner, parent, view["rgb"], masks, view["depth"], view["depth_valid"],
                        view["A"] @ K, c2w, crop_A, view["support"])

    def add_render(self, owner: int, render: dict, size: int) -> int:
        """Append an asset render (``x3dcomposer.data.assets.Asset.view``) resized to ``size``."""
        side = render["rgb"].shape[0]
        A = pixel_map([0, 0, side, side], (size, size))
        foreground = nearest(render["mask"].astype(np.uint8), A, (size, size))[0].astype(bool)
        depth = nearest(render["depth"], A, (size, size))[0]
        masks = np.zeros((self.objects, size, size), bool)
        masks[owner] = foreground
        rgb = resize_rgb(render["rgb"], [0, 0, side, side], (size, size), "constant")
        return self.add(ROLE_RENDER, owner, -1, rgb, masks, depth, foreground & (depth > 0), A @ render["K"],
                        render["c2w"])

    def canvas(self) -> tuple[int, int]:
        """(H, W) of the largest view."""
        return (max(r["view_size"][0] for r in self.rows), max(r["view_size"][1] for r in self.rows))

    def stack(self) -> dict[str, np.ndarray]:
        """The sample as arrays (all conditions marked as given; see ``data.conditions.apply_conditions``)."""
        H, W = self.canvas()

        def pad(a: np.ndarray) -> np.ndarray:
            out = np.zeros(a.shape[:-2] + (H, W), a.dtype)
            out[..., :a.shape[-2], :a.shape[-1]] = a
            return out

        dense = ("images", "instance_masks", "depth", "depth_valid", "image_valid")
        x = {key: np.stack([pad(r[key]) for r in self.rows]) for key in dense}
        for key, dtype in (("K", np.float32), ("camera_c2w", np.float32), ("crop_A", np.float32),
                           ("role", np.int64), ("owner", np.int64), ("parent", np.int64), ("view_size", np.int64)):
            x[key] = np.asarray([r[key] for r in self.rows], dtype)
        count = len(self.rows)
        for key in ("camera_available", "pose_available", "depth_available"):
            x[key] = np.ones(count, bool)
        x["owner_slots"] = np.arange(self.objects, dtype=np.int64)
        return x

