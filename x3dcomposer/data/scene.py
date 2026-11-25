"""Reader of a training scene directory and its visibility table (format: docs/DATA.md).

    <scene>/scene.json      scene_id, split, source, depth_unit, synthetic, optional up / floor_height,
                            instances[]: id, asset_id, R (3x3), t (3), scale (3), optional trainable
    <scene>/cameras.npz     K (V, 3, 3), c2w (V, 4, 4): OpenCV camera-to-world poses in a metric world frame
    <scene>/rgb/VVVV.jpg    image of view V
    <scene>/ids/VVVV.png    uint16 instance id per pixel (0 = no annotated object)
    <scene>/depth/VVVV.png  uint16 depth along the optical axis in ``depth_unit`` metres (0 = invalid)
    <scene>/visibility.npz  one row per (view, instance) with visible pixels (``tools/build_visibility.py``)

The label of an instance places its asset in the world: ``x_world = R @ diag(scale) @ p_asset + t``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

VISIBILITY_FILE = "visibility.npz"
VISIBILITY_COLUMNS = ("view", "instance", "pixels", "box", "depth_median", "visible_fraction", "in_frame_fraction")


def _find(folder: Path, index: int, suffixes: tuple[str, ...]) -> Path:
    for suffix in suffixes:
        file = folder / f"{index:04d}{suffix}"
        if file.is_file():
            return file
    raise FileNotFoundError(f"{folder}/{index:04d}{suffixes}")


class Scene:
    """One scene directory: metadata, cameras, labels and decoded views."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "scene.json").read_text())
        with np.load(self.path / "cameras.npz") as cameras:
            self.K, self.c2w = cameras["K"].astype(np.float64), cameras["c2w"].astype(np.float64)
        self.instances = {int(item["id"]): item for item in self.meta["instances"]}
        self.depth_unit = float(self.meta.get("depth_unit", 0.001))
        self.up = np.asarray(self.meta.get("up", (0.0, 1.0, 0.0)), np.float64)

    @property
    def scene_id(self) -> str:
        """Identifier of the scene (the directory name if ``scene.json`` has none)."""
        return str(self.meta.get("scene_id", self.path.name))

    @property
    def num_views(self) -> int:
        """Number of views."""
        return len(self.K)

    def image_hw(self) -> tuple[int, int]:
        """(H, W) of the stored images (read from the first id map)."""
        with Image.open(_find(self.path / "ids", 0, (".png",))) as image:
            return image.size[1], image.size[0]

    def ids(self, view: int) -> np.ndarray:
        """int32 instance id map of a view."""
        return np.asarray(Image.open(_find(self.path / "ids", view, (".png",)))).astype(np.int32)

    def depth(self, view: int) -> np.ndarray:
        """float32 depth in metres, 0 where invalid."""
        raw = np.asarray(Image.open(_find(self.path / "depth", view, (".png",))))
        return raw.astype(np.float32) * self.depth_unit

    def view(self, view: int) -> dict:
        """``rgb`` uint8 (H, W, 3), ``ids`` int32, ``depth`` float32 metres, ``K`` (3, 3), ``c2w`` (4, 4)."""
        rgb = np.asarray(Image.open(_find(self.path / "rgb", view, (".jpg", ".png"))).convert("RGB"))
        return {"rgb": rgb, "ids": self.ids(view), "depth": self.depth(view), "K": self.K[view].copy(),
                "c2w": self.c2w[view].copy()}

    def label(self, instance: int) -> tuple[np.ndarray, np.ndarray]:
        """World pose (4, 4) = [R | t] and per-axis scale (3,) of an instance."""
        item = self.instances[int(instance)]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = np.asarray(item["R"], np.float64), np.asarray(item["t"], np.float64)
        return pose, np.asarray(item["scale"], np.float64)

    def visibility(self) -> dict[str, np.ndarray]:
        """The visibility table (computed from the id and depth maps if the file is missing)."""
        file = self.path / VISIBILITY_FILE
        if not file.is_file():
            return compute_visibility(self)
        with np.load(file) as table:
            return {key: table[key] for key in VISIBILITY_COLUMNS}


def compute_visibility(scene: Scene) -> dict[str, np.ndarray]:
    """Visibility rows of every (view, labelled instance) with visible pixels.

    ``pixels`` and ``box`` = [x0, y0, x1, y1) describe the visible mask, ``depth_median`` its median depth (NaN
    without valid depth). ``visible_fraction`` (visible / unoccluded silhouette inside the image) and
    ``in_frame_fraction`` (silhouette inside the image / whole silhouette) need the posed mesh; they are NaN
    here, which the sampler reads as "unknown, accept".
    """
    rows = {key: [] for key in VISIBILITY_COLUMNS}
    for view in range(scene.num_views):
        ids, depth = scene.ids(view), scene.depth(view)
        for instance in np.unique(ids):
            if int(instance) not in scene.instances:
                continue
            mask = ids == instance
            ys, xs = np.nonzero(mask)
            z = depth[mask]
            z = z[z > 0]
            rows["view"].append(view)
            rows["instance"].append(int(instance))
            rows["pixels"].append(len(xs))
            rows["box"].append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            rows["depth_median"].append(float(np.median(z)) if len(z) else np.nan)
            rows["visible_fraction"].append(np.nan)
            rows["in_frame_fraction"].append(np.nan)
    dtypes = {"view": np.int32, "instance": np.int32, "pixels": np.int64, "box": np.int32}
    return {key: np.asarray(value, dtypes.get(key, np.float32)).reshape((len(value), 4) if key == "box" else -1)
            for key, value in rows.items()}


def read_scene_list(path: str | Path) -> list[str]:
    """Scene directories listed one per line (relative paths are taken from the list's directory)."""
    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    return [str(p if (p := Path(line)).is_absolute() else path.parent / p) for line in lines]
