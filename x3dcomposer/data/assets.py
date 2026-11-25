"""Reader of a pre-rendered asset directory (format: docs/DATA.md; written by ``tools/render_asset.py``).

    <asset>/asset.json     asset_id, N_asset_from_mesh (4x4), size, depth_scale, views[], optional mesh / symmetry
    <asset>/cameras.npz    K (V, 3, 3), c2w (V, 4, 4): asset-from-camera poses of the V views
    <asset>/rgb/KK.jpg     render on a uniform gray background
    <asset>/mask/KK.png    uint8 foreground mask (0 / 255)
    <asset>/depth/KK.png   uint16 depth in asset units * depth_scale (0 = background)
    <asset>/points.npy     optional (P, 3) surface samples in the asset frame (used by the floor loss)

The asset frame has +Y up, its bounding box centred at the origin and its longest side equal to 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

SYMMETRY_NONE, SYMMETRY_CONTINUOUS = 1, 0     # numeric codes: 1 = none, n >= 2 = n-fold about +Y, 0 = continuous


def symmetry_code(name: str | None) -> int:
    """``None`` / ``"none"`` -> 1, ``"C4"`` -> 4, ``"SO2"`` -> 0 (rotational symmetry about the asset's +Y axis)."""
    if name is None or name.lower() == "none":
        return SYMMETRY_NONE
    if name.upper() == "SO2":
        return SYMMETRY_CONTINUOUS
    if name.upper().startswith("C") and name[1:].isdigit() and 2 <= int(name[1:]) <= 12:
        return int(name[1:])
    raise ValueError(f"Unknown symmetry {name!r}; use none, C2..C12 or SO2")


class Asset:
    """One asset directory: metadata, cameras and decoded views."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "asset.json").read_text())
        with np.load(self.path / "cameras.npz") as cameras:
            self.K, self.c2w = cameras["K"].astype(np.float64), cameras["c2w"].astype(np.float64)
        self.N = np.asarray(self.meta["N_asset_from_mesh"], np.float64)
        self.depth_scale = float(self.meta.get("depth_scale", 1e4))
        self.symmetry = symmetry_code(self.meta.get("symmetry"))

    @property
    def num_views(self) -> int:
        """Number of rendered views."""
        return len(self.K)

    @property
    def mesh_path(self) -> Path | None:
        """The mesh the asset was rendered from, if the asset records it."""
        mesh = self.meta.get("mesh")
        return None if mesh is None else (self.path / mesh)

    def _image(self, folder: str, index: int, suffixes: tuple[str, ...]) -> Image.Image:
        for suffix in suffixes:
            file = self.path / folder / f"{index:02d}{suffix}"
            if file.is_file():
                return Image.open(file)
        raise FileNotFoundError(f"{self.path}/{folder}/{index:02d}{suffixes}")

    def view(self, index: int) -> dict:
        """``rgb`` uint8 (s, s, 3), ``mask`` bool, ``depth`` float32 (asset units, 0 off the object), ``K``, ``c2w``."""
        mask = np.asarray(self._image("mask", index, (".png",))) > 0
        depth = np.asarray(self._image("depth", index, (".png",))).astype(np.float32) / self.depth_scale
        rgb = np.asarray(self._image("rgb", index, (".jpg", ".png")).convert("RGB"))
        return {"rgb": rgb, "mask": mask, "depth": np.where(mask, depth, 0).astype(np.float32),
                "K": self.K[index].copy(), "c2w": self.c2w[index].copy()}

    def points(self) -> np.ndarray | None:
        """(P, 3) surface points in the asset frame, or None if the asset has none."""
        file = self.path / "points.npy"
        return np.load(file).astype(np.float64) if file.is_file() else None
