"""The fixed viewpoints from which an asset is rendered (NumPy).

32 upright views in the asset frame (+Y up; azimuth 0 looks at the asset from +Z):

    0..23   three rings of 8 azimuths at elevations 10, 25 and 45 degrees; ring r is turned by 15 * r degrees
    24..31  4 views from below (-20 degrees) and 4 from above (65 degrees), between the ring azimuths

All rings share one azimuth offset in [0, 45) degrees derived from the asset id, so no asset is always seen
exactly from its front. Every view has a 50 degree field of view and is placed so that the whole mesh fills
0.88 of the image (``fit_view``). Inference uses views 0..15, training all of them.
"""

from __future__ import annotations

import hashlib

import numpy as np

SIZE = 608
FOV_DEG = 50.0
FILL = 0.88
AZIMUTHS = 8
ELEVATIONS_DEG = (10.0, 25.0, 45.0)
RING_STAGGER_DEG = 15.0
EXTRA_ELEVATIONS_DEG = (-20.0, 65.0)
EXTRA_AZIMUTHS = 4
UP = np.array([0.0, 1.0, 0.0])


def view_directions(asset_id: str) -> list[dict]:
    """``azimuth_deg``, ``elevation_deg`` and unit ``direction`` (asset -> camera) of the 32 views."""
    digest = hashlib.sha256(("asset-views:" + asset_id).encode()).digest()
    offset = int.from_bytes(digest[:8], "little") / 2 ** 64 * 360.0 / AZIMUTHS

    def view(azimuth: float, elevation: float) -> dict:
        a, e = np.radians(azimuth), np.radians(elevation)
        return {"azimuth_deg": float(azimuth), "elevation_deg": float(elevation),
                "direction": [float(np.cos(e) * np.sin(a)), float(np.sin(e)), float(np.cos(e) * np.cos(a))]}

    views = [view((offset + 360.0 * k / AZIMUTHS + ring * RING_STAGGER_DEG) % 360.0, elevation)
             for ring, elevation in enumerate(ELEVATIONS_DEG) for k in range(AZIMUTHS)]
    for j, elevation in enumerate(EXTRA_ELEVATIONS_DEG):
        shift = 180.0 / EXTRA_AZIMUTHS * (j + 0.5)
        views += [view((offset + shift + 360.0 * k / EXTRA_AZIMUTHS) % 360.0, elevation) for k in range(EXTRA_AZIMUTHS)]
    return views


def camera_rotation(direction) -> np.ndarray:
    """OpenCV camera axes (columns x, y, z) of an upright camera looking along ``-direction``."""
    z = -np.asarray(direction, np.float64) / np.linalg.norm(direction)
    x = np.cross(z, UP)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def fit_view(vertices: np.ndarray, direction, size: int = SIZE, fill: float = FILL,
             fov_deg: float = FOV_DEG) -> tuple[np.ndarray, np.ndarray]:
    """Camera ``c2w`` (4, 4) and ``K`` (3, 3) on ``direction`` that frames all ``vertices`` (N, 3) within ``fill``
    of the image: the closest camera for which every vertex projects inside, shifted sideways to centre them."""
    R = camera_rotation(direction)
    focal = size / 2 / np.tan(np.radians(fov_deg) / 2)
    q = np.asarray(vertices, np.float64) @ R
    limits = fill * (np.array([size, size]) - 1) / (2 * focal)
    lower = np.max(q[:, :2] - q[:, 2:] * limits, axis=0)
    upper = np.min(q[:, :2] + q[:, 2:] * limits, axis=0)
    distance = max(float(np.max((lower - upper) / (2 * limits))), 0.06 - float(q[:, 2].min()))
    lateral = ((lower - limits * distance) + (upper + limits * distance)) / 2
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = R @ np.array([*lateral, -distance])
    K = np.array([[focal, 0, (size - 1) / 2], [0, focal, (size - 1) / 2], [0, 0, 1.0]])
    return c2w, K


def sample_surface(vertices: np.ndarray, triangles: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """``count`` area-weighted random points on a triangle mesh."""
    rng = np.random.default_rng(seed)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    chosen = rng.choice(len(triangles), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    return a[chosen] + u[:, None] * (b[chosen] - a[chosen]) + v[:, None] * (c[chosen] - a[chosen])
