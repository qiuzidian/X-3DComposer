"""Pinhole cameras, pixel maps and the view rasters of the model (NumPy).

Conventions: OpenCV camera axes (x right, y down, z forward), camera-to-world matrices ``c2w``, pixel centres at
integer coordinates, depth along the optical axis. A *pixel map* A is the 3x3 affine that takes source pixel
coordinates to raster pixel coordinates, so the intrinsics of a resampled view are ``A @ K``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

PATCH = 16
SCENE_PIXELS = 416            # scene views have about 416^2 pixels (676 patches) at any aspect ratio
# Height / width ratios of the scene views (the discrete set of the backbone's training).
ASPECTS = (5 / 9, 10 / 16, 2 / 3, 3 / 4, 4 / 5, 1.0, 5 / 4, 4 / 3, 3 / 2, 16 / 10, 9 / 5)


def scene_shape(aspect: float) -> tuple[int, int]:
    """(h, w) of a scene view with ``aspect`` = h / w, keeping about 676 patches."""
    tokens = (SCENE_PIXELS // PATCH) ** 2
    w_patches = np.sqrt(tokens / aspect)
    h_patches = tokens / w_patches
    return int(np.round(h_patches)) * PATCH, int(np.round(w_patches)) * PATCH


def nearest_aspect(height: int, width: int) -> float:
    """The entry of ``ASPECTS`` closest (in log ratio) to ``height / width``."""
    return min(ASPECTS, key=lambda a: abs(np.log(a) - np.log(height / width)))


def pixel_map(box: Sequence[float], hw: tuple[int, int]) -> np.ndarray:
    """Pixel map of cropping the source region ``box`` = [x0, y0, x1, y1) (pixel-edge coordinates: pixel i
    covers [i, i + 1)) and resizing it to ``hw``."""
    x0, y0, x1, y1 = map(float, box)
    h, w = hw
    if not (x1 > x0 and y1 > y0):
        raise ValueError("Empty box")
    ax, ay = w / (x1 - x0), h / (y1 - y0)
    return np.array([[ax, 0, -ax * x0 + (ax - 1) / 2], [0, ay, -ay * y0 + (ay - 1) / 2], [0, 0, 1]], np.float64)


def scene_window(K: np.ndarray, source_hw: tuple[int, int], hw: tuple[int, int]) -> np.ndarray:
    """Source region [x0, y0, x1, y1) of a scene view of raster ``hw``: the largest window of the raster's
    aspect ratio that fits the image, centred on the principal point (so the raster's principal point is
    its centre). The window leaves the image where the principal point is off-centre."""
    H, W = source_hw
    h, w = hw
    s = min(W / w, H / h)                      # source pixels per raster pixel
    cx, cy = K[0, 2] + 0.5, K[1, 2] + 0.5
    return np.array([cx - s * w / 2, cy - s * h / 2, cx + s * w / 2, cy + s * h / 2])


def centered_K(K: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """``K`` with zero skew and the principal point at (w / 2, h / 2): the form the camera encoding expresses."""
    h, w = hw
    return np.array([[K[0, 0], 0, w / 2], [0, K[1, 1], h / 2], [0, 0, 1]], np.float64)


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 (or 3x4) affine to row points (N, 3)."""
    return np.asarray(points, np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def project(points: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points -> pixel coordinates (N, 2) and depth (N,)."""
    cam = transform(points, np.linalg.inv(c2w))
    hom = cam @ np.asarray(K, np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2:3], cam[:, 2]


def unproject(depth: np.ndarray, valid: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World points (M, 3) of the valid pixels of a depth map."""
    yy, xx = np.nonzero(valid)
    cam = (np.linalg.inv(K) @ np.stack([xx, yy, np.ones(len(xx))])).T * depth[yy, xx, None]
    return transform(cam, c2w)


def decode_pose_encoding(pose_enc: np.ndarray, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """9-d camera encodings (..., 9) -> ``c2w`` (..., 4, 4) and centred intrinsics (..., 3, 3) for raster ``hw``.

    The encoding holds the world-to-camera translation, the world-to-camera XYZW quaternion and the vertical /
    horizontal field of view.
    """
    e = np.asarray(pose_enc, np.float64)
    q = e[..., 3:7]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(e).all() or np.any(norm <= 1e-8):
        raise ValueError("Invalid camera encoding")
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(e.shape[:-1] + (3, 3))
    c2w = np.broadcast_to(np.eye(4), e.shape[:-1] + (4, 4)).copy()
    c2w[..., :3, :3] = np.swapaxes(R, -1, -2)
    c2w[..., :3, 3] = -(c2w[..., :3, :3] @ e[..., :3, None])[..., 0]
    H, W = hw
    K = np.zeros(e.shape[:-1] + (3, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        K[..., 0, 0] = W / (2 * np.tan(e[..., 8] / 2))
        K[..., 1, 1] = H / (2 * np.tan(e[..., 7] / 2))
    K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = W / 2, H / 2, 1
    return c2w, K


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal mean of rotation matrices (M, 3, 3): their average projected onto SO(3) by SVD."""
    U, _, Vt = np.linalg.svd(np.asarray(rotations, np.float64).mean(0))
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def canonical_frame(vertices: np.ndarray) -> np.ndarray:
    """4x4 map from mesh coordinates to the asset frame: bounding box centred at the origin, longest side 1."""
    low, high = vertices.min(0), vertices.max(0)
    extent = float((high - low).max())
    if not np.isfinite(extent) or extent <= 1e-8:
        raise ValueError("Degenerate mesh bounds")
    N = np.eye(4)
    N[:3, :3] /= extent
    N[:3, 3] = -(low + high) / 2 / extent
    return N
