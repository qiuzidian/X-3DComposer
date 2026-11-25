"""Training-time appearance augmentation of an assembled sample (NumPy + Pillow).

* ``degrade_source``: slight blur or JPEG compression of a source scene image (before scene views and crops are
  cut from it, so they stay consistent);
* ``scene_exposure``: one exposure / white-balance gain shared by a scene view and its crops;
* ``render_color_gain``: one colour gain shared by all renders of an object;
* ``render_jitter``: brightness, contrast, saturation, hue, gamma and blur / sharpening of a render's object
  pixels (the gray background stays), which mimics the texture mismatch between an asset and the real object.

Geometry, masks, depth and cameras are never changed.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

from ..constants import ROLE_RENDER, ROLE_SCENE

DEGRADE_PROBABILITY = 0.25
JITTER_PROBABILITY = 0.8
EDGE_BAND = 2                 # pixels on each side of the mask boundary where the alpha is estimated
BACKGROUND_MARGIN = 3
MIN_CONTRAST = 0.04
JITTER = dict(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.7, 1.3), hue=(-0.04, 0.04),
              gamma=(0.85, 1.15), blur_p=0.25, blur_sigma=(0.4, 1.0), sharpen_p=0.2, sharpen_amount=(0.3, 0.8))
_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)
_RGB = np.linalg.inv(_YIQ).astype(np.float32)


def degrade_source(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """uint8 (H, W, 3) image, with probability 0.25 slightly blurred or JPEG-compressed."""
    if rng.random() >= DEGRADE_PROBABILITY:
        return rgb
    image = Image.fromarray(rgb)
    if rng.random() < 0.5:
        return np.asarray(image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.4, 1.0)))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(rng.integers(92, 99)))
    return np.asarray(Image.open(buffer).convert("RGB"))


def scene_exposure(x: dict, rng: np.random.Generator) -> None:
    """In place: with probability 0.5 per scene view, a linear RGB gain on the view and its crops (no clipping)."""
    common = x["role"] != ROLE_RENDER
    for j in np.flatnonzero(x["role"] == ROLE_SCENE):
        if rng.random() < 0.5:
            gain = rng.uniform(0.9, 1.1, 3)[:, None, None] * rng.uniform(0.85, 1.15)
            rows = (x["parent"] == j) & common
            ceiling = x["images"][rows].max(axis=(0, 2, 3))[:, None, None]
            x["images"][rows] *= np.minimum(gain, 1 / np.maximum(ceiling, 1e-6))


def render_color_gain(x: dict, rng: np.random.Generator) -> None:
    """In place: with probability 0.5 per object, one RGB gain on all of its renders."""
    render = x["role"] == ROLE_RENDER
    for index in range(len(x["owner_slots"])):
        if rng.random() < 0.5:
            gain = rng.uniform(0.9, 1.1, 3)[:, None, None]
            rows = (x["owner"] == index) & render
            x["images"][rows] = np.clip(x["images"][rows] * gain, 0, 1)


def _morph(mask: np.ndarray, radius: int, grow: bool) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    size = 2 * radius + 1
    return np.asarray(image.filter(ImageFilter.MaxFilter(size) if grow else ImageFilter.MinFilter(size))) > 0


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    return _morph(np.pad(mask, radius), radius, False)[radius:-radius, radius:-radius]


def _box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over a k x k window (zero padded), same shape as ``a``."""
    r = k // 2
    c = np.pad(a.astype(np.float64), ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    h, w = a.shape
    return ((c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]) / (k * k)).astype(np.float32)


def _gauss(channel: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(np.ceil(3 * sigma)))
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    g /= g.sum()
    p = np.pad(channel, r, mode="edge")
    rows = sum(g[i] * p[:, i:i + channel.shape[1]] for i in range(2 * r + 1))
    return sum(g[i] * rows[i:i + channel.shape[0], :] for i in range(2 * r + 1)).astype(np.float32)


def edge_alpha(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(h, w) object opacity of a render on a uniform background: the mask, refined in a band around its
    boundary from ``I = a F + (1 - a) B`` (F = local mean object colour, B = median background colour)."""
    alpha = mask.astype(np.float32)
    outside = ~_morph(mask, BACKGROUND_MARGIN, True)
    band = _morph(mask, EDGE_BAND, True) & ~_erode(mask, EDGE_BAND)
    if outside.sum() < 64 or not band.any():
        return alpha
    background = np.median(image[:, outside], axis=1).astype(np.float32)[:, None, None]
    inner = _erode(mask, EDGE_BAND).astype(np.float32)
    if inner.sum() == 0:
        inner = mask.astype(np.float32)
    k = 4 * EDGE_BAND + 3
    weight = _box_mean(inner, k)
    foreground = np.stack([_box_mean(c * inner, k) for c in image]) / np.maximum(weight, 1e-6)
    d_image, d_fore = image - background, foreground - background
    norm = (d_fore ** 2).sum(0)
    estimate = np.clip((d_image * d_fore).sum(0) / np.maximum(norm, 1e-12), 0, 1)
    ok = band & (weight > 1e-3) & (np.sqrt(norm) >= MIN_CONTRAST)
    alpha[ok] = estimate[ok]
    return alpha


def jitter(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Appearance-jittered copy of a (3, h, w) image in [0, 1] (statistics from the ``mask`` pixels)."""
    b, c, s, h, g = (float(rng.uniform(*JITTER[key])) for key in ("brightness", "contrast", "saturation", "hue",
                                                                   "gamma"))
    mode, r = rng.random(), rng.random()
    out = image * b
    mean = out[:, mask].mean(axis=1)[:, None, None] if mask.any() else out.mean(axis=(1, 2), keepdims=True)
    out = np.clip((out - mean) * c + mean, 0, 1).astype(np.float32)
    angle = 2 * np.pi * h
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]], np.float32)
    out = np.einsum("ij,jhw->ihw", _RGB @ np.diag(np.array([1, s, s], np.float32)) @ rotation @ _YIQ, out)
    out = np.clip(out, 0, 1) ** g
    if mode < JITTER["blur_p"]:
        low, high = JITTER["blur_sigma"]
        out = np.stack([_gauss(ch, low + r * (high - low)) for ch in out])
    elif mode < JITTER["blur_p"] + JITTER["sharpen_p"]:
        low, high = JITTER["sharpen_amount"]
        out = np.stack([ch + (low + r * (high - low)) * (ch - _gauss(ch, 1.0)) for ch in out])
    return np.clip(out, 0, 1).astype(np.float32)


def render_jitter(x: dict, rng: np.random.Generator) -> None:
    """In place: with probability 0.8 per render, ``jitter`` on its object pixels (alpha-blended at the edge)."""
    for row in np.flatnonzero(x["role"] == ROLE_RENDER):
        if rng.random() >= JITTER_PROBABILITY:
            continue
        h, w = (int(v) for v in x["view_size"][row])
        image = x["images"][row, :, :h, :w]
        mask = x["instance_masks"][row, int(x["owner"][row]), :h, :w] > 0
        alpha = edge_alpha(image, mask)[None]
        x["images"][row, :, :h, :w] = np.clip(alpha * jitter(image, mask, rng) + (1 - alpha) * image, 0, 1)
