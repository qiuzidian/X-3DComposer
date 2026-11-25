"""Training sample selection: which scene, views, objects, crops and renders make up a sample (NumPy only).

Per optimizer step one *layout* is drawn: the number of scene views S and objects N from ``LAYOUTS``, the number
of renders per object M from ``RENDER_COUNTS`` and the aspect ratio of the scene views. The step then holds as
many samples of that layout as fit the token budget. Selection reads scene metadata and the visibility table
only; pixels are decoded later by ``x3dcomposer.data.assemble``.

Rules (thresholds in pixels of the stored images, which should have about twice the resolution of the 416^2
scene views):

* An object can be a target in a view if its visible mask has at least 64 pixels and 0.25 % of the image area,
  its mask box is at most 0.8 of the shorter image side, at most 55 % of it is occluded and at most 20 % of its
  silhouette is outside the image (the last two only where the table knows them), and the mask box stays 8
  pixels inside the scene window.
* Any two scene views of a sample are at least ``max(0.6 m, 0.3 * mean median object depth)`` apart, except
  pairs less than 0.2 m apart whose optical axes differ by at least 30 degrees (co-centric rigs).
* Views are chosen greedily: until N objects are covered, views that add objects are preferred; afterwards
  views that show the covered objects again. Objects are drawn with probability proportional to the number of
  chosen views that show them.
* Crops: ``views.crop_candidates`` per object over the chosen views; the crop box has side ``k * L``
  (k ~ U(1.15, 1.6), L = long side of the mask box) and a random offset that keeps the mask inside.
* Renders: M of the asset's views, without replacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..geometry.camera import ASPECTS, scene_shape, scene_window
from .scene import Scene
from .views import MAX_CROPS_PER_OBJECT, crop_candidates

# (weight, ((S, N), ...)): tiers of layouts; weights are shared equally inside a tier and summed per (S, N).
_TIERS = (
    (0.09, ((1, 1), (1, 2), (1, 1))),
    (0.26, ((2, 2), (2, 3), (2, 1), (4, 2), (4, 1), (2, 2))),
    (0.31, ((4, 3), (4, 4), (4, 6), (4, 3), (6, 3), (8, 4), (6, 6), (8, 6))),
    (0.34, ((8, 4), (8, 6), (8, 8), (8, 12), (8, 6), (4, 8), (8, 10), (10, 6), (10, 8))),
)
RENDER_COUNTS = (1, 2, 3, 4, 6, 8)
SCENE_TOKENS = 693            # 26 x 26 patches + camera + 16 register tokens (upper bound over aspects)
OBJECT_TOKENS = 458           # a 336-pixel crop or render
MIN_PIXELS, MIN_AREA, MAX_BOX = 64, 0.0025, 0.8
MIN_VISIBLE_FRACTION, MIN_IN_FRAME_FRACTION = 0.45, 0.8
WINDOW_MARGIN = 8
BASELINE_FLOOR_M, BASELINE_DEPTH_FRACTION = 0.6, 0.3
RIG_BASELINE_M, RIG_ANGLE_DEG = 0.2, 30.0
CROP_CONTEXT = (1.15, 1.6)
CROP_MARGIN = 0.02
VIEW_ATTEMPTS, SCENE_ATTEMPTS, LAYOUT_ATTEMPTS = 8, 64, 32


def layouts() -> dict[tuple[int, int], float]:
    """(S, N) -> draw weight."""
    merged: dict[tuple[int, int], float] = {}
    for weight, entries in _TIERS:
        for entry in entries:
            merged[entry] = merged.get(entry, 0.0) + weight / len(entries)
    return merged


def layout_tokens(S: int, N: int, M: int) -> int:
    """Worst-case tokens of a sample: every object with ``min(S, 3)`` crops and M renders at 336 pixels."""
    return S * SCENE_TOKENS + N * (min(S, MAX_CROPS_PER_OBJECT) + M) * OBJECT_TOKENS


@dataclass
class SceneSummary:
    """What selection needs from one scene (no pixels)."""

    path: str
    source: str
    image_hw: tuple[int, int]
    num_views: int
    visible: list[list[frozenset]]            # [aspect][view] -> instances that can be targets
    box: dict[tuple[int, int], tuple]         # (view, instance) -> mask box
    pixels: dict[tuple[int, int], int]        # (view, instance) -> mask pixels
    asset: dict[int, str]                     # instance -> asset id
    ok_pair: np.ndarray                       # (V, V) view pairs that may share a sample
    max_views: int
    objects: list[frozenset]                  # [aspect] -> union of ``visible``


def _inside_window(boxes: np.ndarray, window: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Boxes at least ``WINDOW_MARGIN`` inside every window side that lies inside the image."""
    ok = np.ones(len(boxes), bool)
    H, W = image_hw
    for axis, extent in ((0, W), (1, H)):
        low, high = window[axis], window[axis + 2]
        if low > 0:
            ok &= boxes[:, axis] >= low + WINDOW_MARGIN
        if high < extent:
            ok &= boxes[:, axis + 2] <= high - WINDOW_MARGIN
    return ok


def _largest_view_set(ok_pair: np.ndarray, starts: int = 8) -> int:
    """Size of the largest of a few greedy sets of mutually compatible views (a cheap lower bound)."""
    best, count = 0, len(ok_pair)
    rng = np.random.default_rng(0)
    for _ in range(min(starts, count)):
        allowed, size = np.ones(count, bool), 0
        for v in rng.permutation(count):
            if allowed[v]:
                size += 1
                allowed &= ok_pair[v]
        best = max(best, size)
    return best


def summarize(scene: Scene, assets_root: str | Path) -> SceneSummary:
    """Read one scene's metadata into a ``SceneSummary``."""
    rows = scene.visibility()
    H, W = image_hw = scene.image_hw()
    box = rows["box"].astype(np.int64)
    side = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    with np.errstate(invalid="ignore"):
        keep = ((rows["pixels"] >= MIN_PIXELS) & (rows["pixels"] >= MIN_AREA * H * W) & (side <= MAX_BOX * min(H, W))
                & ~(rows["visible_fraction"] < MIN_VISIBLE_FRACTION)
                & ~(rows["in_frame_fraction"] < MIN_IN_FRAME_FRACTION))
    asset = {}
    for instance in np.unique(rows["instance"][keep]).tolist():
        item = scene.instances.get(int(instance))
        rendered = item is not None and (Path(assets_root) / str(item["asset_id"]) / "asset.json").is_file()
        if rendered and item.get("trainable", True):
            asset[int(instance)] = str(item["asset_id"])
    keep &= np.isin(rows["instance"], list(asset))
    views, instances, boxes = rows["view"][keep].astype(int), rows["instance"][keep].astype(int), box[keep]
    V = scene.num_views
    visible = []
    for aspect in ASPECTS:
        fits = np.ones(len(views), bool)
        for v in np.unique(views):
            here = views == v
            fits[here] = _inside_window(boxes[here].astype(np.float64),
                                        scene_window(scene.K[v], image_hw, scene_shape(aspect)), image_hw)
        per_view = [set() for _ in range(V)]
        for v, i in zip(views[fits], instances[fits]):
            per_view[v].add(int(i))
        visible.append([frozenset(s) for s in per_view])
    with np.errstate(all="ignore"):
        depth = np.array([np.nanmedian(rows["depth_median"][rows["view"] == v]) if (rows["view"] == v).any()
                          else np.nan for v in range(V)])
    depth = np.where(np.isfinite(depth), depth, np.nanmedian(depth) if np.isfinite(depth).any() else 2.0)
    centres, axes = scene.c2w[:, :3, 3], scene.c2w[:, :3, 2]
    baseline = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    angle = np.degrees(np.arccos(np.clip(axes @ axes.T, -1, 1)))
    floor = np.maximum(BASELINE_FLOOR_M, BASELINE_DEPTH_FRACTION * (depth[:, None] + depth[None]) / 2)
    ok_pair = (baseline >= floor) | ((baseline < RIG_BASELINE_M) & (angle >= RIG_ANGLE_DEG))
    np.fill_diagonal(ok_pair, False)
    every = np.flatnonzero(np.isin(rows["instance"], list(asset)))
    return SceneSummary(
        str(scene.path), str(scene.meta.get("source", "default")), image_hw, V, visible,
        {(int(rows["view"][r]), int(rows["instance"][r])): tuple(int(b) for b in box[r]) for r in every},
        {(int(rows["view"][r]), int(rows["instance"][r])): int(rows["pixels"][r]) for r in every},
        asset, ok_pair, _largest_view_set(ok_pair) if V > 1 else V, [frozenset().union(*p) for p in visible])


@dataclass(frozen=True)
class Selection:
    """One sample's choice of scene, views, objects, crops and renders."""

    scene_path: str
    views: tuple                 # scene view indices; views[0] defines the scene frame
    objects: tuple               # instance ids; object index = position
    assets: tuple                # asset id per object
    crops: tuple                 # (object index, position in ``views``, box [x0, y0, x1, y1))
    renders: tuple               # (object index, asset view index)
    scene_hw: tuple              # (h, w) of the scene views
    seed: int                    # seed of the sample's condition and augmentation draws


def choose_views(s: SceneSummary, S: int, N: int, aspect: int, rng: np.random.Generator):
    """S mutually compatible views that show at least N target objects: ``(views, covered objects)`` or None."""
    visible = s.visible[aspect]
    starts = [v for v in range(s.num_views) if len(visible[v]) >= (N if S == 1 else 1)]
    if not starts:
        return None
    for _ in range(VIEW_ATTEMPTS):
        first = int(starts[rng.integers(len(starts))])
        chosen, covered, allowed = [first], set(visible[first]), s.ok_pair[first].copy()
        while len(chosen) < S:
            candidates = np.flatnonzero(allowed)
            if not len(candidates):
                break
            if len(covered) < N:
                weight = 1 + 4 * np.array([len(visible[v] - covered) for v in candidates], np.float64)
            else:
                weight = 1 + 4 * np.array([len(visible[v] & covered) for v in candidates], np.float64)
            v = int(candidates[rng.choice(len(candidates), p=weight / weight.sum())])
            chosen.append(v)
            covered |= visible[v]
            allowed &= s.ok_pair[v]
        if len(chosen) == S and len(covered) >= N:
            return chosen, covered
    return None


def train_crop_box(box: Sequence[int], rng: np.random.Generator) -> list[int]:
    """Square training crop: side ``ceil(k * L)`` with k ~ U(1.15, 1.6), random offset, mask inside with a 2 %
    margin. The box is not clipped to the image."""
    x0, y0, x1, y1 = (int(b) for b in box)
    long_side = max(x1 - x0, y1 - y0)
    side = int(math.ceil(rng.uniform(*CROP_CONTEXT) * long_side))
    start = []
    for low_b, high_b in ((x0, x1), (y0, y1)):
        centre = (low_b + high_b - side) / 2 + rng.uniform(-0.5, 0.5) * (side - long_side)
        pad = CROP_MARGIN * side
        low, high = math.ceil(high_b + pad - side), math.floor(low_b - pad)
        if low > high:
            low, high = high_b - side, low_b
        start.append(int(np.clip(round(centre), low, high)))
    return [start[0], start[1], start[0] + side, start[1] + side]


def _draw_objects(s: SceneSummary, views: list, pool: list, N: int, aspect: int, same_asset: bool,
                  rng: np.random.Generator) -> list[int] | None:
    """N objects of ``pool``, weighted by the number of chosen views that show them; with ``same_asset`` the
    first two share an asset (None if the pool has no such pair)."""
    shown = np.array([sum(p in s.visible[aspect][v] for v in views) for p in pool], np.float64)
    forced: list[int] = []
    if same_asset:
        pairs = [(a, b) for i, a in enumerate(pool) for b in pool[i + 1:] if s.asset[a] == s.asset[b]]
        if not pairs or N < 2:
            return None
        forced = list(pairs[int(rng.integers(len(pairs)))])
    rest = [p for p in pool if p not in forced]
    weight = np.array([shown[pool.index(p)] for p in rest], np.float64)
    if N == len(forced):
        return forced
    extra = rng.choice(rest, N - len(forced), replace=False, p=weight / weight.sum())
    return forced + [int(p) for p in extra]


def select(catalog: Sequence[SceneSummary], candidates: Sequence[int], S: int, N: int, M: int, aspect: int,
           asset_views: int, same_asset: bool, rng: np.random.Generator) -> Selection | None:
    """Draw a scene among ``candidates`` and choose its views, objects, crops and renders (None if none fits)."""
    for _ in range(SCENE_ATTEMPTS):
        s = catalog[candidates[int(rng.integers(len(candidates)))]]
        picked = choose_views(s, S, N, aspect, rng)
        if picked is None:
            continue
        views, covered = picked
        views = [views[i] for i in rng.permutation(len(views))]
        objects = _draw_objects(s, views, sorted(covered), N, aspect, same_asset, rng)
        if objects is None:
            continue
        crops = []
        for index, instance in enumerate(objects):
            boxes = {j: s.box[v, instance] for j, v in enumerate(views) if (v, instance) in s.box}
            pixels = {j: s.pixels[views[j], instance] for j in boxes}
            chosen = crop_candidates(boxes, pixels, s.image_hw)
            if not chosen:       # no view passes the crop rule: the views in which the object is a target
                chosen = [j for j, v in enumerate(views) if instance in s.visible[aspect][v]][:MAX_CROPS_PER_OBJECT]
            crops += [(index, j, tuple(train_crop_box(boxes[j], rng))) for j in chosen]
        renders = [(index, int(k)) for index in range(N) for k in rng.choice(asset_views, M, replace=False)]
        return Selection(s.path, tuple(int(v) for v in views), tuple(objects), tuple(s.asset[p] for p in objects),
                         tuple(crops), tuple(renders), scene_shape(ASPECTS[aspect]),
                         int(rng.integers(2 ** 31)))
    return None


class StepSampler:
    """Deterministic per-step sample plans: ``plan(step, rank)`` -> the selections of that step on that rank."""

    def __init__(self, catalog: Sequence[SceneSummary], seed: int, token_budget: int,
                 source_weights: dict[str, float] | None = None, same_asset_probability: float = 0.0,
                 asset_views: int = 32):
        self.catalog, self.seed, self.asset_views = list(catalog), seed, asset_views
        self.token_budget, self.same_asset_probability = token_budget, same_asset_probability
        self.source_weights = dict(source_weights or {})
        self.layouts = {}
        for (S, N), weight in layouts().items():
            if any(min(s.max_views, s.num_views) >= S and len(s.objects[a]) >= N
                   for s in self.catalog for a in range(len(ASPECTS))):
                self.layouts[S, N] = weight
        if not self.layouts:
            raise ValueError("No sample layout is feasible on this scene list")

    def _candidates(self, S: int, N: int, aspect: int, rng: np.random.Generator) -> list[int]:
        """Scenes that can hold the layout, restricted to one source drawn by ``source_weights`` if given."""
        fits = [i for i, s in enumerate(self.catalog)
                if min(s.max_views, s.num_views) >= S and len(s.objects[aspect]) >= N]
        if not self.source_weights or not fits:
            return fits
        sources = sorted({self.catalog[i].source for i in fits})
        weight = np.array([self.source_weights.get(name, 0.0) for name in sources], np.float64)
        if weight.sum() <= 0:
            return fits
        source = sources[int(rng.choice(len(sources), p=weight / weight.sum()))]
        return [i for i in fits if self.catalog[i].source == source]

    def plan(self, step: int, rank: int = 0, attempt: int = 0) -> list[Selection]:
        """The samples of ``step`` on ``rank`` (at least one; all share S, N, M and the scene raster).

        The layout depends on (seed, step) only, so every rank of a distributed run draws the same one; the
        samples depend on the rank too. ``attempt`` > 0 gives an independent redraw of the samples.
        """
        layout_rng = np.random.default_rng([self.seed, step])
        rng = np.random.default_rng([self.seed, step, rank, attempt])
        keys = list(self.layouts)
        weights = np.array([self.layouts[k] for k in keys])
        for _ in range(LAYOUT_ATTEMPTS):
            S, N = keys[int(layout_rng.choice(len(keys), p=weights / weights.sum()))]
            M = int(layout_rng.choice(RENDER_COUNTS))
            aspect = int(layout_rng.integers(len(ASPECTS)))
            selections = []
            for _ in range(max(1, self.token_budget // layout_tokens(S, N, M))):
                candidates = self._candidates(S, N, aspect, rng)
                if not candidates:
                    break
                same = N >= 2 and rng.random() < self.same_asset_probability
                picked = select(self.catalog, candidates, S, N, M, aspect, self.asset_views, same, rng)
                if picked is None and same:
                    picked = select(self.catalog, candidates, S, N, M, aspect, self.asset_views, False, rng)
                if picked is not None:
                    selections.append(picked)
            if selections:
                return selections
        raise RuntimeError(f"No sample could be drawn at step {step}; check the scene list and visibility tables")

