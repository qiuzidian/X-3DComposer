"""Check scene and asset directories against the format of docs/DATA.md (NumPy only, no GPU).

    python tools/validate_dataset.py --scenes data/scenes_train.txt --assets data/assets [--samples 4]

Per scene: files and array shapes, valid camera poses, labels and assets of the listed instances, and the
agreement of labels and masks: the asset's surface points, placed by the label and projected into a view, must
fall on the instance's mask. Per asset: files, camera poses, and mask / depth consistency. With ``--samples N``
it also assembles N training samples, which exercises the whole data path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from x3dcomposer.data.assemble import assemble
from x3dcomposer.data.assets import Asset
from x3dcomposer.data.sampling import StepSampler, summarize
from x3dcomposer.data.scene import Scene, read_scene_list
from x3dcomposer.geometry.camera import project, transform

MIN_MASK_AGREEMENT = 0.5      # fraction of the projected, in-image asset points that must land on the mask


def is_rigid(c2w: np.ndarray) -> bool:
    """True if every 4x4 matrix is a rotation plus a translation."""
    R = c2w[..., :3, :3]
    return bool(np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-4)
                and np.allclose(np.linalg.det(R), 1, atol=1e-4)
                and np.allclose(c2w[..., 3, :], [0, 0, 0, 1], atol=1e-6))


def check_asset(path: Path) -> list[str]:
    """Problems of one asset directory (empty list = fine)."""
    problems = []
    try:
        asset = Asset(path)
    except (OSError, KeyError, ValueError) as error:
        return [f"cannot be read: {error}"]
    if asset.K.shape[1:] != (3, 3) or asset.c2w.shape != (asset.num_views, 4, 4) or not is_rigid(asset.c2w):
        problems.append("cameras.npz needs K (V, 3, 3) and rigid c2w (V, 4, 4)")
    if asset.N.shape != (4, 4):
        problems.append("asset.json N_asset_from_mesh must be 4x4")
    for index in (0, asset.num_views - 1):
        try:
            view = asset.view(index)
        except OSError as error:
            problems.append(f"view {index}: {error}")
            continue
        if view["rgb"].shape[:2] != view["mask"].shape or view["mask"].shape != view["depth"].shape:
            problems.append(f"view {index}: rgb, mask and depth differ in size")
        elif not view["mask"].any() or (view["depth"][view["mask"]] <= 0).mean() > 0.05:
            problems.append(f"view {index}: empty mask or missing depth on the object")
    points = asset.points()
    if points is not None and (np.abs(points).max() > 0.51 or points.shape[1] != 3):
        problems.append("points.npy must hold (P, 3) points inside the unit asset frame")
    return problems


def check_scene(path: str, assets_root: Path, known: dict[str, list[str]]) -> list[str]:
    """Problems of one scene directory; ``known`` caches the asset checks."""
    problems = []
    try:
        scene = Scene(path)
        hw = scene.image_hw()
    except (OSError, KeyError, ValueError) as error:
        return [f"cannot be read: {error}"]
    views = scene.num_views
    if scene.K.shape != (views, 3, 3) or scene.c2w.shape != (views, 4, 4) or not is_rigid(scene.c2w):
        problems.append("cameras.npz needs K (V, 3, 3) and rigid c2w (V, 4, 4)")
    if np.abs(scene.K[:, 0, 1]).max() > 1e-6:
        problems.append("intrinsics must have zero skew")
    for instance, item in scene.instances.items():
        missing = [key for key in ("asset_id", "R", "t", "scale") if key not in item]
        if missing:
            problems.append(f"instance {instance}: missing {missing}")
            continue
        asset_id = str(item["asset_id"])
        if asset_id not in known:
            known[asset_id] = check_asset(assets_root / asset_id)
        problems += [f"asset {asset_id}: {p}" for p in known[asset_id]]
        if min(item["scale"]) <= 0 or not is_rigid(scene.label(instance)[0]):
            problems.append(f"instance {instance}: scale must be positive and R a rotation")
    table = scene.visibility()
    for view in sorted({0, scene.num_views - 1}):
        try:
            data = scene.view(view)
        except OSError as error:
            problems.append(f"view {view}: {error}")
            continue
        if data["rgb"].shape[:2] != hw or data["ids"].shape != hw or data["depth"].shape != hw:
            problems.append(f"view {view}: rgb, ids and depth differ in size")
            continue
        for row in np.flatnonzero((table["view"] == view) & (table["pixels"] >= 1024)):
            instance = int(table["instance"][row])
            item = scene.instances.get(instance)
            if item is None or known.get(str(item.get("asset_id"))):
                continue
            points = Asset(assets_root / str(item["asset_id"])).points()
            if points is None:
                continue
            pose, scale = scene.label(instance)
            uv, z = project(transform(points * scale, pose), data["K"], data["c2w"])
            u, v = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
            inside = (z > 0) & (u >= 0) & (v >= 0) & (u < hw[1]) & (v < hw[0])
            # Occluded points fall on other objects; count only the points whose depth matches the depth map.
            surface = inside.copy()
            surface[inside] = np.abs(data["depth"][v[inside], u[inside]] - z[inside]) < 0.05 * np.maximum(z[inside], 1)
            if surface.sum() >= 20 and (data["ids"][v[surface], u[surface]] == instance).mean() < MIN_MASK_AGREEMENT:
                problems.append(f"view {view}, instance {instance}: the label does not project onto the mask "
                                "(check the pose / scale / camera conventions)")
    return problems


def main() -> None:
    """Check every listed scene and the assets it uses; exit with status 1 if any check fails."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", required=True, help="text file listing scene directories")
    parser.add_argument("--assets", required=True, type=Path, help="directory of asset directories")
    parser.add_argument("--samples", type=int, default=0, help="also assemble this many training samples")
    args = parser.parse_args()
    paths = read_scene_list(args.scenes)
    known: dict[str, list[str]] = {}
    failed = 0
    for path in paths:
        problems = check_scene(path, args.assets, known)
        failed += bool(problems)
        print(("FAIL " if problems else "ok   ") + path)
        for problem in problems:
            print("     " + problem)
    print(f"{len(paths) - failed} of {len(paths)} scenes passed; {len(known)} assets checked")
    if args.samples and not failed:
        sampler = StepSampler([summarize(Scene(p), args.assets) for p in paths], 0, 56_000)
        selections = [s for step in range(args.samples) for s in sampler.plan(step)][:args.samples]
        for selection in selections:
            x, t = assemble(selection, str(args.assets))
            print(f"sample: {Path(selection.scene_path).name}, views {len(selection.views)}, objects "
                  f"{len(selection.objects)}, model views {len(x['role'])}, canvas {x['images'].shape[-2:]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
