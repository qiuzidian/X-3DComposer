"""Command-line demo: predict the layout of assets in a scene and export it.

From images and id masks (mask pixel value i = the i-th asset, 0 = background):

    python demo.py --weights weights/ --backbone checkpoints/vggt_omega_1b_416_reproduce.pt \\
        --images view0.jpg view1.jpg --masks view0_ids.png view1_ids.png \\
        --assets assets/chair assets/table --output out/

Optional conditions: ``--cameras cameras.npz`` (``K`` (3, 3) or (S, 3, 3) and optionally ``c2w`` (S, 4, 4)) and
``--depths`` (``.npy`` float maps, or 16-bit PNGs scaled by ``--depth-unit``). ``--mode auto`` uses everything
that is given.

From a scene directory of the training format (docs/DATA.md), e.g. to try the model on held-out data:

    python demo.py --weights weights/ --backbone ... --scene data/scenes/room_0001 --views 0 4 \\
        --instances 3 7 12 --assets-root data/assets --output out/

Outputs: ``layout.json`` and, if every asset records its mesh, ``scene.glb``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from x3dcomposer import Asset, SceneObservation, X3DComposer, compose, export_glb, save_layout
from x3dcomposer.constants import CONDITION_MODES
from x3dcomposer.data.scene import Scene


def observation_from_files(args: argparse.Namespace) -> tuple[SceneObservation, list[Asset]]:
    """Scene and assets from image / mask files and optional camera and depth files."""
    images = [np.asarray(Image.open(p).convert("RGB")) for p in args.images]
    masks = [np.asarray(Image.open(p)).astype(np.int32) for p in args.masks]
    intrinsics = poses = depths = None
    if args.cameras:
        with np.load(args.cameras) as cameras:
            intrinsics = cameras["K"]
            poses = cameras["c2w"] if "c2w" in cameras else None
    if args.depths:
        depths = [np.load(p) if p.endswith(".npy") else np.asarray(Image.open(p)).astype(np.float32) * args.depth_unit
                  for p in args.depths]
    return SceneObservation(images, masks, intrinsics, poses, depths), [Asset(p) for p in args.assets]


def observation_from_scene(args: argparse.Namespace) -> tuple[SceneObservation, list[Asset]]:
    """Scene and assets from a scene directory of the training format (labels are not read)."""
    scene = Scene(args.scene)
    views = [scene.view(v) for v in args.views]
    lookup = np.zeros(max(int(v["ids"].max()) for v in views) + 1, np.int32)
    for index, instance in enumerate(args.instances):
        lookup[instance] = index + 1
    assets = [Asset(Path(args.assets_root) / str(scene.instances[i]["asset_id"])) for i in args.instances]
    observation = SceneObservation([v["rgb"] for v in views], [lookup[v["ids"]] for v in views],
                                   np.stack([v["K"] for v in views]), np.stack([v["c2w"] for v in views]),
                                   [v["depth"] for v in views])
    return observation, assets


def main() -> None:
    """Parse the command line, run the model once and write the layout."""
    parser = argparse.ArgumentParser(description="X-3DComposer demo", formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--weights", required=True, help="directory or Hugging Face repository of the released weights")
    parser.add_argument("--backbone", help="VGGT-Omega checkpoint (vggt_omega_1b_416_reproduce.pt)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--masks", nargs="+")
    parser.add_argument("--assets", nargs="+", help="asset directories, in the order of the mask ids")
    parser.add_argument("--cameras")
    parser.add_argument("--depths", nargs="+")
    parser.add_argument("--depth-unit", type=float, default=0.001, help="metres per unit of a PNG depth map")
    parser.add_argument("--scene", help="scene directory of the training format (instead of --images / --masks)")
    parser.add_argument("--views", nargs="+", type=int, default=[0])
    parser.add_argument("--instances", nargs="+", type=int)
    parser.add_argument("--assets-root")
    parser.add_argument("--mode", default="auto", choices=("auto",) + CONDITION_MODES)
    parser.add_argument("--renders", type=int, default=4, help="renders per object (1-8)")
    parser.add_argument("--estimated", action="store_true", help="the given K / depth are estimates, not measurements")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene, assets = observation_from_scene(args) if args.scene else observation_from_files(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = X3DComposer.from_pretrained(args.weights, args.backbone).to(device)
    layout = compose(model, scene, assets, mode=args.mode, renders_per_object=args.renders,
                     conditions_estimated=args.estimated, seed=args.seed)
    save_layout(layout, args.output / "layout.json")
    if all(a.mesh_path is not None and a.mesh_path.is_file() for a in assets):
        try:
            export_glb(layout, args.output / "scene.glb")
        except ImportError:
            print("scene.glb was not written: trimesh is missing (pip install -r requirements.txt)")
    for item in layout.objects:
        print(f"{item['asset_id']}: t = {np.round(item['t'], 3).tolist()}, "
              f"scale = {np.round(item['scale'], 3).tolist()}")
    print(f"mode {layout.mode}, unit {layout.unit}; wrote {args.output}")


if __name__ == "__main__":
    main()
