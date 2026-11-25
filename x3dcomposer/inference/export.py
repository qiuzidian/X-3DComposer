"""Writers for a predicted layout: ``layout.json`` and a GLB of the placed meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pipeline import Layout

# glTF cameras look along -Z with +Y up; the scene frame is the first camera in OpenCV axes (+Z forward, +Y down).
OPENCV_TO_GLTF = np.diag([1.0, -1.0, -1.0, 1.0])


def layout_dict(layout: Layout) -> dict:
    """JSON-serializable form of a layout (matrices as nested lists, row-major)."""
    objects = [{"asset_id": o["asset_id"], "rotation": o["R"].tolist(), "translation": o["t"].tolist(),
                "scale": o["scale"].tolist(), "pose_asset_to_scene": o["pose"].tolist(),
                "pose_mesh_to_scene": o["mesh_pose"].tolist(), "rotation_spread_deg": o["rotation_spread_deg"]}
               for o in layout.objects]
    return {"frame": "first scene camera, OpenCV axes (x right, y down, z forward)", "unit": layout.unit,
            "condition_mode": layout.mode, "objects": objects, "camera_to_scene": layout.camera_c2w.tolist(),
            "camera_intrinsics": layout.camera_K.tolist()}


def save_layout(layout: Layout, path: str | Path) -> None:
    """Write ``layout_dict`` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout_dict(layout), indent=2) + "\n")


def export_glb(layout: Layout, path: str | Path, gltf_axes: bool = True) -> None:
    """Write the assets' meshes, placed by the predicted poses, as one GLB (requires ``trimesh``).

    Every asset must record its mesh (``asset.json`` ``mesh``). With ``gltf_axes`` the scene is rotated from
    OpenCV to glTF camera axes, so viewers show it upright as seen from the first camera.
    """
    import trimesh

    scene = trimesh.Scene()
    world = OPENCV_TO_GLTF if gltf_axes else np.eye(4)
    for index, (item, asset) in enumerate(zip(layout.objects, layout.assets)):
        if asset.mesh_path is None or not asset.mesh_path.is_file():
            raise FileNotFoundError(f"Asset {item['asset_id']} has no mesh file to export")
        loaded = trimesh.load(asset.mesh_path, force="scene")
        loaded.apply_transform(world @ item["mesh_pose"])
        for part, geometry in enumerate(loaded.dump(concatenate=False)):
            scene.add_geometry(geometry, node_name=f"object{index:02d}_{part}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(path))
