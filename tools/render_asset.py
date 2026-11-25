"""Render the 32 views of an asset with Blender (Cycles) and write an asset directory (format: docs/DATA.md).

    python tools/render_asset.py chair.glb --output data/assets/chair [--asset-id chair] [--device CPU]

Requires the ``bpy`` Python module (Blender 4.2 - 4.5) in the environment that has this package installed.
Input: a ``.glb`` / ``.gltf`` / ``.obj`` mesh with +Y up (the convention of these formats). The mesh is moved to
the asset frame (bounding box centred, longest side 1) and rendered from the viewpoints of
``x3dcomposer.geometry.asset_views`` with the recipe of the training data:

* 608 x 608 pixels, 50 degree field of view, the object fills 0.88 of the image;
* uniform white environment of strength 0.35 plus a 10 W disk light (size 2.5) just above the camera, exposure
  -1 stop, "Standard" view transform; the background comes out as uniform gray (sRGB 116);
* per view: RGB (JPEG), foreground mask and depth (uint16, asset units * 10000).

It also writes ``asset.json``, ``cameras.npz``, ``points.npy`` (512 surface samples) and a copy of the mesh.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from PIL import Image

from x3dcomposer.geometry.asset_views import FILL, FOV_DEG, SIZE, fit_view, sample_surface, view_directions
from x3dcomposer.geometry.camera import canonical_frame

DEPTH_SCALE = 1e4
WORLD_STRENGTH, LIGHT_WATTS, LIGHT_SIZE, EXPOSURE = 0.35, 10.0, 2.5, -1.0
OPENCV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])            # camera axes: OpenCV (y down, z forward) -> Blender
# The glTF / OBJ importers turn the file's +Y up into Blender's +Z up: (x, y, z)_file -> (x, -z, y)_blender.
FILE_FROM_BLENDER = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def import_mesh(path: Path) -> list:
    """Import the mesh file; returns its mesh objects (in Blender's converted axes)."""
    if path.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError("Supported mesh formats: .glb, .gltf, .obj")
    objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objects:
        raise ValueError(f"No mesh in {path}")
    return objects


def mesh_arrays(objects: list, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (after each object's world matrix and ``transform``) and triangles of all mesh objects."""
    vertices, triangles, offset = [], [], 0
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        v = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", v)
        t = np.empty(len(mesh.loop_triangles) * 3, np.int64)
        mesh.loop_triangles.foreach_get("vertices", t)
        world = transform @ np.array(obj.matrix_world)
        vertices.append(v.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3])
        triangles.append(t.reshape(-1, 3) + offset)
        offset += len(mesh.vertices)
    return np.concatenate(vertices), np.concatenate(triangles)


def setup_scene(device: str, samples: int, work: Path) -> dict:
    """Empty Cycles scene with the lighting above, a camera, a camera-attached light and depth / index outputs."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU" if device == "CPU" else "GPU"
    if device != "CPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = device
        preferences.get_devices()
        for item in preferences.devices:
            item.use = item.type == device
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = EXPOSURE
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs[1].default_value = WORLD_STRENGTH
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.clip_start, camera.data.clip_end = 0.01, 100.0
    camera.data.sensor_fit, camera.data.sensor_width = "HORIZONTAL", 36.0
    scene.collection.objects.link(camera)
    scene.camera = camera
    light_data = bpy.data.lights.new("fill", "AREA")
    light_data.shape, light_data.energy, light_data.size = "DISK", LIGHT_WATTS, LIGHT_SIZE
    light = bpy.data.objects.new("fill", light_data)
    light.visible_camera = False
    scene.collection.objects.link(light)
    layer = scene.view_layers[0]
    layer.use_pass_z = True
    layer.use_pass_object_index = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    tree.links.new(layers.outputs["Image"], tree.nodes.new("CompositorNodeComposite").inputs[0])
    for socket, name in (("Depth", "depth"), ("IndexOB", "index")):
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.format.file_format = "OPEN_EXR"
        node.format.color_depth = "32"
        node.format.color_mode = "RGB"
        node.base_path = str(work)
        node.file_slots[0].path = name + "_"
        tree.links.new(layers.outputs[socket], node.inputs[0])
    return {"scene": scene, "camera": camera, "light": light}


def read_exr(path: Path) -> np.ndarray:
    """First channel of an EXR as (SIZE, SIZE) float32 with row 0 at the top."""
    image = bpy.data.images.load(str(path))
    pixels = np.empty(SIZE * SIZE * image.channels, np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(SIZE, SIZE, -1)[::-1, :, 0].copy()


def render_view(setup: dict, index: int, c2w: np.ndarray, K: np.ndarray, work: Path) -> tuple[np.ndarray, ...]:
    """RGB uint8 (SIZE, SIZE, 3), foreground mask and depth (asset units) of one view."""
    scene, camera, light = setup["scene"], setup["camera"], setup["light"]
    camera.data.lens = float(K[0, 0] * 36.0 / SIZE)
    camera.matrix_world = Matrix((c2w @ OPENCV_TO_BLENDER).tolist())
    above = c2w.copy()
    above[:3, 3] -= 0.25 * c2w[:3, 1]                           # the light sits 0.25 above the camera
    light.matrix_world = Matrix((above @ OPENCV_TO_BLENDER).tolist())
    scene.frame_set(index + 1)
    scene.render.filepath = str(work / f"rgb_{index:02d}.png")
    bpy.ops.render.render(write_still=True)
    rgb = np.asarray(Image.open(scene.render.filepath).convert("RGB"))
    mask = read_exr(work / f"index_{index + 1:04d}.exr") > 0.5
    depth = np.where(mask, read_exr(work / f"depth_{index + 1:04d}.exr"), 0.0)
    return rgb, mask, depth


def render_asset(mesh_path: Path, output: Path, asset_id: str, device: str, samples: int) -> None:
    """Render all views of ``mesh_path`` and write the asset directory ``output``."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objects = import_mesh(mesh_path)
    file_vertices, _ = mesh_arrays(objects, FILE_FROM_BLENDER)
    N = canonical_frame(file_vertices)                          # mesh (file) coordinates -> asset frame
    placed = [N @ FILE_FROM_BLENDER @ np.array(obj.matrix_world) for obj in objects]
    for obj, matrix in zip(objects, placed):
        obj.parent = None
        obj.matrix_world = Matrix(matrix.tolist())
        obj.pass_index = 1
    bpy.context.view_layer.update()
    vertices, triangles = mesh_arrays(objects, np.eye(4))       # now in the asset frame
    views = view_directions(asset_id)
    for name in ("rgb", "mask", "depth"):
        (output / name).mkdir(parents=True, exist_ok=True)
    intrinsics, cameras = [], []
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        setup = setup_scene(device, samples, work)
        for index, view in enumerate(views):
            c2w, K = fit_view(vertices, view["direction"])
            rgb, mask, depth = render_view(setup, index, c2w, K, work)
            Image.fromarray(rgb).save(output / "rgb" / f"{index:02d}.jpg", quality=93, subsampling=0)
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / "mask" / f"{index:02d}.png")
            quantized = np.clip(np.rint(depth * DEPTH_SCALE), 0, 65535).astype(np.uint16)
            Image.fromarray(quantized).save(output / "depth" / f"{index:02d}.png")
            intrinsics.append(K)
            cameras.append(c2w)
    np.savez(output / "cameras.npz", K=np.stack(intrinsics), c2w=np.stack(cameras))
    np.save(output / "points.npy", sample_surface(vertices, triangles, 512).astype(np.float32))
    shutil.copyfile(mesh_path, output / ("mesh" + mesh_path.suffix.lower()))
    meta = {"asset_id": asset_id, "N_asset_from_mesh": N.tolist(), "mesh": "mesh" + mesh_path.suffix.lower(),
            "size": SIZE, "fov_deg": FOV_DEG, "fill": FILL, "depth_scale": DEPTH_SCALE, "symmetry": "none",
            "views": views}
    (output / "asset.json").write_text(json.dumps(meta, indent=1) + "\n")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="asset directory to create")
    parser.add_argument("--asset-id", help="identifier stored in asset.json (default: the output directory's name)")
    parser.add_argument("--device", default="CPU", choices=["CPU", "CUDA", "OPTIX"])
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel")
    args = parser.parse_args()
    render_asset(args.mesh, args.output, args.asset_id or args.output.name, args.device, args.samples)


if __name__ == "__main__":
    main()
