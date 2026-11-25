# Data format

Training reads two kinds of directories: **scenes** (images with per-pixel object ids, cameras, depth and 9-DoF
object labels) and **assets** (pre-rendered views of each 3D asset). Inference uses the same asset directories.
Converters from specific datasets are not part of this repository; any dataset that can be written in this
format can be used.

```
data/
  scenes_train.txt            # one scene directory per line (relative to this file)
  scenes/<scene_id>/
    scene.json
    cameras.npz
    rgb/0000.jpg  rgb/0001.jpg  ...
    ids/0000.png  ...
    depth/0000.png ...
    visibility.npz            # written by tools/build_visibility.py
  assets/<asset_id>/
    asset.json
    cameras.npz
    rgb/00.jpg ... rgb/31.jpg
    mask/00.png ...
    depth/00.png ...
    points.npy                # optional
    mesh.glb                  # optional
```

## Conventions

- **Cameras** follow OpenCV: x right, y down, z forward. Poses are camera-to-world 4x4 matrices (`c2w`).
  Intrinsics `K` are 3x3 with zero skew, in pixels of the stored image, with pixel centres at integer
  coordinates (the centre of the top-left pixel is (0, 0)).
- **World frame** of a scene: any rigid frame in **metres**. The view-selection rules and the floor loss use
  metric lengths.
- **Depth** is measured along the optical axis (z of the camera frame), not along the ray.
- **Asset frame**: +Y up, bounding box centred at the origin, longest bounding-box side equal to 1. The matrix
  `N_asset_from_mesh` in `asset.json` maps the coordinates of the original mesh file into this frame.
- **Object label**: rotation `R`, translation `t` and per-axis scale `s` place the asset in the world,
  `x_world = R @ diag(s) @ p_asset + t`. For a true-to-size mesh, `s` is isotropic and equals the longest side
  of its bounding box in metres.
- **Image resolution**: the model's scene views have about 416 x 416 pixels (the exact size depends on the
  aspect ratio). Object crops are cut from the stored images, so store them at about twice that resolution,
  e.g. 832 x 832 or 832 x 1248. The pixel thresholds of the sampler (64-pixel masks, 336-pixel crops) refer to
  stored pixels.

## Scene directory

`scene.json`

```json
{
  "scene_id": "room_0001",
  "split": "train",
  "source": "my_dataset",
  "depth_unit": 0.001,
  "synthetic": true,
  "up": [0.0, 1.0, 0.0],
  "floor_height": 0.0,
  "instances": [
    {"id": 2, "asset_id": "chair_a",
     "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [1.20, 0.45, -0.30], "scale": [0.9, 0.9, 0.9],
     "trainable": true}
  ]
}
```

| Field | Meaning |
|---|---|
| `source` | Free-form name; `data.source_weights` in the training configuration sets sampling shares per source. |
| `depth_unit` | Metres per unit of the 16-bit depth PNGs. |
| `synthetic` | `false` for real captures: the depth-gradient losses of the scene views are then skipped. |
| `up` | Gravity-up direction in the world frame (default +Y). Used by the floor loss. |
| `floor_height` | Height of the floor along `up`, or `null` / absent if unknown. |
| `instances[].id` | The object's value in the id maps (>= 1). |
| `instances[].trainable` | Optional; `false` keeps the object visible but never selects it as a target. |

`cameras.npz`: `K` (V, 3, 3) and `c2w` (V, 4, 4), float64, one row per view.

Per view `VVVV` (four digits, starting at `0000`):

| File | Content |
|---|---|
| `rgb/VVVV.jpg` (or `.png`) | The image. |
| `ids/VVVV.png` | uint16, the instance id of every pixel; 0 where no listed object is visible. |
| `depth/VVVV.png` | uint16, depth in `depth_unit` metres; 0 = invalid. Needed in every view (targets and the normalization length). |

`visibility.npz` has one row per (view, instance) with visible pixels:

| Column | Meaning |
|---|---|
| `view`, `instance` | Row key. |
| `pixels`, `box` | Number of mask pixels and mask box `[x0, y0, x1, y1)`. |
| `depth_median` | Median depth of the mask in metres. |
| `visible_fraction` | Visible pixels / unoccluded silhouette inside the image; NaN = unknown. |
| `in_frame_fraction` | Silhouette inside the image / whole silhouette; NaN = unknown. |

`python tools/build_visibility.py <scene dirs>` computes the table from the id and depth maps and writes NaN
into the last two columns. If you can rasterize the posed assets, fill them in: the sampler then skips objects
that are more than 55 % occluded or more than 20 % outside the image.

## Asset directory

`asset.json`

```json
{
  "asset_id": "chair_a",
  "N_asset_from_mesh": [[0.9, 0, 0, 0.0], [0, 0.9, 0, -0.41], [0, 0, 0.9, 0.02], [0, 0, 0, 1]],
  "mesh": "mesh.glb",
  "size": 608,
  "depth_scale": 10000.0,
  "symmetry": "none",
  "views": [{"azimuth_deg": 12.3, "elevation_deg": 10.0, "direction": [0.21, 0.17, 0.96]}]
}
```

| File | Content |
|---|---|
| `cameras.npz` | `K` (32, 3, 3) and `c2w` (32, 4, 4): asset-from-camera poses of the views. |
| `rgb/KK.jpg` | Render on a uniform gray background (sRGB 116). |
| `mask/KK.png` | uint8 foreground mask, 0 or 255. |
| `depth/KK.png` | uint16 depth in asset units times `depth_scale`; 0 on the background. |
| `points.npy` | Optional (P, 3) float32 surface samples in the asset frame. Without it the object is ignored by the floor loss. |
| `mesh.*` | Optional copy of the mesh; needed only to export a GLB of the predicted layout. |

`symmetry` describes rotational symmetry about the asset's +Y axis: `"none"`, `"C2"` ... `"C12"` (n-fold) or
`"SO2"` (continuous, e.g. a round table). Training then accepts every symmetric equivalent of the label.

### The 32 views

`tools/render_asset.py` renders an asset directory from a mesh with Blender:

```bash
python tools/render_asset.py chair.glb --output data/assets/chair_a
```

All views are upright (the asset's +Y is up in the image), 608 x 608 pixels, with a 50 degree field of view, and
the camera is placed so that the mesh fills 0.88 of the image.

| Views | Elevation | Azimuths |
|---|---|---|
| 0 - 7 | 10 degrees | 8, evenly spaced |
| 8 - 15 | 25 degrees | 8, shifted by 15 degrees |
| 16 - 23 | 45 degrees | 8, shifted by 30 degrees |
| 24 - 27 | -20 degrees | 4 |
| 28 - 31 | 65 degrees | 4 |

Inference uses views 0 - 15 (evenly spread in azimuth, alternating between the first two elevations); training
draws from all 32. Other renderers can be used as long as the layout, the gray background and the files above
are kept; the model was trained on Blender Cycles renders with soft, uniform lighting.

## Checking a dataset

```bash
python tools/validate_dataset.py --scenes data/scenes_train.txt --assets data/assets --samples 4
```

checks files, shapes and camera matrices, verifies that every label projects onto its object's mask, and
assembles a few training samples. It needs only NumPy and Pillow.

## What a training sample looks like

`x3dcomposer.data.sampling` picks, per sample, S scene views, N target objects, up to three crops per object and
M renders per object; `x3dcomposer.data.assemble` turns them into the model inputs listed in
`x3dcomposer/constants.py` and the targets listed at the top of `x3dcomposer/data/assemble.py`. All targets are
expressed in the camera frame of the first chosen view and divided by the mean distance of the scene points to
that camera.
