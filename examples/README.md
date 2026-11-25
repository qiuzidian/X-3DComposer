# Example data

## `ycbv_tabletop`

Two photos of six household objects on a table, with everything `compose` / `demo.py` can take as input:

```
ycbv_tabletop/
  images/view0.jpg, view1.jpg      640 x 480 RGB photos
  masks/view0.png, view1.png       object id per pixel: 0 = background, 1..6 = the assets in the order below
  depth/view0.png, view1.png       16-bit depth in millimetres (0 = no measurement); optional input
  cameras.npz                      K (2, 3, 3) intrinsics and c2w (2, 4, 4) camera-to-world poses; optional input
  assets/<name>/                   one pre-rendered asset directory per object (format: docs/DATA.md)
```

Mask ids: 1 `cracker_box`, 2 `pudding_box`, 3 `gelatin_box`, 4 `banana`, 5 `mug`, 6 `scissors`.

To keep the repository small, each asset directory holds only the 16 views that inference uses, stored at
304 x 304 pixels, and a vertex-coloured mesh (`mesh.ply`, millimetres). `tools/render_asset.py` writes all 32
views at 608 x 608 pixels.

### Source and license

The photos, masks, depth maps, camera parameters and object models come from the **YCB-V** dataset in the
[BOP benchmark](https://bop.felk.cvut.cz/datasets/) distribution (training video 70, two frames), which the
BOP benchmark lists under the **MIT license**. YCB-V is the BOP version of the YCB-Video dataset; its objects
are from the YCB Object and Model Set. The asset renders and the vertex-coloured meshes in this directory were
derived from the YCB-V object models. These files are redistributed under the same MIT license; they are not
covered by the license of the code in this repository.

Please cite the original works when using this data:

```bibtex
@inproceedings{xiang2018posecnn,
  title     = {{PoseCNN}: A Convolutional Neural Network for {6D} Object Pose Estimation in Cluttered Scenes},
  author    = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2018}
}
@article{calli2015ycb,
  title   = {Benchmarking in Manipulation Research: Using the {Yale-CMU-Berkeley} Object and Model Set},
  author  = {Calli, Berk and Walsman, Aaron and Singh, Arjun and Srinivasa, Siddhartha and Abbeel, Pieter and Dollar, Aaron M.},
  journal = {IEEE Robotics \& Automation Magazine},
  year    = {2015}
}
@inproceedings{hodan2018bop,
  title     = {{BOP}: Benchmark for {6D} Object Pose Estimation},
  author    = {Hoda{\v{n}}, Tom{\'a}{\v{s}} and Michel, Frank and Brachmann, Eric and Kehl, Wadim and Buch, Anders Glent and Kraft, Dirk and Drost, Bertram and Vidal, Joel and Ihrke, Stephan and Zabulis, Xenophon and Sahin, Caner and Manhardt, Fabian and Tombari, Federico and Kim, Tae-Kyun and Matas, Ji{\v{r}}{\'\i} and Rother, Carsten},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2018}
}
```
