"""Write ``visibility.npz`` for scene directories (format: docs/DATA.md).

    python tools/build_visibility.py data/scenes/scene_0001 [more scene directories ...]

The table is computed from the id and depth maps. Its columns ``visible_fraction`` and ``in_frame_fraction``
need the unoccluded silhouette of each object and are written as NaN ("unknown"); fill them in if you can
render the posed assets, so that the sampler can skip heavily occluded or truncated objects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from x3dcomposer.data.scene import VISIBILITY_FILE, Scene, compute_visibility


def main() -> None:
    """Write visibility.npz into every given scene directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenes", nargs="+", help="scene directories")
    args = parser.parse_args()
    for path in args.scenes:
        table = compute_visibility(Scene(path))
        np.savez_compressed(Path(path) / VISIBILITY_FILE, **table)
        print(f"{path}: {len(table['view'])} rows")


if __name__ == "__main__":
    main()
