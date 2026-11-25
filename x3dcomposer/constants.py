"""Names and constants of the model input contract, shared by the data, model and inference code.

A sample is a list of N *views* ordered by role: scene views first, then object crops, then asset renders.
Every dense array has the canvas size (H, W) = the largest view; a view of size (h, w) occupies ``[:h, :w]``.

Model inputs (one sample; a leading batch axis is added by the collate function):

    images              (N, 3, H, W)  float32  RGB in [0, 1]
    instance_masks      (N, O, H, W)  float32  0/1 mask of each of the O objects (renders: own foreground only)
    role                (N,)          int64    ROLE_SCENE / ROLE_CROP / ROLE_RENDER
    owner               (N,)          int64    object index of a crop / render, -1 for scene views
    parent              (N,)          int64    scene views: own index; crops: index of the scene view they are cut from
    crop_A              (N, 3, 3)     float32  crops: pixel map from the parent scene view to the crop; else identity
    K                   (N, 3, 3)     float32  intrinsics of the view raster (zeros if not given)
    camera_c2w          (N, 4, 4)     float32  scene / crop: camera pose relative to scene view 0 (identity if only K
                                               is given, zeros if nothing is given); render: asset-from-camera pose
    camera_available    (N,)          bool     intrinsics are given (always True for renders)
    pose_available      (N,)          bool     the camera pose is given (always True for renders)
    depth               (N, 1, H, W)  float32  optical-axis depth, 0 where invalid
    depth_valid         (N, 1, H, W)  bool
    depth_available     (N,)          bool     depth is given (always True for renders)
    image_valid         (N, 1, H, W)  bool     pixels that carry image content (False in the zero padding)
    view_size           (N, 2)        int64    (h, w) of each view
    owner_slots         (O,)          int64    distinct identity-embedding indices in [0, 64)
"""

ROLE_SCENE, ROLE_CROP, ROLE_RENDER = 0, 1, 2

PATCH = 16
IDENTITY_SLOTS = 64          # number of identity embeddings = maximum number of objects per forward pass
IDENTITY_DIM = 64

FLOAT_KEYS = ("images", "instance_masks", "K", "camera_c2w", "depth", "crop_A")
INT_KEYS = ("role", "owner", "parent", "owner_slots", "view_size")
BOOL_KEYS = ("camera_available", "pose_available", "depth_available", "depth_valid", "image_valid")
INPUT_KEYS = FLOAT_KEYS + INT_KEYS + BOOL_KEYS
SPATIAL_KEYS = ("images", "instance_masks", "depth", "depth_valid", "image_valid")

# Condition modes: which of intrinsics (K), camera poses (P) and depth (D) the scene views carry.
CONDITION_MODES = ("K+P+D", "K+P", "K+D", "K", "none")


def mode_flags(mode: str) -> tuple[bool, bool, bool]:
    """``(K, P, D)`` given to the model under ``mode``."""
    if mode not in CONDITION_MODES:
        raise ValueError(f"Unknown condition mode {mode!r}; choose one of {CONDITION_MODES}")
    parts = set() if mode == "none" else set(mode.split("+"))
    return "K" in parts, "P" in parts, "D" in parts


def mode_name(k: bool, pose: bool, depth: bool) -> str:
    """Mode string of the flags (poses and depth require intrinsics)."""
    if (pose or depth) and not k:
        raise ValueError("Camera poses and depth require intrinsics")
    parts = [name for name, on in (("K", k), ("P", pose), ("D", depth)) if on]
    return "+".join(parts) if parts else "none"
