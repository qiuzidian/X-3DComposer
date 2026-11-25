"""The VGGT-Omega backbone: construction, weight loading and the few places where its internals are used.

X-3DComposer keeps the architecture of the 1B VGGT-Omega model (https://github.com/facebookresearch/vggt-omega).
The ``vggt_omega`` package is imported from the user's installation; no upstream code is copied here. This module
owns the architecture constants, the strict weight load, the 9 -> 12 extension of the camera head output and the
replay of the camera-head trunk that leaves the three added log-scale outputs un-activated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

DEPTH = 24                    # frame-attention blocks == global-attention blocks
WIDTH = 1024                  # token width of the aggregator; the heads read 2 * WIDTH
PATCH = 16
REGISTERS = 16
SPECIAL_TOKENS = 1 + REGISTERS            # camera token + register tokens precede the patch tokens
REGISTER_LAYERS = frozenset({2, 6, 9, 14, 20})   # global blocks that attend to the special tokens only
TAP_LAYERS = frozenset({4, 11, 17, 23})          # layers whose [frame || global] tokens feed the heads
CAMERA_OUTPUTS = 12                       # 9-d camera encoding (t, quaternion, FoV) + per-axis log-scale


def file_sha256(path: str | Path) -> str:
    """Hex sha256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_backbone(weights: str | Path | None, gradient_checkpointing: bool = False,
                   expected_sha256: str | None = None) -> nn.Module:
    """A CPU ``VGGTOmega`` with ``weights`` loaded strictly (``None``: the constructor's random initialization).

    With ``expected_sha256`` the file is hashed first and a different checkpoint is refused: adapter weights
    only work with the backbone checkpoint they were trained on.
    """
    from vggt_omega.models import VGGTOmega

    backbone = VGGTOmega(use_checkpoint=gradient_checkpointing)
    if weights is not None:
        if expected_sha256 is not None and file_sha256(weights) != expected_sha256:
            raise ValueError(f"{weights} is not the backbone checkpoint these weights were trained on "
                             f"(expected sha256 {expected_sha256})")
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        backbone.load_state_dict(state, strict=True)
    check_architecture(backbone)
    return backbone


def check_architecture(backbone: nn.Module) -> None:
    """Raise unless ``backbone`` is the 1B model this code was written for."""
    a = backbone.aggregator
    schedule = ["register" if i in REGISTER_LAYERS else "global" for i in range(DEPTH)]
    if (a.depth != DEPTH or a.patch_size != PATCH or a.patch_token_start != SPECIAL_TOKENS
            or set(a.cached_layer_indices) != TAP_LAYERS or a.inter_frame_attention_types != schedule):
        raise ValueError("Unexpected VGGT-Omega architecture; X-3DComposer needs the 1B model")


def extend_camera_branch(camera_head: nn.Module) -> None:
    """Replace the 9-output Linear of the camera head by a 12-output one (rows 9..11 = log-scale, zero-init)."""
    old = camera_head.camera_branch[-1]
    if not isinstance(old, nn.Linear) or old.out_features != 9:
        raise ValueError("Expected the original 9-d camera head")
    new = nn.Linear(old.in_features, CAMERA_OUTPUTS, bias=old.bias is not None,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:9].copy_(old.weight)
        if old.bias is not None:
            new.bias.zero_()
            new.bias[:9].copy_(old.bias)
    camera_head.camera_branch[-1] = new


def camera_head_raw(camera_head: nn.Module, special_tokens: torch.Tensor) -> torch.Tensor:
    """Camera-head trunk and output branch on the final special tokens (B, N, 17, 2048) -> raw (B, N, 12).

    The same computation as ``CameraHead.forward`` without its final activation, so that the log-scale outputs
    stay raw; ``activate_camera`` applies the original activation to the first nine.
    """
    b, n, count, width = special_tokens.shape
    tokens = camera_head.token_norm(special_tokens.float()).reshape(b, n * count, width)
    for block in camera_head.trunk:
        tokens = block(tokens, None)
    camera = camera_head.trunk_norm(tokens.reshape(b, n, count, width)[:, :, 0])
    return camera_head.camera_branch(camera)


def activate_camera(raw: torch.Tensor) -> torch.Tensor:
    """The original activation of the 9-d camera encoding (translation, quaternion, relu(FoV) + 0.01)."""
    from vggt_omega.models.heads.camera_head import _apply_camera_activation

    return _apply_camera_activation(raw)


def expand_special_token(token: torch.Tensor, batch: int, views: int) -> torch.Tensor:
    """(1, 2, T, C) learned token -> (batch * views, T, C): slot 0 for the first view, slot 1 for the others."""
    from vggt_omega.models.aggregator import slice_expand_and_flatten

    return slice_expand_and_flatten(token, batch, views)


def rotation_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrices -> XYZW quaternions in the backbone's convention."""
    from vggt_omega.utils.rotation import mat_to_quat

    return mat_to_quat(matrix)
