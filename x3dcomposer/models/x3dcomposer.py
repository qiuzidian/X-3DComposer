"""X-3DComposer: VGGT-Omega with objects as views.

Scene views, object crops and asset renders are processed together as views of one multi-view geometry model.
The camera head predicts a camera for every view; object poses follow in closed form from the cameras predicted
for the renders (``x3dcomposer.inference.readout``).

Architecture (1B VGGT-Omega, token width 1024, patch 16)::

    tokens     per view: 1 camera + 16 register + (h / 16) * (w / 16) patch tokens
    trunk      24 x [frame attention (per view, RoPE) -> global attention (all views)];
               five of the global blocks attend to the camera / register tokens only
    added      ConditionEncoder   camera token and patch tokens += encoded intrinsics / pose / depth
               IdentityEmbeddings identity vector per token, projected and added to V in all 48 blocks
                                  and to Q, K in the 24 global blocks
               LoRA (rank 32)     on the Q and V projections of the 48 trunk blocks and the 4 camera-head blocks
    camera     head output extended 9 -> 12: camera encoding (t, quaternion, FoV) + per-axis log-scale;
               the log-scale of an object is the mean over its renders
    depth      the original dense head, on scene views (and crops during training)
    grounding  training-only head on the final tokens (``x3dcomposer.models.grounding``)

Trainable: the output branch of the camera head, all LoRA weights, the condition encoder, the identity
embeddings and their projections, and the grounding head. Everything else is frozen, and the DINO patch
embedding always runs in eval mode. Views are processed in groups of equal resolution: the patch embedding,
the frame-attention blocks (RoPE grid) and the dense head run per group, the global blocks on all tokens.
The trunk runs under the caller's autocast (bfloat16); the heads always run in float32.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from ..constants import ROLE_RENDER, ROLE_SCENE
from .backbone import (DEPTH, PATCH, SPECIAL_TOKENS, WIDTH, activate_camera, build_backbone, camera_head_raw,
                       expand_special_token, extend_camera_branch)
from .conditions import ConditionEncoder, group_inputs, prepare_inputs, view_groups
from .grounding import GroundingHead
from .identity import IdentityBlock, IdentityEmbeddings
from .lora import QVLoRA

MAX_LOG_SCALE = 8.0
WEIGHTS_FILE, CONFIG_FILE = "model.safetensors", "config.json"
BACKBONE_REPO, BACKBONE_FILE = "facebook/VGGT-Omega", "vggt_omega_1b_416_reproduce.pt"


def _rows(role: torch.Tensor, wanted, start: int = 0, stop: int | None = None) -> torch.Tensor:
    """(B, R) indices of the views in ``start:stop`` whose role satisfies ``wanted`` (same count per sample)."""
    return torch.stack([torch.where(wanted(r))[0] for r in role[:, start:stop]])


def object_log_scales(view_log_scale: torch.Tensor, role: torch.Tensor, owner: torch.Tensor,
                      objects: int) -> torch.Tensor:
    """Mean log-scale over the renders of each object: (B, N, 3) -> (B, O, 3); NaN for an object without renders."""
    render = role == ROLE_RENDER
    selected = render[..., None] & (owner[..., None] == torch.arange(objects, device=owner.device))
    finite = torch.isfinite(view_log_scale).all(-1)
    used = torch.where((render & finite)[..., None], view_log_scale, torch.zeros_like(view_log_scale))
    mean = torch.einsum("bno,bnd->bod", selected.float(), used) / selected.sum(1)[..., None]
    invalid = (selected & ~finite[..., None]).any(1)
    return torch.where(invalid[..., None], torch.full_like(mean, float("nan")), mean)


class X3DComposer(nn.Module):
    """The layout model (module docstring). Build with ``from_pretrained`` or ``from_backbone``."""

    def __init__(self, backbone: nn.Module, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = backbone
        aggregator, camera_head = backbone.aggregator, backbone.camera_head
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        extend_camera_branch(camera_head)
        rank = self.config.lora_rank
        aggregator.frame_blocks = nn.ModuleList(IdentityBlock(b, rank, False) for b in aggregator.frame_blocks)
        aggregator.inter_frame_blocks = nn.ModuleList(
            IdentityBlock(b, rank, True) for b in aggregator.inter_frame_blocks)
        for block in camera_head.trunk:
            block.attn.qkv = QVLoRA(block.attn.qkv, rank)
        for parameter in camera_head.camera_branch.parameters():
            parameter.requires_grad_(True)
        self.conditions = ConditionEncoder()
        self.identity = IdentityEmbeddings()
        self.grounding_head = GroundingHead()
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------ construction and weights

    @classmethod
    def from_backbone(cls, backbone_weights: str | Path | None, config: ModelConfig | None = None,
                      gradient_checkpointing: bool = False) -> "X3DComposer":
        """A model with freshly initialized adapters on a pretrained backbone (the starting point of training)."""
        model = cls(build_backbone(backbone_weights), config)
        model.gradient_checkpointing = gradient_checkpointing
        return model

    @classmethod
    def from_pretrained(cls, weights: str | Path, backbone_weights: str | Path | None = None,
                        verify_backbone: bool = True) -> "X3DComposer":
        """Load released weights: ``weights`` is a directory or Hugging Face repository holding ``model.safetensors``
        and ``config.json``; ``backbone_weights`` is the VGGT-Omega checkpoint (downloaded from its gated
        repository when omitted, which needs an authenticated ``huggingface_hub``). ``verify_backbone`` compares
        the checkpoint's sha256 with the one recorded in ``config.json`` (a few seconds for the 4.6 GB file)."""
        from safetensors.torch import load_file

        directory = Path(weights)
        if not directory.is_dir():
            from huggingface_hub import snapshot_download

            directory = Path(snapshot_download(str(weights), allow_patterns=[WEIGHTS_FILE, CONFIG_FILE]))
        if backbone_weights is None:
            from huggingface_hub import hf_hub_download

            backbone_weights = hf_hub_download(BACKBONE_REPO, BACKBONE_FILE)
        info = json.loads((directory / CONFIG_FILE).read_text())
        expected = info.get("backbone", {}).get("sha256") if verify_backbone else None
        model = cls(build_backbone(backbone_weights, expected_sha256=expected), ModelConfig(**info["model"]))
        model.load_adapter_state_dict(load_file(str(directory / WEIGHTS_FILE)))
        return model.eval()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """The trainable parameters by name: everything a checkpoint stores (the backbone is loaded separately)."""
        return {name: p.detach() for name, p in self.named_parameters() if p.requires_grad}

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Strict load of ``adapter_state_dict``: the key sets and all shapes must match exactly."""
        own = dict(self.named_parameters())
        expected = {name for name, p in own.items() if p.requires_grad}
        if set(state) != expected:
            missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
            raise KeyError(f"Adapter weights mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        with torch.no_grad():
            for name, value in state.items():
                if own[name].shape != value.shape:
                    raise ValueError(f"{name}: expected {tuple(own[name].shape)}, got {tuple(value.shape)}")
                own[name].copy_(value)

    def train(self, mode: bool = True) -> "X3DComposer":
        super().train(mode)
        self.backbone.aggregator.patch_embed.eval()
        return self

    def parameter_groups(self, new_lr: float, pretrained_lr: float, weight_decay: float,
                         head_lr_factor: float = 10.0) -> list[dict]:
        """AdamW groups: new modules (condition encoder, identity) at ``new_lr``; LoRA and the camera-head output
        branch at ``pretrained_lr``; the grounding head and the Q/K identity projections at
        ``head_lr_factor * new_lr``. Matrices decay; biases, vectors and the identity embeddings do not."""
        groups: dict[tuple[str, bool], list] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("grounding_head.") or ".id_query." in name or ".id_key." in name:
                kind = "head"
            elif name.startswith(("conditions.", "identity.")) or ".id_value." in name:
                kind = "new"
            else:
                kind = "pretrained"
            decay = p.ndim >= 2 and not name.startswith("identity.") and name != "conditions.pose_unknown"
            groups.setdefault((kind, decay), []).append(p)
        peak = {"new": new_lr, "pretrained": pretrained_lr, "head": new_lr * head_lr_factor}
        return [{"params": params, "lr": peak[kind], "peak_lr": peak[kind],
                 "weight_decay": weight_decay if decay else 0.0, "name": f"{kind}_{'decay' if decay else 'no_decay'}"}
                for (kind, decay), params in groups.items()]

    # ------------------------------------------------------------------ forward

    def _patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Frozen DINO patch tokens (B * N, P, 1024) of one resolution group."""
        aggregator = self.backbone.aggregator
        b, n, c, h, w = images.shape
        with torch.no_grad():
            normalized = ((images - aggregator._resnet_mean) / aggregator._resnet_std).reshape(b * n, c, h, w)
            tokens = aggregator.patch_embed(normalized)
        return tokens["x_norm_patchtokens"] if isinstance(tokens, dict) else tokens

    def _run(self, block: nn.Module, tokens: torch.Tensor, identity: torch.Tensor, rope, global_block: bool):
        if self.gradient_checkpointing and self.training and global_block:
            return checkpoint(block, tokens, identity, rope, use_reentrant=False)
        return block(tokens, identity, rope)

    def aggregate(self, x: dict, groups: list) -> list[list]:
        """Trunk forward. Returns, per resolution group, the 24 layer outputs (B, n_g, T_g, 2048) = [frame ||
        global] tokens at the tap layers and ``None`` elsewhere."""
        aggregator = self.backbone.aggregator
        b, n = x["images"].shape[:2]
        s = SPECIAL_TOKENS
        camera_all = expand_special_token(aggregator.camera_token, b, n).reshape(b, n, 1, WIDTH)
        register_all = expand_special_token(aggregator.register_token, b, n).reshape(b, n, -1, WIDTH)
        tokens, identity, ropes, counts = [], [], [], []
        for start, stop, hw in groups:
            xg = group_inputs(x, start, stop, hw)
            patch = self._patch_tokens(xg["images"])
            patch_delta, camera_delta = self.conditions(xg)
            patch = patch + patch_delta.reshape_as(patch).to(patch.dtype)
            camera = camera_all[:, start:stop].flatten(0, 1) + camera_delta.flatten(0, 1)
            register = register_all[:, start:stop].flatten(0, 1)
            tokens.append(torch.cat([camera, register, patch], 1).reshape(b, stop - start, -1, WIDTH))
            identity.append(self.identity(xg))
            with torch.no_grad():
                sin, cos = aggregator.rope_embed(H=hw[0] // PATCH, W=hw[1] // PATCH)
            ropes.append((sin.to(patch.device), cos.to(patch.device)))
            counts.append(stop - start)
        taps = [[] for _ in groups]
        for i in range(DEPTH):
            frames = [self._run(aggregator.frame_blocks[i], t.flatten(0, 1), d.flatten(0, 1), r, False).reshape(t.shape)
                      for t, d, r in zip(tokens, identity, ropes)]
            block = aggregator.inter_frame_blocks[i]
            if aggregator.inter_frame_attention_types[i] == "global":
                lengths = [f.shape[1] * f.shape[2] for f in frames]
                joint = self._run(block, torch.cat([f.flatten(1, 2) for f in frames], 1),
                                  torch.cat([d.flatten(1, 2) for d in identity], 1), None, True)
                tokens = [part.reshape(f.shape) for part, f in zip(joint.split(lengths, 1), frames)]
            else:       # camera and register tokens of all views only
                special = self._run(block, torch.cat([f[:, :, :s] for f in frames], 1).flatten(1, 2),
                                    torch.cat([d[:, :, :s] for d in identity], 1).flatten(1, 2), None, True)
                special = special.reshape(b, n, s, WIDTH).split(counts, 1)
                tokens = [torch.cat([sp, f[:, :, s:]], 2) for sp, f in zip(special, frames)]
            for g, (frame, token) in enumerate(zip(frames, tokens)):
                taps[g].append(torch.cat([frame, token], -1) if i in aggregator.cached_layer_indices else None)
        return taps

    def forward(self, inputs: dict, dense_crops: bool | None = None, validate: bool = True) -> dict:
        """One forward pass over all views of a batch (input keys: ``x3dcomposer.constants``).

        Returns ``pose_enc`` (B, N, 9) camera encodings (world-to-camera translation, XYZW quaternion, vertical
        and horizontal FoV), ``view_log_scale`` (B, N, 3), ``object_log_scale`` (B, O, 3), ``depth`` /
        ``depth_conf`` of the scene views on the canvas (B, S, H, W, 1) / (B, S, H, W) with ``scene_indices``,
        and with ``dense_crops`` (default: in training mode) ``crop_depth`` / ``crop_indices``. In training mode
        also ``grounding_logits`` (B, N, S * P).
        """
        x = prepare_inputs(inputs, device=next(self.parameters()).device, validate=validate)
        groups = view_groups(x)
        taps = self.aggregate(x, groups)
        with torch.autocast(device_type=x["images"].device.type, enabled=False):
            special = torch.cat([group[-1][:, :, :SPECIAL_TOKENS] for group in taps], 1)
            raw = camera_head_raw(self.backbone.camera_head, special)
            view_log_scale = raw[..., 9:].clamp(-MAX_LOG_SCALE, MAX_LOG_SCALE)
            out = {"pose_enc": activate_camera(raw[..., :9]), "view_log_scale": view_log_scale,
                   "object_log_scale": object_log_scales(view_log_scale, x["role"], x["owner"],
                                                         x["owner_slots"].shape[1]),
                   "scene_indices": _rows(x["role"], lambda r: r == ROLE_SCENE)}
            if self.training:
                out["grounding_logits"] = self._grounding(x, groups, taps)
            self._dense(out, x, groups, taps, self.training if dense_crops is None else dense_crops)
        return out

    def _grounding(self, x: dict, groups: list, taps: list) -> torch.Tensor:
        """Grounding logits from the final camera tokens of all views and the final scene-view patch tokens."""
        cameras = torch.cat([group[-1][:, :, 0] for group in taps], 1)
        patches = []
        for (first, stop, _), group in zip(groups, taps):
            scene = x["role"][0, first:stop] == ROLE_SCENE
            if scene.any():
                patches.append(group[-1][:, scene][:, :, SPECIAL_TOKENS:])
        return self.grounding_head(torch.cat(patches, 1), cameras)

    def _dense(self, out: dict, x: dict, groups: list, taps: list, crops: bool) -> None:
        """Depth of the scene views (and crops) from the original dense head, zero-padded to the canvas."""
        H, W = x["images"].shape[-2:]
        batch = torch.arange(x["images"].shape[0], device=x["images"].device)[:, None]
        wanted = (lambda r: r != ROLE_RENDER) if crops else (lambda r: r == ROLE_SCENE)
        depths, confidences, rows = [], [], []
        for (first, stop, (h, w)), group in zip(groups, taps):
            local = _rows(x["role"], wanted, first, stop)
            if not local.shape[1]:
                continue
            images = x["images"][:, first:stop, :, :h, :w][batch, local]
            depth, confidence = self.backbone.dense_head([None if t is None else t[batch, local] for t in group],
                                                         images=images, patch_token_start=SPECIAL_TOKENS)
            depths.append(F.pad(depth, (0, 0, 0, W - w, 0, H - h)))
            confidences.append(F.pad(confidence, (0, W - w, 0, H - h), value=1.0))
            rows.append(local + first)
        depth, confidence, rows = torch.cat(depths, 1), torch.cat(confidences, 1), torch.cat(rows, 1)
        scenes = out["scene_indices"].shape[1]
        out.update(depth=depth[:, :scenes], depth_conf=confidence[:, :scenes])
        if crops:
            out.update(crop_depth=depth[:, scenes:], crop_indices=rows[:, scenes:])

