"""Configuration dataclasses. The default hyperparameters follow the paper; a YAML file overrides them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Architecture options stored next to the weights (``config.json``)."""

    lora_rank: int = 32


@dataclass
class DataConfig:
    """Training data and sampling (``x3dcomposer.data.sampling``)."""

    scenes: str = "data/scenes_train.txt"        # text file: one scene directory per line
    assets: str = "data/assets"                  # directory of asset directories, named by asset id
    source_weights: dict[str, float] = field(default_factory=dict)   # sampling share per scene ``source``
    token_budget: int = 56_000                   # tokens per GPU and optimizer step
    same_asset_probability: float = 0.09         # samples forced to contain two objects with the same asset
    augment: bool = True
    workers: int = 8


@dataclass
class LossConfig:
    """Loss weights (``x3dcomposer.losses.objective``)."""

    scale: float = 5.0                           # per-render log-scale L1 (plus ``shape`` times its shape part)
    shape: float = 3.0                           # relative weight of the axis-ratio part inside the scale term
    consistency: float = 1.0                     # per-render log-scale vs the object's mean
    candidate: float = 2.0                       # pose candidates vs ground truth
    fused: float = 1.0                           # fused pose vs ground truth
    agreement: float = 1.0                       # pose candidates vs fused pose
    crop: float = 0.5                            # crop camera and crop depth
    grounding: float = 0.5
    floor: float = 0.3
    small_object: float = 2.0                    # extra weight of the scale terms for small objects
    symmetry: bool = True                        # use the symmetric-equivalent label closest to the prediction


@dataclass
class OptimConfig:
    """AdamW and schedule (``x3dcomposer.train``)."""

    steps: int = 200_000
    new_lr: float = 2.5e-5                       # condition encoders, identity embeddings and projections
    pretrained_lr: float = 5e-6                  # LoRA and the output branch of the camera head
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 500
    final_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.9998
    gradient_checkpointing: bool = True


@dataclass
class TrainConfig:
    """Everything ``python -m x3dcomposer.train`` needs."""

    backbone_weights: str = "checkpoints/vggt_omega_1b_416_reproduce.pt"
    vggt_omega_root: str = "third_party/vggt-omega"   # clone of the backbone repository (its training losses)
    output: str = "runs/x3dcomposer"
    seed: int = 0
    save_every: int = 500
    log_every: int = 50
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def _update(target: Any, values: dict) -> None:
    names = {f.name for f in fields(target)}
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"Unknown configuration key {key!r} in {type(target).__name__}")
        current = getattr(target, key)
        if is_dataclass(current):
            _update(current, value)
        else:
            setattr(target, key, tuple(value) if isinstance(current, tuple) else value)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> TrainConfig:
    """Default ``TrainConfig`` updated by a YAML file and by ``overrides`` (nested dictionaries)."""
    config = TrainConfig()
    if path is not None:
        import yaml

        _update(config, yaml.safe_load(Path(path).read_text()) or {})
    _update(config, overrides or {})
    return config


def config_dict(config: Any) -> dict:
    """Plain-dictionary form of a configuration dataclass."""
    return asdict(config)
