"""Training of X-3DComposer.

    python -m x3dcomposer.train --config configs/release.yaml                       # one GPU
    torchrun --nproc_per_node 8 -m x3dcomposer.train --config configs/release.yaml  # several GPUs (DDP)
    python -m x3dcomposer.train --config configs/release.yaml --resume runs/x3dcomposer/checkpoint.pt

Each optimizer step draws one sample layout and as many samples as fit the token budget (``data.sampling``).
The samples of a step are processed one by one and their gradients are accumulated, then clipped and applied
once. The trunk runs in bfloat16 autocast; the heads and the losses run in float32. An exponential moving
average of the trainable weights is kept and saved with every checkpoint; ``tools/export_weights.py`` turns a
checkpoint into release weights.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import OptimConfig, TrainConfig, config_dict, load_config
from .data.dataset import make_loader, to_device
from .losses.objective import Objective
from .models.conditions import prepare_inputs
from .models.x3dcomposer import X3DComposer

CHECKPOINT = "checkpoint.pt"


def lr_factor(step: int, config: OptimConfig) -> float:
    """Learning-rate multiplier: linear warm-up from ``final_lr_ratio`` to 1, then cosine decay back to it."""
    low, warm = config.final_lr_ratio, max(1, config.warmup_steps)
    if step < warm:
        return low + (1 - low) * step / warm
    progress = min(1.0, (step - warm) / max(1, config.steps - warm))
    return low + (1 - low) * 0.5 * (1 + math.cos(math.pi * progress))


class WeightEMA:
    """Exponential moving average of a model's trainable parameters (float32 shadow on the parameters' device)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self.shadow = {name: p.detach().float().clone() for name, p in self.parameters.items()}

    @torch.no_grad()
    def update(self) -> None:
        """One EMA step towards the current weights (call after every optimizer step)."""
        for name, p in self.parameters.items():
            self.shadow[name].lerp_(p.detach().float(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.cpu().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if set(state) != set(self.shadow):
            raise KeyError("EMA state does not match the model's trainable parameters")
        for name, value in state.items():
            self.shadow[name].copy_(value)


def save_checkpoint(path: Path, model: X3DComposer, ema: WeightEMA, optimizer: torch.optim.Optimizer, step: int,
                    config: TrainConfig) -> None:
    """Atomically write the trainable weights, their EMA, the optimizer state, the step and the configuration."""
    state = {"model": {k: v.cpu().clone() for k, v in model.adapter_state_dict().items()}, "ema": ema.state_dict(),
             "optimizer": optimizer.state_dict(), "step": step, "config": config_dict(config)}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, model: X3DComposer, ema: WeightEMA, optimizer: torch.optim.Optimizer) -> int:
    """Restore a ``save_checkpoint`` file; returns the number of completed steps."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_adapter_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def train(config: TrainConfig, resume: str | None = None) -> None:
    """Run (or resume) training; under ``torchrun`` every process calls this and they train with DDP."""
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda")
    torch.manual_seed(config.seed + rank)
    output = Path(config.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config_dict(config), indent=2) + "\n")

    optim = config.optim
    model = X3DComposer.from_backbone(config.backbone_weights, config.model, optim.gradient_checkpointing).to(device)
    objective = Objective(config.vggt_omega_root, config.loss).to(device)
    groups = model.parameter_groups(optim.new_lr, optim.pretrained_lr, optim.weight_decay)
    optimizer = torch.optim.AdamW(groups, betas=tuple(optim.betas))
    ema = WeightEMA(model, optim.ema_decay)
    start = load_checkpoint(resume, model, ema, optimizer) if resume else 0
    wrapped = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()]) if distributed else model
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train()

    tick = time.perf_counter()
    for step, samples in make_loader(config.data, config.seed, start, optim.steps, rank):
        factor = lr_factor(step, optim)
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * factor
        optimizer.zero_grad(set_to_none=True)
        logged: dict[str, float] = {}
        for index, (x, t) in enumerate(samples):
            last = index == len(samples) - 1
            # Gradients are accumulated locally and synchronized across ranks once, with the last sample.
            sync = contextlib.nullcontext() if last or not distributed else wrapped.no_sync()
            with sync:
                x = prepare_inputs(x, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = wrapped(x, validate=False)
                losses = objective(prediction, x, to_device(t, device), step / optim.steps)
                (losses["loss"] / len(samples)).backward()
            for name, value in losses.items():
                logged[name] = logged.get(name, 0.0) + float(value.detach()) / len(samples)
        norm = torch.nn.utils.clip_grad_norm_(trainable, optim.grad_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradients at step {step}")
        optimizer.step()
        ema.update()
        done = step + 1
        if rank == 0 and (done % config.log_every == 0 or done == optim.steps):
            row = {"step": done, "samples": len(samples), "grad_norm": float(norm), "lr_factor": factor,
                   "seconds_per_step": (time.perf_counter() - tick) / config.log_every, **logged}
            tick = time.perf_counter()
            with (output / "log.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"step {done}  loss {row['loss']:.4f}  grad {row['grad_norm']:.2f}  "
                  f"{row['seconds_per_step']:.2f} s/step", flush=True)
        if rank == 0 and (done % config.save_every == 0 or done == optim.steps):
            save_checkpoint(output / CHECKPOINT, model, ema, optimizer, done, config)
    if distributed:
        dist.destroy_process_group()


def _overrides(pairs: list[str]) -> dict:
    """``["optim.steps=20", "data.workers=0"]`` -> nested dictionary (values parsed as JSON when possible)."""
    out: dict = {}
    for pair in pairs:
        key, _, text = pair.partition("=")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        node = out
        *parents, leaf = key.split(".")
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = value
    return out


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train X-3DComposer")
    parser.add_argument("--config", help="YAML file overriding the default configuration")
    parser.add_argument("--resume", help="checkpoint.pt of an interrupted run")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. optim.steps=20 data.workers=0")
    args = parser.parse_args()
    train(load_config(args.config, _overrides(args.overrides)), args.resume)


if __name__ == "__main__":
    main()
