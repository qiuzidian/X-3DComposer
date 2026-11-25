"""Turn a training checkpoint into release weights: ``model.safetensors`` + ``config.json``.

    python tools/export_weights.py runs/x3dcomposer/checkpoint.pt --output weights/ [--raw]

Only the trainable parameters are kept (the EMA weights by default, ``--raw`` for the non-averaged ones); the
optimizer state, the step counter and the training configuration are dropped. The output holds the LoRA
weights, the condition encoders, the identity embeddings and their projections, the camera-head output branch
and the grounding head. The VGGT-Omega backbone weights are not included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

FORMAT = "x3dcomposer-adapter-v1"
BACKBONE = {"repository": "facebook/VGGT-Omega", "file": "vggt_omega_1b_416_reproduce.pt",
            "sha256": "e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896"}


def write_release(state: dict[str, torch.Tensor], lora_rank: int, output: str | Path) -> dict:
    """Write ``model.safetensors`` and ``config.json`` for a trainable-parameter state; returns the config."""
    state = {name: value.detach().to(torch.float32).contiguous() for name, value in state.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output / "model.safetensors"), metadata={"format": "pt"})
    config = {"format": FORMAT, "model": {"lora_rank": lora_rank}, "backbone": BACKBONE, "tensors": len(state),
              "parameters": int(sum(v.numel() for v in state.values()))}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def export(checkpoint_path: str | Path, output: str | Path, raw: bool = False) -> dict:
    """Export a checkpoint written by ``x3dcomposer.train`` and return the content of ``config.json``."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if raw else checkpoint["ema"]
    return write_release(dict(state), int(checkpoint["config"]["model"]["lora_rank"]), output)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True, help="directory for model.safetensors and config.json")
    parser.add_argument("--raw", action="store_true", help="export the non-averaged weights instead of the EMA")
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output, args.raw), indent=2))


if __name__ == "__main__":
    main()
