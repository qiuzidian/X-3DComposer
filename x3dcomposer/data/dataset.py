"""The training data stream: one item per optimizer step, holding that step's assembled samples."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..config import DataConfig
from .assemble import assemble
from .sampling import StepSampler, summarize
from .scene import Scene, read_scene_list

REDRAWS = 4


class StepDataset(IterableDataset):
    """Yields ``(step, [(x, t), ...])`` for ``step`` in ``[start, stop)``: the samples of one optimizer step.

    Sampling is a pure function of (seed, step, rank), so a resumed run continues the same stream. With several
    loader workers, worker w produces every ``workers``-th step starting at ``start + w``; the loader collects
    from the workers in turn, so the steps arrive in order.
    """

    def __init__(self, config: DataConfig, seed: int, start: int, stop: int, rank: int = 0):
        self.config, self.start, self.stop, self.rank = config, start, stop, rank
        catalog = [summarize(Scene(path), config.assets) for path in read_scene_list(config.scenes)]
        self.sampler = StepSampler(catalog, seed, config.token_budget, config.source_weights,
                                   config.same_asset_probability)

    def samples(self, step: int) -> list[tuple[dict, dict]]:
        """The assembled samples of ``step`` (samples that cannot be assembled are redrawn)."""
        for attempt in range(REDRAWS):
            out = []
            for selection in self.sampler.plan(step, self.rank, attempt):
                try:
                    out.append(assemble(selection, self.config.assets, self.config.augment))
                except ValueError:
                    continue
            if out:
                return out
        raise RuntimeError(f"Step {step}: no sample could be assembled")

    def __iter__(self):
        info = get_worker_info()
        workers, index = (1, 0) if info is None else (info.num_workers, info.id)
        for step in range(self.start, self.stop):
            if (step - self.start) % workers == index:
                yield step, self.samples(step)


def make_loader(config: DataConfig, seed: int, start: int, stop: int, rank: int = 0) -> DataLoader:
    """Loader over ``StepDataset`` (arrays arrive as CPU tensors; one item = one optimizer step)."""
    if not Path(config.scenes).is_file():
        raise FileNotFoundError(f"Scene list not found: {config.scenes}")
    dataset = StepDataset(config, seed, start, stop, rank)
    return DataLoader(dataset, batch_size=None, num_workers=config.workers,
                      prefetch_factor=2 if config.workers else None,
                      multiprocessing_context="spawn" if config.workers else None)


def to_device(sample: dict, device: torch.device) -> dict:
    """Targets of one sample -> batched (leading axis of one) tensors on ``device``."""
    return {key: torch.as_tensor(value).unsqueeze(0).to(device, non_blocking=True) for key, value in sample.items()}
