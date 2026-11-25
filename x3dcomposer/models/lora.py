"""LoRA on the query and value projections of a frozen fused ``qkv`` Linear."""

from __future__ import annotations

import torch
from torch import nn


class QVLoRA(nn.Module):
    """``qkv(x)`` with low-rank updates ``x A^T B^T`` added to the query and value thirds (keys untouched).

    The LoRA scale is 1 (alpha == rank). ``B`` starts at zero, so the wrapped layer is unchanged at initialization.
    The frozen Linear stays at ``base``; ``in_features`` is kept because the backbone's attention reads it.
    """

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        width = base.in_features
        self.q_A = nn.Parameter(torch.empty(rank, width))
        self.v_A = nn.Parameter(torch.empty(rank, width))
        self.q_B = nn.Parameter(torch.zeros(width, rank))
        self.v_B = nn.Parameter(torch.zeros(width, rank))
        nn.init.normal_(self.q_A, std=0.02)
        nn.init.normal_(self.v_A, std=0.02)

    def parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q, k, v)`` with the LoRA updates added to q and v."""
        q, k, v = self.base(x).chunk(3, -1)
        return q + (x @ self.q_A.T) @ self.q_B.T, k, v + (x @ self.v_A.T) @ self.v_B.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.parts(x), -1)
