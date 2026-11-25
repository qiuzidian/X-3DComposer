"""Grounding head (training only): the camera token of a crop / render predicts where its object is in the scene.

``p_r(u) = softmax_u( tau * cos(W1 c_r, W2 h_u) )`` with ``c_r`` the final camera token of view r and ``h_u`` the
final token of scene-view patch u (both from the last backbone layer, width 2048). The loss is in
``x3dcomposer.losses.grounding``. The head is not used at inference.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

TOKEN_WIDTH = 2048
GROUNDING_DIM = 256
INITIAL_TEMPERATURE = 10.0


class GroundingHead(nn.Module):
    """Cosine-similarity logits between every view's camera token and all scene-view patches."""

    def __init__(self, dim: int = GROUNDING_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_WIDTH)
        self.key = nn.Linear(TOKEN_WIDTH, dim)
        self.query = nn.Linear(TOKEN_WIDTH, dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(INITIAL_TEMPERATURE)))

    def forward(self, patches: torch.Tensor, cameras: torch.Tensor) -> torch.Tensor:
        """``patches`` (B, S, P, 2048), ``cameras`` (B, N, 2048) -> logits (B, N, S * P)."""
        keys = F.normalize(self.key(self.norm(patches.float())), dim=-1).flatten(1, 2)
        queries = F.normalize(self.query(self.norm(cameras.float())), dim=-1)
        return self.log_temperature.exp() * torch.einsum("bnd,bkd->bnk", queries, keys)
