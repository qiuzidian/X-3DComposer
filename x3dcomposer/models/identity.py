"""Identity embeddings: which object a token belongs to, and how that enters the attention blocks.

``IdentityEmbeddings`` turns the object masks of a sample into one identity vector per token (Eq. 3 of the paper);
``IdentityBlock`` wraps a backbone attention block, adds LoRA to its query / value projections and adds the
projected identity vector to the values (and, in the global-attention blocks, to the queries and keys).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..constants import IDENTITY_DIM, IDENTITY_SLOTS, ROLE_SCENE
from .backbone import PATCH, REGISTERS, WIDTH
from .lora import QVLoRA


class IdentityEmbeddings(nn.Module):
    """64 learnable embeddings; each object of a sample is assigned one of them at random (``owner_slots``).

    Per view the result is (1 + 16 + P, 64): the camera token of a crop / render carries the embedding of its
    object (zero for scene views), the register tokens carry zero, and patch u carries the sum of the object
    embeddings weighted by the fraction of the patch that each object's mask covers.
    """

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(IDENTITY_SLOTS, IDENTITY_DIM)

    def forward(self, x: dict) -> torch.Tensor:
        b, n, o, h, w = x["instance_masks"].shape
        table = self.embeddings(x["owner_slots"])                                   # (B, O, 64)
        coverage = F.avg_pool2d(x["instance_masks"].reshape(b * n, o, h, w), PATCH, PATCH)
        coverage = coverage.flatten(2).transpose(1, 2).reshape(b, n, -1, o)
        patch = torch.einsum("bnpo,bod->bnpd", coverage, table)
        own = table[torch.arange(b, device=table.device)[:, None], x["owner"].clamp_min(0)]
        camera = own * (x["role"] != ROLE_SCENE)[..., None]
        return torch.cat([camera[:, :, None], table.new_zeros(b, n, REGISTERS, IDENTITY_DIM), patch], 2)


def _zero_linear() -> nn.Linear:
    layer = nn.Linear(IDENTITY_DIM, WIDTH, bias=False)
    nn.init.zeros_(layer.weight)
    return layer


class IdentityBlock(nn.Module):
    """A backbone ``SelfAttentionBlock`` (kept at ``base``) with Q/V LoRA and identity injection.

    The forward pass is the block's own (pre-norm attention and MLP with layer scale); the only additions are the
    LoRA terms and ``v += id_value(identity)`` (plus ``q += id_query(identity)``, ``k += id_key(identity)`` when
    ``query_key`` is set). All three projections are zero-initialized.
    """

    def __init__(self, base: nn.Module, rank: int, query_key: bool):
        super().__init__()
        if base.sample_drop_ratio:
            raise ValueError("Blocks with stochastic depth are not supported")
        self.base = base
        base.attn.qkv = QVLoRA(base.attn.qkv, rank)
        self.id_value = _zero_linear()
        self.id_query = _zero_linear() if query_key else None
        self.id_key = _zero_linear() if query_key else None

    def forward(self, x: torch.Tensor, identity: torch.Tensor, rope=None) -> torch.Tensor:
        block, attention = self.base, self.base.attn
        q, k, v = attention.qkv.parts(block.norm1(x))
        v = v + self.id_value(identity).to(v.dtype)
        if self.id_query is not None:
            q = q + self.id_query(identity).to(q.dtype)
            k = k + self.id_key(identity).to(k.dtype)
        attended = attention.compute_attention(torch.cat([q, k, v], -1), rope=rope)
        x = x + block.ls1(attention.proj_drop(attention.proj(attended)))
        return x + block.ls2(block.mlp(block.norm2(x)))
