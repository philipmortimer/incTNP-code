from abc import ABC
from functools import partial
from typing import Optional, Tuple, Union

import torch
from check_shapes import check_shapes
from torch import nn

from .attention import (
    BaseMultiHeadAttention,
    MultiHeadAttention,
    MultiHeadCrossAttention,
    MultiHeadKRAttention,
    MultiHeadSelfAttention,
)


class BaseMultiHeadAttentionLayer(nn.Module, ABC):
    def __init__(
        self,
        embed_dim: int,
        attention: Union[BaseMultiHeadAttention, partial[BaseMultiHeadAttention]],
        feedforward_dim: Optional[int] = None,
        p_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_first: bool = False,
        **kwargs,
    ):
        super().__init__()
        feedforward_dim = embed_dim if feedforward_dim is None else feedforward_dim

        self.embed_dim = embed_dim
        self.attn = attention(**kwargs)

        # Feedforward model.
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation,
            nn.Dropout(p_dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(p_dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first

        self.attn_dropout = nn.Dropout(p_dropout)


class MultiHeadAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(
        self,
        *,
        qk_dim: int,
        v_dim: int,
        **kwargs,
    ):
        attention = partial(MultiHeadAttention, qk_dim=qk_dim, v_dim=v_dim)
        super().__init__(embed_dim=v_dim, attention=attention, **kwargs)

    @check_shapes(
        "xq: [m, nq, dqk]",
        "xk: [m, nkv, dqk]",
        "xv: [m, nkv, dv]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dv]",
    )
    def attn_block(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(xq, xk, xv, mask=mask)
        return self.attn_dropout(x)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "xv: [m, nkv, dv]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dv]",
    )
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # An MHA block is just the MHA operation.
        xq = self.attn_block(xq, xk, xv, mask)

        return xq


class MultiHeadSelfAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadSelfAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes("x: [m, n, d]", "mask: [nq, nkv]", "return: [m, n, d]")
    def attn_block(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        kv_tag: Optional[str] = None,
        use_causal: bool = False, # Whether to set causal flag in SDPA
        use_fixed_kv: bool = False, # Whether to use more optimised fixed kv cache or not - less safe but potentially faster
        use_flash: bool = False,
        tnpa_kv: bool=False,
    ) -> torch.Tensor:
        x = self.attn(x, mask=mask, kv_cache=kv_cache, kv_tag=kv_tag, use_causal=use_causal, use_fixed_kv=use_fixed_kv, use_flash=use_flash, tnpa_kv=tnpa_kv)
        return self.attn_dropout(x)

    @check_shapes("x: [m, n, d]", "mask: [nq, nkv]", "return: [m, n, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, kv_cache: Optional[dict] = None, kv_tag: Optional[str] = None,
        use_causal: bool = False, # Whether to set causal flag in SDPA
        use_fixed_kv: bool = False, # Whether to use more optimised fixed kv cache or not - less safe but potentially faster
        use_flash: bool = False,
        tnpa_kv: bool=False,
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.attn_block(self.norm1(x), mask, kv_cache=kv_cache, kv_tag=kv_tag, use_causal=use_causal, use_fixed_kv=use_fixed_kv, use_flash=use_flash, tnpa_kv=tnpa_kv)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.attn_block(x, mask, kv_cache=kv_cache, kv_tag=kv_tag, use_causal=use_causal, use_fixed_kv=use_fixed_kv, use_flash=use_flash, tnpa_kv=tnpa_kv))
            x = self.norm2(x + self.ff_block(x))

        return x


class MultiHeadCrossAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadCrossAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes(
        "xq: [m, nq, d]", "xkv: [m, nkv, d]", "mask: [nq, nkv]", "return: [m, n, d]"
    )
    def attn_block(
        self,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_flash: bool = False,
        precomputed_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.attn(xq, xkv, mask=mask, use_flash=use_flash, precomputed_kv=precomputed_kv)
        return self.attn_dropout(x)

    @check_shapes(
        "xq: [m, nq, d]", "xkv: [m, nkv, d]", "mask: [nq, nkv]", "return: [m, n, d]"
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None,use_flash: bool = False,
        precomputed_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.norm_first:
            xq = xq + self.attn_block(self.norm1(xq), self.norm1(xkv), mask,use_flash=use_flash, precomputed_kv=precomputed_kv)
            xq = xq + self.ff_block(self.norm2(xq))
        else:
            xq = self.norm1(xq + self.attn_block(xq, xkv, mask,use_flash=use_flash, precomputed_kv=precomputed_kv))
            xq = self.norm2(xq + self.ff_block(xq))

        return xq


class MultiHeadKRAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadKRAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes(
        "xq: [m, nq, d]",
        "xkv: [m, nkv, d]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, d]",
        "return[1]: [m, nkv, d]",
    )
    def attn_block(
        self,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outq, outkv = self.attn(xq, xkv, mask=mask)
        return self.attn_dropout(outq), self.attn_dropout(outkv)

    @check_shapes(
        "xq: [m, nq, d]",
        "xkv: [m, nkv, d]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, d]",
        "return[1]: [m, nkv, d]",
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.norm_first:
            outq, outkv = self.attn_block(self.norm1(xq), self.norm1(xkv), mask)
            xq = xq + outq
            xkv = xkv + outkv
            xq = xq + self.ff_block(self.norm2(xq))
            xkv = xkv + self.ff_block(self.norm2(xkv))
        else:
            outq, outkv = self.attn_block(xq, xkv, mask)
            xq = self.norm1(xq + outq)
            xkv = self.norm1(xkv + outkv)
            xq = self.norm2(xq + self.ff_block(xq))
            xkv = self.norm2(xkv + self.ff_block(xkv))

        return xq, xkv
