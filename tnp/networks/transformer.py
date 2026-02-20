import copy
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn
from .kv_cache import update_ctx_cache, update_kv_cache
from .kv_cache_fixed import get_layer_ctx, update_ctx_cache_fixed, get_layer_id

from .attention_layers import (
    MultiHeadCrossAttentionLayer,
    MultiHeadKRAttentionLayer,
    MultiHeadSelfAttentionLayer,
)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

    @check_shapes("x: [m, n, d]", "mask: [nq, nkv]", "return: [m, n, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        tnpa_kv: bool = False,
    ) -> torch.Tensor:
        for (i, mhsa_layer) in enumerate(self.mhsa_layers):
            tag = f"layer_{i}" if kv_cache is not None else None
            x = mhsa_layer(x, mask, kv_cache=kv_cache, kv_tag=tag, tnpa_kv=tnpa_kv)

        return x


class TNPTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: Optional[MultiHeadSelfAttentionLayer] = None,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = (
            self.mhca_layers
            if mhsa_layer is None
            else _get_clones(mhsa_layer, num_layers)
        )

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, MultiHeadSelfAttentionLayer):
                xc = mhsa_layer(xc)
            elif isinstance(mhsa_layer, MultiHeadCrossAttentionLayer):
                xc = mhsa_layer(xc, xc)
            else:
                raise TypeError("Unknown layer type.")

            xt = mhca_layer(xt, xc)

        return xt


# Tnp Encoder that supports masked self attention and cross attention.
class TNPTransformerFullyMaskedEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: MultiHeadSelfAttentionLayer,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask_sa: [nc, nc]", "mask_ca: [nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask_sa: Optional[torch.Tensor] = None, mask_ca: Optional[torch.Tensor] = None,
        use_causal: bool = False,
    ) -> torch.Tensor:

        for i, (mhsa_layer, mhca_layer) in enumerate(zip(self.mhsa_layers, self.mhca_layers)):
            if isinstance(mhsa_layer, MultiHeadSelfAttentionLayer):
                xc = mhsa_layer(xc, mask=mask_sa, use_causal=use_causal)
            else:
                raise TypeError("Unknown layer type.")

            xt = mhca_layer(xt, xc, mask=mask_ca)

        return xt

    # Fixed KV (more opt) - Computes the MHSA representation with causal plus kv
    @check_shapes(
        "zc_new: [m, nc_new, dz]"
    )
    def encode_context_fixedkv(self, zc_new: torch.Tensor, kv_cache: dict, use_flash: bool = False) -> torch.Tensor:
        L = len(self.mhsa_layers)
        assert False, "Currently not supporting fixed setup"
        m, nc_new, dz = zc_new.shape
        use_causal = nc_new > 1 # No need for causal masking with a single new point
        ctx_vals = torch.empty((L, m, nc_new, dz), device=zc_new.device)
        for i, mhsa_layer in enumerate(self.mhsa_layers):
            self_attention_layer_tag = get_layer_id(i)
            zc_new = mhsa_layer(zc_new, kv_cache=kv_cache, kv_tag=self_attention_layer_tag, use_fixed_kv=True, use_flash=use_flash, use_causal=use_causal)
            update_ctx_cache_fixed(zc_new, kv_cache, i) # Writes updated context

    # Fixed KV - Query - runs MHCA pathway assuming MHSA attention has already been computed
    @check_shapes(
        "zt: [m, nt, dz]", "return: [m, nt, dz]"
    )
    def query_fixedkv(self, zt, kv_cache: dict, use_flash: bool = False) -> torch.Tensor:
        for i, mhca_layer in enumerate(self.mhca_layers):
            zt = mhca_layer(zt, get_layer_ctx(i, kv_cache), use_flash=use_flash)
        return zt

    # Computes the MHSA representation with causal plus kv
    @check_shapes(
        "zc_new: [m, nc_new, dz]"
    )
    def encode_context(self, zc_new: torch.Tensor, kv_cache: dict, use_flash: bool = False,
                       use_mhca_kv_cache: bool = False,):
        L = len(self.mhsa_layers)
        m, nc_new, dz = zc_new.shape
        #ctx_vals = torch.empty((L, m, nc_new, dz), device=zc_new.device)
        # Request causal when more than one new token. attention will override this if past is >= 1 with a shifted mask.
        use_causal = (nc_new > 1)
        # Three cases. 1) nc_new = 1 then we need no mask and just do flash. 2) nc_new > 1 and past = 0 then we just use_causal=True. 3) nc_new > 1 and past >=1 then we needed shifted causal mask. This is built for us inside attention.py

        for i, mhsa_layer in enumerate(self.mhsa_layers):
            self_attention_layer_tag = f"layer_{i}_sa" # Layer tag for KV
            zc_new = mhsa_layer(zc_new, kv_cache=kv_cache, kv_tag=self_attention_layer_tag,use_flash=use_flash, use_causal=use_causal)
            # Writes updated context
            ctx_tag = f"context_layer_{i}"
            update_ctx_cache(zc_new, kv_cache, ctx_tag)

            if use_mhca_kv_cache:
                self._cache_mhca_kv(i=i, zc_new=zc_new, kv_cache=kv_cache)

    # Caches the K/V for MHCA also for the current layer
    def _cache_mhca_kv(self, i: int, zc_new: torch.Tensor, kv_cache: dict):
        cross_tag = f"cross_layer_{i}"
        mhca_layer = self.mhca_layers[i]

        xkv_used = mhca_layer.norm1(zc_new) if mhca_layer.norm_first else zc_new # Gets the kv input to be used. Matches the code path in cross attention - so if this changes this will need manual updating

        attn = mhca_layer.attn
        k_new = attn.to_k(xkv_used)
        v_new = attn.to_v(xkv_used)

        h = attn.num_heads
        k_new = einops.rearrange(k_new, "m n (h d) -> m h n d", h=h)
        v_new = einops.rearrange(v_new, "m n (h d) -> m h n d", h=h)
        update_kv_cache(k_new, v_new, kv_cache, cross_tag)

    # Query - runs MHCA pathway assuming MHSA attention has already been computed
    @check_shapes(
        "zt: [m, nt, dz]", "return: [m, nt, dz]"
    )
    def query(self, zt, kv_cache: dict, use_flash: bool = False,
              use_mhca_kv_cache: bool = False,) -> torch.Tensor:
        for i, mhca_layer in enumerate(self.mhca_layers):
            ctx_tag = f"context_layer_{i}"
            if use_mhca_kv_cache:
                cross_tag = f"cross_layer_{i}"
                precomputed_kv = kv_cache[cross_tag]
                zt = mhca_layer(zt, kv_cache[ctx_tag], use_flash=use_flash, precomputed_kv=precomputed_kv)
            else:
                zt = mhca_layer(zt, kv_cache[ctx_tag],use_flash=use_flash)
        return zt


# TNPTransformerEncoder with causal self attention
class TNPTransformerMaskedEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: MultiHeadSelfAttentionLayer,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, MultiHeadSelfAttentionLayer):
                xc = mhsa_layer(xc, use_causal=True)
            else:
                raise TypeError("Unknown layer type.")

            xt = mhca_layer(xt, xc)

        return xt


# Takes a TNPTransformerMaskedEncoder and converts it to TNPTransformerFullyMaskedEncoder or raises an error
# Takes the current transformer encodeer and converts into a different type that all the other models do.
# This is a legacy change to make code work. In future, simply accept and train with a TNPTransformerFullyMaskedEncoder
# by default. This is done to allow already trained models to leverage the TNPTransformerFullyMaskedEncoder features for KV.
def convert_transformer_encoder(curr_encoder: TNPTransformerMaskedEncoder) -> TNPTransformerFullyMaskedEncoder:
    num_layers = len(curr_encoder.mhca_layers)
    assert len(curr_encoder.mhca_layers) > 0 and len(curr_encoder.mhca_layers) > 0, "Invalid layer numbers to convert"
    # Checks that the provided layers are actually mhsa layers
    for lay in curr_encoder.mhsa_layers: assert isinstance(lay, MultiHeadSelfAttentionLayer), "Cant convert without true MHSA layers"
    return TNPTransformerFullyMaskedEncoder(num_layers=num_layers, mhca_layer=curr_encoder.mhca_layers[0], 
        mhsa_layer=curr_encoder.mhsa_layers[0])


class TNPKRTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhkr_layer: MultiHeadKRAttentionLayer,
    ):
        super().__init__()

        self.mhkr_layers = _get_clones(mhkr_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhkr_layer in self.mhkr_layers:
            xt, xc = mhkr_layer(xt, xc)

        return xt


class BasePerceiverEncoder(nn.Module, ABC):
    def __init__(
        self,
        num_latents: int,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        mhca_ctoq_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        """Base class for the Perceiver encoder.

        Args:
            num_latents (int): Number of latents.
            mhsa_layer (MultiHeadSelfAttentionLayer): MHSA layer between latents.
            mhca_ctoq_layer (MultiHeadCrossAttentionLayer): MHCA layer from context to latents.
            mhca_qtot_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to target.
            num_layers (int): Number of layers.
        """
        super().__init__()

        # Initialise latents.
        embed_dim = mhsa_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)


class PerceiverEncoder(BasePerceiverEncoder):
    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        for mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers, self.mhca_ctoq_layers, self.mhca_qtot_layers
        ):
            xq = mhca_ctoq_layer(xq, xc)
            xq = mhsa_layer(xq)
            xt = mhca_qtot_layer(xt, xq)

        return xt


class BaseISTEncoder(nn.Module, ABC):
    def __init__(
        self,
        num_latents: int,
        mhca_ctoq_layer: MultiHeadSelfAttentionLayer,
        mhca_qtoc_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        """Base class for the IST encoder.

        Args:
            num_latents (int): Number of latents.
            mhca_ctoq_layer (MultiHeadSelfAttentionLayer): MHCA layer from context to latents.
            mhca_qtoc_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to context.
            mhca_qtot_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to target.
            num_layers (int): Number of layers.
        """
        super().__init__()

        embed_dim = mhca_ctoq_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers - 1)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)


class ISTEncoder(BaseISTEncoder):
    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        for i, (mhca_ctoq_layer, mhca_qtot_layer) in enumerate(
            zip(self.mhca_ctoq_layers, self.mhca_qtot_layers)
        ):
            xq = mhca_ctoq_layer(xq, xc)
            xt = mhca_qtot_layer(xt, xq)

            if i < len(self.mhca_qtoc_layers):
                xc = self.mhca_qtoc_layers[i](xc, xq)

        return xt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
