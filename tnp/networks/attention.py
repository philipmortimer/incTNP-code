from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn
from .kv_cache import update_kv_cache
from .kv_cache_fixed import update_kv_cache_fixed, get_mask_fixed
from torch.nn.attention import SDPBackend, sdpa_kernel


class BaseMultiHeadAttention(nn.Module, ABC):
    def __init__(
        self,
        qk_dim: int,
        v_dim: int,
        num_heads: int,
        head_dim: int,
        p_dropout: float = 0.0,
        linear: bool = False,
    ):
        super().__init__()

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == v_dim)

        self.to_q = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, v_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

        self.linear = linear

    @check_shapes(
        "xq: [m, nq, dqk]",
        "xk: [m, nkv, dqk]",
        "xv: [m, nkv, dv]",
        #"mask: [nq, nkv]",
        "return: [m, nq, dv]",
    )
    def propagate(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None, # Stores cached KV values if being used
        kv_tag: Optional[str] = None, # Layer ID to look up in kv_cache,
        use_causal: bool = False, # Whether to set causal flag in SDPA,
        use_fixed_kv: bool = False, # Whether to use more optimised fixed kv cache or not - less safe but potentially faster
        use_flash: bool = False,
        tnpa_kv: bool=False, # Mask for TNPA unroll
        precomputed_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q = self.to_q(xq)
        q = einops.rearrange(q, "m n (h d) -> m h n d", h=self.num_heads)

        if precomputed_kv is not None:
            k, v = precomputed_kv
            assert kv_tag is None and kv_cache is None and not use_fixed_kv, "Calling for two things at once"
        else:
            k_new = self.to_k(xk)
            v_new = self.to_v(xv)

            k_new, v_new = map(
                lambda x: einops.rearrange(x, "m n (h d) -> m h n d", h=self.num_heads),
                (k_new, v_new),
            )

            # This is a little hard to navigate because its been tinkered with. But here is the logic
            # k_v tag means no kv caching (k and v are passed as is). use_flash signals to structure the call
            # so that flash attention can be used. Flash needs mask=None. For pure square matrices it also supports use_causal=True.
            # Otherwise it needs use_causal=False. use_causal=False works for bidir attention (TNP-D or unmasked MHCA) or for AR TNP unrolls as there is a single new arriving point.
            # For context points arriving in chunks, an offset square mask needs to be made (which flash attention wont support.)
            # Thus use_flash=True should only be set in the case where incremental AR updates are happening or TNP-D or unmakes MHCA (target pathway). This function is definitely fiddly so use with care.
            # Support is also added here to allow for use_flash for the initi for AR incTNPs where we need use_causal=True.

            if kv_tag is None: # KV caching not used if no tag provided
                k, v = k_new, v_new
                if use_flash: 
                    mask = None
                    if q.shape[2] == 1: use_causal = False # This ONLY works for the case of one token added at time (which is true for AR NPs) - use with care
                        
            else:
                if use_fixed_kv:
                    k, v = update_kv_cache_fixed(k_new, v_new, kv_cache, kv_tag)
                    if use_flash:
                        mask = None
                        if q.shape[2] == 1: use_causal = False # This ONLY works for the case of one token added at time (which is true for AR NPs) - use with care
                    else:
                        if kv_cache is not None and kv_tag is not None:
                            m, _, k_len, _ = k.shape
                            _, _, q_len, _ = q.shape
                            mask = get_mask_fixed(kv_cache, q_len, k_len)
                            use_causal = False
                else:
                    k, v = update_kv_cache(k_new, v_new, kv_cache, kv_tag)
                    if use_flash:
                        mask = None
                        if q.shape[2] == 1: use_causal = False # This ONLY works for the case of one token added at time (which is true for AR NPs) - use with care
                    else:
                        # Loads cached mask in case of KV caching - https://github.com/pytorch/pytorch/issues/144858
                        if kv_cache is not None and kv_tag is not None:
                            m, _, k_len, _ = k.shape
                            _, _, q_len, _ = q.shape

                            if tnpa_kv:
                                use_causal = False
                                if mask is None:
                                    mask = torch.tril(torch.ones(k_len, k_len, dtype=torch.bool, device=k.device))[-q_len:]
                                #if q_len == 1:
                                #    mask = torch.ones((1, k_len), dtype=torch.bool, device=k.device)
                                #    mask[:, -1] = False
                                #else: # fallback case that should never happen
                                #   mask = torch.tril(torch.ones(k_len, k_len, dtype=torch.bool, device=k.device))[-q_len:]
                            else:
                                mask = torch.tril(torch.ones(k_len, k_len, dtype=torch.bool, device=k.device))[-q_len:]
                                use_causal = False

        #if mask is not None:
            # Shape goes from [m, nq, nkv] -> [m, h, nq, nkv] by only changing view (no new memory allocated)
            # Code used mask = einops.repeat(mask, "m n1 n2 -> m h n1 n2", h=self.num_heads) previously. More readable but uses more memory.
        #    mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        #    mask = mask.contiguous()

        if self.linear:
            out = linear_attention(q, k, v, attn_mask=mask, scale=self.scale)
        else:
            out = nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                    q, k, v, attn_mask=mask, scale=self.scale, is_causal=use_causal
                )
        out = einops.rearrange(out, "m h n d -> m n (h d)")
        out = self.to_out(out)
        return out


class MultiHeadAttention(BaseMultiHeadAttention):
    @check_shapes(
        "xq: [m, nq, dqk]",
        "xk: [m, nkv, dqk]",
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
    ):
        return super().propagate(xq, xk, xv, mask)


class MultiHeadSelfAttention(BaseMultiHeadAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes("x: [m, n, d]", "mask: [nq, nkv]", "return: [m, n, d]")
    def forward(
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
        return super().propagate(x, x, x, mask, kv_cache=kv_cache, kv_tag=kv_tag, use_causal=use_causal, use_fixed_kv=use_fixed_kv,use_flash=use_flash, tnpa_kv=tnpa_kv)


class MultiHeadCrossAttention(BaseMultiHeadAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [nq, nkv]",
        "return: [m, nq, dx]",
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None, use_flash: bool = False,
        precomputed_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        return super().propagate(xq, xkv, xkv, mask, use_flash=use_flash, precomputed_kv=precomputed_kv)


class MultiHeadKRAttention(BaseMultiHeadAttention):
    """https://arxiv.org/abs/2411.12502."""
    def __init__(self, *, embed_dim: int, **kwargs):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dx]",
        "return[1]: [m, nkv, dx]",
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        # Concatenate queries and keys.
        xqk = torch.cat([xq, xkv], dim=-2)

        out = super().propagate(xqk, xkv, xkv, mask)

        # Split into query and key output.
        outq, outk = torch.split(out, [xq.shape[-2], xkv.shape[-2]], dim=-2)

        return outq, outk


@check_shapes(
    "q: [m, h, nq, dqk]",
    "k: [m, h, nkv, dqk]",
    "v: [m, h, nkv, dq]",
)
def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    scale: float = 1.0,
):
    if attn_mask is not None:
        # TODO: What is going on here.
        raise NotImplementedError("Not implemented yet.")

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-1)
    q = q * scale

    kv = k.transpose(-1, -2) @ v
    out = q @ kv
    return out