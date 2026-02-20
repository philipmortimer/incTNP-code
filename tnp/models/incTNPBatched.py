# incTNP-Seq  - incTNP with batching strategy explored.
from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import TNPTransformerFullyMaskedEncoder
from ..utils.helpers import preprocess_observations
from .base import BatchedCausalTNP
from .tnp import TNPDecoder
from ..utils.helpers import preprocess_observations
from .incUpdateBase import IncUpdateEff
from ..networks.kv_cache import init_kv_cache, repeat_kv_cache_batch
import torch.distributions as td


class IncTNPBatchedEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TNPTransformerFullyMaskedEncoder],
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]",
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]",
        "return: [m, n_t_or_n_minus_one, dz]",
    )
    def forward(
        self, x: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None,
        xc: Optional[torch.Tensor] = None, yc: Optional[torch.Tensor] = None, xt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Checks that it either provides (x,y) OR (xc, yc, xt) but not both. This is used to determine whether train / prediction is happening
        assert (xc is None and yc is None and xt is None and y is not None and x is not None) or (xc is not None and yc is not None and xt is not None and x is None and y is None), "Invalid encoder call. Can't differentiate between prediction or training call"

        if x is not None and y is not None: return self.train_encoder(x, y)
        else: return self.predict_encoder(xc, yc , xt)

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def predict_encoder(self, xc: torch.Tensor, yc:torch.Tensor, xt:torch.Tensor):
        # At prediction time we essentially become identically to incTNP basic
        # (I.e.) just self attention over the context points and no cross attention mask.
        m, nc, _ = xc.shape
        zc = self._preprocess_context(xc, yc)
        zt = self._preprocess_targets(xt, yc.shape[2])

        zt = self.transformer_encoder(zc, zt, mask_sa=None, use_causal=True, mask_ca=None)
        
        return zt

    @check_shapes(
        "xt: [m, nt, dx]", "return: [m, nt, dz]"
    )
    def _preprocess_targets(self, xt: torch.Tensor, dy: int):
        m, nt, _ = xt.shape
        # Creates yt of zeros plus a bool flag
        yt = torch.zeros(m, nt, dy).to(xt)
        yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,)).to(yt)), dim=-1)
        # Encodes
        xt_encoded = self.x_encoder(xt)
        yt_encoded = self.y_encoder(yt)
        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        return self.xy_encoder(zt) 

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "return: [m, nc, dz]"
    )
    def _preprocess_context(self, xc: torch.Tensor, yc:torch.Tensor):
        yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,)).to(yc)), dim=-1) # Adds flag
        xc_encoded = self.x_encoder(xc)
        yc_encoded = self.y_encoder(yc)
        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        return self.xy_encoder(zc) 


    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]","return: [m, n_minus_one, dz]"
    )
    def train_encoder(self, x: torch.Tensor, y:torch.Tensor):
        m, n, dy = y.shape
        # Treats sequence as just x and y. y_tgt is set to just be 0s to
        # y_like vector with one fewer target point. This is because we dont want to train with prior (i.e empty context).
        # May want to consider changing to having a learnable dummy variable in the context to learn a prior and include this within the loss.
        y_like = torch.zeros((m, n-1, dy)).to(y)
        y_tgt = torch.cat((y_like, torch.ones(y_like.shape[:-1] + (1,)).to(y)), dim=-1)

        y_ctx = torch.cat((y, torch.zeros(y.shape[:-1] + (1,)).to(y)), dim=-1)

        # Encodes x and y
        x_encoded = self.x_encoder(x)
        x_tgt_encoded = x_encoded[:, 1:, :] # Same as before - we dont use x_0 as a target currently due to zero shot (also technically assumes x is iid encoded which is fair)
        y_ctx_encoded = self.y_encoder(y_ctx)
        y_tgt_encoded = self.y_encoder(y_tgt)

        # Embeds data
        zc = torch.cat((x_encoded, y_ctx_encoded), dim=-1)
        zt = torch.cat((x_tgt_encoded, y_tgt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        # Creates masks. 
        # A target point can only attend to preceding context points.
        mask_ca = torch.tril(torch.ones(n-1, n, dtype=torch.bool, device=zc.device), diagonal=0)
        #mask_ca = mask_ca.unsqueeze(0).expand(m, -1, -1) # [m, n, n]
        # Causal masking for context -> a context point can only attend to itself and previous context points.
        #mask_sa = torch.tril(torch.ones(n, n, dtype=torch.bool, device=zc.device), diagonal=0)
        #mask_sa = mask_sa.unsqueeze(0).expand(m, -1, -1) # [m, n, n]

        zt = self.transformer_encoder(zc, zt, mask_sa=None, use_causal=True, mask_ca=mask_ca)
        
        assert len(zt.shape) == 3 and zt.shape[0] == m and zt.shape[1] == n - 1, "Return encoder shape wrong"
        return zt



class IncTNPBatched(BatchedCausalTNP, IncUpdateEff):
    def __init__(
        self,
        encoder: IncTNPBatchedEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)

    # Logic for effecient incremental context updates
    def init_inc_structs(self, m: int, max_nc: int, device: str, use_flash: bool=False, cache_mhca: bool=False, persist_small: bool=False):
        self.kv_cache_inc = init_kv_cache()
        self.kv_small = init_kv_cache() if persist_small else None

   # Repeats context representation
    def repeat_ctx(self, repeat_times: int, persist_small: bool=False):
        if persist_small: 
            self.kv_cache_inc = init_kv_cache()
            repeat_kv_cache_batch(cache_in=self.kv_small, cache_out=self.kv_cache_inc, repeat_times=repeat_times)
        else: repeat_kv_cache_batch(cache_in=self.kv_cache_inc, cache_out=self.kv_cache_inc, repeat_times=repeat_times)

    # Adds new context points
    def update_ctx(self, xc: torch.Tensor, yc: torch.Tensor, use_flash: bool=False, cache_mhca: bool=False, persist_small: bool=False):
        zc = self.encoder._preprocess_context(xc, yc)
        cache = self.kv_small if persist_small else self.kv_cache_inc # Can just update the smaller context rep. Set persist small=True when just conditioning and False for this call when unrolling with stacked samples
        self.encoder.transformer_encoder.encode_context(zc, cache, use_flash=use_flash, use_mhca_kv_cache=cache_mhca)

    def query(self, xt: torch.Tensor, dy: int, use_flash: bool=False, cache_mhca: bool=False, persist_small: bool=False) -> td.Normal:
        zt = self.encoder._preprocess_targets(xt, dy)
        return self.likelihood(self.decoder(self.encoder.transformer_encoder.query(zt, self.kv_cache_inc, use_flash=use_flash, use_mhca_kv_cache=cache_mhca), xt))

