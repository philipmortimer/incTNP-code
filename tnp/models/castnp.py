# incTNP model code
# (castnp => causal TNP => incTNP)
from typing import Optional, Union
import torch
from check_shapes import check_shapes
from torch import nn
from ..networks.transformer import ISTEncoder, PerceiverEncoder, TNPTransformerMaskedEncoder, TNPTransformerFullyMaskedEncoder, convert_transformer_encoder
from ..utils.helpers import preprocess_observations
from .base import ConditionalNeuralProcess
from .tnp import TNPDecoder
import warnings
from .incUpdateBase import IncUpdateEff
from ..networks.kv_cache import init_kv_cache, repeat_kv_cache_batch
import torch.distributions as td


# TNP using causal attention mask - breaking context permutation invariance
class TNPEncoderMasked(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TNPTransformerMaskedEncoder],# This is now converted to TNPTransformerFullyMaskedEncoder in code
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = convert_transformer_encoder(transformer_encoder) # Type TNPTransformerFullyMaskedEncoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

        if not isinstance(self.transformer_encoder, TNPTransformerMaskedEncoder): # TODO: add support for perceiver encoder and IST encoder
            warnings.warn("Perceiver Encoder and IST Encoder not currently supported for masked TNP encoder.")

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        m, nc, _ = xc.shape
        zc = self._preprocess_context(xc, yc)
        zt = self._preprocess_targets(xt, yc.shape[2])

        # Causal masked attention for context - using mask=None will get same behaviour as non masked
        #m, nc, _ = xc.shape # Number of context points
        #causal_mask = nn.Transformer.generate_square_subsequent_mask(nc, device=zc.device)
        #causal_mask = causal_mask.unsqueeze(0).expand(m, -1, -1).contiguous() # [m, nc, nc]
        zt = self.transformer_encoder(zc, zt, use_causal=True)
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


class TNPCausal(ConditionalNeuralProcess, IncUpdateEff):
    def __init__(
        self,
        encoder: TNPEncoderMasked,
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
