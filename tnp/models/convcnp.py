import einops
import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional

from ..networks.setconv import SetConvGridDecoder, SetConvGridEncoder, setconv_to_grid
from .base import ConditionalNeuralProcess
from .tnp import TNPDecoder
from ..networks.fourier_embed import FourierEmbedderHadISD
from .incUpdateBase import IncUpdateEff
import torch.distributions as td


class ConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        grid_encoder: SetConvGridEncoder,
        grid_decoder: SetConvGridDecoder,
        z_encoder: nn.Module,
        hadisd_mode: bool = False, 
        fourier_encoder: Optional[FourierEmbedderHadISD] = None,
    ):
        super().__init__()

        self.conv_net = conv_net
        self.grid_encoder = grid_encoder
        self.grid_decoder = grid_decoder
        self.z_encoder = z_encoder

        self.hadisd_mode = hadisd_mode # hadisd is special case
        self.fourier_encoder = fourier_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dz]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        if self.hadisd_mode: # handles this data type in a bespoke way - hacky but quick cos input is 4d
            flag = torch.ones_like(yc[..., :1])
            elev = xc[..., 2:3]
            time = xc[..., 3:4]
            # Passes these features into fourier embedder
            elev_time_vec = torch.cat((elev, time), dim=-1)
            elev_time_fourier = self.fourier_encoder(elev_time_vec)
            
            z_feats = torch.cat((yc, flag, elev_time_fourier), dim=-1) 
            xc_coords = xc[..., :2] # Cuts out time and elevation for CNN
            xt_coords = xt[..., :2]

            x_grid, z_grid = self.grid_encoder(xc_coords, z_feats)
            z_grid = self.z_encoder(z_grid)
            z_grid = self.conv_net(z_grid)
            zt = self.grid_decoder(x_grid, z_grid, xt_coords)
            return zt
        else: # Original path
            # Add density.
            yc = torch.cat((yc, torch.ones(yc.shape[:-1] + (1,)).to(yc)), dim=-1)

            # Encode to grid.
            x_grid, z_grid = self.grid_encoder(xc, yc)

            # Encode to z.
            z_grid = self.z_encoder(z_grid)

            # Convolve.
            z_grid = self.conv_net(z_grid)

            # Decode.
            zt = self.grid_decoder(x_grid, z_grid, xt)
            return zt


class GriddedConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        z_encoder: nn.Module,
    ):
        super().__init__()
        self.conv_net = conv_net
        self.z_encoder = z_encoder

    @check_shapes(
        "mc: [m, ...]",
        "y: [m, ..., dy]",
        "mt: [m, ...]",
        "return: [m, dt, dz]",
    )
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.Tensor:
        mc_ = einops.repeat(mc, "m n1 n2 -> m n1 n2 d", d=y.shape[-1])
        yc = y * mc_
        z_grid = torch.cat((yc, mc_), dim=-1)
        z_grid = self.z_encoder(z_grid)
        z_grid = self.conv_net(z_grid)
        zt = torch.stack([z_grid[i][mt[i]] for i in range(mt.shape[0])])
        return zt


#class ConvCNP(ConditionalNeuralProcess, IncUpdateEff):
class ConvCNP(ConditionalNeuralProcess, IncUpdateEff):
    def __init__(
        self,
        encoder: ConvCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
        self.likelihood.min_noise = 1e-5 # Adds little noise here because sometimes scale is exactly 0 - check to ensure this is small enough to not impact other perf - hacky

    # Effecient incremental updates should only be used for hadIsd where this results in measurable speedup
    def init_inc_structs(self, m: int, max_nc: int, device: str, use_flash: bool=False, cache_mhca: bool=False, persist_small: bool=False):
        assert not persist_small, "Persist small not implemented her eyet"
        self.inc_cache = {} # This is an empty cache structure used soley for storing incremental update objects
        # Creates x grid with correct dimensionaility
        grid = self.encoder.grid_encoder.grid
        grid_str = " ".join([f"n{i}" for i in range(grid.ndim - 1)])
        self.inc_cache["x_grid"] = einops.repeat(grid, f"{grid_str} d -> m {grid_str} d", m=m).to(device)
        
        self.inc_cache["z_grid"] = None # Inits on first call to context update

    # Adds new context points
    def update_ctx(self, xc: torch.Tensor, yc: torch.Tensor, use_flash: bool=False, cache_mhca: bool=False, persist_small: bool=False):
        assert not persist_small, "Persist small not implemented her eyet"
        if not self.encoder.hadisd_mode:
            flag = torch.ones_like(yc[..., :1])
            z_feats = torch.cat((yc, flag), dim=-1) 
            # Init z grid for first call now that shape is known
            if self.inc_cache["z_grid"] is None:
                z_grid_shape = self.inc_cache["x_grid"].shape[:-1] + (z_feats.shape[-1],)
                self.inc_cache["z_grid"] = torch.zeros(z_grid_shape, device=xc.device)
            
            # Adds points to set conv grid
            self.inc_cache["z_grid"] = setconv_to_grid(xc, z_feats, self.inc_cache["x_grid"], 
                self.encoder.grid_encoder.lengthscale, z_grid=self.inc_cache["z_grid"], dist_fn=self.encoder.grid_encoder.dist_fn)
        else:
            # Builds hadISD feature vector
            flag = torch.ones_like(yc[..., :1])
            elev = xc[..., 2:3]
            time = xc[..., 3:4]
            # Passes these features into fourier embedder
            elev_time_vec = torch.cat((elev, time), dim=-1)
            elev_time_fourier = self.encoder.fourier_encoder(elev_time_vec)
            
            z_feats = torch.cat((yc, flag, elev_time_fourier), dim=-1) 

            xc_coords = xc[..., :2] # Coords we incrementally add to set conv grid (lat and lon)

            # Init z grid for first call now that shape is known
            if self.inc_cache["z_grid"] is None:
                z_grid_shape = self.inc_cache["x_grid"].shape[:-1] + (z_feats.shape[-1],)
                self.inc_cache["z_grid"] = torch.zeros(z_grid_shape, device=xc.device)
            
            # Adds points to set conv grid
            self.inc_cache["z_grid"] = setconv_to_grid(xc_coords, z_feats, self.inc_cache["x_grid"], 
                self.encoder.grid_encoder.lengthscale, z_grid=self.inc_cache["z_grid"], dist_fn=self.encoder.grid_encoder.dist_fn)

    def query(self, xt: torch.Tensor, dy: int, use_flash: bool=False, cache_mhca: bool=False, persist_small: bool=False) -> td.Normal:
        assert not persist_small, "Persist small not implemented her eyet"
        if not self.encoder.hadisd_mode:
            xt_coords = xt
            z_grid = self.encoder.z_encoder(self.inc_cache["z_grid"])
            z_grid = self.encoder.conv_net(z_grid)
            zt = self.encoder.grid_decoder(self.inc_cache["x_grid"], z_grid, xt_coords)
            return self.likelihood(self.decoder(zt, xt))
        else:
            # Runs CNN and z encoder as before
            xt_coords = xt[..., :2]
            z_grid = self.encoder.z_encoder(self.inc_cache["z_grid"])
            z_grid = self.encoder.conv_net(z_grid)
            zt = self.encoder.grid_decoder(self.inc_cache["x_grid"], z_grid, xt_coords)
            return self.likelihood(self.decoder(zt, xt))
        
    def repeat_ctx(self, repeat_times: int, persist_small: bool=False):
        assert not persist_small, "Persist small not implemented her eyet"
        if self.inc_cache["z_grid"] is None: 
            raise RuntimeError("Cache empty. Run update_ctx before expanding.")
        self.inc_cache["z_grid"] = self.inc_cache["z_grid"].repeat_interleave(repeat_times, dim=0).contiguous()
        self.inc_cache["x_grid"] = self.inc_cache["x_grid"].repeat_interleave(repeat_times, dim=0).contiguous()


class GriddedConvCNP(nn.Module):
    def __init__(
        self,
        encoder: GriddedConvCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood

    @check_shapes("mc: [m, ...]", "y: [m, ..., dy]", "mt: [m, ...]")
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(mc, y, mt)))
