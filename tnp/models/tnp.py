from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import ISTEncoder, PerceiverEncoder, TNPTransformerEncoder
from ..utils.helpers import preprocess_observations
from .base import ConditionalNeuralProcess


class TNPDecoder(nn.Module):
    def __init__(
        self,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes("z: [m, ..., n, dz]", "xt: [m, nt, dx]", "return: [m, ..., nt, dy]")
    def forward(
        self, z: torch.Tensor, xt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if xt is not None:
            zt = z[..., -xt.shape[-2] :, :]
        else:
            zt = z
        return self.z_decoder(zt)


class TNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TNPTransformerEncoder, PerceiverEncoder, ISTEncoder],
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
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        zt = self.transformer_encoder(zc, zt)
        return zt


class TNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: TNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
