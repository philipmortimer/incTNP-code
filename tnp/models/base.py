from abc import ABC

import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional, Union

from ..likelihoods.base import UniformMixtureLikelihood


class BaseNeuralProcess(nn.Module, ABC):
    """Represents a neural process base class"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood


class ConditionalNeuralProcess(BaseNeuralProcess):
    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt), xt))


class LatentNeuralProcess(BaseNeuralProcess):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, likelihood: nn.Module):
        likelihood = UniformMixtureLikelihood(likelihood)
        super().__init__(encoder, decoder, likelihood)

    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, num_samples: int = 1
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt, num_samples), xt))


class ARConditionalNeuralProcess(BaseNeuralProcess):
    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "yt: [m, nt_, dy]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: torch.Tensor,
    ) -> torch.distributions.Distribution:
        if self.training:
            # Train in AR mode.
            return self.likelihood(self.decoder(self.encoder(xc, yc, xt, yt), xt))

        # Test in normal mode.
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt), xt))

# Used specifically for tnpa only at the moment
# Used specifically for tnpa only at the moment
class ARTNPNeuralProcess(BaseNeuralProcess):
    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "yt: [m, nt_, dy]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: torch.Tensor,
        predict_without_yt_tnpa: bool,
    ) -> torch.distributions.Distribution:
        if predict_without_yt_tnpa:
            # Uses a monte carlo sampled distribution (unrolling over multiple samples)
            return self.predictive_distribution_monte_carlo(xc, yc, xt)
        else:
            # Uses teach forcing setup
            return self.likelihood(self.decoder(self.encoder(xc, yc, xt, yt), xt))

# Used for the batched causal TNP
class BatchedCausalTNP(BaseNeuralProcess):
    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "yt: [m, nt_, dy]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: torch.Tensor,
    ) -> torch.distributions.Distribution:
        # AR style training
        if self.training:
            assert xt.shape[1] == yt.shape[1], "xt and yt must both be same length when training"
            # Consider permuting this in future?
            x = torch.cat((xc, xt), dim=1)
            y = torch.cat((yc, yt), dim=1)
            # Decoder doesnt look at zero shot context (first point)
            return self.likelihood(self.decoder(self.encoder(x=x, y=y)))
        else: # Inference time
            return self.likelihood(self.decoder(self.encoder(xc=xc, yc=yc, xt=xt), xt))

# Used for the batched causal TNP with prior - so can handle an empty context set but seperated for different loss function
class BatchedCausalTNPPrior(BaseNeuralProcess):
    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Distribution:
        if yt is not None: assert yt.shape[0] == xc.shape[0] and yt.shape[2] == yc.shape[2], "Invalid yt shape"
        if self.order_ctx_greedy != "False":
            print("greedy ordering")
            policy = self.order_ctx_greedy.split("-")[0]
            var = self.order_ctx_greedy.split("-")[1]
            xc, yc = self.kv_cached_greedy_variance_ctx_builder(xc=xc, yc=yc, policy=policy, select=var)
        # AR style training
        if self.training:
            assert yt is not None and xt.shape[1] == yt.shape[1], "xt and yt must both be same length when training"
            # Consider permuting this in future?
            x = torch.cat((xc, xt), dim=1)
            y = torch.cat((yc, yt), dim=1)
            return self.likelihood(self.decoder(self.encoder(x=x, y=y)))
        else: # Inference time
            return self.likelihood(self.decoder(self.encoder(xc=xc, yc=yc, xt=xt), xt))
        
