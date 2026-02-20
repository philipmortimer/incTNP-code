import torch
from torch import nn

from ..data.base import Batch, ImageBatch
from ..models.base import (
    ARConditionalNeuralProcess,
    ConditionalNeuralProcess,
    LatentNeuralProcess,
    ARTNPNeuralProcess,
    BatchedCausalTNP,
    BatchedCausalTNPPrior,
)
from ..models.convcnp import GriddedConvCNP

# Tries to import wiski model
try:
    from ..models.wiskigp import WiskiGP
    wiskigp_available = True
except ImportError:
    wiskigp_available = False


def np_pred_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
    predict_without_yt_tnpa: bool = False, # Used for tnpa to allow teacher forcing by default but support predictions without access to yt
) -> torch.distributions.Distribution:
    unwrapped_model = getattr(model, '_orig_mod', model) # If model is compiled with torch.compile needs to be unwrapped

    if isinstance(unwrapped_model, GriddedConvCNP):
        assert isinstance(batch, ImageBatch)
        pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
    elif isinstance(unwrapped_model, ConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    elif isinstance(unwrapped_model, LatentNeuralProcess):
        pred_dist = model(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
        )
    elif isinstance(unwrapped_model, ARConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    elif isinstance(unwrapped_model, ARTNPNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt, predict_without_yt_tnpa=predict_without_yt_tnpa)
    elif isinstance(unwrapped_model, BatchedCausalTNP):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    elif isinstance(unwrapped_model, BatchedCausalTNPPrior):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    elif wiskigp_available and isinstance(unwrapped_model, WiskiGP):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    #elif isinstance(unwrapped_model, GPStreamSparseWrapperRBF):
    #    pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    else:
        raise ValueError

    return pred_dist


def np_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """Perform a single training step, returning the loss, i.e.
    the negative log likelihood.

    Arguments:
        model: model to train.
        batch: batch of data.

    Returns:
        loss: average negative log likelihood.
    """
    pred_dist = np_pred_fn(model, batch, num_samples) # Normal dist
    # BatchedCausalTNP uses different loss factorisation - loss is computed by weighting loss for each point AR style
    if isinstance(model, BatchedCausalTNP):
        x = torch.cat((batch.xc, batch.xt), dim=1)
        y = torch.cat((batch.yc, batch.yt), dim=1)
        m, N, dy = y.shape
        log_p = pred_dist.log_prob(y[:, 1:, :]) # [m, N - 1, dy] - zero context point not included
        log_p = log_p.mean(-1) # [m, N - 1] - takes mean over dy dimension
        mask = torch.ones((m, N - 1), device=log_p.device) # Can use this specify what lengths we care about / don't (and how much)
        # We average over all valid tokens in the calculation (and normalise by length)
        nll = - (log_p * mask).sum() / mask.sum()
    elif isinstance(model, BatchedCausalTNPPrior):
        # Similar loss to BatchedCausalTNP but incoproating a loss from zero shot example
        x = torch.cat((batch.xc, batch.xt), dim=1)
        y = torch.cat((batch.yc, batch.yt), dim=1)
        m, N, dy = y.shape
        log_p = pred_dist.log_prob(y) # [m, N, dy]
        log_p = log_p.mean(-1) # [m, N] - takes mean over dy dimension
        mask = torch.ones((m, N), device=log_p.device) # Can use this specify what lengths we care about / don't (and how much) - may want to wait prior lower but is probably important for greedy order selection
        # We average over all valid tokens in the calculation (and normalise by length)
        nll = - (log_p * mask).sum() / mask.sum()
    else:
        nll = -(pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel())
    return nll
