from typing import Tuple

import torch
from check_shapes import check_shapes


@check_shapes("xt: [m, nt, dx]", "yc: [m, nc, dy]")
def preprocess_observations(
    xt: torch.Tensor,
    yc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    yt = torch.zeros(xt.shape[:-1] + yc.shape[-1:]).to(yc)
    yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,)).to(yc)), dim=-1)
    yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,)).to(yt)), dim=-1)

    return yc, yt
