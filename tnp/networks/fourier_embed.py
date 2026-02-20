# Embedds fourier data for atmosphere - based off of embeddings from  https://arxiv.org/pdf/2405.13063v1
import torch
from torch import nn
from check_shapes import check_shapes
from typing import List, Tuple


# A fourier embedder
class FourierEmbedderHadISD(nn.Module):
    def __init__(
        self,
        embed_dim_lambdamin_lambda_max: List[Tuple[int, float , float]]
    ):
        super().__init__()
        
        self.embed_dim_lambdamin_lambda_max = embed_dim_lambdamin_lambda_max

        # Precomputes wavelength grid
        grids = []
        for D, lambda_min, lambda_max in self.embed_dim_lambdamin_lambda_max:
            assert D % 2 == 0, "Embedding dimension must be even"
            i = torch.arange(D // 2, dtype=torch.float64)
            l_min = torch.log10(torch.tensor(lambda_min, dtype=torch.float64))
            l_max = torch.log10(torch.tensor(lambda_max, dtype=torch.float64))
            log_l_i = l_min + i * (l_max - l_min) / (D // 2 - 1)
            grids.append(torch.exp(log_l_i))
        wave_grid = torch.cat(grids)
        self.register_buffer("wave_grid", wave_grid)

        # Stores slice for each feature in
        self.slices = []
        start = 0
        for D, _, _ in self.embed_dim_lambdamin_lambda_max:
            self.slices.append(slice(start, start + D // 2))
            start += D // 2

    @check_shapes(
        "x: [m, n, dx]", "return: [m, n, de]"
    )
    def forward(self, x: torch.Tensor):
        m, n, dx = x.shape
        assert dx == len(self.embed_dim_lambdamin_lambda_max), "Mismatch between embed list and feature length"

        embeds = []
        for i in range(dx):
            lambdas = self.wave_grid[self.slices[i]].view(1, 1, -1) # [1, 1, D / 2]
            angle = 2 * torch.pi * x[:, :, i:i+1] / lambdas
            emb_x = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
            embeds.append(emb_x)
        return torch.cat(embeds, dim=-1).to(dtype=x.dtype) # Casts back to lower datatype if needed