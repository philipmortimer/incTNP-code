# Autoregressive TNP - using the outline from  (https://arxiv.org/pdf/2207.04179) https://github.com/tung-nd/TNP-pytorch
from typing import Optional, Union

from tnp.networks.kv_cache import init_kv_cache
import torch
from check_shapes import check_shapes
from torch import nn
import torch.distributions as td

from .tnp import TNPDecoder
from ..utils.helpers import preprocess_observations
from ..networks.transformer import TransformerEncoder
from .base import ARTNPNeuralProcess

from ..likelihoods.gaussian import HeteroscedasticNormalLikelihood



class ARTNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TransformerEncoder],
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
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor
    ) -> torch.Tensor:
        m = xc.shape[0]
        # Preprocesses observations
        yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,)).to(yc)), dim=-1)
        yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,)).to(yt)), dim=-1)

        # Encodes x and y
        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)
        
        # Concats ctx with fake and real target points
        inp = self._construct_input(xc_encoded, yc_encoded, xt_encoded, yt_encoded)
        mask, num_tar = self._create_mask(num_ctx=xc.shape[1], num_tar=xt.shape[1], device=xt.device)
       # mask = mask.unsqueeze(0).expand(m, -1, -1) # [m, nc + 2*nt, nc+2*nt] Broadcast mask to batch

        # Embeds data and runs through transformer encoder
        embeddings = self.xy_encoder(inp)
        out = self.transformer_encoder(embeddings, mask=mask)

        return out[:, -num_tar:]


    def _construct_input(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor):
        x_y_ctx = torch.cat((xc, yc), dim=-1)
        x_0_tar = torch.cat((xt, torch.zeros_like(yt)), dim=-1)

        x_y_tar = torch.cat((xt, yt), dim=-1) # Note currently no support for bound_std but may want to add (think this is handled by the lilkelihood dist)
        #if self.training and bound_std:
        #    yt_noise = yt + 0.05 * torch.randn_like(yt) # add noise to the past to smooth the model
        #    x_y_tar = torch.cat((xt, yt_noise), dim=-1)
        #else:
        #    x_y_tar = torch.cat((xt, yt), dim=-1)
        inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1) # [m, nc + 2*nt, dx + dy + 1] (probably - depends on encoders etc)
        return inp

    def _create_mask(self, num_ctx, num_tar, device):
        num_all = num_ctx + num_tar
        mask = torch.zeros((num_all+num_tar, num_all+num_tar), device=device).fill_(float('-inf'))
        mask[:, :num_ctx] = 0.0 # all points attend to context points
        mask[num_ctx:num_all, num_ctx:num_all].triu_(diagonal=1) # each real target point attends to itself and precedding real target points
        mask[num_all:, num_ctx:num_all].triu_(diagonal=0) # each fake target point attends to preceeding real target points

        return mask, num_tar
    
    # KV Caching unroll helpers
    def _get_cache_len(self, kv_cache):
        if not kv_cache: return 0
        first_key = next(iter(kv_cache))
        return kv_cache[first_key][0].shape[2]

    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]")
    def forward_ctx_cache(self, xc, yc, kv_cache):
        yc_flag = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,)).to(yc)), dim=-1)
        xc_enc = self.x_encoder(xc)
        yc_enc = self.y_encoder(yc_flag)
        inp = torch.cat((xc_enc, yc_enc), dim=-1)
        embeddings = self.xy_encoder(inp)

        self.cached_y_enc_dim = yc_enc.shape[-1]

        nc = embeddings.shape[1]
        full_mask = torch.ones((nc, nc), dtype=torch.bool, device=xc.device)
        
        self.transformer_encoder(embeddings, mask=full_mask, kv_cache=kv_cache, tnpa_kv=True)

    @check_shapes("xt: [m, 1, dx]", "return: [m, 1, dz]")
    def forward_query_fake(self, xt, kv_cache):
        xt_enc = self.x_encoder(xt)
        
        if not hasattr(self, 'cached_y_enc_dim'):
            raise RuntimeError("Run forward_ctx_cache first.")
            
        y_zeros = torch.zeros((xt.shape[0], 1, self.cached_y_enc_dim), device=xt.device, dtype=xt_enc.dtype)
        
        inp = torch.cat((xt_enc, y_zeros), dim=-1)
        embeddings = self.xy_encoder(inp)
        
        current_cache_len = self._get_cache_len(kv_cache)
        total_len = current_cache_len + 1
        
        mask = torch.ones((1, total_len), dtype=torch.bool, device=xt.device)
        mask[0, -1] = False
        #mask = torch.zeros((1, total_len), dtype=xt.dtype, device=xt.device)
        #mask[0, -1] = float('-inf')
        
        out = self.transformer_encoder(embeddings, mask=mask, kv_cache=kv_cache, tnpa_kv=True)
        return out

    @check_shapes("xt: [m, 1, dx]", "yt_real: [m, 1, dy]")
    def forward_append_real(self, xt, yt_real, kv_cache):
        flag = torch.ones((xt.shape[0], 1, 1), device=xt.device, dtype=xt.dtype)
        yt_flag = torch.cat((yt_real, flag), dim=-1)
        
        xt_enc = self.x_encoder(xt)
        yt_enc = self.y_encoder(yt_flag)
        inp = torch.cat((xt_enc, yt_enc), dim=-1)
        embeddings = self.xy_encoder(inp)
        
        self.transformer_encoder(embeddings, mask=None, kv_cache=kv_cache, tnpa_kv=True)

class TNPA(ARTNPNeuralProcess):
    def __init__(
        self,
        encoder: ARTNPEncoder,
        decoder: TNPDecoder,
        likelihood: Union[HeteroscedasticNormalLikelihood],
        permute: bool = True,
        no_samples_rollout_pred: int = 50 # The number of samples to be used when doing predictive rollout. Note could change on fly if wanted
    ):
        super().__init__(encoder, decoder, likelihood)

        self.permute= permute
        self.num_samples = no_samples_rollout_pred
        self.rollout_mode = "normal" # Experimental flag to try different modes

    @check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
    )
    def _predict(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, num_samples) -> td.Normal:
        batch_size = xc.shape[0]
        num_target = xt.shape[1]
        assert self.rollout_mode in ("cache", "fast", "normal"), "TNP-A rollout must be a valid pre defined mode"
        def squeeze(x):
            return x.view(-1, x.shape[-2], x.shape[-1])
        def unsqueeze(x):
            return x.view(num_samples, batch_size, x.shape[-2], x.shape[-1])

        xc_stacked = self._stack_tnpapaper(xc, num_samples)
        yc_stacked = self._stack_tnpapaper(yc, num_samples)
        xt_stacked = self._stack_tnpapaper(xt, num_samples)
        yt_pred = torch.zeros((batch_size, num_target, yc.shape[2]), device=xt.device)
        yt_stacked = self._stack_tnpapaper(yt_pred, num_samples)
        if self.permute:
            xt_stacked, yt_stacked, dim_sample, dim_batch, deperm_ids = self._permute_sample_batch(xt_stacked, yt_stacked, num_samples, batch_size, num_target)

        batch_xc = squeeze(xc_stacked) # [m * num_samples, nc, dx]
        batch_yc = squeeze(yc_stacked)
        batch_xt = squeeze(xt_stacked)
        batch_yt = squeeze(yt_stacked)

        mean_buffer = unsqueeze(torch.zeros_like(batch_yt))
        std_buffer = unsqueeze(torch.zeros_like(batch_yt))


        if self.rollout_mode == "cache":
            kv_cache = init_kv_cache()
            
            # 1. Prefill Context (Bidirectional)
            self.encoder.forward_ctx_cache(batch_xc, batch_yc, kv_cache)
            
            for step in range(xt.shape[1]):
                curr_xt = batch_xt[:, step:step+1]
                
                z_step = self.encoder.forward_query_fake(curr_xt, kv_cache)

                out = self.likelihood(self.decoder(z_step))
                sample = out.sample()
                
                for key in list(kv_cache.keys()):
                    val = kv_cache[key]
                    if isinstance(val, tuple) and len(val) == 2:
                        k, v = val
                        k_new = k[:, :, :-1, :]
                        v_new = v[:, :, :-1, :]
                        kv_cache[key] = (k_new, v_new)
                
                self.encoder.forward_append_real(curr_xt, sample, kv_cache)
                batch_yt[:, step:step+1] = sample 
                mean, std = unsqueeze(out.mean), unsqueeze(out.stddev)
                mean_buffer[:, :, step] = mean[:, :, 0]
                std_buffer[:, :, step] = std[:, :, 0]
        else:
            # Old code based on Nguyen and Grover paper and optimised a bit in places
            for step in range(xt.shape[1]):
                if self.rollout_mode == "fast":
                    curr_xt = batch_xt[:, :step+1]
                    curr_yt = batch_yt[:, :step+1]
                    z_target_stacked = self.encoder(batch_xc, batch_yc, curr_xt, curr_yt)
                    z_target_stacked = z_target_stacked[:, -1:]
                    sample_idx = 0
                else:
                    z_target_stacked = self.encoder(batch_xc, batch_yc, batch_xt, batch_yt) # [m * num_samples, nt, dz]
                    z_target_stacked = z_target_stacked[..., -xt.shape[-2] :, :] # [m * num_samples, nt, dz] - appears to do nothing
                    sample_idx = step
                out = self.decoder(z_target_stacked) # [m * num_samples, nt, 2 * dy]
                out = self.likelihood(out)

                assert isinstance(out, td.Normal), "TNPAR must predict a Gaussian"
                mean, std = out.mean, out.stddev
                mean, std = unsqueeze(mean), unsqueeze(std)
                batch_yt = unsqueeze(batch_yt)

                batch_yt[:, :, step] = td.Normal(mean[:, :, sample_idx], std[:, :, sample_idx]).sample()

                mean_buffer[:, :, step] = mean[:, :, sample_idx]
                std_buffer[:, :, step] = std[:, :, sample_idx]

                batch_yt = squeeze(batch_yt)
                # Note currently no support for bound_std but may want to consider in future

        if self.permute:
            mean_buffer, std_buffer = mean_buffer[dim_sample, dim_batch, deperm_ids], std_buffer[dim_sample, dim_batch, deperm_ids]

        return td.Normal(mean_buffer, std_buffer)

    # Unrolls monte carlo estimate of a predictive distribution into a mixture. Can use this to access mean and std dev e.g. for plotting
    @check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
    )
    def predictive_distribution_monte_carlo(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor) -> td.MixtureSameFamily:
        dist = self._predict(xc, yc, xt, self.num_samples)
        mean, std = dist.mean, dist.stddev # mean and std both have shape [s, m, nt, dy] where m = 1 probably
        s, m, nt, dy = mean.shape
        # Reorders to [m, nt, dy, s] - needed for mixture
        mean = mean.permute(1,2,3,0)
        std = std.permute(1,2,3,0)
        
        mix = td.Categorical(torch.full((m, nt, dy, s), 1.0 / s, device=xt.device))
        comp = td.Normal(mean, std)
        approx_dist = td.MixtureSameFamily(mix, comp)
        return approx_dist
        
        

    def _stack_tnpapaper(self, x, num_samples=None, dim=0):
        return x if num_samples is None \
                else torch.stack([x]*num_samples, dim=dim)

    def _permute_sample_batch(self, xt, yt, num_samples, batch_size, num_target):
        # data in each batch is permuted identically
        perm_ids = torch.rand(num_samples, num_target, device=xt.device).unsqueeze(1).repeat((1, batch_size, 1))
        perm_ids = torch.argsort(perm_ids, dim=-1)
        deperm_ids = torch.argsort(perm_ids, dim=-1)
        dim_sample = torch.arange(num_samples, device=xt.device).unsqueeze(-1).unsqueeze(-1).repeat((1,batch_size,num_target))
        dim_batch = torch.arange(batch_size, device=xt.device).unsqueeze(0).unsqueeze(-1).repeat((num_samples,1,num_target))
        return xt[dim_sample, dim_batch, perm_ids], yt[dim_sample, dim_batch, perm_ids], dim_sample, dim_batch, deperm_ids