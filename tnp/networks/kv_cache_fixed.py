# KV cache helper but using fixed sized cache to optimise for cases when this is known.
# Also stores masks and context reps. This is lower level but faster (esp for small cases) - use with care!
from typing import Optional
import torch

_CTX_REP_ID = "CTXTensor"
_SA_REP_ID = "SaLayer"
_MASK_ID = "CachedCausalMask"

# Updates key and value pairs
def update_kv_cache_fixed(k_new, v_new, cache: Optional[dict], cache_id):
    if cache is None: return k_new, v_new # Training - in case of an empty cache, k and v are simply returned

    m, h, new_pts, k_dim = k_new.shape 
    _, _, _, v_dim = v_new.shape
    k, v, write_idx = cache[cache_id]
    total = write_idx + new_pts
    k[:,:,write_idx:total,:].copy_(k_new)
    v[:,:,write_idx:total,:].copy_(v_new)
    cache[cache_id] = (k, v, total)

    return k[:, :,:total,:], v[:,:,:total,:] # some reason this can be slow and not work with high level kernels (flash maybe because of non contig view of 4d tensor?)

def get_ctx_id(layer: int) -> str:
    return _CTX_REP_ID + str(layer)

def get_layer_id(layer: int) -> str:
    return _SA_REP_ID + str(layer)

def get_layer_ctx(layer: int, cache):
    return cache[get_ctx_id(layer)][0]

# Updates context rep stored
def update_ctx_cache_fixed(zc_new, cache, l):
    new_pts = zc_new.shape[1]
    l_id = get_ctx_id(l)
    zc, write_idx = cache[l_id]
    zc[:, write_idx: write_idx+new_pts, :] = zc_new
    write_idx += new_pts
    cache[l_id] = (zc, write_idx)

def get_mask_fixed(cache, q_len, k_len):

    return (cache[_MASK_ID])[-q_len:, :k_len]


# Initialises a KV cache
def init_kv_cache_fixed(layers: int, batch_size: int, max_nc: int, dz: int, heads: int, k_dim: int, v_dim: int,
     device: str) -> dict:
    kv_cache = {} # Empty cache
    # Initialises context representation of size [m, nc, dz] for each layer
    for l in range(layers):
        kv_cache[get_ctx_id(l)] = (torch.empty((batch_size, max_nc, dz), device=device), 0)
    # Intialises a causal mask to use that can be sliced at run time
    kv_cache[_MASK_ID] = torch.tril(torch.ones(max_nc, max_nc, dtype=torch.bool, device=device))
    # Initialises K and V tensors to be cached
    for l in range(layers):
        kv_cache[get_layer_id(l)] = (torch.empty((batch_size, heads, max_nc, k_dim), device=device),
            torch.empty((batch_size, heads, max_nc, v_dim), device=device), 0)
    return kv_cache