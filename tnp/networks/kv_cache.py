# Helper to make KV cache updates easy
from typing import Optional
import torch

# Updates key and value pairs
def update_kv_cache(k_new, v_new, cache: Optional[dict], cache_id):
    if cache is None: return k_new, v_new # Training - in case of an empty cache, k and v are simply returned

    k, v = cache.get(cache_id, (None, None)) # Gets previously cached k and v values (or (None, None) if layer has not been cached yet)

    if k is not None:
        # Adds k and v to cache history
        k_new = torch.cat((k, k_new), dim=2)
        v_new = torch.cat((v, v_new), dim=2)
    cache[cache_id] = (k_new, v_new)
    return k_new, v_new

# Updates context rep stored
def update_ctx_cache(zc_new, cache, cache_id):
    zc_old = cache.get(cache_id, None)

    # Adds new representation
    if zc_old is not None:
        zc_new = torch.cat((zc_old, zc_new), dim=1)
    cache[cache_id] = zc_new



# Initialises a KV cache
def init_kv_cache() -> dict:
    kv_cache = {} # Empty cache
    return kv_cache

# Repeats the kv values stored by repeat_times. Used when a shared computation needs to be recylced for AR unrolls
def repeat_kv_cache_batch(cache_in, cache_out, repeat_times:int):
    for key, val in cache_in.items():
        if isinstance(val, tuple) and len(val) == 2:
            k, v = val
            k_new = k.repeat_interleave(repeat_times, dim=0)
            v_new = v.repeat_interleave(repeat_times, dim=0)
            #k_new = k_new.contiguous()
            #v_new = v_new.contiguous()
            cache_out[key] = (k_new, v_new)
        elif isinstance(val, torch.Tensor):
            val_new = val.repeat_interleave(repeat_times, dim=0)
            #val_new = val_new.contiguous()
            cache_out[key] = val_new