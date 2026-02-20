# Helper functions for HadISD
from .base import Batch, DataGenerator
from .count_obs import cache_n_rows
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np
from pathlib import Path
import random
from bisect import bisect_left
import torch
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
import ast


# HadISD Batch - used to recover real values for things like plotting / correct scale
@dataclass
class HadISDBatch(Batch):
    mean_temp: float
    std_temp: float
    mean_elev: float
    std_elev: float
    unnormalised_time: torch.Tensor
    lat_range: Tuple[float, float]
    long_range: Tuple[float, float]
    ordering: str

def normalise_time(x):
    # Encodes time
    return x % (365 * 24) # Modulo to make year definitely not input  - though mostly unneeded 

# Gets true temp for given y
def get_true_temp(batch: HadISDBatch, y_in: torch.tensor):
    return y_in * batch.std_temp + batch.mean_temp

# Converts a temp pred dist to correct scale
def scale_pred_temp_dist(batch: HadISDBatch, pred_dist: torch.distributions.Normal):
    mean_scaled = get_true_temp(batch, pred_dist.mean)
    std_scaled = pred_dist.stddev * batch.std_temp
    return torch.distributions.Normal(loc=mean_scaled, scale=std_scaled)

# Gets correct scale elevation
def get_true_elev(batch: HadISDBatch, y_in: torch.tensor):
    return y_in * batch.std_elev + batch.mean_elev