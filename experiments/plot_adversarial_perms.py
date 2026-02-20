# Helper function used to load model weights
import copy
import glob
import os
import random
import time
from functools import partial
from typing import Callable, Optional, Tuple, Union

import hiyapyco
import lightning.pytorch as pl
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import torch
from check_shapes import check_shapes
from hydra.utils import instantiate
from torch import nn
import wandb
from tnp.utils.experiment_utils import deep_convert_dict, extract_config, unwrap_wandb_config
from tnp.utils.lightning_utils import LitWrapper
from tnp.utils.np_functions import np_pred_fn


# Loads model with config file and either W&B link or local ckpt file path.
def get_model(config_path, weights_and_bias_ref, device='cuda', seed: bool = True, instantiate_only_model: bool = False, load_mod_weights: bool = True, local_weights: bool = False, weights_only_evalhad_call: bool = False):
    raw_config = deep_convert_dict(
        hiyapyco.load(
            config_path,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    if local_weights: raw_config = unwrap_wandb_config(raw_config)

    # Initialise experiment, make path.
    config, _ = extract_config(raw_config, None)
    config = deep_convert_dict(config)

    # Instantiate experiment and load checkpoint.
    if seed: pl.seed_everything(config.misc.seed)
    if instantiate_only_model:
        experiment = instantiate(config.model)
        model = experiment
    else:
        experiment = instantiate(config)
        model = experiment.model
    experiment.config = config
    if seed: pl.seed_everything(experiment.misc.seed)

    # Loads weights and bias model
    if load_mod_weights:
        if local_weights:
            ckpt_file = weights_and_bias_ref
        else:
            artifact = wandb.Api().artifact(weights_and_bias_ref, type='model')
            artifact_dir = artifact.download()
            ckpt_file = os.path.join(artifact_dir, "model.ckpt")
        if weights_only_evalhad_call:
            pattern = os.path.join(artifact_dir, "*.ckpt")
            matching_files = glob.glob(pattern)
            assert len(matching_files) == 1, "must be exactly one ckpt file to load"
            ckpt_file = matching_files[0]
            lit_model = (
                LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                    ckpt_file, model=model, weights_only=False,
                )
            )
        else:
            lit_model = (
                LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                    ckpt_file, model=model, weights_only=False,
                )
            )
        model = lit_model.model
    model.to(device)
    return model
