import argparse
import itertools
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
    SequentialLR,
)

import wandb

from .lightning_utils import LitWrapper


def create_default_config() -> DictConfig:
    default_config = {
        "misc": {
            "resume_from_checkpoint": None,
            "logging": True,
            "seed": 0,
            "plot_interval": 1,
            "lightning_eval": True,
            "num_plots": 5,
            "gradient_clip_val": 0.5,
            "only_plots": False,
            "savefig": False,
            "subplots": True,
            "loss_fn": {
                "_target_": "tnp.utils.np_functions.np_loss_fn",
                "_partial_": True,
            },
            "pred_fn": {
                "_target_": "tnp.utils.np_functions.np_pred_fn",
                "_partial_": True,
            },
            "num_workers": 1,
            "num_val_workers": 1,
            "log_interval": 10,
            "checkpoint_interval": 1,
            "check_val_every_n_epoch": 1,
        }
    }
    return OmegaConf.create(default_config)


def _product_resolver(target: str, param_dict: Dict[str, List[Any]]) -> List[Dict]:
    """Resolver for itertools.product that creates object configs with named arguments.

    Args:
        target: The _target_ path for object initialization
        param_dict: Dictionary of argument names to lists of values
    """
    # Get the names and value lists
    names = list(param_dict.keys())
    value_lists = [
        (
            param_dict[name]
            if isinstance(param_dict[name], Iterable)
            else [param_dict[name]]
        )
        for name in names
    ]

    # Create a list of object configs
    return [
        {"_target_": target, **dict(zip(names, combo))}
        for combo in itertools.product(*value_lists)
    ]


def extract_config(
    raw_config: Union[str, Dict],
    config_changes: Optional[List[str]] = None,
    combine_default: bool = True,
) -> Tuple[DictConfig, Dict]:
    """Extract the config from the config file and the config changes.

    Arguments:
        config_file: path to the config file.
        config_changes: list of config changes.

    Returns:
        config: config object.
        config_dict: config dictionary.
    """
    # Register eval.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    # Register product.
    if not OmegaConf.has_resolver("product"):
        OmegaConf.register_new_resolver("product", _product_resolver)

    if isinstance(raw_config, str):
        config = OmegaConf.load(raw_config)
    else:
        config = OmegaConf.create(raw_config)

    if combine_default:
        default_config = create_default_config()
        config = OmegaConf.merge(default_config, config)

    config_changes = OmegaConf.from_cli(config_changes)
    config = OmegaConf.merge(config, config_changes)
    config_dict = OmegaConf.to_container(config, resolve=True)

    return config, config_dict


def deep_convert_dict(layer: Any):
    to_ret = layer
    if isinstance(layer, OrderedDict):
        to_ret = dict(layer)

    try:
        for key, value in to_ret.items():
            to_ret[key] = deep_convert_dict(value)
    except AttributeError:
        pass

    return to_ret


def initialize_experiment() -> DictConfig:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+")
    parser.add_argument("--learning_rate", type=float, default=None)
    args, config_changes = parser.parse_known_args()

    # Adds learning rate if its a passed argument
    if args.learning_rate is not None:
        print(f"Set learning rate to {args.learning_rate}")
        config_changes.append(f"optimiser.lr={args.learning_rate}")
    else:
        print("using template lr")

    raw_config = deep_convert_dict(
        hiyapyco.load(
            args.config,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    # Initialise experiment, make path.
    config, _ = extract_config(raw_config, config_changes)
    config = deep_convert_dict(config)

    # Instantiate experiment and load checkpoint.
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)
    experiment.config = config
    pl.seed_everything(experiment.misc.seed)

    return experiment

# Unwraaps local weights and bias yamls to be properly parsed
def unwrap_wandb_config(config_dict):
    if not isinstance(config_dict, dict):
        return config_dict
    
    new_config = {}
    for key, val in config_dict.items():
        if isinstance(val, dict) and 'value' in val and 'desc' in val: # Indicates yaml is download from W&B and needs to be unwrapped
            unwrapped_val = val['value']
            if isinstance(unwrapped_val, dict):
                new_config[key] = unwrap_wandb_config(unwrapped_val)
            else:
                new_config[key] = unwrapped_val
        elif isinstance(val, dict): # Recurse deeper if needed
            new_config[key] = unwrap_wandb_config(val)
        else:
            new_config[key] = val
    return new_config


def initialize_evaluation() -> DictConfig:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, help="e.g. user/project/run")
    parser.add_argument("--config", type=str, nargs="+", help="e.g. config.yaml")
    parser.add_argument("--checkpoint", type=str, help="e.g. users/project/artifact")
    args, config_changes = parser.parse_known_args()

    raw_config = deep_convert_dict(
        hiyapyco.load(
            args.config,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    # Initialise evaluation, make path.
    config, _ = extract_config(raw_config, config_changes)

    # Initialise wandb.
    api = wandb.Api()
    run = api.run(args.run_path)
    run = wandb.init(
        resume="allow",
        project=run.project,
        name=run.name,
        id=run.id,
    )

    # Set model to run.config.model.
    if hasattr(run.config, "model") and run.config.model is not None:
        config.model = run.config.model

    # Instantiate.
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)
    pl.seed_everything(config.misc.seed)

    # Load checkpoint from run artifact.
    artifact = run.use_artifact(args.checkpoint)
    artifact_dir = artifact.download()
    ckpt_file = os.path.join(artifact_dir, "model.ckpt")

    ckpt = torch.load(ckpt_file, map_location="cpu")
    print(f"Checkpoint epochs: {ckpt['epoch']}")

    # Load in the checkpoint.
    experiment.lit_model = LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=ckpt_file,
        map_location="cpu",
        strict=True,
        model=experiment.model,
    )

    return experiment


##############################################################################
# Learning rate scheduler utils
##############################################################################


def _calculate_total_steps(experiment: DictConfig, gen_train: Any) -> Optional[int]:
    """Calculates total training steps, handling potential config errors."""
    try:
        epochs = experiment.params.epochs
        # Assuming gen_train exposes num_batches correctly
        train_batches = gen_train.num_batches
        total_steps = epochs * train_batches

        print(
            f"Total training steps: {total_steps} (epochs={epochs}, batches_per_epoch={train_batches})"
        )
        return total_steps
    except AttributeError as e:
        print(
            f"Failed to calculate total steps (Missing config or generator property): {e}"
        )
        return None


def _get_warmup_steps(scheduler_config: DictConfig, total_steps: int) -> int:
    """Calculates warmup steps from 'steps' or 'fraction'."""
    if not hasattr(scheduler_config, "warmup"):
        return 0

    warmup_config = scheduler_config.warmup
    if hasattr(warmup_config, "steps") and warmup_config.steps is not None:
        return int(warmup_config.steps)
    if hasattr(warmup_config, "fraction") and warmup_config.fraction is not None:
        return int(total_steps * float(warmup_config.fraction))
    return 0


def _get_cosine_params(scheduler_config: DictConfig, total_steps: int) -> dict:
    """Gets T_max and eta_min for CosineAnnealingLR."""
    cosine_config = getattr(scheduler_config, "cosine", {})

    eta_min = getattr(cosine_config, "eta_min", 0.0)
    T_max = getattr(cosine_config, "T_max", total_steps)
    if T_max is None:
        T_max = total_steps

    return {"eta_min": eta_min, "T_max": T_max}


# --- Scheduler Creation Functions ---


def _create_cosine_scheduler(
    optimiser: torch.optim.Optimizer, scheduler_config: DictConfig, total_steps: int
) -> LRScheduler:
    """Creates a CosineAnnealingLR scheduler."""
    params = _get_cosine_params(scheduler_config, total_steps)
    scheduler = CosineAnnealingLR(
        optimiser,
        T_max=params["T_max"],
        eta_min=params["eta_min"],
    )
    print(
        f"Created cosine scheduler: T_max={params['T_max']}, eta_min={params['eta_min']}"
    )
    return scheduler


def _create_warmup_scheduler(
    optimiser: torch.optim.Optimizer, scheduler_config: DictConfig, total_steps: int
) -> Optional[LRScheduler]:
    """Creates a simple Warmup scheduler."""
    warmup_steps = _get_warmup_steps(scheduler_config, total_steps)

    if warmup_steps <= 0:
        print(
            "Warning: Warmup scheduler requested but warmup_steps <= 0. Using constant rate."
        )
        return None

    def warmup_lambda(current_step: int):
        return float(current_step) / float(max(1, warmup_steps))

    scheduler = LambdaLR(optimiser, warmup_lambda)
    print(f"Created warmup scheduler: warmup_steps={warmup_steps}")
    return scheduler


def _create_warmup_cosine_scheduler(
    optimiser: torch.optim.Optimizer, scheduler_config: DictConfig, total_steps: int
) -> LRScheduler:
    """Creates a Warmup followed by Cosine Annealing scheduler using SequentialLR."""
    warmup_steps = _get_warmup_steps(scheduler_config, total_steps)
    cosine_params = _get_cosine_params(scheduler_config, total_steps)
    T_max_original = cosine_params["T_max"]

    if warmup_steps <= 0 or T_max_original <= warmup_steps:
        # Fallback to cosine only or constant if T_max is too short
        print(
            "Warning: Warmup-Cosine requested but warmup_steps is invalid/too long. Using Cosine only."
        )
        return _create_cosine_scheduler(optimiser, scheduler_config, total_steps)

    # 1. Warmup Scheduler
    warmup_scheduler = LambdaLR(
        optimiser,
        lr_lambda=lambda step: float(step) / float(max(1, warmup_steps)),
    )

    # 2. Cosine Scheduler (T_max must be adjusted for the warmup period)
    cosine_T_max_adjusted = T_max_original - warmup_steps
    cosine_scheduler = CosineAnnealingLR(
        optimiser,
        T_max=cosine_T_max_adjusted,
        eta_min=cosine_params["eta_min"],
    )

    # 3. Combined Scheduler
    scheduler = SequentialLR(
        optimiser,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    print(
        f"Created Warmup+Cosine scheduler: Warmup={warmup_steps} steps, "
        f"Cosine T_max={cosine_T_max_adjusted}, eta_min={cosine_params['eta_min']}"
    )
    return scheduler


def create_lr_scheduler(
    optimiser: torch.optim.Optimizer, experiment: DictConfig, gen_train: Any
) -> Optional[LRScheduler]:
    """
    Creates a learning rate scheduler based on experiment configuration.
    """
    scheduler_config = getattr(experiment, "scheduler", None)

    if scheduler_config is None:
        print("No scheduler configuration found, using constant learning rate.")
        return None

    scheduler_type = getattr(scheduler_config, "type", "constant").lower()

    if scheduler_type == "constant":
        print("Using constant learning rate (no scheduler).")
        return None

    total_steps = _calculate_total_steps(experiment, gen_train)
    if total_steps is None:
        return None

    # Map scheduler types to their creation functions
    scheduler_factory = {
        "cosine": _create_cosine_scheduler,
        "warmup": _create_warmup_scheduler,
        "warmup_cosine": _create_warmup_cosine_scheduler,
    }

    try:
        # Dispatch to the appropriate creation function
        creator = scheduler_factory.get(scheduler_type)

        if creator is None:
            print(f"Unknown scheduler type: {scheduler_type}")
            return None

        # Execute the creation function
        return creator(optimiser, scheduler_config, total_steps)

    except Exception as e:
        # Catch exceptions during creation (e.g., invalid parameters)
        print(f"Failed to create scheduler of type '{scheduler_type}': {e}")
        return None
