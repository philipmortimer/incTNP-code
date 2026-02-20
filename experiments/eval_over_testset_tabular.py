# Eval script for tabular models on synthetic training task
import argparse
import itertools
import os
import re
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from plot_adversarial_perms import get_model
from tqdm import tqdm

import wandb
from tnp.data.base import Batch
from tnp.utils.np_functions import np_pred_fn


def get_epoch_number(artifact):
    """
    Parses 'model-epoch-0399-gj03lc2w:v0' and returns 399.
    Returns -1 if 'epoch-' is not found (e.g. if name is 'model-best').
    """
    # Look for the specific pattern "epoch-" followed by digits
    match = re.search(r"epoch-(\d+)", artifact.name)

    if match:
        return int(match.group(1))

    # If the artifact is named "model-best" or doesn't have an epoch,
    # treat it as -1 so it is never selected as the "last" one.
    return -1


def shuffle_batch(model, batch, shuffle_strategy: str, device: str = "cuda"):
    assert shuffle_strategy in {
        "random",
        "GreedyBestPriorLogP",
        "GreedyWorstPriorLogP",
        "GreedyMedianPriorLogP",
        "GreedyBestPriorVar",
        "GreedyWorstPriorVar",
        "GreedyMedianPriorVar",
    }, "Invalid context shuffle strategy"
    m, nc, dx = batch.xc.shape
    _, nt, dy = batch.yt.shape
    # Converts batch to cuda
    # batch.xc, batch.yc, batch.xt, batch.yt, batch.x, batch.y = batch.xc.to(device), batch.yc.to(device), batch.xt.to(device), batch.yt.to(device), batch.x.to(device), batch.y.to(device)
    xc_new, yc_new = None, None
    if shuffle_strategy == "random":
        perms = torch.rand(m, nc, device=batch.xc.device).argsort(dim=1)
        perm_x = perms.unsqueeze(-1).expand(-1, -1, dx)
        perm_y = perms.unsqueeze(-1).expand(-1, -1, dy)
        xc_new = torch.gather(batch.xc, 1, perm_x)
        yc_new = torch.gather(batch.yc, 1, perm_y)
    elif shuffle_strategy == "GreedyBestPriorLogP":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(
            batch.xc, batch.yc, policy="best", select="logp"
        )
    elif shuffle_strategy == "GreedyWorstPriorLogP":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(
            batch.xc, batch.yc, policy="worst", select="logp"
        )
    elif shuffle_strategy == "GreedyMedianPriorLogP":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(
            batch.xc, batch.yc, policy="median", select="logp"
        )
    elif shuffle_strategy == "GreedyBestPriorVar":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(
            batch.xc, batch.yc, policy="best", select="var"
        )
    elif shuffle_strategy == "GreedyWorstPriorVar":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(
            batch.xc, batch.yc, policy="worst", select="var"
        )
    elif shuffle_strategy == "GreedyMedianPriorVar":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(
            batch.xc, batch.yc, policy="median", select="var"
        )

    x = torch.cat((xc_new, batch.xt), dim=1)
    y = torch.cat((yc_new, batch.yt), dim=1)
    batch_new = Batch(xc=xc_new, yc=yc_new, xt=batch.xt, yt=batch.yt, y=y, x=x)
    if hasattr(batch, "y_mean"):
        batch_new.y_mean = batch.y_mean
    if hasattr(batch, "y_std"):
        batch_new.y_std = batch.y_std
    return batch_new


def denormalize_predictions(pred_dist, batch):
    """Denormalize predictions back to original scale."""
    if hasattr(batch, "y_mean") and hasattr(batch, "y_std"):
        # Create new distribution with denormalized parameters
        from torch.distributions import Normal

        # Denormalize mean and stddev
        denorm_mean = pred_dist.mean * batch.y_std + batch.y_mean
        denorm_stddev = pred_dist.stddev * batch.y_std

        # Return new distribution with denormalized values
        return Normal(denorm_mean, denorm_stddev)
    return pred_dist


def denormalize_targets(yt, batch):
    """Denormalize targets back to original scale."""
    if hasattr(batch, "y_mean") and hasattr(batch, "y_std"):
        return yt * batch.y_std + batch.y_mean
    return yt


# Evaluates a given models performance. Includes the option for a small number of defined strategies.
@torch.no_grad
def eval_model(model, test_set, shuffle_strategy, device="cuda", denormalise=True):
    shuffle_time = 0
    inf_time = 0
    stat_time = 0
    N = len(test_set)
    # Collect all individual values
    all_log_liks = []
    all_rmse_vals = []

    i = 0
    for batch_test in tqdm(test_set, desc="Evaluating Model on One Train Set"):
        # Shuffles data using defined permute strategy
        start_shuff_t = time.time()
        batch = shuffle_batch(model, batch_test, shuffle_strategy)
        shuffle_time += time.time() - start_shuff_t
        # Model LL and rmse
        m, nt, _ = batch.yt.shape
        # Gets predictive distribution from model
        start_inf_t = time.time()
        if batch.xc.device != device:
            batch = batch.to(device)
        pred_dist = np_pred_fn(model, batch, predict_without_yt_tnpa=True)
        if denormalise:
            pred_dist = denormalize_predictions(pred_dist, batch)
            yt_denorm = denormalize_targets(batch.yt, batch)
        else:
            yt_denorm = batch.yt
        inf_time += time.time() - start_inf_t
        stat_start_time = time.time()
        ll = pred_dist.log_prob(yt_denorm).sum(dim=(1, 2)) / nt
        rmse = torch.sqrt(((pred_dist.mean - yt_denorm) ** 2).mean(dim=(1, 2)))
        # GT LL - may need to wrap this in if statement in future datasets
        stat_time += time.time() - stat_start_time

        # Append all values from this batch
        all_log_liks.append(ll)
        all_rmse_vals.append(rmse)
        i += 1

    # Print times
    print_times_tracked = True
    if print_times_tracked:
        print(
            f"Inf time {inf_time:.2f}, Stat time {stat_time:.2f}, Shuffle time {shuffle_time:.2f}"
        )

    # Concatenate all batches into single tensor
    log_liks = torch.cat(all_log_liks)
    rmse_vals = torch.cat(all_rmse_vals)

    # Gathers results - return individual values
    results = {
        "all_lls": log_liks,
        "all_rmses": rmse_vals,
        "mean_ll": torch.mean(log_liks).item(),
        "std_ll": torch.std(log_liks).item() / (len(log_liks) ** 0.5),
        "mean_rmse": torch.mean(rmse_vals).item(),
        "std_rmse": torch.std(rmse_vals).item() / (len(rmse_vals) ** 0.5),
    }
    return results


# Evaluates model performance over a number of passes of the test set (e.g. to account for noise etc)
def eval_model_over_permutations(
    model,
    test_set,
    no_reps: int = 1,
    shuffle_strategy: str = "random",
    denormalise: bool = False,
):
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())

    # Collect all individual values across repetitions
    all_lls = []
    all_rmses = []
    timings = []

    for _ in tqdm(range(no_reps), desc="Evaluating model over multiple test sets"):
        start_t = time.time()
        result = eval_model(model, test_set, shuffle_strategy, denormalise=denormalise)
        timings.append(time.time() - start_t)

        # Accumulate all values
        all_lls.append(result["all_lls"])
        all_rmses.append(result["all_rmses"])

    # Stack all values from all repetitions into single tensors
    all_lls_stacked = torch.cat(all_lls)
    all_rmses_stacked = torch.cat(all_rmses)

    # Gathers results
    results = {
        "num_params": num_params,
        "all_lls": all_lls_stacked.cpu().tolist(),  # Single flat list
        "all_rmses": all_rmses_stacked.cpu().tolist(),  # Single flat list
        "mean_lls": torch.mean(all_lls_stacked).item(),
        "std_lls": torch.std(all_lls_stacked).item() / (len(all_lls_stacked) ** 0.5),
        "mean_rmse_vals": torch.mean(all_rmses_stacked).item(),
        "std_rmse_vals": torch.std(all_rmses_stacked).item()
        / (len(all_rmses_stacked) ** 0.5),
        "time": timings,
    }
    return results


# Loads data effeciently for fast computation
def load_data(data_gen, device="cuda"):
    start_t = time.time()
    for b in data_gen:
        b.xc = b.xc.to(device, non_blocking=True)
        b.yc = b.yc.to(device, non_blocking=True)
        b.xt = b.xt.to(device, non_blocking=True)
        b.yt = b.yt.to(device, non_blocking=True)
        b.x = b.x.to(device, non_blocking=True)
        b.y = b.y.to(device, non_blocking=True)
        b.gtll = None
        b.gt_pred = None
    data = [b for b in data_gen]
    print(f"Data Time {time.time() - start_t:.2f}")
    return data


def get_test_set_from_config(config_path: str, Nc: int = None):
    """Load test set directly from config with optional Nc override."""

    config_dir = str(Path(config_path).parent.absolute())
    config_name = Path(config_path).stem

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

        # Override Nc if specified
        if Nc is not None:
            cfg.generators.test.max_nc = Nc

        # Instantiate test generator
        gen_test = instantiate(cfg.generators.test)

    data = load_data(gen_test)
    return data


# Main function used to handle flow of evaluating model and plotting the results
def models_perf_main(
    model_list,
    test_data,
    Nc,
    checkpoint_identifier="0499",
):
    folder_name = "experiments/plot_results/eval_set/"
    folder_name += "tabular/"

    txt_file_summary = "Summary over eval data set tabular"
    all_results = {}  # Store all results for JSON export

    for (
        yml_path,
        wandb_id,
        shuffle_strategy,
        model_name,
        special_args,
        no_reps,
    ) in model_list:
        model = get_model(yml_path, wandb_id, seed=False)  # Loads model
        if special_args.startswith("TNPAR_"):
            model.num_samples = int(special_args.split("_")[1])
        results = eval_model_over_permutations(
            model,
            test_data,
            no_reps,
            shuffle_strategy,
        )

        summary_block = f"""
        ----------------------------
        Model: {model_name}
        Params: {results["num_params"]}
        Mean_LLs: [{results["mean_lls"]}]
        Std_LLs: [{results["std_lls"]}]
        Mean_RMSEs: [{results["mean_rmse_vals"]}]
        Std_RMSEs: [{results["std_rmse_vals"]}]
        Times: {results["time"]}
        Num_samples: {len(results["all_lls"])}
        """
        txt_file_summary += summary_block
        print(summary_block)

        # Store results for this model
        all_results[model_name] = results

    os.makedirs(folder_name, exist_ok=True)

    # Save text summary
    file_name = (
        folder_name + f"eval_tabular_Nc{Nc}_ckpt{checkpoint_identifier}" + ".txt"
    )
    with open(file_name, "w") as file_object:
        file_object.write(txt_file_summary)
    print("Saved summary to ", file_name)

    # Save full results with all individual values as JSON
    import json

    json_file_name = (
        folder_name
        + f"eval_results_tabular_Nc{Nc}_ckpt{checkpoint_identifier}"
        + ".json"
    )
    with open(json_file_name, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved full results to ", json_file_name)


# List of models to be tested, adjust this as required
def get_model_list(
    Nc, wandb_project, checkpoint_identifier="0499", model_combinations=None
):
    # Initialise list of models
    models = []

    # Initialize wandb api
    wandb_entity = "REMOVED"
    # wandb_project = "incTNP-RBF"
    api = wandb.Api()
    runs = api.runs(f"{wandb_entity}/{wandb_project}")

    # Get the names of the runs we are interested in
    lrs = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
    lr_schedulers = ["", "LRSched-"]
    model_types = ["plain", "mask", "mask-batched"]
    Nc_list = [Nc]

    # Get all combinations
    if model_combinations is None:
        combinations = list(itertools.product(lrs, lr_schedulers, model_types, Nc_list))
    else:
        combinations = model_combinations

    combinations_names = []
    configs = []
    for lr, lr_scheduler, model_type, nc in combinations:
        name = f"{model_type}-TNP-{lr_scheduler}L5-H8-D128-LR{lr}"
        combinations_names.append(name)
        config_name = ""
        if model_type == "plain":
            config_name = (
                f"experiments/configs/tabular_data/tab_{model_type}_tnp_lr_scheduler.yaml"
                if lr_scheduler != ""
                else f"experiments/configs/tabular_data/tab_{model_type}_tnp.yaml"
            )
        elif model_type == "mask":
            config_name = (
                "experiments/configs/tabular_data/tab_causal_tnp_lr_scheduler.yaml"
                if lr_scheduler != ""
                else "experiments/configs/tabular_data/tab_causal_tnp.yaml"
            )
        elif model_type == "mask-batched":
            config_name = (
                "experiments/configs/tabular_data/tab_batched_causal_tnp_lr_scheduler.yaml"
                if lr_scheduler != ""
                else "experiments/configs/tabular_data/tab_batched_causal_tnp.yaml"
            )
        configs.append(config_name)

    if model_combinations is None:
        additional_combinations_names = [
            "plain-TNP-LRSched-L5-H8-D128-LR0.003-E680",
            "mask-TNP-LRSched-L5-H8-D128-LR0.005-E680",
        ]
        combinations_names.extend(additional_combinations_names)
        configs.extend(
            [
                "experiments/configs/tabular_data/tab_plain_tnp_lr_scheduler.yaml",
                "experiments/configs/tabular_data/tab_causal_tnp_lr_scheduler.yaml",
            ]
        )
    assert len(combinations_names) == len(configs)
    # Fetch runs from wandb to figure out artifact names
    for run in runs:
        if run.name in combinations_names:
            print(f"Found run: {run.name} with id: {run.id}")

            idx = combinations_names.index(run.name)
            config = configs[idx]

            # Get artifacts created (logged) by this run
            artifacts = run.logged_artifacts()
            if "-E" in run.name:
                checkpoint_identifier_run = "best"
            else:
                checkpoint_identifier_run = checkpoint_identifier
            if checkpoint_identifier_run == "last":
                try:
                    # 1. Filter for models first
                    model_artifacts = [a for a in artifacts if a.type == "model"]

                    if not model_artifacts:
                        raise ValueError("No model artifacts found.")

                    selected_artifact = max(model_artifacts, key=get_epoch_number)

                    print(f"Selected latest: {selected_artifact.name})")
                    new_model = (
                        config,
                        f"{wandb_entity}/{wandb_project}/{selected_artifact.name}",
                        "random",
                        run.name,
                        "",
                        1,
                    )
                    models.append(new_model)
                except ValueError:
                    print(f"  - No model artifacts found for run {run.name}")
                    continue
            else:
                for artifact in artifacts:
                    if artifact.type == "model" and (
                        "v19" in artifact.name
                        or checkpoint_identifier_run in artifact.name
                    ):
                        print(f"  - Artifact: {artifact.name}, Type: {artifact.type}")
                        new_model = (
                            config,
                            f"{wandb_entity}/{wandb_project}/{artifact.name}",
                            "random",
                            run.name,
                            "",
                            1,
                        )
                        models.append(new_model)

    return models


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Evaluate models on tabular test set")
    parser.add_argument(
        "--ckpt_identifier",
        type=str,
        default="last",
        help="Checkpoint identifier to use",
    )
    args = parser.parse_args()
    start_t = time.time()
    pl.seed_everything(1)  # Sets seed of randomness for reproducibility
    wandb_project = "incTNP-tab"
    Nc_max = [1024]
    config = "experiments/configs/generators/tabular_data.yaml"

    for Nc in Nc_max:
        print(f"\nEvaluating models with Nc max = {Nc}\n")
        print("Using synthetic data generator for test set.")
        # Use synthetic generator
        gen_test = get_test_set_from_config(config, Nc)
        model_combinations = None
        models_perf_main(
            get_model_list(
                Nc,
                wandb_project,
                checkpoint_identifier=args.ckpt_identifier,
                model_combinations=model_combinations,
            ),
            gen_test,
            Nc,
            checkpoint_identifier=args.ckpt_identifier,
        )
    print(f"Runtime: {time.time() - start_t:.2f}s")
