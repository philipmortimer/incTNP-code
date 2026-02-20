# Eval script over UCI datasets
import argparse
import itertools
import json
import os
import re
import time
import gc

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from plot_adversarial_perms import get_model
from scipy.io import arff
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
def eval_model(
    model,
    test_set_chunk,  # Accepts a LIST (Chunk) of batches
    shuffle_strategy,
    device="cuda",
    denormalise=True,
):
    all_log_liks = []
    all_rmse_vals = []

    # Iterate over the small chunk
    for batch_test in test_set_chunk:
        # Shuffle
        batch = shuffle_batch(model, batch_test, shuffle_strategy)

        # Move to GPU
        batch = batch.to(device)

        # Predict
        pred_dist = np_pred_fn(model, batch, predict_without_yt_tnpa=True)

        if denormalise:
            pred_dist = denormalize_predictions(pred_dist, batch)
            yt_denorm = denormalize_targets(batch.yt, batch)
        else:
            yt_denorm = batch.yt

        # Metrics
        ll = pred_dist.log_prob(yt_denorm).sum(dim=(1, 2)) / batch.yt.shape[1]
        rmse = torch.sqrt(((pred_dist.mean - yt_denorm) ** 2).mean(dim=(1, 2)))

        # Move results to CPU to save GPU memory
        all_log_liks.append(ll.cpu())
        all_rmse_vals.append(rmse.cpu())

    return torch.cat(all_log_liks), torch.cat(all_rmse_vals)


def eval_model_over_permutations(
    model,
    data_gen_fn,  # Pass the GENERATOR FUNCTION, not the data list
    data_gen_args,  # Arguments for the generator
    no_reps: int = 1,
    shuffle_strategy: str = "random",
    denormalise: bool = False,
):
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())

    # We will accumulate results across all chunks and reps
    grand_lls = []
    grand_rmses = []

    # Get the Generator
    # We create the generator ONCE per model evaluation to stream through data
    data_generator = data_gen_fn(**data_gen_args)

    total_processed = 0
    start_t = time.time()

    # Process Data in Chunks
    for chunk in tqdm(data_generator, desc="Processing Data Chunks"):
        # chunk is a list of e.g., 100 batches

        # Perform Repetitions on this specific chunk
        # (Usually 1 rep for standard eval, multiple for stability checks)
        chunk_lls = []
        chunk_rmses = []

        for _ in range(no_reps):
            lls, rmses = eval_model(
                model,
                chunk,
                shuffle_strategy,
                device="cuda",
                denormalise=denormalise,
            )
            chunk_lls.append(lls)
            chunk_rmses.append(rmses)

        # Accumulate results (Lists of Tensors)
        grand_lls.extend(chunk_lls)
        grand_rmses.extend(chunk_rmses)

        # The 'chunk' variable goes out of scope here and memory is freed!
        total_processed += len(chunk)

    # Flatten results
    all_lls_stacked = torch.cat(grand_lls)
    all_rmses_stacked = torch.cat(grand_rmses)

    results = {
        "num_params": num_params,
        "mean_lls": torch.mean(all_lls_stacked).item(),
        "std_lls": torch.std(all_lls_stacked).item() / (len(all_lls_stacked) ** 0.5),
        "mean_rmse_vals": torch.mean(all_rmses_stacked).item(),
        "std_rmse_vals": torch.std(all_rmses_stacked).item()
        / (len(all_rmses_stacked) ** 0.5),
        "all_lls": all_lls_stacked.tolist(),  # Convert to list for JSON serialization
        "all_rmses": all_rmses_stacked.tolist(),
        "time": time.time() - start_t,
    }
    return results


# Load real life data
def data_chunk_generator(
    data_path: str,
    train_split: float = 0.8,
    num_batches: int = 10,
    batch_size: int = 16,
    max_features: int = 20,
    device="cuda",
    random_split: bool = True,
    chunk_size: int = 50,  # NEW: How many batches to yield at once
):
    """
    Generator that yields lists of batches (chunks) to prevent OOM.
    """
    start_t = time.time()

    # 1. Load Data (Raw data is usually small enough to keep in RAM)
    if "skillcraft" in data_path.lower():
        print(f"Loading Skillcraft CSV data from: {data_path}")
        data = pd.read_csv(data_path, na_values=["?"]).dropna().values
    elif "powerplant" in data_path.lower():
        print(f"Loading Power Plant Excel data from: {data_path}")
        data = np.array(pd.read_excel(data_path, engine="openpyxl"))
    elif "elevators" in data_path.lower():
        print(f"Loading Elevators ARFF data from: {data_path}")
        data, meta = arff.loadarff(data_path)
        data = pd.DataFrame(data).values
        chunk_size = 25 # Smaller due to mem issues
        # for col in data.select_dtypes([object]):
        #     data[col] = data[col].str.decode("utf-8")
    elif "protein" in data_path.lower():
        print(f"Loading Protein CSV data from: {data_path}")
        data = pd.read_csv(data_path)
        # Move target from first to last column
        first_col = data.columns[0]
        data = data[[col for col in data.columns if col != first_col] + [first_col]]
        data = data.values
        chunk_size = 10  # Smaller chunks for larger dataset
    else:
        # Fallback or generic CSV
        data = pd.read_csv(data_path, na_values=["?"]).dropna().values

    x_raw = data[:, :-1]
    y_raw = data[:, -1].reshape(-1, 1)

    # Log-transform fix for heavy tails (Optional, based on previous discussion)
    # x_raw = np.log1p(np.abs(x_raw)) * np.sign(x_raw)
    # y_raw = np.log1p(np.abs(y_raw)) * np.sign(y_raw)

    N, dx = x_raw.shape
    if dx > max_features:
        raise ValueError(f"Dataset has {dx} features, max is {max_features}")

    current_chunk = []

    # Iterate through total required batches
    for batch_idx in range(num_batches):
        xc_list, yc_list, xt_list, yt_list = [], [], [], []
        x_list, y_list = [], []
        x_means, x_stds, y_means, y_stds = [], [], [], []

        # Randomize Context Size
        if random_split:
            nc = np.random.randint(int(0.5 * N), int(train_split * N))
        else:
            nc = int(train_split * N)

        # Create one Batch (contains 'batch_size' permutations)
        for i in range(batch_size):
            indices = np.arange(N)
            np.random.shuffle(indices)

            x_shuffled = x_raw[indices]
            y_shuffled = y_raw[indices]

            xc = x_shuffled[:nc]
            yc = y_shuffled[:nc]
            xt = x_shuffled[nc:]
            yt = y_shuffled[nc:]

            # 3. MIN-MAX NORMALIZATION ([-1, 1])

            # Calculate Min and Max on Context (xc) only
            # Shapes: [features]
            c_min = np.min(xc, axis=0)
            c_max = np.max(xc, axis=0)

            # Calculate Center (Mid-range) and Scale (Half-range)
            x_center = (c_max + c_min) / 2.0
            x_scale = (c_max - c_min) / 2.0

            # Safety: Prevent division by zero if max == min (constant feature)
            x_scale[x_scale < 1e-8] = 1.0

            # Save stats into your existing list structure
            # We treat 'center' like mean and 'scale' like std
            x_means.append(x_center)
            x_stds.append(x_scale)

            # Do the same for Y (Targets)
            y_min = np.min(yc, axis=0)
            y_max = np.max(yc, axis=0)

            y_center = (y_max + y_min) / 2.0
            y_scale = (y_max - y_min) / 2.0
            y_scale[y_scale < 1e-8] = 1.0

            y_means.append(y_center)
            y_stds.append(y_scale)

            # Apply Normalization
            # This maps xc exactly to [-1, 1]
            xc_norm = (xc - x_center) / x_scale
            yc_norm = (yc - y_center) / y_scale

            # Apply to Target (Note: xt might go outside -1/1 if it has outliers)
            xt_norm = (xt - x_center) / x_scale
            yt_norm = (yt - y_center) / y_scale

            # Clamp
            xc_norm = np.clip(xc_norm, -5, 5)
            xt_norm = np.clip(xt_norm, -5, 5)

            # Padding
            if dx < max_features:
                pad_width = max_features - dx
                xc_norm = np.pad(xc_norm, ((0, 0), (0, pad_width)), mode="constant")
                xt_norm = np.pad(xt_norm, ((0, 0), (0, pad_width)), mode="constant")

            x_full = np.concatenate([xc_norm, xt_norm], axis=0)
            y_full = np.concatenate([yc_norm, yt_norm], axis=0)

            xc_list.append(xc_norm)
            yc_list.append(yc_norm)
            xt_list.append(xt_norm)
            yt_list.append(yt_norm)
            x_list.append(x_full)
            y_list.append(y_full)

        # Convert to Tensor (CPU)
        batch_obj = Batch(
            xc=torch.from_numpy(np.stack(xc_list)).float(),
            yc=torch.from_numpy(np.stack(yc_list)).float(),
            xt=torch.from_numpy(np.stack(xt_list)).float(),
            yt=torch.from_numpy(np.stack(yt_list)).float(),
            x=torch.from_numpy(np.stack(x_list)).float(),
            y=torch.from_numpy(np.stack(y_list)).float(),
        )

        # Attach Stats
        batch_obj.x_mean = torch.from_numpy(np.stack(x_means)).float().to(device)
        batch_obj.x_std = torch.from_numpy(np.stack(x_stds)).float().to(device)
        batch_obj.y_mean = (
            torch.from_numpy(np.stack(y_means)).float().to(device).unsqueeze(-1)
        )
        batch_obj.y_std = (
            torch.from_numpy(np.stack(y_stds)).float().to(device).unsqueeze(-1)
        )

        current_chunk.append(batch_obj)

        # Yield Logic: If chunk full, yield and clear memory
        if len(current_chunk) >= chunk_size:
            yield current_chunk
            current_chunk = []  # Reset list, freeing memory

    # Yield remaining
    if current_chunk:
        yield current_chunk


# Main function used to handle flow of evaluating model and plotting the results
def models_perf_main(
    model_list,
    data_path,  # Pass path, not loaded data
    Nc,
    dataset_name=None,
    denormalise=False,
    num_batches=10,
    batch_size=128,
    shuffle_strategy="random",
    no_reps=1,
):
    folder_name = "experiments/plot_results/eval_set/"
    if dataset_name:
        folder_name += f"{dataset_name}/"
    else:
        folder_name += "tabular/"

    txt_file_summary = "Summary over eval data set tabular"
    all_results = {}

    for (yml, ckpt, model_name) in model_list:
        # CRITICAL: Reset Seed before every model so they get the exact same "random" data chunks
        pl.seed_everything(1)

        model = get_model(yml, ckpt, seed=False, local_weights=True)

        # Define args for generator
        data_args = {
            "data_path": data_path,
            "train_split": 0.8,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "max_features": 20,
            "device": "cuda",
            "chunk_size": 50,  # Process 100 (or specified) batches at a time to save RAM
        }

        # Pass generator FUNCTION, not data
        # When we load real data
        results = eval_model_over_permutations(
            model,
            data_gen_fn=data_chunk_generator,
            data_gen_args=data_args,
            no_reps=no_reps,
            shuffle_strategy=shuffle_strategy,
            denormalise=denormalise,
        )

        summary_block = f"""
        ----------------------------
        Model: {model_name}
        Params: {results["num_params"]}
        Mean_LLs: [{results["mean_lls"]}]
        Std_LLs: [{results["std_lls"]}]
        Mean_RMSEs: [{results["mean_rmse_vals"]}]
        Std_RMSEs: [{results["std_rmse_vals"]}]
        """
        txt_file_summary += summary_block
        print(summary_block)
        all_results[model_name] = results

        del model
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs(folder_name, exist_ok=True)

    # File saving logic (same as before)
    file_base = (
        f"{folder_name}eval_results_tabular_Nc{Nc}"
        + ("_denormalised" if denormalise else "")
    )

    with open(file_base + ".txt", "w") as f:
        f.write(txt_file_summary)

    with open(file_base + ".json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved results.")


# List of models to be tested, adjust this as required
def get_model_list():
    tnp_plain = ('experiments/configs/incTNPCheckpoints/Tabular/config_tnpd_tab_epoch_0679.yaml',
                 'experiments/configs/incTNPCheckpoints/Tabular/tnpd_tab_epoch_0679.ckpt', 'TNP_D')
    incTNP = ('experiments/configs/incTNPCheckpoints/Tabular/config_inctnp_tab_epoch_0499.yaml',
        'experiments/configs/incTNPCheckpoints/Tabular/inctnp_tab_epoch_0499.ckpt', 'incTNP')
    batchedTNP = ('experiments/configs/incTNPCheckpoints/Tabular/config_inctnpb_tab_epoch_0499.yaml',
        'experiments/configs/incTNPCheckpoints/Tabular/inctnpb_tab_epoch_0499.ckpt', 'incTNP-Batched')
    return [tnp_plain, incTNP, batchedTNP]


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Evaluate models on tabular test set")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to CSV file (if None, uses synthetic generator)",
    )
    parser.add_argument(
        "--shuffle_strategy",
        type=str,
        default="random",
        help="Data shuffling strat",
    )
    parser.add_argument(
        "--no_reps",
        type=int,
        default="1",
        help="No reps data is shuffled",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=500,
        help="Number of batches to create (default: 500)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of shufflings per batch (default: 128)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data to use as context (default: 0.8)",
    )
    parser.add_argument(
        "--denormalise",
        action="store_true",
        help="Whether to denormalise predictions before evaluation",
    )
    args = parser.parse_args()
    start_t = time.time()
    pl.seed_everything(1)  # Sets seed of randomness for reproducibility
    wandb_project = "incTNP-tab"
    Nc_max = [1024]
    config = "experiments/configs/generators/tabular_data.yaml"


    for Nc in Nc_max:
        print(f"\nEvaluating models with Nc max = {Nc}\n")
        # Run evaluation
        dataset_name = (
            args.data_path.split("/")[-1].split(".")[0] if args.data_path else None
        )

        models_perf_main(
            get_model_list(),
            args.data_path,  # gen_test,
            Nc,
            dataset_name=dataset_name,
            denormalise=args.denormalise,
            num_batches=args.num_batches,
            shuffle_strategy=args.shuffle_strategy,
            no_reps=args.no_reps
        )
    print(f"Runtime: {time.time() - start_t:.2f}s")
