# Evaluates the GP task models comparing normal performance to AR performance
from arnp_hadTemporal import eval_had_ar_model
import lightning.pytorch as pl
from tnp.data.hadISDTemporal import TemporalHadISDDataGenerator
from tnp.utils.lightning_utils import LitWrapper
import torch
import csv
import os
from datetime import datetime
import wandb
from tnp.utils.experiment_utils import initialize_evaluation
from tnp.utils.data_loading import adjust_num_batches
from plot_adversarial_perms import get_model
import gc
import re


def test_model(lit_model, test_loader, wandb_run, wandpath, model_name, eval_name="eval_gpspy_ev"):
    lit_model.eval()

    # Store number of parameters.
    num_params = sum(p.numel() for p in lit_model.parameters())

    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        logger=False,
    )

    trainer.test(model=lit_model, dataloaders=test_loader)
    test_result = {
        k: [result[k] for result in lit_model.test_outputs]
        for k in lit_model.test_outputs[0].keys()
    }
    loglik = torch.stack(test_result["loglik"])
    test_result["mean_loglik"] = loglik.mean()
    test_result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)

    rmse = torch.stack(test_result["rmse"])
    test_result["mean_rmse"] = rmse.mean()
    test_result["std_rmse"] = rmse.std() / (len(rmse) ** 0.5)

    if "loglik_temp" in test_result:
        loglik_temp = torch.stack(test_result["loglik_temp"])
        test_result["mean_loglik_temp"] = loglik_temp.mean()
        test_result["std_loglik_temp"] = loglik_temp.std() / (len(loglik_temp) ** 0.5)
        rmse_temp = torch.stack(test_result["rmse_temp"])
        test_result["mean_rmse_temp"] = rmse_temp.mean()
        test_result["std_rmse_temp"] = rmse_temp.std() / (len(rmse_temp) ** 0.5)

    if "gt_loglik" in test_result:
        gt_loglik = torch.stack(test_result["gt_loglik"])
        test_result["mean_gt_loglik"] = gt_loglik.mean()
        test_result["std_gt_loglik"] = gt_loglik.std() / (len(gt_loglik) ** 0.5)
    
    # W&B logging
    if wandb_run is not None:
        summary = wandb_run.summary() if callable(wandb_run.summary) else wandb_run.summary
        summary.update({"num_params": num_params})
        summary.update({f"test/{eval_name}/loglik": test_result["mean_loglik"]})
        summary.update({f"test/{eval_name}/std_loglik": test_result["std_loglik"]})
        summary.update({f"test/{eval_name}/rmse": test_result["mean_rmse"]})
        summary.update({f"test/{eval_name}/std_rmse": test_result["std_rmse"]})
        if "loglik_temp" in test_result:
            summary.update({f"test/{eval_name}/loglik_temp": test_result["mean_loglik_temp"]})
            summary.update({f"test/{eval_name}/std_loglik_temp": test_result["std_loglik_temp"]})
            summary.update({f"test/{eval_name}/rmse_temp": test_result["mean_rmse_temp"]})
            summary.update({f"test/{eval_name}/std_rmse_temp": test_result["std_rmse_temp"]})

    # Return stats
    stats = {
        "model": model_name,
        "loglik": test_result["mean_loglik"].item(),
        "std_loglik": test_result["std_loglik"].item(),
        "rmse": test_result["mean_rmse"].item(),
        "std_rmse": test_result["std_rmse"].item(),
        "wandb_path": wandpath,
    }

    if "loglik_temp" in test_result:
        stats["loglik_temp"] = test_result["mean_loglik_temp"].item()
        stats["std_loglik_temp"] = test_result["std_loglik_temp"].item(),
        stats["rmse_temp"] = test_result["mean_rmse_temp"].item()
        stats["std_rmse_temp"] = test_result["std_rmse_temp"].item(),
    
    if "gt_loglik" in test_result:
        stats["gt_loglik"] = test_result["mean_gt_loglik"].item()
        stats["gt_loglik_std"] = test_result["std_gt_loglik"].item()
    print(f"\n--- Results for {eval_name} ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("-" * 30)
    return stats


# Converts w&b artefact string into the model string using string replacement
def get_run_string(wab_artefact_string):
    parts = wab_artefact_string.split('/')
    entity_project = "/".join(parts[:2])
    artifact_full_name = parts[2].split(':')[0]
    match = re.search(r'model-([a-zA-Z0-9]+)', artifact_full_name)
    if match:
        run_id = match.group(1)
    else:
        run_id = artifact_full_name.replace("model-", "").split('-')[0]
        
    return f"{entity_project}/runs/{run_id}"

def test_all_models(data_gen_callable, eval_function_callable, output_file, model_lists):
    if os.path.exists(output_file):
            print(f"Removing old results file: {output_file}")
            os.remove(output_file)
    api = wandb.Api()
    for (model_cfg, model_name), model_vars_list in  model_list:
        for model_wandb_path, model_train_description, model_type in model_vars_list:
            if samples_per_epoch is not None:
                assert model_type in ("ckpt", "wandb"), "Model must either be wandb or ckpt"
                weights_only_evalhad_call = model_type == "ckpt"
                for ordering_type in ordering_list:
                    for (h, delta) in h_delta_list:
                        print(f"Testing {model_cfg} {model_train_description} {model_wandb_path} on {ordering_type} {h} {delta}")
                        eval_name = f"ord_{ordering_type}-H_{h}_delta_{delta}"
                        model = get_model(model_cfg, model_wandb_path, seed=True, weights_only_evalhad_call=weights_only_evalhad_call, device=device)
                        # Data loader
                        gen_test = data_gen_callable()
                        test_loader = torch.utils.data.DataLoader(gen_test, batch_size=None, num_workers=num_workers, worker_init_fn=(adjust_num_batches if num_workers > 0 else None),persistent_workers=False, pin_memory=True,)
                        lit_model = LitWrapper(model=model, optimiser=None,)
                        run = api.run(get_run_string(model_wandb_path))
                        run = wandb.init(resume="allow", project=run.project, name=run.name, id=run.id,)
                        with torch.no_grad():
                            stats = eval_function_callable()
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        file_exists = os.path.isfile(output_file)
                        with open(output_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=stats.keys())
                            if not file_exists:
                                writer.writeheader()
                            writer.writerow(stats)
                        # Tries to clean up stuff to prevent script being killed due to mem
                        run.finish()
                        lit_model.test_outputs = []
                        lit_model.val_batches = []
                        del lit_model
                        del test_loader
                        del gen_test
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    # ---- eval hypers -----------
    samples_per_epoch = None # None for no NP eval
    split = "test"
    N_c_min = 100
    N_c_max = 2100
    N_t_min = 250
    N_t_max = 250
    batch_size = 32
    num_workers = 1
    output_file = "experiments/plot_results/eval_hadTemp/eval.txt"
    ar_output_file = "experiments/plot_results/eval_hadTemp/areval.txt"
    data_root = "REMOVED" # remove data root for anonymity
    usewandb = True #whether to log results to wandb
    # ----- NP Evals to run-----------
    h_delta_list = [(8, 6), (4, 6), (24, 7)] # List of (h, delta pairs to run eval over)
    ordering_list = ["ctx_time", "random", "full_time", "forecasting"] # Data ordering evals to run
    # -------- AR settings ---------
    ar_samples = None # Number of samples per run to use for ar evaluation - None denotes not to run ar eval
    device = "cuda"
    ar_rollout_rmse = False
    num_samples_ar = 50
    prioritise_fixed_ar = True
    use_flash_ar = False
    max_no_batches_ar = None
    ar_orderingmode_list = ["given"] # AR odering modes to try
    h_delta_list_ar = h_delta_list
    ordering_list_ar = ordering_list
    # ---------- models to try -----------------
    folder = "hadISDTemporal"
    incTNP_config_name = (f"experiments/configs/{folder}/hadtemp_incTNP.yml", "incTNP")
    incTNP_models = [('REMOVED', 'ctx_time-H8-D6', 'ckpt')] # List of models - each entry is a tuple consisting of the W&B link first and then also a string explaining the task type it was trained on (to make it readable)
    model_list = [(incTNP_config_name, incTNP_models)]
    # -----------------------------------
