# Script for evaluation of models on the HadISD temporal task
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


def test_model(lit_model, test_loader, wandb_run, eval_name, wandpath, ordering_type, h, delta, model_name, model_train_description):
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

    loglik_temp = torch.stack(test_result["loglik_temp"])
    test_result["mean_loglik_temp"] = loglik_temp.mean()
    test_result["std_loglik_temp"] = loglik_temp.std() / (len(loglik_temp) ** 0.5)
    rmse_temp = torch.stack(test_result["rmse_temp"])
    test_result["mean_rmse_temp"] = rmse_temp.mean()
    test_result["std_rmse_temp"] = rmse_temp.std() / (len(rmse_temp) ** 0.5)
    
    # W&B logging
    if wandb_run is not None:
        summary = wandb_run.summary() if callable(wandb_run.summary) else wandb_run.summary
        summary.update({"num_params": num_params})
        summary.update({f"test/{eval_name}/loglik": test_result["mean_loglik"]})
        summary.update({f"test/{eval_name}/std_loglik": test_result["std_loglik"]})
        summary.update({f"test/{eval_name}/rmse": test_result["mean_rmse"]})
        summary.update({f"test/{eval_name}/std_rmse": test_result["std_rmse"]})
        summary.update({f"test/{eval_name}/loglik_temp": test_result["mean_loglik_temp"]})
        summary.update({f"test/{eval_name}/std_loglik_temp": test_result["std_loglik_temp"]})
        summary.update({f"test/{eval_name}/rmse_temp": test_result["mean_rmse_temp"]})
        summary.update({f"test/{eval_name}/std_rmse_temp": test_result["std_rmse_temp"]})

    # Return stats
    stats = {
        "model": model_name,
        "training_format": model_train_description,
        "eval_name": eval_name,
        "loglik_temp": test_result["mean_loglik_temp"].item(),
        "std_loglik_temp": test_result["std_loglik_temp"].item(),
        "rmse_temp": test_result["mean_rmse_temp"].item(),
        "std_rmse_temp": test_result["std_rmse_temp"].item(),
        "ordering": ordering_type,
        "h": h,
        "delta": delta,
        "wandb_path": wandpath,
    }
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


def test_all_models_np(samples_per_epoch, split, N_c_min, N_c_max, N_t_min, N_t_max, batch_size, h_delta_list, ordering_list, ar_samples, model_list, data_root, num_workers, output_file, usewandb):
    if samples_per_epoch is None: return
    if os.path.exists(output_file):
            print(f"Removing old results file: {output_file}")
            os.remove(output_file)
    if usewandb: api = wandb.Api()
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
                        gen_test = TemporalHadISDDataGenerator(N_c_min=N_c_min, N_c_max=N_c_max, N_t_min=N_t_min, N_t_max=N_t_max, split=split, samples_per_epoch=samples_per_epoch, batch_size=batch_size, data_root=data_root, ordering=ordering_type, delta_hours=delta, h_window=h)
                        test_loader = torch.utils.data.DataLoader(gen_test, batch_size=None, num_workers=num_workers, worker_init_fn=(adjust_num_batches if num_workers > 0 else None),persistent_workers=False, pin_memory=True,)
                        lit_model = LitWrapper(model=model, optimiser=None,)
                        if usewandb:
                            # Inits weights and biases logging
                            print(get_run_string(model_wandb_path))
                            run = api.run(get_run_string(model_wandb_path))
                            run = wandb.init(resume="allow", project=run.project, name=run.name, id=run.id,)
                        with torch.no_grad():
                            stats = test_model(lit_model=lit_model, test_loader=test_loader, wandb_run=(wandb.run if usewandb else None), eval_name=eval_name, wandpath=model_wandb_path, ordering_type=ordering_type, h=h, delta=delta, model_name=model_name, model_train_description=model_train_description)
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        file_exists = os.path.isfile(output_file)
                        with open(output_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=stats.keys())
                            if not file_exists:
                                writer.writeheader()
                            writer.writerow(stats)
                        # Tries to clean up stuff to prevent script being killed due to mem
                        if usewandb: run.finish()
                        lit_model.test_outputs = []
                        lit_model.val_batches = []
                        del lit_model
                        del test_loader
                        del gen_test
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                    

def test_all_models_ar(split, N_c_min, N_c_max, N_t_min, N_t_max, batch_size, h_delta_list, ordering_list, ar_samples, model_list, data_root, num_workers,
        device, ar_rollout_rmse, num_samples_ar, prioritise_fixed_ar, use_flash_ar, max_no_batches_ar, ar_orderingmode_list, ar_output_file):
    if ar_samples is None: return
    if os.path.exists(ar_output_file):
            print(f"Removing old results file: {ar_output_file}")
            os.remove(ar_output_file)
    for (model_cfg, model_name), model_vars_list in  model_list:
        for model_wandb_path, model_train_description, model_type in model_vars_list:
            assert model_type in ("ckpt", "wandb"), "Model must either be wandb or ckpt"
            weights_only_evalhad_call = model_type == "ckpt"
            for ordering_type in ordering_list:
                for (h, delta) in h_delta_list:
                    # AR runs
                    if ar_samples is not None:
                        print(f"Testing AR  {model_cfg} {model_train_description} {model_wandb_path} on {ordering_type} {h} {delta}")
                        model = get_model(model_cfg, model_wandb_path, seed=True, weights_only_evalhad_call=weights_only_evalhad_call, device=device)
                        # AR data
                        gen_test_ar = TemporalHadISDDataGenerator(N_c_min=N_c_min, N_c_max=N_c_max, N_t_min=N_t_min, N_t_max=N_t_max, split=split, samples_per_epoch=ar_samples, batch_size=batch_size, data_root=data_root, ordering=ordering_type, delta_hours=delta, h_window=h)
                        test_loader_ar = torch.utils.data.DataLoader(gen_test_ar, batch_size=None, num_workers=num_workers, worker_init_fn=(adjust_num_batches if num_workers > 0 else None),persistent_workers=False, pin_memory=True,)
                        for ordering_ar in ar_orderingmode_list:
                            eval_name = f"{model_name}_AR_ord_{ordering_type}-H_{h}_delta_{delta}_arorder_{ordering_ar}"
                            model.to(device)
                            with torch.no_grad():
                                _, stats = eval_had_ar_model(model=model, data=test_loader_ar, model_name=eval_name, device=device, ordering=ordering_ar, rollout_rmse=ar_rollout_rmse, num_samples=num_samples_ar, prioritise_fixed=prioritise_fixed_ar, use_flash=use_flash_ar, max_no_batches=max_no_batches_ar)
                            os.makedirs(os.path.dirname(ar_output_file), exist_ok=True)
                            file_exists = os.path.isfile(ar_output_file)
                            with open(ar_output_file, 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=stats.keys())
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerow(stats)
                        del test_loader_ar
                        del gen_test_ar
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    # ---- eval hypers -----------
    samples_per_epoch = 80_000 # None for no NP eval
    split = "test"
    N_c_min = 100
    N_c_max = 2100
    N_t_min = 250
    N_t_max = 250
    batch_size = 32
    num_workers = 1
    output_file = "experiments/plot_results/eval_hadTemp/eval_main_incsrandup.txt"
    ar_output_file = "experiments/plot_results/eval_hadTemp/areval_incsrandup.txt"
    data_root = "REDACTED"
    usewandb = True #whether to log results to wandb
    # ----- NP Evals to run-----------
    h_delta_list = [(8, 6), (4, 6), (7, 24), (12, 6)] # List of (h, delta pairs to run eval over)
    ordering_list = ["ctx_time", "forecasting", "random", "full_time"] # Data ordering evals to run
    # -------- AR settings ---------
    ar_samples = 4_096 # Number of samples per run to use for ar evaluation - None denotes not to run ar eval
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
    incTNP_batched_rand_config_name = (f"experiments/configs/{folder}/rand_new/hadtemp_incTNP_batched.yml", "incTNP-Batched (rand)")
    inc_batched_rand_models = [('REMOVED', 'random-H8-D6', 'wandb')]
    model_list = [(incTNP_batched_rand_config_name, [])] # removed list of model lists as W&B links break anonymity
    #test_all_models_np(samples_per_epoch=samples_per_epoch, split=split, N_c_min=N_c_min, N_c_max=N_c_max, N_t_min=N_t_min, N_t_max=N_t_max, batch_size=batch_size, h_delta_list=h_delta_list, ordering_list=ordering_list, ar_samples=ar_samples, model_list=model_list, data_root=data_root, num_workers=num_workers, output_file=output_file, usewandb=usewandb)
    test_all_models_ar(split=split, N_c_min=N_c_min, N_c_max=N_c_max, N_t_min=N_t_min, N_t_max=N_t_max, batch_size=batch_size, h_delta_list=h_delta_list_ar, ordering_list=ordering_list_ar, ar_samples=ar_samples, model_list=model_list, data_root=data_root, num_workers=num_workers,
        device=device, ar_rollout_rmse=ar_rollout_rmse, num_samples_ar=num_samples_ar, prioritise_fixed_ar=prioritise_fixed_ar, use_flash_ar=use_flash_ar, max_no_batches_ar=max_no_batches_ar, ar_orderingmode_list=ar_orderingmode_list, ar_output_file=ar_output_file)
