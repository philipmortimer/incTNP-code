# Eval Script
import lightning.pytorch as pl
import torch

import wandb
from tnp.utils.experiment_utils import initialize_evaluation
from tnp.utils.data_loading import adjust_num_batches


def test_model(lit_model, experiment, wandb_run=None):
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.test

    if hasattr(experiment.misc, "shuffle_greedy_val"):
        lit_model.model.order_ctx_greedy = experiment.misc.shuffle_greedy_val
    lit_model.eval()

    # Store number of parameters.
    num_params = sum(p.numel() for p in lit_model.parameters())

    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        logger=False,
    )

    test_loader = torch.utils.data.DataLoader(
        gen_test,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_val_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_val_workers > 0 else False,
        pin_memory=True,
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


    # Handles hadISD case
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

    if experiment.misc.logging:
        summary = wandb_run.summary() if callable(wandb_run.summary) else wandb_run.summary
        summary.update({"num_params": num_params})
        summary.update({f"test/{eval_name}/loglik": test_result["mean_loglik"]})
        summary.update({f"test/{eval_name}/std_loglik": test_result["std_loglik"]})

        summary.update({f"test/{eval_name}/rmse": test_result["mean_rmse"]})
        summary.update({f"test/{eval_name}/std_rmse": test_result["std_rmse"]})
        if "mean_gt_loglik" in test_result:
            summary.update({f"test/{eval_name}/gt_loglik": test_result["mean_gt_loglik"]})
            summary.update({f"test/{eval_name}/std_gt_loglik": test_result["std_gt_loglik"]})

        # Handles HadISD case
        if "loglik_temp" in test_result:
            summary.update({f"test/{eval_name}/loglik_temp": test_result["mean_loglik_temp"]})
            summary.update({f"test/{eval_name}/std_loglik_temp": test_result["std_loglik_temp"]})
            summary.update({f"test/{eval_name}/rmse_temp": test_result["mean_rmse_temp"]})
            summary.update({f"test/{eval_name}/std_rmse_temp": test_result["std_rmse_temp"]})


def main():
    experiment = initialize_evaluation()

    lit_model = experiment.lit_model

    test_model(lit_model, experiment, wandb_run=wandb.run)


if __name__ == "__main__":
    main()
