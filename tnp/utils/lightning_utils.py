import dataclasses
import os
import time
from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch
from torch import nn
import numpy as np
import wandb

from ..data.base import Batch
from ..data.hadISD import HadISDBatch
from ..data.hadISDTemporal import TemporalHadISDBatch
from ..data.hadISD import get_true_temp, scale_pred_temp_dist
from .np_functions import np_loss_fn, np_pred_fn



class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Callable = np_loss_fn,
        pred_fn: Callable = np_pred_fn,
        plot_fn: Optional[Callable] = None,
        plot_interval: int = 1,
    ):
        super().__init__()

        self.model = model
        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn
        self.plot_fn = plot_fn
        self.plot_interval = plot_interval

        # Keep these for plotting.
        self.val_batches: List[Batch] = []

        # Keep these for analysing.
        self.test_outputs: List[Any] = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> torch.Tensor:
        _ = batch_idx
        loss = self.loss_fn(self.model, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log current learning rate
        if self.lr_scheduler is not None:
            # Get the current learning rate from the optimizer (assumes only one parameter group)
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log("train/lr", current_lr, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    # Val step for had isd
    def _val_had(self, batch, pred_dist, base_name: str = "val") -> None:
        # Reconstructs mean and variance vectors to be the correct units
        pred_dist_temp = scale_pred_temp_dist(batch, pred_dist)
        yt_correct_units = get_true_temp(batch, batch.yt)

        # Computes track statistics
        loglik_temp = pred_dist_temp.log_prob(yt_correct_units).sum() / yt_correct_units[..., 0].numel()
        rmse_temp = nn.functional.mse_loss(pred_dist_temp.mean, yt_correct_units).sqrt().cpu().mean()
        self.log(f"{base_name}/loglik_temp", loglik_temp, on_step=False, on_epoch=True, prog_bar=True,)
        self.log(f"{base_name}/rmse_temp", rmse_temp, on_step=False, on_epoch=True, prog_bar=True)

        # Tracks measure of predicted variance
        sigma_flat = pred_dist_temp.stddev.detach().cpu().numpy().flatten()
        stats = {
            f"{base_name}_var/sigma_min": float(sigma_flat.min()),
            f"{base_name}_var/sigma_max": float(sigma_flat.max()),
            f"{base_name}_var/sigma_mean": float(sigma_flat.mean()),
            f"{base_name}_var/sigma_med": float(np.median(sigma_flat)),
            f"{base_name}_var/sigma_spread": float(sigma_flat.std()), 
            f"{base_name}_var/sigma_p05": float(np.percentile(sigma_flat, 5)),
            f"{base_name}_var/sigma_p95": float(np.percentile(sigma_flat, 95)),
            f"{base_name}_var/num_nans": int(np.isnan(sigma_flat).sum()),
            f"{base_name}_var/num_infs": int(np.isinf(sigma_flat).sum()),
        }
        self.log_dict(stats, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        if batch_idx < 5:
            # Only keep first 5 batches for logging.
            self.val_batches.append(batch)

        pred_dist = self.pred_fn(self.model, batch)

        # Compute metrics to track.
        loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        rmse = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu().mean()

        self.log("val/loglik", loglik, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            self.log(
                "val/gt_loglik", gt_loglik, on_step=False, on_epoch=True, prog_bar=True
            )

        # Handles temperature prediction case - distribution must be normal but is for the models we use.
        if isinstance(batch, HadISDBatch) or isinstance(batch, TemporalHadISDBatch):
            self._val_had(batch=batch, pred_dist=pred_dist, base_name="val")



    def test_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        _ = batch_idx
        result = {}
        pred_dist = self.pred_fn(self.model, batch)

        # Compute metrics to track.
        loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        result["loglik"] = loglik.cpu()
        rmse = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu()
        result["rmse"] = rmse

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            result["gt_loglik"] = gt_loglik.cpu()

        # Handles temperature prediction case - distribution must be normal but is for the models we use.
        if isinstance(batch, HadISDBatch) or isinstance(batch, TemporalHadISDBatch):
            # Reconstructs mean and variance vectors to be the correct units
            pred_dist_temp = scale_pred_temp_dist(batch, pred_dist)
            yt_correct_units = get_true_temp(batch, batch.yt)

            # Computes track statistics
            loglik_temp = (
                pred_dist_temp.log_prob(yt_correct_units).sum()
                / yt_correct_units[..., 0].numel()
            )
            rmse_temp = (
                nn.functional.mse_loss(pred_dist_temp.mean, yt_correct_units)
                .sqrt()
                .cpu()
                .mean()
            )

            result["loglik_temp"] = loglik_temp
            result["rmse_temp"] = rmse_temp

        self.test_outputs.append(result)

    def on_validation_epoch_end(self) -> None:
        if len(self.val_batches) == 0:
            return

        if (
            self.plot_fn is not None
            and (self.current_epoch + 1) % self.plot_interval == 0
        ):
            self.plot_fn(
                self.model, self.val_batches, f"epoch-{self.current_epoch:04d}"
            )

        self.val_batches = []

    def configure_optimizers(self):
        # If no scheduler was provided, return the optimizer directly.
        if self.lr_scheduler is None:
            return self.optimiser

        # Otherwise return the optimizer + scheduler in the format Lightning expects.
        return {
            "optimizer": self.optimiser,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class LogPerformanceCallback(pl.Callback):
    def __init__(self):
        super().__init__()

        self.start_time = 0.0
        self.last_batch_end_time = 0.0
        self.update_count = 0.0
        self.backward_start_time = 0.0
        self.forward_start_time = 0.0
        self.between_step_time = 0.0

    @pl.utilities.rank_zero_only
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @pl.utilities.rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        pl_module.log(
            "performance/between_step_time",
            time.time() - self.between_step_time,
            on_step=True,
            on_epoch=False,
        )
        self.forward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: torch.Tensor,
    ):
        super().on_before_backward(trainer, pl_module, loss)
        forward_time = time.time() - self.forward_start_time
        pl_module.log(
            "performance/forward_time",
            forward_time,
            on_step=True,
            on_epoch=False,
        )
        self.backward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_after_backward(trainer, pl_module)
        backward_time = time.time() - self.backward_start_time
        pl_module.log(
            "performance/backward_time",
            backward_time,
            on_step=True,
            on_epoch=False,
        )

    @pl.utilities.rank_zero_only
    def on_train_epoch_start(self, *args, **kwargs) -> None:
        super().on_train_epoch_start(*args, **kwargs)
        self.update_count = 0.0
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @pl.utilities.rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.update_count += 1

        # Calculate total elapsed time
        total_elapsed_time = time.time() - self.start_time
        last_elapsed_time = time.time() - self.last_batch_end_time
        self.last_batch_end_time = time.time()

        # Calculate updates per second
        average_updates_per_second = self.update_count / total_elapsed_time
        last_updates_per_second = 1 / last_elapsed_time

        # Log updates per second to wandb using pl_module.log
        pl_module.log(
            "performance/average_updates_per_second",
            average_updates_per_second,
            on_step=True,
            on_epoch=False,
        )
        pl_module.log(
            "performance/last_updates_per_second",
            last_updates_per_second,
            on_step=True,
            on_epoch=False,
        )
        self.between_step_time = time.time()


class WandbCheckpointCallback(pl.Callback):
    def __init__(self, logger, n_epochs=10, monitor="val/loglik", mode="max"):
        self.logger = logger
        self.n_epochs = n_epochs
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_model_path = None
        self.saved_checkpoints = []  # Keep track of saved checkpoint files

    @pl.utilities.rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # Only save if we have a logger and are logging
        if not self.logger or not hasattr(self.logger, "experiment"):
            return

        current_epoch = trainer.current_epoch

        # Check for best model
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            # Convert to float if it's a tensor
            if hasattr(current_score, "item"):
                current_score = current_score.item()

            # Check if this is the best score
            is_better = (self.mode == "min" and current_score < self.best_score) or (
                self.mode == "max" and current_score > self.best_score
            )

            if is_better:
                self.best_score = current_score

                # Clean up previous best checkpoint file if it exists
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)

                # Save new best checkpoint
                self.best_model_path = f"best_model_epoch_{current_epoch:04d}.ckpt"
                trainer.save_checkpoint(self.best_model_path)
                print(
                    f"New best model saved: {self.monitor}={current_score:.4f} at epoch {current_epoch}"
                )

        # Check if we should save periodic checkpoint
        if (current_epoch + 1) % self.n_epochs == 0:  # +1 because epochs are 0-indexed
            # Save checkpoint
            checkpoint_path = f"model_epoch_{current_epoch:04d}.ckpt"
            trainer.save_checkpoint(checkpoint_path)

            # Get run ID for artifact naming
            run_id = (
                self.logger.experiment.id
                if hasattr(self.logger.experiment, "id")
                else "unknown"
            )

            # Create and log artifact
            artifact = wandb.Artifact(
                name=f"model-epoch-{current_epoch:04d}-{run_id}",
                type="model",
                description=f"Model checkpoint at epoch {current_epoch}",
            )
            artifact.add_file(checkpoint_path)
            self.logger.experiment.log_artifact(artifact)

            # Keep track of this checkpoint for cleanup
            self.saved_checkpoints.append(checkpoint_path)

            print(f"Checkpoint saved and logged to W&B at epoch {current_epoch}")

    @pl.utilities.rank_zero_only
    def on_train_end(self, trainer, pl_module):
        # Only save if we have a logger and are logging
        if not self.logger or not hasattr(self.logger, "experiment"):
            return

        # Save the best model as artifact with "_best" suffix
        if self.best_model_path and os.path.exists(self.best_model_path):
            run_id = (
                self.logger.experiment.id
                if hasattr(self.logger.experiment, "id")
                else "unknown"
            )
            artifact = wandb.Artifact(
                name=f"model-best-{run_id}_best",
                type="model",
                description=f"Best model with {self.monitor}={self.best_score:.4f}",
            )
            artifact.add_file(self.best_model_path)
            self.logger.experiment.log_artifact(artifact)
            print(f"Best model logged to W&B: {self.monitor}={self.best_score:.4f}")

        # Clean up all saved checkpoint files (including best model)
        for checkpoint_path in self.saved_checkpoints:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)

        total_cleaned = len(self.saved_checkpoints) + (1 if self.best_model_path else 0)
        print(f"Cleaned up {total_cleaned} checkpoint files")


def _batch_to_cpu(batch: Batch):
    batch_kwargs = {
        field.name: (
            getattr(batch, field.name).cpu()
            if isinstance(getattr(batch, field.name), torch.Tensor)
            else getattr(batch, field.name)
        )
        for field in dataclasses.fields(batch)
    }
    return type(batch)(**batch_kwargs)
