import os

import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from plot import plot

import wandb
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import create_lr_scheduler, initialize_experiment
from tnp.utils.lightning_utils import (
    LitWrapper,
    LogPerformanceCallback,
    WandbCheckpointCallback,
)


def main():
    experiment = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    # Setup learning rate scheduler
    lr_scheduler = create_lr_scheduler(optimiser, experiment, gen_train)
    epochs = experiment.params.epochs

    train_loader = torch.utils.data.DataLoader(
        gen_train,
        batch_size=None,
        num_workers=experiment.misc.num_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_workers > 0 else False,
        pin_memory=False,
        prefetch_factor=(2 if experiment.misc.num_workers > 0 else None),
    )
    val_loader = torch.utils.data.DataLoader(
        gen_val,
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
        pin_memory=False,
        prefetch_factor=(2 if experiment.misc.num_workers > 0 else None),
    )

    def plot_fn(model, batches, name):
        plot(
            model=model,
            batches=batches,
            num_fig=min(5, len(batches)),
            name=name,
            pred_fn=experiment.misc.pred_fn,
        )

    if experiment.misc.resume_from_checkpoint is not None:
        api = wandb.Api()
        artifact = api.artifact(experiment.misc.resume_from_checkpoint)
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")

        lit_model = LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
            ckpt_file,
        )
    else:
        ckpt_file = None
        lit_model = LitWrapper(
            model=model,
            optimiser=optimiser,
            lr_scheduler=lr_scheduler,
            loss_fn=experiment.misc.loss_fn,
            pred_fn=experiment.misc.pred_fn,
            plot_fn=plot_fn
            if hasattr(experiment.misc, "plot_fn")
            and experiment.misc.plot_fn is not None
            else None,
            plot_interval=experiment.misc.plot_interval,
        )

    if experiment.misc.logging:
        logger = pl.loggers.WandbLogger(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=OmegaConf.to_container(experiment.config),
            log_model=False,
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=experiment.misc.checkpoint_interval,
            filename="tnp-{epoch:02d}-{val/loglik:.4f}-{val/rmse:.4f}",
            monitor="val/loglik",
            save_top_k=3,
            mode="max",
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False,
        )
        # Custom W&B artifact callback
        wandb_checkpoint_callback = WandbCheckpointCallback(
            logger=logger,
            monitor="val/loglik",
            mode="max",
            n_epochs=experiment.misc.checkpoint_interval,
        )
        performance_callback = LogPerformanceCallback()
        callbacks = [
            checkpoint_callback,
            performance_callback,
            wandb_checkpoint_callback,
        ]
    else:
        logger = False
        callbacks = None

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        log_every_n_steps=(
            experiment.misc.log_interval if not experiment.misc.logging else None
        ),
        devices="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=(experiment.misc.check_val_every_n_epoch),
        gradient_clip_val=experiment.misc.gradient_clip_val,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_file,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("high")
    main()