import os

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from plot import plot
from plot_hadISD_temporal import plot_hadISD_temporal
import numpy as np
import wandb
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import create_lr_scheduler, initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback
from tnp.data.hadISDTemporal import TemporalHadISDDataGenerator
from eval import test_model
from lightning.pytorch.utilities import rank_zero_only
from data_temp.data_processing.elevations import get_cached_elevation_grid
from tnp.models.tnpa import TNPA
from tnp.utils.ram_data_loader import RamDiskDataLoader


def main():
    experiment = initialize_experiment()
    # Training code
    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    # Setup learning rate scheduler
    lr_scheduler = create_lr_scheduler(optimiser, experiment, gen_train)
    epochs = experiment.params.epochs

    is_hadISD_train = False
    is_hadTemporal_train = isinstance(gen_train, TemporalHadISDDataGenerator)

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
        pin_memory=True,
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
        pin_memory=True,
    )

    def plot_fn_gp(model, batches, name):
        is_training = model.training
        model.eval()
        # Calculates plot range
        min_tgt, max_tgt = np.array(experiment.params.target_range).min(), np.array(experiment.params.target_range).max()
        plot(
            model=model,
            batches=batches,
            num_fig=min(5, len(batches)),
            name=name,
            pred_fn=experiment.misc.pred_fn,
            x_range = (min_tgt, max_tgt)
        )
        if is_training: model.train()

    # Sets up elevation caching if appropriate
    if is_hadISD_train or is_hadTemporal_train:
        lat_mesh, lon_mesh, elev_np = get_cached_elevation_grid(gen_train.lat_range, gen_train.long_range,
            experiment.misc.num_grid_points_plot, experiment.misc.cache_dem_dir,
            experiment.misc.dem_path)

    def plot_fn_hadISDTemporal(model, batches, name):
        is_training = model.training
        huge_grid_plots = False if isinstance(model, TNPA) else True # TNPA models just too expensive to do the 40 k grid point plots every epoch on CBL GPUs
        #huge_grid_plots = False
        model.eval()
        plot_hadISD_temporal(
            model=model,
            batches=batches,
            num_fig=min(5, len(batches)),
            name=name,
            pred_fn=experiment.misc.pred_fn,
            lat_mesh=lat_mesh,
            lon_mesh=lon_mesh,
            elev_np=elev_np,
            huge_grid_plots=huge_grid_plots,
            delta_hours=int(experiment.generators.val.delta_hours),
        )
        if is_training: model.train()

    plot_fn = None if is_hadISD_train else (plot_fn_hadISDTemporal if is_hadTemporal_train else plot_fn_gp)

    if experiment.misc.resume_from_checkpoint is not None:
        api = wandb.Api()
        artifact = api.artifact(experiment.misc.resume_from_checkpoint)
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")

        lit_model = (
            LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                ckpt_file,
            )
        )
    else:
        ckpt_file = None
        lit_model = LitWrapper(
            model=model,
            optimiser=optimiser,
            lr_scheduler=lr_scheduler,
            loss_fn=experiment.misc.loss_fn,
            pred_fn=experiment.misc.pred_fn,
            plot_fn=plot_fn,
            plot_interval=experiment.misc.plot_interval,
        )


    if experiment.misc.logging:
        logger = pl.loggers.WandbLogger(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=OmegaConf.to_container(experiment.config),
            log_model="all",
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=experiment.misc.checkpoint_interval,
            save_last=True,
        )
        performance_callback = LogPerformanceCallback()
        # Ensures model is fully evaluated at train end with callback
        class TestAfterTrainingCallback(pl.Callback):
            def __init__(self, experiment):
                super().__init__()
                self.experiment = experiment

            @rank_zero_only
            def on_fit_end(self, trainer, pl_module):
                print("Running final test on model")
                test_loader = torch.utils.data.DataLoader(
                    experiment.generators.test,
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
                wandb_run = trainer.logger.experiment
                test_model(pl_module, self.experiment, wandb_run=wandb_run)

        callbacks = [checkpoint_callback, performance_callback, TestAfterTrainingCallback(experiment)]

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
    torch.set_float32_matmul_precision("high")
    main()
