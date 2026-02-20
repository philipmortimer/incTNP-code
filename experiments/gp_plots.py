# Simple script to plot GP kernel performance on some examples
from plot_adversarial_perms import get_model
from plot import plot
from tnp.networks.gp import RBFKernel
from functools import partial
from tnp.data.gp import RandomScaleGPGenerator
from typing import List
import lightning.pytorch as pl
import os


if __name__ == "__main__":
    # Loads RBF data
    pl.seed_everything(1) # Sets seed
    plots = 5 # Number of plots
    min_nc, max_nc = 1, 64
    min_nt, max_nt = 128, 128
    samples_per_epoch=plots
    batch_size=1

    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    noise_std=0.1
    deterministic = True

    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                            max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=min_nt, max_nt=max_nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    plot_batches = []
    for data in gen_val: plot_batches.append(data)

    
    base_folder = "experiments/plot_results" # Where to store plot results
    os.makedirs(base_folder, exist_ok=True)
    # List of pretrained models
    tnp_plain = ['experiments/configs/synthetic1dRBF/best_trained/tnpd.yaml', 'experiments/configs/synthetic1dRBF/best_trained/weights/tnpd.ckpt', "TNP-D"]
    inc_tnp = ['experiments/configs/synthetic1dRBF/best_trained/incTNP.yaml', 'experiments/configs/synthetic1dRBF/best_trained/weights/incTNP.ckpt', "incTNP"]
    inc_tnp_seq=['experiments/configs/synthetic1dRBF/best_trained/incTNP-Batched.yaml', 'experiments/configs/synthetic1dRBF/best_trained/weights/incTNP-Batched.ckpt', "incTNP-Seq"]

    # Plotting configs and procedure
    def plot_model(model):
        model_np = get_model(model[0], model[1], local_weights=True, device="cpu") # May need to change device var
        model_np.eval()
        out_folder=f"{base_folder}/{model[2]}"
        plot(model=model_np, batches=plot_batches, num_fig=plots, name=out_folder, savefig=True, logging=False, model_lbl=model[2], x_range=(-2.0, 2.0))

    plot_model(tnp_plain)
    plot_model(inc_tnp)
    plot_model(inc_tnp_seq)