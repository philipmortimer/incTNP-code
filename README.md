# Incremental Transformer Neural Processes

This codebase contains the model implementations, training pipelines, and evaluation scripts for our work on Incremental Transformer Neural Processes.

## Environment Setup
To set up the environment, run the following commands:

```bash
conda env create -f environment.yml
conda activate inc_tnp
pip install -e .

```

## Models

| Model | Implementation | Configuration |
| --- | --- | --- |
| **incTNP** | `tnp/models/castnp.py` | `experiments/configs/models/inctnp.yml` |
| **incTNP-Seq** | `tnp/models/incTNPBatched.py` | `experiments/configs/models/inctnpSeq.yml` |
| **TNP-D** | `tnp/models/tnp.py` | `experiments/configs/models/tnp.yml` |

## GP Training

To run the training pipeline:

```bash
python experiments/train.py --config experiments/configs/synthetic1dRBF/{MODEL_CONFIG}

```

Example (training incTNP):

```bash
python experiments/train.py --config experiments/configs/synthetic1dRBF/gp_causal_tnp_lr_scheduler_rangesame.yaml

```

## GP Plotting

To generate plots with trained models:

```bash
python experiments/gp_plots.py

```

## Tabular Training

To train on tabular datasets:

```bash
python experiments/lightning_train.py --config experiments/configs/generators/tabular_data.yaml experiments/configs/tabular_data/{MODEL_CONFIG}

```

Example (training incTNP-Seq):

```bash
python experiments/lightning_train.py --config experiments/configs/generators/tabular_data.yaml experiments/configs/tabular_data/tab_batched_causal_tnp_lr_scheduler.yaml

```

## HadISD

Due to file size limitations, we do not redistribute the HadISD dataset in this repository. However, the training code is included to outline task details and data processing logic.

## WISKI

We utilize the WISKI library from [online_gp](https://github.com/wjmaddox/online_gp), which is cloned in the `online_gp` folder. This library must be installed separately to run experiments involving WISKI.

**Note:** The WISKI experiments require a different environment than the main `environment.yml`. Specifically, we run WISKI code sections with **Python 3.11** and **GPyTorch 1.8.1**. The script `experiments/exchangeability.py` computes the measure of implicit Bayesianness.

## Other

This submission includes a range of additional scripts and model implementations.

Portions of this codebase are adapted from an MIT-licensed implementation of Transformer Neural Processes.

## Verified Environment
This codebase (speficially the `environment.yml` install) has been verified to run on the following setup:
* **OS:** Ubuntu 24.04.3 LTS
* **Python:** 3.12.12
* **PyTorch:** 2.10.0+cu128 (CUDA 12.8)
* **GPU:** NVIDIA GeForce RTX 2080 Ti
* **CPU:** x86_64 architecture

### WSL Troubleshooting
The codebase has also been verified on **Ubuntu 22.04.4 LTS via WSL2**.

If you encounter the following error:
> `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.31' not found`

You can resolve this by installing the required library and exporting the path:

```bash
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Codebase
This implementation builds upon the original [TNP codebase](https://github.com/cambridge-mlg/tnp) developed by Matthew Ashman and Cristiana Diaconu.

*Note: This `incTNP` repository is the initial code release for our preprint. We intend to further polish, document, and refine the codebase in the near future.*

