from plot import plot

import wandb
from tnp.utils.experiment_utils import initialize_evaluation


def main():
    experiment = initialize_evaluation()

    model = experiment.model
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.test

    model.eval()

    gen_test.batch_size = 1
    gen_test.num_batches = experiment.misc.num_plots
    batches = list(iter(gen_test))

    eval_name = wandb.run.name + "/" + eval_name
    plot(
        model=model,
        batches=batches,
        num_fig=min(experiment.misc.num_plots, len(batches)),
        name=eval_name,
        savefig=experiment.misc.savefig,
        logging=experiment.misc.logging,
        pred_fn=experiment.misc.pred_fn,
    )


if __name__ == "__main__":
    main()
