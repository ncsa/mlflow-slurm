import os
import pkgutil

import click

import mlflow
from mlflow.projects.backend import loader


def eval(num_runs, experiment_id):
    alpha = 0.4
    momentum = 0.2

    with mlflow.start_run(nested=True) as child_run:
        x = [{
            'path': m.module_finder.path,
            'name': m.name,
        } for m in list(pkgutil.iter_modules())]
        print(x)
        p = mlflow.projects.run(
            run_id=child_run.info.run_id,
            uri=os.path.dirname(os.path.realpath(__file__)),
            entry_point="train",
            parameters={
                "alpha": str(alpha),
                "l1_ratio": str(momentum),
            },
            experiment_id=experiment_id,
            synchronous=False,
            backend="slurm",
            backend_config="slurm_config.json"
        )
        succeeded = p.wait()
        print(succeeded)
        return succeeded


@click.command(help="Perform grid search over train (main entry point).")
@click.option("--num-runs", type=click.INT, default=32, help="Maximum number of runs to evaluate.")
def run(num_runs):
    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        result = eval(num_runs, experiment_id)
        print("Result ", result)


if __name__ == "__main__":
    run()