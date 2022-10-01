import os
import click
import numpy as np

import mlflow
from mlflow.entities import Param, RunTag
from mlflow.tracking import MlflowClient

tracking_client = mlflow.tracking.MlflowClient()


def run_train(experiment_id, alpha, l1_ratio, backend_config="slurm_config.json", parent_run_id=None):
    p = mlflow.projects.run(
        uri=os.path.dirname(os.path.realpath(__file__)),
        entry_point="train",
        parameters={
            "alpha": str(alpha),
            "l1_ratio": str(l1_ratio)
        },
        experiment_id=experiment_id,
        synchronous=False,
        backend="slurm",
        backend_config=backend_config
    )
    MlflowClient().log_batch(run_id=p.run_id, metrics=[],
                             params=[Param("alpha", str(alpha)), Param("alpha", str(alpha))],
                             tags=[RunTag(mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID, parent_run_id)])

    return p


@click.command(help="Perform grid search over train (main entry point).")
@click.option("--num-runs", type=click.INT, default=2, help="Maximum number of runs to evaluate.")
@click.option("--train-backend-config", type=click.STRING, default="slurm_config.json", help="Json file for training jobs")
def run(num_runs, train_backend_config):
    provided_run_id = os.environ.get("MLFLOW_RUN_ID", None)
    with mlflow.start_run(run_id=provided_run_id) as run:
        print("Search is run_id ", run.info.run_id)
        experiment_id = run.info.experiment_id
        runs = [(np.random.uniform(1e-5, 1e-1), np.random.uniform(0, 1.0)) for _ in range(num_runs)]
        jobs = []
        for alpha, ll_ratio in runs:
            jobs.append(run_train(
                experiment_id,
                alpha=alpha, l1_ratio=ll_ratio,
                backend_config=train_backend_config,
                parent_run_id=provided_run_id)
            )
        results = map(lambda job: job.wait(), jobs)
        print(list(results))


if __name__ == "__main__":
    run()
