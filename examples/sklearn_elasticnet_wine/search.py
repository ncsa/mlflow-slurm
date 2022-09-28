import os
import click
import mlflow

tracking_client = mlflow.tracking.MlflowClient()


def run_train(experiment_id, alpha, l1_ratio, backend_config="slurm_config.json"):
    with mlflow.start_run(nested=True) as child_run:
        p = mlflow.projects.run(
            run_id=child_run.info.run_id,
            uri=os.path.dirname(os.path.realpath(__file__)),
            entry_point="train",
            parameters={
                "alpha": str(alpha),
                "l1_ratio": str(l1_ratio),
            },
            experiment_id=experiment_id,
            synchronous=False,
            backend="slurm",
            backend_config=backend_config
        )
        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
        return p


@click.command(help="Perform grid search over train (main entry point).")
@click.option("--num-runs", type=click.INT, default=32, help="Maximum number of runs to evaluate.")
@click.option("--train-backend-config", type=click.STRING, default="slurm_config.json", help="Json file for training jobs")
def run(num_runs, train_backend_config):
    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        p = run_train(experiment_id, alpha=0.4, l1_ratio=0.1, backend_config=train_backend_config)
        success = p.wait()
        print("Result is ", success)



if __name__ == "__main__":
    run()