# Example of Sequential Worker Jobs
This simple example shows how to use the `sequential_workers` parameter to submit a job that will be split into multiple jobs that depend on each other.

```shell
mlflow run --backend slurm -c ../../slurm_config.json -P sequential_workers=3 .
```

Each job appends to a file named `restart.log` with the time the job is run.
MLFlow will submit three jobs that depend on each other. As soon as the first job terminates, the next job will start. This will continue until all jobs have completed.

When the jobs are complete, you can check the `restart.log` file to see the order in which the jobs were run.
