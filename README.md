# MLFlow-Slurm
Backend for executing MLFlow projects on Slurm batch system

## Usage
Install this package in the environment from which you will be submitting jobs.
If you are submitting jobs from inside jobs, make sure you have this package
listed in your conda or pip environment.

Just list this as your `--backend` in the job run. You should include a json
config file to control how the batch script is constructed:
```shell
mlflow run --backend slurm \
          --backend-config slurm_config.json \
          examples/sklearn_elasticnet_wine
```

It will generate a batch script named after the job id and submit it via the
Slurm `sbatch` command. It will tag the run with the Slurm JobID

## Configure Jobs
You can set values in a json file to control job submission. The supported
properties in this file are:

|Config File Setting| Use                                                                                                            |
|-------------------|----------------------------------------------------------------------------------------------------------------|
| partition         | Which Slurm partition should the job run in?                                                                   |
| environment       | List of additional environment variables to add to the job                                                     |
| exports           | List of environment variables to export to the job                                                             |
| gpus_per_node     | On GPU partitions how many GPUs to allocate per node                                                           |
| gres              | SLURM Generic RESources requests                                                                               |
| mem               | Amount of memory to allocate to CPU jobs                                                                       |
| modules           | List of modules to load before starting job                                                                    |
| nodes             | Number of nodes to request from SLURM                                                                          |
| time              | Max CPU time job may run                                                                                       |
| sbatch-script-file | Name of batch file to be produced. Leave blank to have service generate a script file name based on the run ID |

## Sequential Worker Jobs
There are occaisions where you have a job that can't finish in the maxiumum
allowable wall time. If you are able to write out a checkpoint file, you can
use sequential worker jobs to continue the job where it left off. This is
useful for training deep learning models or other long running jobs.

To use this, you just need to provide a parameter to the `mlflow run` command
```shell
  mlflow run --backend slurm -c ../../slurm_config.json -P sequential_workers=3 .
```
This will the submit the job as normal, but also submit 3 additional jobs that
each depend on the previous job. As soon as the first job terminates, the next
job will start. This will continue until all jobs have completed.

## Development
The slurm docker deployment is handy for testing and development. You can start
up a slurm environment with the included docker-compose file
