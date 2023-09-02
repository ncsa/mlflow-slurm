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
|partition          | Which Slurm partition should the job run in?                                                                   |
 |account            | What account name to run under                                                                                 |
| gpus_per_node     | On GPU partitions how many GPUs to allocate per node                                                           |
| gres              | SLURM Generic RESources requests                                                                               |
| mem               | Amount of memory to allocate to CPU jobs                                                                       |
| modules           | List of modules to load before starting job                                                                    |
| time              | Max CPU time job may run                                                                                       |
| sbatch-script-file | Name of batch file to be produced. Leave blank to have service generate a script file name based on the run ID |

## Development
The slurm docker deployment is handy for testing and development. You can start
up a slurm environment with the included docker-compose file

