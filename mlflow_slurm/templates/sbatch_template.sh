#!/bin/bash
#SBATCH --job-name=MLFlow{{ run_id }}
{% if config.partition %}
#SBATCH --partition={{ config.partition }}
{% endif %}
{% if config.account %}
#SBATCH --account={{ config.account }}
{% endif %}
#SBATCH --export=MLFLOW_TRACKING_URI,MLFLOW_TRACKING_TOKEN,MLFLOW_TRACKING_USERNAME,MLFLOW_TRACKING_PASSWORD,MLFLOW_S3_ENDPOINT_URL,AWS_SECRET_ACCESS_KEY,AWS_ACCESS_KEY_ID
{% if config.gpus_per_node %}
#SBATCH --gpus-per-node={{ config.gpus_per_node }}
{% endif %}
{% if config.gres %}
#SBATCH --gres={{ config.gres }}
{% endif %}
{% if config.mem %}
#SBATCH --mem={{ config.mem }}
{% endif %}
{% if config.time %}
#SBATCH --time={{ config.time }}
{% endif %}
module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
{% for module in config.modules %}
module load {{ module }}
{% endfor %}
module list  # job documentation and metadata
export MLFLOW_RUN_ID={{ run_id }}
echo "job is starting on `hostname`"
{{ command }}
