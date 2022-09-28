#!/bin/bash
#SBATCH --job-name=MLFlow{{ run_id }}
#SBATCH --partition={{ config.partition }}
#SBATCH --account={{ config.account }}
#SBATCH --export=MLFLOW_TRACKING_URI,MLFLOW_S3_ENDPOINT_URL,AWS_SECRET_ACCESS_KEY,AWS_ACCESS_KEY_ID
{% if config.gpus_per_node %}
#SBATCH --gpus-per-node={{ config.gpus_per_node }}
{% endif %}
{% if config.mem %}
#SBATCH --mem={{ config.mem }}
{% endif %}
{% for module in config.modules %}
module load {{ module }}
{% endfor %}

{{ command }}