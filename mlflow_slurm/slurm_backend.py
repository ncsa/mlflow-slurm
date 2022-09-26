import re
import shlex
import subprocess
import time
from threading import RLock
from pathlib import Path
from typing import Tuple, List

from jinja2 import Environment, BaseLoader
from mlflow.utils.logging_utils import _configure_mlflow_loggers

from mlflow import tracking

import mlflow
import logging
from mlflow.entities import RunStatus
from mlflow.projects.utils import (
    fetch_and_validate_project, get_or_create_run,
    PROJECT_STORAGE_DIR, PROJECT_ENV_MANAGER, get_entry_point_command
)
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects import load_project, env_type
from mlflow.exceptions import ExecutionException
from mlflow.tracking import MlflowClient
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import get_or_create_conda_env, get_conda_command
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.mlflow_tags import MLFLOW_PROJECT_ENV
from mlflow.utils.virtualenv import _install_python, _get_mlflow_virtualenv_root, \
    _get_virtualenv_name, _create_virtualenv

_configure_mlflow_loggers(root_module_name=__name__)
_logger = logging.getLogger(__name__)
_logger.setLevel("DEBUG")


def slurm_backend_builder() -> AbstractBackend:
    return SlurmProjectBackend()


class SlurmSubmittedRun(SubmittedRun):
    """Instance of SubmittedRun
       corresponding to a Slum Job launched through pySlurm to run an MLflow
       project.
    :param slurm_job_id: ID of the submitted Slurm Job.
    :param mlflow_run_id: ID of the MLflow project run.
    """
    def __init__(self, mlflow_run_id: str, slurm_job_id: str) -> None:
        super().__init__()
        self._mlflow_run_id = mlflow_run_id
        self.slurm_job_id = slurm_job_id
        self._status = RunStatus.SCHEDULED
        self._status_lock = RLock()

    # How often to poll run status when waiting on a run
    POLL_STATUS_INTERVAL = 5

    @property
    def run_id(self) -> str:
        return self._mlflow_run_id

    def wait(self):
        while not RunStatus.is_terminated(self._update_status()):
            time.sleep(self.POLL_STATUS_INTERVAL)

        return self._status == RunStatus.FINISHED

    def cancel(self) -> None:
        print("Cancel")

    def get_status(self) -> RunStatus:
        return self._status

    def _update_status(self) -> RunStatus:
        with subprocess.Popen(f"squeue --job={self.slurm_job_id} -o%A,%t",
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, shell=True,
                              universal_newlines=True) as p:
            p.wait()
            output = p.stdout.read().split('\n')
            job = output[1]  # First line is header
            if not job:
                _logger.warning(f"Looking for status of job {self.slurm_job_id}, but it is gone")
                return None

            job_status = output[1].split(",")[1]
            if job_status == "PD":
                return RunStatus.SCHEDULED
            elif job_status == "CD":
                return RunStatus.FINISHED
            elif job_status == "F":
                return RunStatus.FAILED
            elif job_status == "R" \
                    or job_status == "S" \
                    or job_status == "ST" \
                    or job_status == "CG" \
                    or job_status == "PR":
                return RunStatus.RUNNING
            else:
                _logger.warning(f"Job ID {self.slurm_job_id} has an unmapped status of {job_status}")
                return None


class SlurmProjectBackend(AbstractBackend):
    @staticmethod
    def sbatch(script: str):
        job_re = "Submitted batch job (\\d+)"
        with subprocess.Popen(f"sbatch {script}",
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, shell=True) as p:
            return_code = p.wait()
            if return_code == 0:
                sbatch_output = p.stdout.read().decode('utf-8')
                match = re.search(job_re, sbatch_output)
                if not match:
                    print(f"Couldn't parse batch output: {sbatch_output}")
                    return None
                else:
                    return match.group(1)
            else:
                sbatch_err = p.stderr.read().decode('utf-8')
                print(f"SBatch Error:{sbatch_err}")
                return None

    def run(self, project_uri, entry_point, params, version, backend_config, tracking_uri,
            experiment_id):

        work_dir = fetch_and_validate_project(project_uri, version, entry_point, params)
        active_run = get_or_create_run(None, project_uri, experiment_id, work_dir, version,
                                       entry_point, params)
        _logger.info(f"run_id={active_run.info.run_id}")
        _logger.info(f"work_dir={work_dir}")

        project = load_project(work_dir)

        storage_dir = backend_config[PROJECT_STORAGE_DIR]

        entry_point_command = project.get_entry_point(entry_point) \
            .compute_command(params, storage_dir)

        _logger.info(f"entry_point_command={entry_point_command}")

        command_args = []
        command_separator = " "
        env_manager = backend_config[PROJECT_ENV_MANAGER]
        storage_dir = backend_config[PROJECT_STORAGE_DIR]

        if project.env_type == "python_env":
            tracking.MlflowClient().set_tag(
                active_run.info.run_id, MLFLOW_PROJECT_ENV, "virtualenv"
            )
            command_separator = " && "
            python_env = _PythonEnv.from_yaml(project.env_config_path)
            python_bin_path = _install_python(python_env.python)
            env_root = _get_mlflow_virtualenv_root()
            work_dir_path = Path(work_dir)
            env_name = _get_virtualenv_name(python_env, work_dir_path)
            env_dir = Path(env_root).joinpath(env_name)
            activate_cmd = _create_virtualenv(work_dir_path, python_bin_path, env_dir, python_env)
            command_args += [activate_cmd]
        elif project.env_type== "conda_env":
            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, "conda")
            command_separator = " && "
            conda_env_name = get_or_create_conda_env(project.env_config_path)
            command_args += get_conda_command(conda_env_name)

        command_args += get_entry_point_command(project, entry_point, params, storage_dir)
        command_str = command_separator.join(command_args)

        job_template = """#!/bin/bash
#SBATCH --job-name=MLFlow{{ run_id }}
#SBATCH --partition={{ config.partition }}
#SBATCH --account={{ config.account }}
#SBATCH --export=MLFLOW_TRACKING_URI,MLFLOW_S3_ENDPOINT_URL,AWS_SECRET_ACCESS_KEY,AWS_ACCESS_KEY_ID
{% if config.gpus_per_node %}
#SBATCH --gpus-per-node={{ config.gpus_per_node }}
{% endif %}
{% for module in config.modules %}
module load {{ module }}
{% endfor %}


{{ command }}
        """
        template = Environment(loader=BaseLoader()).from_string(job_template)
        with open("generated.sh", "w") as text_file:
            text_file.write(template.render(command=command_str,
                                            config=backend_config,
                                            run_id=active_run.info.run_id))

        job_id = SlurmProjectBackend.sbatch("generated.sh")
        MlflowClient().set_tag(active_run.info.run_id, "slurm_job_id", job_id)

        return SlurmSubmittedRun(active_run.info.run_id, job_id)

    def __init__(self):
        pass


def try_split_cmd(cmd: str) -> Tuple[str, List[str]]:
    parts = []
    found_python = False
    for part in shlex.split(cmd):
        if part == "-m":
            continue
        elif not found_python and part.startswith("python"):
            found_python = True
            continue
        parts.append(part)
    entry_point = ""
    args = []
    if len(parts) > 0:
        entry_point = parts[0]
    if len(parts) > 1:
        args = parts[1:]
    return entry_point, args
