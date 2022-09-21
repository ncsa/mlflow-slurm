import pyslurm
import shlex
from pathlib import Path
from typing import Tuple, List

from jinja2 import Environment, BaseLoader
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

_logger = logging.getLogger(__name__)


def slurm_backend_builder() -> AbstractBackend:
    return SlrumProjectBackend()

class SlurmSubmittedRun(SubmittedRun):
    """Instance of SubmittedRun
       corresponding to a Slum Job launched through pySlurm to run an MLflow
       project.
    :param skein_app_id: ID of the submitted Skein Application.
    :param mlflow_run_id: ID of the MLflow project run.
    """
    def __init__(self, mlflow_run_id: str) -> None:
        super().__init__()
        self._mlflow_run_id = mlflow_run_id

    @property
    def run_id(self) -> str:
        return self._mlflow_run_id

    def wait(self) -> bool:
        print("Wating...")
        return False

    def cancel(self) -> None:
        print("Cancel")

    def get_status(self) -> RunStatus:
        print("Get status")
        return RunStatus.SCHEDULED



class SlrumProjectBackend(AbstractBackend):
    def run(self, project_uri, entry_point, params, version, backend_config, tracking_uri,
            experiment_id):
        print("Ready to sluuuu")
        _logger.warning("Ready to Slurm")
        work_dir = fetch_and_validate_project(project_uri, version, entry_point, params)
        active_run = get_or_create_run(None, project_uri, experiment_id, work_dir, version,
                                       entry_point, params)
        _logger.warning(f"run_id={active_run.info.run_id}")
        _logger.warning(f"work_dir={work_dir}")
        project = load_project(work_dir)
        print(project)

        storage_dir = backend_config[PROJECT_STORAGE_DIR]

        entry_point_command = project.get_entry_point(entry_point) \
            .compute_command(params, storage_dir)

        _logger.warn(f"entry_point_command={entry_point_command}")

        env = {
            "MLFLOW_RUN_ID": active_run.info.run_id,
            "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri(),
            "MLFLOW_EXPERIMENT_ID": experiment_id
        }
        command_args = []
        command_separator = " "
        env_manager = backend_config[PROJECT_ENV_MANAGER]
        storage_dir = backend_config[PROJECT_STORAGE_DIR]

        if project.env_type == "python_env":
            tracking.MlflowClient().set_tag(
                active_run.info.run_id, MLFLOW_PROJECT_ENV, "virtualenv"
            )
            command_separator = " && "
            if project.env_type == env_type.CONDA:
                python_env = _PythonEnv.from_conda_yaml(project.env_config_path)
            else:
                python_env = _PythonEnv.from_yaml(project.env_config_path)
            python_bin_path = _install_python(python_env.python)
            env_root = _get_mlflow_virtualenv_root()
            work_dir_path = Path(work_dir)
            env_name = _get_virtualenv_name(python_env, work_dir_path)
            env_dir = Path(env_root).joinpath(env_name)
            activate_cmd = _create_virtualenv(work_dir_path, python_bin_path, env_dir, python_env)
            command_args += [activate_cmd]
        elif env_manager == _EnvManager.CONDA:
            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, "conda")
            command_separator = " && "
            conda_env_name = get_or_create_conda_env(project.env_config_path)
            command_args += get_conda_command(conda_env_name)



        command_args += get_entry_point_command(project, entry_point, params, storage_dir)
        command_str = command_separator.join(command_args)

        job_template = """#!/bin/bash
{{ command }}
        """
        template = Environment(loader=BaseLoader()).from_string(job_template)
        with open("generated.sh", "w") as text_file:
            text_file.write(template.render(command=command_str))

        jobs = pyslurm.job()
        j = jobs.submit_batch_job({
            'script': "generated.sh",
            'job_name': "pySlurm",
            'partition': 'normal',
            'export': 'MLFLOW_TRACKING_URI,MLFLOW_S3_ENDPOINT_URL,AWS_SECRET_ACCESS_KEY,AWS_ACCESS_KEY_ID'
        })

        return SlurmSubmittedRun(active_run.info.run_id)

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
