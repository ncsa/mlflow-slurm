import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from threading import RLock
from typing import Tuple, List

from jinja2 import Environment, FileSystemLoader

from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.projects import load_project
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import (
    fetch_and_validate_project, get_or_create_run,
    PROJECT_STORAGE_DIR, PROJECT_ENV_MANAGER, get_entry_point_command
)
from mlflow.tracking import MlflowClient
from mlflow.utils.conda import get_or_create_conda_env, get_conda_command
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.logging_utils import _configure_mlflow_loggers
from mlflow.utils.mlflow_tags import MLFLOW_PROJECT_ENV
from mlflow.utils.virtualenv import _install_python, _get_mlflow_virtualenv_root, \
    _get_virtualenv_name, _create_virtualenv

_configure_mlflow_loggers(root_module_name=__name__)
_logger = logging.getLogger(__name__)


# Entrypoint for Project Backend
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

    def is_terminated_or_gone(self):
        self._update_status()
        return not self._status or RunStatus.is_terminated(self._status)

    def wait(self):
        """
        Implements the wait functionality for a slurm job. When we notice that the job
        is complete, attempt to grab the job logs and attach them to the run as an
        artifact
        :return: Boolean success
        """
        while not self.is_terminated_or_gone():
            time.sleep(self.POLL_STATUS_INTERVAL)

        with open(f"slurm-{self.slurm_job_id}.out") as file:
            log_lines = file.read()
            MlflowClient().log_text(self.run_id, log_lines,
                                    f"slurm-{self.slurm_job_id}.txt")
        return self._status == RunStatus.FINISHED

    def cancel(self) -> None:
        print("Cancel")

    def get_status(self) -> RunStatus:
        return self._status

    def _update_status(self) -> RunStatus:
        with subprocess.Popen(f"squeue --state=all --job={self.slurm_job_id} -o%A,%t",
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, shell=True,
                              universal_newlines=True) as p:
            p.wait()
            output = p.stdout.read().split('\n')
            job = output[1]  # First line is header
            with self._status_lock:
                if not job:
                    _logger.warning(
                        f"Looking for status of job {self.slurm_job_id}, but it is gone")
                    self._status = RunStatus.FINISHED

                job_status = output[1].split(",")[1]
                if job_status == "PD":
                    self._status = RunStatus.SCHEDULED
                elif job_status == "CD":
                    self._status = RunStatus.FINISHED
                elif job_status == "F":
                    self._status = RunStatus.FAILED
                elif job_status == "R" \
                        or job_status == "S" \
                        or job_status == "ST" \
                        or job_status == "CG" \
                        or job_status == "PR":
                    self._status = RunStatus.RUNNING
                else:
                    _logger.warning(
                        f"Job ID {self.slurm_job_id} has an unmapped status of {job_status}")
                    self._status = None


class SlurmProjectBackend(AbstractBackend):
    @staticmethod
    def sbatch(script: str) -> str:
        """
        Submit a script to the slurm batch manager
        :param script: The filename of the script
        :return: The Slurm Job ID or None of the submit fails
        """
        job_re = "Submitted batch job (\\d+)"
        with subprocess.Popen(f"sbatch {script}",
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, shell=True) as p:
            return_code = p.wait()
            if return_code == 0:
                # Parse stdout to find the job ID
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

    def run(self, project_uri: str, entry_point: str, params: dict, version: str,
            backend_config: dict, tracking_uri: str,
            experiment_id: str) -> SlurmSubmittedRun:

        work_dir = fetch_and_validate_project(project_uri, version, entry_point, params)
        active_run = get_or_create_run(None, project_uri, experiment_id, work_dir,
                                       version,
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
            activate_cmd = _create_virtualenv(work_dir_path, python_bin_path, env_dir,
                                              python_env)
            command_args += [activate_cmd]
        elif project.env_type == "conda_env":
            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV,
                                            "conda")
            command_separator = " && "
            conda_env = get_or_create_conda_env(project.env_config_path)
            command_args += conda_env.get_activate_command()
        else:
            _logger.fatal(f"Unknown project environment type provided: {project.env_type}")
            return None

        command_args += get_entry_point_command(project, entry_point, params, storage_dir)
        command_str = command_separator.join(command_args)

        # Allow user to specify a filename for the batch file. If none provided then
        # generate one based on the job ID
        sbatch_file = backend_config.get("sbatch-script-file",
                                         f"sbatch-{active_run.info.run_id}.sh")

        generate_sbatch_script(command_str, backend_config, active_run.info.run_id,
                               sbatch_file)

        job_id = SlurmProjectBackend.sbatch(sbatch_file)
        MlflowClient().set_tag(active_run.info.run_id, "slurm_job_id", job_id)
        _logger.info(f"slurm job id={job_id}")

        return SlurmSubmittedRun(active_run.info.run_id, job_id)

    def __init__(self):
        pass


def generate_sbatch_script(command_str: str = None, backend_config: dict = None,
                           run_id: str = None, script_file: str = None):
    # Find the batch script template deployed with this package
    root = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(root, 'templates')

    template = Environment(
        loader=FileSystemLoader(templates_dir),
        trim_blocks=True
    ).get_template("sbatch_template.sh")

    with open(script_file, "w") as text_file:
        text_file.write(template.render(command=command_str,
                                        config=backend_config,
                                        run_id=run_id))


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
