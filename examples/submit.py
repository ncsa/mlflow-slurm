import re
import subprocess

from jinja2 import Environment, BaseLoader

from mlflow.entities import RunStatus

job_template = """#!/bin/bash
#SBATCH --job-name=Popen
#SBATCH --partition=normal
python3.9 -c "print('hello')"
"""
template = Environment(loader=BaseLoader()).from_string(job_template)
with open("generated.sh", "w") as text_file:
    text_file.write(template.render())


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

def squeue(job_id:str):
    with subprocess.Popen(f"squeue --job={job_id} -o%A,%t",
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=True,
                          universal_newlines=True) as p:
        p.wait()
        output = p.stdout.read().split('\n')
        job = output[1]
        if not job:
            return None

        job_status = output[1].split(",")[1]
        print(job_status)
        if job_status == "PD":
            return RunStatus.SCHEDULED
        if job_status == "CD":
            return RunStatus.FINISHED
        if job_status == "F":
            return RunStatus.FAILED
        if job_status == "R" \
                or job_status == "S" \
                or job_status == "ST" \
                or job_status == "CG" \
                or job_status == "PR":
            return RunStatus.RUNNING

job_id = sbatch("generated.sh")
print("Submitted ", job_id)
print("Squeu ", squeue(job_id))
# print("Squeu ", squeue("40"))
