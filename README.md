# MLFlow-Slurm
Backend for executing MLFlow projects on Slurm batch system

To build pySlrum in the docker image:
```
git clone https://github.com/PySlurm/pyslurm.git
cd pyslurm/
git checkout v21.08.4
python3.9 setup.py build 
python3.9 setup.py install
cd /mlflow-slurm
```