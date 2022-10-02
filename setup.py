import os
import setuptools

version = os.getenv("mlflow_slurm_version")
if version is None:
    version = "0.1a1"
else:
    version = version.split("/")[-1]

here = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = "Backend implementation for running MLFlow projects on Slurm"

try:
    LONG_DESCRIPTION = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except Exception:
    LONG_DESCRIPTION = ""


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines()
                if (s.strip() and not s.startswith("#"))]


REQUIREMENTS = _read_reqs("requirements.txt")
TESTS_REQUIREMENTS = _read_reqs("tests-requirements.txt")

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Software Development :: Libraries"
]


setuptools.setup(
    name="mlflow_slurm",
    packages=setuptools.find_packages(),
    version=version,
    install_requires=REQUIREMENTS,
    package_data={'mlflow_slurm.templates': ['sbatch_template.sh']},
    include_package_data=True,
    tests_require=TESTS_REQUIREMENTS,
    python_requires=">=3.6",
    maintainer="Ben Galewsky",
    maintainer_email="bengal1@illinois.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
    keywords="mlflow",
    url="https://github.com/ncsa/mlflow-slurm",
    entry_points={
        "mlflow.project_backend":
            "slurm=mlflow_slurm.slurm_backend:slurm_backend_builder",
    },
)
