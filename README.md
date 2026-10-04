# MENA MLOps Project

Project workspace for the MLOps Practitioner course. The implementation is
organized as a reproducible machine-learning service and can grow from
experimentation into training, serving, and monitoring.

## Repository layout

```text
mena-mlops-project/
├── configs/              # Versioned, non-secret configuration
├── data/
│   ├── raw/              # Source data (not committed)
│   └── processed/        # Generated data (not committed)
├── docs/                 # Architecture and project documentation
├── models/               # Generated model artifacts (not committed)
├── notebooks/            # Exploratory analysis
├── src/mena_mlops/       # Production Python package
├── tests/                # Automated tests
├── .github/workflows/    # Continuous integration
├── Makefile
└── pyproject.toml
```

## Getting started

The project targets Python 3.11 or newer. Create a virtual environment with
your preferred tool, then install the package and development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the checks with:

```bash
make check
```

## Course reference

The course examples and handbooks used to shape this scaffold are kept
outside this repository in
`/home/az-wsl/projects/mlops-course-ressources`.

## Data and secrets

Large data, model binaries, credentials, and local environment files are
intentionally excluded from Git. Store secrets in environment variables or a
secret manager and commit only sanitized configuration examples.
