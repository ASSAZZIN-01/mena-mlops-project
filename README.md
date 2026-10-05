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

The project targets Python 3.11 or newer and uses `uv` for locked,
reproducible environments:

```bash
uv sync
```

Run the checks with:

```bash
make check
```

Pull requests targeting `main` run the same locked-environment Ruff and pytest
checks in GitHub Actions. The workflow cancels superseded runs for the same
branch and exposes a single `checks` status that can be configured as a
required check in repository rules.

## Data pipeline

The data contract and all source mappings live in one file:
[`configs/data.yaml`](configs/data.yaml). Download credentials are read from
`.env` (`HF_API_TOKEN` and `KAGGLE_API_TOKEN`) and are never committed.

Run the stages directly:

```bash
python scripts/run_data_pipeline.py --stage download
python scripts/run_data_pipeline.py --stage process
```

Or reproduce the complete DVC pipeline:

```bash
dvc repro
dvc push
```

Raw downloads and generated datasets are tracked by DVC. The current default
remote is a local cache at
`/home/az-wsl/projects/mlops-course-ressources/mena-mlops-dvc-cache`; this can
be replaced later with shared object storage without changing the pipeline.

The default `debug: true` mode selects one outer fold from a stratified
10-fold split and subdivides it into two training folds, one validation fold,
and one test fold. Set `split.debug` to `false` for the full-data baseline:
fold 8 is validation, fold 9 is test, and the remaining eight folds are
training.

Exact normalized-text duplicates with conflicting labels are removed entirely;
they are not assigned an arbitrary surviving label. Other exact duplicates keep
one row after this conflict filter.

## Course reference

The course examples and handbooks used to shape this scaffold are kept
outside this repository in
`/home/az-wsl/projects/mlops-course-ressources`.

## Data and secrets

Large data, model binaries, credentials, and local environment files are
intentionally excluded from Git. Store secrets in environment variables or a
secret manager and commit only sanitized configuration examples.

## Training

The debug training workflow uses AraBERT and records training plus validation,
test, and per-source evaluation in one MLflow run:

```bash
uv run python -m mena_mlops.training.train
```

MLflow uses the local SQLite tracking database configured in
`configs/training.yaml`; the database and generated artifacts are ignored by
Git.
