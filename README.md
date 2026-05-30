# Federal Heart Failure Identification

Master's thesis core source code for DA-PFL based heart failure identification experiments.

This repository keeps only the implementation files required to describe and run the core model pipeline. Local datasets, generated experiment outputs, plotting scripts, and auxiliary analysis files are not included.

## Module Overview

### `datasets/`

- `datasets/fl_dataset.py`: loads the federated PKL dataset, builds per-client train/validation/test splits, and returns PyTorch `DataLoader` objects.
- `datasets/__init__.py`: package marker for dataset utilities.

### `models/`

- `models/gru.py`: defines the 2-layer GRU backbone used for binary heart failure prediction and exposes parameter groups used by personalization.
- `models/ch4_m1.py`: implements M1 prior-bias calibration for correcting the classifier operating point with client/global prior information.
- `models/ch4_m2.py`: implements M2 adaptive personalization control, including reliability-adjusted personalization budget selection.
- `models/ch4_m3.py`: implements M3 gradient-sensitivity based parameter-group selection and trainable-mask control.
- `models/dapfl_pipeline.py`: connects dataset loading, FedAvg backbone training, M1/M2/M3 components, local personalization, and evaluation.
- `models/main.py`: main experiment entry file for running the DA-PFL comparison pipeline.
- `models/__init__.py`: package marker for model modules.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. If CUDA is available, PyTorch will use GPU automatically in the main experiment code.

## Data

The local dataset file `fl_dataset_final.pkl` is not included in this repository because it is a generated data artifact and may be large or private.

Place the dataset file in the project root before running the main experiment:

```text
fl_dataset_final.pkl
```

## Running Code

Example command:

```bash
python models/main.py
```

Generated outputs such as `out/`, `results/`, and `figures/` are intentionally ignored by Git.

## Repository Scope

This repository is intended as a clean core-code release. The following local files are intentionally excluded:

- raw/generated dataset files
- experiment output folders
- paper figures and result tables
- plotting and auxiliary analysis scripts
- IDE settings and Python cache files
