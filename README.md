# Federal Heart Failure Identification

Core source code for DA-PFL based heart failure identification experiments.

The repository contains the model implementation, dataset loader, method documentation, and reference experimental results.

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

## Documentation

- `docs/method_overview.md`: concise overview of the DA-PFL pipeline and M1/M2/M3 modules.
- `docs/expected_results.md`: thesis reference metrics and reproduction notes.
- `data/README.md`: dataset placement and data-file notes.

## Data

The dataset file `fl_dataset_final.pkl` is not included in this repository.

Place the dataset file in the project root before running the main experiment:

```text
fl_dataset_final.pkl
```

## Running Code

Example command:

```bash
python models/main.py
```

The main reference metrics are listed in `docs/expected_results.md`.
