# DA-PFL Experiment Code

This repository contains the implementation code for the DA-PFL paper experiments.

## Project Structure

- `models/`: model definitions, DA-PFL pipeline, and experiment runners.
- `datasets/`: dataset loading and federated data utilities.
- `scripts/`: scripts for running experiments and generating paper figures.
- `utils/`: metrics, random seed helpers, and statistical analysis utilities.
- `绘图.py`: plotting script used for paper figures.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Python 3.9 or newer is recommended. If CUDA is available, PyTorch will use GPU automatically in the main experiment scripts.

## Data

The local dataset file `fl_dataset_final.pkl` is not included in this repository because it is a generated data artifact and may be large or private.

Place the dataset file in the project root before running the experiment scripts:

```text
fl_dataset_final.pkl
```

## Running Experiments

Example commands:

```bash
python models/main.py
python scripts/run_final_dapfl_updated_m2.py
python scripts/run_m2_mapping_preexperiment.py
```

Generated outputs such as `out/`, `results/`, and `figures/` are intentionally ignored by Git. They can be regenerated from the code and local dataset.
