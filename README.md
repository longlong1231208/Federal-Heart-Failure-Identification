# Federal Heart Failure Identification

Master's thesis core source code for DA-PFL based heart failure identification experiments.

## Project Structure

- `models/`: model definitions and DA-PFL pipeline.
- `datasets/`: dataset loading and federated data utilities required by the model pipeline.

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

## Running Code

Example command:

```bash
python models/main.py
```

Generated outputs such as `out/`, `results/`, and `figures/` are intentionally ignored by Git. Auxiliary local scripts and plotting files are not included in this repository.
