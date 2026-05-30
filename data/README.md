# Data Directory

The dataset is not included in this repository.

Before running the main experiment, place the local dataset file in the project root:

```text
fl_dataset_final.pkl
```

The loader in `datasets/fl_dataset.py` expects a PKL file containing federated client data with train/test splits. The exact data object may be either dictionary-like or object-like, as handled by the dataset loader.

Generated data files and large binary artifacts should not be committed to Git.
