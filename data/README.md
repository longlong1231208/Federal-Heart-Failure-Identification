# Data Directory

This directory documents the expected dataset placement.

Before running the main experiment, place the local dataset file in the project root:

```text
fl_dataset_final.pkl
```

The loader in `datasets/fl_dataset.py` expects a PKL file containing federated client data with train/test splits. The exact data object may be either dictionary-like or object-like, as handled by the dataset loader.
