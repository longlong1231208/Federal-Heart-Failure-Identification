# Method Overview

This document summarizes the DA-PFL pipeline for federated heart failure identification.

## Backbone

The prediction backbone is a two-layer GRU model implemented in `models/gru.py`. It maps sequential clinical features to a binary heart failure prediction logit. The model also exposes parameter groups used by the personalization modules.

## Stage I: Global Federated Training

The pipeline first trains a shared global model with a FedAvg-style procedure. This stage learns a common representation across clients and provides the base model used by later personalization.

## M1: Prior-Bias Calibration

M1 adjusts the classifier bias using prior information from client-level and global label distributions. The goal is to improve the operating point of the classifier under heterogeneous client distributions.

Implementation file:

```text
models/ch4_m1.py
```

## M2: Adaptive Personalization Control

M2 estimates a reliability-adjusted personalization budget for each client. It controls how much personalization should be applied, including the number of trainable parameter groups and local personalization epochs.

Implementation file:

```text
models/ch4_m2.py
```

## M3: Gradient-Sensitivity Parameter Selection

M3 selects parameter groups for personalization based on gradient-sensitivity scores. This allows the method to personalize targeted parts of the GRU model rather than fine-tuning all parameters.

Implementation file:

```text
models/ch4_m3.py
```

## Integrated Pipeline

The integrated workflow is implemented in `models/dapfl_pipeline.py` and invoked from `models/main.py`. It connects dataset loading, global training, prior calibration, adaptive personalization control, sensitivity-based parameter selection, local personalization, and evaluation.
