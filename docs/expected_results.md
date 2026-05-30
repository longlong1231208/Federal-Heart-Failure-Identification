# Expected Results

This file summarizes the main results reported in Chapter 4 of the thesis.

Results are reported as mean +/- standard deviation over five random seeds. Higher values are better for Global AUC, Macro AUC, and Macro F1. Lower values are better for Macro ECE and Gini(AUC).

## Experimental Protocol

- Dataset setting: 10 simulated ICU clients.
- Input representation: 24-hour multivariate ICU time series.
- Backbone: 2-layer GRU, hidden size 64, dropout 0.3.
- Federated training: 30 communication rounds, 2 local epochs per round.
- Aggregation: sample-size weighted FedAvg.
- Optimizer: Adam with weight decay 1e-4.
- Shared-training learning rate: 1e-3.
- Client-adaptation learning rate: 1e-4.
- Validation fraction: 0.1.
- Threshold selection: maximize validation F1 over 0.10, 0.11, ..., 0.60.

## Main Comparison With External Baselines

| Method | Global AUC | Macro AUC | Macro F1 | Macro ECE (lower) | Gini(AUC) (lower) |
|---|---:|---:|---:|---:|---:|
| Local-Only | 0.6875 +/- 0.0020 | 0.7245 +/- 0.0030 | 0.4657 +/- 0.0156 | 0.0658 +/- 0.0040 | 0.0482 +/- 0.0049 |
| FedAvg | 0.7290 +/- 0.0020 | 0.7613 +/- 0.0049 | 0.4759 +/- 0.0180 | 0.1337 +/- 0.0024 | 0.0243 +/- 0.0017 |
| FedAvg + Local FT | 0.8051 +/- 0.0015 | 0.7687 +/- 0.0037 | 0.4946 +/- 0.0181 | 0.0617 +/- 0.0033 | 0.0274 +/- 0.0028 |
| Ditto | 0.8049 +/- 0.0017 | 0.7684 +/- 0.0037 | 0.4937 +/- 0.0171 | 0.0610 +/- 0.0036 | 0.0274 +/- 0.0021 |
| Per-FedAvg (FO) | 0.7898 +/- 0.0012 | 0.7558 +/- 0.0021 | 0.4909 +/- 0.0133 | 0.0649 +/- 0.0017 | 0.0346 +/- 0.0029 |
| pFedMe | 0.8020 +/- 0.0027 | 0.7591 +/- 0.0050 | 0.4744 +/- 0.0152 | 0.0508 +/- 0.0017 | 0.0380 +/- 0.0050 |
| DA-PFL (Ours) | 0.8043 +/- 0.0012 | **0.7785 +/- 0.0015** | **0.5128 +/- 0.0112** | **0.0385 +/- 0.0025** | **0.0195 +/- 0.0018** |

DA-PFL obtains a Global AUC close to the strongest local fine-tuning baselines while achieving the best client-balanced metrics: Macro AUC, Macro F1, Macro ECE, and Gini(AUC).

## Component Analysis

| Method | Global AUC | Macro AUC | Macro F1 | Macro ECE (lower) | Gini(AUC) (lower) |
|---|---:|---:|---:|---:|---:|
| Backbone Only | 0.7269 +/- 0.0011 | 0.7635 +/- 0.0017 | 0.4906 +/- 0.0074 | 0.1376 +/- 0.0026 | 0.0259 +/- 0.0024 |
| Backbone + M1 | 0.7966 +/- 0.0022 | 0.7635 +/- 0.0017 | 0.5015 +/- 0.0100 | 0.0482 +/- 0.0030 | 0.0248 +/- 0.0024 |
| Backbone + M3 | 0.7784 +/- 0.0253 | 0.7680 +/- 0.0016 | 0.4980 +/- 0.0082 | 0.0914 +/- 0.0183 | 0.0210 +/- 0.0030 |
| Backbone + M1 + M3 (fixed scope/depth) | 0.8024 +/- 0.0024 | 0.7725 +/- 0.0015 | 0.5065 +/- 0.0140 | 0.0450 +/- 0.0046 | 0.0225 +/- 0.0023 |
| Standard DA-PFL | **0.8043 +/- 0.0012** | **0.7785 +/- 0.0015** | **0.5128 +/- 0.0112** | **0.0385 +/- 0.0025** | **0.0195 +/- 0.0018** |

The component analysis supports the intended role of each module: M1 mainly improves calibration and score alignment, M3 enables restricted client-specific adaptation, and M2/APC refines personalization intensity according to client mismatch and local data support.

## Reproduction Notes

To reproduce the reported results, place `fl_dataset_final.pkl` in the project root and run:

```bash
python models/main.py
```

Small numerical differences can occur across hardware, PyTorch/CUDA versions, and deterministic settings.
