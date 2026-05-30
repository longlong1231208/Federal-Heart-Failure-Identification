# -*- coding: utf-8 -*-
"""
Unified Chapter-5 experiment runner.

This script provides TWO tables:

1) External baselines
   - Local-Only
   - FedAvg (global)
   - FedAvg + Local Fine-Tuning
   - Ditto
   - Per-FedAvg (first-order practical implementation)
   - pFedMe (practical proximal implementation)
   - Full DA-PFL

2) Internal main-results table (paper-facing component comparison)
   - Global only (Stage I only)
   - Global + M1 only
   - Global + M3 only
   - Global + M1 + M3 with fixed budget (no APC adaptivity)
   - Full DA-PFL

Metrics reported:
   Global AUC, Macro AUC, Macro F1, Macro ECE, Gini
"""
from __future__ import annotations

import copy
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Robust imports: prefer project package paths, then fall back to uploaded files
# -----------------------------------------------------------------------------
try:
    from models.dapfl_pipeline import (
        Ch4Config,
        _aggregate_client_metrics,
        _choose_threshold_from_val,
        _eval_on_test_with_threshold,
        build_ch4_loaders,
        get_priors,
        run_dapfl_stage2,
        run_fedavg_backbone,
        set_seed,
    )
except ImportError:
    from dapfl_pipeline import (  # type: ignore
        Ch4Config,
        _aggregate_client_metrics,
        _choose_threshold_from_val,
        _eval_on_test_with_threshold,
        build_ch4_loaders,
        get_priors,
        run_dapfl_stage2,
        run_fedavg_backbone,
        set_seed,
    )

try:
    from models.gru import GRUModel
except ImportError:
    from gru import GRUModel  # type: ignore


# -----------------------------------------------------------------------------
# Paths & settings
# -----------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "fl_dataset_final.pkl"
OUT_DIR = PROJECT_ROOT / "out" / "ch5_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_NAME = "run_ch5_all_tables.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_REPEATS = 5
VAL_FRAC = 0.1
# External baseline hyperparameters
LOCAL_ONLY_EPOCHS = 20
FINE_TUNE_EPOCHS = 5
DITTO_LAMBDA = 0.1

# Per-FedAvg (practical first-order version)
PERFEDAVG_ROUNDS = 40
PERFEDAVG_INNER_LR = 1e-3
PERFEDAVG_META_LR = 1.0
PERFEDAVG_LOCAL_STEPS = 2
PERFEDAVG_PERSONAL_EPOCHS = 5

# pFedMe (practical proximal version)
PFEDME_ROUNDS = 40
PFEDME_PERSONAL_LR = 1e-3
PFEDME_LAMBDA = 0.05
PFEDME_BETA = 1.0
PFEDME_LOCAL_EPOCHS = 2
PFEDME_PERSONAL_EPOCHS = 5

# Fixed-budget variant used for the internal main-results table
FIXED_K_PERS = 7
FIXED_E_PERS = 5

PAPER_METHOD_LABELS = {
    "External_Local_Only": "Local-Only",
    "External_FedAvg": "FedAvg",
    "External_FedAvg_LocalFT": "FedAvg + Local Fine-Tuning",
    "External_Ditto": "Ditto",
    "External_PerFedAvg_FO": "Per-FedAvg (FO)",
    "External_pFedMe": "pFedMe",
    "External_Full_DAPFL": "Full DA-PFL",
    "Main_Global_Only_StageI": "Stage I global backbone",
    "Main_Global_M1_Only": "Stage I + M1",
    "Main_Global_M3_Only": "Stage I + M3",
    "Main_Global_M1_M3_FixedBudget": "Stage I + M1 + M3 (fixed budget)",
    "Main_Full_DAPFL": "Full DA-PFL",
}

APC_SIGNAL_LABELS = {
    "reliability_prior": "Reliability-Adjusted Prior",
    "need_reliability_product": "Need x Reliability Product",
}

PAPER_METHOD_DESCRIPTIONS = {
    "External_Local_Only": "Each client trains an independent GRU using only local data.",
    "External_FedAvg": "Shared global GRU trained by standard sample-size weighted FedAvg.",
    "External_FedAvg_LocalFT": "Each client starts from the FedAvg backbone and performs local fine-tuning.",
    "External_Ditto": "Proximal personalized FL baseline regularized toward the global model.",
    "External_PerFedAvg_FO": "Practical first-order meta-learning personalized FL baseline.",
    "External_pFedMe": "Proximal personalized FL baseline with iterative personalization around a global anchor.",
    "External_Full_DAPFL": "DA-PFL with M1 prior correction plus need-by-reliability M2/APC and M3 masked personalization.",
    "Main_Global_Only_StageI": "Shared federated backbone without client-specific adaptation.",
    "Main_Global_M1_Only": "Backbone followed only by M1 validation-gated prior correction.",
    "Main_Global_M3_Only": "Backbone followed only by fixed-budget M3 gradient-guided structured personalization.",
    "Main_Global_M1_M3_FixedBudget": "M1 and M3 enabled with uniform budget, removing M2/APC adaptivity.",
    "Main_Full_DAPFL": "Full DA-PFL: M1 followed by need-by-reliability M2/APC and M3 masked personalization.",
}


def _safe_key_part(text: str) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_csv_strings(text: Optional[str]) -> List[str]:
    if text is None or not str(text).strip():
        return []
    return [str(x).strip() for x in str(text).split(",") if str(x).strip()]


def _clean_apc_signal_modes(*items: Optional[str]) -> List[str]:
    allowed = {"reliability_prior", "need_reliability_product"}
    modes: List[str] = []
    for item in items:
        for raw in _parse_csv_strings(item):
            mode = _safe_key_part(raw)
            if mode not in allowed:
                raise ValueError(f"Unsupported APC signal mode '{raw}'. Allowed: {sorted(allowed)}")
            if mode not in modes:
                modes.append(mode)
    return modes or ["reliability_prior"]


def _full_method_key(prefix: str, mode: str, *, multi_mode: bool) -> str:
    if not multi_mode:
        return f"{prefix}_Full_DAPFL"
    return f"{prefix}_Full_DAPFL_{_safe_key_part(mode)}"


def _register_full_method_labels(modes: List[str], *, multi_mode: bool) -> None:
    for mode in modes:
        mode_key = _safe_key_part(mode)
        label = APC_SIGNAL_LABELS.get(mode_key, mode)
        for prefix in ("External", "Main"):
            method_key = _full_method_key(prefix, mode, multi_mode=multi_mode)
            PAPER_METHOD_LABELS[method_key] = f"Full DA-PFL ({label})"
            PAPER_METHOD_DESCRIPTIONS[method_key] = (
                f"Full DA-PFL with M1 prior correction, need-by-reliability M2/APC, "
                f"M3 masked personalization, and signal mode '{mode}'."
            )

PAPER_CORE_METRICS = [
    "global_auc",
    "macro_auc",
    "macro_f1",
    "macro_ece",
    "gini",
]

PAPER_MODULE_NOTES = {
    "Stage I": "Original full-participation FedAvg GRU backbone used as the common initialization.",
    "M1": "Validation-gated prior correction that updates only the classifier-head bias before structural adaptation.",
    "M2/APC": "Need-by-reliability mapper: reliability-adjusted local-global prior mismatch gives adaptation need, sample support gives local reliability, and their product is optionally validation-gated over conservative/neutral/aggressive candidates.",
    "M3": "Gradient-sensitivity based masked personalization over the seven GRU parameter groups.",
}

PAPER_PROTECTION_NOTES = {
    "validation_only_decisions": "Validation data are used for threshold selection, M1 gating, APC candidate selection, and Stage-II early stopping; test data are held out.",
    "m1_bias_scope": "M1 calibration is applied through the classifier-head bias; representation weights are not changed in Stage II-A.",
    "m3_mask_scope": "M3 always keeps the classifier head active and freezes groups outside the selected mask.",
    "adaptation_scope": "M2 maps reliability-adjusted prior mismatch to group scope and epoch depth; the validation gate may select a conservative, neutral, or aggressive neighboring plan.",
    "fallback_checkpoint": "Masked personalization starts from an epoch-0 checkpoint and restores the best validation state when early stopping is active.",
}

M3_GROUP_ORDER = [
    "g1_head",
    "g2_l0_ih",
    "g3_l0_hh",
    "g4_l0_b",
    "g5_l1_ih",
    "g6_l1_hh",
    "g7_l1_b",
]
N_M3_GROUPS = len(M3_GROUP_ORDER)

BUDGET_SUMMARY_METRICS = [
    "E_pers",
    "K_pers",
    "selected_count",
    "trainable_param_ratio",
    "effective_group_epoch_budget",
    "full_model_equiv_epoch_budget",
    "effective_param_epoch_budget",
]


# -----------------------------------------------------------------------------
# JSON sanitization
# -----------------------------------------------------------------------------
def _json_sanitize(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return _json_sanitize(obj.item())
        return _json_sanitize(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


# -----------------------------------------------------------------------------
# Standard training loops for external baselines
# -----------------------------------------------------------------------------
def train_epoch_standard(model: nn.Module, loader, optimizer, criterion, device) -> None:
    model.train()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(x).view(-1) if hasattr(model, "forward_logits") else model(x).view(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def train_epoch_ditto(
    model: nn.Module,
    global_model: nn.Module,
    loader,
    optimizer,
    criterion,
    lam: float,
    device,
) -> None:
    model.train()
    global_model.eval()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(x).view(-1) if hasattr(model, "forward_logits") else model(x).view(-1)
        loss = criterion(logits, y)

        prox_loss = 0.0
        for p_local, p_global in zip(model.parameters(), global_model.parameters()):
            if p_local.requires_grad:
                prox_loss = prox_loss + torch.sum((p_local - p_global.detach().to(device)) ** 2)

        total_loss = loss + 0.5 * float(lam) * prox_loss
        total_loss.backward()
        optimizer.step()


def train_epoch_proximal_to_anchor(
    model: nn.Module,
    anchor_model: nn.Module,
    loader,
    optimizer,
    criterion,
    lam: float,
    device,
) -> None:
    model.train()
    anchor_model.eval()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(x).view(-1) if hasattr(model, "forward_logits") else model(x).view(-1)
        loss = criterion(logits, y)

        prox_loss = 0.0
        for p_local, p_anchor in zip(model.parameters(), anchor_model.parameters()):
            if p_local.requires_grad:
                prox_loss = prox_loss + torch.sum((p_local - p_anchor.detach().to(device)) ** 2)

        total_loss = loss + 0.5 * float(lam) * prox_loss
        total_loss.backward()
        optimizer.step()


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
def _eval_model_on_client(model, name: str, client_loaders, cfg: Ch4Config, device) -> Dict[str, Any]:
    val_loader = client_loaders[name]["val"]
    test_loader = client_loaders[name]["test"]

    thr = _choose_threshold_from_val(model, val_loader, device, cfg)
    return _eval_on_test_with_threshold(model, test_loader, device, cfg, thr)


def _compact_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "global_auc": res.get("global_auc", np.nan),
        "macro_auc": res.get("macro_auc", np.nan),
        "macro_f1": res.get("macro_f1", np.nan),
        "macro_ece": res.get("macro_ece", np.nan),
        "gini": res.get("gini", np.nan),
    }


def _with_client_metrics(agg: Dict[str, Any], client_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(agg)
    out["client_metrics"] = client_metrics
    return out


def _safe_float(obj: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if obj is None:
            return default
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except Exception:
        return default


def _safe_int(obj: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if obj is None:
            return default
        return int(obj)
    except Exception:
        return default


def _baseline_budget_defaults(method_name: str) -> Dict[str, Any]:
    full_ft_epochs = {
        "External_FedAvg_LocalFT": FINE_TUNE_EPOCHS,
        "External_Ditto": FINE_TUNE_EPOCHS,
        "External_PerFedAvg_FO": PERFEDAVG_PERSONAL_EPOCHS,
        "External_pFedMe": PFEDME_PERSONAL_EPOCHS,
    }

    if method_name in full_ft_epochs:
        e = int(full_ft_epochs[method_name])
        return {
            "budget_type": "fixed_full_model_personalization",
            "local_training_epochs": None,
            "personalization_budget_applicable": True,
            "E_pers": e,
            "K_pers": int(N_M3_GROUPS),
            "selected_count": int(N_M3_GROUPS),
            "trainable_param_ratio": 1.0,
            "effective_group_epoch_budget": float(e),
            "full_model_equiv_epoch_budget": float(e),
            "effective_param_epoch_budget": float(e),
        }

    if method_name == "External_Local_Only":
        return {
            "budget_type": "local_only_from_scratch",
            "local_training_epochs": int(LOCAL_ONLY_EPOCHS),
            "personalization_budget_applicable": False,
            "E_pers": None,
            "K_pers": None,
            "selected_count": None,
            "trainable_param_ratio": 1.0,
            "effective_group_epoch_budget": None,
            "full_model_equiv_epoch_budget": None,
            "effective_param_epoch_budget": None,
        }

    if method_name in {"External_FedAvg", "Main_Global_Only_StageI", "Main_Global_M1_Only"}:
        return {
            "budget_type": "no_representation_personalization",
            "local_training_epochs": None,
            "personalization_budget_applicable": True,
            "E_pers": 0,
            "K_pers": 0,
            "selected_count": 0,
            "trainable_param_ratio": 0.0,
            "effective_group_epoch_budget": 0.0,
            "full_model_equiv_epoch_budget": 0.0,
            "effective_param_epoch_budget": 0.0,
        }

    return {
        "budget_type": "apc_m3_masked_personalization",
        "local_training_epochs": None,
        "personalization_budget_applicable": True,
    }


def _client_level_rows(
    *,
    table: str,
    repeat: int,
    seed: int,
    method_name: str,
    result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    client_metrics = result.get("client_metrics", {})
    client_debug = result.get("client_debug", {})
    if not isinstance(client_metrics, dict):
        client_metrics = {}
    if not isinstance(client_debug, dict):
        client_debug = {}

    clients = sorted(set(map(str, client_metrics.keys())) | set(map(str, client_debug.keys())))
    rows: List[Dict[str, Any]] = []
    for client in clients:
        metrics = client_metrics.get(client, {})
        dbg = client_debug.get(client, {})
        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(dbg, dict):
            dbg = {}

        m1 = dbg.get("m1_bias", {}) if isinstance(dbg.get("m1_bias"), dict) else {}
        scores = dbg.get("m3_scores", {}) if isinstance(dbg.get("m3_scores"), dict) else {}
        selected = dbg.get("selected_groups", [])
        selected_set = set(str(x) for x in selected) if isinstance(selected, list) else set()
        budget_defaults = _baseline_budget_defaults(str(method_name))
        selected_count = _safe_int(
            dbg.get("selected_group_count"),
            int(len(selected_set)) if selected_set else budget_defaults.get("selected_count"),
        )

        row: Dict[str, Any] = {
            "table": str(table),
            "repeat": int(repeat),
            "seed": int(seed),
            "method_key": str(method_name),
            "paper_label": PAPER_METHOD_LABELS.get(method_name, method_name),
            "client": str(client),
            "client_auc": _safe_float(metrics.get("auc", metrics.get("roc_auc"))),
            "client_f1": _safe_float(metrics.get("f1")),
            "client_ece": _safe_float(metrics.get("ece")),
            "client_threshold": _safe_float(metrics.get("threshold", metrics.get("thr"))),
            "n_i": _safe_int(dbg.get("n_i", dbg.get("train_size"))),
            "train_size": _safe_int(dbg.get("train_size", dbg.get("n_i"))),
            "p_i_raw": _safe_float(dbg.get("p_i_raw")),
            "p_g_qbar": _safe_float(dbg.get("p_g_qbar")),
            "abs_shift": _safe_float(dbg.get("abs_shift")),
            "apc_shift_mode": str(dbg.get("apc_shift_mode", "")),
            "apc_signal_mode": str(dbg.get("apc_signal_mode", "")),
            "apc_formula": str(dbg.get("apc_formula", "")),
            "apc_formula_normalization": bool(dbg.get("apc_formula_normalization", False)) if dbg else None,
            "apc_pi_k": _safe_float(dbg.get("apc_pi_k")),
            "apc_pi_ref": _safe_float(dbg.get("apc_pi_ref")),
            "apc_delta_b": _safe_float(dbg.get("apc_delta_b")),
            "apc_delta": _safe_float(dbg.get("apc_delta")),
            "apc_sigma_delta": _safe_float(dbg.get("apc_sigma_delta")),
            "apc_q_reliability": _safe_float(dbg.get("apc_q_reliability")),
            "apc_s_reliability": _safe_float(dbg.get("apc_s_reliability")),
            "apc_gamma_s": _safe_float(dbg.get("apc_gamma_s")),
            "apc_gamma_s_scaled": _safe_float(dbg.get("apc_gamma_s_scaled")),
            "apc_r_neutral": _safe_float(dbg.get("apc_r_neutral")),
            "apc_r_reliability": _safe_float(dbg.get("apc_r_reliability")),
            "m1_applied": _safe_float(m1.get("applied")),
            "m1_p_i_tilde": _safe_float(m1.get("p_i_tilde")),
            "m1_p_g_tilde": _safe_float(m1.get("p_g_tilde")),
            "m1_delta_raw": _safe_float(m1.get("delta_raw")),
            "m1_lambda": _safe_float(m1.get("lambda")),
            "m1_delta_eff": _safe_float(m1.get("delta_eff")),
            "m1_bias_before": _safe_float(m1.get("bias_before")),
            "m1_bias_after": _safe_float(m1.get("bias_after")),
            "apc_name": str(dbg.get("apc_name", "")),
            "apc_candidate_selection": bool(dbg.get("apc_candidate_selection", False)) if dbg else None,
            "K_pers": _safe_int(dbg.get("K_pers"), budget_defaults.get("K_pers")),
            "E_pers": _safe_int(dbg.get("E_pers"), budget_defaults.get("E_pers")),
            "rep_off": bool(dbg.get("rep_off", False)) if dbg else None,
            "selected_count": selected_count,
            "selected_groups": ";".join(sorted(selected_set)),
            "budget_type": str(dbg.get("budget_type", budget_defaults.get("budget_type", ""))),
            "local_training_epochs": _safe_int(
                dbg.get("local_training_epochs"),
                budget_defaults.get("local_training_epochs"),
            ),
            "personalization_budget_applicable": bool(
                dbg.get(
                    "personalization_budget_applicable",
                    budget_defaults.get("personalization_budget_applicable", True),
                )
            ),
            "trainable_param_count": _safe_int(dbg.get("trainable_param_count")),
            "total_param_count": _safe_int(dbg.get("total_param_count")),
            "trainable_param_ratio": _safe_float(
                dbg.get("trainable_param_ratio"),
                budget_defaults.get("trainable_param_ratio"),
            ),
            "effective_group_epoch_budget": _safe_float(
                dbg.get("effective_group_epoch_budget"),
                budget_defaults.get("effective_group_epoch_budget"),
            ),
            "full_model_equiv_epoch_budget": _safe_float(
                dbg.get("full_model_equiv_epoch_budget", dbg.get("effective_group_epoch_budget")),
                budget_defaults.get("full_model_equiv_epoch_budget"),
            ),
            "effective_param_epoch_budget": _safe_float(
                dbg.get("effective_param_epoch_budget"),
                budget_defaults.get("effective_param_epoch_budget"),
            ),
            "realized_drift": _safe_float(dbg.get("realized_drift")),
            "best_val_score": _safe_float(dbg.get("best_val_score")),
        }
        for group in M3_GROUP_ORDER:
            row[f"mask_{group}"] = 1 if group in selected_set else 0
            row[f"score_{group}"] = _safe_float(scores.get(group))
        rows.append(row)
    return rows


def _stage1_history_rows(*, repeat: int, seed: int, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        row = {
            "repeat": int(repeat),
            "seed": int(seed),
            "round": _safe_int(item.get("round")),
            "lr": _safe_float(item.get("lr")),
            "train_loss": _safe_float(item.get("train_loss")),
            "val_loss": _safe_float(item.get("val_loss")),
            "global_auc": _safe_float(item.get("global_auc")),
            "macro_auc": _safe_float(item.get("macro_auc")),
            "macro_f1": _safe_float(item.get("macro_f1")),
            "macro_ece": _safe_float(item.get("macro_ece")),
            "gini": _safe_float(item.get("gini")),
        }
        rows.append(row)
    return rows


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in vals if np.isfinite(float(v))]
    if not clean:
        return None, None
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    return mean, std


def _stage1_history_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_round: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        rnd = _safe_int(row.get("round"))
        if rnd is None:
            continue
        by_round.setdefault(int(rnd), []).append(row)

    metrics = ["train_loss", "val_loss", "global_auc", "macro_auc", "macro_f1", "macro_ece", "gini"]
    out: List[Dict[str, Any]] = []
    for rnd in sorted(by_round.keys()):
        group = by_round[rnd]
        row: Dict[str, Any] = {"round": int(rnd), "n_repeats": int(len(group))}
        for metric in metrics:
            vals = []
            for item in group:
                val = _safe_float(item.get(metric))
                if val is not None:
                    vals.append(float(val))
            mean, std = _mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    return out


def _budget_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("table", "")),
            str(row.get("method_key", "")),
            str(row.get("paper_label", row.get("method_key", ""))),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (table, method_key, paper_label), group in sorted(groups.items()):
        budget_types = [str(r.get("budget_type", "")) for r in group if str(r.get("budget_type", ""))]
        budget_type = max(set(budget_types), key=budget_types.count) if budget_types else ""
        applicable = [
            bool(r.get("personalization_budget_applicable", True))
            for r in group
        ]
        row: Dict[str, Any] = {
            "table": table,
            "method_key": method_key,
            "paper_label": paper_label,
            "budget_type": budget_type,
            "n_client_rows": int(len(group)),
            "personalization_budget_applicable": bool(any(applicable)),
        }

        local_epochs = [
            float(v)
            for v in (_safe_float(r.get("local_training_epochs")) for r in group)
            if v is not None
        ]
        mean, std = _mean_std(local_epochs)
        row["local_training_epochs_mean"] = mean
        row["local_training_epochs_std"] = std

        for metric in BUDGET_SUMMARY_METRICS:
            vals = [
                float(v)
                for v in (_safe_float(r.get(metric)) for r in group)
                if v is not None
            ]
            mean, std = _mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    return out


def _client_subset_names(
    *,
    client_names: List[str],
    priors_per_client: Dict[str, Dict[str, float]],
    client_sizes: List[int],
) -> Dict[str, List[str]]:
    p_g = float(priors_per_client.get("_global", {}).get("p_g_raw", 0.5))
    shifts = {
        str(name): abs(float(priors_per_client.get(name, {}).get("p_i_raw", p_g)) - p_g)
        for name in client_names
    }
    sizes = {str(name): int(size) for name, size in zip(client_names, client_sizes)}

    shift_cut = float(np.median(list(shifts.values()))) if shifts else 0.0
    size_cut = float(np.median(list(sizes.values()))) if sizes else 0.0

    high_shift = [name for name in client_names if shifts[str(name)] >= shift_cut]
    small_clients = [name for name in client_names if sizes[str(name)] <= size_cut]

    return {
        "high_shift": high_shift,
        "small_client": small_clients,
    }


def _subgroup_analysis(
    *,
    method_results: Dict[str, Dict[str, Any]],
    client_names: List[str],
    priors_per_client: Dict[str, Dict[str, float]],
    client_sizes: List[int],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    subsets = _client_subset_names(
        client_names=client_names,
        priors_per_client=priors_per_client,
        client_sizes=client_sizes,
    )
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method_name, result in method_results.items():
        client_metrics = result.get("client_metrics", {})
        if not isinstance(client_metrics, dict) or not client_metrics:
            continue
        out[method_name] = {}
        for group_name, names in subsets.items():
            sub_metrics = {
                str(name): client_metrics[str(name)]
                for name in names
                if str(name) in client_metrics
            }
            if not sub_metrics:
                continue
            compact = _compact_metrics(_aggregate_client_metrics(sub_metrics))
            compact["n_clients"] = int(len(sub_metrics))
            compact["clients"] = [str(n) for n in sub_metrics.keys()]
            out[method_name][group_name] = compact
    return out


# -----------------------------------------------------------------------------
# Practical Per-FedAvg / pFedMe helpers
# -----------------------------------------------------------------------------
def _make_model(input_dim: int, cfg: Ch4Config, device) -> nn.Module:
    return GRUModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device)


def _state_dict_weighted_average(
    states: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    assert len(states) == len(weights) and len(states) > 0
    wsum = float(sum(weights)) if sum(weights) > 0 else 1.0

    out = copy.deepcopy(states[0])
    for k in out.keys():
        acc = torch.zeros_like(out[k])
        for st, w in zip(states, weights):
            acc += st[k] * float(w / wsum)
        out[k] = acc
    return out


def _perfedavg_client_update_first_order(
    global_state: Dict[str, torch.Tensor],
    train_loader,
    input_dim: int,
    cfg: Ch4Config,
    device,
    inner_lr: float,
    meta_lr: float,
    local_steps: int,
) -> Dict[str, torch.Tensor]:
    local_model = _make_model(input_dim, cfg, device)
    local_model.load_state_dict(global_state, strict=True)

    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(local_model.parameters(), lr=float(inner_lr), weight_decay=float(cfg.weight_decay))

    step_count = 0
    while step_count < int(local_steps):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1)

            opt.zero_grad(set_to_none=True)
            logits = local_model.forward_logits(x).view(-1) if hasattr(local_model, "forward_logits") else local_model(x).view(-1)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            step_count += 1
            if step_count >= int(local_steps):
                break

    new_state = copy.deepcopy(global_state)
    local_state = local_model.state_dict()
    for k in new_state.keys():
        new_state[k] = global_state[k] + float(meta_lr) * (local_state[k] - global_state[k])
    return new_state


def run_perfedavg_backbone(
    client_loaders,
    client_names: List[str],
    client_sizes: List[int],
    input_dim: int,
    cfg: Ch4Config,
    device,
    *,
    rounds: int,
    inner_lr: float,
    meta_lr: float,
    local_steps: int,
) -> nn.Module:
    global_model = _make_model(input_dim, cfg, device)
    global_state = copy.deepcopy(global_model.state_dict())

    for _ in range(int(rounds)):
        local_states = []
        for name in client_names:
            st = _perfedavg_client_update_first_order(
                global_state=global_state,
                train_loader=client_loaders[name]["train"],
                input_dim=input_dim,
                cfg=cfg,
                device=device,
                inner_lr=float(inner_lr),
                meta_lr=float(meta_lr),
                local_steps=int(local_steps),
            )
            local_states.append(st)

        global_state = _state_dict_weighted_average(local_states, [float(n) for n in client_sizes])
        global_model.load_state_dict(global_state, strict=True)

    return global_model


def _pfedme_personalize_from_global(
    global_model: nn.Module,
    train_loader,
    input_dim: int,
    cfg: Ch4Config,
    device,
    *,
    personal_lr: float,
    lam: float,
    personal_epochs: int,
) -> nn.Module:
    local_model = _make_model(input_dim, cfg, device)
    local_model.load_state_dict(copy.deepcopy(global_model.state_dict()), strict=True)

    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(local_model.parameters(), lr=float(personal_lr), weight_decay=float(cfg.weight_decay))

    for _ in range(int(personal_epochs)):
        train_epoch_proximal_to_anchor(
            model=local_model,
            anchor_model=global_model,
            loader=train_loader,
            optimizer=opt,
            criterion=crit,
            lam=float(lam),
            device=device,
        )
    return local_model


def run_pfedme_global_model(
    client_loaders,
    client_names: List[str],
    client_sizes: List[int],
    input_dim: int,
    cfg: Ch4Config,
    device,
    *,
    rounds: int,
    personal_lr: float,
    lam: float,
    beta: float,
    local_epochs: int,
) -> nn.Module:
    global_model = _make_model(input_dim, cfg, device)
    global_state = copy.deepcopy(global_model.state_dict())

    for _ in range(int(rounds)):
        local_anchor_states = []

        for name in client_names:
            anchor_model = _make_model(input_dim, cfg, device)
            anchor_model.load_state_dict(global_state, strict=True)

            personalized_model = _pfedme_personalize_from_global(
                global_model=anchor_model,
                train_loader=client_loaders[name]["train"],
                input_dim=input_dim,
                cfg=cfg,
                device=device,
                personal_lr=float(personal_lr),
                lam=float(lam),
                personal_epochs=int(local_epochs),
            )

            anchor_state = copy.deepcopy(anchor_model.state_dict())
            personal_state = personalized_model.state_dict()
            for k in anchor_state.keys():
                anchor_state[k] = anchor_state[k] - float(beta) * (anchor_state[k] - personal_state[k])

            local_anchor_states.append(anchor_state)

        global_state = _state_dict_weighted_average(local_anchor_states, [float(n) for n in client_sizes])
        global_model.load_state_dict(global_state, strict=True)

    return global_model


# -----------------------------------------------------------------------------
# Config builders for internal main-results table
# -----------------------------------------------------------------------------
def _cfg_global_m1_only(base_cfg: Ch4Config) -> Ch4Config:
    cfg = copy.deepcopy(base_cfg)
    cfg.use_prior_bias_calib = True
    cfg.freeze_bias_after_calib = True

    cfg.E_pers_min = 0
    cfg.E_pers_max = 0
    cfg.K_pers_min = 1
    cfg.K_pers_max = 1
    cfg.apc_candidate_selection = False
    return cfg


def _cfg_global_m3_only(base_cfg: Ch4Config) -> Ch4Config:
    cfg = _cfg_fixed_budget(base_cfg, FIXED_K_PERS, FIXED_E_PERS)
    cfg.use_prior_bias_calib = False
    cfg.freeze_bias_after_calib = False
    return cfg


def _cfg_fixed_budget(base_cfg: Ch4Config, k_pers: int, e_pers: int) -> Ch4Config:
    cfg = copy.deepcopy(base_cfg)
    cfg.use_prior_bias_calib = True
    cfg.freeze_bias_after_calib = True

    cfg.K_pers_min = int(k_pers)
    cfg.K_pers_max = int(k_pers)
    cfg.E_pers_min = int(e_pers)
    cfg.E_pers_max = int(e_pers)
    cfg.apc_candidate_selection = False
    return cfg


def _cfg_full(base_cfg: Ch4Config) -> Ch4Config:
    cfg = copy.deepcopy(base_cfg)
    cfg.use_prior_bias_calib = True
    cfg.freeze_bias_after_calib = True
    cfg.personalization_select_metric = "tradeoff"
    cfg.apc_signal_mode = "need_reliability_product"
    cfg.apc_mapping_mode = "need_reliability_product"
    cfg.apc_scope_mapping_mode = "floor"
    cfg.apc_candidate_selection = True
    return cfg


def _set_method_seed(seed: int, offset: int) -> None:
    """Reset RNG before a method so results do not depend on table ordering."""
    set_seed(int(seed) * 1000 + int(offset), deterministic=True)


# -----------------------------------------------------------------------------
# Stage I helper
# -----------------------------------------------------------------------------
def _fedavg_qbar(client_names: List[str], client_sizes: List[int]) -> Dict[str, float]:
    total = float(sum(max(0, int(s)) for s in client_sizes))
    if total <= 0.0:
        m = float(max(1, len(client_names)))
        return {str(name): 1.0 / m for name in client_names}
    return {
        str(name): float(max(0, int(size)) / total)
        for name, size in zip(client_names, client_sizes)
    }


def _train_stage1_fedavg_backbone(
    seed: int,
    cfg: Ch4Config,
    *,
    record_history: bool = True,
    history_every: int = 1,
):
    _set_method_seed(seed, 0)
    client_loaders, central, client_names, client_sizes, input_dim = build_ch4_loaders(
        pkl_path=str(DATA_PATH),
        cfg=cfg,
        seed=seed,
        val_frac=VAL_FRAC,
    )

    p_g, priors_per_client = get_priors(client_loaders, client_names, central)

    fedavg_out = run_fedavg_backbone(
        client_loaders=client_loaders,
        client_names=client_names,
        client_sizes=client_sizes,
        input_dim=input_dim,
        cfg=cfg,
        device=DEVICE,
        return_history=bool(record_history),
        history_split="val",
        history_every=int(max(1, history_every)),
    )
    if bool(record_history):
        model, stage1_history = fedavg_out
    else:
        model, stage1_history = fedavg_out, []

    q_bar = _fedavg_qbar(client_names, client_sizes)
    stage1_diag = {
        "stage1_method": "OriginalFedAvg",
        "q_bar": q_bar,
        "avg_active_clients": float(len(client_names)),
        "avg_total_real_samples": float(sum(client_sizes)),
        "fed_rounds": int(cfg.fed_rounds),
        "local_epochs_per_round": int(cfg.local_epochs_per_round),
        "aggregation": "sample_size_weighted",
        "history_split": "val",
        "history_every": int(max(1, history_every)),
    }

    return {
        "client_loaders": client_loaders,
        "central": central,
        "client_names": client_names,
        "client_sizes": client_sizes,
        "input_dim": input_dim,
        "p_g": float(p_g),
        "priors_per_client": priors_per_client,
        "backbone_model": model,
        "stage1_diag": stage1_diag,
        "stage1_history": stage1_history,
    }


def _run_full_dapfl_from_bundle(seed: int, base_cfg: Ch4Config, bundle: Dict[str, Any]) -> Dict[str, Any]:
    _set_method_seed(seed, 700)
    full_cfg = _cfg_full(base_cfg)
    return run_dapfl_stage2(
        backbone_name="StageI-FedAvg",
        backbone_model=copy.deepcopy(bundle["backbone_model"]).to(DEVICE),
        client_loaders=bundle["client_loaders"],
        central=bundle["central"],
        client_names=bundle["client_names"],
        client_sizes=bundle["client_sizes"],
        input_dim=bundle["input_dim"],
        cfg=full_cfg,
        device=DEVICE,
        qbar=bundle["stage1_diag"].get("q_bar", None),
    )


# -----------------------------------------------------------------------------
# External baselines table
# -----------------------------------------------------------------------------
def _run_external_baselines_one_repeat(
    seed: int,
    base_cfg: Ch4Config,
    *,
    bundle: Optional[Dict[str, Any]] = None,
    shared_full_res: Optional[Dict[str, Any]] = None,
    shared_full_res_by_method: Optional[Dict[str, Dict[str, Any]]] = None,
    external_full_method_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if bundle is None:
        bundle = _train_stage1_fedavg_backbone(seed, base_cfg)
    client_loaders = bundle["client_loaders"]
    client_names = bundle["client_names"]
    client_sizes = bundle["client_sizes"]
    input_dim = bundle["input_dim"]
    priors_per_client = bundle["priors_per_client"]

    stage1_model = bundle["backbone_model"]
    qbar = bundle["stage1_diag"].get("q_bar", None)

    out_metrics: Dict[str, Dict[str, Any]] = {}

    # 1) Local-Only
    print("    -> External: Local-Only")
    _set_method_seed(seed, 101)
    local_only_metrics = {}
    for name in client_names:
        model = GRUModel(input_dim, base_cfg.hidden_dim, base_cfg.num_layers, base_cfg.dropout).to(device=DEVICE)
        opt = optim.Adam(model.parameters(), lr=base_cfg.lr, weight_decay=base_cfg.weight_decay)
        crit = nn.BCEWithLogitsLoss()

        for _ in range(int(LOCAL_ONLY_EPOCHS)):
            train_epoch_standard(model, client_loaders[name]["train"], opt, crit, DEVICE)

        local_only_metrics[name] = _eval_model_on_client(model, name, client_loaders, base_cfg, DEVICE)
    out_metrics["External_Local_Only"] = _with_client_metrics(
        _aggregate_client_metrics(local_only_metrics),
        local_only_metrics,
    )

    # 2) FedAvg
    print("    -> External: FedAvg global")
    _set_method_seed(seed, 102)
    fedavg_model = copy.deepcopy(stage1_model).to(DEVICE)

    fedavg_metrics = {}
    for name in client_names:
        fedavg_metrics[name] = _eval_model_on_client(fedavg_model, name, client_loaders, base_cfg, DEVICE)
    out_metrics["External_FedAvg"] = _with_client_metrics(
        _aggregate_client_metrics(fedavg_metrics),
        fedavg_metrics,
    )

    # 3) FedAvg + Local FT
    print("    -> External: FedAvg + Local FT")
    _set_method_seed(seed, 103)
    ft_metrics = {}
    for name in client_names:
        model = copy.deepcopy(fedavg_model).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=base_cfg.personalization_lr, weight_decay=base_cfg.weight_decay)
        crit = nn.BCEWithLogitsLoss()

        for _ in range(int(FINE_TUNE_EPOCHS)):
            train_epoch_standard(model, client_loaders[name]["train"], opt, crit, DEVICE)

        ft_metrics[name] = _eval_model_on_client(model, name, client_loaders, base_cfg, DEVICE)
    out_metrics["External_FedAvg_LocalFT"] = _with_client_metrics(
        _aggregate_client_metrics(ft_metrics),
        ft_metrics,
    )

    # 4) Ditto
    print(f"    -> External: Ditto (lambda={DITTO_LAMBDA})")
    _set_method_seed(seed, 104)
    ditto_metrics = {}
    for name in client_names:
        model = copy.deepcopy(fedavg_model).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=base_cfg.personalization_lr, weight_decay=base_cfg.weight_decay)
        crit = nn.BCEWithLogitsLoss()

        for _ in range(int(FINE_TUNE_EPOCHS)):
            train_epoch_ditto(
                model=model,
                global_model=fedavg_model,
                loader=client_loaders[name]["train"],
                optimizer=opt,
                criterion=crit,
                lam=DITTO_LAMBDA,
                device=DEVICE,
            )
        ditto_metrics[name] = _eval_model_on_client(model, name, client_loaders, base_cfg, DEVICE)
    out_metrics["External_Ditto"] = _with_client_metrics(
        _aggregate_client_metrics(ditto_metrics),
        ditto_metrics,
    )

    # 5) Per-FedAvg
    print("    -> External: Per-FedAvg (FO practical)")
    _set_method_seed(seed, 105)
    perfedavg_model = run_perfedavg_backbone(
        client_loaders=client_loaders,
        client_names=client_names,
        client_sizes=client_sizes,
        input_dim=input_dim,
        cfg=base_cfg,
        device=DEVICE,
        rounds=int(PERFEDAVG_ROUNDS),
        inner_lr=float(PERFEDAVG_INNER_LR),
        meta_lr=float(PERFEDAVG_META_LR),
        local_steps=int(PERFEDAVG_LOCAL_STEPS),
    )
    perfedavg_metrics = {}
    for name in client_names:
        model = copy.deepcopy(perfedavg_model).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=base_cfg.personalization_lr, weight_decay=base_cfg.weight_decay)
        crit = nn.BCEWithLogitsLoss()

        for _ in range(int(PERFEDAVG_PERSONAL_EPOCHS)):
            train_epoch_standard(model, client_loaders[name]["train"], opt, crit, DEVICE)

        perfedavg_metrics[name] = _eval_model_on_client(model, name, client_loaders, base_cfg, DEVICE)
    out_metrics["External_PerFedAvg_FO"] = _with_client_metrics(
        _aggregate_client_metrics(perfedavg_metrics),
        perfedavg_metrics,
    )

    # 6) pFedMe
    print("    -> External: pFedMe (practical proximal)")
    _set_method_seed(seed, 106)
    pfedme_global = run_pfedme_global_model(
        client_loaders=client_loaders,
        client_names=client_names,
        client_sizes=client_sizes,
        input_dim=input_dim,
        cfg=base_cfg,
        device=DEVICE,
        rounds=int(PFEDME_ROUNDS),
        personal_lr=float(PFEDME_PERSONAL_LR),
        lam=float(PFEDME_LAMBDA),
        beta=float(PFEDME_BETA),
        local_epochs=int(PFEDME_LOCAL_EPOCHS),
    )
    pfedme_metrics = {}
    for name in client_names:
        model = _pfedme_personalize_from_global(
            global_model=pfedme_global,
            train_loader=client_loaders[name]["train"],
            input_dim=input_dim,
            cfg=base_cfg,
            device=DEVICE,
            personal_lr=float(PFEDME_PERSONAL_LR),
            lam=float(PFEDME_LAMBDA),
            personal_epochs=int(PFEDME_PERSONAL_EPOCHS),
        )
        pfedme_metrics[name] = _eval_model_on_client(model, name, client_loaders, base_cfg, DEVICE)
    out_metrics["External_pFedMe"] = _with_client_metrics(
        _aggregate_client_metrics(pfedme_metrics),
        pfedme_metrics,
    )

    # 7) Full DA-PFL variants
    full_names = list(external_full_method_names or ["External_Full_DAPFL"])
    for method_name in full_names:
        print(f"    -> External: {method_name}")
        if shared_full_res_by_method is not None and method_name in shared_full_res_by_method:
            full_res = shared_full_res_by_method[method_name]
        else:
            full_res = shared_full_res if shared_full_res is not None else _run_full_dapfl_from_bundle(seed, base_cfg, bundle)
        out_metrics[method_name] = full_res

    out_metrics["_subgroup_analysis"] = _subgroup_analysis(
        method_results={
            k: v
            for k, v in out_metrics.items()
            if isinstance(v, dict) and k.startswith("External_")
        },
        client_names=client_names,
        priors_per_client=priors_per_client,
        client_sizes=client_sizes,
    )

    out_metrics["_stage1_diag"] = {
        "stage1_method": bundle["stage1_diag"].get("stage1_method", "OriginalFedAvg"),
        "q_bar": bundle["stage1_diag"].get("q_bar", {}),
        "avg_total_real_samples": bundle["stage1_diag"].get("avg_total_real_samples", 0.0),
        "avg_active_clients": bundle["stage1_diag"].get("avg_active_clients", 0.0),
        "aggregation": bundle["stage1_diag"].get("aggregation", "sample_size_weighted"),
        "fed_rounds": bundle["stage1_diag"].get("fed_rounds", None),
        "local_epochs_per_round": bundle["stage1_diag"].get("local_epochs_per_round", None),
        "history_split": bundle["stage1_diag"].get("history_split", "val"),
        "history_every": bundle["stage1_diag"].get("history_every", None),
    }
    return out_metrics


# -----------------------------------------------------------------------------
# Internal main-results table
# -----------------------------------------------------------------------------
def _eval_global_only(backbone_model, client_loaders, client_names: List[str], cfg: Ch4Config) -> Dict[str, Any]:
    metrics = {}
    for name in client_names:
        thr = _choose_threshold_from_val(backbone_model, client_loaders[name]["val"], DEVICE, cfg)
        metrics[name] = _eval_on_test_with_threshold(backbone_model, client_loaders[name]["test"], DEVICE, cfg, thr)
    return _with_client_metrics(_aggregate_client_metrics(metrics), metrics)


def _run_internal_main_table_one_repeat(
    seed: int,
    base_cfg: Ch4Config,
    *,
    bundle: Optional[Dict[str, Any]] = None,
    shared_full_res: Optional[Dict[str, Any]] = None,
    shared_full_res_by_method: Optional[Dict[str, Dict[str, Any]]] = None,
    internal_full_method_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if bundle is None:
        bundle = _train_stage1_fedavg_backbone(seed, base_cfg)
    client_loaders = bundle["client_loaders"]
    central = bundle["central"]
    client_names = bundle["client_names"]
    client_sizes = bundle["client_sizes"]
    input_dim = bundle["input_dim"]
    priors_per_client = bundle["priors_per_client"]
    backbone_model = bundle["backbone_model"]
    qbar = bundle["stage1_diag"].get("q_bar", None)

    out: Dict[str, Dict[str, Any]] = {}

    print("    -> Internal: Global only (Stage I)")
    _set_method_seed(seed, 201)
    out["Main_Global_Only_StageI"] = _eval_global_only(
        copy.deepcopy(backbone_model).to(DEVICE),
        client_loaders,
        client_names,
        base_cfg,
    )

    variants: List[Tuple[str, Ch4Config]] = [
        ("Main_Global_M1_Only", _cfg_global_m1_only(base_cfg)),
        ("Main_Global_M3_Only", _cfg_global_m3_only(base_cfg)),
        (
            "Main_Global_M1_M3_FixedBudget",
            _cfg_fixed_budget(base_cfg, FIXED_K_PERS, FIXED_E_PERS),
        ),
    ]

    for idx, (method_name, cfg_variant) in enumerate(variants):
        print(f"    -> Internal: {method_name}")
        _set_method_seed(seed, 202 + idx)
        res = run_dapfl_stage2(
            backbone_name="StageI-FedAvg",
            backbone_model=copy.deepcopy(backbone_model).to(DEVICE),
            client_loaders=client_loaders,
            central=central,
            client_names=client_names,
            client_sizes=client_sizes,
            input_dim=input_dim,
            cfg=cfg_variant,
            device=DEVICE,
            qbar=qbar,
        )
        out[method_name] = res

    full_names = list(internal_full_method_names or ["Main_Full_DAPFL"])
    for method_name in full_names:
        print(f"    -> Internal: {method_name}")
        if shared_full_res_by_method is not None and method_name in shared_full_res_by_method:
            out[method_name] = shared_full_res_by_method[method_name]
        else:
            out[method_name] = (
                shared_full_res if shared_full_res is not None else _run_full_dapfl_from_bundle(seed, base_cfg, bundle)
            )

    out["_subgroup_analysis"] = _subgroup_analysis(
        method_results={
            k: v
            for k, v in out.items()
            if isinstance(v, dict) and k.startswith("Main_")
        },
        client_names=client_names,
        priors_per_client=priors_per_client,
        client_sizes=client_sizes,
    )

    out["_stage1_diag"] = {
        "stage1_method": bundle["stage1_diag"].get("stage1_method", "OriginalFedAvg"),
        "q_bar": bundle["stage1_diag"].get("q_bar", {}),
        "avg_total_real_samples": bundle["stage1_diag"].get("avg_total_real_samples", 0.0),
        "avg_active_clients": bundle["stage1_diag"].get("avg_active_clients", 0.0),
        "aggregation": bundle["stage1_diag"].get("aggregation", "sample_size_weighted"),
        "fed_rounds": bundle["stage1_diag"].get("fed_rounds", None),
        "local_epochs_per_round": bundle["stage1_diag"].get("local_epochs_per_round", None),
        "history_split": bundle["stage1_diag"].get("history_split", "val"),
        "history_every": bundle["stage1_diag"].get("history_every", None),
    }
    return out


# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------
def _summarize_runs(raw_runs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    metrics_to_summarize = [
        "global_auc",
        "macro_auc",
        "macro_f1",
        "macro_ece",
        "gini",
    ]

    summary: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for method_name, runs in raw_runs.items():
        summary[method_name] = {}
        for metric in metrics_to_summarize:
            vals = [r.get(metric, np.nan) for r in runs if np.isfinite(r.get(metric, np.nan))]
            if vals:
                summary[method_name][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
            else:
                summary[method_name][metric] = {"mean": None, "std": None}
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-facing DA-PFL comparison tables."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=N_REPEATS,
        help="Number of repeated seeds used for mean/std tables.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Repeat r uses seed + r.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated explicit seed list. When provided, overrides --repeats/--seed, e.g. 42,43,46,47,49.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=SAVE_NAME,
        help="JSON filename written under out/ch5_compare.",
    )
    parser.add_argument(
        "--stage1-history-every",
        type=int,
        default=1,
        help="Record Stage-I validation diagnostics every N FedAvg rounds.",
    )
    parser.add_argument(
        "--fed-rounds",
        type=int,
        default=40,
        help="Stage-I FedAvg communication rounds. Use 20/30/40/60 for convergence sensitivity.",
    )
    parser.add_argument(
        "--local-epochs-per-round",
        type=int,
        default=2,
        help="Local epochs per FedAvg round.",
    )
    parser.add_argument(
        "--apc-signal-mode",
        type=str,
        default="need_reliability_product",
        help="Full DA-PFL M2 signal mode. Use need_reliability_product for the final method; reliability_prior keeps the old M2 baseline.",
    )
    parser.add_argument(
        "--apc-signal-modes",
        type=str,
        default=None,
        help="Comma-separated Full DA-PFL M2 modes: need_reliability_product,reliability_prior.",
    )
    return parser.parse_args()


def _resolve_run_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds is not None and str(args.seeds).strip():
        seeds: List[int] = []
        for raw in _parse_csv_strings(str(args.seeds)):
            try:
                seeds.append(int(raw))
            except ValueError as exc:
                raise ValueError(f"Invalid seed value in --seeds: {raw!r}") from exc
        if not seeds:
            raise ValueError("--seeds was provided but no valid seeds were parsed.")
        return seeds
    return [int(args.seed) + r for r in range(int(args.repeats))]


def _summary_to_paper_rows(
    summary: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    method_names: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method_name in method_names:
        row: Dict[str, Any] = {
            "method_key": method_name,
            "paper_label": PAPER_METHOD_LABELS.get(method_name, method_name),
            "paper_description": PAPER_METHOD_DESCRIPTIONS.get(method_name, ""),
        }
        for metric in PAPER_CORE_METRICS:
            mm = summary.get(method_name, {}).get(metric, {})
            row[f"{metric}_mean"] = mm.get("mean")
            row[f"{metric}_std"] = mm.get("std")
        rows.append(row)
    return rows


def _collect_subgroup_runs(
    accumulator: Dict[str, List[Dict[str, Any]]],
    subgroup_one_repeat: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    for method_name, groups in subgroup_one_repeat.items():
        for group_name, metrics in groups.items():
            key = f"{method_name}__{group_name}"
            accumulator.setdefault(key, []).append(_compact_metrics(metrics))


def _subgroup_summary_rows(
    summary: Dict[str, Dict[str, Dict[str, Optional[float]]]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(summary.keys()):
        if "__" in key:
            method_name, group_name = key.split("__", 1)
        else:
            method_name, group_name = key, ""
        row: Dict[str, Any] = {
            "method_key": method_name,
            "paper_label": PAPER_METHOD_LABELS.get(method_name, method_name),
            "subgroup": group_name,
        }
        for metric in PAPER_CORE_METRICS:
            mm = summary.get(key, {}).get(metric, {})
            row[f"{metric}_mean"] = mm.get("mean")
            row[f"{metric}_std"] = mm.get("std")
        rows.append(row)
    return rows


def _write_paper_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_live_line(method_name: str, res: Dict[str, Any]) -> None:
    print(
        f"  - {method_name:<34s} | "
        f"gAUC: {res.get('global_auc', float('nan')):.4f} | "
        f"mAUC: {res.get('macro_auc', float('nan')):.4f} | "
        f"mF1: {res.get('macro_f1', float('nan')):.4f} | "
        f"mECE: {res.get('macro_ece', float('nan')):.4f} | "
        f"Gini: {res.get('gini', float('nan')):.4f}"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    run_seeds = _resolve_run_seeds(args)
    n_runs = int(len(run_seeds))
    base_seed = int(run_seeds[0]) if run_seeds else int(args.seed)
    print(f"[Device] {DEVICE}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    base_cfg = Ch4Config(
        seed=int(base_seed),
        fed_rounds=int(args.fed_rounds),
        local_epochs_per_round=int(args.local_epochs_per_round),
    )
    apc_signal_modes = _clean_apc_signal_modes(
        args.apc_signal_modes if args.apc_signal_modes is not None else args.apc_signal_mode
    )
    base_cfg.apc_signal_mode = str(apc_signal_modes[0])
    base_cfg.apc_mapping_mode = (
        "need_reliability_product"
        if str(apc_signal_modes[0]) == "need_reliability_product"
        else "log_median"
    )
    multi_apc_modes = len(apc_signal_modes) > 1
    _register_full_method_labels(apc_signal_modes, multi_mode=multi_apc_modes)
    external_full_method_names = [
        _full_method_key("External", mode, multi_mode=multi_apc_modes)
        for mode in apc_signal_modes
    ]
    internal_full_method_names = [
        _full_method_key("Main", mode, multi_mode=multi_apc_modes)
        for mode in apc_signal_modes
    ]
    stage1_paper_cfg = {
        "method": "OriginalFedAvg",
        "participation": "full",
        "aggregation": "sample_size_weighted",
        "fed_rounds": int(base_cfg.fed_rounds),
        "local_epochs_per_round": int(base_cfg.local_epochs_per_round),
        "base_seed": int(base_cfg.seed),
        "run_seeds": list(run_seeds),
        "full_dapfl_apc_signal_mode": str(base_cfg.apc_signal_mode),
        "full_dapfl_apc_signal_modes": list(apc_signal_modes),
    }

    external_method_names = [
        "External_Local_Only",
        "External_FedAvg",
        "External_FedAvg_LocalFT",
        "External_Ditto",
        "External_PerFedAvg_FO",
        "External_pFedMe",
    ] + external_full_method_names
    internal_method_names = [
        "Main_Global_Only_StageI",
        "Main_Global_M1_Only",
        "Main_Global_M3_Only",
        "Main_Global_M1_M3_FixedBudget",
    ] + internal_full_method_names

    external_raw_runs: Dict[str, List[Dict[str, Any]]] = {k: [] for k in external_method_names}
    internal_raw_runs: Dict[str, List[Dict[str, Any]]] = {k: [] for k in internal_method_names}
    external_subgroup_runs: Dict[str, List[Dict[str, Any]]] = {}
    internal_subgroup_runs: Dict[str, List[Dict[str, Any]]] = {}
    external_stage1_logs: List[Dict[str, Any]] = []
    internal_stage1_logs: List[Dict[str, Any]] = []
    external_client_level_rows: List[Dict[str, Any]] = []
    internal_client_level_rows: List[Dict[str, Any]] = []
    stage1_history_rows: List[Dict[str, Any]] = []

    print(
        "[Config] repeats={} | seeds={} | Stage-I=OriginalFedAvg full participation | M2 modes={} | external={} | internal={} | metrics=gAUC/mAUC/mF1/mECE/Gini".format(
            int(n_runs),
            list(run_seeds),
            list(apc_signal_modes),
            len(external_method_names),
            len(internal_method_names),
        )
    )

    t_all0 = time.perf_counter()
    for r, seed_r in enumerate(run_seeds):
        seed_r = int(seed_r)
        print(f"\n[Repeat {r + 1}/{int(n_runs)}] seed={seed_r}")

        print("  [Shared] Stage I FedAvg backbone")
        bundle = _train_stage1_fedavg_backbone(
            seed_r,
            base_cfg,
            record_history=True,
            history_every=int(max(1, args.stage1_history_every)),
        )
        stage1_history_rows.extend(_stage1_history_rows(
            repeat=r,
            seed=seed_r,
            history=bundle.get("stage1_history", []),
        ))

        shared_full_external: Dict[str, Dict[str, Any]] = {}
        shared_full_internal: Dict[str, Dict[str, Any]] = {}
        for mode, ext_key, int_key in zip(apc_signal_modes, external_full_method_names, internal_full_method_names):
            print(f"  [Shared] Full DA-PFL mode={mode} (reused by Table A and Table B)")
            cfg_mode = copy.deepcopy(base_cfg)
            cfg_mode.apc_signal_mode = str(mode)
            cfg_mode.apc_mapping_mode = "need_reliability_product" if str(mode) == "need_reliability_product" else "log_median"
            full_res = _run_full_dapfl_from_bundle(seed_r, cfg_mode, bundle)
            shared_full_external[ext_key] = full_res
            shared_full_internal[int_key] = full_res

        print("  [Table A] External baselines")
        ext_one = _run_external_baselines_one_repeat(
            seed_r,
            base_cfg,
            bundle=bundle,
            shared_full_res_by_method=shared_full_external,
            external_full_method_names=external_full_method_names,
        )
        ext_subgroup = ext_one.pop("_subgroup_analysis")
        _collect_subgroup_runs(external_subgroup_runs, ext_subgroup)
        external_stage1_logs.append(ext_one.pop("_stage1_diag"))
        for method_name in external_method_names:
            external_client_level_rows.extend(_client_level_rows(
                table="external",
                repeat=r,
                seed=seed_r,
                method_name=method_name,
                result=ext_one[method_name],
            ))
            external_raw_runs[method_name].append(_compact_metrics(ext_one[method_name]))
            _print_live_line(method_name, ext_one[method_name])

        print("  [Table B] Internal main-results table")
        int_one = _run_internal_main_table_one_repeat(
            seed_r,
            base_cfg,
            bundle=bundle,
            shared_full_res_by_method=shared_full_internal,
            internal_full_method_names=internal_full_method_names,
        )
        for ext_key, int_key in zip(external_full_method_names, internal_full_method_names):
            if _compact_metrics(ext_one[ext_key]) != _compact_metrics(int_one[int_key]):
                raise RuntimeError(
                    f"Shared Full DA-PFL result mismatch between Table A and Table B: {ext_key} vs {int_key}."
                )
        int_subgroup = int_one.pop("_subgroup_analysis")
        _collect_subgroup_runs(internal_subgroup_runs, int_subgroup)
        internal_stage1_logs.append(int_one.pop("_stage1_diag"))
        for method_name in internal_method_names:
            internal_client_level_rows.extend(_client_level_rows(
                table="internal",
                repeat=r,
                seed=seed_r,
                method_name=method_name,
                result=int_one[method_name],
            ))
            internal_raw_runs[method_name].append(_compact_metrics(int_one[method_name]))
            _print_live_line(method_name, int_one[method_name])

    t_all = float(time.perf_counter() - t_all0)

    external_summary = _summarize_runs(external_raw_runs)
    internal_summary = _summarize_runs(internal_raw_runs)
    external_subgroup_summary = _summarize_runs(external_subgroup_runs)
    internal_subgroup_summary = _summarize_runs(internal_subgroup_runs)
    stage1_history_summary = _stage1_history_summary_rows(stage1_history_rows)
    budget_summary_rows = _budget_summary_rows(external_client_level_rows + internal_client_level_rows)

    payload = {
        "exp_name": "ch5_all_tables",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_repeats": int(n_runs),
        "run_seeds": list(run_seeds),
        "wall_time_sec": float(t_all),
        "device": str(DEVICE),
        "data_path": str(DATA_PATH),
        "base_cfg": asdict(base_cfg),
        "stage1_cfg": stage1_paper_cfg,
        "stage1_convergence": {
            "raw_rows": stage1_history_rows,
            "summary_rows": stage1_history_summary,
            "record_every_rounds": int(max(1, args.stage1_history_every)),
        },
        "paper_alignment": {
            "stage1_selection": stage1_paper_cfg,
            "method_labels": PAPER_METHOD_LABELS,
            "method_descriptions": PAPER_METHOD_DESCRIPTIONS,
            "module_notes": PAPER_MODULE_NOTES,
            "protection_notes": PAPER_PROTECTION_NOTES,
            "core_metrics": PAPER_CORE_METRICS,
            "core_metric_labels": {
                "global_auc": "Global AUC",
                "macro_auc": "Macro AUC",
                "macro_f1": "Macro F1",
                "macro_ece": "Macro ECE",
                "gini": "Gini",
            },
            "lower_is_better": ["macro_ece", "gini"],
        },
        "notes": {
            "external_baselines_included": external_method_names,
            "fixed_budget_variant": {
                "K_pers": int(FIXED_K_PERS),
                "E_pers": int(FIXED_E_PERS),
            },
            "PerFedAvg_impl": "practical first-order approximation built on the current codebase",
            "pFedMe_impl": "practical proximal personalized implementation built on the current codebase",
            "full_dapfl_reuse": "Within each repeat, External_Full_DAPFL and Main_Full_DAPFL reuse the same Stage-I backbone and the same Full DA-PFL run.",
            "full_dapfl_apc_signal_mode": str(base_cfg.apc_signal_mode),
            "budget_accounting": {
                "group_epoch_budget": "E_pers * selected_group_count / 7; full-model 5-epoch baselines have budget 5.0.",
                "param_epoch_budget": "E_pers * trainable_param_count / total_param_count; this accounts for unequal parameter-group sizes.",
                "local_only_note": "Local-Only is trained from scratch and is not a post-FedAvg personalization budget.",
            },
        },
        "external_table": {
            "methods": external_method_names,
            "summary": external_summary,
            "raw_runs": external_raw_runs,
            "client_level_runs": external_client_level_rows,
            "stage1_logs": external_stage1_logs,
            "subgroup_summary": external_subgroup_summary,
            "subgroup_raw_runs": external_subgroup_runs,
        },
        "internal_table": {
            "methods": internal_method_names,
            "summary": internal_summary,
            "raw_runs": internal_raw_runs,
            "client_level_runs": internal_client_level_rows,
            "stage1_logs": internal_stage1_logs,
            "subgroup_summary": internal_subgroup_summary,
            "subgroup_raw_runs": internal_subgroup_runs,
        },
        "budget_summary": budget_summary_rows,
    }

    out_path = OUT_DIR / str(args.output_name)
    out_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    external_csv = OUT_DIR / "paper_external_summary.csv"
    internal_csv = OUT_DIR / "paper_internal_summary.csv"
    external_subgroup_csv = OUT_DIR / "paper_external_subgroup_summary.csv"
    internal_subgroup_csv = OUT_DIR / "paper_internal_subgroup_summary.csv"
    external_client_csv = OUT_DIR / "paper_external_client_level.csv"
    internal_client_csv = OUT_DIR / "paper_internal_client_level.csv"
    stage1_history_csv = OUT_DIR / "paper_stage1_convergence.csv"
    stage1_history_summary_csv = OUT_DIR / "paper_stage1_convergence_summary.csv"
    budget_summary_csv = OUT_DIR / "paper_budget_summary.csv"
    write_paper_csvs = int(n_runs) > 0
    if write_paper_csvs:
        _write_paper_csv(
            external_csv,
            _summary_to_paper_rows(external_summary, external_method_names),
        )
        _write_paper_csv(
            internal_csv,
            _summary_to_paper_rows(internal_summary, internal_method_names),
        )
        _write_paper_csv(external_subgroup_csv, _subgroup_summary_rows(external_subgroup_summary))
        _write_paper_csv(internal_subgroup_csv, _subgroup_summary_rows(internal_subgroup_summary))
        _write_paper_csv(external_client_csv, external_client_level_rows)
        _write_paper_csv(internal_client_csv, internal_client_level_rows)
        _write_paper_csv(stage1_history_csv, stage1_history_rows)
        _write_paper_csv(stage1_history_summary_csv, stage1_history_summary)
        _write_paper_csv(budget_summary_csv, budget_summary_rows)

    print("\n" + "=" * 96)
    print(f"[Done] Results saved to: {out_path}")
    if write_paper_csvs:
        print(f"[Done] Paper CSV saved to: {external_csv}")
        print(f"[Done] Paper CSV saved to: {internal_csv}")
        print(f"[Done] Subgroup CSV saved to: {external_subgroup_csv}")
        print(f"[Done] Subgroup CSV saved to: {internal_subgroup_csv}")
        print(f"[Done] Client-level CSV saved to: {external_client_csv}")
        print(f"[Done] Client-level CSV saved to: {internal_client_csv}")
        print(f"[Done] Stage-I convergence CSV saved to: {stage1_history_csv}")
        print(f"[Done] Stage-I convergence summary CSV saved to: {stage1_history_summary_csv}")
        print(f"[Done] Budget summary CSV saved to: {budget_summary_csv}")
    else:
        print("[Done] repeats=0 structure check: paper CSV files were not overwritten.")
    print("=" * 96)

    print("\n[External Baseline Summary]")
    for method_name in external_method_names:
        print(f"\n{method_name}")
        for metric in ["global_auc", "macro_auc", "macro_f1", "macro_ece", "gini"]:
            mm = external_summary[method_name].get(metric, {})
            mean_v = mm.get("mean")
            std_v = mm.get("std")
            mean_s = f"{mean_v:.4f}" if mean_v is not None else "None"
            std_s = f"{std_v:.4f}" if std_v is not None else "None"
            print(f"  {metric:<14s}: {mean_s} +/- {std_s}")

    print("\n[Internal Main-Results Summary]")
    for method_name in internal_method_names:
        print(f"\n{method_name}")
        for metric in ["global_auc", "macro_auc", "macro_f1", "macro_ece", "gini"]:
            mm = internal_summary[method_name].get(metric, {})
            mean_v = mm.get("mean")
            std_v = mm.get("std")
            mean_s = f"{mean_v:.4f}" if mean_v is not None else "None"
            std_s = f"{std_v:.4f}" if std_v is not None else "None"
            print(f"  {metric:<14s}: {mean_s} +/- {std_s}")


if __name__ == "__main__":
    main()

