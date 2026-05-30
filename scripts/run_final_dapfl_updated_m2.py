from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dapfl_pipeline import run_dapfl_stage2, set_seed  # noqa: E402
from models.main import (  # noqa: E402
    DEVICE,
    FINE_TUNE_EPOCHS,
    FIXED_E_PERS,
    FIXED_K_PERS,
    Ch4Config,
    _cfg_fixed_budget,
    _cfg_full,
    _cfg_global_m1_only,
    _compact_metrics,
    _eval_global_only,
    _train_stage1_fedavg_backbone,
)


SEEDS = [42, 43, 46, 47, 49]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CORE_METRICS = ["global_auc", "macro_auc", "macro_f1", "macro_ece", "gini"]
METHOD_ORDER = [
    "Backbone Only",
    "Backbone + M1",
    "Backbone + M1 + Head FT",
    "Backbone + M1 + M3 without M2/APC",
    "Full DA-PFL old M2",
    "Full DA-PFL new M2",
    "M1 + new M2 + random mask",
    "M1 + full fine-tuning",
]
GROUP_ORDER = ["g1_head", "g2_l0_ih", "g3_l0_hh", "g4_l0_b", "g5_l1_ih", "g6_l1_hh", "g7_l1_b"]


def _parse_csv(text: Optional[str], cast=str) -> List[Any]:
    if text is None or not str(text).strip():
        return []
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    if vals.size == 1:
        return {"mean": float(vals[0]), "std": 0.0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}


def _summarize(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    methods = [m for m in METHOD_ORDER if any(str(r.get("method")) == m for r in run_rows)]
    for method in methods:
        group = [r for r in run_rows if str(r.get("method")) == method]
        row: Dict[str, Any] = {"method": method, "n_runs": int(len(group))}
        for metric in CORE_METRICS:
            stats = _mean_std(float(r.get(metric, float("nan"))) for r in group)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    return rows


def _metric_row(method: str, seed: int, result: Dict[str, Any]) -> Dict[str, Any]:
    compact = _compact_metrics(result)
    row = {"method": str(method), "seed": int(seed)}
    for metric in CORE_METRICS:
        row[metric] = float(compact.get(metric, float("nan")))
    return row


def _client_rows(method: str, seed: int, result: Dict[str, Any], client_ids: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    client_metrics = result.get("client_metrics", {}) or {}
    client_debug = result.get("client_debug", {}) or {}
    for client in sorted(set(client_metrics.keys()) | set(client_debug.keys())):
        metrics = client_metrics.get(client, {}) or {}
        dbg = client_debug.get(client, {}) or {}
        selected = dbg.get("selected_groups", [])
        if not isinstance(selected, list):
            selected = []
        rows.append({
            "method": str(method),
            "seed": int(seed),
            "client_id": int(client_ids.get(client, -1)),
            "client_name": str(client),
            "test_auc": _safe_float(metrics.get("auc")),
            "test_f1": _safe_float(metrics.get("f1")),
            "test_ece": _safe_float(metrics.get("ece")),
            "mean_pred_prob": _safe_mean(metrics.get("y_prob")),
            "positive_rate": _safe_mean(metrics.get("y_true")),
            "val_bce": _safe_float(dbg.get("stage2_val_bce_after_m3")),
            "threshold": _safe_float(metrics.get("best_threshold", dbg.get("thr"))),
            "m_k": _safe_int(dbg.get("m_k", dbg.get("K_pers"))),
            "e_k": _safe_int(dbg.get("e_k", dbg.get("E_pers"))),
            "selected_groups": ";".join(str(x) for x in selected),
            "active_group_count": _safe_int(dbg.get("selected_group_count"), len(selected)),
            "stage2_drift_norm": _safe_float(dbg.get("stage2_drift_norm", dbg.get("realized_drift"))),
            "realized_drift_global": _safe_float(dbg.get("realized_drift")),
        })
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _safe_mean(value: Any, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(np.mean(arr))
    except Exception:
        return float(default)


def _old_map(q: float, gamma: float, cfg: Ch4Config) -> Dict[str, float]:
    q = max(0.0, float(q))
    gamma = max(float(gamma), float(cfg.apc_eps_num))
    r = q / (q + gamma) if q > 0.0 else 0.0
    g = max(1, int(cfg.apc_G_groups))
    m = 1 + int(math.floor(float(g - 1) * float(r)))
    e = int(cfg.E_pers_min) + int(math.floor(float(cfg.E_pers_max - cfg.E_pers_min) * float(r)))
    return {
        "r": float(np.clip(r, 0.0, 1.0)),
        "m": int(np.clip(m, int(cfg.K_pers_min), int(cfg.K_pers_max))),
        "e": int(np.clip(e, int(cfg.E_pers_min), int(cfg.E_pers_max))),
    }


def _m2_rows(seed: int, result: Dict[str, Any], client_ids: Dict[str, int], cfg: Ch4Config) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client, dbg in sorted((result.get("client_debug", {}) or {}).items()):
        q = _safe_float(dbg.get("apc_q_reliability", dbg.get("q_k")))
        old1 = _old_map(q, 1.0, cfg)
        old3 = _old_map(q, 3.0, cfg)
        rows.append({
            "seed": int(seed),
            "client_id": int(client_ids.get(client, -1)),
            "client_name": str(client),
            "n_k_pos": _safe_int(dbg.get("n_pos_i")),
            "n_k_neg": _safe_int(dbg.get("n_neg_i")),
            "n_k": _safe_int(dbg.get("apc_n_k", dbg.get("n_i"))),
            "n_total_pos": _safe_int(dbg.get("n_pos_g")),
            "n_total_neg": _safe_int(dbg.get("n_neg_g")),
            "pi_k": _safe_float(dbg.get("apc_pi_k", dbg.get("pi_k_tilde"))),
            "pi_ref": _safe_float(dbg.get("apc_pi_ref", dbg.get("pi_g_tilde"))),
            "delta_b": _safe_float(dbg.get("apc_delta_b", dbg.get("delta_b_k"))),
            "delta": _safe_float(dbg.get("apc_delta", dbg.get("delta_k"))),
            "sigma_delta": _safe_float(dbg.get("apc_sigma_delta", dbg.get("sigma_delta_k"))),
            "q_reliability": q,
            "s_reliability": _safe_float(dbg.get("apc_s_reliability", dbg.get("s_k"))),
            "eta_k": _safe_float(dbg.get("apc_s_reliability", dbg.get("s_k"))),
            "u_k": _safe_float(dbg.get("apc_u_k", dbg.get("u_k"))),
            "alpha_k": _safe_float(dbg.get("apc_alpha_k", dbg.get("alpha_k"))),
            "gamma_s": _safe_float(dbg.get("apc_gamma_s", dbg.get("gamma_s"))),
            "gamma_s_scaled": _safe_float(dbg.get("apc_gamma_s_scaled")),
            "r_reliability": _safe_float(dbg.get("apc_r_reliability", dbg.get("r_k"))),
            "r_final": _safe_float(dbg.get("apc_r_final", dbg.get("r_final", dbg.get("apc_r_reliability")))),
            "m_k": _safe_int(dbg.get("m_k")),
            "e_k": _safe_int(dbg.get("e_k")),
            "r_old_gamma_1": old1["r"],
            "r_old_gamma_3": old3["r"],
            "m_old_gamma_1": old1["m"],
            "e_old_gamma_1": old1["e"],
            "m_old_gamma_3": old3["m"],
            "e_old_gamma_3": old3["e"],
        })
    return rows


def _m3_rows(
    seed: int,
    result: Dict[str, Any],
    client_ids: Dict[str, int],
    *,
    before_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    selected_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    metrics = result.get("client_metrics", {}) or {}
    before_metrics = (before_result or {}).get("client_metrics", {}) or {}
    for client, dbg in sorted((result.get("client_debug", {}) or {}).items()):
        selected = dbg.get("selected_groups", [])
        if not isinstance(selected, list):
            selected = []
        scores = dbg.get("m3_scores", {}) or {}
        drift = dbg.get("per_group_normalized_drift", {}) or {}
        row: Dict[str, Any] = {
            "seed": int(seed),
            "client_id": int(client_ids.get(client, -1)),
            "client_name": str(client),
            "m_k": _safe_int(dbg.get("m_k")),
            "e_k": _safe_int(dbg.get("e_k")),
            "selected_groups": ";".join(str(x) for x in selected),
            "active_group_count": int(len(set(selected))),
            "total_drift_norm": _safe_float(dbg.get("stage2_drift_norm", dbg.get("realized_drift"))),
            "val_bce_before_m3": _safe_float(dbg.get("stage2_val_bce_before_m3")),
            "val_bce_after_m3": _safe_float(dbg.get("stage2_val_bce_after_m3")),
            "test_auc_before_m3": _safe_float((before_metrics.get(client, {}) or {}).get("auc")),
            "test_auc_after_m3": _safe_float((metrics.get(client, {}) or {}).get("auc")),
            "test_f1_before_m3": _safe_float((before_metrics.get(client, {}) or {}).get("f1")),
            "test_f1_after_m3": _safe_float((metrics.get(client, {}) or {}).get("f1")),
            "test_ece_before_m3": _safe_float((before_metrics.get(client, {}) or {}).get("ece")),
            "test_ece_after_m3": _safe_float((metrics.get(client, {}) or {}).get("ece")),
            "test_auc": _safe_float((metrics.get(client, {}) or {}).get("auc")),
            "test_f1": _safe_float((metrics.get(client, {}) or {}).get("f1")),
            "test_ece": _safe_float((metrics.get(client, {}) or {}).get("ece")),
        }
        for group in GROUP_ORDER:
            row[f"selected_{group}"] = int(group in set(selected))
            row[f"score_{group}"] = _safe_float(scores.get(group))
            row[f"drift_{group}"] = _safe_float(drift.get(group))
            score_rows.append({
                "seed": int(seed),
                "client_id": int(client_ids.get(client, -1)),
                "client_name": str(client),
                "group_name": group,
                "group": group,
                "selected": int(group in set(selected)),
                "group_score": _safe_float(scores.get(group)),
                "score": _safe_float(scores.get(group)),
                "normalized_drift": _safe_float(drift.get(group)),
            })
        selected_rows.append(row)
    return {"selected": selected_rows, "scores": score_rows}


def _run_stage2(seed: int, method: str, cfg: Ch4Config, bundle: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(int(seed) * 1000 + _method_offset(method), deterministic=True)
    return run_dapfl_stage2(
        backbone_name=f"StageI-FedAvg-{method}",
        backbone_model=copy.deepcopy(bundle["backbone_model"]).to(DEVICE),
        client_loaders=bundle["client_loaders"],
        central=bundle["central"],
        client_names=bundle["client_names"],
        client_sizes=bundle["client_sizes"],
        input_dim=int(bundle["input_dim"]),
        cfg=cfg,
        device=DEVICE,
        qbar=bundle["stage1_diag"].get("q_bar", None),
    )


def _method_offset(method: str) -> int:
    return {
        "Backbone + M1": 301,
        "Backbone + M1 + Head FT": 302,
        "Backbone + M1 + M3 without M2/APC": 303,
        "Full DA-PFL old M2": 304,
        "Full DA-PFL new M2": 305,
        "M1 + new M2 + random mask": 306,
        "M1 + full fine-tuning": 307,
    }.get(str(method), 300)


def _make_cfgs(base_cfg: Ch4Config) -> Dict[str, Ch4Config]:
    head_ft = _cfg_fixed_budget(base_cfg, 1, FINE_TUNE_EPOCHS)

    fixed = _cfg_fixed_budget(base_cfg, FIXED_K_PERS, FIXED_E_PERS)

    old_m2 = _cfg_full(base_cfg)
    old_m2.apc_mapping_mode = "direct_gamma"
    old_m2.apc_direct_gamma = 1.96

    new_m2 = _cfg_full(base_cfg)
    new_m2.apc_mapping_mode = "need_reliability_product"
    new_m2.apc_signal_mode = "need_reliability_product"
    new_m2.apc_scope_mapping_mode = "floor"

    random_mask = copy.deepcopy(new_m2)
    random_mask.m3_selection_strategy = "random"

    full_ft = _cfg_fixed_budget(base_cfg, 7, FINE_TUNE_EPOCHS)
    full_ft.freeze_bias_after_calib = False

    return {
        "Backbone + M1": _cfg_global_m1_only(base_cfg),
        "Backbone + M1 + Head FT": head_ft,
        "Backbone + M1 + M3 without M2/APC": fixed,
        "Full DA-PFL old M2": old_m2,
        "Full DA-PFL new M2": new_m2,
        "M1 + new M2 + random mask": random_mask,
        "M1 + full fine-tuning": full_ft,
    }


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return "not_available"


def _assert_consistency(m2_rows: List[Dict[str, Any]], m3_rows: List[Dict[str, Any]]) -> List[str]:
    messages: List[str] = []
    for seed in sorted({int(r["seed"]) for r in m2_rows}):
        rows = [r for r in m2_rows if int(r["seed"]) == seed]
        s_vals = np.asarray([float(r["s_reliability"]) for r in rows], dtype=float)
        gamma_vals = np.asarray([float(r["gamma_s"]) for r in rows], dtype=float)
        if s_vals.size and gamma_vals.size and np.all(np.isfinite(gamma_vals)):
            med = float(np.median(s_vals))
            max_err = float(np.max(np.abs(gamma_vals - med)))
            assert max_err < 1e-6, f"gamma_s median check failed for seed={seed}: err={max_err}"
            messages.append(f"seed={seed}: gamma_s median check passed (gamma_s={med:.6f}).")
        else:
            messages.append(f"seed={seed}: product M2 check skipped gamma_s median because this mode uses max-normalized need x reliability.")

    for row in m3_rows:
        selected = str(row.get("selected_groups", "")).split(";") if row.get("selected_groups") else []
        assert "g1_head" in selected, f"Head group not selected: seed={row.get('seed')} client={row.get('client_name')}"
        assert int(row.get("active_group_count", 0)) <= int(row.get("m_k", 0)), (
            f"selected groups exceed m_k: seed={row.get('seed')} client={row.get('client_name')}"
        )
    messages.append("M3 checks passed: classifier head is always included and selected_count <= m_k.")
    messages.append("No-test-leakage protocol: threshold/checkpoint/APC candidate decisions are made inside existing validation-only code paths.")
    return messages


def _plot_m2(m2_rows: List[Dict[str, Any]]) -> List[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    q = np.asarray([float(r["q_reliability"]) for r in m2_rows], dtype=float)
    s = np.asarray([float(r["s_reliability"]) for r in m2_rows], dtype=float)

    path = FIGURES_DIR / "m2_q_to_s_compression.png"
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.scatter(q, s, alpha=0.75)
    xs = np.linspace(0, max(1.0, float(np.nanmax(q))), 200)
    ax.plot(xs, np.log1p(xs), color="black", linewidth=1.5)
    ax.set_title("M2: log(1+q) compresses reliability signal")
    ax.set_xlabel("q = delta / sigma_delta")
    ax.set_ylabel("s = log(1 + q)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    by_client = _mean_by_client(m2_rows, ["r_old_gamma_1", "r_old_gamma_3", "r_reliability", "m_k", "e_k"])
    clients = [r["client_name"] for r in by_client]
    x = np.arange(len(clients))

    path = FIGURES_DIR / "m2_old_vs_new_intensity.png"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    ax.plot(x, [r["r_old_gamma_1"] for r in by_client], marker="o", label="old gamma=1")
    ax.plot(x, [r["r_old_gamma_3"] for r in by_client], marker="o", label="old gamma=3")
    ax.plot(x, [r["r_reliability"] for r in by_client], marker="o", label="new log-median")
    ax.set_title("M2: old direct mapping vs new log-median intensity")
    ax.set_ylabel("r")
    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    path = FIGURES_DIR / "m2_new_budget_by_client.png"
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=180, sharex=True)
    axes[0].bar(x, [r["r_reliability"] for r in by_client], color="#4c78a8")
    axes[0].set_ylabel("r")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, [r["m_k"] for r in by_client], color="#59a14f")
    axes[1].set_ylabel("m_k")
    axes[2].bar(x, [r["e_k"] for r in by_client], color="#e15759")
    axes[2].set_ylabel("e_k")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(clients, rotation=35, ha="right")
    axes[0].set_title("M2: new adaptation intensity and budgets by client")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_m3(selected_rows: List[Dict[str, Any]], score_rows: List[Dict[str, Any]]) -> List[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    client_names = sorted({str(r["client_name"]) for r in selected_rows})
    group_index = {g: i for i, g in enumerate(GROUP_ORDER)}
    client_index = {c: i for i, c in enumerate(client_names)}

    selected_mat = np.zeros((len(client_names), len(GROUP_ORDER)), dtype=float)
    selected_counts = np.zeros_like(selected_mat)
    score_mat = np.zeros_like(selected_mat)
    score_counts = np.zeros_like(selected_mat)
    drift_vals: Dict[str, List[float]] = {c: [] for c in client_names}

    for row in selected_rows:
        c = str(row["client_name"])
        drift_vals.setdefault(c, []).append(_safe_float(row.get("total_drift_norm")))
        for g in GROUP_ORDER:
            selected_mat[client_index[c], group_index[g]] += int(row.get(f"selected_{g}", 0))
            selected_counts[client_index[c], group_index[g]] += 1.0
    selected_freq = np.divide(selected_mat, np.maximum(selected_counts, 1.0))

    for row in score_rows:
        c = str(row["client_name"])
        g = str(row["group"])
        if c in client_index and g in group_index:
            score_mat[client_index[c], group_index[g]] += _safe_float(row.get("score"), 0.0)
            score_counts[client_index[c], group_index[g]] += 1.0
    score_avg = np.divide(score_mat, np.maximum(score_counts, 1.0))

    path = FIGURES_DIR / "m3_selected_group_heatmap.png"
    _heatmap(selected_freq, client_names, GROUP_ORDER, "M3 selected group frequency", path, vmin=0.0, vmax=1.0)
    paths.append(path)

    path = FIGURES_DIR / "m3_group_sensitivity_heatmap.png"
    _heatmap(score_avg, client_names, GROUP_ORDER, "M3 average normalized gradient score", path)
    paths.append(path)

    path = FIGURES_DIR / "m3_drift_by_client.png"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    means = [float(np.nanmean([v for v in drift_vals[c] if np.isfinite(v)])) if drift_vals[c] else float("nan") for c in client_names]
    ax.bar(np.arange(len(client_names)), means, color="#8cd17d")
    ax.set_title("M3: Stage-II drift norm by client")
    ax.set_ylabel("drift norm")
    ax.set_xticks(np.arange(len(client_names)))
    ax.set_xticklabels(client_names, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


def _heatmap(mat: np.ndarray, rows: List[str], cols: List[str], title: str, path: Path, *, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(rows))), dpi=180)
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _mean_by_client(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    clients = sorted({str(r["client_name"]) for r in rows})
    out: List[Dict[str, Any]] = []
    for client in clients:
        group = [r for r in rows if str(r["client_name"]) == client]
        row: Dict[str, Any] = {"client_name": client}
        for key in keys:
            vals = [_safe_float(r.get(key)) for r in group]
            vals = [v for v in vals if np.isfinite(v)]
            row[key] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    return out


def _print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n[Summary]")
    for row in summary_rows:
        parts = [f"- {row['method']}"]
        for metric, label in [
            ("global_auc", "gAUC"),
            ("macro_auc", "mAUC"),
            ("macro_f1", "mF1"),
            ("macro_ece", "mECE"),
            ("gini", "Gini"),
        ]:
            parts.append(f"{label} {row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")
        print(" | ".join(parts))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final DA-PFL updated-M2 experiments without external baselines.")
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in SEEDS))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--with-optional-controls", action="store_true", help="Also run random-mask and full-finetune supplementary controls.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = _parse_csv(args.seeds, int) or list(SEEDS)
    global RESULTS_DIR, FIGURES_DIR
    RESULTS_DIR = Path(args.output_dir)
    FIGURES_DIR = Path(args.figures_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_run_rows: List[Dict[str, Any]] = []
    all_client_rows: List[Dict[str, Any]] = []
    full_new_client_rows: List[Dict[str, Any]] = []
    m2_rows: List[Dict[str, Any]] = []
    m3_selected_rows: List[Dict[str, Any]] = []
    m3_score_rows: List[Dict[str, Any]] = []
    optional_client_rows: Dict[str, List[Dict[str, Any]]] = {
        "M1 + new M2 + random mask": [],
        "M1 + full fine-tuning": [],
    }

    t0 = time.perf_counter()
    for repeat_idx, seed in enumerate(seeds, start=1):
        seed = int(seed)
        print(f"\n[Seed {seed}] ({repeat_idx}/{len(seeds)})")
        base_cfg = Ch4Config(seed=seed, fed_rounds=40, local_epochs_per_round=2)
        bundle = _train_stage1_fedavg_backbone(seed, base_cfg, record_history=False)
        client_ids = {str(c): idx for idx, c in enumerate(bundle["client_names"])}
        cfgs = _make_cfgs(base_cfg)

        print("  -> Backbone Only")
        backbone_res = _eval_global_only(
            copy.deepcopy(bundle["backbone_model"]).to(DEVICE),
            bundle["client_loaders"],
            bundle["client_names"],
            base_cfg,
        )
        results_this_seed: Dict[str, Dict[str, Any]] = {"Backbone Only": backbone_res}

        methods = [
            "Backbone + M1",
            "Backbone + M1 + Head FT",
            "Backbone + M1 + M3 without M2/APC",
            "Full DA-PFL old M2",
            "Full DA-PFL new M2",
        ]
        if bool(args.with_optional_controls):
            methods.extend(["M1 + new M2 + random mask", "M1 + full fine-tuning"])

        for method in methods:
            print(f"  -> {method}")
            results_this_seed[method] = _run_stage2(seed, method, cfgs[method], bundle)

        for method, result in results_this_seed.items():
            row = _metric_row(method, seed, result)
            all_run_rows.append(row)
            all_client_rows.extend(_client_rows(method, seed, result, client_ids))
            print(
                f"    - {method:<38s} | "
                f"gAUC={row['global_auc']:.4f} | mAUC={row['macro_auc']:.4f} | "
                f"mF1={row['macro_f1']:.4f} | mECE={row['macro_ece']:.4f} | Gini={row['gini']:.4f}"
            )

        full_new = results_this_seed["Full DA-PFL new M2"]
        full_new_client_rows.extend(_client_rows("Full DA-PFL new M2", seed, full_new, client_ids))
        m2_rows.extend(_m2_rows(seed, full_new, client_ids, cfgs["Full DA-PFL new M2"]))
        m3_pack = _m3_rows(
            seed,
            full_new,
            client_ids,
            before_result=results_this_seed.get("Backbone + M1"),
        )
        m3_selected_rows.extend(m3_pack["selected"])
        m3_score_rows.extend(m3_pack["scores"])
        for opt_name in optional_client_rows.keys():
            if opt_name in results_this_seed:
                optional_client_rows[opt_name].extend(_client_rows(opt_name, seed, results_this_seed[opt_name], client_ids))

    summary_rows = _summarize(all_run_rows)
    full_new_summary = [r for r in summary_rows if r["method"] == "Full DA-PFL new M2"]
    ablation_summary = [r for r in summary_rows if r["method"] in METHOD_ORDER[:6]]
    random_summary = [r for r in summary_rows if r["method"] == "M1 + new M2 + random mask"]
    full_ft_summary = [r for r in summary_rows if r["method"] == "M1 + full fine-tuning"]

    _write_csv(RESULTS_DIR / "full_dapfl_new_m2_summary.csv", full_new_summary)
    _write_csv(RESULTS_DIR / "full_dapfl_new_m2_per_client.csv", full_new_client_rows)
    _write_csv(RESULTS_DIR / "dapfl_component_ablation_summary.csv", ablation_summary)
    _write_csv(RESULTS_DIR / "dapfl_component_ablation_per_client.csv", [
        r for r in all_client_rows if r["method"] in METHOD_ORDER[:6]
    ])
    _write_csv(RESULTS_DIR / "m2_mechanism_table.csv", m2_rows)
    _write_csv(RESULTS_DIR / "m3_selected_groups_table.csv", m3_selected_rows)
    _write_csv(RESULTS_DIR / "m3_group_scores_table.csv", m3_score_rows)
    _write_csv(RESULTS_DIR / "random_mask_control_summary.csv", random_summary)
    _write_csv(RESULTS_DIR / "random_mask_control_per_client.csv", optional_client_rows["M1 + new M2 + random mask"])
    _write_csv(RESULTS_DIR / "full_finetune_after_m1_summary.csv", full_ft_summary)
    _write_csv(RESULTS_DIR / "full_finetune_after_m1_per_client.csv", optional_client_rows["M1 + full fine-tuning"])

    consistency_messages = _assert_consistency(m2_rows, m3_selected_rows)
    config_payload = {
        "seeds": seeds,
        "device": str(DEVICE),
        "git_commit": _git_commit(),
        "fixed_budget_without_m2": {"m_fixed": int(FIXED_K_PERS), "e_fixed": int(FIXED_E_PERS)},
        "old_m2_direct_gamma": 1.96,
        "new_m2_formula": "q=delta/(sigma+eps); s=log(1+q); u=s/max_j(s_j); alpha=log(n+1)/max_j log(n_j+1); r=u*alpha; m=1+floor((G-1)r); e=e_min+floor((e_max-e_min)r), with validation candidate selection.",
        "consistency_checks": consistency_messages,
        "wall_time_min": float((time.perf_counter() - t0) / 60.0),
    }
    (RESULTS_DIR / "final_dapfl_updated_m2_config_and_checks.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure_paths = []
    figure_paths.extend(_plot_m2(m2_rows))
    figure_paths.extend(_plot_m3(m3_selected_rows, m3_score_rows))

    _print_summary(summary_rows)
    print("\n[Outputs]")
    for path in [
        RESULTS_DIR / "full_dapfl_new_m2_summary.csv",
        RESULTS_DIR / "full_dapfl_new_m2_per_client.csv",
        RESULTS_DIR / "dapfl_component_ablation_summary.csv",
        RESULTS_DIR / "dapfl_component_ablation_per_client.csv",
        RESULTS_DIR / "m2_mechanism_table.csv",
        RESULTS_DIR / "m3_selected_groups_table.csv",
        RESULTS_DIR / "m3_group_scores_table.csv",
        RESULTS_DIR / "random_mask_control_summary.csv",
        RESULTS_DIR / "random_mask_control_per_client.csv",
        RESULTS_DIR / "full_finetune_after_m1_summary.csv",
        RESULTS_DIR / "full_finetune_after_m1_per_client.csv",
        RESULTS_DIR / "final_dapfl_updated_m2_config_and_checks.json",
    ]:
        print(f"  - {path}")
    for path in figure_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
