import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


JSON_PATH = r"D:\最终版论文实验\out\ch4_compare\ch4_compare_results_main_ablation.json"

BASELINE_KEY = "0_StageI_Backbone_Only"
OURS_KEY = "1_FULL_NoTS"


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def _pick(d: Dict[str, Any], keys: List[str], default=float("nan")):
    for k in keys:
        if k in d:
            return d[k]
    return default


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 1) 论文口径：方法级指标增益
# =========================================================

def _summary_metric_mean(block: Dict[str, Any], keys: List[str]) -> float:
    """
    从 summary[method_key] 里读 mean
    支持:
      metric: {"mean": ..., "std": ...}
      或 metric: 0.123
    """
    for k in keys:
        if k in block:
            v = block[k]
            if isinstance(v, dict):
                return _safe_float(v.get("mean"))
            return _safe_float(v)
    return float("nan")


def build_paper_gain_table(payload: Dict[str, Any],
                           baseline_key: str,
                           ours_key: str) -> pd.DataFrame:
    summary = payload.get("summary", {})
    base = summary.get(baseline_key, {})
    ours = summary.get(ours_key, {})

    if not isinstance(base, dict) or not base:
        raise ValueError(f"Cannot find baseline summary: {baseline_key}")
    if not isinstance(ours, dict) or not ours:
        raise ValueError(f"Cannot find ours summary: {ours_key}")

    rows = []

    # higher is better
    metric_specs_high = [
        ("Global AUC",    ["global_auc", "global_auroc"]),
        ("Global AUPRC",  ["global_auprc"]),
        ("Macro AUC",     ["macro_auc", "macro_auroc"]),
        ("Macro AUPRC",   ["macro_auprc"]),
        ("Macro F1",      ["macro_f1", "f1"]),
    ]

    # lower is better
    metric_specs_low = [
        ("Macro ECE↓",    ["macro_ece", "ece"]),
        ("Gini↓",         ["gini"]),
    ]

    for metric_name, keys in metric_specs_high:
        b = _summary_metric_mean(base, keys)
        o = _summary_metric_mean(ours, keys)
        rows.append({
            "Metric": metric_name,
            "Baseline": b,
            "Ours": o,
            "AbsoluteGain": o - b,   # higher is better
        })

    for metric_name, keys in metric_specs_low:
        b = _summary_metric_mean(base, keys)
        o = _summary_metric_mean(ours, keys)
        rows.append({
            "Metric": metric_name,
            "Baseline": b,
            "Ours": o,
            "AbsoluteGain": b - o,   # lower is better
        })

    return pd.DataFrame(rows)


# =========================================================
# 2) 每个客户端：对 baseline 的绝对提升
# =========================================================

def _pick_metric(d: Dict[str, Any], keys: List[str]) -> float:
    for k in keys:
        if k in d:
            return _safe_float(d[k])
    return float("nan")


def extract_client_metrics(payload: Dict[str, Any], method_key: str) -> pd.DataFrame:
    rows = []
    runs = payload.get("raw_runs_full", {}).get(method_key, [])

    for rid, run in enumerate(runs):
        client_metrics = run.get("client_metrics", {}) or {}
        for client_name, m in client_metrics.items():
            if not isinstance(m, dict):
                continue

            rows.append({
                "method_key": method_key,
                "repeat": rid,
                "Client": str(client_name),

                # 兼容常见命名
                "AUC": _pick_metric(m, ["auc", "auroc", "roc_auc"]),
                "AUPRC": _pick_metric(m, ["auprc", "pr_auc"]),
                "F1": _pick_metric(m, ["f1", "macro_f1"]),
                "ECE": _pick_metric(m, ["ece"]),
                "Gini": _pick_metric(m, ["gini"]),
            })

    return pd.DataFrame(rows)


def build_client_gain_table(payload: Dict[str, Any],
                            baseline_key: str,
                            ours_key: str) -> pd.DataFrame:
    df_base = extract_client_metrics(payload, baseline_key)
    df_ours = extract_client_metrics(payload, ours_key)

    if df_base.empty:
        raise ValueError(f"No client_metrics found for baseline: {baseline_key}")
    if df_ours.empty:
        raise ValueError(f"No client_metrics found for ours: {ours_key}")

    agg_base = (
        df_base.groupby("Client", as_index=False)
        .agg({
            "AUC": "mean",
            "AUPRC": "mean",
            "F1": "mean",
            "ECE": "mean",
            "Gini": "mean",
        })
        .rename(columns={
            "AUC": "AUC_base",
            "AUPRC": "AUPRC_base",
            "F1": "F1_base",
            "ECE": "ECE_base",
            "Gini": "Gini_base",
        })
    )

    agg_ours = (
        df_ours.groupby("Client", as_index=False)
        .agg({
            "AUC": "mean",
            "AUPRC": "mean",
            "F1": "mean",
            "ECE": "mean",
            "Gini": "mean",
        })
        .rename(columns={
            "AUC": "AUC_ours",
            "AUPRC": "AUPRC_ours",
            "F1": "F1_ours",
            "ECE": "ECE_ours",
            "Gini": "Gini_ours",
        })
    )

    out = agg_base.merge(agg_ours, on="Client", how="inner")

    # higher is better
    out["Gain_AUC"] = out["AUC_ours"] - out["AUC_base"]
    out["Gain_AUPRC"] = out["AUPRC_ours"] - out["AUPRC_base"]
    out["Gain_F1"] = out["F1_ours"] - out["F1_base"]

    # lower is better
    out["Gain_ECE"] = out["ECE_base"] - out["ECE_ours"]
    out["Gain_Gini"] = out["Gini_base"] - out["Gini_ours"]

    # 一个综合判断列：AUPRC 优先排序
    out = out.sort_values("Gain_AUPRC", ascending=False).reset_index(drop=True)
    return out


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    payload = load_json(JSON_PATH)

    # 1) 论文表口径的总体增益
    paper_gain_df = build_paper_gain_table(payload, BASELINE_KEY, OURS_KEY)
    print("\n=== Paper-level absolute gain vs baseline ===")
    print(paper_gain_df.to_string(index=False))
    paper_gain_df.to_csv("paper_level_gain_vs_baseline.csv", index=False, encoding="utf-8-sig")

    # 2) 每个客户端的绝对提升
    client_gain_df = build_client_gain_table(payload, BASELINE_KEY, OURS_KEY)
    print("\n=== Per-client absolute gain vs baseline ===")
    print(client_gain_df[[
        "Client",
        "AUC_base", "AUC_ours", "Gain_AUC",
        "AUPRC_base", "AUPRC_ours", "Gain_AUPRC",
        "F1_base", "F1_ours", "Gain_F1",
        "ECE_base", "ECE_ours", "Gain_ECE",
        "Gini_base", "Gini_ours", "Gain_Gini",
    ]].to_string(index=False))

    client_gain_df.to_csv("client_level_gain_vs_baseline.csv", index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print("  paper_level_gain_vs_baseline.csv")
    print("  client_level_gain_vs_baseline.csv")