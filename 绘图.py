# plot_paper_figures_revised_safe.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import MatplotlibDeprecationWarning

# =========================================================
# Warning / logger control
# =========================================================
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", message=".*Auto-removal of grids by pcolor.*")
warnings.filterwarnings("ignore", message=".*distutils Version classes are deprecated.*")

logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# =========================================================
# Global plot settings
# =========================================================
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

sns.set_style("whitegrid")

# =========================================================
# Paths
# =========================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR   # 关键：脚本就在项目根目录时，不要再 .parent

CANDIDATE_RESULTS_DIRS = [
    PROJECT_ROOT / "out" / "ch4_compare",
    PROJECT_ROOT / "ch4_compare",
]

RESULTS_DIR = None
for cand in CANDIDATE_RESULTS_DIRS:
    if cand.exists():
        RESULTS_DIR = cand
        break

if RESULTS_DIR is None:
    raise FileNotFoundError(
        "Cannot locate results directory. Tried:\n"
        + "\n".join([f"  - {str(p)}" for p in CANDIDATE_RESULTS_DIRS])
    )

BASELINE_JSON = RESULTS_DIR / "run_baselines_results.json"
COMPARE_JSON = RESULTS_DIR / "ch4_compare_results.json"

OUT_DIR = PROJECT_ROOT / "paper_figures_revised"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Colors
# =========================================================
COLOR_AUC = "#4C72B0"
COLOR_AUPRC = "#55A868"
COLOR_ECE = "#C44E52"
COLOR_NEUTRAL = "#6C757D"
COLOR_FULL = "#2A9D8F"
COLOR_SMALL = "#D62728"
COLOR_APC1 = "#8172B3"
COLOR_APC2 = "#CCB974"

# =========================================================
# Method name mapping
# =========================================================
BASELINE_METHOD_MAP = OrderedDict([
    ("Baseline_Local_Only", "Local Only"),
    ("Baseline_Global_FedAvg", "Global FedAvg"),
    ("Baseline_FedAvg_Local_FT", "FedAvg+FT"),
    ("Baseline_Ditto_pFL", "Ditto (pFL)"),
])

FULL_METHOD_KEY = "1_ShiftParticipation_DA-PFL_FULL"

ABLATION_METHOD_MAP = OrderedDict([
    ("1_ShiftParticipation_DA-PFL_FULL", "DA-PFL (FULL)"),
    ("2_ShiftParticipation_no_temp", "w/o Temp Scaling"),
    ("3_ShiftParticipation_no_M1_bias", "w/o M1 Bias"),
    ("4_ShiftParticipation_head_only", "Head Only"),
    ("5_ShiftParticipation_small_penalty_off", "w/o Small Penalty"),
    ("6_ShiftParticipation_size_gate_off", "w/o Size Gate"),
])

BACKBONE_ONLY_KEY = "0_ShiftParticipation_Backbone_Only"

# =========================================================
# Basic utils
# =========================================================
def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found:\n"
            f"  expected: {path}\n"
            f"  cwd: {Path.cwd()}\n"
            f"  script_dir: {Path(__file__).resolve().parent}"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _get_metric(summary: Dict[str, Any], method_key: str, metric: str) -> Dict[str, Any]:
    mm = summary.get(method_key, {}).get(metric, {})
    mean = mm.get("mean", None)
    std = mm.get("std", None)
    n = mm.get("n", None)
    return {
        "mean": _safe_float(mean),
        "std": _safe_float(std, default=0.0) if mean is not None else np.nan,
        "n": n,
    }


def _pick_metric(summary: Dict[str, Any], method_key: str, preferred: List[str]) -> Dict[str, Any]:
    for name in preferred:
        m = _get_metric(summary, method_key, name)
        if np.isfinite(m["mean"]):
            out = dict(m)
            out["metric_used"] = name
            return out
    return {"mean": np.nan, "std": np.nan, "n": None, "metric_used": None}


def _pick(d: Dict[str, Any], candidates: List[str], default=None):
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _finite_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s[np.isfinite(s)]


def _finite_minmax(arr: pd.Series, pad: float = 0.0, lower_clip: Optional[float] = None) -> Optional[Tuple[float, float]]:
    vals = pd.to_numeric(arr, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    lo = float(np.min(vals)) - pad
    hi = float(np.max(vals)) + pad
    if lower_clip is not None:
        lo = max(lower_clip, lo)
    if lo == hi:
        hi = lo + max(0.01, pad if pad > 0 else 0.01)
    return lo, hi


def _size_from_k(k: float) -> float:
    if not np.isfinite(k):
        k = 1.0
    return 140 + 180 * float(k)


def _export_fig(fig: plt.Figure, stem: str) -> None:
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}")


def _label_text(ax, x: float, y: float, text: str, **kwargs):
    if np.isfinite(x) and np.isfinite(y):
        ax.text(x, y, text, **kwargs)


def _annotate_text(ax, x: float, y: float, text: str, **kwargs):
    if np.isfinite(x) and np.isfinite(y):
        ax.annotate(text, (x, y), **kwargs)


# =========================================================
# Data preparation
# =========================================================
def build_main_df(baseline_payload: Dict[str, Any], compare_payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    baseline_summary = baseline_payload.get("summary", {})
    compare_summary = compare_payload.get("summary", {})

    for key, label in BASELINE_METHOD_MAP.items():
        disc = _pick_metric(baseline_summary, key, ["global_auc", "macro_auc"])
        auprc = _pick_metric(baseline_summary, key, ["global_auprc", "macro_auprc"])
        ece = _pick_metric(baseline_summary, key, ["global_ece", "macro_ece"])
        brier = _pick_metric(baseline_summary, key, ["global_brier", "macro_brier"])

        rows.append({
            "method_key": key,
            "label": label,
            "disc": disc["mean"],
            "disc_std": disc["std"],
            "disc_metric_used": disc["metric_used"],
            "auprc": auprc["mean"],
            "auprc_std": auprc["std"],
            "auprc_metric_used": auprc["metric_used"],
            "ece": ece["mean"],
            "ece_std": ece["std"],
            "ece_metric_used": ece["metric_used"],
            "brier": brier["mean"],
            "brier_std": brier["std"],
            "brier_metric_used": brier["metric_used"],
            "source": "run_baselines",
        })

    disc = _pick_metric(compare_summary, FULL_METHOD_KEY, ["global_auc", "macro_auc"])
    auprc = _pick_metric(compare_summary, FULL_METHOD_KEY, ["global_auprc", "macro_auprc"])
    ece = _pick_metric(compare_summary, FULL_METHOD_KEY, ["global_ece", "macro_ece"])
    brier = _pick_metric(compare_summary, FULL_METHOD_KEY, ["global_brier", "macro_brier"])

    rows.append({
        "method_key": FULL_METHOD_KEY,
        "label": "DA-PFL (Ours)",
        "disc": disc["mean"],
        "disc_std": disc["std"],
        "disc_metric_used": disc["metric_used"],
        "auprc": auprc["mean"],
        "auprc_std": auprc["std"],
        "auprc_metric_used": auprc["metric_used"],
        "ece": ece["mean"],
        "ece_std": ece["std"],
        "ece_metric_used": ece["metric_used"],
        "brier": brier["mean"],
        "brier_std": brier["std"],
        "brier_metric_used": brier["metric_used"],
        "source": "compare_main",
    })

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_macro_supp_df(baseline_payload: Dict[str, Any], compare_payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    baseline_summary = baseline_payload.get("summary", {})
    compare_summary = compare_payload.get("summary", {})

    for key, label in BASELINE_METHOD_MAP.items():
        rows.append({
            "method_key": key,
            "label": label,
            "macro_auc": _get_metric(baseline_summary, key, "macro_auc")["mean"],
            "macro_auc_std": _get_metric(baseline_summary, key, "macro_auc")["std"],
            "macro_auprc": _get_metric(baseline_summary, key, "macro_auprc")["mean"],
            "macro_auprc_std": _get_metric(baseline_summary, key, "macro_auprc")["std"],
            "macro_ece": _get_metric(baseline_summary, key, "macro_ece")["mean"],
            "macro_ece_std": _get_metric(baseline_summary, key, "macro_ece")["std"],
        })

    rows.append({
        "method_key": FULL_METHOD_KEY,
        "label": "DA-PFL (Ours)",
        "macro_auc": _get_metric(compare_summary, FULL_METHOD_KEY, "macro_auc")["mean"],
        "macro_auc_std": _get_metric(compare_summary, FULL_METHOD_KEY, "macro_auc")["std"],
        "macro_auprc": _get_metric(compare_summary, FULL_METHOD_KEY, "macro_auprc")["mean"],
        "macro_auprc_std": _get_metric(compare_summary, FULL_METHOD_KEY, "macro_auprc")["std"],
        "macro_ece": _get_metric(compare_summary, FULL_METHOD_KEY, "macro_ece")["mean"],
        "macro_ece_std": _get_metric(compare_summary, FULL_METHOD_KEY, "macro_ece")["std"],
    })

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_ablation_df(compare_payload: Dict[str, Any]) -> pd.DataFrame:
    compare_summary = compare_payload.get("summary", {})
    rows = []

    for key, label in ABLATION_METHOD_MAP.items():
        disc = _pick_metric(compare_summary, key, ["global_auc", "macro_auc"])
        auprc = _pick_metric(compare_summary, key, ["global_auprc", "macro_auprc"])
        ece = _pick_metric(compare_summary, key, ["global_ece", "macro_ece"])
        brier = _pick_metric(compare_summary, key, ["global_brier", "macro_brier"])

        rows.append({
            "method_key": key,
            "label": label,
            "disc": disc["mean"],
            "disc_std": disc["std"],
            "disc_metric_used": disc["metric_used"],
            "auprc": auprc["mean"],
            "auprc_std": auprc["std"],
            "ece": ece["mean"],
            "ece_std": ece["std"],
            "brier": brier["mean"],
            "brier_std": brier["std"],
        })

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    full_row = df[df["method_key"] == FULL_METHOD_KEY].iloc[0]
    df["delta_disc_vs_full"] = df["disc"] - float(full_row["disc"])
    df["delta_auprc_vs_full"] = df["auprc"] - float(full_row["auprc"])
    df["delta_ece_vs_full"] = df["ece"] - float(full_row["ece"])
    df["delta_brier_vs_full"] = df["brier"] - float(full_row["brier"])

    return df


def build_apc_df(compare_payload: Dict[str, Any]) -> pd.DataFrame:
    raw_runs_full = compare_payload.get("raw_runs_full", {})
    runs = raw_runs_full.get(FULL_METHOD_KEY, [])
    rows = []

    for rid, run in enumerate(runs):
        client_debug = run.get("client_debug", {}) or {}
        for client_name, dbg in client_debug.items():
            if str(client_name).startswith("_"):
                continue
            if not isinstance(dbg, dict):
                continue

            rows.append({
                "repeat": rid,
                "Client": str(client_name),
                "AbsShift": _safe_float(_pick(
                    dbg,
                    ["abs_shift", "delta_k", "label_shift_proxy", "shift_proxy", "abs_label_shift"],
                    default=np.nan,
                )),
                "rho_cal": _safe_float(_pick(
                    dbg,
                    ["rho_cal", "p_cal", "calibration_prob", "select_prob_cal"],
                    default=np.nan,
                )),
                "rho_pers": _safe_float(_pick(
                    dbg,
                    ["rho_pers", "p_pers", "personalization_prob", "select_prob_pers"],
                    default=np.nan,
                )),
                "E_pers": _safe_float(_pick(
                    dbg,
                    ["E_pers", "e_pers", "personalization_epochs"],
                    default=np.nan,
                )),
                "K_pers": _safe_float(_pick(
                    dbg,
                    ["K_pers", "k_pers", "trainable_groups"],
                    default=np.nan,
                )),
                "TrainSize": _safe_float(_pick(
                    dbg,
                    ["n_train", "train_size", "client_size", "n_samples"],
                    default=np.nan,
                )),
                "SmallFlag": int(bool(_pick(
                    dbg,
                    ["small_penalty_triggered", "small_size_penalty", "is_small_client", "small_client", "small_rep_off"],
                    default=False,
                ))),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "Client", "AbsShift", "rho_cal", "rho_pers", "E_pers", "K_pers", "TrainSize", "SmallFlag"
        ])

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    agg_spec = {
        "AbsShift": "mean",
        "rho_cal": "mean",
        "rho_pers": "mean",
        "E_pers": "mean",
        "K_pers": "mean",
        "SmallFlag": "max",
    }
    if np.isfinite(df["TrainSize"]).any():
        agg_spec["TrainSize"] = "mean"

    out = df.groupby("Client", as_index=False).agg(agg_spec)
    for col in ["AbsShift", "rho_cal", "rho_pers", "E_pers", "K_pers", "TrainSize"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("AbsShift", na_position="last").reset_index(drop=True)


# =========================================================
# Reporting helpers
# =========================================================
def save_tables(main_df: pd.DataFrame, ablation_df: pd.DataFrame, apc_df: pd.DataFrame, macro_df: pd.DataFrame) -> None:
    main_df.to_csv(OUT_DIR / "table_main_summary.csv", index=False, encoding="utf-8-sig")
    ablation_df.to_csv(OUT_DIR / "table_ablation_summary.csv", index=False, encoding="utf-8-sig")
    macro_df.to_csv(OUT_DIR / "table_macro_summary.csv", index=False, encoding="utf-8-sig")
    if not apc_df.empty:
        apc_df.to_csv(OUT_DIR / "table_apc_outputs.csv", index=False, encoding="utf-8-sig")


def print_key_findings(main_df: pd.DataFrame, ablation_df: pd.DataFrame, apc_df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("[Key Findings]")
    print("=" * 88)

    ours = main_df[main_df["label"] == "DA-PFL (Ours)"].iloc[0]
    baselines_only = main_df[main_df["label"] != "DA-PFL (Ours)"].copy()

    valid_disc = baselines_only[np.isfinite(baselines_only["disc"])].copy()
    valid_ece = baselines_only[np.isfinite(baselines_only["ece"])].copy()

    best_disc_baseline = valid_disc.sort_values("disc", ascending=False).iloc[0] if not valid_disc.empty else None
    best_ece_baseline = valid_ece.sort_values("ece", ascending=True).iloc[0] if not valid_ece.empty else None

    print("\n[Main]")
    print(
        f"Ours: disc={ours['disc']:.4f} ± {ours['disc_std']:.4f} ({ours['disc_metric_used']}), "
        f"AUPRC={ours['auprc']:.4f} ± {ours['auprc_std']:.4f}, "
        f"ECE={ours['ece']:.4f} ± {ours['ece_std']:.4f}"
    )
    if best_disc_baseline is not None:
        print(
            f"Best baseline (disc): {best_disc_baseline['label']} = {best_disc_baseline['disc']:.4f} "
            f"({best_disc_baseline['disc_metric_used']})"
        )
    else:
        print("Best baseline (disc): no valid finite value")

    if best_ece_baseline is not None:
        print(
            f"Best baseline (ECE):  {best_ece_baseline['label']} = {best_ece_baseline['ece']:.4f} "
            f"({best_ece_baseline['ece_metric_used']})"
        )
    else:
        print("Best baseline (ECE): no valid finite value")

    print("\n[Ablation]")
    others = ablation_df[ablation_df["method_key"] != FULL_METHOD_KEY].copy()
    valid_disc_drop = others[np.isfinite(others["delta_disc_vs_full"])].copy()
    valid_ece_increase = others[np.isfinite(others["delta_ece_vs_full"])].copy()

    if not valid_disc_drop.empty:
        worst_drop_disc = valid_disc_drop.sort_values("delta_disc_vs_full").iloc[0]
        print(
            f"Largest discrimination drop vs FULL: {worst_drop_disc['label']} "
            f"({worst_drop_disc['delta_disc_vs_full']:+.4f})"
        )
    if not valid_ece_increase.empty:
        worst_ece_increase = valid_ece_increase.sort_values("delta_ece_vs_full", ascending=False).iloc[0]
        print(
            f"Largest ECE increase vs FULL: {worst_ece_increase['label']} "
            f"({worst_ece_increase['delta_ece_vs_full']:+.4f})"
        )

    if not apc_df.empty:
        print("\n[APC output pattern]")
        for left, right, name in [
            ("AbsShift", "rho_cal", "rho(shift, rho_cal)"),
            ("AbsShift", "rho_pers", "rho(shift, rho_pers)"),
            ("AbsShift", "E_pers", "rho(shift, E_pers)"),
            ("AbsShift", "K_pers", "rho(shift, K_pers)"),
        ]:
            sub = apc_df[[left, right]].dropna()
            if len(sub) >= 2:
                rho = sub[left].corr(sub[right], method="spearman")
                print(f"Spearman {name:<20} = {rho:.3f}")

    print("=" * 88)


# =========================================================
# Plot 1: Main comparison (disc + AUPRC + ECE)
# =========================================================
def plot_main_comparison(main_df: pd.DataFrame) -> None:
    plot_df = main_df[np.isfinite(main_df["disc"]) & np.isfinite(main_df["auprc"]) & np.isfinite(main_df["ece"])].copy()
    if plot_df.empty:
        print("Skipped: fig1_main_comparison (no finite rows)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9))
    x = np.arange(len(plot_df))
    labels = plot_df["label"].tolist()
    colors = [COLOR_FULL if lab == "DA-PFL (Ours)" else COLOR_AUC for lab in labels]

    metric_specs = [
        ("disc", "disc_std", "Discrimination", "Higher is better", COLOR_AUC),
        ("auprc", "auprc_std", "AUPRC", "Higher is better", COLOR_AUPRC),
        ("ece", "ece_std", "ECE", "Lower is better", COLOR_ECE),
    ]

    for ax, (metric, metric_std, title, subtitle, accent_color) in zip(axes, metric_specs):
        bars = ax.bar(
            x,
            plot_df[metric].values,
            yerr=plot_df[metric_std].fillna(0.0).values,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.90,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(f"{title}\n({subtitle})", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.45)

        lims = _finite_minmax(plot_df[metric], pad=0.02 if metric != "ece" else 0.01, lower_clip=0.0)
        if lims is not None:
            ax.set_ylim(*lims)

        for i, bar in enumerate(bars):
            yval = plot_df.iloc[i][metric]
            if np.isfinite(yval):
                _label_text(
                    ax,
                    bar.get_x() + bar.get_width() / 2,
                    yval + (0.002 if metric != "ece" else 0.001),
                    f"{yval:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=COLOR_FULL if labels[i] == "DA-PFL (Ours)" else accent_color,
                    fontweight="bold" if labels[i] == "DA-PFL (Ours)" else None,
                )

    disc_metric_used = plot_df.loc[plot_df["label"] == "DA-PFL (Ours)", "disc_metric_used"].iloc[0]
    axes[0].set_ylabel(disc_metric_used if disc_metric_used else "Metric")
    axes[1].set_ylabel(plot_df.loc[plot_df["label"] == "DA-PFL (Ours)", "auprc_metric_used"].iloc[0] or "Metric")
    axes[2].set_ylabel(plot_df.loc[plot_df["label"] == "DA-PFL (Ours)", "ece_metric_used"].iloc[0] or "Metric")

    fig.suptitle("Main Comparison Across Baselines and DA-PFL", y=1.02, fontweight="bold")
    plt.tight_layout()
    _export_fig(fig, "fig1_main_comparison")


# =========================================================
# Plot 2: Tradeoff scatter (AUPRC vs ECE)
# =========================================================
def plot_tradeoff_pr_ece(main_df: pd.DataFrame) -> None:
    plot_df = main_df[np.isfinite(main_df["auprc"]) & np.isfinite(main_df["ece"])].copy()
    if plot_df.empty:
        print("Skipped: fig2_tradeoff_pr_ece (no finite rows)")
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    colors = [COLOR_FULL if lab == "DA-PFL (Ours)" else COLOR_AUPRC for lab in plot_df["label"]]
    sizes = [180 if lab == "DA-PFL (Ours)" else 130 for lab in plot_df["label"]]

    ax.scatter(
        plot_df["ece"],
        plot_df["auprc"],
        s=sizes,
        c=colors,
        edgecolors="black",
        alpha=0.9,
    )

    for _, row in plot_df.iterrows():
        if np.isfinite(row["ece"]) and np.isfinite(row["auprc"]):
            ax.text(
                row["ece"] + 0.0008,
                row["auprc"] + 0.0008,
                row["label"],
                fontsize=9,
                color="black",
            )

    xlims = _finite_minmax(plot_df["ece"], pad=0.01, lower_clip=0.0)
    ylims = _finite_minmax(plot_df["auprc"], pad=0.01, lower_clip=0.0)
    if xlims is not None:
        ax.set_xlim(*xlims)
    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.set_xlabel("ECE (Lower is better)")
    ax.set_ylabel("AUPRC (Higher is better)")
    ax.set_title("Performance Trade-off: AUPRC vs ECE", fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    handles = [
        Patch(facecolor=COLOR_AUPRC, edgecolor="black", label="Baselines"),
        Patch(facecolor=COLOR_FULL, edgecolor="black", label="DA-PFL (Ours)"),
    ]
    ax.legend(handles=handles, frameon=True, loc="best")

    plt.tight_layout()
    _export_fig(fig, "fig2_tradeoff_pr_ece")


# =========================================================
# Plot 3: Ablation on global-style metrics
# =========================================================
def _plot_barh_with_error(
    ax,
    df: pd.DataFrame,
    metric: str,
    metric_std: str,
    delta_col: str,
    ref_value: float,
    color: str,
    title: str,
    xlabel: str,
):
    plot_df = df[np.isfinite(df[metric])].copy()
    if plot_df.empty:
        ax.set_title(title, fontweight="bold")
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        return

    y = np.arange(len(plot_df))
    colors = [COLOR_FULL if "FULL" in label else color for label in plot_df["label"]]

    bars = ax.barh(
        y,
        plot_df[metric].values,
        xerr=plot_df[metric_std].fillna(0.0).values,
        color=colors,
        edgecolor="black",
        alpha=0.9,
        capsize=4,
    )

    if np.isfinite(ref_value):
        ax.axvline(ref_value, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.2, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].tolist())
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    lims = _finite_minmax(plot_df[metric], pad=0.015, lower_clip=0.0)
    if lims is not None:
        ax.set_xlim(*lims)

    for i, bar in enumerate(bars):
        v = plot_df.iloc[i][metric]
        d = plot_df.iloc[i][delta_col]
        if np.isfinite(v):
            txt = f"{v:.4f}"
            if np.isfinite(d):
                txt += f" ({d:+.4f})"
            _label_text(
                ax,
                v + 0.0015,
                bar.get_y() + bar.get_height() / 2,
                txt,
                va="center",
                ha="left",
                fontsize=9,
                color="black",
            )


def plot_ablation_global(ablation_df: pd.DataFrame) -> None:
    if ablation_df.empty:
        print("Skipped: fig3_ablation_global (empty)")
        return

    full_row = ablation_df[ablation_df["method_key"] == FULL_METHOD_KEY].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    _plot_barh_with_error(
        ax=ax1,
        df=ablation_df,
        metric="disc",
        metric_std="disc_std",
        delta_col="delta_disc_vs_full",
        ref_value=float(full_row["disc"]) if np.isfinite(full_row["disc"]) else np.nan,
        color=COLOR_AUC,
        title="Ablation Impact on Discrimination",
        xlabel="Discrimination Metric",
    )

    _plot_barh_with_error(
        ax=ax2,
        df=ablation_df,
        metric="ece",
        metric_std="ece_std",
        delta_col="delta_ece_vs_full",
        ref_value=float(full_row["ece"]) if np.isfinite(full_row["ece"]) else np.nan,
        color=COLOR_ECE,
        title="Ablation Impact on Calibration",
        xlabel="ECE",
    )

    plt.tight_layout()
    _export_fig(fig, "fig3_ablation_global")


# =========================================================
# Plot 4: APC outputs
# =========================================================
def plot_apc_outputs(apc_df: pd.DataFrame) -> None:
    if apc_df.empty:
        print("Skipped: fig4_apc_outputs (no client_debug/APC data)")
        return

    plot_df = apc_df[np.isfinite(apc_df["AbsShift"])].copy()
    if plot_df.empty:
        print("Skipped: fig4_apc_outputs (no finite shift)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axes = axes.flatten()

    specs = [
        ("rho_cal", COLOR_APC1, r"$\rho_{cal}$"),
        ("rho_pers", COLOR_APC2, r"$\rho_{pers}$"),
        ("E_pers", COLOR_AUPRC, r"$E_k^{pers}$"),
        ("K_pers", COLOR_AUC, r"$K_k$"),
    ]

    has_train_size = "TrainSize" in plot_df.columns and np.isfinite(plot_df["TrainSize"]).any()

    for ax, (metric, color, ylab) in zip(axes, specs):
        sub = plot_df[np.isfinite(plot_df[metric])].copy()
        if sub.empty:
            ax.set_title(f"{ylab} vs Shift", fontweight="bold")
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
            continue

        sizes = [_size_from_k(k) for k in sub["K_pers"].fillna(1.0).values]
        if has_train_size:
            sc = ax.scatter(
                sub["AbsShift"],
                sub[metric],
                s=sizes,
                c=sub["TrainSize"],
                cmap="viridis",
                alpha=0.88,
                edgecolors="black",
                linewidths=0.8,
            )
        else:
            sc = ax.scatter(
                sub["AbsShift"],
                sub[metric],
                s=sizes,
                color=color,
                alpha=0.88,
                edgecolors="black",
                linewidths=0.8,
            )

        for idx, row in sub.iterrows():
            label = row["Client"]
            kw = {}
            if int(row.get("SmallFlag", 0)) == 1:
                kw["color"] = COLOR_SMALL
                kw["fontweight"] = "bold"
            else:
                kw["color"] = "black"
            _label_text(
                ax,
                row["AbsShift"],
                row[metric] + (0.02 if metric.startswith("rho") else 0.12),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                **kw,
            )

        ax.set_title(f"{ylab} vs Shift", fontweight="bold")
        ax.set_xlabel("Shift proxy")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", alpha=0.5)

        xlims = _finite_minmax(sub["AbsShift"], pad=0.02, lower_clip=0.0)
        ylims = _finite_minmax(sub[metric], pad=0.05 if metric.startswith("rho") else 0.25, lower_clip=0.0)
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)

        if len(sub) >= 2:
            rho = sub["AbsShift"].corr(sub[metric], method="spearman")
            ax.text(
                0.02,
                0.98,
                f"Spearman rho = {rho:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
            )

    if has_train_size:
        cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("Average Train Size", fontsize=11)

    fig.suptitle("APC Output Patterns by Client Shift", y=1.01, fontweight="bold")
    plt.tight_layout()
    _export_fig(fig, "fig4_apc_outputs")


# =========================================================
# Plot 5: Macro supplementary figure
# =========================================================
def plot_macro_supplement(macro_df: pd.DataFrame) -> None:
    plot_df = macro_df[np.isfinite(macro_df["macro_auc"]) & np.isfinite(macro_df["macro_ece"])].copy()
    if plot_df.empty:
        print("Skipped: fig5_macro_supplement (no finite rows)")
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    colors = [COLOR_FULL if lab == "DA-PFL (Ours)" else COLOR_AUC for lab in plot_df["label"]]
    sizes = [180 if lab == "DA-PFL (Ours)" else 130 for lab in plot_df["label"]]

    ax.scatter(
        plot_df["macro_ece"],
        plot_df["macro_auc"],
        s=sizes,
        c=colors,
        edgecolors="black",
        alpha=0.9,
    )

    for _, row in plot_df.iterrows():
        _label_text(
            ax,
            row["macro_ece"] + 0.0008,
            row["macro_auc"] + 0.0008,
            row["label"],
            fontsize=9,
            color="black",
        )

    xlims = _finite_minmax(plot_df["macro_ece"], pad=0.01, lower_clip=0.0)
    ylims = _finite_minmax(plot_df["macro_auc"], pad=0.01, lower_clip=0.0)
    if xlims is not None:
        ax.set_xlim(*xlims)
    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.set_xlabel("Macro ECE (Lower is better)")
    ax.set_ylabel("Macro AUC (Higher is better)")
    ax.set_title("Supplementary Macro-Level View", fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    handles = [
        Patch(facecolor=COLOR_AUC, edgecolor="black", label="Baselines"),
        Patch(facecolor=COLOR_FULL, edgecolor="black", label="DA-PFL (Ours)"),
    ]
    ax.legend(handles=handles, frameon=True, loc="best")

    plt.tight_layout()
    _export_fig(fig, "fig5_macro_supplement")
def build_mask_heatmap_df(compare_payload: Dict[str, Any]) -> pd.DataFrame:
    raw_runs_full = compare_payload.get("raw_runs_full", {})
    runs = raw_runs_full.get(FULL_METHOD_KEY, [])

    rows = []
    all_groups = set()

    for rid, run in enumerate(runs):
        client_debug = run.get("client_debug", {}) or {}
        for client_name, dbg in client_debug.items():
            if str(client_name).startswith("_"):
                continue
            if not isinstance(dbg, dict):
                continue

            selected_groups = dbg.get("selected_groups", None)
            mask_vector = dbg.get("m3_mask_vector", None)

            row = {
                "repeat": rid,
                "Client": str(client_name),
            }

            # 方式1：优先用 selected_groups
            if isinstance(selected_groups, list) and len(selected_groups) > 0:
                for g in selected_groups:
                    gname = str(g)
                    row[gname] = 1.0
                    all_groups.add(gname)

            # 方式2：如果没有 selected_groups，则尝试从 m3_mask_vector 解析
            elif isinstance(mask_vector, dict):
                for gname, val in mask_vector.items():
                    gname = str(gname)
                    v = 1.0 if _safe_float(val, default=0.0) > 0 else 0.0
                    row[gname] = v
                    all_groups.add(gname)

            elif isinstance(mask_vector, list):
                # 若 m3_mask_vector 是 list，需要 group 名顺序
                group_order = dbg.get("group_order", None)
                if isinstance(group_order, list) and len(group_order) == len(mask_vector):
                    for gname, val in zip(group_order, mask_vector):
                        gname = str(gname)
                        v = 1.0 if _safe_float(val, default=0.0) > 0 else 0.0
                        row[gname] = v
                        all_groups.add(gname)

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 补齐所有 group 列
    group_cols = sorted(all_groups)
    for g in group_cols:
        if g not in df.columns:
            df[g] = 0.0

    # 缺失补 0
    df[group_cols] = df[group_cols].fillna(0.0)

    # 按 client 聚合，得到平均选择频率
    heatmap_df = df.groupby("Client", as_index=True)[group_cols].mean()

    # 按被选总频率排序，便于展示
    heatmap_df["__sum__"] = heatmap_df.sum(axis=1)
    heatmap_df = heatmap_df.sort_values("__sum__", ascending=False).drop(columns="__sum__")

    return heatmap_df


def plot_mask_heatmap(mask_df: pd.DataFrame) -> None:
    if mask_df.empty:
        print("Skipped: fig6_mask_heatmap (no selected_groups / m3_mask_vector found)")
        return

    fig_w = max(8.0, 0.9 * mask_df.shape[1] + 3.0)
    fig_h = max(4.8, 0.45 * mask_df.shape[0] + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        mask_df,
        ax=ax,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Selection Frequency"},
    )

    ax.set_title("M3 Selected-Group Mask Heatmap", fontweight="bold", pad=12)
    ax.set_xlabel("Parameter Groups")
    ax.set_ylabel("Clients")

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    _export_fig(fig, "fig6_mask_heatmap")

# =========================================================
# Main
# =========================================================
def main():
    print(f"PROJECT_ROOT   = {PROJECT_ROOT}")
    print(f"RESULTS_DIR    = {RESULTS_DIR}")
    print(f"BASELINE_JSON  = {BASELINE_JSON}")
    print(f"COMPARE_JSON   = {COMPARE_JSON}")
    print(f"OUT_DIR        = {OUT_DIR}")
    print(f"CWD            = {Path.cwd()}")

    baseline_payload = _load_json(BASELINE_JSON)
    compare_payload = _load_json(COMPARE_JSON)

    main_df = build_main_df(baseline_payload, compare_payload)
    ablation_df = build_ablation_df(compare_payload)
    apc_df = build_apc_df(compare_payload)
    macro_df = build_macro_supp_df(baseline_payload, compare_payload)
    mask_df = build_mask_heatmap_df(compare_payload)

    save_tables(main_df, ablation_df, apc_df, macro_df)
    if not mask_df.empty:
        mask_df.to_csv(OUT_DIR / "table_mask_heatmap.csv", encoding="utf-8-sig")

    plot_main_comparison(main_df)
    plot_tradeoff_pr_ece(main_df)
    plot_ablation_global(ablation_df)
    plot_apc_outputs(apc_df)
    plot_macro_supplement(macro_df)
    plot_mask_heatmap(mask_df)

    print_key_findings(main_df, ablation_df, apc_df)
    print(f"\nAll figures and tables saved to: {OUT_DIR.resolve()}")
def build_mask_heatmap_df(compare_payload: Dict[str, Any]) -> pd.DataFrame:
    raw_runs_full = compare_payload.get("raw_runs_full", {})
    runs = raw_runs_full.get(FULL_METHOD_KEY, [])

    rows = []
    all_groups = set()

    for rid, run in enumerate(runs):
        client_debug = run.get("client_debug", {}) or {}
        for client_name, dbg in client_debug.items():
            if str(client_name).startswith("_"):
                continue
            if not isinstance(dbg, dict):
                continue

            selected_groups = dbg.get("selected_groups", None)
            mask_vector = dbg.get("m3_mask_vector", None)

            row = {
                "repeat": rid,
                "Client": str(client_name),
            }

            # 方式1：优先用 selected_groups
            if isinstance(selected_groups, list) and len(selected_groups) > 0:
                for g in selected_groups:
                    gname = str(g)
                    row[gname] = 1.0
                    all_groups.add(gname)

            # 方式2：如果没有 selected_groups，则尝试从 m3_mask_vector 解析
            elif isinstance(mask_vector, dict):
                for gname, val in mask_vector.items():
                    gname = str(gname)
                    v = 1.0 if _safe_float(val, default=0.0) > 0 else 0.0
                    row[gname] = v
                    all_groups.add(gname)

            elif isinstance(mask_vector, list):
                # 若 m3_mask_vector 是 list，需要 group 名顺序
                group_order = dbg.get("group_order", None)
                if isinstance(group_order, list) and len(group_order) == len(mask_vector):
                    for gname, val in zip(group_order, mask_vector):
                        gname = str(gname)
                        v = 1.0 if _safe_float(val, default=0.0) > 0 else 0.0
                        row[gname] = v
                        all_groups.add(gname)

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 补齐所有 group 列
    group_cols = sorted(all_groups)
    for g in group_cols:
        if g not in df.columns:
            df[g] = 0.0

    # 缺失补 0
    df[group_cols] = df[group_cols].fillna(0.0)

    # 按 client 聚合，得到平均选择频率
    heatmap_df = df.groupby("Client", as_index=True)[group_cols].mean()

    # 按被选总频率排序，便于展示
    heatmap_df["__sum__"] = heatmap_df.sum(axis=1)
    heatmap_df = heatmap_df.sort_values("__sum__", ascending=False).drop(columns="__sum__")

    return heatmap_df


def plot_mask_heatmap(mask_df: pd.DataFrame) -> None:
    if mask_df.empty:
        print("Skipped: fig6_mask_heatmap (no selected_groups / m3_mask_vector found)")
        return

    fig_w = max(8.0, 0.9 * mask_df.shape[1] + 3.0)
    fig_h = max(4.8, 0.45 * mask_df.shape[0] + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        mask_df,
        ax=ax,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Selection Frequency"},
    )

    ax.set_title("M3 Selected-Group Mask Heatmap", fontweight="bold", pad=12)
    ax.set_xlabel("Parameter Groups")
    ax.set_ylabel("Clients")

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    _export_fig(fig, "fig6_mask_heatmap")

if __name__ == "__main__":
    main()