import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# =========================================================
# Global constants
# =========================================================

FULL_METHOD_KEY = "1_FULL_NoTS"

COLOR_FULL    = "#d62728"
COLOR_AUPRC   = "#4c78a8"
COLOR_ECE     = "#f58518"
COLOR_NEUTRAL = "#7f7f7f"
COLOR_SMALL   = "#c0392b"
COLOR_UK      = "#72b7b2"
COLOR_RK      = "#54a24b"


# =========================================================
# CLI
# =========================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        type=str,
        default=r"D:\最终版论文实验\out\ch4_compare\ch4_compare_results_main_ablation.json"
    )
    p.add_argument(
        "--out",
        type=str,
        default=r"D:\最终版论文实验\out\figures"
    )
    return p.parse_args()


# =========================================================
# Low-level helpers
# =========================================================

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(v: Any) -> float:
    try:
        f = float(v)
        return float("nan") if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return float("nan")


def _pick(d: dict, keys: List[str], default: Any = float("nan")) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _export_fig(fig: plt.Figure, name: str, out_dir: Path) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _size_from_k(k: float) -> float:
    return max(60.0, float(k) * 120.0)


# =========================================================
# DataFrame builders
# =========================================================

def _method_metrics(payload: Dict, method_key: str) -> Optional[Dict[str, float]]:
    """Read mean±std from payload['summary'][method_key]."""
    summary = payload.get("summary", {})
    block = summary.get(method_key)
    if not block or not isinstance(block, dict):
        return None

    def _ms(key_candidates):
        for k in key_candidates:
            if k in block and isinstance(block[k], dict):
                return (
                    _safe_float(block[k].get("mean", float("nan"))),
                    _safe_float(block[k].get("std", 0.0)),
                )
        return float("nan"), 0.0

    auc_m, auc_s = _ms(["macro_auc", "macro_auroc"])
    auprc_m, auprc_s = _ms(["macro_auprc"])
    ece_m, ece_s = _ms(["macro_ece"])
    brier_m, brier_s = _ms(["macro_brier"])

    if math.isnan(auprc_m):
        auprc_m, auprc_s = auc_m, auc_s

    return {
        "macro_auroc": auc_m,
        "macro_auroc_std": auc_s,
        "macro_auprc": auprc_m,
        "macro_auprc_std": auprc_s,
        "macro_ece": ece_m,
        "macro_ece_std": ece_s,
        "macro_brier": brier_m,
        "macro_brier_std": brier_s,
    }


BASELINE_LABELS = {
    "0_StageI_Backbone_Only": "Global Only",
    "1_FULL_NoTS": "DA-PFL (Ours)",
}

ABLATION_LABELS = {
    "0_StageI_Backbone_Only": "0 Global Only",
    "1_FULL_NoTS": "1 FULL (Ours)",
    "2_wo_M1": "2 No M1",
    "3_wo_M3": "3 No M3",
    "4_FixedBudget": "4 Fixed Budget",
    "5_HeterogeneityOnly": "5 Heterogeneity Only",
    "6_SizeOnly": "6 Size Only",
}


def build_main_df(payload: Dict) -> pd.DataFrame:
    rows = []
    for key, label in BASELINE_LABELS.items():
        m = _method_metrics(payload, key)
        if m is None:
            continue
        rows.append({"method_key": key, "label": label, **m})
    return pd.DataFrame(rows)


def build_ablation_df(payload: Dict) -> pd.DataFrame:
    rows = []
    for key, label in ABLATION_LABELS.items():
        m = _method_metrics(payload, key)
        if m is None:
            continue
        rows.append({"method_key": key, "label": label, **m})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if FULL_METHOD_KEY in df["method_key"].values:
        full = df[df["method_key"] == FULL_METHOD_KEY].iloc[0]
        df["delta_auprc_vs_full"] = df["macro_auprc"] - float(full["macro_auprc"])
        df["delta_ece_vs_full"] = df["macro_ece"] - float(full["macro_ece"])
        df["delta_auc_vs_full"] = df["macro_auroc"] - float(full["macro_auroc"])
    return df


def _extract_apc_row(client_name: str, dbg: Dict[str, Any], repeat_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(dbg, dict):
        return None
    if str(client_name).startswith("_"):
        return None

    m1 = dbg.get("m1_bias", {})
    td = dbg.get("temp_diag", {})

    lambda_k = _safe_float(_pick(dbg, ["lambda_k"]))
    if math.isnan(lambda_k) and isinstance(m1, dict):
        lambda_k = _safe_float(m1.get("applied"))

    t_global = _safe_float(_pick(dbg, ["T_global"]))
    if math.isnan(t_global) and isinstance(td, dict):
        t_global = _safe_float(td.get("T_after", td.get("applied")))

    n_train = _safe_float(_pick(dbg, ["train_size", "n_i", "n_train", "client_size"]))

    row = {
        "Client":         str(client_name),
        "AbsShift":       _safe_float(_pick(dbg, ["abs_shift", "delta_k"])),
        "Zeta":           _safe_float(_pick(dbg, ["zeta_k", "zeta"])),
        "u_k":            _safe_float(_pick(dbg, ["rho_cal", "u_k"])),
        "a_k":            _safe_float(_pick(dbg, ["a_k"])),
        "r_k":            _safe_float(_pick(dbg, ["rho_pers", "r_k"])),
        "s_lab":          _safe_float(_pick(dbg, ["s_lab"])),
        "s_cov":          _safe_float(_pick(dbg, ["s_cov"])),
        "E_cal":          _safe_float(_pick(dbg, ["E_cal"])),
        "E_pers":         _safe_float(_pick(dbg, ["E_pers", "e_pers"])),
        "K_pers":         _safe_float(_pick(dbg, ["K_pers", "k_pers"])),
        "lambda_k":       lambda_k,
        "delta_bias":     _safe_float(_pick(dbg, ["delta_bias", "abs_shift"])),
        "T_global":       t_global,
        "realized_drift": _safe_float(_pick(dbg, ["realized_drift"])),
        "drift_bound":    _safe_float(_pick(dbg, ["drift_bound"])),
        "TrainSize":      n_train,
        "SmallFlag":      int(n_train < 400) if not math.isnan(n_train) else 0,
    }

    if repeat_id is not None:
        row["repeat"] = repeat_id

    return row


def build_apc_df(payload: Dict) -> pd.DataFrame:
    """
    Build per-client APC allocation table.

    Priority:
      1. payload["raw_runs_full"][FULL_METHOD_KEY][*]["client_debug"]
         - each repeat is extracted, then averaged per client
      2. payload["__acsp_client_debug__"] as fallback
    """
    raw_runs_full = payload.get("raw_runs_full", {})
    runs = raw_runs_full.get(FULL_METHOD_KEY, [])
    rows = []

    for rid, run in enumerate(runs):
        client_debug = run.get("client_debug") or {}
        for cname, dbg in client_debug.items():
            row = _extract_apc_row(cname, dbg, repeat_id=rid)
            if row is not None:
                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        numeric_cols = [c for c in df.columns if c not in ("Client", "repeat")]
        agg = {c: "mean" for c in numeric_cols}
        agg["SmallFlag"] = "max"
        out = df.groupby("Client", as_index=False).agg(agg)
        return out.sort_values("AbsShift").reset_index(drop=True)

    debug_block = payload.get("__acsp_client_debug__", {})
    if not isinstance(debug_block, dict) or not debug_block:
        return pd.DataFrame()

    rows = []
    for cname, dbg in debug_block.items():
        row = _extract_apc_row(cname, dbg)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("AbsShift").reset_index(drop=True)


# =========================================================
# Save tables
# =========================================================

def save_tables(main_df: pd.DataFrame, ablation_df: pd.DataFrame,
                apc_df: pd.DataFrame, out_dir: Path) -> None:
    main_df.to_csv(out_dir / "table_main_summary.csv", index=False, encoding="utf-8-sig")
    ablation_df.to_csv(out_dir / "table_ablation_summary.csv", index=False, encoding="utf-8-sig")
    if not apc_df.empty:
        apc_df.to_csv(out_dir / "table_apc_allocations.csv", index=False, encoding="utf-8-sig")
    print("  Tables saved.")


# =========================================================
# Print key findings
# =========================================================

def print_key_findings(main_df: pd.DataFrame, ablation_df: pd.DataFrame,
                       apc_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("[Key Findings]")
    print("=" * 80)

    if not main_df.empty and "DA-PFL (Ours)" in main_df["label"].values:
        ours = main_df[main_df["label"] == "DA-PFL (Ours)"].iloc[0]
        bases = main_df[main_df["label"] != "DA-PFL (Ours)"]
        best = bases.sort_values("macro_auprc", ascending=False).iloc[0]
        print(f"\n[Main] DA-PFL: AUPRC={ours['macro_auprc']:.4f} "
              f"ECE={ours['macro_ece']:.4f}")
        print(f"  Best baseline ({best['label']}): AUPRC={best['macro_auprc']:.4f}")
        print(f"  ΔAUPRC={ours['macro_auprc'] - best['macro_auprc']:+.4f} "
              f"ΔECE={ours['macro_ece'] - best['macro_ece']:+.4f}")

    if not ablation_df.empty and "delta_auprc_vs_full" in ablation_df.columns:
        others = ablation_df[ablation_df["method_key"] != FULL_METHOD_KEY]
        if not others.empty:
            worst = others.sort_values("delta_auprc_vs_full").iloc[0]
            print(f"\n[Ablation] Largest AUPRC drop: {worst['label']} "
                  f"({worst['delta_auprc_vs_full']:+.4f})")

    if not apc_df.empty and "AbsShift" in apc_df.columns:
        has_uk = "u_k" in apc_df.columns and apc_df["u_k"].notna().any()
        has_zeta = "Zeta" in apc_df.columns and apc_df["Zeta"].notna().any()
        rho_e = apc_df["AbsShift"].corr(apc_df["E_pers"], method="spearman")
        rho_k = apc_df["AbsShift"].corr(apc_df["K_pers"], method="spearman")
        print(f"\n[APC] Spearman ρ(δ, E_pers)={rho_e:.3f}  ρ(δ, K_pers)={rho_k:.3f}")
        if has_zeta:
            print(f"  ζ_k range: [{apc_df['Zeta'].min():.4f}, {apc_df['Zeta'].max():.4f}]")
        if has_uk:
            print(f"  u_k range: [{apc_df['u_k'].min():.4f}, {apc_df['u_k'].max():.4f}]")
    print("=" * 80)


# =========================================================
# Figure 1 – Main performance trade-off
# =========================================================

def plot_main_performance(main_df: pd.DataFrame, out_dir: Path) -> None:
    if main_df.empty:
        print("  Skipped Fig 1: no main data.")
        return

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(main_df))
    bar_colors = [COLOR_FULL if l == "DA-PFL (Ours)" else COLOR_AUPRC
                  for l in main_df["label"]]

    bars = ax1.bar(
        x,
        main_df["macro_auprc"].fillna(0),
        width=0.58,
        color=bar_colors,
        alpha=0.90,
        yerr=main_df["macro_auprc_std"].fillna(0),
        capsize=4,
        edgecolor="black",
        linewidth=0.8,
    )

    ax1.set_ylabel("Macro AUPRC", color=COLOR_AUPRC, fontweight="bold")
    lo = max(0.35, float(main_df["macro_auprc"].min()) - 0.02)
    hi = float(main_df["macro_auprc"].max()) + 0.02
    ax1.set_ylim(lo, hi)
    ax1.tick_params(axis="y", labelcolor=COLOR_AUPRC)
    ax1.set_xticks(x)
    ax1.set_xticklabels(main_df["label"], rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.errorbar(
        x,
        main_df["macro_ece"].fillna(0),
        yerr=main_df["macro_ece_std"].fillna(0),
        color=COLOR_ECE,
        marker="o",
        ms=9,
        lw=2.5,
        capsize=4,
        zorder=5,
    )
    ax2.set_ylabel("Macro ECE ↓", color=COLOR_ECE, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=COLOR_ECE)
    elo = max(0, float(main_df["macro_ece"].min()) - 0.015)
    ehi = float(main_df["macro_ece"].max()) + 0.02
    ax2.set_ylim(elo, ehi)
    for i, v in enumerate(main_df["macro_ece"]):
        ax2.annotate(f"{v:.4f}", (x[i], v),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, color=COLOR_ECE)

    handles = [
        mpatches.Patch(facecolor=COLOR_AUPRC, edgecolor="black", label="AUPRC ↑"),
        Line2D([0], [0], color=COLOR_ECE, marker="o", lw=2.5, ms=9, label="ECE ↓"),
        mpatches.Patch(facecolor=COLOR_FULL, edgecolor="black", label="DA-PFL (Ours)"),
    ]
    ax1.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)
    plt.title("Performance Trade-off: Discrimination vs. Calibration",
              pad=14, fontweight="bold")
    plt.tight_layout()
    _export_fig(fig, "fig1_main_performance", out_dir)


# =========================================================
# Figure 2 – APC mechanism (4-panel validation)
# =========================================================

def plot_apc_mechanism(apc_df: pd.DataFrame, out_dir: Path) -> None:
    if apc_df.empty:
        print("  Skipped Fig 2: no APC data.")
        return

    has_uk = "u_k" in apc_df.columns and apc_df["u_k"].notna().any()
    has_rk = "r_k" in apc_df.columns and apc_df["r_k"].notna().any()
    has_zeta = "Zeta" in apc_df.columns and apc_df["Zeta"].notna().any()
    has_drift = "realized_drift" in apc_df.columns and apc_df["realized_drift"].notna().any()
    has_n = "TrainSize" in apc_df.columns and apc_df["TrainSize"].notna().any()

    df = apc_df.copy().sort_values("AbsShift").reset_index(drop=True)
    shorts = df["Client"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip().str[:16]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Client-wise APC Allocation Patterns", fontsize=14, fontweight="bold", y=1.01)

    # =====================================================
    # Panel A: AbsShift -> E_pers
    # =====================================================
    ax = axes[0, 0]
    n_vals = df["TrainSize"].values if has_n else np.ones(len(df)) * 1000

    sc = ax.scatter(
        df["AbsShift"], df["E_pers"],
        c=n_vals,
        cmap="YlOrRd_r",
        s=150,
        edgecolors="white",
        linewidths=0.8,
        zorder=3
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Train size $n_k$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    mask = np.isfinite(df["AbsShift"]) & np.isfinite(df["E_pers"])
    if mask.sum() >= 2:
        x = df.loc[mask, "AbsShift"].values
        y = df.loc[mask, "E_pers"].values
        z = np.polyfit(x, y, 1)
        xx = np.linspace(x.min() * 0.9, x.max() * 1.05, 100)
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        ax.plot(xx, np.polyval(z, xx), "--", color="gray", lw=1.3,
                label=f"Trend (Spearman ρ={rho:.2f})")
        ax.legend(fontsize=8, loc="upper left")

    for i, row in df.iterrows():
        ax.annotate(shorts.iloc[i], (row["AbsShift"], row["E_pers"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5,
                    color=COLOR_SMALL if row.get("SmallFlag", 0) else "#2c3e50")

    ax.set_xlabel(r"Label-shift proxy $\delta_k$")
    ax.set_ylabel(r"Personalization epochs $E_k^{\mathrm{pers}}$")
    ax.set_title(r"Panel A: Shift intensity vs. personalization budget")
    ax.grid(alpha=0.25, lw=0.6)

    # =====================================================
    # Panel B: u_k -> r_k (dumbbell plot)
    # =====================================================
    ax = axes[0, 1]
    if has_uk and has_rk:
        tmp = df.sort_values("u_k").reset_index(drop=True)
        y_pos = np.arange(len(tmp))

        for i in range(len(tmp)):
            ax.plot(
                [tmp.loc[i, "r_k"], tmp.loc[i, "u_k"]],
                [i, i],
                color="0.7",
                lw=2,
                zorder=1
            )

        ax.scatter(tmp["u_k"], y_pos, s=70, label=r"$u_k$", zorder=3)
        ax.scatter(tmp["r_k"], y_pos, s=70, label=r"$r_k$", zorder=4)

        for i, row in tmp.iterrows():
            gap = row["u_k"] - row["r_k"]
            if np.isfinite(gap) and gap > 0.015:
                ax.annotate(f"-{gap:.2f}", (row["r_k"], i),
                            textcoords="offset points", xytext=(6, 0),
                            va="center", fontsize=7.5,
                            color=COLOR_SMALL if row.get("SmallFlag", 0) else "0.35")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(tmp["Client"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip().str[:16], fontsize=8)
        ax.set_xlabel("Signal magnitude")
        ax.set_title(r"Panel B: Gating effect from $u_k$ to $r_k$")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.25, lw=0.6)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Panel B unavailable\n($u_k$ or $r_k$ missing)",
                ha="center", va="center", fontsize=11)

    # =====================================================
    # Panel C: TrainSize vs E_pers, bubble size = K_pers
    # =====================================================
    ax = axes[1, 0]
    sizes = df["K_pers"].fillna(1).apply(lambda k: max(80, float(k) * 140))

    sc2 = ax.scatter(
        df["TrainSize"], df["E_pers"],
        s=sizes,
        c=df["AbsShift"],
        cmap="Blues",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.85
    )

    cb2 = fig.colorbar(sc2, ax=ax, pad=0.02, shrink=0.85)
    cb2.set_label(r"Shift intensity $\delta_k$", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    for i, row in df.iterrows():
        ax.annotate(shorts.iloc[i], (row["TrainSize"], row["E_pers"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5)

    ax.set_xlabel(r"Train size $n_k$")
    ax.set_ylabel(r"Personalization epochs $E_k^{\mathrm{pers}}$")
    ax.set_title(r"Panel C: Budget allocation vs. client size"
                 "\n(bubble size $\propto K_k$)")
    ax.grid(alpha=0.25, lw=0.6)

    # =====================================================
    # Panel D: realized drift vs personalization budget
    # =====================================================
    ax = axes[1, 1]
    if has_drift:
        for sf, label in [(0, "Large site"), (1, "Small site")]:
            sub = df[df["SmallFlag"] == sf]
            if sub.empty:
                continue
            ax.scatter(
                sub["E_pers"], sub["realized_drift"],
                s=sub["K_pers"].fillna(1).apply(lambda k: max(80, float(k) * 140)),
                alpha=0.8,
                edgecolors="black",
                linewidths=0.8,
                label=label
            )

        mask = np.isfinite(df["E_pers"]) & np.isfinite(df["realized_drift"])
        if mask.sum() >= 2:
            x = df.loc[mask, "E_pers"].values
            y = df.loc[mask, "realized_drift"].values
            z = np.polyfit(x, y, 1)
            xx = np.linspace(x.min() * 0.9, x.max() * 1.05, 100)
            rho = pd.Series(x).corr(pd.Series(y), method="spearman")
            ax.plot(xx, np.polyval(z, xx), "--", color="gray", lw=1.3,
                    label=f"Trend (Spearman ρ={rho:.2f})")

        for i, row in df.iterrows():
            ax.annotate(shorts.iloc[i], (row["E_pers"], row["realized_drift"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7.5)

        ax.set_xlabel(r"Personalization epochs $E_k^{\mathrm{pers}}$")
        ax.set_ylabel("Realized drift")
        ax.set_title(r"Panel D: Personalization budget vs. realized drift"
                     "\n(bubble size $\propto K_k$)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)

    elif has_zeta:
        tmp = df.sort_values("Zeta").reset_index(drop=True)
        y_pos = np.arange(len(tmp))
        ax.barh(y_pos, tmp["Zeta"], alpha=0.75)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tmp["Client"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip().str[:16], fontsize=8)
        ax.set_xlabel(r"Uncertainty proxy $\zeta_k$")
        ax.set_title(r"Panel D: Uncertainty signal $\zeta_k$")
        ax.grid(axis="x", alpha=0.25, lw=0.6)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Panel D unavailable\n(drift and zeta missing)",
                ha="center", va="center", fontsize=11)

    footer = (
        r"Interpretation: Panel A shows shift-sensitive epoch allocation; "
        r"Panel B shows gating from $u_k$ to $r_k$; "
        r"Panel C links client size, $E_k^{\mathrm{pers}}$, and $K_k$; "
        r"Panel D summarizes the drift associated with allocated personalization budget."
    )
    fig.text(0.5, -0.01, footer, ha="center", fontsize=8, color="0.35")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    _export_fig(fig, "fig2_apc_mechanism_revised", out_dir)


# =========================================================
# Figure 3 – Ablation study
# =========================================================

def _barh_with_err(ax, df, metric, metric_std, ref_val, color, title, xlabel):
    y = np.arange(len(df))
    colors = [COLOR_FULL if "FULL" in str(lbl) else color for lbl in df["label"]]
    bars = ax.barh(
        y, df[metric].fillna(0), xerr=df[metric_std].fillna(0),
        color=colors, edgecolor="black", alpha=0.9, capsize=4
    )
    ax.axvline(ref_val, color=COLOR_NEUTRAL, linestyle="--", lw=1.2, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    delta_col = "delta_auprc_vs_full" if "auprc" in metric else "delta_ece_vs_full"
    for i, bar in enumerate(bars):
        v = df.iloc[i][metric]
        delta = df.iloc[i].get(delta_col, float("nan"))
        suffix = f" ({delta:+.4f})" if not math.isnan(delta) else ""
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}{suffix}", va="center", ha="left", fontsize=9)


def plot_ablation_study(ablation_df: pd.DataFrame, out_dir: Path) -> None:
    if ablation_df.empty:
        print("  Skipped Fig 3: no ablation data.")
        return

    full_row = (
        ablation_df[ablation_df["method_key"] == FULL_METHOD_KEY].iloc[0]
        if FULL_METHOD_KEY in ablation_df["method_key"].values
        else ablation_df.iloc[0]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    _barh_with_err(
        ax1, ablation_df, "macro_auprc", "macro_auprc_std",
        float(full_row["macro_auprc"]), COLOR_AUPRC,
        "Impact on Discrimination (Macro AUPRC)", "Macro AUPRC ↑"
    )
    _barh_with_err(
        ax2, ablation_df, "macro_ece", "macro_ece_std",
        float(full_row["macro_ece"]), COLOR_ECE,
        "Impact on Calibration (Macro ECE)", "Macro ECE ↓"
    )
    plt.tight_layout()
    _export_fig(fig, "fig3_ablation_study", out_dir)


# =========================================================
# Figure 4 – Pareto scatter
# =========================================================

def plot_pareto_scatter(main_df: pd.DataFrame, ablation_df: pd.DataFrame,
                        out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.5))

    if not main_df.empty:
        for _, row in main_df.iterrows():
            c = COLOR_FULL if row["label"] == "DA-PFL (Ours)" else COLOR_AUPRC
            ax.scatter(row["macro_ece"], row["macro_auprc"],
                       s=160, c=c, edgecolors="black", alpha=0.9)
            ax.text(row["macro_ece"] + 0.001, row["macro_auprc"] + 0.001,
                    row["label"], fontsize=8.5)

    if not ablation_df.empty:
        ab_sub = ablation_df[ablation_df["label"].str.find("FULL") == -1]
        ax.scatter(ab_sub["macro_ece"], ab_sub["macro_auprc"],
                   s=80, c=COLOR_NEUTRAL, marker="D",
                   edgecolors="black", alpha=0.65, label="Ablations")

    ax.set_xlabel("Macro ECE ↓")
    ax.set_ylabel("Macro AUPRC ↑")
    ax.set_title("Pareto View: Accuracy–Calibration Frontier",
                 fontweight="bold", pad=12)

    handles = [
        mpatches.Patch(facecolor=COLOR_FULL, edgecolor="black", label="DA-PFL (Ours)"),
        mpatches.Patch(facecolor=COLOR_AUPRC, edgecolor="black", label="Baselines"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_NEUTRAL,
               markeredgecolor="black", ms=8, label="Ablations"),
    ]
    ax.legend(handles=handles, frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _export_fig(fig, "fig4_pareto_scatter", out_dir)


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = get_args()
    json_path = Path(args.json)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {json_path} …")
    payload = _load_json(json_path)

    print("Building DataFrames …")
    main_df = build_main_df(payload)
    ablation_df = build_ablation_df(payload)
    apc_df = build_apc_df(payload)

    print(f"  main_df rows:     {len(main_df)}")
    print(f"  ablation_df rows: {len(ablation_df)}")
    print(f"  apc_df rows:      {len(apc_df)}")

    if not apc_df.empty:
        required_cols = ["Zeta", "u_k", "r_k"]
        missing = [
            c for c in required_cols
            if c not in apc_df.columns or not apc_df[c].notna().any()
        ]
        if missing:
            print(f"  [Warning] APC intermediate fields partially missing: {missing}")
        else:
            print("  APC intermediate fields OK (u_k, r_k, ζ_k present).")

    save_tables(main_df, ablation_df, apc_df, out_dir)

    print("Generating figures …")
    plot_main_performance(main_df, out_dir)
    plot_apc_mechanism(apc_df, out_dir)
    plot_ablation_study(ablation_df, out_dir)
    plot_pareto_scatter(main_df, ablation_df, out_dir)

    print_key_findings(main_df, ablation_df, apc_df)
    print(f"\nAll outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()