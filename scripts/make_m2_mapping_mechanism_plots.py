from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "m2_mapping_preexperiment" / "m2_mapping_per_client.csv"
DEFAULT_FIG_DIR = ROOT / "figures" / "m2_mapping_mechanism"

ABBR: Dict[str, str] = {
    "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
    "Coronary Care Unit (CCU)": "CCU",
    "Medical Intensive Care Unit (MICU)": "MICU",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
    "Neuro Intermediate": "Neuro Inter.",
    "Neuro Stepdown": "Neuro Step.",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro SICU",
    "Surgical Intensive Care Unit (SICU)": "SICU",
    "Trauma SICU (TSICU)": "TSICU",
}


def _abbr(name: Any) -> str:
    return ABBR.get(str(name), str(name))


def _save(fig: plt.Figure, path: Path) -> List[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _agg_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    sub = df[df["variant"].astype(str) == str(variant)].copy()
    if sub.empty:
        raise ValueError(f"Variant not found in input: {variant}")
    cols = [
        "apc_delta",
        "apc_sigma_delta",
        "apc_q_reliability",
        "apc_s_reliability",
        "apc_gamma_s",
        "apc_gamma_s_scaled",
        "apc_r_reliability",
        "m_k",
        "e_k",
        "test_auc",
        "test_f1",
        "test_ece",
        "stage2_drift_norm",
        "val_bce_before_m3",
        "val_bce_after_m3",
    ]
    agg = sub.groupby(["client_id", "client_name"], as_index=False)[cols].mean(numeric_only=True)
    return agg


def plot_need_signal(agg: pd.DataFrame, variant: str, fig_dir: Path) -> List[Path]:
    df = agg.sort_values("apc_s_reliability", ascending=False).reset_index(drop=True)
    labels = [_abbr(x) for x in df["client_name"]]
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.8, 5.8), dpi=160)
    max_height = float(max(df["apc_delta"].max(), df["apc_s_reliability"].max()))
    ax.bar(x - width / 2, df["apc_delta"], width, label=r"Raw prior mismatch $\delta_k$", color="#7aa6c2", edgecolor="white")
    bars = ax.bar(
        x + width / 2,
        df["apc_s_reliability"],
        width,
        label=r"Reliability-adjusted score $s_k$",
        color="#f28e2b",
        edgecolor="white",
    )
    ax.set_ylim(0.0, max_height * 1.25)
    for rect, sigma in zip(bars, df["apc_sigma_delta"]):
        ax.annotate(
            rf"$\sigma$={sigma:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#4d4d4d",
            rotation=90,
        )
    ax.set_title(f"M2 need signal after reliability adjustment ({variant})", fontsize=13.5, fontweight="bold", pad=12)
    ax.set_ylabel("Signal value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(
        0.01,
        0.94,
        r"$s_k=\log(1+\delta_k/(\sigma_{\Delta,k}+\epsilon))$; text annotates $\sigma_{\Delta,k}$.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#525252",
        ha="left",
        va="top",
    )
    return _save(fig, fig_dir / f"{variant}_need_signal")


def plot_intensity_mapping(agg: pd.DataFrame, variant: str, fig_dir: Path) -> List[Path]:
    df = agg.sort_values("apc_r_reliability", ascending=True).reset_index(drop=True)
    labels = [_abbr(x) for x in df["client_name"]]
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=160)
    ax.hlines(y, 0.0, df["apc_r_reliability"], color="#c7c7c7", linewidth=1.8)
    sc = ax.scatter(
        df["apc_r_reliability"],
        y,
        c=df["apc_s_reliability"],
        cmap="YlOrRd",
        s=108,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    for idx, row in df.iterrows():
        ax.annotate(
            f"{int(round(row['m_k']))}/{int(round(row['e_k']))}",
            (row["apc_r_reliability"], idx),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )
    ax.set_title(f"M2 intensity mapped to scope/depth ({variant})", fontsize=13.5, fontweight="bold", pad=12)
    ax.set_xlabel(r"Personalization intensity $r_k$")
    ax.set_ylabel("Client")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, min(1.0, max(0.75, float(df["apc_r_reliability"].max()) + 0.12)))
    ax.grid(axis="x", alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(r"Need score $s_k$", fontsize=9)
    ax.text(0.98, 0.04, "marker label = scope/depth", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#525252")
    return _save(fig, fig_dir / f"{variant}_intensity_scope_depth")


def plot_vs_fixed(df: pd.DataFrame, variant: str, fixed_variant: str, fig_dir: Path) -> List[Path]:
    sub = df[df["variant"].isin([variant, fixed_variant])].copy()
    if sub["variant"].nunique() < 2:
        return []
    piv = sub.pivot_table(
        index=["client_id", "client_name"],
        columns="variant",
        values=["test_auc", "test_ece", "m_k", "e_k"],
        aggfunc="mean",
    )
    rows = []
    for (cid, cname), row in piv.iterrows():
        rows.append({
            "client_id": int(cid),
            "client_name": str(cname),
            "auc_gain": float(row[("test_auc", variant)] - row[("test_auc", fixed_variant)]),
            "ece_gain": float(row[("test_ece", fixed_variant)] - row[("test_ece", variant)]),
            "m_k": float(row[("m_k", variant)]),
            "e_k": float(row[("e_k", variant)]),
        })
    eff = pd.DataFrame(rows).sort_values("auc_gain", ascending=True).reset_index(drop=True)
    labels = [_abbr(x) for x in eff["client_name"]]
    y = np.arange(len(eff))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), dpi=160, sharey=True)
    colors_auc = np.where(eff["auc_gain"] >= 0, "#4e79a7", "#d55e00")
    axes[0].barh(y, eff["auc_gain"], color=colors_auc, alpha=0.88)
    axes[0].axvline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel(f"AUC({variant}) - AUC({fixed_variant})")
    axes[0].set_title("Client AUC difference", fontsize=11, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.22)

    colors_ece = np.where(eff["ece_gain"] >= 0, "#59a14f", "#d55e00")
    axes[1].barh(y, eff["ece_gain"], color=colors_ece, alpha=0.88)
    axes[1].axvline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(f"ECE({fixed_variant}) - ECE({variant})")
    axes[1].set_title("Client calibration difference", fontsize=11, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.22)
    for idx, row in eff.iterrows():
        axes[1].annotate(
            f"{int(round(row['m_k']))}/{int(round(row['e_k']))}",
            (row["ece_gain"], idx),
            xytext=(5 if row["ece_gain"] >= 0 else -5, 0),
            textcoords="offset points",
            va="center",
            ha="left" if row["ece_gain"] >= 0 else "right",
            fontsize=7.8,
        )
    fig.suptitle(f"Adaptive M2 versus fixed full-scope ({variant})", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, fig_dir / f"{variant}_vs_{fixed_variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mechanism plots for M2 mapping pre-experiment.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--variant", type=str, default="m2_product")
    parser.add_argument("--fixed-variant", type=str, default="fixed_fullscope_m7_e5")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    agg = _agg_variant(df, args.variant)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "results" / "m2_mapping_preexperiment" / f"{args.variant}_mechanism_summary.csv"
    agg.to_csv(summary_path, index=False)

    paths: List[Path] = []
    paths.extend(plot_need_signal(agg, args.variant, args.fig_dir))
    paths.extend(plot_intensity_mapping(agg, args.variant, args.fig_dir))
    paths.extend(plot_vs_fixed(df, args.variant, args.fixed_variant, args.fig_dir))
    paths.append(summary_path)

    print("[M2 mapping mechanism plots]")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
