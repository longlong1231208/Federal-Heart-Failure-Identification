from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "apc_need_reliability_full_mechanism_raw.csv"
DEFAULT_CSV = ROOT / "results" / "apc_need_reliability_full_mechanism.csv"
DEFAULT_FIG = ROOT / "figures" / "ch4" / "fig_m2_apc_need_reliability_mechanism_full"


ABBR = {
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


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _read_m2_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"M2 mechanism table not found: {path}")
    df = pd.read_csv(path)
    required = [
        "client_id",
        "client_name",
        "n_k",
        "delta",
        "sigma_delta",
        "q_reliability",
        "u_k",
        "alpha_k",
        "m_k",
        "e_k",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def _build_export_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "eta_k" not in work.columns:
        work["eta_k"] = work.get("s_reliability", np.nan)
    work["r_k"] = work["u_k"].astype(float) * work["alpha_k"].astype(float)
    if "r_selected" not in work.columns:
        if "r_final" in work.columns:
            work["r_selected"] = work["r_final"]
        elif "r_reliability" in work.columns:
            work["r_selected"] = work["r_reliability"]
        else:
            work["r_selected"] = work["r_k"]

    columns = [
        "client_id",
        "client_name",
        "n_k",
        "n_k_pos",
        "n_k_neg",
        "pi_k",
        "pi_g",
        "delta_b",
        "delta",
        "sigma_delta",
        "q_reliability",
        "eta_k",
        "u_k",
        "alpha_k",
        "r_k",
        "r_selected",
        "m_k",
        "e_k",
    ]
    out = work[[c for c in columns if c in work.columns]].copy()
    numeric_cols = [c for c in out.columns if c not in {"client_name"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["client_id", "client_name"], as_index=False).mean(numeric_only=True)
    agg["r_formula"] = (agg["u_k"].astype(float) * agg["alpha_k"].astype(float)).clip(0.0, 1.0)
    return agg.sort_values("r_k", ascending=False).reset_index(drop=True)


def _save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _plot(agg: pd.DataFrame, stem: Path) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(int(x)) for x in agg["client_id"]]
    x = np.arange(len(agg))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8), dpi=180)

    ax = axes[0]
    sc = ax.scatter(
        agg["u_k"],
        agg["e_k"],
        c=agg["n_k"],
        cmap="viridis",
        s=115,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    for _, row in agg.iterrows():
        ax.annotate(
            str(int(row["client_id"])),
            (row["u_k"], row["e_k"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Panel A: Adaptation need and personalization depth", fontsize=12.5, fontweight="bold")
    ax.set_xlabel(r"adaptation-need score $u_k$")
    ax.set_ylabel(r"personalization depth $e_k$")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(r"client size $n_k$")

    ax = axes[1]
    ax.bar(
        x - width / 2,
        agg["u_k"],
        width,
        label=r"$u_k$ (adaptation-need score)",
        color="#76a5c5",
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        agg["r_k"],
        width,
        label=r"$r_k=u_k\alpha_k$ (final personalization intensity)",
        color="#f28e2b",
        edgecolor="white",
    )
    for idx, row in agg.iterrows():
        top = max(_finite_float(row["u_k"]), _finite_float(row["r_k"]))
        ax.annotate(
            rf"$\alpha_k$={row['alpha_k']:.2f}" + f"\n{int(round(row['m_k']))}/{int(round(row['e_k']))}",
            (idx, top),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            rotation=90,
        )
    ax.set_title("Panel B: Need score is moderated by local reliability", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("client_id")
    ax.set_ylabel("signal value in [0, 1]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_ylim(0.0, 1.25)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.text(
        0.01,
        0.95,
        r"annotation = sample-size reliability factor $\alpha_k$ and scope/depth $m_k/e_k$",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#555555",
    )

    fig.suptitle("M2/APC need-by-reliability personalization mechanism", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def _allocations(agg: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in agg.iterrows():
        rows.append({
            "client_id": int(row["client_id"]),
            "client_name": str(row["client_name"]),
            "m_k": int(round(row["m_k"])),
            "e_k": int(round(row["e_k"])),
            "u_k": float(row["u_k"]),
            "alpha_k": float(row["alpha_k"]),
            "r_k": float(row["r_k"]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final Chapter 4 M2/APC need-by-reliability mechanism figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--fig-stem", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()

    raw = _read_m2_table(args.input)
    table = _build_export_table(raw)
    _save_csv(args.csv, table)
    agg = _aggregate(table)
    paths = _plot(agg, args.fig_stem)

    numeric = table.drop(columns=["client_name"], errors="ignore").to_numpy(dtype=float)
    has_bad = bool(np.isnan(numeric).any() or np.isinf(numeric).any())

    print("[M2/APC need-by-reliability mechanism]")
    print(f"input={args.input}")
    print(f"csv={args.csv}")
    for path in paths:
        print(f"figure={path}")
    print(f"clients={int(agg.shape[0])}")
    print(f"has_nan_or_inf={has_bad}")
    print("allocations:")
    for row in _allocations(agg):
        print(
            f"  - {row['client_id']}: {row['client_name']} | "
            f"m/e={row['m_k']}/{row['e_k']} | "
            f"u={row['u_k']:.3f} | alpha={row['alpha_k']:.3f} | r={row['r_k']:.3f}"
        )


if __name__ == "__main__":
    main()
