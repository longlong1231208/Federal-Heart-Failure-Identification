from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures" / "final_mechanism"

METHOD_BACKBONE = "Backbone Only"
METHOD_M1 = "Backbone + M1"
METHOD_FIXED = "Backbone + M1 + M3 without M2/APC"
METHOD_FULL = "Full DA-PFL new M2"
METHOD_RANDOM = "M1 + new M2 + random mask"
METHOD_FULL_FT = "M1 + full fine-tuning"

GROUP_ORDER = ["g1_head", "g2_l0_ih", "g3_l0_hh", "g4_l0_b", "g5_l1_ih", "g6_l1_hh", "g7_l1_b"]
NON_HEAD_GROUPS = [g for g in GROUP_ORDER if g != "g1_head"]

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


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_effect_rows(results_dir: Path) -> pd.DataFrame:
    frames = [_read_csv(results_dir / "dapfl_component_ablation_per_client.csv")]
    for filename in ["random_mask_control_per_client.csv", "full_finetune_after_m1_per_client.csv"]:
        df = _read_csv(results_dir / filename, required=False)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _warn_missing(df: pd.DataFrame, cols: Iterable[str], table_name: str) -> List[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[Warning] {table_name} missing columns: {missing}")
    return missing


def _abbr(name: Any) -> str:
    return ABBR.get(str(name), str(name))


def _client_order(m2: pd.DataFrame, effects: pd.DataFrame) -> List[str]:
    if {"client_id", "client_name"}.issubset(m2.columns):
        return (
            m2[["client_id", "client_name"]]
            .drop_duplicates()
            .sort_values("client_id")["client_name"]
            .astype(str)
            .tolist()
        )
    return (
        effects[["client_id", "client_name"]]
        .drop_duplicates()
        .sort_values("client_id")["client_name"]
        .astype(str)
        .tolist()
    )


def _pearson(x: Iterable[float], y: Iterable[float]) -> float:
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if mask.sum() < 2:
        return float("nan")
    if float(np.std(xx[mask])) == 0.0 or float(np.std(yy[mask])) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx[mask], yy[mask])[0, 1])


def _gini(values: Iterable[float]) -> float:
    arr = np.sort(np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float))
    if arr.size == 0:
        return float("nan")
    idx = np.arange(1, arr.size + 1)
    return float(((2 * idx - arr.size - 1) * arr).sum() / (arr.size * (arr.sum() + 1e-12)))


def _save(fig: plt.Figure, stem: str, outputs: List[Path]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    outputs.extend([png, pdf])


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.tick_params(labelsize=9)


def _bar_err(ax: plt.Axes, x: np.ndarray, mean: Iterable[float], std: Iterable[float], **kwargs) -> None:
    ax.bar(
        np.asarray(x, dtype=float),
        np.asarray(list(mean), dtype=float),
        yerr=np.asarray(list(std), dtype=float),
        capsize=3,
        linewidth=0.6,
        edgecolor="black",
        **kwargs,
    )


def _group_mean(df: pd.DataFrame, value: str) -> pd.Series:
    return df.groupby("client_name")[value].mean()


def _group_std(df: pd.DataFrame, value: str) -> pd.Series:
    return df.groupby("client_name")[value].std(ddof=1).fillna(0.0)


def _m1_summary(effects: pd.DataFrame, m2: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    base = effects[effects["method"] == METHOD_BACKBONE].copy()
    m1 = effects[effects["method"] == METHOD_M1].copy()
    delta = m2.groupby(["client_id", "client_name"], as_index=False).agg(delta_b=("delta_b", "mean"))

    rows: List[Dict[str, Any]] = []
    for client in order:
        b = base[base["client_name"] == client]
        a = m1[m1["client_name"] == client]
        drow = delta[delta["client_name"] == client]
        if b.empty or a.empty or drow.empty:
            continue
        rows.append({
            "client_id": int(drow["client_id"].iloc[0]),
            "client_name": client,
            "delta_b": float(drow["delta_b"].iloc[0]),
            "ece_before_m1_mean": float(b["test_ece"].mean()),
            "ece_before_m1_std": float(b["test_ece"].std(ddof=1)),
            "ece_after_m1_mean": float(a["test_ece"].mean()),
            "ece_after_m1_std": float(a["test_ece"].std(ddof=1)),
            "ece_improvement": float(b["test_ece"].mean() - a["test_ece"].mean()),
            "mean_pred_before_m1": float(b["mean_pred_prob"].mean()) if "mean_pred_prob" in b.columns else float("nan"),
            "mean_pred_after_m1": float(a["mean_pred_prob"].mean()) if "mean_pred_prob" in a.columns else float("nan"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mean_pred_shift"] = out["mean_pred_after_m1"] - out["mean_pred_before_m1"]
    return out


def _selected_score_summary(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["client_id", "client_name", "selected_score_mean"])
    group_col = "group_name" if "group_name" in scores.columns else "group"
    score_col = "group_score" if "group_score" in scores.columns else "score"
    needed = {"seed", "client_id", "client_name", group_col, score_col, "selected"}
    if not needed.issubset(scores.columns):
        return pd.DataFrame(columns=["client_id", "client_name", "selected_score_mean"])

    rows: List[Dict[str, Any]] = []
    for (seed, client_id, client_name), group in scores.groupby(["seed", "client_id", "client_name"]):
        selected = group[(group["selected"].astype(int) == 1) & (group[group_col].astype(str) != "g1_head")]
        if selected.empty:
            selected = group[group["selected"].astype(int) == 1]
        rows.append({
            "seed": seed,
            "client_id": client_id,
            "client_name": client_name,
            "selected_score_mean_seed": float(selected[score_col].mean()) if not selected.empty else float("nan"),
        })
    if not rows:
        return pd.DataFrame(columns=["client_id", "client_name", "selected_score_mean"])
    per_seed = pd.DataFrame(rows)
    return per_seed.groupby(["client_id", "client_name"], as_index=False).agg(
        selected_score_mean=("selected_score_mean_seed", "mean")
    )


def _m2_summary(m2: pd.DataFrame, m3: pd.DataFrame, order: List[str], scores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    m2_mean = m2.groupby(["client_id", "client_name"], as_index=False).agg(
        delta=("delta", "mean"),
        sigma_delta=("sigma_delta", "mean"),
        q_reliability=("q_reliability", "mean"),
        s_reliability=("s_reliability", "mean"),
        gamma_s=("gamma_s", "mean"),
        gamma_s_scaled=("gamma_s_scaled", "mean"),
        r_reliability=("r_reliability", "mean"),
        m_k=("m_k", "mean"),
        e_k=("e_k", "mean"),
    )
    m3_mean = m3.groupby(["client_id", "client_name"], as_index=False).agg(
        total_drift_norm=("total_drift_norm", "mean"),
        val_bce_before_m3=("val_bce_before_m3", "mean"),
        val_bce_after_m3=("val_bce_after_m3", "mean"),
    )
    out = m2_mean.merge(m3_mean, on=["client_id", "client_name"], how="left")
    if scores is not None and not scores.empty:
        out = out.merge(_selected_score_summary(scores), on=["client_id", "client_name"], how="left")
    else:
        out["selected_score_mean"] = np.nan
    out["effective_adaptation_pressure"] = out["r_reliability"] * out["selected_score_mean"]
    out["delta_val_bce"] = out["val_bce_before_m3"] - out["val_bce_after_m3"]
    out["client_name"] = pd.Categorical(out["client_name"], categories=order, ordered=True)
    return out.sort_values("client_name").reset_index(drop=True)


def _m3_summary(m3: pd.DataFrame, scores: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    group_col = "group_name" if "group_name" in scores.columns else "group"
    score_col = "group_score" if "group_score" in scores.columns else "score"
    rows: List[Dict[str, Any]] = []
    for client in order:
        m3c = m3[m3["client_name"] == client]
        sc = scores[scores["client_name"] == client].copy()
        if m3c.empty or sc.empty:
            continue

        selected_ranks: List[float] = []
        unselected_ranks: List[float] = []
        selected_scores: List[float] = []
        unselected_scores: List[float] = []
        for _, group in sc.groupby(["seed", "client_name"]):
            non_head = group[group[group_col] != "g1_head"].copy()
            non_head = non_head.sort_values(score_col, ascending=False)
            rank_map = {str(g): i + 1 for i, g in enumerate(non_head[group_col].tolist())}
            for _, row in non_head.iterrows():
                rank = float(rank_map[str(row[group_col])])
                score = float(row[score_col])
                if int(row["selected"]) == 1:
                    selected_ranks.append(rank)
                    selected_scores.append(score)
                else:
                    unselected_ranks.append(rank)
                    unselected_scores.append(score)

        avg_selected = sc.groupby(group_col)["selected"].mean().to_dict()
        selected_groups = [g for g in GROUP_ORDER if float(avg_selected.get(g, 0.0)) >= 0.5]
        rows.append({
            "client_id": int(m3c["client_id"].iloc[0]),
            "client_name": client,
            "m_k": float(m3c["m_k"].mean()),
            "e_k": float(m3c["e_k"].mean()),
            "selected_groups": ";".join(selected_groups),
            "active_group_count": float(m3c["active_group_count"].mean()),
            "mean_selected_non_head_rank": float(np.mean(selected_ranks)) if selected_ranks else float("nan"),
            "mean_unselected_non_head_rank": float(np.mean(unselected_ranks)) if unselected_ranks else float("nan"),
            "mean_selected_non_head_score": float(np.mean(selected_scores)) if selected_scores else float("nan"),
            "mean_unselected_non_head_score": float(np.mean(unselected_scores)) if unselected_scores else float("nan"),
        })
    return pd.DataFrame(rows)


def fig1_m1(m1: pd.DataFrame, outputs: List[Path]) -> Dict[str, float]:
    has_shift = {"mean_pred_before_m1", "mean_pred_after_m1", "mean_pred_shift"}.issubset(m1.columns)
    has_shift = has_shift and m1["mean_pred_shift"].notna().any()
    ncols = 2 if has_shift else 1
    fig, axes = plt.subplots(1, ncols, figsize=(13.0 if has_shift else 7.0, 4.5), dpi=160)
    if ncols == 1:
        axes = [axes]
    labels = [_abbr(x) for x in m1["client_name"]]
    x = np.arange(len(labels))
    width = 0.38

    _bar_err(
        axes[0],
        x - width / 2,
        m1["ece_before_m1_mean"],
        m1["ece_before_m1_std"].fillna(0.0),
        width=width,
        label="Before M1",
        color="#9ecae1",
    )
    _bar_err(
        axes[0],
        x + width / 2,
        m1["ece_after_m1_mean"],
        m1["ece_after_m1_std"].fillna(0.0),
        width=width,
        label="After M1",
        color="#3182bd",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=40, ha="right")
    axes[0].legend(frameon=False, fontsize=9)
    _style(axes[0], "A. Client-level ECE before and after M1", "Client", "Test ECE")

    corr = float("nan")
    if has_shift:
        corr = _pearson(m1["delta_b"], m1["mean_pred_shift"])
        axes[1].scatter(m1["delta_b"], m1["mean_pred_shift"], s=58, color="#756bb1")
        for _, row in m1.iterrows():
            axes[1].annotate(_abbr(row["client_name"]), (row["delta_b"], row["mean_pred_shift"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].axvline(0, color="black", linewidth=1)
        _style(axes[1], f"B. Prior shift vs prediction shift (r={corr:.2f})", r"$\Delta b_k$", "Mean prediction shift after M1")
    else:
        print("mean_pred_prob missing; skipping M1 prediction-shift panel.")

    fig.suptitle("M1 prior correction changes operating point and calibration", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig1_m1_operating_point_calibration", outputs)
    return {
        "corr_delta_pred_shift": corr,
        "mean_ece_before": float(m1["ece_before_m1_mean"].mean()),
        "mean_ece_after": float(m1["ece_after_m1_mean"].mean()),
        "mean_ece_improvement": float(m1["ece_improvement"].mean()),
    }


def fig2_m2(m2: pd.DataFrame, outputs: List[Path]) -> Dict[str, float]:
    corr_intensity_drift = _pearson(m2["r_reliability"], m2["total_drift_norm"])
    has_pressure = "effective_adaptation_pressure" in m2.columns and m2["effective_adaptation_pressure"].notna().any()
    corr_pressure_drift = _pearson(m2["effective_adaptation_pressure"], m2["total_drift_norm"]) if has_pressure else float("nan")
    corr_drift_gain = _pearson(m2["total_drift_norm"], m2["delta_val_bce"])

    if has_pressure:
        fig, axes_arr = plt.subplots(1, 2, figsize=(12.4, 5.4), dpi=160)
        axes = list(axes_arr)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.4), dpi=160)
        axes = [ax]
        print("[Warning] selected_score_mean unavailable; skipping M2 effective-pressure panel.")
    label_box = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78)

    sorted_m2 = m2.sort_values("r_reliability", ascending=True).reset_index(drop=True)
    y = np.arange(len(sorted_m2))
    sorted_labels = [_abbr(x) for x in sorted_m2["client_name"]]
    sorted_r = sorted_m2["r_reliability"].to_numpy(dtype=float)
    delta = sorted_m2["delta"].to_numpy(dtype=float)
    delta_min = float(np.nanmin(delta))
    delta_max = float(np.nanmax(delta))

    axes[0].hlines(y, 0.0, sorted_r, color="#c7c7c7", linewidth=1.6, zorder=1)
    sc = axes[0].scatter(
        sorted_r,
        y,
        c=delta,
        cmap="YlOrRd",
        s=95,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for idx, row in sorted_m2.iterrows():
        scope = int(round(float(row["m_k"])))
        depth = int(round(float(row["e_k"])))
        axes[0].annotate(
            f"{scope}/{depth}",
            (row["r_reliability"], idx),
            fontsize=8,
            xytext=(8, -1),
            textcoords="offset points",
            va="center",
            bbox=label_box,
        )
    cbar = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.025)
    cbar.set_label(r"Prior mismatch $\delta_k$", fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sorted_labels)
    axes[0].set_xlim(0.0, min(1.0, max(0.78, float(sorted_r.max()) + 0.09)))
    axes[0].set_ylim(-0.6, len(sorted_m2) - 0.4)
    _style(
        axes[0],
        "A. Client-specific personalization intensity and scope/depth",
        r"Personalization intensity $r_k$",
        "Client",
    )
    axes[0].text(
        0.98,
        0.04,
        "marker label = scope/depth",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#525252",
    )

    if has_pressure:
        labels = [_abbr(x) for x in m2["client_name"]]
        axes[1].scatter(
            m2["effective_adaptation_pressure"],
            m2["total_drift_norm"],
            c=m2["delta"],
            cmap="YlOrRd",
            s=78,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        x_vals = m2["effective_adaptation_pressure"].to_numpy(dtype=float)
        y_vals = m2["total_drift_norm"].to_numpy(dtype=float)
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if mask.sum() >= 2 and np.std(x_vals[mask]) > 0:
            coef = np.polyfit(x_vals[mask], y_vals[mask], deg=1)
            x_grid = np.linspace(float(np.min(x_vals[mask])), float(np.max(x_vals[mask])), 80)
            axes[1].plot(x_grid, coef[0] * x_grid + coef[1], color="#636363", linewidth=1.5, linestyle="--", zorder=2)
        offsets = {
            "CVICU": (6, 8),
            "Neuro Inter.": (6, 8),
            "SICU": (6, 4),
            "TSICU": (6, -12),
            "Neuro SICU": (6, 4),
            "MICU/SICU": (6, -10),
            "Neuro Step.": (6, -12),
        }
        for i, row in m2.iterrows():
            axes[1].annotate(
                labels[i],
                (row["effective_adaptation_pressure"], row["total_drift_norm"]),
                fontsize=7.8,
                xytext=offsets.get(labels[i], (5, 4)),
                textcoords="offset points",
                bbox=label_box,
            )
        p_min = float(np.nanmin(m2["effective_adaptation_pressure"]))
        p_max = float(np.nanmax(m2["effective_adaptation_pressure"]))
        p_pad = max((p_max - p_min) * 0.14, 0.001)
        drift_max = float(np.nanmax(m2["total_drift_norm"]))
        axes[1].set_xlim(max(0.0, p_min - p_pad), p_max + p_pad * 1.8)
        axes[1].set_ylim(0.0, drift_max * 1.12)
        _style(
            axes[1],
            f"B. Realized drift depends on intensity and selected-gradient sensitivity (r={corr_pressure_drift:.2f})",
            r"$r_k \times$ mean selected non-head gradient score",
            "Parameter drift from M1 anchor",
        )

    fig.suptitle("M2/APC controls client-specific personalization intensity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig2_m2_apc_controlled_intensity", outputs)
    return {
        "corr_intensity_drift": corr_intensity_drift,
        "corr_pressure_drift": corr_pressure_drift,
        "corr_drift_val_bce": corr_drift_gain,
    }


def fig3_m3(scores: pd.DataFrame, order: List[str], outputs: List[Path]) -> Dict[str, float]:
    group_col = "group_name" if "group_name" in scores.columns else "group"
    score_col = "group_score" if "group_score" in scores.columns else "score"
    selected_freq_all = scores.groupby(group_col)["selected"].mean().to_dict()
    shown_groups = [g for g in GROUP_ORDER if g == "g1_head" or float(selected_freq_all.get(g, 0.0)) > 0.05]

    freq = scores.groupby(["client_name", group_col])["selected"].mean().unstack(group_col).reindex(order).fillna(0.0)
    freq = freq[[g for g in shown_groups if g in freq.columns]]

    selected_ranks: List[float] = []
    unselected_ranks: List[float] = []
    selected_scores: List[float] = []
    unselected_scores: List[float] = []
    for _, group in scores.groupby(["seed", "client_name"]):
        non_head = group[group[group_col] != "g1_head"].copy().sort_values(score_col, ascending=False)
        rank_map = {str(g): i + 1 for i, g in enumerate(non_head[group_col].tolist())}
        for _, row in non_head.iterrows():
            rank = float(rank_map[str(row[group_col])])
            score = float(row[score_col])
            if int(row["selected"]) == 1:
                selected_ranks.append(rank)
                selected_scores.append(score)
            else:
                unselected_ranks.append(rank)
                unselected_scores.append(score)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=160)
    im = axes[0].imshow(freq.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axes[0].set_title("A. Selected-group frequency", fontsize=12, fontweight="bold")
    axes[0].set_xticks(np.arange(len(freq.columns)))
    axes[0].set_xticklabels(freq.columns.tolist(), rotation=40, ha="right")
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([_abbr(x) for x in order])
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("Selection frequency", fontsize=9)

    bp = axes[1].boxplot([selected_ranks, unselected_ranks], labels=["Selected", "Unselected"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#74c476", "#fdae6b"]):
        patch.set_facecolor(color)
    axes[1].invert_yaxis()
    _style(axes[1], "B. Selected vs unselected non-head ranks", "", "Gradient-sensitivity rank (lower is better)")

    fig.suptitle("M3 selects gradient-sensitive parameter groups", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig3_m3_gradient_sensitive_selection", outputs)
    return {
        "mean_selected_rank": float(np.mean(selected_ranks)) if selected_ranks else float("nan"),
        "mean_unselected_rank": float(np.mean(unselected_ranks)) if unselected_ranks else float("nan"),
        "mean_selected_score": float(np.mean(selected_scores)) if selected_scores else float("nan"),
        "mean_unselected_score": float(np.mean(unselected_scores)) if unselected_scores else float("nan"),
    }


def fig4_controlled(effects: pd.DataFrame, m3: pd.DataFrame, order: List[str], outputs: List[Path]) -> Optional[Dict[str, float]]:
    if METHOD_FULL_FT not in set(effects["method"].astype(str)):
        print("Full fine-tuning results not found; skipping Figure 4.")
        return None

    full = effects[effects["method"] == METHOD_FULL].copy()
    ft = effects[effects["method"] == METHOD_FULL_FT].copy()
    merged = full.merge(ft, on=["seed", "client_id", "client_name"], suffixes=("_dapfl", "_fullft"))
    drift = m3.groupby(["seed", "client_name"], as_index=False).agg(drift_dapfl=("total_drift_norm", "mean"))
    merged = merged.merge(drift, on=["seed", "client_name"], how="left")

    by_client = merged.groupby("client_name", as_index=False).agg(
        drift_dapfl=("drift_dapfl", "mean"),
        drift_fullft=("stage2_drift_norm_fullft", "mean"),
        ece_dapfl=("test_ece_dapfl", "mean"),
        ece_fullft=("test_ece_fullft", "mean"),
    )
    by_client["ece_difference"] = by_client["ece_fullft"] - by_client["ece_dapfl"]
    by_client["client_name"] = pd.Categorical(by_client["client_name"], categories=order, ordered=True)
    by_client = by_client.sort_values("client_name")

    labels = [_abbr(x) for x in by_client["client_name"]]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), dpi=160)
    axes[0].bar(x - width / 2, by_client["drift_dapfl"], width=width, label="DA-PFL", color="#3182bd", edgecolor="black", linewidth=0.4)
    axes[0].bar(x + width / 2, by_client["drift_fullft"], width=width, label="Full fine-tuning", color="#fdae6b", edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=40, ha="right")
    axes[0].legend(frameon=False, fontsize=9)
    _style(axes[0], "A. Drift norm by client", "Client", "Total drift norm")

    axes[1].bar(x, by_client["ece_difference"], color="#74c476", edgecolor="black", linewidth=0.4)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    _style(axes[1], "B. Calibration difference", "Client", r"$ECE_{FullFT}-ECE_{DA-PFL}$")

    fig.suptitle("Controlled personalization versus unrestricted fine-tuning", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig4_controlled_vs_unrestricted", outputs)
    return {
        "mean_drift_dapfl": float(by_client["drift_dapfl"].mean()),
        "mean_drift_fullft": float(by_client["drift_fullft"].mean()),
        "mean_ece_difference": float(by_client["ece_difference"].mean()),
    }


def appendix_client_effects(effects: pd.DataFrame, order: List[str], outputs: List[Path]) -> None:
    needed = {METHOD_BACKBONE, METHOD_M1, METHOD_FULL}
    if not needed.issubset(set(effects["method"].astype(str))):
        return
    base = effects[effects["method"] == METHOD_BACKBONE]
    m1 = effects[effects["method"] == METHOD_M1]
    full = effects[effects["method"] == METHOD_FULL]
    rows: List[Dict[str, Any]] = []
    for client in order:
        b = base[base["client_name"] == client]
        a = m1[m1["client_name"] == client]
        f = full[full["client_name"] == client]
        if b.empty or a.empty or f.empty:
            continue
        rows.append({
            "client_name": client,
            "auc_m1": float(a["test_auc"].mean()),
            "auc_full": float(f["test_auc"].mean()),
            "delta_auc": float(f["test_auc"].mean() - a["test_auc"].mean()),
            "delta_f1": float(f["test_f1"].mean() - a["test_f1"].mean()),
            "ece_m1": float(a["test_ece"].mean()),
            "ece_full": float(f["test_ece"].mean()),
        })
    df = pd.DataFrame(rows)
    labels = [_abbr(x) for x in df["client_name"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=160)
    axes[0].plot(df["auc_m1"].to_numpy(dtype=float), x, marker="o", label="After M1", color="#9ecae1")
    axes[0].plot(df["auc_full"].to_numpy(dtype=float), x, marker="o", label="Full DA-PFL", color="#3182bd")
    for i in range(len(df)):
        axes[0].plot([float(df["auc_m1"].iloc[i]), float(df["auc_full"].iloc[i])], [i, i], color="gray", alpha=0.45)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8)
    _style(axes[0], "A. AUC comparison", "Test AUC", "Client")
    axes[1].bar(x - 0.17, df["delta_auc"].to_numpy(dtype=float), width=0.34, label="AUC", color="#756bb1")
    axes[1].bar(x + 0.17, df["delta_f1"].to_numpy(dtype=float), width=0.34, label="F1", color="#fdae6b")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    axes[1].legend(frameon=False, fontsize=8)
    _style(axes[1], "B. Client-level changes", "Client", "Change")
    axes[2].bar(x - 0.17, df["ece_m1"].to_numpy(dtype=float), width=0.34, label="After M1", color="#9ecae1")
    axes[2].bar(x + 0.17, df["ece_full"].to_numpy(dtype=float), width=0.34, label="Full DA-PFL", color="#3182bd")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=40, ha="right")
    axes[2].legend(frameon=False, fontsize=8)
    _style(axes[2], "C. ECE comparison", "Client", "Test ECE")
    fig.suptitle("Appendix: client-level effects of DA-PFL", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "appendix_client_level_effects", outputs)


def appendix_m2_construction(m2: pd.DataFrame, outputs: List[Path]) -> None:
    labels = [_abbr(x) for x in m2["client_name"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), dpi=160)
    axes[0].bar(x - 0.18, m2["delta"].to_numpy(dtype=float), width=0.36, label=r"$\delta$", color="#9ecae1")
    axes[0].bar(x + 0.18, m2["sigma_delta"].to_numpy(dtype=float), width=0.36, label=r"$\sigma_\Delta$", color="#fdae6b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=40, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    _style(axes[0], "A. Prior mismatch and uncertainty scale", "Client", "Value")
    axes[1].plot(x, m2["q_reliability"].to_numpy(dtype=float), marker="o", label=r"$q_k$", color="#756bb1")
    axes[1].plot(x, m2["s_reliability"].to_numpy(dtype=float), marker="o", label=r"$s_k$", color="#31a354")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    axes[1].legend(frameon=False, fontsize=8)
    _style(axes[1], "B. Reliability signal and log-transformed signal", "Client", "Value")
    axes[2].plot(x, m2["r_reliability"].to_numpy(dtype=float), marker="o", label=r"$r_k$", color="#3182bd")
    axes[2].bar(x - 0.16, (m2["m_k"] / 7.0).to_numpy(dtype=float), width=0.32, label=r"scope $m_k/G$", color="#74c476")
    axes[2].bar(x + 0.16, (m2["e_k"] / 10.0).to_numpy(dtype=float), width=0.32, label=r"depth $e_k/E_{max}$", color="#fdae6b")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=40, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(frameon=False, fontsize=8)
    _style(axes[2], "C. Personalization intensity, scope, and depth", "Client", "Normalized value")
    fig.suptitle("Appendix: M2 signal construction", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "appendix_m2_signal_construction", outputs)


def appendix_m2_intensity_drift(m2: pd.DataFrame, outputs: List[Path]) -> None:
    corr = _pearson(m2["r_reliability"], m2["total_drift_norm"])
    labels = [_abbr(x) for x in m2["client_name"]]
    label_box = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78)
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=160)
    ax.scatter(
        m2["r_reliability"],
        m2["total_drift_norm"],
        c=m2["delta"],
        cmap="YlOrRd",
        s=76,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    x_vals = m2["r_reliability"].to_numpy(dtype=float)
    y_vals = m2["total_drift_norm"].to_numpy(dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if mask.sum() >= 2 and np.std(x_vals[mask]) > 0:
        coef = np.polyfit(x_vals[mask], y_vals[mask], deg=1)
        x_grid = np.linspace(float(np.min(x_vals[mask])), float(np.max(x_vals[mask])), 80)
        ax.plot(x_grid, coef[0] * x_grid + coef[1], color="#636363", linewidth=1.5, linestyle="--", zorder=2)
    for i, row in m2.iterrows():
        ax.annotate(
            labels[i],
            (row["r_reliability"], row["total_drift_norm"]),
            fontsize=7.8,
            xytext=(5, 4),
            textcoords="offset points",
            bbox=label_box,
        )
    _style(
        ax,
        f"Appendix: M2 intensity and realized drift show a moderate association (r={corr:.2f})",
        r"Personalization intensity $r_k$",
        "Parameter drift from M1 anchor",
    )
    fig.tight_layout()
    _save(fig, "appendix_m2_intensity_drift_moderate_association", outputs)


def appendix_old_new_mapping(raw_m2: pd.DataFrame, order: List[str], outputs: List[Path]) -> None:
    cols = {"r_old_gamma_1", "r_old_gamma_3", "r_reliability"}
    if not cols.issubset(raw_m2.columns):
        return
    df = raw_m2.groupby("client_name", as_index=False).agg(
        r_old_gamma_1=("r_old_gamma_1", "mean"),
        r_old_gamma_3=("r_old_gamma_3", "mean"),
        r_reliability=("r_reliability", "mean"),
    )
    df["client_name"] = pd.Categorical(df["client_name"], categories=order, ordered=True)
    df = df.sort_values("client_name")
    labels = [_abbr(x) for x in df["client_name"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=160)
    ax.plot(x, df["r_old_gamma_1"].to_numpy(dtype=float), marker="o", label=r"direct $\gamma=1$", color="#de2d26")
    ax.plot(x, df["r_old_gamma_3"].to_numpy(dtype=float), marker="o", label=r"direct $\gamma=3$", color="#fc9272")
    ax.plot(x, df["r_reliability"].to_numpy(dtype=float), marker="o", label="log-median", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Appendix: scale calibration prevents intensity saturation", "Client", "Personalization intensity")
    fig.tight_layout()
    _save(fig, "appendix_old_vs_new_m2_mapping", outputs)


def appendix_m3_score_heatmap(scores: pd.DataFrame, order: List[str], outputs: List[Path]) -> None:
    group_col = "group_name" if "group_name" in scores.columns else "group"
    score_col = "group_score" if "group_score" in scores.columns else "score"
    mat = scores.groupby(["client_name", group_col])[score_col].mean().unstack(group_col).reindex(order).fillna(0.0)
    mat = mat[[g for g in GROUP_ORDER if g in mat.columns]]
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=160)
    im = ax.imshow(mat.values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Appendix: M3 average gradient score heatmap", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), rotation=40, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbr(x) for x in order])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Average normalized gradient score", fontsize=9)
    fig.tight_layout()
    _save(fig, "appendix_m3_gradient_score_heatmap", outputs)


def appendix_random_mask(effects: pd.DataFrame, order: List[str], outputs: List[Path]) -> Optional[Dict[str, float]]:
    if METHOD_RANDOM not in set(effects["method"].astype(str)):
        return None
    full = effects[effects["method"] == METHOD_FULL]
    rand = effects[effects["method"] == METHOD_RANDOM]
    merged = full.merge(rand, on=["seed", "client_id", "client_name"], suffixes=("_dapfl", "_random"))
    by_client = merged.groupby("client_name", as_index=False).agg(
        auc_dapfl=("test_auc_dapfl", "mean"),
        auc_random=("test_auc_random", "mean"),
        ece_dapfl=("test_ece_dapfl", "mean"),
        ece_random=("test_ece_random", "mean"),
    )
    by_client["delta_auc"] = by_client["auc_dapfl"] - by_client["auc_random"]
    by_client["delta_ece"] = by_client["ece_random"] - by_client["ece_dapfl"]
    by_client["client_name"] = pd.Categorical(by_client["client_name"], categories=order, ordered=True)
    by_client = by_client.sort_values("client_name")

    per_seed_rows: List[Dict[str, Any]] = []
    for (method, seed), group in effects[effects["method"].isin([METHOD_FULL, METHOD_RANDOM])].groupby(["method", "seed"]):
        per_seed_rows.append({
            "method": method,
            "seed": int(seed),
            "macro_auc": float(group["test_auc"].mean()),
            "macro_f1": float(group["test_f1"].mean()),
            "macro_ece": float(group["test_ece"].mean()),
            "gini": _gini(group["test_auc"]),
        })
    per_seed = pd.DataFrame(per_seed_rows)

    labels = [_abbr(x) for x in by_client["client_name"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=160)
    axes[0].bar(x, by_client["delta_auc"].to_numpy(dtype=float), color="#3182bd", edgecolor="black", linewidth=0.4)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=40, ha="right")
    _style(axes[0], "A. AUC gain over random mask", "Client", r"$\Delta$ AUC")
    axes[1].bar(x, by_client["delta_ece"].to_numpy(dtype=float), color="#74c476", edgecolor="black", linewidth=0.4)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    _style(axes[1], "B. ECE gain over random mask", "Client", r"$ECE_{random}-ECE_{DA-PFL}$")
    metric_cols = ["macro_auc", "macro_f1", "macro_ece", "gini"]
    metric_labels = ["Macro AUC", "Macro F1", "Macro ECE", "Gini"]
    mx = np.arange(len(metric_cols))
    width = 0.36
    for i, method in enumerate([METHOD_FULL, METHOD_RANDOM]):
        group = per_seed[per_seed["method"] == method]
        axes[2].bar(
            mx + (i - 0.5) * width,
            [group[c].mean() for c in metric_cols],
            yerr=[group[c].std(ddof=1) for c in metric_cols],
            width=width,
            capsize=3,
            label="DA-PFL" if method == METHOD_FULL else "Random mask",
        )
    axes[2].set_xticks(mx)
    axes[2].set_xticklabels(metric_labels, rotation=25, ha="right")
    axes[2].legend(frameon=False, fontsize=8)
    _style(axes[2], "C. Aggregate comparison", "", "Metric value")
    fig.suptitle("Appendix: M3 versus random mask", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "appendix_m3_vs_random_mask", outputs)
    return {
        "macro_auc_delta": float(
            per_seed[per_seed["method"] == METHOD_FULL]["macro_auc"].mean()
            - per_seed[per_seed["method"] == METHOD_RANDOM]["macro_auc"].mean()
        )
    }


def main() -> None:
    global RESULTS_DIR, FIG_DIR
    parser = argparse.ArgumentParser(description="Generate final paper-ready mechanism figures.")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--fig-dir", type=str, default=str(FIG_DIR))
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir)
    FIG_DIR = Path(args.fig_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    effects = _read_effect_rows(RESULTS_DIR)
    m2_raw = _read_csv(RESULTS_DIR / "m2_mechanism_table.csv")
    m3_raw = _read_csv(RESULTS_DIR / "m3_selected_groups_table.csv")
    scores = _read_csv(RESULTS_DIR / "m3_group_scores_table.csv")

    _warn_missing(effects, ["seed", "method", "client_id", "client_name", "test_auc", "test_f1", "test_ece", "val_bce", "threshold"], "dapfl_component_ablation_per_client")
    _warn_missing(m2_raw, ["seed", "client_id", "client_name", "n_k_pos", "n_k_neg", "pi_k", "pi_ref", "delta_b", "delta", "sigma_delta", "q_reliability", "s_reliability", "gamma_s", "gamma_s_scaled", "r_reliability", "m_k", "e_k"], "m2_mechanism_table")
    _warn_missing(m3_raw, ["seed", "client_id", "client_name", "m_k", "e_k", "selected_groups", "active_group_count", "total_drift_norm", "val_bce_before_m3", "val_bce_after_m3", "test_auc_before_m3", "test_auc_after_m3", "test_f1_before_m3", "test_f1_after_m3", "test_ece_before_m3", "test_ece_after_m3"], "m3_selected_groups_table")
    _warn_missing(scores, ["seed", "client_id", "client_name", "group_name", "group_score", "selected", "normalized_drift"], "m3_group_scores_table")

    order = _client_order(m2_raw, effects)

    m1 = _m1_summary(effects, m2_raw, order)
    m2 = _m2_summary(m2_raw, m3_raw, order, scores)
    m3 = _m3_summary(m3_raw, scores, order)

    m1_path = RESULTS_DIR / "final_mechanism_m1_summary.csv"
    m2_path = RESULTS_DIR / "final_mechanism_m2_summary.csv"
    m3_path = RESULTS_DIR / "final_mechanism_m3_summary.csv"
    m1[[
        "client_id", "client_name", "delta_b", "ece_before_m1_mean", "ece_after_m1_mean",
        "ece_improvement", "mean_pred_before_m1", "mean_pred_after_m1", "mean_pred_shift",
    ]].to_csv(m1_path, index=False, encoding="utf-8-sig")
    m2[[
        "client_id", "client_name", "delta", "sigma_delta", "q_reliability", "s_reliability",
        "gamma_s", "r_reliability", "m_k", "e_k", "selected_score_mean",
        "effective_adaptation_pressure", "total_drift_norm", "delta_val_bce",
    ]].to_csv(m2_path, index=False, encoding="utf-8-sig")
    m3[[
        "client_id", "client_name", "m_k", "e_k", "selected_groups", "active_group_count",
        "mean_selected_non_head_rank", "mean_unselected_non_head_rank",
        "mean_selected_non_head_score", "mean_unselected_non_head_score",
    ]].to_csv(m3_path, index=False, encoding="utf-8-sig")
    outputs.extend([m1_path, m2_path, m3_path])

    m1_stats = fig1_m1(m1, outputs)
    m2_stats = fig2_m2(m2, outputs)
    m3_stats = fig3_m3(scores, order, outputs)
    controlled_stats = fig4_controlled(effects, m3_raw, order, outputs)

    appendix_client_effects(effects, order, outputs)
    appendix_m2_construction(m2, outputs)
    appendix_m2_intensity_drift(m2, outputs)
    appendix_m3_score_heatmap(scores, order, outputs)
    random_stats = appendix_random_mask(effects, order, outputs)

    print("\n[Final Mechanism Summary]")
    print("1. M1:")
    print(f"Mean ECE before M1: {m1_stats['mean_ece_before']:.4f}")
    print(f"Mean ECE after M1: {m1_stats['mean_ece_after']:.4f}")
    print(f"Mean ECE improvement: {m1_stats['mean_ece_improvement']:.4f}")
    print(f"Correlation(delta_b, mean prediction shift): {m1_stats['corr_delta_pred_shift']:.3f}")
    print("\n2. M2:")
    print(f"Correlation(effective pressure, drift_norm): {m2_stats['corr_pressure_drift']:.3f}")
    print(f"Correlation(r_k, drift_norm; appendix): {m2_stats['corr_intensity_drift']:.3f}")
    print(f"Correlation(drift_norm, delta_val_bce): {m2_stats['corr_drift_val_bce']:.3f}")
    print("\n3. M3:")
    print(f"Mean selected non-head rank: {m3_stats['mean_selected_rank']:.3f}")
    print(f"Mean unselected non-head rank: {m3_stats['mean_unselected_rank']:.3f}")
    print(f"Mean selected non-head score: {m3_stats['mean_selected_score']:.5f}")
    print(f"Mean unselected non-head score: {m3_stats['mean_unselected_score']:.5f}")
    print("\n4. Controlled personalization:")
    if controlled_stats is not None:
        print(f"Mean drift DA-PFL: {controlled_stats['mean_drift_dapfl']:.4f}")
        print(f"Mean drift Full FT: {controlled_stats['mean_drift_fullft']:.4f}")
        print(f"Mean ECE difference FullFT - DAPFL: {controlled_stats['mean_ece_difference']:.4f}")
    else:
        print("Full fine-tuning results not found; skipped.")
    if random_stats is not None:
        print(f"Appendix random-mask macro-AUC delta: {random_stats['macro_auc_delta']:.4f}")
    print("\n5. Generated files:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
