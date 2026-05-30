from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dapfl_pipeline import run_dapfl_stage2, set_seed  # noqa: E402
from models.main import (  # noqa: E402
    DEVICE,
    Ch4Config,
    _cfg_fixed_budget,
    _cfg_full,
    _compact_metrics,
    _train_stage1_fedavg_backbone,
)


SEEDS = [42, 43, 46, 47, 49]
CORE_METRICS = ["global_auc", "macro_auc", "macro_f1", "macro_ece", "gini"]


def _parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    if vals.size == 1:
        return {"mean": float(vals[0]), "std": 0.0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}


def _cfg_variant(base_cfg: Ch4Config, variant: str) -> Ch4Config:
    if variant == "fixed_medium_m4_e6":
        cfg = _cfg_fixed_budget(base_cfg, 4, 6)
        cfg.apc_candidate_selection = False
        return cfg
    if variant == "fixed_fullscope_m7_e5":
        cfg = _cfg_fixed_budget(base_cfg, 7, 5)
        cfg.apc_candidate_selection = False
        return cfg

    cfg = _cfg_full(base_cfg)
    cfg.apc_controller_type = "label_shift"
    cfg.apc_signal_mode = "reliability_prior"
    cfg.apc_mapping_mode = "log_median"
    cfg.apc_scope_mapping_mode = "floor"
    cfg.apc_candidate_selection = True
    cfg.K_pers_min = 1
    cfg.K_pers_max = 7
    cfg.E_pers_min = 3
    cfg.E_pers_max = 10

    if variant == "m2_current":
        cfg.apc_gamma_s_scale = 1.0
    elif variant == "m2_product":
        cfg.apc_mapping_mode = "need_reliability_product"
        cfg.apc_scope_mapping_mode = "floor"
        cfg.apc_gamma_s_scale = 1.0
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cfg


def _run_stage2(seed: int, variant: str, cfg: Ch4Config, bundle: Dict[str, Any]) -> Dict[str, Any]:
    offsets = {
        "fixed_medium_m4_e6": 501,
        "fixed_fullscope_m7_e5": 502,
        "m2_current": 503,
        "m2_product": 508,
    }
    set_seed(int(seed) * 1000 + offsets.get(variant, 500), deterministic=True)
    return run_dapfl_stage2(
        backbone_name=f"StageI-FedAvg-{variant}",
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


def _metric_row(seed: int, variant: str, result: Dict[str, Any]) -> Dict[str, Any]:
    compact = _compact_metrics(result)
    row = {"seed": int(seed), "variant": str(variant)}
    for metric in CORE_METRICS:
        row[metric] = _safe_float(compact.get(metric))
    debug = result.get("client_debug", {}) or {}
    m_vals = [_safe_float(dbg.get("m_k", dbg.get("K_pers"))) for dbg in debug.values()]
    e_vals = [_safe_float(dbg.get("e_k", dbg.get("E_pers"))) for dbg in debug.values()]
    drift_vals = [_safe_float(dbg.get("stage2_drift_norm", dbg.get("realized_drift"))) for dbg in debug.values()]
    val_before = [_safe_float(dbg.get("stage2_val_bce_before_m3")) for dbg in debug.values()]
    val_after = [_safe_float(dbg.get("stage2_val_bce_after_m3")) for dbg in debug.values()]
    row["mean_m_k"] = float(np.nanmean(m_vals)) if m_vals else float("nan")
    row["mean_e_k"] = float(np.nanmean(e_vals)) if e_vals else float("nan")
    row["mean_drift"] = float(np.nanmean(drift_vals)) if drift_vals else float("nan")
    row["mean_val_bce_before"] = float(np.nanmean(val_before)) if val_before else float("nan")
    row["mean_val_bce_after"] = float(np.nanmean(val_after)) if val_after else float("nan")
    row["mean_val_bce_improvement"] = row["mean_val_bce_before"] - row["mean_val_bce_after"]
    return row


def _client_rows(seed: int, variant: str, result: Dict[str, Any], client_ids: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = result.get("client_metrics", {}) or {}
    debug = result.get("client_debug", {}) or {}
    for client in sorted(set(metrics.keys()) | set(debug.keys())):
        met = metrics.get(client, {}) or {}
        dbg = debug.get(client, {}) or {}
        selected = dbg.get("selected_groups", [])
        if not isinstance(selected, list):
            selected = []
        rows.append({
            "seed": int(seed),
            "variant": str(variant),
            "client_id": int(client_ids.get(client, -1)),
            "client_name": str(client),
            "test_auc": _safe_float(met.get("auc")),
            "test_f1": _safe_float(met.get("f1")),
            "test_ece": _safe_float(met.get("ece")),
            "threshold": _safe_float(met.get("best_threshold", dbg.get("thr"))),
            "m_k": _safe_int(dbg.get("m_k", dbg.get("K_pers"))),
            "e_k": _safe_int(dbg.get("e_k", dbg.get("E_pers"))),
            "selected_groups": ";".join(str(x) for x in selected),
            "stage2_drift_norm": _safe_float(dbg.get("stage2_drift_norm", dbg.get("realized_drift"))),
            "val_bce_before_m3": _safe_float(dbg.get("stage2_val_bce_before_m3")),
            "val_bce_after_m3": _safe_float(dbg.get("stage2_val_bce_after_m3")),
            "apc_pi_k": _safe_float(dbg.get("apc_pi_k", dbg.get("pi_k_tilde"))),
            "apc_pi_g": _safe_float(dbg.get("apc_pi_ref", dbg.get("pi_g_tilde"))),
            "apc_delta_b": _safe_float(dbg.get("apc_delta_b", dbg.get("delta_b_k"))),
            "apc_delta": _safe_float(dbg.get("apc_delta")),
            "apc_sigma_delta": _safe_float(dbg.get("apc_sigma_delta")),
            "apc_q_reliability": _safe_float(dbg.get("apc_q_reliability")),
            "apc_s_reliability": _safe_float(dbg.get("apc_s_reliability")),
            "apc_u_k": _safe_float(dbg.get("apc_u_k", dbg.get("u_k"))),
            "apc_alpha_k": _safe_float(dbg.get("apc_alpha_k", dbg.get("alpha_k"))),
            "apc_r_final": _safe_float(dbg.get("apc_r_final", dbg.get("r_final"))),
            "apc_n_k": _safe_float(dbg.get("apc_n_k")),
            "apc_max_s_reliability": _safe_float(dbg.get("apc_max_s_reliability")),
            "apc_gamma_s": _safe_float(dbg.get("apc_gamma_s")),
            "apc_gamma_s_scaled": _safe_float(dbg.get("apc_gamma_s_scaled")),
            "apc_r_reliability": _safe_float(dbg.get("apc_r_reliability")),
        })
    return rows


def _product_mechanism_rows(client_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in client_rows:
        if str(row.get("variant")) != "m2_product":
            continue
        u = _safe_float(row.get("apc_u_k"))
        alpha = _safe_float(row.get("apc_alpha_k"))
        r_product = float(np.clip(u * alpha, 0.0, 1.0)) if np.isfinite(u) and np.isfinite(alpha) else float("nan")
        rows.append({
            "seed": int(row["seed"]),
            "client_id": int(row["client_id"]),
            "client_name": str(row["client_name"]),
            "n_k": _safe_float(row.get("apc_n_k")),
            "pi_k": _safe_float(row.get("apc_pi_k")),
            "pi_g": _safe_float(row.get("apc_pi_g")),
            "delta_b": _safe_float(row.get("apc_delta_b")),
            "delta": _safe_float(row.get("apc_delta")),
            "sigma_delta": _safe_float(row.get("apc_sigma_delta")),
            "q_reliability": _safe_float(row.get("apc_q_reliability")),
            "s_reliability": _safe_float(row.get("apc_s_reliability")),
            "u_k": u,
            "alpha_k": alpha,
            "r_k": r_product,
            "r_product": r_product,
            "r_selected": _safe_float(row.get("apc_r_final", row.get("apc_r_reliability"))),
            "m_k": _safe_int(row.get("m_k")),
            "e_k": _safe_int(row.get("e_k")),
            "test_auc": _safe_float(row.get("test_auc")),
            "test_f1": _safe_float(row.get("test_f1")),
            "test_ece": _safe_float(row.get("test_ece")),
        })
    return rows


def _plot_product_mechanism(product_rows: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    if not product_rows:
        return []
    import pandas as pd

    df = pd.DataFrame(product_rows)
    agg = df.groupby(["client_id", "client_name"], as_index=False).mean(numeric_only=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    a = agg.sort_values("r_product", ascending=False).reset_index(drop=True)
    labels = [str(x).replace("Cardiac Vascular Intensive Care Unit (CVICU)", "CVICU")
              .replace("Coronary Care Unit (CCU)", "CCU")
              .replace("Medical Intensive Care Unit (MICU)", "MICU")
              .replace("Medical/Surgical Intensive Care Unit (MICU/SICU)", "MICU/SICU")
              .replace("Neuro Intermediate", "Neuro Inter.")
              .replace("Neuro Stepdown", "Neuro Step.")
              .replace("Neuro Surgical Intensive Care Unit (Neuro SICU)", "Neuro SICU")
              .replace("Surgical Intensive Care Unit (SICU)", "SICU")
              .replace("Trauma SICU (TSICU)", "TSICU")
              for x in a["client_name"]]

    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=160)
    sc = ax.scatter(a["u_k"], a["e_k"], c=a["n_k"], cmap="viridis", s=95, edgecolor="white", linewidth=0.8)
    for _, row in a.iterrows():
        label = labels[int(a.index[a["client_name"] == row["client_name"]][0])]
        suppressed = float(row["alpha_k"]) < 0.9 and float(row["u_k"]) > 0.5
        ax.annotate(label + ("*" if suppressed else ""), (row["u_k"], row["e_k"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_title("M2 need signal and selected personalization depth", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"Adaptation need $u_k$")
    ax.set_ylabel(r"Depth $e_k$")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(r"Client train size $n_k$")
    ax.text(0.02, 0.04, "* alpha_k suppresses high-need client", transform=ax.transAxes, fontsize=8, color="#555")
    p = fig_dir / "apc_product_need_vs_depth.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    paths.extend([p, p.with_suffix(".pdf")])

    x = np.arange(len(a))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=160)
    ax.bar(x - width / 2, a["u_k"], width, label=r"Need $u_k$", color="#76a5c5", edgecolor="white")
    ax.bar(x + width / 2, a["r_product"], width, label=r"Product intensity $r_k=u_k\alpha_k$", color="#f28e2b", edgecolor="white")
    for idx, row in a.iterrows():
        ax.annotate(
            f"a={row['alpha_k']:.2f}\n{int(round(row['m_k']))}/{int(round(row['e_k']))}",
            (idx, max(row["u_k"], row["r_product"])),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7.0,
            rotation=90,
        )
    ax.set_title("Need-by-reliability APC allocation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Signal value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    ax.text(0.01, 0.94, "annotation = alpha_k and validation-selected scope/depth", transform=ax.transAxes, fontsize=8, color="#555", va="top")
    p = fig_dir / "apc_product_need_reliability_allocation.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    paths.extend([p, p.with_suffix(".pdf")])
    return paths


def _summarize(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    variants = list(dict.fromkeys(str(r["variant"]) for r in run_rows))
    for variant in variants:
        group = [r for r in run_rows if str(r["variant"]) == variant]
        row: Dict[str, Any] = {"variant": variant, "n_runs": int(len(group))}
        for metric in CORE_METRICS + [
            "mean_m_k",
            "mean_e_k",
            "mean_drift",
            "mean_val_bce_improvement",
        ]:
            stats = _mean_std(float(r.get(metric, float("nan"))) for r in group)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        # Validation-side selector only. Test metrics are reported, not used for
        # choosing the preferred mapping.
        row["validation_selector_score"] = row["mean_val_bce_improvement_mean"]
        rows.append(row)
    rows.sort(key=lambda r: float(r["validation_selector_score"]), reverse=True)
    return rows


def _print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n[M2 mapping pre-experiment summary]")
    for row in summary:
        print(
            f"- {row['variant']:22s} | "
            f"gAUC {row['global_auc_mean']:.4f} +/- {row['global_auc_std']:.4f} | "
            f"mAUC {row['macro_auc_mean']:.4f} +/- {row['macro_auc_std']:.4f} | "
            f"mF1 {row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f} | "
            f"mECE {row['macro_ece_mean']:.4f} +/- {row['macro_ece_std']:.4f} | "
            f"Gini {row['gini_mean']:.4f} +/- {row['gini_std']:.4f} | "
            f"Kmean {row['mean_m_k_mean']:.2f} | Emean {row['mean_e_k_mean']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M2 intensity-mapping pre-experiment.")
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in SEEDS))
    parser.add_argument("--variants", type=str, default="fixed_fullscope_m7_e5,m2_current,m2_product")
    parser.add_argument("--fed-rounds", type=int, default=40)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "m2_mapping_preexperiment")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_runs: List[Dict[str, Any]] = []
    all_clients: List[Dict[str, Any]] = []
    start = time.time()
    for si, seed in enumerate(seeds, start=1):
        print(f"\n[Seed {seed}] ({si}/{len(seeds)})")
        base_cfg = Ch4Config(seed=int(seed), fed_rounds=int(args.fed_rounds), local_epochs_per_round=int(args.local_epochs))
        bundle = _train_stage1_fedavg_backbone(int(seed), base_cfg)
        client_ids = {str(name): idx for idx, name in enumerate(bundle["client_names"])}
        for variant in variants:
            cfg = _cfg_variant(base_cfg, variant)
            print(f"  -> {variant}")
            result = _run_stage2(int(seed), variant, cfg, bundle)
            row = _metric_row(int(seed), variant, result)
            all_runs.append(row)
            all_clients.extend(_client_rows(int(seed), variant, result, client_ids))
            print(
                f"    - {variant:22s} | "
                f"gAUC={row['global_auc']:.4f} | mAUC={row['macro_auc']:.4f} | "
                f"mF1={row['macro_f1']:.4f} | mECE={row['macro_ece']:.4f} | "
                f"Gini={row['gini']:.4f} | Kmean={row['mean_m_k']:.2f} | Emean={row['mean_e_k']:.2f}"
            )

    summary = _summarize(all_runs)
    product_mechanism = _product_mechanism_rows(all_clients)
    _write_csv(args.output_dir / "m2_mapping_runs.csv", all_runs)
    _write_csv(args.output_dir / "m2_mapping_per_client.csv", all_clients)
    _write_csv(args.output_dir / "m2_mapping_summary.csv", summary)
    _write_csv(args.output_dir / "apc_need_reliability_product_mechanism.csv", product_mechanism)
    figure_paths = _plot_product_mechanism(product_mechanism, args.output_dir)
    _print_summary(summary)
    print("\n[Saved]")
    for path in [
        args.output_dir / "m2_mapping_summary.csv",
        args.output_dir / "m2_mapping_runs.csv",
        args.output_dir / "m2_mapping_per_client.csv",
        args.output_dir / "apc_need_reliability_product_mechanism.csv",
    ]:
        print(f"  - {path}")
    for path in figure_paths:
        print(f"  - {path}")
    print(f"[Done] elapsed_sec={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
