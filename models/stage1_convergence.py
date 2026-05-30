# -*- coding: utf-8 -*-
"""
Lightweight Stage-I convergence / round-sensitivity runner.

This script trains only the FedAvg backbone for a grid of communication-round
budgets. It is meant for appendix evidence that the Stage-I model reaches a
stable operating point around the chosen default.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.dapfl_pipeline import Ch4Config
    from models.main import (
        OUT_DIR,
        _safe_float,
        _stage1_history_rows,
        _stage1_history_summary_rows,
        _train_stage1_fedavg_backbone,
        _write_paper_csv,
    )
except ImportError:
    from dapfl_pipeline import Ch4Config  # type: ignore
    from main import (  # type: ignore
        OUT_DIR,
        _safe_float,
        _stage1_history_rows,
        _stage1_history_summary_rows,
        _train_stage1_fedavg_backbone,
        _write_paper_csv,
    )


def _parse_int_grid(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-I FedAvg convergence sensitivity.")
    parser.add_argument("--rounds", type=str, default="20,30,40,60")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-epochs-per-round", type=int, default=2)
    parser.add_argument("--history-every", type=int, default=1)
    parser.add_argument("--default-round", type=int, default=30)
    parser.add_argument("--tail-window", type=int, default=10)
    parser.add_argument("--output-prefix", type=str, default="paper_stage1_round_sensitivity")
    return parser.parse_args()


def _mean_std(vals: List[float]) -> Dict[str, Any]:
    clean = [float(v) for v in vals if np.isfinite(float(v))]
    if not clean:
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
    }


def _maybe_plot_round_sensitivity(summary_rows: List[Dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Warn] matplotlib is unavailable; skip round-sensitivity figure.")
        return

    if not summary_rows:
        return

    xs = np.asarray([float(r["fed_rounds"]) for r in summary_rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    specs = [
        ("val_loss", "Validation loss", "lower is better"),
        ("macro_auc", "Macro AUC", "higher is better"),
        ("macro_f1", "Macro F1", "higher is better"),
        ("macro_ece", "Macro ECE", "lower is better"),
    ]
    for ax, (metric, title, subtitle) in zip(axes.ravel(), specs):
        mean = np.asarray([float(r.get(f"{metric}_mean") or np.nan) for r in summary_rows], dtype=float)
        std = np.asarray([float(r.get(f"{metric}_std") or 0.0) for r in summary_rows], dtype=float)
        ax.plot(xs, mean, marker="o", linewidth=1.8, color="#4C78A8")
        ax.fill_between(xs, mean - std, mean + std, color="#4C78A8", alpha=0.16, linewidth=0)
        ax.set_title(f"{title} ({subtitle})", fontsize=11, fontweight="bold")
        ax.set_xlabel("FedAvg communication rounds")
        ax.grid(alpha=0.24)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _history_for_longest_run(history_summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_rounds = [
        int(r.get("fed_rounds"))
        for r in history_summary_rows
        if r.get("fed_rounds") is not None
    ]
    if not valid_rounds:
        return []
    max_rounds = int(max(valid_rounds))
    rows = [
        r for r in history_summary_rows
        if int(r.get("fed_rounds", -1)) == max_rounds
    ]
    return sorted(rows, key=lambda r: int(r.get("round", 0)))


def _maybe_plot_training_convergence(
    history_summary_rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    default_round: int = 30,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Warn] matplotlib is unavailable; skip training-convergence figure.")
        return

    rows = _history_for_longest_run(history_summary_rows)
    if not rows:
        return

    xs = np.asarray([float(r["round"]) for r in rows], dtype=float)
    max_rounds = int(max(xs)) if xs.size else 0
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    specs = [
        ("train_loss", "Training loss", "lower is better"),
        ("val_loss", "Validation loss", "lower is better"),
        ("macro_auc", "Validation macro AUC", "higher is better"),
        ("macro_f1", "Validation macro F1", "higher is better"),
    ]

    for ax, (metric, title, subtitle) in zip(axes.ravel(), specs):
        mean = np.asarray([float(r.get(f"{metric}_mean") or np.nan) for r in rows], dtype=float)
        std = np.asarray([float(r.get(f"{metric}_std") or 0.0) for r in rows], dtype=float)
        ax.plot(xs, mean, linewidth=1.8, color="#2F6B9A")
        ax.fill_between(xs, mean - std, mean + std, color="#2F6B9A", alpha=0.14, linewidth=0)
        if 1 <= int(default_round) <= max_rounds:
            ax.axvline(int(default_round), color="#D95F02", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(
                int(default_round),
                0.98,
                f"default R={int(default_round)}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="#D95F02",
            )
        ax.set_title(f"{title} ({subtitle})", fontsize=11, fontweight="bold")
        ax.set_xlabel("FedAvg communication round")
        ax.grid(alpha=0.24)

    fig.suptitle(
        f"Stage-I FedAvg round-by-round convergence (trained to R={max_rounds})",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _nearest_round(rows: List[Dict[str, Any]], target_round: int) -> Dict[str, Any]:
    return min(rows, key=lambda r: abs(int(r.get("round", 0)) - int(target_round)))


def _convergence_diagnostics(
    history_summary_rows: List[Dict[str, Any]],
    *,
    default_round: int = 30,
    tail_window: int = 10,
) -> List[Dict[str, Any]]:
    rows = _history_for_longest_run(history_summary_rows)
    if not rows:
        return []

    final_row = rows[-1]
    default_row = _nearest_round(rows, int(default_round))
    tail = rows[-int(max(2, tail_window)):]
    tail_x = np.asarray([float(r.get("round", 0)) for r in tail], dtype=float)

    specs = [
        ("train_loss", "min"),
        ("val_loss", "min"),
        ("macro_auc", "max"),
        ("macro_f1", "max"),
        ("macro_ece", "min"),
        ("gini", "min"),
    ]
    out: List[Dict[str, Any]] = []
    for metric, direction in specs:
        vals = [
            (
                int(r.get("round", 0)),
                _safe_float(r.get(f"{metric}_mean")),
            )
            for r in rows
        ]
        vals = [(rnd, val) for rnd, val in vals if val is not None]
        if not vals:
            continue
        if direction == "min":
            best_round, best_val = min(vals, key=lambda x: float(x[1]))
        else:
            best_round, best_val = max(vals, key=lambda x: float(x[1]))

        final_val = _safe_float(final_row.get(f"{metric}_mean"))
        default_val = _safe_float(default_row.get(f"{metric}_mean"))
        tail_vals = np.asarray([
            float(v)
            for v in (_safe_float(r.get(f"{metric}_mean")) for r in tail)
            if v is not None
        ], dtype=float)
        if tail_vals.size >= 2 and tail_x.size == tail_vals.size:
            tail_slope = float(np.polyfit(tail_x, tail_vals, deg=1)[0])
            tail_delta = float(tail_vals[-1] - tail_vals[0])
        else:
            tail_slope = None
            tail_delta = None

        out.append(
            {
                "fed_rounds_trained": int(final_row.get("fed_rounds", final_row.get("round", 0))),
                "metric": metric,
                "direction": direction,
                "default_round": int(default_row.get("round", default_round)),
                "default_mean": default_val,
                "final_round": int(final_row.get("round", 0)),
                "final_mean": final_val,
                "best_round": int(best_round),
                "best_mean": float(best_val),
                "final_minus_default": None if final_val is None or default_val is None else float(final_val - default_val),
                "final_minus_best": None if final_val is None else float(final_val - float(best_val)),
                "tail_window": int(max(2, tail_window)),
                "tail_delta": tail_delta,
                "tail_slope_per_round": tail_slope,
            }
        )
    return out


def main() -> None:
    args = _parse_args()
    rounds_grid = _parse_int_grid(args.rounds)
    all_history_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    print(
        f"[Config] rounds={rounds_grid} | repeats={int(args.repeats)} | "
        f"local_epochs={int(args.local_epochs_per_round)}"
    )

    for rounds in rounds_grid:
        for rep in range(int(args.repeats)):
            seed = int(args.seed) + rep
            print(f"\n[Stage-I] rounds={int(rounds)} | repeat={rep + 1}/{int(args.repeats)} | seed={seed}")
            cfg = Ch4Config(
                seed=seed,
                fed_rounds=int(rounds),
                local_epochs_per_round=int(args.local_epochs_per_round),
            )
            bundle = _train_stage1_fedavg_backbone(
                seed,
                cfg,
                record_history=True,
                history_every=int(max(1, args.history_every)),
            )
            hist_rows = _stage1_history_rows(
                repeat=rep,
                seed=seed,
                history=bundle.get("stage1_history", []),
            )
            for row in hist_rows:
                row["fed_rounds"] = int(rounds)
            all_history_rows.extend(hist_rows)

            last = hist_rows[-1] if hist_rows else {}
            final_rows.append(
                {
                    "fed_rounds": int(rounds),
                    "repeat": int(rep),
                    "seed": int(seed),
                    "train_loss": _safe_float(last.get("train_loss")),
                    "val_loss": _safe_float(last.get("val_loss")),
                    "global_auc": _safe_float(last.get("global_auc")),
                    "macro_auc": _safe_float(last.get("macro_auc")),
                    "macro_f1": _safe_float(last.get("macro_f1")),
                    "macro_ece": _safe_float(last.get("macro_ece")),
                    "gini": _safe_float(last.get("gini")),
                }
            )

    summary_rows: List[Dict[str, Any]] = []
    metrics = ["train_loss", "val_loss", "global_auc", "macro_auc", "macro_f1", "macro_ece", "gini"]
    for rounds in rounds_grid:
        group = [r for r in final_rows if int(r.get("fed_rounds", -1)) == int(rounds)]
        row: Dict[str, Any] = {"fed_rounds": int(rounds), "n_repeats": int(len(group))}
        for metric in metrics:
            stats = _mean_std([
                float(v)
                for v in (_safe_float(g.get(metric)) for g in group)
                if v is not None
            ])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        summary_rows.append(row)

    history_summary_rows = []
    for rounds in rounds_grid:
        subset = [r for r in all_history_rows if int(r.get("fed_rounds", -1)) == int(rounds)]
        for row in _stage1_history_summary_rows(subset):
            row["fed_rounds"] = int(rounds)
            history_summary_rows.append(row)
    diagnostic_rows = _convergence_diagnostics(
        history_summary_rows,
        default_round=int(args.default_round),
        tail_window=int(args.tail_window),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)
    final_csv = OUT_DIR / f"{prefix}.csv"
    history_csv = OUT_DIR / f"{prefix}_history.csv"
    history_summary_csv = OUT_DIR / f"{prefix}_history_summary.csv"
    diagnostics_csv = OUT_DIR / f"{prefix}_convergence_diagnostics.csv"
    fig_path = OUT_DIR / f"{prefix}.png"
    convergence_fig_path = OUT_DIR / f"{prefix}_convergence_curve.png"

    _write_paper_csv(final_csv, summary_rows)
    _write_paper_csv(history_csv, all_history_rows)
    _write_paper_csv(history_summary_csv, history_summary_rows)
    _write_paper_csv(diagnostics_csv, diagnostic_rows)
    _maybe_plot_round_sensitivity(summary_rows, fig_path)
    _maybe_plot_training_convergence(
        history_summary_rows,
        convergence_fig_path,
        default_round=int(args.default_round),
    )

    print("\n[Done] wall_time_sec={:.1f}".format(float(time.perf_counter() - t0)))
    print(f"[Done] Round sensitivity CSV: {final_csv}")
    print(f"[Done] Round history CSV: {history_csv}")
    print(f"[Done] Round history summary CSV: {history_summary_csv}")
    print(f"[Done] Convergence diagnostics CSV: {diagnostics_csv}")
    print(f"[Done] Round sensitivity figure: {fig_path}")
    print(f"[Done] Training convergence figure: {convergence_fig_path}")


if __name__ == "__main__":
    main()
