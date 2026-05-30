from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.main import (  # noqa: E402
    DEVICE,
    Ch4Config,
    _compact_metrics,
    _json_sanitize,
    _train_stage1_fedavg_backbone,
)
from models.dapfl_pipeline import run_dapfl_stage2, set_seed  # noqa: E402


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


def _m2_rows(result: Dict[str, Any], client_ids: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client, dbg in sorted((result.get("client_debug", {}) or {}).items()):
        u_k = _safe_float(dbg.get("apc_u_k", dbg.get("u_k")))
        alpha_k = _safe_float(dbg.get("apc_alpha_k", dbg.get("alpha_k")))
        r_formula = float(np.clip(u_k * alpha_k, 0.0, 1.0)) if np.isfinite(u_k) and np.isfinite(alpha_k) else float("nan")
        rows.append({
            "client_id": int(client_ids.get(str(client), -1)),
            "client_name": str(client),
            "n_k": _safe_int(dbg.get("apc_n_k", dbg.get("n_i"))),
            "n_k_pos": _safe_int(dbg.get("n_pos_i")),
            "n_k_neg": _safe_int(dbg.get("n_neg_i")),
            "pi_k": _safe_float(dbg.get("apc_pi_k", dbg.get("pi_k_tilde"))),
            "pi_g": _safe_float(dbg.get("apc_pi_ref", dbg.get("pi_g_tilde"))),
            "delta_b": _safe_float(dbg.get("apc_delta_b", dbg.get("delta_b_k"))),
            "delta": _safe_float(dbg.get("apc_delta", dbg.get("delta_k"))),
            "sigma_delta": _safe_float(dbg.get("apc_sigma_delta", dbg.get("sigma_delta_k"))),
            "q_reliability": _safe_float(dbg.get("apc_q_reliability", dbg.get("q_k"))),
            "eta_k": _safe_float(dbg.get("apc_s_reliability", dbg.get("s_k"))),
            "u_k": u_k,
            "alpha_k": alpha_k,
            "r_k": r_formula,
            "r_selected": _safe_float(dbg.get("apc_r_final", dbg.get("r_final", dbg.get("apc_r_reliability")))),
            "m_k": _safe_int(dbg.get("m_k", dbg.get("K_pers"))),
            "e_k": _safe_int(dbg.get("e_k", dbg.get("E_pers"))),
        })
    return rows


def _client_metric_rows(result: Dict[str, Any], client_ids: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client, metrics in sorted((result.get("client_metrics", {}) or {}).items()):
        rows.append({
            "client_id": int(client_ids.get(str(client), -1)),
            "client_name": str(client),
            "test_auc": _safe_float(metrics.get("auc")),
            "test_f1": _safe_float(metrics.get("f1")),
            "test_ece": _safe_float(metrics.get("ece")),
            "threshold": _safe_float(metrics.get("best_threshold")),
        })
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the full DA-PFL pipeline with final need-by-reliability M2/APC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fed-rounds", type=int, default=40)
    parser.add_argument("--local-epochs-per-round", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--personalization-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "full_dapfl_final_m2_seed42")
    parser.add_argument("--m2-csv", type=Path, default=ROOT / "results" / "apc_need_reliability_full_mechanism_raw.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = Ch4Config(
        seed=int(args.seed),
        fed_rounds=int(args.fed_rounds),
        local_epochs_per_round=int(args.local_epochs_per_round),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        lr=float(args.lr),
        personalization_lr=float(args.personalization_lr),
        weight_decay=float(args.weight_decay),
    )
    cfg.apc_signal_mode = "need_reliability_product"
    cfg.apc_mapping_mode = "need_reliability_product"
    cfg.apc_scope_mapping_mode = "floor"
    cfg.apc_candidate_selection = True
    cfg.use_prior_bias_calib = True
    cfg.freeze_bias_after_calib = True
    cfg.personalization_select_metric = "tradeoff"

    set_seed(int(args.seed), deterministic=True)
    print(f"[Full DA-PFL final M2] seed={args.seed} device={DEVICE}")
    print("[Stage I] FedAvg backbone")
    bundle = _train_stage1_fedavg_backbone(int(args.seed), cfg, record_history=False)
    client_ids = {str(c): idx for idx, c in enumerate(bundle["client_names"])}

    print("[Stage II] M1 + need_reliability_product M2/APC + M3")
    set_seed(int(args.seed) * 1000 + 700, deterministic=True)
    result = run_dapfl_stage2(
        backbone_name="StageI-FedAvg-Full-DA-PFL-final-M2",
        backbone_model=bundle["backbone_model"],
        client_loaders=bundle["client_loaders"],
        central=bundle["central"],
        client_names=bundle["client_names"],
        client_sizes=bundle["client_sizes"],
        input_dim=int(bundle["input_dim"]),
        cfg=cfg,
        device=DEVICE,
        qbar=bundle["stage1_diag"].get("q_bar", None),
    )

    metrics = _compact_metrics(result)
    m2_rows = _m2_rows(result, client_ids)
    client_rows = _client_metric_rows(result, client_ids)

    full_json = args.output_dir / "full_dapfl_final_m2_seed42.json"
    metrics_csv = args.output_dir / "full_dapfl_final_m2_metrics.csv"
    client_csv = args.output_dir / "full_dapfl_final_m2_per_client.csv"
    raw_m2_csv = args.output_dir / "m2_mechanism_table.csv"

    full_json.write_text(json.dumps(_json_sanitize(result), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(metrics_csv, [{"seed": int(args.seed), **metrics}])
    _write_csv(client_csv, client_rows)
    _write_csv(raw_m2_csv, m2_rows)
    _write_csv(args.m2_csv, m2_rows)

    print("[Metrics]")
    print(
        f"gAUC={metrics['global_auc']:.4f} | mAUC={metrics['macro_auc']:.4f} | "
        f"mF1={metrics['macro_f1']:.4f} | mECE={metrics['macro_ece']:.4f} | Gini={metrics['gini']:.4f}"
    )
    print("[Saved]")
    for path in [full_json, metrics_csv, client_csv, raw_m2_csv, args.m2_csv]:
        print(f"  - {path}")
    print(f"[Done] elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
