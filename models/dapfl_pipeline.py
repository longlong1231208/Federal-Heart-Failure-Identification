# models/ch4_methods.py
from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from datasets.fl_dataset import FederatedPKLDataset
from models.gru import GRUModel
from models.ch4_m1 import (
    M1Config,
    apply_prior_bias_calibration,
)
from models.ch4_m2 import (
    APCParams,
    compute_gamma_s_from_counts,
    compute_need_reliability_product_outputs,
    get_apc_candidates_from_output,
    get_apc_candidates,
    get_apc_output,
)
from models.ch4_m3 import (
    M3Config,
    build_param_groups,
    clip_gradients_per_selected_group,
    get_group_order,
    infer_head_group_names,
    run_m3_selection,
    set_trainable_by_selected_groups,
)


# ============================================================
# Seed
# ============================================================
def set_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


# ============================================================
# Config (paper-only)
# ============================================================
@dataclass
class Ch4Config:
    seed: int = 42
    batch_size: int = 64

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3

    # -------- Stage I backbone --------
    fed_rounds: int = 40
    local_epochs_per_round: int = 2
    lr: float = 1e-3
    lr_decay: float = 0.98
    weight_decay: float = 1e-4
    clip_grad_fedavg: Optional[float] = None

    # -------- Stage II-B masked ERM --------
    personalization_lr: float = 1e-4
    group_clip_norm: Optional[float] = 5.0
    # -------- M1 --------
    prior_smoothing_alpha: float = 0.5
    prior_smoothing_beta: float = 0.5
    use_prior_bias_calib: bool = True
    freeze_bias_after_calib: bool = True

    # -------- M2/APC: reliability-adjusted prior mismatch --------
    large_threshold: Optional[int] = None

    apc_signal_mode: str = "reliability_prior"
    apc_controller_type: str = "label_shift"
    apc_eps_sm: float = 0.5
    apc_eps_num: float = 1e-8
    apc_eps_gamma: float = 1e-8
    apc_mapping_mode: str = "need_reliability_product"
    apc_scope_mapping_mode: str = "floor"
    apc_direct_gamma: float = 1.96
    apc_gamma_s_scale: float = 1.0
    apc_G_groups: int = 7
    apc_candidate_selection: bool = True

    E_pers_min: int = 3
    E_pers_max: int = 10

    K_pers_min: int = 1
    K_pers_max: int = 7

    # -------- M3 --------
    grad_score_batches: int = 10
    grad_score_batches_large: Optional[int] = 20
    m3_score_mode: str = "dimnorm_l2"
    m3_selection_strategy: str = "gradient"
    m3_eps: float = 1e-8
    personalization_select_metric: str = "tradeoff"

    # -------- Eval --------
    thresh_grid: Tuple[float, float, int] = (0.1, 0.6, 51)
    fallback_threshold_when_no_val: float = 0.5
    early_stop_patience: int = 2

    def lr_at(self, r: int) -> float:
        return float(self.lr) * (float(self.lr_decay) ** int(r))


# ============================================================
# Data
# ============================================================
def _loader_to_numpy_y(loader) -> np.ndarray:
    ys = []
    if loader is None:
        return np.asarray([])
    for _, y in loader:
        ys.append(y.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ys) if ys else np.asarray([])


def build_ch4_loaders(
        pkl_path: str,
        cfg: Ch4Config,
        seed: int,
        val_frac: float = 0.1,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[str], List[int], int]:
    ds = FederatedPKLDataset(pkl_path, val_frac=val_frac, seed=seed)
    client_names = ds.get_clients()

    client_loaders: Dict[str, Dict[str, Any]] = {}
    client_sizes: List[int] = []
    all_train_used_y = []

    for cname in client_names:
        tr_loader, va_loader, te_loader = ds.get_dataloaders(cname, batch_size=cfg.batch_size)

        y_train_used_raw = _loader_to_numpy_y(tr_loader)
        if y_train_used_raw.size > 0:
            all_train_used_y.append(torch.tensor(y_train_used_raw, dtype=torch.float32))

        client_loaders[cname] = {
            "train": tr_loader,
            "val": va_loader,
            "test": te_loader,
            "y_train_used_raw": y_train_used_raw,
        }
        client_sizes.append(len(tr_loader.dataset))

    y_global_used = torch.cat(all_train_used_y).detach().cpu().numpy() if all_train_used_y else np.asarray([])
    central = {"y_train_used_raw": y_global_used}
    return client_loaders, central, client_names, client_sizes, ds.input_dim


# ============================================================
# Numeric / eval helpers (Added ECE & Brier)
# ============================================================
def _sigmoid_stable(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


# <--- 补充: 计算 Expected Calibration Error (ECE)
def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) for binary classification."""
    if y_true.size == 0:
        return 0.0
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i + 1]
        # First bin includes 0.0
        if i == 0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)


# <--- 补充: 手动计算 Brier Score (避免 sklearn 在单一类别时报错)
def calculate_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    if loader is None:
        return np.asarray([]), np.asarray([])
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if hasattr(model, "forward_logits") and callable(getattr(model, "forward_logits")):
            logits_t = model.forward_logits(x)
        else:
            logits_t = model(x)
        logits = logits_t.detach().cpu().numpy().reshape(-1)
        yy = y.detach().cpu().numpy().reshape(-1)
        logits_all.append(logits)
        y_all.append(yy)
    if not logits_all:
        return np.asarray([]), np.asarray([])
    return np.concatenate(logits_all), np.concatenate(y_all)


def _search_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray, grid: Tuple[float, float, int]) -> float:
    lo, hi, n = grid
    best_f1 = -1.0
    best_t = 0.5
    for t in np.linspace(lo, hi, int(n)):
        pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def evaluate_from_logits(
        y_true: np.ndarray,
        logits: np.ndarray,
        threshold: Optional[float],
        grid: Tuple[float, float, int],
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    logits = np.asarray(logits, dtype=float).reshape(-1)

    if y_true.size == 0 or logits.size == 0:
        return {
            "auc": 0.5, "f1": 0.0, "recall": 0.0, "precision": 0.0,
            "brier": 0.0, "ece": 0.0,  # <--- 补充
            "best_threshold": 0.5,
            "y_true": np.asarray([], dtype=int),
            "y_prob": np.asarray([], dtype=float),
        }

    prob = _sigmoid_stable(logits)
    brier = calculate_brier(y_true, prob)  # <--- 补充
    ece = calculate_ece(y_true, prob)  # <--- 补充

    if len(np.unique(y_true)) < 2:
        t = 0.5 if threshold is None else float(threshold)
        pred = (prob > t).astype(int)
        return {
            "auc": 0.5,
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "brier": brier,  # <--- 补充
            "ece": ece,  # <--- 补充
            "best_threshold": float(t),
            "y_true": y_true,
            "y_prob": prob,
        }

    auc = float(roc_auc_score(y_true, prob))

    t = _search_best_threshold_by_f1(y_true, prob, grid) if threshold is None else float(threshold)
    pred = (prob > t).astype(int)

    return {
        "auc": auc,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "brier": brier,  # <--- 补充
        "ece": ece,  # <--- 补充
        "best_threshold": float(t),
        "y_true": y_true,
        "y_prob": prob,
    }


def _choose_threshold_from_val(model: nn.Module, val_loader, device, cfg: Ch4Config) -> float:
    if val_loader is None:
        return float(cfg.fallback_threshold_when_no_val)
    val_logits, val_y = collect_logits_labels(model, val_loader, device)
    if val_y.size == 0 or val_logits.size == 0:
        return float(cfg.fallback_threshold_when_no_val)
    val_res = evaluate_from_logits(val_y, val_logits, threshold=None, grid=cfg.thresh_grid)
    return float(val_res["best_threshold"])


def _eval_on_test_with_threshold(model: nn.Module, test_loader, device, cfg: Ch4Config, thr: float) -> Dict[str, Any]:
    test_logits, test_y = collect_logits_labels(model, test_loader, device)
    return evaluate_from_logits(test_y, test_logits, threshold=float(thr), grid=cfg.thresh_grid)


def calculate_gini(scores: List[float]) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    denom = (s.sum() + 1e-12)
    return float(((2 * idx - n - 1) * s).sum() / (n * denom))


def _aggregate_client_metrics(client_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    ys = [m["y_true"] for m in client_metrics.values() if m.get("y_true", np.asarray([])).size > 0]
    ps = [m["y_prob"] for m in client_metrics.values() if m.get("y_prob", np.asarray([])).size > 0]

    if ys and ps:
        all_y = np.concatenate(ys)
        all_p = np.concatenate(ps)
        if all_y.size > 0 and len(np.unique(all_y)) > 1:
            micro_auc = float(roc_auc_score(all_y, all_p))
        else:
            micro_auc = 0.5

        # <--- 补充: Global Brier & ECE
        global_brier = calculate_brier(all_y, all_p)
        global_ece = calculate_ece(all_y, all_p)
    else:
        micro_auc = 0.5
        global_brier, global_ece = 0.0, 0.0

    client_aucs = [float(m["auc"]) for m in client_metrics.values()]
    client_f1s = [float(m["f1"]) for m in client_metrics.values()]
    client_recalls = [float(m["recall"]) for m in client_metrics.values()]
    client_precs = [float(m["precision"]) for m in client_metrics.values()]

    # <--- 补充: Macro Brier & ECE
    client_briers = [float(m["brier"]) for m in client_metrics.values()]
    client_eces = [float(m["ece"]) for m in client_metrics.values()]

    valid_aucs = []
    for m in client_metrics.values():
        y = np.asarray(m.get("y_true", np.asarray([]))).reshape(-1)
        if y.size > 0 and len(np.unique(y.astype(int))) > 1:
            valid_aucs.append(float(m.get("auc", 0.5)))

    return {
        "global_auc": float(micro_auc),
        "global_brier": float(global_brier),  # <--- 补充
        "global_ece": float(global_ece),  # <--- 补充
        "macro_auc": float(np.mean(client_aucs)) if client_aucs else float("nan"),
        "macro_f1": float(np.mean(client_f1s)) if client_f1s else float("nan"),
        "macro_recall": float(np.mean(client_recalls)) if client_recalls else float("nan"),
        "macro_precision": float(np.mean(client_precs)) if client_precs else float("nan"),
        "macro_brier": float(np.mean(client_briers)) if client_briers else float("nan"),  # <--- 补充
        "macro_ece": float(np.mean(client_eces)) if client_eces else float("nan"),  # <--- 补充
        "gini": float(calculate_gini(valid_aucs)) if valid_aucs else 0.0,
    }


# ============================================================
# Priors
# ============================================================
def get_priors(
        client_loaders,
        client_names: List[str],
        central: Dict[str, Any],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    y_g = np.asarray(central.get("y_train_used_raw", np.asarray([]))).reshape(-1)
    n_g = int(y_g.size)
    n_pos_g = int(np.sum(y_g)) if n_g > 0 else 0
    p_g_raw = float(np.mean(y_g)) if n_g > 0 else 0.5

    per = {}
    for n in client_names:
        y_i = np.asarray(client_loaders[n].get("y_train_used_raw", np.asarray([]))).reshape(-1)
        n_i = int(y_i.size)
        n_pos_i = int(np.sum(y_i)) if n_i > 0 else 0
        p_i_raw = float(np.mean(y_i)) if n_i > 0 else p_g_raw
        per[n] = {
            "p_i_raw": float(p_i_raw),
            "n_i": float(n_i),
            "n_pos_i": float(n_pos_i),
        }

    per["_global"] = {
        "p_g_raw": float(p_g_raw),
        "n_g": float(n_g),
        "n_pos_g": float(n_pos_g),
    }
    return float(p_g_raw), per


def compute_qbar_aligned_global_prior(
        client_names: List[str],
        priors_per_client: Dict[str, Dict[str, float]],
        qbar: Optional[Dict[str, float]],
) -> float:
    """
    Paper-aligned Stage II-A baseline:
      p_g^(qbar) = sum_k qbar_k * p_k
    If qbar is unavailable, fall back to pooled prior.
    """
    if not qbar:
        return float(priors_per_client["_global"]["p_g_raw"])

    total = 0.0
    wsum = 0.0
    for name in client_names:
        wk = float(qbar.get(name, 0.0))
        pk = float(priors_per_client.get(name, {}).get("p_i_raw", priors_per_client["_global"]["p_g_raw"]))
        total += wk * pk
        wsum += wk

    if wsum <= 0.0:
        return float(priors_per_client["_global"]["p_g_raw"])
    return float(total / wsum)


# ============================================================
# Backbone (FedAvg only; keep Stage I separate if desired)
# ============================================================
def _trainable_params(model: nn.Module) -> List[torch.Tensor]:
    return [p for p in model.parameters() if bool(p.requires_grad)]


def train_epoch(
        model: nn.Module,
        loader,
        optimizer,
        criterion,
        device,
        clip_grad: Optional[float] = None,
):
    model.train()
    params_to_clip = _trainable_params(model)
    loss_sum = 0.0
    n_seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x).view(-1)
        loss = criterion(logits, y)
        loss.backward()

        if clip_grad is not None and params_to_clip:
            torch.nn.utils.clip_grad_norm_(params_to_clip, clip_grad)

        optimizer.step()

        bs = int(y.numel())
        loss_sum += float(loss.detach().cpu().item()) * float(bs)
        n_seen += bs

    return float(loss_sum / max(1, n_seen))


def _split_bce_loss(model: nn.Module, loader, device) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    loss_sum = 0.0
    n_seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1)
            logits = model(x).view(-1)
            loss_sum += float(criterion(logits, y).detach().cpu().item())
            n_seen += int(y.numel())
    return float(loss_sum / max(1, n_seen)) if n_seen > 0 else float("nan")


def _evaluate_clients_on_split(
        model: nn.Module,
        client_loaders,
        client_names: List[str],
        cfg: Ch4Config,
        device,
        *,
        split: str = "val",
) -> Dict[str, Any]:
    client_metrics: Dict[str, Dict[str, Any]] = {}
    losses: List[float] = []
    weights: List[float] = []

    for name in client_names:
        loader = client_loaders[name].get(split)
        logits, y = collect_logits_labels(model, loader, device)
        client_metrics[str(name)] = evaluate_from_logits(
            y,
            logits,
            threshold=None,
            grid=cfg.thresh_grid,
        )

        loss = _split_bce_loss(model, loader, device)
        if np.isfinite(float(loss)):
            losses.append(float(loss))
            weights.append(float(max(1, y.size)))

    agg = _aggregate_client_metrics(client_metrics)
    if losses:
        agg[f"{split}_loss"] = float(np.average(np.asarray(losses), weights=np.asarray(weights)))
    else:
        agg[f"{split}_loss"] = float("nan")
    return agg


def run_fedavg_backbone(
        client_loaders,
        client_names: List[str],
        client_sizes: List[int],
        input_dim: int,
        cfg: Ch4Config,
        device,
        *,
        return_history: bool = False,
        history_split: str = "val",
        history_every: int = 1,
) -> Any:
    global_model = GRUModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device)
    global_w = global_model.state_dict()
    history: List[Dict[str, Any]] = []
    history_every = int(max(1, history_every))

    for r in range(cfg.fed_rounds):
        lr = cfg.lr_at(r)
        local_ws = []
        local_losses: List[float] = []
        local_loss_weights: List[float] = []

        for i, name in enumerate(client_names):
            local = GRUModel(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device)
            local.load_state_dict(global_w, strict=True)

            opt = optim.Adam(local.parameters(), lr=lr, weight_decay=cfg.weight_decay)
            crit = nn.BCEWithLogitsLoss()

            epoch_losses: List[float] = []
            for _ in range(cfg.local_epochs_per_round):
                epoch_losses.append(
                    train_epoch(
                        local,
                        client_loaders[name]["train"],
                        opt,
                        crit,
                        device,
                        clip_grad=cfg.clip_grad_fedavg,
                    )
                )
            finite_losses = [float(x) for x in epoch_losses if np.isfinite(float(x))]
            if finite_losses:
                local_losses.append(float(np.mean(finite_losses)))
                local_loss_weights.append(float(max(1, int(client_sizes[i]))))

            local_ws.append(local.state_dict())

        total = float(sum(client_sizes)) if sum(client_sizes) > 0 else 1.0
        new_w = copy.deepcopy(global_w)
        for k in new_w.keys():
            acc = torch.zeros_like(new_w[k])
            for i, w in enumerate(local_ws):
                acc += w[k] * (client_sizes[i] / total)
            new_w[k] = acc

        global_w = new_w
        global_model.load_state_dict(global_w, strict=True)

        if return_history and (((r + 1) % history_every == 0) or (r + 1 == int(cfg.fed_rounds))):
            diag = _evaluate_clients_on_split(
                global_model,
                client_loaders,
                client_names,
                cfg,
                device,
                split=str(history_split),
            )
            if local_losses:
                train_loss = float(np.average(np.asarray(local_losses), weights=np.asarray(local_loss_weights)))
            else:
                train_loss = float("nan")
            history.append(
                {
                    "round": int(r + 1),
                    "lr": float(lr),
                    "train_loss": train_loss,
                    **diag,
                }
            )

    if return_history:
        return global_model, history
    return global_model


# ============================================================
# APC / M3 config builders
# ============================================================
def _build_apc_params_from_cfg(cfg: Ch4Config) -> APCParams:
    return APCParams(
        eps_sm=float(cfg.apc_eps_sm),
        eps_num=float(cfg.apc_eps_num),
        eps_gamma=float(getattr(cfg, "apc_eps_gamma", cfg.apc_eps_num)),
        mapping_mode=str(getattr(cfg, "apc_mapping_mode", "log_median")),
        scope_mapping_mode=str(getattr(cfg, "apc_scope_mapping_mode", "floor")),
        direct_gamma=float(getattr(cfg, "apc_direct_gamma", 1.96)),
        gamma_s_scale=float(cfg.apc_gamma_s_scale),
        G_groups=int(cfg.apc_G_groups),

        K_pers_min=int(cfg.K_pers_min),
        K_pers_max=int(cfg.K_pers_max),
        E_pers_min=int(cfg.E_pers_min),
        E_pers_max=int(cfg.E_pers_max),
    )


def _build_m3_cfg_from_cfg(cfg: Ch4Config) -> M3Config:
    return M3Config(
        enabled=True,
        score_mode=str(cfg.m3_score_mode),
        grad_score_batches=int(cfg.grad_score_batches),
        grad_score_batches_large=int(
            cfg.grad_score_batches_large) if cfg.grad_score_batches_large is not None else None,
        eps=float(cfg.m3_eps),
        force_keep_head=True,
        prefer_forward_logits=True,
        enable_groupwise_clip_helper=True,
        group_clip_norm=float(cfg.group_clip_norm) if cfg.group_clip_norm is not None else None,
    )


# ============================================================
# Stage II-B masked ERM
# ============================================================
def train_epoch_masked_stage2(
        model: nn.Module,
        loader,
        optimizer,
        criterion,
        device,
        *,
        groups: Dict[str, List[str]],
        selected_groups: List[str],
        group_clip_norm: Optional[float] = None,
) -> None:
    model.train()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(x).view(-1) if hasattr(model, "forward_logits") else model(x).view(-1)
        loss = criterion(logits, y)
        loss.backward()

        clip_gradients_per_selected_group(
            model=model,
            groups=groups,
            selected_groups=selected_groups,
            max_norm=group_clip_norm,
        )

        optimizer.step()


def _val_selection_score(model: nn.Module, val_loader, device, cfg: Ch4Config) -> float:
    logits, y = collect_logits_labels(model, val_loader, device)
    if y.size == 0 or logits.size == 0 or len(np.unique(y.astype(int))) < 2:
        return float("-inf")
    prob = _sigmoid_stable(logits)
    yy = y.astype(int)
    auc = float(roc_auc_score(yy, prob))
    thr = _search_best_threshold_by_f1(yy, prob, cfg.thresh_grid)
    pred = (prob > float(thr)).astype(int)
    f1 = float(f1_score(yy, pred, zero_division=0))
    ece = float(calculate_ece(yy, prob))

    metric = str(getattr(cfg, "personalization_select_metric", "tradeoff")).lower().strip()
    if metric == "auc":
        return auc
    if metric == "f1":
        return f1
    if metric == "tradeoff":
        return float(auc + f1 - ece)
    return float(auc + f1 - ece)



def _freeze_to_selected_groups(
        model: nn.Module,
        groups: Dict[str, List[str]],
        selected_groups: List[str],
        *,
        freeze_bias_after_calib: bool = False,
) -> List[str]:
    """
    Enforce true masked personalization:
      - only parameters in `selected_groups` remain trainable
      - optional calibrated classifier bias stays frozen

    This closes the gap between paper semantics (masked ERM) and the
    implementation, where gradient clipping alone is not sufficient to
    stop optimizer updates on unselected groups.
    """
    selected = [str(g) for g in selected_groups if str(g) in groups]
    set_trainable_by_selected_groups(model, groups, set(selected))

    if freeze_bias_after_calib and hasattr(model, "fc") and getattr(model.fc, "bias", None) is not None:
        try:
            model.fc.bias.requires_grad_(False)
        except Exception:
            pass

    trainable = [n for n, p in model.named_parameters() if bool(p.requires_grad)]
    return trainable


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    pp = float(np.clip(float(p), eps, 1.0 - eps))
    return float(np.log(pp / (1.0 - pp)))


def _smoothed_prior(pos: int, n: int, alpha: float, beta: float, fallback: float) -> float:
    nn = int(max(0, n))
    if nn <= 0:
        return float(np.clip(float(fallback), 1e-6, 1.0 - 1e-6))
    pp = int(max(0, min(int(pos), nn)))
    return float((float(pp) + float(alpha)) / (float(nn) + float(alpha) + float(beta)))


def _prior_shift_value(
        *,
        p_i_raw: float,
        p_g_raw: float,
        n_i: int,
        n_g: int,
        n_pos_i: int,
        n_pos_g: int,
        cfg: Ch4Config,
) -> float:
    pi = _smoothed_prior(
        int(n_pos_i),
        int(n_i),
        float(cfg.prior_smoothing_alpha),
        float(cfg.prior_smoothing_beta),
        float(p_i_raw),
    )
    pg = _smoothed_prior(
        int(n_pos_g),
        int(n_g),
        float(cfg.prior_smoothing_alpha),
        float(cfg.prior_smoothing_beta),
        float(p_g_raw),
    )
    return float(abs(_safe_logit(pi) - _safe_logit(pg)))


def _candidate_cost(candidate) -> float:
    return float(max(1, int(getattr(candidate, "K_pers", 1))) * max(0, int(getattr(candidate, "E_pers", 0))))


def _run_group_selection_for_stage2(
        *,
        model: nn.Module,
        loader,
        device,
        cfg: Ch4Config,
        m3_cfg: M3Config,
        scope_groups,
        k_select: int,
        client_size: int,
        large_threshold: int,
        groups: Dict[str, List[str]],
) -> Dict[str, object]:
    strategy = str(getattr(cfg, "m3_selection_strategy", "gradient")).strip().lower()
    if strategy not in {"random", "random_mask"}:
        return run_m3_selection(
            model=model,
            loader=loader,
            device=device,
            cfg=m3_cfg,
            scope_groups=scope_groups,
            k_select=int(k_select),
            client_size=int(client_size),
            large_threshold=int(large_threshold),
            groups=groups,
        )

    groups_used = groups if isinstance(groups, dict) else build_param_groups(model)
    scope = {g for g in set(scope_groups) if g in groups_used}
    stable_order = get_group_order(model, groups_used)
    force_keep = infer_head_group_names(model, groups_used).intersection(scope)

    k = int(max(1, int(k_select)))
    k = int(min(k, len(scope))) if scope else 0
    if k <= len(force_keep):
        selected = set(list(sorted(force_keep, key=lambda g: stable_order.index(g)))[:k])
    else:
        need = int(k - len(force_keep))
        non_forced = [g for g in stable_order if g in scope and g not in force_keep]
        if need > 0 and non_forced:
            picked = list(np.random.choice(non_forced, size=min(need, len(non_forced)), replace=False))
        else:
            picked = []
        selected = set(force_keep)
        selected.update(picked)

    set_trainable_by_selected_groups(model, groups_used, selected)
    return {
        "scores": {g: 0.0 for g in stable_order if g in scope},
        "selected_groups": sorted(list(selected), key=lambda g: stable_order.index(g) if g in stable_order else 10**9),
        "k_select": int(k),
        "ratio": None,
        "forced_keep": sorted(list(force_keep), key=lambda g: stable_order.index(g) if g in stable_order else 10**9),
        "stable_order": stable_order,
        "mask_vector": None,
        "limit_batches": 0,
        "score_mode": "random_mask_control",
        "selection_strategy": "random_mask",
        "skipped": False,
    }


def _select_apc_candidate(
        candidates: Dict[str, Any],
        scores: Dict[str, float],
        *,
        min_gain: float = 1e-6,
):
    if not candidates:
        raise ValueError("APC candidate set is empty.")
    neutral = candidates.get("neutral", next(iter(candidates.values())))
    valid_names = [name for name in candidates.keys() if name in scores and np.isfinite(float(scores[name]))]
    if not valid_names:
        return neutral

    best_score = max(float(scores[name]) for name in valid_names)
    close_names = [
        name for name in valid_names
        if (best_score - float(scores[name])) <= float(min_gain)
    ]
    close_names.sort(key=lambda name: (_candidate_cost(candidates[name]), str(name)))
    best_name = close_names[0]

    neutral_score = scores.get("neutral", None)
    if neutral_score is not None and np.isfinite(float(neutral_score)):
        if float(scores[best_name]) <= float(neutral_score) + float(min_gain):
            if _candidate_cost(neutral) <= _candidate_cost(candidates[best_name]):
                return neutral
    return candidates[best_name]


def _run_candidate_for_val(
        *,
        base_model: nn.Module,
        candidate,
        client_loaders,
        client_name: str,
        groups: Dict[str, List[str]],
        scope_groups,
        m3_cfg: M3Config,
        cfg: Ch4Config,
        device,
        n_i: int,
        large_th: int,
) -> Tuple[float, Dict[str, Any]]:
    trial = copy.deepcopy(base_model).to(device)

    K_pers = int(getattr(candidate, "K_pers", 1))
    E_pers = int(getattr(candidate, "E_pers", cfg.E_pers_min))
    rep_off = bool(getattr(candidate, "rep_off", False))
    if rep_off:
        K_pers = 1
        E_pers = 0

    sel_pack = _run_group_selection_for_stage2(
        model=trial,
        loader=client_loaders[client_name]["train"],
        device=device,
        cfg=cfg,
        m3_cfg=m3_cfg,
        scope_groups=scope_groups,
        k_select=int(K_pers),
        client_size=int(n_i),
        large_threshold=int(large_th),
        groups=groups,
    )
    selected_groups = list(sel_pack["selected_groups"])
    _freeze_to_selected_groups(
        model=trial,
        groups=groups,
        selected_groups=selected_groups,
        freeze_bias_after_calib=bool(cfg.freeze_bias_after_calib),
    )

    params = [p for p in trial.parameters() if p.requires_grad]
    best_state = {k: v.detach().cpu().clone() for k, v in trial.state_dict().items()}
    best_val = _val_selection_score(trial, client_loaders[client_name]["val"], device, cfg)
    bad = 0

    if params and E_pers > 0:
        opt = optim.Adam(
            params,
            lr=float(cfg.personalization_lr),
            weight_decay=float(cfg.weight_decay),
        )
        crit = nn.BCEWithLogitsLoss()
        for _ in range(int(E_pers)):
            train_epoch_masked_stage2(
                model=trial,
                loader=client_loaders[client_name]["train"],
                optimizer=opt,
                criterion=crit,
                device=device,
                groups=groups,
                selected_groups=selected_groups,
                group_clip_norm=cfg.group_clip_norm,
            )
            cur = _val_selection_score(trial, client_loaders[client_name]["val"], device, cfg)
            if np.isfinite(cur) and (cur > best_val + 1e-6):
                best_val = float(cur)
                best_state = {k: v.detach().cpu().clone() for k, v in trial.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= int(max(1, cfg.early_stop_patience)):
                    break

    trial.load_state_dict(best_state, strict=True)
    return float(best_val), {
        "selected_groups": selected_groups,
        "m3_scores": dict(sel_pack.get("scores", {})),
        "m3_mask_vector": sel_pack.get("mask_vector", None),
    }


# ============================================================
# Pure paper Stage II
# ============================================================
def finetune_dapfl_stage2(
        variant_name: str,
        global_model: nn.Module,
        client_loaders,
        client_names: List[str],
        client_sizes: List[int],
        cfg: Ch4Config,
        device,
        *,
        priors_per_client: Dict[str, Dict[str, float]],
        p_g_qbar: float,
) -> Dict[str, Any]:
    client_metrics: Dict[str, Dict[str, Any]] = {}
    val_client_metrics: Dict[str, Dict[str, Any]] = {}
    client_debug: Dict[str, Dict[str, Any]] = {}

    n_med = int(np.median(client_sizes)) if client_sizes else 1
    large_th = int(cfg.large_threshold) if cfg.large_threshold is not None else max(1, n_med)

    apc_params = _build_apc_params_from_cfg(cfg)
    n_g_total = int(priors_per_client["_global"]["n_g"])
    n_pos_g_total = int(priors_per_client["_global"]["n_pos_g"])
    n_neg_g_total = int(max(0, n_g_total - n_pos_g_total))
    apc_count_rows = []
    for cname in client_names:
        n_i_tmp = int(priors_per_client[cname]["n_i"])
        n_pos_i_tmp = int(priors_per_client[cname]["n_pos_i"])
        apc_count_rows.append({
            "client": str(cname),
            "client_name": str(cname),
            "n_k_pos": int(n_pos_i_tmp),
            "n_k_neg": int(max(0, n_i_tmp - n_pos_i_tmp)),
            "n_total_pos": int(n_pos_g_total),
            "n_total_neg": int(n_neg_g_total),
        })
    apc_gamma_s = compute_gamma_s_from_counts(apc_count_rows, params=apc_params)
    apc_controller_type = str(getattr(cfg, "apc_controller_type", "label_shift")).strip().lower()
    use_product_apc = str(getattr(cfg, "apc_mapping_mode", "")).strip().lower() == "need_reliability_product"
    apc_signal_mode = "need_reliability_product" if use_product_apc else "reliability_prior"
    m3_cfg = _build_m3_cfg_from_cfg(cfg)

    product_apc_outputs = {}
    if use_product_apc:
        product_apc_outputs = compute_need_reliability_product_outputs(
            apc_count_rows,
            params=apc_params,
        )

    for i, name in enumerate(client_names):
        local = copy.deepcopy(global_model).to(device)
        groups = build_param_groups(local)
        scope_groups = set(groups.keys())  # paper: selection over all groups G

        p_i_raw = float(priors_per_client[name]["p_i_raw"])
        n_i = int(priors_per_client[name]["n_i"])
        n_pos_i = int(priors_per_client[name]["n_pos_i"])
        n_g = int(priors_per_client["_global"]["n_g"])
        n_pos_g = int(priors_per_client["_global"]["n_pos_g"])

        val_loader = client_loaders[name]["val"]

        # ---------------- M1: prior correction ----------------
        m1_cfg = M1Config(
            enabled=bool(cfg.use_prior_bias_calib),
            smoothing="additive",
            alpha=float(cfg.prior_smoothing_alpha),
            beta=float(cfg.prior_smoothing_beta),
            eps=1e-6,
            apply_mode="set",
            guard_reapply=True,
            enable_lambda_selection=True,
            freeze_bias_after_calib=bool(cfg.freeze_bias_after_calib),
        )

        m1_bias_diag = apply_prior_bias_calibration(
            model=local,
            p_i=float(p_i_raw),
            p_g=float(p_g_qbar),
            n_i=int(n_i),
            n_g=int(n_g),
            pos_i=int(n_pos_i),
            pos_g=int(n_pos_g),
            cfg=m1_cfg,
            val_loader=val_loader,
            device=device,
        )
        m1_anchor_state = {k: v.detach().cpu().clone() for k, v in local.state_dict().items()}
        val_bce_before_m3 = _split_bce_loss(local, val_loader, device)

        # ---------------- M2/APC: reliability-adjusted output + optional validation gate ----------------
        prior_abs_shift = _prior_shift_value(
            p_i_raw=float(p_i_raw),
            p_g_raw=float(p_g_qbar),
            n_i=int(n_i),
            n_g=int(n_g),
            n_pos_i=int(n_pos_i),
            n_pos_g=int(n_pos_g),
            cfg=cfg,
        )
        apc_kwargs = {
            "params": apc_params,
            "n_k_pos": int(n_pos_i),
            "n_k_neg": int(max(0, int(n_i) - int(n_pos_i))),
            "n_total_pos": int(n_pos_g),
            "n_total_neg": int(max(0, int(n_g) - int(n_pos_g))),
            "gamma_s": float(apc_gamma_s),
        }
        if use_product_apc:
            product_neutral = product_apc_outputs.get(str(name))
            if product_neutral is None:
                raise RuntimeError(f"Product APC output missing for client: {name}")
            if bool(cfg.apc_candidate_selection):
                apc_candidates = get_apc_candidates_from_output(
                    base_output=product_neutral,
                    params=apc_params,
                )
                apc_candidate_scores = {}
                for cand_name, candidate in apc_candidates.items():
                    score, _ = _run_candidate_for_val(
                        base_model=local,
                        candidate=candidate,
                        client_loaders=client_loaders,
                        client_name=name,
                        groups=groups,
                        scope_groups=scope_groups,
                        m3_cfg=m3_cfg,
                        cfg=cfg,
                        device=device,
                        n_i=int(n_i),
                        large_th=int(large_th),
                    )
                    apc_candidate_scores[str(cand_name)] = float(score)
                apc_selected = _select_apc_candidate(apc_candidates, apc_candidate_scores)
            else:
                apc_candidates = {"neutral": product_neutral}
                apc_candidate_scores = {}
                apc_selected = product_neutral
        elif bool(cfg.apc_candidate_selection):
            apc_candidates = get_apc_candidates(**apc_kwargs)
            apc_candidate_scores: Dict[str, float] = {}
            for cand_name, candidate in apc_candidates.items():
                score, _ = _run_candidate_for_val(
                    base_model=local,
                    candidate=candidate,
                    client_loaders=client_loaders,
                    client_name=name,
                    groups=groups,
                    scope_groups=scope_groups,
                    m3_cfg=m3_cfg,
                    cfg=cfg,
                    device=device,
                    n_i=int(n_i),
                    large_th=int(large_th),
                )
                apc_candidate_scores[str(cand_name)] = float(score)
            apc_selected = _select_apc_candidate(apc_candidates, apc_candidate_scores)
        else:
            apc_candidates = {
                "neutral": get_apc_output(**apc_kwargs)
            }
            apc_candidate_scores = {}
            apc_selected = apc_candidates["neutral"]

        K_pers = int(getattr(apc_selected, "K_pers", 1))
        E_pers = int(getattr(apc_selected, "E_pers", cfg.E_pers_min))
        rep_off = bool(getattr(apc_selected, "rep_off", False))
        apc_meta = dict(getattr(apc_selected, "metadata", {}) or {})

        if rep_off:
            K_pers = 1
            E_pers = 0

        # ---------------- M3: select M*_k once ----------------
        # Even when E_pers == 0, we still select once for logging/audit.
        sel_pack = _run_group_selection_for_stage2(
            model=local,
            loader=client_loaders[name]["train"],
            device=device,
            cfg=cfg,
            m3_cfg=m3_cfg,
            scope_groups=scope_groups,
            k_select=int(K_pers),
            client_size=int(n_i),
            large_threshold=int(large_th),
            groups=groups,
        )
        selected_groups = list(sel_pack["selected_groups"])
        head_group_selected = "g1_head" in set(selected_groups)
        selected_count_le_budget = len(selected_groups) <= int(max(1, K_pers))

        # bias stays fixed during Stage II-B if requested, and only the
        # selected groups remain trainable. This is the critical fix that
        # makes Stage II-B a *true* masked personalization step.
        trainable_param_names = _freeze_to_selected_groups(
            model=local,
            groups=groups,
            selected_groups=selected_groups,
            freeze_bias_after_calib=bool(cfg.freeze_bias_after_calib),
        )
        total_param_count = int(sum(int(p.numel()) for p in local.parameters()))
        trainable_param_count = int(sum(int(p.numel()) for p in local.parameters() if bool(p.requires_grad)))
        trainable_param_ratio = float(trainable_param_count / max(1, total_param_count))
        selected_group_count = int(len(set(str(g) for g in selected_groups)))
        group_count = int(max(1, len(groups)))
        effective_group_epoch_budget = float(int(E_pers) * selected_group_count / group_count)
        effective_param_epoch_budget = float(int(E_pers) * trainable_param_ratio)

        # ---------------- Stage II-B masked ERM ----------------
        params = [p for p in local.parameters() if p.requires_grad]

        # <--- 补充: Safe Fallback Initialization at Epoch 0
        best_state = {k: v.detach().cpu().clone() for k, v in local.state_dict().items()}
        best_val = float("-inf")
        bad = 0

        use_early_stop = bool(val_loader is not None)
        if use_early_stop:
            v0_logits, v0_y = collect_logits_labels(local, val_loader, device)
            if v0_y.size == 0 or v0_logits.size == 0 or len(np.unique(v0_y.astype(int))) < 2:
                use_early_stop = False
            else:
                best_val = _val_selection_score(local, val_loader, device, cfg)

        if params and E_pers > 0:
            opt = optim.Adam(
                params,
                lr=float(cfg.personalization_lr),
                weight_decay=float(cfg.weight_decay),
            )
            crit = nn.BCEWithLogitsLoss()

            for _ in range(int(E_pers)):
                train_epoch_masked_stage2(
                    model=local,
                    loader=client_loaders[name]["train"],
                    optimizer=opt,
                    criterion=crit,
                    device=device,
                    groups=groups,
                    selected_groups=selected_groups,
                    group_clip_norm=cfg.group_clip_norm,
                )

                if not use_early_stop:
                    # Without early stop, we naturally keep the final epoch state
                    best_state = {k: v.detach().cpu().clone() for k, v in local.state_dict().items()}
                    continue

                cur = _val_selection_score(local, val_loader, device, cfg)
                if np.isfinite(cur) and (cur > best_val + 1e-6):
                    best_val = float(cur)
                    best_state = {k: v.detach().cpu().clone() for k, v in local.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= int(max(1, cfg.early_stop_patience)):
                        break

        # Always load the best state (which might be the fallback state from Epoch 0)
        local.load_state_dict(best_state, strict=True)

        # ---------------- realized drift ‖Θ_k,pers − Θ_global‖₂ (Prop 6.2) ----------------
        # 计算 Stage II-B 结束后参数偏移量，用于验证漂移界 Proposition 6.2。
        # 结果存入 client_debug["realized_drift"]，用于 APC 验证图 Panel D。
        val_bce_after_m3 = _split_bce_loss(local, val_loader, device)

        per_group_normalized_drift: Dict[str, float] = {}
        stage2_drift_norm = float("nan")
        try:
            local_sd = local.state_dict()
            total_stage2_sq = 0.0
            for gname, pnames in groups.items():
                group_sq = 0.0
                group_dim = 0
                for pname in pnames:
                    if pname in local_sd and pname in m1_anchor_state:
                        diff = local_sd[pname].float().cpu() - m1_anchor_state[pname].float().cpu()
                        group_sq += float(diff.pow(2).sum().item())
                        group_dim += int(diff.numel())
                per_group_normalized_drift[str(gname)] = float((group_sq ** 0.5) / ((max(1, group_dim)) ** 0.5))
                total_stage2_sq += float(group_sq)
            stage2_drift_norm = float(total_stage2_sq ** 0.5)
        except Exception:
            per_group_normalized_drift = {}

        realized_drift = float("nan")
        try:
            _drift_sq = 0.0
            g_sd = global_model.state_dict()
            for pname, pval in local.state_dict().items():
                if pname in g_sd:
                    _drift_sq += (
                        pval.float().cpu() - g_sd[pname].float().cpu()
                    ).pow(2).sum().item()
            realized_drift = float(_drift_sq ** 0.5)
        except Exception:
            pass
        # -----------------------------------------------------------------------------------

        # ---------------- threshold on val, final test ----------------
        thr = _choose_threshold_from_val(local, val_loader, device, cfg)
        val_res = _eval_on_test_with_threshold(
            local,
            val_loader,
            device,
            cfg,
            thr,
        )
        test_res = _eval_on_test_with_threshold(
            local,
            client_loaders[name]["test"],
            device,
            cfg,
            thr,
        )

        val_client_metrics[name] = val_res
        client_metrics[name] = test_res
        client_debug[name] = {
            "p_i_raw":            float(p_i_raw),
            "p_g_qbar":           float(p_g_qbar),
            "n_i":                int(n_i),
            "n_pos_i":            int(n_pos_i),
            "n_neg_i":            int(max(0, int(n_i) - int(n_pos_i))),
            "n_g":                int(n_g),
            "n_pos_g":            int(n_pos_g),
            "n_neg_g":            int(max(0, int(n_g) - int(n_pos_g))),
            "abs_shift":          float(prior_abs_shift),
            "apc_need_value":      float(apc_meta.get("r_reliability", np.nan)),
            "apc_need_q10":        None,
            "apc_need_q90":        None,
            "apc_shift_mode":      str(apc_signal_mode),
            "apc_signal_mode":     str(apc_signal_mode),
            "apc_formula":         str(apc_meta.get("apc_formula", "reliability_adjusted_prior_mismatch")),
            "apc_mapping_mode":    str(apc_meta.get("mapping_mode", getattr(cfg, "apc_mapping_mode", "log_median"))),
            "apc_controller_type": str(apc_meta.get("controller_type", apc_controller_type)),
            "apc_formula_normalization": True,
            "apc_pi_k":            float(apc_meta.get("pi_k", np.nan)),
            "apc_pi_ref":          float(apc_meta.get("pi_ref", np.nan)),
            "apc_delta_b":         float(apc_meta.get("delta_b", np.nan)),
            "apc_delta":           float(apc_meta.get("delta", np.nan)),
            "apc_sigma_delta":     float(apc_meta.get("sigma_delta", np.nan)),
            "apc_q_reliability":   float(apc_meta.get("q_reliability", np.nan)),
            "apc_s_reliability":   float(apc_meta.get("s_reliability", np.nan)),
            "apc_u_k":             float(apc_meta.get("u_k", np.nan)),
            "apc_alpha_k":         float(apc_meta.get("alpha_k", np.nan)),
            "apc_r_final":         float(apc_meta.get("r_final", apc_meta.get("r_reliability", np.nan))),
            "apc_n_k":             float(apc_meta.get("n_k", np.nan)),
            "apc_max_s_reliability": float(apc_meta.get("max_s_reliability", np.nan)),
            "apc_gamma_s":         float(apc_meta.get("gamma_s", np.nan)),
            "apc_gamma_s_scaled":  float(apc_meta.get("gamma_s_scaled", np.nan)),
            "apc_direct_gamma":    float(apc_meta.get("direct_gamma", np.nan)),
            "apc_r_neutral":       float(apc_meta.get("r_neutral", np.nan)),
            "apc_r_reliability":   float(apc_meta.get("r_reliability", np.nan)),
            "pi_k_tilde":          float(apc_meta.get("pi_k", np.nan)),
            "pi_g_tilde":          float(apc_meta.get("pi_ref", np.nan)),
            "delta_b_k":           float(apc_meta.get("delta_b", np.nan)),
            "delta_k":             float(apc_meta.get("delta", np.nan)),
            "sigma_delta_k":       float(apc_meta.get("sigma_delta", np.nan)),
            "q_k":                 float(apc_meta.get("q_reliability", np.nan)),
            "s_k":                 float(apc_meta.get("s_reliability", np.nan)),
            "u_k":                 float(apc_meta.get("u_k", np.nan)),
            "alpha_k":             float(apc_meta.get("alpha_k", np.nan)),
            "r_final":             float(apc_meta.get("r_final", apc_meta.get("r_reliability", np.nan))),
            "gamma_s":             float(apc_meta.get("gamma_s", np.nan)),
            "r_k":                 float(apc_meta.get("r_reliability", np.nan)),
            "m_k":                 int(apc_meta.get("m_k", K_pers)),
            "e_k":                 int(apc_meta.get("e_k", E_pers)),
            "adaptation_scope":    int(apc_meta.get("m_k", K_pers)),
            "adaptation_depth":    int(apc_meta.get("e_k", E_pers)),
            "m1_bias":            dict(m1_bias_diag),
            "apc_name":           str(getattr(apc_selected, "name", "neutral")),
            "apc_candidate_selection": bool(cfg.apc_candidate_selection),
            "apc_candidate_scores": dict(apc_candidate_scores),
            "apc_candidate_names": list(apc_candidates.keys()),
            "K_pers":             int(K_pers),
            "E_pers":             int(E_pers),
            "rep_off":            bool(rep_off),
            "selected_groups":    list(selected_groups),
            "selected_group_count": int(selected_group_count),
            "trainable_param_names": list(trainable_param_names),
            "total_param_count": int(total_param_count),
            "trainable_param_count": int(trainable_param_count),
            "trainable_param_ratio": float(trainable_param_ratio),
            "effective_group_epoch_depth": float(effective_group_epoch_budget),
            "full_model_equiv_epoch_depth": float(effective_group_epoch_budget),
            "effective_param_epoch_depth": float(effective_param_epoch_budget),
            "scope_depth_accounting": "E_pers * selected_group_count / total_group_count; param version uses trainable_param_count / total_param_count",
            # Backward-compatible names consumed by existing summary scripts.
            "effective_group_epoch_budget": float(effective_group_epoch_budget),
            "full_model_equiv_epoch_budget": float(effective_group_epoch_budget),
            "effective_param_epoch_budget": float(effective_param_epoch_budget),
            "budget_accounting": "E_pers * selected_group_count / total_group_count; param version uses trainable_param_count / total_param_count",
            "paper_safety": {
                "m1_bias_only_before_stage2": bool(cfg.use_prior_bias_calib),
                "head_group_selected": bool(head_group_selected),
                "selected_count_le_budget": bool(selected_count_le_budget),
                "validation_threshold_fixed_for_test": True,
                "epoch0_fallback_checkpoint": True,
            },
            "m3_scores":          dict(sel_pack.get("scores", {})),
            "m3_mask_vector":     sel_pack.get("mask_vector", None),
            "m3_selection_strategy": str(getattr(cfg, "m3_selection_strategy", "gradient")),
            "stage2_val_bce_before_m3": float(val_bce_before_m3) if np.isfinite(float(val_bce_before_m3)) else None,
            "stage2_val_bce_after_m3": float(val_bce_after_m3) if np.isfinite(float(val_bce_after_m3)) else None,
            "stage2_drift_norm": float(stage2_drift_norm) if np.isfinite(float(stage2_drift_norm)) else None,
            "per_group_normalized_drift": dict(per_group_normalized_drift),
            "thr":                float(thr),
            "train_size":         int(n_i),
            "best_val_score":     float(best_val) if np.isfinite(best_val) else None,
            # ── 新增字段（用于 APC 验证图）──────────────────────────────
            "realized_drift":     float(realized_drift), # ‖Θ_k,pers − Θ_global‖₂ (Prop 6.2)
        }

    agg = _aggregate_client_metrics(client_metrics)
    val_agg = _aggregate_client_metrics(val_client_metrics)
    return {
        "name": variant_name,
        **agg,
        "validation": val_agg,
        "val_client_metrics": val_client_metrics,
        "client_metrics": client_metrics,
        "client_debug": client_debug,
    }


# ============================================================
# Public runner
# ============================================================
def run_dapfl_stage2(
        backbone_name: str,
        backbone_model: nn.Module,
        client_loaders,
        central,
        client_names: List[str],
        client_sizes: List[int],
        input_dim: int,
        cfg: Ch4Config,
        device,
        *,
        qbar: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    qbar should come from Stage I realized exposure logs.
    If absent, we fall back to pooled global prior, but the paper-aligned
    behavior is to use qbar-aligned global prior.
    """
    _, priors_per_client = get_priors(client_loaders, client_names, central)

    p_g_qbar = compute_qbar_aligned_global_prior(
        client_names=client_names,
        priors_per_client=priors_per_client,
        qbar=qbar,
    )

    return finetune_dapfl_stage2(
        variant_name=f"{backbone_name}-DA-PFL",
        global_model=backbone_model,
        client_loaders=client_loaders,
        client_names=client_names,
        client_sizes=client_sizes,
        cfg=cfg,
        device=device,
        priors_per_client=priors_per_client,
        p_g_qbar=float(p_g_qbar),
    )

