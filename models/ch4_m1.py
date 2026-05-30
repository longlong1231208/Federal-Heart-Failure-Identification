# models/ch4_m1.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Config
# ============================================================
@dataclass
class M1Config:
    enabled: bool = True

    # --------------------------------------------------------
    # Prior smoothing
    # --------------------------------------------------------
    smoothing: str = "additive"
    alpha: float = 0.5
    beta: float = 0.5
    eps: float = 1e-6

    # --------------------------------------------------------
    # Bias apply mode
    #   "set" is strongly recommended for idempotence:
    #       b <- b_base + delta_eff
    # --------------------------------------------------------
    apply_mode: str = "set"      # "add" or "set"
    guard_reapply: bool = True
    applied_flag_name: str = "_m1_bias_applied"
    bias_base_name: str = "_m1_bias_base"

    # --------------------------------------------------------
    # Lambda selection for effective bias correction
    #   delta_eff = lambda * delta_raw
    # We choose lambda on validation set with a 1D search:
    #   val_loss(lambda) + reg_eta * lambda^2
    # --------------------------------------------------------
    enable_lambda_selection: bool = True
    lambda_grid: Tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    )
    lambda_reg_eta: float = 0.05
    lambda_min_val_size: int = 8

    # Optional safety: if val labels are degenerate, fall back
    # "one"  -> lambda=1.0
    # "zero" -> lambda=0.0
    fallback_lambda_mode: str = "one"

    freeze_bias_after_calib: bool = False

    # --------------------------------------------------------
    # Internal switches
    # --------------------------------------------------------
    prefer_forward_logits: bool = True


# ============================================================
# Helpers
# ============================================================
def get_head_linear(model: nn.Module) -> nn.Linear:
    """
    Robust head resolver: supports .fc / .head / .classifier.
    Requires out_features == 1 for BCE single-logit binary classification.
    """
    for name in ["fc", "head", "classifier"]:
        if hasattr(model, name):
            module = getattr(model, name)
            if isinstance(module, nn.Linear):
                if int(module.out_features) != 1:
                    raise ValueError(
                        f"Head {name} out_features={module.out_features}, "
                        "but this M1 assumes BCE single-logit head (out_features==1)."
                    )
                return module
    raise ValueError("Could not find a valid Linear head (fc/head/classifier) with out_features==1.")


def _as_scalar_bias(head: nn.Linear) -> torch.Tensor:
    if head.bias is None:
        raise ValueError("Head has no bias; M1 prior-bias calibration requires bias.")
    if int(head.bias.numel()) != 1 or int(head.out_features) != 1:
        raise ValueError("This M1 implementation assumes a binary BCE head with a single bias.")
    return head.bias.view(1)  # shape (1,)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(float(p), eps, 1.0 - eps))
    return float(math.log(p) - math.log(1.0 - p))


def smooth_priors(
    p_i: float,
    n_i: int,
    p_g: float,
    n_g: int,
    cfg: M1Config,
    *,
    pos_i: Optional[int] = None,
    pos_g: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Additive/Beta smoothing:
        p_tilde = (pos + alpha) / (n + alpha + beta)

    If pos_i / pos_g are not provided, fall back to round(p*n).
    """
    a = float(cfg.alpha)
    b = float(cfg.beta)

    ni = max(int(n_i), 0)
    ng = max(int(n_g), 0)

    # client prior
    if ni == 0:
        pi_t = float(np.clip(float(p_i), cfg.eps, 1.0 - cfg.eps))
    else:
        if pos_i is None:
            pos_i_use = int(round(float(p_i) * float(ni)))
        else:
            pos_i_use = int(max(0, min(int(pos_i), ni)))
        denom_i = float(ni) + a + b
        pi_t = (float(pos_i_use) + a) / denom_i if denom_i > 0 else float(p_i)

    # global prior
    if ng == 0:
        pg_t = float(np.clip(float(p_g), cfg.eps, 1.0 - cfg.eps))
    else:
        if pos_g is None:
            pos_g_use = int(round(float(p_g) * float(ng)))
        else:
            pos_g_use = int(max(0, min(int(pos_g), ng)))
        denom_g = float(ng) + a + b
        pg_t = (float(pos_g_use) + a) / denom_g if denom_g > 0 else float(p_g)

    pi_t = float(np.clip(pi_t, cfg.eps, 1.0 - cfg.eps))
    pg_t = float(np.clip(pg_t, cfg.eps, 1.0 - cfg.eps))
    return pi_t, pg_t


def reset_m1_state(model: nn.Module, cfg: Optional[M1Config] = None) -> None:
    """
    Clear M1 bias-related state so the same model instance can be reused safely.
    """
    cfg = cfg or M1Config()
    if hasattr(model, cfg.applied_flag_name):
        delattr(model, cfg.applied_flag_name)
    if hasattr(model, cfg.bias_base_name):
        delattr(model, cfg.bias_base_name)

# ============================================================
# Raw logits collection
# ============================================================
@torch.no_grad()
def _forward_raw_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Prefer forward_logits(x) if the model exposes it.
    Otherwise fall back to forward(x).
    """
    if hasattr(model, "forward_logits") and callable(getattr(model, "forward_logits")):
        return model.forward_logits(x).view(-1)
    return model(x).view(-1)


@torch.no_grad()
def _collect_logits_labels_raw(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: M1Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect raw logits and labels.
    Returns CPU tensors.
    """
    if loader is None:
        return torch.empty(0), torch.empty(0)

    model.eval()
    zs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)
        z = _forward_raw_logits(model, x).detach().view(-1)
        zs.append(z.cpu())
        ys.append(y.cpu())

    if not zs:
        return torch.empty(0), torch.empty(0)
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


# ============================================================
# Lambda selection for effective bias correction
# ============================================================
def _select_lambda_from_logits(
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    delta_raw: float,
    cfg: M1Config,
) -> Dict[str, float]:
    
    out = {
        "lambda": 1.0,
        "delta_raw": float(delta_raw),
        "delta_eff": float(delta_raw),
        "best_obj": float("inf"),
        "best_bce": float("inf"),
    }

    if z_val.numel() == 0 or y_val.numel() == 0:
        lam = 1.0 if cfg.fallback_lambda_mode == "one" else 0.0
        out["lambda"] = float(lam)
        out["delta_eff"] = float(lam * delta_raw)
        return out

    if int(z_val.numel()) < int(cfg.lambda_min_val_size):
        lam = 1.0 if cfg.fallback_lambda_mode == "one" else 0.0
        out["lambda"] = float(lam)
        out["delta_eff"] = float(lam * delta_raw)
        return out

    # If labels are all one class, BCE ranking on val is unstable for calibration choice
    if y_val.unique().numel() < 2:
        lam = 1.0 if cfg.fallback_lambda_mode == "one" else 0.0
        out["lambda"] = float(lam)
        out["delta_eff"] = float(lam * delta_raw)
        return out

    crit = nn.BCEWithLogitsLoss(reduction="mean")

    best_obj = float("inf")
    best_bce = float("inf")
    best_lam = 1.0

    eta = float(cfg.lambda_reg_eta)

    for lam in cfg.lambda_grid:
        lam = float(lam)
        z_adj = z_val + lam * float(delta_raw)
        bce = float(crit(z_adj, y_val).detach().cpu().item())
        reg = eta * (lam * lam)
        obj = bce + reg

        if obj < best_obj:
            best_obj = obj
            best_bce = bce
            best_lam = lam

    out["lambda"] = float(best_lam)
    out["delta_eff"] = float(best_lam * delta_raw)
    out["best_obj"] = float(best_obj)
    out["best_bce"] = float(best_bce)
    return out


@torch.no_grad()
def select_effective_bias_delta(
    model: nn.Module,
    *,
    p_i: float,
    p_g: float,
    n_i: int,
    n_g: int,
    cfg: Optional[M1Config] = None,
    pos_i: Optional[int] = None,
    pos_g: Optional[int] = None,
    val_loader=None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute raw delta from smoothed priors, then optionally choose lambda on val set:
        delta_eff = lambda * delta_raw
    """
    cfg = cfg or M1Config()

    pi_t, pg_t = smooth_priors(
        p_i=float(p_i),
        n_i=int(n_i),
        p_g=float(p_g),
        n_g=int(n_g),
        cfg=cfg,
        pos_i=pos_i,
        pos_g=pos_g,
    )
    delta_raw = _logit(pi_t, cfg.eps) - _logit(pg_t, cfg.eps)

    out = {
        "p_i_tilde": float(pi_t),
        "p_g_tilde": float(pg_t),
        "delta_raw": float(delta_raw),
        "lambda": 1.0,
        "delta_eff": float(delta_raw),
        "best_obj": float("inf"),
        "best_bce": float("inf"),
    }

    if (not cfg.enable_lambda_selection) or (val_loader is None) or (device is None):
        return out

    z_val_cpu, y_val_cpu = _collect_logits_labels_raw(model, val_loader, device, cfg)
    if z_val_cpu.numel() == 0 or y_val_cpu.numel() == 0:
        return out

    z_val = z_val_cpu.to(device)
    y_val = y_val_cpu.to(device)

    lam_out = _select_lambda_from_logits(
        z_val=z_val,
        y_val=y_val,
        delta_raw=float(delta_raw),
        cfg=cfg,
    )

    out.update(lam_out)
    return out


# ============================================================
# Core M1a: prior-bias calibration
# ============================================================
@torch.no_grad()
def apply_prior_bias_calibration(
    model: nn.Module,
    p_i: float,
    p_g: float,
    n_i: int,
    n_g: int,
    pos_i: Optional[int] = None,
    pos_g: Optional[int] = None,
    cfg: Optional[M1Config] = None,
    val_loader=None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Paper-aligned bias calibration:

        delta_raw = logit(p_i_tilde) - logit(p_g_tilde)
        delta_eff = lambda * delta_raw

    apply_mode:
      - "add": b <- b + delta_eff
      - "set": b <- b_base + delta_eff   (recommended / idempotent)
    """
    cfg = cfg or M1Config()

    if not cfg.enabled:
        return {"applied": 0.0}

    if cfg.guard_reapply and getattr(model, cfg.applied_flag_name, False):
        return {"applied": 0.0, "status": "skipped_already_applied"}

    head = get_head_linear(model)
    b = _as_scalar_bias(head)
    bias_before = float(b.detach().cpu().item())

    if not hasattr(model, cfg.bias_base_name):
        base = b.detach().clone()
        model.register_buffer(cfg.bias_base_name, base)

    base = getattr(model, cfg.bias_base_name)

    delta_info = select_effective_bias_delta(
        model=model,
        p_i=float(p_i),
        p_g=float(p_g),
        n_i=int(n_i),
        n_g=int(n_g),
        pos_i=pos_i,
        pos_g=pos_g,
        cfg=cfg,
        val_loader=val_loader,
        device=device,
    )

    delta_eff = float(delta_info["delta_eff"])
    delta_t = b.new_tensor([delta_eff])

    mode = (cfg.apply_mode or "set").lower().strip()
    if mode == "add":
        b.add_(delta_t)
    else:
        b.copy_(base.to(device=b.device, dtype=b.dtype) + delta_t)

    bias_after = float(b.detach().cpu().item())

    if cfg.freeze_bias_after_calib and head.bias is not None:
        head.bias.requires_grad_(False)

    if cfg.guard_reapply:
        setattr(model, cfg.applied_flag_name, True)

    out = {
        "applied": 1.0,
        "p_i_tilde": float(delta_info["p_i_tilde"]),
        "p_g_tilde": float(delta_info["p_g_tilde"]),
        "delta_raw": float(delta_info["delta_raw"]),
        "lambda": float(delta_info["lambda"]),
        "delta_eff": float(delta_info["delta_eff"]),
        "best_obj": float(delta_info["best_obj"]),
        "best_bce": float(delta_info["best_bce"]),
        "bias_before": float(bias_before),
        "bias_after": float(bias_after),
    }
    if pos_i is not None:
        out["pos_i"] = float(int(pos_i))
    if pos_g is not None:
        out["pos_g"] = float(int(pos_g))
    return out


