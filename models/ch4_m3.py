# models/ch4_m3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Config
# ============================================================
@dataclass
class M3Config:
    enabled: bool = True

    # --------------------------------------------------------
    # Paper-aligned sensitivity score:
    #   s_{k,g} = E_B [ ||grad_g||_2 / (sqrt(d_g) + eps) ]
    # robustly estimated by median over mini-batches.
    # --------------------------------------------------------
    score_mode: str = "dimnorm_l2"   # recommended / paper-aligned
    grad_score_batches: int = 10
    grad_score_batches_large: Optional[int] = 20
    eps: float = 1e-8

    # always keep Group 1 (head) trainable
    force_keep_head: bool = True

    # selection
    allow_k_zero: bool = False   # paper effectively keeps head => K>=1

    # criterion consistency
    pos_weight: Optional[float] = None

    # deterministic
    seed: Optional[int] = None

    # --------------------------------------------------------
    # Optional support for downstream masked ERM
    # This is NOT used during selection itself, but provided
    # because Theorem 6.2 assumes per-group clipping.
    # --------------------------------------------------------
    enable_groupwise_clip_helper: bool = True
    group_clip_norm: Optional[float] = None

    # --------------------------------------------------------
    # If true, prefer model.forward_logits(x) for raw-logit scoring.
    # --------------------------------------------------------
    prefer_forward_logits: bool = True


# ============================================================
# Group / ordering helpers
# ============================================================
def build_param_groups(model: nn.Module) -> Dict[str, List[str]]:
    """
    Preferred: model.named_param_groups()
    Fallback: group by top-level module prefix.
    """
    if hasattr(model, "named_param_groups") and callable(getattr(model, "named_param_groups")):
        out = model.named_param_groups()
        return {str(k): list(v) for k, v in out.items()}

    out: Dict[str, List[str]] = {}
    for name, _ in model.named_parameters():
        top = name.split(".")[0]
        out.setdefault(top, []).append(name)
    return out


def get_group_order(model: nn.Module, groups: Dict[str, List[str]]) -> List[str]:
    """
    Stable order for deterministic tie-breaking.
    Prefer model.get_group_order() if available.
    """
    if hasattr(model, "get_group_order") and callable(getattr(model, "get_group_order")):
        order = list(getattr(model, "get_group_order")())
        order = [g for g in order if g in groups]
        remaining = sorted([g for g in groups.keys() if g not in order])
        return order + remaining
    return sorted(groups.keys())


def infer_head_group_names(model: nn.Module, groups: Dict[str, List[str]]) -> Set[str]:
    """
    Robustly infer head groups. In your GRUModel this should resolve to g1_head.
    """
    keep: Set[str] = set()

    # explicit common names
    for g in ("g1_head", "head", "fc", "classifier"):
        if g in groups:
            keep.add(g)

    # fallback by parameter-name prefixes
    if not keep:
        for g, pnames in groups.items():
            if any(
                pn.startswith("fc.") or pn.startswith("head.") or pn.startswith("classifier.")
                for pn in pnames
            ):
                keep.add(g)

    return keep


def snapshot_requires_grad(model: nn.Module) -> Dict[str, bool]:
    return {n: bool(p.requires_grad) for n, p in model.named_parameters()}


def restore_requires_grad(model: nn.Module, snap: Dict[str, bool]) -> None:
    for n, p in model.named_parameters():
        if n in snap:
            p.requires_grad_(bool(snap[n]))


def set_trainable_by_selected_groups(
    model: nn.Module,
    groups: Dict[str, List[str]],
    selected: Set[str],
) -> None:
    """
    Freeze everything except selected groups.
    """
    allow: Set[str] = set()
    for g in selected:
        for pname in groups.get(g, []):
            allow.add(pname)

    for name, p in model.named_parameters():
        p.requires_grad_(name in allow)


def _group_numel(
    groups: Dict[str, List[str]],
    name_map: Dict[str, torch.Tensor],
    gname: str,
) -> int:
    total = 0
    for pname in groups.get(gname, []):
        p = name_map.get(pname)
        if p is not None:
            total += int(p.numel())
    return int(total)


# ============================================================
# Forward helper
# ============================================================
def _forward_raw_logits(model: nn.Module, x: torch.Tensor, prefer_forward_logits: bool = True) -> torch.Tensor:
    if prefer_forward_logits and hasattr(model, "forward_logits") and callable(getattr(model, "forward_logits")):
        return model.forward_logits(x).view(-1)
    return model(x).view(-1)


# ============================================================
# Dropout control for stable gradient scoring
# ============================================================
def _set_dropout_p(model: nn.Module, p: float) -> Dict[int, float]:
    token: Dict[int, float] = {}
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            token[id(m)] = float(m.p)
            m.p = float(p)
    return token


def _restore_dropout_p(model: nn.Module, token: Dict[int, float]) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            mid = id(m)
            if mid in token:
                m.p = float(token[mid])


# ============================================================
# Paper-aligned scoring
# ============================================================
def _tensor_dimnorm_l2_score(g: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Paper Eq. (5.18)/(5.19) practical form:
        ||g||_2 / (sqrt(d) + eps)
    """
    if g is None:
        return 0.0
    numel = int(g.numel())
    if numel <= 0:
        return 0.0
    return float(torch.norm(g.detach(), p=2).item() / (np.sqrt(float(numel)) + float(eps)))


def _tensor_abs_mean_score(g: torch.Tensor, eps: float = 1e-8) -> float:
    if g is None:
        return 0.0
    return float(torch.mean(torch.abs(g.detach())).item())


def _tensor_l2_rms_score(g: torch.Tensor, eps: float = 1e-8) -> float:
    if g is None:
        return 0.0
    gg = g.detach()
    return float(torch.sqrt(torch.mean(gg * gg)).item())


def _score_group_from_current_grads(
    model: nn.Module,
    groups: Dict[str, List[str]],
    gname: str,
    score_mode: str,
    eps: float,
) -> float:
    """
    Aggregate all params inside one group into a single group score.
    For paper mode, concatenate effect is equivalent to using
    total group L2 norm and total group dimension.
    """
    named = dict(model.named_parameters())
    mode = str(score_mode).lower().strip()

    grads: List[torch.Tensor] = []
    total_numel = 0

    for pname in groups.get(gname, []):
        p = named.get(pname)
        if p is None or p.grad is None:
            continue
        grads.append(p.grad.detach().reshape(-1))
        total_numel += int(p.grad.numel())

    if total_numel <= 0 or not grads:
        return 0.0

    gvec = torch.cat(grads, dim=0)

    if mode == "abs":
        return _tensor_abs_mean_score(gvec, eps=eps)
    if mode == "l2":
        return _tensor_l2_rms_score(gvec, eps=eps)

    # default: paper-aligned
    return _tensor_dimnorm_l2_score(gvec, eps=eps)


def compute_group_grad_scores_within_scope(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    groups: Dict[str, List[str]],
    scope_groups: Set[str],
    limit_batches: int,
    score_mode: str,
    eps: float = 1e-8,
    pos_weight: Optional[float] = None,
    prefer_forward_logits: bool = True,
) -> Dict[str, float]:
    """
    Paper-aligned M3 sensitivity estimation.

    Important:
      - gradients are computed on raw logits
      - we use TRAIN mode for cuDNN RNN backward
      - dropout is temporarily disabled for stabler scoring
      - robust estimator = median across mini-batches (Eq. 5.19)
    """
    scope_groups = set(scope_groups)
    if loader is None or not scope_groups:
        return {g: 0.0 for g in scope_groups}

    snap = snapshot_requires_grad(model)
    was_training = bool(model.training)
    dropout_token: Dict[int, float] = {}

    per_group_batch_scores: Dict[str, List[float]] = {g: [] for g in scope_groups}

    try:
        # Only scope groups need grads during sensitivity estimation
        set_trainable_by_selected_groups(model, groups, scope_groups)

        # cuDNN RNN backward requires train mode
        model.train()

        # Disable dropout for more stable sensitivity estimates
        dropout_token = _set_dropout_p(model, p=0.0)

        if pos_weight is not None:
            pw = torch.tensor(float(pos_weight), device=device, dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()

        count = 0
        for i, batch in enumerate(loader):
            if limit_batches > 0 and i >= int(limit_batches):
                break

            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1)

            model.zero_grad(set_to_none=True)

            logits = _forward_raw_logits(model, x, prefer_forward_logits=prefer_forward_logits)
            if logits.numel() != y.numel():
                raise ValueError(
                    f"[M3] logits and labels size mismatch: logits={tuple(logits.shape)}, y={tuple(y.shape)}."
                )

            loss = criterion(logits, y)
            loss.backward()

            for gname in scope_groups:
                s = _score_group_from_current_grads(
                    model=model,
                    groups=groups,
                    gname=gname,
                    score_mode=score_mode,
                    eps=eps,
                )
                per_group_batch_scores[gname].append(float(s))

            count += 1

        if count <= 0:
            return {g: 0.0 for g in scope_groups}

        # paper robust estimator: median over batch scores
        out: Dict[str, float] = {}
        for g in scope_groups:
            vals = per_group_batch_scores.get(g, [])
            out[g] = float(np.median(np.asarray(vals, dtype=float))) if vals else 0.0
        return out

    finally:
        if dropout_token:
            _restore_dropout_p(model, dropout_token)
        restore_requires_grad(model, snap)
        model.train(was_training)


# ============================================================
# Deterministic Top-K selection (paper-aligned Eq. 5.20)
# ============================================================
def select_groups_by_k(
    *,
    scores: Dict[str, float],
    scope_groups: Set[str],
    k_select: int,
    force_keep: Set[str],
    stable_order: List[str],
    allow_k_zero: bool = False,
) -> Set[str]:
    """
    Paper-aligned selection:
      - head anchor is always included via force_keep
      - among remaining groups, choose Top-(K-#force_keep)
      - deterministic tie-breaking follows stable_order
    """
    scope_groups = set(scope_groups)
    force_keep = set(force_keep).intersection(scope_groups)

    non_forced = [g for g in stable_order if g in scope_groups and g not in force_keep]
    non_forced = [g for g in non_forced if g in scores]

    K = int(k_select)
    if allow_k_zero:
        K = max(0, K)
    else:
        K = max(1, K)

    # if forced groups already exceed K, keep forced only
    if K <= len(force_keep):
        return set(list(sorted(force_keep, key=lambda g: stable_order.index(g)))[:K])

    need = K - len(force_keep)

    # sort by descending score, then stable order
    order_index = {g: i for i, g in enumerate(stable_order)}
    ranked = sorted(
        non_forced,
        key=lambda g: (-float(scores.get(g, 0.0)), order_index.get(g, 10**9)),
    )

    selected = set(force_keep)
    selected.update(ranked[:need])
    return selected


def select_groups_by_ratio(
    *,
    scores: Dict[str, float],
    scope_groups: Set[str],
    ratio: float,
    force_keep: Set[str],
    stable_order: List[str],
    allow_k_zero: bool = False,
) -> Tuple[Set[str], int]:
    """
    Backward-compatible wrapper:
      K = ceil(ratio * |scope|)
    Still uses exact Top-K selection logic afterwards.
    """
    N = len([g for g in stable_order if g in scope_groups])
    r = float(np.clip(float(ratio), 0.0, 1.0))
    K = int(np.ceil(r * float(N)))
    if allow_k_zero:
        K = max(0, min(K, N))
    else:
        K = max(1, min(K, N)) if N > 0 else 0

    selected = select_groups_by_k(
        scores=scores,
        scope_groups=scope_groups,
        k_select=K,
        force_keep=force_keep,
        stable_order=stable_order,
        allow_k_zero=allow_k_zero,
    )
    return selected, int(K)


# ============================================================
# Optional helper for Stage II-B training (Theorem 6.2 support)
# ============================================================
@torch.no_grad()
def clip_gradients_per_selected_group(
    model: nn.Module,
    groups: Dict[str, List[str]],
    selected_groups: Iterable[str],
    max_norm: Optional[float],
) -> Dict[str, float]:
    """
    Optional helper to enforce per-group L2 clipping during masked ERM.
    This supports the theorem-side assumption used in the drift bound.

    Returns:
        dict[group_name] = pre-clip norm
    """
    out: Dict[str, float] = {}
    if max_norm is None or float(max_norm) <= 0.0:
        return out

    named = dict(model.named_parameters())
    for g in selected_groups:
        params: List[torch.Tensor] = []
        for pname in groups.get(g, []):
            p = named.get(pname)
            if p is not None and p.grad is not None:
                params.append(p)

        if not params:
            out[str(g)] = 0.0
            continue

        total_sq = 0.0
        for p in params:
            total_sq += float(torch.sum(p.grad.detach() * p.grad.detach()).item())
        norm = float(np.sqrt(total_sq))
        out[str(g)] = float(norm)

        if norm > float(max_norm):
            scale = float(max_norm) / (norm + 1e-12)
            for p in params:
                p.grad.mul_(scale)

    return out


# ============================================================
# One-stop runner
# ============================================================
def run_m3_selection(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    cfg: M3Config,
    scope_groups: Set[str],
    ratio: Optional[float] = None,
    k_select: Optional[int] = None,
    client_size: Optional[int] = None,
    large_threshold: Optional[int] = None,
    groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, object]:
    """
    Paper-aligned M3 selection runner.

    Preferred usage:
      - pass k_select = K_pers from APC
    Backward compatibility:
      - if k_select is None, use ratio to derive K = ceil(ratio * |scope|)

    Behavior:
      1) compute robust dimension-normalized gradient sensitivities
      2) keep head fixed as anchor
      3) select Top-(K-1) among non-head groups with deterministic tie-breaking
      4) set trainable params exactly to selected groups
    """
    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

    groups_used = groups if isinstance(groups, dict) else build_param_groups(model)
    scope_groups = {g for g in set(scope_groups) if g in groups_used}
    stable_order = get_group_order(model, groups_used)

    # force keep head
    force_keep: Set[str] = set()
    if cfg.force_keep_head:
        force_keep = infer_head_group_names(model, groups_used).intersection(scope_groups)

    if (not cfg.enabled) or (not scope_groups):
        set_trainable_by_selected_groups(model, groups_used, set())
        return {
            "scores": {},
            "selected_groups": [],
            "k_select": 0,
            "ratio": None if ratio is None else float(ratio),
            "forced_keep": sorted(list(force_keep)),
            "stable_order": stable_order,
            "skipped": True,
        }

    # determine scoring batch limit
    limit = int(cfg.grad_score_batches)
    if (
        cfg.grad_score_batches_large is not None
        and client_size is not None
        and large_threshold is not None
        and int(client_size) >= int(large_threshold)
    ):
        limit = int(cfg.grad_score_batches_large)

    # compute scores
    scores = compute_group_grad_scores_within_scope(
        model=model,
        loader=loader,
        device=device,
        groups=groups_used,
        scope_groups=scope_groups,
        limit_batches=int(limit),
        score_mode=str(cfg.score_mode),
        eps=float(cfg.eps),
        pos_weight=cfg.pos_weight,
        prefer_forward_logits=bool(cfg.prefer_forward_logits),
    )

    # exact K from APC is preferred
    if k_select is not None:
        K = int(k_select)
        if not cfg.allow_k_zero:
            K = max(1, K)
        K = min(K, len(scope_groups)) if len(scope_groups) > 0 else 0

        selected = select_groups_by_k(
            scores=scores,
            scope_groups=scope_groups,
            k_select=K,
            force_keep=force_keep,
            stable_order=stable_order,
            allow_k_zero=bool(cfg.allow_k_zero),
        )
    else:
        # backward compatibility: derive K from ratio
        rr = 0.0 if ratio is None else float(ratio)
        selected, K = select_groups_by_ratio(
            scores=scores,
            scope_groups=scope_groups,
            ratio=rr,
            force_keep=force_keep,
            stable_order=stable_order,
            allow_k_zero=bool(cfg.allow_k_zero),
        )

    # final application
    set_trainable_by_selected_groups(model, groups_used, selected)

    # convenient mask vector if model exposes stable paper order
    mask_vector = None
    if hasattr(model, "get_mask_vector") and callable(getattr(model, "get_mask_vector")):
        try:
            mask_vector = model.get_mask_vector(selected)
        except Exception:
            mask_vector = None

    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "selected_groups": sorted(list(selected), key=lambda g: stable_order.index(g) if g in stable_order else 10**9),
        "k_select": int(K),
        "ratio": None if ratio is None else float(ratio),
        "forced_keep": sorted(list(force_keep), key=lambda g: stable_order.index(g) if g in stable_order else 10**9),
        "stable_order": stable_order,
        "mask_vector": mask_vector,
        "limit_batches": int(limit),
        "score_mode": str(cfg.score_mode),
        "skipped": False,
    }






