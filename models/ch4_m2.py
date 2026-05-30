from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


def _safe_logit(p: float, eps: float = 1e-8) -> float:
    pp = float(np.clip(float(p), float(eps), 1.0 - float(eps)))
    return float(np.log(pp / (1.0 - pp)))


@dataclass(frozen=True)
class APCParams:
    """Paper-facing M2 parameters for reliability-adjusted personalization."""

    eps_sm: float = 0.5
    eps_num: float = 1e-8
    eps_gamma: float = 1e-8
    mapping_mode: str = "log_median"
    scope_mapping_mode: str = "floor"
    direct_gamma: float = 1.96
    gamma_s_scale: float = 1.0
    G_groups: int = 7

    K_pers_min: int = 1
    K_pers_max: int = 7
    E_pers_min: int = 3
    E_pers_max: int = 10


@dataclass
class APCOutput:
    """M2 output consumed by M3."""

    name: str
    K_pers: int
    E_pers: int
    rep_off: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            f"apc_{self.name}_K_pers": float(self.K_pers),
            f"apc_{self.name}_E_pers": float(self.E_pers),
            f"apc_{self.name}_rep_off": float(1.0 if self.rep_off else 0.0),
        }


def _base_reliability_terms(
    *,
    n_k_pos: int,
    n_k_neg: int,
    n_total_pos: int,
    n_total_neg: int,
    params: APCParams,
) -> Dict[str, float]:
    eps_sm = float(params.eps_sm)
    eps_num = float(params.eps_num)

    nkp = int(max(0, int(n_k_pos)))
    nkn = int(max(0, int(n_k_neg)))
    ntp = int(max(0, int(n_total_pos)))
    ntn = int(max(0, int(n_total_neg)))

    pi_k = float((float(nkp) + eps_sm) / (float(nkp + nkn) + 2.0 * eps_sm))
    pi_ref = float((float(ntp) + eps_sm) / (float(ntp + ntn) + 2.0 * eps_sm))

    delta_b = float(_safe_logit(pi_k) - _safe_logit(pi_ref))
    delta = float(abs(delta_b))
    sigma_delta = float(
        np.sqrt(
            1.0 / (float(nkp) + eps_sm)
            + 1.0 / (float(nkn) + eps_sm)
            + 1.0 / (float(ntp) + eps_sm)
            + 1.0 / (float(ntn) + eps_sm)
        )
    )
    q = float(delta / (sigma_delta + eps_num))
    s = float(np.log1p(max(0.0, q)))

    return {
        "pi_k": float(pi_k),
        "pi_ref": float(pi_ref),
        "delta_b": float(delta_b),
        "delta": float(delta),
        "sigma_delta": float(sigma_delta),
        "q": float(q),
        "s": float(s),
        "n_ref_pos": float(ntp),
        "n_ref_neg": float(ntn),
    }


def compute_gamma_s_from_counts(
    count_rows: Iterable[Dict[str, int]],
    *,
    params: APCParams,
) -> float:
    """Median scale gamma_s = median_j log(1 + q_j)."""

    s_values: List[float] = []
    for row in count_rows:
        terms = _base_reliability_terms(
            n_k_pos=int(row["n_k_pos"]),
            n_k_neg=int(row["n_k_neg"]),
            n_total_pos=int(row["n_total_pos"]),
            n_total_neg=int(row["n_total_neg"]),
            params=params,
        )
        s_values.append(float(terms["s"]))
    if not s_values:
        return float(params.eps_num)
    gamma_s = float(np.median(np.asarray(s_values, dtype=float)))
    return float(max(gamma_s, float(params.eps_num)))


def compute_reliability_adjusted_intensity(
    *,
    n_k_pos: int,
    n_k_neg: int,
    n_total_pos: int,
    n_total_neg: int,
    params: APCParams,
    gamma_s: Optional[float] = None,
) -> Dict[str, float]:
    """Compute q_k, s_k, and the configured M2 personalization intensity."""

    terms = _base_reliability_terms(
        n_k_pos=int(n_k_pos),
        n_k_neg=int(n_k_neg),
        n_total_pos=int(n_total_pos),
        n_total_neg=int(n_total_neg),
        params=params,
    )
    raw_gamma_s = float(terms["s"] if gamma_s is None else gamma_s)
    gamma_s_scaled = float(
        max(float(params.gamma_s_scale) * raw_gamma_s, float(params.eps_num))
    )
    s = float(max(0.0, terms["s"]))
    mapping_mode = str(getattr(params, "mapping_mode", "log_median")).strip().lower()
    if mapping_mode in {"direct", "direct_gamma", "old", "old_direct", "old_gamma"}:
        gamma_direct = float(
            max(float(getattr(params, "direct_gamma", 1.96)), float(params.eps_num))
        )
        q = float(max(0.0, terms["q"]))
        r = float(q / (q + gamma_direct)) if q > 0.0 else 0.0
        formula = "direct_gamma_reliability_prior_mismatch"
    else:
        gamma_direct = float("nan")
        r = float(s / (s + gamma_s_scaled)) if s > 0.0 else 0.0
        formula = "log_median_reliability_prior_mismatch"
    r = float(np.clip(r, 0.0, 1.0))

    return {
        **terms,
        "gamma_s": float(raw_gamma_s),
        "gamma_s_scaled": float(gamma_s_scaled),
        "direct_gamma": float(gamma_direct),
        "mapping_mode": str(mapping_mode),
        "formula": str(formula),
        "r": float(r),
    }


def _map_intensity_to_scope_depth(r: float, params: APCParams) -> Dict[str, int]:
    r = float(np.clip(float(r), 0.0, 1.0))
    g = int(max(1, int(params.G_groups)))
    k_min = int(max(1, int(params.K_pers_min)))
    k_max = int(max(k_min, int(params.K_pers_max)))
    e_min = int(max(0, int(params.E_pers_min)))
    e_max = int(max(e_min, int(params.E_pers_max)))

    scope_mapping_mode = (
        str(getattr(params, "scope_mapping_mode", "floor")).strip().lower()
    )
    if scope_mapping_mode in {"ceil", "ceiling"}:
        k_pers = int(1 + np.ceil(float(g - 1) * r - 1e-12))
    else:
        k_pers = int(1 + np.floor(float(g - 1) * r))
    k_pers = int(np.clip(k_pers, k_min, k_max))

    e_pers = int(e_min + np.floor(float(e_max - e_min) * r))
    e_pers = int(np.clip(e_pers, e_min, e_max))

    return {"K_pers": int(k_pers), "E_pers": int(e_pers)}


def compute_need_reliability_product_outputs(
    count_rows: Iterable[Dict[str, int]],
    *,
    params: APCParams,
) -> Dict[str, APCOutput]:
    """Joint M2 mode: personalization intensity = need * reliability."""

    prepared: List[Dict[str, float]] = []
    for idx, raw in enumerate(count_rows):
        client = str(raw.get("client", raw.get("client_name", idx)))
        n_k_pos = int(raw["n_k_pos"])
        n_k_neg = int(raw["n_k_neg"])
        n_total_pos = int(raw["n_total_pos"])
        n_total_neg = int(raw["n_total_neg"])
        terms = _base_reliability_terms(
            n_k_pos=n_k_pos,
            n_k_neg=n_k_neg,
            n_total_pos=n_total_pos,
            n_total_neg=n_total_neg,
            params=params,
        )
        n_k = int(max(0, n_k_pos + n_k_neg))
        prepared.append(
            {
                "client": client,
                "n_k_pos": float(n_k_pos),
                "n_k_neg": float(n_k_neg),
                "n_k": float(n_k),
                "log_n_k": float(np.log(float(n_k) + 1.0)),
                **terms,
            }
        )

    if not prepared:
        return {}

    eps_num = float(params.eps_num)
    max_s = float(max(float(row["s"]) for row in prepared))
    max_log_n = float(max(float(row["log_n_k"]) for row in prepared))

    outputs: Dict[str, APCOutput] = {}
    for row in prepared:
        u_k = float(float(row["s"]) / (max_s + eps_num)) if max_s > 0.0 else 0.0
        alpha_k = (
            float(float(row["log_n_k"]) / (max_log_n + eps_num))
            if max_log_n > 0.0
            else 0.0
        )
        u_k = float(np.clip(u_k, 0.0, 1.0))
        alpha_k = float(np.clip(alpha_k, 0.0, 1.0))
        r_k = float(np.clip(u_k * alpha_k, 0.0, 1.0))
        mapped = _map_intensity_to_scope_depth(r_k, params)
        metadata = {
            "signal_mode": "need_reliability_product",
            "controller_type": "label_shift",
            "apc_formula": "need_reliability_product",
            "mapping_mode": "need_reliability_product",
            "n_k_pos": float(row["n_k_pos"]),
            "n_k_neg": float(row["n_k_neg"]),
            "n_k": float(row["n_k"]),
            "pi_k": float(row["pi_k"]),
            "pi_ref": float(row["pi_ref"]),
            "pi_g": float(row["pi_ref"]),
            "delta_b": float(row["delta_b"]),
            "delta": float(row["delta"]),
            "sigma_delta": float(row["sigma_delta"]),
            "q_reliability": float(row["q"]),
            "s_reliability": float(row["s"]),
            "max_s_reliability": float(max_s),
            "u_k": float(u_k),
            "alpha_k": float(alpha_k),
            "r_neutral": float(r_k),
            "r_reliability": float(r_k),
            "r_final": float(r_k),
            "n_ref_pos": float(row["n_ref_pos"]),
            "n_ref_neg": float(row["n_ref_neg"]),
            "m_k": float(mapped["K_pers"]),
            "e_k": float(mapped["E_pers"]),
            "direct_gamma": float("nan"),
            "gamma_s": float("nan"),
            "gamma_s_scaled": float("nan"),
        }
        outputs[str(row["client"])] = APCOutput(
            name="neutral",
            K_pers=int(mapped["K_pers"]),
            E_pers=int(mapped["E_pers"]),
            rep_off=False,
            metadata=metadata,
        )
    return outputs


def get_apc_candidates_from_output(
    *,
    base_output: APCOutput,
    params: APCParams,
) -> Dict[str, APCOutput]:
    """Generate conservative/neutral/aggressive candidates around an output's r."""

    metadata = dict(getattr(base_output, "metadata", {}) or {})
    r_neu = float(metadata.get("r_neutral", metadata.get("r_reliability", 0.0)))
    g = int(max(2, int(params.G_groups)))
    step = float(1.0 / float(g - 1))

    def make(name: str, r_value: float) -> APCOutput:
        r = float(np.clip(float(r_value), 0.0, 1.0))
        mapped = _map_intensity_to_scope_depth(r, params)
        meta = dict(metadata)
        meta.update(
            {
                "candidate_name": str(name),
                "r_reliability": float(r),
                "r_final": float(r),
                "m_k": float(mapped["K_pers"]),
                "e_k": float(mapped["E_pers"]),
            }
        )
        return APCOutput(
            name=str(name),
            K_pers=int(mapped["K_pers"]),
            E_pers=int(mapped["E_pers"]),
            rep_off=bool(getattr(base_output, "rep_off", False)),
            metadata=meta,
        )

    return {
        "conservative": make("conservative", max(0.0, r_neu - step)),
        "neutral": make("neutral", r_neu),
        "aggressive": make("aggressive", min(1.0, r_neu + step)),
    }


def _make_output(
    *,
    name: str,
    reliability: Dict[str, float],
    r_value: float,
    params: APCParams,
) -> APCOutput:
    r = float(np.clip(float(r_value), 0.0, 1.0))
    mapped = _map_intensity_to_scope_depth(r, params)
    metadata = {
        "signal_mode": "reliability_prior",
        "apc_formula": str(
            reliability.get("formula", "log_median_reliability_prior_mismatch")
        ),
        "mapping_mode": str(reliability.get("mapping_mode", "log_median")),
        "pi_k": float(reliability["pi_k"]),
        "pi_ref": float(reliability["pi_ref"]),
        "delta_b": float(reliability["delta_b"]),
        "delta": float(reliability["delta"]),
        "sigma_delta": float(reliability["sigma_delta"]),
        "q_reliability": float(reliability["q"]),
        "s_reliability": float(reliability["s"]),
        "gamma_s": float(reliability["gamma_s"]),
        "gamma_s_scaled": float(reliability["gamma_s_scaled"]),
        "direct_gamma": float(reliability.get("direct_gamma", np.nan)),
        "r_neutral": float(reliability["r"]),
        "r_reliability": float(r),
        "n_ref_pos": float(reliability["n_ref_pos"]),
        "n_ref_neg": float(reliability["n_ref_neg"]),
        "m_k": float(mapped["K_pers"]),
        "e_k": float(mapped["E_pers"]),
    }
    return APCOutput(
        name=str(name),
        K_pers=int(mapped["K_pers"]),
        E_pers=int(mapped["E_pers"]),
        rep_off=False,
        metadata=metadata,
    )


def get_apc_output(
    *,
    params: APCParams,
    n_k_pos: int,
    n_k_neg: int,
    n_total_pos: int,
    n_total_neg: int,
    gamma_s: Optional[float] = None,
) -> APCOutput:
    """Return the neutral deterministic M2 output."""

    reliability = compute_reliability_adjusted_intensity(
        n_k_pos=int(n_k_pos),
        n_k_neg=int(n_k_neg),
        n_total_pos=int(n_total_pos),
        n_total_neg=int(n_total_neg),
        params=params,
        gamma_s=gamma_s,
    )
    return _make_output(
        name="neutral",
        reliability=reliability,
        r_value=float(reliability["r"]),
        params=params,
    )


def get_apc_candidates(
    *,
    params: APCParams,
    n_k_pos: int,
    n_k_neg: int,
    n_total_pos: int,
    n_total_neg: int,
    gamma_s: Optional[float] = None,
) -> Dict[str, APCOutput]:
    """Generate conservative/neutral/aggressive M2 candidates around r_k."""

    reliability = compute_reliability_adjusted_intensity(
        n_k_pos=int(n_k_pos),
        n_k_neg=int(n_k_neg),
        n_total_pos=int(n_total_pos),
        n_total_neg=int(n_total_neg),
        params=params,
        gamma_s=gamma_s,
    )
    g = int(max(2, int(params.G_groups)))
    step = float(1.0 / float(g - 1))
    r_neu = float(reliability["r"])

    return {
        "conservative": _make_output(
            name="conservative",
            reliability=reliability,
            r_value=max(0.0, r_neu - step),
            params=params,
        ),
        "neutral": _make_output(
            name="neutral",
            reliability=reliability,
            r_value=r_neu,
            params=params,
        ),
        "aggressive": _make_output(
            name="aggressive",
            reliability=reliability,
            r_value=min(1.0, r_neu + step),
            params=params,
        ),
    }
