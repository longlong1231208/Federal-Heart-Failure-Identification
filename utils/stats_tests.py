# src/utils/stats_tests.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def paired_signflip_pvalue(
    diffs: np.ndarray,
    *,
    n_perm: int = 20000,
    seed: int = 123,
    exact_if_small: bool = True,
    exact_max_n: int = 20,
) -> float:
    """
    Paired sign-flip permutation test (two-sided) for H0: mean(diffs)=0.

    diffs: shape [R] paired differences across repeats (may include NaN/Inf).
    Returns: two-sided p-value with add-one smoothing for Monte Carlo.

    Notes:
    - This is the standard randomization test for paired samples:
      flip the sign of each paired difference independently with prob 0.5.
    - If exact_if_small and R<=exact_max_n, compute exact p-value by enumerating all 2^R sign patterns.
    """
    diffs = _finite_1d(diffs)
    R = int(diffs.size)
    if R < 2:
        return float("nan")

    obs = float(np.mean(diffs))

    # Exact enumeration for small R (paper-friendly, deterministic).
    if exact_if_small and R <= int(exact_max_n):
        # Enumerate all sign vectors in {+1,-1}^R
        # Use integers 0..(2^R-1) as bitmasks.
        total = 1 << R
        count = 0
        for mask in range(total):
            # signs: +1 default, -1 where bit=1
            signs = np.ones(R, dtype=float)
            # bit trick
            bits = np.fromiter(((mask >> i) & 1 for i in range(R)), count=R, dtype=int)
            signs[bits == 1] = -1.0
            m = float(np.mean(diffs * signs))
            if abs(m) >= abs(obs):
                count += 1
        return float(count / total)

    # Monte Carlo sign-flip
    rng = np.random.default_rng(seed)
    count = 0
    n_perm = int(n_perm)

    for _ in range(n_perm):
        # Rademacher signs: ±1 equally likely, independent
        signs = rng.integers(0, 2, size=R, dtype=np.int8)  # {0,1}
        signs = (signs * 2 - 1).astype(float)             # {-1,+1}
        m = float(np.mean(diffs * signs))
        if abs(m) >= abs(obs):
            count += 1

    # add-one smoothing (avoid 0 p-value)
    return float((count + 1) / (n_perm + 1))


def bootstrap_ci_mean(
    diffs: np.ndarray,
    *,
    n_boot: int = 20000,
    ci: float = 0.95,
    seed: int = 123,
) -> Tuple[float, float]:
    """
    Bootstrap percentile CI for mean(diffs).

    diffs: shape [R], may include NaN/Inf.
    Returns: (lo, hi). If insufficient samples, returns (nan, nan).
    """
    diffs = _finite_1d(diffs)
    R = int(diffs.size)
    if R < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    idx = np.arange(R)

    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        b = rng.choice(idx, size=R, replace=True)
        means[i] = float(np.mean(diffs[b]))

    lo_q = (1.0 - float(ci)) / 2.0 * 100.0
    hi_q = (1.0 + float(ci)) / 2.0 * 100.0
    lo = float(np.percentile(means, lo_q))
    hi = float(np.percentile(means, hi_q))
    return lo, hi


def cohens_d_paired(diffs: np.ndarray) -> float:
    """
    Cohen's d for paired samples: mean(diffs) / std(diffs, ddof=1).
    """
    diffs = _finite_1d(diffs)
    if diffs.size < 2:
        return float("nan")

    s = float(np.std(diffs, ddof=1))
    if not np.isfinite(s) or s <= 1e-12:
        return float("nan")
    return float(np.mean(diffs) / s)


def summarize_paired_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_perm: int = 20000,
    n_boot: int = 20000,
    seed_perm: int = 123,
    seed_boot: int = 456,
    ci: float = 0.95,
    strict_pairing: bool = True,
) -> Dict[str, Any]:
    """
    Compare paired arrays a vs b using:
      - mean difference
      - sign-flip permutation p-value (two-sided)
      - bootstrap CI on mean diff
      - paired Cohen's d

    Pairing policy:
      - strict_pairing=True (recommended): require same length; otherwise raise.
      - strict_pairing=False: truncate to min length (use only if you are sure ordering matches).
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if strict_pairing and a.size != b.size:
        raise ValueError(f"Paired test requires same length, got a={a.size}, b={b.size}")

    n = int(min(a.size, b.size))
    diffs = a[:n] - b[:n] if n > 0 else np.asarray([], dtype=float)

    diffs_f = _finite_1d(diffs)
    mean_diff = float(np.mean(diffs_f)) if diffs_f.size else float("nan")

    p = paired_signflip_pvalue(diffs, n_perm=n_perm, seed=seed_perm)
    lo, hi = bootstrap_ci_mean(diffs, n_boot=n_boot, ci=ci, seed=seed_boot)
    d = cohens_d_paired(diffs)

    return {
        "mean_diff": mean_diff,
        "perm_pvalue_two_sided": float(p),
        "bootstrap_ci": [float(lo), float(hi)],
        "cohens_d_paired": float(d),
        "n_effective": int(diffs_f.size),
        "n_input": int(n),
    }
