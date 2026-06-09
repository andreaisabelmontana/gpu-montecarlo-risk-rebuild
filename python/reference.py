"""reference.py — NumPy reference implementation of the risk engine.

This is the "slow" baseline in the speedup story and, more importantly, the
*independent* implementation that validates the C/CUDA backends: it uses a
different RNG (NumPy's PCG64) yet must land on the same VaR/CVaR to within
Monte Carlo noise, which confirms the model math — not just the code — is right.

Usage:
    python reference.py [n_sims] [seed] [alpha] [--pure]

    --pure  run the naive Python-loop version (orders of magnitude slower);
            only do this for small n to feel why the C/GPU backends exist.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

# Must match src/common.c :: portfolio_demo
S0 = np.array([100, 120, 80, 60, 100], dtype=float)
MU = np.array([0.08, 0.10, 0.06, 0.05, 0.03])
SIGMA = np.array([0.20, 0.28, 0.18, 0.30, 0.08])
WEIGHTS = np.array([10, 8, 15, 5, 20], dtype=float)
CORR = np.array([
    [1.00, 0.55, 0.40, 0.30, -0.10],
    [0.55, 1.00, 0.35, 0.45, -0.05],
    [0.40, 0.35, 1.00, 0.25, 0.00],
    [0.30, 0.45, 0.25, 1.00, 0.05],
    [-0.10, -0.05, 0.00, 0.05, 1.00],
])
T = 1.0


def var_cvar(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR = alpha-quantile of losses; CVaR = mean of the worst (1-alpha) tail."""
    losses = np.sort(losses)
    idx = min(int(alpha * len(losses)), len(losses) - 1)
    var = losses[idx]
    cvar = losses[idx:].mean()
    return float(var), float(cvar)


def simulate_vectorized(n_sims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(CORR)                      # lower-triangular factor
    x = rng.standard_normal((n_sims, len(S0)))        # i.i.d. normals
    z = x @ L.T                                       # correlated normals
    drift = (MU - 0.5 * SIGMA**2) * T
    diffusion = SIGMA * np.sqrt(T) * z
    ST = S0 * np.exp(drift + diffusion)               # terminal prices
    V_T = ST @ WEIGHTS
    V_0 = float(S0 @ WEIGHTS)
    return V_0 - V_T                                  # losses


def simulate_pure_python(n_sims: int, seed: int) -> np.ndarray:
    """Deliberately naive scalar loop — the 'why we need C/CUDA' exhibit."""
    import math
    import random

    random.seed(seed)
    L = np.linalg.cholesky(CORR)
    n = len(S0)
    V_0 = float(S0 @ WEIGHTS)
    losses = []
    for _ in range(n_sims):
        x = [random.gauss(0, 1) for _ in range(n)]
        v_t = 0.0
        for i in range(n):
            zi = sum(L[i, j] * x[j] for j in range(i + 1))
            drift = (MU[i] - 0.5 * SIGMA[i] ** 2) * T
            st = S0[i] * math.exp(drift + SIGMA[i] * math.sqrt(T) * zi)
            v_t += WEIGHTS[i] * st
        losses.append(V_0 - v_t)
    return np.array(losses)


def main(argv: list[str]) -> int:
    n_sims = int(argv[1]) if len(argv) > 1 else 1_000_000
    seed = int(argv[2]) if len(argv) > 2 else 12345
    alpha = float(argv[3]) if len(argv) > 3 else 0.95
    pure = "--pure" in argv

    t0 = time.perf_counter()
    losses = simulate_pure_python(n_sims, seed) if pure else simulate_vectorized(n_sims, seed)
    elapsed = time.perf_counter() - t0

    var, cvar = var_cvar(losses, alpha)
    result = {
        "backend": "python-pure" if pure else "python-numpy",
        "n_sims": n_sims,
        "threads": 1,
        "seconds": round(elapsed, 6),
        "sims_per_sec": round(n_sims / elapsed, 1),
        "V0": round(float(S0 @ WEIGHTS), 4),
        "mean_loss": round(float(losses.mean()), 4),
        "alpha": alpha,
        "VaR": round(var, 4),
        "CVaR": round(cvar, 4),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
