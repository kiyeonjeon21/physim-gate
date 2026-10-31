"""Which interval keeps its error rate when trials cluster by training run?

This study chose the t interval behind ``cluster_by`` (see docs/comparison-semantics.md). It
needs NumPy and SciPy, which the ``dev`` extra installs, and takes about half a minute:

    python scripts/cluster_method_study.py --reps 4000

Model, on the logit scale: the success of run k of arm a on episode e is
Bernoulli(expit(mu_a + b_ak + d_e)). The episode effect d_e ~ N(0, SD_EPISODE) is shared by
every run (fixed initial states, as in LIBERO), and the run effect b_ak ~ N(0, tau) is
independent across arms. The target is the difference in success between the arms over runs
and episodes. At the margin, P(lower > m) is the wrong-FAIL rate and P(upper < m) the
wrong-PASS rate; each should be at most 2.5%.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import optimize, special, stats

E = 500
SD_EPISODE = (
    2.2  # reproduces LeBoard clean: about 15% of episodes always solved, 28% never (5 seeds)
)
MU_BASE = -0.45  # about 42% success


_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(200)


def marginal(mu: float, tau: float) -> float:
    """E[expit(mu + Z)], Z ~ N(0, sd_episode^2 + tau^2), by Gauss-Hermite quadrature."""
    sd = np.hypot(SD_EPISODE, tau)
    return float((_WEIGHTS * special.expit(mu + sd * _NODES)).sum() / _WEIGHTS.sum())


def mu_for_rate(rate: float, tau: float) -> float:
    return optimize.brentq(lambda mu: marginal(mu, tau) - rate, -10, 10, xtol=1e-12)


def seed_sd_pp(tau: float, rng: np.random.Generator) -> float:
    """Spread of a seed's success rate over a fixed, infinite episode set, in points."""
    b = rng.normal(0, tau, 4000)
    d = rng.normal(0, SD_EPISODE, 4000)
    return float(special.expit(MU_BASE + b[:, None] + d[None, :]).mean(1).std() * 100)


def simulate(k: int, tau: float, delta: float, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_base = marginal(MU_BASE, tau)
    mu_c = mu_for_rate(p_base - delta, tau)
    d = rng.normal(0, SD_EPISODE, (reps, 1, E))
    bb = rng.normal(0, tau, (reps, k, 1))
    bc = rng.normal(0, tau, (reps, k, 1))
    yb = rng.random((reps, k, E)) < special.expit(MU_BASE + bb + d)
    yc = rng.random((reps, k, E)) < special.expit(mu_c + bc + d)
    # Degradation per pair: positive is worse.
    return yb.astype(float) - yc.astype(float)


def intervals(diff: np.ndarray, z: float = 1.959964) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    reps, k, e = diff.shape
    out = {}
    # A. Pool every (seed, episode) pair as independent: what the gate does today.
    flat = diff.reshape(reps, k * e)
    m = flat.mean(1)
    se = flat.std(1, ddof=1) / np.sqrt(k * e)
    out["pooled pairs"] = (m - z * se, m + z * se)
    if k < 2:
        return out
    # B. Seed-level paired t on per-seed mean differences.
    dk = diff.mean(2)
    m = dk.mean(1)
    se = dk.std(1, ddof=1) / np.sqrt(k)
    t = stats.t.ppf(0.975, k - 1)
    out["seed t"] = (m - t * se, m + t * se)
    # C. Random-effects meta-analysis over seeds, Hartung-Knapp; D. modified HK (q >= 1).
    vk = diff.var(2, ddof=1) / e + 1e-12
    w = 1 / vk
    mu_fe = (w * dk).sum(1) / w.sum(1)
    q = (w * (dk - mu_fe[:, None]) ** 2).sum(1)
    c = w.sum(1) - (w**2).sum(1) / w.sum(1)
    tau2 = np.maximum(0, (q - (k - 1)) / c)
    wr = 1 / (vk + tau2[:, None])
    mu = (wr * dk).sum(1) / wr.sum(1)
    qhk = (wr * (dk - mu[:, None]) ** 2).sum(1) / (k - 1)
    se = np.sqrt(qhk / wr.sum(1))
    out["HK"] = (mu - t * se, mu + t * se)
    se = np.sqrt(np.maximum(qhk, 1) / wr.sum(1))
    out["modified HK"] = (mu - t * se, mu + t * se)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=4000)
    args = parser.parse_args()
    margin = 0.02
    print(
        "| tau (logit) | runs | method | wrong FAIL at margin | wrong PASS at margin "
        "| FAIL at margin + 4 points | mean width (points) |"
    )
    print("|---|---|---|---|---|---|---|")
    for tau in (0.0, 0.1, 0.3):
        spread = seed_sd_pp(tau, np.random.default_rng(5))
        print(f"<!-- tau {tau}: sd of a run's success rate {spread:.1f} points -->")
        for k in (1, 3, 5, 10):
            at = intervals(simulate(k, tau, margin, args.reps, 11))
            far = intervals(simulate(k, tau, margin + 0.04, args.reps, 12))
            for name, (lo, hi) in at.items():
                flo, _ = far[name]
                wrong_fail, wrong_pass = np.mean(lo > margin), np.mean(hi < margin)
                print(
                    f"| {tau} | {k} | {name} | {wrong_fail:.3f} | {wrong_pass:.3f} "
                    f"| {np.mean(flo > margin):.3f} | {np.mean(hi - lo) * 100:.1f} |"
                )


if __name__ == "__main__":
    main()
