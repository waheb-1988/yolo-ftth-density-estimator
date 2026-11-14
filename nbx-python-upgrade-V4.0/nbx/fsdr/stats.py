"""Statistics shared by evaluation, uplift, offer arbitration and reports (V1.0, review #2 #3 #4 #10 #13).

    off-policy value      ips_snips()      IPS / SNIPS of a policy given as action probabilities per row
                          paired_bootstrap()  CI of (challenger - champion) on the same logged rows
    logistic regression   LogReg           L2-penalised, standardised inputs, scipy L-BFGS (no sklearn)
    two-arm tests         diff_ci()        difference of proportions / means with a normal 95 % CI and p-value
                          mde()            minimum detectable uplift for given arm sizes (power analysis)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from nbx.fsdr import config


# ---------------------------------------------------------------------------
# Off-policy evaluation
# ---------------------------------------------------------------------------
def ips_snips(pi_logged: np.ndarray, rewards: np.ndarray, propensities: np.ndarray) -> tuple[float, float]:
    """pi_logged[i] = probability the evaluated policy gives to the action logged on row i.
    IPS = mean(pi / p * r); SNIPS = sum(pi / p * r) / sum(pi / p)."""
    w = np.asarray(pi_logged, float) / np.asarray(propensities, float)
    r = np.asarray(rewards, float)
    if not len(w):
        return 0.0, 0.0
    return float(np.mean(w * r)), float((w * r).sum() / w.sum()) if w.sum() > 0 else 0.0


def bootstrap_snips(pi_logged, rewards, propensities, samples: int = config.BOOTSTRAP_SAMPLES,
                    confidence: float = config.CONFIDENCE, seed: int = 0) -> tuple[float, float]:
    w = np.asarray(pi_logged, float) / np.asarray(propensities, float)
    r = np.asarray(rewards, float)
    if len(w) < 2:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(seed).integers(0, len(w), (samples, len(w)))
    num, den = (w * r)[idx].sum(axis=1), w[idx].sum(axis=1)
    est = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    a = (1 - confidence) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


def paired_bootstrap(pi_new, pi_old, rewards, propensities, samples: int = config.BOOTSTRAP_SAMPLES,
                     confidence: float = config.CONFIDENCE, seed: int = 0) -> dict:
    """SNIPS(new) - SNIPS(old) on the same rows, resampled together: the CI of the difference, not of each."""
    p = np.asarray(propensities, float)
    w1, w0, r = np.asarray(pi_new, float) / p, np.asarray(pi_old, float) / p, np.asarray(rewards, float)
    n = len(r)
    if n < 2:
        return {"n": n, "delta": None, "ci": [None, None]}

    def snips(w, ix):
        den = w[ix].sum(axis=-1)
        return np.where(den > 0, (w * r)[ix].sum(axis=-1) / np.where(den > 0, den, 1), 0.0)

    full = np.arange(n)
    idx = np.random.default_rng(seed).integers(0, n, (samples, n))
    d = snips(w1, idx) - snips(w0, idx)
    a = (1 - confidence) / 2
    return {"n": int(n), "delta": round(float(snips(w1, full) - snips(w0, full)), 4),
            "ci": [round(float(np.quantile(d, a)), 4), round(float(np.quantile(d, 1 - a)), 4)]}


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
@dataclass
class LogReg:
    """P(y = 1 | x) = sigmoid(b + x_std . w), L2 penalty l2 on w. Columns with zero variance are kept (weight 0)."""
    l2: float = 1.0
    columns: list[str] = field(default_factory=list)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    coef: np.ndarray | None = None
    intercept: float = 0.0
    n: int = 0
    positives: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, columns: list[str] | None = None) -> "LogReg":
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.columns = list(columns or [f"x{i}" for i in range(X.shape[1])])
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)
        self.std = np.where(self.std > 1e-9, self.std, 1.0)
        Z = (X - self.mean) / self.std
        n, k = Z.shape

        def loss(theta):
            b, w = theta[0], theta[1:]
            z = b + Z @ w
            nll = np.sum(np.logaddexp(0, z) - y * z) / n + 0.5 * self.l2 * (w @ w) / n
            g = 1 / (1 + np.exp(-z)) - y
            return nll, np.concatenate([[g.sum() / n], (Z.T @ g) / n + self.l2 * w / n])

        p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        res = optimize.minimize(loss, np.concatenate([[math.log(p0 / (1 - p0))], np.zeros(k)]), jac=True,
                                method="L-BFGS-B", options={"maxiter": 500})
        self.intercept, self.coef = float(res.x[0]), res.x[1:]
        self.n, self.positives = int(n), int(y.sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("model not fitted")
        # row-wise sum, not a matrix product: a row gets the same value alone or in a batch of any size
        z = self.intercept + (((np.asarray(X, float) - self.mean) / self.std) * self.coef).sum(axis=1)
        return 1 / (1 + np.exp(-z))

    def weights(self) -> dict[str, float]:
        """Effect of +1 standard deviation of each input on the log-odds (readable, for reports)."""
        return {c: round(float(w), 4) for c, w in zip(self.columns, self.coef)}


# ---------------------------------------------------------------------------
# Two-arm comparisons (report) and power
# ---------------------------------------------------------------------------
def diff_ci(x_t: np.ndarray, x_c: np.ndarray, confidence: float = config.CONFIDENCE) -> dict:
    """Difference of means treated - control (a proportion when x is 0/1), Welch normal CI and two-sided p."""
    x_t, x_c = np.asarray(x_t, float), np.asarray(x_c, float)
    if len(x_t) < 2 or len(x_c) < 2:
        return {"delta": None, "ci": [None, None], "p_value": None}
    d = x_t.mean() - x_c.mean()
    se = math.sqrt(x_t.var(ddof=1) / len(x_t) + x_c.var(ddof=1) / len(x_c))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = 2 * (1 - stats.norm.cdf(abs(d) / se)) if se > 0 else (0.0 if d else 1.0)
    return {"delta": round(float(d), 4), "ci": [round(float(d - z * se), 4), round(float(d + z * se), 4)],
            "p_value": round(float(p), 4)}


def mde(p_control: float, n_t: int, n_c: int, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Smallest take-rate uplift detectable with these arm sizes (two-sided alpha, given power)."""
    if n_t < 2 or n_c < 2:
        return None
    p = min(max(p_control, 1e-4), 1 - 1e-4)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return round(float(z * math.sqrt(p * (1 - p) * (1 / n_t + 1 / n_c))), 4)


def control_size_for(p_control: float, uplift: float, n_t: int, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """Control lines needed to detect `uplift` against an arm of n_t lines (None if n_t alone cannot)."""
    p = min(max(p_control, 1e-4), 1 - 1e-4)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    inv_c = (uplift / z) ** 2 / (p * (1 - p)) - 1 / n_t
    return int(math.ceil(1 / inv_c)) if inv_c > 0 else None
