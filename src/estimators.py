"""
VaR and CVaR estimators for the DS603 Monte Carlo study.

All functions operate on a 1-D array of *losses* L = -RET (positive = bad).
Confidence level alpha is stated as a probability of non-exceedance,
e.g. alpha=0.99 means VaR_{0.99}: the loss not exceeded 99% of the time.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Historical simulation
# ---------------------------------------------------------------------------

def historical_var(losses: np.ndarray, alpha: float = 0.99) -> float:
    """Empirical alpha-quantile of the loss distribution."""
    return float(np.quantile(losses, alpha))


def historical_cvar(losses: np.ndarray, alpha: float = 0.99) -> float:
    """Mean of losses that exceed the empirical VaR (Expected Shortfall)."""
    var = historical_var(losses, alpha)
    tail = losses[losses > var]
    if len(tail) == 0:
        return var  # edge case: no exceedances
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Parametric Normal
# ---------------------------------------------------------------------------

def parametric_normal_var(losses: np.ndarray, alpha: float = 0.99) -> float:
    """
    VaR under a Normal assumption using the sample mean and standard deviation.
    VaR_alpha = mu + sigma * Phi^{-1}(alpha)
    """
    mu, sigma = losses.mean(), losses.std(ddof=1)
    return float(mu + sigma * stats.norm.ppf(alpha))


def parametric_normal_cvar(losses: np.ndarray, alpha: float = 0.99) -> float:
    """
    CVaR (ES) under a Normal assumption.
    CVaR_alpha = mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)
    where phi is the standard Normal PDF and Phi^{-1} is its quantile function.
    """
    mu, sigma = losses.mean(), losses.std(ddof=1)
    z = stats.norm.ppf(alpha)
    return float(mu + sigma * stats.norm.pdf(z) / (1.0 - alpha))


# ---------------------------------------------------------------------------
# Parametric Student-t
# ---------------------------------------------------------------------------

def fit_student_t(losses: np.ndarray) -> tuple[float, float, float]:
    """
    Fit a Student-t distribution to losses by MLE.

    Returns
    -------
    df : float   — degrees of freedom (nu)
    loc : float  — location (mean when df > 1)
    scale : float — scale parameter
    """
    df, loc, scale = stats.t.fit(losses, floc=losses.mean())
    return float(df), float(loc), float(scale)


def parametric_t_var(
    losses: np.ndarray,
    alpha: float = 0.99,
    params: tuple | None = None,
) -> float:
    """
    VaR under a fitted Student-t distribution.

    Parameters
    ----------
    losses : array of losses
    alpha  : confidence level
    params : (df, loc, scale) tuple; if None, MLE is run on losses.
    """
    if params is None:
        params = fit_student_t(losses)
    df, loc, scale = params
    return float(stats.t.ppf(alpha, df=df, loc=loc, scale=scale))


def parametric_t_cvar(
    losses: np.ndarray,
    alpha: float = 0.99,
    params: tuple | None = None,
) -> float:
    """
    CVaR under a fitted Student-t distribution.

    CVaR_alpha = loc + scale * [t_{df}.pdf(t_{df}^{-1}(alpha)) / (1 - alpha)]
                 * (df + t_{df}^{-1}(alpha)^2) / (df - 1)

    This is the closed-form Expected Shortfall for a Student-t.
    """
    if params is None:
        params = fit_student_t(losses)
    df, loc, scale = params
    q = stats.t.ppf(alpha, df=df)  # quantile of standardised t
    # Closed-form ES for standardised t, then scale and shift
    es_std = (stats.t.pdf(q, df=df) / (1.0 - alpha)) * (df + q**2) / (df - 1.0)
    return float(loc + scale * es_std)


# ---------------------------------------------------------------------------
# True (analytical / large-simulation) VaR and CVaR for a given Student-t DGP
# ---------------------------------------------------------------------------

def true_t_var(df: float, loc: float, scale: float, alpha: float = 0.99) -> float:
    """Exact VaR for a t(df, loc, scale) distribution."""
    return float(stats.t.ppf(alpha, df=df, loc=loc, scale=scale))


def true_t_cvar(df: float, loc: float, scale: float, alpha: float = 0.99) -> float:
    """Exact CVaR (Expected Shortfall) for a t(df, loc, scale) distribution."""
    q = stats.t.ppf(alpha, df=df)
    es_std = (stats.t.pdf(q, df=df) / (1.0 - alpha)) * (df + q**2) / (df - 1.0)
    return float(loc + scale * es_std)


# ---------------------------------------------------------------------------
# Convenience: compute all estimators at once
# ---------------------------------------------------------------------------

ALPHAS = (0.95, 0.99)


def compute_all_estimators(
    losses: np.ndarray,
    alphas: tuple = ALPHAS,
    t_params: tuple | None = None,
) -> dict:
    """
    Returns a flat dict of {method_measure_alpha: value} for all combinations.

    t_params : pre-fitted (df, loc, scale) to avoid repeated MLE; if None, runs MLE.
    """
    if t_params is None:
        t_params = fit_student_t(losses)

    results = {}
    for alpha in alphas:
        a_str = f"{int(alpha * 100)}"
        results[f"hist_var_{a_str}"]    = historical_var(losses, alpha)
        results[f"hist_cvar_{a_str}"]   = historical_cvar(losses, alpha)
        results[f"norm_var_{a_str}"]    = parametric_normal_var(losses, alpha)
        results[f"norm_cvar_{a_str}"]   = parametric_normal_cvar(losses, alpha)
        results[f"t_var_{a_str}"]       = parametric_t_var(losses, alpha, t_params)
        results[f"t_cvar_{a_str}"]      = parametric_t_cvar(losses, alpha, t_params)
    return results
