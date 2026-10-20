"""
Monte Carlo finite-sample study for the DS603 project.

The DGP is a calibrated Student-t distribution.  For each sample size n and
each replication, we draw n losses from t(df, loc, scale) and compute every
estimator. Bias, SD, and RMSE are reported relative to the analytic true value.
"""

import numpy as np
import pandas as pd
from typing import Sequence
from scipy import stats

from src.estimators import (
    historical_var, historical_cvar,
    parametric_normal_var, parametric_normal_cvar,
    parametric_t_var, parametric_t_cvar,
    fit_student_t, true_t_var, true_t_cvar,
)

SEED = 42
SAMPLE_SIZES = (60, 250, 1000)
N_REPS = 1_000
ALPHAS = (0.95, 0.99)


def _estimators_for_sample(losses: np.ndarray, alpha: float) -> dict[str, float]:
    """Run all three VaR and CVaR estimators on a single loss sample."""
    try:
        t_params = fit_student_t(losses)
    except Exception:
        t_params = None

    return {
        "hist_var":  historical_var(losses, alpha),
        "hist_cvar": historical_cvar(losses, alpha),
        "norm_var":  parametric_normal_var(losses, alpha),
        "norm_cvar": parametric_normal_cvar(losses, alpha),
        "t_var":     parametric_t_var(losses, alpha, t_params) if t_params else np.nan,
        "t_cvar":    parametric_t_cvar(losses, alpha, t_params) if t_params else np.nan,
    }


def run_monte_carlo(
    df: float,
    loc: float,
    scale: float,
    sample_sizes: Sequence[int] = SAMPLE_SIZES,
    n_reps: int = N_REPS,
    alphas: Sequence[float] = ALPHAS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Run the full Monte Carlo experiment.

    Parameters
    ----------
    df, loc, scale : Student-t DGP parameters (from MLE on real data)
    sample_sizes   : iterable of n values to test
    n_reps         : number of Monte Carlo replications per (n, alpha)
    alphas         : confidence levels
    seed           : base random seed for reproducibility

    Returns
    -------
    DataFrame with columns:
        alpha, n, estimator, measure (var|cvar),
        bias, sd, rmse, true_value, mean_estimate
    """
    rng = np.random.default_rng(seed)
    records = []

    for alpha in alphas:
        true_var  = true_t_var(df, loc, scale, alpha)
        true_cvar = true_t_cvar(df, loc, scale, alpha)

        for n in sample_sizes:
            # Pre-draw all replications at once for speed
            all_draws = rng.standard_t(df=df, size=(n_reps, n)) * scale + loc

            var_cols  = {k: [] for k in ("hist_var", "norm_var", "t_var")}
            cvar_cols = {k: [] for k in ("hist_cvar", "norm_cvar", "t_cvar")}

            for rep in range(n_reps):
                sample = all_draws[rep]
                est = _estimators_for_sample(sample, alpha)
                for k in var_cols:
                    var_cols[k].append(est[k])
                for k in cvar_cols:
                    cvar_cols[k].append(est[k])

            for est_name, vals in var_cols.items():
                arr = np.array(vals, dtype=float)
                bias = float(np.nanmean(arr) - true_var)
                sd   = float(np.nanstd(arr, ddof=1))
                rmse = float(np.sqrt(np.nanmean((arr - true_var) ** 2)))
                records.append({
                    "alpha": alpha,
                    "n": n,
                    "estimator": est_name.replace("_var", ""),
                    "measure": "var",
                    "true_value": true_var,
                    "mean_estimate": float(np.nanmean(arr)),
                    "bias": bias,
                    "sd": sd,
                    "rmse": rmse,
                })

            for est_name, vals in cvar_cols.items():
                arr = np.array(vals, dtype=float)
                bias = float(np.nanmean(arr) - true_cvar)
                sd   = float(np.nanstd(arr, ddof=1))
                rmse = float(np.sqrt(np.nanmean((arr - true_cvar) ** 2)))
                records.append({
                    "alpha": alpha,
                    "n": n,
                    "estimator": est_name.replace("_cvar", ""),
                    "measure": "cvar",
                    "true_value": true_cvar,
                    "mean_estimate": float(np.nanmean(arr)),
                    "bias": bias,
                    "sd": sd,
                    "rmse": rmse,
                })

    return pd.DataFrame(records)
