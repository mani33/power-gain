# -*- coding: utf-8 -*-
"""
Sensitivity of n* for ONE observed pair of datasets, and the ceiling N*.
Uses utils.py only.

Procedure (simple version):
  1. k is fixed (from the calibration step).
  2. observed n* from the real x̄_0, x̄_h, variances and sample sizes.
  3. redraw x̄_0*, x̄_h* around the observed values (N(x̄, var/n)), compute lambda and n*
     for each draw, and summarise.
  N* = min(observed n*, median of the draws).

n* = n_0 + lambda * (var_0 / var_h) * n_h,   lambda = (A * P)^k

This is a sensitivity check around the observed values. It gives no coverage guarantee for
the true means; protection against over-borrowing comes from the calibration of k.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut


def _lambda(xbar_0, xbar_h, var_0, var_h, n_0, n_h, k, use_sd_scale):
    A = ut.get_overlap(xbar_0, xbar_h, var_0, var_h)
    P = ut.get_pvalue(xbar_0, xbar_h, var_0, var_h, n_0, n_h, use_sd_scale=use_sd_scale)
    return (A * P) ** k


def get_n_star(xbar_0, xbar_h, var_0, var_h, n_0, n_h, k, n_draws=2000, seed=2026,
               use_sd_scale=False, sample_variances=False, margin=0.10):
    """
    Inputs:
        xbar_0, xbar_h - observed means of the drone and external data
        var_0, var_h - observed (sample) variances
        n_0, n_h - sample sizes
        k - exponent in lambda = (A * P)^k
        n_draws - number of redraws
        seed - random seed
        use_sd_scale - False: P on the SE scale (variances / n); True: SD scale
        sample_variances - False: variances fixed at the observed values in the redraws;
                           True: also redraw them from chi-square (as get_conf_int does)
        margin - the share of draws with n* above (1 + margin) * n_0 is reported
    Output (dict):
        observed_lambda, observed_n_star
        median, p10, p90 - of n* over the redraws; mean_n_star
        ci95_low, ci95_high - 2.5th and 97.5th percentiles of n* over the redraws
        share_borrowing - share of draws with n* > (1 + margin) * n_0
        N_star - min(observed_n_star, median)
        d_star, n_star_draws, lambda_draws - arrays of the redraws (for plotting)
    """
    ratio = var_0 / var_h
    lam_obs = _lambda(xbar_0, xbar_h, var_0, var_h, n_0, n_h, k, use_sd_scale)
    n_star_obs = n_0 + lam_obs * ratio * n_h

    np.random.seed(seed)
    se_0, se_h = np.sqrt(var_0 / n_0), np.sqrt(var_h / n_h)
    d_star = np.empty(n_draws)
    n_star = np.empty(n_draws)
    lam_draws = np.empty(n_draws)
    for i in range(n_draws):
        x0 = np.random.normal(xbar_0, se_0)
        xh = np.random.normal(xbar_h, se_h)
        if sample_variances:
            v0 = ut.get_sampled_variance(n_0, var_0)
            vh = ut.get_sampled_variance(n_h, var_h)
        else:
            v0, vh = var_0, var_h
        lam = _lambda(x0, xh, v0, vh, n_0, n_h, k, use_sd_scale)
        d_star[i] = xh - x0
        lam_draws[i] = lam
        n_star[i] = n_0 + lam * (v0 / vh) * n_h

    med = np.median(n_star)
    return {
        "observed_lambda": lam_obs,
        "observed_n_star": n_star_obs,
        "median": med,
        "p10": np.percentile(n_star, 10),
        "p90": np.percentile(n_star, 90),
        "ci95_low": np.percentile(n_star, 2.5),
        "ci95_high": np.percentile(n_star, 97.5),
        "mean_n_star": n_star.mean(),
        "share_borrowing": np.mean(n_star > (1 + margin) * n_0),
        "N_star": min(n_star_obs, med),
        "d_star": d_star,
        "lambda_draws": lam_draws,
        "n_star_draws": n_star,
    }


def get_n_star_from_data(drone_values, external_values, k, **kwargs):
    """Same as get_n_star, starting from the raw values (1-D arrays) of each dataset."""
    x0, xh = np.asarray(drone_values, float), np.asarray(external_values, float)
    return get_n_star(x0.mean(), xh.mean(), x0.var(ddof=1), xh.var(ddof=1),
                      len(x0), len(xh), k, **kwargs)


def plot_n_star(res, n_0, var_0, n_h, var_h, xbar_0, xbar_h, save_path=None):
    """Scatter of n* against the redrawn difference d*, in SEs of the difference."""
    se_diff = np.sqrt(var_0 / n_0 + var_h / n_h)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(res["d_star"] / se_diff, res["n_star_draws"], s=6, alpha=0.4, color="#2a6fb0")
    ax.axvline((xbar_h - xbar_0) / se_diff, color="#c8402f", ls="--", label="observed difference")
    ax.axhline(res["N_star"], color="k", ls=":", label=f"N* = {res['N_star']:.0f}")
    ax.set_xlabel("redrawn difference d* (SEs of the difference)")
    ax.set_ylabel("n*")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def plot_n_star_hist(res, n_0, bins=60, log_y=True, save_path=None):
    """
    Histogram of the n* values from the redraws, with vertical lines for the mean, the
    median and the 95% interval (2.5th to 97.5th percentile of the redraws).
    log_y - log-scale counts, so that the long tail stays visible next to the large
            pile of draws at n_0 (set False for a plain count axis)
    """
    n_star = res["n_star_draws"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(n_star, bins=bins, color="#9aa5b1", edgecolor="white", log=log_y)
    ax.axvline(res["mean_n_star"], color="#c8402f", lw=2, label=f"mean = {res['mean_n_star']:.0f}")
    ax.axvline(res["median"], color="#2a6fb0", lw=2, ls="--", label=f"median = {res['median']:.0f}")
    ax.axvline(res["ci95_low"], color="k", lw=1.5, ls=":",
               label=f"95% interval = [{res['ci95_low']:.0f}, {res['ci95_high']:.0f}]")
    ax.axvline(res["ci95_high"], color="k", lw=1.5, ls=":")
    ax.set_xlabel("n* (effective sample size) over the redraws")
    ax.set_ylabel("number of draws" + (" (log scale)" if log_y else ""))
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def make_xbar_h_grid(xbar_0, var_0, var_h, n_0, n_h, max_se=4.0, step_se=0.5):
    """External means in fixed steps of SE_diff around x̄_0: x̄_0 + j * step_se * SE_diff,
    for j from -max_se/step_se to +max_se/step_se. SE_diff = sqrt(var_0/n_0 + var_h/n_h)."""
    se_diff = np.sqrt(var_0 / n_0 + var_h / n_h)
    m = int(round(max_se / step_se))
    return xbar_0 + step_se * se_diff * np.arange(-m, m + 1)


def get_xbar_h_table(xbar_0, var_0, var_h, n_0, n_h, k, xbar_h_values,
                     xbar_h_observed=None, n_draws=2000, seed=2026,
                     use_sd_scale=False, sample_variances=False):
    """
    Table of lambda and n* as the external mean varies, with x̄_0 and the variances fixed.
    Each column treats its x̄_h as the "observed" external mean: the observed lambda and n*
    are computed from it, and the redraws are x̄_0* ~ N(x̄_0, var_0/n_0) and
    x̄_h* ~ N(that x̄_h, var_h/n_h) (see get_n_star). Every column uses the same seed.

    Inputs:
        xbar_0 - observed drone mean (fixed)
        var_0, var_h, n_0, n_h, k - as in get_n_star
        xbar_h_values - iterable of external means (e.g. from make_xbar_h_grid)
        xbar_h_observed - your actual observed x̄_h; added as its own column
                          (the is_observed row is 1 there)
        n_draws, seed, use_sd_scale, sample_variances - as in get_n_star
    Output:
        DataFrame, one column per x̄_h (sorted), rows:
            xbar_h, diff_in_SEs, is_observed,
            observed_lambda, observed_n_star,
            lambda_median, lambda_mean, lambda_ci95_low, lambda_ci95_high,
            n_star_median, n_star_mean, n_star_ci95_low, n_star_ci95_high
        The 95% interval is the 2.5th to 97.5th percentile of the redraws.
    """
    values = [float(v) for v in xbar_h_values]
    if xbar_h_observed is not None:
        values.append(float(xbar_h_observed))
    values = sorted(set(np.round(values, 10)))
    se_diff = np.sqrt(var_0 / n_0 + var_h / n_h)
    cols = {}
    for xbar_h in values:
        r = get_n_star(xbar_0, xbar_h, var_0, var_h, n_0, n_h, k, n_draws=n_draws,
                       seed=seed, use_sd_scale=use_sd_scale,
                       sample_variances=sample_variances)
        lam, ns = r["lambda_draws"], r["n_star_draws"]
        cols[xbar_h] = {
            "xbar_h": xbar_h,
            "diff_in_SEs": (xbar_h - xbar_0) / se_diff,
            "is_observed": float(xbar_h_observed is not None
                                 and np.isclose(xbar_h, xbar_h_observed)),
            "observed_lambda": r["observed_lambda"],
            "observed_n_star": r["observed_n_star"],
            "lambda_median": np.median(lam),
            "lambda_mean": lam.mean(),
            "lambda_ci95_low": np.percentile(lam, 2.5),
            "lambda_ci95_high": np.percentile(lam, 97.5),
            "n_star_median": np.median(ns),
            "n_star_mean": ns.mean(),
            "n_star_ci95_low": np.percentile(ns, 2.5),
            "n_star_ci95_high": np.percentile(ns, 97.5),
        }
    return pd.DataFrame(cols)


#%% Run
if __name__ == "__main__":
    # ---- ILLUSTRATIVE numbers: replace with your observed values ----
    XBAR_0, XBAR_H = 1.75, 1.77
    VAR_0, VAR_H = 2.0, 2.0
    N_0, N_H = 2000, 98000
    K = 20

    res = get_n_star(XBAR_0, XBAR_H, VAR_0, VAR_H, N_0, N_H, K)
    print(f"k = {K}")
    print(f"observed lambda = {res['observed_lambda']:.3g},  observed n* = {res['observed_n_star']:.1f}")
    print(f"redraws: median = {res['median']:.1f}, p10 = {res['p10']:.1f}, p90 = {res['p90']:.1f}, "
          f"mean = {res['mean_n_star']:.1f}")
    print(f"share of draws with n* > 1.1 * n_0: {res['share_borrowing']:.3f}")
    print(f"95% interval of n*: [{res['ci95_low']:.1f}, {res['ci95_high']:.1f}]")
    print(f"N* = {res['N_star']:.1f}")
    plot_n_star(res, N_0, VAR_0, N_H, VAR_H, XBAR_0, XBAR_H, save_path="n_star_sensitivity.png")
    plot_n_star_hist(res, N_0, save_path="n_star_histogram.png")

    # ---- table over a grid of external means (fixed steps of SE_diff around x̄_0) ----
    grid = make_xbar_h_grid(XBAR_0, VAR_0, VAR_H, N_0, N_H, max_se=4.0, step_se=0.5)
    table = get_xbar_h_table(XBAR_0, VAR_0, VAR_H, N_0, N_H, K, grid,
                             xbar_h_observed=XBAR_H)
    pd.set_option("display.width", 300, "display.max_columns", 40,
                  "display.float_format", lambda v: f"{v:.4g}")
    print(table)
    table.to_csv("xbar_h_table.csv")
    plt.show()
