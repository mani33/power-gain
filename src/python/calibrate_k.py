# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:00:58 2026

@author: Mani Subramaniyan
Calibrate the K value in the calculation of lambda, the exponent of power prior
described in Shi, Li & Liu 2023, SMMR 32(3):493-508

"""

import numpy as np
import matplotlib.pyplot as plt
import utils as ut
#%% Functions

def get_conf_int(k, mu_0_true, gap, var_0, var_h, n_0, n_h, use_P=False,
                 use_sd_scale=False, known_var=False, var_overlap=None,
                 mu_1_true=None, var_1=None, n_1=None):
    """
    Get 95% CI for the posterior mean when we simulate different scenarios of
    separation between current and historical/external data
    Inputs:
        k - candidate calibration exponent in lambda = (A * P)^ k
        mu_0_true - the true "current data" mean used to generate the simulated data;
                    it equals mu_null (the benchmark mean) for Type I error analysis
                    and equals mu_null + delta for Power analysis. Remember, this
                    true mean is used for simulation purposes only, so its absolute
                    value can be anything, including an actually measured mean value.
                    What matters is its offset from mu_null, which the caller sets
                    and the function never sees.
        gap - gap is in current data's SD units, so the true external mean is
                a shifted version of mu_0_true (mu_0_true + gap × sqrt(var_0)).
        var_0 - variance of current dataset
        var_h - variance of historical dataset
        n_0 - number of samples of current dataset
        n_h - number of samples of external or historic dataset
        use_P - boolean; if true, we will use lambda = (A * P)^k; otherwise
                will use lambda = A^k
    Optional inputs (the defaults reproduce the single-group analysis exactly):
        use_sd_scale - passed to ut.get_pvalue; True (as printed in Shi et al.)
                scales the mean difference by sqrt(var_h + var_0); False scales it
                by sqrt(var_h/n_h + var_0/n_0). Only used if use_P is True
        known_var - if True, the true variances var_0, var_h (and var_1) are used
                as they are, instead of sampled variances. Shi et al.'s Figures 2
                and 3 use known variances
        var_overlap - if given, the area A is computed for two normal curves that
                both have this variance (the sampled means are still used). None
                uses the variances of the two datasets. Setting var_overlap = 2.0
                (= var_h + var_0 when both are 1) reproduces Figures 2 and 3 of
                Shi et al.; this was found by fitting their curves and is not
                stated in their text
        mu_1_true, var_1, n_1 - true mean, variance and sample size of a treatment
                arm. If mu_1_true is given, the function simulates the treatment
                arm and returns the CI for the DIFFERENCE (treatment - current
                control) instead of the CI for the current-data mean. The CI
                is (x1bar - posterior mean) +/- 1.96 * sqrt(var_1/n_1 + posterior
                variance), as in the test statistic of Shi et al.
    Output:
        ci - list of size 2; confidence interval

    """

    """
    Assuming mu_0_true as the true population mean and var_0 as the true population
    variance, sample two things - a sample mean and a sample variance - for
    both current data and history data
    """
    # Sample means
    se_0 = np.sqrt(var_0/n_0)
    se_h = np.sqrt(var_h/n_h)
    # r suffix - for emphasizing that this is randomly sampled mean
    xbar_0_r = np.random.normal(mu_0_true, se_0)
    mu_h_true = mu_0_true + gap * np.sqrt(var_0)
    xbar_h_r = np.random.normal(mu_h_true, se_h)
    # Sample variances (or the known true variances)
    if known_var:
        var_0_r, var_h_r = var_0, var_h
    else:
        var_0_r = ut.get_sampled_variance(n_0, var_0)
        var_h_r = ut.get_sampled_variance(n_h, var_h)
    # Treatment arm (optional); drawn after the control data so that the random
    # numbers of the single-group analysis are not changed
    if mu_1_true is not None:
        xbar_1_r = np.random.normal(mu_1_true, np.sqrt(var_1/n_1))
        var_1_r = var_1 if known_var else ut.get_sampled_variance(n_1, var_1)
    # Compute overlap of the two normal distributions created by the sampled
    # means and variances
    if var_overlap is None:
        A = ut.get_overlap(xbar_0_r, xbar_h_r, var_0_r, var_h_r) # overlap
    else:
        A = ut.get_overlap(xbar_0_r, xbar_h_r, var_overlap, var_overlap)
    P = 1.0
    if use_P:
        # Get p-value for a two-sided z-test for a null hypothesis that
        # mu_0_true = mu_h_true
        P = ut.get_pvalue(xbar_0_r, xbar_h_r, var_0_r, var_h_r, n_0, n_h,
                          use_sd_scale=use_sd_scale)
    # Compute lambda
    lambda_r = (A * P)**k
    # Compute posterior mean and variance
    xbar_n_r = ut.get_posterior_mean(n_0, n_h, var_0_r, var_h_r,
                                    lambda_r, xbar_0_r, xbar_h_r)
    var_n_r = ut.get_posterior_variance(n_0, n_h, var_0_r, var_h_r, lambda_r)

    # Confidence interval
    if mu_1_true is None:
        center, var_ci = xbar_n_r, var_n_r
    else:                       # CI for (treatment - borrowed control)
        center, var_ci = xbar_1_r - xbar_n_r, var_1_r/n_1 + var_n_r
    ci = (np.sqrt(var_ci) * np.array([-1, 1]) * 1.96) + center

    return ci


def run_ci_grid(mu_null, delta, gaps, ks, var_0, var_h, n_0, n_h, n_rep,
                seed=2026, two_arm=False, var_1=None, n_1=None, **ci_options):
    """
    Call get_conf_int for every analysis, gap, k and replicate and keep the CIs.
    No type I error or power is computed here (see get_fraction_outside).

    The same random seed is set before every (analysis, gap, k) cell. Because
    get_conf_int draws its random numbers in the same order on every call,
    replicate r then sees the same underlying noise in every cell. Differences
    between k values (and between gaps) are therefore not blurred by Monte
    Carlo noise.

    Inputs:
        mu_null - benchmark mean claimed by the null hypothesis (single-group
                  analysis). In the two-arm analysis it is the true mean of the
                  current control arm, and the null value to check is 0
        delta - difference to detect. Single-group: the power analysis uses
                mu_0_true = mu_null + delta. Two-arm: the power analysis uses
                a true treatment mean of mu_null + delta (control stays mu_null)
        gaps - list of gap values (current data's SD units)
        ks - list of candidate exponents k
        var_0, var_h, n_0, n_h - as in get_conf_int
        n_rep - number of replicates R per cell
        seed - random seed
        two_arm - if True, simulate a treatment arm (var_1, n_1) and return CIs
                  for the difference (treatment - current control)
        var_1, n_1 - variance and sample size of the treatment arm (two_arm only)
        **ci_options - passed to get_conf_int (use_P, use_sd_scale, known_var,
                  var_overlap)
    Output:
        ci - array of shape (2, len(gaps), len(ks), n_rep, 2)
             axis 0: 0 = Type I error analysis
                     1 = Power analysis
             last axis: 0 = lower limit, 1 = upper limit
    """
    if two_arm:
        # control true mean is always mu_null; treatment differs by 0 or delta
        mu_0_true_by_analysis = [mu_null, mu_null]
        mu_1_true_by_analysis = [mu_null, mu_null + delta]
    else:
        mu_0_true_by_analysis = [mu_null, mu_null + delta]
        mu_1_true_by_analysis = [None, None]
    ci = np.empty((2, len(gaps), len(ks), n_rep, 2))
    for a in range(2):
        arm = {} if not two_arm else dict(mu_1_true=mu_1_true_by_analysis[a],
                                          var_1=var_1, n_1=n_1)
        for g, gap in enumerate(gaps):
            for j, k in enumerate(ks):
                np.random.seed(seed)
                for r in range(n_rep):
                    ci[a, g, j, r, :] = get_conf_int(
                        k, mu_0_true_by_analysis[a], gap, var_0, var_h, n_0, n_h,
                        **arm, **ci_options)
        print(f"  finished analysis {a + 1} of 2 "
              f"({'Type I error' if a == 0 else 'Power'})")

    return ci


def get_fraction_outside(ci, mu_null, alternative="two-sided"):
    """
    Fraction of replicates in which the null value lies outside the CI, counting
    only the side that matters for the test.
    The fraction inside is 1 minus this value (two-sided).
    Inputs:
        ci - array from run_ci_grid, shape (2, n_gaps, n_ks, n_rep, 2)
        mu_null - null value: the benchmark mean (single group), or 0 for the
                  treatment - control difference (two-arm)
        alternative - "two-sided": null value below the lower limit or above the
                      upper limit (95% CI, alpha = 0.05)
                      "greater": H0: mean <= null value versus Ha: mean > null
                      value; reject if the null value is below the lower limit
                      "less": reject if the null value is above the upper limit
                      The limits are the same +/- 1.96 limits, so a one-sided
                      test uses alpha = 0.025
    Output:
        frac_outside - array of shape (2, n_gaps, n_ks)
            [0] = Type I error (null is true, so outside = false positive)
            [1] = Power (null is false, so outside = correct detection)
    """
    if alternative == "two-sided":
        outside = (mu_null < ci[..., 0]) | (mu_null > ci[..., 1])
    elif alternative == "greater":
        outside = mu_null < ci[..., 0]
    elif alternative == "less":
        outside = mu_null > ci[..., 1]
    else:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")

    return outside.mean(axis=-1)


def plot_results(gaps, ks, type1, power, alpha=0.05, nominal_power=0.80,
                 zoom=0.1, save_path=None,
                 xlabel="gap = (true external mean - true subset mean) / subset SD"):
    """
    Plot Type I error and power against gap, one line per k, in the layout of
    Figures 2 and 3 of Shi, Li & Liu 2023 (left: wide range; right: zoom).
    Inputs:
        gaps - 1-D array of gap values
        ks - list of k values (k = 0 means lambda = 1, i.e. full pooling)
        type1 - array (n_gaps, n_ks) of Type I error
        power - array (n_gaps, n_ks) of power
        alpha - nominal Type I error (dashed reference line)
        nominal_power - power without borrowing (dashed reference line)
        zoom - half-width of the gap range shown in the right-hand panels
        save_path - if given, the figure is saved there
        xlabel - label of the horizontal axis
    """
    gaps = np.asarray(gaps)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for j, k in enumerate(ks):
        color = 'k' if k == 0 else colors[j]
        label = "k = 0 (full pooling)" if k == 0 else f"k = {k}"
        for col in range(2):
            axs[0, col].plot(gaps, type1[:, j], color=color, label=label)
            axs[1, col].plot(gaps, power[:, j], color=color)
    in_zoom = np.abs(gaps) <= zoom
    for col in range(2):
        axs[0, col].axhline(alpha, color='gray', ls='--', lw=1)
        axs[1, col].axhline(nominal_power, color='gray', ls='--', lw=1)
        axs[1, col].set_xlabel(xlabel)
        axs[0, col].set_ylabel("Type I error")
        axs[1, col].set_ylabel("Power")
    axs[0, 0].set_title("Type I error, wide range")
    axs[0, 1].set_title("Type I error, zoom")
    axs[1, 0].set_title("Power, wide range")
    axs[1, 1].set_title("Power, zoom")
    for row, data in ((0, type1), (1, power)):
        axs[row, 1].set_xlim(-zoom, zoom)
        lo, hi = data[in_zoom].min(), data[in_zoom].max()
        pad = 0.05 * (hi - lo + 1e-9)
        axs[row, 1].set_ylim(lo - pad, hi + pad)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='lower center',
               ncol=min(len(ks), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


#%% Run
if __name__ == "__main__":
    # "benchmark": single-group CI check, two-sided (your calibration)
    # "shi_fig2": Shi et al. Figure 2 setup, lambda = A^k, one-sided two-arm test
    # "shi_fig3": Shi et al. Figure 3 setup, lambda = (A*P)^k, one-sided two-arm test
    # "shi_fig3_corrected": same as shi_fig3 but P scaled by sample sizes
    MODE = "benchmark" # "shi_fig3"
    N_REP = 1000                 # replicates per (analysis, gap, k) cell
    SEED = 2026

    if MODE == "benchmark":
        # ---- Settings: replace with values from your own data ----
        MU_NULL = 1.75               # benchmark mean (e.g., observed subset mean)
        VAR_0 = 2.0                  # subset variance (illustrative: SD = 2)
        VAR_H = 2.0                  # external variance
        N_0, N_H = 2000, 98000       # subset and external sample sizes
        DELTA = 2.8 * np.sqrt(VAR_0 / N_0)   # ~80% power without borrowing
        KS = [0, 1, 2, 3, 5, 8, 10, 20, 50]  # k = 0: lambda = 1 (full pooling)
        GAPS = np.unique(np.round(np.concatenate([
            np.linspace(-0.3, -0.1, 5),
            np.linspace(-0.1, 0.1, 21),
            np.linspace(0.1, 0.3, 5)]), 4))  # in subset-SD units
        run_kw = dict(use_P=True, use_sd_scale=False)                 # lambda = (A*P)^k
        TWO_ARM, ALT, NULL_VALUE = False, "two-sided", MU_NULL
        ALPHA_LINE, ZOOM = 0.05, 0.1
        XLABEL = "gap = (true external mean - true subset mean) / subset SD"
    else:
        # ---- Shi et al. (2023) simulation setup ----
        MU_NULL = 0.0                # true mean of the current control arm
        VAR_0 = VAR_H = VAR_1 = 1.0  # known variances
        N_0 = N_H = N_1 = 175
        DELTA = 0.3                  # power analysis: treatment mean = 0.3
        KS = [1, 2, 3]
        GAPS = np.unique(np.round(np.concatenate([
            np.arange(-3, 3.0001, 0.1),
            np.arange(-0.3, 0.3001, 0.02)]), 4))
        run_kw = dict(two_arm=True, var_1=VAR_1, n_1=N_1, known_var=True,
                      var_overlap=2.0,
                      use_P=(MODE != "shi_fig2"),
                      use_sd_scale=(MODE != "shi_fig3_corrected"))
        TWO_ARM, ALT, NULL_VALUE = True, "greater", 0.0   # H0: treatment - control <= 0
        ALPHA_LINE, ZOOM = 0.025, 0.3
        XLABEL = "mu_h - mu_0"

    print(f"MODE = {MODE}: running {2 * len(GAPS) * len(KS) * N_REP} CI simulations...")
    ci = run_ci_grid(MU_NULL, DELTA, GAPS, KS, VAR_0, VAR_H, N_0, N_H,
                     N_REP, seed=SEED, **run_kw)
    frac = get_fraction_outside(ci, NULL_VALUE, alternative=ALT)
    type1, power = frac[0], frac[1]

    print("\n    k   worst Type I   power at gap = 0")
    i0 = int(np.argmin(np.abs(GAPS)))
    for j, k in enumerate(KS):
        print(f"{k:>5} {type1[:, j].max():>13.3f} {power[i0, j]:>17.3f}")

    plot_results(GAPS, KS, type1, power, alpha=ALPHA_LINE, zoom=ZOOM,
                 xlabel=XLABEL, save_path=f"type1_power_{MODE}.png")
    plt.show()
