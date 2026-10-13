# -*- coding: utf-8 -*-
"""
Diagnostic for ONE k (default k = 5): follow each factor behind the Type I error
hump as the gap changes. Type I error analysis only (true drone mean = mu_null).

Chain of factors, per replicate:
    A, P  ->  lambda = (A*P)^k  ->  weight w on the external mean
          ->  bias of the pooled mean and width of the CI
          ->  |bias| / CI half-width  ->  miss mu_null (Type I error)

The steps inside simulate_cell() repeat those in calibrate_k.get_conf_int, so that
A, P, lambda, ... can be recorded. check_against_get_conf_int() confirms that the CIs
are identical to those returned by get_conf_int for the same seed.
"""
import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import calibrate_k as ck


def simulate_cell(k, mu_0_true, gap, var_0, var_h, n_0, n_h, n_rep, seed=2026):
    """Return a dict of per-replicate arrays for one (k, gap) cell."""
    np.random.seed(seed)
    out = {name: np.empty(n_rep) for name in
           ("A", "P", "lam", "w", "bias", "ci_ratio", "z_off", "miss", "lo", "hi")}
    se_0, se_h = np.sqrt(var_0 / n_0), np.sqrt(var_h / n_h)
    for r in range(n_rep):
        xbar_0 = np.random.normal(mu_0_true, se_0)
        xbar_h = np.random.normal(mu_0_true + gap * np.sqrt(var_0), se_h)
        var_0_r = ut.get_sampled_variance(n_0, var_0)
        var_h_r = ut.get_sampled_variance(n_h, var_h)
        A = ut.get_overlap(xbar_0, xbar_h, var_0_r, var_h_r)
        P = ut.get_pvalue(xbar_0, xbar_h, var_0_r, var_h_r, n_0, n_h)
        lam = (A * P) ** k
        mean_n = ut.get_posterior_mean(n_0, n_h, var_0_r, var_h_r, lam,
                                       xbar_0, xbar_h)
        var_n = ut.get_posterior_variance(n_0, n_h, var_0_r, var_h_r, lam)
        prec_0, prec_h = n_0 / var_0_r, lam * n_h / var_h_r
        half = 1.96 * np.sqrt(var_n)
        out["A"][r], out["P"][r], out["lam"][r] = A, P, lam
        out["w"][r] = prec_h / (prec_0 + prec_h)           # share of the pooled mean
        out["bias"][r] = (mean_n - mu_0_true) / np.sqrt(var_0)   # in drone SDs
        out["ci_ratio"][r] = np.sqrt(var_n / (var_0_r / n_0))    # vs drone-only CI
        out["z_off"][r] = abs(mean_n - mu_0_true) / half
        out["miss"][r] = abs(mean_n - mu_0_true) > half
        out["lo"][r], out["hi"][r] = mean_n - half, mean_n + half
    return out


def check_against_get_conf_int(k, mu_0_true, gap, var_0, var_h, n_0, n_h, n=20):
    d = simulate_cell(k, mu_0_true, gap, var_0, var_h, n_0, n_h, n)
    np.random.seed(2026)
    ref = np.array([ck.get_conf_int(k, mu_0_true, gap, var_0, var_h, n_0, n_h)
                    for _ in range(n)])
    assert np.allclose(ref[:, 0], d["lo"]) and np.allclose(ref[:, 1], d["hi"])


def plot_diagnostics(gaps, cells, k, save_path=None):
    g = np.asarray(gaps)
    mean = lambda key: np.array([c[key].mean() for c in cells])
    fig, axs = plt.subplots(2, 4, figsize=(17, 8))
    ax = axs.ravel()
    ax[0].plot(g, mean("A"), 'k'); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("A: overlap of the two normal curves\n(mean over replicates)")
    ax[1].plot(g, mean("P"), 'k'); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("P: p-value that the two means differ\n(mean over replicates)")
    ax[2].plot(g, mean("lam"), 'k', label="mean")
    ax[2].plot(g, [np.median(c["lam"]) for c in cells], 'k--', label="median")
    ax[2].set_yscale('log'); ax[2].legend(frameon=False)
    ax[2].set_title(f"lambda = (A*P)^{k}\n(across replicates)")
    ax[3].plot(g, mean("w"), 'k', label="mean weight")
    ax[3].plot(g, [(c["w"] > 0.5).mean() for c in cells], 'C3',
               label="share of replicates\nwith weight > 0.5")
    ax[3].set_ylim(0, 1.05); ax[3].legend(frameon=False, fontsize=8)
    ax[3].set_title("Weight of the external mean in\nthe pooled mean")
    def cond(key, strong):
        return np.array([c[key][(c["w"] > 0.5) == strong].mean()
                         if ((c["w"] > 0.5) == strong).any() else np.nan
                         for c in cells])
    ax[4].plot(g, cond("bias", True), 'C3', label="weight > 0.5 (borrowing strongly)")
    ax[4].plot(g, cond("bias", False), 'C0', label="weight <= 0.5 (mostly drone only)")
    ax[4].plot(g, g, color='gray', ls=':', label="gap itself")
    ax[4].axhline(0, color='gray', lw=0.8); ax[4].legend(frameon=False, fontsize=7)
    ax[4].set_title("Error of pooled mean vs true drone mean\n(drone SDs, averaged within each group)")
    ax[5].plot(g, mean("ci_ratio"), 'k'); ax[5].set_ylim(0, 1.05)
    ax[5].set_title("CI half-width / drone-only half-width\n(mean; smaller = narrower)")
    ax[6].plot(g, mean("z_off"), 'k')
    ax[6].axhline(1.96 / 1.96, color='gray', ls='--', lw=1)
    ax[6].set_title("|bias| / CI half-width (mean)\nabove 1 = mu_null outside the CI")
    ax[7].plot(g, mean("miss"), 'k', lw=2, label="all replicates")
    ax[7].plot(g, cond("miss", True), 'C3', label="weight > 0.5")
    ax[7].plot(g, cond("miss", False), 'C0', label="weight <= 0.5")
    ax[7].axhline(0.05, color='gray', ls='--', lw=1)
    ax[7].set_ylim(0, 1.05); ax[7].legend(frameon=False, fontsize=8)
    ax[7].set_title("Type I error (share of replicates where\nmu_null is outside the CI)")
    for a in ax:
        a.set_xlabel("gap (drone SDs)")
        a.set_xlim(g.min(), g.max())
    fig.suptitle(f"How the Type I error hump is built (k = {k}, true drone mean = mu_null)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


if __name__ == "__main__":
    MU_NULL, VAR_0, VAR_H, N_0, N_H = 1.75, 4.0, 4.0, 2000, 98000
    K, N_REP = 5, 2000
    GAPS = np.round(np.arange(-0.10, 0.1001, 0.005), 4)
    check_against_get_conf_int(K, MU_NULL, 0.02, VAR_0, VAR_H, N_0, N_H)
    cells = [simulate_cell(K, MU_NULL, gp, VAR_0, VAR_H, N_0, N_H, N_REP)
             for gp in GAPS]
    print("gap    A     P   lambda  w_mean  share_w>.5  bias  ci_ratio  z_off  TypeI"
          "  TypeI|w>.5  TypeI|w<=.5  err|w>.5  err|w<=.5")
    for gp, c in zip(GAPS, cells):
        if gp in (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10):
            hi = c["w"] > 0.5
            t_hi = c["miss"][hi].mean() if hi.any() else float('nan')
            t_lo = c["miss"][~hi].mean() if (~hi).any() else float('nan')
            print(f"{gp:5.2f} {c['A'].mean():5.3f} {c['P'].mean():5.3f} "
                  f"{c['lam'].mean():7.4f} {c['w'].mean():7.3f} {hi.mean():10.3f} "
                  f"{c['bias'].mean():7.4f} {c['ci_ratio'].mean():8.3f} "
                  f"{c['z_off'].mean():6.2f} {c['miss'].mean():6.3f} "
                  f"{t_hi:10.3f} {t_lo:12.3f} "
                  f"{(c['bias'][hi].mean() if hi.any() else float('nan')):9.4f} "
                  f"{c['bias'][~hi].mean():10.4f}")
    plot_diagnostics(GAPS, cells, K, save_path="diagnose_k5.png")
    plt.show()
