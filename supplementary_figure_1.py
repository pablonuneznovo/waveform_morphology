import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_2 as M

C_DIR, C_SYM, C_BASE = "#E64B35", "#3C5488", "#8C8C8C"


def fit_var(X, p):
    n, T = X.shape
    Z = np.vstack([X[:, p - k:T - k] for k in range(1, p + 1)])
    Y = X[:, p:]
    B, *_ = np.linalg.lstsq(Z.T, Y.T, rcond=None)
    B = B.T
    resid = Y - B @ Z
    Sigma = np.cov(resid)
    return [B[:, (k - 1) * n:k * n] for k in range(1, p + 1)], Sigma


def var_pdc(Aks, Sigma, f_vec, dt_s, i_src, i_tgt):
    n = Aks[0].shape[0]
    sig2 = np.diag(Sigma).real
    out = np.empty(f_vec.size)
    for k, f in enumerate(f_vec):
        Af = np.eye(n, dtype=complex)
        for lag, Ak in enumerate(Aks, start=1):
            Af -= Ak * np.exp(-1j * 2 * np.pi * f * lag * dt_s)
        col = np.abs(Af[:, i_src]) ** 2 / sig2
        out[k] = (np.abs(Af[i_tgt, i_src]) ** 2 / sig2[i_tgt]) / col.sum()
    return out


def band_pdc(Aks, Sigma, dt_s, band, i_src, i_tgt, nfreq=40):
    fb = np.linspace(band[0], band[1], nfreq)
    return var_pdc(Aks, Sigma, fb, dt_s, i_src, i_tgt).mean()


def stage_mvar():
    kappa = 0.50; kHalf = kappa / 2
    P_val = M.P_hopf - 0.30
    _, _, gEv, gIv = M.fp_newton(P_val)
    M_ens = 30
    T_c = 80.0
    D_c = 0.0008
    ds = 20
    dt_s = M.dt * ds
    burn = int(round(4.0 / dt_s))

    yAd, yBd = M.sim_two_dir_ens(M_ens, M.tau_r_E, M.tau_d_E, M.tau_r_I, M.tau_d_I,
                                  P_val, 0, kappa, T_c, D_c)
    yAs, yBs = M.sim_two_dir_ens(M_ens, M.tau_r_E, M.tau_d_E, M.tau_r_I, M.tau_d_I,
                                  P_val, kHalf, kHalf, T_c, D_c)

    orders = [2, 5, 10, 20, 40, 80, 150]
    band = (2.0, 8.0)
    configs = [("Directed", yAd, yBd, C_DIR), ("Symmetric", yAs, yBs, C_SYM)]

    pdc_AB_o = {lab: np.full((len(orders), M_ens), np.nan) for lab, *_ in configs}
    pdc_BA_o = {lab: np.full((len(orders), M_ens), np.nan) for lab, *_ in configs}

    for lab, yA, yB, c in configs:
        for oi, p in enumerate(orders):
            for m in range(M_ens):
                X = np.vstack([yA[m, ::ds][burn:], yB[m, ::ds][burn:]])
                X = (X - X.mean(axis=1, keepdims=True))
                try:
                    Aks, Sigma = fit_var(X, p)
                except np.linalg.LinAlgError:
                    continue
                pdc_AB_o[lab][oi, m] = band_pdc(Aks, Sigma, dt_s, band, i_src=0, i_tgt=1)
                pdc_BA_o[lab][oi, m] = band_pdc(Aks, Sigma, dt_s, band, i_src=1, i_tgt=0)

    p_rep = 20
    f_plot = np.linspace(0.5, 20, 60)
    Aks_rep = []
    for m in range(min(8, M_ens)):
        X = np.vstack([yAd[m, ::ds][burn:], yBd[m, ::ds][burn:]])
        X = X - X.mean(axis=1, keepdims=True)
        Aks_rep.append(fit_var(X, p_rep))
    pdc_AB_rep = np.mean([var_pdc(Aks, Sigma, f_plot, dt_s, 0, 1) for Aks, Sigma in Aks_rep], axis=0)
    pdc_BA_rep = np.mean([var_pdc(Aks, Sigma, f_plot, dt_s, 1, 0) for Aks, Sigma in Aks_rep], axis=0)
    pdc_AB_analytic = np.array([M.pdc2(2 * np.pi * f, M.tau_r_E, M.tau_d_E, gEv, 0, kappa)[0]
                                 for f in f_plot])
    pdc_BA_analytic = np.array([M.pdc2(2 * np.pi * f, M.tau_r_E, M.tau_d_E, gEv, 0, kappa)[1]
                                 for f in f_plot])

    fig = M.figure(width="double", height=2.6)
    axes = fig.subplots(1, 2)

    ax = axes[0]
    ax.plot(f_plot, pdc_AB_analytic, color=C_DIR, label="Analytic A$\\to$B")
    ax.plot(f_plot, pdc_BA_analytic, "--", color=C_DIR, lw=1, label="Analytic B$\\to$A")
    ax.plot(f_plot, pdc_AB_rep, "o", color=C_DIR, ms=2.5, alpha=0.6, label=f"MVAR($p$={p_rep}) A$\\to$B")
    ax.plot(f_plot, pdc_BA_rep, "s", color=C_DIR, ms=2.5, alpha=0.6, mfc="none", label=f"MVAR($p$={p_rep}) B$\\to$A")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Squared generalised PDC")
    ax.legend(fontsize=5.5, loc="upper right")
    M.despine(ax); M.panel_label(ax, "a")

    ax = axes[1]
    for lab, *_rest, c in configs:
        mAB = np.nanmean(pdc_AB_o[lab], axis=1); sAB = np.nanstd(pdc_AB_o[lab], axis=1)
        mBA = np.nanmean(pdc_BA_o[lab], axis=1); sBA = np.nanstd(pdc_BA_o[lab], axis=1)
        ax.errorbar(orders, mAB, sAB, fmt="o-", color=c, capsize=2, label=f"{lab}, A$\\to$B")
        ax.errorbar(orders, mBA, sBA, fmt="s--", color=c, capsize=2, mfc="none", label=f"{lab}, B$\\to$A")
    ax.set_xscale("log")
    ax.set_xlabel("VAR model order $p$"); ax.set_ylabel(f"Band-mean gPDC$^2$ ({band[0]:.0f}-{band[1]:.0f} Hz)")
    ax.legend(fontsize=5.5, loc="upper left")
    M.despine(ax); M.panel_label(ax, "b")

    fig.subplots_adjust(wspace=0.4)
    M.save_figure(fig, os.path.join(M.FIGURES_OUT, "figS1"), formats=["png", "svg"], dpi=600)
    plt.close(fig)

    print("\nMVAR-PDC order-dependence summary (band-mean, 2-8 Hz)")
    print(f"{'order':>7}" + "".join(f"{'dir A->B':>12}{'dir B->A':>12}{'sym A->B':>12}{'sym B->A':>12}"))
    for oi, p in enumerate(orders):
        row = f"{p:>7}"
        for lab, *_ in configs:
            row += f"{np.nanmean(pdc_AB_o[lab][oi]):>12.4f}{np.nanmean(pdc_BA_o[lab][oi]):>12.4f}"
        print(row)
    print(f"\nSaved: {os.path.join(M.FIGURES_OUT, 'figS1.png')}")
    return pdc_AB_o, pdc_BA_o, orders


if __name__ == "__main__":
    stage_mvar()

