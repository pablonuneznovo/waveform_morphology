import os
import numpy as np
from scipy.signal import welch, csd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_2 as M

C_DIR, C_SYM, C_BASE = "#E64B35", "#3C5488", "#8C8C8C"


def per_realisation_spectra(yA, yB, winC, ovC):
    Pxx_list, Pyy_list, Pxy_list = [], [], []
    fw = None
    for m in range(yA.shape[0]):
        fw, pxx = welch(yA[m], fs=M.fs, window="hann", nperseg=winC, noverlap=ovC)
        _, pyy = welch(yB[m], fs=M.fs, window="hann", nperseg=winC, noverlap=ovC)
        _, pxy = csd(yA[m], yB[m], fs=M.fs, window="hann", nperseg=winC, noverlap=ovC)
        Pxx_list.append(pxx); Pyy_list.append(pyy); Pxy_list.append(np.conj(pxy))
    return fw, np.array(Pxx_list), np.array(Pyy_list), np.array(Pxy_list)


def band_icoh_from_pooled(Pxx, Pyy, Pxy, fw, band):
    Sxx = Pxx.sum(axis=0); Syy = Pyy.sum(axis=0); Sxy = Pxy.sum(axis=0)
    icoh = np.imag(Sxy) / np.sqrt(Sxx * Syy)
    fmask = (fw >= band[0]) & (fw <= band[1])
    return float(np.mean(icoh[fmask]))


def bootstrap_ci_fast(Pxx, Pyy, Pxy, fw, band, nboot=2000, seed=0):
    rng = np.random.default_rng(seed)
    Mens = Pxx.shape[0]
    vals = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, Mens, size=Mens)
        vals[b] = band_icoh_from_pooled(Pxx[idx], Pyy[idx], Pxy[idx], fw, band)
    return np.percentile(vals, [2.5, 97.5])


def estimate_peak_freq(P, kappa, tdE, D_c):
    yA, yB = M.sim_two_dir_ens(4, M.tau_r_E, tdE, M.tau_r_I, M.tau_d_I, P, 0, kappa, 20.0, D_c)
    winQ = int(round(4 / M.dt))
    fw, pxx = welch(yA[0], fs=M.fs, window="hann", nperseg=winQ, noverlap=winQ // 2)
    lo = np.searchsorted(fw, 0.5)
    hi = np.searchsorted(fw, 30.0)
    return float(fw[lo + np.argmax(pxx[lo:hi])])


def stage_imcoh_nonlinear():
    kappa = 0.50; kHalf = kappa / 2
    tdE = 0.015
    Ph, fh0 = M.scan_bif(M.sigma_sig, M.theta_E, M.theta_I, M.tau_r_E, tdE, M.tau_r_I, M.tau_d_I, 6)
    margins = np.linspace(0.2, 2.6, 7)
    M_ens = 60
    T_c = 45.0
    D_c = 0.0008
    winC = int(round(4 / M.dt)); ovC = winC // 2

    sym_mean = np.full(len(margins), np.nan); sym_lo = np.full_like(sym_mean, np.nan); sym_hi = np.full_like(sym_mean, np.nan)
    dir_mean = np.full(len(margins), np.nan); dir_lo = np.full_like(sym_mean, np.nan); dir_hi = np.full_like(sym_mean, np.nan)
    peak_freqs = np.full(len(margins), np.nan)

    for i, margin in enumerate(margins):
        P = Ph + margin
        peak_f = estimate_peak_freq(P, kappa, tdE, D_c)
        peak_freqs[i] = peak_f
        band = (max(0.5, peak_f - 3), peak_f + 3)

        yAs, yBs = M.sim_two_dir_ens(M_ens, M.tau_r_E, tdE, M.tau_r_I, M.tau_d_I, P, kHalf, kHalf, T_c, D_c)
        yAd, yBd = M.sim_two_dir_ens(M_ens, M.tau_r_E, tdE, M.tau_r_I, M.tau_d_I, P, 0, kappa, T_c, D_c)

        fw_s, Pxx_s, Pyy_s, Pxy_s = per_realisation_spectra(yAs, yBs, winC, ovC)
        fw_d, Pxx_d, Pyy_d, Pxy_d = per_realisation_spectra(yAd, yBd, winC, ovC)

        sym_mean[i] = band_icoh_from_pooled(Pxx_s, Pyy_s, Pxy_s, fw_s, band)
        sym_lo[i], sym_hi[i] = bootstrap_ci_fast(Pxx_s, Pyy_s, Pxy_s, fw_s, band)
        dir_mean[i] = band_icoh_from_pooled(Pxx_d, Pyy_d, Pxy_d, fw_d, band)
        dir_lo[i], dir_hi[i] = bootstrap_ci_fast(Pxx_d, Pyy_d, Pxy_d, fw_d, band)

        print(f"margin={margin:.2f}  peak_f={peak_f:.2f}Hz  band=({band[0]:.1f},{band[1]:.1f})  "
              f"sym={sym_mean[i]:.4f} [{sym_lo[i]:.4f},{sym_hi[i]:.4f}]  "
              f"dir={dir_mean[i]:.4f} [{dir_lo[i]:.4f},{dir_hi[i]:.4f}]")

    unreliable = (dir_lo <= 0) & (0 <= dir_hi)

    fig = M.figure(width="double", height=2.6)
    axes = fig.subplots(1, 2)

    ax = axes[0]
    dm = np.diff(margins).mean() if len(margins) > 1 else 0.35
    for m in margins[unreliable]:
        ax.axvspan(m - dm / 2, m + dm / 2, color=C_BASE, alpha=0.15, zorder=0)
    sym_err = np.vstack([sym_mean - sym_lo, sym_hi - sym_mean])
    dir_err = np.vstack([dir_mean - dir_lo, dir_hi - dir_mean])
    ax.errorbar(margins, dir_mean, dir_err, fmt="o-", color=C_DIR,
                capsize=3, ms=4, label="Directed coupling", zorder=3)
    ax.errorbar(margins, sym_mean, sym_err, fmt="s-", color=C_SYM,
                capsize=3, ms=4, label="Symmetric coupling", zorder=3)
    if unreliable.any():
        ax.errorbar(margins[unreliable], dir_mean[unreliable], dir_err[:, unreliable],
                     fmt="o", color=C_DIR, mfc="none", capsize=3, ms=4, zorder=4)
        ax.errorbar(margins[unreliable], sym_mean[unreliable], sym_err[:, unreliable],
                     fmt="s", color=C_SYM, mfc="none", capsize=3, ms=4, zorder=4)
    ax.axhline(0, color=C_BASE, lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(r"Supercritical margin, $P-P_{\mathrm{Hopf}}$")
    ax.set_ylabel("Band-mean imaginary coherence")
    ax.legend(fontsize=5.5, loc="center left")
    M.despine(ax); M.panel_label(ax, "a")

    ax = axes[1]
    ax.plot(margins, peak_freqs, "o-", color=C_BASE)
    ax.axhline(fh0, color=C_BASE, lw=0.6, ls=":", label=f"$f_{{\\mathrm{{Hopf}}}}$={fh0:.2f} Hz")
    ax.set_xlabel(r"Supercritical margin, $P-P_{\mathrm{Hopf}}$")
    ax.set_ylabel("Oscillation peak frequency (Hz)")
    ax.legend(fontsize=5.5, loc="upper right")
    M.despine(ax); M.panel_label(ax, "b")

    fig.subplots_adjust(wspace=0.4)
    M.save_figure(fig, os.path.join(M.FIGURES_OUT, "figS2"), formats=["png", "svg"], dpi=600)
    plt.close(fig)
    print(f"\nSaved: {os.path.join(M.FIGURES_OUT, 'figS2.png')}")


if __name__ == "__main__":
    stage_imcoh_nonlinear()

