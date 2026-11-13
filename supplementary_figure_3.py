import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_2 as M

C_DIR, C_SYM, C_BASE = "#E64B35", "#3C5488", "#8C8C8C"


def run():
    kappa = 0.50
    kHalf = kappa / 2
    tdE = 0.030
    Ph, fh = M.scan_bif(M.sigma_sig, M.theta_E, M.theta_I, M.tau_r_E, tdE, M.tau_r_I, M.tau_d_I, 6)
    P = Ph + 0.8
    M_ens = 12
    T_c = 45.0
    D_c = 0.0008
    n_surr = 40

    print(f"tau_d={tdE*1e3:.0f}ms, Hopf at P={Ph:.3f}, f0={fh:.2f}Hz, running at P={P:.3f}")

    yAs, yBs = M.sim_two_dir_ens(M_ens, M.tau_r_E, tdE, M.tau_r_I, M.tau_d_I, P, kHalf, kHalf, T_c, D_c)
    yAd, yBd = M.sim_two_dir_ens(M_ens, M.tau_r_E, tdE, M.tau_r_I, M.tau_d_I, P, 0, kappa, T_c, D_c)
    yAs = yAs[:, ::M.ds]; yBs = yBs[:, ::M.ds]
    yAd = yAd[:, ::M.ds]; yBd = yBd[:, ::M.ds]

    period_period = 1.0 / fh
    fracs = np.array([1/16, 1/8, 1/4, 3/8, 1/2, 3/4, 1.0])
    rng = np.random.default_rng(11)

    Isym = np.empty(fracs.size); Idir = np.empty(fracs.size); Inull = np.empty(fracs.size)
    Ts = np.empty(fracs.size, dtype=int)
    for k, frac in enumerate(fracs):
        T = max(1, int(round((frac * period_period) / M.dt_s)))
        Ts[k] = T
        Is = Id = 0.0
        null_vals = []
        for m in range(M_ens):
            a, b = yAd[m], yBd[m]
            Is += M.reversibility_deco(np.vstack([yAs[m], yBs[m]]), T)
            Id += M.reversibility_deco(np.vstack([a, b]), T)
            for _ in range(n_surr):
                null_vals.append(M.reversibility_deco(
                    np.vstack([M._phase_rand(a, rng), M._phase_rand(b, rng)]), T))
        Isym[k] = Is / M_ens
        Idir[k] = Id / M_ens
        Inull[k] = np.percentile(null_vals, 95)
        print(f"  frac={frac:6.3f} (T={T:3d} samp, {frac/0.25:.2f}x quarter-period)  "
              f"Isym={Isym[k]:.5f}  Idir={Idir[k]:.5f}  null95={Inull[k]:.5f}  "
              f"dir>null: {Idir[k] > Inull[k]}  sym<null: {Isym[k] < Inull[k]}")

    fig = M.figure(width="single", height=2.8)
    ax = fig.subplots(1, 1)
    x = fracs / 0.25
    ax.plot(x, Idir, "o-", color=C_DIR, label="Directed coupling")
    ax.plot(x, Isym, "s-", color=C_SYM, label="Symmetric coupling")
    ax.plot(x, Inull, "--", color=C_BASE, label="Surrogate null (95th pct.)")
    ax.axvline(1.0, color=C_BASE, lw=0.6, ls=":")
    ax.set_xlabel("Lag $T$ / quarter-period lag")
    ax.set_ylabel("Bivariate irreversibility index $I(T)$")
    ax.legend(fontsize=6, loc="upper right")
    M.despine(ax)
    fig.subplots_adjust(bottom=0.18, left=0.18)
    M.save_figure(fig, os.path.join(M.FIGURES_OUT, "figS3"), formats=["png", "svg"], dpi=600)
    plt.close(fig)
    print(f"\nSaved: {os.path.join(M.FIGURES_OUT, 'figS3.png')}")


if __name__ == "__main__":
    run()

