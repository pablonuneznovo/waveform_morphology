import os
from math import exp
import numpy as np
from scipy.signal import welch, csd, find_peaks
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

NPG_COLORS = {'red': '#E64B35', 'teal': '#4DBBD5', 'green': '#00A087', 'navy': '#3C5488', 'salmon': '#F39B7F', 'lavender': '#8491B4', 'yellow': '#FED439', 'purple': '#7E6148', 'gray': '#8C8C8C', 'lightgray': '#E5E5E5'}

NPG_PALETTE = [NPG_COLORS['red'], NPG_COLORS['teal'], NPG_COLORS['green'], NPG_COLORS['navy'], NPG_COLORS['salmon'], NPG_COLORS['lavender'], NPG_COLORS['yellow'], NPG_COLORS['purple']]

NN_RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.linewidth': 0.5, 'axes.labelsize': 8, 'axes.labelpad': 3, 'axes.titlesize': 8, 'axes.titlepad': 4, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.prop_cycle': mpl.cycler(color=NPG_PALETTE), 'xtick.major.size': 3, 'xtick.minor.size': 1.5, 'xtick.major.width': 0.5, 'xtick.minor.width': 0.5, 'xtick.direction': 'in', 'xtick.labelsize': 7, 'xtick.major.pad': 2, 'ytick.major.size': 3, 'ytick.minor.size': 1.5, 'ytick.major.width': 0.5, 'ytick.minor.width': 0.5, 'ytick.direction': 'in', 'ytick.labelsize': 7, 'ytick.major.pad': 2, 'lines.linewidth': 1.0, 'lines.markersize': 4, 'legend.fontsize': 7, 'legend.frameon': False, 'legend.handlelength': 1.2, 'legend.handletextpad': 0.4, 'legend.labelspacing': 0.3, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'image.cmap': 'RdBu_r'}

def set_style():
    mpl.rcParams.update(NN_RC)

FIG_WIDTH = {'single': 3.346, 'onehalf': 4.488, 'double': 6.929}

def figure(width='double', height=None, panels_rows=1):
    w = FIG_WIDTH.get(width, FIG_WIDTH['double'])
    h = height if height is not None else 2.0 * panels_rows
    return plt.figure(figsize=(w, h))

def despine(ax, which=('top', 'right')):
    for spine in which:
        ax.spines[spine].set_visible(False)

def panel_label(ax, label, x=-0.18, y=1.05):
    ax.text(x, y, str(label), transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', ha='right')

def save_figure(fig, path_stem, formats=('pdf', 'tiff'), dpi=300):
    import os
    os.makedirs(os.path.dirname(path_stem) if os.path.dirname(path_stem) else '.', exist_ok=True)
    for fmt in formats:
        out = f'{path_stem}.{fmt}'
        kwargs = dict(dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        if fmt == 'tiff':
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(out, format=fmt, **kwargs)
        print(f'Saved: {out}')

rng = np.random.default_rng(13)

C_DIR = NPG_COLORS['red']

C_SYM = NPG_COLORS['navy']

C_MORPH = NPG_COLORS['teal']

C_BASE = NPG_COLORS['gray']

tau_fast = 0.005

tau_r_E, tau_d_E = (0.003, 0.035)

tau_r_I, tau_d_I = (0.005, 0.025)

c_EE, c_IE, c_EI, c_II = (16.0, 12.0, 15.0, 3.0)

sigma_sig = 1.0

theta_E, theta_I = (4.5, 4.0)

dt = 0.0001

ds = 10

dt_s = dt * ds

fs = 1.0 / dt

NF = 3000

f_vec = np.linspace(0.1, 30, NF)

w_vec = 2 * np.pi * f_vec

def S_np(v):
    return 1.0 / (1.0 + np.exp(-v / sigma_sig))

def dS_np(v):
    s = S_np(v)
    return s * (1.0 - s) / sigma_sig

def fp_newton(P, thE=theta_E, thI=theta_I, sg=sigma_sig, n_iter=80):
    RE, RI = (0.5, 0.5)
    for _ in range(n_iter):
        uE = c_EE * RE - c_IE * RI - thE + P
        uI = c_EI * RE - c_II * RI - thI
        SE = 1.0 / (1.0 + exp(-uE / sg))
        SI = 1.0 / (1.0 + exp(-uI / sg))
        g1 = RE - SE
        g2 = RI - SI
        dSE = SE * (1.0 - SE) / sg
        dSI = SI * (1.0 - SI) / sg
        J11 = 1.0 - dSE * c_EE
        J12 = dSE * c_IE
        J21 = -dSI * c_EI
        J22 = 1.0 + dSI * c_II
        det = J11 * J22 - J12 * J21
        if abs(det) < 1e-12:
            break
        RE -= (J22 * g1 - J12 * g2) / det
        RI -= (-J21 * g1 + J11 * g2) / det
        RE = min(1.0 - 1e-12, max(1e-12, RE))
        RI = min(1.0 - 1e-12, max(1e-12, RI))
    uE = c_EE * RE - c_IE * RI - thE + P
    uI = c_EI * RE - c_II * RI - thI
    SE = 1.0 / (1.0 + exp(-uE / sg))
    SI = 1.0 / (1.0 + exp(-uI / sg))
    return (RE, RI, SE * (1.0 - SE) / sg, SI * (1.0 - SI) / sg)

def scan_bif(sg, thE, thI, trE, tdE, trI, tdI, P_max, tf=tau_fast):
    Ps = np.linspace(0, P_max, 250)
    mRc = np.full(Ps.size, np.nan)
    fc = np.full(Ps.size, np.nan)
    for k, Pk in enumerate(Ps):
        _, _, gE, gI = fp_newton(Pk, thE, thI, sg)
        J = np.zeros((6, 6))
        J[0, 0] = -1 / tf
        J[0, 3] = gE * c_EE / tf
        J[0, 5] = -gE * c_IE / tf
        J[1, 1] = -1 / tf
        J[1, 3] = gI * c_EI / tf
        J[1, 5] = -gI * c_II / tf
        J[2, 0] = 1 / trE
        J[2, 2] = -1 / trE
        J[3, 2] = 1 / tdE
        J[3, 3] = -1 / tdE
        J[4, 1] = 1 / trI
        J[4, 4] = -1 / trI
        J[5, 4] = 1 / tdI
        J[5, 5] = -1 / tdI
        ev = np.linalg.eigvals(J)
        cx = ev[(np.abs(ev.imag) > 1e-06) & (ev.imag > 0)]
        if cx.size:
            idx = np.argmax(cx.real)
            mRc[k] = cx[idx].real
            fc[k] = cx[idx].imag / (2 * np.pi)
    cross = np.where((mRc[:-1] < 0) & (mRc[1:] > 0))[0]
    if cross.size:
        return (Ps[cross[0]], fc[cross[0]])
    return (np.nan, np.nan)

def sim_node(trE, tdE, trI, tdI, P, T, tf=tau_fast):
    N = int(round(T / dt))
    RE = RI = xE = yE = xI = yI = 0.5
    out = np.empty(N)
    for n in range(N):
        uE = c_EE * yE - c_IE * yI - theta_E + P
        uI = c_EI * yE - c_II * yI - theta_I
        SE = 1.0 / (1.0 + exp(-uE / sigma_sig))
        SI = 1.0 / (1.0 + exp(-uI / sigma_sig))
        RE = min(1.0, max(0.0, RE + dt * (-RE + SE) / tf))
        RI = min(1.0, max(0.0, RI + dt * (-RI + SI) / tf))
        xE += dt * (-xE + RE) / trE
        yE += dt * (-yE + xE) / tdE
        xI += dt * (-xI + RI) / trI
        yI += dt * (-yI + xI) / tdI
        yE = min(1.0, max(0.0, yE))
        out[n] = yE
    return out

def sim_two_dir_ens(M, trE, tdE, trI, tdI, P, kAB, kBA, T, D):
    N = int(round(T / dt))
    sq = np.sqrt(2 * D / dt)
    z = 0.5 * np.ones(M)
    REA, RIA, xEA, yEA, xIA, yIA = (z.copy() for _ in range(6))
    REB, RIB, xEB, yEB, xIB, yIB = (z.copy() for _ in range(6))
    yA = np.empty((M, N))
    yB = np.empty((M, N))
    P = np.broadcast_to(P, (M,)).astype(float)
    tdE = np.broadcast_to(tdE, (M,)).astype(float)
    kAB = np.broadcast_to(kAB, (M,)).astype(float)
    kBA = np.broadcast_to(kBA, (M,)).astype(float)
    for n in range(N):
        uEA = c_EE * yEA - c_IE * yIA - theta_E + P
        uIA = c_EI * yEA - c_II * yIA - theta_I
        uEB = c_EE * yEB - c_IE * yIB - theta_E + P
        uIB = c_EI * yEB - c_II * yIB - theta_I
        cA = kAB / tau_fast * dS_np(uEA) * yEB
        cB = kBA / tau_fast * dS_np(uEB) * yEA
        nz = sq * rng.standard_normal((4, M))
        REA = np.clip(REA + dt * ((-REA + S_np(uEA)) / tau_fast + cA + nz[0]), 0, 1)
        RIA = np.clip(RIA + dt * ((-RIA + S_np(uIA)) / tau_fast + nz[1]), 0, 1)
        xEA = xEA + dt * (-xEA + REA) / trE
        yEA = yEA + dt * (-yEA + xEA) / tdE
        xIA = xIA + dt * (-xIA + RIA) / trI
        yIA = yIA + dt * (-yIA + xIA) / tdI
        REB = np.clip(REB + dt * ((-REB + S_np(uEB)) / tau_fast + cB + nz[2]), 0, 1)
        RIB = np.clip(RIB + dt * ((-RIB + S_np(uIB)) / tau_fast + nz[3]), 0, 1)
        xEB = xEB + dt * (-xEB + REB) / trE
        yEB = yEB + dt * (-yEB + xEB) / tdE
        xIB = xIB + dt * (-xIB + RIB) / trI
        yIB = yIB + dt * (-yIB + xIB) / tdI
        yA[:, n] = yEA
        yB[:, n] = yEB
    return (yA, yB)

def wf_features(y):
    y = y - y.mean()
    amp = y.std() or np.finfo(float).eps
    mpd = int(round(0.04 / dt))
    pk, _ = find_peaks(y, distance=mpd, prominence=0.3 * amp)
    tr, _ = find_peaks(-y, distance=mpd, prominence=0.3 * amp)
    rises, decays = ([], [])
    for p in pk:
        prev = tr[tr < p]
        nxt = tr[tr > p]
        if prev.size and nxt.size:
            rises.append((p - prev[-1]) * dt)
            decays.append((nxt[0] - p) * dt)
    if not rises:
        raise ValueError('No valid rise and decay times were detected.')
    else:
        rt, dc = (np.median(rises), np.median(decays))
        rise, decay = (rt, dc)
        rdsym = rt / (rt + dc)
    dy = np.diff(y) / dt
    dsk = np.mean((dy - dy.mean()) ** 3) / dy.std() ** 3
    return dict(rise=rise, decay=decay, rdsym=rdsym, dsk=dsk)

def one_cycle(y):
    ny = len(y)
    yy = y - np.mean(y[ny // 2:])
    zc = np.where((yy[:-1] < 0) & (yy[1:] >= 0))[0]
    zc = zc[zc > ny // 2]
    if zc.size < 2:
        raise ValueError('Fewer than two waveform cycles were detected.')
    seg = y[zc[-2]:zc[-1]]
    p = np.linspace(0, 1, seg.size)
    ph = np.linspace(0, 1, 200)
    return (ph, np.interp(ph, p, seg))

def twonode_spec(w, trE, tdE, trI, tdI, gE, gI, kAB, kBA):
    n = w.size
    coh = np.empty(n)
    icoh = np.empty(n)
    for k in range(n):
        wk = w[k]
        HE = 1 / ((1 + 1j * wk * trE) * (1 + 1j * wk * tdE))
        HI = 1 / ((1 + 1j * wk * trI) * (1 + 1j * wk * tdI))
        a = 1 + 1j * wk * tau_fast - gE * c_EE * HE
        b = gE * c_IE * HI
        c = -gI * c_EI * HE
        d = 1 + 1j * wk * tau_fast + gI * c_II * HI
        M = np.array([[a, b, -gE * kAB * HE, 0], [c, d, 0, 0], [-gE * kBA * HE, 0, a, b], [0, 0, c, d]], dtype=complex)
        G = np.linalg.inv(M)
        Sm = G @ G.conj().T
        den = np.sqrt(Sm[0, 0].real * Sm[2, 2].real)
        coh[k] = abs(Sm[0, 2]) / den
        icoh[k] = Sm[0, 2].imag / den
    return (coh, icoh)

def pdc2(w, trE, tdE, gE, kAB, kBA):
    HE = 1 / ((1 + 1j * w * trE) * (1 + 1j * w * tdE))
    As = abs(1 + 1j * w * tau_fast - gE * c_EE * HE)
    ABA = abs(gE * kBA * HE)
    AAB = abs(gE * kAB * HE)
    pdc_AB = ABA ** 2 / (As ** 2 + ABA ** 2)
    pdc_BA = AAB ** 2 / (As ** 2 + AAB ** 2)
    return (pdc_AB, pdc_BA)

def reversibility_deco(X, T):
    X = X - X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < np.finfo(float).eps] = np.finfo(float).eps
    X = X / sd
    Xr = X[:, ::-1]
    Dn = X.shape[0]
    Cf = np.empty((Dn, Dn))
    Cr = np.empty((Dn, Dn))
    for i in range(Dn):
        for j in range(Dn):
            Cf[i, j] = np.mean(X[i, :-T] * X[j, T:])
            Cr[i, j] = np.mean(Xr[i, :-T] * Xr[j, T:])
    Cf = np.clip(Cf, -0.999999, 0.999999)
    Cr = np.clip(Cr, -0.999999, 0.999999)
    FSf = -0.5 * np.log(1 - Cf ** 2)
    FSr = -0.5 * np.log(1 - Cr ** 2)
    return np.mean((FSf - FSr) ** 2)

def ensemble_spectra(yA, yB, winC, ovC):
    Pxx = Pyy = Pxy = 0.0
    for m in range(yA.shape[0]):
        fw, pxx = welch(yA[m], fs=fs, window='hann', nperseg=winC, noverlap=ovC)
        _, pyy = welch(yB[m], fs=fs, window='hann', nperseg=winC, noverlap=ovC)
        _, pxy = csd(yA[m], yB[m], fs=fs, window='hann', nperseg=winC, noverlap=ovC)
        Pxx = Pxx + pxx
        Pyy = Pyy + pyy
        Pxy = Pxy + np.conj(pxy)
    coh = np.abs(Pxy) / np.sqrt(Pxx * Pyy)
    icoh = np.imag(Pxy) / np.sqrt(Pxx * Pyy)
    return (fw, coh, icoh)

def smooth(v, n=5):
    return np.convolve(v, np.ones(n) / n, mode='same')

P_hopf, f_hopf = scan_bif(sigma_sig, theta_E, theta_I, tau_r_E, tau_d_E, tau_r_I, tau_d_I, 6)

mph = 1.8

def simulate_morphology():

    def compute():
        td_ex = np.array([0.015, 0.025, 0.033, 0.04])
        td_sw = np.linspace(0.012, 0.042, 12)
        tr_ex = 0.003
        cyc = []
        for td in td_ex:
            Ph, fhj = scan_bif(sigma_sig, theta_E, theta_I, tr_ex, td, tau_r_I, tau_d_I, 6)
            if np.isnan(Ph) or fhj < 2.5 or fhj > 7:
                Ph = P_hopf
            y = sim_node(tr_ex, td, tau_r_I, tau_d_I, Ph + mph, 8)
            ph, wc = one_cycle(y)
            cyc.append((wc - wc.min()) / (wc.max() - wc.min()))
        rise = np.full(td_sw.size, np.nan)
        dec = np.full(td_sw.size, np.nan)
        rds = np.full(td_sw.size, np.nan)
        dsk = np.full(td_sw.size, np.nan)
        for k, td in enumerate(td_sw):
            Ph, fhk = scan_bif(sigma_sig, theta_E, theta_I, tr_ex, td, tau_r_I, tau_d_I, 6)
            if np.isnan(Ph) or fhk < 2.5 or fhk > 7:
                Ph = P_hopf
            y = sim_node(tr_ex, td, tau_r_I, tau_d_I, Ph + mph, 8)
            F = wf_features(y[-int(5 / dt):])
            rise[k] = F['rise'] * 1000.0
            dec[k] = F['decay'] * 1000.0
            rds[k] = F['rdsym']
            dsk[k] = F['dsk']
        ph = np.linspace(0, 1, 200)
        return dict(td_ex=td_ex, td_sw=td_sw, ph=ph, cyc=np.array(cyc), rise=rise, dec=dec, rds=rds, dsk=dsk)
    return compute()

def simulate_connectivity():

    def compute():
        kappa = 0.5
        kHalf = kappa / 2
        P_val = P_hopf - 0.3
        _, _, gEv, gIv = fp_newton(P_val)
        coh_sym, ic_sym = twonode_spec(w_vec, tau_r_E, tau_d_E, tau_r_I, tau_d_I, gEv, gIv, kHalf, kHalf)
        coh_dir, ic_dir = twonode_spec(w_vec, tau_r_E, tau_d_E, tau_r_I, tau_d_I, gEv, gIv, 0, kappa)
        M_ens = 30
        T_c = 80
        D_c = 0.0008
        winC = int(round(4 / dt))
        ovC = winC // 2
        yAd, yBd = sim_two_dir_ens(M_ens, tau_r_E, tau_d_E, tau_r_I, tau_d_I, P_val, 0, kappa, T_c, D_c)
        yAs, yBs = sim_two_dir_ens(M_ens, tau_r_E, tau_d_E, tau_r_I, tau_d_I, P_val, kHalf, kHalf, T_c, D_c)
        fw, coh_sd, ic_sd = ensemble_spectra(yAd, yBd, winC, ovC)
        _, coh_ss, ic_ss = ensemble_spectra(yAs, yBs, winC, ovC)
        pdc_AB = np.empty(f_vec.size)
        pdc_BA = np.empty(f_vec.size)
        pdc_AB_sym = np.empty(f_vec.size)
        pdc_BA_sym = np.empty(f_vec.size)
        for k in range(f_vec.size):
            pdc_AB[k], pdc_BA[k] = pdc2(w_vec[k], tau_r_E, tau_d_E, gEv, 0, kappa)
            pdc_AB_sym[k], pdc_BA_sym[k] = pdc2(w_vec[k], tau_r_E, tau_d_E, gEv, kHalf, kHalf)
        return dict(f_vec=f_vec, coh_sym=coh_sym, ic_sym=ic_sym, coh_dir=coh_dir, ic_dir=ic_dir, fw=fw, coh_sd=coh_sd, ic_sd=ic_sd, coh_ss=coh_ss, ic_ss=ic_ss, pdc_AB=pdc_AB, pdc_BA=pdc_BA, pdc_AB_sym=pdc_AB_sym, pdc_BA_sym=pdc_BA_sym)
    return compute()

def dskew(y):
    y = y - y.mean()
    dyv = np.diff(y) / dt
    sd = dyv.std()
    return np.mean((dyv - dyv.mean()) ** 3) / sd ** 3 if sd > 0 else np.nan

def _phase_rand(x, rng):
    n = x.size
    X = np.fft.rfft(x)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, X.size))
    ph[0] = 1.0
    if n % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(X * ph, n=n)

def simulate_frequency_bands():

    def compute():
        bands = [(1.0, 'theta (~4 Hz)'), (0.5, '~9 Hz'), (0.25, '~17 Hz'), (0.125, '~34 Hz')]
        mgrid = np.linspace(0.35, 1.1, 9)
        labels, X, Y = ([], [], [])
        for c, lbl in bands:
            xs, ys = ([], [])
            for m in mgrid:
                tf = tau_fast * c
                trE = tau_r_E * c
                tdE = tau_d_E * c * m
                trI = tau_r_I * c
                tdI = tau_d_I * c
                Ph, fh = scan_bif(sigma_sig, theta_E, theta_I, trE, tdE, trI, tdI, 6, tf)
                if np.isnan(Ph) or not 0.5 < fh * c / max(f_hopf, 1e-09) < 1.9:
                    continue
                T = min(12.0, max(4.0, 45.0 / fh))
                y = sim_node(trE, tdE, trI, tdI, Ph + mph, T, tf)
                xs.append(2 * np.pi * fh * tdE)
                ys.append(dskew(y[int(min(1.5, T * 0.2) / dt):]))
            labels.append(lbl)
            X.append(np.array(xs))
            Y.append(np.array(ys))
        return dict(labels=np.array(labels, dtype=object), X=np.array(X, dtype=object), Y=np.array(Y, dtype=object))
    return compute()

def simulate_pair_irreversibility():

    def compute():
        kappa = 0.5
        kHalf = kappa / 2
        td_b = np.linspace(0.018, 0.042, 6)
        T_b = 45
        D_b = 0.0008
        nRepB = 30
        Isym_reps = np.empty((td_b.size, nRepB))
        Idir_reps = np.empty((td_b.size, nRepB))
        Inull = np.empty(td_b.size)
        rng_s = np.random.default_rng(7)
        n_surr = 40
        for k, td in enumerate(td_b):
            print(f'Bivariate irreversibility: decay {td * 1000:.1f} ms', flush=True)
            PhB, fhk = scan_bif(sigma_sig, theta_E, theta_I, tau_r_E, td, tau_r_I, tau_d_I, 6)
            if np.isnan(PhB) or fhk < 2 or fhk > 7:
                PhB, fhk = (P_hopf, f_hopf)
            Pp = PhB + 0.8
            Tk = max(1, int(round(1 / (4 * max(fhk, 1)) / dt_s)))
            yAs, yBs = sim_two_dir_ens(nRepB, tau_r_E, td, tau_r_I, tau_d_I, Pp, kHalf, kHalf, T_b, D_b)
            yAd, yBd = sim_two_dir_ens(nRepB, tau_r_E, td, tau_r_I, tau_d_I, Pp, 0, kappa, T_b, D_b)
            null_vals = []
            for m in range(nRepB):
                a = yAd[m, ::ds]
                b = yBd[m, ::ds]
                Isym_reps[k, m] = reversibility_deco(np.vstack([yAs[m, ::ds], yBs[m, ::ds]]), Tk)
                Idir_reps[k, m] = reversibility_deco(np.vstack([a, b]), Tk)
                for _ in range(n_surr):
                    null_vals.append(reversibility_deco(np.vstack([_phase_rand(a, rng_s), _phase_rand(b, rng_s)]), Tk))
            Inull[k] = np.percentile(null_vals, 95)
            del yAs, yBs, yAd, yBd
        return dict(td_b=td_b, Isym=Isym_reps.mean(1), Idir=Idir_reps.mean(1), Isym_sem=Isym_reps.std(1, ddof=1) / np.sqrt(nRepB), Idir_sem=Idir_reps.std(1, ddof=1) / np.sqrt(nRepB), Inull=Inull)
    return compute()

ORIG_DIR = os.path.dirname(os.path.abspath(__file__))

FIGURES_OUT = os.path.join(ORIG_DIR, 'figures')

SCHEMATICS = os.path.join(ORIG_DIR, 'abstracts')

SCHEM_SCALE = 0.75

def load_schematic(name):
    return plt.imread(os.path.join(SCHEMATICS, name))

def add_schematic_panel(fig, gs_slice, img, label='a'):
    ax = fig.add_subplot(gs_slice)
    ax.imshow(img, aspect='equal')
    ax.axis('off')
    panel_label(ax, label, x=-0.02, y=1.03)
    return ax

def schem_height(aspect, usable_w, scale=SCHEM_SCALE):
    return scale * usable_w / aspect

def two_part_gridspec(fig, total_h, schem_h, data_h, data_rows, data_cols, hspace=0.35, wspace=1.0):
    top_margin = 0.025 * total_h
    bottom_margin = 0.09 * total_h
    gap = 0.035 * total_h
    schem_top = 1 - top_margin / total_h
    schem_bottom = schem_top - schem_h / total_h
    data_top = schem_bottom - gap / total_h
    data_bottom = bottom_margin / total_h
    inset = 0.91 * (1 - SCHEM_SCALE) / 2
    gs_top = fig.add_gridspec(1, 6, left=0.07 + inset, right=0.98 - inset, top=schem_top, bottom=schem_bottom)
    gs_bot = fig.add_gridspec(data_rows, data_cols, left=0.07, right=0.98, top=data_top, bottom=data_bottom, hspace=hspace, wspace=wspace)
    return (gs_top, gs_bot)

def plot_figure(d, d1, d4, db):
    img = load_schematic('Figure 3.png')
    aspect = img.shape[1] / img.shape[0]
    usable_w = FIG_WIDTH['double'] * 0.91
    schem_h = schem_height(aspect, usable_w)
    data_h = 4.1
    total_h = schem_h + data_h + 0.25
    fig = figure(width='double', height=total_h)
    gs_top, gs_bot = two_part_gridspec(fig, total_h, schem_h, data_h, data_rows=2, data_cols=3, hspace=0.4, wspace=0.5)
    add_schematic_panel(fig, gs_top[0, :], img, 'a')
    fw = d['fw']
    fmw = fw <= 20
    ax = fig.add_subplot(gs_bot[0, 0])
    ax.plot(d['f_vec'], d['coh_dir'], color=C_DIR, label='Theory, directed')
    ax.plot(fw[fmw], smooth(d['coh_sd'])[fmw], lw=0.7, color=C_DIR, label='Sim, directed')
    ax.plot(d['f_vec'], d['coh_sym'], color=C_SYM, label='Theory, symmetric')
    ax.plot(fw[fmw], smooth(d['coh_ss'])[fmw], lw=0.7, color=C_SYM, label='Sim, symmetric')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.3)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Coherence')
    ax.legend(loc='upper right', fontsize=5)
    despine(ax)
    panel_label(ax, 'b')
    ax = fig.add_subplot(gs_bot[0, 1])
    ax.plot(d['f_vec'], d['ic_dir'], color=C_DIR)
    ax.plot(fw[fmw], smooth(d['ic_sd'])[fmw], lw=0.7, color=C_DIR)
    ax.plot(d['f_vec'], d['ic_sym'], color=C_SYM)
    ax.plot(fw[fmw], smooth(d['ic_ss'])[fmw], lw=0.7, color=C_SYM)
    ax.axhline(0.0, ls=':', lw=0.5, color=C_BASE)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Imaginary coherence')
    despine(ax)
    panel_label(ax, 'c')
    ax = fig.add_subplot(gs_bot[0, 2])
    ax.plot(d['f_vec'], d['pdc_AB'], color=C_DIR, label='Directed, A$\\rightarrow$B')
    ax.plot(d['f_vec'], d['pdc_BA'], '--', color=C_DIR, lw=1, label='Directed, B$\\rightarrow$A')
    if 'pdc_AB_sym' in d:
        ax.plot(d['f_vec'], d['pdc_AB_sym'], color=C_SYM, label='Symmetric (both directions)')
        ax.plot(d['f_vec'], d['pdc_BA_sym'], ':', color=C_SYM, lw=1.2)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Squared generalised PDC')
    ax.legend(loc='upper right', fontsize=5)
    despine(ax)
    panel_label(ax, 'd')
    ax = fig.add_subplot(gs_bot[1, 0])
    ax.plot(d1['td_sw'] * 1000.0, d1['dsk'], 'o-', color=C_MORPH)
    ax.set_ylim(0.1, 0.205)
    ax.set_xlabel('$\\tau_d$ (ms)')
    ax.set_ylabel('Derivative skewness')
    despine(ax)
    panel_label(ax, 'e')
    ax = fig.add_subplot(gs_bot[1, 1])
    if 'Inull' in d4:
        ax.plot(d4['td_b'] * 1000.0, d4['Inull'], ':', color=C_BASE, lw=1.2, label='surrogate null (95th pct)')
    ax.errorbar(d4['td_b'] * 1000.0, d4['Idir'], yerr=d4['Idir_sem'], fmt='o-', capsize=2, color=C_DIR, label='Directed A$\\rightarrow$B')
    ax.errorbar(d4['td_b'] * 1000.0, d4['Isym'], yerr=d4['Isym_sem'], fmt='s-', capsize=2, color=C_SYM, label='Symmetric')
    ax.axhline(0.0, ls=':', lw=0.5, color=C_BASE)
    ymax = max((d4['Idir'] + d4['Idir_sem']).max(), (d4['Isym'] + d4['Isym_sem']).max(), d4['Inull'].max())
    ax.set_ylim(-0.02, ymax * 1.25)
    ax.set_xlabel('$\\tau_d$ (ms)')
    ax.set_ylabel('Bivariate irreversibility')
    ax.legend(loc='upper left', fontsize=5)
    despine(ax)
    panel_label(ax, 'f')
    ax = fig.add_subplot(gs_bot[1, 2])
    cols = NPG_PALETTE[:4]
    for lbl, xs, ys, col in zip(db['labels'], db['X'], db['Y'], cols):
        ax.plot(xs, ys, 'o-', color=col, markersize=3.5, label=str(lbl))
    ax.set_xlabel('$\\omega\\,\\tau_d$')
    ax.set_ylabel('Derivative skewness')
    ax.legend(title='band', loc='upper right', fontsize=5, title_fontsize=5)
    despine(ax)
    panel_label(ax, 'g')
    save_figure(fig, os.path.join(FIGURES_OUT, 'fig3'), formats=['png', 'svg'], dpi=600)
    plt.close(fig)

set_style()

if __name__ == '__main__':
    morphology = simulate_morphology()
    connectivity = simulate_connectivity()
    irreversibility = simulate_pair_irreversibility()
    bands = simulate_frequency_bands()
    plot_figure(connectivity, morphology, irreversibility, bands)
