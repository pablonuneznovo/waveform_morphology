import os
from math import exp
import numpy as np
from scipy.signal import welch, csd
from matplotlib.lines import Line2D
from numba import njit
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

P_hopf, f_hopf = scan_bif(sigma_sig, theta_E, theta_I, tau_r_E, tau_d_E, tau_r_I, tau_d_I, 6)

mph = 1.8

sig = sigma_sig

EPS = 0.05

def integrate(s, N, tdE, P, record=False):
    RE, RI, xE, yE, xI, yI = s
    out = np.empty(N)
    St = np.empty((6, N)) if record else None
    for n in range(N):
        uE = c_EE * yE - c_IE * yI - theta_E + P
        uI = c_EI * yE - c_II * yI - theta_I
        SE = 1.0 / (1.0 + exp(-uE / sig))
        SI = 1.0 / (1.0 + exp(-uI / sig))
        RE = min(1.0, max(0.0, RE + dt * (-RE + SE) / tau_fast))
        RI = min(1.0, max(0.0, RI + dt * (-RI + SI) / tau_fast))
        xE += dt * (-xE + RE) / tau_r_E
        yE += dt * (-yE + xE) / tdE
        xI += dt * (-xI + RI) / tau_r_I
        yI += dt * (-yI + xI) / tau_d_I
        yE = min(1.0, max(0.0, yE))
        out[n] = yE
        if record:
            St[:, n] = (RE, RI, xE, yE, xI, yI)
    return (out, St) if record else out

def working_point(tdE):
    Ph, fh = scan_bif(sig, theta_E, theta_I, tau_r_E, tdE, tau_r_I, tau_d_I, 6)
    if np.isnan(Ph) or fh < 2.5 or fh > 7:
        Ph, fh = (P_hopf, f_hopf)
    return (Ph + mph, fh)

def hopf_point(tdE):
    Ph, fh = scan_bif(sig, theta_E, theta_I, tau_r_E, tdE, tau_r_I, tau_d_I, 6)
    if np.isnan(Ph) or fh < 2.5 or fh > 7:
        Ph, fh = (P_hopf, f_hopf)
    return (Ph, fh)

def last_up(y):
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    return i - y[i] / (y[i + 1] - y[i])

def cycle_states(tdE, P, fh, nph=40, settle=6.0):
    s0 = [0.5] * 6
    _, St = integrate(s0, int(round(settle / dt)), tdE, P, record=True)
    s = St[:, -1].copy()
    Tc = 1.0 / fh
    Nrec = int(round(3.0 * Tc / dt))
    y, St2 = integrate(s, Nrec, tdE, P, record=True)
    yd = y - y.mean()
    up = np.where((yd[:-1] < 0) & (yd[1:] >= 0))[0]
    c0 = up[0]
    per_min = int(round(0.5 * Tc / dt))
    c1 = up[up > c0 + per_min][0]
    period = int(c1 - c0)
    idx = c0 + np.round(np.linspace(0, period, nph, endpoint=False)).astype(int)
    return (St2[:, idx], period, yd[c0:c0 + period])

def prc(states, period, tdE, P):
    nph = states.shape[1]
    L = int(round(3 * period))
    Z = np.full(nph, np.nan)
    for k in range(nph):
        s0 = states[:, k].copy()
        yu = integrate(s0, L, tdE, P)
        sp = s0.copy()
        sp[0] = min(1.0, sp[0] + EPS)
        yp = integrate(sp, L, tdE, P)
        ta = last_up(yu - yu.mean())
        tb = last_up(yp - yp.mean())
        if np.isnan(ta) or np.isnan(tb):
            continue
        dT = tb - ta
        dT = (dT + period / 2) % period - period / 2
        Z[k] = 2 * np.pi * dT / period / EPS
    return Z

def _pair_step(st, tdE, P, W01, W10):
    REa, RIa, xEa, yEa, xIa, yIa, REb, RIb, xEb, yEb, xIb, yIb = st
    uEa = c_EE * yEa - c_IE * yIa - theta_E + P
    uIa = c_EI * yEa - c_II * yIa - theta_I
    uEb = c_EE * yEb - c_IE * yIb - theta_E + P
    uIb = c_EI * yEb - c_II * yIb - theta_I
    coupA = 1.0 / (1.0 + exp(-uEa / sig)) * (1.0 - 1.0 / (1.0 + exp(-uEa / sig))) / sig * (W01 * yEb) / tau_fast
    coupB = 1.0 / (1.0 + exp(-uEb / sig)) * (1.0 - 1.0 / (1.0 + exp(-uEb / sig))) / sig * (W10 * yEa) / tau_fast
    REa = min(1.0, max(0.0, REa + dt * ((-REa + 1.0 / (1.0 + exp(-uEa / sig))) / tau_fast + coupA)))
    RIa = min(1.0, max(0.0, RIa + dt * (-RIa + 1.0 / (1.0 + exp(-uIa / sig))) / tau_fast))
    xEa += dt * (-xEa + REa) / tau_r_E
    yEa += dt * (-yEa + xEa) / tdE
    xIa += dt * (-xIa + RIa) / tau_r_I
    yIa += dt * (-yIa + xIa) / tau_d_I
    REb = min(1.0, max(0.0, REb + dt * ((-REb + 1.0 / (1.0 + exp(-uEb / sig))) / tau_fast + coupB)))
    RIb = min(1.0, max(0.0, RIb + dt * (-RIb + 1.0 / (1.0 + exp(-uIb / sig))) / tau_fast))
    xEb += dt * (-xEb + REb) / tau_r_E
    yEb += dt * (-yEb + xEb) / tdE
    xIb += dt * (-xIb + RIb) / tau_r_I
    yIb += dt * (-yIb + xIb) / tau_d_I
    yEa = min(1.0, max(0.0, yEa))
    yEb = min(1.0, max(0.0, yEb))
    return [REa, RIa, xEa, yEa, xIa, yIa, REb, RIb, xEb, yEb, xIb, yIb]

def integrate_pair(s, N, tdE, P, W, kick_node=None, record=False):
    st = list(s)
    if kick_node == 'A':
        st[0] = min(1.0, st[0] + EPS)
    elif kick_node == 'B':
        st[6] = min(1.0, st[6] + EPS)
    W01, W10 = (W[0, 1], W[1, 0])
    yA = np.empty(N)
    yB = np.empty(N)
    St = np.empty((12, N)) if record else None
    for n in range(N):
        st = _pair_step(st, tdE, P, W01, W10)
        yA[n] = st[3]
        yB[n] = st[9]
        if record:
            St[:, n] = st
    return (yA, yB, St) if record else (yA, yB)

def pair_cycle(tdE, P, W, fh, settle=6.0):
    _, _, St = integrate_pair([0.5] * 12, int(round(settle / dt)), tdE, P, W, record=True)
    s = list(St[:, -1])
    Tc = 1.0 / fh
    yA, _, St2 = integrate_pair(s, int(round(3.0 * Tc / dt)), tdE, P, W, record=True)
    yd = yA - yA.mean()
    up = np.where((yd[:-1] < 0) & (yd[1:] >= 0))[0]
    c0 = up[0]
    c1 = up[up > c0 + int(round(0.5 * Tc / dt))][0]
    period = int(c1 - c0)
    return (St2[:, c0:c0 + period], period, St2[3, c0:c0 + period], St2[9, c0:c0 + period])

def imagcoh_band(yA, yB, fs, nseg, band):
    f, pxx = welch(yA, fs=fs, window='hann', nperseg=nseg)
    _, pyy = welch(yB, fs=fs, window='hann', nperseg=nseg)
    _, pxy = csd(yA, yB, fs=fs, window='hann', nperseg=nseg)
    ic = np.imag(np.conj(pxy)) / np.sqrt(pxx * pyy)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.mean(np.abs(ic[m])))

@njit(cache=True)
def step_chunk(state, noise, uniforms, w01, w10, targets, phase, last, fired, previous, thr, period, offset, td, P, D, eps):
    steps, _, C = noise.shape
    out = np.empty((C, 2, steps // 10))
    sq = np.sqrt(2 * D / 0.0001)
    for t in range(steps):
        n = offset + t
        for c in range(C):
            s = state[c]
            ue = 16 * s[3] - 12 * s[5] - 4.5 + P
            ui = 15 * s[3] - 3 * s[5] - 4
            ve = 16 * s[9] - 12 * s[11] - 4.5 + P
            vi = 15 * s[9] - 3 * s[11] - 4
            se = 1 / (1 + np.exp(-ue))
            si = 1 / (1 + np.exp(-ui))
            te = 1 / (1 + np.exp(-ve))
            ti = 1 / (1 + np.exp(-vi))
            ca = se * (1 - se) * w01[c] * s[9] / 0.005
            cb = te * (1 - te) * w10[c] * s[3] / 0.005
            s[0] = min(1.0, max(0.0, s[0] + 0.0001 * ((-s[0] + se) / 0.005 + ca + sq * noise[t, 0, c])))
            s[1] = min(1.0, max(0.0, s[1] + 0.0001 * ((-s[1] + si) / 0.005 + sq * noise[t, 1, c])))
            s[6] = min(1.0, max(0.0, s[6] + 0.0001 * ((-s[6] + te) / 0.005 + cb + sq * noise[t, 2, c])))
            s[7] = min(1.0, max(0.0, s[7] + 0.0001 * ((-s[7] + ti) / 0.005 + sq * noise[t, 3, c])))
            s[2] += 0.0001 * (-s[2] + s[0]) / 0.003
            s[3] = min(1.0, max(0.0, s[3] + 0.0001 * (-s[3] + s[2]) / td))
            s[4] += 0.0001 * (-s[4] + s[1]) / 0.005
            s[5] += 0.0001 * (-s[5] + s[4]) / 0.025
            s[8] += 0.0001 * (-s[8] + s[6]) / 0.003
            s[9] = min(1.0, max(0.0, s[9] + 0.0001 * (-s[9] + s[8]) / td))
            s[10] += 0.0001 * (-s[10] + s[7]) / 0.005
            s[11] += 0.0001 * (-s[11] + s[10]) / 0.025
            cross = previous[c] < thr and s[3] >= thr
            if cross:
                last[c] = n
                fired[c] = False
                if targets[c] < 0:
                    phase[c] = uniforms[t, c]
            if not np.isnan(targets[c]) and (not fired[c]) and (n - last[c] >= phase[c] * period) and (n - last[c] < period):
                s[0] = min(1.0, max(0.0, s[0] + eps))
                fired[c] = True
            previous[c] = s[3]
            if (t + 1) % 10 == 0:
                out[c, 0, t // 10] = s[3]
                out[c, 1, t // 10] = s[9]
    return out

def initialize(C, period, targets):
    return (np.full((C, 12), 0.5), np.where(np.isnan(targets) | (targets < 0), 0.0, targets), np.full(C, -10 * period, dtype=np.int64), np.ones(C, dtype=np.bool_), np.full(C, 0.5))

def run_pair_perturbed(w01, w10, targets, td, P, D, T, rng, thr, period, eps):
    C = len(targets)
    steps = int(round(T / 0.0001))
    state, phase, last, fired, previous = initialize(C, period, targets)
    out = np.empty((C, 2, steps // 10))
    randomized = np.any(~np.isnan(targets) & (targets < 0))
    for offset in range(0, steps, 10000):
        count = min(10000, steps - offset)
        noise = np.empty((count, 4, C))
        uniforms = np.zeros((count, C))
        for t in range(count):
            for k in range(4):
                noise[t, k] = rng.standard_normal(C)
            if randomized:
                uniforms[t] = rng.random(C)
        out[:, :, offset // 10:(offset + count) // 10] = step_chunk(state, noise, uniforms, w01, w10, targets, phase, last, fired, previous, thr, period, offset, td, P, D, eps)
    return (out[:, 0], out[:, 1])

ORIG_DIR = os.path.dirname(os.path.abspath(__file__))

FIGURES_OUT = os.path.join(ORIG_DIR, 'figures')

SCHEMATICS = os.path.join(ORIG_DIR, 'abstracts')

def load_schematic(name):
    return plt.imread(os.path.join(SCHEMATICS, name))

SCHEM_SCALE = 0.75

def add_schematic_panel(fig, gs_slice, img, label='a'):
    ax = fig.add_subplot(gs_slice)
    ax.imshow(img, aspect='equal')
    ax.axis('off')
    panel_label(ax, label, x=-0.02, y=1.03)
    return ax

def schem_height(aspect, usable_w, scale=SCHEM_SCALE):
    return scale * usable_w / aspect

def two_part_gridspec(fig, total_h, schem_h, data_h, data_rows, data_cols, hspace=0.4, wspace=0.5):
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

def plot_figure():
    img = load_schematic('Figure 4.png')
    aspect = img.shape[1] / img.shape[0]
    usable_w = FIG_WIDTH['double'] * 0.91
    schem_h = schem_height(aspect, usable_w)
    data_h = 4.1
    total_h = schem_h + data_h + 0.25
    fig = figure(width='double', height=total_h)
    gs_top, gs_bot = two_part_gridspec(fig, total_h, schem_h, data_h, data_rows=2, data_cols=2, hspace=0.4, wspace=0.5)
    add_schematic_panel(fig, gs_top[0, :], img, 'a')
    kappa = 0.5
    ax = fig.add_subplot(gs_bot[0, 0])
    for tdE, c, lab in [(0.015, C_DIR, 'fast $\\tau_d$ (skewed)'), (0.04, C_SYM, 'slow $\\tau_d$ (sinusoidal)')]:
        Pn, fhn = working_point(tdE)
        states, period, _ = cycle_states(tdE, Pn, fhn)
        Z = prc(states, period, tdE, Pn)
        ph = np.linspace(0, 1, states.shape[1], endpoint=False)
        ax.plot(ph, Z, 'o-', color=c, ms=3, label=lab)
    ax.axhline(0, ls=':', lw=0.5, color=C_BASE)
    ax.set_xlabel('Absolute phase of perturbation')
    ax.set_ylabel('Phase response $Z(\\phi)$')
    ax.legend(fontsize=6, loc='upper right')
    despine(ax)
    panel_label(ax, 'b', y=1.12)
    tdE = 0.015
    Ph, fh = hopf_point(tdE)
    P = Ph + 0.8
    nph = 40
    ph = np.linspace(0, 1, nph, endpoint=False)
    ax = fig.add_subplot(gs_bot[0, 1])
    b_curves = {}
    for W, tl, c, mk in [(np.array([[0.0, 0.0], [kappa, 0.0]]), 'Directed', C_DIR, 'o'), (np.array([[0.0, kappa / 2], [kappa / 2, 0.0]]), 'Symmetric', C_SYM, 's')]:
        Stc, period, _, _ = pair_cycle(tdE, P, W, fh)
        L = 2 * period
        idx = np.round(np.linspace(0, period, nph, endpoint=False)).astype(int)
        Pab = np.empty(nph)
        Pba = np.empty(nph)
        for k, ii in enumerate(idx):
            s0 = list(Stc[:, ii])
            bA, bB = integrate_pair(s0, L, tdE, P, W)
            _, kAB = integrate_pair(s0, L, tdE, P, W, kick_node='A')
            kBA, _ = integrate_pair(s0, L, tdE, P, W, kick_node='B')
            Pab[k] = np.sqrt(np.mean((kAB - bB)[:period] ** 2))
            Pba[k] = np.sqrt(np.mean((kBA - bA)[:period] ** 2))
        b_curves[tl] = (Pab, Pba)
        ax.plot(ph, Pab, mk + '-', color=c, ms=3, label=f'{tl} A$\\to$B')
        ax.plot(ph, Pba, mk + '--', color=c, ms=3, alpha=0.55, label=f'{tl} B$\\to$A')
    ax.set_xlabel('Absolute phase of perturbation')
    ax.set_ylabel('Propagated response (rms)')
    ax.legend(fontsize=5.2, loc='upper left', ncol=1)
    despine(ax)
    panel_label(ax, 'c', y=1.12)
    peak_ph_sym = ph[np.argmax(b_curves['Symmetric'][0])]
    D = 0.0008
    T = 80.0
    eps_train = 0.12
    nRepC = 30
    W_dir_ref = np.array([[0.0, 0.0], [kappa, 0.0]])
    _, _, yAc, _ = pair_cycle(tdE, P, W_dir_ref, fh)
    thr = float(yAc.mean())
    period_nom = int(round(1.0 / fh / dt))
    phis = np.linspace(0, 1, 8, endpoint=False)
    tops = [('Directed', (0.0, kappa), C_DIR), ('Symmetric', (kappa / 2, kappa / 2), C_SYM)]
    W01v, W10v, phiv, meta = ([], [], [], [])
    for tl, (w01, w10), c in tops:
        for pp in [np.nan, -1.0] + list(phis):
            W01v.append(w01)
            W10v.append(w10)
            phiv.append(pp)
            meta.append((tl, pp))
    W01v = np.array(W01v)
    W10v = np.array(W10v)
    phiv = np.array(phiv)
    fs_s = 1.0 / dt_s
    deco_lag = max(1, int(round(0.05 / dt_s)))
    band = (2.0, 8.0)
    irr_reps = np.empty((nRepC, len(meta)))
    ico_reps = np.empty((nRepC, len(meta)))
    for r in range(nRepC):
        if (r + 1) % 5 == 0:
            print(f'Noisy-pair realization {r + 1}/{nRepC}', flush=True)
        rng = np.random.default_rng(1000 + r)
        YA, YB = run_pair_perturbed(W01v, W10v, phiv, tdE, P, D, T, rng, thr, period_nom, eps_train)
        nseg = min(YA.shape[1], int(round(4.0 * fs_s)))
        irr_reps[r] = [reversibility_deco(np.vstack([YA[j], YB[j]]), deco_lag) for j in range(len(meta))]
        ico_reps[r] = [imagcoh_band(YA[j], YB[j], fs_s, nseg, band) for j in range(len(meta))]
    irr = irr_reps.mean(0)
    irr_sem = irr_reps.std(0, ddof=1) / np.sqrt(nRepC)
    ico = ico_reps.mean(0)
    ico_sem = ico_reps.std(0, ddof=1) / np.sqrt(nRepC)
    for vals, sems, ax, ylab, plab in [(irr, irr_sem, fig.add_subplot(gs_bot[1, 0]), 'Bivariate irreversibility', 'd'), (ico, ico_sem, fig.add_subplot(gs_bot[1, 1]), '$|$Imaginary coherence$|$', 'e')]:
        for tl, (w01, w10), c in tops:
            base = [vals[j] for j, m in enumerate(meta) if m[0] == tl and np.isnan(m[1])][0]
            randv = [vals[j] for j, m in enumerate(meta) if m[0] == tl and m[1] == -1.0][0]
            php = np.array([m[1] for j, m in enumerate(meta) if m[0] == tl and (not np.isnan(m[1])) and (m[1] >= 0)])
            vv = np.array([vals[j] for j, m in enumerate(meta) if m[0] == tl and (not np.isnan(m[1])) and (m[1] >= 0)])
            se = np.array([sems[j] for j, m in enumerate(meta) if m[0] == tl and (not np.isnan(m[1])) and (m[1] >= 0)])
            o = np.argsort(php)
            ax.plot(php[o], vv[o], 'o-', color=c, ms=3, label=tl)
            ax.fill_between(php[o], (vv - se)[o], (vv + se)[o], color=c, alpha=0.15, lw=0)
            ax.axhline(base, ls='--', lw=0.7, color=c, alpha=0.6)
            ax.axhline(randv, ls=':', lw=1.1, color=c, alpha=0.9)
        ax.set_xlabel('Phase of perturbation $\\phi$')
        ax.set_ylabel(ylab)
        despine(ax)
        panel_label(ax, plab, y=1.12)
    handles = [Line2D([], [], color=C_DIR, marker='o', ms=3, label='Directed'), Line2D([], [], color=C_SYM, marker='o', ms=3, label='Symmetric'), Line2D([], [], color=C_BASE, ls='--', label='unperturbed'), Line2D([], [], color=C_BASE, ls=':', label='random phase')]
    fig.axes[3].legend(handles=handles, fontsize=5, loc='upper left', ncol=1)
    save_figure(fig, os.path.join(FIGURES_OUT, 'fig4'), formats=['png', 'svg'], dpi=600)
    plt.close(fig)

set_style()

if __name__ == '__main__':
    plot_figure()
