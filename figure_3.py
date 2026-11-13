import os
from math import exp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numba import njit, prange
from scipy.optimize import root

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

C_DIR = NPG_COLORS['red']

C_SYM = NPG_COLORS['navy']

C_BASE = NPG_COLORS['gray']

def n_shade_ramp(base_hex, n):
    cmap = LinearSegmentedColormap.from_list('n_ramp', ['#FFFFFF', base_hex], N=256)
    return cmap(np.linspace(0.35, 0.95, n))

tau_fast = 0.005

tau_r_E, tau_d_E = (0.003, 0.035)

tau_r_I, tau_d_I = (0.005, 0.025)

c_EE, c_IE, c_EI, c_II = (16.0, 12.0, 15.0, 3.0)

sigma_sig = 1.0

theta_E, theta_I = (4.5, 4.0)

dt = 0.0001

ds = 10

dt_s = dt * ds

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

@njit(parallel=True)
def run_networks(weights, tr_e, td_e, tr_i, td_i, drive, duration, noise, dt, ds, tau_fast, c_ee, c_ie, c_ei, c_ii, theta_e, theta_i, sigma, seed):
    n_rep, n_nodes, _ = weights.shape
    n_steps = int(round(duration / dt))
    n_samples = n_steps // ds
    output = np.empty((n_rep, n_nodes, n_samples))
    noise_scale = np.sqrt(2 * noise / dt)
    for rep in prange(n_rep):
        np.random.seed(seed + rep)
        re = np.full(n_nodes, 0.5)
        ri = np.full(n_nodes, 0.5)
        xe = re.copy()
        ye = re.copy()
        xi = ri.copy()
        yi = ri.copy()
        ue = np.empty(n_nodes)
        ui = np.empty(n_nodes)
        coupling = np.empty(n_nodes)
        sample = 0
        for step in range(n_steps):
            for node in range(n_nodes):
                ue[node] = c_ee * ye[node] - c_ie * yi[node] - theta_e + drive
                ui[node] = c_ei * ye[node] - c_ii * yi[node] - theta_i
            for node in range(n_nodes):
                weighted = 0.0
                for source in range(n_nodes):
                    weighted += weights[rep, node, source] * ye[source]
                se = 1.0 / (1.0 + np.exp(-ue[node] / sigma))
                coupling[node] = se * (1.0 - se) / sigma * weighted / tau_fast
            for node in range(n_nodes):
                se = 1.0 / (1.0 + np.exp(-ue[node] / sigma))
                si = 1.0 / (1.0 + np.exp(-ui[node] / sigma))
                re_new = re[node] + dt * ((-re[node] + se) / tau_fast + coupling[node] + noise_scale * np.random.standard_normal())
                ri_new = ri[node] + dt * ((-ri[node] + si) / tau_fast + noise_scale * np.random.standard_normal())
                re[node] = min(1.0, max(0.0, re_new))
                ri[node] = min(1.0, max(0.0, ri_new))
                xe[node] += dt * (-xe[node] + re[node]) / tr_e
                ye[node] += dt * (-ye[node] + xe[node]) / td_e
                xi[node] += dt * (-xi[node] + ri[node]) / tr_i
                yi[node] += dt * (-yi[node] + xi[node]) / td_i
            if (step + 1) % ds == 0 and sample < n_samples:
                for node in range(n_nodes):
                    output[rep, node, sample] = ye[node]
                sample += 1
    return output

P_hopf, f_hopf = scan_bif(sigma_sig, theta_E, theta_I, tau_r_E, tau_d_E, tau_r_I, tau_d_I, 6)
N_REP = 60
KAPPA = 0.5
noise_rng = np.random.default_rng(81099)


def exact_symmetric_weights(base):
    sym = (base + base.T) / 2

    def residual(x):
        scale = np.exp(x)
        return scale * (sym @ scale) - KAPPA

    with np.errstate(over='ignore', invalid='ignore'):
        fit = root(residual, np.zeros(sym.shape[0]), method='hybr')
        if not fit.success or np.max(np.abs(residual(fit.x))) > 1e-9:
            return None
        scale = np.exp(fit.x)
    weights = scale[:, None] * sym * scale[None, :]
    if not np.allclose(weights, weights.T, atol=1e-12):
        raise RuntimeError('Symmetry check failed')
    if not np.allclose(weights.sum(axis=1), KAPPA, atol=1e-9):
        raise RuntimeError('Input-strength check failed')
    return weights


def directed_weights(base):
    totals = base.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError('A node has no incoming connection')
    return KAPPA * base / totals


def weighted_reciprocity(weights):
    return np.minimum(weights, weights.T).sum() / weights.sum()


def network_pairs(size, base_rng):
    symmetric = []
    directed = []
    attempts = 0
    while len(symmetric) < N_REP:
        attempts += 1
        if attempts > 20 * N_REP:
            raise RuntimeError(f'Could not construct {N_REP} balanced networks for N={size}')
        base = base_rng.random((size, size)) * (base_rng.random((size, size)) < 0.5)
        np.fill_diagonal(base, 0)
        if np.any(base.sum(axis=1) == 0):
            continue
        exact = exact_symmetric_weights(base)
        if exact is None:
            continue
        symmetric.append(exact)
        directed.append(directed_weights(base))
    print(f'N={size}: {N_REP} networks from {attempts} draws', flush=True)
    return symmetric, directed


def reversibility_deco(signals, lag):
    signals = signals - signals.mean(axis=1, keepdims=True)
    sd = signals.std(axis=1, keepdims=True)
    sd[sd < np.finfo(float).eps] = np.finfo(float).eps
    signals = signals / sd
    forward = signals[:, :-lag] @ signals[:, lag:].T / (signals.shape[1] - lag)
    forward = np.clip(forward, -0.999999, 0.999999)
    transformed = -0.5 * np.log(1 - forward ** 2)
    return np.mean((transformed - transformed.T) ** 2)


def simulate(weights, drive, lag, tr_e=tau_r_E, td_e=tau_d_E):
    values = []
    for start in range(0, len(weights), 15):
        chunk = np.stack(weights[start:start + 15])
        signals = run_networks(chunk, tr_e, td_e, tau_r_I, tau_d_I, drive, 25, 0.0004, dt, ds, tau_fast, c_EE, c_IE, c_EI, c_II, theta_E, theta_I, sigma_sig, int(noise_rng.integers(1, 1_000_000_000)))
        values.extend(reversibility_deco(signal, lag) for signal in signals)
    return np.asarray(values)


def simulate_network():
    n_vec = np.array([8, 16, 24, 32, 48])
    nb_vec = np.array([8, 16, 32])
    alpha_vec = np.linspace(0, 1, 6)
    rho_vec = np.array([1, 2, 4, 7, 11], float)
    drive = P_hopf + 0.4
    lag = max(1, int(round(1 / (4 * max(f_hopf, 1)) / dt_s)))

    i1 = np.empty((len(n_vec), N_REP, len(alpha_vec)))
    r1 = np.empty_like(i1)
    base_rng = np.random.default_rng(20261105)
    for k, size in enumerate(n_vec):
        symmetric, directed = network_pairs(int(size), base_rng)
        for j, alpha in enumerate(alpha_vec):
            weights = [(1 - alpha) * x + alpha * y for x, y in zip(symmetric, directed)]
            r1[k, :, j] = [weighted_reciprocity(weight) for weight in weights]
            i1[k, :, j] = simulate(weights, drive, lag)
            print(f'Reciprocity: N={size}, alpha={alpha:.1f}', flush=True)

    tau_m = np.sqrt(tau_r_E * tau_d_E)
    isym2 = np.empty((len(nb_vec), N_REP, len(rho_vec)))
    idir2 = np.empty_like(isym2)
    base_rng = np.random.default_rng(20262105)
    for k, size in enumerate(nb_vec):
        symmetric, directed = network_pairs(int(size), base_rng)
        for j, rho in enumerate(rho_vec):
            tr_e = tau_m / np.sqrt(rho)
            td_e = tau_m * np.sqrt(rho)
            hopf, frequency = scan_bif(sigma_sig, theta_E, theta_I, tr_e, td_e, tau_r_I, tau_d_I, 6)
            if not np.isfinite(hopf) or not np.isfinite(frequency):
                raise RuntimeError(f'Could not locate bifurcation for rho={rho}')
            condition_lag = max(1, int(round(1 / (4 * max(frequency, 1)) / dt_s)))
            condition_drive = hopf + 0.4
            isym2[k, :, j] = simulate(symmetric, condition_drive, condition_lag, tr_e, td_e)
            idir2[k, :, j] = simulate(directed, condition_drive, condition_lag, tr_e, td_e)
            print(f'Synaptic asymmetry: N={size}, rho={rho:g}', flush=True)

    isym3 = np.empty((N_REP, len(n_vec)))
    idir3 = np.empty_like(isym3)
    base_rng = np.random.default_rng(20263105)
    for j, size in enumerate(n_vec):
        symmetric, directed = network_pairs(int(size), base_rng)
        isym3[:, j] = simulate(symmetric, drive, lag)
        idir3[:, j] = simulate(directed, drive, lag)
        print(f'Network size: N={size}', flush=True)

    return dict(N_vec=n_vec, Nb_vec=nb_vec, alpha_vec=alpha_vec, I1_byN=i1, R1_byN=r1, rho_vec=rho_vec, Isym2_byN=isym2, Idir2_byN=idir2, Isym3=isym3, Idir3=idir3)

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

def plot_figure(d):
    img = load_schematic('Figure 3.png')
    aspect = img.shape[1] / img.shape[0]
    usable_w = FIG_WIDTH['double'] * 0.91
    schem_h = schem_height(aspect, usable_w)
    data_h = 4.1
    total_h = schem_h + data_h + 0.25
    fig = figure(width='double', height=total_h)
    gs_top, gs_bot = two_part_gridspec(fig, total_h, schem_h, data_h, data_rows=2, data_cols=2, hspace=0.4, wspace=0.5)
    add_schematic_panel(fig, gs_top[0, :], img, 'a')
    ax = fig.add_subplot(gs_bot[0, 0])
    N_ramp = n_shade_ramp(NPG_COLORS['teal'], d['N_vec'].size)
    for k, Nn in enumerate(d['N_vec']):
        rvec = d['R1_byN'][k].mean(0)
        o = np.argsort(rvec)
        ax.plot(rvec[o], d['I1_byN'][k].mean(0)[o], 'o-', color=N_ramp[k], ms=3, lw=1.2, label=f'N={int(Nn)}')
    ax.set_xlabel('Network reciprocity $r$')
    ax.set_ylabel('Global irreversibility')
    ax.legend(fontsize=5.5, loc='upper right', title='Network size', title_fontsize=5.5)
    despine(ax)
    panel_label(ax, 'b')
    ax = fig.add_subplot(gs_bot[0, 1])
    Nb_ramp_dir = n_shade_ramp(NPG_COLORS['red'], d['Nb_vec'].size)
    Nb_ramp_sym = n_shade_ramp(NPG_COLORS['navy'], d['Nb_vec'].size)
    for k, Nn in enumerate(d['Nb_vec']):
        ax.plot(d['rho_vec'], d['Idir2_byN'][k].mean(0), 'o-', color=Nb_ramp_dir[k], ms=3, lw=1.2, label=f'Directed, N={int(Nn)}')
    for k, Nn in enumerate(d['Nb_vec']):
        ax.plot(d['rho_vec'], d['Isym2_byN'][k].mean(0), 's--', color=Nb_ramp_sym[k], ms=3, lw=1.2, label=f'Symmetric, N={int(Nn)}')
    ax.axvline(1, ls='--', lw=0.5, color=C_BASE)
    ax.set_xlabel('Synaptic asymmetry $\\rho=\\tau_d/\\tau_r$')
    ax.set_ylabel('Global irreversibility')
    ax.legend(fontsize=5, loc='upper right', ncol=2, columnspacing=0.8, handletextpad=0.4)
    despine(ax)
    panel_label(ax, 'c')
    ax = fig.add_subplot(gs_bot[1, 0])
    ax.errorbar(d['N_vec'], d['Idir3'].mean(0), d['Idir3'].std(0), fmt='o-', color=C_DIR, capsize=2, label='Directed')
    ax.errorbar(d['N_vec'], d['Isym3'].mean(0), d['Isym3'].std(0), fmt='s-', color=C_SYM, capsize=2, label='Symmetric')
    ax.set_yscale('log')
    ax.set_xlabel('Network size $N$')
    ax.set_ylabel('Global irreversibility')
    ax.legend(loc='lower right')
    despine(ax)
    panel_label(ax, 'd')
    ax = fig.add_subplot(gs_bot[1, 1])
    ratio = d['Idir3'].mean(0) / np.clip(d['Isym3'].mean(0), 1e-30, None)
    ax.plot(d['N_vec'], ratio, 'o-', color=C_BASE)
    ax.axhline(1.0, ls='--', lw=0.5, color=C_BASE)
    ax.set_ylim(0, max(1.6, np.nanmax(ratio) * 1.15))
    ax.set_xlabel('Network size $N$')
    ax.set_ylabel('Directed / symmetric ratio')
    despine(ax)
    panel_label(ax, 'e')
    save_figure(fig, os.path.join(FIGURES_OUT, 'fig3'), formats=['png', 'svg'], dpi=600)
    plt.close(fig)

set_style()

if __name__ == '__main__':
    plot_figure(simulate_network())
