import os
from math import exp
import numpy as np
from scipy.signal import find_peaks
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

NPG_COLORS = {'red': '#E64B35', 'teal': '#4DBBD5', 'green': '#00A087', 'navy': '#3C5488', 'salmon': '#F39B7F', 'lavender': '#8491B4', 'yellow': '#FED439', 'purple': '#7E6148', 'gray': '#8C8C8C', 'lightgray': '#E5E5E5'}

NPG_PALETTE = [NPG_COLORS['red'], NPG_COLORS['teal'], NPG_COLORS['green'], NPG_COLORS['navy'], NPG_COLORS['salmon'], NPG_COLORS['lavender'], NPG_COLORS['yellow'], NPG_COLORS['purple']]

def _make_npg_sequential():
    return LinearSegmentedColormap.from_list('npg_sequential', ['#FFFFFF', NPG_COLORS['navy']], N=256)

def _make_npg_diverging():
    return LinearSegmentedColormap.from_list('npg_diverging', [NPG_COLORS['teal'], '#FFFFFF', NPG_COLORS['red']], N=256)

NPG_CMAPS = {'sequential': _make_npg_sequential(), 'diverging': _make_npg_diverging()}

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

C_MORPH = NPG_COLORS['teal']

C_BASE = NPG_COLORS['gray']

def td_ramp(n):
    return NPG_CMAPS['sequential'](np.linspace(0.35, 0.95, n))

tau_fast = 0.005

tau_r_E, tau_d_E = (0.003, 0.035)

tau_r_I, tau_d_I = (0.005, 0.025)

c_EE, c_IE, c_EI, c_II = (16.0, 12.0, 15.0, 3.0)

sigma_sig = 1.0

theta_E, theta_I = (4.5, 4.0)

dt = 0.0001

NF = 3000

f_vec = np.linspace(0.1, 30, NF)

w_vec = 2 * np.pi * f_vec

def Gam(w, tr, td):
    return 1.0 / ((1 + 1j * w * tr) * (1 + 1j * w * td))

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
    img = load_schematic('Figure 1.png')
    aspect = img.shape[1] / img.shape[0]
    usable_w = FIG_WIDTH['double'] * 0.91
    schem_h = schem_height(aspect, usable_w)
    data_h = 3.9
    total_h = schem_h + data_h + 0.25
    fig = figure(width='double', height=total_h)
    gs_top, gs_bot = two_part_gridspec(fig, total_h, schem_h, data_h, data_rows=2, data_cols=6, hspace=0.4, wspace=1.0)
    add_schematic_panel(fig, gs_top[0, :], img, 'a')
    cols = td_ramp(len(d['td_ex']))
    ax = fig.add_subplot(gs_bot[0, 0:2])
    for j, td in enumerate(d['td_ex']):
        ax.plot(d['ph'], d['cyc'][j], color=cols[j], label=f'{td * 1000.0:.0f} ms')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Phase within cycle')
    ax.set_ylabel('$y_E$ (normalised)')
    ax.legend(title='$\\tau_d$', loc='upper right', fontsize=6, title_fontsize=6)
    despine(ax)
    panel_label(ax, 'b')
    ax = fig.add_subplot(gs_bot[0, 2:4])
    ax.plot(d['td_sw'] * 1000.0, d['rise'], 'o-', color=C_DIR, label='Rise')
    ax.plot(d['td_sw'] * 1000.0, d['dec'], 's-', color=C_SYM, label='Decay')
    ax.set_xlabel('$\\tau_d$ (ms)')
    ax.set_ylabel('Measured time (ms)')
    ax.legend(loc='upper right', fontsize=6)
    despine(ax)
    panel_label(ax, 'c')
    ax = fig.add_subplot(gs_bot[0, 4:6])
    ax.plot(d['td_sw'] * 1000.0, d['rds'], 'o-', color=C_MORPH)
    ax.axhline(0.5, ls='--', lw=0.5, color=C_BASE)
    ax.set_xlabel('$\\tau_d$ (ms)')
    ax.set_ylabel('Rise–decay symmetry')
    despine(ax)
    panel_label(ax, 'd')
    td2 = [0.015, 0.025, 0.033, 0.04]
    cols2 = td_ramp(len(td2))
    ax = fig.add_subplot(gs_bot[1, 1:3])
    for j, td in enumerate(td2):
        ax.plot(f_vec, np.abs(Gam(w_vec, tau_r_E, td)), color=cols2[j], label=f'{td * 1000.0:.0f} ms')
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$|\\Gamma_E(\\omega)|$')
    ax.legend(title='$\\tau_d$', loc='upper right', fontsize=6, title_fontsize=6)
    despine(ax)
    panel_label(ax, 'e')
    ax = fig.add_subplot(gs_bot[1, 3:5])
    for j, td in enumerate(td2):
        ax.plot(f_vec, np.angle(Gam(w_vec, tau_r_E, td)) * 180 / np.pi, color=cols2[j])
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$\\varphi_E(\\omega)$ (deg)')
    despine(ax)
    panel_label(ax, 'f')
    save_figure(fig, os.path.join(FIGURES_OUT, 'fig1'), formats=['png', 'svg'], dpi=600)
    plt.close(fig)

set_style()

if __name__ == '__main__':
    morphology = simulate_morphology()
    plot_figure(morphology)
