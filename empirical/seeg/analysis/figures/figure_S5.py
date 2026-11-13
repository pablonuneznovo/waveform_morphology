import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RED = "#E64B35"
CYAN = "#4DBBD5"
TEAL = "#00A087"
NAVY = "#3C5488"
GREY = "#8A8A8A"
REGION_CYCLE = [NAVY, RED, TEAL, CYAN, "#F39B7F", "#8491B4",
                "#91D1C2", "#7E6148", "#DC0000", "#B09C85"]

def apply_style(saved_width_in):
    scale = saved_width_in / 6.27
    sizes = {name: round(points * scale, 2) for name, points in {
        "tick": 6.5, "label": 7.5, "title": 7.5, "panel": 8.5,
        "legend": 6.5, "annot": 6.0, "small": 5.5}.items()}
    sizes["scale"] = scale
    sizes["spine"] = round(0.75 * scale, 2)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": sizes["tick"],
        "axes.labelsize": sizes["label"], "axes.titlesize": sizes["title"],
        "xtick.labelsize": sizes["tick"], "ytick.labelsize": sizes["tick"],
        "legend.fontsize": sizes["legend"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": sizes["spine"],
        "xtick.major.width": sizes["spine"],
        "ytick.major.width": sizes["spine"],
        "xtick.major.size": 2.5 * scale, "ytick.major.size": 2.5 * scale,
        "legend.frameon": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    return sizes

NN = apply_style(11.98)
K  = NN["tick"] / 10.0

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

TABLE_DIR = ROOT / "seeg" / "pair_tables" / "tables" / "bipolar"
OUT_DIR   = ROOT / "seeg" / "analysis" / "outputs"
FIG_DIR   = Path(__file__).resolve().parents[3] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRE_COLOR   = NAVY
POST_COLOR  = RED
NSTIM_COLOR = NAVY
STIM_COLOR  = RED

def panel_label(ax, letter, x=-0.13, y=1.05):
    ax.text(x, y, letter.lower(), transform=ax.transAxes,
            fontsize=13*K, fontweight="bold", va="top")

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

wf = pd.read_csv(TABLE_DIR / "channel_epoch_imf4_features.csv", low_memory=False)

wv = np.load(OUT_DIR / "S5_waveforms_full.npz", allow_pickle=False)
pre_all, post_all = wv["pre"], wv["post"]
is_stim_wv = wv["is_stimulated"].astype(bool)

stim_pre,    stim_post    = pre_all[is_stim_wv],  post_all[is_stim_wv]
nonstim_pre, nonstim_post = pre_all[~is_stim_wv], post_all[~is_stim_wv]
print(f"waveforms - stimulated: {len(stim_pre):,}  non-stimulated: {len(nonstim_pre):,}")

KEY = ["file", "entry_index", "subject", "stim_frequency_hz",
       "channel", "is_stimulated_channel"]
pre_f  = wf[wf["epoch"] == "pre" ][KEY + ["asc2desc", "peak2trough"]]
post_f = wf[wf["epoch"] == "post"][KEY + ["asc2desc", "peak2trough"]]
merged = pre_f.merge(post_f, on=KEY, suffixes=("_pre", "_post"))
merged["delta_asc2desc"]    = merged["asc2desc_post"]    - merged["asc2desc_pre"]
merged["delta_peak2trough"] = merged["peak2trough_post"] - merged["peak2trough_pre"]
merged = merged.dropna(subset=["delta_asc2desc", "delta_peak2trough"])
merged["is_stim"] = merged["is_stimulated_channel"].astype(str).str.lower().eq("true")
stim_full, nonstim_full = merged[merged["is_stim"]], merged[~merged["is_stim"]]
print(f"deltas     - stimulated: {len(stim_full):,}  non-stimulated: {len(nonstim_full):,}")

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
ax_D = fig.add_subplot(gs[1, 1])

x_phase = np.linspace(0, 1, pre_all.shape[1])

def plot_waveform_panel(ax, pre_arr, post_arr, title, label):
    n = pre_arr.shape[0]
    mean_pre,  sem_pre  = pre_arr.mean(axis=0),  pre_arr.std(axis=0)  / np.sqrt(n)
    mean_post, sem_post = post_arr.mean(axis=0), post_arr.std(axis=0) / np.sqrt(n)

    ax.fill_between(x_phase, mean_pre - sem_pre, mean_pre + sem_pre,
                    color=PRE_COLOR, alpha=0.20)
    ax.fill_between(x_phase, mean_post - sem_post, mean_post + sem_post,
                    color=POST_COLOR, alpha=0.20)
    ax.plot(x_phase, mean_pre,  color=PRE_COLOR,  lw=2.2, label="Pre")
    ax.plot(x_phase, mean_post, color=POST_COLOR, lw=2.2, label="Post")

    ax.axhline(0, color="0.75", lw=0.8, ls="--")
    ax.set_xlabel("Cycle phase  (0 → 1)", fontsize=11*K)
    ax.set_ylabel("Normalised amplitude", fontsize=11*K)

    ax.set_xlim(0, 1)
    ax.tick_params(axis="both", labelsize=10*K)
    ax.legend(fontsize=10*K, frameon=False, loc="lower left")
    panel_label(ax, label)

plot_waveform_panel(
    ax_A, stim_pre, stim_post,
    "Stimulated electrodes\nGrand mean ± SEM pre vs post stimulation", "A")
plot_waveform_panel(
    ax_B, nonstim_pre, nonstim_post,
    "Non-stimulated electrodes\nGrand mean ± SEM pre vs post stimulation", "B")

y_min = min(ax_A.get_ylim()[0], ax_B.get_ylim()[0])
y_max = max(ax_A.get_ylim()[1], ax_B.get_ylim()[1])
ax_A.set_ylim(y_min, y_max)
ax_B.set_ylim(y_min, y_max)

rng = np.random.default_rng(42)
for ax, feat, ylabel, letter in [
    (ax_C, "delta_asc2desc",    "Δ Ascent/descent ratio\n(post − pre)", "C"),
    (ax_D, "delta_peak2trough", "Δ Peak/trough ratio\n(post − pre)",    "D"),
]:
    ns_vals = nonstim_full[feat].values
    st_vals = stim_full[feat].values

    parts = ax.violinplot([ns_vals, st_vals], positions=[0, 1],
                          widths=0.55, showmedians=False, showextrema=False)
    for body, col in zip(parts["bodies"], [NSTIM_COLOR, STIM_COLOR]):
        body.set_facecolor(col); body.set_alpha(0.35)
        body.set_edgecolor(col); body.set_linewidth(1)

    for xi, vals, col in zip([0, 1], [ns_vals, st_vals], [NSTIM_COLOR, STIM_COLOR]):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([xi - 0.08, xi + 0.08], [med, med], color=col, lw=2.5, zorder=4)
        ax.plot([xi, xi], [q1, q3], color=col, lw=6,
                solid_capstyle="butt", alpha=0.6, zorder=3)
        n_j = min(len(vals), 2000)
        idx = rng.choice(len(vals), size=n_j, replace=False)
        ax.scatter(xi + rng.uniform(-0.12, 0.12, n_j), vals[idx], s=8,
                   color=col, alpha=0.35, zorder=2, linewidths=0)

    ax.axhline(0, color="0.6", lw=1, ls="--", zorder=1)

    _, p_mw = stats.mannwhitneyu(st_vals, ns_vals, alternative="two-sided")
    y_top = np.percentile(np.concatenate([ns_vals, st_vals]), 99)
    y_sig = y_top * 1.08
    ax.plot([0, 0, 1, 1], [y_top, y_sig, y_sig, y_top], color="0.3", lw=1.2)

    p_label = "p < 0.001" if p_mw < 0.001 else f"p = {p_mw:.3f}"
    ax.text(0.5, y_sig * 1.04, f"{p_label}  {sig_stars(p_mw)}",
            ha="center", fontsize=10*K, color="0.2")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-stimulated", "Stimulated"], fontsize=11*K)
    ax.tick_params(axis="y", labelsize=10*K)
    ax.set_ylabel(ylabel, fontsize=11*K)
    ax.set_xlim(-0.5, 1.5)
    panel_label(ax, letter)

fig.savefig(FIG_DIR / "figS5.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "figS5.svg",           bbox_inches="tight")
plt.close(fig)
print(f"Figure S5 saved to {FIG_DIR}")
