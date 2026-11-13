import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

NN = apply_style(9.55)
K  = NN["tick"] / 10.0

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
FIG_DIR = Path(__file__).resolve().parents[3] / "figures"

NROW, NCOL = 2, 2
LEFT_COL_LETTERS = ("A", "C")     # panels that keep the y tick labels
FIG_DIR.mkdir(parents=True, exist_ok=True)

post = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)
# normalise stimulated-channel flag (stored as "True"/"False" strings)
post["is_stimulated_channel"] = (post["is_stimulated_channel"].astype(str)
                                   .str.lower().eq("true"))
# Common electrode subset - all waveform + connectivity metrics valid, same set
# as Fig 1/2/3 and the LME. Electrode N is reported in the manuscript text.
post = post.dropna(subset=["mean_if", "asc2desc", "peak2trough",
                           "mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive"]).reset_index(drop=True)
print(f"Common-subset electrodes: {len(post):,}  |  Subjects: {post['subject'].nunique()}")

WAVEFORM_FEATURES = [
    {"col": "mean_if",     "label": "Instantaneous frequency (Hz)"},
    {"col": "asc2desc",    "label": "Ascent/descent ratio"},
    {"col": "peak2trough", "label": "Peak/trough ratio"},
]
# One acronym set across every figure in the paper (see S6_scatter.py).
CONNECTIVITY = [
    {"col": "mean_coh",       "label": "Coh",     "short": "Coh"},
    {"col": "mean_abs_imcoh", "label": "|ImCoh|", "short": "|ImCoh|"},
    {"col": "pdc_asymmetry",  "label": "PDC",     "short": "PDC"},
    {"col": "mean_io_adaptive",    "label": "BTI", "short": "BTI"},
]

REGIONS_KEEP  = ["frontal","temporal","insula","hippocampus","parietal","occipital","amygdala"]
REGION_COLORS = {r: REGION_CYCLE[i % len(REGION_CYCLE)]
                 for i, r in enumerate(REGIONS_KEEP)}
FREQ_COLORS   = {1.0: NAVY, 50.0: RED}
FREQ_LABELS   = {1.0: "1 Hz", 50.0: "50 Hz"}
STIM_COLORS   = {True: RED, False: NAVY}
STIM_LABELS   = {True: "Stimulated", False: "Non-stimulated"}

rng = np.random.default_rng(42)
SCATTER_N = 3000

def clean(data, x, y):
    return data[[x, y]].dropna()

def spearman(data, x, y):
    c = clean(data, x, y)
    if len(c) < 5:
        return np.nan, np.nan, 0
    r, p = stats.spearmanr(c[x], c[y])
    return r, p, len(c)

def ols(x_arr, y_arr):
    m, b = np.polyfit(x_arr, y_arr, 1)
    xs = np.array([x_arr.min(), x_arr.max()])
    return xs, m * xs + b

def pfmt(p: float) -> str:
    """p rounded to four places reads as 0.0000 when it is merely very small."""
    return "< 0.0001" if p < 1e-4 else "= %.4f" % p

def stars(p):
    if not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def sub_scatter(ax, c, x, y, color, label=None, alpha=0.5, s=18):
    idx = (rng.choice(len(c), size=min(SCATTER_N, len(c)), replace=False)
           if len(c) > SCATTER_N else np.arange(len(c)))
    ax.scatter(c[x].values[idx], c[y].values[idx], s=s, alpha=alpha,
               color=color, label=label, zorder=2, linewidths=0)

def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")

def panel_letter(ax, letter):
    ax.text(-0.12, 1.05, letter.lower(), transform=ax.transAxes,
            fontsize=13*K, fontweight="bold", va="bottom", ha="left")

# --- S10 - region-level bar chart ---
print("S10: region-level ...")
data_r = post[post["region_family"].isin(REGIONS_KEEP)].copy()
fig, axes = plt.subplots(NROW, NCOL, figsize=(9.6, 9.0))
axes = axes.ravel()
for ax, conn, letter in zip(axes, CONNECTIVITY, "ABCDEFG"):
    region_rhos, region_ps, region_ns = [], [], []
    for region in REGIONS_KEEP:
        grp = data_r[data_r["region_family"] == region]
        rho, p, n = spearman(grp, "mean_if", conn["col"])
        region_rhos.append(rho); region_ps.append(p); region_ns.append(n)

    valid_mask = [np.isfinite(p) for p in region_ps]
    p_fdr = [np.nan] * len(region_ps)
    valid_ps = [p for p, v in zip(region_ps, valid_mask) if v]
    if len(valid_ps) > 1:
        corrected = multipletests(valid_ps, method="fdr_bh")[1]
        ci = 0
        for i, v in enumerate(valid_mask):
            if v: p_fdr[i] = corrected[ci]; ci += 1

    colors = [REGION_COLORS[r] for r in REGIONS_KEEP]
    y_pos = np.arange(len(REGIONS_KEEP))
    ax.barh(y_pos, region_rhos, color=colors, alpha=0.85,
            edgecolor="0.3", lw=0.6, height=0.6)
    for yi, (rho, p, fdr, n) in enumerate(zip(region_rhos, region_ps, p_fdr, region_ns)):
        if not np.isfinite(rho):
            continue
        x_off = 0.01 if rho >= 0 else -0.01
        ha = "left" if rho >= 0 else "right"
        ax.text(rho + x_off, yi, f"{stars(fdr)}", va="center",
                ha=ha, fontsize=9*K)
    # With a 2x2 layout the region names belong on both left-column
    # panels (a and c), not only the first.
    ax.set_yticks(y_pos, REGIONS_KEEP if letter in LEFT_COL_LETTERS
                  else [""] * len(REGIONS_KEEP))
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("Spearman rho")
    ax.set_title(f"IF vs {conn['short']}", fontsize=10*K, fontweight="bold")
    ax.set_xlim(-0.55, 0.55)
    panel_letter(ax, letter)

fig.tight_layout()
save(fig, "figS10")

