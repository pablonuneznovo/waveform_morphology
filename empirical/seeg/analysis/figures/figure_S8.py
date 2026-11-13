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
LEFT_COL_LETTERS = ("A", "C")
FIG_DIR.mkdir(parents=True, exist_ok=True)

post = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)

post["is_stimulated_channel"] = (post["is_stimulated_channel"].astype(str)
                                   .str.lower().eq("true"))

post = post.dropna(subset=["mean_if", "asc2desc", "peak2trough",
                           "mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive"]).reset_index(drop=True)
print(f"Common-subset electrodes: {len(post):,}  |  Subjects: {post['subject'].nunique()}")

WAVEFORM_FEATURES = [
    {"col": "mean_if",     "label": "Instantaneous frequency (Hz)"},
    {"col": "asc2desc",    "label": "Ascent/descent ratio"},
    {"col": "peak2trough", "label": "Peak/trough ratio"},
]

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

print("S8: stimulated vs non-stimulated ...")
fig, axes = plt.subplots(NROW, NCOL, figsize=(9.6, 8.6))
axes = axes.ravel()
wf = WAVEFORM_FEATURES[0]
for ax, conn, letter in zip(axes, CONNECTIVITY, "ABCDEFG"):
    for is_stim in (False, True):
        grp = post[post["is_stimulated_channel"] == is_stim]
        c = clean(grp, wf["col"], conn["col"])
        if len(c) >= 5:
            sub_scatter(ax, c, wf["col"], conn["col"], STIM_COLORS[is_stim],
                        label=STIM_LABELS[is_stim])
            xs, ys = ols(c[wf["col"]].values, c[conn["col"]].values)
            ax.plot(xs, ys, color=STIM_COLORS[is_stim], lw=2.2, zorder=4)
            rho, p, n = spearman(grp, wf["col"], conn["col"])
            lbl = "Stim" if is_stim else "Non-stim"
            yoff = 0.97 if not is_stim else 0.78
            ax.text(0.03, yoff,
                    f"{lbl}: rho={rho:.3f}, p {pfmt(p)} {stars(p)}",
                    transform=ax.transAxes, va="top", fontsize=8*K,
                    color=STIM_COLORS[is_stim],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=STIM_COLORS[is_stim], alpha=0.9, lw=0.8))
    ax.axhline(0, color="0.75", lw=0.7, ls="--")
    ax.set_xlabel(wf["label"], fontsize=9*K)
    ax.set_title(conn["short"], fontsize=10*K, fontweight="bold")
    panel_letter(ax, letter)
    if letter == "A":
        ax.legend(fontsize=8*K, frameon=False, loc="lower right")

fig.tight_layout()
save(fig, "figS8")

