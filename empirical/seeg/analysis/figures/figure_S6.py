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

NN = apply_style(12.16)
K  = NN["tick"] / 10.0

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
FIG_DIR = Path(__file__).resolve().parents[3] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = NAVY
RED  = RED

post = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)

COMMON_COLS = ["mean_if", "asc2desc", "peak2trough",
               "mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive"]
post = post.dropna(subset=COMMON_COLS).reset_index(drop=True)

REPEAT_SESSIONS = {"HEJ_Subject004": "HEJ_Subject001",
                   "HEJ_Subject015": "HEJ_Subject014"}
post["record"]  = post["subject"]
post["subject"] = post["subject"].replace(REPEAT_SESSIONS)
print(f"Common-subset electrodes: {len(post):,}  |  "
      f"records: {post['record'].nunique()}  patients: {post['subject'].nunique()}")

WAVEFORM = [
    {"col": "mean_if",     "label": "Instantaneous\nfrequency (Hz)"},
    {"col": "asc2desc",    "label": "Ascent/descent\nratio"},
    {"col": "peak2trough", "label": "Peak/trough\nratio"},
]

CONNECTIVITY = [
    {"col": "mean_coh",       "label": "Coh",     "short": "Coh"},
    {"col": "mean_abs_imcoh", "label": "|ImCoh|", "short": "|ImCoh|"},
    {"col": "pdc_asymmetry",  "label": "PDC",     "short": "PDC"},
    {"col": "mean_io_adaptive",    "label": "BTI", "short": "BTI"},
]

def clean(df, x, y):
    return df[[x, y]].dropna()

def spearman_rho(df, x, y):
    c = clean(df, x, y)
    if len(c) < 5:
        return np.nan, np.nan, 0
    r, p = stats.spearmanr(c[x], c[y])
    return r, p, len(c)

def ols_line(x_arr, y_arr):
    m, b = np.polyfit(x_arr, y_arr, 1)
    xs = np.array([x_arr.min(), x_arr.max()])
    return xs, m * xs + b

def stars(p):
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def fmt_p(p):

    if not np.isfinite(p):
        return "= n/a"
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"

def within_subject_stats(df, x, y):
    rhos = []
    for _, grp in df.groupby("subject"):
        r, p, n = spearman_rho(grp, x, y)
        if np.isfinite(r):
            rhos.append(r)
    if len(rhos) < 3:
        return np.nan, np.nan, 0
    rhos = np.array(rhos)
    _, p_wilcox = stats.wilcoxon(rhos, alternative="two-sided")
    return float(np.median(rhos)), p_wilcox, len(rhos)

def panel_label(ax, letter):

    ax.annotate(letter.lower(), xy=(0, 1), xycoords="axes fraction",
                xytext=(-40, 12), textcoords="offset points",
                fontsize=16*K, fontweight="bold", ha="right", va="bottom")

rng = np.random.default_rng(42)
SCATTER_N = 3000

NROW = len(CONNECTIVITY)
fig = plt.figure(figsize=(14, 4 * NROW))
gs  = gridspec.GridSpec(NROW, 3, figure=fig, hspace=0.52, wspace=0.38)
panel_letters = iter("ABCDEFGHIJKLMNO")

for row, conn in enumerate(CONNECTIVITY):
    for col, wf in enumerate(WAVEFORM):
        ax = fig.add_subplot(gs[row, col])
        letter = next(panel_letters)

        c = clean(post, wf["col"], conn["col"])
        idx = (rng.choice(len(c), size=min(SCATTER_N, len(c)), replace=False)
               if len(c) > SCATTER_N else np.arange(len(c)))
        ax.scatter(c[wf["col"]].values[idx], c[conn["col"]].values[idx],
                   s=12, alpha=0.30, color=BLUE, linewidths=0, zorder=2)

        if len(c) >= 5:
            xs, ys = ols_line(c[wf["col"]].values, c[conn["col"]].values)
            ax.plot(xs, ys, color=RED, lw=2.0, zorder=4)

        rho_pool, p_pool, n_pool = spearman_rho(post, wf["col"], conn["col"])
        med_rho, p_wilcox, n_subj = within_subject_stats(post, wf["col"], conn["col"])

        txt = (
            f"ρ = {rho_pool:.3f} (pooled)\n"
            f"med ρ = {med_rho:.3f} (within-patient)\n"
            f"Wilcoxon p {fmt_p(p_wilcox)} {stars(p_wilcox)}"
        )
        ax.text(0.03, 0.97, txt,
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5*K,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="0.65", alpha=0.93, lw=0.7))

        ax.tick_params(axis="both", labelsize=11*K)

        if conn["col"] == "pdc_asymmetry":
            ax.axhline(0, color="0.75", lw=0.7, ls="--", zorder=1)

        if row == NROW - 1:
            ax.set_xlabel(wf["label"], fontsize=12*K)
        if col == 0:
            ax.set_ylabel(conn["label"], fontsize=12*K)
        if row == 0:
            ax.set_title(wf["label"].replace("\n", " "),
                         fontsize=12*K, fontweight="bold", pad=8)
        panel_label(ax, letter)

for row, conn in enumerate(CONNECTIVITY):
    ax_last = fig.axes[row * 3 + 2]
    ax_last.annotate(conn["short"], xy=(1.05, 0.5), xycoords="axes fraction",
                     fontsize=14*K, fontweight="bold", rotation=270,
                     va="center", ha="left")

fig.savefig(FIG_DIR / "figS6.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "figS6.svg",           bbox_inches="tight")
plt.close(fig)
print(f"Figure S6 saved to {FIG_DIR}")
