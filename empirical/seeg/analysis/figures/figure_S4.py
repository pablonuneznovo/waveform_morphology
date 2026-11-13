import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# STYLE: the ggsci "npg" palette and type scale.
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

NN = apply_style(11.27)
K  = NN["tick"] / 10.0

# Data lives outside the repository: WAVEFORM_ALPHA points at the folder
# holding seeg/ and ecog/. Without it the paths resolve against the working
# directory, so run from that folder instead.
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
FIG_DIR = Path(__file__).resolve().parents[3] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HILITE  = CYAN   # univariate - same colour it carries in Figs 6-7
NEUTRAL = "#9AA6AE"    # the four between-channel measures

N_BOOT = 500
RNG    = np.random.default_rng(42)

# --- load + common subset ---
uni  = pd.read_csv(OUT_DIR / "univariate_irreversibility.csv")
uni  = uni[uni.status == "ok"]
elec = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)
for df in (uni, elec):
    df["entry_index"] = df["entry_index"].astype(int)
KEY = ["file", "entry_index", "subject", "channel"]
m = uni.merge(elec, on=KEY, how="inner", suffixes=("", "_e"))

COMMON = ["mean_if", "asc2desc", "peak2trough", "mean_abs_imcoh",
          "pdc_asymmetry", "mean_io_adaptive", "uni_irrev", "mean_coh"]
m = m.dropna(subset=COMMON).reset_index(drop=True)
print(f"electrodes: {len(m):,}  subjects: {m.subject.nunique()}")

REPEAT_SESSIONS = {"HEJ_Subject004": "HEJ_Subject001",
                   "HEJ_Subject015": "HEJ_Subject014"}
m["subject"] = m["subject"].replace(REPEAT_SESSIONS)
print(f"  -> resampling {m.subject.nunique()} patients")

# Panel layout:
#   - one colour across the bars in panel c: UTI's bar is 0.347 against 0.050
#     for the next largest, so the separation reads from bar length alone and
#     a colour highlight would be redundant.
MEASURES = [
    ("uni_irrev",      "UTI",     True),
    ("pdc_asymmetry",  "PDC",     False),
    ("mean_io_adaptive",    "BTI", False),
    ("mean_coh",       "Coh",     False),
    ("mean_abs_imcoh", "|ImCoh|", False),
]

FEAT = "asc2desc"

def pooled_rho(frame: pd.DataFrame, col: str) -> float:
    r, _ = stats.spearmanr(frame[col], frame[FEAT])
    return float(r)

def bootstrap_ci(col: str, subject_groups: dict, subjects: np.ndarray) -> tuple[float, float]:
    """95% CI for the pooled rho, resampling patients with replacement."""
    vals = []
    for _ in range(N_BOOT):
        pick = RNG.choice(subjects, size=subjects.size, replace=True)
        x = np.concatenate([subject_groups[s][col] for s in pick])
        y = np.concatenate([subject_groups[s][FEAT] for s in pick])
        r, _ = stats.spearmanr(x, y)
        if np.isfinite(r):
            vals.append(abs(r))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

subjects = m["subject"].unique()
cols_needed = [c for c, _, _ in MEASURES] + [FEAT]
subject_groups = {
    s: {c: g[c].to_numpy(float) for c in cols_needed}
    for s, g in m.groupby("subject")
}

# --- figure ---
fig = plt.figure(figsize=(13, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.9],
                       hspace=0.34, wspace=0.30)

def panel_letter(ax, letter: str) -> None:
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-42, 12),
                textcoords="offset points", fontsize=16*K, fontweight="bold",
                ha="right", va="bottom")

def scatter_panel(ax, ycol, ylabel, color, letter):
    x = m[FEAT].to_numpy(float)
    y = m[ycol].to_numpy(float)
    idx = RNG.choice(len(x), size=min(4000, len(x)), replace=False)
    ax.scatter(x[idx], y[idx], s=10, alpha=0.25, color=color, linewidths=0, zorder=2)
    good = np.isfinite(x) & np.isfinite(y)
    b, a = np.polyfit(x[good], y[good], 1)
    xs = np.array([np.percentile(x, 1), np.percentile(x, 99)])
    ax.plot(xs, a + b * xs, color="0.15", lw=2.2, zorder=4)
    ax.text(0.03, 0.97, f"ρ = {pooled_rho(m, ycol):+.3f}", transform=ax.transAxes,
            va="top", fontsize=12*K, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel("Waveform asymmetry (ascent/descent ratio)", fontsize=12*K)
    ax.set_ylabel(ylabel, fontsize=12*K)
    ax.tick_params(labelsize=11*K)
    ax.set_xlim(np.percentile(x, 0.5), np.percentile(x, 99.5))
    ax.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.5))
    panel_letter(ax, letter)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
scatter_panel(ax_a, "uni_irrev",
              "Univariate irreversibility\n(derivative skewness)", HILITE, "a")
scatter_panel(ax_b, "mean_io_adaptive", "Bivariate irreversibility", NEUTRAL, "b")

# --- panel c ---
ax_c = fig.add_subplot(gs[1, :])
labels, rhos, los, his, colors = [], [], [], [], []
for col, lab, is_hi in MEASURES:
    r = abs(pooled_rho(m, col))
    lo, hi = bootstrap_ci(col, subject_groups, subjects)
    labels.append(lab.replace("\n", " "))
    rhos.append(r)
    los.append(lo)
    his.append(hi)
    colors.append(NEUTRAL)   # no highlight: one colour across the bars
    print(f"  {lab.replace(chr(10), ' '):<44} |rho| {r:.3f}  CI [{lo:.3f}, {hi:.3f}]")

rhos = np.array(rhos)
err = np.vstack([np.maximum(rhos - np.array(los), 0),
                 np.maximum(np.array(his) - rhos, 0)])
y = np.arange(len(labels))[::-1]

ax_c.barh(y, rhos, color=colors, alpha=0.9, edgecolor="0.3", lw=0.6,
          height=0.62, zorder=2)
ax_c.errorbar(rhos, y, xerr=err, fmt="none", ecolor="0.25", elinewidth=1.4,
              capsize=4, zorder=4)
ax_c.set_yticks(y)
ax_c.set_yticklabels(labels, fontsize=11*K)
ax_c.set_xlabel("|Spearman ρ|  with waveform asymmetry (ascent/descent ratio)",
                fontsize=12*K)
ax_c.tick_params(labelsize=11*K)
ax_c.set_xlim(0, float(np.max(his)) * 1.10)
panel_letter(ax_c, "c")

fig.savefig(FIG_DIR / "figS4.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "figS4.svg", bbox_inches="tight")
plt.close(fig)
print(f"Figure S4 saved to {FIG_DIR}")
