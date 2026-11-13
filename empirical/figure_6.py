import os
from pathlib import Path
import numpy as np
import pandas as pd
FS = 1.15

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

NAVY = "#1B3A57"
CYAN = "#3FB8DE"
RED = "#C44536"
GREY = "#8A97A0"

CODE = Path(__file__).resolve().parent

BRAIN_PNG = CODE / "figures_assets" / "brain_seeg.png"

BOX_W = 0.204
_GAP = (0.990 - 0.008 - 4 * BOX_W) / 3
BOX_X = [0.008 + i * (BOX_W + _GAP) for i in range(4)]
BOX_Y0, BOX_Y1 = 0.06, 0.94

def _box(ax, x0, title, fs):
    ax.add_patch(
        FancyBboxPatch(
            (x0, BOX_Y0), BOX_W, BOX_Y1 - BOX_Y0,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            linewidth=1.4, edgecolor=NAVY, facecolor="white",
            transform=ax.transAxes, zorder=2, clip_on=False,
        )
    )
    ax.text(x0 + BOX_W / 2, BOX_Y1 - 0.055, title, transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5 * fs, fontweight="bold",
            color=NAVY, zorder=4)

def _arrow(ax, x_from, x_to, y=0.5):
    ax.add_patch(
        FancyArrowPatch(
            (x_from, y), (x_to, y),
            transform=ax.transAxes, zorder=3,
            arrowstyle="-|>", mutation_scale=15,
            linewidth=2.6, color=NAVY, shrinkA=0, shrinkB=0,
        )
    )

def draw_schematic(ax, fs: float = 1.0) -> None:

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    titles = [
        "sEEG recording",
        "Bipolar montage, EMD",
        "Cycle waveform",
        "Connectivity, irreversibility",
    ]
    for x0, t in zip(BOX_X, titles):
        _box(ax, x0, t, fs)
    for i in range(3):
        _arrow(ax, BOX_X[i] + BOX_W + 0.004, BOX_X[i + 1] - 0.004)

    x0 = BOX_X[0]
    if BRAIN_PNG.exists():
        img = mpimg.imread(str(BRAIN_PNG))
        iax = ax.inset_axes([x0 + 0.0414, 0.2262, 0.1150, 0.4145],
                            transform=ax.transAxes, zorder=5)
        iax.imshow(img, aspect="auto")
        iax.axis("off")
    ax.text(x0 + BOX_W / 2, BOX_Y0 + 0.10,
            "23 patients\n5,602 stimulation runs",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.6 * fs, color="0.25", zorder=6)

    x0 = BOX_X[1]
    cx = x0 + BOX_W / 2
    shaft_y = BOX_Y1 - 0.20
    ax.plot([cx - 0.055, cx + 0.055], [shaft_y, shaft_y],
            transform=ax.transAxes, color="0.55", lw=3.2, solid_capstyle="round",
            zorder=4)
    for dx, lab in ((-0.028, "1"), (0.028, "2")):

        ax.text(cx + dx, shaft_y, lab, transform=ax.transAxes, ha="center",
                va="center", fontsize=6.4 * fs, color="white",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.34", facecolor=NAVY,
                          edgecolor="white", linewidth=1.0))
    ax.text(cx, shaft_y - 0.10, "contact 1 − contact 2", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4 * fs, color="0.25", zorder=5)

    t = np.linspace(0, 4 * np.pi, 400)
    trace = np.sin(t) + 0.28 * np.sin(2 * t + 0.6)
    tx = cx - 0.075 + 0.15 * (t - t.min()) / (t.max() - t.min())
    ty = BOX_Y0 + 0.20 + 0.052 * trace / np.abs(trace).max()
    ax.plot(tx, ty, transform=ax.transAxes, color=CYAN, lw=1.7, zorder=5)
    ax.text(cx, BOX_Y0 + 0.075, "IMF4  ≈ 10 Hz", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4 * fs, color="0.25", zorder=5)

    x0 = BOX_X[2]
    cx = x0 + BOX_W / 2
    ph = np.linspace(0, 1, 300)

    cyc = np.sin(2 * np.pi * ph)
    cyc = np.sign(cyc) * np.abs(cyc) ** 0.78
    kx = cx - 0.078 + 0.156 * ph
    ky = BOX_Y1 - 0.34 + 0.075 * cyc
    ax.plot(kx, ky, transform=ax.transAxes, color=CYAN, lw=2.0, zorder=5)
    ax.plot([kx[0], kx[-1]], [BOX_Y1 - 0.34, BOX_Y1 - 0.34], transform=ax.transAxes,
            color="0.75", lw=0.8, ls="--", zorder=4)
    i_pk = int(np.argmax(cyc))
    i_tr = int(np.argmin(cyc))
    i_dz = i_pk + int(np.argmin(np.abs(cyc[i_pk:i_tr])))
    for idx, col in ((i_pk, RED), (i_tr, "#E8A838"), (i_dz, "#5BAD6F")):
        ax.plot([kx[idx]], [ky[idx]], transform=ax.transAxes, marker="o",
                markersize=4.6, color=col, markeredgecolor="white",
                markeredgewidth=0.7, zorder=6)
    ax.text(cx, BOX_Y0 + 0.185, "peak · trough · descending zero",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0 * fs, color="0.25", zorder=5)
    ax.text(cx, BOX_Y0 + 0.085, "IF        A2D        P2T", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.2 * fs, color=NAVY,
            fontweight="bold", zorder=5)

    x0 = BOX_X[3]
    cx = x0 + BOX_W / 2
    ya = BOX_Y1 - 0.22
    node_bbox = dict(boxstyle="circle,pad=0.38", facecolor="white",
                     edgecolor=NAVY, linewidth=1.4)
    ax.text(cx - 0.052, ya, "A", transform=ax.transAxes, ha="center", va="center",
            fontsize=8.0 * fs, color=NAVY, fontweight="bold", zorder=6,
            bbox=node_bbox)
    ax.text(cx + 0.052, ya, "B", transform=ax.transAxes, ha="center", va="center",
            fontsize=8.0 * fs, color=NAVY, fontweight="bold", zorder=6,
            bbox=dict(node_bbox))
    ax.add_patch(FancyArrowPatch((cx - 0.022, ya), (cx + 0.022, ya),
                                 transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.8, color=NAVY,
                                 zorder=5))
    ax.text(cx, ya - 0.115, "Coh   |ImCoh|   PDC   BTI", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.8 * fs, color=NAVY,
            fontweight="bold", zorder=5)
    ax.text(cx - 0.052, BOX_Y0 + 0.215, "A", transform=ax.transAxes, ha="center",
            va="center", fontsize=8.0 * fs, color=RED, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.38", facecolor="white",
                      edgecolor=RED, linewidth=1.4))
    ax.text(cx + 0.040, BOX_Y0 + 0.215, "UTI", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.8 * fs, color=RED,
            fontweight="bold", zorder=5)
    ax.text(cx, BOX_Y0 + 0.085, "mixed-effects model", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4 * fs, color="0.25", zorder=5)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10 * FS,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
FIG_DIR = CODE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

elec  = pd.read_csv(OUT_DIR / "bipolar_lme_results.csv")
subj  = pd.read_csv(OUT_DIR / "bipolar_subject_level_results.csv")
betas = pd.read_csv(OUT_DIR / "bipolar_subject_level_betas.csv")

OUTCOME_ORDER = ["mean_coh", "mean_abs_imcoh", "pdc_asymmetry",
                 "mean_io_adaptive", "uni_irrev"]

OUTCOME_LABELS = {
    "mean_coh"      : "Coh",
    "mean_abs_imcoh": "|ImCoh|",
    "pdc_asymmetry" : "PDC",
    "mean_io_adaptive"   : "BTI",
    "uni_irrev"     : "UTI",
}
OUTCOME_COLORS = {
    "mean_coh"      : "#8E6BB5",
    "mean_abs_imcoh": "#4C8BB5",
    "pdc_asymmetry" : "#E8A838",
    "mean_io_adaptive"   : "#5BAD6F",
    "uni_irrev"     : "#C44536",
}
FEATURES = ["mean_if_z", "asc2desc_z", "peak2trough_z"]
FEATURE_TITLES = {
    "mean_if_z"    : "Instantaneous frequency",
    "asc2desc_z"   : "Ascent/descent ratio",
    "peak2trough_z": "Peak/trough ratio",
}

Y_POS   = {c: float(i) for i, c in enumerate(OUTCOME_ORDER[:4])}
Y_POS[OUTCOME_ORDER[4]] = 4.7
Y_ORDER = [Y_POS[c] for c in OUTCOME_ORDER]

XLIM     = 0.25
STAR_PAD = 0.012
JITTER   = 0.20

def sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return ("***" if p < 0.001 else "**" if p < 0.01 else
            "*"   if p < 0.05  else "ns")

def panel_letter(ax, letter: str) -> None:
    ax.annotate(letter.lower(), xy=(0, 1), xycoords="axes fraction",
                xytext=(-46, 14), textcoords="offset points",
                fontsize=15*FS, fontweight="bold", va="top", ha="left")

fig = plt.figure(figsize=(13.2, 11.0))
gs = fig.add_gridspec(3, 3, height_ratios=[0.60, 1.0, 1.0],
                      hspace=0.16, wspace=0.08,
                      left=0.085, right=0.985, top=0.975, bottom=0.085)

ax_schematic = fig.add_subplot(gs[0, :])
draw_schematic(ax_schematic, FS)

axes = np.empty((2, 3), dtype=object)
for r in range(2):
    for c in range(3):
        share = {} if (r == 0 and c == 0) else {"sharex": axes[0][0],
                                                "sharey": axes[0][0]}
        axes[r][c] = fig.add_subplot(gs[r + 1, c], **share)

rng = np.random.default_rng(11)
n_clipped = 0

for col, feat in enumerate(FEATURES):

    ax = axes[0][col]
    for outcome in OUTCOME_ORDER:
        row   = elec[(elec["outcome"] == outcome) & (elec["feature"] == feat)].iloc[0]
        color = OUTCOME_COLORS[outcome]
        yi    = Y_POS[outcome]
        xerr  = np.array([[row.beta - row.ci_lo], [row.ci_hi - row.beta]])
        fc    = color if row.p_fdr < 0.05 else "white"
        ax.errorbar(row.beta, yi, xerr=xerr, fmt="o", color=color,
                    markerfacecolor=fc, markeredgecolor=color, markersize=11,
                    elinewidth=2.0, capsize=5, capthick=2.0, zorder=3)
        s = sig_stars(row.p_fdr)
        if s and s != "ns":
            ax.text(row.beta + np.sign(row.beta or 1) * STAR_PAD * 1.6, yi - 0.34,
                    s, va="center", ha="center", fontsize=12*FS,
                    color="black", fontweight="bold")
    ax.set_title(FEATURE_TITLES[feat], fontsize=13*FS, fontweight="bold", pad=10)

    ax = axes[1][col]
    for outcome in OUTCOME_ORDER:
        srow  = subj[(subj["outcome"] == outcome) & (subj["feature"] == feat)].iloc[0]
        color = OUTCOME_COLORS[outcome]
        yi    = Y_POS[outcome]
        vals  = betas[(betas["outcome"] == outcome) &
                      (betas["feature"] == feat)]["beta"].to_numpy(float)
        vals  = vals[np.isfinite(vals)]

        ax.plot([max(srow.ci_lo, -XLIM), min(srow.ci_hi, XLIM)], [yi, yi],
                color=color, lw=3.5, alpha=0.30, solid_capstyle="round", zorder=2)

        jit    = rng.uniform(-JITTER, JITTER, size=vals.size)
        inside = np.abs(vals) <= XLIM
        ax.scatter(vals[inside], yi + jit[inside], s=26, facecolor=color,
                   alpha=0.55, edgecolor="white", linewidths=0.5, zorder=3)
        for v, j in zip(vals[~inside], jit[~inside]):
            n_clipped += 1
            ax.scatter(np.sign(v) * XLIM, yi + j, s=95,
                       marker=">" if v > 0 else "<", facecolor=color,
                       alpha=1.0, edgecolor="0.15", linewidths=0.9,
                       zorder=6, clip_on=False)

        ax.scatter(srow.median_beta, yi, s=110, marker="D", facecolor=color,
                   edgecolor="0.15", linewidths=1.2, zorder=5)
        s = sig_stars(srow.p_fdr)
        if s and s != "ns":
            ax.text(srow.median_beta, yi - 0.42, s, va="center", ha="center",
                    fontsize=12*FS, color="black", fontweight="bold")
    ax.set_xlabel("Standardised β", fontsize=12*FS)

for r in range(2):
    for c in range(3):
        ax = axes[r][c]
        ax.axvline(0, color="0.5", lw=1, ls="--", zorder=1)

        ax.axhline(4.0, color="0.85", lw=1.0, ls="-", zorder=0)
        ax.set_xlim(-XLIM, XLIM)
        ax.set_ylim(-0.7, 5.5)
        ax.tick_params(axis="x", labelsize=11*FS)

        panel_letter(ax, "BCDEFG"[r * 3 + c])
        if r == 0:
            plt.setp(ax.get_xticklabels(), visible=False)

        if c > 0:
            plt.setp(ax.get_yticklabels(), visible=False)

panel_letter(ax_schematic, "A")

axes[0][0].set_yticks(Y_ORDER)
axes[0][0].set_yticklabels([OUTCOME_LABELS[c] for c in OUTCOME_ORDER], fontsize=11.5*FS)
for r in range(2):
    for lbl, outcome in zip(axes[r][0].get_yticklabels(), OUTCOME_ORDER):
        lbl.set_color(OUTCOME_COLORS[outcome])

axes[0][0].set_ylabel("Electrode level", fontsize=11.5*FS, labelpad=10)
axes[1][0].set_ylabel("Patient level", fontsize=11.5*FS, labelpad=10)

legend_handles = [
    Line2D([0], [0], marker="o", color="0.45", markerfacecolor="0.45",
           markersize=8, lw=0, label="FDR < 0.05"),
    Line2D([0], [0], marker="o", color="0.45", markerfacecolor="white",
           markeredgecolor="0.45", markersize=8, lw=0, label="ns"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45",
           markeredgecolor="white", markersize=6, lw=0, label="individual patient"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="0.45",
           markeredgecolor="0.15", markersize=9, lw=0, label="median across patients"),
    Line2D([0], [0], marker="<", color="none", markerfacecolor="0.45",
           markeredgecolor="0.15", markersize=11, lw=0, label="beyond axis"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=10*FS, frameon=False, bbox_to_anchor=(0.5, 0.008))

stem = "fig6"
fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
plt.close(fig)
print(f"{stem} saved.  ({n_clipped} patient slopes at the axis boundary)")
