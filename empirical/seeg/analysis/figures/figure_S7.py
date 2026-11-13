import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

NN = apply_style(16.04)
K  = NN["tick"] / 10.0

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
FIG_DIR = Path(__file__).resolve().parents[3] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

res     = pd.read_csv(OUT_DIR / "bipolar_lme_results.csv")
partial = pd.read_csv(OUT_DIR / "bipolar_lme_partial_resid.csv")

OUTCOME_LABELS = {
    "mean_coh"      : "Coh",
    "mean_abs_imcoh": "|ImCoh|",
    "pdc_asymmetry" : "PDC",
    "mean_io_adaptive"   : "BTI",
    "uni_irrev"     : "UTI",
}
FEATURE_LABELS = {
    "mean_if_z"    : "Instantaneous\nFrequency",
    "asc2desc_z"   : "Ascent/Descent\nRatio",
    "peak2trough_z": "Peak/Trough\nRatio",
}
FEATURE_XLABELS = {
    "mean_if_z"    : "IF (z)",
    "asc2desc_z"   : "A2D ratio (z)",
    "peak2trough_z": "P2T ratio (z)",
}

OUTCOME_COLORS = {
    "mean_coh"        : "#8E6BB5",
    "mean_abs_imcoh"  : "#4C8BB5",
    "pdc_asymmetry"   : "#E8A838",
    "mean_io_adaptive": "#5BAD6F",
    "uni_irrev"       : "#C44536",
}

def pfmt(p: float) -> str:

    return "< 0.0001" if p < 1e-4 else "= %.4f" % p

def sig_stars(p: float) -> str:
    if not np.isfinite(p): return ""
    return ("***" if p < 0.001 else "**" if p < 0.01 else
            "*"   if p < 0.05  else "ns")

NOUT = len(OUTCOME_LABELS)

feat_keys_b  = list(FEATURE_LABELS.keys())
outcome_keys = list(OUTCOME_LABELS.keys())

fig, axes = plt.subplots(3, NOUT, figsize=(4.3 * NOUT, 11))
rng = np.random.default_rng(42)

for col_i, outcome_col in enumerate(outcome_keys):
    color = OUTCOME_COLORS[outcome_col]
    for row_i, feat_col in enumerate(feat_keys_b):
        ax = axes[row_i][col_i]

        sub = partial[(partial["outcome"] == outcome_col) &
                      (partial["feature"] == feat_col)]
        x_vals    = sub["x_z"].values
        y_partial = sub["y_partial"].values

        n   = len(x_vals)
        idx = (rng.choice(n, size=min(n, 600), replace=False)
               if n > 600 else np.arange(n))
        ax.scatter(x_vals[idx], y_partial[idx],
                   s=14, alpha=0.35, color=color, zorder=2, linewidths=0)

        row_stat = res[(res["outcome"] == outcome_col) &
                       (res["feature"] == feat_col)].iloc[0]
        x_line = np.array([x_vals.min(), x_vals.max()])
        ax.plot(x_line, row_stat.beta * x_line, color="0.25", lw=2, zorder=4)

        s = sig_stars(row_stat.p_fdr)
        txt = (f"β = {row_stat.beta:+.3f}\n"
               f"p {pfmt(row_stat.p)}\n"
               f"FDR-p {pfmt(row_stat.p_fdr)}"
               + (f" {s}" if s != "ns" else ""))
        ax.text(0.04, 0.97, txt, transform=ax.transAxes, va="top",
                fontsize=9.5*K,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))

        ax.axhline(0, color="0.75", lw=0.8, ls="--")
        ax.axvline(0, color="0.75", lw=0.8, ls="--")

        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.42 * (hi - lo))

        ax.tick_params(axis="both", labelsize=11*K)

        if row_i == 0:
            ax.set_title(OUTCOME_LABELS[outcome_col], fontsize=11*K,
                         fontweight="bold", color="black", pad=8)
        if col_i == 0:
            ax.set_ylabel("Partial residuals", fontsize=11*K)
        ax.set_xlabel(FEATURE_XLABELS[feat_col], fontsize=11*K)

fig.tight_layout()
fig.savefig(FIG_DIR / "figS7.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "figS7.svg", bbox_inches="tight")
plt.close(fig)
print("Figure S7 saved.")
