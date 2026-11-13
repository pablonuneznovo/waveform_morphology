import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import emd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import matplotlib.image as mpimg

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "seeg"))
from seeg_waveform.config import load_config
from seeg_waveform.io import iter_mat_files
from seeg_waveform.preprocess import preprocess_segment, window_mask
from seeg_waveform.primary_bipolar_mne import make_bipolar_run, analysis_windows
from seeg_waveform.quinn import analyze_segment, asc2desc, peak2trough

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

NN = apply_style(12.27)

CFG_PATH = CODE / "seeg" / "analysis" / "configs" / "bipolar_npz_windows.yaml"
OUT_DIR  = ROOT / "seeg" / "analysis" / "outputs"
FIG_DIR  = CODE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ABSTRACT_PNG = CODE / "figures_assets" / "graphical_abstract.png"

FILE_NAME   = "HEJ_Subject005_F=1Hz_I=3mA_stims"
ENTRY_INDEX = 62
CHANNEL     = "Ep7-Ep8"

IMF4_IDX = 3

BLUE = NAVY
GREEN = TEAL
GOLD = CYAN

def panel_label(ax, letter, x=-0.10, y=1.05):
    ax.text(x, y, letter.lower(), transform=ax.transAxes,
            fontsize=NN["panel"], fontweight="bold", va="top")

cfg = load_config(CFG_PATH)
paths = [p for p in iter_mat_files(cfg["project"]["data_dir"]) if p.stem == FILE_NAME]
assert len(paths) == 1, f"expected 1 recording, found {len(paths)}"

run = make_bipolar_run(paths[0], cfg, entry_index=ENTRY_INDEX)
assert run.status == "ok", f"bipolar run failed: {run.status} {run.message}"
rec = run.recording
ch_idx = rec.channel_names.index(CHANNEL)
print(f"{FILE_NAME} entry {ENTRY_INDEX}  channel {CHANNEL} (idx {ch_idx})")
print(f"  fs={rec.fs} Hz  stim={rec.stim_frequency_hz} Hz  nodes={rec.signal.shape[1]}")

_, post_win = analysis_windows(rec, cfg)
mask = window_mask(rec.time, *post_win)
t_post = rec.time[mask]
x_post, _, _ = preprocess_segment(rec.signal[mask, ch_idx], t_post, rec.fs, cfg["preprocessing"])

res = analyze_segment(x_post, rec.fs, cfg["emd"])
assert res.status == "ok", f"analyze_segment failed: {res.status} {res.message}"
imf, ip, ifreq, iamp = res.imf, res.ip, res.ifreq, res.iamp
pa_if = res.phase_aligned_if
print(f"  status={res.status}  n_cycles={res.n_cycles}  IMFs={imf.shape[1]}  PA-IF={pa_if.shape}")

fs = float(rec.fs)

ecfg = cfg["emd"]
cycles = emd.cycles.Cycles(ip[:, IMF4_IDX])
cycles.compute_cycle_metric("max_amp", iamp[:, IMF4_IDX], np.max)
cycles.compute_cycle_metric("mean_if", ifreq[:, IMF4_IDX], np.mean)
cycles.compute_cycle_metric("max_if",  ifreq[:, IMF4_IDX], np.max)
cycles.compute_cycle_metric("if_range", ifreq[:, IMF4_IDX],
                            lambda v: float(np.nanmax(v) - np.nanmin(v)))
cycles.compute_cycle_metric("asc2desc", imf[:, IMF4_IDX], asc2desc)
cycles.compute_cycle_metric("peak2trough", imf[:, IMF4_IDX], peak2trough)
amp_thresh = np.nanpercentile(iamp[:, IMF4_IDX], float(ecfg.get("amp_percentile", 10.0)))
cycles.pick_cycle_subset([
    "is_good==1",
    f"max_amp>{amp_thresh}",
    f"max_if<{float(ecfg.get('max_if_hz', 20.0))}",
    f"if_range<{float(ecfg.get('max_if_range_hz', 20.0))}",
])

subset_vect = np.asarray(cycles.subset_vect)
orig_cycle_inds = np.where(subset_vect > -1)[0]
good_cycles = []
for ii in orig_cycle_inds:
    inds = cycles.get_inds_of_cycle(int(ii), mode="cycle")
    if inds is not None and len(inds) > 10:
        good_cycles.append(np.asarray(inds))
print(f"  subset cycles: {len(orig_cycle_inds)}  with usable sample indices: {len(good_cycles)}")
assert good_cycles, "no retained cycles resolved to sample indices"

fig = plt.figure(figsize=(14, 13.6))
gs  = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1.55, 1.6, 1.0],
                        hspace=0.36, wspace=0.92)
ax_abs = fig.add_subplot(gs[0, 0:3])
ax_B   = fig.add_subplot(gs[0, 3:4])
ax_C   = fig.add_subplot(gs[0, 4:6])
ax_A   = fig.add_subplot(gs[1, :])
ax_D1  = fig.add_subplot(gs[2, 0:2])
ax_D2  = fig.add_subplot(gs[2, 2:4])
ax_D3  = fig.add_subplot(gs[2, 4:6])

ax_abs.imshow(mpimg.imread(str(ABSTRACT_PNG)))
ax_abs.axis("off")
panel_label(ax_abs, "A", x=-0.02)

SHOW_SEC = 6.0
n_show = int(SHOW_SEC * fs)
t_s = t_post[:n_show]
x_s = x_post[:n_show]

labels  = ["Signal", "IMF 1", "IMF 2", "IMF 3", "IMF 4 (α)", "IMF 5", "IMF 6"]
signals = [x_s] + [imf[:n_show, i] for i in range(min(6, imf.shape[1]))]

TRACE_H, GAP = 1.0, 0.35
norm_sigs = []
for s in signals:
    rng = np.ptp(s)
    norm_sigs.append(s / rng if rng > 0 else s)
offsets = [0.0]
for i in range(1, len(norm_sigs)):
    offsets.append(offsets[-1] - (TRACE_H + GAP))

for i, (sig, off, lab) in enumerate(zip(norm_sigs, offsets, labels)):
    color = BLUE if i == 4 else ("0.15" if i == 0 else GREY)
    lw    = 1.6  if i == 4 else (1.1 if i == 0 else 0.9)
    ax_A.plot(t_s, sig + off, color=color, lw=lw, zorder=3 if i == 4 else 2)
    ax_A.text(t_s[0] - 0.18, off, lab, ha="right", va="center", fontsize=NN["annot"],
              color=color if i in (0, 4) else GREY,
              fontweight="bold" if i in (0, 4) else "normal")

bar_y = offsets[-1] - 0.55
ax_A.plot([t_s[-1] - 1.0, t_s[-1]], [bar_y, bar_y], color="k", lw=2,
          solid_capstyle="butt", clip_on=False)
ax_A.text(t_s[-1] - 0.5, bar_y + 0.12, "1 s", ha="center", va="bottom",
          fontsize=NN["small"])

ax_A.axhspan(offsets[4] - 0.55, offsets[4] + 0.55, color=BLUE, alpha=0.07, zorder=0)
ax_A.set_xlim(t_s[0], t_s[-1])
ax_A.set_xlabel("Time (s)", fontsize=NN["label"])
ax_A.set_yticks([])
ax_A.spines["left"].set_visible(False)
panel_label(ax_A, "D")

imf4 = imf[:, IMF4_IDX]
a2d_vals = np.array([asc2desc(imf4[c]) for c in good_cycles])
target = np.nanmedian(a2d_vals)
k = int(np.nanargmin(np.abs(a2d_vals - target)))

inds  = good_cycles[k]
seg   = imf4[inds]
T     = len(seg)
x_cyc = np.linspace(0, 1, T)
seg_n = (seg - seg.min()) / (seg.max() - seg.min()) * 2 - 1

i_peak = emd.cycles.cf_peak_sample(seg, interp=True)
i_tro  = emd.cycles.cf_trough_sample(seg, interp=True)
i_desc = emd.cycles.cf_descending_zero_sample(seg, interp=True)
a2d = asc2desc(seg)
p2t = peak2trough(seg)
print(f"  representative cycle: asc2desc={a2d:.3f}  peak2trough={p2t:.3f}  ({T} samples)")

for kk in range(max(0, k - 3), min(len(good_cycles), k + 4)):
    if kk == k:
        continue
    s = imf4[good_cycles[kk]]
    s_n = (s - s.min()) / (s.max() - s.min() + 1e-14) * 2 - 1
    ax_B.plot(np.linspace(0, 1, len(s_n)), s_n, color=GREY, lw=0.6, alpha=0.4)

ax_B.plot(x_cyc, seg_n, color=BLUE, lw=2, zorder=3)

def _interp_at(frac):
    return np.interp(frac, x_cyc, seg_n)

for pos, col, lab in ((i_peak, RED, "Peak"), (i_tro, GOLD, "Trough"),
                      (i_desc, GREEN, "Desc. zero")):
    if pos is None:
        continue
    f = pos / T
    ax_B.scatter([f], [_interp_at(f)], s=60, color=col, zorder=5, label=lab)

ax_B.axhline(0, color="0.75", lw=0.8, ls="--", zorder=1)
ax_B.set_xlim(-0.02, 1.05)
ax_B.set_ylim(-1.35, 1.35)
ax_B.set_xlabel("Cycle phase (0 → 1)", fontsize=NN["label"])
ax_B.set_ylabel("Normalised amplitude", fontsize=NN["label"])
ax_B.set_title("Representative\ncycle", fontsize=NN["title"], pad=6)
ax_B.legend(fontsize=NN["legend"], frameon=False, loc="lower left",
            handlelength=0.8, handletextpad=0.4)
panel_label(ax_B, "B", x=-0.46, y=1.12)

n_bins, n_cyc = pa_if.shape
phase = np.linspace(0, 360, n_bins)
mean_profile = np.nanmean(pa_if, axis=1)
sd_profile = np.nanstd(pa_if, axis=1)

sem_profile = sd_profile / np.sqrt(np.sum(np.isfinite(pa_if), axis=1))
print(f"  PA-IF mean profile: range {mean_profile.max()-mean_profile.min():.3f} Hz  "
      f"mean SEM {sem_profile.mean():.3f} Hz  "
      f"range/SEM {(mean_profile.max()-mean_profile.min())/sem_profile.mean():.1f}")

ax_C.fill_between(phase, mean_profile - sem_profile, mean_profile + sem_profile,
                  color=BLUE, alpha=0.30, lw=0, zorder=2)
ax_C.axhline(float(np.nanmean(mean_profile)), color="0.35", lw=1.2, ls="--",
             zorder=3)
ax_C.plot(phase, mean_profile, color=BLUE, lw=2.2, zorder=4)

ax_C.legend(handles=[
    Line2D([0], [0], color=BLUE, lw=2.2, label="mean ± SEM"),
    Line2D([0], [0], color="0.35", lw=1.2, ls="--", label="sinusoid (flat)"),
], fontsize=NN["legend"], frameon=False, loc="upper right",
   handlelength=1.1, handletextpad=0.5)
ax_C.set_xlabel("Phase (°)", fontsize=NN["label"])
ax_C.set_ylabel("Instantaneous\nfrequency (Hz)", fontsize=NN["label"])
ax_C.set_xticks([0, 90, 180, 270, 360])
ax_C.set_xlim(0, 360)

_c = float(np.nanmean(mean_profile)); _s = float(sem_profile.mean())
ax_C.set_ylim(_c - 4 * _s, _c + 4 * _s)
ax_C.set_title("Phase-aligned IF profile", fontsize=NN["title"], pad=6)
panel_label(ax_C, "C", x=-0.06)

pop = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)

pop = pop.dropna(subset=["mean_if", "asc2desc", "peak2trough", "mean_coh",
                         "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive"])
FEAT_CONFIGS = [
    ("mean_if",     ax_D1, "Instantaneous\nfrequency (Hz)", BLUE,  (6, 14)),
    ("asc2desc",    ax_D2, "Ascent/descent\nratio",          RED,   (0.46, 0.56)),
    ("peak2trough", ax_D3, "Peak/trough\nratio",            GREEN, (0.46, 0.52)),
]
for feat, ax, xlabel, color, xlim in FEAT_CONFIGS:
    vals = pop[feat].dropna().values
    ax.hist(vals, bins=60, color=color, alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(np.median(vals), color="k", lw=1.5, ls="--", zorder=5)
    ax.text(0.97, 0.95, f"median\n{np.median(vals):.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=NN["annot"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9))
    ax.set_xlabel(xlabel, fontsize=NN["label"])
    ax.set_ylabel("Electrode count" if ax is ax_D1 else "", fontsize=NN["label"])
    ax.set_xlim(xlim)
    ax.tick_params(axis="both", labelsize=NN["tick"])
panel_label(ax_D1, "E")

fig.savefig(FIG_DIR / "fig5.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig5.svg", bbox_inches="tight")
plt.close(fig)
print(f"Figure 5 saved to {FIG_DIR}")
