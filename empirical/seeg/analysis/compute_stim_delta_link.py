"""
compute_stim_delta_link.py - 25 records from 23 patients, bipolar montage
Delta_waveform  = post - pre   (mean_if, asc2desc, peak2trough)   [features table]
Delta_conn      = post - pre   (|ImCoh|, PDC asym, TII, coherence)
                  post: electrode_level_summary_bipolar.csv
                  pre : electrode_level_connectivity_pre.csv

Tests, per (Delta_waveform x Delta_conn) cell:
  - pooled Spearman rho
  - within-subject median rho + Wilcoxon across the 25 records
  - separately for stimulated vs non-stimulated electrodes
Outputs: outputs/stim_delta_link_results.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

TABLE_DIR = ROOT / "seeg" / "pair_tables" / "tables" / "bipolar"
OUT_DIR   = ROOT / "seeg" / "analysis" / "outputs"

KEY = ["file", "entry_index", "subject", "stim_frequency_hz", "channel"]

# --- delta waveform ---
wf = pd.read_csv(TABLE_DIR / "channel_epoch_imf4_features.csv", low_memory=False)
WK = KEY + ["is_stimulated_channel"]
pre_w  = wf[wf.epoch == "pre" ][WK + ["mean_if", "asc2desc", "peak2trough"]]
post_w = wf[wf.epoch == "post"][WK + ["mean_if", "asc2desc", "peak2trough"]]
dwave = pre_w.merge(post_w, on=WK, suffixes=("_pre", "_post"))
for f in ["mean_if", "asc2desc", "peak2trough"]:
    dwave[f"d_{f}"] = dwave[f"{f}_post"] - dwave[f"{f}_pre"]
dwave["is_stim"] = dwave["is_stimulated_channel"].astype(str).str.lower().eq("true")

# --- delta connectivity ---
CC = ["mean_abs_imcoh", "pdc_asymmetry", "mean_tra", "mean_coh"]
post_c = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)
if "mean_coh" not in post_c.columns:
    coh = pd.read_csv(OUT_DIR / "coh_electrode_bipolar.csv")
    for d in (post_c, coh): d["entry_index"] = d["entry_index"].astype(int)
    post_c = post_c.merge(coh[KEY + ["mean_coh"]], on=KEY, how="left")
pre_c = pd.read_csv(OUT_DIR / "electrode_level_connectivity_pre.csv", low_memory=False)
for d in (post_c, pre_c, dwave): d["entry_index"] = d["entry_index"].astype(int)
dconn = pre_c[KEY + CC].merge(post_c[KEY + CC], on=KEY, suffixes=("_pre", "_post"))
for c in CC:
    dconn[f"d_{c}"] = dconn[f"{c}_post"] - dconn[f"{c}_pre"]

df = dwave.merge(dconn, on=KEY, how="inner")
DW = ["d_mean_if", "d_asc2desc", "d_peak2trough"]
DC = [f"d_{c}" for c in CC]
df = df.dropna(subset=DW + DC)
print(f"electrodes with pre&post: {len(df):,}  subjects: {df.subject.nunique()}")
print(f"  stimulated: {int(df.is_stim.sum()):,}  non-stim: {int((~df.is_stim).sum()):,}")

WLAB = {"d_mean_if": "Δ IF", "d_asc2desc": "Δ asc2desc", "d_peak2trough": "Δ peak2trough"}
CLAB = {"d_mean_abs_imcoh": "Δ|ImCoh|", "d_pdc_asymmetry": "Δ PDC",
        "d_mean_tra": "Δ TII", "d_mean_coh": "Δ Coherence"}

def stars(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

def within(sub, x, y):
    rr = []
    for _, g in sub.groupby("subject"):
        if len(g) >= 20:
            r, _ = stats.spearmanr(g[x], g[y])
            if np.isfinite(r): rr.append(r)
    if len(rr) < 5: return np.nan, np.nan, 0
    _, p = stats.wilcoxon(rr)
    return float(np.median(rr)), p, len(rr)

rows = []
for grp_name, sub in [("all", df), ("stim", df[df.is_stim]), ("nonstim", df[~df.is_stim])]:
    for x in DW:
        for y in DC:
            r, p = stats.spearmanr(sub[x], sub[y])
            mr, wp, ns = within(sub, x, y)
            rows.append({"group": grp_name, "dwave": WLAB[x], "dconn": CLAB[y],
                         "n": len(sub), "rho_pooled": r, "p_pooled": p,
                         "rho_ws_median": mr, "p_ws": wp, "n_subj": ns})
res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "stim_delta_link_results.csv", index=False)

print("\n=== Δwaveform -> Δconnectivity  (ALL electrodes) ===")
a = res[res.group == "all"]
print(f"{'Δwaveform':<14}{'Δconn':<12}{'rho':>8}{'ws-med':>9}{'ws-p':>10}")
for _, r in a.iterrows():
    print(f"{r.dwave:<14}{r.dconn:<12}{r.rho_pooled:>+8.3f}{r.rho_ws_median:>+9.3f}"
          f"{r.p_ws:>10.1e} {stars(r.p_ws)}")

print("\n=== STIMULATED electrodes only (largest waveform change) ===")
s = res[res.group == "stim"]
for _, r in s.iterrows():
    print(f"{r.dwave:<14}{r.dconn:<12}{r.rho_pooled:>+8.3f}  p={r.p_pooled:.3g} {stars(r.p_pooled)}  (n={r.n})")
print(f"\nSaved: stim_delta_link_results.csv")
