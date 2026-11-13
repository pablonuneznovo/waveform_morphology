"""
build_bipolar_electrode_summary.py
===================================
Build electrode_level_summary_bipolar.csv for the 25-record dataset (23 patients).

Sources (pair_tables/tables/bipolar/)
  channel_epoch_imf4_features.csv  - waveform features per electrode per epoch
  all_pairs_connectivity.csv       - pair-level ImCoh / gPDC / TRA  (~12 GB)

Output
  outputs/electrode_level_summary_bipolar.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Data lives outside the repository: WAVEFORM_ALPHA points at the folder
# holding seeg/ and ecog/. Without it the paths resolve against the working
# directory, so run from that folder instead.
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

TABLE_DIR  = ROOT / "seeg" / "pair_tables" / "tables" / "bipolar"
OUT_DIR    = ROOT / "seeg" / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500_000
KEY_COLS   = ["file", "entry_index", "subject", "stim_frequency_hz"]

# --- 1. Waveform features - post epoch only ---
print("Loading waveform features...")
wf = pd.read_csv(TABLE_DIR / "channel_epoch_imf4_features.csv", low_memory=False)
wf_post = wf[wf["epoch"] == "post"].copy()
print(f"  Post-epoch electrodes: {len(wf_post):,}")

# --- 2. Aggregate connectivity to electrode level - vectorised chunks ---
print("Aggregating connectivity (chunked, vectorised)...")

imcoh_parts  = []
gpdc_parts   = []
tra_parts    = []

conn_path = TABLE_DIR / "all_pairs_connectivity.csv"
n_chunks  = 0
total_rows = 0

CONN_COLS = [
    "file", "entry_index", "subject", "stim_frequency_hz",
    "channel_a", "channel_b",
    "abs_imcoh", "gpdc_a_to_b", "gpdc_b_to_a", "tra_unsigned",
    "pdc_valid", "epoch",
]

for chunk in pd.read_csv(conn_path, usecols=CONN_COLS,
                          low_memory=False, chunksize=CHUNK_SIZE):
    chunk = chunk[chunk["epoch"] == "post"].copy()
    if chunk.empty:
        continue
    total_rows += len(chunk)
    n_chunks   += 1

    chunk["pdc_valid"] = chunk["pdc_valid"].astype(str).str.lower() == "true"

    # --- ImCoh (symmetric: same value for both electrodes in a pair) ---
    base = chunk[KEY_COLS + ["channel_a", "channel_b", "abs_imcoh"]].copy()
    a_side = base.rename(columns={"channel_a": "channel", "channel_b": "_partner"})
    b_side = base.rename(columns={"channel_b": "channel", "channel_a": "_partner"})
    stacked = pd.concat([a_side, b_side], ignore_index=True).drop(columns="_partner")
    stacked = stacked[stacked["abs_imcoh"].notna()]
    stacked["_imcoh_n"] = 1
    imcoh_parts.append(
        stacked.groupby(KEY_COLS + ["channel"], as_index=False)
               .agg(imcoh_sum=("abs_imcoh", "sum"), imcoh_n=("_imcoh_n", "sum"))
    )

    # --- gPDC (directional: channel_a -> channel_b) ---
    pdc = chunk[chunk["pdc_valid"]].copy()
    pdc = pdc[pdc["gpdc_a_to_b"].notna() & pdc["gpdc_b_to_a"].notna()]
    if not pdc.empty:
        # channel_a perspective: out=gpdc_a_to_b, in=gpdc_b_to_a
        pa = pdc[KEY_COLS + ["channel_a", "gpdc_a_to_b", "gpdc_b_to_a"]].copy()
        pa = pa.rename(columns={"channel_a": "channel",
                                 "gpdc_a_to_b": "gpdc_out", "gpdc_b_to_a": "gpdc_in"})
        # channel_b perspective: out=gpdc_b_to_a, in=gpdc_a_to_b
        pb = pdc[KEY_COLS + ["channel_b", "gpdc_b_to_a", "gpdc_a_to_b"]].copy()
        pb = pb.rename(columns={"channel_b": "channel",
                                 "gpdc_b_to_a": "gpdc_out", "gpdc_a_to_b": "gpdc_in"})
        pdc_stacked = pd.concat([pa, pb], ignore_index=True)
        pdc_stacked["_n"] = 1
        gpdc_parts.append(
            pdc_stacked.groupby(KEY_COLS + ["channel"], as_index=False)
                       .agg(gpdc_out_sum=("gpdc_out", "sum"),
                            gpdc_in_sum=("gpdc_in", "sum"),
                            gpdc_n=("_n", "sum"))
        )

    # --- TRA (symmetric) ---
    tra = chunk[KEY_COLS + ["channel_a", "channel_b", "tra_unsigned"]].copy()
    ta = tra.rename(columns={"channel_a": "channel", "channel_b": "_p"})
    tb = tra.rename(columns={"channel_b": "channel", "channel_a": "_p"})
    tra_stacked = pd.concat([ta, tb], ignore_index=True).drop(columns="_p")
    tra_stacked = tra_stacked[tra_stacked["tra_unsigned"].notna()]
    tra_stacked["_n"] = 1
    tra_parts.append(
        tra_stacked.groupby(KEY_COLS + ["channel"], as_index=False)
                   .agg(tra_sum=("tra_unsigned", "sum"), tra_n=("_n", "sum"))
    )

    if n_chunks % 20 == 0:
        print(f"  ...chunk {n_chunks}, {total_rows:,} post rows, "
              f"{len(imcoh_parts)} imcoh parts")

print(f"  Finished {n_chunks} chunks, {total_rows:,} total post-epoch pair rows.")

# --- 3. Combine chunk parts and compute per-electrode means ---
print("Combining and computing means...")
GRP = KEY_COLS + ["channel"]

imcoh_agg = (pd.concat(imcoh_parts, ignore_index=True)
               .groupby(GRP, as_index=False)
               .agg(imcoh_sum=("imcoh_sum", "sum"), imcoh_n=("imcoh_n", "sum")))
imcoh_agg["mean_abs_imcoh"] = imcoh_agg["imcoh_sum"] / imcoh_agg["imcoh_n"]

gpdc_agg = (pd.concat(gpdc_parts, ignore_index=True)
              .groupby(GRP, as_index=False)
              .agg(gpdc_out_sum=("gpdc_out_sum", "sum"),
                   gpdc_in_sum=("gpdc_in_sum", "sum"),
                   n_valid_pdc=("gpdc_n", "sum")))
gpdc_agg["mean_gpdc_out"]  = gpdc_agg["gpdc_out_sum"] / gpdc_agg["n_valid_pdc"]
gpdc_agg["mean_gpdc_in"]   = gpdc_agg["gpdc_in_sum"]  / gpdc_agg["n_valid_pdc"]
gpdc_agg["pdc_asymmetry"]  = gpdc_agg["mean_gpdc_out"] - gpdc_agg["mean_gpdc_in"]

tra_agg = (pd.concat(tra_parts, ignore_index=True)
             .groupby(GRP, as_index=False)
             .agg(tra_sum=("tra_sum", "sum"), tra_n=("tra_n", "sum")))
tra_agg["mean_tra"] = tra_agg["tra_sum"] / tra_agg["tra_n"]

# --- 4. Merge waveform + connectivity ---
print("Merging...")
summary = wf_post.merge(
    imcoh_agg[GRP + ["mean_abs_imcoh"]], on=GRP, how="left"
).merge(
    gpdc_agg[GRP + ["mean_gpdc_out", "mean_gpdc_in", "pdc_asymmetry", "n_valid_pdc"]],
    on=GRP, how="left"
).merge(
    tra_agg[GRP + ["mean_tra"]], on=GRP, how="left"
)

summary["montage"] = "bipolar"
summary["epoch"]   = "post"

col_order = [
    "file", "entry_index", "subject", "stim_frequency_hz", "montage", "epoch",
    "channel", "region_family", "is_stimulated_channel",
    "n_cycles", "mean_if", "asc2desc", "peak2trough",
    "mean_abs_imcoh", "mean_gpdc_out", "mean_gpdc_in",
    "pdc_asymmetry", "n_valid_pdc", "mean_tra",
]
summary = summary[col_order]

out_path = OUT_DIR / "electrode_level_summary_bipolar.csv"
summary.to_csv(out_path, index=False)

# --- 5. Report ---
print(f"\nSaved: {out_path}")
print(f"  Total electrodes      : {len(summary):,}")
print(f"  Subjects              : {summary['subject'].nunique()}")
print(f"  mean_abs_imcoh (n)    : {summary['mean_abs_imcoh'].notna().sum():,}")
print(f"  pdc_asymmetry  (n)    : {summary['pdc_asymmetry'].notna().sum():,}")
print(f"  mean_tra       (n)    : {summary['mean_tra'].notna().sum():,}")
print(f"  mean_if  mean         : {summary['mean_if'].mean():.3f} Hz")
print(f"  mean_abs_imcoh mean   : {summary['mean_abs_imcoh'].mean():.4f}")
print(f"  pdc_asymmetry  mean   : {summary['pdc_asymmetry'].mean():.6f}")
print(f"  mean_tra mean         : {summary['mean_tra'].mean():.4f}")
print(f"  stim_frequency_hz     : {sorted(summary['stim_frequency_hz'].unique())}")
print(f"  region_family         : {sorted(summary['region_family'].dropna().unique())}")
