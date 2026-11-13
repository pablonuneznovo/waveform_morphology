"""
build_coherence_electrode.py - 25-record bipolar (23 patients)

Output: outputs/coh_electrode_bipolar.csv  (key + mean_coh, n_coh)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

TABLE_DIR = ROOT / "seeg" / "pair_tables" / "tables" / "bipolar"
OUT_DIR   = ROOT / "seeg" / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500_000
KEY_COLS   = ["file", "entry_index", "subject", "stim_frequency_hz"]
GRP        = KEY_COLS + ["channel"]
CONN_COLS  = KEY_COLS + ["epoch", "channel_a", "channel_b", "coh"]

conn_path = TABLE_DIR / "all_pairs_connectivity.csv"
parts = []
n_chunks = total = 0
print("Aggregating coherence (chunked)...")
for chunk in pd.read_csv(conn_path, usecols=CONN_COLS, low_memory=False, chunksize=CHUNK_SIZE):
    chunk = chunk[chunk["epoch"] == "post"].copy()
    if chunk.empty:
        continue
    total += len(chunk); n_chunks += 1
    base = chunk[KEY_COLS + ["channel_a", "channel_b", "coh"]]
    a = base.rename(columns={"channel_a": "channel", "channel_b": "_p"})
    b = base.rename(columns={"channel_b": "channel", "channel_a": "_p"})
    st = pd.concat([a, b], ignore_index=True).drop(columns="_p")
    st = st[st["coh"].notna()]
    st["_n"] = 1
    parts.append(st.groupby(GRP, as_index=False)
                   .agg(coh_sum=("coh", "sum"), coh_n=("_n", "sum")))
    if n_chunks % 20 == 0:
        print(f"  ...chunk {n_chunks}, {total:,} post rows")

print(f"  finished {n_chunks} chunks, {total:,} post pair rows")
agg = (pd.concat(parts, ignore_index=True)
         .groupby(GRP, as_index=False)
         .agg(coh_sum=("coh_sum", "sum"), coh_n=("coh_n", "sum")))
agg["mean_coh"] = agg["coh_sum"] / agg["coh_n"]
agg = agg.rename(columns={"coh_n": "n_coh"})[GRP + ["mean_coh", "n_coh"]]

out = OUT_DIR / "coh_electrode_bipolar.csv"
agg.to_csv(out, index=False)
print(f"\nSaved {out}")
print(f"  electrodes with coherence: {len(agg):,}")
print(f"  mean_coh: mean {agg['mean_coh'].mean():.4f}  median {agg['mean_coh'].median():.4f}")
