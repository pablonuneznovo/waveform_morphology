

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

TABLE_DIR = ROOT / "seeg" / "pair_tables" / "tables" / "bipolar"
OUT_DIR   = ROOT / "seeg" / "analysis" / "outputs"

CHUNK = 500_000
KEY   = ["file", "entry_index", "subject", "stim_frequency_hz"]
GRP   = KEY + ["channel"]
COLS  = KEY + ["epoch", "channel_a", "channel_b",
               "abs_imcoh", "coh", "gpdc_a_to_b", "gpdc_b_to_a", "tra_unsigned", "pdc_valid"]

imcoh_p, coh_p, gpdc_p, tra_p = [], [], [], []
n = tot = 0
print("Aggregating PRE connectivity (chunked)...")
for ch in pd.read_csv(TABLE_DIR / "all_pairs_connectivity.csv", usecols=COLS,
                      low_memory=False, chunksize=CHUNK):
    ch = ch[ch["epoch"] == "pre"].copy()
    if ch.empty:
        continue
    tot += len(ch); n += 1
    ch["pdc_valid"] = ch["pdc_valid"].astype(str).str.lower() == "true"

    for col, store in (("abs_imcoh", imcoh_p), ("coh", coh_p), ("tra_unsigned", tra_p)):
        base = ch[KEY + ["channel_a", "channel_b", col]]
        a = base.rename(columns={"channel_a": "channel", "channel_b": "_p"})
        b = base.rename(columns={"channel_b": "channel", "channel_a": "_p"})
        st = pd.concat([a, b], ignore_index=True).drop(columns="_p")
        st = st[st[col].notna()]; st["_n"] = 1
        store.append(st.groupby(GRP, as_index=False)
                       .agg(s=(col, "sum"), c=("_n", "sum")))

    pdc = ch[ch["pdc_valid"] & ch["gpdc_a_to_b"].notna() & ch["gpdc_b_to_a"].notna()]
    if not pdc.empty:
        pa = pdc[KEY + ["channel_a", "gpdc_a_to_b", "gpdc_b_to_a"]].rename(
            columns={"channel_a": "channel", "gpdc_a_to_b": "out", "gpdc_b_to_a": "in_"})
        pb = pdc[KEY + ["channel_b", "gpdc_b_to_a", "gpdc_a_to_b"]].rename(
            columns={"channel_b": "channel", "gpdc_b_to_a": "out", "gpdc_a_to_b": "in_"})
        ps = pd.concat([pa, pb], ignore_index=True); ps["_n"] = 1
        gpdc_p.append(ps.groupby(GRP, as_index=False)
                        .agg(out=("out", "sum"), in_=("in_", "sum"), c=("_n", "sum")))
    if n % 20 == 0:
        print(f"  ...chunk {n}, {tot:,} pre rows")

print(f"  finished {n} chunks, {tot:,} pre pair rows")

def finalize(parts, valcols):
    g = pd.concat(parts, ignore_index=True).groupby(GRP, as_index=False).sum()
    return g

im = finalize(imcoh_p, None); im["mean_abs_imcoh"] = im["s"] / im["c"]
co = finalize(coh_p, None);   co["mean_coh"]        = co["s"] / co["c"]
tr = finalize(tra_p, None);   tr["mean_tra"]        = tr["s"] / tr["c"]
gp = finalize(gpdc_p, None)
gp["pdc_asymmetry"] = gp["out"] / gp["c"] - gp["in_"] / gp["c"]

out = (im[GRP + ["mean_abs_imcoh"]]
       .merge(co[GRP + ["mean_coh"]], on=GRP, how="outer")
       .merge(tr[GRP + ["mean_tra"]], on=GRP, how="outer")
       .merge(gp[GRP + ["pdc_asymmetry"]], on=GRP, how="outer"))
out.to_csv(OUT_DIR / "electrode_level_connectivity_pre.csv", index=False)
print(f"\nSaved electrode_level_connectivity_pre.csv  ({len(out):,} electrodes)")
print(out[["mean_abs_imcoh", "mean_coh", "mean_tra", "pdc_asymmetry"]].describe().round(4).to_string())
