"""
merge_electrode_tables.py - 25 records from 23 patients, bipolar montage

Joins the three measures that are computed on their own onto the electrode
table, so that one row per channel carries all five outcomes the models use.

  mean_coh          build_coherence_electrode.py      coh_electrode_bipolar.csv
  mean_io_adaptive  compute_insideout_adaptive.py     electrode_level_irrev_adaptiveT.csv
  uni_irrev         compute_univariate_irreversibility.py  univariate_irreversibility.csv

The remaining outcomes, mean_abs_imcoh and pdc_asymmetry, are already in the
table from build_bipolar_electrode_summary.py.

Every table is keyed on the channel within a recording, file + entry_index +
subject + channel, which is the grouping each of the four scripts aggregates to.

Run after build_bipolar_electrode_summary.py and the compute_* scripts, and
before run_bipolar_lme.py. 

Output: outputs/electrode_level_summary_bipolar.csv, in place
"""
import os
import sys
from pathlib import Path

import pandas as pd

# Data lives outside the repository: WAVEFORM_ALPHA points at the folder
# holding seeg/ and ecog/. Without it the paths resolve against the working
# directory, so run from that folder instead.
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()
WORK = Path(os.environ.get("WAVEFORM_ALPHA_WORK", ROOT / "work")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
SUMMARY = OUT_DIR / "electrode_level_summary_bipolar.csv"

KEY = ["file", "entry_index", "subject", "channel"]


def find(name):
    """The irreversibility batches are written to whichever scratch folder the
    compute scripts used, so look there as well as in outputs/."""
    for folder in (OUT_DIR, ROOT / "seeg" / "irrev_intermediates", WORK):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


SOURCES = [
    ("coh_electrode_bipolar.csv", "mean_coh"),
    ("electrode_level_irrev_adaptiveT.csv", "mean_io_adaptive"),
    ("univariate_irreversibility.csv", "uni_irrev"),
]

# --- load ---
if not SUMMARY.exists():
    raise SystemExit(f"run build_bipolar_electrode_summary.py first: {SUMMARY} is needed")

elec = pd.read_csv(SUMMARY, low_memory=False)
print(f"electrode table: {len(elec):,} rows")

missing = [name for name, _ in SOURCES if find(name) is None]
if missing:
    raise SystemExit("still to be computed: " + ", ".join(missing))

# --- merge ---
for name, column in SOURCES:
    path = find(name)
    part = pd.read_csv(path, low_memory=False)
    part = part[KEY + [column]].drop_duplicates(subset=KEY)
    elec = elec.drop(columns=[column], errors="ignore").merge(part, on=KEY, how="left")
    print(f"  {column:<17} {part[column].notna().sum():>7,} values from {path.name}")

# --- report and save ---
OUTCOMES = ["mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive", "uni_irrev"]
WAVEFORM = ["mean_if", "asc2desc", "peak2trough"]
complete = elec.dropna(subset=OUTCOMES + WAVEFORM)
print(f"\nrows with all five outcomes and all three waveform features: {len(complete):,}"
      f"  ({complete['subject'].nunique()} records)")

elec.to_csv(SUMMARY, index=False)
print(f"saved: {SUMMARY}")
