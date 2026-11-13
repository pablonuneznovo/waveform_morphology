

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()
WORK = Path(os.environ.get("WAVEFORM_ALPHA_WORK", ROOT / "work")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
SUMMARY = OUT_DIR / "electrode_level_summary_bipolar.csv"

KEY = ["file", "entry_index", "subject", "channel"]

def find(name):

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

if not SUMMARY.exists():
    raise SystemExit(f"run build_bipolar_electrode_summary.py first: {SUMMARY} is needed")

elec = pd.read_csv(SUMMARY, low_memory=False)
print(f"electrode table: {len(elec):,} rows")

missing = [name for name, _ in SOURCES if find(name) is None]
if missing:
    raise SystemExit("still to be computed: " + ", ".join(missing))

for name, column in SOURCES:
    path = find(name)
    part = pd.read_csv(path, low_memory=False)
    part = part[KEY + [column]].drop_duplicates(subset=KEY)
    elec = elec.drop(columns=[column], errors="ignore").merge(part, on=KEY, how="left")
    print(f"  {column:<17} {part[column].notna().sum():>7,} values from {path.name}")

OUTCOMES = ["mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive", "uni_irrev"]
WAVEFORM = ["mean_if", "asc2desc", "peak2trough"]
complete = elec.dropna(subset=OUTCOMES + WAVEFORM)
print(f"\nrows with all five outcomes and all three waveform features: {len(complete):,}"
      f"  ({complete['subject'].nunique()} records)")

elec.to_csv(SUMMARY, index=False)
print(f"saved: {SUMMARY}")
