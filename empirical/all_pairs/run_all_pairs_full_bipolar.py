"""
run_all_pairs_full_bipolar.py - the full bipolar all-pairs connectivity run.

This is the first step of the analysis: it reads the raw recordings and writes
the pair-level tables that every later script works from. Everything it does is
controlled by the config beside this file, which is the one used for the
manuscript (gPDC model order 30, downsampling to 128 Hz, 20,000 permutations,
the two excluded recordings).

Run it from the data root, or set WAVEFORM_ALPHA to that folder first.
"""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "seeg"))

from seeg_waveform.all_pairs_waveform_connectivity import run_all_pairs

CONFIG = Path(__file__).with_name(
    "all_pairs_config.yaml")

if __name__ == "__main__":
    paths = run_all_pairs(str(CONFIG), montage="bipolar")
    print("\nFULL BIPOLAR RUN COMPLETE")
    for name, p in paths.items():
        print(f"  {name}: {p}")
