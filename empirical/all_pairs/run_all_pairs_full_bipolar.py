

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
