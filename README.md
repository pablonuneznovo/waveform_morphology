# Reproducing the paper figures

This folder contains the code and graphical abstracts for Figures 2–5 of the paper.

## Contents

- `figure_2.py` — Figure 2: waveform morphology.
- `figure_3.py` — Figure 3: interregional measures and irreversibility.
- `figure_4.py` — Figure 4: network irreversibility.
- `figure_5.py` — Figure 5: phase-dependent perturbation responses.
- `abstracts/Figure 2.png` through `abstracts/Figure 5.png` — graphical abstracts used in the figures.
- `requirements.txt` — Python dependencies.

Each figure script contains its own simulation and plotting code. No participant data or external model source is required.

## Python setup and execution

Python 3.11 was used to run the scripts. From this folder:

```bash
python -m pip install -r requirements.txt
python figure_2.py
python figure_3.py
python figure_4.py
python figure_5.py
```

The scripts write 600 dpi PNG and SVG files to `figures/`, named `fig2` through `fig5`. Figures 3 and 4 reproduce the simulation analyses from scratch and may take substantially longer than the other figures.
