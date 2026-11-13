# Reproducing the paper figures

This folder contains the code and graphical abstracts for Figures 1–4 of the paper.

## Contents

- `figure_1.py` — Figure 1: waveform morphology.
- `figure_2.py` — Figure 2: interregional measures and irreversibility.
- `figure_3.py` — Figure 3: network irreversibility.
- `figure_4.py` — Figure 4: phase-dependent perturbation responses.
- `abstracts/Figure 1.png` through `abstracts/Figure 4.png` — graphical abstracts used in the figures.
- `requirements.txt` — Python dependencies.

Each figure script contains its own simulation and plotting code. No participant data or external model source is required.

## Python setup and execution

Python 3.11 was used to run the scripts. From this folder:

```bash
python -m pip install -r requirements.txt
python figure_1.py
python figure_2.py
python figure_3.py
python figure_4.py
```

The scripts write 600 dpi PNG and SVG files to `figures/`, named `fig1` through `fig4`.
