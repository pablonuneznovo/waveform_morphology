# Reproducing the simulation figures

The main folder contains the code and graphical abstracts for Figures 1-4 of the paper.

## Contents

- `figure_1.py` - Figure 1: waveform morphology.
- `figure_2.py` - Figure 2: interregional measures and irreversibility.
- `figure_3.py` - Figure 3: network irreversibility.
- `figure_4.py` - Figure 4: phase-dependent perturbation responses.
- `abstracts/Figure 1.png` through `abstracts/Figure 4.png` - graphical abstracts used in the figures.
- `requirements.txt` - Python dependencies.

Each figure script contains its own simulation and plotting code. No participant data or external model source is required.

## Python setup and execution

Python 3.11 was used to run the scripts. From this folder:

```bash
python -m pip install -r requirements.txt
python figure_1.py
python figure_2.py
python figure_3.py
python figure_4.py
python supplementary_figure_1.py
python supplementary_figure_2.py
python supplementary_figure_3.py
```

The scripts write 600 dpi PNG and SVG files to `figures/`, named `fig1` through `fig4`.

# Reproducing the empirical figures

The folder `empirical` contains human sEEG and macaque ECoG analysis code for Figures 5–7 and Supplementary Figures S4–S10. These scripts were authored by Gulshan Kumar.

## Contents

- `figure_5.py` - sEEG decomposition and waveform features.
- `figure_6.py` - sEEG mixed-model results.
- `figure_7.py` - macaque ECoG mixed-model results.
- `seeg/analysis/figures/figure_S4.py` through `figure_S10.py` - one script per supplementary figure.
- `seeg/seeg_waveform/`, `seeg/analysis/`, `ecog/`, and `all_pairs/` - analysis pipeline and run configurations.
- `figures_assets/` — image assets used in the figure schematics.

Raw recordings, clinical channel exclusions, contact-localisation outputs, atlas labels, and generated tables are not included. The pipeline starts from their exported files.

## Setup

Use Python 3.11. From this folder:

```bash
python -m pip install -r requirements.txt
```

Set `WAVEFORM_ALPHA` to the data root containing `seeg/` and `ecog/`. Paths in both YAML files are resolved from this root. On PowerShell:

```powershell
$env:WAVEFORM_ALPHA = 'D:\path\to\data'
```

## Analysis order

```bash
python all_pairs/run_all_pairs_full_bipolar.py
python seeg/analysis/build_bipolar_electrode_summary.py
python seeg/analysis/build_coherence_electrode.py
python seeg/analysis/compute_univariate_irreversibility.py
python seeg/analysis/compute_insideout_adaptive.py
python seeg/analysis/compute_if_profile_shape.py
python seeg/analysis/merge_electrode_tables.py
python seeg/analysis/run_bipolar_lme.py
python seeg/analysis/run_bipolar_subject_level.py
python seeg/analysis/build_pre_connectivity.py
python seeg/analysis/compute_stim_delta_link.py
python ecog/scripts/ecog_montage.py
python ecog/analysis/ecog_run_all.py
python ecog/analysis/ecog_primary_analysis.py
```

## Figures

```bash
python figure_5.py
python figure_6.py
python figure_7.py
python seeg/analysis/figures/figure_S4.py
python seeg/analysis/figures/figure_S5.py
python seeg/analysis/figures/figure_S6.py
python seeg/analysis/figures/figure_S7.py
python seeg/analysis/figures/figure_S8.py
python seeg/analysis/figures/figure_S9.py
python seeg/analysis/figures/figure_S10.py
```

