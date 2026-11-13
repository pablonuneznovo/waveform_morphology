# Reproducing the empirical figures

Gulshan Kumar's human sEEG and macaque ECoG analysis code for Figures 5–7 and Supplementary Figures S4–S10.

## Contents

- `figure_5.py` — sEEG decomposition and waveform features.
- `figure_6.py` — sEEG mixed-model results.
- `figure_7.py` — macaque ECoG mixed-model results.
- `seeg/analysis/figures/figure_S4.py` through `figure_S10.py` — one script per supplementary figure.
- `seeg/seeg_waveform/`, `seeg/analysis/`, `ecog/`, and `all_pairs/` — analysis pipeline and run configurations.
- `figures_assets/` — image assets used in the figure schematics.

Raw recordings, clinical channel exclusions, contact-localisation outputs, atlas labels, and generated tables are not included. The pipeline starts from their exported files. Figure 5 also expects `figures_assets/graphical_abstract.png`, which will be added separately.

## Setup

Use Python 3.11. From this folder:

```bash
python -m pip install -r requirements.txt
```

Set `WAVEFORM_ALPHA` to the data root containing `seeg/` and `ecog/`. Paths in both YAML files are resolved from this root. On PowerShell:

```powershell
$env:WAVEFORM_ALPHA = 'D:\path\to\data'
```

For the large irreversibility intermediates, `WAVEFORM_ALPHA_WORK` may point to separate scratch storage.

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

The sEEG all-pairs run is the long step. The macaque pipeline uses the awake eyes-closed sessions.

## Figures

```bash
python figure_5.py
python figure_6.py
python figure_7.py
```

Each main script writes a 600 dpi PNG and an SVG named `fig5`–`fig7` to `figures/`. Supplementary scripts write `figS4`–`figS10` to the same folder. They read the analysis outputs from the data root. Figure 5 also reads the selected raw recording to recompute the decomposition.

To build the supplementary figures:

```bash
python seeg/analysis/figures/figure_S4.py
python seeg/analysis/figures/figure_S5.py
python seeg/analysis/figures/figure_S6.py
python seeg/analysis/figures/figure_S7.py
python seeg/analysis/figures/figure_S8.py
python seeg/analysis/figures/figure_S9.py
python seeg/analysis/figures/figure_S10.py
```
