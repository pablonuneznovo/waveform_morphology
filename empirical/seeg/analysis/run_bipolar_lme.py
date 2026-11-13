"""
run_bipolar_lme.py
==================
Linear Mixed Effects analysis - bipolar montage, 25 records from 23 patients.

Model (one per connectivity outcome):
  connectivity_z ~ mean_if_z + asc2desc_z + peak2trough_z
                 + C(region, Treatment('frontal'))
                 + C(stim_freq, Treatment('1.0'))
                 + is_stim
                 + (1 | patient)

Outputs
-------
outputs/bipolar_lme_results.csv
outputs/bipolar_lme_partial_resid.csv
outputs/bipolar_lme_model_summary.txt
"""
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Data lives outside the repository: WAVEFORM_ALPHA points at the folder
# holding seeg/ and ecog/. Without it the paths resolve against the working
# directory, so run from that folder instead.
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Load ---
elec = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)

KEY_COLS = [
    "mean_if", "asc2desc", "peak2trough",
    "mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive", "uni_irrev",
    "region_family", "stim_frequency_hz", "is_stimulated_channel", "subject",
]
df = elec.dropna(subset=KEY_COLS).copy()
print(f"N electrodes: {len(df):,}  |  N subjects: {df['subject'].nunique()}")

# --- Patient grouping ---
# SEEG_Sociodemo.csv holds 25 sEEG records from 23 distinct patients: its Title
# field marks HEJ_Subject004 as the second sEEG session of HEJ_Subject001, and
# HEJ_Subject015 as the second session of HEJ_Subject014. Grouping on the record
# would enter those two people twice as independent patients, so the grouping
# variable is the patient.
REPEAT_SESSIONS = {
    "HEJ_Subject004": "HEJ_Subject001",
    "HEJ_Subject015": "HEJ_Subject014",
}
df["record"]  = df["subject"]
df["subject"] = df["subject"].replace(REPEAT_SESSIONS)
print(f"records: {df['record'].nunique()}  ->  patients: {df['subject'].nunique()}")


# --- 2. Prepare variables ---
df["region"]    = df["region_family"].replace({"amygdala": "other"})
df["stim_freq"] = df["stim_frequency_hz"].astype(str)
df["is_stim"]   = (df["is_stimulated_channel"].astype(str)
                     .str.lower().eq("true").astype(int))

WAVEFORM = ["mean_if", "asc2desc", "peak2trough"]
OUTCOMES  = ["mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive", "uni_irrev"]

for col in WAVEFORM + OUTCOMES:
    mu, sd = df[col].mean(), df[col].std()
    df[col + "_z"] = (df[col] - mu) / sd

print(f"\nRegion counts:\n{df['region'].value_counts().to_string()}")
print(f"\nStim freq counts:\n{df['stim_freq'].value_counts().to_string()}")
print(f"is_stim counts:\n{df['is_stim'].value_counts().to_string()}")

# --- 3. Fit LME models ---
OUTCOME_LABELS = {
    "mean_coh"      : "Coherence",
    "mean_abs_imcoh": "|ImCoh|",
    "pdc_asymmetry" : "PDC asymmetry",
    "mean_io_adaptive"   : "Bivariate irreversibility",
    "uni_irrev"     : "Univariate irreversibility",
}
FEATURE_LABELS = {
    "mean_if_z"    : "Instantaneous frequency",
    "asc2desc_z"   : "Ascent/descent ratio",
    "peak2trough_z": "Peak/trough ratio",
}

records       = []
model_fits    = {}
summary_lines = []

for outcome_col, outcome_label in OUTCOME_LABELS.items():
    print(f"\nFitting: {outcome_label} ...")
    formula = (
        f"{outcome_col}_z ~ mean_if_z + asc2desc_z + peak2trough_z"
        f" + C(region, Treatment('frontal'))"
        f" + C(stim_freq, Treatment('1.0'))"
        f" + is_stim"
    )
    model = smf.mixedlm(formula, df, groups=df["subject"])
    fit   = model.fit(reml=True, method="powell", maxiter=3000, disp=False)
    if fit.cov_re.values[0][0] < 1e-6:
        fit = model.fit(reml=False, method="powell", maxiter=3000, disp=False)

    model_fits[outcome_col] = fit
    summary_lines.append(f"\n{'='*70}\nOutcome: {outcome_label}\n{'='*70}")
    summary_lines.append(str(fit.summary()))

    ci = fit.conf_int()
    for feat_col, feat_label in FEATURE_LABELS.items():
        records.append({
            "outcome"      : outcome_col,
            "outcome_label": outcome_label,
            "feature"      : feat_col,
            "feature_label": feat_label,
            "beta"         : fit.params[feat_col],
            "se"           : fit.bse[feat_col],
            "ci_lo"        : ci.loc[feat_col, 0],
            "ci_hi"        : ci.loc[feat_col, 1],
            "p"            : fit.pvalues[feat_col],
        })

# --- 4. FDR across all 15 tests (5 outcomes x 3 waveform features) ---
# NOTE: family grew 9 -> 12 (coherence added) -> 15 (univariate irreversibility
# added), so that every measure the theory paper analyses - including the
# single-channel one - is estimated in the SAME mixed-effects framework with the
# same covariates. This makes the univariate/bivariate dissociation directly
# comparable in standardised beta rather than across different statistics.
res = pd.DataFrame(records)
_, p_fdr, _, _ = multipletests(res["p"].values, method="fdr_bh")
res["p_fdr"] = p_fdr
res.to_csv(OUT_DIR / "bipolar_lme_results.csv", index=False)

with open(OUT_DIR / "bipolar_lme_model_summary.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(summary_lines))

# --- 5. Print summary table ---
print("\nBipolar LME fixed effects -- waveform features")
print("-" * 75)
print(f"{'Outcome':<18} {'Feature':<28} {'beta':>7} {'95% CI':>18} {'p':>8} {'FDR-p':>8}")
print("-" * 75)
for _, row in res.iterrows():
    sig = ("***" if row.p_fdr < 0.001 else
           "**"  if row.p_fdr < 0.01  else
           "*"   if row.p_fdr < 0.05  else "")
    print(f"{row.outcome_label:<18} {row.feature_label:<28} "
          f"{row.beta:+7.3f}  [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]  "
          f"{row.p:8.4f}  {row.p_fdr:8.4f} {sig}")

print(f"\nFDR survivors: {(res.p_fdr < 0.05).sum()} / {len(res)}")
print(f"Results saved to {OUT_DIR}")

# --- 6. Partial residuals ---
partial_rows = []
for outcome_col in OUTCOME_LABELS:
    fit   = model_fits[outcome_col]
    resid = fit.resid.values
    for feat_col in FEATURE_LABELS:
        beta_j    = fit.params[feat_col]
        x_vals    = df[feat_col].values
        y_partial = resid + beta_j * x_vals
        partial_rows.append(
            df[["subject"]].assign(
                outcome=outcome_col,
                feature=feat_col,
                x_z=x_vals,
                y_partial=y_partial,
            )
        )

partial_df = pd.concat(partial_rows, ignore_index=True)
partial_df.to_csv(OUT_DIR / "bipolar_lme_partial_resid.csv", index=False)
print("bipolar_lme_partial_resid.csv saved.")
