

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

elec = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)

KEY_COLS = [
    "mean_if", "asc2desc", "peak2trough",
    "mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive", "uni_irrev",
    "region_family", "stim_frequency_hz", "is_stimulated_channel", "subject",
]
df = elec.dropna(subset=KEY_COLS).copy()
print(f"N electrodes: {len(df):,}  |  N subjects: {df['subject'].nunique()}")

REPEAT_SESSIONS = {
    "HEJ_Subject004": "HEJ_Subject001",
    "HEJ_Subject015": "HEJ_Subject014",
}
df["record"]  = df["subject"]
df["subject"] = df["subject"].replace(REPEAT_SESSIONS)
print(f"records: {df['record'].nunique()}  ->  patients: {df['subject'].nunique()}")

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

res = pd.DataFrame(records)
_, p_fdr, _, _ = multipletests(res["p"].values, method="fdr_bh")
res["p_fdr"] = p_fdr
res.to_csv(OUT_DIR / "bipolar_lme_results.csv", index=False)

with open(OUT_DIR / "bipolar_lme_model_summary.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(summary_lines))

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
