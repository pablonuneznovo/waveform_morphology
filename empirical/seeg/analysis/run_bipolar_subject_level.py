

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT_DIR = ROOT / "seeg" / "analysis" / "outputs"

elec = pd.read_csv(OUT_DIR / "electrode_level_summary_bipolar.csv", low_memory=False)

KEY_COLS = [
    "mean_if", "asc2desc", "peak2trough",
    "mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive", "uni_irrev",
    "region_family", "stim_frequency_hz", "is_stimulated_channel", "subject",
]
df = elec.dropna(subset=KEY_COLS).copy()

df["region"]    = df["region_family"].replace({"amygdala": "other"})
df["stim_freq"] = df["stim_frequency_hz"].astype(str)
df["is_stim"]   = (df["is_stimulated_channel"].astype(str)
                     .str.lower().eq("true").astype(int))

REPEAT_SESSIONS = {
    "HEJ_Subject004": "HEJ_Subject001",
    "HEJ_Subject015": "HEJ_Subject014",
}
df["record"]  = df["subject"]
df["subject"] = df["subject"].replace(REPEAT_SESSIONS)
print(f"records: {df['record'].nunique()}  ->  patients: {df['subject'].nunique()}")

WAVEFORM = ["mean_if", "asc2desc", "peak2trough"]
OUTCOMES = ["mean_coh", "mean_abs_imcoh", "pdc_asymmetry", "mean_io_adaptive", "uni_irrev"]

for col in WAVEFORM + OUTCOMES:
    mu, sd = df[col].mean(), df[col].std()
    df[col + "_z"] = (df[col] - mu) / sd

subjects = sorted(df["subject"].unique())
print(f"N electrodes: {len(df):,}  |  N subjects: {len(subjects)}")

OUTCOME_LABELS = {"mean_coh": "Coherence", "mean_abs_imcoh": "|ImCoh|",
                  "pdc_asymmetry": "PDC asymmetry", "mean_io_adaptive": "Bivariate irreversibility",
                  "uni_irrev": "Univariate irreversibility"}
FEATURE_LABELS = {"mean_if_z": "Instantaneous frequency",
                  "asc2desc_z": "Ascent/descent ratio",
                  "peak2trough_z": "Peak/trough ratio"}
WAVE_Z = list(FEATURE_LABELS.keys())

beta_rows = []
for outcome in OUTCOMES:
    for subj in subjects:
        sd = df[df["subject"] == subj]

        rhs = ["mean_if_z", "asc2desc_z", "peak2trough_z"]
        if sd["region"].nunique() > 1:
            rhs.append("C(region, Treatment('frontal'))"
                       if (sd["region"] == "frontal").any()
                       else "C(region)")
        if sd["stim_freq"].nunique() > 1:
            rhs.append("C(stim_freq, Treatment('1.0'))"
                       if (sd["stim_freq"] == "1.0").any()
                       else "C(stim_freq)")
        if sd["is_stim"].nunique() > 1:
            rhs.append("is_stim")

        formula = f"{outcome}_z ~ " + " + ".join(rhs)
        rec = {"outcome": outcome, "subject": subj, "n_elec": len(sd)}
        try:
            fit = smf.ols(formula, data=sd).fit()
            for feat in WAVE_Z:
                rec[feat] = float(fit.params.get(feat, np.nan))
        except Exception as exc:
            for feat in WAVE_Z:
                rec[feat] = np.nan
            rec["error"] = str(exc)
        beta_rows.append(rec)

betas = pd.DataFrame(beta_rows)
betas_long = betas.melt(id_vars=["outcome", "subject", "n_elec"],
                        value_vars=WAVE_Z, var_name="feature", value_name="beta")
betas_long.to_csv(OUT_DIR / "bipolar_subject_level_betas.csv", index=False)

records = []
for outcome in OUTCOMES:
    for feat in WAVE_Z:
        vals = betas.loc[betas["outcome"] == outcome, feat].dropna().to_numpy()
        n = len(vals)
        mean_b, med_b = float(np.mean(vals)), float(np.median(vals))
        sem = float(stats.sem(vals))
        tcrit = stats.t.ppf(0.975, n - 1)
        ci_lo, ci_hi = mean_b - tcrit * sem, mean_b + tcrit * sem
        t_stat, t_p = stats.ttest_1samp(vals, 0.0)
        try:
            _, w_p = stats.wilcoxon(vals, alternative="two-sided")
        except ValueError:
            w_p = np.nan
        frac_pos = float(np.mean(vals > 0))
        records.append({
            "outcome": outcome, "outcome_label": OUTCOME_LABELS[outcome],
            "feature": feat, "feature_label": FEATURE_LABELS[feat],
            "n_subj": n, "mean_beta": mean_b, "median_beta": med_b,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "t_p": float(t_p), "wilcoxon_p": float(w_p),
            "frac_positive": frac_pos,
            "sign_consistency": float(max(frac_pos, 1 - frac_pos)),
        })

res = pd.DataFrame(records)
res["p_fdr"]   = multipletests(res["wilcoxon_p"].values, method="fdr_bh")[1]
res["t_p_fdr"] = multipletests(res["t_p"].values,        method="fdr_bh")[1]
res.to_csv(OUT_DIR / "bipolar_subject_level_results.csv", index=False)

def stars(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else
            "*" if p < 0.05 else "ns")

print("\nTwo-stage patient-level results (n = %d patients)" % len(subjects) + "")
print("-" * 108)
print(f"{'Outcome':<14} {'Feature':<26} {'mean β':>8} {'95% CI':>18} "
      f"{'sign':>6} {'Wilcox-p':>9} {'FDR':>8} {'':<4} {'(t-FDR)':>9}")
print("-" * 108)
for _, r in res.iterrows():
    sc = f"{int(r.frac_positive*r.n_subj)}/{r.n_subj}"
    print(f"{r.outcome_label:<14} {r.feature_label:<26} {r.mean_beta:>+8.3f} "
          f"[{r.ci_lo:>+6.3f},{r.ci_hi:>+6.3f}] {sc:>6} "
          f"{r.wilcoxon_p:>9.4f} {r.p_fdr:>8.4f} {stars(r.p_fdr):<4} {r.t_p_fdr:>9.4f}")
print("-" * 108)
print(f"FDR survivors (Wilcoxon): {(res.p_fdr < 0.05).sum()} / {len(res)}")
print(f"\nSaved: bipolar_subject_level_results.csv  +  bipolar_subject_level_betas.csv")
