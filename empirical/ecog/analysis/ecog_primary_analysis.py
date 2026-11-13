"""
ecog_primary_analysis.py - the primary ECoG result: AwakeEyesClosed, all monkeys.

Mirrors the sEEG design so the two datasets are directly comparable:

  ELECTRODE LEVEL (top row of the figure)
    outcome_z ~ mean_if_z + asc2desc_z + peak2trough_z + (1 | session)
    Mixed model over all bipolar channels from every session. Session is the random
    effect - the ECoG analog of the sEEG's (1 | subject). Monkey is added as a fixed
    covariate when more than one monkey is present, the analog of the sEEG's region /
    protocol covariates.
    FDR-BH across all 15 tests (5 outcomes x 3 waveform features), exactly as in the sEEG.

  SESSION LEVEL (bottom row) - the two-stage analog of the sEEG subject-level
    The same model is fitted WITHIN each session by OLS, and the resulting slopes are
    tested against zero with a Wilcoxon signed-rank test across sessions.

Outputs: outputs/ecog_primary_lme.csv, ecog_primary_session_betas.csv,
         outputs/fig_ecog_primary.png / .pdf
"""
import warnings; warnings.filterwarnings("ignore")
import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Data lives outside the repository: WAVEFORM_ALPHA points at the folder
# holding seeg/ and ecog/. Without it the paths resolve against the working
# directory, so run from that folder instead.
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

OUT = ROOT / "ecog" / "analysis" / "outputs"
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "pdf.fonttype": 42})

OUTCOMES = {"mean_coh": "Coherence", "mean_abs_imcoh": "|ImCoh|",
            "pdc_asymmetry": "PDC asymmetry",
            "mean_io_adaptive": "Bivariate\nirreversibility",
            "uni_irrev": "Univariate\nirreversibility"}
FEATS = {"mean_if_z": "Instantaneous frequency",
         "asc2desc_z": "Ascent/descent ratio",
         "peak2trough_z": "Peak/trough ratio"}
COLORS = {"Coherence": "#8E6BB5", "|ImCoh|": "#4C8BB5", "PDC asymmetry": "#E8A838",
          "Bivariate\nirreversibility": "#5BAD6F",
          "Univariate\nirreversibility": "#C44536"}
# sEEG alpha reference betas (IF / asc2desc), for the comparison print
SEEG = {"Coherence": (-0.131, -0.023), "|ImCoh|": (-0.098, -0.019),
        "PDC asymmetry": (-0.034, -0.016),
        "Bivariate\nirreversibility": (-0.113, 0.001),
        "Univariate\nirreversibility": (0.017, -0.107)}

files = sorted(OUT.glob("electrodes_*_AwakeEyesClosed.csv"))
E = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
E = E.dropna(subset=["mean_if", "asc2desc", "peak2trough"] + list(OUTCOMES)).copy()
print(f"files {len(files)}  channels {len(E):,}  sessions {E.session.nunique()}  "
      f"monkeys {sorted(E.monkey.unique())}")
print(E.groupby("monkey").agg(sessions=("session", "nunique"),
                              channels=("bip", "size")).to_string())

for c in ["mean_if", "asc2desc", "peak2trough"] + list(OUTCOMES):
    sd = E[c].std()
    E[c + "_z"] = (E[c] - E[c].mean()) / (sd if sd > 0 else 1)

# --- electrode level ---
rows = []
for oc, lab in OUTCOMES.items():
    rhs = "mean_if_z + asc2desc_z + peak2trough_z"
    if E.monkey.nunique() > 1:
        rhs += " + C(monkey)"
    try:
        fit = smf.mixedlm(f"{oc}_z ~ {rhs}", E, groups=E["session"]).fit(
            reml=True, method="powell", maxiter=3000, disp=False)
        ci = fit.conf_int()
        for fk, fl in FEATS.items():
            rows.append({"outcome": oc, "outcome_label": lab, "feature": fk,
                         "feature_label": fl, "beta": fit.params[fk],
                         "ci_lo": ci.loc[fk, 0], "ci_hi": ci.loc[fk, 1],
                         "p": fit.pvalues[fk]})
    except Exception as e:
        print(f"  LME failed for {lab}: {type(e).__name__}")
R = pd.DataFrame(rows)
R["p_fdr"] = multipletests(R["p"], method="fdr_bh")[1]
R.to_csv(OUT / "ecog_primary_lme.csv", index=False)

def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

print("\nELECTRODE LEVEL - AwakeEyesClosed, session as random effect")
print("=" * 96)
print(f"{'Outcome':<28}{'Feature':<26}{'beta':>8}{'95% CI':>20}{'FDR':>10}")
print("-" * 96)
for _, r in R.iterrows():
    print(f"{r.outcome_label.replace(chr(10),' '):<28}{r.feature_label:<26}{r.beta:>+8.3f}"
          f"  [{r.ci_lo:+.3f},{r.ci_hi:+.3f}]{r.p_fdr:>10.4f} {stars(r.p_fdr)}")
print(f"FDR survivors: {(R.p_fdr < .05).sum()} / {len(R)}")

# --- session level (two-stage) ---
brows = []
for oc in OUTCOMES:
    for s, g in E.groupby("session"):
        if len(g) < 12:
            continue
        try:
            fit = smf.ols(f"{oc}_z ~ mean_if_z + asc2desc_z + peak2trough_z", data=g).fit()
            for fk in FEATS:
                brows.append({"outcome": oc, "session": s,
                              "monkey": g.monkey.iloc[0], "feature": fk,
                              "beta": float(fit.params.get(fk, np.nan))})
        except Exception:
            pass
B = pd.DataFrame(brows)
B.to_csv(OUT / "ecog_primary_session_betas.csv", index=False)

srows = []
for (oc, fk), g in B.groupby(["outcome", "feature"]):
    v = g.beta.dropna().to_numpy()
    if len(v) < 5:
        continue
    try:
        p = stats.wilcoxon(v)[1]
    except Exception:
        p = np.nan
    # column names match the sEEG's bipolar_subject_level_results.csv, so one figure
    # script draws both datasets and the two panels cannot drift apart
    mean_b = float(np.mean(v)); sem = float(stats.sem(v))
    tcrit = stats.t.ppf(0.975, len(v) - 1)
    srows.append({"outcome": oc, "outcome_label": OUTCOMES[oc], "feature": fk,
                  "feature_label": FEATS[fk], "median": float(np.median(v)),
                  "median_beta": float(np.median(v)), "mean_beta": mean_b,
                  "ci_lo": mean_b - tcrit * sem, "ci_hi": mean_b + tcrit * sem,
                  "n": len(v), "n_neg": int((v < 0).sum()), "p": p})
S = pd.DataFrame(srows)
S["p_fdr"] = multipletests(S["p"].fillna(1), method="fdr_bh")[1]
S.to_csv(OUT / "ecog_primary_session_results.csv", index=False)

print("\nSESSION LEVEL - model fitted within each session, Wilcoxon across sessions")
print("=" * 96)
print(f"{'Outcome':<28}{'Feature':<26}{'median':>8}{'neg':>8}{'FDR':>10}")
print("-" * 96)
for _, r in S.iterrows():
    print(f"{r.outcome_label.replace(chr(10),' '):<28}{r.feature_label:<26}"
          f"{r['median']:>+8.3f}{f'{r.n_neg}/{r.n}':>8}{r.p_fdr:>10.4f} {stars(r.p_fdr)}")
print(f"FDR survivors: {(S.p_fdr < .05).sum()} / {len(S)}")

# --- per-monkey sign consistency ---
print("\nPER-MONKEY median session slope  (n=4 animals - consistency only, not inference)")
print("=" * 96)
piv = (B.groupby(["outcome", "feature", "monkey"]).beta.median()
         .unstack("monkey"))
for oc, lab in OUTCOMES.items():
    for fk, fl in FEATS.items():
        if (oc, fk) not in piv.index:
            continue
        r = piv.loc[(oc, fk)]
        cells = "".join(f"{r[m]:>+10.3f}" if m in r and np.isfinite(r[m]) else f"{'-':>10}"
                        for m in piv.columns)
        agree = np.sum(np.sign(r.dropna()) == np.sign(r.dropna()).mode()[0]) if len(r.dropna()) else 0
        print(f"{lab.replace(chr(10),' '):<28}{fl:<26}{cells}   {agree}/{r.notna().sum()} same sign")
print(f"  columns: {list(piv.columns)}")

# --- comparison with the sEEG ---
print("\nvs sEEG alpha (beta: IF / asc2desc)")
print("=" * 96)
print(f"{'Outcome':<28}{'sEEG IF':>10}{'ECoG IF':>10}{'':<4}{'sEEG a2d':>10}{'ECoG a2d':>10}")
for oc, lab in OUTCOMES.items():
    e_if = R[(R.outcome == oc) & (R.feature == "mean_if_z")]
    e_ad = R[(R.outcome == oc) & (R.feature == "asc2desc_z")]
    if not len(e_if):
        continue
    s_if, s_ad = SEEG[lab]
    print(f"{lab.replace(chr(10),' '):<28}{s_if:>+10.3f}{e_if.beta.iloc[0]:>+10.3f}{'':<4}"
          f"{s_ad:>+10.3f}{e_ad.beta.iloc[0]:>+10.3f}")

# --- figure ---
order = list(OUTCOMES.values())[::-1]
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey="row")
for ci, (fk, fl) in enumerate(FEATS.items()):
    ax = axes[0][ci]
    sub = R[R.feature == fk].set_index("outcome_label").reindex(order)
    yy = np.arange(len(order))
    for yi, oc in enumerate(order):
        if oc not in sub.index or not np.isfinite(sub.loc[oc, "beta"]):
            continue
        r = sub.loc[oc]; col = COLORS[oc]
        ax.errorbar(r.beta, yi, xerr=[[r.beta - r.ci_lo], [r.ci_hi - r.beta]], fmt="o",
                    color=col, markerfacecolor=col if r.p_fdr < .05 else "white",
                    markeredgecolor=col, markersize=11, elinewidth=1.8, capsize=4, zorder=3)
        st = stars(r.p_fdr)
        if st != "ns":
            ax.text(r.beta, yi - 0.32, st, ha="center", fontsize=11, color=col, fontweight="bold")
    ax.axvline(0, color="0.5", lw=1, ls="--", zorder=1)
    ax.axhline(0.5, color="0.85", lw=1, zorder=0)
    ax.set_title(fl, fontsize=13, fontweight="bold", pad=8)
    ax.tick_params(labelsize=11)
    if ci == 0:
        ax.set_yticks(yy); ax.set_yticklabels(order, fontsize=11)
        ax.set_ylabel("Electrode level\n(session as random effect)", fontsize=11)

    ax = axes[1][ci]
    for yi, oc in enumerate(order):
        key = [k for k, v in OUTCOMES.items() if v == oc][0]
        v = B[(B.outcome == key) & (B.feature == fk)].beta.dropna().to_numpy()
        if not len(v):
            continue
        col = COLORS[oc]
        ax.scatter(v, np.full(len(v), yi) + np.random.default_rng(0).uniform(-.12, .12, len(v)),
                   s=22, color=col, alpha=0.5, linewidths=0, zorder=2)
        ax.plot([np.min(v), np.max(v)], [yi, yi], color=col, alpha=0.3, lw=3, zorder=1)
        ax.scatter([np.median(v)], [yi], marker="D", s=110, color=col,
                   edgecolor="0.2", linewidths=0.8, zorder=4)
        row = S[(S.outcome == key) & (S.feature == fk)]
        if len(row) and row.p_fdr.iloc[0] < .05:
            ax.text(np.median(v), yi - 0.34, stars(row.p_fdr.iloc[0]), ha="center",
                    fontsize=12, color=col, fontweight="bold")
    ax.axvline(0, color="0.5", lw=1, ls="--", zorder=1)
    ax.axhline(0.5, color="0.85", lw=1, zorder=0)
    ax.set_xlabel("Standardised beta", fontsize=12)
    ax.tick_params(labelsize=11)
    if ci == 0:
        ax.set_yticks(yy); ax.set_yticklabels(order, fontsize=11)
        ax.set_ylabel("Session level\n(model fitted within each session)", fontsize=11)

for ax, L in zip(axes.ravel(), "abcdef"):
    ax.annotate(L, xy=(0, 1), xycoords="axes fraction", xytext=(-46, 14),
                textcoords="offset points", fontsize=16, fontweight="bold",
                ha="right", va="bottom")
fig.legend(handles=[
    Line2D([0], [0], marker="o", color="0.4", markerfacecolor="0.4", markersize=9, lw=0, label="FDR < 0.05"),
    Line2D([0], [0], marker="o", color="0.4", markerfacecolor="white", markeredgecolor="0.4",
           markersize=9, lw=0, label="ns"),
    Line2D([0], [0], marker="o", color="0.55", markersize=6, lw=0, label="individual session"),
    Line2D([0], [0], marker="D", color="0.35", markersize=9, lw=0, label="median across sessions"),
], loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Macaque ECoG, awake eyes-closed - waveform shape vs connectivity and irreversibility\n"
             f"alpha (IMF4), spatial bipolar montage, {E.session.nunique()} sessions / "
             f"{E.monkey.nunique()} monkeys, {len(E):,} channels", fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig_ecog_primary.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_ecog_primary.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nsaved {OUT/'fig_ecog_primary.png'}")
