"""
ecog_pipeline.py - waveform / connectivity pipeline for the Neurotycho ECoG.

Same waveform descriptors, same five outcomes and the same adaptive-T INSIDEOUT as the
sEEG analysis, so the two datasets are directly comparable.

  * NOTCH 50/100/150 Hz, where the sEEG uses 60/120/180. RIKEN is eastern Japan;
    measured peak/neighbour ratio 25.6 at 50 Hz against 1.18 at 60 Hz.
  * IMF_IDX = 3 (IMF4). Verified empirically: 11.8 Hz awake eyes-closed, cascade
    ratios near 2.0. Same index as the sEEG, because the EMD masks are pinned in
    absolute Hz.
  * BIPOLAR montage from the published X/Y map: consecutive channel numbering
    corresponds to mutual nearest neighbours for only 23% of pairs. Built by
    ecog_montage.py.

Per condition -> per epoch -> per bipolar channel: IMF4 waveform features + univariate
irreversibility; then every bipolar pair: coherence, |ImCoh|, gPDC asymmetry, INSIDEOUT
with per-pair adaptive T. Pair values are averaged over epochs, then aggregated to the
electrode level over all that channel's pairs (exactly as in the sEEG).

Outputs -> outputs/{channels,pairs,electrodes}_<session>_<condition>.csv
"""
import argparse, sys, time, itertools, warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps, stats
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

# Data lives outside the repository: WAVEFORM_ALPHA points at the folder
# holding seeg/ and ecog/. Without it the paths resolve against the working
# directory, so run from that folder instead.
CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

sys.path.insert(0, str(CODE / "seeg"))

RAW  = ROOT / "ecog" / "raw"
OUT  = ROOT / "ecog" / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FS         = 1000.0
NOTCH      = [50.0, 100.0, 150.0]
NOTCH_Q    = 30.0
IMF_IDX    = 3               # IMF4 = alpha (verified)
MAX_IMFS   = 6
EPOCH_S    = 28.0            # same window as the sEEG analysis
WELCH_S    = 4.0
BAND       = (8.0, 13.0)
MAX_EPOCHS = 10
N_PDC      = 500
SEED       = 0

EMD_CFG = {"max_imfs": MAX_IMFS, "nphases": 4, "imf_index": IMF_IDX, "smooth_phase": 3,
           "min_cycles_required": 5, "max_if_hz": 20.0, "max_if_range_hz": 20.0,
           "amp_percentile": 10.0}

# --- helpers ---
def notch(x, fs=FS):
    for f0 in NOTCH:
        if f0 < fs / 2:
            b, a = sps.iirnotch(f0, NOTCH_Q, fs)
            x = sps.filtfilt(b, a, x)
    return x

def band_pair(x, y, fs=FS):
    """Coherence and |ImCoh| averaged over the alpha band."""
    nper = int(WELCH_S * fs)
    f, Pxy = sps.csd(x, y, fs, nperseg=nper)
    _, Pxx = sps.welch(x, fs, nperseg=nper)
    _, Pyy = sps.welch(y, fs, nperseg=nper)
    m = (f >= BAND[0]) & (f <= BAND[1])
    denom = np.sqrt(Pxx[m] * Pyy[m])
    denom[denom < 1e-30] = 1e-30
    C = Pxy[m] / denom
    return float(np.mean(np.abs(C))), float(np.mean(np.abs(C.imag)))

def insideout(a, b, T):
    """Reference INSIDEOUT (Deco et al. 2023), the same estimator used for the sEEG."""
    X = np.vstack([a, b]).astype(float)
    X -= X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True); sd[sd < np.finfo(float).eps] = np.finfo(float).eps
    X /= sd
    Xr = X[:, ::-1]
    Cf = np.empty((2, 2)); Cr = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            Cf[i, j] = np.mean(X[i, :-T] * X[j, T:])
            Cr[i, j] = np.mean(Xr[i, :-T] * Xr[j, T:])
    Cf = np.clip(Cf, -0.999999, 0.999999); Cr = np.clip(Cr, -0.999999, 0.999999)
    FSf = -0.5 * np.log(1 - Cf ** 2); FSr = -0.5 * np.log(1 - Cr ** 2)
    return float(np.mean((FSf - FSr) ** 2))

def conditions_of(sess: Path):
    c = sio.loadmat(sess / "Condition.mat")
    lab = [str(v[0]) for v in c["ConditionLabel"][0]]
    idx = np.asarray(c["ConditionIndex"][0], int)
    out = {}
    for i, L in enumerate(lab):
        if L.endswith("-Start"):
            nm = L[:-6]
            e = [j for j, M in enumerate(lab) if M == nm + "-End"]
            if e:
                out[nm] = (int(idx[i]), int(idx[e[0]]))
    return out

# --- one epoch: all channels + all pairs ---
def do_epoch(sig, ep_i, want_pdc_pairs):
    """sig: dict bip_id -> raw bipolar segment for this epoch."""
    import numpy as np
    from scipy import stats as st
    from seeg_waveform.quinn import analyze_segment
    from seeg_waveform.all_pairs_waveform_connectivity import generalized_pdc_pair

    feats, imf = {}, {}
    for bid, raw in sig.items():
        x = notch(sps.detrend(raw.astype(float)))
        try:
            res = analyze_segment(x, FS, EMD_CFG)
        except Exception:
            continue
        if res.status != "ok" or res.imf is None or res.imf.shape[1] <= IMF_IDX:
            continue
        t = res.cycle_table
        if t is None or not len(t):
            continue
        m = np.asarray(res.imf[:, IMF_IDX], float)
        d = np.diff(m)
        feats[bid] = {
            "bip": bid, "epoch": ep_i, "n_cycles": int(len(t)),
            "mean_if": float(np.nanmean(t["mean_if"])),
            "asc2desc": float(np.nanmean(t["asc2desc"])),
            "peak2trough": float(np.nanmean(t["peak2trough"])),
            "uni_irrev": float(st.skew(d, bias=False)) if d.size > 5 and np.std(d) > 1e-12 else np.nan,
        }
        imf[bid] = m

    rows = []
    ids = sorted(imf)
    for a, b in itertools.combinations(ids, 2):
        x, y = imf[a], imf[b]
        try:
            coh, imc = band_pair(x, y)
        except Exception:
            continue
        # ADAPTIVE T: quarter period of THIS pair's own frequency
        fp = np.nanmean([feats[a]["mean_if"], feats[b]["mean_if"]])
        if not np.isfinite(fp) or fp <= 0:
            fp = 10.0
        T = max(1, int(round(FS / (4.0 * fp))))
        io = insideout(x, y, T) if T < len(x) // 4 else np.nan
        pdc = np.nan
        if (a, b) in want_pdc_pairs:
            try:
                # max_order 30, as in the sEEG: at target_fs 128 alpha has a
                # ~11.6-sample period, and measured pdc_valid is 100% at this order.
                pr = generalized_pdc_pair(x, y, FS, target_fs=128.0, max_order=30,
                                          whiteness_alpha=0.001, epsilon=1e-12)
                if pr.get("pdc_valid"):
                    pdc = float(pr["gpdc_a_to_b"] - pr["gpdc_b_to_a"])
            except Exception:
                pass
        rows.append({"epoch": ep_i, "bip_a": a, "bip_b": b, "coh": coh,
                     "imcoh": imc, "insideout": io, "T": T, "pdc_asym": pdc})
    return list(feats.values()), rows

# --- main ---
def run_session(folder: Path, monkey: str, jobs: int, limit_cond=None):
    mont = pd.read_csv(OUT / f"ecog_montage_{monkey}.csv")
    print(f"montage: {len(mont)} bipolar channels")

    inner = [d for d in folder.iterdir() if d.is_dir()]
    base = inner[0] if len(inner) == 1 and not (folder / "Session1").exists() else folder
    sessions = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("Session")])

    for sess in sessions:
        cond = conditions_of(sess)
        if not cond:
            continue
        tag = f"{folder.name.split('_')[0]}_{sess.name}"
        # load every electrode this montage needs, once
        need = sorted(set(mont.ch_a) | set(mont.ch_b))
        raw = {}
        for ch in need:
            fp = sess / f"ECoG_ch{ch}.mat"
            if fp.exists():
                raw[ch] = sio.loadmat(fp)[f"ECoGData_ch{ch}"][0]
        print(f"\n{tag}: loaded {len(raw)} electrodes, conditions {list(cond)}")

        for cname, (i0, i1) in cond.items():
            if limit_cond and cname not in limit_cond:
                continue
            n = int(EPOCH_S * FS)
            n_ep = min(MAX_EPOCHS, max(0, (i1 - i0) // n))
            if n_ep == 0:
                print(f"  {cname}: too short, skipped"); continue
            starts = np.linspace(i0, i1 - n, n_ep).astype(int)

            ids = list(mont.bip)
            allp = list(itertools.combinations(sorted(ids), 2))
            rng = np.random.default_rng(SEED)
            sel = rng.choice(len(allp), size=min(N_PDC, len(allp)), replace=False)
            want_pdc = {allp[k] for k in sel}

            t0 = time.time()
            payload = []
            for ei, s0 in enumerate(starts):
                sig = {}
                for r in mont.itertuples():
                    if r.ch_a in raw and r.ch_b in raw:
                        sig[r.bip] = (raw[r.ch_a][s0:s0 + n].astype(float)
                                      - raw[r.ch_b][s0:s0 + n].astype(float))
                payload.append((sig, ei))
            res = Parallel(n_jobs=jobs, backend="loky")(
                delayed(do_epoch)(s, e, want_pdc) for s, e in payload)
            get_reusable_executor().shutdown(wait=True)

            ch_rows = [r for f, _ in res for r in f]
            pr_rows = [r for _, p in res for r in p]
            if not ch_rows:
                print(f"  {cname}: no usable channels"); continue
            C = pd.DataFrame(ch_rows); P = pd.DataFrame(pr_rows)
            C["session"] = tag; C["condition"] = cname; C["monkey"] = monkey
            P["session"] = tag; P["condition"] = cname; P["monkey"] = monkey
            C.to_csv(OUT / f"channels_{tag}_{cname}.csv", index=False)
            P.to_csv(OUT / f"pairs_{tag}_{cname}.csv", index=False)

            # electrode level: average each channel's waveform over epochs, and its
            # connectivity over every pair it belongs to (same as the sEEG)
            wf = C.groupby("bip").agg(mean_if=("mean_if", "mean"),
                                      asc2desc=("asc2desc", "mean"),
                                      peak2trough=("peak2trough", "mean"),
                                      uni_irrev=("uni_irrev", "mean"),
                                      n_cycles=("n_cycles", "mean")).reset_index()
            pm = P.groupby(["bip_a", "bip_b"]).agg(
                coh=("coh", "mean"), imcoh=("imcoh", "mean"),
                insideout=("insideout", "mean"), pdc_asym=("pdc_asym", "mean")).reset_index()
            a = pm.rename(columns={"bip_a": "bip", "bip_b": "_o"})
            b = pm.rename(columns={"bip_b": "bip", "bip_a": "_o"})
            stack = pd.concat([a, b], ignore_index=True)
            el = stack.groupby("bip").agg(mean_coh=("coh", "mean"),
                                          mean_abs_imcoh=("imcoh", "mean"),
                                          mean_io_adaptive=("insideout", "mean"),
                                          pdc_asymmetry=("pdc_asym", "mean"),
                                          n_pairs=("coh", "size")).reset_index()
            E = wf.merge(el, on="bip", how="inner")
            E["session"] = tag; E["condition"] = cname; E["monkey"] = monkey
            E = E.merge(mont[["bip", "x", "y", "dist_px"]], on="bip", how="left")
            E.to_csv(OUT / f"electrodes_{tag}_{cname}.csv", index=False)
            print(f"  {cname:<22} {n_ep:>2} epochs  {len(E):>3} channels  {len(pm):>5} pairs  "
                  f"IMF4 {E.mean_if.median():5.2f} Hz  T {P['T'].median():.0f}  "
                  f"PDC valid {P[P.pdc_asym.notna() | P.set_index(['bip_a','bip_b']).index.isin(want_pdc)].pdc_asym.notna().mean()*100:4.1f}%"
                  f" of {len(want_pdc)} sampled  {time.time()-t0:5.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="20110112KTMD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128")
    ap.add_argument("--monkey", default="George")
    ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--cond", default=None, help="comma-separated subset")
    a = ap.parse_args()
    run_session(RAW / a.folder, a.monkey, a.jobs,
                a.cond.split(",") if a.cond else None)
