"""
compute_insideout_adaptive.py - bivariate time-irreversibility, 25 records from
23 patients, bipolar montage.

The INSIDEOUT measure (Deco et al. 2023) compares the forward and reversed
cross-correlation of a pair at a shift T, and T is defined as a quarter period of
the pair's own dominant frequency. That detail decides the result here. Electrode
mean instantaneous frequency in this dataset spans roughly 6 to 14 Hz, so a single
global T would only be correct near 10 Hz: holding the coupling identical and
varying the frequency alone from 8 to 13 Hz moves a fixed-T value by a factor of
about 70. Since instantaneous frequency is one of the predictors in the models,
a fixed T would build the frequency effect into the outcome.

T is therefore computed per pair from that pair's own mean instantaneous
frequency, f_pair = mean(mean_if_a, mean_if_b), giving T = round(fs / (4 *
f_pair)); the mean_if values come from the post-epoch feature table.

Output: electrode_level_irrev_adaptiveT.csv (column mean_io_adaptive)
        Batches: insideout_adaptive_batches/
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()
WORK = Path(os.environ.get("WAVEFORM_ALPHA_WORK", ROOT / "work")).expanduser()

sys.path.insert(0, str(CODE / "seeg"))

CFG_PATH  = CODE / "seeg" / "analysis" / "configs" / "bipolar_npz_windows.yaml"
TABLE_DIR = ROOT / "seeg" / "pair_tables" / "tables" / "bipolar"
def _work_dir():
    """Batch files are large; keep them next to the data if that folder
    exists, otherwise in WAVEFORM_ALPHA_WORK."""
    for d in (str(ROOT / "seeg" / "irrev_intermediates"), str(WORK)):
        try:
            p = Path(d)
            if p.exists():
                return p
        except OSError:
            continue
    raise SystemExit("no scratch folder: set WAVEFORM_ALPHA_WORK")

OUT_DIR   = _work_dir()
BATCH_DIR = OUT_DIR / "insideout_adaptive_batches"
IMF4_IDX  = 3
FALLBACK_HZ = 10.0     # only if a channel has no mean_if


def _insideout(a, b, T):
    """Reference INSIDEOUT (Deco et al. 2023), 2x2 form incl. diagonal terms."""
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


def process_recording(path_str, file_stem, entry_index, whitelist, mif_map):
    import numpy as np
    from seeg_waveform.config import load_config
    from seeg_waveform.preprocess import preprocess_segment, window_mask
    from seeg_waveform.primary_bipolar_mne import make_bipolar_run, analysis_windows
    from seeg_waveform.quinn import analyze_segment

    cfg = load_config(CFG_PATH)
    try:
        run = make_bipolar_run(path_str, cfg, entry_index=entry_index)
    except Exception:
        return []
    if run.status != "ok":
        return []
    rec = run.recording
    fs = float(rec.fs)
    _, post_win = analysis_windows(rec, cfg)
    mask = window_mask(rec.time, *post_win)
    wl = set(whitelist)

    sig = {}
    for i, ch in enumerate(rec.channel_names):
        if ch not in wl:
            continue
        try:
            proc, _, _ = preprocess_segment(rec.signal[mask, i], rec.time[mask], fs, cfg["preprocessing"])
            res = analyze_segment(proc, fs, cfg["emd"])
        except Exception:
            continue
        if res.status != "ok" or res.imf is None or res.imf.shape[1] <= IMF4_IDX:
            continue
        sig[ch] = np.asarray(res.imf[:, IMF4_IDX], float)

    chans = list(sig.keys())
    io_sum = {c: 0.0 for c in chans}; cnt = {c: 0 for c in chans}
    Tsum = {c: 0.0 for c in chans}
    for ii in range(len(chans)):
        for jj in range(ii + 1, len(chans)):
            a, b = chans[ii], chans[jj]
            fa = mif_map.get(a, np.nan); fb = mif_map.get(b, np.nan)
            f_pair = np.nanmean([fa, fb])
            if not np.isfinite(f_pair) or f_pair <= 0:
                f_pair = FALLBACK_HZ
            # ADAPTIVE T: quarter period of THIS pair's own frequency
            T = max(1, int(round(fs / (4.0 * f_pair))))
            if T >= len(sig[a]) // 4:
                continue
            try:
                io = _insideout(sig[a], sig[b], T)
            except Exception:
                continue
            if not np.isfinite(io):
                continue
            for c in (a, b):
                io_sum[c] += io; cnt[c] += 1; Tsum[c] += T
    out = []
    for c in chans:
        if cnt[c] == 0:
            continue
        out.append({"file": file_stem, "entry_index": entry_index, "subject": rec.subject,
                    "channel": c, "mean_io_adaptive": io_sum[c] / cnt[c],
                    "mean_T_used": Tsum[c] / cnt[c], "n_pairs": cnt[c]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=150)
    args = ap.parse_args()

    from seeg_waveform.config import load_config
    from seeg_waveform.io import iter_mat_files
    cfg = load_config(CFG_PATH)
    path_map = {p.stem: str(p) for p in iter_mat_files(cfg["project"]["data_dir"])}

    feats = pd.read_csv(TABLE_DIR / "channel_epoch_imf4_features.csv",
                        usecols=["file", "entry_index", "channel", "epoch", "mean_if"],
                        low_memory=False)
    post = feats[feats["epoch"] == "post"]
    wl_map, mif_maps = {}, {}
    for (f, e), g in post.groupby(["file", "entry_index"]):
        wl_map[(f, int(e))] = tuple(g["channel"].unique())
        mif_maps[(f, int(e))] = dict(zip(g["channel"], g["mean_if"]))

    groups = (post[["file", "entry_index"]].drop_duplicates()
              .sort_values(["file", "entry_index"]).reset_index(drop=True))
    groups = groups[groups["file"].isin(path_map)]
    if args.limit:
        groups = groups.head(args.limit)
    print(f"recordings: {len(groups):,}  workers: {args.jobs}  -> {BATCH_DIR}")
    print(f"ADAPTIVE T: per pair, T = round(fs / (4 * mean_if_pair))")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batches = [groups.iloc[i:i+args.batch_size] for i in range(0, len(groups), args.batch_size)]
    t0 = time.time(); all_rows = []
    for bi, batch in enumerate(batches):
        bpath = BATCH_DIR / f"batch_{bi:04d}.csv"
        if bpath.exists():
            all_rows.append(pd.read_csv(bpath)); print(f"[{bi+1}/{len(batches)}] cached", flush=True)
            continue
        res = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(process_recording)(path_map[r.file], r.file, int(r.entry_index),
                                       wl_map[(r.file, int(r.entry_index))],
                                       mif_maps[(r.file, int(r.entry_index))])
            for r in batch.itertuples())
        rows = pd.DataFrame([x for g in res for x in g])
        rows.to_csv(bpath, index=False); all_rows.append(rows)
        get_reusable_executor().shutdown(wait=True)
        el = time.time() - t0; done = bi + 1
        print(f"[{done}/{len(batches)}] {len(rows):,} electrodes | {el/60:.1f}m | "
              f"ETA {el/done*(len(batches)-done)/60:.1f}m", flush=True)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUT_DIR / "electrode_level_irrev_adaptiveT.csv", index=False)
    print(f"\nelectrodes: {len(df):,}")
    print(f"  mean_io_adaptive : mean {df.mean_io_adaptive.mean():.4e}")
    print(f"  mean T used      : {df.mean_T_used.mean():.1f} samples "
          f"(a quarter period at 10 Hz would be {round(512/(4*10.0))})")
    print(f"saved: {OUT_DIR / 'electrode_level_irrev_adaptiveT.csv'}")


if __name__ == "__main__":
    main()
