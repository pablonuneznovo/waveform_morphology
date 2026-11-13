

import argparse, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

sys.path.insert(0, str(CODE / "seeg"))

CFG_PATH  = CODE / "seeg" / "analysis" / "configs" / "bipolar_npz_windows.yaml"
TABLE_DIR = ROOT / "seeg" / "pair_tables" / "tables" / "bipolar"
OUT_DIR   = ROOT / "seeg" / "analysis" / "outputs"
BATCH_DIR = OUT_DIR / "uni_irrev_batches"
IMF4_IDX  = 3

def process_group(path_str, file_stem, entry_index, whitelist):
    import numpy as np
    from scipy import stats
    from seeg_waveform.config import load_config
    from seeg_waveform.preprocess import preprocess_segment, window_mask
    from seeg_waveform.primary_bipolar_mne import make_bipolar_run, analysis_windows
    from seeg_waveform.quinn import analyze_segment

    cfg = load_config(CFG_PATH)
    try:
        run = make_bipolar_run(path_str, cfg, entry_index=entry_index)
    except Exception as exc:
        return [{"file": file_stem, "entry_index": entry_index, "channel": "",
                 "status": f"run_exception:{type(exc).__name__}"}]
    if run.status != "ok":
        return [{"file": file_stem, "entry_index": entry_index, "channel": "",
                 "status": f"run_{run.status}"}]

    rec = run.recording
    wl = set(whitelist)
    selected = [i for i, ch in enumerate(rec.channel_names) if ch in wl]
    _, post_win = analysis_windows(rec, cfg)
    mask = window_mask(rec.time, *post_win)

    out = []
    for idx in selected:
        row = {"file": file_stem, "entry_index": entry_index,
               "channel": rec.channel_names[idx], "subject": rec.subject,
               "status": "ok", "uni_irrev": np.nan, "uni_irrev_abs": np.nan}
        try:
            proc, _, _ = preprocess_segment(rec.signal[mask, idx], rec.time[mask],
                                            rec.fs, cfg["preprocessing"])
            res = analyze_segment(proc, rec.fs, cfg["emd"])
        except Exception as exc:
            row["status"] = f"seg_exception:{type(exc).__name__}"; out.append(row); continue
        if res.status != "ok" or res.imf is None or res.imf.shape[1] <= IMF4_IDX:
            row["status"] = f"seg_{res.status}"; out.append(row); continue
        imf4 = np.asarray(res.imf[:, IMF4_IDX], float)
        d = np.diff(imf4)
        sk = float(stats.skew(d, bias=False)) if d.size > 5 and np.std(d) > 1e-12 else np.nan
        row["uni_irrev"] = sk
        row["uni_irrev_abs"] = abs(sk)
        out.append(row)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--out", default="univariate_irreversibility.csv")
    args = ap.parse_args()

    from seeg_waveform.config import load_config
    from seeg_waveform.io import iter_mat_files

    cfg = load_config(CFG_PATH)
    path_map = {p.stem: str(p) for p in iter_mat_files(cfg["project"]["data_dir"])}

    feats = pd.read_csv(TABLE_DIR / "channel_epoch_imf4_features.csv",
                        usecols=["file", "entry_index", "channel", "epoch"], low_memory=False)
    post = feats[feats["epoch"] == "post"]
    wl_map = {(f, int(e)): tuple(g["channel"].unique())
              for (f, e), g in post.groupby(["file", "entry_index"])}
    groups = (post[["file", "entry_index"]].drop_duplicates()
              .sort_values(["file", "entry_index"]).reset_index(drop=True))
    groups = groups[groups["file"].isin(path_map)]
    if args.limit:
        groups = groups.head(args.limit)
    print(f"recordings: {len(groups):,}   workers: {args.jobs}")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batches = [groups.iloc[i:i+args.batch_size] for i in range(0, len(groups), args.batch_size)]
    t0 = time.time()
    all_rows = []
    for bi, batch in enumerate(batches):
        bpath = BATCH_DIR / f"batch_{bi:04d}.csv"
        if bpath.exists():
            all_rows.append(pd.read_csv(bpath)); print(f"[{bi+1}/{len(batches)}] cached", flush=True)
            continue
        res = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(process_group)(path_map[r.file], r.file, int(r.entry_index),
                                   wl_map[(r.file, int(r.entry_index))])
            for r in batch.itertuples())
        rows = pd.DataFrame([x for g in res for x in g])
        rows.to_csv(bpath, index=False)
        all_rows.append(rows)
        get_reusable_executor().shutdown(wait=True)
        el = time.time() - t0; done = bi + 1
        print(f"[{done}/{len(batches)}] {len(rows):,} rows | {el/60:.1f}m | "
              f"ETA {el/done*(len(batches)-done)/60:.1f}m", flush=True)

    df = pd.concat(all_rows, ignore_index=True)
    ok = df[df["status"] == "ok"]
    print(f"\nelectrodes: {len(df):,}  usable: {len(ok):,}")
    print(f"status: {df['status'].value_counts().head().to_dict()}")
    print(f"uni_irrev (signed): mean {ok['uni_irrev'].mean():+.4f}  sd {ok['uni_irrev'].std():.4f}")
    print(f"uni_irrev_abs     : mean {ok['uni_irrev_abs'].mean():.4f}")
    df.to_csv(OUT_DIR / args.out, index=False)
    print(f"saved: {OUT_DIR / args.out}")

if __name__ == "__main__":
    main()
