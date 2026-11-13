"""
ecog_run_all.py - run the primary analysis over every session: AwakeEyesClosed only.

Design:
    which is 99% power for the sEEG's univariate~asymmetry effect (beta 0.107).

Outputs land in outputs/ as {channels,pairs,electrodes}_<date>_<Session>_AwakeEyesClosed.csv
"""
import re, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ecog_pipeline import run_session, RAW, OUT

MONKEYS = ("George", "Chibi", "Kin2", "Su")
COND = ["AwakeEyesClosed"]

def main(jobs=13):
    folders = []
    for d in sorted(RAW.iterdir()):
        if not d.is_dir() or d.name == "position_locations":
            continue
        m = re.match(r"(\d{8})(KTMD|PF|KT|MD|SLP)_.*?_(" + "|".join(MONKEYS) + r")_", d.name)
        if m:
            folders.append((d, m.group(3), m.group(1), m.group(2)))
        else:
            print(f"SKIP (cannot parse monkey): {d.name}")

    print(f"{len(folders)} session folders to process, condition = {COND}\n")
    t0 = time.time()
    done, failed = 0, []
    for i, (d, monkey, date, exp) in enumerate(folders, 1):
        tag = f"{date}{exp}_{monkey}"
        # skip if this folder already produced electrode output
        if list(OUT.glob(f"electrodes_{date}*_AwakeEyesClosed.csv")):
            print(f"[{i}/{len(folders)}] {tag:<22} cached"); done += 1; continue
        print(f"[{i}/{len(folders)}] {tag:<22} ...", flush=True)
        try:
            run_session(d, monkey, jobs, COND)
            done += 1
        except Exception:
            failed.append(tag)
            print(f"    FAILED {tag}\n{traceback.format_exc(limit=3)}", flush=True)
        el = (time.time() - t0) / 60
        print(f"    elapsed {el:.1f} min | ETA {el/i*(len(folders)-i):.1f} min", flush=True)

    print(f"\ndone {done}/{len(folders)}   failed: {failed if failed else 'none'}")
    n = len(list(OUT.glob("electrodes_*_AwakeEyesClosed.csv")))
    print(f"electrode files written: {n}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 13)
