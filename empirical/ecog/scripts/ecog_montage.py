"""
ecog_montage.py - build and verify the spatial bipolar montage for the Neurotycho ECoG.

The 128 electrodes follow their own numbering: only 23% of consecutive channel
pairs are mutual nearest neighbours, and the median consecutive-channel distance is 43%
larger than the median nearest-neighbour distance. So a bipolar montage must be built
from the published X/Y coordinates, not from channel numbering.

Montage: minimum-distance matching over the electrode graph - repeatedly take the
closest still-unused pair, so each electrode appears in at most one bipolar channel.

Outputs
  outputs/ecog_montage_<monkey>.csv   bipolar channel table (a, b, distance, midpoint)
  outputs/fig_ecog_montage_<monkey>.png  electrodes + pairing drawn on the brain image
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAPS = {}
# Data lives outside the repository: WAVEFORM_ALPHA points at the folder
# holding seeg/ and ecog/. Without it the paths resolve against the working
# directory, so run from that folder instead.
ROOT = Path(os.environ.get("WAVEFORM_ALPHA", ".")).expanduser()

for _d in sorted((ROOT / "ecog" / "raw" / "position_locations").iterdir()):
    if not _d.is_dir():
        continue
    for _m in _d.rglob("*.mat"):
        for _name in ("George", "Chibi", "Kin2", "Su"):
            if _m.name.lower() == _name.lower() + "map.mat" and _name not in MAPS:
                MAPS[_name] = str(_m)
OUT = ROOT / "ecog" / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MAX_PAIR_FACTOR = 1.6      # reject a bipolar pair farther than this x median NN distance

def find_map(p):
    p = Path(p)
    if p.is_file():
        return p
    hits = list(p.rglob("*Map.mat")) + list(p.rglob("*MAP.mat")) + list(p.rglob("*map.mat"))
    return hits[0] if hits else None

def greedy_matching(P, max_d):
    """Closest-first non-overlapping pairing."""
    n = len(P)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu = np.triu_indices(n, 1)
    order = np.argsort(D[iu])
    used = np.zeros(n, bool)
    pairs = []
    for k in order:
        i, j = iu[0][k], iu[1][k]
        if used[i] or used[j] or D[i, j] > max_d:
            continue
        used[i] = used[j] = True
        pairs.append((i, j, D[i, j]))
    return pairs, D

for monkey, path in MAPS.items():
    mp = find_map(path)
    if mp is None:
        print(f"{monkey:<8} no map file found under {path}")
        continue
    d = sio.loadmat(mp)
    X = d["X"].ravel(); Y = d["Y"].ravel(); I = d.get("I")
    P = np.c_[X, Y]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    med = float(np.median(nn))

    pairs, D = greedy_matching(P, MAX_PAIR_FACTOR * med)
    dist = np.array([p[2] for p in pairs])
    used = sorted({i for p in pairs for i in p[:2]})

    rows = []
    for k, (i, j, dd) in enumerate(pairs):
        rows.append({
            "monkey": monkey, "bip": k + 1,
            "ch_a": i + 1, "ch_b": j + 1,          # .mat files are 1-indexed
            "dist_px": dd, "dist_rel": dd / med,
            "x": (X[i] + X[j]) / 2, "y": (Y[i] + Y[j]) / 2,
        })
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f"ecog_montage_{monkey}.csv", index=False)

    print(f"\n{monkey}: {len(X)} electrodes  median NN {med:.1f} px")
    print(f"  bipolar channels : {len(pairs)}  (electrodes used {len(used)}/{len(X)})")
    print(f"  pair distance px : median {np.median(dist):.1f}  "
          f"IQR {np.percentile(dist,25):.1f}-{np.percentile(dist,75):.1f}  max {dist.max():.1f}")
    print(f"  as multiples of median NN: median {np.median(dist)/med:.2f}  max {dist.max()/med:.2f}")
    print(f"  -> {len(pairs)*(len(pairs)-1)//2:,} bipolar pairs available for connectivity")

    fig, ax = plt.subplots(figsize=(7, 8))
    if I is not None:
        ax.imshow(np.asarray(I))
    ax.scatter(X, Y, s=26, c="#4C8BB5", edgecolor="white", linewidths=0.6, zorder=3)
    for i, j, dd in pairs:
        ax.plot([X[i], X[j]], [Y[i], Y[j]], "-", color="#C44536", lw=2, zorder=2)
    unused = [k for k in range(len(X)) if k not in used]
    if unused:
        ax.scatter(X[unused], Y[unused], s=46, facecolor="none",
                   edgecolor="#E8A838", linewidths=1.8, zorder=4)
    ax.set_title(f"{monkey}: {len(pairs)} bipolar channels from {len(X)} electrodes\n"
                 f"red = bipolar pairing, orange ring = unpaired", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_ecog_montage_{monkey}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT / f'fig_ecog_montage_{monkey}.png'}")
