

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .localization import normalize_channel_name

ATLAS_COLUMNS = [
    "HCPEX",
    "HCP_FSL",
    "AAL",
    "ANATOMY3",
    "COBRA",
    "HAMMERS",
    "ISBR",
    "JULICHBRAIN",
    "LPBA40",
    "MORI",
    "SUIT",
    "NEUROMORPHOMETRICS",
    "THALAMUS",
    "THALAMIC_NUCLEI",
    "TIAN_SUBCORTEX",
]

NON_GRAY_PATTERNS = [
    "white_matter",
    "out_of_atlas",
    "out of atlas",
    "csf",
    "ventricle",
    "cerebral_white",
    "unknown",
    "not found",
]

def _norm_region_text(value: object) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower())

def is_non_gray_label(value: object) -> bool:
    text = _norm_region_text(value)
    return any(pattern in text for pattern in NON_GRAY_PATTERNS)

def load_brainstorm_atlas_csv(path: Path, primary_atlas: str = "AAL") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["channel"] = df["Name"].astype(str)
    df["channel_norm"] = df["channel"].map(normalize_channel_name)
    df["x_mni"] = pd.to_numeric(df["MNICoordinate_1"], errors="coerce")
    df["y_mni"] = pd.to_numeric(df["MNICoordinate_2"], errors="coerce")
    df["z_mni"] = pd.to_numeric(df["MNICoordinate_3"], errors="coerce")
    df["x_world"] = pd.to_numeric(df["WORLDCoordinate_1"], errors="coerce")
    df["y_world"] = pd.to_numeric(df["WORLDCoordinate_2"], errors="coerce")
    df["z_world"] = pd.to_numeric(df["WORLDCoordinate_3"], errors="coerce")
    df["x_scs_mm"] = pd.to_numeric(df["SCSCoordinate_1"], errors="coerce") * 1000.0
    df["y_scs_mm"] = pd.to_numeric(df["SCSCoordinate_2"], errors="coerce") * 1000.0
    df["z_scs_mm"] = pd.to_numeric(df["SCSCoordinate_3"], errors="coerce") * 1000.0
    if primary_atlas not in df.columns:
        raise ValueError(f"Primary atlas {primary_atlas!r} not found. Available: {ATLAS_COLUMNS}")
    df["primary_atlas"] = primary_atlas
    df["primary_region"] = df[primary_atlas].astype(str)
    df["primary_non_gray_flag"] = df["primary_region"].map(is_non_gray_label)
    df["any_out_of_atlas_flag"] = df[ATLAS_COLUMNS].astype(str).apply(
        lambda row: any(_norm_region_text(v) == "out_of_atlas" for v in row), axis=1
    )
    df["any_white_matter_flag"] = df[ATLAS_COLUMNS].astype(str).apply(
        lambda row: any("white_matter" in _norm_region_text(v) for v in row), axis=1
    )
    df["analysis_gray_matter_flag"] = ~df["primary_non_gray_flag"]
    df["hemisphere"] = np.where(df["x_mni"] < 0, "left", "right")
    return df

def coords_matrix(df: pd.DataFrame, space: str = "mni") -> np.ndarray:
    if space == "mni":
        cols = ["x_mni", "y_mni", "z_mni"]
    elif space == "world":
        cols = ["x_world", "y_world", "z_world"]
    elif space == "scs":
        cols = ["x_scs_mm", "y_scs_mm", "z_scs_mm"]
    else:
        raise ValueError("space must be 'mni', 'world', or 'scs'")
    return df[cols].to_numpy(dtype=float)

def nearest_stim_distance(
    atlas: pd.DataFrame,
    stimulated_channels: str,
    space: str = "mni",
) -> tuple[pd.Series, pd.Series]:
    stim_norms = [normalize_channel_name(x) for x in str(stimulated_channels).split(";") if str(x).strip()]
    stim = atlas.loc[atlas["channel_norm"].isin(stim_norms)]
    if stim.empty:
        return pd.Series(np.nan, index=atlas.index), pd.Series("", index=atlas.index)
    all_xyz = coords_matrix(atlas, space)
    stim_xyz = coords_matrix(stim, space)
    distances = np.linalg.norm(all_xyz[:, None, :] - stim_xyz[None, :, :], axis=2)
    nearest_idx = np.nanargmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(atlas)), nearest_idx]
    nearest_channel = stim.iloc[nearest_idx]["channel"].to_numpy()
    return pd.Series(nearest_distance, index=atlas.index), pd.Series(nearest_channel, index=atlas.index)

def add_waveform_spatial_features(
    waveform: pd.DataFrame,
    atlas: pd.DataFrame,
    subject_id: str = "Subject13",
    space: str = "mni",
) -> pd.DataFrame:
    wf = waveform.loc[waveform["subject"].astype(str).str.contains(subject_id, case=False, na=False)].copy()
    wf["channel_norm"] = wf["channel"].map(normalize_channel_name)
    atlas_keep = atlas.drop_duplicates("channel_norm").copy()
    rows = []
    for _, run in wf.groupby("file", dropna=False):
        run = run.copy()
        dist, nearest = nearest_stim_distance(atlas_keep, run["stimulated_channels"].dropna().iloc[0], space=space)
        dist_table = atlas_keep[["channel_norm"]].copy()
        dist_table[f"distance_to_stim_{space}_mm"] = dist.to_numpy()
        dist_table["nearest_stim_channel"] = nearest.to_numpy()
        rows.append(run.merge(dist_table, on="channel_norm", how="left"))
    out = pd.concat(rows, ignore_index=True) if rows else wf
    atlas_cols = [
        "channel_norm",
        "channel",
        "x_mni",
        "y_mni",
        "z_mni",
        "x_world",
        "y_world",
        "z_world",
        "primary_atlas",
        "primary_region",
        "primary_non_gray_flag",
        "any_out_of_atlas_flag",
        "any_white_matter_flag",
        "analysis_gray_matter_flag",
        "hemisphere",
    ]
    return out.merge(atlas[atlas_cols].drop_duplicates("channel_norm"), on="channel_norm", how="left", suffixes=("", "_atlas"))

def add_connectivity_spatial_features(
    connectivity: pd.DataFrame,
    atlas: pd.DataFrame,
    subject_id: str = "Subject13",
    space: str = "mni",
) -> pd.DataFrame:
    conn = connectivity.loc[connectivity["subject"].astype(str).str.contains(subject_id, case=False, na=False)].copy()
    if conn.empty:
        return conn
    conn["seed_norm"] = conn["seed_channel"].map(normalize_channel_name)
    conn["target_norm"] = conn["target_channel"].map(normalize_channel_name)
    atlas = atlas.drop_duplicates("channel_norm").copy()
    if space == "mni":
        xyz_cols = ["x_mni", "y_mni", "z_mni"]
    elif space == "world":
        xyz_cols = ["x_world", "y_world", "z_world"]
    else:
        xyz_cols = ["x_scs_mm", "y_scs_mm", "z_scs_mm"]
    seed = atlas[["channel_norm", "primary_region", "analysis_gray_matter_flag", *xyz_cols]].rename(
        columns={
            "channel_norm": "seed_norm",
            "primary_region": "seed_region",
            "analysis_gray_matter_flag": "seed_gray_matter_flag",
            xyz_cols[0]: "seed_x",
            xyz_cols[1]: "seed_y",
            xyz_cols[2]: "seed_z",
        }
    )
    target = atlas[["channel_norm", "primary_region", "analysis_gray_matter_flag", *xyz_cols]].rename(
        columns={
            "channel_norm": "target_norm",
            "primary_region": "target_region",
            "analysis_gray_matter_flag": "target_gray_matter_flag",
            xyz_cols[0]: "target_x",
            xyz_cols[1]: "target_y",
            xyz_cols[2]: "target_z",
        }
    )
    out = conn.merge(seed, on="seed_norm", how="left").merge(target, on="target_norm", how="left")
    out[f"seed_target_distance_{space}_mm"] = np.linalg.norm(
        out[["seed_x", "seed_y", "seed_z"]].to_numpy(float)
        - out[["target_x", "target_y", "target_z"]].to_numpy(float),
        axis=1,
    )
    return out

def spearman_summary(df: pd.DataFrame, x: str, y: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        sub = group[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        row = dict(zip(group_cols, keys))
        row["x"] = x
        row["y"] = y
        row["n"] = int(len(sub))
        if len(sub) >= 4 and sub[x].nunique() > 1 and sub[y].nunique() > 1:
            rho, p = stats.spearmanr(sub[x], sub[y])
            row["spearman_rho"] = float(rho)
            row["p_value"] = float(p)
        else:
            row["spearman_rho"] = np.nan
            row["p_value"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return pd.Series(out, index=p_values.index)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[ok])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return pd.Series(out, index=p_values.index)
