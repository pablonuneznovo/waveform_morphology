"""The bipolar montage and the channel retention rules.

This is where the analysed set of channels is defined. Bipolar channels are
formed from adjacent contacts on the same shaft; a channel is kept only if its
own atlas label is grey matter and both constituent contacts resolve to one
region family matching that label. Channels sharing a contact with the
stimulation pair are dropped, while the stimulated channel itself is kept and
flagged so it can enter the models as a covariate.
"""

from dataclasses import dataclass
import hashlib
from itertools import combinations, product
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from .config import load_config
from .io import SeegRecording, count_entries, iter_mat_files, load_recording, parse_subject_from_name
from .preprocess import classify_post_stim_response, post_window_for_frequency, preprocess_segment, window_mask
from .quinn import SegmentResult, analyze_segment
from .quinn_figures import set_publication_style
from .region_waveform import atlas_region_family, subject_number
from .spatial_analysis import load_brainstorm_atlas_csv


BANDS = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]
BAND_LABELS = {
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "low_gamma": "Low gamma",
    "high_gamma": "High gamma",
}
DISTANCE_BIN_EDGES = [0.0, 30.0, 60.0, 90.0, np.inf]
DISTANCE_BIN_LABELS = ["<30", "30-<60", "60-<90", ">=90"]


def normalize_contact(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def split_bipolar_node(value: object) -> tuple[str, str] | None:
    parts = [normalize_contact(part) for part in re.split(r"\s*-\s*", str(value)) if str(part).strip()]
    return (parts[0], parts[1]) if len(parts) == 2 else None


@dataclass
class BipolarRun:
    recording: SeegRecording
    nodes: pd.DataFrame
    source_recording: SeegRecording
    status: str
    message: str = ""


@dataclass
class NodeWindow:
    processed: np.ndarray
    result: SegmentResult
    drift_flag: bool
    noise_flag: bool


def filename_stim_amplitude_ma(path: str | Path) -> float:
    """Return the stimulation current encoded in a recording filename."""
    match = re.search(r"_I=([0-9]+(?:\.[0-9]+)?)", Path(path).stem)
    return float(match.group(1)) if match else np.nan


def recording_fingerprint(recording: SeegRecording) -> str:
    """Fingerprint recorded data to prevent duplicated runs entering summaries."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(recording.signal).tobytes())
    digest.update(np.ascontiguousarray(recording.time).tobytes())
    digest.update("|".join(recording.channel_names).encode("utf-8"))
    digest.update("|".join(recording.stimulated_channel_names).encode("utf-8"))
    digest.update(f"{recording.stim_frequency_hz:g}|{recording.stim_amplitude_ma:g}".encode("ascii"))
    return digest.hexdigest()


def duplicate_recording_exclusions(files: list[Path]) -> dict[tuple[Path, int], str]:
    """Identify exact data duplicates across every run in every file, and retain the
    (file, entry_index) agreeing with metadata. Keyed by (path, entry_index) since one
    file can hold many runs and duplicates can occur at the run level, not just the file
    level.
    """
    grouped: dict[str, list[tuple[Path, int, float]]] = {}
    for path in files:
        try:
            n_entries = count_entries(path)
        except Exception as exc:
            print(f"  [SKIP unreadable] {path.name}: {exc}")
            continue
        for entry_index in range(n_entries):
            try:
                recording = load_recording(path, entry_index=entry_index)
            except Exception as exc:
                print(f"  [SKIP unreadable] {path.name}[{entry_index}]: {exc}")
                continue
            fingerprint = recording_fingerprint(recording)
            stim_amplitude_ma = recording.stim_amplitude_ma
            del recording
            grouped.setdefault(fingerprint, []).append((path, entry_index, stim_amplitude_ma))
    excluded: dict[tuple[Path, int], str] = {}
    for group in grouped.values():
        if len(group) < 2:
            continue
        ranked = sorted(
            group,
            key=lambda item: (
                not np.isclose(filename_stim_amplitude_ma(item[0]), item[2], equal_nan=False),
                item[0].name,
                item[1],
            ),
        )
        retained_path, retained_index, _ = ranked[0]
        for path, entry_index, _ in ranked[1:]:
            excluded[(path, entry_index)] = (
                f"Exact duplicate recording; retained {retained_path.stem}[{retained_index}] "
                "based on embedded stimulation metadata."
            )
    return excluded


def _atlas_path(atlas_dir: Path, subject: str, montage: str) -> Path | None:
    number = subject_number(subject)
    if number is None:
        return None
    hits = sorted(atlas_dir.glob(f"*Subject{number:03d}_{montage}_Coordinates_and_Atlas.csv"))
    return hits[0] if hits else None


def _unipolar_path(atlas_dir: Path, subject: str) -> Path | None:
    return _atlas_path(atlas_dir, subject, "unipolar_filtered")


def _component_region_qc(nodes: pd.DataFrame, atlas_dir: Path, subject: str, primary_atlas: str) -> pd.DataFrame:
    out = nodes.copy()
    unipolar_path = _unipolar_path(atlas_dir, subject)
    if unipolar_path is None:
        out["constituent_region_families"] = ""
        out["region_unambiguous_flag"] = False
        return out
    uni = load_brainstorm_atlas_csv(unipolar_path, primary_atlas)
    uni["contact_norm"] = uni["channel"].map(normalize_contact)
    uni["region_family"] = uni["primary_region"].map(atlas_region_family)
    family_map = uni.set_index("contact_norm")["region_family"].to_dict()
    labels: list[str] = []
    flags: list[bool] = []
    for row in out.itertuples():
        families = [family_map.get(row.contact_a_norm), family_map.get(row.contact_b_norm)]
        meaningful = {x for x in families if x and x != "non_gray"}
        labels.append(";".join([x or "missing" for x in families]))
        flags.append(len(meaningful) == 1 and row.region_family in meaningful)
    out["constituent_region_families"] = labels
    out["region_unambiguous_flag"] = flags
    return out


def make_bipolar_run(path: str | Path, cfg: dict, entry_index: int = 0) -> BipolarRun:
    """Create atlas-defined adjacent bipolar channels from unipolar recordings."""
    source = load_recording(path, entry_index=entry_index)
    minimum = int(cfg["preprocessing"].get("min_channels_per_file", 5))
    if source.signal.shape[1] < minimum:
        return BipolarRun(source, pd.DataFrame(), source, "excluded_few_channels", f"n_channels < {minimum}")
    atlas_dir = Path(cfg["project"]["atlas_dir"])
    montage = str(cfg["design"].get("montage", "bipolar2"))
    atlas_path = _atlas_path(atlas_dir, source.subject, montage)
    if atlas_path is None:
        return BipolarRun(source, pd.DataFrame(), source, "missing_bipolar_atlas", "No matching bipolar atlas CSV")
    atlas = load_brainstorm_atlas_csv(atlas_path, str(cfg["design"].get("primary_atlas", "AAL")))
    atlas["region_family"] = atlas["primary_region"].map(atlas_region_family)
    raw_lookup = {normalize_contact(name): index for index, name in enumerate(source.channel_names)}
    stim_contacts = {normalize_contact(name) for name in source.stimulated_channel_names}
    signals: list[np.ndarray] = []
    rows: list[dict] = []
    for atlas_row in atlas.itertuples():
        parts = split_bipolar_node(atlas_row.channel)
        if parts is None or parts[0] not in raw_lookup or parts[1] not in raw_lookup:
            continue
        i, j = raw_lookup[parts[0]], raw_lookup[parts[1]]
        signals.append(source.signal[:, i] - source.signal[:, j])
        rows.append(
            {
                "channel": atlas_row.channel,
                "channel_norm": normalize_contact(atlas_row.channel),
                "contact_a_norm": parts[0],
                "contact_b_norm": parts[1],
                "source_index_a": i,
                "source_index_b": j,
                "is_stimulated_channel": set(parts) == stim_contacts,
                "shares_stim_contact": bool(set(parts) & stim_contacts),
                "primary_region": atlas_row.primary_region,
                "region_family": atlas_row.region_family,
                "analysis_gray_matter_flag": bool(atlas_row.analysis_gray_matter_flag),
                "x_mni": atlas_row.x_mni,
                "y_mni": atlas_row.y_mni,
                "z_mni": atlas_row.z_mni,
                "atlas_csv": str(atlas_path),
            }
        )
    nodes = pd.DataFrame(rows)
    if nodes.empty:
        return BipolarRun(source, nodes, source, "no_matched_bipolar_nodes", "No atlas nodes matched recorded contacts")
    nodes = _component_region_qc(nodes, atlas_dir, source.subject, str(cfg["design"].get("primary_atlas", "AAL")))
    if int(nodes["is_stimulated_channel"].sum()) != 1:
        return BipolarRun(source, nodes, source, "missing_or_ambiguous_seed", "Expected exactly one bipolar stimulation seed")
    seed_contacts = set(nodes.loc[nodes["is_stimulated_channel"], ["contact_a_norm", "contact_b_norm"]].iloc[0])
    nodes["overlaps_seed"] = nodes.apply(
        lambda row: bool({row["contact_a_norm"], row["contact_b_norm"]} & seed_contacts)
        and not bool(row["is_stimulated_channel"]),
        axis=1,
    )
    nodes["primary_region_include"] = (
        nodes["analysis_gray_matter_flag"].astype(bool)
        & nodes["region_unambiguous_flag"].astype(bool)
        & ~nodes["overlaps_seed"].astype(bool)
    )
    bipolar = SeegRecording(
        path=source.path,
        entry_index=source.entry_index,
        file_stem=source.file_stem,
        signal=np.column_stack(signals),
        time=source.time,
        fs=source.fs,
        channel_names=nodes["channel"].tolist(),
        stimulated_channel_names=nodes.loc[nodes["is_stimulated_channel"], "channel"].tolist(),
        stim_frequency_hz=source.stim_frequency_hz,
        stim_amplitude_ma=source.stim_amplitude_ma,
        subject=source.subject,
    )
    return BipolarRun(bipolar, nodes, source, "ok")


def analysis_windows(rec: SeegRecording, cfg: dict, early: bool = False) -> tuple[tuple[float, float], tuple[float, float]]:
    settings = cfg["windows"]
    if early:
        settings = {**settings, **settings["early_sensitivity"]}
    pre = tuple(float(x) for x in settings["pre"])
    post = post_window_for_frequency(
        rec.stim_frequency_hz,
        settings["stim_end_by_frequency"],
        float(settings["post_duration_sec"]),
        float(settings["post_artifact_buffer_sec"]),
    )
    return pre, post


def _decompose_window(rec: SeegRecording, mask: np.ndarray, cfg: dict) -> list[NodeWindow]:
    outputs: list[NodeWindow] = []
    for index in range(rec.signal.shape[1]):
        processed, drift, noise = preprocess_segment(
            rec.signal[mask, index], rec.time[mask], rec.fs, cfg["preprocessing"]
        )
        outputs.append(NodeWindow(processed, analyze_segment(processed, rec.fs, cfg["emd"]), drift.flagged, noise.flagged))
    return outputs


def _selected_imf(result: SegmentResult, imf_number: int) -> np.ndarray | None:
    index = int(imf_number) - 1
    if result.imf is None or result.imf.ndim != 2 or result.imf.shape[1] <= index:
        return None
    return np.asarray(result.imf[:, index], dtype=float)


def _metric_mean(result: SegmentResult, metric: str) -> float:
    if result.cycle_table is None or metric not in result.cycle_table:
        return np.nan
    return float(pd.to_numeric(result.cycle_table[metric], errors="coerce").mean())


def _profile_rows(run: BipolarRun, pre: list[NodeWindow], post: list[NodeWindow]) -> list[dict]:
    rows: list[dict] = []
    for index, node in run.nodes.iterrows():
        if pre[index].result.phase_aligned_if is None or post[index].result.phase_aligned_if is None:
            continue
        include = (
            bool(node["primary_region_include"])
            and not pre[index].noise_flag
            and not post[index].noise_flag
            and pre[index].result.status == "ok"
            and post[index].result.status == "ok"
        )
        for epoch, result in [("pre", pre[index].result), ("post", post[index].result)]:
            if_prof = np.nanmean(result.phase_aligned_if, axis=1)
            wave_prof = np.nanmean(result.normalized_waveform, axis=1)
            for phase_index, (if_value, wave_value) in enumerate(zip(if_prof, wave_prof)):
                rows.append(
                    {
                        "file": run.recording.file_stem,
                        "subject": run.recording.subject,
                        "stim_frequency_hz": run.recording.stim_frequency_hz,
                        "channel": node["channel"],
                        "epoch": epoch,
                        "phase_index": phase_index,
                        "phase_rad": 2 * np.pi * phase_index / len(if_prof),
                        "mean_if_hz": if_value,
                        "normalized_waveform": wave_value,
                        "is_stimulated_channel": node["is_stimulated_channel"],
                        "primary_region_include": node["primary_region_include"],
                        "analysis_include": include,
                    }
                )
    return rows


def waveform_rows(run: BipolarRun, pre: list[NodeWindow], post: list[NodeWindow], cfg: dict | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    rec = run.recording
    for index, node in run.nodes.iterrows():
        p, q = pre[index], post[index]
        noise = bool(p.noise_flag or q.noise_flag)
        status = "ok" if p.result.status == "ok" and q.result.status == "ok" and not noise else "excluded_qc"
        baseline_rms = float(np.sqrt(np.mean(np.asarray(p.processed, dtype=float) ** 2))) if p.processed.size else np.nan
        response_class = classify_post_stim_response(q.processed, rec.fs, baseline_rms, cfg)
        row = {
            "file": rec.file_stem,
            "entry_index": rec.entry_index,
            "stimulated_channel_names": "-".join(rec.stimulated_channel_names),
            "subject": rec.subject,
            "stim_frequency_hz": rec.stim_frequency_hz,
            "stim_amplitude_ma": rec.stim_amplitude_ma,
            **node.to_dict(),
            "pre_noise_flag": p.noise_flag,
            "post_noise_flag": q.noise_flag,
            "pre_drift_flag": p.drift_flag,
            "post_drift_flag": q.drift_flag,
            "pre_status": p.result.status,
            "post_status": q.result.status,
            "pre_n_cycles": p.result.n_cycles,
            "post_n_cycles": q.result.n_cycles,
            "post_stim_response_class": response_class,
            "status": status,
        }
        for metric in ["mean_if", "asc2desc", "peak2trough"]:
            row[f"pre_{metric}"] = _metric_mean(p.result, metric)
            row[f"post_{metric}"] = _metric_mean(q.result, metric)
            row[f"delta_{metric}"] = row[f"post_{metric}"] - row[f"pre_{metric}"]
        row["analysis_include"] = status == "ok" and bool(node["primary_region_include"])
        rows.append(row)
    return pd.DataFrame(rows)


def _band_power(freqs: np.ndarray, pxx: np.ndarray, band: list[float]) -> float:
    mask = (freqs >= float(band[0])) & (freqs < float(band[1]))
    return float(np.trapezoid(pxx[mask], freqs[mask])) if mask.sum() >= 2 else np.nan


def spectral_rows(run: BipolarRun, pre: list[NodeWindow], post: list[NodeWindow], cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    spec_cfg = cfg["spectral"]
    for index, node in run.nodes.iterrows():
        row = {
            "file": run.recording.file_stem,
            "entry_index": run.recording.entry_index,
            "stimulated_channel_names": "-".join(run.recording.stimulated_channel_names),
            "subject": run.recording.subject,
            "stim_frequency_hz": run.recording.stim_frequency_hz,
            "stim_amplitude_ma": run.recording.stim_amplitude_ma,
            **node.to_dict(),
            "pre_noise_flag": pre[index].noise_flag,
            "post_noise_flag": post[index].noise_flag,
        }
        total: dict[str, float] = {}
        for epoch, values in [("pre", pre[index].processed), ("post", post[index].processed)]:
            nperseg = min(len(values), int(round(float(spec_cfg["welch_nperseg_sec"]) * run.recording.fs)))
            freqs, pxx = signal.welch(values, run.recording.fs, nperseg=nperseg, noverlap=nperseg // 2)
            total[epoch] = _band_power(freqs, pxx, spec_cfg["total_power_band"])
            for band, limits in spec_cfg["frequency_bands"].items():
                power = _band_power(freqs, pxx, limits)
                row[f"{epoch}_{band}_log_power"] = np.log10(power) if power > 0 else np.nan
                row[f"{epoch}_{band}_relative_power"] = power / total[epoch] if total[epoch] > 0 else np.nan
        for band in spec_cfg["frequency_bands"]:
            for metric in ["log_power", "relative_power"]:
                row[f"delta_{band}_{metric}"] = row[f"post_{band}_{metric}"] - row[f"pre_{band}_{metric}"]
        row["analysis_include"] = (
            bool(node["primary_region_include"]) and not pre[index].noise_flag and not post[index].noise_flag
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _mne_values(data: np.ndarray, pairs: list[tuple[int, int]], fs: float, cfg: dict) -> dict[str, np.ndarray]:
    from mne_connectivity import spectral_connectivity_epochs

    if not pairs:
        return {"imcoh": np.array([]), "coh": np.array([])}
    rows, cols = zip(*pairs)
    low, high = cfg["connectivity"]["primary_frequency_band_hz"]
    outputs = spectral_connectivity_epochs(
        data,
        method=list(cfg["connectivity"]["metrics"]),
        indices=(np.asarray(rows), np.asarray(cols)),
        mode=str(cfg["connectivity"]["method"]),
        sfreq=float(fs),
        fmin=float(low),
        fmax=float(high),
        faverage=bool(cfg["connectivity"].get("faverage", True)),
        mt_adaptive=bool(cfg["connectivity"].get("mt_adaptive", False)),
        n_jobs=int(cfg["connectivity"].get("n_jobs", 1)),
        verbose=False,
    )
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return {
        method: np.asarray(output.get_data()).reshape(len(pairs), -1).mean(axis=1)
        for method, output in zip(cfg["connectivity"]["metrics"], outputs)
    }


def summarize_imcoh_change(pre_signed: float, post_signed: float) -> dict[str, float]:
    """Keep signed imaginary coherence separate from the magnitude endpoint."""
    return {
        "pre_imcoh_signed": float(pre_signed),
        "post_imcoh_signed": float(post_signed),
        "delta_imcoh_signed": float(post_signed - pre_signed),
        "pre_abs_imcoh": float(abs(pre_signed)),
        "post_abs_imcoh": float(abs(post_signed)),
        "delta_abs_imcoh": float(abs(post_signed) - abs(pre_signed)),
    }


def _constituent_contacts(node: pd.Series) -> set[str]:
    """Return normalized unipolar contacts that form one adjacent bipolar node."""
    return {str(node["contact_a_norm"]), str(node["contact_b_norm"])}


def _share_constituent_contact(left_node: pd.Series, right_node: pd.Series) -> bool:
    """Identify bipolar edge pairs that are not independent contact pairs."""
    return bool(_constituent_contacts(left_node) & _constituent_contacts(right_node))


def _segmented_imfs(
    windows: list[NodeWindow], valid: list[int], rec: SeegRecording, cfg: dict, imf_number: int
) -> np.ndarray:
    signals = np.column_stack([_selected_imf(windows[index].result, imf_number) for index in valid])
    samples = int(round(float(cfg["connectivity"]["segment_length_sec"]) * rec.fs))
    count = min(int(cfg["connectivity"]["n_segments"]), signals.shape[0] // samples)
    if count < 2:
        raise ValueError("Connectivity requires at least two complete analysis segments.")
    return np.stack([signals[start * samples : (start + 1) * samples].T for start in range(count)], axis=0)


def connectivity_rows(run: BipolarRun, pre: list[NodeWindow], post: list[NodeWindow], cfg: dict) -> pd.DataFrame:
    nodes = run.nodes
    rows: list[dict] = []
    imf_numbers = [int(value) for value in cfg["design"].get("connectivity_imfs", [cfg["design"]["primary_imf"]])]
    primary_imf = int(cfg["design"]["primary_imf"])
    for imf_number in imf_numbers:
        valid = [
            index
            for index in range(len(nodes))
            if not pre[index].noise_flag
            and not post[index].noise_flag
            and _selected_imf(pre[index].result, imf_number) is not None
            and _selected_imf(post[index].result, imf_number) is not None
        ]
        seed_hits = [index for index in valid if bool(nodes.iloc[index]["is_stimulated_channel"])]
        if len(seed_hits) != 1:
            continue
        seed = seed_hits[0]
        targets = [index for index in valid if index != seed and not bool(nodes.iloc[index]["overlaps_seed"])]
        remotes = [index for index in targets if not bool(nodes.iloc[index]["is_stimulated_channel"])]
        edge_pairs = [(seed, target, "seed_to_target") for target in targets]
        edge_pairs += [
            (left, right, "remote_to_remote")
            for left, right in combinations(remotes, 2)
            if not _share_constituent_contact(nodes.iloc[left], nodes.iloc[right])
        ]
        if not edge_pairs:
            continue
        local_index = {node_index: i for i, node_index in enumerate(valid)}
        pairs = [(local_index[left], local_index[right]) for left, right, _ in edge_pairs]
        pre_vals = _mne_values(_segmented_imfs(pre, valid, run.recording, cfg, imf_number), pairs, run.recording.fs, cfg)
        post_vals = _mne_values(_segmented_imfs(post, valid, run.recording, cfg, imf_number), pairs, run.recording.fs, cfg)
        samples = int(round(float(cfg["connectivity"]["segment_length_sec"]) * run.recording.fs))
        n_segments_used = min(int(cfg["connectivity"]["n_segments"]), len(pre[valid[0]].processed) // samples)
        seed_xyz = nodes.loc[seed, ["x_mni", "y_mni", "z_mni"]].to_numpy(float)
        for edge_index, (left, right, edge_type) in enumerate(edge_pairs):
            left_node, right_node = nodes.iloc[left], nodes.iloc[right]
            left_xyz = left_node[["x_mni", "y_mni", "z_mni"]].to_numpy(float)
            right_xyz = right_node[["x_mni", "y_mni", "z_mni"]].to_numpy(float)
            pre_signed = float(pre_vals["imcoh"][edge_index])
            post_signed = float(post_vals["imcoh"][edge_index])
            analysis_include = (
                bool(right_node["primary_region_include"])
                if edge_type == "seed_to_target"
                else bool(left_node["primary_region_include"] and right_node["primary_region_include"])
            )
            row = {
                "file": run.recording.file_stem,
                "entry_index": run.recording.entry_index,
                "stimulated_channel_names": "-".join(run.recording.stimulated_channel_names),
                "subject": run.recording.subject,
                "stim_frequency_hz": run.recording.stim_frequency_hz,
                "stim_amplitude_ma": run.recording.stim_amplitude_ma,
                "imf": imf_number,
                "imf_role": "primary" if imf_number == primary_imf else ("secondary" if imf_number <= 3 else "exploratory"),
                "edge_type": edge_type,
                "n_segments_used": n_segments_used,
                "seed_channel": nodes.loc[seed, "channel"],
                "node_a": left_node["channel"],
                "node_b": right_node["channel"],
                "target_channel": right_node["channel"] if edge_type == "seed_to_target" else "",
                "node_a_contact_a": left_node["contact_a_norm"],
                "node_a_contact_b": left_node["contact_b_norm"],
                "node_b_contact_a": right_node["contact_a_norm"],
                "node_b_contact_b": right_node["contact_b_norm"],
                "shared_constituent_contact": _share_constituent_contact(left_node, right_node),
                "node_a_region_family": left_node["region_family"],
                "node_b_region_family": right_node["region_family"],
                "node_a_primary_include": bool(left_node["primary_region_include"]),
                "node_b_primary_include": bool(right_node["primary_region_include"]),
                "distance_mm": float(np.linalg.norm(left_xyz - right_xyz)),
                "node_a_distance_to_seed_mm": float(np.linalg.norm(left_xyz - seed_xyz)),
                "node_b_distance_to_seed_mm": float(np.linalg.norm(right_xyz - seed_xyz)),
                "edge_distance_to_seed_mm": float(min(np.linalg.norm(left_xyz - seed_xyz), np.linalg.norm(right_xyz - seed_xyz))),
                "pre_coh": float(pre_vals["coh"][edge_index]),
                "post_coh": float(post_vals["coh"][edge_index]),
                "delta_coh": float(post_vals["coh"][edge_index] - pre_vals["coh"][edge_index]),
                "analysis_include": analysis_include,
            }
            row.update(summarize_imcoh_change(pre_signed, post_signed))
            rows.append(row)
    return pd.DataFrame(rows)


def _exact_signflip(values: pd.Series) -> float:
    x = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if x.size == 0:
        return np.nan
    observed = abs(float(x.mean()))
    means = [abs(float(np.mean(x * np.asarray(signs)))) for signs in product([-1.0, 1.0], repeat=x.size)]
    return float(np.mean(np.asarray(means) >= observed - np.finfo(float).eps))


def _holm(p_values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p_values.index)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return out
    adjusted = np.maximum.accumulate([(len(valid) - i) * value for i, value in enumerate(valid)])
    out.loc[valid.index] = np.minimum(adjusted, 1.0)
    return out


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction for prespecified exploratory families."""
    out = pd.Series(np.nan, index=p_values.index)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return out
    n_tests = len(valid)
    raw = np.asarray([value * n_tests / rank for rank, value in enumerate(valid, start=1)], dtype=float)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    out.loc[valid.index] = np.minimum(adjusted, 1.0)
    return out


def paired_protocol_contrasts(subject: pd.DataFrame) -> pd.DataFrame:
    """Estimate paired 50 Hz minus 1 Hz contrasts without treating runs as independent."""
    rows: list[dict] = []
    for endpoint in ["delta_mean_if", "delta_abs_imcoh"]:
        pivot = subject.pivot(index="subject", columns="stim_frequency_hz", values=endpoint)
        if 1.0 not in pivot.columns or 50.0 not in pivot.columns:
            differences = pd.Series(dtype=float)
        else:
            differences = (pivot[50.0] - pivot[1.0]).dropna()
        rows.append(
            {
                "endpoint": endpoint,
                "contrast": "50Hz_minus_1Hz",
                "n_paired_subjects": int(len(differences)),
                "mean_difference": float(differences.mean()) if len(differences) else np.nan,
                "median_difference": float(differences.median()) if len(differences) else np.nan,
                "exact_signflip_p": _exact_signflip(differences),
            }
        )
    out = pd.DataFrame(rows)
    out["holm_p"] = _holm(out["exact_signflip_p"])
    return out


def _secondary_subject_summaries(tables: dict[str, tuple[pd.DataFrame, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate key secondary connectivity outcomes without edge-level inference."""
    subject_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for analysis, (run_table, metric) in tables.items():
        if run_table.empty:
            continue
        subject = (
            run_table.groupby(["subject", "stim_frequency_hz"], as_index=False)
            .agg(value=(metric, "mean"), n_runs=(metric, "count"))
            .dropna(subset=["value"])
        )
        subject["analysis"] = analysis
        subject_frames.append(subject[["analysis", "subject", "stim_frequency_hz", "value", "n_runs"]])
        for freq, values in subject.groupby("stim_frequency_hz")["value"]:
            summary_rows.append(
                {
                    "analysis": analysis,
                    "comparison": f"{freq:g} Hz post minus pre",
                    "n_subjects": int(values.size),
                    "mean_value": float(values.mean()),
                    "median_value": float(values.median()),
                    "exact_signflip_p_descriptive": _exact_signflip(values),
                }
            )
        paired = subject.pivot(index="subject", columns="stim_frequency_hz", values="value")
        differences = (
            (paired[50.0] - paired[1.0]).dropna()
            if 1.0 in paired.columns and 50.0 in paired.columns
            else pd.Series(dtype=float)
        )
        summary_rows.append(
            {
                "analysis": analysis,
                "comparison": "50 Hz minus 1 Hz paired contrast",
                "n_subjects": int(differences.size),
                "mean_value": float(differences.mean()) if differences.size else np.nan,
                "median_value": float(differences.median()) if differences.size else np.nan,
                "exact_signflip_p_descriptive": _exact_signflip(differences),
            }
        )
    subject_out = pd.concat(subject_frames, ignore_index=True) if subject_frames else pd.DataFrame()
    return subject_out, pd.DataFrame(summary_rows)


def _nonstimulated_waveform_sensitivity(waveform: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe the distributed waveform endpoint after omitting local seed nodes."""
    clean = waveform.loc[
        waveform["analysis_include"].fillna(False).astype(bool)
        & ~waveform["is_stimulated_channel"].fillna(False).astype(bool)
    ].copy()
    runs = clean.groupby(["file", "entry_index", "subject", "stim_frequency_hz"], as_index=False).agg(
        delta_mean_if=("delta_mean_if", "mean"), n_nodes=("channel", "nunique")
    )
    subject = runs.groupby(["subject", "stim_frequency_hz"], as_index=False).agg(
        delta_mean_if=("delta_mean_if", "mean"), n_runs=("file", "size")
    )
    rows: list[dict] = []
    for freq, values in subject.groupby("stim_frequency_hz")["delta_mean_if"]:
        rows.append(
            {
                "comparison": f"{freq:g} Hz post minus pre",
                "n_subjects": int(values.size),
                "mean_delta_mean_if_hz": float(values.mean()),
                "median_delta_mean_if_hz": float(values.median()),
                "exact_signflip_p_descriptive": _exact_signflip(values),
            }
        )
    across_protocols = subject.groupby("subject", as_index=False)["delta_mean_if"].mean()["delta_mean_if"]
    rows.append(
        {
            "comparison": "Across-protocol subject average post minus pre",
            "n_subjects": int(across_protocols.size),
            "mean_delta_mean_if_hz": float(across_protocols.mean()),
            "median_delta_mean_if_hz": float(across_protocols.median()),
            "exact_signflip_p_descriptive": _exact_signflip(across_protocols),
        }
    )
    paired = subject.pivot(index="subject", columns="stim_frequency_hz", values="delta_mean_if")
    differences = (
        (paired[50.0] - paired[1.0]).dropna()
        if 1.0 in paired.columns and 50.0 in paired.columns
        else pd.Series(dtype=float)
    )
    rows.append(
        {
            "comparison": "50 Hz minus 1 Hz paired contrast",
            "n_subjects": int(differences.size),
            "mean_delta_mean_if_hz": float(differences.mean()) if differences.size else np.nan,
            "median_delta_mean_if_hz": float(differences.median()) if differences.size else np.nan,
            "exact_signflip_p_descriptive": _exact_signflip(differences),
        }
    )
    return subject, pd.DataFrame(rows)


def _spectral_subject_summaries(sp_runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize supportive spectral changes with subjects, not runs, as the unit."""
    metrics = [f"delta_{band}_log_power" for band in BANDS]
    if sp_runs.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    subject = (
        sp_runs.groupby(["subject", "stim_frequency_hz"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in metrics}, n_runs=("file", "size"))
    )
    protocol_rows: list[dict] = []
    for band in BANDS:
        metric = f"delta_{band}_log_power"
        for freq, values in subject.groupby("stim_frequency_hz")[metric]:
            clean = values.replace([np.inf, -np.inf], np.nan).dropna()
            protocol_rows.append(
                {
                    "band": band,
                    "band_label": BAND_LABELS[band],
                    "stim_frequency_hz": freq,
                    "n_subjects": int(clean.size),
                    "mean_delta_log_power": float(clean.mean()) if clean.size else np.nan,
                    "median_delta_log_power": float(clean.median()) if clean.size else np.nan,
                    "sem_delta_log_power": float(clean.sem()) if clean.size > 1 else np.nan,
                    "exact_signflip_p_descriptive": _exact_signflip(clean),
                }
            )
    contrast_rows: list[dict] = []
    for band in BANDS:
        metric = f"delta_{band}_log_power"
        paired = subject.pivot(index="subject", columns="stim_frequency_hz", values=metric)
        differences = (
            (paired[50.0] - paired[1.0]).dropna()
            if 1.0 in paired.columns and 50.0 in paired.columns
            else pd.Series(dtype=float)
        )
        contrast_rows.append(
            {
                "band": band,
                "band_label": BAND_LABELS[band],
                "contrast": "50Hz_minus_1Hz",
                "n_paired_subjects": int(differences.size),
                "mean_difference_delta_log_power": float(differences.mean()) if differences.size else np.nan,
                "median_difference_delta_log_power": float(differences.median()) if differences.size else np.nan,
                "exact_signflip_p": _exact_signflip(differences),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["bh_fdr_p"] = _bh_fdr(contrasts["exact_signflip_p"])
    return subject, pd.DataFrame(protocol_rows), contrasts


def _imf_connectivity_summaries(connectivity: pd.DataFrame, primary_imf: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize IMF1-IMF6 connectivity while preserving IMF4 as the primary endpoint."""
    seed = connectivity.loc[
        connectivity["edge_type"].eq("seed_to_target") & connectivity["analysis_include"].fillna(False).astype(bool)
    ].copy()
    if seed.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    run = (
        seed.groupby(["file", "entry_index", "subject", "stim_frequency_hz", "stim_amplitude_ma", "imf", "imf_role"], as_index=False)
        .agg(
            delta_abs_imcoh=("delta_abs_imcoh", "mean"),
            delta_imcoh_signed=("delta_imcoh_signed", "mean"),
            delta_coh=("delta_coh", "mean"),
            n_edges=("target_channel", "nunique"),
        )
    )
    subject = (
        run.groupby(["subject", "stim_frequency_hz", "imf", "imf_role"], as_index=False)
        .agg(
            delta_abs_imcoh=("delta_abs_imcoh", "mean"),
            delta_imcoh_signed=("delta_imcoh_signed", "mean"),
            delta_coh=("delta_coh", "mean"),
            n_runs=("file", "size"),
            n_edges=("n_edges", "sum"),
        )
    )
    rows: list[dict] = []
    for imf_number, imf_group in subject.groupby("imf"):
        role = "primary" if int(imf_number) == primary_imf else ("secondary" if int(imf_number) <= 3 else "exploratory")
        for freq, values in imf_group.groupby("stim_frequency_hz")["delta_abs_imcoh"]:
            clean = values.replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "imf": int(imf_number),
                    "imf_role": role,
                    "comparison": f"{freq:g} Hz post minus pre",
                    "n_subjects": int(clean.size),
                    "mean_delta_abs_imcoh": float(clean.mean()) if clean.size else np.nan,
                    "median_delta_abs_imcoh": float(clean.median()) if clean.size else np.nan,
                    "exact_signflip_p_descriptive": _exact_signflip(clean),
                }
            )
        paired = imf_group.pivot(index="subject", columns="stim_frequency_hz", values="delta_abs_imcoh")
        differences = (
            (paired[50.0] - paired[1.0]).dropna()
            if 1.0 in paired.columns and 50.0 in paired.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "imf": int(imf_number),
                "imf_role": role,
                "comparison": "50 Hz minus 1 Hz paired contrast",
                "n_subjects": int(differences.size),
                "mean_delta_abs_imcoh": float(differences.mean()) if differences.size else np.nan,
                "median_delta_abs_imcoh": float(differences.median()) if differences.size else np.nan,
                "exact_signflip_p_descriptive": _exact_signflip(differences),
            }
        )
    summary = pd.DataFrame(rows)
    paired_mask = summary["comparison"].eq("50 Hz minus 1 Hz paired contrast")
    summary["bh_fdr_p_within_imf_family"] = np.nan
    summary.loc[paired_mask, "bh_fdr_p_within_imf_family"] = _bh_fdr(
        summary.loc[paired_mask, "exact_signflip_p_descriptive"]
    )
    return run, subject, summary


def _distance_bin_subject_summary(edges: pd.DataFrame, distance_column: str) -> pd.DataFrame:
    """Create subject-balanced descriptive profiles across prespecified distance bins."""
    if edges.empty:
        return pd.DataFrame()
    clean = edges.copy()
    clean["distance_bin_mm"] = pd.cut(
        clean[distance_column],
        bins=DISTANCE_BIN_EDGES,
        labels=DISTANCE_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    runs = (
        clean.dropna(subset=["distance_bin_mm"])
        .groupby(["file", "entry_index", "subject", "stim_frequency_hz", "distance_bin_mm"], observed=True, as_index=False)
        .agg(delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_edges=("delta_abs_imcoh", "size"))
    )
    return (
        runs.groupby(["subject", "stim_frequency_hz", "distance_bin_mm"], observed=True, as_index=False)
        .agg(delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_runs=("file", "size"), n_edges=("n_edges", "sum"))
    )


def summarize_outputs(waveform: pd.DataFrame, connectivity: pd.DataFrame, spectral: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary_imf = int(connectivity["imf"].dropna().mode().iloc[0]) if "imf" in connectivity and not connectivity["imf"].dropna().empty else 4
    if "imf_role" in connectivity and connectivity["imf_role"].eq("primary").any():
        primary_imf = int(connectivity.loc[connectivity["imf_role"].eq("primary"), "imf"].dropna().iloc[0])
    wf = waveform.loc[waveform["analysis_include"]].copy()
    wf_runs = (
        wf.groupby(["file", "entry_index", "subject", "stim_frequency_hz", "stim_amplitude_ma"], as_index=False)
        .agg(delta_mean_if=("delta_mean_if", "mean"), delta_asc2desc=("delta_asc2desc", "mean"), delta_peak2trough=("delta_peak2trough", "mean"), n_nodes=("channel", "nunique"))
    )
    primary_connectivity = connectivity.loc[connectivity["imf"].eq(primary_imf)].copy() if "imf" in connectivity else connectivity
    imf_seed_runs, imf_subject, imf_summary = _imf_connectivity_summaries(connectivity, primary_imf)
    seed_edges = primary_connectivity.loc[
        primary_connectivity["edge_type"].eq("seed_to_target") & primary_connectivity["analysis_include"]
    ].copy()
    seed_runs = (
        seed_edges.groupby(["file", "entry_index", "subject", "stim_frequency_hz", "stim_amplitude_ma"], as_index=False)
        .agg(delta_abs_imcoh=("delta_abs_imcoh", "mean"), delta_imcoh_signed=("delta_imcoh_signed", "mean"), delta_coh=("delta_coh", "mean"), n_edges=("target_channel", "nunique"))
    )
    remote_edges = primary_connectivity.loc[
        primary_connectivity["edge_type"].eq("remote_to_remote") & primary_connectivity["analysis_include"]
    ].copy()
    remote_runs = (
        remote_edges.groupby(["file", "entry_index", "subject", "stim_frequency_hz", "stim_amplitude_ma"], as_index=False)
        .agg(delta_abs_imcoh=("delta_abs_imcoh", "mean"), delta_imcoh_signed=("delta_imcoh_signed", "mean"), delta_coh=("delta_coh", "mean"), n_edges=("node_a", "size"))
    )
    remote_matrix = remote_edges.copy()
    if not remote_matrix.empty:
        ordered_regions = remote_matrix.apply(
            lambda row: sorted([str(row["node_a_region_family"]), str(row["node_b_region_family"])]), axis=1
        )
        remote_matrix["region_a"] = ordered_regions.map(lambda value: value[0])
        remote_matrix["region_b"] = ordered_regions.map(lambda value: value[1])
        matrix_run = (
            remote_matrix.groupby(["file", "entry_index", "subject", "stim_frequency_hz", "region_a", "region_b"], as_index=False)
            .agg(delta_abs_imcoh=("delta_abs_imcoh", "mean"))
        )
        remote_matrix_summary = (
            matrix_run.groupby(["stim_frequency_hz", "region_a", "region_b"], as_index=False)
            .agg(mean_delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_subjects=("subject", "nunique"), n_runs=("file", "size"))
        )
    else:
        remote_matrix_summary = pd.DataFrame()
    sp = spectral.loc[spectral["analysis_include"] & ~spectral["is_stimulated_channel"]].copy()
    sp_runs = (
        sp.groupby(["file", "entry_index", "subject", "stim_frequency_hz", "stim_amplitude_ma"], as_index=False)
        .agg(**{f"delta_{band}_log_power": (f"delta_{band}_log_power", "mean") for band in BANDS})
    )
    spectral_subject, spectral_protocol_summary, spectral_protocol_contrasts = _spectral_subject_summaries(sp_runs)
    subject_wf = wf_runs.groupby(["subject", "stim_frequency_hz"], as_index=False)["delta_mean_if"].mean()
    nonstim_subject_wf, nonstim_waveform_sensitivity = _nonstimulated_waveform_sensitivity(waveform)
    subject_conn = seed_runs.groupby(["subject", "stim_frequency_hz"], as_index=False)["delta_abs_imcoh"].mean()
    subject = subject_wf.merge(subject_conn, on=["subject", "stim_frequency_hz"], how="outer")
    waveform_connectivity_runs = wf_runs.merge(
        seed_runs[["file", "entry_index", "subject", "stim_frequency_hz", "delta_abs_imcoh", "n_edges"]],
        on=["file", "entry_index", "subject", "stim_frequency_hz"],
        how="inner",
        suffixes=("_waveform", "_connectivity"),
    )
    waveform_connectivity_subject = (
        waveform_connectivity_runs.groupby(["subject", "stim_frequency_hz"], as_index=False)
        .agg(
            delta_mean_if=("delta_mean_if", "mean"),
            delta_abs_imcoh=("delta_abs_imcoh", "mean"),
            n_runs=("file", "size"),
        )
    )
    endpoint_subject = subject.groupby("subject", as_index=False)[["delta_mean_if", "delta_abs_imcoh"]].mean()
    inference = pd.DataFrame(
        [
            {"endpoint": "delta_mean_if", "n_subjects": endpoint_subject["delta_mean_if"].notna().sum(), "mean_delta": endpoint_subject["delta_mean_if"].mean(), "exact_signflip_p": _exact_signflip(endpoint_subject["delta_mean_if"])},
            {"endpoint": "delta_abs_imcoh", "n_subjects": endpoint_subject["delta_abs_imcoh"].notna().sum(), "mean_delta": endpoint_subject["delta_abs_imcoh"].mean(), "exact_signflip_p": _exact_signflip(endpoint_subject["delta_abs_imcoh"])},
        ]
    )
    inference["holm_p"] = _holm(inference["exact_signflip_p"])
    protocol_contrasts = paired_protocol_contrasts(subject)
    distance_rows = []
    for keys, group in seed_edges.groupby(["file", "entry_index", "subject", "stim_frequency_hz"], dropna=False):
        value = stats.spearmanr(group["distance_mm"], group["delta_abs_imcoh"]).statistic if len(group) >= 4 else np.nan
        distance_rows.append({"file": keys[0], "entry_index": keys[1], "subject": keys[2], "stim_frequency_hz": keys[3], "n_edges": len(group), "spearman_rho": value})
    distance = pd.DataFrame(distance_rows)
    seed_distance_eligible = seed_edges.groupby(["file", "entry_index"], group_keys=False).filter(lambda group: len(group) >= 4)
    seed_distance_bins = _distance_bin_subject_summary(seed_distance_eligible, "distance_mm")
    remote_proximity_rows = []
    for keys, group in remote_edges.groupby(["file", "entry_index", "subject", "stim_frequency_hz"], dropna=False):
        value = (
            stats.spearmanr(group["edge_distance_to_seed_mm"], group["delta_abs_imcoh"]).statistic
            if len(group) >= 4
            else np.nan
        )
        remote_proximity_rows.append(
            {"file": keys[0], "entry_index": keys[1], "subject": keys[2], "stim_frequency_hz": keys[3], "n_edges": len(group), "spearman_rho": value}
        )
    remote_proximity = pd.DataFrame(remote_proximity_rows)
    remote_proximity_eligible = remote_edges.groupby(["file", "entry_index"], group_keys=False).filter(lambda group: len(group) >= 4)
    remote_proximity_bins = _distance_bin_subject_summary(remote_proximity_eligible, "edge_distance_to_seed_mm")
    target_wf = wf[["file", "entry_index", "channel", "delta_mean_if"]].rename(columns={"channel": "target_channel"})
    coupling_edges = seed_edges.merge(target_wf, on=["file", "entry_index", "target_channel"], how="left").dropna(subset=["delta_mean_if"])
    coupling_rows = []
    for keys, group in coupling_edges.groupby(["file", "entry_index", "subject", "stim_frequency_hz"], dropna=False):
        value = stats.spearmanr(group["delta_abs_imcoh"], group["delta_mean_if"]).statistic if len(group) >= 4 else np.nan
        coupling_rows.append({"file": keys[0], "entry_index": keys[1], "subject": keys[2], "stim_frequency_hz": keys[3], "n_edges": len(group), "spearman_rho": value})
    coupling = pd.DataFrame(coupling_rows)
    secondary_subject, secondary_summary = _secondary_subject_summaries(
        {
            "seed_to_target_distance_dependence": (distance, "spearman_rho"),
            "remote_network_change": (remote_runs, "delta_abs_imcoh"),
            "remote_edge_proximity_dependence": (remote_proximity, "spearman_rho"),
            "waveform_connectivity_coupling": (coupling, "spearman_rho"),
        }
    )
    current_rows = []
    for table, endpoint in [(wf_runs, "delta_mean_if"), (seed_runs, "delta_abs_imcoh"), (sp_runs, "delta_alpha_log_power")]:
        clean = table[[endpoint, "stim_amplitude_ma", "stim_frequency_hz"]].dropna()
        if len(clean) >= 5 and clean["stim_amplitude_ma"].nunique() > 1 and clean["stim_frequency_hz"].nunique() > 1:
            import statsmodels.formula.api as smf

            model = smf.ols(f"{endpoint} ~ stim_amplitude_ma + C(stim_frequency_hz)", data=clean).fit(cov_type="HC3")
            for term in ["stim_amplitude_ma", "C(stim_frequency_hz)[T.50.0]"]:
                if term in model.params:
                    current_rows.append(
                        {
                            "endpoint": endpoint,
                            "term": term,
                            "estimate": model.params[term],
                            "standard_error": model.bse[term],
                            "p_value": model.pvalues[term],
                            "n_runs": len(clean),
                            "note": "Run-level HC3 sensitivity; protocol, duration, pulse count and current remain incompletely separable.",
                        }
                    )
    return {
        "waveform_run_summary": wf_runs,
        "seed_connectivity_run_summary": seed_runs,
        "remote_network_run_summary": remote_runs,
        "remote_region_matrix_summary": remote_matrix_summary,
        "spectral_run_summary": sp_runs,
        "spectral_subject_protocol_summary": spectral_subject,
        "spectral_band_protocol_summary": spectral_protocol_summary,
        "spectral_band_protocol_contrasts": spectral_protocol_contrasts,
        "subject_protocol_summary": subject,
        "waveform_nonstimulated_subject_protocol": nonstim_subject_wf,
        "waveform_nonstimulated_sensitivity": nonstim_waveform_sensitivity,
        "coprimary_inference": inference,
        "coprimary_protocol_contrasts": protocol_contrasts,
        "distance_run_summary": distance,
        "seed_distance_bin_subject_summary": seed_distance_bins,
        "remote_proximity_run_summary": remote_proximity,
        "remote_proximity_bin_subject_summary": remote_proximity_bins,
        "waveform_connectivity_coupling": coupling,
        "waveform_connectivity_run_summary": waveform_connectivity_runs,
        "waveform_connectivity_subject_protocol_summary": waveform_connectivity_subject,
        "secondary_connectivity_subject_protocol": secondary_subject,
        "secondary_connectivity_descriptive_summary": secondary_summary,
        "imf_connectivity_seed_run_summary": imf_seed_runs,
        "imf_connectivity_subject_protocol_summary": imf_subject,
        "imf_connectivity_descriptive_summary": imf_summary,
        "current_sensitivity": pd.DataFrame(current_rows),
    }


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")


def _protocol_points(ax: plt.Axes, table: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    for freq, color in [(1, "#2A9D8F"), (50, "#D62828")]:
        values = table.loc[table["stim_frequency_hz"].eq(freq), metric].dropna()
        ax.scatter(np.repeat(freq, len(values)), values, color=color, alpha=.8)
        if not values.empty:
            ax.plot([freq - 4, freq + 4], [values.median(), values.median()], color=color, lw=1.7)
    ax.axhline(0, color="0.25", lw=.7)
    ax.set_xticks([1, 50], ["1 Hz", "50 Hz"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _protocol_n_label(table: pd.DataFrame, metric: str) -> str:
    counts = {
        freq: table.loc[table["stim_frequency_hz"].eq(freq), metric].notna().sum()
        for freq in (1, 50)
    }
    return f"1 Hz n={counts[1]}; 50 Hz n={counts[50]}"


def _distance_profile(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    if data.empty or "stim_frequency_hz" not in data.columns:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data\n(no run had >=4 eligible edges)", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    positions = np.arange(len(DISTANCE_BIN_LABELS), dtype=float)
    tick_labels: list[str] = []
    for label in DISTANCE_BIN_LABELS:
        counts = [
            int(data.loc[data["stim_frequency_hz"].eq(freq) & data["distance_bin_mm"].eq(label), "subject"].nunique())
            for freq in (1, 50)
        ]
        tick_labels.append(f"{label}\nn={counts[0]}/{counts[1]}")
    for freq, offset, color in [(1, -0.10, "#2A9D8F"), (50, 0.10, "#D62828")]:
        means: list[float] = []
        for index, label in enumerate(DISTANCE_BIN_LABELS):
            values = data.loc[
                data["stim_frequency_hz"].eq(freq) & data["distance_bin_mm"].eq(label),
                "delta_abs_imcoh",
            ].dropna()
            ax.scatter(np.repeat(positions[index] + offset, len(values)), values, s=16, color=color, alpha=.5)
            means.append(float(values.mean()) if not values.empty else np.nan)
        ax.plot(positions + offset, means, marker="o", lw=1.7, color=color, label=f"{freq:g} Hz stimulation protocol")
    ax.axhline(0, color="0.25", lw=.7)
    ax.set_xticks(positions, tick_labels)
    ax.set_xlabel("Distance to stimulated seed (mm)\n(n = 1 Hz / 50 Hz subjects)")
    ax.set_ylabel("Post - pre |imaginary coherence|")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6, loc="best")


def _imf_summary_heatmap(ax: plt.Axes, summary: pd.DataFrame, metric: str, title: str) -> plt.Axes:
    imfs = [1, 2, 3, 4, 5, 6]
    comparisons = ["1 Hz post minus pre", "50 Hz post minus pre", "50 Hz minus 1 Hz paired contrast"]
    matrix = np.full((len(comparisons), len(imfs)), np.nan)
    support = np.full_like(matrix, np.nan)
    for row in summary.itertuples():
        if int(row.imf) not in imfs or row.comparison not in comparisons:
            continue
        row_index = comparisons.index(row.comparison)
        col_index = imfs.index(int(row.imf))
        matrix[row_index, col_index] = float(getattr(row, metric))
        support[row_index, col_index] = float(row.n_subjects)
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    limit = max(float(limit), 0.01)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            if not np.isfinite(matrix[row_index, col_index]):
                continue
            slow_note = "*" if imfs[col_index] in (5, 6) else ""
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:+.3f}\nn={int(support[row_index, col_index])}{slow_note}",
                ha="center",
                va="center",
                fontsize=6,
            )
    ax.set_xticks(np.arange(len(imfs)), [f"IMF{value}" for value in imfs])
    ax.set_yticks(np.arange(len(comparisons)), ["1 Hz", "50 Hz", "50 - 1 Hz"])
    ax.set_title(title)
    ax.set_xlabel("Intrinsic mode function")
    return image


def _relationship_scatter(ax: plt.Axes, runs: pd.DataFrame, subject: pd.DataFrame) -> None:
    for freq, color in [(1, "#2A9D8F"), (50, "#D62828")]:
        run = runs.loc[runs["stim_frequency_hz"].eq(freq)]
        summary = subject.loc[subject["stim_frequency_hz"].eq(freq)]
        ax.scatter(
            run["delta_mean_if"], run["delta_abs_imcoh"], s=23, facecolor="none", edgecolor=color,
            alpha=.55, label=f"{freq:g} Hz run summary",
        )
        ax.scatter(
            summary["delta_mean_if"], summary["delta_abs_imcoh"], s=42, marker="D", color=color,
            edgecolor="white", lw=.45, label=f"{freq:g} Hz subject summary",
        )
    ax.axhline(0, color="0.55", lw=.7)
    ax.axvline(0, color="0.55", lw=.7)
    ax.set_xlabel("Post - pre IMF4 mean instantaneous frequency (Hz)")
    ax.set_ylabel("Post - pre seed-to-target |imaginary coherence|")
    ax.set_title("Waveform change in relation to connectivity change")
    ax.legend(frameon=False, fontsize=6, loc="best")


def make_figures(
    qc: pd.DataFrame,
    waveform: pd.DataFrame,
    connectivity: pd.DataFrame,
    spectral: pd.DataFrame,
    profiles: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    paths: list[Path] = []
    run_qc = qc.loc[qc["status"].eq("ok")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), constrained_layout=True)
    counts = run_qc["stim_frequency_hz"].value_counts().sort_index()
    axes[0].bar([f"{x:g} Hz" for x in counts.index], counts, color="#4C78A8")
    axes[0].set_ylabel("Eligible runs"); axes[0].set_title("Included protocol runs"); _panel(axes[0], "a")
    for freq, color in [(1, "#2A9D8F"), (50, "#D62828")]:
        sub = run_qc.loc[run_qc["stim_frequency_hz"].eq(freq)]
        axes[1].scatter([f"{freq:g} Hz"] * len(sub), sub["stim_amplitude_ma"], color=color, alpha=0.75)
    axes[1].set_ylabel("Current (mA)"); axes[1].set_title("Stimulation current"); _panel(axes[1], "b")
    axes[2].broken_barh([(-30, 28)], (1.1, .35), facecolors="#4C78A8")
    axes[2].broken_barh([(6, 28)], (.6, .35), facecolors="#D62828")
    axes[2].broken_barh([(31, 28)], (.1, .35), facecolors="#2A9D8F")
    axes[2].set_yticks([1.27, .77, .27], ["Pre", "50 Hz post", "1 Hz post"])
    axes[2].set_xlabel("Time from stimulation onset (s)"); axes[2].set_title("Matched analysis windows"); _panel(axes[2], "c")
    path = out_dir / "figure_01_overview.png"; fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig); paths.append(path)

    included_profiles = profiles.loc[profiles["analysis_include"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    colors = {(1, "pre"): "#72B7B2", (1, "post"): "#2A9D8F", (50, "pre"): "#F4A3A3", (50, "post"): "#D62828"}
    for (freq, epoch), group in included_profiles.groupby(["stim_frequency_hz", "epoch"]):
        curve = group.groupby("phase_rad")[["mean_if_hz", "normalized_waveform"]].mean().reset_index()
        axes[0].plot(curve["phase_rad"], curve["mean_if_hz"], color=colors[(freq, epoch)], label=f"{freq:g} Hz {epoch}")
        axes[1].plot(curve["phase_rad"], curve["normalized_waveform"], color=colors[(freq, epoch)], label=f"{freq:g} Hz {epoch}")
    axes[0].set_title("Phase-aligned IF"); axes[0].set_xlabel("Cycle phase (rad)"); axes[0].set_ylabel("Instantaneous frequency (Hz)"); _panel(axes[0], "a")
    axes[1].set_title("Normalized waveform"); axes[1].set_xlabel("Cycle phase (rad)"); axes[1].set_ylabel("Amplitude"); axes[1].legend(frameon=False, fontsize=6); _panel(axes[1], "b")
    subject = summaries["subject_protocol_summary"]
    for freq, color in [(1, "#2A9D8F"), (50, "#D62828")]:
        vals = subject.loc[subject["stim_frequency_hz"].eq(freq), "delta_mean_if"].dropna()
        axes[2].scatter(np.repeat(freq, len(vals)), vals, color=color, alpha=.8)
        axes[2].plot([freq - 4, freq + 4], [vals.mean(), vals.mean()], color=color)
    axes[2].axhline(0, color="0.25", lw=.7); axes[2].set_xticks([1, 50], ["1 Hz", "50 Hz"]); axes[2].set_ylabel("Post - pre mean IF (Hz)"); axes[2].set_title("Subject-level IMF4 change"); _panel(axes[2], "c")
    path = out_dir / "figure_02_quinn_waveform.png"; fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.05), constrained_layout=True)
    seed_subject = summaries["seed_connectivity_run_summary"].groupby(["subject", "stim_frequency_hz"], as_index=False)["delta_abs_imcoh"].mean()
    remote_subject = summaries["remote_network_run_summary"].groupby(["subject", "stim_frequency_hz"], as_index=False)["delta_abs_imcoh"].mean()
    _protocol_points(
        axes[0], seed_subject, "delta_abs_imcoh", "Post - pre |imaginary coherence|",
        f"Stimulated seed to target\n({_protocol_n_label(seed_subject, 'delta_abs_imcoh')})",
    )
    _distance_profile(axes[1], summaries["seed_distance_bin_subject_summary"], "Seed-target change by distance\n(descriptive subject summaries)")
    _protocol_points(
        axes[2], remote_subject, "delta_abs_imcoh", "Post - pre |imaginary coherence|",
        f"Non-stimulated recorded network\n(descriptive; {_protocol_n_label(remote_subject, 'delta_abs_imcoh')})",
    )
    _distance_profile(axes[3], summaries["remote_proximity_bin_subject_summary"], "Remote change by proximity to seed\n(descriptive subject summaries)")
    for label, ax in zip("abcd", axes):
        _panel(ax, label)
    path = out_dir / "figure_03_mne_connectivity_distance_remote.png"; fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.35), constrained_layout=True)
    _relationship_scatter(
        axes[0],
        summaries["waveform_connectivity_run_summary"],
        summaries["waveform_connectivity_subject_protocol_summary"],
    )
    _panel(axes[0], "a")
    spectral_subject = (
        summaries["spectral_run_summary"].groupby(["subject", "stim_frequency_hz"], as_index=False)
        [[f"delta_{band}_log_power" for band in BANDS]]
        .mean()
    )
    x = np.arange(len(BANDS), dtype=float)
    for freq, offset, color in [(1, -0.12, "#2A9D8F"), (50, 0.12, "#D62828")]:
        selected = spectral_subject.loc[spectral_subject["stim_frequency_hz"].eq(freq)]
        values = selected[[f"delta_{band}_log_power" for band in BANDS]].to_numpy(float)
        for row in values:
            axes[1].plot(x + offset, row, color=color, alpha=.18, lw=.7)
        axes[1].plot(
            x + offset, np.nanmean(values, axis=0), marker="o", lw=1.8, color=color,
            label=f"{freq:g} Hz stimulation protocol",
        )
    axes[1].axhline(0, color="0.25", lw=.7)
    axes[1].set_xticks(x, [BAND_LABELS[band] for band in BANDS], rotation=35, ha="right")
    axes[1].set_ylabel("Post - pre log10 band power")
    axes[1].set_title("Supportive spectral changes\n(non-stimulated recording sites)")
    axes[1].legend(frameon=False, fontsize=6)
    _panel(axes[1], "b")
    path = out_dir / "figure_04_coupling_spectral.png"; fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig); paths.append(path)

    if "imf_connectivity_descriptive_summary" in summaries and not summaries["imf_connectivity_descriptive_summary"].empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.25), constrained_layout=False)
        image = _imf_summary_heatmap(
            ax,
            summaries["imf_connectivity_descriptive_summary"],
            "mean_delta_abs_imcoh",
            "IMF1-IMF6 seed-to-target connectivity summary\n(IMF4 primary; IMF1-IMF3 secondary; IMF5-IMF6 exploratory)",
        )
        fig.colorbar(image, ax=ax, label="Post - pre |imaginary coherence|")
        fig.subplots_adjust(left=0.09, right=0.86, bottom=0.28, top=0.78)
        fig.text(
            0.09,
            0.05,
            "* IMF5-IMF6 are exploratory because 4-second multitaper segments provide limited slow-component cycle support.",
            fontsize=6,
        )
        path = out_dir / "extended_imf1_imf6_seed_connectivity_summary.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def run_analysis(
    config_path: str | Path,
    max_files: int | None = None,
    early: bool = False,
    file_path: str | Path | None = None,
    subjects: list[str] | None = None,
) -> dict[str, Path]:
    cfg = load_config(config_path)
    root = Path(cfg["project"]["output_dir"]) / ("early_10s_guarded_sensitivity" if early else "primary_28s_guarded")
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    waveform_frames: list[pd.DataFrame] = []
    spectral_frames: list[pd.DataFrame] = []
    connectivity_frames: list[pd.DataFrame] = []
    profile_rows: list[dict] = []
    qc_rows: list[dict] = []
    connectivity_qc_rows: list[dict] = []
    files = [Path(file_path)] if file_path is not None else list(iter_mat_files(cfg["project"]["data_dir"]))
    if file_path is None:
        exclude_patterns = cfg["project"].get("exclude_file_patterns", [])
        files = [p for p in files if not any(pattern in p.name for pattern in exclude_patterns)]
        if subjects is not None:
            subject_numbers = {int(s) for s in subjects}
            files = [p for p in files if subject_number(parse_subject_from_name(p)) in subject_numbers]
        if max_files is not None:
            files = files[:max_files]
    duplicate_exclusions = duplicate_recording_exclusions(files) if file_path is None else {}
    for path in files:
        try:
            n_entries = count_entries(path)
        except Exception as exc:
            qc_rows.append({"file": path.stem, "entry_index": -1, "status": "excluded_unreadable", "message": str(exc)})
            connectivity_qc_rows.append({"file": path.stem, "entry_index": -1, "status": "excluded_unreadable", "connectivity_status": "excluded_unreadable", "n_edges": 0, "n_primary_edges": 0, "message": str(exc)})
            continue
        for entry_index in range(n_entries):
            try:
                run = make_bipolar_run(path, cfg, entry_index=entry_index)
            except Exception as exc:
                qc_rows.append({"file": path.stem, "entry_index": entry_index, "status": "excluded_unreadable", "message": str(exc)})
                connectivity_qc_rows.append({"file": path.stem, "entry_index": entry_index, "status": "excluded_unreadable", "connectivity_status": "excluded_unreadable", "n_edges": 0, "n_primary_edges": 0, "message": str(exc)})
                continue
            filename_current = filename_stim_amplitude_ma(path)
            metadata_match = bool(np.isclose(filename_current, run.source_recording.stim_amplitude_ma, equal_nan=False))
            dup_key = (path, entry_index)
            status = "excluded_duplicate_recording" if dup_key in duplicate_exclusions else run.status
            message = duplicate_exclusions[dup_key] if dup_key in duplicate_exclusions else run.message
            qc = {"file": run.source_recording.file_stem, "entry_index": entry_index, "stimulated_channel_names": "-".join(run.source_recording.stimulated_channel_names), "subject": run.source_recording.subject, "stim_frequency_hz": run.source_recording.stim_frequency_hz, "stim_amplitude_ma": run.source_recording.stim_amplitude_ma, "filename_stim_amplitude_ma": filename_current, "filename_current_matches_metadata": metadata_match, "status": status, "message": message, "n_unipolar_channels": run.source_recording.signal.shape[1], "n_bipolar_nodes": len(run.nodes)}
            qc_rows.append(qc)
            if status != "ok":
                connectivity_qc_rows.append({**qc, "connectivity_status": status, "n_edges": 0, "n_primary_edges": 0})
                continue
            pre_range, post_range = analysis_windows(run.recording, cfg, early=early)
            pre = _decompose_window(run.recording, window_mask(run.recording.time, *pre_range), cfg)
            post = _decompose_window(run.recording, window_mask(run.recording.time, *post_range), cfg)
            waveform_frames.append(waveform_rows(run, pre, post, cfg))
            spectral_frames.append(spectral_rows(run, pre, post, cfg))
            connectivity = connectivity_rows(run, pre, post, cfg)
            connectivity_frames.append(connectivity)
            seed_index = int(run.nodes.index[run.nodes["is_stimulated_channel"]][0])
            primary_imf = int(cfg["design"]["primary_imf"])
            seed_imf_available = (
                _selected_imf(pre[seed_index].result, primary_imf) is not None
                and _selected_imf(post[seed_index].result, primary_imf) is not None
            )
            if pre[seed_index].noise_flag or post[seed_index].noise_flag:
                conn_status = "excluded_seed_noise"
            elif not seed_imf_available:
                conn_status = "excluded_seed_missing_imf"
            elif connectivity.empty:
                conn_status = "excluded_no_valid_edges"
            else:
                conn_status = "ok"
            connectivity_qc_rows.append(
                {
                    **qc,
                    "connectivity_status": conn_status,
                    "seed_channel": run.nodes.loc[seed_index, "channel"],
                    "seed_pre_noise_flag": pre[seed_index].noise_flag,
                    "seed_post_noise_flag": post[seed_index].noise_flag,
                    "seed_pre_emd_status": pre[seed_index].result.status,
                    "seed_post_emd_status": post[seed_index].result.status,
                    "seed_imf_available": seed_imf_available,
                    "n_edges": len(connectivity),
                    "n_primary_edges": int(
                        connectivity.loc[connectivity["imf"].eq(primary_imf), "analysis_include"].sum()
                    )
                    if not connectivity.empty
                    else 0,
                }
            )
            profile_rows.extend(_profile_rows(run, pre, post))
    qc = pd.DataFrame(qc_rows)
    waveform = pd.concat(waveform_frames, ignore_index=True) if waveform_frames else pd.DataFrame()
    spectral = pd.concat(spectral_frames, ignore_index=True) if spectral_frames else pd.DataFrame()
    connectivity = pd.concat(connectivity_frames, ignore_index=True) if connectivity_frames else pd.DataFrame()
    profiles = pd.DataFrame(profile_rows)
    outputs = {"run_qc": qc, "connectivity_qc": pd.DataFrame(connectivity_qc_rows), "waveform_nodes": waveform, "spectral_nodes": spectral, "connectivity_edges": connectivity, "quinn_profiles": profiles}
    paths: dict[str, Path] = {}
    for name, table in outputs.items():
        path = tables / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    if not waveform.empty and not connectivity.empty and not spectral.empty:
        summaries = summarize_outputs(waveform, connectivity, spectral)
        for name, table in summaries.items():
            path = tables / f"{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = path
        for index, path in enumerate(make_figures(qc, waveform, connectivity, spectral, profiles, summaries, figures, int(cfg["outputs"].get("figure_dpi", 300))), start=1):
            paths[f"figure_{index:02d}"] = path
    return paths
