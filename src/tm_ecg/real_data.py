"""Executable helpers for real PTB-XL and LUDB pipeline stages."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tm_ecg.config import ProjectConfig
from tm_ecg.constants import LEADS_12, PROJECT_LABELS
from tm_ecg.features.formulas import BeatMeasurement
from tm_ecg.io.common import read_json, write_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.modeling.classifier import build_model
from tm_ecg.modeling.triads import build_triad_memberships
from tm_ecg.signal.filtering import preprocess_signal
from tm_ecg.signal.rpeaks import detect_r_peaks
from tm_ecg.types import BeatAcceptance


def _runtime():
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import wfdb  # type: ignore
    from scipy import signal as sp_signal  # type: ignore

    return np, torch, wfdb, sp_signal


def _parse_labels(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    text = str(raw or "")
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text] if text else []


def split_entries(config: ProjectConfig, dataset: str) -> dict[str, list[dict[str, Any]]]:
    if dataset == "ptbxl":
        payload = read_json(config.paths.manifests / "split_manifest_ptbxl.json")
        grouped = {"train": [], "val": [], "test": []}
        for row in payload["split_assignments"]:
            grouped[str(row["split"])].append(row)
        return grouped

    payload = read_json(config.paths.manifests / "split_manifest_ludb_repeat_1.json")
    grouped = {"train": [], "val": [], "test": []}
    for row in payload["split_assignments"]:
        split = str(row["split"])
        prefix = "repeat_1_fold_1_"
        if split.startswith(prefix):
            grouped[split[len(prefix) :]].append(row)
    return grouped


def _resolved_dataset_root(config: ProjectConfig, dataset: str) -> Path:
    base = config.paths.raw / config.datasets[dataset].extract_dir
    children = sorted(child for child in base.iterdir() if child.is_dir())
    return children[0] if children else base


def _lead_index(sig_names: list[str], target: str) -> int:
    lookup = {name.lower(): idx for idx, name in enumerate(sig_names)}
    if target.lower() in lookup:
        return lookup[target.lower()]
    alias = target.lower().replace("av", "a")
    return lookup.get(alias, 1 if len(sig_names) > 1 else 0)


def _load_record(config: ProjectConfig, dataset: str, entry: dict[str, Any]):
    np, _torch, wfdb, _sp_signal = _runtime()
    if dataset == "ptbxl":
        root = _resolved_dataset_root(config, dataset)
        record_path = root / str(entry["source_path"])
    else:
        record_path = Path(str(entry["source_path"])).with_suffix("")
    record = wfdb.rdrecord(str(record_path))
    return np.asarray(record.p_signal, dtype=np.float32), float(record.fs), list(record.sig_name)


def _window(signal, center: int, left: int, right: int, np):
    start = center - left
    end = center + right
    width = left + right
    if start >= 0 and end <= signal.shape[0]:
        return signal[start:end]
    output = np.zeros((width, signal.shape[1]), dtype=np.float32)
    valid_start = max(0, start)
    valid_end = min(signal.shape[0], end)
    out_start = valid_start - start
    out_end = out_start + (valid_end - valid_start)
    output[out_start:out_end] = signal[valid_start:valid_end]
    return output


def _count_secondary_extrema(values, np) -> int:
    if values.size < 5:
        return 0
    derivative = np.diff(values)
    signs = np.sign(derivative)
    return int(np.sum(np.abs(np.diff(signs)) > 1))


def _lead_quality(values, peaks: list[int], np) -> float:
    if not peaks:
        return 0.0
    qrs_energy = float(np.mean(np.abs(values[peaks])))
    baseline = float(np.std(values))
    return 20.0 * math.log10((qrs_energy + 1e-6) / (baseline + 1e-6))


def _st_offset_samples(rr_s: float | None, fs: float) -> int:
    hr = 60.0 / rr_s if rr_s and rr_s > 0 else 0.0
    return int((0.06 if hr >= 100.0 else 0.08) * fs)


def _one_record_measurements(signal, fs: float, sig_names: list[str], record_id: str, config: ProjectConfig):
    np, _torch, _wfdb, sp_signal = _runtime()
    diagnostic = preprocess_signal(signal, fs, config.filters["diagnostic"])
    lead_ii = _lead_index(sig_names, "II")
    lead_i = _lead_index(sig_names, "I")
    lead_avf = _lead_index(sig_names, "aVF")
    lead_v1 = _lead_index(sig_names, "V1")
    lead_v2 = _lead_index(sig_names, "V2")
    lead_v3 = _lead_index(sig_names, "V3")
    lead_v5 = _lead_index(sig_names, "V5")

    peaks, _meta = detect_r_peaks(diagnostic[:, lead_ii], fs)
    peaks = [int(peak) for peak in peaks if int(0.25 * fs) <= int(peak) < signal.shape[0] - int(0.45 * fs)]
    if len(peaks) < 3:
        fallback = np.linspace(int(0.6 * fs), max(int(0.6 * fs), signal.shape[0] - int(0.6 * fs)), num=5, dtype=int)
        peaks = [int(value) for value in fallback]

    measurements: list[dict[str, Any]] = []
    acceptances: list[BeatAcceptance] = []
    tq_ratios: list[float] = []

    for idx, peak in enumerate(peaks):
        rr_prev = (peak - peaks[idx - 1]) / fs if idx > 0 else None
        rr_next = (peaks[idx + 1] - peak) / fs if idx < len(peaks) - 1 else None
        rr_s = rr_next if rr_next is not None else rr_prev
        qrs_on = max(0, peak - int(0.04 * fs))
        qrs_off = min(signal.shape[0] - 1, peak + int(0.06 * fs))
        p_on = max(0, peak - int(0.20 * fs))
        p_peak = max(0, peak - int(0.16 * fs))
        p_off = max(0, peak - int(0.12 * fs))
        t_on = min(signal.shape[0] - 1, peak + int(0.08 * fs))
        t_peak = min(signal.shape[0] - 1, peak + int(0.24 * fs))
        t_off = min(signal.shape[0] - 1, peak + int(0.36 * fs))

        baseline_window = diagnostic[max(0, qrs_on - int(0.06 * fs)) : max(qrs_on - int(0.02 * fs), 1)]
        baseline = np.median(baseline_window, axis=0) if baseline_window.size else np.zeros(signal.shape[1], dtype=np.float32)

        qrs_sig_ii = diagnostic[qrs_on:qrs_off, lead_ii]
        qrs_sig_i = diagnostic[qrs_on:qrs_off, lead_i]
        qrs_sig_avf = diagnostic[qrs_on:qrs_off, lead_avf]
        t_sig_v5 = diagnostic[t_on:t_off, lead_v5]
        st_idx = min(signal.shape[0] - 1, qrs_off + _st_offset_samples(rr_s, fs))

        tq_start = min(signal.shape[0] - 1, t_off + int(0.04 * fs))
        tq_end = min(
            signal.shape[0],
            (peaks[idx + 1] - int(0.04 * fs)) if idx < len(peaks) - 1 else tq_start + int(0.2 * fs),
        )
        tq_segment = diagnostic[tq_start:tq_end, lead_ii]
        if tq_segment.size >= 16:
            freqs, psd = sp_signal.welch(tq_segment, fs=fs, nperseg=min(128, tq_segment.size))
            f_mask = (freqs >= 4.0) & (freqs <= 10.0)
            all_mask = (freqs >= 0.5) & (freqs <= 20.0)
            p_f = float(np.trapezoid(psd[f_mask], freqs[f_mask])) if np.any(f_mask) else 0.0
            p_all = float(np.trapezoid(psd[all_mask], freqs[all_mask])) if np.any(all_mask) else 0.0
            tq_ratios.append(p_f / p_all if p_all > 0 else 0.0)

        qrs_dur_ms = 1000.0 * (qrs_off - qrs_on) / fs
        secondary_extrema = _count_secondary_extrema(qrs_sig_ii, np)
        qrs_def_prob = 1.0 / (1.0 + math.exp(-((qrs_dur_ms - 110.0) / 10.0 + 0.5 * secondary_extrema)))
        right_t = diagnostic[t_on:t_off, [lead_v1, lead_v2, lead_v3]]
        min_t = float(np.min(right_t)) if right_t.size else 0.0
        neg_duration_ms = float(np.sum(right_t[:, 0] < float(config.thresholds["t_inverted_threshold_mv"]))) * 1000.0 / fs if right_t.size else 0.0
        u_idx = min(signal.shape[0] - 1, t_off + int(0.08 * fs))

        beat = BeatMeasurement(
            beat_id=f"{record_id}-beat-{idx:04d}",
            rr_s=rr_s,
            rr_prev_s=rr_prev,
            rr_next_s=rr_next,
            p_present=True,
            p_amp_ii_mv=float(diagnostic[p_peak, lead_ii] - baseline[lead_ii]),
            p_dur_ms=1000.0 * (p_off - p_on) / fs,
            pr_ms=1000.0 * (qrs_on - p_on) / fs,
            q_amp_ii_mv=float(np.min(qrs_sig_ii) - baseline[lead_ii]) if qrs_sig_ii.size else 0.0,
            r_amp_ii_mv=float(diagnostic[peak, lead_ii] - baseline[lead_ii]),
            s_amp_ii_mv=float(qrs_sig_ii[-1] - baseline[lead_ii]) if qrs_sig_ii.size else 0.0,
            qrs_dur_ms=qrs_dur_ms,
            qrs_deformed_prob=float(max(0.0, min(1.0, qrs_def_prob))),
            qrs_secondary_extrema=secondary_extrema,
            r_prime_v1=bool(np.max(diagnostic[qrs_on:qrs_off, lead_v1] - baseline[lead_v1]) > 0.15) if qrs_off > qrs_on else False,
            broad_r_v6=bool(np.max(diagnostic[qrs_on:qrs_off, lead_v5] - baseline[lead_v5]) > 0.2) if qrs_off > qrs_on else False,
            st_level_v1_mv=float(diagnostic[st_idx, lead_v1] - baseline[lead_v1]),
            st_level_v5_mv=float(diagnostic[st_idx, lead_v5] - baseline[lead_v5]),
            st_slope_v5_uv_per_ms=float(((diagnostic[st_idx, lead_v5] - diagnostic[qrs_off, lead_v5]) / max(1, st_idx - qrs_off)) * fs * 1000.0),
            t_amp_v5_mv=float(np.max(t_sig_v5) - baseline[lead_v5]) if t_sig_v5.size else 0.0,
            t_dur_ms=1000.0 * (t_off - t_on) / fs,
            t_amp_right_mv=min_t,
            t_negative_duration_ms=neg_duration_ms,
            qt_ms=1000.0 * (t_off - qrs_on) / fs,
            qrs_net_area_i_mv_ms=float((1000.0 / fs) * np.sum(qrs_sig_i - baseline[lead_i])) if qrs_sig_i.size else 0.0,
            qrs_net_area_avf_mv_ms=float((1000.0 / fs) * np.sum(qrs_sig_avf - baseline[lead_avf])) if qrs_sig_avf.size else 0.0,
            lead_quality_db=_lead_quality(diagnostic[:, lead_ii], peaks, np),
            delineation_confidence=float(max(0.0, min(1.0, 0.4 + 0.06 * min(len(peaks), 6) + 0.02 * min(10.0, _lead_quality(diagnostic[:, lead_ii], peaks, np))))),
            u_present_v2=bool(fs >= 500 and diagnostic[u_idx, lead_v2] > baseline[lead_v2] + 0.02),
            u_amp_v2_mv=float(diagnostic[u_idx, lead_v2] - baseline[lead_v2]) if fs >= 500 else None,
            pvc_like=bool(qrs_dur_ms >= float(config.thresholds["qrs_wide_ms"]) and rr_prev is not None and rr_s is not None and rr_prev < rr_s),
            apb_like=bool(rr_prev is not None and rr_s is not None and rr_prev < rr_s and qrs_dur_ms < float(config.thresholds["qrs_wide_ms"])),
            paced_like=False,
            is_ectopic=bool(rr_prev is not None and rr_s is not None and rr_prev < 0.85 * rr_s),
            is_paced=False,
            is_artifact=False,
        )
        measurements.append(asdict(beat))
        acceptances.append(
            BeatAcceptance(
                beat_id=beat.beat_id,
                record_id=record_id,
                accepted=True,
                reasons=["accepted"],
                lead_quality_db=beat.lead_quality_db,
                fiducial_completeness=1.0,
                pacing_corrected=True,
            )
        )

    triad_rows = [item.to_dict() for item in build_triad_memberships(record_id, acceptances)]
    return measurements, triad_rows, tq_ratios


def build_measurement_records(config: ProjectConfig, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = split_entries(config, dataset)
    records: list[dict[str, Any]] = []
    triads: list[dict[str, Any]] = []
    for split, entries in grouped.items():
        for entry in entries:
            signal, fs, sig_names = _load_record(config, dataset, entry)
            measurements, triad_rows, tq_ratios = _one_record_measurements(signal, fs, sig_names, str(entry["record_id"]), config)
            records.append(
                {
                    "record_id": str(entry["record_id"]),
                    "split": split,
                    "sampling_rate_hz": fs,
                    "qrs_def_threshold": 0.5,
                    "labels": _parse_labels(entry["labels"]),
                    "beats": measurements,
                    "tq_power_ratios": tq_ratios,
                }
            )
            triads.extend(triad_rows)
    return records, triads


def _label_vector(labels: list[str]) -> list[float]:
    return [1.0 if label in labels else 0.0 for label in PROJECT_LABELS]


def representative_triad_tensor(signal, fs: float, sig_names: list[str], config: ProjectConfig):
    np, _torch, _wfdb, sp_signal = _runtime()
    detection = preprocess_signal(signal, fs, config.filters["detection"])
    lead_ii = _lead_index(sig_names, "II")
    peaks, _meta = detect_r_peaks(detection[:, lead_ii], fs)
    peaks = [int(peak) for peak in peaks if int(0.25 * fs) <= int(peak) < signal.shape[0] - int(0.45 * fs)]
    if len(peaks) < 3:
        peaks = [int(value) for value in np.linspace(int(0.6 * fs), max(int(0.6 * fs), signal.shape[0] - int(0.6 * fs)), num=3, dtype=int)]
    center = len(peaks) // 2
    triad_peaks = peaks[max(0, center - 1) : min(len(peaks), center + 2)]
    while len(triad_peaks) < 3:
        triad_peaks.append(triad_peaks[-1])
    left = int(float(config.training["pre_r_seconds"]) * fs)
    right = int(float(config.training["post_r_seconds"]) * fs)
    samples_per_beat = int(config.training["samples_per_beat"])
    parts = []
    for peak in triad_peaks[:3]:
        beat = _window(detection, peak, left, right, np)
        part = sp_signal.resample(beat, samples_per_beat, axis=0).T.astype(np.float32)
        parts.append(part)
    return np.concatenate(parts, axis=0), triad_peaks[:3]


def _build_split_samples(config: ProjectConfig, dataset: str) -> dict[str, list[dict[str, Any]]]:
    grouped = split_entries(config, dataset)
    output = {"train": [], "val": [], "test": []}
    for split, entries in grouped.items():
        for entry in entries:
            signal, fs, sig_names = _load_record(config, dataset, entry)
            tensor, triad_peaks = representative_triad_tensor(signal, fs, sig_names, config)
            output[split].append(
                {
                    "record_id": str(entry["record_id"]),
                    "tensor": tensor,
                    "target": _label_vector(_parse_labels(entry["labels"])),
                    "labels": _parse_labels(entry["labels"]),
                    "triad_peaks": triad_peaks,
                }
            )
    return output


def train_ptbxl_classifier(config: ProjectConfig) -> dict[str, str]:
    np, torch, _wfdb, _sp_signal = _runtime()
    samples = _build_split_samples(config, "ptbxl")
    model = build_model(
        in_leads=len(LEADS_12),
        triad_length=3,
        samples_per_beat=int(config.training["samples_per_beat"]),
        latent_dim=int(config.latents["penultimate_dim"]),
        num_classes=len(PROJECT_LABELS),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training["learning_rate"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    batch_size = int(config.training["batch_size"])

    model.train()
    for _epoch in range(int(config.training["epochs"])):
        for start in range(0, len(samples["train"]), batch_size):
            batch = samples["train"][start : start + batch_size]
            x = torch.tensor(np.stack([item["tensor"] for item in batch]), dtype=torch.float32)
            y = torch.tensor(np.stack([item["target"] for item in batch]), dtype=torch.float32)
            optimizer.zero_grad()
            logits, _latent = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    checkpoint_path = config.paths.latents / "ptbxl_classifier.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": int(config.latents["penultimate_dim"]),
            "samples_per_beat": int(config.training["samples_per_beat"]),
        },
        checkpoint_path,
    )
    metrics_path = config.paths.reports / "ptbxl_training_metrics.json"
    write_json(
        metrics_path,
        {
            "epochs": int(config.training["epochs"]),
            "train_records": len(samples["train"]),
            "val_records": len(samples["val"]),
            "test_records": len(samples["test"]),
        },
    )
    return {"checkpoint": str(checkpoint_path), "metrics": str(metrics_path)}


def _load_checkpoint(config: ProjectConfig, checkpoint_dataset: str = "ptbxl"):
    _np, torch, _wfdb, _sp_signal = _runtime()
    checkpoint = torch.load(config.paths.latents / f"{checkpoint_dataset}_classifier.pt", map_location="cpu")
    model = build_model(
        in_leads=len(LEADS_12),
        triad_length=3,
        samples_per_beat=int(config.training["samples_per_beat"]),
        latent_dim=int(config.latents["penultimate_dim"]),
        num_classes=len(PROJECT_LABELS),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_samples_for_dataset(
    config: ProjectConfig,
    dataset: str,
    include_targets: bool = True,
    checkpoint_dataset: str = "ptbxl",
) -> dict[str, list[dict[str, Any]]]:
    np, torch, _wfdb, _sp_signal = _runtime()
    model = _load_checkpoint(config, checkpoint_dataset=checkpoint_dataset)
    split_samples = _build_split_samples(config, dataset)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    with torch.no_grad():
        for split, items in split_samples.items():
            for item in items:
                x = torch.tensor(item["tensor"][None, ...], dtype=torch.float32)
                _logits, latent = model(x)
                output[split].append(
                    {
                        "record_id": item["record_id"],
                        "split": split,
                        "latent": latent.squeeze(0).cpu().numpy().astype(np.float32).tolist(),
                        "labels": item["labels"] if include_targets else [],
                    }
                )
    return output


def save_latent_rows(config: ProjectConfig, dataset: str, latent_rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    written: dict[str, str] = {}
    for split, rows in latent_rows_by_split.items():
        if not rows:
            continue
        formatted = []
        for row in rows:
            payload = {"record_id": row["record_id"], "split": split}
            for idx, value in enumerate(row["latent"]):
                payload[f"latent_{idx:04d}"] = value
            formatted.append(payload)
        dataset_path = write_records_table(config.paths.latents / f"A_{dataset}_{split}.parquet", formatted)
        generic_path = write_records_table(config.paths.latents / f"A_{split}.parquet", formatted)
        written[f"{dataset}_{split}"] = str(dataset_path)
        written[split] = str(generic_path)
    return written


