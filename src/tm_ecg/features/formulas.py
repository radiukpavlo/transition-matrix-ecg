"""Pure-Python implementations of the locked feature formulas."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import mean, median, pstdev

from tm_ecg.constants import QT_CORRECTION_CODE


def _clean(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _median(values: list[float | None]) -> float | None:
    cleaned = _clean(values)
    return median(cleaned) if cleaned else None


def _iqr(values: list[float | None]) -> float | None:
    cleaned = sorted(_clean(values))
    if len(cleaned) < 4:
        return None
    lower_idx = int(0.25 * (len(cleaned) - 1))
    upper_idx = int(0.75 * (len(cleaned) - 1))
    return cleaned[upper_idx] - cleaned[lower_idx]


def _std(values: list[float | None]) -> float | None:
    cleaned = _clean(values)
    return pstdev(cleaned) if len(cleaned) >= 2 else None


def median_angle_deg(angles_deg: list[float]) -> float | None:
    if not angles_deg:
        return None
    sin_med = median([math.sin(math.radians(angle)) for angle in angles_deg])
    cos_med = median([math.cos(math.radians(angle)) for angle in angles_deg])
    return math.degrees(math.atan2(sin_med, cos_med))


def st_offset_seconds(hr_bpm: float) -> float:
    return 0.06 if hr_bpm >= 100.0 else 0.08


@dataclass(slots=True)
class BeatMeasurement:
    beat_id: str
    rr_s: float | None = None
    rr_prev_s: float | None = None
    rr_next_s: float | None = None
    p_present: bool = False
    p_amp_ii_mv: float | None = None
    p_dur_ms: float | None = None
    pr_ms: float | None = None
    q_amp_ii_mv: float | None = None
    r_amp_ii_mv: float | None = None
    s_amp_ii_mv: float | None = None
    qrs_dur_ms: float | None = None
    qrs_deformed_prob: float | None = None
    qrs_secondary_extrema: int = 0
    r_prime_v1: bool = False
    broad_r_v6: bool = False
    st_level_v1_mv: float | None = None
    st_level_v5_mv: float | None = None
    st_slope_v5_uv_per_ms: float | None = None
    t_amp_v5_mv: float | None = None
    t_dur_ms: float | None = None
    t_amp_right_mv: float | None = None
    t_negative_duration_ms: float | None = None
    qt_ms: float | None = None
    qrs_net_area_i_mv_ms: float | None = None
    qrs_net_area_avf_mv_ms: float | None = None
    lead_quality_db: float | None = None
    delineation_confidence: float | None = None
    u_present_v2: bool | None = None
    u_amp_v2_mv: float | None = None
    pvc_like: bool = False
    apb_like: bool = False
    paced_like: bool = False
    is_ectopic: bool = False
    is_paced: bool = False
    is_artifact: bool = False


@dataclass(slots=True)
class RecordMeasurements:
    record_id: str
    beats: list[BeatMeasurement] = field(default_factory=list)
    tq_power_ratios: list[float] = field(default_factory=list)
    sampling_rate_hz: float = 500.0
    qrs_def_threshold: float = 0.5


def _nn_intervals(beats: list[BeatMeasurement]) -> list[float]:
    values = []
    for beat in beats:
        if beat.rr_s is None:
            continue
        if beat.is_ectopic or beat.is_paced or beat.is_artifact:
            continue
        values.append(beat.rr_s)
    return values


def _qtc_fridericia_ms(qt_ms: float | None, rr_s: float | None) -> float | None:
    if qt_ms is None or rr_s is None or rr_s <= 0:
        return None
    return 1000.0 * ((qt_ms / 1000.0) / (rr_s ** (1.0 / 3.0)))


def compute_record_features(record: RecordMeasurements, thresholds: dict[str, object]) -> dict[str, float | int | None]:
    beats = record.beats
    rr_values = [beat.rr_s for beat in beats]
    hr_values = [60.0 / rr for rr in _clean(rr_values) if rr > 0]
    nn_values = _nn_intervals(beats)
    rr_n = median(nn_values) if nn_values else None

    prematurity = []
    comp_pause = []
    for beat in beats:
        if rr_n and beat.rr_prev_s is not None:
            prematurity.append(beat.rr_prev_s / rr_n)
        if rr_n and beat.is_ectopic and beat.rr_prev_s is not None and beat.rr_next_s is not None:
            comp_pause.append((beat.rr_prev_s + beat.rr_next_s) / (2.0 * rr_n))

    qrs_angles = []
    for beat in beats:
        if beat.qrs_net_area_i_mv_ms is None or beat.qrs_net_area_avf_mv_ms is None:
            continue
        qrs_angles.append(math.degrees(math.atan2(beat.qrs_net_area_avf_mv_ms, beat.qrs_net_area_i_mv_ms)))

    qtc_values = [_qtc_fridericia_ms(beat.qt_ms, beat.rr_s) for beat in beats]
    axis_deg = median_angle_deg(qrs_angles)

    features: dict[str, float | int | None] = {
        "hr_med_bpm": _median(hr_values),
        "rr_med_ms": None if _median(rr_values) is None else 1000.0 * _median(rr_values),  # type: ignore[arg-type]
        "rr_iqr_ms": None if _iqr(rr_values) is None else 1000.0 * _iqr(rr_values),  # type: ignore[arg-type]
        "rr_sdnn_ms": None if _std(nn_values) is None or len(nn_values) < 5 else 1000.0 * _std(nn_values),  # type: ignore[arg-type]
        "prematurity_index_min": min(prematurity) if prematurity else None,
        "comp_pause_ratio_max": max(comp_pause) if comp_pause else None,
        "pvc_like_beat_count": sum(1 for beat in beats if beat.pvc_like),
        "apb_like_beat_count": sum(1 for beat in beats if beat.apb_like),
        "paced_like_beat_count": sum(1 for beat in beats if beat.paced_like),
        "af_irregularity_cv": None if not _clean(rr_values) or mean(_clean(rr_values)) == 0 else (pstdev(_clean(rr_values)) / mean(_clean(rr_values)) if len(_clean(rr_values)) >= 2 else 0.0),
        "f_wave_power_ratio": _median(record.tq_power_ratios),
        "p_present_ratio": (sum(1 for beat in beats if beat.p_present) / len(beats)) if beats else None,
        "p_amp_ii_med_mV": _median([beat.p_amp_ii_mv for beat in beats]),
        "p_dur_med_ms": _median([beat.p_dur_ms for beat in beats]),
        "pr_med_ms": _median([beat.pr_ms for beat in beats]),
        "pr_iqr_ms": _iqr([beat.pr_ms for beat in beats]),
        "q_amp_ii_med_mV": _median([beat.q_amp_ii_mv for beat in beats]),
        "r_amp_ii_med_mV": _median([beat.r_amp_ii_mv for beat in beats]),
        "s_amp_ii_med_mV": _median([beat.s_amp_ii_mv for beat in beats]),
        "qrs_dur_med_ms": _median([beat.qrs_dur_ms for beat in beats]),
        "qrs_dur_iqr_ms": _iqr([beat.qrs_dur_ms for beat in beats]),
        "qrs_deformed_prob": _median([beat.qrs_deformed_prob for beat in beats]),
        "qrs_deformed_any": int(any((beat.qrs_deformed_prob or 0.0) >= record.qrs_def_threshold for beat in beats)),
        "qrs_fragmented_any": int(any(beat.qrs_secondary_extrema >= 2 for beat in beats)),
        "qrs_wide_any": int(any((beat.qrs_dur_ms or 0.0) >= float(thresholds["qrs_wide_ms"]) for beat in beats)),
        "r_prime_v1_any": int(any(beat.r_prime_v1 for beat in beats)),
        "broad_r_v6_any": int(any(beat.broad_r_v6 for beat in beats)),
        "st_level_v1_mV": _median([beat.st_level_v1_mv for beat in beats]),
        "st_level_v5_mV": _median([beat.st_level_v5_mv for beat in beats]),
        "st_slope_v5_uV_per_ms": _median([beat.st_slope_v5_uv_per_ms for beat in beats]),
        "t_amp_v5_med_mV": _median([beat.t_amp_v5_mv for beat in beats]),
        "t_dur_med_ms": _median([beat.t_dur_ms for beat in beats]),
        "t_inverted_right_any": int(
            any(
                (beat.t_amp_right_mv is not None and beat.t_amp_right_mv <= float(thresholds["t_inverted_threshold_mv"]))
                and (beat.t_negative_duration_ms is not None and beat.t_negative_duration_ms >= float(thresholds["t_inverted_duration_ms"]))
                for beat in beats
            )
        ),
        "qt_med_ms": _median([beat.qt_ms for beat in beats]),
        "qtc_med_ms": _median(qtc_values),
        "qrs_net_area_i_mV_ms": _median([beat.qrs_net_area_i_mv_ms for beat in beats]),
        "qrs_net_area_avf_mV_ms": _median([beat.qrs_net_area_avf_mv_ms for beat in beats]),
        "qrs_axis_deg": axis_deg,
        "qrs_axis_sin": None if axis_deg is None else math.sin(math.radians(axis_deg)),
        "qrs_axis_cos": None if axis_deg is None else math.cos(math.radians(axis_deg)),
        "lead_quality_min_db": min(_clean([beat.lead_quality_db for beat in beats])) if _clean([beat.lead_quality_db for beat in beats]) else None,
        "delineation_confidence": _median([beat.delineation_confidence for beat in beats]),
        "u_present_v2_any": None if record.sampling_rate_hz < 500 else int(any(bool(beat.u_present_v2) for beat in beats if beat.u_present_v2 is not None)),
        "u_amp_v2_mV": None if record.sampling_rate_hz < 500 else _median([beat.u_amp_v2_mv for beat in beats]),
    }
    features["qtc_formula_code"] = QT_CORRECTION_CODE
    return features
