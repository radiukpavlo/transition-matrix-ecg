"""Feature dictionary and B-matrix schema helpers."""

from __future__ import annotations

from tm_ecg.constants import B_COLUMNS
from tm_ecg.features.formulas import RecordMeasurements, compute_record_features
from tm_ecg.types import FeatureDictionaryRow


FEATURE_SPECS: dict[str, tuple[str, str, str, str]] = {
    "hr_med_bpm": ("rhythm", "bpm", "continuous", "F1"),
    "rr_med_ms": ("rhythm", "ms", "continuous", "F2"),
    "rr_iqr_ms": ("rhythm", "ms", "continuous", "F2"),
    "rr_sdnn_ms": ("rhythm", "ms", "continuous", "F3"),
    "prematurity_index_min": ("rhythm", "ratio", "bounded", "F4"),
    "comp_pause_ratio_max": ("rhythm", "ratio", "bounded", "F5"),
    "pvc_like_beat_count": ("burden", "count", "count", "F6"),
    "apb_like_beat_count": ("burden", "count", "count", "F6"),
    "paced_like_beat_count": ("burden", "count", "count", "F6"),
    "af_irregularity_cv": ("rhythm", "ratio", "bounded", "F7"),
    "f_wave_power_ratio": ("atrial", "ratio", "bounded", "F8"),
    "p_present_ratio": ("atrial", "ratio", "bounded", "F9"),
    "p_amp_ii_med_mV": ("atrial", "mV", "continuous", "F10"),
    "p_dur_med_ms": ("atrial", "ms", "continuous", "F11"),
    "pr_med_ms": ("atrial", "ms", "continuous", "F12"),
    "pr_iqr_ms": ("atrial", "ms", "continuous", "F12"),
    "q_amp_ii_med_mV": ("qrs", "mV", "continuous", "F10"),
    "r_amp_ii_med_mV": ("qrs", "mV", "continuous", "F10"),
    "s_amp_ii_med_mV": ("qrs", "mV", "continuous", "F10"),
    "qrs_dur_med_ms": ("qrs", "ms", "continuous", "F13"),
    "qrs_dur_iqr_ms": ("qrs", "ms", "continuous", "F13"),
    "qrs_deformed_prob": ("qrs", "prob", "bounded", "F14"),
    "qrs_deformed_any": ("qrs", "binary", "binary", "F14"),
    "qrs_fragmented_any": ("qrs", "binary", "binary", "F15"),
    "qrs_wide_any": ("qrs", "binary", "binary", "F15"),
    "r_prime_v1_any": ("qrs", "binary", "binary", "F16"),
    "broad_r_v6_any": ("qrs", "binary", "binary", "F16"),
    "st_level_v1_mV": ("st", "mV", "continuous", "F17"),
    "st_level_v5_mV": ("st", "mV", "continuous", "F17"),
    "st_slope_v5_uV_per_ms": ("st", "uV/ms", "continuous", "F18"),
    "t_amp_v5_med_mV": ("t", "mV", "continuous", "F19"),
    "t_dur_med_ms": ("t", "ms", "continuous", "F19"),
    "t_inverted_right_any": ("t", "binary", "binary", "F20"),
    "qt_med_ms": ("qt", "ms", "continuous", "F21"),
    "qtc_med_ms": ("qt", "ms", "continuous", "F21"),
    "qrs_net_area_i_mV_ms": ("axis", "mV*ms", "continuous", "F22"),
    "qrs_net_area_avf_mV_ms": ("axis", "mV*ms", "continuous", "F22"),
    "qrs_axis_deg": ("axis", "deg", "circular", "F23"),
    "qrs_axis_sin": ("axis", "unitless", "continuous", "F23"),
    "qrs_axis_cos": ("axis", "unitless", "continuous", "F23"),
    "rbbb_signature_score": ("signature", "logodds", "continuous", "F24"),
    "lbbb_signature_score": ("signature", "logodds", "continuous", "F24"),
    "pvc_signature_score": ("signature", "logodds", "continuous", "F24"),
    "af_signature_score": ("signature", "logodds", "continuous", "F24"),
    "paced_signature_score": ("signature", "logodds", "continuous", "F24"),
    "lead_quality_min_db": ("quality", "dB", "continuous", "F25"),
    "delineation_confidence": ("quality", "0-1", "bounded", "F26"),
    "u_present_v2_any": ("u", "binary", "binary", "F27"),
    "u_amp_v2_mV": ("u", "mV", "continuous", "F27"),
}


OPTIONAL_COLUMNS = {"u_present_v2_any", "u_amp_v2_mV"}


def feature_dictionary_rows() -> list[FeatureDictionaryRow]:
    return [
        FeatureDictionaryRow(
            column=column,
            family=FEATURE_SPECS[column][0],
            unit=FEATURE_SPECS[column][1],
            value_type=FEATURE_SPECS[column][2],
            level="record",
            formula=FEATURE_SPECS[column][3],
            included=True,
            notes="Optional" if column in OPTIONAL_COLUMNS else "",
        )
        for column in B_COLUMNS
    ]


def feature_types() -> dict[str, str]:
    return {column: FEATURE_SPECS[column][2] for column in B_COLUMNS}


def build_raw_feature_row(record: RecordMeasurements, thresholds: dict[str, object]) -> dict[str, object]:
    features = compute_record_features(record, thresholds)
    row: dict[str, object] = {column: features.get(column) for column in B_COLUMNS}
    row["record_id"] = record.record_id
    row["qtc_formula_code"] = features["qtc_formula_code"]
    return row


def fit_columns(rows: list[dict[str, object]], optional_missingness_threshold: float = 0.2) -> list[str]:
    selected = []
    row_count = max(len(rows), 1)
    for column in B_COLUMNS:
        if column == "qrs_axis_deg":
            continue
        missingness = sum(1 for row in rows if row.get(column) is None) / row_count
        if column in OPTIONAL_COLUMNS and missingness > optional_missingness_threshold:
            continue
        selected.append(column)
    return selected
