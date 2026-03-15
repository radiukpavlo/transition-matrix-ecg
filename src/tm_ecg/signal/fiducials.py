"""Fiducial validation and beat acceptance logic."""

from __future__ import annotations

from tm_ecg.types import BeatAcceptance, BeatFiducials


def verify_fiducial_order(fiducials: BeatFiducials) -> bool:
    values = [
        fiducials.p_on,
        fiducials.p_peak,
        fiducials.p_off,
        fiducials.qrs_on,
        fiducials.r_peak,
        fiducials.qrs_off,
        fiducials.t_on,
        fiducials.t_peak,
        fiducials.t_off,
    ]
    cleaned = [value for value in values if value is not None]
    return cleaned == sorted(cleaned)


def accept_beat(
    fiducials: BeatFiducials,
    lead_quality_db: float,
    delineation_confidence: float,
    pacing_contaminated: bool,
) -> BeatAcceptance:
    reasons = []
    if not verify_fiducial_order(fiducials):
        reasons.append("invalid_fiducial_order")
    if pacing_contaminated:
        reasons.append("pacing_contamination")
    accepted = not reasons
    return BeatAcceptance(
        beat_id=fiducials.beat_id,
        record_id=fiducials.record_id,
        accepted=accepted,
        reasons=reasons or ["accepted"],
        lead_quality_db=lead_quality_db,
        fiducial_completeness=sum(value is not None for value in fiducials.to_dict().values() if not isinstance(value, str)) / 10.0,
        pacing_corrected=not pacing_contaminated,
    )
