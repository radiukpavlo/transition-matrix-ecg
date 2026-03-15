"""Adaptive R-peak detection stage policy."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    payload = {
        "dataset": dataset,
        "lead_ii_snr_db": config.thresholds["lead_ii_snr_db"],
        "status": "algorithm_ready",
        "notes": [
            "Use adaptive RR-horizon updates instead of a fixed 260-sample window.",
            "Accepted peaks must satisfy cross-lead temporal consistency.",
        ],
    }
    write_stage_manifest(config, f"rpeaks_{dataset}", payload)
    print(f"R-peak policy manifest written for {dataset}")
    return 0
