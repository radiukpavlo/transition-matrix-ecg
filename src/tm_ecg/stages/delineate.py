"""Fiducial delineation and beat acceptance stage policy."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    payload = {
        "dataset": dataset,
        "minimum_valid_beats": config.thresholds["minimum_valid_beats"],
        "minimum_analyzable_fraction": config.thresholds["minimum_analyzable_fraction"],
        "status": "policy_frozen",
        "notes": [
            "PTB-XL+ fiducials are verified against waveform consistency.",
            "LUDB manual fiducials are the source of truth.",
        ],
    }
    write_stage_manifest(config, f"delineate_{dataset}", payload)
    print(f"Delineation policy manifest written for {dataset}")
    return 0
