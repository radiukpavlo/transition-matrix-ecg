"""Pacing artifact policy stage."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    payload = {
        "dataset": dataset,
        "status": "policy_frozen",
        "notes": [
            "Pacing spikes must be detected before routine morphology filtering.",
            "Spike-removed waveforms must preserve adjacent physiologic morphology.",
        ],
    }
    write_stage_manifest(config, f"pace_{dataset}", payload)
    print(f"Pacing policy manifest written for {dataset}")
    return 0
