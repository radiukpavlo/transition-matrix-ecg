"""Waveform preprocessing policy stage."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.signal.filtering import branch_parameters
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    payload = {
        "dataset": dataset,
        "detection_branch": branch_parameters(config.filters["detection"]),
        "diagnostic_branch": branch_parameters(config.filters["diagnostic"]),
        "status": "policy_frozen",
        "notes": [
            "Detection branch allows 0.67 Hz high-pass for anchor robustness.",
            "Diagnostic branch preserves 0.05 Hz high-pass for ST/QT fidelity.",
        ],
    }
    write_stage_manifest(config, f"preprocess_{dataset}", payload)
    print(f"Preprocessing policy manifest written for {dataset}")
    return 0
