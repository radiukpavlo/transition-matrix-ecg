"""Ontology helpers shared across datasets and reports."""

from __future__ import annotations

import ast

from tm_ecg.constants import LUDB_TO_PROJECT, PROJECT_LABELS, PTBXL_TO_PROJECT
from tm_ecg.types import OntologyMapping


def parse_ptbxl_scp_codes(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, dict):
        return [str(key) for key in parsed.keys()]
    return []


def map_ptbxl_labels(row: dict[str, str]) -> list[str]:
    labels = []
    for code in parse_ptbxl_scp_codes(row.get("scp_codes")):
        mapped = PTBXL_TO_PROJECT.get(code.upper())
        if mapped and mapped not in labels:
            labels.append(mapped)
    pacemaker = (row.get("pacemaker") or row.get("pacemaker_present") or "").strip().lower()
    if pacemaker in {"1", "true", "yes", "y"} and "Paced" not in labels:
        labels.append("Paced")
    return labels or ["Other / unmapped"]


def map_ludb_text(text: str | None) -> list[str]:
    if not text:
        return ["Other / unmapped"]
    lowered = text.strip().lower()
    labels = []
    for source, mapped in LUDB_TO_PROJECT.items():
        if source in lowered and mapped not in labels:
            labels.append(mapped)
    return labels or ["Other / unmapped"]


def normalize_project_labels(labels: list[str]) -> list[str]:
    normalized = []
    for label in labels:
        if label in PROJECT_LABELS and label not in normalized:
            normalized.append(label)
    return normalized or ["Other / unmapped"]


def appendix_d_mapping() -> list[OntologyMapping]:
    rows = [
        ("ptbxl", "PTB-XL NORM", "Normal", "Appendix D"),
        ("ludb", "normal sinus rhythm / no pathologic rhythm label", "Normal", "Appendix D"),
        ("ptbxl", "PTB-XL PVC / VPB / ventricular ectopy statements", "PVC", "Appendix D"),
        ("ludb", "ventricular extrasystole / PVC diagnosis", "PVC", "Appendix D"),
        ("ptbxl", "PTB-XL PAC/APB/SPAC/SVPB", "APB", "Appendix D"),
        ("ludb", "atrial extrasystole / supraventricular ectopy diagnosis", "APB", "Appendix D"),
        ("ptbxl", "PTB-XL RBBB / CRBBB / IRBBB", "RBBB spectrum", "Appendix D"),
        ("ludb", "right bundle branch block diagnosis", "RBBB spectrum", "Appendix D"),
        ("ptbxl", "PTB-XL LBBB / CLBBB", "LBBB spectrum", "Appendix D"),
        ("ludb", "left bundle branch block diagnosis", "LBBB spectrum", "Appendix D"),
        ("ptbxl", "PTB-XL AFIB", "AF", "Appendix D"),
        ("ludb", "atrial fibrillation rhythm field", "AF", "Appendix D"),
        ("ptbxl", "PTB-XL AFLT", "AFL", "Appendix D"),
        ("ludb", "atrial flutter rhythm field", "AFL", "Appendix D"),
        ("ptbxl", "PTB-XL paced / pacemaker metadata", "Paced", "Appendix D"),
        ("ludb", "pacemaker-present rhythm/metadata", "Paced", "Appendix D"),
        ("ptbxl", "PTB-XL other rhythm/form statements", "Other / unmapped", "Appendix D"),
        ("ludb", "unmatched diagnoses", "Other / unmapped", "Appendix D"),
    ]
    return [
        OntologyMapping(
            source_dataset=dataset,
            source_label=source_label,
            project_label=project_label,
            notes=notes,
        )
        for dataset, source_label, project_label, notes in rows
    ]
