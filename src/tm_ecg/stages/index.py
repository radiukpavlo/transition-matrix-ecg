"""Dataset indexing and source file manifest generation."""

from __future__ import annotations

import csv
from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import write_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.ontology import map_ludb_text, map_ptbxl_labels
from tm_ecg.stages.shared import dataset_root, find_single_file, write_stage_manifest


def _suffix_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower() or "<no_suffix>"
            counts[suffix] = counts.get(suffix, 0) + 1
    return counts


def _index_ptbxl(config: ProjectConfig) -> list[dict[str, object]]:
    root = dataset_root(config, "ptbxl")
    metadata_path = find_single_file(root, config.datasets["ptbxl"].metadata_csv or "ptbxl_database.csv")
    rows: list[dict[str, object]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels = map_ptbxl_labels(row)
            record_id = row.get("ecg_id") or row.get("filename_hr") or row.get("filename_lr") or "unknown"
            rows.append(
                {
                    "dataset": "ptbxl",
                    "record_id": str(record_id),
                    "patient_id": row.get("patient_id"),
                    "strat_fold": row.get("strat_fold"),
                    "sex": row.get("sex"),
                    "age": row.get("age"),
                    "labels": "|".join(labels),
                    "filename_hr": row.get("filename_hr"),
                    "filename_lr": row.get("filename_lr"),
                    "pacemaker": row.get("pacemaker", ""),
                }
            )
    return rows


def _parse_ludb_header(header_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with header_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                continue
            body = line[1:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
    return metadata


def _index_ludb(config: ProjectConfig) -> list[dict[str, object]]:
    root = dataset_root(config, "ludb")
    records_file = find_single_file(root, "RECORDS")
    base_dir = records_file.parent
    rows: list[dict[str, object]] = []
    with records_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            record_path = line.strip()
            if not record_path:
                continue
            header_path = base_dir / f"{record_path}.hea"
            metadata = _parse_ludb_header(header_path)
            diagnosis_text = " ".join(
                value for key, value in metadata.items() if key in {"diagnosis", "diagnoses", "dx"}
            )
            labels = map_ludb_text(diagnosis_text)
            rows.append(
                {
                    "dataset": "ludb",
                    "record_id": Path(record_path).name,
                    "patient_id": metadata.get("patient"),
                    "sex": metadata.get("sex"),
                    "age": metadata.get("age"),
                    "labels": "|".join(labels),
                    "header_path": str(header_path),
                }
            )
    return rows


def run(config: ProjectConfig, args: object) -> int:
    summaries = {}
    datasets = {
        "ptbxl": _index_ptbxl(config),
        "ludb": _index_ludb(config),
        "ptbxl_plus": [],
    }
    for dataset_key, rows in datasets.items():
        root = dataset_root(config, dataset_key)
        summaries[dataset_key] = {
            "root": str(root),
            "suffix_counts": _suffix_counts(root),
            "row_count": len(rows),
        }
        if rows:
            destination = config.paths.manifests / f"{dataset_key}_index.parquet"
            actual = write_records_table(destination, rows)
            summaries[dataset_key]["index_path"] = str(actual)

    write_json(config.paths.manifests / "dataset_file_inventory.json", summaries)
    write_stage_manifest(config, "index", summaries)
    print("Dataset indexes written to manifests/")
    return 0
