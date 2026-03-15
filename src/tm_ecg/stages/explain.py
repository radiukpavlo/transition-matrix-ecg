"""Apply the trained transition operator to validation or test latents."""

from __future__ import annotations

import csv
from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.io.common import read_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.stages.shared import write_stage_manifest
from tm_ecg.transition.ridge import apply_transition
from tm_ecg.transition.typed_transforms import inverse_rows
from tm_ecg.types import TransformBundle, TransformColumnStats


def _read_table(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    import pyarrow.parquet as pq  # type: ignore

    table = pq.read_table(path)
    columns = {name: table[name].to_pylist() for name in table.column_names}
    row_count = len(next(iter(columns.values()), []))
    return [{name: columns[name][idx] for name in table.column_names} for idx in range(row_count)]


def _find_table(base_dir: Path, stem: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        candidate = base_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _dataset_a_path(config: ProjectConfig, dataset: str, split: str) -> Path | None:
    dataset_key = {"b1": "ptbxl", "b2": "ludb"}.get(dataset, dataset)
    preferred = _find_table(config.paths.latents, f"A_{dataset_key}_{split}")
    if preferred is not None:
        return preferred
    return _find_table(config.paths.latents, f"A_{split}")


def _load_bundle(path: Path) -> TransformBundle:
    payload = read_json(path)
    return TransformBundle(
        dataset=str(payload["dataset"]),
        fit_columns=list(payload["fit_columns"]),
        dropped_columns=list(payload.get("dropped_columns", [])),
        stats=[TransformColumnStats(**item) for item in payload["stats"]],
    )


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    split = getattr(args, "split")
    a_path = _dataset_a_path(config, dataset, split)
    bundle_path = config.paths.transition / f"{dataset.upper()}_transform_bundle.json"
    operator_path = config.paths.transition / f"{dataset.upper()}_T_ridge.json"
    if a_path is None or not bundle_path.exists() or not operator_path.exists():
        write_stage_manifest(
            config,
            f"explain_{dataset}_{split}",
            {"dataset": dataset, "split": split, "status": "waiting_for_dependencies"},
        )
        print(f"Missing artifacts for explain {dataset} {split}")
        return 0

    a_rows = _read_table(a_path)
    a_columns = [column for column in a_rows[0].keys() if column not in {"record_id", "split"}]
    matrix = [[float(row[column]) for column in a_columns] for row in a_rows]
    operator_payload = read_json(operator_path)
    predicted_fit = apply_transition(matrix, operator_payload["operator"])
    bundle = _load_bundle(bundle_path)
    fit_rows = []
    for row, predicted in zip(a_rows, predicted_fit, strict=False):
        fit_row = {"record_id": row["record_id"]}
        for column, value in zip(bundle.fit_columns, predicted, strict=False):
            fit_row[column] = value
        fit_rows.append(fit_row)
    raw_rows = inverse_rows(fit_rows, bundle)

    fit_output = write_records_table(config.paths.transition / f"{dataset.upper()}_B_hat_fit_{split}.parquet", fit_rows)
    raw_output = write_records_table(config.paths.transition / f"{dataset.upper()}_B_hat_raw_{split}.parquet", raw_rows)
    write_stage_manifest(
        config,
        f"explain_{dataset}_{split}",
        {
            "dataset": dataset,
            "split": split,
            "status": "complete",
            "fit_output": str(fit_output),
            "raw_output": str(raw_output),
        },
    )
    print(f"Predicted feature outputs written for {dataset} {split}")
    return 0
