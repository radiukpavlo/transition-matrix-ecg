"""Typed B-space transform fitting and transition-operator estimation."""

from __future__ import annotations

import csv
from pathlib import Path

from tm_ecg.config import ProjectConfig
from tm_ecg.features.registry import fit_columns
from tm_ecg.io.common import write_json
from tm_ecg.io.tabular import write_records_table
from tm_ecg.stages.shared import write_stage_manifest
from tm_ecg.transition.ridge import apply_transition, fit_ridge_transition, save_operator_package
from tm_ecg.transition.typed_transforms import fit_transform_bundle, transform_rows


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


def _align(
    a_rows: list[dict[str, object]],
    b_rows: list[dict[str, object]],
    b_columns: list[str],
) -> tuple[list[list[float]], list[list[float]]]:
    a_map = {str(row["record_id"]): row for row in a_rows}
    b_map = {str(row["record_id"]): row for row in b_rows}
    common = sorted(set(a_map) & set(b_map))
    if not common:
        return [], []
    a_columns = [column for column in a_map[common[0]].keys() if column not in {"record_id", "split"}]
    a_matrix = []
    b_matrix = []
    for record_id in common:
        a_row = a_map[record_id]
        b_row = b_map[record_id]
        if any(a_row.get(column) is None for column in a_columns):
            continue
        if any(b_row.get(column) is None for column in b_columns):
            continue
        a_matrix.append([float(a_row[column]) for column in a_columns])
        b_matrix.append([float(b_row[column]) for column in b_columns])
    return a_matrix, b_matrix


def _mae(y_true: list[list[float]], y_pred: list[list[float]]) -> float:
    errors = []
    for true_row, pred_row in zip(y_true, y_pred, strict=False):
        for true_value, pred_value in zip(true_row, pred_row, strict=False):
            errors.append(abs(true_value - pred_value))
    return sum(errors) / len(errors) if errors else float("inf")


def run(config: ProjectConfig, args: object) -> int:
    dataset = getattr(args, "dataset")
    raw_train_path = _find_table(config.paths.features, f"{dataset.upper()}_raw_train")
    if raw_train_path is None:
        write_stage_manifest(config, f"fit_transition_{dataset}", {"dataset": dataset, "status": "waiting_for_raw_features"})
        print(f"No raw training features found for {dataset}")
        return 0

    train_rows = _read_table(raw_train_path)
    columns = fit_columns(train_rows)
    bundle = fit_transform_bundle(train_rows, columns, eps=float(config.thresholds["eps"]))
    bundle.dataset = dataset

    fit_outputs = {}
    transformed_rows_by_split = {}
    for path in sorted(config.paths.features.glob(f"{dataset.upper()}_raw_*.*")):
        split = path.stem.replace(f"{dataset.upper()}_raw_", "")
        rows = _read_table(path)
        transformed = transform_rows(rows, bundle)
        transformed_rows_by_split[split] = transformed
        fit_path = write_records_table(config.paths.features / f"{dataset.upper()}_fit_{split}.parquet", transformed)
        fit_outputs[split] = str(fit_path)

    bundle_path = config.paths.transition / f"{dataset.upper()}_transform_bundle.json"
    write_json(bundle_path, bundle.to_dict())

    a_train_path = _dataset_a_path(config, dataset, "train")
    if a_train_path is None:
        write_stage_manifest(
            config,
            f"fit_transition_{dataset}",
            {
                "dataset": dataset,
                "status": "waiting_for_a_train",
                "transform_bundle": str(bundle_path),
                "fit_outputs": fit_outputs,
            },
        )
        print(f"Typed transforms written for {dataset}; no A_train found yet.")
        return 0

    a_train_rows = _read_table(a_train_path)
    a_train, b_train = _align(a_train_rows, transformed_rows_by_split["train"], bundle.fit_columns)
    if not a_train or not b_train:
        raise RuntimeError("No aligned rows between A_train and B_fit_train")

    best = None
    for lambda_value in config.transition["lambda_grid"]:
        candidate = fit_ridge_transition(a_train, b_train, float(lambda_value), rank_cap=int(config.transition["rank_cap"]))
        score = 0.0
        a_val_path = _dataset_a_path(config, dataset, "val")
        if a_val_path is not None and "val" in transformed_rows_by_split:
            a_val_rows = _read_table(a_val_path)
            a_val, b_val = _align(a_val_rows, transformed_rows_by_split["val"], bundle.fit_columns)
            if a_val and b_val:
                predicted = apply_transition(a_val, candidate["operator"])
                score = _mae(b_val, predicted)
        if best is None or score < best["score"]:
            best = {"score": score, "payload": candidate}

    operator_path = save_operator_package(config.paths.transition / f"{dataset.upper()}_T_ridge.json", best["payload"])
    write_stage_manifest(
        config,
        f"fit_transition_{dataset}",
        {
            "dataset": dataset,
            "status": "complete",
            "transform_bundle": str(bundle_path),
            "fit_outputs": fit_outputs,
            "operator_path": str(operator_path),
            "selected_lambda": best["payload"]["lambda_value"],
        },
    )
    print(f"Transition operator written for {dataset}")
    return 0

