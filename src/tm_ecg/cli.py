"""Command-line interface for the locked ECG baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from tm_ecg.config import ProjectConfig
from tm_ecg.stages import (
    bootstrap_env,
    delineate,
    explain,
    features,
    fit_transition,
    freeze,
    index,
    ingest,
    pace,
    preprocess,
    report,
    rpeaks,
    splits,
    train_classifier,
    triads,
)


StageFn = Callable[[ProjectConfig, argparse.Namespace], int]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tm-ecg")
    parser.add_argument("--config", default="configs/defaults.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap-env")
    bootstrap.set_defaults(handler=bootstrap_env.run)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--source", choices=("zip", "download"), default="zip")
    ingest_parser.set_defaults(handler=ingest.run)

    index_parser = subparsers.add_parser("index")
    index_parser.set_defaults(handler=index.run)

    split_parser = subparsers.add_parser("splits")
    split_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    split_parser.set_defaults(handler=splits.run)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    preprocess_parser.set_defaults(handler=preprocess.run)

    pace_parser = subparsers.add_parser("pace")
    pace_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    pace_parser.set_defaults(handler=pace.run)

    rpeak_parser = subparsers.add_parser("rpeaks")
    rpeak_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    rpeak_parser.set_defaults(handler=rpeaks.run)

    delineate_parser = subparsers.add_parser("delineate")
    delineate_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    delineate_parser.set_defaults(handler=delineate.run)

    triads_parser = subparsers.add_parser("triads")
    triads_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    triads_parser.set_defaults(handler=triads.run)

    train_parser = subparsers.add_parser("train-classifier")
    train_parser.add_argument("--dataset", choices=("ptbxl",), required=True)
    train_parser.set_defaults(handler=train_classifier.run)

    extract_parser = subparsers.add_parser("extract-a")
    extract_parser.add_argument("--dataset", choices=("ptbxl", "ludb"), required=True)
    extract_parser.set_defaults(handler=triads.extract_latents)

    build_b = subparsers.add_parser("build-b")
    build_b.add_argument("--dataset", choices=("b1", "b2"), required=True)
    build_b.set_defaults(handler=features.run)

    fit_parser = subparsers.add_parser("fit-transition")
    fit_parser.add_argument("--dataset", choices=("b1", "b2"), required=True)
    fit_parser.set_defaults(handler=fit_transition.run)

    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("--split", choices=("val", "test"), required=True)
    explain_parser.add_argument("--dataset", choices=("b1", "b2"), default="b1")
    explain_parser.set_defaults(handler=explain.run)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--experiment", required=True)
    report_parser.set_defaults(handler=report.run)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--experiment", default="default")
    freeze_parser.set_defaults(handler=freeze.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = ProjectConfig.load(Path(args.config))
    config.ensure_directories()
    handler: StageFn = args.handler
    return handler(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
