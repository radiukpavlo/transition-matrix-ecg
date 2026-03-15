# transition-matrix-ecg

Locked baseline implementation for explainable ECG arrhythmia analysis with
clinician-facing feature matrices, triad latents, and a reduced-rank ridge
transition operator.

## Status

This repository contains a package-oriented research pipeline scaffold with:

- a `tm-ecg` CLI covering the end-to-end stages in `AGENTS.md`
- typed project configuration and manifests
- dataset ingestion and split-freezing infrastructure
- feature-registry and typed transform support
- model, transition, explanation, and report entrypoints
- smoke-testable pure-Python components

Heavy training and full dataset processing are intentionally deferred until the
scientific Python stack is installed.

## Quick Start

1. Review the defaults in `configs/defaults.toml`.
2. Use an editable install or set `PYTHONPATH=src`.
3. Bootstrap the environment:

```powershell
$env:PYTHONPATH = "src"
python -m tm_ecg.cli bootstrap-env
```

4. Ingest the locked archives:

```powershell
$env:PYTHONPATH = "src"
python -m tm_ecg.cli ingest --source zip
```

5. Freeze ontology and splits:

```powershell
$env:PYTHONPATH = "src"
python -m tm_ecg.cli splits --dataset ptbxl
python -m tm_ecg.cli splits --dataset ludb
```
