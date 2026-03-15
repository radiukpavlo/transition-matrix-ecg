# Data Lock Directory

This directory (`data_lock/`) is designated for the secure, version-locked storage of the original ECG dataset archives. By maintaining unaltered, hashed copies of the source data, the project ensures strict reproducibility and compliance with the implementation specifications outlined in `AGENTS.md`.

## 1. Purpose

The primary goal of this directory is to serve as the local offline repository for the version-locked ECG archives. While the baseline program supports downloading official datasets via the terminal by default, storing the `.zip` archives here provides a faster, guaranteed-reproducible alternative (using the `--source zip` ingestion pathway).

## 2. Required Datasets

To fully support the matrix generation and transition fitting pipeline, the following datasets must be downloaded and placed intact into this directory:

| Dataset | Version | PhysioNet/Kaggle Source | Role |
| :--- | :---: | :--- | :--- |
| **PTB-XL** | `v1.0.3` | [PhysioNet PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | Main latency-to-clinician benchmark dataset (`B1` matrix family). |
| **PTB-XL+** | `v1.0.1` | [PhysioNet PTB-XL+ 1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/#files-panel) | Required side-information and metadata harmonization for PTB-XL. |
| **LUDB** | `v1.0.1` | [PhysioNet LUDB 1.0.1](https://physionet.org/content/ludb/1.0.1/) | Gold-standard manual delineation dataset (`B2` matrix family). |

*Note: The Chapman/Shaoxing/Ningbo datasets and MIT-BIH are explicitly not required for building the baseline `B1` and `B2` matrices.*

## 3. Data Management and Pipeline Integration

### Do Not Unzip Manually

Do **not** manually extract these archives into the `raw/` directory. The project pipeline handles unzipping and validation automatically to ensure data integrity.

### Ingestion Workflow

1. Place the untouched official `.zip` files into this `data_lock/` folder.
2. Run the ingestion script using the local archive mode:

   ```bash
   # Example command (depends on the exact CLI implementation)
   python src/tm_ecg/real_data.py ingest --source zip
   ```

3. The pipeline will automatically:
   - Read the archives from `data_lock/`.
   - Verify their cryptographic hashes against `manifests/source_manifest.json`.
   - Unpack the verified contents into the project's `raw/` directory (which is git-ignored).

## 4. Derived Data Directories

To understand how data flows outward from this lock directory, be aware of the following ignored or tracked staging areas:

- `raw/`: Where the pipeline extracts the `.zip` files. (Git-ignored)
- `interim/`: Extracted per-record measurement payloads (e.g., `ptbxl_record_measurements.json`). (Git-ignored)
- `features/`: The final materialized `B_raw` and `B_fit` matrices in `.parquet`, `.csv`, and `.xlsx` formats. (Currently tracked in Git, as it was removed from `.gitignore`).

## 5. Version Control Notice

These `.zip` files can be extremely large. Ensure that if you track this directory via Git, you are using **Git LFS (Large File Storage)**, or simply ensure that standard Git ignores the `.zip` extensions to avoid bloating the repository. Data integrity should be maintained relying on the cryptographic hashes in `manifests/source_manifest.json` rather than direct Git history of the binary files.
