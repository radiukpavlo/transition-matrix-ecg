# B-Matrix Generation Report

This report explains the clinician-facing `B` matrices currently materialized in the repository, how they differ technically, how the underlying datasets are used in the project, and where the deliverables exist in Parquet, CSV, and Excel formats.

For feature-by-feature formulas, null counts, and detailed interpretations of the 52 columns in `B1_raw_train`, see [b1_raw_train_null_report.md](/D:/GitHub/transition-matrix-ecg/reports/b1_raw_train_null_report.md).

## Matrix Families

- `B_raw` is the clinician-readable matrix in native units. It preserves nullable values and keeps the reporting metadata columns `record_id`, `qtc_formula_code`, and `split`.
- `B_fit` is the typed transform of `B_raw` used for transition fitting. It is generated with training-only statistics and uses family-specific transforms implemented in [typed_transforms.py](/D:/GitHub/transition-matrix-ecg/src/tm_ecg/transition/typed_transforms.py).
- In the current run, each `B_raw_*` matrix has `52` columns total, while each `B_fit_*` matrix has `44` columns total because metadata and non-fittable targets are excluded or collapsed for modeling.

## Data

How to download the data?

| Dataset Name | Version | PhysioNet Repository | Kaggle Repository |
| :--- | :---: | :--- | :--- |
| **PTB-XL** | `v1.0.3` | [View on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) | [View on Kaggle](https://www.kaggle.com/datasets/garethwmch/ptb-xl-1-0-3) |
| **PTB-XL+** | `v1.0.1` | [View on PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/#files-panel) | [View on Kaggle](https://www.kaggle.com/datasets/antonymgitau/ptb-xl-a-comprehensive-ecg-feature-dataset) |
| **LUDB** | `v1.0.1` | [View on PhysioNet](https://physionet.org/content/ludb/1.0.1/) | None |

## Dataset Roles In This Project

| Dataset | Project role | Why it is used | Resulting matrix family | Current exported realization |
| --- | --- | --- | --- | --- |
| `PTB-XL v1.0.3 + PTB-XL+ v1.0.1` | Large-scale benchmark dataset for the main latent-to-clinician bridge. | It provides broad cohort scale, official fold structure, 10-second 12-lead signals, and PTB-XL+ side information for feature harmonization and measurement verification. | `B1_*` | `train=10000`, `val=2183`, `test=2198` |
| `LUDB v1.0.1` | Gold-standard delineation dataset for morphology-faithful feature extraction and secondary explanatory validation. | It provides manual P/QRS/T boundaries and peaks, which makes it the preferred source for high-trust clinician feature construction despite its smaller size. | `B2_*` | Current executable fold realization: `train=120`, `val=40`, `test=40` |

PTB-XL/PTB-XL+ and LUDB are therefore not interchangeable in the project:

- `B1` is the scalable benchmark matrix used to mirror the 10,000-row transition-matrix methodology on a modern large ECG cohort.
- `B2` is the smaller but higher-fidelity matrix used when manual delineation quality matters more than sample count.
- The classifier and latent extraction path is anchored on PTB-XL; LUDB is used to generate a second, morphology-grounded `B` space for validation of the explanatory bridge.
- MIT-BIH remains auxiliary only and is not part of the baseline `B1` or `B2` artifact family.

## Medical Protocols And Standards

The project does not treat ECG feature engineering as ad hoc heuristic work. Under the governing specification in [AGENTS.md](/D:/GitHub/transition-matrix-ecg/AGENTS.md), especially the sections `Locked decisions`, `5.1 Standards anchor and evidence discipline`, `5.2 Design principles for the clinician feature space`, and `Appendix A`, the populated ECG features in `B1_*` and `B2_*` are required to be extracted, selected, and calculated against recognized electrocardiographic standards rather than convenience rules.

In short, the standards basis is:

- the AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram, especially Parts II-VI
- ANSI/AAMI EC57 rhythm and ST-measurement evaluation/reporting practice where applicable
- the official dataset documentation for PTB-XL, PTB-XL+, LUDB, and the auxiliary benchmark corpora cited by the specification

In practical terms, [AGENTS.md](/D:/GitHub/transition-matrix-ecg/AGENTS.md) locks the following protocol-critical rules for the feature space:

- `qtc_med_ms` uses Fridericia correction and stores the sidecar code `QTcF`
- ST displacement is measured at `J+60 ms` when `HR >= 100 bpm` and at `J+80 ms` otherwise, always relative to the PR/PQ isoelectric baseline
- `rr_sdnn_ms` is computed only from verified NN intervals after ectopic, paced, and artifact rejection
- right-precordial T-wave inversion requires `<= -0.1 mV` and duration `>= 80 ms`
- frontal QRS axis is derived from net QRS area in leads I and aVF using `atan2`
- U-wave extraction is optional, 500 Hz-only, and forbidden in 100 Hz branches
- `B_fit` is constructed with typed family-specific transforms instead of raw unconstrained least squares

This gives the matrix family a standards-based clinical audit trail: every populated feature is expected to be clinically relevant, explicitly measurable on the waveform or fiducials, numerically precise, and version-locked. In the current artifact set, that standards alignment applies to all populated extracted features. Schema-defined but currently unpopulated items, most visibly some `F24` composite signature scores, remain governed by the same specification and should only be populated using the training-split `L1`-regularized logistic rule defined in `AGENTS.md` Appendix A.

## Difference Between `B1_raw_train`, `B1_raw_val`, and `B1_raw_test`

| Matrix | Same schema? | Split semantics | What it is allowed to influence | What it must not influence |
| --- | --- | --- | --- | --- |
| `B1_raw_train.parquet` | Yes, same 52-column schema as val/test. | Frozen PTB-XL training cohort restricted to exactly 10,000 records. | Training-only transforms, future composite-score fitting, and the `A -> B_fit` operator fit after conversion to `B1_fit_train`. | It must not contain validation/test rows or leak fold-9/fold-10 information into fitted statistics. |
| `B1_raw_val.parquet` | Yes. | Frozen PTB-XL validation cohort from the official fold allocation. | Validation-only model selection, thresholding, and ridge-penalty choice after conversion to `B1_fit_val`. | It must not be used to fit transforms, PCA, composite signatures, or the transition operator. |
| `B1_raw_test.parquet` | Yes. | Final PTB-XL holdout cohort. | Final explanatory evaluation and clinician review after predictions are generated. | It must remain untouched during training and validation-time selection. |

All three matrices share the same column ordering and semantic definitions. The difference is not in the feature definitions but in the split membership, row count, and allowed statistical use.

## Difference Between `B1_*` and `B2_*`

| Aspect | `B1_*` | `B2_*` |
| --- | --- | --- |
| Source dataset | PTB-XL v1.0.3 augmented by PTB-XL+ v1.0.1 | LUDB v1.0.1 |
| Primary purpose | Large-scale benchmark clinician matrix | Gold-standard delineation clinician matrix |
| Annotation basis | Real waveforms plus PTB-XL+ harmonization and waveform-side verification | Real waveforms plus LUDB manual fiducials as source of truth |
| Scale in current run | `10000 / 2183 / 2198` rows across train/val/test | `120 / 40 / 40` rows in the current executable fold realization |
| Strength | Cohort size and benchmark comparability | Fiducial trust and morphology fidelity |
| Limitation | Some feature families are sparse or dependent on heuristic measurement quality | Much smaller cohort size, so statistical power is lower |
| Alignment counterpart | `A_ptbxl_train/val/test` | `A_ludb_train/val/test` |

## Technical Inventory Of All Calculated Matrices

| Matrix | Dataset source | Split | Representation | Rows | Cols | Technical meaning | Parquet | CSV | Excel |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `B1_raw_train` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Train | Raw clinician space | 10000 | 52 | Native-unit benchmark matrix built from the frozen 10,000-record PTB-XL training cohort. Contains 49 clinician features plus `record_id`, `qtc_formula_code`, and `split`. This is the authoritative raw `B` matrix used to fit training-only transforms and to align with `A_ptbxl_train` via `record_id`. | [B1_raw_train.parquet](D:/GitHub/transition-matrix-ecg/features/B1_raw_train.parquet) | [B1_raw_train.csv](D:/GitHub/transition-matrix-ecg/features/B1_raw_train.csv) | [B1_raw_train.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_raw_train.xlsx) |
| `B1_raw_val` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Validation | Raw clinician space | 2183 | 52 | Native-unit validation counterpart of `B1_raw_train` with the identical schema. It is held out from all training-only fitting steps and is used for validation-time model selection, thresholding, and explanation diagnostics only. | [B1_raw_val.parquet](D:/GitHub/transition-matrix-ecg/features/B1_raw_val.parquet) | [B1_raw_val.csv](D:/GitHub/transition-matrix-ecg/features/B1_raw_val.csv) | [B1_raw_val.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_raw_val.xlsx) |
| `B1_raw_test` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Test | Raw clinician space | 2198 | 52 | Native-unit final holdout matrix for PTB-XL fold 10. It has the same column contract as the train and validation raw matrices, but it is reserved for end-of-pipeline evaluation and clinician-facing review of predicted versus measured features. | [B1_raw_test.parquet](D:/GitHub/transition-matrix-ecg/features/B1_raw_test.parquet) | [B1_raw_test.csv](D:/GitHub/transition-matrix-ecg/features/B1_raw_test.csv) | [B1_raw_test.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_raw_test.xlsx) |
| `B1_fit_train` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Train | Typed transformed fit space | 10000 | 44 | Training-space projection of `B1_raw_train` used for transition estimation. Continuous features are winsorized and z-scored, count features use `log1p` then z-score, binary and bounded features use logit-space transforms, `qrs_axis_deg` is replaced by `qrs_axis_sin/cos`, and columns without usable training statistics are dropped. | [B1_fit_train.parquet](D:/GitHub/transition-matrix-ecg/features/B1_fit_train.parquet) | [B1_fit_train.csv](D:/GitHub/transition-matrix-ecg/features/B1_fit_train.csv) | [B1_fit_train.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_fit_train.xlsx) |
| `B1_fit_val` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Validation | Typed transformed fit space | 2183 | 44 | Validation split transformed with the training-only `B1` transform bundle. This matrix is used to select the ridge penalty and to score how well the latent-to-clinician operator generalizes beyond the fitting cohort. | [B1_fit_val.parquet](D:/GitHub/transition-matrix-ecg/features/B1_fit_val.parquet) | [B1_fit_val.csv](D:/GitHub/transition-matrix-ecg/features/B1_fit_val.csv) | [B1_fit_val.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_fit_val.xlsx) |
| `B1_fit_test` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Test | Typed transformed fit space | 2198 | 44 | Test split transformed with the same training-only bundle as `B1_fit_train`. This is the held-out target space for final transition-operator evaluation and inverse mapping back into clinician units. | [B1_fit_test.parquet](D:/GitHub/transition-matrix-ecg/features/B1_fit_test.parquet) | [B1_fit_test.csv](D:/GitHub/transition-matrix-ecg/features/B1_fit_test.csv) | [B1_fit_test.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_fit_test.xlsx) |
| `B2_raw_train` | LUDB v1.0.1 | Train | Raw clinician space | 120 | 52 | Native-unit LUDB matrix for the current executable train fold. It shares the same 52-column schema as `B1_raw_*`, but its values are derived from LUDB manual fiducials and therefore provide a smaller, morphology-grounded reference matrix. | [B2_raw_train.parquet](D:/GitHub/transition-matrix-ecg/features/B2_raw_train.parquet) | [B2_raw_train.csv](D:/GitHub/transition-matrix-ecg/features/B2_raw_train.csv) | [B2_raw_train.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_raw_train.xlsx) |
| `B2_raw_val` | LUDB v1.0.1 | Validation | Raw clinician space | 40 | 52 | Validation counterpart of `B2_raw_train` for the current LUDB fold realization. It stays out of training-only statistics and provides a clean morphology-oriented check on the explanatory pipeline. | [B2_raw_val.parquet](D:/GitHub/transition-matrix-ecg/features/B2_raw_val.parquet) | [B2_raw_val.csv](D:/GitHub/transition-matrix-ecg/features/B2_raw_val.csv) | [B2_raw_val.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_raw_val.xlsx) |
| `B2_raw_test` | LUDB v1.0.1 | Test | Raw clinician space | 40 | 52 | Final LUDB holdout matrix for the current executable fold. It is the manual-annotation-grounded clinician-space target for out-of-sample explanation checks on the smaller gold-standard dataset. | [B2_raw_test.parquet](D:/GitHub/transition-matrix-ecg/features/B2_raw_test.parquet) | [B2_raw_test.csv](D:/GitHub/transition-matrix-ecg/features/B2_raw_test.csv) | [B2_raw_test.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_raw_test.xlsx) |
| `B2_fit_train` | LUDB v1.0.1 | Train | Typed transformed fit space | 120 | 44 | Typed transform of `B2_raw_train` produced with LUDB training-only statistics. The same raw-to-fit rules are applied as in `B1_fit_train`, enabling a consistent transition-operator formulation on the gold-standard delineation dataset. | [B2_fit_train.parquet](D:/GitHub/transition-matrix-ecg/features/B2_fit_train.parquet) | [B2_fit_train.csv](D:/GitHub/transition-matrix-ecg/features/B2_fit_train.csv) | [B2_fit_train.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_fit_train.xlsx) |
| `B2_fit_val` | LUDB v1.0.1 | Validation | Typed transformed fit space | 40 | 44 | Validation fold transformed with the `B2` training bundle. It is used to assess transition-fit stability for the LUDB realization without leaking validation rows into training statistics. | [B2_fit_val.parquet](D:/GitHub/transition-matrix-ecg/features/B2_fit_val.parquet) | [B2_fit_val.csv](D:/GitHub/transition-matrix-ecg/features/B2_fit_val.csv) | [B2_fit_val.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_fit_val.xlsx) |
| `B2_fit_test` | LUDB v1.0.1 | Test | Typed transformed fit space | 40 | 44 | Held-out LUDB fit-space matrix used for final gold-standard transition evaluation within the currently exported fold realization. | [B2_fit_test.parquet](D:/GitHub/transition-matrix-ecg/features/B2_fit_test.parquet) | [B2_fit_test.csv](D:/GitHub/transition-matrix-ecg/features/B2_fit_test.csv) | [B2_fit_test.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_fit_test.xlsx) |

## How The Matrices Are Built

The matrix-construction path is stage-based and is implemented primarily in [real_data.py](/D:/GitHub/transition-matrix-ecg/src/tm_ecg/real_data.py), [features.py](/D:/GitHub/transition-matrix-ecg/src/tm_ecg/stages/features.py), and [typed_transforms.py](/D:/GitHub/transition-matrix-ecg/src/tm_ecg/transition/typed_transforms.py).

1. `ingest --source zip` unpacks the locked ECG archives into [raw](/D:/GitHub/transition-matrix-ecg/raw) and records source hashes in [source_manifest.json](/D:/GitHub/transition-matrix-ecg/manifests/source_manifest.json).
2. `index` builds metadata indexes for PTB-XL and LUDB in [manifests](/D:/GitHub/transition-matrix-ecg/manifests).
3. `splits --dataset ptbxl|ludb` freezes row membership. PTB-XL is reduced to the locked `10,000`-row training cohort for `B1_train`; LUDB is materialized as the current executable train/val/test fold realization used for `B2_*` export.
4. `triads --dataset ptbxl|ludb` extracts per-record measurement payloads into [interim](/D:/GitHub/transition-matrix-ecg/interim), for example [ptbxl_record_measurements.json](/D:/GitHub/transition-matrix-ecg/interim/ptbxl_record_measurements.json).
5. `build-b --dataset b1|b2` computes the Appendix A clinician features in native units and writes the `B_raw_*` matrices to [features](/D:/GitHub/transition-matrix-ecg/features).
6. `fit-transition --dataset b1|b2` fits the training-only typed transform bundle and emits the `B_fit_*` matrices used for the reduced-rank ridge transition operator.

## Raw Space Versus Fit Space

| Property | `B_raw_*` | `B_fit_*` |
| --- | --- | --- |
| Units | Native clinician units such as `ms`, `mV`, counts, binary flags, and axis degrees. | Modeling-space values after family-specific transforms and standardization. |
| Null handling | Nullable values are preserved as measured. | Columns with no usable training statistics are dropped; remaining rows are transformed with training-only statistics. |
| Axis handling | Keeps `qrs_axis_deg` for clinician readability. | Uses `qrs_axis_sin` and `qrs_axis_cos` instead of fitting the raw angle directly. |
| Primary consumer | Clinicians, review packets, direct inspection, export. | Transition operator fitting and prediction in typed Euclidean space. |
| Row alignment key | `record_id` | `record_id` |

## Exported Deliverables

Every `.parquet` file in [features](/D:/GitHub/transition-matrix-ecg/features) has now been exported to two sibling formats:

- `.csv` for plain tabular interoperability
- `.xlsx` for spreadsheet-based review in Excel-compatible tools

The Excel files are single-sheet equivalents of the Parquet feature tables and preserve the same header order and row content. The dictionary files already existed as CSV and were not part of the Parquet-to-CSV/XLSX conversion request.

## Direct File References

If you need the main deliverables immediately, use these files:

- Raw benchmark training matrix: [B1_raw_train.parquet](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.parquet), [B1_raw_train.csv](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.csv), [B1_raw_train.xlsx](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.xlsx)
- Raw benchmark validation matrix: [B1_raw_val.parquet](/D:/GitHub/transition-matrix-ecg/features/B1_raw_val.parquet), [B1_raw_val.csv](/D:/GitHub/transition-matrix-ecg/features/B1_raw_val.csv), [B1_raw_val.xlsx](/D:/GitHub/transition-matrix-ecg/features/B1_raw_val.xlsx)
- Raw benchmark test matrix: [B1_raw_test.parquet](/D:/GitHub/transition-matrix-ecg/features/B1_raw_test.parquet), [B1_raw_test.csv](/D:/GitHub/transition-matrix-ecg/features/B1_raw_test.csv), [B1_raw_test.xlsx](/D:/GitHub/transition-matrix-ecg/features/B1_raw_test.xlsx)
- Gold-standard raw training matrix: [B2_raw_train.parquet](/D:/GitHub/transition-matrix-ecg/features/B2_raw_train.parquet), [B2_raw_train.csv](/D:/GitHub/transition-matrix-ecg/features/B2_raw_train.csv), [B2_raw_train.xlsx](/D:/GitHub/transition-matrix-ecg/features/B2_raw_train.xlsx)
- Gold-standard raw validation matrix: [B2_raw_val.parquet](/D:/GitHub/transition-matrix-ecg/features/B2_raw_val.parquet), [B2_raw_val.csv](/D:/GitHub/transition-matrix-ecg/features/B2_raw_val.csv), [B2_raw_val.xlsx](/D:/GitHub/transition-matrix-ecg/features/B2_raw_val.xlsx)
- Gold-standard raw test matrix: [B2_raw_test.parquet](/D:/GitHub/transition-matrix-ecg/features/B2_raw_test.parquet), [B2_raw_test.csv](/D:/GitHub/transition-matrix-ecg/features/B2_raw_test.csv), [B2_raw_test.xlsx](/D:/GitHub/transition-matrix-ecg/features/B2_raw_test.xlsx)

## Bottom Line

If you need the main raw benchmark matrix with selected clinician features as columns and exactly `10,000` training samples, use [B1_raw_train.parquet](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.parquet) or its exported siblings [B1_raw_train.csv](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.csv) and [B1_raw_train.xlsx](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.xlsx). For detailed per-column formulas, null counts, and interpretation, consult [b1_raw_train_null_report.md](/D:/GitHub/transition-matrix-ecg/reports/b1_raw_train_null_report.md).
