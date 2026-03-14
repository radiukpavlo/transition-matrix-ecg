# AGENTS.md - Implementation Specification for Explainable ECG Arrhythmia Analysis with Clinician-Facing Feature Matrices, Reduced-Rank Latent Spaces, and Regularized Transition Operators

## 2. Executive overview

This specification supersedes the earlier version and converts the uploaded thesis and eight companion papers into a stricter, implementation-grade research blueprint for ECG analysis, arrhythmia detection, and explainable deep learning. It explicitly resolves the methodological, statistical, and clinical validity issues listed in the review block that accompanied the rewrite request.

The baseline program is:

1. **Download official datasets via terminal by default**, but keep the option to use version-locked ECG archives.
2. **Preprocess ECGs through two distinct branches**: a morphology-preserving diagnostic branch and a detection-optimized branch.
3. **Detect R-peaks with a knowledge-integrated, adaptive-window method**, not a hard-coded 260-sample scan.
4. **Delineate P/QRS/T fiducials** and reject low-confidence beats before feature construction.
5. **Represent each object with a triad of neighboring cardio cycles**, then aggregate triad latents to a record-level deep representation.
6. **Construct matrix `A` from preactivation penultimate latent vectors**, reduce its dimension, and regularize it before fitting any transition operator.
7. **Construct matrix `B` from clinically meaningful ECG features** whose thresholds and operational definitions are anchored to the uploaded corpus and to external electrocardiographic standards.
8. **Fit a typed transition operator on a transformed target space `B_fit`**, so binary, bounded, count, and continuous features are treated mathematically correctly.
9. **Evaluate both prediction quality and explanation quality with confidence intervals, agreement metrics, and null-model statistical tests**.
10. **Treat Equivariant Transition Matrices (ETM) as a future extension only**, and forbid clinically invalid temporal scaling transformations.

The scientific backbone remains the uploaded corpus:

- the **thesis** defines the end-to-end goal: robust arrhythmia classification plus interpretation by features used in medical practice;
- the **R-peak paper** supplies the knowledge-integration principle and shows that peak localization is the mandatory anchor for all downstream cycle-based reasoning;
- the **feature-vector paper** supplies a clinician-legible ontology of amplitudes, durations, RR context, and waveform fragments;
- the **arrhythmia-classification paper** and the **integrated transparent-AI paper** establish the triad-of-cycles idea and the feature-wise interpretation logic;
- the **transition-matrix papers** provide the `A -> B` explanatory bridge; and
- the **equivariance paper** provides a future pathway to symmetry-aware explanation stability.

### Locked decisions in this version

1. **Dataset 1 is PTB-XL v1.0.3 augmented by PTB-XL+ v1.0.1.**
2. **Dataset 2 is LUDB v1.0.1.**
3. **Chapman/Shaoxing/Ningbo is not used to build `B2`**, because it lacks the fiducial ground truth required for a morphology-faithful expert feature matrix.
4. **`B1` is the large-scale benchmark matrix** and is fixed to exactly **10,000 training records** to remain comparable to the `10,000-row` transition-matrix methodology in the 2024 Mathematics paper.
5. **`B2` is the gold-standard delineation matrix** and therefore uses **all eligible LUDB records** rather than artificial duplication to 10,000 rows.
6. **The primary QT correction is Fridericia (`QTcF`)**, explicitly named and stored in metadata.
7. **ST displacement is measured at J+60 ms when HR >= 100 bpm and at J+80 ms otherwise**, relative to a PR/PQ isoelectric baseline.
8. **`rr_sdnn_ms` is computed only from verified NN intervals** after ectopic and paced beat rejection.
9. **U-wave extraction is optional and 500 Hz-only**; it is prohibited in any 100 Hz processing branch.
10. **Matrix `A` uses preactivation latent features**, not post-ReLU activations.
11. **Transition fitting uses reduced-rank, ridge-regularized estimation with explicit singular-value truncation.**
12. **Binary and bounded features are never fit in raw 0/1 or raw proportion space.**
13. **All scaling, PCA, imputation, score fitting, and transition estimation are fit on training data only.**

### Deliverables defined by this specification

The implementation team must produce at least the following artifacts for each experiment:

- `A_train.parquet`, `A_val.parquet`, `A_test.parquet`
- `B_raw_train.parquet`, `B_fit_train.parquet`, and their validation/test analogues
- `B_dictionary.csv`
- `T_ridge.npz` (or equivalent serialized operator package)
- `ontology_mapping.csv`
- `split_manifest.json`
- `metrics_report.md`
- `bootstrap_ci_report.csv`
- `error_analysis_workbook` or equivalent review packet

The present Markdown file is the governing specification for producing those artifacts.

## 3. Integrated synthesis of the uploaded thesis and papers

The uploaded corpus is not nine unrelated publications. It is a single escalating program that moves from **domain knowledge integration**, to **cycle-centered ECG representation**, to **triad-aware deep classification**, to **clinician-facing interpretation**, and finally to **transition-matrix-based explainability**.

The thesis states the core research problem plainly: improve arrhythmia classification from ECG using deep learning, then interpret those results through the features that clinicians actually use in practice. It also states the three core scientific contributions of the program: a new R-peak detector using a synchronized auxiliary signal, a triad-based arrhythmia classifier, and a method for interpreting deep-learning decisions through medically used ECG features. The thesis further makes the deployment intention explicit by reporting a nine-class client-server information system intended for practical clinical use.

The 2022 **human-in-the-loop** paper is the philosophical root of the project. It does not yet provide the final arrhythmia pipeline, but it establishes the premise that human knowledge should not be bolted on after the model is finished. Instead, clinically meaningful structures - P and T peaks, the QRS complex, and PQ and ST segments - can be embedded into the data representation itself. That observation justifies the later knowledge-integration approach used for R-peak localization and the broader clinician-facing mental model used for explanation.

The 2023 **R-peak detection** paper converts that philosophy into the first operational module. It introduces a three-stage detector: knowledge integration into the ECG, CNN processing, and post-processing. Its crucial insight is that R-peak identification should be supported by domain regularities rather than left entirely to a generic detector. In the original paper, the knowledge signal `K` is built through fixed-length scanning and local-max verification. The specification preserves the **knowledge-channel principle** but replaces the clinically brittle fixed scan length with an adaptive, heart-rate-aware search process.

The 2023 **feature-vector** paper supplies the clinician-understandable vocabulary that later becomes matrix `B`. It shows that one can represent each cycle not only as a raw waveform but as a structured vector comprising wave amplitudes, wave durations, the PR/PQ interval, QRS duration, ST duration, T duration, RR context, and signal fragments for the P-wave, QRS complex, ST segment, and T wave. This paper is indispensable because it proves that a human-readable feature space can remain predictive while being much easier for clinicians to audit than opaque latent channels.

The 2024 **arrhythmia-classification** paper makes the strongest ECG-specific argument for contextual classification. It argues that a single current beat is not always enough, especially for phenomena such as PVCs where the diagnostic pattern depends on what happens before and after the current beat. This is the origin of the **triad of cardio cycles**: preceding, current, and following. The same paper also defines a pragmatic interpretation workflow: determine the attention zone, then decide how the feature can be confirmed - visually, statistically, formulaically, through visual analytics, via a small ML model, or via a dedicated DL model. That exact logic is reused here for the formal construction of matrix `B`.

The 2025 **integrated transparent-AI** paper consolidates the prior ECG work into a single pipeline: robust R-peak detection, triad-based deep classification, and explanation through clinically meaningful features. It is the most directly applicable ECG systems paper in the corpus because it connects the theory to actual clinical trials, reports kappa agreement values for both R-peak detection and classification, and demonstrates how clinicians can be shown feature-level evidence rather than only class labels. It also makes a key limitation explicit: if only a few features are available, overlapping pathologies can still remain under-explained. That limitation motivates the broader feature ontology in this revision.

The 2024 **transition-matrix** paper supplies the mathematical bridge. It defines the idea that a deep formal model can be connected to a more interpretable mental model by building matrices `A` and `B` over the **same set of training objects** and solving for a transition operator `T`. In the paper, this is shown on MNIST, FNC-1, and Iris. The role of that paper in the current ECG project is not the particular benchmark datasets it uses, but the formalism: the same objects must populate both feature spaces, and the transition operator must be trained only after those spaces are rigorously aligned.

The 2024 **healthcare transition-matrix** paper adapts that formalism to medicine. Its critical innovation is that matrix `B` need not come from another conventional ML model. Instead, `B` can be built directly from **expert-defined user-friendly features**. This is the direct methodological authorization for constructing ECG matrix `B` from clinical ECG features rather than from a separate shallow classifier. It also lays out the sequential logic of matrix-B construction: list the features, define their intervals and meaning, decide how to measure each feature, validate measurability, and only then assemble the matrix.

The 2026 **equivariant transition-matrix** manuscript extends the transition operator by imposing symmetry consistency under small transformations. This is conceptually attractive for ECG explanation stability, but its validation in the uploaded manuscript is on synthetic data and MNIST rather than on ECG. The manuscript also explicitly acknowledges limitations: the linear transition assumption may fail in complex settings, finite-difference generator estimation can be fragile, and the human-in-the-loop integration remains unexplored. For these reasons, ETM is not adopted as a baseline requirement here. It is reserved for a later phase once the ECG-specific baseline is stable, validated, and clinically credible.

The unified interpretation is therefore as follows:

1. **From the thesis and ECG papers**: the medically meaningful ECG object is an R-anchored cardio cycle, and diagnostically meaningful context often spans neighboring cycles.
2. **From the feature-vector paper**: clinician reasoning can be encoded numerically through amplitudes, durations, RR context, and selected signal fragments.
3. **From the transition-matrix papers**: a deep model's latent state can be translated into clinician-readable features if `A` and `B` are built over the same samples and linked by a well-defined operator.
4. **From the ETM paper**: explanation operators should eventually be stable under clinically benign transformations, but not at the cost of violating physiology.

That integrated reading yields the blueprint that follows.

## 4. Systemized cross-document contribution map

| Source | Research goal | Data type | Preprocessing / preparation | Feature extraction | Classifier / formal model | Explainability method | Evaluation protocol | Contribution to blueprint |
|---|---|---|---|---|---|---|---|---|
| 01_Thesis.pdf | Umbrella ECG XAI program | ECG | R-peak detection; triad construction; client-server deployment | Medical-practice features | Deep learning for 9 classes | Interpretation of DL results via clinical ECG features | Academic results plus clinical-system implementation | Defines the end-to-end research agenda and nine-class deployment target |
| 02_Human-in-the-loop-approach.pdf | Human-centric knowledge integration | ECG and MRI | Embedding clinical knowledge into signal/image | P/T peaks, QRS, PQ, ST | Proof-of-concept CNN/autoencoder pathway | HITL framing | Visual analysis | Supplies the knowledge-integration philosophy for ECG preprocessing |
| 03_A-novel-feature-vector-for-ECG.pdf | Transparent ECG feature-vector design | ECG (MIT-BIH derivative subset) | db6 wavelet denoising; beat segmentation by R-peaks | 195-element feature vector of amplitudes, durations, RR fragments, waveform fragments | DL classifier on handcrafted features | Transparency through clinician-style inputs | Inter-patient evaluation | Primary source for clinician-facing feature ontology |
| 04_ECG-arrhythmia-classification.pdf | Triad-based arrhythmia classification and feature interpretation | ECG (MIT-BIH) | R-peak-based fragmentation; triad of cycles | Attention zones; formula/PCA/ML/DL confirmation of features | Improved CNN with BatchNorm and extra conv layer | Feature-wise interpretation after classification | Accuracy, precision, recall, F1 | Supplies the triad logic and interpretation workflow |
| 05_Robust-R-peak-detection.pdf | Robust R-peak detection with domain knowledge | ECG (MIT-BIH, QT, CPSC-2020, UoG) | Knowledge channel K + CNN + post-processing | R-peak localization | CNN detector | Domain-guided preprocessing, not post-hoc XAI | Performance under +/-25 ms tolerance | Critical first-stage detector for all downstream features |
| 06_Towards-transparent-AI.pdf | Integrated transparent ECG pipeline | ECG | Knowledge-integrated R-peak detection; triad classifier; clinical raster-to-signal preprocessing | Normal/PVC/RBBB/LBBB/Fusion feature explanations | Modified CNN | Clinician-facing interpretation with clinical trials | Benchmark comparison and clinical trials; kappa | Most complete ECG-specific integrated implementation paper |
| 07_Explainable-deep-learning.pdf | Generic transition-matrix theory | Benchmark image/text/tabular data | Not ECG-specific | Matrix A from DL; matrix B from mental/interpretable model | Any formal DL model | Transition matrix via pseudo-inverse / SVD | Qualitative and quantitative fidelity metrics | Mathematical basis for A, B, T |
| 08_Toward-explainable-deep-learning.pdf | Scalable healthcare transition-matrix method | ECG and MRI | Matrix B construction steps; same training objects in A and B | Expert-defined user-friendly features | DLECG and DLMRI | Transition matrix from deep features to expert features | Cohen kappa plus CI/p-values | Direct methodological authority for matrix B construction in healthcare |
| 09_Equivariant-transition-matrices.pdf | Symmetry-aware post-hoc explanation | Synthetic + MNIST | Perturbation-based generator estimation | Formal and mental features under small transformations | Post-hoc ETM | Equivariance-constrained transition operator | MSE, SSIM, PSNR, SymDef | Future-stage extension, not baseline ECG requirement |

## 5. ECG feature ontology for clinicians and deep learning systems

### 5.1 Standards anchor and evidence discipline

This feature ontology is not allowed to rely only on the uploaded ECG papers. The corpus provides the project-specific logic, but the **operational thresholds, interval definitions, and morphology criteria** must also be anchored to recognized electrocardiographic standards. Accordingly, the feature space in this specification is constrained by:

- the uploaded thesis and ECG papers (`01`-`06`);
- the transition-matrix methodology papers (`07`-`09`);
- the **AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram**, especially Parts II-VI;
- ECG algorithm evaluation and reporting practice such as **ANSI/AAMI EC57** where applicable; and
- official dataset documentation for PTB-XL, PTB-XL+, LUDB, and MIT-BIH.

Every feature in this document is labeled implicitly by three evidence layers:

- **Grounded**: directly supported by the uploaded corpus;
- **Implementation inference**: required to make the grounded idea executable and reproducible; and
- **Optional extension**: useful but not mandatory for the baseline implementation.

### 5.2 Design principles for the clinician feature space

The expert feature space must satisfy all of the following conditions:

1. **Clinical relevance**: each feature must correspond to something a cardiologist or trained ECG reader can justify conceptually.
2. **Measurability**: every feature must have an explicit extraction path from the digital waveform or its fiducials.
3. **Distinctiveness**: the feature should add information rather than duplicate many others.
4. **Auditability**: a human should be able to trace where the value came from.
5. **Compatibility with deep explanation**: the feature should be stable enough to be predicted from latent representations without becoming meaningless after inverse transformation.
6. **Versionability**: each feature needs a locked definition, a unit, a normalization rule, and a missingness rule.

### 5.3 Consolidated ECG feature inventory

| Feature family | Clinical meaning | Direct or derived | Unit | Extraction source | Arrhythmia relevance | Usefulness for clinicians | Usefulness for DL | Support in corpus |
|---|---|---|---|---|---|---|---|---|
| P-wave presence / absence | Atrial depolarization before QRS; absence supports ectopy/AF reasoning | Derived | binary or proportion | P delineation in lead II with supporting leads | PVC/APB/AF context | Very high | High | 03, 04, 06, 08 |
| P-wave amplitude | Magnitude of atrial depolarization | Direct | mV | P peak relative to isoelectric baseline | Atrial conduction and visibility | High | Moderate | 03 |
| P-wave duration | Atrial conduction duration | Direct | ms | P onset to P offset | Atrial enlargement/conduction context | High | Moderate | 03 |
| PR / PQ interval | AV conduction time | Direct | ms | P onset to QRS onset | AV delay, atrial timing | Very high | High | 03, 02 |
| Q/R/S amplitudes | Depolarization morphology | Direct | mV | QRS fiducials | PVC/BBB morphology | High | High | 03 |
| QRS duration | Width of ventricular depolarization | Direct | ms | QRS onset to offset | PVC/RBBB/LBBB/paced | Very high | Very high | 03, 04, 06, 08 |
| QRS deformation probability | Abnormal ventricular morphology beyond simple width | Derived | probability | Attention zone around QRS | PVC/fusion/BBB | Very high | Very high | 04, 06, 08 |
| QRS fragmentation / notching | Conduction heterogeneity and abnormal depolarization | Derived | binary | QRS fragment morphology in precordial leads | BBB/scar/conduction disorder context | High | High | Implementation inference anchored to 04/06 |
| ST level | Repolarization deviation relative to baseline | Direct | mV | J+60/J+80 sample relative to PR baseline | Ischemia/ST-T abnormalities | Very high | High | 03 plus external AHA |
| ST slope | Direction of early repolarization trajectory | Derived | uV/ms | Regression from J to J+offset | ST depression/elevation characterization | High | Moderate | Implementation inference anchored to 03 and standards |
| T-wave amplitude | Magnitude of repolarization | Direct | mV | T peak relative to baseline | Repolarization abnormalities | High | High | 03 |
| T-wave duration | Temporal width of repolarization | Direct | ms | T onset to T offset | Repolarization changes | High | Moderate | 03 |
| T-wave inversion | Pathologic negative repolarization | Derived | binary | Right precordial / selected leads | RBBB/ST-T abnormalities | Very high | High | Implementation inference anchored to 06 and AHA |
| QT interval | Total ventricular depolarization + repolarization duration | Direct | ms | QRS onset to T offset | Long/short QT risk | Very high | High | External AHA + standard ECG practice |
| QTc (Fridericia default) | Heart-rate-corrected QT | Derived | ms | QT and RR | Rate-adjusted repolarization | Very high | High | External evidence; reproducibility lock |
| RR interval statistics | Rhythm regularity | Direct/derived | ms | R-peak series | PVC/APB/AF/normal rhythm | Very high | Very high | 03, 04, 05, 06, 08 |
| Compensatory pause ratio | PVC hallmark based on surrounding RR intervals | Derived | ratio | RR before and after ectopic beat | PVC discrimination | Very high | Very high | 04, 06, 08 |
| f-wave power ratio | Baseline atrial fibrillatory activity during T-Q interval | Derived | ratio | T-Q interval spectral power 4-10 Hz | AF discrimination | Very high | High | Implementation extension required by clinical gap |
| Frontal QRS axis | Electrical axis of ventricular depolarization | Derived | degrees / sin / cos | Net QRS area in I and aVF | Conduction disorder and chamber strain context | High | High | PTB-XL metadata + standard ECG practice |
| Bundle-branch signature scores | Composite rule-based conduction evidence | Derived | log-odds score | Clinically constrained feature subset | RBBB/LBBB identification | Very high | High | Implementation inference anchored to 06/08 |
| Paced-beat evidence | Evidence for pacing | Derived | count/score | Pacemaker metadata + pacing spike detector + QRS morphology | Paced rhythm identification | Very high | High | 06 + external pacing-artifact literature |
| U-wave features | Late repolarization / electrolyte effect indicator | Direct | binary / mV | High-resolution 500 Hz signal only | Supplementary repolarization context | Moderate | Low | Optional extension; 500 Hz only |
| Waveform fragments | Fine-grained shape information for P/QRS/ST/T | Direct | sample arrays | Beat-centered fragments | Transparent shape descriptors | Moderate | Very high | 03 |
| Triad latent representation | Contextual representation of preceding-current-following cycles | Derived | latent vector | Three-cycle encoder | PVC and contextual arrhythmias | Indirect | Very high | Thesis, 04, 06 |

### 5.4 Deep-learning-aligned feature representations

The deep model does not need to predict exactly the same primitives that clinicians read directly from paper ECGs. However, its latent structure must be alignable with those primitives. The corpus suggests four classes of deep-learning-aligned representations:

1. **R-anchored cycle embeddings**: latent vectors centered on robust R-peak timing.
2. **Triad embeddings**: contextual latent vectors for preceding-current-following cycles.
3. **Morphology attention-zone embeddings**: local embeddings specialized to P, QRS, ST, or T fragments.
4. **Record-level pooled embeddings**: robust aggregation of triad embeddings across a 10-second 12-lead record.

The specification uses all four. The clinician never sees the latent vector directly. Instead, the latent vector contributes to a translation into the feature blocks that mirror cardiology reasoning:

- rhythm regularity and ectopy burden;
- atrial evidence;
- ventricular depolarization width and morphology;
- ST/T repolarization;
- axis and conduction signatures; and
- syndrome-level composite scores.

### 5.5 Negative design rules

The following choices are prohibited:

- using undefined binary features without thresholds;
- computing QTc without naming the correction formula;
- using raw RR-based SDNN on ectopic-containing sequences and calling it SDNN;
- measuring ST displacement at an unspecified “fixed point” after the J point;
- extracting U-wave features from 100 Hz signals;
- using median imputation for missing leads in the main benchmark matrices; and
- introducing feature thresholds that are not either grounded in the corpus or explicitly tied to recognized ECG standards.

## 6. Clinician-understandable mental model based on transition matrices

### 6.1 Interpretable-space definition

In this project, the **interpretable space** is not an arbitrary feature bank. It is a structured clinical reasoning surface built to answer the same questions a healthcare professional asks when reading an ECG:

1. **Is the signal reliable enough to trust?**
2. **Is the rhythm regular or irregular?**
3. **Is there atrial evidence before the ventricular event?**
4. **Is the QRS narrow, wide, normal, deformed, fragmented, or paced?**
5. **Do the ST segment and T wave support a repolarization abnormality?**
6. **Does the frontal axis or precordial morphology support a conduction syndrome?**
7. **Do the combined features support a recognizable syndrome such as PVC, AF, RBBB, LBBB, or paced rhythm?**

Matrix `B` is the numerical embodiment of this reasoning surface.

### 6.2 Matrix `A`: the formal-model latent representation

Matrix `A` represents the formal deep model (FM). In the ECG implementation:

- each accepted central beat is embedded together with its preceding and following beats, producing a **triad latent vector** `a_triad^(i)`;
- the penultimate-layer vector must be extracted **before** the final nonlinearity (preactivation);
- the recommended preactivation width is **512** features, not thousands of unconstrained activations;
- all triad latent vectors belonging to the same 10-second ECG record are then aggregated into a **record-level latent vector** `a_record`.

The baseline aggregation operator is:

`a_record = mean_trim_10%( a_triad^(1), ..., a_triad^(n) )`

where `mean_trim_10%` denotes a robust trimmed mean over triads. Two ablation alternatives are allowed:

- `max` pooling, and
- attention-weighted pooling.

The baseline trimmed mean is locked for the main report because it is robust, deterministic, and easy to audit.

To stabilize the transition fit, `A` is processed in four steps:

1. remove zero-variance latent dimensions;
2. standardize the remaining dimensions using training-only statistics;
3. apply PCA to retain 99% of the training variance, subject to a maximum retained rank chosen on validation data; and
4. truncate near-zero singular values explicitly during transition estimation.

This yields `A_red`, the reduced-rank version of the latent space used for fitting `T`.

### 6.3 Matrix `B`: the mental-model / clinician-feature representation

Matrix `B` represents the clinician-facing mental model (MM). It is stored in two synchronized forms:

- `B_raw`: human-readable clinical values in native units (`ms`, `mV`, counts, probabilities, angles);
- `B_fit`: typed transformed values used for operator fitting.

This distinction is essential. Clinicians review `B_raw`. The transition operator is fit on `B_fit`, because a single Euclidean least-squares objective is not valid on heterogeneous targets unless the target families are transformed appropriately.

### 6.4 The transition operator

The original transition-matrix papers solve `AT ≈ B` using a pseudo-inverse. That idea is retained, but the implementation is tightened substantially.

#### 6.4.1 Why the naive formulation is insufficient

A raw least-squares fit from `A` to a mixed `B` containing:

- continuous measurements (e.g., `qrs_dur_med_ms`),
- binary flags (e.g., `qrs_deformed_any`),
- bounded proportions (e.g., `p_present_ratio`),
- counts (e.g., `pvc_like_beat_count`),

is mathematically incoherent if all targets are treated in the same raw scale. The implementation therefore fits the operator on a **typed transformed target space**.

#### 6.4.2 Typed transformation of `B`

For each feature column `b_j`:

- if `b_j` is continuous: winsorize on training data and z-score;
- if `b_j` is a count: apply `log(1 + x)`, then z-score;
- if `b_j` is binary: apply epsilon-smoothing followed by a logit link, then z-score;
- if `b_j` is a bounded proportion in `(0,1)`: clip to `[eps, 1-eps]`, apply the logit link, then z-score;
- if the feature is circular (frontal axis): store `qrs_axis_deg` for reporting but fit only `sin(theta)` and `cos(theta)`.

This produces `B_fit`.

#### 6.4.3 Reduced-rank ridge solution

Let `A_red in R^(m x r)` be the reduced latent matrix on the training split, and `B_fit in R^(m x l)` the transformed expert-feature matrix on the same rows. The baseline transition operator is:

`T = argmin_T ||A_red T - B_fit||_F^2 + lambda ||T||_F^2`

with closed form:

`T = (A_red^T A_red + lambda I)^(-1) A_red^T B_fit`

or, equivalently via truncated SVD:

`A_red = U_r Sigma_r V_r^T`

`T = V_r diag( sigma_i / (sigma_i^2 + lambda) ) U_r^T B_fit`

Only singular values satisfying

`sigma_i > max(m, r) * eps_machine * sigma_1`

may be retained. Lower singular values must be discarded.

This keeps the model within the transition-matrix family while avoiding the instability of a raw pseudo-inverse.

#### 6.4.4 Inverse maps back to clinician space

For a new latent vector `a*`, compute:

`z* = a* T`

Then recover `B_hat_raw` feature-wise:

- continuous: inverse z-score (and reverse winsorization only conceptually; do not unclip),
- counts: inverse z-score then `exp(z) - 1`,
- binary: inverse z-score, logistic projection, threshold at the validation-calibrated cutoff,
- proportions: inverse z-score then logistic projection,
- circular axis: recover `theta_hat = atan2( sin_hat, cos_hat )`.

This produces a clinician-readable predicted feature vector `B_hat_raw`.

#### 6.4.5 Optional sensitivity analysis

A secondary sensitivity analysis may replace the single ridge map with a family-specific bridge:

- ridge heads for continuous variables,
- logistic heads for binary variables,
- beta- or logit-space heads for proportions.

That sensitivity analysis is optional. The main report must remain with the single typed transition operator described above for comparability and cognitive clarity.

### 6.5 What makes a feature understandable

A feature is considered understandable only if all five conditions hold:

1. the clinician can name it in ordinary ECG language;
2. the extraction point on the waveform is inspectable;
3. the measurement rule is numerically precise;
4. the feature contributes to a recognizable pathophysiologic narrative; and
5. the model can predict it without producing non-physiologic values after inverse transformation.

### 6.6 How healthcare professionals use the mental model

The clinical reasoning loop that the explainer must support is:

1. inspect signal quality and pacing artifacts;
2. inspect RR regularity and ectopic burden;
3. inspect P-wave presence and AV timing;
4. inspect QRS width, deformation, fragmentation, R' patterns, and broad terminal forces;
5. inspect ST displacement and T-wave inversion;
6. inspect frontal axis;
7. inspect syndrome scores;
8. compare model-implied features `B_hat_raw` with measured features `B_raw`;
9. accept, challenge, or qualify the model's class prediction.

### 6.7 ETM classification for this project

The **Equivariant Transition Matrix** idea is classified here as an **advanced future enhancement**, not a baseline requirement.

It may be explored only after the baseline transition operator is stable and only with transformations that do **not** create false physiology. The allowed future perturbations are:

- small baseline offsets,
- mild global amplitude gain changes (for example +/-5%),
- low-amplitude isoelectric noise injection.

The following are explicitly forbidden in ETM experiments for ECG:

- linear temporal scaling,
- generic time warping,
- interval stretching that alters PR, QRS, or QT morphology.

Those transformations are not benign symmetries in ECG; they create artificial pathologies rather than clinically acceptable invariances.

## 7. Selection and justification of two modern ECG datasets

### 7.1 Final dataset selection

The two datasets selected for matrix-`B` construction are:

1. **Dataset 1: PTB-XL v1.0.3 + PTB-XL+ v1.0.1**
2. **Dataset 2: LUDB v1.0.1**

### 7.2 Why these two datasets are selected

`PTB-XL` is the modern large-scale, multi-label, 12-lead benchmark that best supports reproducible explainability research. It provides 21,799 clinical 12-lead ECGs of 10-second duration from 18,869 patients, raw waveforms at 500 Hz (plus downsampled 100 Hz versions), SCP-ECG statements, signal-quality metadata, a pacemaker flag, and patient-respecting stratified folds with recommended use of folds 1-8 for training, fold 9 for validation, and fold 10 for test. `PTB-XL+` then adds what the original PTB-XL release lacks for a clinician-feature matrix: harmonized tabular ECG features, median beats, fiducial points, and feature descriptions with common naming and units. That combination makes PTB-XL/PTB-XL+ the correct primary dataset for a modern matrix `B`.

`LUDB` is selected as Dataset 2 because it solves the exact weakness that makes Chapman unsuitable for this role: LUDB provides **manual cardiologist annotations of P-wave, QRS-complex, and T-wave boundaries and peaks** across 200 10-second 12-lead recordings, digitized at 500 Hz. That makes LUDB a gold-standard morphology and delineation dataset for validating matrix-`B` features that depend on exact fiducials.

### 7.3 Explicit rejection of Chapman/Shaoxing/Ningbo as `B2`

The Chapman/Shaoxing/Ningbo database remains a useful external transportability dataset for large-scale rhythm modeling, but it is **not** selected for `B2` in this specification. The reason is not that Chapman is a bad ECG dataset. The reason is narrower and stricter: a clinician-grounded matrix `B` intended to contain explicit amplitudes, durations, intervals, ST measurements, axis calculations, and wave-level morphology should not rely entirely on automatically inferred fiducials when a cardiologist-annotated alternative is available. LUDB provides that alternative.

### 7.4 Dataset comparison table

| Criterion | PTB-XL + PTB-XL+ | LUDB | MIT-BIH | Chapman/Shaoxing/Ningbo |
|---|---|---|---|---|
| Public availability | Yes | Yes | Yes | Yes |
| Modern suitability | Very high | High for morphology validation | Moderate; classic but older and mostly 2-lead | High for scale, lower for fiducial truth |
| Lead configuration | 12-lead | 12-lead | 2-channel ambulatory | 12-lead |
| Sampling rate | 500 Hz (+100 Hz copy) | 500 Hz | 360 Hz | 500 Hz |
| Label richness | SCP diagnostic, form, rhythm statements | Diagnosis + rhythm metadata | Beat annotations | Record-level rhythm/condition labels |
| Fiducial ground truth | PTB-XL+ provides extracted fiducials/median beats | Manual P/QRS/T boundaries and peaks | Beat labels, not full 12-lead wave boundaries | No manual fiducial boundaries |
| Suitability for clinician-feature matrix | Excellent | Excellent for morphology ground truth | Useful auxiliary beat dataset | Inadequate as primary ground-truth `B2` |
| Suitability for transition-matrix explainability | Excellent | Excellent for morphology validation | Good auxiliary pretraining/validation | Secondary transportability only |
| Reproducibility strength | Very high | High | High | High |
| Primary role in this project | `B1`, main benchmark | `B2`, gold-standard delineation matrix | Auxiliary beat-level pretraining and sanity checks | Optional future transport test only |

### 7.5 Dataset acquisition policy

By default, the required datasets must be obtained by downloading them through the terminal using their official links:

- `wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/`
- `wget -r -N -c -np https://physionet.org/files/ptb-xl-plus/1.0.1/`
- `wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/`

Optionally, for strictly reproducible version-locked benchmark runs, the implementation team may consume the following version-locked archives supplied with the project package:

- `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip`
- `ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip`
- `lobachevsky-university-electrocardiography-database-1.0.1.zip`
- `DATA_LOCK_SHA256SUMS.txt`

When using the optional ZIP archives, the team must verify checksums before ingestion.

## 8. Matrix B for dataset 1

### 8.1 Dataset-1 definition

**Dataset 1** is the combined PTB-XL/PTB-XL+ bundle. The waveform source of truth is PTB-XL 500 Hz. PTB-XL+ is used to supplement, not replace, the waveform source. Whenever a PTB-XL+ fiducial or feature conflicts with a waveform recomputation, the discrepancy must be logged and adjudicated by the feature-QA pipeline.

### 8.2 What the rows represent

Rows in `B1` represent **10-second 12-lead ECG records**, not individual beats.

This is a deliberate change from the naive beat-only interpretation of the original ECG corpus. The corpus is still honored through the triad encoder, but the transition-matrix object for PTB-XL must be record-level because:

- PTB-XL labels are record-level and multi-label;
- PTB-XL+ features are provided with one row per ECG record; and
- the transition papers require that rows of `A` and `B` correspond to the same objects.

#### Locked row count for the primary benchmark

For the **transition-fitting training matrix only**, `B1_train` must contain **exactly 10,000 rows**, sampled from PTB-XL folds 1-8 using:

- patient-disjoint sampling,
- multilabel stratification across project ontology labels,
- preservation of sex distribution where feasible, and
- preservation of the noise/pacemaker distribution where feasible.

This locked 10,000-row choice is adopted to remain directly comparable to the reference transition-matrix methodology in the 2024 Mathematics paper.

Validation and test matrices are **not** forced to 10,000 rows:

- `B1_val`: all eligible fold-9 records
- `B1_test`: all eligible fold-10 records

### 8.3 What the columns represent

Columns in `B1` represent clinically meaningful features measured or derived from the same record represented by the corresponding row. `B1` is stored in two forms:

- `B1_raw`: native units and native semantics;
- `B1_fit`: transformed/scaled values for transition fitting.

The actual numeric matrix excludes non-numeric identifiers. Record identifiers, patient identifiers, split labels, formula codes, and provenance metadata must be stored in a synchronized **sidecar index table**.

### 8.4 Authoritative feature block for `B1`

The following columns constitute the normative `B1` schema.

| Column | Family | Unit | Type | Level | Formula | Included in B1 | Primary extraction source | Notes |
|---|---|---|---|---|---|---|---|---|
| hr_med_bpm | rhythm | bpm | continuous | record | F1 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median instantaneous heart rate from accepted RR intervals. |
| rr_med_ms | rhythm | ms | continuous | record | F2 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median RR over accepted beats. |
| rr_iqr_ms | rhythm | ms | continuous | record | F2 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Interquartile range of RR over accepted beats. |
| rr_sdnn_ms | rhythm | ms | continuous | record | F3 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Standard deviation of NN intervals only after ectopic rejection. |
| prematurity_index_min | rhythm | ratio | bounded | record | F4 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Minimum coupling ratio RR_prev / RRn. |
| comp_pause_ratio_max | rhythm | ratio | bounded | record | F5 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Maximum (RR_prev + RR_next)/(2*RRn) across ectopic candidates. |
| pvc_like_beat_count | burden | count | count | record | F6 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Rule-based count of PVC-like beats. |
| apb_like_beat_count | burden | count | count | record | F6 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Rule-based count of atrial premature-like beats. |
| paced_like_beat_count | burden | count | count | record | F6 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Count of paced-beat candidates after spike detection and morphology checks. |
| af_irregularity_cv | rhythm | ratio | bounded | record | F7 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Coefficient of variation of accepted RR intervals after artifact rejection. |
| f_wave_power_ratio | atrial | ratio | bounded | record | F8 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Welch PSD ratio in 4-10 Hz over 0.5-20 Hz during T-Q interval. |
| p_present_ratio | atrial | ratio | bounded | record | F9 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Fraction of accepted beats with valid P-wave preceding QRS. |
| p_amp_ii_med_mV | atrial | mV | continuous | record | F10 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median lead II P-wave amplitude. |
| p_dur_med_ms | atrial | ms | continuous | record | F11 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median P-wave duration. |
| pr_med_ms | atrial | ms | continuous | record | F12 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median PR interval. |
| pr_iqr_ms | atrial | ms | continuous | record | F12 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | PR interval variability. |
| q_amp_ii_med_mV | qrs | mV | continuous | record | F10 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median lead II Q-wave amplitude. |
| r_amp_ii_med_mV | qrs | mV | continuous | record | F10 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median lead II R-wave amplitude. |
| s_amp_ii_med_mV | qrs | mV | continuous | record | F10 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median lead II S-wave amplitude. |
| qrs_dur_med_ms | qrs | ms | continuous | record | F13 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median QRS duration. |
| qrs_dur_iqr_ms | qrs | ms | continuous | record | F13 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | QRS duration variability. |
| qrs_deformed_prob | qrs | prob | bounded | record | F14 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median probability from shallow morphology detector trained on attention-zone QRS snippets. |
| qrs_deformed_any | qrs | binary | binary | record | F14 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | 1 if qrs_deformed_prob >= tau_qrs_def on any accepted beat. |
| qrs_fragmented_any | qrs | binary | binary | record | F15 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Binary indicator of fragmented or notched QRS in at least one analyzable lead. |
| qrs_wide_any | qrs | binary | binary | record | F15 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | 1 if any accepted beat has QRS duration >= 120 ms. |
| r_prime_v1_any | qrs | binary | binary | record | F16 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | R' / rsR' morphology present in V1. |
| broad_r_v6_any | qrs | binary | binary | record | F16 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Broad/notched terminal R in V6 or I. |
| st_level_v1_mV | st | mV | continuous | record | F17 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median ST displacement in V1 at J+60/J+80 relative to PR baseline. |
| st_level_v5_mV | st | mV | continuous | record | F17 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median ST displacement in V5 at J+60/J+80 relative to PR baseline. |
| st_slope_v5_uV_per_ms | st | uV/ms | continuous | record | F18 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Regression slope from J to J+offset in V5. |
| t_amp_v5_med_mV | t | mV | continuous | record | F19 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median T-wave amplitude in V5. |
| t_dur_med_ms | t | ms | continuous | record | F19 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median T-wave duration. |
| t_inverted_right_any | t | binary | binary | record | F20 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Negative deflection <= -0.1 mV lasting >= 80 ms in V1-V3. |
| qt_med_ms | qt | ms | continuous | record | F21 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median QT interval. |
| qtc_med_ms | qt | ms | continuous | record | F21 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median QT corrected by Fridericia formula; formula code stored in sidecar. |
| qrs_net_area_i_mV_ms | axis | mV*ms | continuous | record | F22 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median net QRS area in lead I. |
| qrs_net_area_avf_mV_ms | axis | mV*ms | continuous | record | F22 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Median net QRS area in lead aVF. |
| qrs_axis_deg | axis | deg | circular | record | F23 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Frontal QRS axis for clinician reporting; not used directly in regression. |
| qrs_axis_sin | axis | unitless | continuous | record | F23 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Sine encoding of frontal QRS axis. |
| qrs_axis_cos | axis | unitless | continuous | record | F23 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Cosine encoding of frontal QRS axis. |
| rbbb_signature_score | signature | logodds | continuous | record | F24 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | L1-logistic composite score using clinically constrained features. |
| lbbb_signature_score | signature | logodds | continuous | record | F24 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | L1-logistic composite score using clinically constrained features. |
| pvc_signature_score | signature | logodds | continuous | record | F24 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Composite score for PVC evidence. |
| af_signature_score | signature | logodds | continuous | record | F24 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Composite score for atrial fibrillation evidence. |
| paced_signature_score | signature | logodds | continuous | record | F24 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Composite score for paced rhythm evidence. |
| lead_quality_min_db | quality | dB | continuous | record | F25 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Minimum SNR across mandatory leads; lead II fallback threshold = 5 dB. |
| delineation_confidence | quality | 0-1 | bounded | record | F26 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Aggregate confidence from fiducial availability, beat acceptance, and lead quality. |
| u_present_v2_any | u | binary | binary | record | F27 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Only in 500 Hz pathways with high-confidence manual or algorithmic confirmation. |
| u_amp_v2_mV | u | mV | continuous | record | F27 | Yes | PTB-XL+ harmonized features + median beats/fiducials + 500 Hz raw waveform recomputation | Only in 500 Hz pathways; excluded categorically from any 100 Hz branch. |

### 8.5 Extraction rules and feature-family normalization for `B1`

`B1` must follow the family-specific transformation rules below.

#### 8.5.1 Continuous morphology and interval measurements

Applies to measurements in `ms`, `mV`, `mV*ms`, `dB`, or similar continuous scales.

1. Compute the raw feature from accepted beats.
2. Aggregate by robust median at record level unless the feature definition states otherwise.
3. On the training split only, winsorize to the `[0.5, 99.5]` percentile range.
4. Z-score using training-only mean and standard deviation.

#### 8.5.2 Count features

Applies to `*_count` variables.

1. Compute the raw non-negative integer count.
2. Transform using `log(1 + x)`.
3. Z-score using training-only mean and standard deviation.

No percentile clipping is allowed on count features, because high ectopic burden is a true pathology signal, not necessarily an outlier artifact.

#### 8.5.3 Binary and bounded features

Applies to binary flags and proportions.

1. Clip proportions to `[eps, 1-eps]` with `eps = 1e-3`.
2. For binaries, use epsilon-smoothed values `eps` and `1-eps`.
3. Apply the logit transform.
4. Z-score on training statistics.

This prevents the transition fit from treating 0/1 targets as ordinary unbounded Euclidean variables and also prevents continuous features from dominating the loss merely because they have larger raw variance.

#### 8.5.4 Circular features

`qrs_axis_deg` is stored for clinician reporting only. The fitting matrix uses `qrs_axis_sin` and `qrs_axis_cos`.

### 8.6 Handling missing or unavailable features in `B1`

The following rules are mandatory.

1. **Mandatory-lead failure**: if required leads for a feature family are absent or unusable after quality control, exclude the record from the primary `B1` transition-fitting cohort.
2. **Beat insufficiency**: if fewer than five valid beats remain, or if accepted beats cover less than 50% of the nominal 10-second record, exclude the record.
3. **Optional-feature failure**: optional columns such as U-wave features may remain `NA` in `B1_raw`; they must not be used in the core `B1_fit` matrix unless the missingness rate is low enough and the imputation model is validated.
4. **Non-mandatory feature imputation**: if an optional continuous feature is missing and the record is otherwise eligible, the team may use training-only MICE or a physiologic lead-reconstruction model. Global median imputation is prohibited.
5. **Missingness provenance**: every imputed value must be accompanied by a missingness flag in the sidecar metadata.

### 8.7 How `B1` interacts with matrix `A`

Each row in `B1` must align exactly with one row in `A1`, the record-level latent matrix built from pooled triad embeddings. The alignment contract is:

- identical record index,
- identical split membership,
- identical inclusion/exclusion status,
- identical preprocessing version hash,
- identical ontology version.

The transition operator is trained only on the intersection of rows that are valid in both `A1_train` and `B1_train`.

### 8.8 Output format for `B1`

The primary machine format for `B1` is:

- `B1_index.parquet`
- `B1_raw.parquet`
- `B1_fit.parquet`
- `B1_dictionary.csv`

A compact CSV header preview may be used for documentation, but the research artifact of record is the typed Parquet representation.

#### Representative structural preview (schema only)

```text
row_id,ecg_id,patient_id,split,hr_med_bpm,rr_med_ms,rr_iqr_ms,rr_sdnn_ms,prematurity_index_min,comp_pause_ratio_max,pvc_like_beat_count,apb_like_beat_count,paced_like_beat_count,af_irregularity_cv,f_wave_power_ratio,p_present_ratio,p_amp_ii_med_mV,p_dur_med_ms,pr_med_ms,pr_iqr_ms,q_amp_ii_med_mV,r_amp_ii_med_mV,s_amp_ii_med_mV,qrs_dur_med_ms,qrs_dur_iqr_ms,qrs_deformed_prob,qrs_deformed_any,qrs_fragmented_any,qrs_wide_any,r_prime_v1_any,broad_r_v6_any,st_level_v1_mV,st_level_v5_mV,st_slope_v5_uV_per_ms,t_amp_v5_med_mV,t_dur_med_ms,t_inverted_right_any,qt_med_ms,qtc_med_ms,qrs_net_area_i_mV_ms,qrs_net_area_avf_mV_ms,qrs_axis_deg,qrs_axis_sin,qrs_axis_cos,rbbb_signature_score,lbbb_signature_score,pvc_signature_score,af_signature_score,paced_signature_score,lead_quality_min_db,delineation_confidence,u_present_v2_any,u_amp_v2_mV
```

## 9. Matrix B for dataset 2

### 9.1 Dataset-2 definition

**Dataset 2** is LUDB v1.0.1, used because it provides manually annotated P-wave, QRS-complex, and T-wave boundaries and peaks across 12 leads. It is the morphology-grounded counterweight to the scale of PTB-XL/PTB-XL+.

### 9.2 What the rows represent

Rows in `B2` also represent **10-second 12-lead ECG records**.

Unlike `B1`, `B2` is **not** forced to exactly 10,000 rows. The reason is methodological, not logistical: LUDB is a gold-standard delineation corpus. Artificial duplication or excessive overlapping-window inflation would contaminate uncertainty estimates and undermine the point of using a manual-fiducial dataset in the first place.

Accordingly:

- `B2_train`, `B2_val`, and `B2_test` are generated within a repeated stratified 5-fold protocol at the record level;
- all eligible records are preserved in each training fold;
- no row duplication is allowed in the main analysis.

### 9.3 What the columns represent

The target ontology of `B2` is intentionally matched to `B1` as closely as possible. This makes `B2` usable for:

- morphology-grounded validation of the same feature definitions,
- cross-dataset transport studies after ontology mapping, and
- calibration of extraction error in features that are noisier in PTB-XL.

The crucial difference is provenance:

- in `B1`, many fiducials come from PTB-XL+ or recomputation;
- in `B2`, the fiducial ground truth comes from manual cardiologist annotations.

### 9.4 Authoritative feature block for `B2`

`B2` uses the same column definitions as `B1`, but with LUDB-specific provenance and one additional lock:

- whenever a feature can be derived directly from manual boundaries, the manual boundary is the source of truth;
- burden counts and syndrome scores may still be rule-derived because LUDB is not a beat-label corpus in the MIT-BIH sense.

The normative schema is:

| Column | Family | Unit | Type | Level | Formula | Included in B2 | Primary extraction source | Notes |
|---|---|---|---|---|---|---|---|---|
| hr_med_bpm | rhythm | bpm | continuous | record | F1 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median instantaneous heart rate from accepted RR intervals. |
| rr_med_ms | rhythm | ms | continuous | record | F2 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median RR over accepted beats. |
| rr_iqr_ms | rhythm | ms | continuous | record | F2 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Interquartile range of RR over accepted beats. |
| rr_sdnn_ms | rhythm | ms | continuous | record | F3 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Standard deviation of NN intervals only after ectopic rejection. |
| prematurity_index_min | rhythm | ratio | bounded | record | F4 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Minimum coupling ratio RR_prev / RRn. |
| comp_pause_ratio_max | rhythm | ratio | bounded | record | F5 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Maximum (RR_prev + RR_next)/(2*RRn) across ectopic candidates. |
| pvc_like_beat_count | burden | count | count | record | F6 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Rule-based count of PVC-like beats. |
| apb_like_beat_count | burden | count | count | record | F6 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Rule-based count of atrial premature-like beats. |
| paced_like_beat_count | burden | count | count | record | F6 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Count of paced-beat candidates after spike detection and morphology checks. |
| af_irregularity_cv | rhythm | ratio | bounded | record | F7 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Coefficient of variation of accepted RR intervals after artifact rejection. |
| f_wave_power_ratio | atrial | ratio | bounded | record | F8 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Welch PSD ratio in 4-10 Hz over 0.5-20 Hz during T-Q interval. |
| p_present_ratio | atrial | ratio | bounded | record | F9 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Fraction of accepted beats with valid P-wave preceding QRS. |
| p_amp_ii_med_mV | atrial | mV | continuous | record | F10 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median lead II P-wave amplitude. |
| p_dur_med_ms | atrial | ms | continuous | record | F11 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median P-wave duration. |
| pr_med_ms | atrial | ms | continuous | record | F12 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median PR interval. |
| pr_iqr_ms | atrial | ms | continuous | record | F12 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | PR interval variability. |
| q_amp_ii_med_mV | qrs | mV | continuous | record | F10 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median lead II Q-wave amplitude. |
| r_amp_ii_med_mV | qrs | mV | continuous | record | F10 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median lead II R-wave amplitude. |
| s_amp_ii_med_mV | qrs | mV | continuous | record | F10 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median lead II S-wave amplitude. |
| qrs_dur_med_ms | qrs | ms | continuous | record | F13 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median QRS duration. |
| qrs_dur_iqr_ms | qrs | ms | continuous | record | F13 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | QRS duration variability. |
| qrs_deformed_prob | qrs | prob | bounded | record | F14 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median probability from shallow morphology detector trained on attention-zone QRS snippets. |
| qrs_deformed_any | qrs | binary | binary | record | F14 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | 1 if qrs_deformed_prob >= tau_qrs_def on any accepted beat. |
| qrs_fragmented_any | qrs | binary | binary | record | F15 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Binary indicator of fragmented or notched QRS in at least one analyzable lead. |
| qrs_wide_any | qrs | binary | binary | record | F15 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | 1 if any accepted beat has QRS duration >= 120 ms. |
| r_prime_v1_any | qrs | binary | binary | record | F16 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | R' / rsR' morphology present in V1. |
| broad_r_v6_any | qrs | binary | binary | record | F16 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Broad/notched terminal R in V6 or I. |
| st_level_v1_mV | st | mV | continuous | record | F17 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median ST displacement in V1 at J+60/J+80 relative to PR baseline. |
| st_level_v5_mV | st | mV | continuous | record | F17 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median ST displacement in V5 at J+60/J+80 relative to PR baseline. |
| st_slope_v5_uV_per_ms | st | uV/ms | continuous | record | F18 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Regression slope from J to J+offset in V5. |
| t_amp_v5_med_mV | t | mV | continuous | record | F19 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median T-wave amplitude in V5. |
| t_dur_med_ms | t | ms | continuous | record | F19 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median T-wave duration. |
| t_inverted_right_any | t | binary | binary | record | F20 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Negative deflection <= -0.1 mV lasting >= 80 ms in V1-V3. |
| qt_med_ms | qt | ms | continuous | record | F21 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median QT interval. |
| qtc_med_ms | qt | ms | continuous | record | F21 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median QT corrected by Fridericia formula; formula code stored in sidecar. |
| qrs_net_area_i_mV_ms | axis | mV*ms | continuous | record | F22 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median net QRS area in lead I. |
| qrs_net_area_avf_mV_ms | axis | mV*ms | continuous | record | F22 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Median net QRS area in lead aVF. |
| qrs_axis_deg | axis | deg | circular | record | F23 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Frontal QRS axis for clinician reporting; not used directly in regression. |
| qrs_axis_sin | axis | unitless | continuous | record | F23 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Sine encoding of frontal QRS axis. |
| qrs_axis_cos | axis | unitless | continuous | record | F23 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Cosine encoding of frontal QRS axis. |
| rbbb_signature_score | signature | logodds | continuous | record | F24 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | L1-logistic composite score using clinically constrained features. |
| lbbb_signature_score | signature | logodds | continuous | record | F24 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | L1-logistic composite score using clinically constrained features. |
| pvc_signature_score | signature | logodds | continuous | record | F24 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Composite score for PVC evidence. |
| af_signature_score | signature | logodds | continuous | record | F24 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Composite score for atrial fibrillation evidence. |
| paced_signature_score | signature | logodds | continuous | record | F24 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Composite score for paced rhythm evidence. |
| lead_quality_min_db | quality | dB | continuous | record | F25 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Minimum SNR across mandatory leads; lead II fallback threshold = 5 dB. |
| delineation_confidence | quality | 0-1 | bounded | record | F26 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Aggregate confidence from fiducial availability, beat acceptance, and lead quality. |
| u_present_v2_any | u | binary | binary | record | F27 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Only in 500 Hz pathways with high-confidence manual or algorithmic confirmation. |
| u_amp_v2_mV | u | mV | continuous | record | F27 | Yes | LUDB manual P/QRS/T boundaries and peaks + 500 Hz raw waveforms; burden/signature features are rule-derived from manual fiducials | Only in 500 Hz pathways; excluded categorically from any 100 Hz branch. |

### 9.5 LUDB-specific rules

1. **Manual fiducials dominate**: use manual P/QRS/T boundaries and peaks whenever present.
2. **Waveform recomputation must respect manual anchors**: any recomputed amplitude, interval, or area feature must use the manual fiducial indices as its boundaries.
3. **Rule-derived burden features are allowed**: `pvc_like_beat_count`, `apb_like_beat_count`, and syndrome scores may be derived from the manually anchored morphology and RR context.
4. **No synthetic row expansion in the main analysis**: do not create pseudo-records to mimic PTB-XL scale.
5. **Optional U-wave module**: allow only if visually confirmed and if the 500 Hz branch supports it robustly.

### 9.6 Composite rule scores in `B2`

The following composite scores must be trained quantitatively and not hand-weighted:

- `rbbb_signature_score`
- `lbbb_signature_score`
- `pvc_signature_score`
- `af_signature_score`
- `paced_signature_score`

For each score:

1. define a clinically constrained candidate feature subset;
2. fit an `L1`-regularized logistic regression on the training split only;
3. retain the learned log-odds coefficients;
4. compute the score as `beta_0 + sum_j beta_j z_j`;
5. calibrate the decision threshold on validation data.

### 9.7 Handling missing features in `B2`

The same missingness rules as `B1` apply, with one additional LUDB-specific principle:

- if a manual fiducial is missing or contradictory across leads for a supposedly mandatory measurement, the feature should be marked missing and the record reviewed before inclusion.

### 9.8 Output format for `B2`

The primary machine format for `B2` is:

- `B2_index.parquet`
- `B2_raw.parquet`
- `B2_fit.parquet`
- `B2_dictionary.csv`

The same typed Parquet and sidecar structure used for `B1` must be used for `B2`.

#### Representative structural preview (schema only)

```text
row_id,record_id,split,hr_med_bpm,rr_med_ms,rr_iqr_ms,rr_sdnn_ms,prematurity_index_min,comp_pause_ratio_max,pvc_like_beat_count,apb_like_beat_count,paced_like_beat_count,af_irregularity_cv,f_wave_power_ratio,p_present_ratio,p_amp_ii_med_mV,p_dur_med_ms,pr_med_ms,pr_iqr_ms,q_amp_ii_med_mV,r_amp_ii_med_mV,s_amp_ii_med_mV,qrs_dur_med_ms,qrs_dur_iqr_ms,qrs_deformed_prob,qrs_deformed_any,qrs_fragmented_any,qrs_wide_any,r_prime_v1_any,broad_r_v6_any,st_level_v1_mV,st_level_v5_mV,st_slope_v5_uV_per_ms,t_amp_v5_med_mV,t_dur_med_ms,t_inverted_right_any,qt_med_ms,qtc_med_ms,qrs_net_area_i_mV_ms,qrs_net_area_avf_mV_ms,qrs_axis_deg,qrs_axis_sin,qrs_axis_cos,rbbb_signature_score,lbbb_signature_score,pvc_signature_score,af_signature_score,paced_signature_score,lead_quality_min_db,delineation_confidence,u_present_v2_any,u_amp_v2_mV
```

## 10. Detailed implementation roadmap

Execute the project in the exact order below. Do not change the order unless a dependency explicitly requires it.

### 10.1 Step 1 - Provision the dataset package

1. Create a project root with the following subdirectories:

```text
project_root/
  data_lock/
  raw/
  interim/
  features/
  latents/
  transition/
  reports/
  manifests/
```

2. By default, download the datasets directly via terminal using their official links:
   - `wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/`
   - `wget -r -N -c -np https://physionet.org/files/ptb-xl-plus/1.0.1/`
   - `wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/`
3. (Optional) If using the version-locked ZIP archives instead:
   - Copy the locked archives into `data_lock/`.
   - Verify SHA256 checksums against `DATA_LOCK_SHA256SUMS.txt`.
   - Abort the run if any checksum mismatch occurs.

### 10.2 Step 2 - Unpack and index the source datasets

1. If using the default terminal download, ensure the downloaded files are correctly placed in `raw/ptbxl/`, `raw/ptbxl_plus/`, and `raw/ludb/`.
2. If using the optional ZIP archives, unpack `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip` into `raw/ptbxl/`, `ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip` into `raw/ptbxl_plus/`, and `lobachevsky-university-electrocardiography-database-1.0.1.zip` into `raw/ludb/`.
3. Build file manifests for every waveform and annotation file.
4. Record data provenance (download timestamp or archive version/checksum) and unpack timestamp in `manifests/source_manifest.json`.

### 10.3 Step 3 - Build the ontology and data splits before any scaling

1. Create the project ontology table mapping source labels into the project classes.
2. Build the explicit PTB-XL -> LUDB ontological mapping matrix before any cross-dataset evaluation.
3. For PTB-XL, use folds 1-8 as the candidate training pool, fold 9 as validation, and fold 10 as test.
4. From the PTB-XL training pool, draw the locked 10,000-record `B1_train` cohort using patient-disjoint multilabel stratification.
5. For LUDB, run repeated stratified 5-fold record-level splits.
6. Freeze the split assignment files.
7. Only after the splits are frozen, compute any scaling statistics, imputation models, PCA transforms, score weights, or transition operators.

### 10.4 Step 4 - Preprocess the waveforms using two branches

Create two preprocessing branches.

#### 10.4.1 Detection branch

Use this branch for R-peak detection and beat indexing.

- Remove mains interference with a narrow notch filter only if line noise is present.
- Use a zero-phase drift-suppression configuration suited to robust peak detection.
- The detection branch may use a zero-phase digital high-pass drift-suppression setting up to **0.67 Hz** when required for robust peak localization.
- Use a low-pass ceiling appropriate to the acquisition chain, typically **<= 150 Hz** for 500 Hz diagnostic recordings.
- If resampling is required, perform it with an FIR polyphase anti-aliasing filter.
- Use this branch to maximize temporal anchor reliability, not ST fidelity.

#### 10.4.2 Diagnostic-feature branch

Use this branch for amplitudes, intervals, ST level, QT, axis, and morphology measurements.

- Preserve the native 500 Hz waveform when available.
- Use a **zero-phase linear high-pass filter with a 0.05 Hz low-cut** for ST/QT-critical measurements.
- Use a low-pass ceiling appropriate to the device bandwidth, typically **<= 150 Hz** for 500 Hz diagnostic recordings.
- Do not use a blanket 0.67 Hz low-cut for ST-critical features.
- Apply the same lead ordering and scaling across all records.
- Keep a record of the exact filter coefficients and version hash.

### 10.5 Step 5 - Detect and remove pacing artifacts before routine morphology analysis

1. Identify records with explicit pacemaker metadata where available.
2. Run a pacing-spike detector on all candidate paced records before standard linear filtering.
3. Estimate pacing-spike onset and offset from high-slope transients.
4. Remove the pacing spike while preserving the physiologic waveform immediately adjacent to it.
5. Store a pacing-artifact provenance flag in the record metadata.
6. Do not compute paced morphology features from uncorrected spike-contaminated signals.

### 10.6 Step 6 - Detect R-peaks with a knowledge-integrated adaptive-window method

1. Compute a coarse heart-period estimate `RR_hat^(0)` from autocorrelation or another global periodicity estimator on the best-quality lead.
2. Score all mandatory leads for quality.
3. Use lead II as the default rhythm anchor only if its QRS-to-baseline SNR is at least 5 dB.
4. If lead II fails the threshold, switch to the best available rhythm lead or a composite rhythm channel.
5. Build the knowledge signal `K` around candidate peak regions, but do not scan with a hard-coded 260-sample window.
6. Set the adaptive search horizon from the current `RR_hat` and update it after every accepted R-peak.
7. Impose an adaptive refractory interval derived from `RR_hat`, never a fixed skip length alone.
8. Validate accepted peaks by local morphology constraints and by cross-lead temporal consistency.
9. Store both the final R-peaks and the lead/provenance information used to detect them.

### 10.7 Step 7 - Delineate P/QRS/T fiducials and accept or reject beats

1. For PTB-XL/PTB-XL+, ingest fiducials from PTB-XL+ where available, then verify them against waveform consistency.
2. For LUDB, use manual fiducials as the source of truth.
3. For each candidate beat, verify the physiologic order:

`P_on <= P_peak <= P_off < QRS_on <= R <= QRS_off <= T_on <= T_peak <= T_off`

4. Reject beats with impossible ordering, extreme saturation, unresolved pacing contamination, or severe baseline corruption.
5. Compute per-beat lead quality and fiducial completeness.
6. Retain only accepted beats for downstream feature aggregation.
7. Reject any record with fewer than five accepted beats or less than 50% analyzable duration.

### 10.8 Step 8 - Construct triad objects

1. For each accepted central beat, locate the nearest accepted previous beat and accepted following beat.
2. Form a triad object `[beat_(i-1), beat_i, beat_(i+1)]`.
3. Align the three beats on their R-peaks.
4. Use consistent windowing around each R-peak in physical time, not arbitrary sample counts detached from heart rate.
5. If a triad cannot be formed because one neighbor is unavailable, exclude that central beat from triad encoding.
6. Persist the triad membership file so that record-level pooling can be reproduced exactly.

### 10.9 Step 9 - Train the deep classifier with class-imbalance control

1. Use a 1D convolutional architecture that preserves the corpus logic: convolution + BatchNorm + additional convolutional depth as needed.
2. Extend the architecture from the legacy single-/few-lead setting to a multilead triad tensor.
3. Use focal loss or dynamically weighted cross-entropy to prevent minority classes from collapsing in latent space.
4. Track macro-F1, class-wise recall, and calibration during training.
5. Early-stop on validation performance, not test performance.
6. Save the exact model weights, training seed, and optimizer configuration.

### 10.10 Step 10 - Extract matrix `A`

1. Extract the **preactivation** penultimate-layer vector for every triad object.
2. Set the nominal penultimate dimensionality to 512 unless a validated ablation justifies a smaller value.
3. Pool triad vectors to record-level `A_record` using the locked trimmed-mean baseline.
4. Remove zero-variance columns on the training split only.
5. Standardize the remaining columns using training-only statistics.
6. Fit PCA on the training split only.
7. Retain the smallest rank `r` that explains at least 99% of training variance, subject to validation-based stability checks.
8. Persist:
   - the zero-variance mask,
   - training means and standard deviations,
   - PCA loadings,
   - retained rank.

### 10.11 Step 11 - Compute matrix `B_raw`

1. Compute every feature listed in Section 8 or Section 9 from the accepted beats and verified fiducials.
2. Use the exact formulas in Appendix A.
3. Use Fridericia for `qtc_med_ms`.
4. Use J+60/J+80 rules for ST measurement.
5. Use the NN-only mask for `rr_sdnn_ms`.
6. Use the 4-10 Hz T-Q interval spectrum for `f_wave_power_ratio`.
7. For composite scores, fit the required L1-logistic models on the training split only and then apply them to validation/test.
8. Persist the raw native-unit values as `B_raw`.

### 10.12 Step 12 - Transform `B_raw` into `B_fit` and fit the transition operator

1. Apply the family-specific transforms from Sections 8.5 and 9.5.
2. Fit all transformation statistics on the training split only.
3. Build `B_fit_train`.
4. Fit the ridge-regularized transition operator `T`.
5. Tune `lambda` on validation data using a combination of:
   - feature reconstruction error,
   - agreement metrics,
   - clinical plausibility checks.
6. Truncate singular values explicitly.
7. Persist the full operator package:
   - `T`,
   - `lambda`,
   - retained rank,
   - transform statistics,
   - inverse-map metadata.

### 10.13 Step 13 - Generate explanations for validation and test data

1. Apply the trained deep encoder to validation/test triads.
2. Pool to record-level `A`.
3. Transform `A` through `T` into `B_hat_fit`.
4. Invert the family-specific transforms into `B_hat_raw`.
5. For binary features, apply the validation-calibrated thresholds.
6. Store both the raw feature estimates and the thresholded flags.

### 10.14 Step 14 - Produce clinician-facing explanation packets

For each reviewed ECG record, generate a compact packet containing:

- the record identifier,
- the model prediction,
- the measured feature vector `B_raw`,
- the model-implied feature vector `B_hat_raw`,
- discrepancy highlights,
- waveform snapshots marking the relevant fiducials or attention zones.

### 10.15 Step 15 - Freeze outputs and reproducibility manifests

1. Write all matrices and sidecars to Parquet/CSV.
2. Save the software environment specification.
3. Save all random seeds.
4. Save a final manifest with:
   - source archive hashes,
   - split hashes,
   - model hash,
   - transition-operator hash,
   - feature-dictionary hash.

No main-result figure or table may be produced from an un-frozen manifest state.

## 11. Validation, evaluation, and risk controls

### 11.1 Prediction-quality evaluation

The classifier must be evaluated at minimum with:

- accuracy,
- balanced accuracy,
- macro-precision,
- macro-recall,
- macro-F1,
- micro-F1 where relevant,
- AUROC and AUPRC for one-vs-rest settings,
- calibration error and reliability diagrams.

Every metric must be reported with **95% confidence intervals** computed by non-parametric bootstrap. The resampling unit must respect the independence structure:

- patient-level bootstrap for PTB-XL when patient grouping is available,
- record-level bootstrap for LUDB if patient identity is not recoverable beyond record scope.

Use at least 1000 bootstrap replicates in the main analysis.

### 11.2 Explanation-quality evaluation

Explanation quality must be evaluated feature-family-wise.

#### 11.2.1 Continuous features

Report, at minimum:

- MAE,
- RMSE,
- Pearson correlation,
- Spearman correlation,
- calibration plots where clinically meaningful.

#### 11.2.2 Binary features

Report, at minimum:

- AUROC,
- sensitivity,
- specificity,
- precision,
- recall,
- Brier score,
- threshold used for binarization.

#### 11.2.3 Agreement-oriented metrics

For clinically interpreted binary or categorical explanations, report both:

- **Cohen kappa**, and
- **Gwet AC1**

because kappa alone can be misleading under strong class imbalance.

### 11.3 Statistical testing of explanation validity

Do not report explanation fidelity as point estimates only. Perform formal hypothesis testing.

Required tests:

1. **Null-model comparison**: compare the residual error of the proposed operator against a label-shuffled or row-shuffled null mapping.
2. **Naive-baseline comparison**: compare the typed, reduced-rank, regularized operator against a naive unconstrained linear fit.
3. **Paired significance test**: use paired Wilcoxon signed-rank or a paired permutation test on per-record residual summaries.

At least one null test and one paired comparison must appear in the main report.

### 11.4 Robustness checks

Run the following robustness studies:

1. adaptive R-window vs legacy fixed-window detection;
2. with vs without pacing-artifact removal on paced records;
3. diagnostic branch low-cut settings;
4. trimmed-mean vs max vs attention pooling for record-level `A`;
5. no-PCA vs PCA-reduced `A`;
6. naive pseudo-inverse vs reduced-rank ridge operator;
7. with vs without `f_wave_power_ratio`;
8. with vs without count-feature log transform;
9. PTB-XL-trained operator evaluated on LUDB after ontology mapping;
10. optional ETM pilot under allowed transformations only.

### 11.5 Error analysis

Every main experiment must include structured error analysis by:

- pathology class,
- heart-rate range,
- paced vs non-paced records,
- signal quality strata,
- sex and age strata where metadata are available,
- records with missing optional features,
- records with large disagreement `||B_raw - B_hat_raw||`.

### 11.6 Interpretability validation with experts

Where clinical collaboration is available, perform a blinded review in which experts assess:

- whether the highlighted features are correctly measured,
- whether the translated features support the predicted rhythm/pathology,
- whether discrepancies reveal model failure, feature-space insufficiency, or labeling ambiguity.

The purpose is not to replace statistical validation but to test whether the mental model is usable in real expert reasoning.

## 12. Limitations and future extensions

### 12.1 Current limitations

1. **The baseline explainer is still linear in latent space.** The typed transformations make it mathematically defensible, but the bridge itself remains linear after transformation.
2. **PTB-XL is large but primarily record-labeled.** Some beat-level burden features are therefore still rule-derived.
3. **LUDB is morphology-rich but small.** It is ideal for validation of feature extraction, not for replacing large-scale training.
4. **Signal quality heterogeneity remains clinically real.** The project reduces but does not eliminate the impact of noise, pacing, and ambiguous waves.
5. **The exact 10,000-row benchmark lock is applied only to `B1`.** This is a design choice for comparability, not a universal requirement for all ECG explainer studies.

### 12.2 Immediate optional extensions after the baseline is stable

1. Add demographic features (age, sex, height, weight) exactly as suggested in the 2023 feature-vector paper, but store them outside the strict ECG-only core until a fairness analysis is completed.
2. Add lead-group summary features (inferior, lateral, anterior aggregates).
3. Add concept-level uncertainty intervals for `B_hat_raw`.
4. Add beat-level explanation packets in addition to record-level packets.

### 12.3 ETM as a future-stage methodological enhancement

If the baseline operator is validated, ETM may be investigated under the following restrictions:

- only use clinically benign transformations;
- do not use linear temporal scaling;
- keep ETM as a post-hoc explainer, not a replacement for the validated predictive model;
- compare ETM against the baseline operator on both fidelity and symmetry metrics;
- abort ETM adoption if physiologic plausibility deteriorates.

### 12.4 External dataset transport beyond LUDB

After PTB-XL -> LUDB transport is validated, the project may extend to:

- Chapman/Shaoxing/Ningbo,
- challenge-style multihospital 12-lead corpora,
- legacy beat-level corpora such as MIT-BIH for auxiliary tasks.

Those studies must reuse the locked ontology and must not silently redefine feature thresholds.

## 13. Final conclusion

The project specification preserves the scientific identity of the uploaded ECG/XAI corpus while making the implementation clinically defensible, mathematically stable, and reproducible.

The central idea remains unchanged: translate deep latent ECG representations into a feature space that clinicians can understand and audit. What has changed in this version is the rigor of that translation. The operator is now fit on a typed target space rather than on raw heterogeneous targets; the latent space is reduced and regularized before transition fitting; R-peak detection is adaptive rather than hard-coded; QT, ST, T-wave, and SDNN definitions are locked to explicit rules; missingness handling is physiologically aware; uncertainty and agreement are reported with confidence intervals and prevalence-robust metrics; and the second dataset for matrix `B` is now a true fiducial ground-truth corpus.

As a result, this specification is not merely a literature synthesis. It is an executable research-and-development plan for building, validating, and auditing explainable ECG deep-learning systems in a way that is intelligible to both data scientists and healthcare professionals.

## Appendix A. Mathematical operationalization tutorial

### A.1 Notation

Let:

- `x_(r,l)[n]` be the diagnostic-branch ECG signal for record `r`, lead `l`, sample `n`;
- `f_s` be the sampling frequency in Hz;
- `i = 1, ..., N_r` index accepted beats in record `r`;
- `t_Pon^(i,l), t_Ppk^(i,l), t_Poff^(i,l), t_QRSon^(i,l), t_R^(i,l), t_QRSoff^(i,l), t_Ton^(i,l), t_Tpk^(i,l), t_Toff^(i,l)` be fiducial sample indices for beat `i`, lead `l`;
- `beta_(r,l)^(i)` be the beat-specific isoelectric baseline in lead `l`;
- `RR_i = ( t_R^(i) - t_R^(i-1) ) / f_s` in seconds.

Use the PR/PQ isoelectric segment for `beta` whenever available. If the PR segment is unavailable, use a validated TP-segment fallback.

### A.2 Beat-level primitives

#### F1. Heart rate

For each accepted beat:

`HR_i = 60 / RR_i`

Record-level feature:

`hr_med_bpm = median_i( HR_i )`

#### F2. RR summary statistics

`rr_med_ms = 1000 * median_i( RR_i )`

`rr_iqr_ms = 1000 * IQR_i( RR_i )`

#### F3. NN-only SDNN

Define an NN mask:

`M_NN(i) = 1` if beat `i` and its adjacent intervals are not ectopic, paced, or artifact-corrupted; otherwise `0`.

Then:

`NN = { RR_i : M_NN(i) = 1 }`

`rr_sdnn_ms = 1000 * std( NN )`

If `|NN| < 5`, set `rr_sdnn_ms` missing and exclude the record from core HRV analyses.

#### F4. Prematurity index

Let `RR_prev^(i)` be the interval before beat `i`. Let `RR_n` be the median NN interval for the record.

`prematurity_index^(i) = RR_prev^(i) / RR_n`

`prematurity_index_min = min_i( prematurity_index^(i) )`

#### F5. Compensatory pause ratio

For each ectopic candidate beat `i` with valid neighboring intervals:

`comp_pause_ratio^(i) = ( RR_prev^(i) + RR_next^(i) ) / ( 2 * RR_n )`

`comp_pause_ratio_max = max_i( comp_pause_ratio^(i) )`

#### F6. Burden counts

Define rule-based beat labels from morphology and rhythm logic. Then:

`pvc_like_beat_count = sum_i 1[ beat_i is PVC-like ]`

`apb_like_beat_count = sum_i 1[ beat_i is APB-like ]`

`paced_like_beat_count = sum_i 1[ beat_i is paced-like ]`

#### F7. AF irregularity coefficient

`af_irregularity_cv = std_i(RR_i) / mean_i(RR_i)`

#### F8. f-wave power ratio

For each analyzable beat interval, define a T-Q segment:

`S_i = x_residual[ t_Toff^(i-1) + delta1 : t_QRSon^(i) - delta2 ]`

with guard margins `delta1 = delta2 = 0.04 * f_s` when feasible.

Compute Welch power:

`P_f(i) = integral_(4 Hz to 10 Hz) PSD(S_i) df`

`P_all(i) = integral_(0.5 Hz to 20 Hz) PSD(S_i) df`

Then:

`f_wave_power_ratio = median_i( P_f(i) / P_all(i) )`

#### F9. P-wave presence ratio

For each accepted beat define:

`P_present^(i) = 1` if a physiologically valid P-wave precedes the QRS within allowed timing; else `0`.

Then:

`p_present_ratio = (1 / N_r) * sum_i P_present^(i)`

#### F10. Amplitude features

For any wave peak `w` in lead `l`:

`amp_w^(i,l) = x_(r,l)[ t_w^(i,l) ] - beta_(r,l)^(i)`

Record-level median examples:

`p_amp_ii_med_mV = median_i amp_P^(i,II)`

`q_amp_ii_med_mV = median_i amp_Q^(i,II)`

`r_amp_ii_med_mV = median_i amp_R^(i,II)`

`s_amp_ii_med_mV = median_i amp_S^(i,II)`

`t_amp_v5_med_mV = median_i amp_T^(i,V5)`

All amplitudes are stored in `mV`.

#### F11. P-wave duration

For each beat:

`P_dur_i_ms = 1000 * ( t_Poff^(i) - t_Pon^(i) ) / f_s`

Then:

`p_dur_med_ms = median_i( P_dur_i_ms )`

#### F12. PR interval

For each beat:

`PR_i_ms = 1000 * ( t_QRSon^(i) - t_Pon^(i) ) / f_s`

Then:

`pr_med_ms = median_i( PR_i_ms )`

`pr_iqr_ms = IQR_i( PR_i_ms )`

#### F13. QRS duration

For each beat:

`QRS_i_ms = 1000 * ( t_QRSoff^(i) - t_QRSon^(i) ) / f_s`

Then:

`qrs_dur_med_ms = median_i( QRS_i_ms )`

`qrs_dur_iqr_ms = IQR_i( QRS_i_ms )`

#### F14. QRS deformation probability and flag

Let `g_QRS( . )` be the dedicated shallow morphology detector trained on attention-zone QRS snippets.

For beat `i`:

`qrs_deformed_prob^(i) = g_QRS( snippet_QRS^(i) )`

Record-level probability:

`qrs_deformed_prob = median_i qrs_deformed_prob^(i)`

Binary flag:

`qrs_deformed_any = 1` if `max_i qrs_deformed_prob^(i) >= tau_qrs_def`, else `0`

where `tau_qrs_def` is calibrated on the validation split.

#### F15. Fragmentation and wide-QRS flags

For each beat and lead, smooth the QRS snippet with a short Savitzky-Golay or equivalent zero-phase local smoother and count secondary extrema inside `[t_QRSon, t_QRSoff]` whose prominence is at least `0.05 mV`, excluding the dominant R and dominant S/Q extrema.

`qrs_fragmented_any = 1` if there exists at least one analyzable lead with `>= 2` such secondary extrema in any accepted beat; else `0`

`qrs_wide_any = 1` if `max_i QRS_i_ms >= 120`, else `0`

#### F16. Bundle-branch morphology primitives

`r_prime_v1_any = 1` if an `R'` morphology is present in lead V1 in any accepted beat according to the locked morphology rule.

`broad_r_v6_any = 1` if a broad terminal R is present in lead I or V6 in any accepted beat according to the locked morphology rule.

#### F17. ST level

Let `t_J^(i,l) = t_QRSoff^(i,l)`.

Define the ST offset:

`delta_ST^(i) = 0.06 * f_s` if `HR_i >= 100`, else `0.08 * f_s`

Then for lead `l`:

`ST_level^(i,l) = x_(r,l)[ t_J^(i,l) + delta_ST^(i) ] - beta_(r,l)^(i)`

Record-level medians:

`st_level_v1_mV = median_i ST_level^(i,V1)`

`st_level_v5_mV = median_i ST_level^(i,V5)`

#### F18. ST slope

For lead `l`, fit a least-squares line from `t_J^(i,l)` to `t_J^(i,l) + delta_ST^(i)`.

If `a_i^(l)` is the slope coefficient:

`st_slope_v5_uV_per_ms = median_i a_i^(V5)`

stored in `uV/ms`.

#### F19. T-wave duration

`T_dur_i_ms = 1000 * ( t_Toff^(i) - t_Ton^(i) ) / f_s`

`t_dur_med_ms = median_i( T_dur_i_ms )`

#### F20. T-wave inversion

For the target right-precordial set `L_R = {V1, V2, V3}`:

`t_inverted_right_any = 1`

iff there exists a beat `i` and lead `l in L_R` such that the T-wave remains negative with peak amplitude

`amp_T^(i,l) <= -0.1 mV`

and the contiguous negative deflection duration is at least `80 ms`.

Otherwise:

`t_inverted_right_any = 0`

#### F21. QT and QTc

For each beat:

`QT_i_ms = 1000 * ( t_Toff^(i) - t_QRSon^(i) ) / f_s`

`QT_i_s = QT_i_ms / 1000`

`RR_i_s = RR_i`

Fridericia correction:

`QTcF_i = QT_i_s / RR_i_s^(1/3)`

Then:

`qt_med_ms = median_i( QT_i_ms )`

`qtc_med_ms = 1000 * median_i( QTcF_i )`

The formula code `QTcF` must be stored in the sidecar metadata.

#### F22. Net QRS area

For lead `l`:

`QRS_net^(i,l) = (1000 / f_s) * sum_(n=t_QRSon^(i,l))^(t_QRSoff^(i,l)) [ x_(r,l)[n] - beta_(r,l)^(i) ]`

Examples:

`qrs_net_area_i_mV_ms = median_i QRS_net^(i,I)`

`qrs_net_area_avf_mV_ms = median_i QRS_net^(i,aVF)`

#### F23. Frontal QRS axis

For each beat:

`theta_i = atan2( QRS_net^(i,aVF), QRS_net^(i,I) )`

Record-level reporting angle in degrees:

`qrs_axis_deg = median_angle_i( theta_i ) * 180 / pi`

Encoded features:

`qrs_axis_sin = sin( qrs_axis_deg * pi / 180 )`

`qrs_axis_cos = cos( qrs_axis_deg * pi / 180 )`

#### F24. Composite signature scores

Let `z_j` denote standardized training-split features from a clinically constrained subset.

For a target syndrome `c`:

`score_c = beta_0^(c) + sum_j beta_j^(c) z_j`

where the coefficients are learned by `L1`-regularized logistic regression on the training split.

This defines:

- `rbbb_signature_score`
- `lbbb_signature_score`
- `pvc_signature_score`
- `af_signature_score`
- `paced_signature_score`

#### F25. Lead quality

For a lead `l`:

`SNR_l_db = 20 * log10( A_QRS_pp / RMS_baseline )`

where `A_QRS_pp` is the peak-to-peak QRS amplitude in the chosen rhythm window and `RMS_baseline` is the RMS of an isoelectric baseline segment.

Then:

`lead_quality_min_db = min_l SNR_l_db`

Lead II is considered unacceptable for rhythm anchoring if `SNR_II_db < 5`.

#### F26. Delineation confidence

Define three normalized components:

- `c_fid`: fraction of required fiducials present,
- `c_beats`: fraction of beats accepted,
- `c_leads`: fraction of mandatory leads above quality threshold.

Then:

`delineation_confidence = ( c_fid + c_beats + c_leads ) / 3`

#### F27. U-wave features (500 Hz only)

`u_present_v2_any = 1` only if a high-confidence U wave is visually or algorithmically confirmed in V2 on the 500 Hz branch.

`u_amp_v2_mV = median_i amp_U^(i,V2)` for confirmed U waves.

These features are excluded from any 100 Hz branch by protocol.

### A.3 Typed transformation into `B_fit`

Let `mu_j`, `sigma_j` be training-only mean and standard deviation after the feature-family transform.

#### G1. Continuous features

Winsorize:

`x'_j = min( max( x_j, q_0.5 ), q_99.5 )`

then z-score:

`z_j = ( x'_j - mu_j ) / sigma_j`

#### G2. Count features

`z_j = ( log(1 + x_j) - mu_j ) / sigma_j`

#### G3. Binary and bounded features

Epsilon smoothing with `eps = 1e-3`:

`p'_j = clip( p_j, eps, 1 - eps )`

`g_j = log( p'_j / (1 - p'_j) )`

`z_j = ( g_j - mu_j ) / sigma_j`

For a binary `y in {0,1}` use `p_j = eps` if `y=0` and `p_j = 1-eps` if `y=1`.

### A.4 Inverse maps from `B_hat_fit` to `B_hat_raw`

#### I1. Continuous

`x_hat_j = sigma_j * z_hat_j + mu_j`

#### I2. Counts

`x_hat_j = exp( sigma_j * z_hat_j + mu_j ) - 1`

#### I3. Binary / bounded

`g_hat_j = sigma_j * z_hat_j + mu_j`

`p_hat_j = 1 / (1 + exp( -g_hat_j ))`

For binary outputs:

`y_hat_j = 1[ p_hat_j >= tau_j ]`

where `tau_j` is calibrated on validation data.

### A.5 Adaptive R-peak search update

Compute the initial coarse period on the detection branch:

`RR_hat^(0) = argmax_(tau in [0.24 f_s, 1.5 f_s]) R_x(tau)`

where `R_x` is the autocorrelation of the selected rhythm lead or composite rhythm channel.

For the `k`-th predicted beat after the first accepted peak:

`t_pred^(k) = t_R^(k-1) + RR_hat^(k-1)`

`w_k = clip( 0.35 * RR_hat^(k-1), 0.12 * f_s, 0.45 * f_s )`

Search only in:

`W_k = [ t_pred^(k) - w_k , t_pred^(k) + w_k ]`

Accept the candidate R-peak as the dominant positive deviation within `W_k` that also satisfies the local morphology and refractory checks.

Update the running period estimate with `alpha = 0.2`:

`RR_hat^(k) = alpha * RR_obs^(k) + (1 - alpha) * RR_hat^(k-1)`

Refractory period:

`ref_k = max( 0.18 * RR_hat^(k-1), 0.15 * f_s )`

### A.6 Transition operator

With `A_red` and `B_fit` on the same training rows:

`T = argmin_T ||A_red T - B_fit||_F^2 + lambda ||T||_F^2`

Use the reduced-rank ridge solution from Section 6.4.3.

## Appendix B. Issue-resolution crosswalk

| Issue | Resolution in specification | Where implemented |
|---|---|---|
| 1 | Binary/bounded B targets no longer use raw unconstrained least squares. B_fit uses family-specific link transforms (logit for binary/proportions, log1p for counts) and inverse projections after T estimation. | 6.4, 8.5, 9.5, Appendix A |
| 2 | Matrix A is reduced and regularized before T fitting: preactivation 512-dim latent, zero-variance pruning, PCA to 99% variance, ridge penalty. | 6.2, 10.10-10.12, Appendix A |
| 3 | Matrix A uses preactivation penultimate features, never post-ReLU activations. | 6.2, 10.10 |
| 4 | Fixed 260-sample scanning is replaced by an autocorrelation-initialized adaptive RR search window and adaptive refractory period. | 10.6, Appendix A |
| 5 | rr_sdnn_ms is restricted to NN intervals after ectopic/paced rejection. | 8.4, 9.4, Appendix A |
| 6 | qtc_med_ms is locked to Fridericia as the primary reproducible default; sensitivity analyses may add Framingham/Hodges. | 8.4, 9.4, Appendix A |
| 7 | ST levels are measured at J+60 ms for HR >= 100 bpm and J+80 ms otherwise, relative to PR baseline. | 8.4, 9.4, Appendix A |
| 8 | T inversion requires <= -0.1 mV and duration >= 80 ms in the target leads. | 8.4, 9.4, Appendix A |
| 9 | QRS axis is operationalized from net QRS area in leads I and aVF using atan2. | 8.4, 9.4, Appendix A |
| 10 | Global median imputation for missing leads is prohibited; mandatory-lead failures trigger exclusion, otherwise training-only MICE/lead reconstruction is allowed. | 8.6, 9.5, 10.7 |
| 11 | Minimum analyzable content is raised to >= 5 valid beats or >= 50% of the 10-second record duration. | 8.6, 9.5, 10.7 |
| 12 | U-wave extraction is categorically forbidden in 100 Hz branches and only optional in 500 Hz pathways. | 5.2, 8.3, 9.3, Appendix A |
| 13 | A two-branch preprocessing design is imposed: a diagnostic branch with 0.05 Hz zero-phase low-cut for ST/QT fidelity, and a detection branch that may use stronger drift suppression; blanket 0.67 Hz filtering is rejected for ST measurement. | 10.4, 12.1 |
| 14 | Pacing artifact detection/removal is inserted before routine linear filtering for paced rhythms. | 10.5 |
| 15 | Triad latent vectors are aggregated to record-level A via robust mean pooling; max/attention pooling are ablation variants. | 6.2, 10.8-10.10 |
| 16 | All predictive and explanation metrics must report 95% bootstrap confidence intervals with patient/record-level resampling. | 11.1-11.3 |
| 17 | Gwet AC1 is reported alongside Cohen kappa for imbalanced agreement analysis. | 11.2 |
| 18 | Residuals of the proposed explainer are tested against null/shuffled baselines using paired non-parametric or permutation tests. | 11.3 |
| 19 | Composite rule weights are learned from L1-regularized logistic regression, not set heuristically. | 8.4, 9.4, Appendix A |
| 20 | f_wave_power_ratio is added explicitly to cover AF physiology beyond absent P waves. | 5.2, 8.4, 9.4, Appendix A |
| 21 | Count features use log1p before z-scoring; percentile clipping is reserved for continuous morphology measurements. | 8.5, 9.5, Appendix A |
| 22 | Binary features are standardized within the fitting copy after link transformation so they are not dominated by high-variance continuous targets. | 6.3-6.4, 8.5, 9.5 |
| 23 | SVD rank truncation is explicit: discard singular values <= max(m,r)*eps*sigma_1 or use tuned rank r. | 6.4, 10.12, Appendix A |
| 24 | Lead II is discarded for detection when SNR < 5 dB; fallback lead selection is quantitative, not subjective. | 10.6, Appendix A |
| 25 | Cross-dataset transport requires a predeclared PTB-XL-to-LUDB ontology mapping matrix before evaluation. | 7.4, 11.4, Appendix D |
| 26 | Deep classifier training uses focal loss or dynamic class-weighted cross-entropy. | 10.9 |
| 27 | All scaling, PCA, imputation, composite-score fitting, and T estimation are fit on training splits only. | 10.3, 10.11-10.12 |
| 28 | Any resampling must use FIR polyphase anti-aliasing. | 10.4 |
| 29 | ETM forbids linear temporal scaling; only baseline offsets, mild gain perturbations, and isoelectric noise are admissible future transformations. | 6.6, 12.3 |
| 30 | Feature thresholds and diagnostic rules are anchored to AHA/ACCF/HRS ECG standardization recommendations and EC57 reporting practice. | 5.1, 8.4, 9.4, Appendix E |
| 31 | A full mathematical tutorial with feature formulas and transforms is appended. | Appendix A |
| 32 | Data ingestion is performed via terminal downloads by default; the option to use versioned ZIP archives with checksum verification is kept as an alternative. | 7.5, 10.1-10.2, Appendix C |
| 33 | The roadmap is rewritten as direct imperative execution steps. | Section 10 |
| 34 | Exactly 10,000 rows are locked for B1/PTB-XL training as the large-scale benchmark; LUDB B2 intentionally preserves all eligible gold-standard records rather than duplicating rows. | 8.2, 9.2, 12.1 |
| 35 | Chapman is rejected as B2 and replaced by LUDB because B2 must rest on fiducial ground truth. | 7.1-7.4, 9.1 |

## Appendix C. Optional version-locked data package manifest

If using the optional ZIP archives, the implementation team must package and track at least the following files:

```text
data_lock/
  ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
  ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip
  lobachevsky-university-electrocardiography-database-1.0.1.zip
  DATA_LOCK_SHA256SUMS.txt
  LOCKFILE.md
```

`LOCKFILE.md` must contain:

- archive filenames,
- source citation,
- source version,
- archive SHA256,
- unpack path,
- unpack timestamp,
- operator who created the package.

When using this optional method, the benchmark pipeline must refuse to run if:

- an archive version differs,
- a checksum fails,
- or an expected archive is missing.

## Appendix D. PTB-XL <-> LUDB ontology mapping template

The following template must be finalized before any cross-dataset transportability analysis.

| PTB-XL source ontology | LUDB source ontology | Project label |
|---|---|---|
| PTB-XL NORM | LUDB normal sinus rhythm / no pathologic rhythm label | Normal |
| PTB-XL PVC / VPB / ventricular ectopy statements | LUDB ventricular extrasystole / PVC diagnosis or rule-derived PVC morphology | PVC |
| PTB-XL PAC/APB/SPAC/SVPB | LUDB atrial extrasystole / supraventricular ectopy diagnosis or rule-derived atrial premature morphology | APB |
| PTB-XL RBBB / CRBBB / IRBBB | LUDB right bundle branch block diagnosis | RBBB spectrum |
| PTB-XL LBBB / CLBBB | LUDB left bundle branch block diagnosis | LBBB spectrum |
| PTB-XL AFIB | LUDB atrial fibrillation rhythm field | AF |
| PTB-XL AFLT | LUDB atrial flutter rhythm field | AFL |
| PTB-XL paced / pacemaker metadata | LUDB pacemaker-present rhythm/metadata | Paced |
| PTB-XL other rhythm/form statements | LUDB unmatched diagnoses | Other / unmapped |

## Appendix E. References and standards used in this revision

### Uploaded primary corpus

1. `01_Thesis.pdf` - thesis on arrhythmia detection from ECG using explainable AI.
2. `02_Human-in-the-loop-approach.pdf` - human-centric approach for embedding clinically meaningful ECG knowledge.
3. `03_A-novel-feature-vector-for-ECG.pdf` - 195-element ECG feature-vector design.
4. `04_ECG-arrhythmia-classification.pdf` - triad-based arrhythmia classification and feature interpretation.
5. `05_Robust-R-peak-detection.pdf` - knowledge-integrated deep-learning R-peak detection.
6. `06_Towards-transparent-AI.pdf` - integrated explainable ECG pipeline with clinical trials.
7. `07_Explainable-deep-learning.pdf` - formal transition-matrix methodology.
8. `08_Toward-explainable-deep-learning.pdf` - scalable healthcare transition-matrix method.
9. `09_Equivariant-transition-matrices.pdf` - symmetry-aware transition operators.

### External standards and authoritative supporting sources

1. Wagner P, et al. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*.
2. Strodthoff N, et al. PTB-XL+, a comprehensive electrocardiographic feature dataset. *PhysioNet* / *Scientific Data* companion release.
3. Kalyakulina A, et al. Lobachevsky University Electrocardiography Database (LUDB). *PhysioNet*.
4. Moody GB, Mark RG. MIT-BIH Arrhythmia Database. *PhysioNet*.
5. AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram, Parts II-VI.
6. ANSI/AAMI EC57: testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.
7. Haq KT, Javadekar N, Tereshchenko LG. Detection and removal of pacing artifacts prior to automated analysis of 12-lead ECG. *Computers in Biology and Medicine*.
8. Recent comparative QT-correction studies supporting explicit formula locking and showing the instability of Bazett relative to Fridericia/Framingham at heart-rate extremes.

### Practical interpretation of the standards inside this specification

- 0.05 Hz low-cut is preserved for the diagnostic-feature branch when ST/QT fidelity matters.
- 0.67 Hz zero-phase digital drift suppression is treated only as a monitoring/detection-side compromise, not as the universal morphology branch.
- J+60/J+80 measurement conventions are explicitly locked for reproducibility.
- Fridericia is the mandatory project default for QTc reporting, with sensitivity analyses allowed.
- Conduction and repolarization thresholds are treated as rule primitives for feature construction, not as substitutes for full clinical diagnosis.