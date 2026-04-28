# Medical Protocols and Standards Used in the ECG Transition Matrix Research Repository

## 1. Step-by-Step Discovery and Analysis

1.  **Repository Analysis:** I analyzed the core specification files of the research repository, specifically targeting `AGENTS.md` (the implementation blueprint) and the project reports (`reports/b_matrix_generation_report.md`).
2.  **Identification of Standards:** During the analysis, I discovered that the repository strictly enforces and relies on two exact medical standards for ECG feature construction, clinical baseline definitions, and algorithm validation:
    *   **AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram** (Specifically Parts II-VI)
    *   **ANSI/AAMI EC57**
3.  **Internet Research:** I queried the internet for the exact titles of these standards to retrieve their official context, publication authorities, and references.

---

## 2. Standard 1: AHA/ACCF/HRS Recommendations

### Utilization in the Repository
This standard is utilized as the clinical foundation for the project's **ECG feature ontology**. The repository explicitly mandates that all operational thresholds, interval definitions, and morphology criteria for matrix `B` (the clinician-facing features) must be anchored to this standard. 

Specific implementations in the codebase based on this standard include:
*   Defining what constitutes a valid ST-segment level measurement (e.g., measuring at the J+60 or J+80 sample relative to the PR isoelectric baseline).
*   Formally defining criteria for pathologic T-wave inversion.
*   Standardizing measurements of the QT interval (total ventricular depolarization and repolarization duration).
*   Evaluating intraventricular conduction delays and bundle branch blocks.

### Overview and Official References
The AHA/ACCF/HRS Recommendations constitute a comprehensive, multi-part scientific statement endorsed by the **American Heart Association (AHA)**, the **American College of Cardiology Foundation (ACCF)**, and the **Heart Rhythm Society (HRS)**. It is the definitive modern guideline for how digital electrocardiograms should be measured, categorized, and interpreted.

*   **Part I:** The Electrocardiogram and Its Technology
*   **Part II:** Electrocardiography Diagnostic Statement List
*   **Part III:** Intraventricular Conduction Disturbances
*   **Part IV:** The ST Segment, T and U Waves, and the QT Interval
*   **Part V:** Electrocardiogram Changes Associated With Cardiac Chamber Hypertrophy
*   **Part VI:** Acute Ischemia and Infarction

**Official Link / Citation Example:**
*   [Circulation (AHA Journals) - Part IV: The ST Segment, T and U Waves, and the QT Interval](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.108.191092)

---

## 3. Standard 2: ANSI/AAMI EC57

### Utilization in the Repository
The `ANSI/AAMI EC57` standard is utilized for **algorithm evaluation and reporting practice**. The repository mandates that testing, benchmarking, and reporting the performance results of the AI's cardiac rhythm and ST segment measurement algorithms must follow these guidelines where applicable. This ensures that the deep learning models are evaluated using rigorous, industry-recognized, medical-device-grade statistical methodologies, rather than purely academic metrics.

### Overview and Official References
`ANSI/AAMI EC57:2012/(R)2020` is a widely recognized medical device standard developed by the **Association for the Advancement of Medical Instrumentation (AAMI)** and approved by the **American National Standards Institute (ANSI)**. It is often cited in FDA regulatory submissions for medical software.

The standard establishes standardized methods for evaluating and reporting the accuracy of algorithms used to measure cardiac rhythm and ST segments, ensuring consistency across different devices and research implementations.

**Official Link / Citation:**
*   **Standard Title:** *ANSI/AAMI EC57:2012/(R)2020: Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms*
*   [ANSI Webstore - ANSI/AAMI EC57](https://webstore.ansi.org/Standards/AAMI/ANSIAAMIEC572012R2020)
