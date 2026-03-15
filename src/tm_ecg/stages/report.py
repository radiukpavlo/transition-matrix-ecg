"""Render experiment-level report artifacts."""

from __future__ import annotations

from tm_ecg.config import ProjectConfig
from tm_ecg.reporting.reports import write_bootstrap_report, write_metrics_markdown
from tm_ecg.stages.shared import write_stage_manifest


def run(config: ProjectConfig, args: object) -> int:
    experiment = getattr(args, "experiment")
    metrics_path = write_metrics_markdown(
        config.paths.reports / "metrics_report.md",
        {
            "Prediction Quality": [
                "Populate accuracy, balanced accuracy, macro-F1, AUROC, AUPRC, and calibration after model runs.",
            ],
            "Explanation Quality": [
                "Populate MAE, RMSE, rank correlations, AUROC, Brier score, Cohen kappa, and Gwet AC1.",
            ],
            "Robustness": [
                "Run all Section 11.4 ablations against the frozen manifests before final publication.",
            ],
        },
    )
    bootstrap_path = write_bootstrap_report(
        config.paths.reports / "bootstrap_ci_report.csv",
        [
            {
                "experiment": experiment,
                "metric": "placeholder",
                "point_estimate": "",
                "ci_lower": "",
                "ci_upper": "",
                "notes": "Populate after running evaluation.",
            }
        ],
    )
    write_stage_manifest(
        config,
        f"report_{experiment}",
        {
            "experiment": experiment,
            "status": "report_templates_written",
            "metrics_report": str(metrics_path),
            "bootstrap_report": str(bootstrap_path),
        },
    )
    print(f"Report templates written for {experiment}")
    return 0
