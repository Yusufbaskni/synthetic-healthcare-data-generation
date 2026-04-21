"""End-to-end pipeline for synthetic healthcare data generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data_loader import generate_mock_ehr_data
from src.evaluator import compute_fidelity_metrics, evaluate_ml_utility
from src.generator import HealthcareDataGenerator, SynthesizerConfig
from src.utils import ensure_directory, save_correlation_heatmaps


def configure_logging() -> None:
    """Configure console logging for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_pipeline() -> None:
    """Run full workflow: data generation, model training, evaluation, and export."""
    configure_logging()
    logger = logging.getLogger(__name__)

    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    synthetic_dir = data_dir / "synthetic"
    reports_dir = data_dir / "reports"
    figures_dir = reports_dir / "figures"

    for directory in [raw_dir, synthetic_dir, reports_dir, figures_dir]:
        ensure_directory(directory)

    logger.info("Step 1/5 - Generating mock EHR dataset")
    real_path = raw_dir / "ehr_real_mock.csv"
    real_data = generate_mock_ehr_data(n_samples=1000, random_state=42, output_path=real_path)
    logger.info("Real-like dataset created at %s", real_path.as_posix())

    logger.info("Step 2/5 - Training SDV synthesizer")
    config = SynthesizerConfig(
        model_type="gaussian_copula",
        epochs=300,
        batch_size=256,
        verbose=True,
    )
    generator = HealthcareDataGenerator(config=config)
    generator.fit(real_data=real_data)
    logger.info("Model training complete: %s", generator.get_model_info())

    logger.info("Step 3/5 - Sampling synthetic dataset")
    synthetic_data = generator.sample(n_rows=len(real_data))
    synthetic_path = synthetic_dir / "ehr_synthetic.csv"
    synthetic_data.to_csv(synthetic_path, index=False)
    logger.info("Synthetic dataset saved to %s", synthetic_path.as_posix())

    logger.info("Step 4/5 - Computing fidelity metrics")
    fidelity_df = compute_fidelity_metrics(real_data=real_data, synthetic_data=synthetic_data)
    fidelity_path = reports_dir / "fidelity_metrics.csv"
    fidelity_df.to_csv(fidelity_path, index=False)
    logger.info("Fidelity report saved to %s", fidelity_path.as_posix())

    logger.info("Step 5/5 - Evaluating downstream ML utility")
    utility_results = evaluate_ml_utility(
        real_data=real_data,
        synthetic_data=synthetic_data,
        target_column="chronic_disease",
        random_state=42,
    )
    utility_path = reports_dir / "utility_metrics.json"
    utility_path.write_text(json.dumps(utility_results, indent=2), encoding="utf-8")
    logger.info("Utility report saved to %s", utility_path.as_posix())

    logger.info("Generating correlation comparison heatmaps")
    plot_columns = [
        "age",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol",
        "glucose",
        "hba1c",
        "chronic_disease",
    ]
    heatmap_path = figures_dir / "correlation_heatmaps.png"
    save_correlation_heatmaps(
        real_data=real_data,
        synthetic_data=synthetic_data,
        columns=plot_columns,
        output_path=heatmap_path,
    )
    logger.info("Heatmap figure saved to %s", heatmap_path.as_posix())
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()

