"""Streamlit app for synthetic healthcare data generation and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import generate_mock_ehr_data
from src.evaluator import compute_fidelity_metrics, evaluate_ml_utility
from src.generator import HealthcareDataGenerator, SynthesizerConfig
from src.utils import ensure_directory, save_correlation_heatmaps


def _prepare_output_dirs() -> dict[str, Path]:
    """Create and return project output directories."""
    data_dir = Path("data")
    paths = {
        "raw": data_dir / "raw",
        "synthetic": data_dir / "synthetic",
        "reports": data_dir / "reports",
        "figures": data_dir / "reports" / "figures",
    }
    for directory in paths.values():
        ensure_directory(directory)
    return paths


def _run_generation_flow(
    model_type: str, epochs: int, batch_size: int, n_real: int, n_synthetic: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]], Path]:
    """Run train-generate-evaluate flow and save outputs to disk."""
    paths = _prepare_output_dirs()

    real_path = paths["raw"] / "ehr_real_mock.csv"
    real_data = generate_mock_ehr_data(
        n_samples=n_real,
        random_state=seed,
        output_path=real_path,
    )

    config = SynthesizerConfig(
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )
    generator = HealthcareDataGenerator(config=config)
    generator.fit(real_data)

    synthetic_data = generator.sample(n_rows=n_synthetic)
    synthetic_path = paths["synthetic"] / "ehr_synthetic.csv"
    synthetic_data.to_csv(synthetic_path, index=False)

    fidelity_df = compute_fidelity_metrics(real_data=real_data, synthetic_data=synthetic_data)
    fidelity_df.to_csv(paths["reports"] / "fidelity_metrics.csv", index=False)

    utility = evaluate_ml_utility(real_data=real_data, synthetic_data=synthetic_data)
    (paths["reports"] / "utility_metrics.json").write_text(
        json.dumps(utility, indent=2), encoding="utf-8"
    )

    heatmap_path = paths["figures"] / "correlation_heatmaps.png"
    save_correlation_heatmaps(
        real_data=real_data,
        synthetic_data=synthetic_data,
        columns=[
            "age",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "cholesterol",
            "glucose",
            "hba1c",
            "chronic_disease",
        ],
        output_path=heatmap_path,
    )

    return real_data, synthetic_data, fidelity_df, utility, heatmap_path


def main() -> None:
    """Render Streamlit UI and execute generation workflow."""
    st.set_page_config(page_title="Synthetic Healthcare Data Generator", layout="wide")
    st.title("Synthetic Healthcare Data Generation")
    st.write(
        "Train an SDV synthesizer on real-like EHR data, generate synthetic records, "
        "and evaluate fidelity and utility."
    )

    with st.sidebar:
        st.header("Configuration")
        model_type = st.selectbox("Model Type", options=["gaussian_copula", "ctgan"], index=0)
        epochs = st.slider("CTGAN Epochs", min_value=100, max_value=1000, value=300, step=50)
        batch_size = st.select_slider("Batch Size", options=[64, 128, 256, 512], value=256)
        n_real = st.number_input("Mock Real Dataset Rows", min_value=500, max_value=20000, value=1000, step=100)
        n_synthetic = st.number_input(
            "Synthetic Dataset Rows", min_value=500, max_value=50000, value=1000, step=100
        )
        seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

    if st.button("Run Pipeline", type="primary"):
        with st.spinner("Training model and generating synthetic data..."):
            real_data, synthetic_data, fidelity_df, utility, heatmap_path = _run_generation_flow(
                model_type=model_type,
                epochs=int(epochs),
                batch_size=int(batch_size),
                n_real=int(n_real),
                n_synthetic=int(n_synthetic),
                seed=int(seed),
            )

        st.success("Pipeline completed successfully.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Real-like Data Preview")
            st.dataframe(real_data.head(10), use_container_width=True)
        with col2:
            st.subheader("Synthetic Data Preview")
            st.dataframe(synthetic_data.head(10), use_container_width=True)

        st.subheader("Utility Scores")
        util_col1, util_col2, util_col3 = st.columns(3)
        util_col1.metric("Accuracy Gap", f"{abs(utility['trained_on_real']['accuracy'] - utility['trained_on_synthetic']['accuracy']):.4f}")
        util_col2.metric("F1 Gap", f"{abs(utility['trained_on_real']['f1'] - utility['trained_on_synthetic']['f1']):.4f}")
        util_col3.metric("ROC-AUC Gap", f"{abs(utility['trained_on_real']['roc_auc'] - utility['trained_on_synthetic']['roc_auc']):.4f}")

        st.json(utility)

        st.subheader("Fidelity Metrics (Top 20 Rows)")
        st.dataframe(fidelity_df.head(20), use_container_width=True)

        st.subheader("Correlation Comparison")
        st.image(str(heatmap_path), caption="Real vs Synthetic Correlation Heatmaps")

        st.download_button(
            label="Download Synthetic CSV",
            data=synthetic_data.to_csv(index=False).encode("utf-8"),
            file_name="ehr_synthetic.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

