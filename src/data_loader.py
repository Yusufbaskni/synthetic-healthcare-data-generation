"""Data loading and mock EHR data generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def generate_mock_ehr_data(
    n_samples: int = 1000,
    random_state: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate a realistic mock EHR dataset with meaningful correlations.

    Args:
        n_samples: Number of patient records to generate.
        random_state: Seed for reproducibility.
        output_path: Optional path to save generated data as CSV.

    Returns:
        A pandas DataFrame containing synthetic "real-like" EHR records.
    """
    rng = np.random.default_rng(seed=random_state)

    age = rng.normal(loc=52, scale=16, size=n_samples).clip(18, 90).round().astype(int)
    gender = rng.choice(["Female", "Male"], size=n_samples, p=[0.52, 0.48])
    bmi = (rng.normal(27, 4.5, n_samples) + (age - 50) * 0.03).clip(16, 45).round(1)
    systolic_bp = (95 + age * 0.55 + bmi * 0.85 + rng.normal(0, 9, n_samples)).clip(90, 210).round(0)
    diastolic_bp = (55 + age * 0.28 + bmi * 0.45 + rng.normal(0, 7, n_samples)).clip(55, 130).round(0)
    cholesterol = (130 + age * 0.9 + bmi * 1.8 + rng.normal(0, 22, n_samples)).clip(110, 360).round(0)
    glucose = (70 + bmi * 1.2 + age * 0.35 + rng.normal(0, 15, n_samples)).clip(60, 290).round(0)
    hba1c = (4.5 + (glucose - 90) / 45 + rng.normal(0, 0.45, n_samples)).clip(4.0, 13.0).round(2)
    hemoglobin = (13.8 + (gender == "Male") * 1.0 - (age - 50) * 0.01 + rng.normal(0, 1.1, n_samples)).clip(9.0, 18.5).round(1)
    creatinine = (
        0.65
        + (gender == "Male") * 0.2
        + (age - 50) * 0.004
        + rng.normal(0, 0.12, n_samples)
    ).clip(0.4, 2.8).round(2)

    # Disease risk is tied to clinical indicators to create useful signal.
    logit = (
        -11.0
        + 0.045 * age
        + 0.06 * bmi
        + 0.025 * glucose
        + 0.012 * systolic_bp
        + 0.008 * cholesterol
    )
    disease_prob = 1 / (1 + np.exp(-logit))
    chronic_disease = rng.binomial(1, disease_prob, n_samples)

    smoking_status = rng.choice(
        ["Never", "Former", "Current"],
        size=n_samples,
        p=[0.52, 0.27, 0.21],
    )

    df = pd.DataFrame(
        {
            "patient_id": np.arange(1, n_samples + 1),
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "systolic_bp": systolic_bp.astype(int),
            "diastolic_bp": diastolic_bp.astype(int),
            "cholesterol": cholesterol.astype(int),
            "glucose": glucose.astype(int),
            "hba1c": hba1c,
            "hemoglobin": hemoglobin,
            "creatinine": creatinine,
            "smoking_status": smoking_status,
            "chronic_disease": chronic_disease.astype(int),
        }
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def load_csv_data(input_path: Path) -> pd.DataFrame:
    """Load tabular data from CSV.

    Args:
        input_path: Absolute or relative path to CSV file.

    Returns:
        DataFrame loaded from disk.
    """
    return pd.read_csv(input_path)

