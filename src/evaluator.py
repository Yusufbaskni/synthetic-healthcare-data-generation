"""Evaluation routines for fidelity and utility of synthetic data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def compute_fidelity_metrics(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> pd.DataFrame:
    """Compute column-wise descriptive similarity metrics.

    For numerical features: compare mean, std and quartiles.
    For categorical features: compare top category frequency.
    """
    results: list[dict[str, Any]] = []

    for column in real_data.columns:
        if column not in synthetic_data.columns:
            continue

        real_col = real_data[column]
        syn_col = synthetic_data[column]

        if pd.api.types.is_numeric_dtype(real_col):
            real_stats = real_col.describe(percentiles=[0.25, 0.5, 0.75])
            syn_stats = syn_col.describe(percentiles=[0.25, 0.5, 0.75])
            results.append(
                {
                    "column": column,
                    "type": "numeric",
                    "mean_abs_diff": abs(real_stats["mean"] - syn_stats["mean"]),
                    "std_abs_diff": abs(real_stats["std"] - syn_stats["std"]),
                    "median_abs_diff": abs(real_stats["50%"] - syn_stats["50%"]),
                    "q1_abs_diff": abs(real_stats["25%"] - syn_stats["25%"]),
                    "q3_abs_diff": abs(real_stats["75%"] - syn_stats["75%"]),
                }
            )
        else:
            real_top = real_col.value_counts(normalize=True, dropna=False)
            syn_top = syn_col.value_counts(normalize=True, dropna=False)
            real_mode = real_top.index[0] if not real_top.empty else None
            syn_mode = syn_top.index[0] if not syn_top.empty else None
            results.append(
                {
                    "column": column,
                    "type": "categorical",
                    "mode_match": int(real_mode == syn_mode),
                    "top_freq_abs_diff": abs(
                        (real_top.iloc[0] if len(real_top) else 0.0)
                        - (syn_top.iloc[0] if len(syn_top) else 0.0)
                    ),
                }
            )

    return pd.DataFrame(results)


def evaluate_ml_utility(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str = "chronic_disease",
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Train and compare disease prediction models on real vs synthetic data.

    Both models are evaluated on the same held-out split from real data.
    """
    if target_column not in real_data.columns or target_column not in synthetic_data.columns:
        raise ValueError(f"Target column '{target_column}' must exist in both datasets.")

    x_real = real_data.drop(columns=[target_column, "patient_id"], errors="ignore")
    y_real = real_data[target_column].astype(int)

    x_syn = synthetic_data.drop(columns=[target_column, "patient_id"], errors="ignore")
    y_syn = synthetic_data[target_column].astype(int)

    x_train_real, x_test_real, y_train_real, y_test_real = train_test_split(
        x_real, y_real, test_size=0.25, random_state=random_state, stratify=y_real
    )

    numeric_features = x_real.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in x_real.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    def _fit_and_score(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test_real)
        y_proba = model.predict_proba(x_test_real)[:, 1]

        return {
            "accuracy": float(accuracy_score(y_test_real, y_pred)),
            "f1": float(f1_score(y_test_real, y_pred)),
            "roc_auc": float(roc_auc_score(y_test_real, y_proba)),
        }

    real_scores = _fit_and_score(x_train_real, y_train_real)
    synthetic_scores = _fit_and_score(x_syn, y_syn)

    return {"trained_on_real": real_scores, "trained_on_synthetic": synthetic_scores}

