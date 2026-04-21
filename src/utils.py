"""Visualization and helper utilities for synthetic data evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_directory(path: Path) -> None:
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_correlation_heatmaps(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    columns: Iterable[str],
    output_path: Path,
) -> None:
    """Save side-by-side correlation heatmaps for real and synthetic data."""
    selected = list(columns)
    real_corr = real_data[selected].corr(numeric_only=True)
    synthetic_corr = synthetic_data[selected].corr(numeric_only=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(real_corr, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", square=True)
    axes[0].set_title("Real Data Correlation")
    sns.heatmap(synthetic_corr, ax=axes[1], annot=True, fmt=".2f", cmap="coolwarm", square=True)
    axes[1].set_title("Synthetic Data Correlation")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

