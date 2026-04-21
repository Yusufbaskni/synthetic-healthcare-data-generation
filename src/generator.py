"""Model training and synthetic data generation using SDV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer


@dataclass
class SynthesizerConfig:
    """Configuration object for SDV synthesizer training."""

    model_type: str = "gaussian_copula"
    epochs: int = 300
    batch_size: int = 256
    verbose: bool = True


class HealthcareDataGenerator:
    """Train SDV models and sample synthetic healthcare data."""

    def __init__(self, config: SynthesizerConfig) -> None:
        self.config = config
        self.metadata = SingleTableMetadata()
        self._synthesizer: GaussianCopulaSynthesizer | CTGANSynthesizer | None = None

    def _build_synthesizer(self) -> GaussianCopulaSynthesizer | CTGANSynthesizer:
        """Create synthesizer instance according to configuration."""
        model_type = self.config.model_type.lower()
        if model_type == "gaussian_copula":
            return GaussianCopulaSynthesizer(metadata=self.metadata)
        if model_type == "ctgan":
            return CTGANSynthesizer(
                metadata=self.metadata,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                verbose=self.config.verbose,
            )
        raise ValueError("model_type must be either 'gaussian_copula' or 'ctgan'.")

    def fit(self, real_data: pd.DataFrame) -> None:
        """Detect metadata and fit the synthesizer.

        Args:
            real_data: Real-like source dataset used for model training.
        """
        self.metadata.detect_from_dataframe(data=real_data)
        self._synthesizer = self._build_synthesizer()
        self._synthesizer.fit(real_data)

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic records from the trained model.

        Args:
            n_rows: Number of synthetic rows to sample.

        Returns:
            Synthetic DataFrame with the same schema as training data.

        Raises:
            RuntimeError: If called before the model is trained.
        """
        if self._synthesizer is None:
            raise RuntimeError("Synthesizer is not trained. Call fit() first.")
        return self._synthesizer.sample(num_rows=n_rows)

    def get_model_info(self) -> dict[str, Any]:
        """Return basic metadata about the current generator instance."""
        return {
            "model_type": self.config.model_type,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "fitted": self._synthesizer is not None,
        }

