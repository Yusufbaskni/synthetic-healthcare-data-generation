# Synthetic Healthcare Data Generation

An end-to-end Python project to generate privacy-preserving synthetic healthcare data while preserving statistical patterns and useful clinical relationships from a real-like Electronic Health Record (EHR) dataset.

This repository is designed as a portfolio-grade data science project with modular architecture, type hints, and reproducible experimentation.

## Project Goals

- Simulate realistic patient-level healthcare records with clinically meaningful correlations.
- Train an SDV synthesizer (Gaussian Copula or CTGAN) on real-like data.
- Generate synthetic records that preserve statistical fidelity.
- Evaluate downstream machine learning utility using disease prediction.
- Produce reports and visual assets for transparent model comparison.

## Tech Stack

- Python
- Pandas / NumPy
- SDV (Synthetic Data Vault)
- Scikit-learn
- Matplotlib / Seaborn

## Repository Structure

```text
.
├── data/
│   ├── raw/                # Real-like source data
│   ├── synthetic/          # Generated synthetic datasets
│   └── reports/            # Evaluation outputs and figures
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Mock EHR generation and CSV loading
│   ├── generator.py        # SDV model setup, training, and sampling
│   ├── evaluator.py        # Fidelity and utility evaluation
│   └── utils.py            # Plotting and helper utilities
├── main.py                 # Full pipeline entry point
├── app.py                  # Streamlit web application
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Execute the full pipeline:

```bash
python main.py
```

## Run as Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

Features in the app:

- Model selection (`gaussian_copula` or `ctgan`)
- Configurable rows, epochs, batch size, and random seed
- Real vs synthetic preview tables
- Fidelity report preview
- Utility score comparison
- Correlation heatmap visualization
- One-click synthetic CSV download

## Pipeline Steps

1. Generate a 1,000-row mock EHR dataset with realistic correlations.
2. Detect metadata and train an SDV synthesizer.
3. Sample synthetic healthcare records.
4. Compute fidelity metrics (column-level statistical similarity).
5. Evaluate utility by comparing Logistic Regression performance:
   - model trained on real data
   - model trained on synthetic data
   - both tested on the same held-out real test set
6. Save correlation heatmaps for side-by-side comparison.

## Outputs

After running `main.py`, outputs are written to:

- `data/raw/ehr_real_mock.csv`
- `data/synthetic/ehr_synthetic.csv`
- `data/reports/fidelity_metrics.csv`
- `data/reports/utility_metrics.json`
- `data/reports/figures/correlation_heatmaps.png`

## Notes for Extension

- Switch `model_type` in `main.py` from `gaussian_copula` to `ctgan` for deep generative modeling.
- Add privacy risk checks (for example nearest-neighbor distance or membership inference tests).
- Add hyperparameter search for synthesizer and downstream models.
- Integrate experiment tracking (MLflow / Weights & Biases) for versioned benchmarking.

## License

This project is intended for educational and portfolio purposes.

