# Synthetic Healthcare Data Generation

End-to-end Python project for generating privacy-aware synthetic healthcare data from a realistic Electronic Health Record (EHR)-like dataset while preserving statistical structure and downstream ML usefulness.

## Why this project?

Real healthcare data is sensitive and hard to share. This project demonstrates a reproducible workflow to:

- Generate realistic mock patient records.
- Train SDV-based synthesizers (`GaussianCopulaSynthesizer` or `CTGANSynthesizer`).
- Evaluate statistical fidelity between real and synthetic data.
- Compare ML utility via disease prediction performance.
- Export reports and visual diagnostics for transparent analysis.

## Key Features

- Modular pipeline (`src/`) with clear separation of data generation, synthesis, evaluation, and utilities.
- CLI-style pipeline execution via `main.py`.
- Interactive Streamlit interface via `app.py`.
- Automatic export of synthetic dataset, fidelity metrics, utility metrics, and correlation heatmaps.

## Tech Stack

- Python
- Pandas / NumPy
- SDV (Synthetic Data Vault)
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit

## Project Structure

```text
.
├── app.py                   # Streamlit UI
├── main.py                  # End-to-end pipeline entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Mock EHR generation / loading
│   ├── generator.py         # SDV model setup, fit, sample
│   ├── evaluator.py         # Fidelity + ML utility metrics
│   └── utils.py             # Plotting and helper utilities
└── data/
    ├── raw/
    ├── synthetic/
    └── reports/
```

## Quickstart

1) Create and activate virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run the full pipeline:

```bash
python main.py
```

4) Or run the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit App Capabilities

- Select synthesizer type (`gaussian_copula` or `ctgan`)
- Configure epochs, batch size, sample size, and seed
- Inspect real vs synthetic data previews
- View fidelity and utility results
- Display correlation comparison heatmaps
- Download synthetic CSV directly

## Output Artifacts

When pipeline execution completes, outputs are saved to:

- `data/raw/ehr_real_mock.csv`
- `data/synthetic/ehr_synthetic.csv`
- `data/reports/fidelity_metrics.csv`
- `data/reports/utility_metrics.json`
- `data/reports/figures/correlation_heatmaps.png`

## Suggested Improvements

- Add privacy risk metrics (nearest-neighbor distance, membership inference proxies)
- Add experiment tracking (MLflow / Weights & Biases)
- Add configuration file support (YAML/TOML) for reproducible runs
- Add tests and CI checks for pipeline reliability

## License

This project is intended for educational and portfolio use.
## Connect


