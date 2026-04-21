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

You can also run custom experiments from CLI:

```bash
python main.py --model-type ctgan --epochs 500 --batch-size 128 --n-real 2000 --n-synthetic 2000 --seed 7
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

## Approach and Lessons Learned

This project started as a baseline Gaussian Copula pipeline and then evolved with comparative CTGAN runs.
The main challenge was balancing visual fidelity with downstream model utility: synthetic data that "looks right"
does not always preserve predictive behavior. I added explicit utility tracking (accuracy, F1, ROC-AUC) alongside
distribution-level checks to avoid over-optimizing only for visual similarity.

From experimentation, reproducibility controls (seed, row counts, and model configuration) were the most useful
improvement for debugging and comparison. The CLI arguments in `main.py` reflect that lesson and make fast,
traceable experiment loops easier.

## License

This project is intended for educational and portfolio use.
## Connect
www.linkedin.com/in/yusuf-başkani-0782553a1
Markdown
# Synthetic Healthcare Data Generation

Hi! Welcome to my repository. 

I started this project because I was looking into how difficult it is to get access to real medical records (EHR) for data science projects due to privacy constraints. I wanted to see if I could build a pipeline that generates realistic mock data—preserving the statistical structure of the original dataset without leaking any sensitive patient information.

This project relies on SDV (Synthetic Data Vault) to train generative models on a sample dataset and output synthetic records that can actually be used for downstream machine learning tasks.

## What's inside?
Instead of dumping everything into one massive Jupyter Notebook, I tried to structure this more like a proper software project:
* `src/`: The core logic. I split the data loading, model training (`generator.py`), and evaluation (`evaluator.py`) into their own modules.
* `app.py`: A Streamlit dashboard. I built this so it's easier to visually compare the real vs. fake data distributions without running CLI commands every time.
* `main.py`: The entry point if you want to run the end-to-end pipeline from your terminal.

## Tech Stack
* Python (Pandas, NumPy, Scikit-learn)
* SDV (Synthetic Data Vault) - specifically testing `GaussianCopula` and `CTGAN`
* Streamlit for the UI
* Matplotlib & Seaborn for correlation heatmaps

## How to run it locally
Clone the repo and set up your virtual environment:

```bash
python -m venv venv
# On Windows use: venv\Scripts\activate
source venv/bin/activate  
pip install -r requirements.txt
You can run the pipeline from your terminal. I added some CLI arguments so you can easily play around with the epochs and batch sizes without changing the code:

Bash
python main.py --model-type ctgan --epochs 300 --batch-size 128
Or, just spin up the interactive UI:

Bash
streamlit run app.py
Lessons Learned & Realities
When I first started building this, I thought that if the synthetic data looked similar on a histogram, my job was done. I quickly realized that visual fidelity doesn't mean the data is actually useful for machine learning. A predictive model trained on the fake data would sometimes tank in accuracy when tested on real data. That's why I ended up explicitly tracking ML utility metrics (like ROC-AUC and F1 scores) alongside the standard visual comparisons.

Also, CTGAN is heavy. Running it locally with high epochs really tested my machine's patience. That's why I made sure the CLI lets you easily drop the sample size or switch back to the much faster Gaussian Copula for quick debugging.

To-Do List / Next Steps
I'm actively using this repo to learn and build my portfolio for software and data science internships, so there are a few things I plan to improve:

Privacy checks: Right now, I check if the data works, but I don't mathematically prove that a real patient's data wasn't just copy-pasted by the model (overfitting). I need to implement nearest-neighbor distance metrics.

Better config management: Some file paths are a bit hardcoded right now. I want to move these to a proper config.yaml file.

Unit Testing: I need to write some pytest scripts, especially for the data loader and evaluator modules, to catch bugs earlier.

Connect
Feel free to reach out, open an issue, or suggest improvements!
LinkedIn
