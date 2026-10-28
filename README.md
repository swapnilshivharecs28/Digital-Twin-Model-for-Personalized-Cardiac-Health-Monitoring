# Digital Twin Model for Personalized Cardiac Health Monitoring

A runnable academic prototype for the Mid-Term Evaluation and Phase III implementation submissions.

## Features
- Reproducible ML training with the UCI Heart Disease dataset (ID 45).
- Data preprocessing, stratified train/test split, logistic-regression baseline and held-out metrics.
- Streamlit dashboard with patient input and model output.
- Simulated 14-day trend chart to demonstrate the visualization module.
- Saved model and metrics under `artifacts/`.

## Important limitation
This is an educational prototype, not a medical device or diagnostic system. The UCI Cleveland dataset is small, historical, and not longitudinal wearable data. The trend chart is simulated, not measured patient data. Never use the model for medical decisions.

## Windows setup
1. Install Python 3.10–3.12.
2. Extract the ZIP and open this folder in VS Code.
3. Open Terminal in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat` instead. The UCI data fetch requires internet access on the first training run.

## Generated artifacts
- `artifacts/heart_disease_model.joblib`
- `artifacts/metrics.json`

Run `train_model.py` yourself and use its actual metrics in the report. Do not invent results.

## Dataset and citation
UCI Machine Learning Repository, Heart Disease (ID 45), CC BY 4.0: https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease

Janosi, A., Steinbrunn, M., Pfisterer, J., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.

## Troubleshooting
- `ModuleNotFoundError`: activate `.venv` and run `pip install -r requirements.txt`.
- Dataset download error: check internet and rerun `python train_model.py`; alternatively place a correctly sourced CSV in `data/heart.csv` with the 13 feature columns and target column `num` (or `target`).
- Streamlit command not found: run `python -m streamlit run app.py`.
- Port busy: `python -m streamlit run app.py --server.port 8502`.

## Status at handoff
Code is prepared but has not yet been executed on your computer. Only describe training, testing, and results as completed after you run the commands and capture the outputs.
