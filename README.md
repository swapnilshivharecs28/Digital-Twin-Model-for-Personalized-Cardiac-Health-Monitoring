
# Digital Twin Model for Personalized Cardiac Health Monitoring

An academic machine learning prototype that demonstrates cardiac-related data classification and visualization through a Streamlit dashboard.

## Project Overview

This project uses a Logistic Regression classifier trained on the UCI Heart Disease dataset and provides an interactive dashboard for exploring model predictions, evaluation metrics, and illustrative trends.

**Project status:** Working software prototype; real-time monitoring and clinical validation are not implemented.

## Features

- Data preprocessing using median imputation and feature standardization
- Logistic Regression classification pipeline
- Interactive Streamlit dashboard with 13 input features
- Prediction output and estimated positive-class probability
- Model evaluation metrics and confusion matrix
- Simulated 14-day trend visualization

## Technology Stack

- Python
- Pandas and NumPy
- scikit-learn
- Streamlit
- Plotly
- Joblib

## Dataset and Methodology

Dataset: UCI Machine Learning Repository — Heart Disease (ID 45).

- Records used: 303
- Training records: 227
- Test records: 76
- Split: Stratified, random_state=42
- Model: Logistic Regression
- Preprocessing: Median imputation and StandardScaler

The positive target class is defined as original `num > 0`; otherwise, it is negative.

## Reported Experimental Results

| Metric | Result |
|---|---:|
| Accuracy | 86.84% |
| Precision | 83.78% |
| Recall | 88.57% |
| F1 score | 86.11% |
| ROC-AUC | 92.82% |

These results come from one held-out test split and are not evidence of clinical performance.

## Setup and Usage

1. Install Python and create a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model, if the dataset retrieval and training script are configured:

   ```bash
   python train_model.py
   ```

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

Run the training step before starting the dashboard if the required saved model and metrics artifacts are not already present.

## Current Limitations

- The dataset is small and historical.
- Trend data is simulated, not live patient data.
- No physical sensors, wearables, or real-time data pipeline are integrated.
- Automated edge-case testing and independent-dataset validation remain pending.
- The model has not undergone clinical validation.

## Disclaimer

This project is for educational and academic demonstration only. It is not a medical device and must not be used for diagnosis, treatment, or medical decision-making.

## Reference

UCI Machine Learning Repository, Heart Disease dataset (ID 45): https://doi.org/10.24432/C52P4X
