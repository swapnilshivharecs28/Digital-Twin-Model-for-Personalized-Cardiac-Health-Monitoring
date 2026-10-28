from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / 'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
COLS = FEATURES + ['num']

def load_data():
    local = ROOT / 'data' / 'heart.csv'
    if local.exists():
        df = pd.read_csv(local)
        if 'target' in df.columns and 'num' not in df.columns:
            df = df.rename(columns={'target':'num'})
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(f'Local CSV missing columns: {missing}')
        source = 'Local CSV (verify source in README)'
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=45)
        df = ds.data.features.copy()
        df['num'] = ds.data.targets.iloc[:, 0]
        source = 'UCI Machine Learning Repository, Heart Disease dataset (ID 45)'
    df = df[COLS].copy()
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['num','age','sex'])
    y = (df['num'] > 0).astype(int)
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    return X, y, source, len(df)

def main():
    X, y, source, n_rows = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'dataset': source, 'dataset_rows_used': int(n_rows), 'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)), 'random_state': 42,
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred, zero_division=0)),
        'f1': float(f1_score(y_test, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, proba)),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
        'target_definition': '1 if UCI num > 0, otherwise 0',
        'disclaimer': 'Exploratory educational model; not for clinical use.'
    }
    joblib.dump(model, ARTIFACTS / 'heart_disease_model.joblib')
    (ARTIFACTS / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('TRAINING COMPLETE')
    print(json.dumps(metrics, indent=2))
    print('\nSaved model and metrics in artifacts/. Now run: streamlit run app.py')

if __name__ == '__main__':
    main()
