import pandas as pd
import mlflow
import xgboost as xgb
import joblib
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, roc_auc_score

print("Starting the training script...")

# --- NEW: Auto-Download Data ---
data_file = 'creditcard.csv'
if not os.path.exists(data_file):
    print(f"'{data_file}' not found. Downloading from Kaggle...")
    try:
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'mlg-ulb/creditcardfraud', '--unzip'], check=True)
        print("Download complete.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Failed to download data. Make sure the 'kaggle' command is available and your API key is set up.")
        exit()

# Load and Prepare Data
df = pd.read_csv(data_file)

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Resample Training Data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Data prepared.")

# Train and Log Model with MLflow
mlflow.set_tracking_uri("file:./mlflow_runs")
mlflow.set_experiment("Fraud_Detection_XGBoost")

with mlflow.start_run() as run:
    print(f"Starting MLflow run with ID: {run.info.run_id}")
    params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'seed': 42}
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)

    # --- NEW, SIMPLER SAVING METHOD ---
    # 1. Save model to a local file using joblib
    model_filename = "model.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved locally to {model_filename}")

    # 2. Log that single file as a simple artifact
    mlflow.log_artifact(model_filename)
    print(f"Logged {model_filename} as an MLflow artifact.")
    # --- END NEW METHOD ---

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, (y_pred_proba > 0.5).astype(int))
    auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metrics({'recall': recall, 'auc': auc})
    print(f"✅ Training complete. Recall: {recall:.4f}, AUC: {auc:.4f}")