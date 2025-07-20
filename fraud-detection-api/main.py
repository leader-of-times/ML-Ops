import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

class Transaction(BaseModel):
    Time: float; V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float; Amount: float

# --- THE ULTIMATE FIX: Load the model file directly ---
# Docker copies model.joblib to the app's root directory, so we can just load it.
try:
    logged_model = joblib.load("model.joblib")
    print("✅ Model loaded successfully from model.joblib!")
except FileNotFoundError:
    print("CRITICAL ERROR: model.joblib not found. Build the Docker image again.")
    logged_model = None
# --- END FIX ---

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if logged_model is None:
        return {"error": "Model is not loaded."}
    
    input_df = pd.DataFrame([transaction.model_dump()])
    prediction_proba = logged_model.predict_proba(input_df)[0][1]
    prediction = int(prediction_proba > 0.5)
    return {"is_fraud": prediction, "fraud_probability": float(prediction_proba)}

@app.get("/")
def read_root():
    return {"status": "API is running"}