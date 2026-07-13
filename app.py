from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
app = FastAPI()
model = joblib.load("customer_churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.get("/health")
def health():
    return {"status": "Healthy"}

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    result = "Churn" if prediction == 1 else "No Churn"
    return {
        "Prediction": result,
        "Probability": round(float(probability), 4)
    }