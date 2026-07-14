# 🚀 Customer Churn Prediction API

A Machine Learning REST API built using **FastAPI**, **XGBoost**, and **Docker** to predict customer churn.

## 🌐 Live Demo
**API:** https://customer-churn-prediction-or62.onrender.com

**Swagger UI:** https://customer-churn-prediction-or62.onrender.com/docs

## 📌 Features
- Customer Churn Prediction
- FastAPI REST API
- XGBoost Model
- Dockerized Application
- Deployed on Render
- Interactive Swagger UI
- Feature Importance Visualization
- Pydantic Input Validation

## 🛠️ Tech Stack
- Python
- FastAPI
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Uvicorn
- Docker
- Render

## 📂 Project Structure

```text
customer-churn-prediction/
│── app.py
│── training.py
│── customer_churn_model.pkl
│── model_columns.pkl
│── Dataset.csv
│── Dockerfile
│── requirements.txt
│── README.md
│── feature_importance.png
│── churn_distribution.png
│── monthlycharges_churn.png
└── .gitignore
```

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/rajeshwarispatil/customer-churn-prediction.git
cd customer-churn-prediction
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## 🐳 Docker

Build

```bash
docker build -t customer-churn-api .
```

Run

```bash
docker run -p 8000:8000 customer-churn-api
```

## ☁️ Deployment

The application is deployed on **Render** using Docker.

Live URL:

https://customer-churn-prediction-or62.onrender.com

## 📊 Model

**Algorithm:** XGBoost Classifier

**Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## 📈 Visualizations
- Customer Churn Distribution
- Monthly Charges vs Churn
- Feature Importance

## 📥 Sample Request

```json
{
  "gender":"Female",
  "SeniorCitizen":0,
  "Partner":"No",
  "Dependents":"No",
  "tenure":5,
  "PhoneService":"Yes",
  "MultipleLines":"No",
  "InternetService":"Fiber optic",
  "OnlineSecurity":"No",
  "OnlineBackup":"No",
  "DeviceProtection":"No",
  "TechSupport":"No",
  "StreamingTV":"No",
  "StreamingMovies":"No",
  "Contract":"Month-to-month",
  "PaperlessBilling":"Yes",
  "PaymentMethod":"Electronic check",
  "MonthlyCharges":90.5,
  "TotalCharges":452.5
}
```

## 📤 Sample Response

```json
{
  "Prediction":"Churn",
  "Probability":0.9234
}

