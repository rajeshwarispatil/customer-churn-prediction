import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
from xgboost import XGBClassifier

#Load dataset
df = pd.read_csv("Dataset.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())

#Data cleaning
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)
print(df.isnull().sum())

# Churn Distribution
plt.figure(figsize=(6,5))
sns.countplot(x="Churn",data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.savefig("churn_distribution.png")
plt.show()
# Monthly Charges vs Churn
plt.figure(figsize=(8,5))
sns.boxplot(x="Churn",y="MonthlyCharges",data=df)
plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.savefig("monthlycharges_churn.png")
plt.show()
#correlation heatmap
temp = df.copy()
for col in temp.columns:
    if temp[col].dtype == "object":
        temp[col] = LabelEncoder().fit_transform(temp[col])
plt.figure(figsize=(12,10))
sns.heatmap(temp.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
#Prepocessing
df.drop("customerID", axis=1, inplace=True)
label = LabelEncoder()
df["Churn"] = label.fit_transform(df["Churn"])
X = df.drop("Churn", axis=1)
y = df["Churn"]
X = pd.get_dummies(X,drop_first=True)
joblib.dump(list(X.columns),"model_columns.pkl")
#train-est split
X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.2,random_state=42,stratify=y)
#ML Flow
mlflow.set_experiment("Customer_Churn_Prediction")
#Train XGBoost
model = XGBClassifier( n_estimators=300,learning_rate=0.05,max_depth=6,random_state=42,eval_metric="logloss")
model.fit( X_train, y_train)
#Predictions
prediction = model.predict(X_test)
probability = model.predict_proba(X_test)[:,1]
#Evaluation
accuracy = accuracy_score(y_test,prediction)
precision = precision_score(y_test,prediction)
recall = recall_score(y_test,prediction)
f1 = f1_score(y_test,prediction)
roc = roc_auc_score(y_test,probability)
print(classification_report(y_test, prediction))
#confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, prediction, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
#ROC curve
fpr,tpr,_=roc_curve(y_test,probability)
plt.figure(figsize=(7,5))
plt.plot(fpr,tpr,label="XGBoost")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)
importance.sort_values(
    ascending=False
).head(10).plot.barh(figsize=(8,6))
plt.title("Top 10 Important Features")
plt.show()
joblib.dump(
    model,
    "customer_churn_model.pkl"
)
#ML logging
with mlflow.start_run():
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)
    mlflow.log_metric("ROC AUC", roc)

    mlflow.xgboost.log_model(
        xgb_model=model,
        name="Customer_Churn_Model"
    )
#ML registery
with mlflow.start_run():

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)
    mlflow.log_metric("ROC AUC", roc)

    model_info = mlflow.xgboost.log_model(
        xgb_model=model,
        name="Customer_Churn_Model"
    )

    mlflow.register_model(
        model_uri=model_info.model_uri,
        name="Customer_Churn_Model"
    )