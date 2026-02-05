import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
mlflow.set_experiment("churn_experiment")

df = pd.read_csv("data/data.csv")
X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="ChurnModel"
    )

    print("Accuracy:", acc)
