import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

def train_model():
    mlflow.set_tracking_uri("file:///home/jovyan/mlruns")
    mlflow.set_experiment("smallbiz-credit-score")

    df = pd.read_csv("train_data.csv")
    X = df.drop("credit_score", axis=1)
    y = df["credit_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="CreditScoreModel")

    print("Model trained and logged to MLflow")

if __name__ == "__main__":
    train_model()
