import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import shutil

def train_model():
    # Set tracking URI and experiment
    mlflow.set_tracking_uri("file:///home/jovyan/mlruns")
    mlflow.set_experiment("smallbiz-credit-score")

    # Load dataset
    df = pd.read_csv("train_data.csv")
    X = df.drop("credit_score", axis=1)
    y = df["credit_score"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")

        shutil.copytree("../demo", "demo", dirs_exist_ok=True)
        mlflow.log_artifacts("demo", artifact_path="demo")

        # Add input example and signature for Swagger
        input_example = X_train.iloc[:1]
        signature = infer_signature(X_train, model.predict(X_train))

        # Log model with metadata
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="CreditScoreModel",
            signature=signature,
            input_example=input_example
        )

    print("Model trained, registered, and ready for Swagger")

if __name__ == "__main__":
    train_model()
