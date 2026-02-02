import os
import yaml
import mlflow
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

## Load environment variables from .env file
load_dotenv()

## Define MLflow remote server credentials
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

## Load evaluation parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["evaluate"]

## Extract parameters
data_path = params["data_path"]  # Path to evaluation dataset
model_path = params["model_path"]  # Path to trained model pickle
target = params["target"]  # Target column name
experiment = params["experiment"]  # MLflow experiment name
run_name = params["run_name"]


def evaluate(data_path, model_path):
    ## Load evaluation data
    data = pd.read_csv(data_path)
    ## Separate features (X) and target (y)
    X = data.drop(columns=[target])
    y = data[target]
    ## Set MLflow tracking URI (remote server or local)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    ## Set the experiment in MLflow
    ## If experiment does not exist, MLflow will create it automatically
    mlflow.set_experiment(experiment)
    ## Start an MLflow run context to log metrics
    with mlflow.start_run(run_name=run_name):
        ## Load the trained model from disk (pickle)
        model = pickle.load(open(model_path, "rb"))
        ## Generate predictions on evaluation data
        predictions = model.predict(X)
        ## Compute evaluation metric (accuracy)
        ac = accuracy_score(y, predictions)
        print("Model Accuracy:", ac)
        ## Log the metric to MLflow
        mlflow.log_metric("accuracy", ac)


## Entry point of the script
if __name__ == "__main__":
    evaluate(data_path=data_path, model_path=model_path)
