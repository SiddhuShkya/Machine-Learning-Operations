import os
import pickle
import mlflow
import yaml
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Load environment variables from .env file
load_dotenv()

## Define MLflow remote server credentials
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

## Load training parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

## Extract required parameters
input_path = params["train"]["input"]
target = params["train"]["target"]
model_path = params["train"]["model"]
random_state = params["train"]["random_state"]
param_grid = params["train"]["param_grid"]
experiment = params["train"]["experiment"]
run_name = params["train"]["run_name"]


## Function to perform hyperparameter tuning using GridSearchCV
def hyperparameter_tuning(X_train, y_train, param_grid):
    ## Initialize Random Forest model
    rf = RandomForestClassifier()
    ## Setup GridSearch with cross-validation
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )
    ## Train models for all parameter combinations
    grid_search.fit(X_train, y_train)
    return grid_search


## Main training function
def train(data_path, target, model_path, random_state, param_grid):
    ## Load dataset
    data = pd.read_csv(data_path)
    ## Separate features and target
    X = data.drop(columns=[target])
    y = data[target]
    ## Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    ## Infer MLflow model signature from training data
    signature = infer_signature(X_train, y_train)
    ## Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    ## Set MLflow experiment name
    mlflow.set_experiment(experiment)
    ## Start MLflow experiment run
    with mlflow.start_run(run_name=run_name):
        ## Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(
            X_train=X_train, y_train=y_train, param_grid=param_grid
        )

        ## Retrieve the best model from GridSearch
        best_model = grid_search.best_estimator_
        ## Make predictions on test data
        y_pred = best_model.predict(X_test)
        ## Evaluate model performance
        ac = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        ## Log evaluation metrics to MLflow
        mlflow.log_metric("accuracy", ac)
        ## Log best hyperparameters
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param(
            "best_sample_split", grid_search.best_params_["min_samples_split"]
        )
        mlflow.log_param(
            "best_min_samples_leaf", grid_search.best_params_["min_samples_leaf"]
        )
        ## Log confusion matrix and classification report as artifacts
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")
        ## Log model and its artifacts
        mlflow.sklearn.log_model(sk_model=best_model, name="model", signature=signature)
        ## Create directory to save the trained model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ## Save the best model using pickle
        pickle.dump(best_model, open(model_path, "wb"))
        print(f"Best Model saved to : {model_path}")


## Entry point of the script
if __name__ == "__main__":
    train(
        data_path=input_path,
        target=target,
        model_path=model_path,
        random_state=random_state,
        param_grid=param_grid,
    )
