import mlflow  # Import the MLflow library for experiment tracking

mlflow.set_tracking_uri("http://localhost:5000")  # Set the MLflow tracking server URI

mlflow.set_experiment(
    "my_experiment"
)  # Specify or create an experiment named "my_experiment"

with (
    mlflow.start_run()
):  # Start a new MLflow run (context manager ensures it ends properly)
    mlflow.log_param("param1", 5)  # Log a parameter named "param1" with value 5
    mlflow.log_metric("metric1", 0.85)  # Log a metric named "metric1" with value 0.85
