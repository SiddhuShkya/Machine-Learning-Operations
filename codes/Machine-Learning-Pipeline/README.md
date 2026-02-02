# Machine Learning Pipeline With DVC, MLflow, and DagsHub

This project demonstrates how to build a robust, end-to-end machine learning pipeline using **DVC (Data Version Control)** for data and pipeline management, **MLflow** for experiment tracking and model registry, and **DagsHub** as the central collaboration platform.

The pipeline is designed to train a Random Forest Classifier on the Pima Indians Diabetes Dataset. It is structured into clear, reproducible stages: data preprocessing, model training, and evaluation.

## Key Technologies

*   **DVC (Data Version Control)**: Used for versioning datasets and defining the pipeline stages (`preprocess`, `train`, `evaluate`). It ensures reproducibility by tracking data, code, and parameters.
*   **MLflow**: Used for tracking experiments, logging metrics (accuracy, confusion matrix), parameters (hyperparameters), and artifacts (trained models). It also serves as a model registry.
*   **DagsHub**: Acts as the remote storage for DVC and the hosting server for MLflow. It provides a unified interface to view code, data, and experiments.

## Pipeline Architecture

The pipeline is defined in `dvc.yaml` and consists of the following stages:

1.  **Preprocess** (`src/preprocess.py`):
    *   **Input**: Raw data (`data/raw/data.csv`).
    *   **Action**: Cleans and prepares the data.
    *   **Output**: Processed data (`data/processed/data.csv`).

2.  **Train** (`src/train.py`):
    *   **Input**: Processed data.
    *   **Action**: Trains a Random Forest Classifier. It performs hyperparameter tuning using GridSearchCV.
    *   **Tracking**: Logs parameters, metrics, and the trained model to MLflow.
    *   **Output**: Trained model (`models/model.pkl`).

3.  **Evaluate** (`src/evaluate.py`):
    *   **Input**: Raw data and the trained model.
    *   **Action**: Evaluates the model on the dataset (or a separate holdout set if configured).
    *   **Output**: Evaluation metrics.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://dagshub.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory to store your MLflow credentials (you can copy `.env.example` if it exists, or use the template below):

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo-name>.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

> **Note**: You can find these credentials in your DagsHub repository settings under the "Remote" button.

## Configuration

The pipeline parameters are managed in `params.yaml`. You can modify these values to experiment with different settings without changing the code:

*   **preprocess**: Input and output paths.
*   **train**: Hyperparameter grids for Random Forest (`n_estimators`, `max_depth`, etc.) and random seed.
*   **evaluate**: Data and model paths.

## Running the Pipeline

To run the entire pipeline (or only the stages that have changed), use DVC:

```bash
dvc repro
```

This command will:
1.  Check dependencies for changes.
2.  Execute the necessary stages (`preprocess`, `train`, `evaluate`).
3.  Update `dvc.lock` with the new state.

## Experiment Tracking

After running the pipeline, you can view your experiments on the DagsHub MLflow server.
1.  Go to your DagsHub repository.
2.  Click on the **Experiments** tab.
3.  You will see the runs logged by MLflow, including:
    *   **Parameters**: `n_estimators`, `max_depth`, etc.
    *   **Metrics**: `accuracy`.
    *   **Artifacts**: `model`, `confusion_matrix.txt`, `classification_report.txt`.