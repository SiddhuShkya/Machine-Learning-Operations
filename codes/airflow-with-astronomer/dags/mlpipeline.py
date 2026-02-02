from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


# Define task 1
def preprocess_data():
    print("preprocessing data...")


# Define task 2
def train_model():
    print("Training model...")


# Define task 3
def evaluate_model():
    print("Evaluating model...")


## Defining the DAG

with DAG("ml_pipline", start_date=datetime(2025, 1, 1), schedule="@weekly") as dag:
    # Define the task
    preprocess = PythonOperator(
        task_id="preporcess_task", python_callable=preprocess_data
    )
    train = PythonOperator(task_id="train_task", python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluet_model", python_callable=evaluate_model)

    # set dependencies (defining order)
    preprocess >> train >> evaluate
