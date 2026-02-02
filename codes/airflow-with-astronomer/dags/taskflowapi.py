from airflow import DAG
from airflow.decorators import task
from datetime import datetime

# Define the DAG

with DAG(
    "taskflow_api_dag",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:
    # Define task 1 : Start wth an initial number
    @task
    def start_number():
        initial_value = 15
        print(f"Starting number: {initial_value}")
        return initial_value

    # Define task 2 : Increment the number
    @task
    def add_ten(number: int):
        incremented_value = number + 10
        print(f"Add 10 : {number} + 10 = {incremented_value}")
        return incremented_value

    # Define task 3 : Multiply the number
    @task
    def multiply_by_two(number: int):
        multiplied_value = number * 2
        print(f"Multiply by 2 : {number} * 2 = {multiplied_value}")
        return multiplied_value

    # Define task 4 : Subtract five from the number
    @task
    def subtract_five(number: int):
        subtracted_value = number - 5
        print(f"Subtract 5 : {number} - 5 = {subtracted_value}")
        return subtracted_value

    # Define task 5 : Square the number
    @task
    def square_number(number: int):
        squared_value = number * number
        print(f"Square : {number} ^ 2 = {squared_value}")
        return squared_value

    # Set the task dependencies
    initial_number = start_number()
    incremented_number = add_ten(initial_number)
    multiplied_number = multiply_by_two(incremented_number)
    subtracted_number = subtract_five(multiplied_number)
    squared_number = square_number(subtracted_number)
