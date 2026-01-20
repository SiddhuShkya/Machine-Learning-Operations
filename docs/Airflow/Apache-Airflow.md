## Apache Airflow

Apache  Airflow  is  an  open-source  platform  used  to  programmatically  author, 
schedule, and monitor workflows. It allows you to define complex workflows as code 
and manage their execution. Airflow is commonly used for data pipelines, where tasks 
like  data  extraction,  transformation,  and  loading  (ETL)  are  orchestrated  across 
multiple systems

---

### Setting Up Airflow With Astronomer (Astro)

Astronomer is a managed platform for Apache Airflow that simplifies running and scaling airflow while providing additional enterprises features like monitoring, security and automation. The best thing about this platform is that it will be running your airflow within a docker container.

- Website Link : [astronomer.io](https://www.astronomer.io/)
- Dcoumentation Link : [Astronomer Docs](https://www.astronomer.io/docs/home/astronomer-documentation)

> Before setting up make sure you have astro CLI installed in your local machine.

- Run the following command to install the latest version of the Astro CLI directly to PATH:
```bash
siddhu@ubuntu:~/Desktop$ curl -sSL install.astronomer.io | sudo bash -s
```
- To install a specific version of the CLI, specify the version number as a flag at the end of the command. For example, to install the most recent release of the CLI, you would run:
```bash
siddhu@ubuntu:~/Desktop$ curl -sSL install.astronomer.io | sudo bash -s -- v1.36.0
```
> After installing astro CLI to your local machine, you may start setting up your airflow by following the below instructions : 

1. Create a new directory (airflow-astro)

```bash
siddhu@ubuntu:~/Desktop$ mkdir airflow-astro
```

2. Go inside the newly created directory and initialize an astro project

```bash
siddhu@ubuntu:~/Desktop$ cd airflow-astro/
siddhu@ubuntu:~/Desktop/airflow-astro$ astro dev init
Initialized empty Astro project in /home/siddhu/Desktop/airflow-astro
```

3. Open your VS-Code

```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ code .
```

<img src="../../images/astro-project-template.png"
     alt="Astro project template"
     style="border:1px solid white; padding:1px; background:#fff;" />

> You should see that automatically the astro project template has been initialized and it has pulled the airflow development files.

Astro Airflow Project Structure (Tree View)

```text
.
├── 📄 airflow_settings.yaml        # Airflow config overrides
├── 📁 .astro/                      # Astronomer internal configuration
│   ├── 📄 config.yaml              # Astro CLI project config
│   ├── 📄 dag_integrity_exceptions.txt  # DAGs excluded from integrity checks
│   └── 📄 test_dag_integrity_default.py # Default DAG integrity test
├── 📁 dags/                        # Airflow DAG definitions
│   ├── 📄 .airflowignore           # Files Airflow should ignore
│   └── 📄 exampledag.py            # Sample DAG
├── 📄 Dockerfile                   # Docker image definition
├── 📄 .dockerignore                # Docker ignore rules
├── 📄 .env                         # Environment variables
├── 📄 .gitignore                   # Git ignore rules
├── 📁 include/                     # Shared helpers, SQL, utilities
├── 📄 packages.txt                 # OS-level dependencies
├── 📁 plugins/                     # Custom Airflow plugins
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
└── 📁 tests/                       # Tests and validations
    └── 📁 dags/                    # DAG-related tests
        └── 📄 test_dag_example.py  # Example DAG test
```

4. Run the project

```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ astro dev start
```

> The above command will start building the Dockefile and after building it will run the airflow. You can see this if you have docker desktop in your machine.

<img src="../../images/airflow-docker-desktop.png"
     alt="Astro Conatienr in Docker Desktop"
     style="border:1px solid white; padding:1px; background:#fff;" />

5. Open Airflow UI

```bash
http://localhost:8080/
```

> Once the airflow has been started you can now access the airflow ui from the above link which is completely managed by the astro. This ui is entirely running in the docker container.

<img src="../../images/astro-airflow-ui.png"
     alt="Astro Conatienr in Docker Desktop"
     style="border:1px solid white; padding:1px; background:#fff;" />

> Below are some of the things you can do with airflow ui : 

- You can view the dags you have created (by default they are disabled).
- You can see the tasks that are inside your dag.
- You can view the tasks in the form of graphs.
- You can manually run/trigger your dag.
- You can see the schedule and the exact time when it basically has to run.
- You can also see the event logs and the code.
- Similarly, you can also view gantt, run duration, task duration, calendar and Xcom.

Note : Xcom tells you what information is being passed from one task to another task.

6. Stopping the Project/Airflow UI 

```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ sudo astro dev stop
[sudo] password for siddhu: 
✔ Project stopped
```

---

### Building your first DAG with Airflow

In here, we will be creating a basic ML pipeline, we won't be defining what kind of functionality it is basically going to do. Instead we will be discussing the basic skeleton of how we can create our DAG, define our tasks and also how we ran run it.

1. Create a new file (mlpipeline.py) inside the dags folder of your astro project directory

```text
.
├── 📄 airflow_settings.yaml       
├── 📁 .astro/                      
│   ├── 📄 config.yaml              
│   ├── 📄 dag_integrity_exceptions.txt  
│   └── 📄 test_dag_integrity_default.py 
├── 📁 dags/                        
│   ├── 📄 .airflowignore           
│   └── 📄 exampledag.py    
|   └── 📄 mlpipeline.py         # Your newly created file here 
├── 📄 Dockerfile                   
├── 📄 .dockerignore                
├── 📄 .env                         
├── 📄 .gitignore                   
├── 📁 include/                     
├── 📄 packages.txt                 
├── 📁 plugins/                     
├── 📄 README.md                    
├── 📄 requirements.txt             
└── 📁 tests/                       
    └── 📁 dags/                    
        └── 📄 test_dag_example.py  
```

2. Copy and paste the below code to your newly created python file (mlpipeline.py)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define task 1
def preprocess_data():
    print('preprocessing data...')

# Define task 2
def train_model():
    print('Training model...')

# Define task 3
def evaluate_model():
    print('Evaluating model...')


## Defining the DAG

with DAG('ml_pipline', start_date=datetime(2025, 1, 1), schedule='@weekly') as dag:
    # Define the task
    preprocess = PythonOperator(task_id='preporcess_task', python_callable=preprocess_data)
    train = PythonOperator(task_id='train_task', python_callable=train_model)
    evaluate = PythonOperator(task_id='evaluet_model', python_callable=evaluate_model)

    # set dependencies (defining order)
    preprocess >> train >> evaluate

```

3. Run the project

```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ astro dev start
```

Possible Error : 

```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ astro dev start
✔ Project image has been updated
Error: error building, (re)creating or starting project containers: Error response from daemon: failed to set up container networking: driver failed programming external connectivity on endpoint airflow-astro_ece713-postgres-1 (1d587611fb6f13e6cb4ac783859f74aa0e9aa1c4b94354c52622f1c288b20057): Bind for 127.0.0.1:5432 failed: port is already allocated
```

> If you ran into an error similarly to the above one, run the below commands to fix it :

- Option 1 : Stop existing containers (clean start)
```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ docker stop $(docker ps -q --filter "name=airflow-astro")
siddhu@ubuntu:~/Desktop/airflow-astro$ docker rm $(docker ps -aq --filter "name=airflow-astro")
```
- Option 2 : Use a different Postgres port
You can change the port mapping in your docker-compose.yaml (usually in .astro folder) for Postgres:

```bash
ports:
  - "5433:5432"
```
- Then start Astro dev again:
```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ astro dev start
```

4. Open the Airflow UI

```bash
http://localhost:8080/
```

5. In the dags, you will be able to see that you have 2 dags including the one you just created (ml_pipleline)

<img src="../../images/airflow-ui-dags.png"
     alt="Astro Conatienr in Docker Desktop"
     style="border:1px solid white; padding:1px; background:#fff;" />

6. Go inside your newly created dag and run/trigger it to see the running tasks and some important informations related to it.

<table>
  <tr>
    <td>
      <img src="../../images/triggered-dag.png" 
           alt="Astro Container in Docker Desktop" 
           style="border:1px solid white; padding:1px; background:#fff; width:800px;" />
    </td>
    <td>
      <img src="../../images/triggered-dag-logs.png" 
           alt="Astro Container in Docker Desktop" 
           style="border:1px solid white; padding:1px; background:#fff; width:800px;" />
    </td>
  </tr>
</table>

7. Stop the Project/Airflow UI 

```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ sudo astro dev stop
[sudo] password for siddhu: 
✔ Project stopped
```

> Additionally, If you want to change the code and run this again, while the container is still running, you can do this using the below command

```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ astro dev restart
```
---

### Apache Airflow With TaskFlow API

Apache Airflow introduced the TaskFlow API which allows you to create tasks using Python decorators like @task. This is a cleaner and more intuitive way of writing tasks without needing to manually use operators like PythonOperator. Airflow with taskflow api example : 

1. Create a new dag file (taskflowapi.py) inside the dags folder of your astro project directory

```text
.
├── 📄 airflow_settings.yaml       
├── 📁 .astro/                      
│   ├── 📄 config.yaml              
│   ├── 📄 dag_integrity_exceptions.txt  
│   └── 📄 test_dag_integrity_default.py 
├── 📁 dags/                        
│   ├── 📄 .airflowignore           
│   └── 📄 exampledag.py    
|   └── 📄 mlpipeline.py    
|   └── 📄 taskflowapi.py            # Your newly created dag  
├── 📄 Dockerfile                   
├── 📄 .dockerignore                
├── 📄 .env                         
├── 📄 .gitignore                   
├── 📁 include/                     
├── 📄 packages.txt                 
├── 📁 plugins/                     
├── 📄 README.md                    
├── 📄 requirements.txt             
└── 📁 tests/                       
    └── 📁 dags/                    
        └── 📄 test_dag_example.py  
```

2. Copy and paste the below code to your newly created dag file (taskflowapi.py)

```python
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

```

3. Run the project

```bash
siddhu@ubuntu:~/Desktop/airflow-astro$ astro dev start
```

4. Open the Airflow UI

```bash
http://localhost:8080/
```

5. In the dags tab, you will be able to see your newly created dag (taskflow_api_dag)

<img src="../../images/taskflow-api-dag.png"
     alt="Taskflow-API-DAG"
     style="border:1px solid white; padding:1px; background:#fff;" />


6. Go inside your newly created dag (taskflow_api_dag) and run/trigger it. If you see all tasks green after execution it means that all the tasks has been executed successfully.

<img src="../../images/taskflow-api-dag-status.png"
     alt="Taskflow-API-DAG-status"
     style="border:1px solid white; padding:1px; background:#fff;" />

7. You can now view the logs and Xcom of all your tasks. Examples :

  - Task 1 : start_number
  <table>
  <tr>
    <td>
      <img src="../../images/task1-logs.png" 
           alt="Astro Container in Docker Desktop" 
           style="border:1px solid white; padding:1px; background:#fff; width:800px;" />
    </td>
    <td>
      <img src="../../images/task1-xcom.png" 
           alt="Astro Container in Docker Desktop" 
           style="border:1px solid white; padding:1px; background:#fff; width:800px;" />
    </td>
    </tr>
  </table>

  - Task 4 : subtract_five
  <table>
  <tr>
    <td>
      <img src="../../images/task4-logs.png" 
           alt="Astro Container in Docker Desktop" 
           style="border:1px solid white; padding:1px; background:#fff; width:800px;" />
    </td>
    <td>
      <img src="../../images/task4-xcom.png" 
           alt="Astro Container in Docker Desktop" 
           style="border:1px solid white; padding:1px; background:#fff; width:800px;" />
    </td>
    </tr>
  </table>

---

# <div align="center">Thank You for Going Through This Guide! 🙏✨</div>