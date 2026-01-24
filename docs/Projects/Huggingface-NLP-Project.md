## Text Summarization : An NLP Project With Huggingface 


In this project, we are going to implement a text summarization using the various open-source models that are provided by huggingface. We will also be fine tuning the models for summarization

---

### 1. Project Structure & GitHub Repo Setup

In this section, we are going start our project implementation. We are going to set up our project github repository and create an initial project structure suitable for the project.

1.1 Firstly, let's create an github repositorty with the project name 'Text-Summarizer-With-HF'

<img src="../../images/text-summarizer-repo.png"
    alt="Image Caption"
    style="border:1px solid white; padding:1px; background:#fff; width: 3000px;" />

> Also clone the newly created project repo into our local machine, from withing the colned repo open up your vs-code

```sh
siddhu@ubuntu:~/Desktop$ git clone git@github.com:SiddhuShkya/Text-Summarizer-With-HF.git
Cloning into 'Text-Summarizer-With-HF'...
remote: Enumerating objects: 4, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 4 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
Receiving objects: 100% (4/4), 12.75 KiB | 12.75 MiB/s, done.
siddhu@ubuntu:~/Desktop$ cd Text-Summarizer-With-HF/
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ code .
```

<img src="../../images/text-summarizer-vscode.png"
    alt="Image Caption"
    style="border:1px solid white; padding:1px; background:#fff; width: 3000px;" />

1.2 Create and activate an environment for this project using the below commands

```sh
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ conda create -p venv python==3.10
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ conda activate venv/
```

1.3 Create 2 new files requirements.txt and template.py, and copy paste the below dependences and python script to them respectively.

> Copy paste the below to the requirements.txt file

```text
## requirements.txt

transformers
transformers[sentencepiece]
datasets
sacrebleu
rouge_score
py7zr
pandas
nltk
tqdm
PyYAML
matplotlib
torch
notebook
boto3
mypy-boto3-s3
python-box==6.0.2
ensure==1.0.2
uvicorn==0.18.3
Jinja2==3.1.2
```

> Install the dependencies to your conda environment, using the requirements.txt file

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ pip install -r requirements.txt 
```

> While the installion is being done, lets also complete our template.py, copy paste the below script to template.py

```python
## template.py

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "textSummarizer"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "params.yaml",
    "config/config.yaml",
    ".gitignore",
    "app.py",
    "main.py",
    "requirements.txt",
    "Dockerfile",
    "setup.py",
    "research/research-notebook.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}, skipping creation.")
```

> This file helps to create all required directories and empty files for our project in a consistent structure, instead of having to do it manually. This is especially useful in:

- MLOps projects
- Production-grade ML pipelines
- Team environments

1.4 Initiate our project structure using the template.py

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ python template.py 
```

> After, running the above command our project structure should look something like this:

```sh
.
├── app.py
├── config
│   └── config.yaml
├── Dockerfile
├── .gitignore
├── .git
├── .github
│   └── workflows
├── LICENSE
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
│   └── research-notebook.ipynb
├── setup.py
├── src
│   └── textSummarizer
├── template.py
└── venv
```

1.5 Add the below two folders to .gitignore file as we dont have to track them 

```text
## .gitignore

/venv
/artifcats
```

1.6 Add, commit and push our changes to the main branch of our github repository

```sh
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git add .
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git commit -m 'Initial Project Structure'
(/home/siddhu/Desktop/Text-Summarizer-With-HF/venv) siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git push origin main
```

> Verify the commit from the github repository page by reloading it

<img src="../../images/text-summarizer-git-commit.png"
    alt="Image Caption"
    style="border:1px solid white; padding:1px; background:#fff; width: 3000px;" />

> The Project Structure setup for this project has been completed.

### 2. Logging & Utils Common Functionalities

In this section we are going to implement the common utility functions and logging setup required across the project. These components will help us:

- Track training progress
- Save model checkpoints
- Log evaluation metrics
- Handle reusable helper functions
- Improve reproducibility

2.1 Let's begin by implementing our basic logging functionalites. 

> For logging we will be using the file shown below 

```text
├── src
│   └── textSummarizer
│       ├── components
│       ├── config
│       ├── constants
│       ├── entity
│       ├── __init__.py
│       ├── logging
│       │   └── __init__.py <------------ # This file 
│       ├── pipeline
│       └── utils
```

> Copy paste the below python script to the above mentioned file

```python
## src/logging/__init__.py

import os
import sys
import logging

log_dir = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_file_path = os.path.join(log_dir, "text_summarizer.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("summarizer_logger")
```

2.2 Verify if the logging has been implemented or not

> Copy paste the below python code to main.py

```python
## main.py

from src.textSummarizer.logging import logger

logger.info("Starting the text summarization process.")
logger.info("Text summarization process completed successfully.")
```
> Run the python main.py file, you should see output similar to the below one

```sh
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ python main.py 
[2026-01-23 12:34:00,557: INFO: main: Starting the text summarization process.]
[2026-01-23 12:34:00,557: INFO: main: Text summarization process completed successfully.]
```

> In our case, its working fine

2.3 Now, let's go ahead and implement some of the common functionalities we will be using for the project

> For defining our common utilities/functions we will be using the file shown below 

```text
├── src
│   └── textSummarizer
│       ├── components
│       ├── config
│       ├── constants
│       ├── entity
│       └── utils
│       │    └── __init__.py  
│       │    └── common.py   <------------ # This file 
│       ├── __init__.py
│       ├── logging
│       ├── pipeline
```

> Copy paste the below python script to the above mentioned file

```python
## src/utils/common.py

import os
from box.exceptions import BoxValueError
import yaml
from src.textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' read successfully.")
            return ConfigBox(content)
    except BoxValueError as box_error:
        logger.error(
            f"BoxValueError while converting YAML content to ConfigBox: {box_error}"
        )
        raise box_error
    except Exception as e:
        logger.error(f"Error reading YAML file '{path_to_yaml}': {e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: list[Path]) -> None:
    """Creates directories if they do not exist.

    Args:
        path_to_directories (list[Path]): A list of directory paths to create.
    """
    for path in path_to_directories:
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory '{path}' created successfully or already exists.")
        except Exception as e:
            logger.error(f"Error creating directory '{path}': {e}")
            raise e
```

## 📌 Important Notes

- Why use ConfigBox ?

> Without ConfigBox:

```python
dict_info = {"name": "Siddhartha", "age": 22, "city": "Bhaktapur"}
print(dict_info["name"])
print(dict_info.name)
```
```sh
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[6], line 3
      1 dict_info = {"name": "Siddhartha", "age": 22, "city": "Bhaktapur"}
      2 print(dict_info["name"])
----> 3 print(dict_info.name)

AttributeError: 'dict' object has no attribute 'name'
```

> With ConfigBox:

```python
from box import ConfigBox

dict_info = ConfigBox({"name": "Siddhartha", "age": 22, "city": "Bhaktapur"})
print(dict_info['name'])
print(dict_info.name)
```

```sh
Siddhartha
Siddhartha
```

- Why @ensure_annotations?

> Without @ensure_annotations:

```python
def multiply(a :int, b :int) -> int:
    return a * b

result = multiply(3, "4")
print(result)
```
```sh
444
```

> With @ensure_annotations:

```python
from ensure import ensure_annotations

@ensure_annotations
def multiply(a: int, b: int) -> int:
    return a * b

result = multiply(3, "4")
print(result)
```
```sh
EnsureError: Argument b of type <class 'str'> to <function multiply at 0x7ddb4806a950> does not match annotation type <class 'int'>
```

Throws an Error, making sure that only the correct values are passed and returned.

---

### 3. Finetuning Huggingface Models

In this section, we are going to finetune a huggingface model using a dataset with the help of jupyter notebook.

3.1 Create the new jupyter notebook file (text-summarizer.ipynb) inside our research folder 

```text
.
├── app.py
├── config
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
│   └── research-notebook.ipynb
│   └── text-summarizer.ipynb   <-------- # Your new jupyter notebook file
├── setup.py
├── src
├── template.py
└── venv
```

> For our project we are going to use the below model for summarization and dataset for finetuning

- Model -> [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)
- Dataset -> [knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum)

> *Note that I will be using the google colab notebook for this finetuning process as it has more GPU RAM (15 GB) compared to my local machines (3.6 GB).*

3.2 Copy, paste and run the below codes from cell to cell to your jupyter notebook file (text-summarizer.ipynb)

> Check for Nvidia Driver 
```python
## Cell 1

!nvidia-smi
```
```
Sat Jan 24 04:49:00 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   56C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```



> Install some dependencies if not already

```python
## Cell 2

!pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
```

> Update the accelerate 

```python
## Cell 3

!pip install --upgrade accelerate
!pip uninstall -y transformers accelerate
!pip install transformers accelerate
```

*`Purpose of accelerate :` A library by Hugging Face is designed to simplify the process of running PyTorch models on any type of hardware setup—whether it's a single CPU, a single GPU, or multiple GPUs (Multi-GPU/TPU) across several nodes.*


> Import the necessary dependencies

```python
## Cell 4

import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import pipeline, set_seed
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
```

> Lets also now download our [dataset](https://huggingface.co/datasets/knkarthick/samsum/tree/main), and upload it to our github repo as a zip file (summarizer-data.zip). The zip will have the below downloaded .csv files:

```text
train.csv
validation.csv
test.csv
```

> Create a new folder inside our project directory and move the summarizer-data.zip to it

```text
.
├── app.py
├── config
├── data
│   └── summarizer-data.zip  # <-------- Your dataset here
├── Dockerfile
├── .git
├── .github
├── .gitignore
├── LICENSE
├── logs
├── main.py
├── params.yaml
├── README.md
├── requirements.txt
├── research
├── setup.py
├── src
├── template.py
└── venv
```

> Add, commit and push the data to our githup repo

```sh
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git add data/
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git commit -m 'Added dataset used to fine
tune our hf model'
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF$ git push origin main
```

> Verify the push from the github repo page

<img src="../../images/summarizer-data.png"
    alt="Image Caption"
    style="border:1px solid white; padding:1px; background:#fff; width: 3000px;" />

✅ Now that the data has been successfully uploaded, lets jump back to coding

> Check whether we are going to use GPU or not

```python
## Cell 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : ", device)
```
```text
Device :  cuda
```

> Basic Functionality of HuggingFace Model (Optional)

```python
## Cell 6

from transformers import PegasusForConditionalGeneration

model_name = "google/pegasus-xsum"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ARTICLE_TO_SUMMARIZE = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
    "Natural language processing enables computers to understand and interpret human language.",
]

# Note: We added padding=True for batch processing
inputs = tokenizer(
    ARTICLE_TO_SUMMARIZE,
    max_length=1024,
    return_tensors="pt",
    truncation=True,
    padding=True,
).to(device)

# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
summaries = tokenizer.batch_decode(
    summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(summaries[0])
```
```text
This is the moment a fox meets a dog for the first time.
```

> Fine Tuning

- Loading the model and the tokenizer used for this particular model
```python
## Cell 7

model = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

print(f"Model loaded successfully on {device}")
```
```text
Model loaded successfully on cuda
```

- Download & Unzip our data

```python
## Cell 8

# Download the actual raw binary file
!wget https://github.com/SiddhuShkya/Text-Summarizer-With-HF/raw/main/data/summarizer-data.zip

# Now unzip will work
!unzip summarizer-data.zip
```

> Verify if the data has been downloaded and extracted successfully or not

```sh
siddhu@ubuntu:~/Desktop/Text-Summarizer-With-HF/research$ tree -a -L 3
.
├── research-notebook.ipynb
├── summarizer-data        <---------- # Extracted zip folder
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
├── summarizer-data.zip    <---------- # Downlaoded zip file
└── text-summarizer.ipynb
```

Once the data has been successfully loaded, you can follow the below coding steps :

> Load our dataset

```python
## Cell 9

train_df = pd.read_csv('./summarizer-data/train.csv')
test_df = pd.read_csv('./summarizer-data/test.csv')
val_df = pd.read_csv('./summarizer-data/validation.csv')

print("Features in the dataset: ", train_df.columns.tolist())
print("=" * 70)
print("Number of samples in each dataset:")
print("Train data samples: ", len(train_df))
print("Test data samples: ", len(test_df))
print("Validation data samples: ", len(val_df))
```
```text
Features in the dataset:  ['id', 'dialogue', 'summary']
======================================================================
Number of samples in each dataset:
Train data samples:  14731
Test data samples:  819
Validation data samples:  818
```

> Display one record

```python
## Cell 10

print(train_df["dialogue"][0])
print("\nSummary: ", train_df["summary"][0])
```
```text
Amanda: I baked  cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)

Summary:  Amanda baked cookies and will bring Jerry some tomorrow.
```

> Prepare Data For Training Seq To Seq Model

- Python function to prepare our raw data for model training

```python
## Cell 11

def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(
        example_batch["dialogue"],
        max_length=512,
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch["summary"],
            max_length=128,
            truncation=True,
        )
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }
```

- To use the above function with our dataframe, we need to first convert it to a data using the below codes:

```python
## Cell 12

from datasets import Dataset

# Convert dataframes → Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)
# Apply the function with .map()
train_dataset = train_dataset.map(convert_examples_to_features, batched=True)
test_dataset = test_dataset.map(convert_examples_to_features, batched=True)
val_dataset = val_dataset.map(convert_examples_to_features, batched=True)
```

- Check your new dataset

```python
## Cell 13

print("Train Dataset:\n", train_dataset)
print("Test Dataset:\n", test_dataset)
print("Val Dataset:\n", val_dataset)
```
```text
Train Dataset:
 Dataset({
    features: ['id', 'dialogue', 'summary', 'input_ids', 'attention_mask', 'labels'],
    num_rows: 14731
})
Test Dataset:
 Dataset({
    features: ['id', 'dialogue', 'summary', 'input_ids', 'attention_mask', 'labels'],
    num_rows: 819
})
Val Dataset:
 Dataset({
    features: ['id', 'dialogue', 'summary', 'input_ids', 'attention_mask', 'labels'],
    num_rows: 818
})
```

*Data Before Preparation:*

```python
{
    'id': '13862856',
    'dialogue': "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.",
    'summary': "Hannah needs Betty's number but Amanda doesn't have it."
}
```

*Data After Preparation:*

```python
{
    'id': '13862856',
    'dialogue': "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.",
    'summary': "Hannah needs Betty's number but Amanda doesn't have it."
    'input_ids': [123, 456, 789, .....] # Token IDs for the dialogue
    'attention_mask': [1, 1, 1, .....]  # Attention mask for the input
    'labels': [321, 654, 987, .....]    # Token Ids for the summary (target)
}
```

> Training Our Model

- Since, we are using a seq2seq model, we need to use `DataCollatorForSeq2Seq`, to make sure that our data is converted into batches before providing it for model training.

```python
## Cell 14

from transformers import DataCollatorForSeq2Seq

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
```

- Defining our training arguments with the help of TrainingArguments

```python
## Cell 15

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="pegasus-finetuned",
    num_train_epochs=1,
    warmup_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=100000,
    gradient_accumulation_steps=16,
)
```

- Defining our trainer with the help of Trainer

```python
## Cell 16

trainer = Trainer(
    model=model_pegasus,
    args=training_args,
    processing_class=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=test_dataset,
    eval_dataset=val_dataset,
)
```

- Train the model

```python
## Cell 17

trainer.train()
```
```sh
TrainOutput(global_step=52, training_loss=2.9682579774122972, metrics={'train_runtime': 288.3112, 'train_samples_per_second': 2.841, 'train_steps_per_second': 0.18, 'total_flos': 313176745058304.0, 'train_loss': 2.9682579774122972, 'epoch': 1.0})
```

> Evaluate our model

- We are going to use `ROUGE` evaluation for our model, because it is the standard automatic metric for summarization.

```python
## Cell 18

import evaluate

rouge_metric = evaluate.load("rouge")
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
```

- Defining a batch generator function that splits a specified dataset column to process data in batches.

```python
## Cell 19

def generate_batch_sized_chunks(list_of_elements, batch_size):
    """split the dataset into smaller batches"""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]
```

- Defining the function that computes the ROUGE scores for a dataset by generating summaries and comparing them to the reference summaries.

```python
## Cell 20

def calculate_metric_on_test_ds(
    dataset,
    metric,
    model,
    tokenizer,
    batch_size=1,
    device=device,
    column_text="article",
    column_summary="highlights",
):
    model.eval() 

    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    with torch.no_grad(): 
        for article_batch, target_batch in zip(article_batches, target_batches):

            inputs = tokenizer(
                article_batch,
                max_length=256,          
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                max_new_tokens=128,
                num_beams=1,             
                do_sample=False,
                use_cache=True,
            )

            decoded_summaries = tokenizer.batch_decode(
                summaries, skip_special_tokens=True
            )

            metric.add_batch(
                predictions=decoded_summaries,
                references=target_batch,
            )

    return metric.compute()
```

- Evaluate our model on the valisdation set using the ROUGE metric.
```python
## Cell 21

score = calculate_metric_on_test_ds(
    dataset=val_dataset[0:10],
    metric=rouge_metric,
    model=trainer.model,
    tokenizer=tokenizer,
    column_text="dialogue",
    column_summary="summary"
)

rouge_dict = {name: score[name] for name in rouge_names}
pd.DataFrame(rouge_dict, index=[f'pegasus-finetuned'])
```
| index             | rouge1      | rouge2      | rougeL       | rougeLsum    |
|------------------|-------------|-------------|--------------|--------------|
| pegasus-finetuned | 0.3056143469 | 0.1114997623 | 0.2479984793 | 0.2484580193 |

*We are getting values less than 0.5 because, we have trained our model for 1 epoch only*

## 📌 Important Note

1. `Scores close to 1:` Indicates a strong overlap between the generated summary and the reference summary, which is desirable in summarization tasks.

2. `Scores between 0.5  and 0.7:` Indicates moderate overlap. The summary might be capturing the key points but is likely missing some structure or important information.

3. `Scores below 0.5:` Suggests a poor match between the generated and reference summaries. The model might be generating irrelevant or incomplete summaries that dont capture the ky ideas well.

---

> Save the model & tokenizer

```python
## Cell 22

model_pegasus.save_pretrained("pegasus-model")
tokenizer.save_pretrained("pegasus-tokenizer")
```