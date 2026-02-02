import os
import pandas as pd
import yaml

## Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

input_path = params["preprocess"]["input"]
output_path = params["preprocess"]["output"]


## Your main preocessing function
def preprocess_data(input_path, output_path):
    # Load the raw data
    data = pd.read_csv(input_path)
    # preprocessing steps
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    preprocess_data(input_path, output_path)
