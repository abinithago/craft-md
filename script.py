
from src.utils import get_choices
from src.craftmd import craftmd_opensource
from tqdm import tqdm
import pandas as pd
import openai
import argparse

def execute_script(dataset):
    model_name = "llama3-8b"

    cases = [(dataset.loc[idx,"case_id"], 
            dataset.loc[idx,"case_vignette"], 
            dataset.loc[idx,"category"],
            get_choices(dataset,idx)) for idx in dataset.index]

    path_dir = f"/data/healthy-ml/scratch/abinitha/craft-md/craft-md/results/{model_name}/baseline"

    for i in tqdm(range(len(dataset))):
        if i < 250: 
            case = cases[i]
            craftmd_opensource(case, path_dir, model_name)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Modify gender markers in text using LLaMA 3.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file.")

    args = parser.parse_args()


    deployment_name = "gpt-4"
    openai.api_base = "https://codesidebias.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview"
    openai.api_key = "766aeef6717d4c09b997c943f75ce7d5"

    dataset = pd.read_csv(args.input_file)
    execute_script(dataset)

