
from src.utils import get_choices
from src.craftmd import craftmd_gpt
from src.graderai_eval import graderai_evaluation
from tqdm import tqdm
import pandas as pd
import openai
import argparse
from pathlib import Path

def execute_script(args):
    model_name = "llama3-8b"

    dataset = pd.read_csv("data_augmented/baseline.csv", index_col=0)
    path_dir = f"/data/healthy-ml/scratch/abinitha/craft-md/craft-md/results/{model_name}/{args.attribute}"
    experiment_names = ["vignette_frq", "multiturn_frq", "singleturn_frq", "summarized_frq"]

    # Iterate through each file in the directory
    path = Path(path_dir)
    files = sorted([f for f in path.iterdir() if f.is_file()],
    key=lambda x: int(x.stem.split('_')[1])  # Extract the number after "case_"
    )[:250]

    for file_path in tqdm(files):
        case = file_path.stem  # Use filename as the case
        graderai_evaluation(case, dataset, path_dir, experiment_names)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Modify gender markers in text using LLaMA 3.")
    parser.add_argument("--attribute", type=str, choices=['baseline','exclamation', 'gender', 'lowercase', 'no_gender', 'typo', 'uppercase', 'whitespace'])

    args = parser.parse_args()

    deployment_name = "gpt-4"
    openai.api_base = "https://codesidebias.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview"
    openai.api_key = "766aeef6717d4c09b997c943f75ce7d5"

    execute_script(args)

