from openai import AzureOpenAI
import pandas as pd 
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime
import re
from huggingface_hub import login
import torch
import transformers
import sys

def modify_texts(df, text_column, mode):
    """
    Modify texts in a DataFrame based on the specified mode using LLaMA 3.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the column containing text.
        mode (str): Either "gender" for swapping genders or "no-gender" for removing genders.
        model_name (str): Name of the LLaMA 3 model on Hugging Face.
        hf_token (str): Hugging Face access token.

    Returns:
        pd.DataFrame: DataFrame with modified texts.
    """
    hf_token = "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" 
    login(hf_token, add_to_git_credential=True)
    
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print("cuda available", torch.cuda.is_available())

    print(sys.executable)

    # Load the language model
    text_generator = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    def process_text(text):
        if mode == "gender":
            prompt = f"Rewrite the following text by swapping all gender-specific references (male ↔ female):\n{text}"
        elif mode == "no-gender":
            prompt = f"Rewrite the following text to remove all gender-specific references:\n{text}"
        else:
            raise ValueError("Invalid mode. Choose 'gender' or 'no-gender'.")
        
        result = text_generator(prompt, max_length=512, do_sample=False)
        return result[0]['generated_text'].split(":", 1)[-1].strip()  # Extract generated text

    df[text_column] = df[text_column].apply(process_text)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify gender markers in text using LLaMA 3.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file.")
    parser.add_argument("--mode", type=str, choices=["gender", "no-gender"], help="Mode of modification: 'gender' to swap genders, 'no-gender' to remove genders.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column containing text.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3", help="Hugging Face model name.")

    args = parser.parse_args()

    # Load the input DataFrame
    df = pd.read_csv(args.input_file)

    # Modify texts based on the mode
    df = modify_texts(df, args.text_column, args.mode)

    # Save the modified DataFrame
    df.to_csv(args.output_file, index=False)
    print(f"Processed file saved to {args.output_file}")
