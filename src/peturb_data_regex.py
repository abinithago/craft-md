import os
from openai import AzureOpenAI
import pandas as pd 
import numpy as np
import argparse
import random
import json
from tqdm import tqdm
import datetime
import string
import math
import utils
import sys 

def random_perturbations(text, type_pert, prob):

  random.seed(0)
  assert type(text) == str
  list_chars = list(text)

  if type_pert=="lowercase":
    for i in range(len(list_chars)):
      char_value = random.choices([list_chars[i].lower(), list_chars[i]], weights=[prob, 1-prob], k=1)[0]
      list_chars[i] = char_value

  if type_pert=="uppercase":
    for i in range(len(list_chars)):
      char_value = random.choices([list_chars[i].upper(), list_chars[i]], weights=[prob, 1-prob], k=1)[0]
      list_chars[i] = char_value

  if type_pert=="exclamation":
    indices = [i for i, letter in enumerate(list_chars) if letter == "."]
    for i in indices:
      char_value = random.choices(["!", "."], weights=[prob, 1-prob], k=1)[0]
      list_chars[i] = char_value
  
  if type_pert == "typo":
    # Get the indices of all non-space characters
    nonspace_indices = [i for i, char in enumerate(list_chars) if not char.isspace()]

    # Calculate the number of indices to flip based on the probability
    num_indices = math.floor(len(nonspace_indices) * prob)
    
    # Randomly select indices to flip
    flipping_indices = random.sample(nonspace_indices, k=num_indices)
    
    # Perform the flips
    for i in flipping_indices:
      # Randomly choose a replacement character from the alphabet
      list_chars[i] = random.choice(string.ascii_letters)

  if type_pert == "whitespace":
    new_text = []
    for char in list_chars:
        # Randomly add whitespace before the character
        add_space = random.choices([True, False], weights=[prob, 1-prob], k=1)[0]
        if add_space:
            whitespace = " " * random.randint(1, 3)  # Add 1 to 3 spaces
            new_text.append(whitespace)
        new_text.append(char)
    list_chars = new_text

  return "".join(list_chars)

def regex_perturb(df, attribute, probability):
  """
  gender swap using regex 
  """
  perturbed_messages = []
  for i, row in df.iterrows(): 
    perturbed_text = random_perturbations(row['case_vignette'], attribute, probability)
    perturbed_messages.append(perturbed_text)

  df['case_vignette'] = [r for r in perturbed_messages]
  return df
