# src/utils/common.py
import json
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl(file_path):
    """
    Load data from a JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """
    Save data to a JSONL file.
    """
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
