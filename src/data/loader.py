# src/data/loader.py
import json
import os

def load_supervised_data(data_dir, split="train"):
    """
    Load supervised data from jsonl files.

    Args:
        data_dir (str): Path to the directory containing .jsonl files.
        split (str): Split to load (train, test, val).

    Returns:
        list of dict: List of examples with 'text' and 'label'.
    """
    file_path = os.path.join(data_dir, f"{split}.jsonl")
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append({"text": item["text"], "label": item["label"]})
            except json.JSONDecodeError:
                continue
    return data

def load_jsonl_file(file_path):
    """
    Load data from a specific jsonl file.

    Args:
        file_path (str): Absolute path to the .jsonl file.

    Returns:
        list of dict: List of examples with 'text' and 'label'.
    """
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append({"text": item["text"], "label": item["label"]})
            except json.JSONDecodeError:
                continue
    return data
