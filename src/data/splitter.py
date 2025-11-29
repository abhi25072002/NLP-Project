# src/data/splitter.py
import json
import random
import os
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def create_master_table(dataset, prompt_length=50, max_length=512):
    """
    Create a master table from the TuringBench dataset.
    Extracts prompts and creates a unique ID for each entry.
    
    Args:
        dataset: HuggingFace dataset object.
        prompt_length (int): Number of tokens to use as prompt.
        max_length (int): Maximum length of human text to keep.
        
    Returns:
        list: List of dictionaries (the master table).
    """
    logger.info("Creating master table...")
    master_table = []
    
    # Initialize tokenizer for prompt extraction (using GPT-2 as generic tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    for i, entry in enumerate(dataset):
        # TuringBench structure: 'text', 'label', 'src' (generator name)
        # We only want human texts to start with (label usually 0 or 'human')
        # Adjust based on actual TuringBench structure. 
        # Assuming we filter for human texts first if we want to generate from them.
        # However, TuringBench has both. The user plan says:
        # "Start from N human texts (TuringBench)."
        
        # Check if it's human
        if entry['src'] != 'human':
            continue
            
        text = entry['text']
        
        # Truncate text if needed
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
        human_text = tokenizer.decode(tokens)
        
        # Extract prompt
        prompt_tokens = tokens[:prompt_length]
        prompt = tokenizer.decode(prompt_tokens)
        
        row = {
            "id": f"tb_{i:06d}",
            "prompt": prompt,
            "human_text": human_text,
            "split": None # To be assigned
        }
        master_table.append(row)
        
    logger.info(f"Created master table with {len(master_table)} entries.")
    return master_table

def assign_splits(master_table, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Assign train/val/test splits to the master table.
    
    Args:
        master_table (list): The master table.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
        seed (int): Random seed.
        
    Returns:
        list: Updated master table with 'split' field assigned.
    """
    logger.info("Assigning splits...")
    random.seed(seed)
    
    # Shuffle IDs/Indices
    indices = list(range(len(master_table)))
    random.shuffle(indices)
    
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train:n_train+n_val])
    test_indices = set(indices[n_train+n_val:])
    
    for i, row in enumerate(master_table):
        if i in train_indices:
            row['split'] = 'train'
        elif i in val_indices:
            row['split'] = 'val'
        elif i in test_indices:
            row['split'] = 'test'
        else:
            # Should not happen if ratios sum to 1.0
            row['split'] = 'test' 
            
    logger.info("Splits assigned.")
    return master_table
