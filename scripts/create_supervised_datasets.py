#!/usr/bin/env python3
"""
create_supervised_datasets.py

This script takes the fully populated `master_table.jsonl` (containing human, machine_default, and all OOD variants)
and creates the specific train/val/test datasets required for supervised training and evaluation.

It implements:
1.  **Prompt-level Splitting**: Randomly assigns IDs to train/val/test.
2.  **Base Dataset Creation**: Creates balanced train/val/test sets (Human vs Machine Default).
3.  **Knobbed Test Sets**: Creates separate test sets for each OOD variant (e.g., Test_T0.1_p0.9).
4.  **Back-Translation Test Sets**: Creates Test_BT_both and Test_BT_attack.

Usage:
    python scripts/create_supervised_datasets.py
"""

import argparse
import logging
import yaml
import os
import random
import sys

# Add project root to path so `src` imports work when run as `python scripts/...`
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.common import load_jsonl, save_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_dataset_file(data, output_path, positive_key, negative_key):
    """
    Helper to create a standard dataset file (text, label).
    Label 0 = Human (Positive), Label 1 = Machine (Negative)
    """
    dataset = []
    for row in data:
        # Add Positive (Human)
        if positive_key in row and row[positive_key]:
            dataset.append({"text": row[positive_key], "label": 0, "id": row["id"], "source": positive_key})
        
        # Add Negative (Machine)
        if negative_key in row and row[negative_key]:
            dataset.append({"text": row[negative_key], "label": 1, "id": row["id"], "source": negative_key})
            
    save_jsonl(dataset, output_path)
    return len(dataset)

def main():
    parser = argparse.ArgumentParser(description="Create Supervised Datasets")
    parser.add_argument("--data_config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--master_table", type=str, default="data/master_table.jsonl")
    args = parser.parse_args()
    
    config = load_config(args.data_config)
    
    # 1. Load Master Table
    logger.info(f"Loading master table from {args.master_table}")
    try:
        master_table = load_jsonl(args.master_table)
    except FileNotFoundError:
        logger.error("Master table not found. Run generate_default_dataset.py and generate_variants.py first.")
        return

    # 2. Prompt-level Splitting
    logger.info("Assigning splits...")
    random.seed(config["splitting"]["seed"])
    
    # Get unique IDs
    ids = list(set(row['id'] for row in master_table))
    random.shuffle(ids)
    
    n = len(ids)
    n_train = int(n * config["splitting"]["train_ratio"])
    n_val = int(n * config["splitting"]["val_ratio"])
    
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train+n_val])
    test_ids = set(ids[n_train+n_val:])
    
    # Assign split to each row in master table (in memory)
    train_rows = []
    val_rows = []
    test_rows = []
    
    for row in master_table:
        if row['id'] in train_ids:
            row['split'] = 'train'
            train_rows.append(row)
        elif row['id'] in val_ids:
            row['split'] = 'val'
            val_rows.append(row)
        elif row['id'] in test_ids:
            row['split'] = 'test'
            test_rows.append(row)
            
    logger.info(f"Splits: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}")
    
    # Save updated master table with splits
    save_jsonl(master_table, args.master_table)

    # 3. Create Base Supervised Datasets (Human vs Machine Default)
    output_dir = "data/supervised/base"
    os.makedirs(output_dir, exist_ok=True)
    
    create_dataset_file(train_rows, os.path.join(output_dir, "train.jsonl"), "human_text", "machine_default")
    create_dataset_file(val_rows, os.path.join(output_dir, "val.jsonl"), "human_text", "machine_default")
    create_dataset_file(test_rows, os.path.join(output_dir, "test.jsonl"), "human_text", "machine_default")
    
    logger.info(f"Base datasets created in {output_dir}")

    # 4. Create Knobbed Test Sets (Robustness Analysis)
    # We only create these for the TEST split
    knob_dir = "data/supervised/knobs"
    os.makedirs(knob_dir, exist_ok=True)
    
    # Identify all machine variant columns
    # Assuming columns starting with 'machine_' are variants
    sample_row = master_table[0]
    variant_cols = [k for k in sample_row.keys() if k.startswith("machine_") and k != "machine_default"]
    
    for col in variant_cols:
        test_filename = f"test_{col.replace('machine_', '')}.jsonl"
        create_dataset_file(test_rows, os.path.join(knob_dir, test_filename), "human_text", col)
        logger.info(f"Created knobbed test set: {test_filename}")

    # 5. Create Back-Translation Test Sets
    bt_dir = "data/supervised/backtrans"
    os.makedirs(bt_dir, exist_ok=True)
    
    # Scenario A: Test_BT_both (Human BT vs Machine Default BT)
    if "human_BT" in sample_row and "machine_default_BT" in sample_row:
        create_dataset_file(test_rows, os.path.join(bt_dir, "test_bt_both.jsonl"), "human_BT", "machine_default_BT")
        logger.info("Created Test_BT_both")
        
    # Scenario B: Test_BT_attack (Human vs Machine Default BT)
    if "machine_default_BT" in sample_row:
        create_dataset_file(test_rows, os.path.join(bt_dir, "test_bt_attack.jsonl"), "human_text", "machine_default_BT")
        logger.info("Created Test_BT_attack")

    logger.info("All supervised datasets created successfully.")

if __name__ == "__main__":
    main()
