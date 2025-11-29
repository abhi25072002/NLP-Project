#!/usr/bin/env python3
"""
create_splits.py

This script creates the train/val/test splits for the project.
Crucially, it implements 'prompt-level splitting' to ensure that all variants 
(human, machine, different decodings) of a single prompt remain in the same split.
This prevents data leakage where the model sees a prompt in training and its variant in testing.

Steps:
1. Load raw TuringBench data.
2. Group by prompt/ID.
3. Split IDs into Train (60%), Val (20%), Test (20%).
4. Save the split indices or IDs to `data/splits/`.

Usage:
    python scripts/create_splits.py
"""

import argparse
import logging
import yaml
import os
import json
import random
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/data_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml", help="Path to data config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set seed for reproducibility
    random.seed(config["splitting"]["seed"])
    
    logger.info("Loading dataset...")
    # TODO: Implement dataset loading logic
    # ds = load_dataset(...)
    
    logger.info("Grouping by prompt/ID...")
    # TODO: Implement grouping logic
    # unique_ids = ...
    
    logger.info("Creating splits...")
    # TODO: Implement splitting logic
    # train_ids, val_ids, test_ids = ...
    
    output_dir = config["splitting"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving splits to {output_dir}...")
    # TODO: Save splits to JSON files
    # with open(os.path.join(output_dir, "train_ids.json"), "w") as f: ...
    
    logger.info("Splits created successfully.")

if __name__ == "__main__":
    main()
