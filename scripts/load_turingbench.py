#!/usr/bin/env python3
"""
load_turingbench.py

This script demonstrates how to load the TuringBench dataset using the HuggingFace `datasets` library.
It serves as a utility to verify data access and inspect the raw data structure.

Usage:
    python scripts/load_turingbench.py
"""

import argparse
import logging
from datasets import load_dataset
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/data_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Load and inspect TuringBench dataset")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml", help="Path to data config")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config["dataset"]["name"]
    hf_path = config["dataset"]["hf_path"]
    
    logger.info(f"Loading dataset: {dataset_name} from {hf_path}")
    
    try:
        # Load dataset
        # Note: Adjust split if needed (e.g., 'train', 'validation', 'test')
        ds = load_dataset(hf_path, split="train", cache_dir=config["dataset"]["cache_dir"])
        
        logger.info(f"Successfully loaded dataset. Size: {len(ds)}")
        logger.info(f"Sample entry:\n{ds[0]}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

if __name__ == "__main__":
    main()
