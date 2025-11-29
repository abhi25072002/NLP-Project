#!/usr/bin/env python3
"""
generate_default_dataset.py

This script builds the initial "Master Table" for the project.
It performs the following steps:
1. Load TuringBench dataset (Human samples only).
2. Extract prompts (first k tokens).
3. Generate `machine_default` text for EVERY sample using the default generator config.
4. Save the master table to `data/master_table.jsonl`.

Note: This script does NOT perform splitting. Splitting is handled by `create_supervised_datasets.py`.
"""

import argparse
import logging
import yaml
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from src.data.splitter import create_master_table
from src.generation.generator import VariantGenerator
from src.utils.common import save_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Build Master Dataset (Human + Machine Default)")
    parser.add_argument("--data_config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--gen_config", type=str, default="configs/generation_config.yaml")
    args = parser.parse_args()
    
    data_config = load_config(args.data_config)
    gen_config = load_config(args.gen_config)
    
    # 1. Load TuringBench (Human Only)
    logger.info("Loading TuringBench dataset...")
    try:
        # Load local CSV prepared by prepare_local_turingbench.py
        local_csv = data_config.get("local", {}).get("consolidated_csv", "data/human_turingbench.csv")
        logger.info(f"Loading local data from {local_csv}")
        ds = load_dataset("csv", data_files=local_csv, split="train")
        
        # Rename 'Generation' to 'text' for compatibility if needed, or handle in create_master_table
        # create_master_table expects 'text' and 'src' (or we filter earlier)
        # The local CSV has 'Generation' and 'label'
        ds = ds.rename_column("Generation", "text")
        ds = ds.rename_column("label", "src")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 2. Create Base Master Table (ID, Prompt, Human Text)
    master_table = create_master_table(
        ds, 
        prompt_length=data_config["processing"]["prompt_length"],
        max_length=data_config["processing"]["max_length"]
    )
    
    logger.info(f"Base table created with {len(master_table)} human samples.")

    # 3. Generate machine_default
    logger.info("Initializing Generator for machine_default...")
    generator = VariantGenerator(
        model_name=gen_config["generator_model"]["name"], 
        device=gen_config["generator_model"]["device"]
    )
    
    default_cfg = gen_config["default"]
    logger.info(f"Generating machine_default with: {default_cfg}")
    
    prompts = [row['prompt'] for row in master_table]
    batch_size = 8 # Adjust based on GPU
    
    generated_texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Default Variants"):
        batch_prompts = prompts[i:i+batch_size]
        batch_out = generator.generate(
            batch_prompts,
            max_new_tokens=default_cfg["max_new_tokens"],
            temperature=default_cfg["temperature"],
            top_p=default_cfg["top_p"],
            do_sample=default_cfg["do_sample"]
        )
        generated_texts.extend(batch_out)
        
    # Attach to master table
    for row, text in zip(master_table, generated_texts):
        row['machine_default'] = text
        
    # 4. Save Master Table
    output_path = data_config["paths"]["master_table"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_jsonl(master_table, output_path)
    
    logger.info(f"Master table saved to {output_path}")

if __name__ == "__main__":
    main()
