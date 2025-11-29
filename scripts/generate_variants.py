#!/usr/bin/env python3
"""
generate_variants.py

This script generates Out-of-Distribution (OOD) variants of the machine-generated text.
It loads the master table, iterates through prompts, and generates text using
different decoding strategies (temperature, top-p, etc.).

The generated texts are added as new columns to the master table (e.g., 'machine_T0.7_p0.9').
"""

import argparse
import logging
import yaml
import os
import json
from tqdm import tqdm
from src.generation.generator import VariantGenerator
from src.utils.common import load_jsonl, save_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/generation_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate OOD variants")
    parser.add_argument("--config", type=str, default="configs/generation_config.yaml", help="Path to generation config")
    parser.add_argument("--master_table", type=str, default="data/master_table.jsonl", help="Path to master table")
    parser.add_argument("--mode", type=str, required=True, choices=["temperature", "topp", "decoding", "heldout", "all"], help="Generation mode")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load Master Table
    logger.info(f"Loading master table from {args.master_table}")
    try:
        master_table = load_jsonl(args.master_table)
    except FileNotFoundError:
        logger.error("Master table not found. Run create_splits.py first.")
        return

    # Initialize Generator
    model_name = config["generator_model"]["name"]
    generator = VariantGenerator(model_name=model_name, device=config["generator_model"]["device"])
    
    # Prepare prompts
    # We process in batches for efficiency
    batch_size = 8 # Adjust based on GPU memory
    prompts = [row['prompt'] for row in master_table]
    
    # ... (previous code)

    # Load default generation settings
    default_kwargs = config.get("default", {})
    logger.info(f"Global default settings: {default_kwargs}")

    # Define generation tasks based on mode
    tasks = []
    
    if args.mode in ["temperature", "all"]:
        temps = config["variants"]["temperature"]["values"]
        base_top_p = config["variants"]["temperature"]["base_top_p"]
        for t in temps:
            # Start with defaults, then override
            kwargs = default_kwargs.copy()
            kwargs.update({"temperature": t, "top_p": base_top_p, "do_sample": True})
            tasks.append({
                "name": f"machine_T{t}_p{base_top_p}",
                "kwargs": kwargs
            })
            
    if args.mode in ["topp", "all"]:
        top_ps = config["variants"]["top_p"]["values"]
        base_temp = config["variants"]["top_p"]["base_temperature"]
        for p in top_ps:
            kwargs = default_kwargs.copy()
            kwargs.update({"temperature": base_temp, "top_p": p, "do_sample": True})
            tasks.append({
                "name": f"machine_T{base_temp}_p{p}",
                "kwargs": kwargs
            })
            
    if args.mode in ["decoding", "all"]:
        strategies = config["variants"]["decoding_strategies"]
        for s in strategies:
            # Start with defaults
            kwargs = default_kwargs.copy()
            # Override with strategy-specific settings
            strategy_kwargs = {k: v for k, v in s.items() if k != 'name'}
            kwargs.update(strategy_kwargs)
            
            tasks.append({
                "name": f"machine_{s['name']}",
                "kwargs": kwargs
            })

    # Execute Generation Tasks
    import time
    total_start_time = time.time()
    
    for task in tasks:
        task_start_time = time.time()
        col_name = task["name"]
        gen_kwargs = task["kwargs"]
        
        logger.info(f"--- Starting Task: {col_name} ---")
        logger.info(f"Configuration: {json.dumps(gen_kwargs, indent=2)}")
        
        # Check if column already exists
        if len(master_table) > 0 and col_name in master_table[0]:
             logger.warning(f"Column '{col_name}' already exists in master table. Overwriting...")

        generated_texts = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating {col_name}"):
            batch_prompts = prompts[i:i+batch_size]
            batch_texts = generator.generate(
                batch_prompts, 
                max_new_tokens=config["prompts"]["max_new_tokens"],
                **gen_kwargs
            )
            generated_texts.extend(batch_texts)
            
        # Update Master Table
        for row, text in zip(master_table, generated_texts):
            row[col_name] = text
            
        # Save intermediate results
        save_jsonl(master_table, args.master_table)
        
        task_duration = time.time() - task_start_time
        logger.info(f"Task {col_name} completed in {task_duration:.2f} seconds ({task_duration/60:.2f} minutes).")
        
    total_duration = time.time() - total_start_time
    logger.info(f"All generation tasks completed in {total_duration:.2f} seconds ({total_duration/3600:.2f} hours).")
    logger.info("Master table updated.")

if __name__ == "__main__":
    main()
