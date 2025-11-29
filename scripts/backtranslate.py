#!/usr/bin/env python3
"""
backtranslate.py

This script performs back-translation (paraphrase attack) on the dataset.
It translates English text to German and back to English.
It updates the master table with `human_BT` and `machine_default_BT`.
"""

import argparse
import logging
import yaml
import os
from tqdm import tqdm
from src.generation.paraphrase import BackTranslator
from src.utils.common import load_jsonl, save_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/generation_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Perform back-translation")
    parser.add_argument("--config", type=str, default="configs/generation_config.yaml", help="Path to generation config")
    parser.add_argument("--master_table", type=str, default="data/master_table.jsonl", help="Path to master table")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load Master Table
    logger.info(f"Loading master table from {args.master_table}")
    try:
        master_table = load_jsonl(args.master_table)
    except FileNotFoundError:
        logger.error("Master table not found. Run generate_default_dataset.py first.")
        return

    # Initialize BackTranslator
    translator = BackTranslator(device=config["generator_model"]["device"])
    
    # Prepare data
    batch_size = 16 # Translation is usually faster/lighter than generation
    
    # We need to back-translate 'human_text' and 'machine_default'
    # We can do this in one pass or two. Let's do two passes for clarity.
    
    targets = [
        ("human_text", "human_BT"),
        ("machine_default", "machine_default_BT")
    ]
    
    for src_col, tgt_col in targets:
        logger.info(f"Back-translating {src_col} -> {tgt_col}...")
        
        # Check if target column already exists
        if len(master_table) > 0 and tgt_col in master_table[0]:
            logger.warning(f"Column {tgt_col} already exists. Overwriting...")
            
        texts = [row[src_col] for row in master_table if src_col in row]
        
        # If some rows are missing the source column (shouldn't happen), handle it
        if len(texts) != len(master_table):
            logger.error(f"Source column {src_col} missing in some rows!")
            return

        translated_texts = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_translated = translator.backtranslate(batch_texts)
            translated_texts.extend(batch_translated)
            
        # Update Master Table
        for row, text in zip(master_table, translated_texts):
            row[tgt_col] = text
            
        # Save intermediate results
        save_jsonl(master_table, args.master_table)
        
    logger.info("Back-translation complete. Master table updated.")

if __name__ == "__main__":
    main()
