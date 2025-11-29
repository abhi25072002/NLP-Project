#!/usr/bin/env python3
"""
generate_variants.py

This script generates Out-of-Distribution (OOD) variants of the machine-generated text.
It takes the prompts from the test set (and optionally train/val) and re-generates text
using different decoding strategies (temperature, top-p, etc.) as defined in `configs/generation_config.yaml`.

Modes:
- temperature: Generate variants with different temperature values.
- topp: Generate variants with different top-p values.
- decoding: Generate variants with greedy, beam search, etc.

Usage:
    python scripts/generate_variants.py --mode temperature
    python scripts/generate_variants.py --mode topp
"""

import argparse
import logging
import yaml
import os
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/generation_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate OOD variants")
    parser.add_argument("--config", type=str, default="configs/generation_config.yaml", help="Path to generation config")
    parser.add_argument("--mode", type=str, required=True, choices=["temperature", "topp", "decoding", "heldout"], help="Generation mode")
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger.info(f"Starting generation in mode: {args.mode}")
    
    # TODO: Load model and tokenizer
    # model_name = config["generator_model"]["name"]
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # TODO: Load prompts (from master table or splits)
    
    # TODO: Loop through prompts and generate variants based on mode
    if args.mode == "temperature":
        temps = config["variants"]["temperature"]["values"]
        logger.info(f"Generating for temperatures: {temps}")
        # for temp in temps: ...
        
    elif args.mode == "topp":
        top_ps = config["variants"]["top_p"]["values"]
        logger.info(f"Generating for top-p: {top_ps}")
        # for p in top_ps: ...
        
    # TODO: Save generated texts to data/generated/
    
    logger.info("Generation complete.")

if __name__ == "__main__":
    main()
