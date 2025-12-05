#!/usr/bin/env python3
"""
train_roberta.py

This script fine-tunes a RoBERTa model for binary classification.
It uses the HuggingFace Trainer API for efficient training and evaluation.

Usage:
    python scripts/train_roberta.py
"""

import argparse
import logging
import yaml
import os
from src.data.loader import load_turingbench
from src.models.roberta import train_roberta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train RoBERTa model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    training_config = config["training"]
    
    logger.info("Fine-tuning RoBERTa model...")
    
    # Load dataset
    logger.info("Loading data...")
    dataset = load_turingbench(split="train")
    
    # Split into train/val
    logger.info("Splitting data...")
    train_test_split = dataset.train_test_split(test_size=0.2, seed=training_config["seed"])
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Train
    trainer = train_roberta(train_dataset, val_dataset, config)
    
    # Save model
    output_dir = os.path.join(training_config["output_dir"], "roberta")
    trainer.save_model(output_dir)
    
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
