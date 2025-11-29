#!/usr/bin/env python3
"""
train_cnn.py

This script trains a CNN-based text classifier (Kim CNN).
It uses word embeddings (randomly initialized or pre-trained) and convolutional filters
to capture local patterns in text.

Usage:
    python scripts/train_cnn.py
"""

import argparse
import logging
import yaml
import os
import torch
# from torch.utils.data import DataLoader
# from src.models.cnn import CNNClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    params = config["supervised"]["cnn"]
    
    logger.info("Training CNN model...")
    
    # TODO: Load training data and create vocabulary
    
    # TODO: Initialize Model
    # model = CNNClassifier(...)
    
    # TODO: Training Loop
    # for epoch in range(epochs): ...
    
    # TODO: Save model
    output_dir = os.path.join(config["training"]["output_dir"], "cnn")
    os.makedirs(output_dir, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
