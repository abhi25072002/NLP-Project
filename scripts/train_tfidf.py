#!/usr/bin/env python3
"""
train_tfidf.py

This script trains a TF-IDF + Logistic Regression baseline model.
It loads the training data, extracts TF-IDF features, trains the classifier,
and saves the model and vectorizer.

Usage:
    python scripts/train_tfidf.py
"""



import argparse
import logging
import yaml
import os
import joblib
from src.data.loader import load_turingbench
from src.models.tfidf import train_tfidf_lr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LR model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger.info("Training TF-IDF + LR model...")
    
    # Load training data
    logger.info("Loading data...")
    dataset = load_turingbench(split="train")
    
    texts = dataset["text"]
    labels = dataset["label"]
    
    logger.info(f"Loaded {len(texts)} samples.")
    
    # Train pipeline
    logger.info("Training pipeline...")
    pipeline = train_tfidf_lr(texts, labels, config)
    
    # Save model
    output_dir = os.path.join(config["training"]["output_dir"], "tfidf_lr")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    
    # Also save vectorizer separately if needed, but pipeline has it.
    # The prompt asked for "model.joblib, vectorizer *.joblib".
    # I'll save the vectorizer separately too.
    vectorizer_path = os.path.join(output_dir, "vectorizer.joblib")
    joblib.dump(pipeline.named_steps['tfidf'], vectorizer_path)
    
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
