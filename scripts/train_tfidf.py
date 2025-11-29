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
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report

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
    params = config["supervised"]["tfidf_lr"]
    
    logger.info("Training TF-IDF + LR model...")
    
    # TODO: Load training data
    
    # TODO: Initialize Vectorizer and Classifier
    # vectorizer = TfidfVectorizer(...)
    # clf = LogisticRegression(...)
    
    # TODO: Train pipeline
    # X_train = vectorizer.fit_transform(texts)
    # clf.fit(X_train, labels)
    
    # TODO: Save model
    output_dir = os.path.join(config["training"]["output_dir"], "tfidf_lr")
    os.makedirs(output_dir, exist_ok=True)
    # joblib.dump(pipeline, os.path.join(output_dir, "model.joblib"))
    
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
