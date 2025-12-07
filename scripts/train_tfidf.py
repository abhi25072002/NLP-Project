import argparse
import logging
import yaml
import os
import joblib
import sys

# Add src to path if needed (though usually processed by setup or user pythonpath, explicit add helps in scripts)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_supervised_data
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
    parser.add_argument("--data_dir", type=str, default="data/supervised/base", help="Directory containing supervised train.jsonl")
    args = parser.parse_args()
    
    config = load_config(args.config)
    tfidf_params = config["supervised"]["tfidf_lr"]
    
    logger.info("Loading training data...")
    try:
        train_data = load_supervised_data(args.data_dir, split="train")
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Loaded {len(train_data)} training examples.")
    
    texts = [d['text'] for d in train_data]
    labels = [d['label'] for d in train_data]
    
    logger.info("Training TF-IDF + LR model...")
    # Train
    pipeline = train_tfidf_lr(texts, labels, tfidf_params)
    
    # Save model
    output_dir = os.path.join(config["training"]["output_dir"], "tfidf_lr")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save vectorizer vocabulary size for inspection
    try:
        vocab_size = len(pipeline.named_steps['features'].transformer_list[0][1].vocabulary_) + \
                     len(pipeline.named_steps['features'].transformer_list[1][1].vocabulary_)
        logger.info(f"Total features: ~{vocab_size} (Word + Char)")
    except Exception as e:
        logger.warning(f"Could not calculate vocab size: {e}")

if __name__ == "__main__":
    main()
