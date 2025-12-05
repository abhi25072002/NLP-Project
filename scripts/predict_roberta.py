import argparse
import logging
import yaml
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from src.data.loader import load_turingbench

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Predict with RoBERTa model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--output", type=str, default="predictions_roberta.csv", help="Output file for predictions")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to predict on")
    args = parser.parse_args()
    
    config = load_config(args.config)
    params = config["supervised"]["roberta"]
    
    logger.info(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    
    logger.info(f"Loading data ({args.split})...")
    dataset = load_turingbench(split=args.split)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=params.get("max_seq_length", 512))
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Use Trainer for prediction
    trainer = Trainer(model=model)
    
    logger.info("Generating predictions...")
    predictions_output = trainer.predict(tokenized_dataset)
    logits = predictions_output.predictions
    
    # Softmax to get probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    
    # Save Results
    df = pd.DataFrame({
        "text": dataset["text"],
        "label": dataset["label"],
        "prediction": preds,
        "probability": probs[:, 1] # Prob of class 1
    })
    
    df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
