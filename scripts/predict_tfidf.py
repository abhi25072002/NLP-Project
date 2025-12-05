import argparse
import logging
import yaml
import os
import joblib
import pandas as pd
from src.data.loader import load_turingbench

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Predict with TF-IDF + LR model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (joblib)")
    parser.add_argument("--output", type=str, default="predictions_tfidf.jsonl", help="Output file for predictions")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to predict on")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger.info(f"Loading model from {args.checkpoint}...")
    pipeline = joblib.load(args.checkpoint)
    
    logger.info(f"Loading data ({args.split})...")
    dataset = load_turingbench(split=args.split)
    texts = dataset["text"]
    
    logger.info("Generating predictions...")
    # Get probabilities
    probs = pipeline.predict_proba(texts)
    # Get class predictions
    preds = pipeline.predict(texts)
    
    # Save Results
    # "Outputs probabilities and predictions for datasets; write result JSONL."
    # We'll write a JSONL file.
    
    results = []
    for i in range(len(texts)):
        results.append({
            "text": texts[i],
            "label": dataset["label"][i],
            "prediction": int(preds[i]),
            "probability": float(probs[i][1]) # Prob of class 1
        })
        
    df = pd.DataFrame(results)
    
    # Check extension to decide format, but prompt said JSONL
    if args.output.endswith(".jsonl"):
        df.to_json(args.output, orient="records", lines=True)
    else:
        df.to_csv(args.output, index=False)
        
    logger.info(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
