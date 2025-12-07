#!/usr/bin/env python3
"""
eval_tfidf.py

This script evaluates the trained TF-IDF + LR model on multiple test datasets.
Calculates Precision, Recall, F1, and AUC.
"""

import argparse
import logging
import os
import joblib
import sys
import numpy as np
import glob
import json

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
except ImportError:
    precision_score = recall_score = f1_score = roc_auc_score = accuracy_score = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_jsonl_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_file(pipeline, file_path):
    try:
        data = load_jsonl_file(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

    if not data:
        return None

    texts = [d['text'] for d in data]
    labels = [d['label'] for d in data]
    
    # Predict
    # Probability of positive class (machine)
    # Check classes_ to identify index of 1
    pos_index = 1
    if hasattr(pipeline, "classes_"):
        classes = pipeline.classes_
        if 1 in classes:
            pos_index = np.where(classes == 1)[0][0]
        else:
             # Fallback if only one class (unlikely for trained model but possible if only 0s in train)
             # If only 0s, index 0 is class 0. If only 1s, index 0 is class 1.
             if len(classes) == 1 and classes[0] == 1:
                 pos_index = 0
             else:
                 pos_index = -1 # Error state or 0 only
    
    try:
        if pos_index != -1 and len(pipeline.classes_) > 1:
            probs = pipeline.predict_proba(texts)[:, pos_index]
        elif len(pipeline.classes_) == 1:
            # If model only knows one class, prob is 1.0 if class matches, else 0.0
            # This is edge case.
            if pipeline.classes_[0] == 1:
                 probs = np.ones(len(texts))
            else:
                 probs = np.zeros(len(texts))
        else:
            probs = np.zeros(len(texts))
            
        preds = pipeline.predict(texts)
    except Exception as e:
        logger.error(f"Prediction failed for {file_path}: {e}")
        return None

    # Calculate Metrics
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0 # Handle case with only one class in y_true
        
    return {
        "File": os.path.basename(file_path),
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "Num_Samples": len(data)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model.joblib")
    parser.add_argument("--data_dir", type=str, default="data/supervised/knobs", help="Directory containing test .jsonl files")
    args = parser.parse_args()
    
    # Load Model
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}")
        return
        
    logger.info(f"Loading model from {args.model_path}...")
    try:
        pipeline = joblib.load(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    logger.info("Model loaded.")
    
    search_dir = args.data_dir
    if not os.path.exists(search_dir):
         # Try project root relative
         candidate = os.path.join("data", "supervised", "knobs")
         if os.path.exists(candidate):
             search_dir = candidate

    logger.info(f"Searching for test files in {search_dir}")
    test_files = glob.glob(os.path.join(search_dir, "*.jsonl"))
    
    if not test_files:
        logger.error(f"No .jsonl files found in {search_dir}")
        return
        
    results = []
    
    for file_path in sorted(test_files):
        logger.info(f"Evaluating {os.path.basename(file_path)}...")
        metrics = evaluate_file(pipeline, file_path)
        if metrics:
            results.append(metrics)
            print(f"Results for {metrics['File']}:")
            print(f"  Acc: {metrics['Accuracy']:.4f}, P: {metrics['Precision']:.4f}, R: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}")

    if results:
        # Save to CSV
        result_dir = os.path.dirname(args.model_path)
        result_path = os.path.join(result_dir, "evaluation_results.csv")
        
        if pd:
            df = pd.DataFrame(results)
            df.to_csv(result_path, index=False)
            logger.info(f"Results saved to {result_path}")
            print("\nAll Results:")
            print(df.to_string())
        else:
            # Fallback if pandas not available
            with open(result_path, 'w') as f:
                header = ",".join(results[0].keys())
                f.write(header + "\n")
                for res in results:
                    f.write(",".join(str(x) for x in res.values()) + "\n")
            logger.info(f"Results saved to {result_path} (csv)")
            print(json.dumps(results, indent=2))
            
    else:
        logger.info("No results to save.")

if __name__ == "__main__":
    main()
