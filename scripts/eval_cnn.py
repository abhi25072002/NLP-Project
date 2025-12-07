#!/usr/bin/env python3
"""
eval_cnn.py

This script evaluates the trained CNN model on multiple test datasets.
Calculates Precision, Recall, F1, and AUC.
"""

import argparse
import logging
import os
# import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
import sys
import numpy as np
import pickle
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
from src.models.cnn import CNNClassifier
from scripts.train_cnn import TextDataset  # Import Dataset class to reuse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    default_config = {
            "supervised": {
                "cnn": {
                    "embedding_dim": 100,
                    "num_filters": 100,
                    "filter_sizes": [3, 4, 5],
                    "output_dim": 2,
                    "dropout": 0.5,
                    "max_len": 200
                }
            }
        }
    
    if not os.path.exists(config_path):
        return default_config

    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return default_config

def evaluate_file(model, file_path, vocab, max_len, device):
    try:
        data = load_jsonl_file(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

    if not data:
        return None

    dataset = TextDataset(data, vocab, max_len=max_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for text, labels in loader:
            text = text.to(device)
            labels = labels.to(device)
            
            logits = model(text)
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of positive class (machine)
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Calculate Metrics
    # Assumes label 1 is positive (machine) -- verify mapping. typically 1=machine, 0=human
    # metrics average='binary' is default
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0 # Handle case with only one class
        
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
    parser = argparse.ArgumentParser(description="Evaluate CNN model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to vocab.pkl (default: same dir as model)")
    parser.add_argument("--data_dir", type=str, default="supervised/knobs", help="Directory containing test .jsonl files")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    params = config["supervised"].get("cnn", {})
    
    # Load Vocab
    if args.vocab_path:
        vocab_path = args.vocab_path
    else:
        vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.pkl")

    if not os.path.exists(vocab_path):
        logger.error(f"Vocab not found at {vocab_path}")
        return
        
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    logger.info(f"Loaded vocab size: {len(vocab)}")
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    EMBEDDING_DIM = params.get("embedding_dim", 100)
    NUM_FILTERS = params.get("num_filters", 100)
    FILTER_SIZES = params.get("filter_sizes", [3, 4, 5])
    OUTPUT_DIM = params.get("output_dim", 2)
    DROPOUT = params.get("dropout", 0.5)
    MAX_LEN = params.get("max_len", 200)

    # We must instantiate the model structure to load the weights
    model = CNNClassifier(len(vocab), EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded.")
    
    # Find all jsonl files
    # Try provided path first
    search_dir = args.data_dir
    if not os.path.exists(search_dir):
         # Specific user fallback
        search_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "supervised", "supervised", "knobs"))
        
    logger.info(f"Searching for test files in {search_dir}")
    test_files = glob.glob(os.path.join(search_dir, "*.jsonl"))
    
    if not test_files:
        logger.error("No .jsonl files found.")
        return
        
    results = []
    
    for file_path in test_files:
        logger.info(f"Evaluating {os.path.basename(file_path)}...")
        metrics = evaluate_file(model, file_path, vocab, MAX_LEN, device)
        if metrics:
            results.append(metrics)
            print(f"Results for {metrics['File']}:")
            print(f"  Acc: {metrics['Accuracy']:.4f}, P: {metrics['Precision']:.4f}, R: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}")

    if results:
        df = pd.DataFrame(results)
        result_dir = os.path.dirname(args.model_path)
        result_path = os.path.join(result_dir, "evaluation_results.csv")
        df.to_csv(result_path, index=False)
        logger.info(f"Results saved to {result_path}")
        print("\nAll Results:")
        print(df.to_string())
    else:
        logger.info("No results to save.")

if __name__ == "__main__":
    main()
