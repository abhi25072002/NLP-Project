#!/usr/bin/env python3
"""
eval_roberta.py

This script evaluates the trained RoBERTa model on multiple test datasets.
"""

import argparse
import logging
import os
import torch
import sys
import numpy as np
import pickle
import glob
import json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import pandas as pd
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_jsonl_file
from src.models.roberta import get_roberta_model, get_tokenizer
from scripts.train_roberta import RobertaDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_file(model, tokenizer, file_path, max_len, device):
    try:
        data = load_jsonl_file(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

    if not data:
        return None

    dataset = RobertaDataset(data, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Calculate Metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        
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
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint directory (containing pytorch_model.bin or safe_tensors)")
    parser.add_argument("--data_dir", type=str, default="supervised/knobs", help="Directory containing test .jsonl files")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()
    
    # Initialize Model & Tokenizer
    # When loading from checkpoint, we usually use from_pretrained with the checkpoint dir
    logger.info(f"Loading model from {args.model_path} (Type: {type(args.model_path)})")
    if not os.path.exists(args.model_path):
        logger.error(f"Path {args.model_path} does not exist!")
        return

    import traceback
    try:
        model = get_roberta_model(args.model_path, num_labels=2)
        tokenizer = get_tokenizer(args.model_path) # Assumes tokenizer is saved in same dir
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        traceback.print_exc()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info("Model loaded.")
    
    # Find all jsonl files
    search_dir = args.data_dir
    if not os.path.exists(search_dir):
         # Specific user fallback
        print("Hi")
        search_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "supervised", "supervised", "knobs"))
        
    logger.info(f"Searching for test files in {search_dir}")
    test_files = glob.glob(os.path.join(search_dir, "*.jsonl"))
    
    if not test_files:
        logger.error("No .jsonl files found.")
        return
        
    results = []
    
    for file_path in test_files:
        logger.info(f"Evaluating {os.path.basename(file_path)}...")
        metrics = evaluate_file(model, tokenizer, file_path, args.max_len, device)
        if metrics:
            results.append(metrics)
            print(f"Results for {metrics['File']}:")
            print(f"  Acc: {metrics['Accuracy']:.4f}, P: {metrics['Precision']:.4f}, R: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}")

    if results:
        df = pd.DataFrame(results)
        result_path = os.path.join(args.model_path, "evaluation_results.csv")
        df.to_csv(result_path, index=False)
        logger.info(f"Results saved to {result_path}")
        print("\nAll Results:")
        print(df.to_string())
    else:
        logger.info("No results to save.")

if __name__ == "__main__":
    main()
