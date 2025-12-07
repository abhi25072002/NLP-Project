#!/usr/bin/env python3
"""
train_roberta.py

This script fine-tunes a RoBERTa model for binary classification.
It uses the HuggingFace Trainer API for efficient training and evaluation.

Usage:
    python scripts/train_roberta.py
"""

# scripts/train_roberta.py

import argparse
import logging
# import yaml
import os
import torch
import sys
import numpy as np
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_supervised_data
from src.models.roberta import get_roberta_model, get_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    # Fallback default if file doesn't exist to avoid import error issues if yaml missing
    if not os.path.exists(config_path):
        return {
            "supervised": {
                "roberta": {
                    "model_name": "roberta-base",
                    "learning_rate": 2.0e-5,
                    "batch_size": 16,
                    "epochs": 3,
                    "weight_decay": 0.01,
                    "max_seq_length": 512
                }
            },
            "training": {
                "output_dir": "checkpoints"
            }
        }
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback to defaults
        pass
        
    return {
            "supervised": {
                "roberta": {
                    "model_name": "roberta-base",
                    "learning_rate": 2.0e-5,
                    "batch_size": 16,
                    "epochs": 3,
                    "weight_decay": 0.01,
                    "max_seq_length": 512
                }
            },
            "training": {
                "output_dir": "checkpoints"
            }
        }

class RobertaDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description="Train RoBERTa model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--data_dir", type=str, default="supervised/base", help="Path to data directory")
    args = parser.parse_args()
    
    config = load_config(args.config)
    # Safely get params
    params = config.get("supervised", {}).get("roberta", {
        "model_name": "roberta-base",
        "learning_rate": 2.0e-5,
        "batch_size": 16,
        "epochs": 3,
        "weight_decay": 0.01,
        "max_seq_length": 512
    })
    
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(params.get("model_name", "roberta-base"))
    
    logger.info("Loading data...")
    if not os.path.exists(args.data_dir):
        # Fallback
        args.data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "supervised", "supervised", "base"))
    
    try:
        train_data = load_supervised_data(args.data_dir, "train")
        val_data = load_supervised_data(args.data_dir, "val")
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    train_dataset = RobertaDataset(train_data, tokenizer, params.get("max_seq_length", 512))
    val_dataset = RobertaDataset(val_data, tokenizer, params.get("max_seq_length", 512))
    
    logger.info("Initializing Model...")
    model = get_roberta_model(params.get("model_name", "roberta-base"), num_labels=2)
    
    output_dir = os.path.join(config.get("training", {}).get("output_dir", "checkpoints"), "roberta")
    
    # dynamic args creation to handle version differences
    common_args = {
        "output_dir": output_dir,
        "num_train_epochs": params.get("epochs", 3),
        "per_device_train_batch_size": params.get("batch_size", 16),
        "per_device_eval_batch_size": params.get("batch_size", 16),
        "warmup_steps": params.get("warmup_steps", 500),
        "weight_decay": params.get("weight_decay", 0.01),
        "logging_dir": os.path.join(output_dir, "logs"),
        "logging_steps": 100,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "learning_rate": float(params.get("learning_rate", 2e-5)),
        "save_total_limit": 2
    }
    
    try:
        # Try new/standard argument first
        training_args = TrainingArguments(
            **common_args,
            evaluation_strategy="epoch"
        )
    except TypeError:
        try:
            # Try new alias (4.41+)
            training_args = TrainingArguments(
                **common_args,
                eval_strategy="epoch"
            )
        except TypeError:
             # Fallback for very old versions (pre 3.0) - unlikely given roberta-base usage but possible
             # Removing save_strategy if that's also an issue?
             # Assuming 'evaluate_during_training' is the culprit if really old
            common_args.pop("save_strategy", None) # Might not be supported
            training_args = TrainingArguments(
                **common_args,
                evaluate_during_training=True
            )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
