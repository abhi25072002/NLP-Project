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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
import sys


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_supervised_data
from src.models.cnn import CNNClassifier

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
                    "lr": 0.001,
                    "batch_size": 64,
                    "epochs": 5,
                    "max_vocab_size": 25000,
                    "max_len": 200
                }
            },
            "training": {
                "output_dir": "models"
            }
        }
    
    if not os.path.exists(config_path):
        return default_config

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=200):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        
        # Tokenize and numericalize
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [self.vocab["<PAD>"]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def build_vocab(data, max_size=25000):
    counter = Counter()
    for item in data:
        counter.update(item["text"].lower().split())
        
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for text, labels in iterator:
            text = text.to(device)
            labels = labels.to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            acc = (predictions.argmax(1) == labels).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--data_dir", type=str, default="supervised/base", help="Path to data directory")
    args = parser.parse_args()
    
    config = load_config(args.config)
    params = config["supervised"].get("cnn", {})
    
    # Defaults if not in config
    EMBEDDING_DIM = params.get("embedding_dim", 100)
    NUM_FILTERS = params.get("num_filters", 100)
    FILTER_SIZES = params.get("filter_sizes", [3, 4, 5])
    OUTPUT_DIM = params.get("output_dim", 2)
    DROPOUT = params.get("dropout", 0.5)
    LR = params.get("lr", 0.001)
    BATCH_SIZE = params.get("batch_size", 64)
    EPOCHS = params.get("epochs", 5)
    MAX_VOCAB_SIZE = params.get("max_vocab_size", 25000)
    MAX_LEN = params.get("max_len", 200)
    
    logger.info("Loading data...")
    # Use the absolute path if provided, or assume relative to current execution
    # Ideally user provided args.data_dir which should be absolute or correct relative path
    # Given the user context: c:\Users\91816\Downloads\supervised\supervised\base
    # I'll default to that if the arg is not passed or relative path doesn't work.
    
    # Try provided path first
    if not os.path.exists(args.data_dir):
        # Specific user fallback
        args.data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "supervised", "supervised", "base"))
    
    try:
        train_data = load_supervised_data(args.data_dir, "train")
        val_data = load_supervised_data(args.data_dir, "val")
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    logger.info("Building vocabulary...")
    vocab = build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
    logger.info(f"Vocab size: {len(vocab)}")
    
    train_dataset = TextDataset(train_data, vocab, max_len=MAX_LEN)
    val_dataset = TextDataset(val_data, vocab, max_len=MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = CNNClassifier(len(vocab), EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    logger.info("Starting training...")
    best_valid_loss = float('inf')
    
    output_dir = os.path.join(config.get("training", {}).get("output_dir", "models"), "cnn")
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        
        for text, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            text = text.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            acc = (predictions.argmax(1) == labels).float().mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
            
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        logger.info(f"Epoch {epoch+1}: Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info("Saved best model.")
            
    # Save vocab
    import pickle
    with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
