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
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data.loader import load_turingbench
from src.models.cnn import CNNClassifier
from src.utils.embeddings import Vocabulary, load_glove_vectors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def tokenize(text):
    return text.split()

def collate_batch(batch, vocab, max_len, device):
    labels = []
    text_list = []
    for item in batch:
        labels.append(item['label'])
        tokens = tokenize(item['text'])
        indices = vocab.lookup_indices(tokens)
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices += [vocab["<pad>"]] * (max_len - len(indices))
        text_list.append(indices)
    
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    text_list = torch.tensor(text_list, dtype=torch.long).to(device)
    return text_list, labels

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for text, labels in iterator:
            predictions = model(text)
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    cnn_config = config["supervised"]["cnn"]
    training_config = config["training"]
    
    device = torch.device(training_config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Data
    logger.info("Loading data...")
    dataset = load_turingbench(split="train")
    # Splitting train into train/val if not already split, but load_turingbench returns one split.
    # We should probably load train and validation separately or split here.
    # The config says val_ratio: 0.2. 
    # For simplicity, let's assume we split the train set here or load a validation split if available.
    # The loader returns whatever split we ask. Let's try to load 'validation' or split manually.
    # Since I don't know if 'validation' split exists in the HF dataset, I'll split manually for safety.
    
    train_test_split = dataset.train_test_split(test_size=0.2, seed=training_config["seed"])
    train_data = train_test_split['train']
    valid_data = train_test_split['test']
    
    logger.info(f"Train size: {len(train_data)}, Val size: {len(valid_data)}")
    
    # Build Vocabulary
    logger.info("Building vocabulary...")
    vocab = Vocabulary(max_size=25000) # Reasonable default
    vocab.build_vocabulary([item['text'] for item in train_data])
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Load Embeddings
    embedding_matrix = None
    # Assuming embeddings are at data/embeddings/glove.6B.100d.txt or similar
    # We need to define where to look. Config doesn't specify embedding path explicitly for CNN.
    # I'll check a common path or skip if not found.
    embedding_path = f"data/embeddings/glove.6B.{cnn_config['embedding_dim']}d.txt"
    if os.path.exists(embedding_path):
        logger.info(f"Loading embeddings from {embedding_path}...")
        embedding_matrix = load_glove_vectors(embedding_path, vocab, cnn_config['embedding_dim'])
    else:
        logger.warning(f"Embedding file not found at {embedding_path}. Training from scratch.")
    
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=cnn_config['batch_size'], shuffle=True, 
                              collate_fn=lambda b: collate_batch(b, vocab, 512, device))
    valid_loader = DataLoader(valid_data, batch_size=cnn_config['batch_size'], shuffle=False, 
                              collate_fn=lambda b: collate_batch(b, vocab, 512, device))
    
    # Initialize Model
    model = CNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=cnn_config['embedding_dim'],
        num_filters=cnn_config['num_filters'],
        filter_sizes=cnn_config['filter_sizes'],
        output_dim=2, # Binary classification (Human vs Machine)
        dropout=cnn_config['dropout'],
        embedding_matrix=embedding_matrix
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cnn_config['learning_rate'])
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Training Loop
    best_valid_loss = float('inf')
    patience_counter = 0
    output_dir = os.path.join(training_config["output_dir"], "cnn")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting training...")
    for epoch in range(cnn_config['epochs']):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        for text, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cnn_config['epochs']}"):
            optimizer.zero_grad()
            predictions = model(text)
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_acc / len(train_loader)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        logger.info(f"          Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
            patience_counter = 0
            logger.info("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= training_config['early_stopping_patience']:
                logger.info("Early stopping triggered.")
                break
                
    logger.info(f"Training complete. Best model saved to {output_dir}")

if __name__ == "__main__":
    main()
