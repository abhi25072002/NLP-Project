import argparse
import logging
import yaml
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.data.loader import load_turingbench
from src.models.cnn import CNNClassifier
from src.utils.embeddings import Vocabulary, load_glove_vectors
from scripts.train_cnn import collate_batch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Predict with CNN model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file for predictions")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to predict on")
    args = parser.parse_args()
    
    config = load_config(args.config)
    cnn_config = config["supervised"]["cnn"]
    training_config = config["training"]
    
    device = torch.device(training_config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Data
    logger.info(f"Loading data ({args.split})...")
    dataset = load_turingbench(split=args.split)
    
    # Build Vocabulary (Must match training!)
    # Ideally we should load the vocab used during training.
    # Since we didn't save the vocab explicitly in train_cnn.py (my bad), we have to rebuild it 
    # assuming the same training data or save/load it.
    # For now, I will rebuild it using the TRAIN split, which is inefficient but ensures consistency 
    # if the seed is same. 
    # TODO: In a real scenario, save/load vocab.pickle.
    logger.info("Rebuilding vocabulary from training data (for consistency)...")
    train_dataset = load_turingbench(split="train")
    # Note: If we split train/val in train script, we used the whole 'train' split from loader 
    # and then split it. So vocab was built on the 'train' part of that split.
    # To be exact, I should replicate the split logic.
    train_test_split = train_dataset.train_test_split(test_size=0.2, seed=training_config["seed"])
    train_data = train_test_split['train']
    
    vocab = Vocabulary(max_size=25000)
    vocab.build_vocabulary([item['text'] for item in train_data])
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Load Model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = CNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=cnn_config['embedding_dim'],
        num_filters=cnn_config['num_filters'],
        filter_sizes=cnn_config['filter_sizes'],
        output_dim=2,
        dropout=cnn_config['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # DataLoader
    loader = DataLoader(dataset, batch_size=cnn_config['batch_size'], shuffle=False, 
                        collate_fn=lambda b: collate_batch(b, vocab, 512, device))
    
    # Predict
    logger.info("Generating predictions...")
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for text, labels in tqdm(loader):
            outputs = model(text)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of class 1 (Machine?)
            
    # Save Results
    df = pd.DataFrame({
        "text": dataset["text"], # Assuming dataset is indexable like this or list
        "label": dataset["label"],
        "prediction": all_preds,
        "probability": all_probs
    })
    
    df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
