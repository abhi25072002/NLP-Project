import torch
import numpy as np
import os
from collections import Counter

class Vocabulary:
    def __init__(self, max_size=None, min_freq=1, specials=["<pad>", "<unk>"]):
        self.max_size = max_size
        self.min_freq = min_freq
        self.specials = specials
        self.stoi = {}
        self.itos = []
        
    def build_vocabulary(self, texts):
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                tokens = text.split() # Simple whitespace tokenization
            else:
                tokens = text
            counter.update(tokens)
            
        # Add specials first
        for s in self.specials:
            self.stoi[s] = len(self.itos)
            self.itos.append(s)
            
        # Sort by frequency
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if freq < self.min_freq:
                break
            if self.max_size and len(self.itos) >= self.max_size:
                break
            
            if word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                
    def __len__(self):
        return len(self.itos)
        
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get("<unk>"))
        
    def lookup_indices(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split()
        return [self[token] for token in tokens]

def load_glove_vectors(path, vocab, embedding_dim=100):
    """
    Load GloVe vectors and create an embedding matrix for the vocabulary.
    """
    if not os.path.exists(path):
        print(f"Warning: Embedding file not found at {path}. Initializing randomly.")
        return torch.randn(len(vocab), embedding_dim)
        
    embeddings_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    hits = 0
    misses = 0
    
    for i, word in enumerate(vocab.itos):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Check if dimension matches
            if len(embedding_vector) == embedding_dim:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))
        else:
            misses += 1
            # Initialize random vector for unknown words
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))
            
    print(f"Loaded {hits} vectors. {misses} words not found in embeddings.")
    return torch.tensor(embedding_matrix, dtype=torch.float32)
