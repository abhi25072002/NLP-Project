# src/models/cnn.py
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        """
        Initialize the Kim CNN model.
        """
        super().__init__()
        pass
        
    def forward(self, text):
        """
        Forward pass.
        """
        pass
