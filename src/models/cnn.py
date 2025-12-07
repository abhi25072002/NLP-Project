# src/models/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        """
        Initialize the Kim CNN model.
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        """
        Forward pass.
        Args:
            text: [batch_size, sent_len]
        Returns:
            logits: [batch_size, output_dim]
        """
        # [batch_size, sent_len, emb_dim]
        embedded = self.embedding(text)
        
        # [batch_size, emb_dim, sent_len]
        embedded = embedded.permute(0, 2, 1)
        
        # Apply conv and relu
        # [batch_size, num_filters, sent_len - filter_sizes[n] + 1]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        
        # Max pooling over time
        # [batch_size, num_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate
        # [batch_size, num_filters * len(filter_sizes)]
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        # FC layer
        return self.fc(cat)
