# src/models/cnn.py
import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout, embedding_matrix=None):
        """
        Initialize the Kim CNN model.
        """
        super().__init__()
        
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
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
        text: (batch_size, seq_len)
        """
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text)
        
        # (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolution and max pooling
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        
        # Max pooling over time
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate pooled features
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        return self.fc(cat)
