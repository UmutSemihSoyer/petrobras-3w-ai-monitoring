"""
Petrobras 3W AI Monitoring - Foundation Time-Series Transformer (PatchTST / TimesFM)

This module implements a Patch-based Time-Series Transformer architecture
for sequence anomaly classification across 3W sensor signals.
"""

import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformerClassifier(nn.Module):
    """
    PatchTST-style Time-Series Transformer Classifier for 3W sensor sequences.
    """
    def __init__(self, num_channels: int = 8, d_model: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.input_proj = nn.Linear(num_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, sequence_length, num_channels)
        """
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        out = self.transformer(h)
        # Global Average Pooling
        pooled = torch.mean(out, dim=1)
        logits = self.classifier(pooled)
        return logits
