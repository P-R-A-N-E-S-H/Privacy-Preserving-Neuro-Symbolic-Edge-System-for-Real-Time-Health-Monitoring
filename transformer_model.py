import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Inject temporal information into the sequence
    """
    def __init__(self, d_model, max_len=24):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return x


class SepsisTransformer(nn.Module):
    """
    Time-Series Transformer for Sepsis Prediction
    
    Architecture:
    - Input: [batch, seq_len=24, features=6] vital signs
    - Embedding layer to project features to d_model dimensions
    - Positional encoding to inject temporal order
    - Multi-head self-attention layers to capture long-range dependencies
    - Classification head for binary sepsis prediction
    """
    def __init__(
        self,
        n_features=6,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        # Input projection: [batch, seq, features] -> [batch, seq, d_model]
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=24)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # CRITICAL: Use batch_first=True for [batch, seq, feat]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary classification: healthy vs sepsis
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Xavier initialization for better convergence
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch, seq_len=24, features=6]
        
        Returns:
            logits: [batch, 2] - raw class scores
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # [batch, seq, d_model]
        
        # Transpose for positional encoding: [batch, seq, d_model] -> [seq, batch, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # [seq, batch, d_model]
        
        # Transpose back for transformer: [seq, batch, d_model] -> [batch, seq, d_model]
        x = x.transpose(0, 1)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # [batch, seq, d_model]
        
        # Global average pooling over time dimension
        x = x.mean(dim=1)  # [batch, d_model]
        
        # Classification
        logits = self.classifier(x)  # [batch, 2]
        
        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = SepsisTransformer(
        n_features=6,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    )
    
    print("SepsisTransformer Architecture:")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  d_model: {model.d_model}")
    
    # Test forward pass
    batch_size = 16
    seq_len = 24
    n_features = 6
    
    x_test = torch.randn(batch_size, seq_len, n_features)
    logits = model(x_test)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x_test.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output: {logits[0]}")
    
    probs = torch.softmax(logits, dim=1)
    print(f"  Probabilities: {probs[0]}")