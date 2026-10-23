"""
Action Unit (AU) Encoder Module
Transformer-based encoder for facial AU time-series
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AUEncoder(nn.Module):
    """
    Video AU encoder using transformer architecture
    Processes temporal sequences of Action Unit intensities
    """
    
    def __init__(
        self,
        au_dim: int = 16,  # Number of AU features
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        out_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()
        
        self.au_dim = au_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(au_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(
        self,
        aus: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            aus: (B, T, au_dim) AU intensity sequences
            mask: Optional (B, T) padding mask
            
        Returns:
            utterance_emb: (B, out_dim) utterance-level embedding
            frame_emb: (B, T, out_dim) frame-level embeddings
        """
        batch_size, seq_len, _ = aus.shape
        
        # Project input
        x = self.input_proj(aus)  # (B, T, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask if needed
        if mask is not None:
            # Convert padding mask to attention mask
            # True values are masked (not attended to)
            attn_mask = mask.bool()
        else:
            attn_mask = None
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)  # (B, T, d_model)
        
        # Project to output dimension
        frame_emb = self.output_proj(encoded)  # (B, T, out_dim)
        
        # Utterance-level pooling (masked mean if mask provided)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = (~mask).unsqueeze(-1).float()  # (B, T, 1)
            utterance_emb = (frame_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            utterance_emb = frame_emb.mean(dim=1)  # (B, out_dim)
        
        return utterance_emb, frame_emb


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def test_au_encoder():
    """Test AU encoder"""
    print("Testing AU Encoder...")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 300  # 3 seconds at 100Hz
    au_dim = 16
    
    aus = torch.randn(batch_size, seq_len, au_dim)
    
    # Create padding mask (simulate variable length sequences)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[0, 250:] = True  # First sample has 250 frames
    mask[1, 200:] = True  # Second sample has 200 frames
    
    # Initialize encoder
    encoder = AUEncoder(
        au_dim=au_dim,
        d_model=256,
        nhead=8,
        num_layers=3,
        out_dim=512
    )
    
    # Forward pass
    with torch.no_grad():
        utterance_emb, frame_emb = encoder(aus, mask)
    
    print(f"Input AU shape: {aus.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output utterance embedding shape: {utterance_emb.shape}")
    print(f"Output frame embedding shape: {frame_emb.shape}")
    
    # Check shapes
    assert utterance_emb.shape == (batch_size, 512)
    assert frame_emb.shape == (batch_size, seq_len, 512)
    
    print("✓ AU Encoder test passed!")


if __name__ == "__main__":
    test_au_encoder()
