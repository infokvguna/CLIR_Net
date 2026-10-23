"""
Emotion Classification Head
"""

import torch
import torch.nn as nn


class EmotionHead(nn.Module):
    """
    Emotion classification head
    Maps latent emotion embedding to emotion class probabilities
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        n_classes: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            E: (B, latent_dim) emotion embeddings
            
        Returns:
            logits: (B, n_classes) class logits
        """
        return self.classifier(E)


if __name__ == "__main__":
    # Test
    batch_size = 2
    latent_dim = 256
    n_classes = 6
    
    E = torch.randn(batch_size, latent_dim)
    head = EmotionHead(latent_dim, 128, n_classes)
    
    logits = head(E)
    print(f"Input shape: {E.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, n_classes)
    print("✓ Emotion Head test passed!")
