"""
Audio Encoder Module
Combines pretrained Wav2Vec2 with prosodic feature processing
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional, Tuple


class AudioEncoder(nn.Module):
    """
    Audio encoder combining:
    1. Pretrained Wav2Vec2/XLS-R for waveform embeddings
    2. Prosody projection network for F0, energy, jitter, shimmer
    3. Temporal transformer for frame-level fusion
    4. Adaptive pooling to utterance-level representation
    """
    
    def __init__(
        self,
        pretrained_model: str = "facebook/wav2vec2-base",
        freeze_layers: int = 6,
        prosody_dim: int = 6,  # F0, voiced_prob, energy, centroid, rolloff, flux
        prosody_hidden: int = 128,
        out_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load pretrained Wav2Vec2
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        self.wav2vec_dim = self.wav2vec.config.hidden_size
        
        # Freeze early layers
        if freeze_layers > 0:
            # Freeze feature extractor
            for param in self.wav2vec.feature_extractor.parameters():
                param.requires_grad = False
            
            # Freeze first N transformer layers
            for layer in self.wav2vec.encoder.layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Prosody projection
        self.prosody_proj = nn.Sequential(
            nn.Linear(prosody_dim, prosody_hidden),
            nn.LayerNorm(prosody_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prosody_hidden, prosody_hidden)
        )
        
        # Combined feature dimension
        combined_dim = self.wav2vec_dim + prosody_hidden
        
        # Temporal fusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=combined_dim,
            nhead=8,
            dim_feedforward=combined_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Adaptive pooling for utterance-level representation
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(
        self,
        waveform: torch.Tensor,
        prosody_frames: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            waveform: (B, T_wave) raw audio waveform
            prosody_frames: (B, T_frames, prosody_dim) frame-level prosody features
            attention_mask: Optional (B, T_wave) mask for waveform
            
        Returns:
            utterance_emb: (B, out_dim) utterance-level embedding
            frame_emb: (B, T_frames, out_dim) frame-level embeddings
        """
        batch_size = waveform.size(0)
        
        # Extract Wav2Vec2 features
        wav_outputs = self.wav2vec(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        wav_features = wav_outputs.last_hidden_state  # (B, T_w, wav2vec_dim)
        
        # Process prosody features
        prosody_emb = self.prosody_proj(prosody_frames)  # (B, T_frames, prosody_hidden)
        
        # Align wav2vec features with prosody frames
        # Wav2Vec2 downsamples by ~320x (for 16kHz), prosody is at 100Hz (160 samples/frame)
        # We need to match dimensions
        T_wav = wav_features.size(1)
        T_prosody = prosody_emb.size(1)
        
        if T_wav != T_prosody:
            # Interpolate wav features to match prosody length
            wav_features = wav_features.permute(0, 2, 1)  # (B, C, T)
            wav_features = nn.functional.interpolate(
                wav_features,
                size=T_prosody,
                mode='linear',
                align_corners=False
            )
            wav_features = wav_features.permute(0, 2, 1)  # (B, T, C)
        
        # Concatenate features
        combined = torch.cat([wav_features, prosody_emb], dim=-1)  # (B, T, combined_dim)
        
        # Temporal fusion
        fused = self.temporal_encoder(combined)  # (B, T, combined_dim)
        
        # Project to output dimension
        frame_emb = self.output_proj(fused)  # (B, T, out_dim)
        
        # Utterance-level pooling
        utterance_emb = frame_emb.mean(dim=1)  # (B, out_dim)
        
        return utterance_emb, frame_emb


def test_audio_encoder():
    """Test audio encoder"""
    print("Testing Audio Encoder...")
    
    # Create dummy inputs
    batch_size = 2
    T_wave = 16000 * 3  # 3 seconds at 16kHz
    T_frames = 300  # 3 seconds at 100Hz
    prosody_dim = 6
    
    waveform = torch.randn(batch_size, T_wave)
    prosody = torch.randn(batch_size, T_frames, prosody_dim)
    
    # Initialize encoder
    encoder = AudioEncoder(
        pretrained_model="facebook/wav2vec2-base",
        freeze_layers=6,
        prosody_dim=prosody_dim,
        out_dim=512
    )
    
    # Forward pass
    with torch.no_grad():
        utterance_emb, frame_emb = encoder(waveform, prosody)
    
    print(f"Input waveform shape: {waveform.shape}")
    print(f"Input prosody shape: {prosody.shape}")
    print(f"Output utterance embedding shape: {utterance_emb.shape}")
    print(f"Output frame embedding shape: {frame_emb.shape}")
    
    # Check shapes
    assert utterance_emb.shape == (batch_size, 512)
    assert frame_emb.shape[0] == batch_size
    assert frame_emb.shape[2] == 512
    
    print("✓ Audio Encoder test passed!")


if __name__ == "__main__":
    test_audio_encoder()
