"""
CLIR-Net: Cross-Lingual Invariant Representation Network for Emotion Recognition
Full model integrating Audio Encoder, Action Unit Encoder, Structural Causal Model (SCM),
and Emotion Classification Head with Causal Interventions.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from models.audio_encoder import AudioEncoder
from models.au_encoder import AUEncoder
from models.scm import SCM
from models.emotion_head import EmotionHead


class CLIRNet(nn.Module):
    """
    Complete CLIR-Net model

    Architecture:
        1. Audio Encoder (Wav2Vec2/XLS-R + frame-level prosodic features)
        2. Action Unit (AU) Encoder (Transformer for facial dynamics)
        3. Speaker & Language Embeddings
        4. SCM (Structural Causal Model with learnable edge gates)
        5. Emotion Classifier (predicts categorical emotion states)
    """

    def __init__(
        self,
        # Audio encoder config
        audio_pretrained: str = "facebook/wav2vec2-base",
        audio_freeze_layers: int = 6,
        prosody_dim: int = 6,
        audio_out_dim: int = 512,
        # AU encoder config
        au_dim: int = 16,
        au_encoder_dim: int = 256,
        au_encoder_heads: int = 8,
        au_encoder_layers: int = 3,
        au_out_dim: int = 512,
        # SCM config
        n_speakers: int = 100,  # Dynamically set based on dataset vocabulary
        n_languages: int = 5,
        speaker_dim: int = 64,
        lang_dim: int = 16,
        latent_dim: int = 256,
        scm_hidden_dim: int = 512,
        # Emotion head config
        n_classes: int = 6,
        emotion_hidden_dim: int = 128,
        # General
        dropout: float = 0.1,
        learn_structure: bool = True,
        audio_only: bool = False
    ):
        super().__init__()

        self.n_speakers = n_speakers
        self.n_languages = n_languages
        self.n_classes = n_classes
        self.audio_only = audio_only

        # Encoders
        self.audio_encoder = AudioEncoder(
            pretrained_model=audio_pretrained,
            freeze_layers=audio_freeze_layers,
            prosody_dim=prosody_dim,
            prosody_hidden=128,
            out_dim=audio_out_dim,
            dropout=dropout
        )

        if not audio_only:
            self.au_encoder = AUEncoder(
                au_dim=au_dim,
                d_model=au_encoder_dim,
                nhead=au_encoder_heads,
                num_layers=au_encoder_layers,
                out_dim=au_out_dim,
                dropout=dropout
            )
        else:
            self.au_encoder = None

        # Embeddings
        self.speaker_embedding = nn.Embedding(n_speakers, speaker_dim)
        self.language_embedding = nn.Embedding(n_languages, lang_dim)

        # SCM
        self.scm = SCM(
            audio_dim=audio_out_dim,
            au_dim=au_out_dim,
            speaker_dim=speaker_dim,
            lang_dim=lang_dim,
            latent_dim=latent_dim,
            hidden_dim=scm_hidden_dim,
            dropout=dropout,
            learn_structure=learn_structure,
            audio_only=audio_only
        )

        # Emotion classifier
        self.emotion_head = EmotionHead(
            latent_dim=latent_dim,
            hidden_dim=emotion_hidden_dim,
            n_classes=n_classes,
            dropout=dropout
        )

    def forward(
        self,
        waveform: torch.Tensor,
        prosody_frames: torch.Tensor,
        aus: torch.Tensor,
        speaker_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        au_mask: Optional[torch.Tensor] = None,
        return_counterfactual: bool = False,
        cf_lang_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            waveform: (B, T_wave) raw audio
            prosody_frames: (B, T_frames, prosody_dim) prosodic features
            aus: (B, T_au, au_dim) AU intensities
            speaker_ids: (B,) speaker IDs
            lang_ids: (B,) language IDs
            audio_mask: Optional (B, T_wave) audio mask
            au_mask: Optional (B, T_au) AU mask
            return_counterfactual: Whether to compute counterfactual under intervention
            cf_lang_ids: Optional (B,) counterfactual language IDs

        Returns:
            Dictionary containing:
                - logits: (B, n_classes) emotion logits
                - E: (B, latent_dim) emotion embedding
                - scm_outputs: SCM node values
                - (optional) E_cf, logits_cf: counterfactual outputs
        """
        batch_size = waveform.size(0)

        # Encode audio
        audio_emb, _ = self.audio_encoder(waveform, prosody_frames, audio_mask)

        # Encode AUs
        if not self.audio_only:
            au_emb, _ = self.au_encoder(aus, au_mask)
        else:
            # Dummy AU embedding (zeros)
            # SCM will ignore it anyway due to audio_only=True
            au_emb = torch.zeros(batch_size, self.scm.au_dim, device=waveform.device)

        # Get embeddings
        speaker_emb = self.speaker_embedding(speaker_ids)
        lang_emb = self.language_embedding(lang_ids)

        # SCM forward
        scm_outputs = self.scm(audio_emb, au_emb, speaker_emb, lang_emb)
        E = scm_outputs['E']

        # Classify emotion
        logits = self.emotion_head(E)

        # Prepare output
        outputs = {
            'logits': logits,
            'E': E,
            'scm_outputs': scm_outputs,
            'audio_emb': audio_emb,
            'au_emb': au_emb
        }

        # Counterfactual intervention: do(L = L')
        if return_counterfactual:
            if cf_lang_ids is None:
                raise ValueError("cf_lang_ids must be provided when return_counterfactual=True")

            cf_lang_emb = self.language_embedding(cf_lang_ids)

            # Intervene: do(L = L')
            E_original, E_cf = self.scm.intervene(
                audio_emb, au_emb, speaker_emb, lang_emb, cf_lang_emb
            )

            # Classify counterfactual
            logits_cf = self.emotion_head(E_cf)

            outputs['E_cf'] = E_cf
            outputs['logits_cf'] = logits_cf

        return outputs

    def get_causal_structure(self) -> torch.Tensor:
        """Get learned causal adjacency matrix"""
        return self.scm.get_adjacency_matrix()

    def compute_acyclicity_penalty(self) -> torch.Tensor:
        """Compute acyclicity penalty for DAG constraint"""
        return self.scm.acyclicity_penalty()





def test_clirnet():
    """Test full CLIR-Net model"""
    print("Testing CLIR-Net...")

    # Config
    batch_size = 2
    T_wave = 16000 * 3  # 3 seconds
    T_frames = 300
    T_au = 300
    prosody_dim = 6
    au_dim = 16
    n_speakers = 50
    n_languages = 5
    n_classes = 6

    # Create dummy inputs
    waveform = torch.randn(batch_size, T_wave)
    prosody = torch.randn(batch_size, T_frames, prosody_dim)
    aus = torch.randn(batch_size, T_au, au_dim)
    speaker_ids = torch.randint(0, n_speakers, (batch_size,))
    lang_ids = torch.randint(0, n_languages, (batch_size,))
    cf_lang_ids = torch.randint(0, n_languages, (batch_size,))

    # Initialize model
    model = CLIRNet(
        audio_pretrained="facebook/wav2vec2-base",
        audio_freeze_layers=6,
        prosody_dim=prosody_dim,
        au_dim=au_dim,
        n_speakers=n_speakers,
        n_languages=n_languages,
        n_classes=n_classes,
        learn_structure=True
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Forward pass without counterfactual
    with torch.no_grad():
        outputs = model(
            waveform, prosody, aus, speaker_ids, lang_ids,
            return_counterfactual=False
        )

    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"E shape: {outputs['E'].shape}")

    # Forward pass with counterfactual
    with torch.no_grad():
        outputs_cf = model(
            waveform, prosody, aus, speaker_ids, lang_ids,
            return_counterfactual=True,
            cf_lang_ids=cf_lang_ids
        )

    print(f"E_cf shape: {outputs_cf['E_cf'].shape}")
    print(f"Logits_cf shape: {outputs_cf['logits_cf'].shape}")

    # Check causal structure
    adj = model.get_causal_structure()
    print(f"Adjacency matrix:\n{adj}")

    # Check acyclicity
    h = model.compute_acyclicity_penalty()
    print(f"Acyclicity penalty: {h.item():.4f}")

    print("✓ CLIR-Net test passed!")





if __name__ == "__main__":
    test_clirnet()
