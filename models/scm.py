"""
Structural Causal Model (SCM) Module
Implements learnable causal structure with interventions for CLIR-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import math


class SCM(nn.Module):
    """
    Structural Causal Model for emotion recognition
    
    Causal Graph (assumed structure):
        L (Language) → A (Acoustic features)
        L → F (Facial features)
        S (Speaker) → A
        S → F
        S → VQ (Voice Quality)
        A → VQ
        A → E (Emotion)
        F → E
        VQ → E
        N (Noise) → A, F
    
    We learn the strength of these edges via learnable gates
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        au_dim: int = 512,
        speaker_dim: int = 64,
        lang_dim: int = 16,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        learn_structure: bool = True,
        audio_only: bool = False
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.au_dim = au_dim
        self.speaker_dim = speaker_dim
        self.lang_dim = lang_dim
        self.latent_dim = latent_dim
        self.learn_structure = learn_structure
        self.audio_only = audio_only
        
        # Learnable edge weights (gates)
        # Initialize with small positive values
        if learn_structure:
            self.w_LA = nn.Parameter(torch.tensor(0.5))  # Language → Acoustic
            self.w_SA = nn.Parameter(torch.tensor(0.5))  # Speaker → Acoustic
            self.w_SVQ = nn.Parameter(torch.tensor(0.5))  # Speaker → VoiceQuality
            self.w_AVQ = nn.Parameter(torch.tensor(0.5))  # Acoustic → VoiceQuality
            self.w_AE = nn.Parameter(torch.tensor(0.5))  # Acoustic → Emotion
            self.w_VQE = nn.Parameter(torch.tensor(0.5))  # VoiceQuality → Emotion
            
            if not audio_only:
                self.w_LF = nn.Parameter(torch.tensor(0.5))  # Language → Facial
                self.w_SF = nn.Parameter(torch.tensor(0.5))  # Speaker → Facial
                self.w_FE = nn.Parameter(torch.tensor(0.5))  # Facial → Emotion
            else:
                # Register as buffers (fixed to 0) if audio only
                self.register_buffer('w_LF', torch.tensor(-10.0))  # Sigmoid(-10) ~ 0
                self.register_buffer('w_SF', torch.tensor(-10.0))
                self.register_buffer('w_FE', torch.tensor(-10.0))
        else:
            # Fixed weights (all edges active)
            self.register_buffer('w_LA', torch.tensor(1.0))
            self.register_buffer('w_SA', torch.tensor(1.0))
            self.register_buffer('w_SVQ', torch.tensor(1.0))
            self.register_buffer('w_AVQ', torch.tensor(1.0))
            self.register_buffer('w_AE', torch.tensor(1.0))
            self.register_buffer('w_VQE', torch.tensor(1.0))
            
            if not audio_only:
                self.register_buffer('w_LF', torch.tensor(1.0))
                self.register_buffer('w_SF', torch.tensor(1.0))
                self.register_buffer('w_FE', torch.tensor(1.0))
            else:
                self.register_buffer('w_LF', torch.tensor(-10.0))
                self.register_buffer('w_SF', torch.tensor(-10.0))
                self.register_buffer('w_FE', torch.tensor(-10.0))
        
        # Node functions (structural equations)
        # A_node = g_A(a, l, s, n)
        self.g_A = nn.Sequential(
            nn.Linear(audio_dim + lang_dim + speaker_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # F_node = g_F(f, l, s, n)
        self.g_F = nn.Sequential(
            nn.Linear(au_dim + lang_dim + speaker_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # VQ_node = g_VQ(a, s)
        self.g_VQ = nn.Sequential(
            nn.Linear(audio_dim + speaker_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2)
        )
        
        # E = g_E(A_node, F_node, VQ_node)
        self.g_E = nn.Sequential(
            nn.Linear(latent_dim * 2 + latent_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(
        self,
        a: torch.Tensor,  # Audio embedding
        f: torch.Tensor,  # Facial (AU) embedding
        s: torch.Tensor,  # Speaker embedding
        l: torch.Tensor,  # Language embedding
        n: Optional[torch.Tensor] = None  # Noise (optional)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SCM
        
        Args:
            a: (B, audio_dim) audio embeddings
            f: (B, au_dim) facial embeddings
            s: (B, speaker_dim) speaker embeddings
            l: (B, lang_dim) language embeddings
            n: Optional noise
            
        Returns:
            Dictionary with all node values and emotion embedding
        """
        batch_size = a.size(0)
        
        # Apply edge gates (sigmoid to keep in [0, 1])
        gate_LA = torch.sigmoid(self.w_LA)
        gate_LF = torch.sigmoid(self.w_LF)
        gate_SA = torch.sigmoid(self.w_SA)
        gate_SF = torch.sigmoid(self.w_SF)
        gate_SVQ = torch.sigmoid(self.w_SVQ)
        gate_AVQ = torch.sigmoid(self.w_AVQ)
        gate_AE = torch.sigmoid(self.w_AE)
        gate_FE = torch.sigmoid(self.w_FE)
        gate_VQE = torch.sigmoid(self.w_VQE)
        
        # Compute A_node (acoustic node)
        # Influenced by: audio embedding, language, speaker
        a_input = torch.cat([a, l * gate_LA, s * gate_SA], dim=-1)
        A_node = self.g_A(a_input)
        
        # Compute F_node (facial node)
        if not self.audio_only:
            # Influenced by: facial embedding, language, speaker
            f_input = torch.cat([f, l * gate_LF, s * gate_SF], dim=-1)
            F_node = self.g_F(f_input)
        else:
            # Zero out F_node
            F_node = torch.zeros(batch_size, self.latent_dim, device=a.device)
        
        # Compute VQ_node (voice quality node)
        # Influenced by: audio embedding, speaker
        vq_input = torch.cat([a * gate_AVQ, s * gate_SVQ], dim=-1)
        VQ_node = self.g_VQ(vq_input)
        
        # Compute E (emotion node)
        # Influenced by: A_node, F_node, VQ_node
        e_input = torch.cat([
            A_node * gate_AE,
            F_node * gate_FE,
            VQ_node * gate_VQE
        ], dim=-1)
        E = self.g_E(e_input)
        
        return {
            'A_node': A_node,
            'F_node': F_node,
            'VQ_node': VQ_node,
            'E': E,
            'gates': {
                'LA': gate_LA,
                'LF': gate_LF,
                'SA': gate_SA,
                'SF': gate_SF,
                'SVQ': gate_SVQ,
                'AVQ': gate_AVQ,
                'AE': gate_AE,
                'FE': gate_FE,
                'VQE': gate_VQE
            }
        }
    
    def intervene(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        s: torch.Tensor,
        l: torch.Tensor,
        l_prime: torch.Tensor,
        n: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform intervention do(L = l')
        
        Args:
            a, f, s, l: Original inputs
            l_prime: Intervened language embedding
            n: Optional noise
            
        Returns:
            E: Original emotion embedding
            E_cf: Counterfactual emotion embedding under intervention
        """
        # Original
        out_original = self.forward(a, f, s, l, n)
        E = out_original['E']
        
        # Counterfactual
        out_cf = self.forward(a, f, s, l_prime, n)
        E_cf = out_cf['E']
        
        return E, E_cf
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Get adjacency matrix of learned causal structure
        
        Returns:
            Adjacency matrix (7x7) for nodes: [L, S, N, A, F, VQ, E]
        """
        # Node order: L(0), S(1), N(2), A(3), F(4), VQ(5), E(6)
        adj = torch.zeros(7, 7)
        
        gates = {
            'LA': torch.sigmoid(self.w_LA).item(),
            'LF': torch.sigmoid(self.w_LF).item(),
            'SA': torch.sigmoid(self.w_SA).item(),
            'SF': torch.sigmoid(self.w_SF).item(),
            'SVQ': torch.sigmoid(self.w_SVQ).item(),
            'AVQ': torch.sigmoid(self.w_AVQ).item(),
            'AE': torch.sigmoid(self.w_AE).item(),
            'FE': torch.sigmoid(self.w_FE).item(),
            'VQE': torch.sigmoid(self.w_VQE).item()
        }
        
        # Fill adjacency matrix
        adj[0, 3] = gates['LA']  # L → A
        adj[0, 4] = gates['LF']  # L → F
        adj[1, 3] = gates['SA']  # S → A
        adj[1, 4] = gates['SF']  # S → F
        adj[1, 5] = gates['SVQ']  # S → VQ
        adj[3, 5] = gates['AVQ']  # A → VQ
        adj[3, 6] = gates['AE']  # A → E
        adj[4, 6] = gates['FE']  # F → E
        adj[5, 6] = gates['VQE']  # VQ → E
        
        return adj
    
    def acyclicity_penalty(self) -> torch.Tensor:
        """
        Compute acyclicity penalty (NOTEARS-style)
        h(W) = tr(e^(W ⊙ W)) - d
        
        For DAG constraint: h(W) = 0
        """
        adj = self.get_adjacency_matrix()
        
        # Compute matrix exponential
        d = adj.size(0)
        W_squared = adj * adj
        
        # Use matrix exponential approximation
        # exp(W²) ≈ I + W² + W⁴/2 + W⁶/6 + ...
        expm = torch.eye(d, device=adj.device)
        W_power = torch.eye(d, device=adj.device)
        
        for i in range(1, d):
            W_power = W_power @ W_squared
            expm = expm + W_power / math.factorial(i)
        
        h = torch.trace(expm) - d
        
        return h


def test_scm():
    """Test SCM module"""
    print("Testing SCM...")
    
    batch_size = 2
    audio_dim = 512
    au_dim = 512
    speaker_dim = 64
    lang_dim = 16
    
    # Create dummy inputs
    a = torch.randn(batch_size, audio_dim)
    f = torch.randn(batch_size, au_dim)
    s = torch.randn(batch_size, speaker_dim)
    l = torch.randn(batch_size, lang_dim)
    l_prime = torch.randn(batch_size, lang_dim)
    
    # Initialize SCM
    scm = SCM(
        audio_dim=audio_dim,
        au_dim=au_dim,
        speaker_dim=speaker_dim,
        lang_dim=lang_dim,
        latent_dim=256,
        learn_structure=True
    )
    
    # Forward pass
    with torch.no_grad():
        out = scm(a, f, s, l)
        E, E_cf = scm.intervene(a, f, s, l, l_prime)
    
    print(f"A_node shape: {out['A_node'].shape}")
    print(f"F_node shape: {out['F_node'].shape}")
    print(f"VQ_node shape: {out['VQ_node'].shape}")
    print(f"E shape: {out['E'].shape}")
    print(f"E_cf shape: {E_cf.shape}")
    
    # Check adjacency matrix
    adj = scm.get_adjacency_matrix()
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix:\n{adj}")
    
    # Check acyclicity penalty
    h = scm.acyclicity_penalty()
    print(f"Acyclicity penalty: {h.item():.4f}")
    
    print("✓ SCM test passed!")


if __name__ == "__main__":
    test_scm()
