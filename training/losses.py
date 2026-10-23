"""
Loss Functions for CLIR-Net Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CLIRNetLoss(nn.Module):
    """
    Combined loss for CLIR-Net training
    
    L_total = L_cls + alpha * L_inv + beta * L_cf_pred + gamma * L_acyc
    """
    
    def __init__(
        self,
        alpha_inv: float = 1.0,
        beta_cf_pred: float = 1.0,
        gamma_acyc: float = 1e-2,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.alpha_inv = alpha_inv
        self.beta_cf_pred = beta_cf_pred
        self.gamma_acyc = gamma_acyc
        
        # Classification loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy classification loss
        
        Args:
            logits: (B, n_classes)
            labels: (B,)
        """
        return self.ce_loss(logits, labels)
    
    def invariance_loss(
        self,
        E: torch.Tensor,
        E_cf: torch.Tensor
    ) -> torch.Tensor:
        """
        Counterfactual invariance loss
        L_inv = ||E - E_cf||^2
        
        Encourages emotion embedding to be stable under language interventions
        
        Args:
            E: (B, latent_dim) original emotion embeddings
            E_cf: (B, latent_dim) counterfactual emotion embeddings
        """
        return F.mse_loss(E, E_cf)
    
    def cf_prediction_loss(
        self,
        logits: torch.Tensor,
        logits_cf: torch.Tensor
    ) -> torch.Tensor:
        """
        Counterfactual prediction consistency loss
        L_cf_pred = KL(p(y|E) || p(y|E_cf))
        
        Encourages consistent predictions under interventions
        
        Args:
            logits: (B, n_classes) original logits
            logits_cf: (B, n_classes) counterfactual logits
        """
        # Convert to log probabilities
        log_p = F.log_softmax(logits, dim=-1)
        p_cf = F.softmax(logits_cf, dim=-1)
        
        # KL divergence
        kl = F.kl_div(log_p, p_cf, reduction='batchmean')
        
        return kl
    
    def acyclicity_penalty(
        self,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Acyclicity penalty for DAG constraint
        
        Args:
            h: Acyclicity measure from SCM
        """
        return h
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        acyc_penalty: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            outputs: Model outputs dictionary containing:
                - logits: (B, n_classes)
                - E: (B, latent_dim)
                - E_cf: (B, latent_dim) if counterfactual computed
                - logits_cf: (B, n_classes) if counterfactual computed
            labels: (B,) ground truth labels
            acyc_penalty: Optional acyclicity penalty
            
        Returns:
            Dictionary of losses
        """
        # Classification loss
        loss_cls = self.classification_loss(outputs['logits'], labels)
        
        # Initialize other losses
        loss_inv = torch.tensor(0.0, device=loss_cls.device)
        loss_cf_pred = torch.tensor(0.0, device=loss_cls.device)
        loss_acyc = torch.tensor(0.0, device=loss_cls.device)
        
        # Counterfactual losses (if available)
        if 'E_cf' in outputs and 'logits_cf' in outputs:
            loss_inv = self.invariance_loss(outputs['E'], outputs['E_cf'])
            loss_cf_pred = self.cf_prediction_loss(outputs['logits'], outputs['logits_cf'])
        
        # Acyclicity penalty
        if acyc_penalty is not None:
            loss_acyc = self.acyclicity_penalty(acyc_penalty)
        
        # Total loss
        loss_total = (
            loss_cls +
            self.alpha_inv * loss_inv +
            self.beta_cf_pred * loss_cf_pred +
            self.gamma_acyc * loss_acyc
        )
        
        return {
            'loss': loss_total,
            'loss_cls': loss_cls,
            'loss_inv': loss_inv,
            'loss_cf_pred': loss_cf_pred,
            'loss_acyc': loss_acyc
        }


def test_losses():
    """Test loss functions"""
    print("Testing CLIR-Net Loss...")
    
    batch_size = 4
    n_classes = 6
    latent_dim = 256
    
    # Create dummy outputs
    logits = torch.randn(batch_size, n_classes)
    logits_cf = torch.randn(batch_size, n_classes)
    E = torch.randn(batch_size, latent_dim)
    E_cf = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, n_classes, (batch_size,))
    acyc_penalty = torch.tensor(0.5)
    
    outputs = {
        'logits': logits,
        'E': E,
        'E_cf': E_cf,
        'logits_cf': logits_cf
    }
    
    # Initialize loss
    criterion = CLIRNetLoss(
        alpha_inv=1.0,
        beta_cf_pred=1.0,
        gamma_acyc=1e-2
    )
    
    # Compute losses
    losses = criterion(outputs, labels, acyc_penalty)
    
    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"Classification loss: {losses['loss_cls'].item():.4f}")
    print(f"Invariance loss: {losses['loss_inv'].item():.4f}")
    print(f"CF prediction loss: {losses['loss_cf_pred'].item():.4f}")
    print(f"Acyclicity loss: {losses['loss_acyc'].item():.4f}")
    
    # Test backward
    losses['loss'].backward()
    
    print("✓ Loss test passed!")


if __name__ == "__main__":
    test_losses()
