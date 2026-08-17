"""
Loss Functions
==============
VAE loss functions for training.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Compute reconstruction loss (cross-entropy).
    
    Args:
        logits: Decoder output (batch, seq_len, vocab_size)
        targets: Target sequences (batch, seq_len)
        pad_idx: Padding token index to ignore
        
    Returns:
        Reconstruction loss
    """
    batch_size, seq_len, vocab_size = logits.shape
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=pad_idx,
        reduction='mean'
    )


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence from standard normal.
    
    Args:
        mu: Latent mean
        logvar: Latent log variance
        
    Returns:
        KL divergence
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.1,
    pad_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = Reconstruction + KL divergence.
    
    Args:
        logits: Decoder output (batch, seq_len, vocab_size)
        targets: Target sequences (batch, seq_len)
        mu: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL term (beta-VAE)
        pad_idx: Padding token index
        
    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon_loss = reconstruction_loss(logits, targets, pad_idx)
    kl_loss = kl_divergence(mu, logvar)
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss
