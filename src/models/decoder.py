"""
SELFIES Decoder
===============
Autoregressive decoder for SELFIES sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SELFIESDecoder(nn.Module):
    """
    Autoregressive decoder for SELFIES sequences.
    
    Uses GRU with attention over latent+condition vector.
    """
    
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 64,
        condition_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_len: int = 100,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        
        # Combine latent + condition
        self.latent_condition_proj = nn.Linear(latent_dim + condition_dim, hidden_dim)
        
        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        target_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent vector to SELFIES sequence.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            condition: Condition vector (batch_size, condition_dim)
            target_seq: Target sequence for teacher forcing (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Combine latent and condition
        combined = torch.cat([z, condition], dim=-1)
        h0 = self.latent_condition_proj(combined)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        if target_seq is not None:
            # Teacher forcing
            embedded = self.embedding(target_seq)
            embedded = self.dropout(embedded)
            output, _ = self.gru(embedded, h0)
            logits = self.output_proj(output)
            return logits
        else:
            raise ValueError("Use decode() method for generation")
            
    def init_hidden(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Initialize hidden state from latent and condition."""
        combined = torch.cat([z, condition], dim=-1)
        h0 = self.latent_condition_proj(combined)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0
