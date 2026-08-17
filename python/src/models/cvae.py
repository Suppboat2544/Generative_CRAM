"""
Property-Conditioned CRAM VAE
=============================
Complete CVAE model combining encoder, decoder, and condition encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .encoder import DeepCRAMEncoder
from .decoder import SELFIESDecoder
from .condition import PropertyConditionEncoder


class PropertyConditionedCRAMVAE(nn.Module):
    """
    Complete Property-Conditioned Conditional VAE for CRAM generation.
    
    Architecture:
    - Graph encoder (GAT) -> latent space
    - Property condition encoder -> condition vector
    - SELFIES decoder (GRU) -> molecule sequence
    
    Token IDs (CRITICAL - aligned with SELFIESTokenizer):
    - PAD (Padding): 0
    - SOS (Start of Sequence): 1
    - EOS (End of Sequence): 2
    - UNK (Unknown): 3
    """
    
    def __init__(
        self,
        vocab_size: int,
        encoder_config: Dict,
        decoder_config: Dict,
        condition_dim: int = 32
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.latent_dim = encoder_config.get('latent_dim', 64)
        self.condition_dim = condition_dim
        
        # Encoder
        self.encoder = DeepCRAMEncoder(
            in_dim=encoder_config.get('in_dim', 163),
            hidden_dim=encoder_config.get('hidden_dim', 256),
            latent_dim=self.latent_dim,
            num_heads=encoder_config.get('num_heads', 4),
            num_layers=encoder_config.get('num_layers', 3),
            dropout=encoder_config.get('dropout', 0.1)
        )
        
        # Property condition encoder
        self.condition_encoder = PropertyConditionEncoder(condition_dim)
        
        # Decoder
        self.decoder = SELFIESDecoder(
            vocab_size=vocab_size,
            latent_dim=self.latent_dim,
            condition_dim=condition_dim,
            hidden_dim=decoder_config.get('hidden_dim', 256),
            num_layers=decoder_config.get('num_layers', 2),
            max_len=decoder_config.get('max_len', 100),
            dropout=decoder_config.get('dropout', 0.1)
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(
        self,
        g,
        node_feats: Optional[torch.Tensor] = None,
        properties: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode molecule graph to latent z and property condition."""
        enc = self.encoder(g)
        mu, logvar = enc["mu"], enc["logvar"]
        z = self.reparameterize(mu, logvar)
        if properties is None:
            batch_size = mu.size(0)
            device = mu.device
            properties = {
                "n_cooh": torch.full((batch_size,), 2, dtype=torch.long, device=device),
                "ring_class": torch.full((batch_size,), 3, dtype=torch.long, device=device),
                "aliphatic_pct": torch.full((batch_size,), 0.7, device=device),
                "mw": torch.full((batch_size,), 0.4, device=device),
                "has_benzene": torch.zeros(batch_size, device=device),
            }
        condition = self.condition_encoder(
            properties["n_cooh"],
            properties["ring_class"],
            properties["aliphatic_pct"],
            properties["mw"],
            properties["has_benzene"],
            properties.get("dbe_c"),
            properties.get("dbe_o"),
            properties.get("dbe_h"),
            properties.get("avg_ring_size"),
            properties.get("ring_size_class"),
        )
        return z, mu, logvar, condition
    
    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        target_seq: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """
        Decode latent vector to SELFIES sequence.
        
        Args:
            z: Latent vector
            condition: Condition vector
            target_seq: Target for teacher forcing (training)
            temperature: Sampling temperature (generation)
            sos_idx: Start of sequence token index (default: 1)
            eos_idx: End of sequence token index (default: 2)
        """
        if target_seq is not None:
            return self.decoder(z, condition, target_seq)
        else:
            return self._generate_autoregressive(z, condition, temperature, sos_idx, eos_idx)
    
    def _generate_autoregressive(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        temperature: float = 1.0,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """Generate SELFIES sequence autoregressively."""
        batch_size = z.size(0)
        device = z.device
        max_len = self.decoder.max_len
        
        current_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        hidden = self.decoder.init_hidden(z, condition)
        
        generated = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            embedded = self.decoder.embedding(current_token)
            embedded = self.decoder.dropout(embedded)
            output, hidden = self.decoder.gru(embedded, hidden)
            logits = self.decoder.output_proj(output[:, -1, :])
            
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                current_token = torch.multinomial(probs, 1)
            else:
                current_token = logits.argmax(dim=-1, keepdim=True)
            
            generated.append(current_token)
            finished = finished | (current_token.squeeze(-1) == eos_idx)
            if finished.all():
                break
        
        return torch.cat(generated, dim=1)
    
    def forward(
        self,
        g,
        node_feats: torch.Tensor,
        properties: Dict[str, torch.Tensor],
        target_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        z, mu, logvar, condition = self.encode(g, node_feats, properties)
        logits = self.decode(z, condition, target_seq)
        return logits, mu, logvar
    
    def generate(
        self,
        properties: Dict[str, torch.Tensor],
        num_samples: int = 1,
        temperature: float = 1.0,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """Generate molecules from property conditions."""
        device = next(self.parameters()).device
        
        batch_size = properties['n_cooh'].size(0)
        if num_samples > 1 and batch_size == 1:
            properties = {k: v.repeat(num_samples) for k, v in properties.items()}
            batch_size = num_samples

        condition = self.condition_encoder(
            properties["n_cooh"],
            properties["ring_class"],
            properties["aliphatic_pct"],
            properties["mw"],
            properties["has_benzene"],
            properties.get("dbe_c"),
            properties.get("dbe_o"),
            properties.get("dbe_h"),
            properties.get("avg_ring_size"),
            properties.get("ring_size_class"),
        )
        z = torch.randn(batch_size, self.latent_dim, device=device)

        return self.decode(z, condition, temperature=temperature, sos_idx=sos_idx, eos_idx=eos_idx)

    def generate_conditioned(
        self,
        z: torch.Tensor,
        conditions: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        start_token: int = 1,
        eos_idx: int = 2,
    ) -> torch.Tensor:
        """Decode latent + condition vectors to token indices (generation API)."""
        return self._generate_autoregressive(
            z, conditions, temperature=temperature, sos_idx=start_token, eos_idx=eos_idx
        )


def create_cvae_model(
    vocab_size: int,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    condition_dim: int = 32,
    in_dim: int = 163,
    max_len: int = 100
) -> PropertyConditionedCRAMVAE:
    """Factory function to create CVAE model."""
    encoder_config = {
        'in_dim': in_dim,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'num_heads': 4,
        'num_layers': 3,
        'dropout': 0.1
    }
    
    decoder_config = {
        'hidden_dim': hidden_dim,
        'num_layers': 2,
        'max_len': max_len,
        'dropout': 0.1
    }
    
    return PropertyConditionedCRAMVAE(
        vocab_size=vocab_size,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        condition_dim=condition_dim
    )
