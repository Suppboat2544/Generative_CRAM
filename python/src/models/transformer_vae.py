"""Notebook Transformer VAEs: DeepCRAMVAE and sweep variants + registry."""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    import dgl
except ImportError:
    dgl = None

from .deep_cram_encoder import DeepCRAMEncoder
from .deep_cram_vae import DeepCRAMDecoderVAE, training_style_vocab

class DeepCRAMVAE(nn.Module):
    """
    Complete Deep CRAM VAE with encoder, decoder, and multi-task learning.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 max_len: int = 100,
                 encoder_config: dict = None,
                 decoder_config: dict = None):
        super().__init__()
        
        encoder_config = encoder_config or {}
        decoder_config = decoder_config or {}
        
        # Encoder with all deep improvements
        self.encoder = DeepCRAMEncoder(**encoder_config)
        
        # Get latent dim from encoder
        latent_dim = self.encoder.latent_dim
        
        # Decoder (reuse existing Transformer decoder)
        self.decoder_embedding = nn.Embedding(vocab_size, decoder_config.get('hidden_dim', 256))
        self.decoder_pos_encoding = nn.Parameter(
            torch.randn(max_len, decoder_config.get('hidden_dim', 256)) * 0.02
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_config.get('hidden_dim', 256),
            nhead=decoder_config.get('num_heads', 8),
            dim_feedforward=decoder_config.get('ff_dim', 1024),
            dropout=decoder_config.get('dropout', 0.1),
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_config.get('num_layers', 6))
        
        # Latent to decoder hidden
        self.latent_proj = nn.Linear(latent_dim, decoder_config.get('hidden_dim', 256))
        
        # Output projection
        self.output_proj = nn.Linear(decoder_config.get('hidden_dim', 256), vocab_size)
    
    def forward(self, g: dgl.DGLGraph, target_seq: torch.Tensor = None):
        """Forward pass through VAE."""
        # Encode
        enc_output = self.encoder(g)
        z = enc_output['z']
        mu = enc_output['mu']
        logvar = enc_output['logvar']
        
        output = {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'predictions': enc_output['predictions']
        }
        
        # Decode if target provided
        if target_seq is not None:
            # Ensure z has batch dimension
            if z.dim() == 1:
                z = z.unsqueeze(0)
            
            # Get batch size from target_seq
            batch_size = target_seq.size(0)
            
            # If z batch doesn't match target batch, expand it
            if z.size(0) == 1 and batch_size > 1:
                z = z.expand(batch_size, -1)
            elif z.size(0) != batch_size:
                raise ValueError(f"Batch size mismatch: z has batch {z.size(0)}, target_seq has batch {batch_size}")
            
            # Project latent to decoder space
            memory = self.latent_proj(z).unsqueeze(1)  # (batch, 1, hidden)
            
            # Embed target
            tgt = self.decoder_embedding(target_seq)
            tgt = tgt + self.decoder_pos_encoding[:tgt.size(1)]
            
            # Generate causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            # Decode
            decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(decoded)
            
            output['logits'] = logits
        
        return output


class PropertyConditionedTransformerVAE(DeepCRAMVAE):
    """Sweep variant: slightly deeper decoder (property-conditioned naming)."""

    def __init__(self, vocab_size, max_len=150, encoder_config=None, decoder_config=None, **kwargs):
        encoder_config = encoder_config or {
            'in_dim': 163, 'hidden_dim': 256, 'latent_dim': 64,
            'num_gat_layers': 4, 'num_heads': 4, 'dropout': 0.1, 'use_contrastive': True,
        }
        decoder_config = decoder_config or {
            'hidden_dim': 256, 'num_heads': 8, 'ff_dim': 1024, 'num_layers': 6, 'dropout': 0.1,
        }
        super().__init__(vocab_size=vocab_size, max_len=max_len,
                         encoder_config=encoder_config, decoder_config=decoder_config)


class MultimodalTransformerVAE(DeepCRAMVAE):
    """Sweep variant: larger 128D latent (multimodal naming)."""

    def __init__(self, vocab_size, max_len=150, encoder_config=None, decoder_config=None, **kwargs):
        encoder_config = encoder_config or {
            'in_dim': 163, 'hidden_dim': 256, 'latent_dim': 128,
            'num_gat_layers': 4, 'num_heads': 4, 'dropout': 0.1, 'use_contrastive': True,
        }
        decoder_config = decoder_config or {
            'hidden_dim': 256, 'num_heads': 8, 'ff_dim': 1024, 'num_layers': 4, 'dropout': 0.1,
        }
        super().__init__(vocab_size=vocab_size, max_len=max_len,
                         encoder_config=encoder_config, decoder_config=decoder_config)


class TransformerVAE(DeepCRAMVAE):
    """Sweep variant: wider hidden / 128D latent (legacy transformer constructor)."""

    def __init__(
        self,
        vocab_size,
        max_len=150,
        encoder_config=None,
        decoder_config=None,
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        latent_size=128,
        dropout=0.1,
        **kwargs,
    ):
        hidden = max(256, d_model // 2)
        encoder_config = encoder_config or {
            'in_dim': 163, 'hidden_dim': hidden, 'latent_dim': latent_size,
            'num_gat_layers': num_encoder_layers, 'num_heads': 4, 'dropout': dropout,
            'use_contrastive': True,
        }
        decoder_config = decoder_config or {
            'hidden_dim': hidden, 'num_heads': nhead, 'ff_dim': dim_feedforward,
            'num_layers': num_decoder_layers, 'dropout': dropout,
        }
        super().__init__(vocab_size=vocab_size, max_len=max_len,
                         encoder_config=encoder_config, decoder_config=decoder_config)


def cram_model_registry():
    return {
        'DeepCRAMVAE': DeepCRAMVAE,
        'PropertyConditionedTransformerVAE': PropertyConditionedTransformerVAE,
        'MultimodalTransformerVAE': MultimodalTransformerVAE,
        'TransformerVAE': TransformerVAE,
    }


