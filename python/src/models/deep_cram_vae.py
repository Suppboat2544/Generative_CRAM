"""
DeepCRAMVAE decoder used by best_cram_model.pt.

The production checkpoint is a Transformer decoder VAE (notebook DeepCRAMVAE),
not the GRU PropertyConditionedCRAMVAE. Generation samples z ~ N(0, I) and
does not need DGL or the graph encoder.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import selfies as sf
from rdkit import Chem


class DeepCRAMDecoderVAE(nn.Module):
    """Transformer decoder half of DeepCRAMVAE (latent_dim=64, hidden=256)."""

    def __init__(
        self,
        vocab_size: int = 157,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        max_len: int = 150,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.decoder_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_pos_encoding = nn.Parameter(torch.randn(max_len, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    @torch.no_grad()
    def generate_tokens(
        self,
        n: int = 8,
        max_length: int = 100,
        temperature: float = 1.0,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        max_length = min(max_length, self.max_len)
        z = torch.randn(n, self.latent_dim, device=device)
        memory = self.latent_proj(z).unsqueeze(1)
        tokens = torch.full((n, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            tgt = self.decoder_embedding(tokens)
            tgt = tgt + self.decoder_pos_encoding[: tokens.size(1)]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1)).to(device)
            decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(decoded[:, -1, :])
            if temperature and temperature > 0:
                next_token = torch.multinomial(
                    F.softmax(logits / temperature, dim=-1), 1
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_idx)
            if bool(finished.all()):
                break
        return tokens

    def decode_smiles(
        self,
        token_batch: torch.Tensor,
        idx_to_token: Dict[int, str],
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0,
    ) -> List[str]:
        smiles_out: List[str] = []
        skip = {sos_idx, eos_idx, pad_idx}
        for seq in token_batch.cpu().tolist():
            parts = []
            for idx in seq:
                if idx == eos_idx:
                    break
                if idx in skip:
                    continue
                tok = idx_to_token.get(idx) or idx_to_token.get(str(idx), "")
                if not tok or tok.startswith("<"):
                    continue
                parts.append(tok)
            if not parts:
                continue
            try:
                smiles = sf.decoder("".join(parts))
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is not None:
                    smiles_out.append(Chem.MolToSmiles(mol))
            except Exception:
                continue
        return smiles_out

    def sample_smiles(
        self,
        idx_to_token: Dict[int, str],
        n: int = 3,
        temperature: float = 1.0,
        max_length: int = 100,
        batch_size: int = 16,
        max_rounds: int = 20,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0,
    ) -> List[str]:
        unique: List[str] = []
        seen = set()
        for _ in range(max_rounds):
            if len(unique) >= n:
                break
            tokens = self.generate_tokens(
                n=batch_size,
                max_length=max_length,
                temperature=temperature,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
            )
            for smiles in self.decode_smiles(
                tokens, idx_to_token, sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx
            ):
                if smiles not in seen:
                    seen.add(smiles)
                    unique.append(smiles)
                if len(unique) >= n:
                    break
        return unique[:n]


def training_style_vocab(token_set, vocab_size: int) -> Dict[str, int]:
    """Match notebook build_vocab: PAD/SOS/EOS then sorted SELFIES tokens."""
    specials = ["<PAD>", "<SOS>", "<EOS>"]
    tokens = sorted(t for t in token_set if t and not t.startswith("<"))
    try:
        extra = sorted(sf.get_semantic_robust_alphabet())
    except Exception:
        extra = []
    merged = []
    seen = set()
    for tok in tokens + extra:
        if tok not in seen:
            seen.add(tok)
            merged.append(tok)
    vocab_list = specials + merged
    if len(vocab_list) < vocab_size:
        vocab_list.extend(f"<EXTRA_{i}>" for i in range(vocab_size - len(vocab_list)))
    else:
        vocab_list = vocab_list[:vocab_size]
    return {t: i for i, t in enumerate(vocab_list)}
