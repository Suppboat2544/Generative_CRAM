"""
CRAM SELFIES Tokenizer
======================

Tokenization utilities for molecular SELFIES representation.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import selfies as sf
from rdkit import Chem
from tqdm.auto import tqdm


class SELFIESTokenizer:
    """
    SELFIES tokenizer for molecular sequences.
    
    SELFIES (Self-Referencing Embedded Strings) provide 100% valid
    molecule generation by construction.
    """
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_len: int = 150):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Pre-built vocabulary {token: idx}
            max_len: Maximum sequence length
        """
        self.max_len = max_len
        
        if vocab is not None:
            self.token_to_idx = vocab
            self.idx_to_token = {v: k for k, v in vocab.items()}
        else:
            self.token_to_idx = {
                self.PAD_TOKEN: 0,
                self.SOS_TOKEN: 1,
                self.EOS_TOKEN: 2,
                self.UNK_TOKEN: 3,
            }
            self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)
    
    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[self.PAD_TOKEN]
    
    @property
    def sos_idx(self) -> int:
        return self.token_to_idx[self.SOS_TOKEN]
    
    @property
    def eos_idx(self) -> int:
        return self.token_to_idx[self.EOS_TOKEN]
    
    @property
    def unk_idx(self) -> int:
        return self.token_to_idx[self.UNK_TOKEN]
    
    def build_vocab(self, smiles_list: List[str], min_freq: int = 1) -> Dict[str, int]:
        """
        Build vocabulary from SMILES list.
        
        Args:
            smiles_list: List of SMILES strings
            min_freq: Minimum token frequency
            
        Returns:
            Vocabulary dictionary
        """
        all_tokens = set()
        max_selfies_len = 0
        
        for smi in tqdm(smiles_list, desc="Building vocab"):
            try:
                selfies_str = sf.encoder(smi)
                if selfies_str:
                    tokens = list(sf.split_selfies(selfies_str))
                    all_tokens.update(tokens)
                    max_selfies_len = max(max_selfies_len, len(tokens))
            except:
                continue
        
        # Build vocabulary
        vocab_list = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        vocab_list += sorted(list(all_tokens))
        
        self.token_to_idx = {t: i for i, t in enumerate(vocab_list)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        
        # Update max_len if needed
        self.max_len = min(max_selfies_len + 2, self.max_len)  # +2 for SOS/EOS
        
        return self.token_to_idx
    
    def encode(self, smiles: str) -> List[int]:
        """
        Encode SMILES to token indices.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of token indices
        """
        try:
            selfies_str = sf.encoder(smiles)
            if selfies_str is None:
                return [self.sos_idx, self.eos_idx]
            
            tokens = [self.SOS_TOKEN] + list(sf.split_selfies(selfies_str)) + [self.EOS_TOKEN]
            indices = [self.token_to_idx.get(t, self.unk_idx) for t in tokens]
            
            # Pad or truncate
            if len(indices) > self.max_len:
                indices = indices[:self.max_len]
            else:
                indices = indices + [self.pad_idx] * (self.max_len - len(indices))
            
            return indices
            
        except:
            return [self.pad_idx] * self.max_len
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode token indices to SMILES.
        
        Args:
            indices: List of token indices
            remove_special: Remove special tokens
            
        Returns:
            SMILES string (or empty if invalid)
        """
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, self.UNK_TOKEN)
            
            if remove_special:
                if token in [self.PAD_TOKEN, self.SOS_TOKEN, self.UNK_TOKEN]:
                    continue
                if token == self.EOS_TOKEN:
                    break
            
            tokens.append(token)
        
        selfies_str = ''.join(tokens)
        
        try:
            smiles = sf.decoder(selfies_str)
            # Validate
            if smiles and Chem.MolFromSmiles(smiles):
                return smiles
            return ""
        except:
            return ""
    
    def batch_encode(self, smiles_list: List[str]) -> np.ndarray:
        """Encode batch of SMILES."""
        return np.array([self.encode(s) for s in smiles_list], dtype=np.int64)
    
    def batch_decode(self, indices: np.ndarray) -> List[str]:
        """Decode batch of token indices."""
        return [self.decode(seq.tolist()) for seq in indices]


class BeamSearchDecoder:
    """
    Beam search decoder for improved molecule generation.
    
    Uses beam search to find higher-quality sequences during
    autoregressive decoding.
    """
    
    def __init__(
        self,
        tokenizer: SELFIESTokenizer,
        beam_width: int = 5,
        max_len: int = 100,
        length_penalty: float = 0.7
    ):
        """
        Initialize beam search decoder.
        
        Args:
            tokenizer: SELFIES tokenizer
            beam_width: Number of beams to keep
            max_len: Maximum sequence length
            length_penalty: Penalty for sequence length
        """
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_len = max_len
        self.length_penalty = length_penalty
    
    def decode(
        self,
        model,
        z: 'torch.Tensor',
        device: 'torch.device'
    ) -> List[Tuple[str, float]]:
        """
        Perform beam search decoding.
        
        Args:
            model: DeepCRAMVAE model
            z: Latent vector [latent_dim]
            device: torch device
            
        Returns:
            List of (smiles, score) tuples
        """
        import torch
        
        # Initialize beams: (sequence, log_prob)
        beams = [([self.tokenizer.sos_idx], 0.0)]
        completed = []
        
        for step in range(self.max_len):
            all_candidates = []
            
            for seq, log_prob in beams:
                if seq[-1] == self.tokenizer.eos_idx:
                    completed.append((seq, log_prob))
                    continue
                
                # Get next token probabilities
                seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    memory = model.latent_proj(z.unsqueeze(0)).unsqueeze(1)
                    tgt = model.decoder_emb(seq_tensor) + model.decoder_pos[:len(seq)]
                    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)
                    decoded = model.decoder(tgt, memory, tgt_mask=tgt_mask)
                    logits = model.output_proj(decoded[0, -1])
                    log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get top-k next tokens
                top_log_probs, top_indices = log_probs.topk(self.beam_width)
                
                for i in range(self.beam_width):
                    new_seq = seq + [top_indices[i].item()]
                    new_log_prob = log_prob + top_log_probs[i].item()
                    all_candidates.append((new_seq, new_log_prob))
            
            if not all_candidates:
                break
            
            # Select top beams
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:self.beam_width]
        
        # Add incomplete beams
        completed.extend(beams)
        
        # Decode to SMILES and apply length penalty
        results = []
        for seq, log_prob in completed:
            smiles = self.tokenizer.decode(seq)
            if smiles:
                # Length-normalized score
                score = log_prob / (len(seq) ** self.length_penalty)
                results.append((smiles, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


def build_vocab_from_smiles(smiles_list: List[str], max_len: int = 150) -> Tuple[Dict[str, int], int]:
    """
    Build SELFIES vocabulary from SMILES list.
    
    Args:
        smiles_list: List of SMILES strings
        max_len: Maximum sequence length
        
    Returns:
        Tuple of (vocabulary dict, actual max length)
    """
    tokenizer = SELFIESTokenizer(max_len=max_len)
    vocab = tokenizer.build_vocab(smiles_list)
    return vocab, tokenizer.max_len
