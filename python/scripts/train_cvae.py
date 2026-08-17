#!/usr/bin/env python3
"""Train PropertyConditionedCRAMVAE (notebook-aligned).

z ∈ ℝ⁶⁴; concat(z, c)=96; GRU 256×2; c ∈ ℝ³².
Writes a full checkpoint (vocab, history, n_cycles, temperature) plus
{stem}_meta.json with vocab_size beside the .pt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import create_dataloaders
from src.models import create_cvae_model
from src.tokenizer import SELFIESTokenizer
from src.utils import get_device


def _default_properties(batch_size: int, device: torch.device) -> dict:
    return {
        "n_cooh": torch.full((batch_size,), 2, dtype=torch.long, device=device),
        "ring_class": torch.full((batch_size,), 3, dtype=torch.long, device=device),
        "aliphatic_pct": torch.full((batch_size,), 0.7, device=device),
        "mw": torch.full((batch_size,), 0.4, device=device),
        "has_benzene": torch.zeros(batch_size, device=device),
    }


def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def cyclical_beta(epoch: int, num_epochs: int, n_cycles: int, beta_min: float, beta_max: float) -> float:
    """Same schedule as the notebook IntegratedCRAMTrainer (linear rise, then hold)."""
    if n_cycles <= 0 or num_epochs <= 0:
        return beta_max
    cycle_length = max(1, num_epochs // n_cycles)
    pos = (epoch - 1) % cycle_length
    ratio = 0.5
    if pos < cycle_length * ratio:
        return beta_min + (beta_max - beta_min) * (pos / max(1, cycle_length * ratio))
    return beta_max


def _load_smiles(csv_path: Path, column: str | None, max_rows: int | None) -> list[str]:
    df = pd.read_csv(csv_path, nrows=max_rows)
    if column and column in df.columns:
        col = column
    else:
        for candidate in ("CanonicalSMILES", "smiles", "SMILES", "canonical_smiles"):
            if candidate in df.columns:
                col = candidate
                break
        else:
            raise ValueError(f"No SMILES column in {csv_path}. Columns: {list(df.columns)}")
    smiles = [str(s) for s in df[col].dropna().tolist() if str(s).strip()]
    if not smiles:
        raise ValueError(f"No SMILES found in column '{col}'")
    return smiles


def _token_accuracy(logits: torch.Tensor, tokens: torch.Tensor, pad_idx: int) -> float:
    pred = logits[:, :-1].argmax(dim=-1)
    tgt = tokens[:, 1:]
    mask = tgt != pad_idx
    if not mask.any():
        return float("nan")
    return float((pred[mask] == tgt[mask]).float().mean().item())


def save_full_checkpoint(
    path: Path,
    model,
    tokenizer: SELFIESTokenizer,
    args: argparse.Namespace,
    optimizer=None,
    epoch=None,
    val_loss=None,
    history=None,
    extra=None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "selfies_vocab": tokenizer.token_to_idx,
        "vocab": tokenizer.token_to_idx,
        "token_to_idx": tokenizer.token_to_idx,
        "idx_to_token": tokenizer.idx_to_token,
        "latent_dim": args.latent_dim,
        "condition_dim": args.condition_dim,
        "hidden_dim": args.hidden_dim,
        "max_len": args.max_len,
        "n_cycles": args.n_cycles,
        "kl_cycles": args.n_cycles,
        "decoder_temperature": args.temperature,
        "architecture": "PropertyConditionedCRAMVAE z64 concat(z,c)=96 GRU 256x2 c32",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if val_loss is not None:
        payload["val_loss"] = val_loss
        payload["loss"] = val_loss
    if history is not None:
        payload["history"] = history
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    meta = {
        "checkpoint": str(path),
        "vocab_size": tokenizer.vocab_size,
        "latent_dim": args.latent_dim,
        "condition_dim": args.condition_dim,
        "hidden_dim": args.hidden_dim,
        "max_len": args.max_len,
        "epoch": epoch,
        "val_loss": val_loss,
        "n_kl_cycles": args.n_cycles,
        "decoder_temperature": args.temperature,
        "note": "VAE z is R^64; concat(z,c)=96 GRU. latent_vectors_128d.npz is Morgan residual PCA, not VAE z.",
    }
    path.with_name(path.stem + "_meta.json").write_text(
        json.dumps({k: v for k, v in meta.items() if v is not None}, indent=2)
    )
    path.with_name(path.stem + "_vocab.json").write_text(
        json.dumps({"vocab_size": tokenizer.vocab_size, "token_to_idx": tokenizer.token_to_idx, "max_len": args.max_len}, indent=2)
    )
    return path


def train(args: argparse.Namespace) -> None:
    device = get_device()
    if device.type == "mps":
        print("DGL training is more stable on CPU than MPS; using CPU.")
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles = _load_smiles(Path(args.smiles_csv), args.smiles_column, args.max_rows)
    n_val = max(1, int(len(smiles) * args.val_fraction))
    val_smiles = smiles[:n_val]
    train_smiles = smiles[n_val:] or smiles

    tokenizer = SELFIESTokenizer(max_len=args.max_len)
    tokenizer.build_vocab(train_smiles + val_smiles)

    train_loader, val_loader = create_dataloaders(
        train_smiles,
        val_smiles,
        tokenizer,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augment,
    )

    model = create_cvae_model(
        vocab_size=tokenizer.vocab_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        condition_dim=args.condition_dim,
        max_len=args.max_len,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_val = float("inf")
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "betas": [],
        "token_accuracy": [],
    }

    print(f"Device: {device}")
    print(f"Train/val: {len(train_smiles)}/{len(val_smiles)}")
    print(f"Vocab: {tokenizer.vocab_size}  latent_dim: {args.latent_dim}  concat(z,c)={args.latent_dim + args.condition_dim}")
    print(f"KL cycles: {args.n_cycles}  β∈[{args.beta_min}, {args.beta_max}]")

    for epoch in range(1, args.epochs + 1):
        beta = cyclical_beta(epoch, args.epochs, args.n_cycles, args.beta_min, args.beta_max)
        model.train()
        train_loss = train_recon = train_kl = train_acc = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            graph = batch["graph"].to(device)
            tokens = batch["tokens"].to(device)
            node_feats = graph.ndata["feat"]
            properties = _default_properties(tokens.size(0), device)

            logits, mu, logvar = model(graph, node_feats, properties, tokens)
            recon = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_idx,
            )
            kl = _kl(mu, logvar)
            loss = recon + beta * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
            train_acc += _token_accuracy(logits, tokens, tokenizer.pad_idx)
            n_batches += 1

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                graph = batch["graph"].to(device)
                tokens = batch["tokens"].to(device)
                node_feats = graph.ndata["feat"]
                properties = _default_properties(tokens.size(0), device)
                logits, mu, logvar = model(graph, node_feats, properties, tokens)
                recon = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    ignore_index=tokenizer.pad_idx,
                )
                val_loss += (recon + beta * _kl(mu, logvar)).item()
                n_val_batches += 1

        n_batches = max(n_batches, 1)
        n_val_batches = max(n_val_batches, 1)
        train_avg = train_loss / n_batches
        val_avg = val_loss / n_val_batches
        history["train_loss"].append(train_avg)
        history["train_recon"].append(train_recon / n_batches)
        history["train_kl"].append(train_kl / n_batches)
        history["val_loss"].append(val_avg)
        history["betas"].append(beta)
        history["token_accuracy"].append(train_acc / n_batches)
        print(
            f"Epoch {epoch:03d}  train={train_avg:.4f}  val={val_avg:.4f}  "
            f"recon={history['train_recon'][-1]:.4f}  kl={history['train_kl'][-1]:.4f}  "
            f"β={beta:.4f}  acc={history['token_accuracy'][-1]:.4f}"
        )

        save_full_checkpoint(
            output_dir / "last.pt",
            model,
            tokenizer,
            args,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_avg,
            history=history,
        )
        if val_avg < best_val:
            best_val = val_avg
            save_full_checkpoint(
                output_dir / "cvae_best.pt",
                model,
                tokenizer,
                args,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=best_val,
                history=history,
            )
            print(f"  saved best checkpoint (val={best_val:.4f}) vocab_size={tokenizer.vocab_size}")

    config_path = output_dir / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "latent_dim": args.latent_dim,
                "condition_dim": args.condition_dim,
                "hidden_dim": args.hidden_dim,
                "in_dim": 163,
                "vocab_size": tokenizer.vocab_size,
                "max_len": args.max_len,
                "n_cycles": args.n_cycles,
                "decoder_temperature": args.temperature,
                "architecture": "PropertyConditionedCRAMVAE z64 concat(z,c)=96 GRU 256x2 c32",
            },
            indent=2,
        )
    )
    print(f"Wrote {output_dir / 'cvae_best.pt'} and {config_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CRAM GRU CVAE (latent_dim=64, concat(z,c)=96)")
    p.add_argument("--smiles-csv", required=True, help="CSV with SMILES column")
    p.add_argument("--smiles-column", default=None, help="Column name (auto-detect if omitted)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=None, help="Unused if cyclical β is on; kept for compatibility")
    p.add_argument("--beta-min", type=float, default=0.00005)
    p.add_argument("--beta-max", type=float, default=0.003)
    p.add_argument("--n-cycles", type=int, default=4, help="KL annealing cycles (notebook default 4)")
    p.add_argument("--temperature", type=float, default=0.8, help="Decoder sampling temperature stored with the checkpoint")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--condition-dim", type=int, default=32)
    p.add_argument("--max-len", type=int, default=100)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--output-dir", default="models/cvae_run")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
