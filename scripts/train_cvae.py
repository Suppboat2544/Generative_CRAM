#!/usr/bin/env python3
"""
Train PropertyConditionedCRAMVAE on a SMILES CSV.

Latent dimension defaults to 64 to match the trained production checkpoint.
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


def _load_smiles(csv_path: Path, column: str | None) -> list[str]:
    df = pd.read_csv(csv_path)
    if column and column in df.columns:
        col = column
    else:
        for candidate in ("CanonicalSMILES", "smiles", "SMILES", "canonical_smiles"):
            if candidate in df.columns:
                col = candidate
                break
        else:
            raise ValueError(
                f"No SMILES column in {csv_path}. Columns: {list(df.columns)}"
            )
    smiles = [str(s) for s in df[col].dropna().tolist() if str(s).strip()]
    if not smiles:
        raise ValueError(f"No SMILES found in column '{col}'")
    return smiles


def train(args: argparse.Namespace) -> None:
    device = get_device()
    if device.type == "mps":
        print("DGL training is more stable on CPU than MPS; using CPU.")
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles = _load_smiles(Path(args.smiles_csv), args.smiles_column)
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

    print(f"Device: {device}")
    print(f"Train/val: {len(train_smiles)}/{len(val_smiles)}")
    print(f"Vocab: {tokenizer.vocab_size}  latent_dim: {args.latent_dim}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
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
            loss = recon + args.kl_weight * _kl(mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
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
                val_loss += (recon + args.kl_weight * _kl(mu, logvar)).item()
                n_val_batches += 1

        train_avg = train_loss / max(n_batches, 1)
        val_avg = val_loss / max(n_val_batches, 1)
        print(f"Epoch {epoch:03d}  train={train_avg:.4f}  val={val_avg:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "selfies_vocab": tokenizer.token_to_idx,
            "idx_to_token": tokenizer.idx_to_token,
            "latent_dim": args.latent_dim,
            "condition_dim": args.condition_dim,
            "hidden_dim": args.hidden_dim,
            "max_len": args.max_len,
            "loss": val_avg,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(ckpt, output_dir / "cvae_best.pt")
            print(f"  saved best checkpoint (val={best_val:.4f})")

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
            },
            indent=2,
        )
    )
    print(f"Wrote {output_dir / 'cvae_best.pt'} and {config_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CRAM CVAE (latent_dim=64)")
    p.add_argument("--smiles-csv", required=True, help="CSV with SMILES column")
    p.add_argument("--smiles-column", default=None, help="Column name (auto-detect if omitted)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=0.1)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--condition-dim", type=int, default=32)
    p.add_argument("--max-len", type=int, default=100)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--output-dir", default="models/cvae_run")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
