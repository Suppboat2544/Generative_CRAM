#!/usr/bin/env python3
"""Paper blocking criticals A1–A3 (honest artifacts only).

Implemented VAE: z ∈ ℝ⁶⁴; concat(z, c)=96; GRU 256×2; c ∈ ℝ³².
latent_dim=128 and latent_vectors_128d.npz are Morgan residual PCA, not VAE z.

Does not restore unpublished percentages (72.3 / 69.7 / 87.5), dual persistence
66–68%, carboxyl shielding, ANOVA F=18.4, or KS “indistinguishable”.
Table 1 A is n=10,000 only. Headline EPR stays TRFU 31.8 vs 38.0 (n=18,513)
until a per-ppm table exists. 10⁷×0.576×0.700 is not assumed to equal 5.04M.

Deliverables (under --out-dir):
  A1  vae_z64_mu_logvar.npz, dual_z_alignment.csv
  A2  kl_per_dim.csv, collapse_summary.json, training_curve.csv (if logs exist)
  A3  production_funnel.csv, checkpoint_meta.json (vocab_size beside the .pt)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import smiles_to_graph  # noqa: E402
from src.models import create_cvae_model  # noqa: E402
from src.tokenizer import SELFIESTokenizer  # noqa: E402
from src.utils import get_device  # noqa: E402
from src.validator import CRAMValidator  # noqa: E402

try:
    import dgl
except ImportError as e:  # pragma: no cover
    raise SystemExit("DGL is required") from e

try:
    import selfies as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit("selfies is required") from e

ACTIVE_KL_NATS = 0.01
ENV_OC = (0.40, 0.60)
ENV_HC = (1.20, 1.70)
PAPER_CONSTRAINTS = {
    "vae_z": "R^64",
    "concat_z_c": 96,
    "gru": "256x2",
    "c": "R^32",
    "latent_vectors_128d": "Morgan residual PCA, NOT VAE z",
    "table_1_a_n": 10000,
    "do_not_restore": [
        "72.3",
        "69.7",
        "87.5",
        "dual persistence 66-68%",
        "carboxyl shielding",
        "ANOVA F=18.4",
        "KS indistinguishable",
    ],
    "headline_epr": "TRFU 31.8 vs 38.0 (n=18513) until a per-ppm table exists",
    "sirius": "SIRIUS 5 / CSI:FingerID 5; MS1 in {1,3,5,10} ppm with MS2 matched; adducts [M+H]+ / [M-H]- by polarity",
    "metfrag": "MetFrag 2/18 and 0/36 remain native coverage; identifiers must be recovered, not guessed",
}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _first_existing(paths) -> Path | None:
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.is_file():
            return p
    return None


def _search_named(name: str, roots: list[Path]) -> Path | None:
    """Look for `name` at root and two directory levels down. No Drive-wide rglob."""
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            return direct
        for pat in (name, f"*/{name}", f"*/*/{name}"):
            for hit in root.glob(pat):
                if hit.is_file():
                    return hit
    return None


def load_smiles_list(csv_path: Path, column: str | None, max_rows: int | None) -> list[str]:
    df = pd.read_csv(csv_path, nrows=max_rows)
    if column and column in df.columns:
        col = column
    else:
        col = next(
            (c for c in ("CanonicalSMILES", "SMILES", "smiles", "canonical_smiles") if c in df.columns),
            None,
        )
    if col is None:
        raise ValueError(f"No SMILES column in {csv_path}: {list(df.columns)}")
    out = []
    seen = set()
    for s in df[col].dropna().astype(str):
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def default_properties(batch: int, device: torch.device) -> dict:
    return {
        "n_cooh": torch.full((batch,), 2, dtype=torch.long, device=device),
        "ring_class": torch.full((batch,), 3, dtype=torch.long, device=device),
        "aliphatic_pct": torch.full((batch,), 0.7, device=device),
        "mw": torch.full((batch,), 0.4, device=device),
        "has_benzene": torch.zeros(batch, device=device),
    }


def _as_int_keyed(d) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def load_cvae(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        ckpt = {"model_state_dict": ckpt}
    state = ckpt.get("model_state_dict", ckpt)
    if "decoder_embedding.weight" in state:
        raise RuntimeError(
            f"{ckpt_path} is the Transformer DeepCRAMVAE, not concat(z,c)=96 GRU. "
            "Use the PropertyConditionedCRAMVAE checkpoint (decoder.gru / latent_condition_proj)."
        )
    if "decoder.gru.weight_ih_l0" not in state and "decoder.latent_condition_proj.weight" not in state:
        print(f"WARNING: {ckpt_path} may not be concat(z,c)=96 GRU (missing decoder.gru / latent_condition_proj)")
    vocab = ckpt.get("selfies_vocab") or ckpt.get("char2idx") or ckpt.get("vocab") or ckpt.get("token_to_idx")
    idx_to_token = ckpt.get("idx_to_token") or ckpt.get("idx2char")
    if idx_to_token is None and vocab:
        idx_to_token = {int(v): k for k, v in vocab.items()}
    else:
        idx_to_token = _as_int_keyed(idx_to_token)
    vocab_size = int(ckpt.get("vocab_size") or (len(vocab) if vocab else 0) or state["decoder.output_proj.weight"].shape[0])
    latent_dim = int(ckpt.get("latent_dim") or 64)
    if "encoder.fc_mu.2.weight" in state:
        latent_dim = int(state["encoder.fc_mu.2.weight"].shape[0])
    elif "encoder.fc_mu.1.weight" in state:
        latent_dim = int(state["encoder.fc_mu.1.weight"].shape[0])
    elif "encoder.fc_mu.weight" in state:
        latent_dim = int(state["encoder.fc_mu.weight"].shape[0])
    if latent_dim != 64:
        print(f"WARNING: checkpoint latent_dim={latent_dim}, paper VAE z is R^64")
    hidden_dim = int(ckpt.get("hidden_dim") or 256)
    condition_dim = int(ckpt.get("condition_dim") or 32)
    max_len = int(ckpt.get("max_len") or 100)
    model = create_cvae_model(
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        condition_dim=condition_dim,
        max_len=max_len,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"load_state_dict missing ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"load_state_dict unexpected ({len(unexpected)}): {unexpected[:8]}")
    model = model.to(device).eval()
    tok = None
    if vocab:
        vocab = dict(vocab)
        if "<UNK>" not in vocab and "UNK" not in vocab:
            vocab["<UNK>"] = vocab.get("<PAD>", 0)
        tok = SELFIESTokenizer(vocab=vocab, max_len=max_len)
    meta = {
        "checkpoint": str(ckpt_path),
        "vocab_size": vocab_size,
        "latent_dim": latent_dim,
        "condition_dim": condition_dim,
        "hidden_dim": hidden_dim,
        "max_len": max_len,
        "epoch": ckpt.get("epoch"),
        "n_cycles": ckpt.get("n_cycles") or ckpt.get("kl_cycles"),
        "decoder_temperature": ckpt.get("temperature") or ckpt.get("decoder_temperature"),
        "history_keys": sorted(ckpt["history"].keys()) if isinstance(ckpt.get("history"), dict) else [],
        "architecture": "PropertyConditionedCRAMVAE z64 concat(z,c)=96 GRU 256x2 c32",
        "has_decoder_gru": "decoder.gru.weight_ih_l0" in state,
        "has_latent_condition_proj": "decoder.latent_condition_proj.weight" in state,
    }
    return model, tok, idx_to_token or {}, ckpt, meta


@torch.no_grad()
def encode_smiles(model, smiles_list, device, properties_fn=default_properties):
    mus, logvars, conditions, ok_smiles, fail = [], [], [], [], []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is None:
            fail.append(smi)
            continue
        g = g.to(device)
        node_feats = g.ndata["feat"]
        props = properties_fn(1, device)
        try:
            z, mu, logvar, cond = model.encode(g, node_feats, props)
        except Exception:
            fail.append(smi)
            continue
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
            logvar = logvar.unsqueeze(0)
            cond = cond.unsqueeze(0) if cond.dim() == 1 else cond
        mus.append(mu.float().cpu())
        logvars.append(logvar.float().cpu())
        conditions.append(cond.float().cpu())
        ok_smiles.append(smi)
    if not mus:
        raise RuntimeError("No graphs encoded")
    return {
        "mu": torch.cat(mus, 0).numpy(),
        "logvar": torch.cat(logvars, 0).numpy(),
        "condition": torch.cat(conditions, 0).numpy(),
        "smiles": np.array(ok_smiles, dtype=object),
        "failed": fail,
    }


def kl_per_dim(mu, logvar):
    logvar = np.clip(logvar, -10.0, 10.0)
    return -0.5 * (1.0 + logvar - mu ** 2 - np.exp(logvar))


def twonn_dint(x: np.ndarray) -> float:
    n = x.shape[0]
    if n < 8:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(x)
    dist, _ = nn.kneighbors(x)
    r1 = np.maximum(dist[:, 1], 1e-12)
    r2 = dist[:, 2]
    mu = np.maximum(r2 / r1, 1.0 + 1e-12)
    mu = np.sort(mu)
    keep = max(8, int(0.99 * n))
    return float(1.0 / np.mean(np.log(mu[:keep])))


def n_pcs_95(x: np.ndarray) -> int:
    pca = PCA().fit(x)
    c = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(c, 0.95) + 1)


def rv_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean(0, keepdims=True)
    yc = y - y.mean(0, keepdims=True)
    xty = xc.T @ yc
    xtx = xc.T @ xc
    yty = yc.T @ yc
    num = float(np.sum(xty ** 2))
    den = float(np.sqrt(np.sum(xtx ** 2) * np.sum(yty ** 2)))
    return num / den if den > 0 else float("nan")


def procrustes_disparity(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.spatial import procrustes

    if x.shape[0] != y.shape[0]:
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n], y[:n]
    if x.shape[1] != y.shape[1]:
        # Same-n Procrustes needs equal columns: project the wider matrix onto
        # x.shape[1] PCs (μ is 64-D; deposited PCA is 128-D).
        k = min(x.shape[1], y.shape[1], x.shape[0] - 1)
        if y.shape[1] > x.shape[1]:
            y = PCA(n_components=k, random_state=0).fit_transform(y)
            x = x[:, :k]
        else:
            x = PCA(n_components=k, random_state=0).fit_transform(x)
            y = y[:, :k]
    _, _, d = procrustes(x, y)
    return float(d)


def load_smiles_file(path: Path, max_n: int | None = None) -> list[str]:
    text = path.read_text(errors="ignore").strip()
    if path.suffix.lower() == ".csv":
        return load_smiles_list(path, None, max_n)
    out = []
    seen = set()
    for line in text.splitlines():
        s = line.strip().split(",")[0].strip()
        if not s or s.lower() in {"smiles", "canonicalsmiles"} or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if max_n and len(out) >= max_n:
            break
    return out


def discover_logs(roots: list[Path], extra: list[Path] | None = None) -> list[Path]:
    found: list[Path] = []
    seen = set()
    for p in extra or []:
        if p and Path(p).is_file() and p not in seen:
            found.append(Path(p))
            seen.add(Path(p))
    names = (
        "production_10M.log",
        "generation_10M.log",
        "funnel.log",
        "table1a.log",
        "table1_a.log",
        "generation_10k.log",
    )
    patterns = ("*10M*.log", "*10m*.log", "*funnel*.log", "*production*.log", "*10k*.log")
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            hit = _search_named(name, [root])
            if hit and hit not in seen:
                found.append(hit)
                seen.add(hit)
        for pat in patterns:
            for hit in root.glob(pat):
                if hit.is_file() and hit not in seen:
                    found.append(hit)
                    seen.add(hit)
            # one level of subdirs only — avoid Drive-wide rglob
            for hit in root.glob(f"*/{pat}"):
                if hit.is_file() and hit not in seen:
                    found.append(hit)
                    seen.add(hit)
    return found


def load_history_sidecar(ckpt_path: Path, ckpt: dict) -> dict | None:
    hist = ckpt.get("history") if isinstance(ckpt, dict) else None
    if isinstance(hist, dict) and (hist.get("train_recon") or hist.get("train_loss")):
        return hist
    for p in (
        ckpt_path.with_name(ckpt_path.stem + "_history.json"),
        ckpt_path.with_name("training_curve.csv"),
        ckpt_path.parent / "history.json",
        ckpt_path.parent / "training_history.json",
    ):
        if not p.is_file():
            continue
        if p.suffix == ".json":
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        if p.suffix == ".csv":
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            hist = {}
            for dst, aliases in (
                ("train_recon", ("recon", "train_recon", "reconstruction")),
                ("train_kl", ("kl", "train_kl")),
                ("betas", ("beta", "betas", "kl_beta")),
                ("token_accuracy", ("token_accuracy", "train_acc", "acc")),
            ):
                for a in aliases:
                    if a in cols:
                        hist[dst] = df[cols[a]].tolist()
                        break
            if hist:
                return hist
    return None


def tokens_to_selfies(indices, idx_to_token) -> str:
    parts = []
    for i in indices:
        tok = idx_to_token.get(int(i), "<PAD>")
        if tok in ("<EOS>", "[end]", "EOS"):
            break
        if tok in ("<PAD>", "<SOS>", "[start]", "[nop]", "PAD", "SOS", "UNK", "<UNK>"):
            continue
        parts.append(tok)
    return "".join(parts)


@torch.no_grad()
def valid_decode_along_pc1(model, mu, condition, idx_to_token, n_steps=21, scale=2.0, temperature=0.0):
    pca = PCA(n_components=1, random_state=0).fit(mu)
    direction = pca.components_[0]
    center = mu.mean(axis=0)
    device = next(model.parameters()).device
    n_ok = 0
    n_tot = 0
    cond = torch.as_tensor(condition.mean(axis=0, keepdims=True), device=device, dtype=torch.float32)
    for t in np.linspace(-scale, scale, n_steps):
        z = torch.as_tensor((center + t * direction)[None, :], device=device, dtype=torch.float32)
        tokens = model.decode(z, cond.expand(z.size(0), -1), temperature=temperature)
        sfi = tokens_to_selfies(tokens[0].tolist(), idx_to_token)
        n_tot += 1
        try:
            smi = sf.decoder(sfi)
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                n_ok += 1
        except Exception:
            pass
    return n_ok / n_tot if n_tot else float("nan"), n_ok, n_tot


@torch.no_grad()
def reconstruction_nlls(model, smiles_list, tokenizer, device, max_n=256):
    """Teacher-forced NLL under z~q(z|x) vs z=0 (c kept)."""
    nll_q, nll_c, acc_q, n = [], [], [], 0
    for smi in smiles_list[:max_n]:
        g = smiles_to_graph(smi)
        if g is None or tokenizer is None:
            continue
        try:
            tokens = tokenizer.encode(smi)
        except Exception:
            continue
        if tokens is None:
            continue
        tokens = torch.as_tensor(tokens, dtype=torch.long).view(1, -1).to(device)
        pad_idx = int(getattr(tokenizer, "pad_idx", 0))
        g = g.to(device)
        props = default_properties(1, device)
        logits_q, mu, logvar = model(g, g.ndata["feat"], props, tokens)
        z0 = torch.zeros_like(mu)
        _, _, _, cond = model.encode(g, g.ndata["feat"], props)
        logits_c = model.decode(z0, cond, tokens)
        tgt = tokens[:, 1:]
        pred_q = logits_q[:, :-1]
        pred_c = logits_c[:, :-1]
        nll_q.append(F.cross_entropy(pred_q.reshape(-1, pred_q.size(-1)), tgt.reshape(-1), ignore_index=pad_idx).item())
        nll_c.append(F.cross_entropy(pred_c.reshape(-1, pred_c.size(-1)), tgt.reshape(-1), ignore_index=pad_idx).item())
        with torch.no_grad():
            hat = pred_q.argmax(-1)
            mask = tgt != pad_idx
            acc_q.append(float((hat[mask] == tgt[mask]).float().mean().item()) if mask.any() else float("nan"))
        n += 1
    if not nll_q:
        return {"nll_qzx": None, "nll_c_only": None, "token_accuracy_q": None, "n": 0}
    return {
        "nll_qzx": float(np.mean(nll_q)),
        "nll_c_only": float(np.mean(nll_c)),
        "token_accuracy_q": float(np.nanmean(acc_q)),
        "n": n,
    }


def elemental_ratios(mol) -> tuple[float, float] | None:
    if mol is None:
        return None
    n_c = n_h = n_o = 0
    try:
        mol_h = Chem.AddHs(mol)
    except Exception:
        mol_h = mol
    for a in mol_h.GetAtoms():
        z = a.GetAtomicNum()
        if z == 6:
            n_c += 1
        elif z == 1:
            n_h += 1
        elif z == 8:
            n_o += 1
    if n_c == 0:
        return None
    return n_o / n_c, n_h / n_c


def env_plausible(mol) -> bool:
    r = elemental_ratios(mol)
    if r is None:
        return False
    oc, hc = r
    return ENV_OC[0] <= oc <= ENV_OC[1] and ENV_HC[0] <= hc <= ENV_HC[1]


def load_npz_128(path: Path):
    data = np.load(path, allow_pickle=True)
    keys = list(data.files)
    vec = None
    for k in ("z", "latent", "X", "pca", "vectors", "arr_0"):
        if k in data.files and np.asarray(data[k]).ndim == 2 and np.asarray(data[k]).shape[1] == 128:
            vec = np.asarray(data[k], dtype=np.float32)
            break
    if vec is None:
        for k in keys:
            a = np.asarray(data[k])
            if a.ndim == 2 and a.shape[1] == 128:
                vec = a.astype(np.float32)
                break
    smiles = None
    for k in ("smiles", "SMILES", "index", "ids"):
        if k in data.files:
            smiles = np.asarray(data[k]).astype(str)
            break
    return vec, smiles, keys


def write_training_curve(ckpt: dict, out: Path, ckpt_path: Path | None = None) -> dict:
    hist = load_history_sidecar(ckpt_path, ckpt) if ckpt_path else (
        ckpt.get("history") if isinstance(ckpt, dict) else None
    )
    rows = []
    note = "no training history in checkpoint or sidecar"
    epoch_178 = {"present": False, "kl_beta": None, "matches_0.0498": None}
    recon = None
    if isinstance(hist, dict):
        recon = hist.get("train_recon") or hist.get("recon")
    if isinstance(hist, dict) and recon:
        n = len(recon)
        betas = hist.get("betas") or hist.get("beta") or hist.get("kl_beta") or [None] * n
        kl = hist.get("train_kl") or hist.get("kl") or [None] * n
        acc = hist.get("token_accuracy") or hist.get("train_acc") or hist.get("acc") or [None] * n
        for i in range(n):
            rows.append(
                {
                    "epoch": i + 1,
                    "recon": recon[i],
                    "kl": kl[i] if i < len(kl) else None,
                    "beta": betas[i] if i < len(betas) else None,
                    "token_accuracy": acc[i] if i < len(acc) else None,
                }
            )
        note = f"from checkpoint/sidecar history ({n} epochs)"
        if n >= 178:
            b = rows[177].get("beta")
            epoch_178 = {
                "present": True,
                "kl_beta": b,
                "matches_0.0498": (abs(float(b) - 0.0498) < 1e-4) if b is not None else None,
            }
        else:
            note += "; epoch 178 not present — do not confirm kl_beta=0.0498"
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "recon", "kl", "beta", "token_accuracy"])
            w.writeheader()
            w.writerows(rows)
    else:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "recon", "kl", "beta", "token_accuracy", "note"])
            w.writeheader()
            w.writerow({"epoch": "", "recon": "", "kl": "", "beta": "", "token_accuracy": "", "note": note})
    return {"n_rows": len(rows), "note": note, "epoch_178": epoch_178}


def parse_funnel_from_logs(log_paths: list[Path]) -> dict | None:
    """Recover integer funnel counts from logs. Never invent 72.3/69.7/87.5."""
    keys = (
        "sampled from prior",
        "selfies-valid",
        "rdkit-sanitized",
        "strict cram",
        "env-plausible",
        "unique inchikeys",
        "unique murcko scaffolds",
    )
    found = {k: None for k in keys}
    text = ""
    for p in log_paths:
        try:
            text += "\n" + p.read_text(errors="ignore")
        except Exception:
            continue
    if not text.strip():
        return None
    import re

    patterns = {
        "sampled from prior": r"sampled(?: from prior)?[^\d]{0,40}(\d[\d,]*)",
        "selfies-valid": r"selfies[- ]valid[^\d]{0,40}(\d[\d,]*)",
        "rdkit-sanitized": r"(?:rdkit[- ])?sanitiz[^\d]{0,40}(\d[\d,]*)",
        "strict cram": r"strict cram[^\d]{0,40}(\d[\d,]*)",
        "env-plausible": r"env(?:ironmentally)?[- ]plausible[^\d]{0,40}(\d[\d,]*)",
        "unique inchikeys": r"unique inchikey[s]?[^\d]{0,40}(\d[\d,]*)",
        "unique murcko scaffolds": r"unique murcko[^\d]{0,40}(\d[\d,]*)",
    }
    low = text.lower()
    for k, pat in patterns.items():
        m = re.search(pat, low)
        if m:
            found[k] = int(m.group(1).replace(",", ""))
    if all(v is None for v in found.values()):
        return None
    return found


def funnel_from_generated_csv(path: Path, max_rows: int | None = None) -> dict:
    """Count CRAM/env/InChI/Murcko from an existing generated SMILES file. Not a prior-sample log."""
    df = pd.read_csv(path, nrows=max_rows)
    col = next((c for c in ("SMILES", "smiles", "CanonicalSMILES") if c in df.columns), None)
    if col is None:
        return {"error": f"no SMILES column in {path}"}
    validator = CRAMValidator()
    n = 0
    rdkit_ok = 0
    cram_ok = 0
    env_ok = 0
    keys = set()
    scaffolds = set()
    for smi in df[col].astype(str):
        n += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rdkit_ok += 1
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        if validator.validate(mol).get("is_cram"):
            cram_ok += 1
        if env_plausible(mol):
            env_ok += 1
        try:
            keys.add(Chem.MolToInchiKey(mol))
        except Exception:
            pass
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if sc:
                scaffolds.add(sc)
        except Exception:
            pass
    return {
        "sampled from prior": None,
        "selfies-valid": None,
        "rdkit-sanitized": rdkit_ok,
        "strict cram": cram_ok,
        "env-plausible": env_ok,
        "unique inchikeys": len(keys),
        "unique murcko scaffolds": len(scaffolds),
        "n_rows": n,
        "source_file": str(path),
    }


def write_funnel(out: Path, log_counts: dict | None, notes: list[str], csv_counts: dict | None = None, run_meta: dict | None = None):
    steps = [
        "sampled from prior",
        "SELFIES-valid",
        "RDKit-sanitized",
        "strict CRAM (all four criteria)",
        "env-plausible",
        "unique InChIKeys",
        "unique Murcko scaffolds",
    ]
    key_map = {
        "sampled from prior": "sampled from prior",
        "SELFIES-valid": "selfies-valid",
        "RDKit-sanitized": "rdkit-sanitized",
        "strict CRAM (all four criteria)": "strict cram",
        "env-plausible": "env-plausible",
        "unique InChIKeys": "unique inchikeys",
        "unique Murcko scaffolds": "unique murcko scaffolds",
    }
    product_check = (
        "10^7 * 0.576 * 0.700 = 4,032,000, not 5.04e6. "
        "Do not treat 5.04 million as that product unless logs show a different path."
    )
    notes = list(notes) + [
        "Table 1 A is n=10,000 only; do not restore 72.3/69.7/87.5 unless those percentages are in the logs.",
        product_check,
        "Headline EPR stays TRFU 31.8 vs 38.0 (n=18513) until a per-ppm table exists.",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "count", "source", "note"])
        w.writeheader()
        for step in steps:
            raw = (log_counts or {}).get(key_map[step]) if log_counts else None
            csv_raw = (csv_counts or {}).get(key_map[step]) if csv_counts else None
            if raw is not None:
                count, source, note = raw, "generation log", ""
            elif csv_raw is not None:
                count, source, note = csv_raw, "generated CSV (not prior-sample log)", csv_counts.get("source_file", "")
            else:
                count, source, note = "", "MISSING", "no log integer recovered; not guessed"
            w.writerow({"step": step, "count": count, "source": source, "note": note})
        w.writerow(
            {
                "step": "Table 1 A",
                "count": 10000,
                "source": "paper constraint",
                "note": "n=10,000 only",
            }
        )
        w.writerow(
            {
                "step": "reported 5.04 million unique",
                "count": "",
                "source": "unverified",
                "note": product_check,
            }
        )
        rm = run_meta or {}
        temp = (rm.get("decoder_temperature") or {})
        cyc = (rm.get("n_kl_cycles") or {})
        cvec = (rm.get("property_target_c_10M") or {})
        w.writerow(
            {
                "step": "decoder temperature",
                "count": "" if temp.get("value") is None else temp.get("value"),
                "source": temp.get("source", "MISSING"),
                "note": "10M run; code defaults 0.8 / 1.0 are not treated as recovered",
            }
        )
        w.writerow(
            {
                "step": "n_kl_cycles",
                "count": "" if cyc.get("value") is None else cyc.get("value"),
                "source": cyc.get("source", "MISSING"),
                "note": "notebook IntegratedCRAMTrainer uses 4; train_cvae.py has no cycles — not assumed for 10M",
            }
        )
        w.writerow(
            {
                "step": "property-target c (10M)",
                "count": "",
                "source": cvec.get("source", "MISSING"),
                "note": "c ∈ R^32; do not guess vectors. See generation_run_meta.json",
            }
        )
        w.writerow(
            {
                "step": "vocab_size",
                "count": "" if rm.get("vocab_size") is None else rm.get("vocab_size"),
                "source": "checkpoint + {stem}_meta.json beside the .pt",
                "note": rm.get("vocab_size_sidecar", ""),
            }
        )
    (out.with_name("production_funnel_notes.json")).write_text(
        json.dumps(_jsonable({
            "notes": notes,
            "log_counts": log_counts,
            "csv_counts": csv_counts,
            "generation_run_meta": run_meta,
            "constraints": PAPER_CONSTRAINTS,
        }), indent=2)
    )


def recover_generation_run_meta(ckpt: dict, meta: dict, log_files: list[Path]) -> dict:
    """Decoder T, KL cycles, and 10M c — recovered only; code defaults labeled as such."""
    import re

    text = ""
    for p in log_files:
        try:
            text += "\n" + p.read_text(errors="ignore")
        except Exception:
            continue
    low = text.lower()

    def _from_log(patterns):
        for pat in patterns:
            m = re.search(pat, low)
            if m:
                return m.group(1)
        return None

    t_log = _from_log(
        [
            r"decoder temperature[:\s=]+([0-9.]+)",
            r"sampling temperature[:\s=]+([0-9.]+)",
            r"temperature[:\s=]+([0-9.]+)",
        ]
    )
    cyc_log = _from_log(
        [
            r"(?:n_)?kl[_ ]cycles?[:\s=]+(\d+)",
            r"annealing cycles[:\s=]+(\d+)",
            r"n_cycles[:\s=]+(\d+)",
        ]
    )
    temp = {
        "value": None,
        "source": "MISSING from 10M logs and checkpoint",
        "code_defaults_not_used_as_recovery": {
            "CVAE_GENERATION_CONFIG.temperature": 0.8,
            "generate_unique() default": 1.0,
            "model.decode default": 1.0,
        },
    }
    if meta.get("decoder_temperature") is not None:
        temp = {"value": meta["decoder_temperature"], "source": "checkpoint"}
    elif t_log is not None:
        temp = {"value": float(t_log), "source": "generation log"}

    cycles = {
        "value": None,
        "source": "MISSING from 10M logs and checkpoint",
        "code_defaults_not_used_as_recovery": {
            "notebook IntegratedCRAMTrainer.n_cycles": 4,
            "train_cvae.py": "constant kl_weight, no cyclical beta",
        },
    }
    if meta.get("n_cycles") is not None:
        cycles = {"value": meta["n_cycles"], "source": "checkpoint"}
    elif cyc_log is not None:
        cycles = {"value": int(cyc_log), "source": "generation log"}

    c_10m = {
        "recovered": False,
        "value": None,
        "source": "MISSING — do not guess the 10M property-target vectors",
        "code_default_sampler_not_used_as_recovery": {
            "n_cooh": [2, 3, 3, 4, 4, 5],
            "ring_class": ["fused", "fused", "mixed", "benzene_aliphatic", "pure_alicyclic"],
            "aliphatic_pct": "U(0.5, 0.8)",
            "mw": "U(200, 600) in generate_million; U(200, 500) in generate_unique",
            "has_benzene": "True/False (generate_million weights True)",
            "dbe_c": "U(0.3, 0.7)",
            "dbe_o": "U(0.5, 2.0)",
            "dbe_h": "U(0.1, 0.4)",
            "encoded_c": "R^32 via PropertyConditionEncoder",
        },
    }
    if "n_cooh" in low and "ring_class" in low:
        c_10m["log_mentions_property_keys"] = True
        c_10m["source"] = "logs mention property keys but integer/vector table not parsed"

    return {
        "decoder_temperature": temp,
        "n_kl_cycles": cycles,
        "property_target_c_10M": c_10m,
        "vocab_size": meta.get("vocab_size"),
        "vocab_size_sidecar": "{checkpoint_stem}_meta.json beside the .pt",
        "logs_searched": [str(p) for p in log_files],
        "constraints": PAPER_CONSTRAINTS,
    }


def write_checkpoint_sidecar(ckpt_path: Path, meta: dict, extra: dict):
    payload = {**meta, **extra, "constraints": PAPER_CONSTRAINTS}
    side = ckpt_path.with_name(ckpt_path.stem + "_meta.json")
    try:
        side.write_text(json.dumps(_jsonable(payload), indent=2))
        return side
    except OSError:
        return None


def run(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu") if args.cpu else get_device()
    if device.type == "mps":
        device = torch.device("cpu")

    search_roots = [Path(p) for p in args.search_root] if args.search_root else [
        Path.cwd(),
        ROOT,
        ROOT.parent,
        ROOT.parent.parent,
        Path("/content/drive/MyDrive/CRAM_Generator"),
    ]

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _first_existing(
        [
            ROOT / "models" / "cvae_best.pt",
            Path("/content/drive/MyDrive/CRAM_Generator/models/cvae_best.pt"),
        ]
    )
    if ckpt_path is None:
        raise SystemExit(
            "No GRU CVAE checkpoint (cvae_best.pt). "
            "Do not pass Transformer best_cram_model.pt (decoder_embedding.weight). "
            "Pass --checkpoint to a PropertyConditionedCRAMVAE .pt."
        )

    smiles_csv = Path(args.smiles_csv) if args.smiles_csv else _first_existing(
        [
            ROOT / "data" / "cram_molecules.csv",
            ROOT.parent.parent / "cram_data" / "CRAM.csv",
            Path("/content/drive/MyDrive/CRAM_Generator/cram_data/CRAM.csv"),
        ]
    )
    if smiles_csv is None and not args.train_smiles:
        raise SystemExit("No SMILES CSV. Pass --smiles-csv or --train-smiles")

    npz128 = Path(args.latent_128) if args.latent_128 else _search_named("latent_vectors_128d.npz", search_roots)
    extra_logs = [Path(p) for p in args.generation_log] if args.generation_log else []
    log_files = discover_logs(search_roots, extra_logs)

    print(f"device={device}")
    print(f"checkpoint={ckpt_path}")
    print(f"smiles_csv={smiles_csv}")
    print(f"latent_vectors_128d.npz={npz128 or 'MISSING (Morgan residual PCA deposit)'}")
    print(f"generation logs={log_files or 'MISSING'}")

    model, tokenizer, idx_to_token, ckpt, meta = load_cvae(ckpt_path, device)
    run_meta = recover_generation_run_meta(ckpt, meta, log_files)
    sidecar = write_checkpoint_sidecar(
        ckpt_path,
        meta,
        {
            "vocab_size_next_to_checkpoint": meta["vocab_size"],
            "generation_run": run_meta,
        },
    )
    (out / "checkpoint_meta.json").write_text(
        json.dumps(
            _jsonable(
                {
                    **meta,
                    "sidecar": str(sidecar) if sidecar else None,
                    "vocab_size": meta["vocab_size"],
                    "note": "vocab_size is also written beside the .pt as {stem}_meta.json when writable",
                }
            ),
            indent=2,
        )
    )
    (out / "generation_run_meta.json").write_text(json.dumps(_jsonable(run_meta), indent=2))
    print(f"vocab_size={meta['vocab_size']}  latent_dim={meta['latent_dim']}  condition_dim={meta['condition_dim']}")
    print(
        f"decoder_temperature={run_meta['decoder_temperature']['value']} "
        f"[{run_meta['decoder_temperature']['source']}]  "
        f"n_kl_cycles={run_meta['n_kl_cycles']['value']} [{run_meta['n_kl_cycles']['source']}]"
    )

    max_train = args.max_train or 5842
    if args.train_smiles:
        train_smiles = load_smiles_file(Path(args.train_smiles), max_n=max_train)
        train_source = f"--train-smiles {args.train_smiles}"
    else:
        smiles = load_smiles_list(smiles_csv, args.smiles_column, max_rows=args.csv_nrows)
        train_smiles = smiles[:max_train]
        train_source = (
            f"first {len(train_smiles)} unique SMILES from {smiles_csv} "
            "(not a recovered 5842-row train index)"
        )
    print(f"encoding n={len(train_smiles)} training SMILES ({train_source})")
    bundle = encode_smiles(model, train_smiles, device)
    mu, logvar = bundle["mu"], bundle["logvar"]
    if mu.shape[1] != 64:
        print(f"WARNING: encoded μ is {mu.shape[1]}-D; paper VAE z is 64-D")
    npz_payload = {
        "mu": mu.astype(np.float32),
        "logvar": logvar.astype(np.float32),
        "logsigma": 0.5 * logvar.astype(np.float32),
        "condition": bundle["condition"].astype(np.float32),
        "smiles": bundle["smiles"],
        "n": np.array(len(bundle["smiles"])),
        "latent_dim": np.array(mu.shape[1]),
        "train_source": np.array(train_source),
        "note": np.array("VAE encoder μ, logσ for training graphs; not Morgan PCA"),
    }

    # A1 dual-z vs deposited 128-D PCA
    align = {
        "encoder_n": int(mu.shape[0]),
        "encoder_dim": int(mu.shape[1]),
        "twonn_d_int": twonn_dint(mu),
        "n_PCs_95": n_pcs_95(mu),
        "valid_decode_PC1": None,
        "valid_decode_PC1_n_ok": None,
        "valid_decode_PC1_n": None,
        "RV": None,
        "Procrustes": None,
        "paired_n": None,
        "pca128_path": str(npz128) if npz128 else None,
        "pca128_note": "Morgan residual PCA, NOT VAE z" if npz128 else "latent_vectors_128d.npz not found",
        "umap": None,
    }
    frac, n_ok, n_tot = valid_decode_along_pc1(
        model, mu, bundle["condition"], idx_to_token, temperature=0.0
    )
    align["valid_decode_PC1"] = frac
    align["valid_decode_PC1_n_ok"] = n_ok
    align["valid_decode_PC1_n"] = n_tot
    align["valid_decode_note"] = "walk along PC1 of encoder μ (64-D), not the 128-D PCA"

    if npz128 is not None:
        z128, smi128, npz_keys = load_npz_128(npz128)
        align["pca128_keys"] = npz_keys
        if z128 is None:
            align["pca128_note"] = f"file found but no (n,128) array; keys={npz_keys}"
        elif smi128 is None:
            n = min(len(mu), len(z128), 5000)
            align["paired_n"] = n
            align["RV"] = rv_coefficient(mu[:n], z128[:n])
            align["Procrustes"] = procrustes_disparity(mu[:n], z128[:n])
            align["pca128_note"] = (
                "npz has no SMILES index; paired by row order for first "
                f"{n} (NOT the same 5,000 SMILES unless order matches)"
            )
        else:
            smi128_list = [str(s) for s in smi128.tolist()[:5000]]
            print(f"encoding the same {len(smi128_list)} SMILES as latent_vectors_128d.npz → μ ∈ R^64")
            paired_bundle = encode_smiles(model, smi128_list, device)
            npz_payload["mu_5000"] = paired_bundle["mu"].astype(np.float32)
            npz_payload["logvar_5000"] = paired_bundle["logvar"].astype(np.float32)
            npz_payload["smiles_5000"] = paired_bundle["smiles"]
            n_fail = len(paired_bundle["failed"])
            index128 = {s: i for i, s in enumerate(smi128.tolist())}
            paired_mu, paired_128 = [], []
            for s, row in zip(paired_bundle["smiles"].tolist(), paired_bundle["mu"]):
                if s in index128:
                    paired_mu.append(row)
                    paired_128.append(z128[index128[s]])
            if paired_mu:
                pm = np.stack(paired_mu)
                p128 = np.stack(paired_128)
                align["paired_n"] = int(len(paired_mu))
                align["RV"] = rv_coefficient(pm, p128)
                align["Procrustes"] = procrustes_disparity(pm, p128)
                align["pca128_note"] = (
                    "Morgan residual PCA vs encoder μ on the npz SMILES index "
                    f"(n={align['paired_n']}; encode failures={n_fail})"
                )
            else:
                align["paired_n"] = 0
                align["pca128_note"] = "npz SMILES encoded but none matched the 128-D index"
    np.savez_compressed(out / "vae_z64_mu_logvar.npz", **npz_payload)
    print(f"wrote {out / 'vae_z64_mu_logvar.npz'}  train μ shape={mu.shape}")
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=0)
        um = reducer.fit_transform(mu)
        np.save(out / "umap_mu64.npy", um)
        align["umap"] = str(out / "umap_mu64.npy")
    except Exception as e:
        align["umap"] = f"not computed ({type(e).__name__}: {e})"

    with (out / "dual_z_alignment.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "note"])
        w.writerow(["TwoNN_d_int", align["twonn_d_int"], "on encoder μ ∈ R^64"])
        w.writerow(["n_PCs_95", align["n_PCs_95"], "PCA of μ to 95% variance"])
        w.writerow(["valid_decode_PC1", align["valid_decode_PC1"], align["valid_decode_note"]])
        w.writerow(["RV", align["RV"], "μ vs deposited 128-D PCA (Morgan residual, not VAE z)"])
        w.writerow(["Procrustes", align["Procrustes"], "disparity on min(64,128) leading columns"])
        w.writerow(["paired_n", align["paired_n"], "same SMILES as latent_vectors_128d.npz if index present"])
        w.writerow(["encoder_n", align["encoder_n"], "training graphs encoded"])
        w.writerow(["pca128_path", align["pca128_path"] or "", align["pca128_note"]])
        w.writerow(["umap", align["umap"], "UMAP of encoder μ (64-D)"])
    (out / "dual_z_alignment.json").write_text(json.dumps(_jsonable(align), indent=2))

    # A2 collapse
    kl = kl_per_dim(mu, logvar)
    mean_kl = kl.mean(axis=0)
    active = mean_kl > ACTIVE_KL_NATS
    sigma2 = np.exp(np.clip(logvar, -10, 10))
    nll = reconstruction_nlls(model, bundle["smiles"].tolist(), tokenizer, device, max_n=args.nll_n)
    collapse_present = int(active.sum()) < 8 or float(mean_kl.mean()) < ACTIVE_KL_NATS
    collapse = {
        "n_samples": int(mu.shape[0]),
        "latent_dim": int(mu.shape[1]),
        "kl_threshold_nats": ACTIVE_KL_NATS,
        "n_active_units": int(active.sum()),
        "n_collapsed_units": int((~active).sum()),
        "frac_active": float(active.mean()),
        "mean_kl": float(mean_kl.mean()),
        "sum_kl": float(mean_kl.sum()),
        "mean_mu_norm": float(np.linalg.norm(mu, axis=1).mean()),
        "mean_sigma2": float(sigma2.mean()),
        "reconstruction_nll_qzx": nll["nll_qzx"],
        "reconstruction_nll_c_only_z_zeroed": nll["nll_c_only"],
        "token_accuracy_q": nll["token_accuracy_q"],
        "nll_n": nll["n"],
        "collapse_present": collapse_present,
        "collapse_statement": (
            f"Posterior collapse is present: {int(active.sum())}/{mu.shape[1]} units "
            f"exceed {ACTIVE_KL_NATS} nats (mean KL={float(mean_kl.mean()):.4f})."
            if collapse_present
            else f"Most units are active: {int(active.sum())}/{mu.shape[1]} above {ACTIVE_KL_NATS} nats."
        ),
        "do_not_paper_over": True,
        "generated_10k_with_recorded_c": "not run — no recorded c vectors for generated molecules",
        "constraints": PAPER_CONSTRAINTS,
    }
    with (out / "kl_per_dim.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "KL_i", "mu_mean", "sigma2_mean", "active"])
        for i in range(mu.shape[1]):
            w.writerow(
                [
                    i,
                    float(mean_kl[i]),
                    float(mu[:, i].mean()),
                    float(sigma2[:, i].mean()),
                    int(bool(active[i])),
                ]
            )
    (out / "collapse_summary.json").write_text(json.dumps(_jsonable(collapse), indent=2))
    print(collapse["collapse_statement"])

    curve = write_training_curve(ckpt, out / "training_curve.csv", ckpt_path)
    if curve["epoch_178"]["present"] is False:
        print("epoch-178 kl_beta=0.0498 NOT confirmed (history shorter or missing)")
    else:
        print(f"epoch 178 kl_beta={curve['epoch_178']['kl_beta']} matches_0.0498={curve['epoch_178']['matches_0.0498']}")

    log_counts = parse_funnel_from_logs(log_files)
    csv_counts = None
    gen_csv = Path(args.generated_csv) if args.generated_csv else _search_named("generated_smiles.csv", search_roots)
    if gen_csv and Path(gen_csv).is_file():
        csv_counts = funnel_from_generated_csv(Path(gen_csv), max_rows=args.funnel_csv_nrows)
        print(f"funnel from generated CSV {gen_csv}: {csv_counts}")
    write_funnel(
        out / "production_funnel.csv",
        log_counts,
        notes=[f"logs={log_files or 'none'}", f"generated_csv={gen_csv}"],
        csv_counts=csv_counts,
        run_meta=run_meta,
    )
    print(f"wrote funnel to {out / 'production_funnel.csv'}  log_counts={log_counts}")
    print(f"done. outputs in {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A1 Dual-z, A2 collapse, A3 10M funnel (no restored numbers)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--smiles-csv", default=None)
    p.add_argument("--smiles-column", default=None)
    p.add_argument("--latent-128", default=None, help="path to latent_vectors_128d.npz (Morgan residual PCA)")
    p.add_argument("--generation-log", action="append", default=None)
    p.add_argument("--search-root", action="append", default=None)
    p.add_argument("--out-dir", default="outputs/paper_blocking")
    p.add_argument("--max-train", type=int, default=5842)
    p.add_argument("--csv-nrows", type=int, default=None)
    p.add_argument("--nll-n", type=int, default=256)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--train-smiles", default=None, help="exact training SMILES list (5842); otherwise first N from --smiles-csv")
    p.add_argument("--generated-csv", default=None, help="generated SMILES CSV to count CRAM/env/InChI (not a prior log)")
    p.add_argument("--funnel-csv-nrows", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
