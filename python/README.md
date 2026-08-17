# Python edition

Property-conditioned CVAE for CRAM generation. Run all commands from this folder (`python/`).

The trained production CVAE uses a **64-dimensional latent space**.

---

## What this package does

1. Validate SMILES against CRAM chemical criteria.
2. Extract 163-dimensional per-atom features for graph neural networks.
3. Train a property-conditioned CVAE (GAT encoder → 64D latent → SELFIES decoder).
4. Generate molecules with CVAE (checkpoint required) or SELFIES mutation.
5. Search PubChem for CRAM-like seeds.

Sibling editions: [Java REST/CLI](../java/README.md) · [Julia CLI](../julia/README.md) · [repo root](../README.md) · [design](../DESIGN.md).

The **trained 64D CVAE is the better generator** (property-conditioned, ~100× faster than mutation). Java/Julia mutation is expected at ~30% Hertkorn-window; recorded Java `--seed 42`: 50% in 0.05 s (CDK), 56% in 0.35 s (RDKit).

**Assets:** [Google Drive](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing) (`cram_data`, `models`, `Generated_Structures`) — copy into `data/`, `models/`, and `outputs/Generated_Structures/` as described in [DATA.md](../DATA.md).

---

## Requirements

- Python 3.9–3.11
- PyTorch 2.x, DGL, RDKit, SELFIES

GPU (CUDA) is recommended for training. DGL graph training is most reliable on CPU or CUDA.

---

## Installation

After clone, download Drive assets (see [DATA.md](../DATA.md)), then:

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -U pip torch
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
pip install -e .
```

CUDA: keep the PyTorch build that matches the GPU (Colab: do **not** `pip install torch==2.2.1`). Then install a DGL wheel for that CUDA:

```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

If the DGL GAT probe fails, reinstall DGL for the installed `torch.version.cuda`. Use CPU only if you set that intentionally.

---

## Quick start

```bash
python main.py validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
python main.py features "CC1CCC(CC1)(C(=O)O)C(=O)O" --verbose
python main.py generate --mode mutation --count 100 --output outputs/cram_mutants.csv
python scripts/compare_mutation_smoke.py -n 40 --seed 42
python main.py ui --port 7860
python main.py info
```

CVAE sampling (needs a 64D checkpoint):

```bash
python main.py generate --mode vae \
  --checkpoint models/cvae_best.pt \
  --count 1000 --n-cooh 3 --ring-class pure_alicyclic \
  --output outputs/cram_cvae.csv
```

Train:

```bash
python scripts/train_cvae.py \
  --smiles-csv data/cram_molecules.csv \
  --epochs 50 --latent-dim 64 \
  --n-cycles 4 --output-dir models/cvae_run
python scripts/paper_blocking_criticals.py \
  --checkpoint models/cvae_best.pt \
  --smiles-csv data/cram_molecules.csv \
  --out-dir outputs/paper_blocking
```

PubChem harvest:

```bash
python scripts/pubchem_cram_search.py
```

---

## Model

```text
SMILES → molecular graph (163D atom features)
        ↓
DeepCRAMEncoder (GAT, hidden 256)
        ↓
z ~ N(μ, σ²)          latent dim = 64
        ↓
condition vector      condition dim = 32
        ↓
SELFIESDecoder (GRU, hidden 256)
        ↓
SELFIES tokens → SMILES
```

| Parameter | Value |
|-----------|-------|
| Atom feature dimension | 163 |
| Encoder hidden / GAT | 256 wide, 3 layers, 4 heads × 64 |
| Pooling | gated mean |
| **Latent dimension** | **64** |
| Property concat / MLP / **c** | 35 / 64 / **32** |
| GRU decoder | 256 hidden, 2 layers; embed 256 → vocab |
| Max length | 100 |
| Special tokens | PAD=0, SOS=1, EOS=2, UNK=3 |

### 163D atom feature groups

Notebook order (same as `DeepCRAMFeatureExtractor.FEATURE_DIMS`):

| Group | Dims | Description |
|-------|:----:|-------------|
| Base atom features | 45 | Element, hybridization, bonds, charge, ring |
| Synthetic accessibility | 28 | Complexity / SA-related flags |
| Marine DOM | 15 | Ocean-chemistry indicators |
| Ring–chain | 20 | Alicyclic rings, junctions, carboxyl proximity |
| Stereochemistry | 12 | Chirality, ring conformation, strain |
| Terpenoid | 10 | Isoprene / natural-product ancestry |
| Functional groups | 13 | COOH, OH, carbonyl, etc. |
| Enhanced oxidation | 20 | Oxidation state, COOH α/β/γ/δ proximity |
| **Total** | **163** | |

Production sampling still uses the GRU CVAE (`PropertyConditionedCRAMVAE`, `decoder.gru`). Notebook Transformer VAEs (`DeepCRAMVAE`, `PropertyConditionedTransformerVAE`, `MultimodalTransformerVAE`, `TransformerVAE`) live in `src/models/transformer_vae.py` and use the GAT + ring-chain + FG encoder (`src/models/deep_cram_encoder.py`). Do not load `best_cram_model.pt` into the GRU CVAE.

---

## Python API

Run from `python/` so `src` imports resolve:

```python
from src.validator import CRAMValidator
from src.features import DeepCRAMFeatureExtractor
from src.generators import SELFIESMutationGenerator

validator = CRAMValidator()
print(validator.validate_smiles("CC1CCC(CC1)(C(=O)O)C(=O)O"))
```

---

## Layout

```text
python/
├── main.py                 CLI: validate / generate / features / ui / info
├── cli.py
├── config/settings.py      latent_dim=64
├── src/models/             GRU CVAE + notebook Transformer / GAT encoders
├── src/generators/         mutation + CVAE sampling + notebook HT generators
├── src/features/           163D atom features (notebook layout)
├── ui/                     Gradio
├── scripts/                train_cvae.py, PubChem, figures
├── data/                   from Drive `cram_data` (not committed)
├── models/                 from Drive `models` (not committed)
└── outputs/                from Drive `Generated_Structures` (not committed)
```

---

## Docker

```bash
cd python
docker compose up --build          # http://localhost:7860
docker compose --profile gpu up
```

---

## License

MIT — see [../LICENSE](../LICENSE)
