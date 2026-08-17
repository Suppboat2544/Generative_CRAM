# Generative CRAM

Python toolkit for **validating, encoding, and generating Carboxyl-Rich Alicyclic Molecules (CRAM)** — refractory components of marine dissolved organic matter (DOM).

This repository contains **Python source only** (no Jupyter notebooks). The trained CVAE uses a **64-dimensional latent space**.

---

## What this package does

1. **Validate** SMILES against CRAM chemical criteria.
2. **Extract** 163-dimensional per-atom features for graph neural networks.
3. **Train** a property-conditioned CVAE (GAT encoder → 64D latent → SELFIES decoder).
4. **Generate** new CRAM-like molecules:
   - CVAE (property-conditioned, requires a checkpoint)
   - SELFIES mutation (no checkpoint required)
5. **Search PubChem** for CRAM-like seeds.

---

## CRAM criteria

| Criterion | Requirement |
|-----------|-------------|
| Carboxyl groups (COOH / COO⁻) | ≥ 2 |
| Ring system | ≥ 1 alicyclic (non-aromatic) ring |
| Aliphatic carbon | ≥ 60% |
| Molecular weight | 100–750 Da |

Example (valid CRAM-like dicarboxylic alicycle):

```text
CC1CCC(CC1)(C(=O)O)C(=O)O
```

---

## Model (trained configuration)

The production CVAE checkpoint was trained with **latent dimension 64**, not 128.

```
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

### Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Atom feature dimension | **163** |
| Encoder hidden size | 256 |
| **Latent dimension** | **64** |
| Condition dimension | 32 |
| GAT heads / layers | 4 / 3 |
| GRU decoder layers | 2 |
| SELFIES max length | 100–150 |
| Token IDs | PAD=0, SOS=1, EOS=2, UNK=3 |

A later notebook draft raised `latent_dim` to 128 in source, but **weights that were actually trained and loaded are 64D**. Loading 64D checkpoints into a 128D architecture will fail. This package matches the trained model (**64**).

### 163D atom feature groups

| Group | Dims | Description |
|-------|:----:|-------------|
| Base atom features | 45 | Atom type, bonds, hybridization |
| Ring–chain | 20 | Ring membership, chain context |
| Stereochemistry | 12 | Chirality / stereo flags |
| Marine DOM | 15 | Ocean-chemistry indicators |
| Synthetic accessibility | 28 | Complexity / SA-related flags |
| Terpenoid | 10 | Natural-product ancestry |
| Functional groups | 13 | COOH, OH, carbonyl, etc. |
| Oxidation | 20 | Redox / heteroatom environment |
| **Total** | **163** | |

---

## Requirements

- Python 3.9–3.11
- PyTorch 2.x
- DGL (Deep Graph Library)
- RDKit, SELFIES

GPU (CUDA) is recommended for training. Apple Silicon can use MPS for some ops; DGL graph training is most reliable on CPU or CUDA.

---

## Installation

```bash
git clone https://github.com/Suppboat2544/Generative_CRAM.git
cd Generative_CRAM

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install torch
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
pip install -e .
```

CUDA 12.1 example:

```bash
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu121/repo.html
```

---

## Quick start

### Validate a molecule

```bash
python main.py validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
```

### Extract 163D features

```bash
python main.py features "CC1CCC(CC1)(C(=O)O)C(=O)O" --verbose
```

### Generate by SELFIES mutation (no checkpoint)

```bash
python main.py generate --mode mutation --count 100 --output outputs/cram_mutants.csv
```

### Generate with a trained CVAE

```bash
python main.py generate --mode vae \
  --checkpoint models/cvae_best.pt \
  --count 1000 \
  --n-cooh 3 \
  --ring-class pure_alicyclic \
  --output outputs/cram_cvae.csv
```

### Train a CVAE

```bash
python scripts/train_cvae.py \
  --smiles-csv data/cram_molecules.csv \
  --smiles-column CanonicalSMILES \
  --epochs 50 \
  --latent-dim 64 \
  --output-dir models/cvae_run
```

Place a SMILES CSV in `data/` (column name defaults to `smiles` or `CanonicalSMILES`). Do not commit large datasets or `.pt` weights.

### PubChem seed search

```bash
python scripts/pubchem_cram_search.py
```

### Gradio UI

```bash
python main.py ui --port 7860
```

### System check

```bash
python main.py info
```

---

## Python API

```python
from src.validator import CRAMValidator
from src.features import DeepCRAMFeatureExtractor
from src.models import create_cvae_model
from src.generators import FastPropertyConditionedGenerator, SELFIESMutationGenerator

validator = CRAMValidator()
print(validator.validate_smiles("CC1CCC(CC1)(C(=O)O)C(=O)O"))

extractor = DeepCRAMFeatureExtractor()
atom_feats = extractor.extract_features("CC1CCC(CC1)(C(=O)O)C(=O)O")
# atom_feats.shape == (n_atoms, 163)

mut = SELFIESMutationGenerator(seed_smiles=[
    "CC1CCC(CC1)(C(=O)O)C(=O)O",
    "OC(=O)C1CCCCC1C(=O)O",
])
print(mut.generate(target_count=10))
```

CVAE generation after you load a matching 64D checkpoint:

```python
import torch
from src.models import create_cvae_model
from src.generators import FastPropertyConditionedGenerator

ckpt = torch.load("models/cvae_best.pt", map_location="cpu")
model = create_cvae_model(vocab_size=ckpt["vocab_size"], latent_dim=64)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

gen = FastPropertyConditionedGenerator(
    model, ckpt["selfies_vocab"], ckpt.get("idx_to_token")
)
smiles = gen.generate_fast(n=100, n_cooh=3, ring_class="pure_alicyclic")
```

Checkpoint keys vary by training script. Inspect with `ckpt.keys()` and adapt the load block.

---

## Repository layout

```text
Generative_CRAM/
├── main.py                 CLI: validate / generate / features / ui / info
├── cli.py                  Alternate rich CLI
├── setup.py / pyproject.toml
├── requirements.txt
├── config/settings.py      Architecture + CRAM thresholds (latent_dim=64)
├── src/
│   ├── validator.py        CRAM rules
│   ├── tokenizer.py        SELFIES vocabulary + beam search
│   ├── dataset.py          Graph + sequence datasets
│   ├── training.py         Trainer, AMP, checkpoints
│   ├── models/
│   │   ├── encoder.py      DeepCRAMEncoder (GAT → 64D)
│   │   ├── decoder.py      SELFIES GRU decoder
│   │   ├── condition.py    PropertyConditionEncoder (32D)
│   │   ├── cvae.py         PropertyConditionedCRAMVAE
│   │   └── losses.py       Reconstruction + KL
│   ├── generators/
│   │   ├── mutation_gen.py SELFIES mutation
│   │   └── cvae_gen.py     Fast property-conditioned sampling
│   └── features/           163D atom features
├── ui/                     Gradio app
├── scripts/
│   ├── train_cvae.py                 Training entry point
│   ├── pubchem_cram_search.py        PubChem CRAM harvest
│   ├── generate_results_figures.py   Result figures
│   └── van_krevelen_ocean_*.py       DOM vs CRAM van Krevelen plots
├── enhanced_polycyclic_generator.py
├── data/                   Put SMILES CSVs here (not committed)
├── models/                 Put .pt checkpoints here (not committed)
└── outputs/                Generation results (not committed)
```

Jupyter notebooks from the research workspace are **intentionally excluded**.

---

## Docker

```bash
docker compose up --build
# UI: http://localhost:7860

docker compose --profile gpu up    # NVIDIA
```

---

## Data and checkpoints

Not included in git (see `.gitignore`):

- `*.pt` / `*.pth` model weights
- large CSV/Excel datasets
- `outputs/`

Provide your own CRAM SMILES list and, for CVAE sampling, a checkpoint trained with **the same `latent_dim` (64), vocab, and token IDs**.

---

## Citation

If you use this code, please also cite the CRAM / DOM literature, e.g.:

- Hertkorn, N. et al. (2006). Characterization of a major refractory component of marine dissolved organic matter. *Geochimica et Cosmochimica Acta*.
- Krenn, M. et al. (2020). SELFIES: a robust representation of semantically constrained graphs. *Machine Learning: Science and Technology*.
- Kingma, D. P. & Welling, M. (2013). Auto-Encoding Variational Bayes.

```
Klabklaydee, S. Generative CRAM: property-conditioned generation of
carboxyl-rich alicyclic molecules. https://github.com/Suppboat2544/Generative_CRAM
```

---

## License

MIT — see [LICENSE](LICENSE).

## Author

**Klabklaydee Supaporn**  
Fujii Laboratory, Institute of Science Tokyo  
Supervised by Assoc. Prof. Manabu Fujii
