# Data, models, and generated structures

Weights, training tables, and generated molecule files are **not** stored in git. Download them from Google Drive:

**[CRAM Drive folder](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing)**

| Drive folder | What it is | Place it here after download |
|--------------|------------|------------------------------|
| `cram_data` | Training / reference SMILES and property tables | `python/data/` |
| `models` | CVAE checkpoints (`.pt`, latent dim **64**) | `python/models/` |
| `Generated_Structures` | Generated CRAM structures (CSV / SDF / images) | `python/outputs/Generated_Structures/` |

## Setup

1. Open the [Drive folder](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing).
2. Download `cram_data`, `models`, and `Generated_Structures`.
3. Copy into the Python edition:

```bash
cd Generative_CRAM
# after unzipping the Drive downloads:
cp -R cram_data/* python/data/
cp -R models/* python/models/
mkdir -p python/outputs/Generated_Structures
cp -R Generated_Structures/* python/outputs/Generated_Structures/
```

If a Drive subfolder is already named `data` or contains `cram_molecules.csv`, keep that CSV at:

```text
python/data/cram_molecules.csv
```

A typical checkpoint path for CVAE sampling:

```text
python/models/cvae_best.pt
```

Use the actual filename from the Drive `models/` folder if it differs.

## Use after download

```bash
cd python
python scripts/train_cvae.py --smiles-csv data/cram_molecules.csv --latent-dim 64
python main.py generate --mode vae --checkpoint models/<checkpoint>.pt --count 100
```

Java and Julia use SMILES from `cram_data` for validate, 163D features, and mutation generate. They do not load PyTorch checkpoints. The property-conditioned CVAE stays in `python/`.
