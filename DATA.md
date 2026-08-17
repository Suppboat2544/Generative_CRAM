# Data, models, and generated structures

Weights, training tables, and generated molecule files are **not** stored in git. Download them from Google Drive. Architecture: [DESIGN.md](DESIGN.md).

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

Use the actual filename from the Drive `models/` folder if it differs. **Paper A1–A3 require this GRU CVAE** (`z∈ℝ⁶⁴`, `concat(z,c)=96`). Transformer `best_cram_model.pt` (`decoder_embedding.weight`) is a different model.

## Paper blocking deposits (A1–A3)

Place these next to the CVAE checkpoint or under `cram_data/` / `results/` so `scripts/paper_blocking_criticals.py` can find them. If a file is absent the script writes **MISSING**; it does not restore unpublished numbers.

| File | Role |
|------|------|
| `cvae_best.pt` | GRU CVAE weights; `{stem}_meta.json` with **vocab_size** is written beside it |
| `train_smiles_5842.txt` (optional) | Exact 5,842 training SMILES; otherwise first N unique from the CSV, labeled as such |
| `latent_vectors_128d.npz` | Morgan residual PCA **(not VAE z)** plus SMILES index for the 5,000 paired encodings |
| `production_10M.log` / `generation_10M.log` | Integer funnel: prior → SELFIES-valid → RDKit → strict CRAM → env-plausible → unique InChIKeys → unique Murcko |
| 10k Table 1 A generation log | Same funnel steps; Table 1 A is **n=10,000 only** |

Do not treat `10⁷×0.576×0.700` as 5.04 million (that product is 4,032,000). Do not restore 72.3 / 69.7 / 87.5 unless those percentages are in the logs.

Outputs: `vae_z64_mu_logvar.npz`, `dual_z_alignment.csv`, `kl_per_dim.csv`, `collapse_summary.json`, `training_curve.csv`, `production_funnel.csv`, `generation_run_meta.json` (decoder temperature, KL cycles, 10M **c**).

## Use after download

```bash
cd python
python scripts/train_cvae.py --smiles-csv data/cram_molecules.csv --latent-dim 64
python scripts/paper_blocking_criticals.py --checkpoint models/<cvae>.pt --smiles-csv data/cram_molecules.csv --out-dir outputs/paper_blocking
python main.py generate --mode vae --checkpoint models/<checkpoint>.pt --count 100
```

Java and Julia use SMILES from `cram_data` for validate, 163D features, and mutation generate. They do not load PyTorch checkpoints. The property-conditioned CVAE stays in `python/`.
