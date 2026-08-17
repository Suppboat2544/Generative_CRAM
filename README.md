# Generative CRAM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Toolkit for **validating, encoding, and generating Carboxyl-Rich Alicyclic Molecules (CRAM)** — refractory components of marine dissolved organic matter (DOM).

Public libraries under-sample this space. This repo **filters** SMILES against CRAM rules and **generates** new structures. The intended generator is the **trained 64-D Python CVAE**. Java and Julia share the chemistry rules and provide mutation / REST / CLI; they do not load `.pt` checkpoints.

| | |
|--|--|
| Architecture | [DESIGN.md](DESIGN.md) |
| Data, weights, generated files | [DATA.md](DATA.md) — [Google Drive](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing) |
| Python | [python/README.md](python/README.md) |
| Java | [java/README.md](java/README.md) |
| Julia | [julia/README.md](julia/README.md) |

```text
SMILES  →  CRAM criteria  →  163D atom features  →  generate / serve
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         python/       java/        julia/
         GAT CVAE      CDK REST     MolecularGraph
         Gradio        Picocli      CLI
         train .pt     mutation     mutation
```

| Folder | Role | Not for |
|--------|------|---------|
| [`python/`](python/README.md) | Property-conditioned CVAE (64D), training, Gradio, SELFIES mutation | JVM API |
| [`java/`](java/README.md) | Validate, 163D CDK features, mutation, REST | `.pt` / training |
| [`julia/`](julia/README.md) | Validate, 163D features, mutation CLI | CVAE |

The **trained CVAE is better** than Java/Julia mutation: it is property-conditioned, samples marine-like CRAM without a SMILES reject loop, and is **~100× faster**. Mutation is CRAM-valid by construction but typically only **~30% environmentally plausible** (Hertkorn O/C 0.40–0.60, H/C 1.20–1.70).

Notebooks and large binaries are not in git.

---

## 1. Chemistry

### Structural CRAM (all languages)

| Criterion | Requirement |
|-----------|-------------|
| Carboxyl groups (COOH / COO⁻) | ≥ 2 |
| Ring system | ≥ 1 alicyclic (non-aromatic) ring |
| Aliphatic carbon | ≥ 60% |
| Molecular weight | 100–750 Da |

Example: `CC1CCC(CC1)(C(=O)O)C(=O)O` — O/C **0.44**, H/C **1.56**, MW **186.2 Da**.

### Environmentally plausible (Hertkorn window)

Structural CRAM **and**:

| Ratio | Window |
|-------|--------|
| O/C | 0.40–0.60 |
| H/C | 1.20–1.70 |

Java/Julia mutation often passes the first table and misses the second. The CVAE is trained to hit both.

---

## 2. Production CVAE (Python, latent 64)

```text
SMILES → RDKit mol → DGL graph
       → 163D features / atom
       → Linear 163 → 256
       → GAT × 3 (4 heads × 64 → 256), residual + LayerNorm
       → gated mean pool → 256
       → μ, log σ² → z ∈ R⁶⁴
       → property encoder → c ∈ R³²
       → concat (z, c) = 96 → Linear → GRU hidden 256
       → GRU × 2, embed 256, out vocab
       → SELFIES tokens → SMILES
```

### GAT encoder

| | |
|--|--|
| Atom input | **163** |
| Width after input projection | **256** |
| Layers | **3** `GATConv` |
| Heads | **4** |
| Per-head width | **64** (`256 / 4`), flattened back to 256 |
| Residual | yes, plus LayerNorm |
| Dropout | 0.1 (features and attention) |
| **Pooling** | **gated mean**: `σ(Linear(256 → 1))` per atom, then `dgl.mean_nodes(h * gate)` |
| Latent | two MLPs from the 256-D graph vector → **μ, log σ² ∈ R⁶⁴** |

### GRU decoder

| | |
|--|--|
| Token embedding | `vocab_size → 256` (padding index 0) |
| Init hidden | concat **z (64) + c (32) = 96** → Linear → **256**, copied to both GRU layers |
| GRU | input **256**, hidden **256**, **2** layers, `batch_first`, dropout 0.1 |
| Output | Linear **256 → vocab_size** |
| Max length | **100** (CVAE); tokenizer cap **150** |

### Vocabulary (SELFIES)

Built from the training SMILES (`SELFIESTokenizer.build_vocab`). Not a fixed integer.

| Index | Token |
|------:|-------|
| 0 | `<PAD>` |
| 1 | `<SOS>` |
| 2 | `<EOS>` |
| 3 | `<UNK>` |
| 4… | sorted unique SELFIES tokens from the data |

`vocab_size = 4 + N_selfies_tokens`. It is stored next to the checkpoint when you train (`train_cvae.py` writes it in the run config).

### Property embedding → condition **c ∈ R³²**

| Piece | Size |
|-------|------|
| `n_cooh` Embedding(0–6) | 8 |
| Ring class Embedding(6) | 16 |
| Ring-size class Embedding(3) | 4 |
| Continuous: aliphatic %, MW, benzene, DBE/C, DBE/O, DBE/H, mean ring size | 7 |
| **Concat** | **35** |
| MLP hidden | 64 |
| **Condition vector c** | **32** |

Ring classes: fused aro–alicyclic, mixed, benzene+aliphatic, pure alicyclic, pure aromatic, acyclic. Ring-size class: strained (3–4) / CRAM-preferred (5–7) / large (8+).

### 163D atom feature groups

| Group | Dims | Role |
|-------|:----:|------|
| Base | 45 | Element, bonds, hybridization |
| Ring–chain | 20 | Ring membership, sizes |
| Stereo | 12 | Chirality / stereo |
| Marine DOM | 15 | Ocean-chemistry flags |
| Synthetic accessibility | 28 | Complexity |
| Terpenoid | 10 | Natural-product ancestry |
| Functional groups | 13 | COOH, OH, carbonyl, … |
| Oxidation | 20 | Carbon redox / O neighbors |
| **Total** | **163** | |

Python (RDKit), Java (CDK), and Julia (MolecularGraph) use the same group widths. Vectors are aligned for the GNN, not bit-identical.

---

## 3. Mutation vs CVAE

```text
seed SMILES ─┬─ mutation (Java / Julia / Python SELFIES)
             │     edit string → parse → keep structural CRAM
             │     ~30% in Hertkorn window (smoke often 30–50%)
             │     reject loop; some CDK SMILES fail RDKit valence
             │
             └─ trained CVAE (Python only)     ← use this
                   condition on n_COOH, ring class, MW, …
                   batch decode, no validation loop
                   ~100× faster
```

Recorded mutation smoke (`--seed 42`, 18 Aug 2026):

| Check | n | Hertkorn window | Time |
|-------|---|-----------------|------|
| Java CLI `smoke` (CDK) | 40 | 20/40 = **50%** | **0.05 s** (824 mol/s) |
| Java JUnit | 40 | 50% | 0.05 s |
| RDKit referee (`compare_mutation_smoke.py --java-only`) | 20 | 9/16 = **56%** (4 unparseable) | **0.35 s** wall |
| Julia `cli.jl smoke` | 40 | same protocol | needs Julia 1.10+ |
| **Trained Python CVAE** | — | **better** (conditioned) | **~100×** vs mutation reject loops |

---

## 4. Install and run

### A. Python CVAE

```bash
git clone https://github.com/Suppboat2544/Generative_CRAM.git
cd Generative_CRAM/python
python -m venv .venv && source .venv/bin/activate
pip install -U pip torch
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
pip install -e .
```

Copy Drive folders first: `cram_data` → `python/data/`, `models` → `python/models/`, `Generated_Structures` → `python/outputs/Generated_Structures/` ([DATA.md](DATA.md)).

```bash
python main.py validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
python main.py features "CC1CCC(CC1)(C(=O)O)C(=O)O" --verbose
python main.py generate --mode vae --checkpoint models/cvae_best.pt --count 100 --n-cooh 3 --ring-class pure_alicyclic
python main.py generate --mode mutation --count 100 --output outputs/cram_mutants.csv
python main.py ui --port 7860
python scripts/train_cvae.py --smiles-csv data/cram_molecules.csv --epochs 50 --latent-dim 64
python scripts/compare_mutation_smoke.py -n 40 --seed 42
```

Needs Python 3.9–3.11, PyTorch 2.x, DGL, RDKit, SELFIES. GPU is recommended for training; DGL is more stable on CPU or CUDA than MPS.

### B. Java (REST + mutation)

Java 21+, Maven 3.8+. Does **not** run the CVAE.

```bash
cd java
mvn package -DskipTests
java -jar target/cram-generator-1.0.0.jar validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
java -jar target/cram-generator-1.0.0.jar features "CC1CCC(CC1)(C(=O)O)C(=O)O" -v
java -jar target/cram-generator-1.0.0.jar generate -n 50 --seed 42
java -jar target/cram-generator-1.0.0.jar smoke -n 40 --seed 42
java -jar target/cram-generator-1.0.0.jar --server
```

| Method | Path | Role |
|--------|------|------|
| GET | `/api/v1/health` | Health |
| GET | `/api/v1/validate?smiles=` | Validate |
| GET | `/api/v1/features?smiles=` | 163D molecule mean |
| POST | `/api/v1/generate` | Mutation generate `{"n":20}` (includes elapsed seconds) |
| GET | `/api/v1/criteria` | CRAM rules |

### C. Julia (CLI + mutation)

Julia 1.10+.

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. cli.jl validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
julia --project=. cli.jl generate -n 50 -o generated.csv
julia --project=. cli.jl smoke -n 40 --seed 42
```

---

## 5. Repository

```text
Generative_CRAM/
├── README.md                 this file
├── DESIGN.md                 architecture narrative
├── DATA.md                   Drive copy paths
├── LICENSE
├── python/
│   ├── main.py               validate / generate / features / ui
│   ├── config/settings.py    latent 64, 163D, CRAM rules
│   ├── src/models/           encoder, decoder, cvae, condition
│   ├── src/generators/       CVAE sampler + SELFIES mutation
│   ├── src/features/         163D
│   ├── src/tokenizer.py      SELFIES vocab
│   ├── ui/                   Gradio
│   └── scripts/              train_cvae.py, compare_mutation_smoke.py
├── java/                     Spring Boot 3.4 + CDK 2.9 + Picocli
│   └── src/test/.../MutationEnvironmentalSmokeTest.java
└── julia/                    MolecularGraph.jl
    ├── cli.jl
    └── test/smoke.jl
```

Git: source only. No `.ipynb`, no `.pt`, no `java/target`.

---

## Citation

- Hertkorn, N. et al. (2006). *Geochimica et Cosmochimica Acta* — CRAM in marine DOM.
- Krenn, M. et al. (2020). SELFIES.
- Kingma, D. P. & Welling, M. (2013). Auto-Encoding Variational Bayes.
- Veličković, P. et al. (2018). Graph Attention Networks.

```
Klabklaydee, S. Generative CRAM.
https://github.com/Suppboat2544/Generative_CRAM
```

## License

**MIT** — [LICENSE](LICENSE). SPDX: `MIT`.

Anyone may use, copy, modify, and distribute this software, including commercially, if they keep the copyright notice. Weights and data stay on Drive ([DATA.md](DATA.md)); they are not covered as git contents.

## Author

**Klabklaydee Supaporn**  
Fujii Laboratory, Institute of Science Tokyo  
Supervised by Assoc. Prof. Manabu Fujii
