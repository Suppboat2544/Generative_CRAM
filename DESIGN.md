# Design

How Generative CRAM is put together. Install and commands live in [README.md](README.md). Assets live in [DATA.md](DATA.md).

## 1. Purpose

Generate **Carboxyl-Rich Alicyclic Molecules (CRAM)** that sit in the marine DOM region of Hertkorn et al. (2006). Public libraries under-sample this space, so the toolkit both **filters** known SMILES and **proposes** new structures.

Two jobs are separate on purpose:

| Job | What it means | Who does it |
|-----|----------------|-------------|
| Structural CRAM | ≥2 COOH, alicyclic ring, ≥60% aliphatic C, MW 100–750 Da | Python, Java, Julia |
| Environmentally plausible CRAM | Structural CRAM **and** van Krevelen **O/C 0.40–0.60**, **H/C 1.20–1.70** | Scored everywhere; **sampled well only by the trained CVAE** |

Java and Julia mutation can satisfy the first row and still miss the second. The **trained 64D Python CVAE is the designed generator** for marine-like CRAM.

## 2. Layout of editions

```text
                    SMILES
                      │
                      ▼
              CRAM criteria  ─── shared chemistry ───┐
                      │                              │
                      ▼                              │
              163D atom features                     │
                      │                              │
         ┌────────────┼────────────┐                 │
         ▼            ▼            ▼                 │
      python/       java/        julia/              │
         │            │            │                 │
    GAT + CVAE     CDK REST     MolecularGraph       │
    Gradio UI      Picocli      CLI / package        │
    train .pt      mutation     mutation             │
         │            │            │                 │
         └────────────┼────────────┘                 │
                      ▼                              │
              generate / serve                       │
              (CVAE only in python/)                 │
```

| Edition | Designed for | Not designed for |
|---------|--------------|------------------|
| [`python/`](python/README.md) | Train and sample the property-conditioned CVAE; Gradio; SELFIES mutation | Serving a JVM API |
| [`java/`](java/README.md) | Validate, 163D CDK features, REST, mutation generate | Loading `.pt` / training |
| [`julia/`](julia/README.md) | Fast CLI validate / features / mutation | CVAE |

All three share the same structural CRAM rules. Only Python owns **GAT → 64D latent → SELFIES decoder** and checkpoints.

## 3. Generation design

```text
                 ┌─ mutation (Java / Julia / Python SELFIES)
  seed SMILES ───┤     edit string → parse → keep if structural CRAM
                 │     expected ~30% in Hertkorn window (smoke 30–50%)
                 │     reject loop, unconstrained O/C–H/C
                 │
                 └─ trained CVAE (Python only)          ← preferred
                       condition on n_COOH, ring class, MW, …
                       decode SELFIES in batch
                       no validation loop
                       ~100× faster than mutation reject loops
```

**Why the trained model is better**

1. It is **property-conditioned** (carboxyl count, ring class, aliphatic fraction, MW, DBE ratios), so samples aim at marine CRAM instead of wandering around dicarboxylic cycloalkanes.
2. It **does not mutate SMILES text**, so it avoids CDK-accepted / RDKit-rejected valence errors seen in the Java smoke (4/20 on `--seed 42`).
3. It generates in **batches** from \(z \sim \mathcal{N}(\mu,\sigma^2)\) plus a 32D condition; mutation must try many invalid strings (Java: 170 attempts for 40 CRAM-valid).

Use Java/Julia mutation for services and smoke tests. Use **`python/` + Drive `models/`** when you need environmentally plausible CRAM.

## 4. Python CVAE (production)

Trained latent size is **64** (not 128).

```text
SMILES → RDKit mol → DGL graph
       → 163D / atom → Linear 163→256
       → GAT × 3, 4 heads × 64, residual + LayerNorm
       → gated mean pool → 256 → μ, log σ² → z ∈ R⁶⁴
       → property MLP → c ∈ R³²
       → concat 96 → GRU 256 × 2 → SELFIES → SMILES
```

| Piece | Size |
|-------|------|
| Atom features | 163 |
| GAT width | 256 |
| GAT layers / heads / per-head | 3 / 4 / 64 |
| Pooling | gated mean (`σ(Linear(256→1))` then `mean_nodes`) |
| **Latent z** | **64** |
| Property concat | 35 (emb 8+16+4 + continuous 7) |
| Condition MLP hidden | 64 |
| **Condition c** | **32** |
| GRU | 256 hidden, 2 layers |
| Token embed / output | 256 ↔ vocab |
| Max length | 100 (CVAE) / 150 (tokenizer) |
| Special tokens | PAD=0, SOS=1, EOS=2, UNK=3 |

Vocabulary size is **4 + unique SELFIES tokens** from the training CSV (saved with the checkpoint).

Property embeddings: `n_cooh` 7×8, ring class 6×16, ring-size class 3×4, plus 7 continuous values (aliphatic %, MW, benzene, DBE/C, DBE/O, DBE/H, mean ring size).

Training: `python/scripts/train_cvae.py --latent-dim 64`. Sampling: `python main.py generate --mode vae --checkpoint models/<file>.pt`. Weights are **not** in git — [DATA.md](DATA.md).

### 163D atom features

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

Python (RDKit), Java (CDK), and Julia (MolecularGraph) all emit 163D vectors with the same group boundaries. Implementations are not bit-identical; they are aligned for the same GNN width.

## 5. Java and Julia

```text
java/     Spring Boot 3.4 + CDK 2.9 + Picocli, Java 21
          CLI: validate | features | generate | smoke | info | --server
          REST: /api/v1/{health,validate,features,generate,criteria}

julia/    MolecularGraph.jl
          CLI: validate | features | generate | smoke | info
```

Mutation (both): start from alicyclic dicarboxylic seeds, insert alkyl / carboxyl fragments, keep canonical SMILES that pass the structural CRAM validator.

Environmental overlay (both): Hertkorn window O/C 0.40–0.60, H/C 1.20–1.70. Smoke commands print **pass rate and wall time** and state that the trained CVAE is better.

## 6. Mutation smoke results

| Check | n | Hertkorn window | Time |
|-------|---|-----------------|------|
| Java CLI `smoke` (CDK) | 40 | 20/40 = **50%** | **0.05 s** (824 mol/s) |
| Java JUnit | 40 | 50% | 0.05 s |
| RDKit referee (`compare_mutation_smoke.py --java-only`) | 20 | 9/16 = **56%** (4 unparseable) | **0.35 s** wall (JVM start) |
| Julia smoke | 40 | same protocol | needs Julia 1.10+ |
| **Trained Python CVAE** | — | **better** (conditioned) | **~100×** vs mutation reject loops |

Expected mutation window rate is **~30%** (this smoke often 30–50%). That is a design limit of unconstrained SMILES edits, not a failed test.

```bash
cd java && mvn test && java -jar target/cram-generator-1.0.0.jar smoke -n 40 --seed 42
cd julia && julia --project=. cli.jl smoke -n 40 --seed 42
cd python && python scripts/compare_mutation_smoke.py -n 40 --seed 42
```

## 7. Data flow

```text
Google Drive
  cram_data              →  python/data/           training SMILES
  models (*.pt, 64D)     →  python/models/         CVAE weights
  Generated_Structures   →  python/outputs/Generated_Structures/
```

Git holds source only (no notebooks, no `.pt`, no `java/target`).

## 8. Repository map

```text
Generative_CRAM/
├── README.md                 start here
├── DESIGN.md                 this file
├── DATA.md                   Drive copy instructions
├── LICENSE
├── python/
│   ├── main.py               validate / generate / features / ui
│   ├── config/settings.py    latent_dim=64, 163D, CRAM rules
│   ├── src/models/           encoder, decoder, cvae, condition, losses
│   ├── src/generators/       CVAE sampler + SELFIES mutation
│   ├── src/features/         163D
│   ├── src/validator.py
│   ├── ui/                   Gradio
│   └── scripts/              train_cvae.py, compare_mutation_smoke.py
├── java/                     CDK + Spring + Picocli
│   └── src/test/.../MutationEnvironmentalSmokeTest.java
└── julia/                    MolecularGraph.jl
    ├── cli.jl
    └── test/smoke.jl
```

## 9. References

- Hertkorn, N. et al. (2006). *Geochimica et Cosmochimica Acta* — CRAM in marine DOM.
- Krenn, M. et al. (2020). SELFIES.
- Kingma, D. P. & Welling, M. (2013). Auto-Encoding Variational Bayes.
- Veličković, P. et al. (2018). Graph Attention Networks.
