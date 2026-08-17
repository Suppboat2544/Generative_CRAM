# Generative CRAM

Toolkit for **validating, encoding, and generating Carboxyl-Rich Alicyclic Molecules (CRAM)** — refractory components of marine dissolved organic matter (DOM).

**Design:** [DESIGN.md](DESIGN.md) — editions, CVAE vs mutation, 163D features, smoke timings.  
**Data / weights:** [DATA.md](DATA.md) — Google Drive (`cram_data`, `models`, `Generated_Structures`).

```text
SMILES  →  CRAM criteria  →  163D features  →  generate / serve
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Python       Java        Julia
 CVAE+UI     REST+CLI    CLI+Pkg
```

| Folder | Role | Typical command |
|--------|------|-----------------|
| [`python/`](python/README.md) | **Trained CVAE** (64D latent), Gradio, training | `cd python && python main.py generate --mode vae ...` |
| [`java/`](java/README.md) | Validate, 163D features, mutation, REST | `cd java && java -jar ... smoke -n 40` |
| [`julia/`](julia/README.md) | Validate, 163D features, mutation | `cd julia && julia cli.jl smoke -n 40` |

The deep generator (GAT → 64D latent → SELFIES decoder) lives only in **Python**. Java and Julia do not load `.pt` files.

The **trained 64D CVAE is the better generator**: property-conditioned marine CRAM, no SMILES reject loop, **~100× faster** than mutation. Java/Julia mutation is CRAM-valid by construction but typically only **~30% environmentally plausible** (Hertkorn O/C 0.40–0.60, H/C 1.20–1.70; recorded Java `--seed 42`: 50% in 0.05 s). Details: [DESIGN.md](DESIGN.md#6-mutation-smoke-results).

Notebooks and large binaries are not in git.

---

## 1. CRAM criteria (all languages)

| Criterion | Requirement |
|-----------|-------------|
| Carboxyl groups (COOH / COO⁻) | ≥ 2 |
| Ring system | ≥ 1 alicyclic (non-aromatic) ring |
| Aliphatic carbon | ≥ 60% |
| Molecular weight | 100–750 Da |

Example: `CC1CCC(CC1)(C(=O)O)C(=O)O`

Environmentally plausible overlay (Hertkorn): **O/C 0.40–0.60**, **H/C 1.20–1.70**.

---

## 2. Choose a path

### A. Train or sample the CVAE (Python)

```bash
git clone https://github.com/Suppboat2544/Generative_CRAM.git
cd Generative_CRAM/python
python -m venv .venv && source .venv/bin/activate
pip install -U pip torch
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
pip install -e .
```

Place Drive downloads first (`cram_data` → `python/data/`, `models` → `python/models/`). Then:

```bash
python main.py validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
python main.py generate --mode vae --checkpoint models/cvae_best.pt --count 100
python main.py ui --port 7860
python scripts/train_cvae.py --smiles-csv data/cram_molecules.csv --epochs 50 --latent-dim 64
```

Full guide: [python/README.md](python/README.md)

### B. HTTP API / mutation (Java)

Java 21+, Maven 3.8+. Does **not** run the CVAE.

```bash
cd java
mvn package -DskipTests
java -jar target/cram-generator-1.0.0.jar smoke -n 40 --seed 42
java -jar target/cram-generator-1.0.0.jar --server   # http://localhost:8080
```

Full guide: [java/README.md](java/README.md)

### C. Fast CLI (Julia)

Julia 1.10+.

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. cli.jl smoke -n 40 --seed 42
```

Full guide: [julia/README.md](julia/README.md)

Compare mutation editions (after building the Java JAR):

```bash
cd python
python scripts/compare_mutation_smoke.py -n 40 --seed 42
```

---

## 3. Production CVAE size

| Parameter | Value |
|-----------|-------|
| Atom features | 163D |
| GAT encoder | 3 layers, 4 heads, hidden 256 |
| **Latent** | **64** |
| Condition | 32D |
| Decoder | GRU, 2 layers, hidden 256 |

Architecture, feature groups, and REST map: [DESIGN.md](DESIGN.md).

---

## 4. Layout

```text
Generative_CRAM/
├── README.md                 this file
├── DESIGN.md                 architecture
├── DATA.md                   Drive assets
├── LICENSE
├── python/                   CVAE, UI, training
├── java/                     Spring Boot + CDK
└── julia/                    MolecularGraph.jl
```

---

## Citation

- Hertkorn, N. et al. (2006). *Geochimica et Cosmochimica Acta* — CRAM in marine DOM.
- Krenn, M. et al. (2020). SELFIES.
- Kingma, D. P. & Welling, M. (2013). Auto-Encoding Variational Bayes.

```
Klabklaydee, S. Generative CRAM.
https://github.com/Suppboat2544/Generative_CRAM
```

## License

MIT — [LICENSE](LICENSE)

## Author

**Klabklaydee Supaporn**  
Fujii Laboratory, Institute of Science Tokyo  
Supervised by Assoc. Prof. Manabu Fujii
