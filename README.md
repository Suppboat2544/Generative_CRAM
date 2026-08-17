# Generative CRAM

Toolkit for **validating, encoding, and generating Carboxyl-Rich Alicyclic Molecules (CRAM)** — refractory components of marine dissolved organic matter (DOM).

Three nested language editions share the same chemistry rules. Start with Python if you need the CVAE; use Java for a REST API; use Julia for fast CLI validation and mutation.

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
| [`python/`](python/README.md) | **CVAE** (64D latent), Gradio, training | `cd python && python main.py generate --mode vae ...` |
| [`java/`](java/README.md) | Validate, 163D features, **mutation** generate, REST | `cd java && java -jar ... generate -n 50` |
| [`julia/`](julia/README.md) | Validate, 163D features, **mutation** generate | `cd julia && julia cli.jl generate -n 50` |

The deep generator (GAT encoder → 64D latent → SELFIES decoder) lives only in **Python**. Java and Julia do not load `.pt` checkpoints.

Mutation in Java/Julia is CRAM-valid by construction but typically only **~30% environmentally plausible** (Hertkorn O/C 0.40–0.60, H/C 1.20–1.70). Property-conditioned CVAE is the path for higher rates.

Compare:

```bash
# from repo root, after building java/target/cram-generator-1.0.0.jar
cd python
python scripts/compare_mutation_smoke.py -n 40 --seed 42
```

Notebooks and large binaries are not in git. Download **data, models, and generated structures** from [Google Drive](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing) — see [DATA.md](DATA.md).

---

## 1. CRAM criteria (all languages)

| Criterion | Requirement |
|-----------|-------------|
| Carboxyl groups (COOH / COO⁻) | ≥ 2 |
| Ring system | ≥ 1 alicyclic (non-aromatic) ring |
| Aliphatic carbon | ≥ 60% |
| Molecular weight | 100–750 Da |

Example SMILES:

```text
CC1CCC(CC1)(C(=O)O)C(=O)O
```

---

## 2. Choose a path

### A. Generate or train (Python)

```bash
git clone https://github.com/Suppboat2544/Generative_CRAM.git
cd Generative_CRAM/python
python -m venv .venv && source .venv/bin/activate
pip install -U pip torch
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
pip install -e .

python main.py validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
python main.py generate --mode mutation --count 100 --output outputs/cram_mutants.csv
python main.py ui --port 7860
```

Train (latent dim **64**, matching the production checkpoint):

```bash
cd python
python scripts/train_cvae.py --smiles-csv data/cram_molecules.csv --epochs 50 --latent-dim 64
```

Full Python guide: [python/README.md](python/README.md)

Place Drive downloads first (`cram_data` → `python/data/`, `models` → `python/models/`). Details: [DATA.md](DATA.md).

### B. HTTP API and mutation generate (Java)

Requires **Java 21** and Maven 3.8+. Does **not** run the CVAE.

```bash
cd java
mvn clean package -DskipTests
java -jar target/cram-generator-1.0.0.jar validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
java -jar target/cram-generator-1.0.0.jar features "CC1CCC(CC1)(C(=O)O)C(=O)O"
java -jar target/cram-generator-1.0.0.jar generate -n 50
java -jar target/cram-generator-1.0.0.jar smoke -n 40 --seed 42
java -jar target/cram-generator-1.0.0.jar --server   # http://localhost:8080
```

```bash
curl "http://localhost:8080/api/v1/validate?smiles=CC1CCC(CC1)(C(=O)O)C(=O)O"
curl "http://localhost:8080/api/v1/features?smiles=CC1CCC(CC1)(C(=O)O)C(=O)O"
curl -X POST http://localhost:8080/api/v1/generate -H "Content-Type: application/json" -d '{"n":20}'
```

Full Java guide: [java/README.md](java/README.md)

### C. Fast CLI / package (Julia)

Requires **Julia 1.10+**.

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. cli.jl validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
julia --project=. cli.jl generate -n 50 -o generated.csv
julia --project=. cli.jl smoke -n 40 --seed 42
```

Full Julia guide: [julia/README.md](julia/README.md)

---

## 3. Python CVAE (trained size)

Production latent space is **64**.

```text
SMILES → graph (163D atom features)
       → GAT encoder (hidden 256)
       → z ~ N(μ, σ²)     64D
       → condition          32D
       → GRU SELFIES decoder
       → SMILES
```

| Parameter | Value |
|-----------|-------|
| Atom features | 163D |
| Hidden | 256 |
| **Latent** | **64** |
| Condition | 32 |

---

## 4. Mutation smoke (Java / Julia)

Environmentally plausible = structural CRAM **and** Hertkorn van Krevelen window **O/C 0.40–0.60**, **H/C 1.20–1.70**. Mutation is not property-conditioned, so only about **30%** of CRAM-valid outputs are expected in that window (this smoke often lands 30–50%). CVAE stays in `python/`.

Reference seed `CC1CCC(CC1)(C(=O)O)C(=O)O`: O/C **0.44**, H/C **1.56**, MW **186.2 Da** (in-window).

Recorded runs (`--seed 42`):

| Check | n | Result |
|-------|---|---------|
| Java CLI `smoke` (CDK) | 40 | 40 CRAM-valid; **20/40 = 50%** environmentally plausible |
| Java JUnit `MutationEnvironmentalSmokeTest` | 40 | same 50% window rate |
| Python `compare_mutation_smoke.py --java-only` (RDKit referee) | 20 | 16 parseable, 4 invalid valence; **9/16 = 56%** in-window |
| Julia `cli.jl smoke` / `test/smoke.jl` | 40 | same protocol; run where Julia 1.10+ is installed |

RDKit may reject a few CDK-accepted mutants (explicit oxygen valence). That is expected for unconstrained SMILES mutation.

```bash
cd java && mvn test && java -jar target/cram-generator-1.0.0.jar smoke -n 40 --seed 42
cd julia && julia --project=. cli.jl smoke -n 40 --seed 42
cd python && python scripts/compare_mutation_smoke.py -n 40 --seed 42
```

---

## 5. Layout

```text
Generative_CRAM/
├── README.md                 this file
├── DATA.md                   Google Drive: cram_data, models, Generated_Structures
├── LICENSE
├── python/                   CVAE, UI, training, analysis scripts
│   ├── main.py
│   ├── src/{models,generators,features,validator,tokenizer,...}
│   ├── ui/
│   ├── scripts/
│   └── README.md
├── java/                     Spring Boot + CDK
│   ├── pom.xml
│   └── src/main/java/com/cram/generator/
└── julia/                    MolecularGraph.jl
    ├── Project.toml
    ├── cli.jl
    ├── test/smoke.jl
    └── src/
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
