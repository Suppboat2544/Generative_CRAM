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
| [`python/`](python/README.md) | Property-conditioned CVAE (64D latent), Gradio UI, training | `cd python && python main.py validate "..."` |
| [`java/`](java/README.md) | Spring Boot REST API + Picocli | `cd java && mvn spring-boot:run` |
| [`julia/`](julia/README.md) | MolecularGraph.jl validation, features, mutation | `cd julia && julia cli.jl info` |

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

### B. HTTP API (Java)

Requires **Java 21** and Maven 3.8+.

```bash
cd java
mvn clean package -DskipTests
java -jar target/cram-generator-1.0.0.jar validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
java -jar target/cram-generator-1.0.0.jar --server   # http://localhost:8080
```

```bash
curl "http://localhost:8080/api/v1/validate?smiles=CC1CCC(CC1)(C(=O)O)C(=O)O"
```

Full Java guide: [java/README.md](java/README.md)

### C. Fast CLI / package (Julia)

Requires **Julia 1.10+**.

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. cli.jl validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
julia --project=. cli.jl generate -n 50 -o generated.csv
```

Full Julia guide: [julia/README.md](julia/README.md)

---

## 3. Python CVAE (trained size)

Production latent space is **64**, not 128.

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

## 4. Layout

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
