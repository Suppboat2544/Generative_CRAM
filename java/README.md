# Java edition

Validate CRAM molecules, extract 163D CDK features, and **mutation-generate** SMILES.  
The property-conditioned **CVAE is Python-only** (`python/`).

Mutation is CRAM-valid by construction but typically only **~30% environmentally plausible**
(Hertkorn O/C 0.40–0.60, H/C 1.20–1.70). Recorded `--seed 42`: Java CLI **20/40 = 50%** in **0.05 s**
(CDK); RDKit comparison **9/16 = 56%** in **0.35 s** wall (4/20 unparseable).
The **trained Python CVAE (64D) is better** (property-conditioned, ~100× faster).
Compare with Julia via `python/scripts/compare_mutation_smoke.py`.

Sibling editions: [Python CVAE](../python/README.md) · [Julia CLI](../julia/README.md) · [repo root](../README.md) · [design](../DESIGN.md)

SMILES tables for batch validation: download **`cram_data`** from [Google Drive](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing) ([DATA.md](../DATA.md)).

## Requirements

- Java 21+ (tested with Temurin 25)
- Maven 3.8+

## Quick Start

### Build

```bash
cd java   # this folder
mvn clean package -DskipTests
```

### Run as CLI

```bash
# Validate a molecule
java -jar target/cram-generator-1.0.0.jar validate "CC1CCC(CC1)(C(=O)O)C(=O)O"

# 163D features
java -jar target/cram-generator-1.0.0.jar features "CC1CCC(CC1)(C(=O)O)C(=O)O" -v

# Mutation generate (not CVAE)
java -jar target/cram-generator-1.0.0.jar generate -n 50 -o generated.csv

# Smoke: validate + 163D features + generate; reports ~30% Hertkorn window
java -jar target/cram-generator-1.0.0.jar smoke -n 40 --seed 42
mvn test

# Batch validation
java -jar target/cram-generator-1.0.0.jar batch input.txt -o results.csv

# Show CRAM criteria
java -jar target/cram-generator-1.0.0.jar info
```

### Run as REST API Server

```bash
java -jar target/cram-generator-1.0.0.jar --server
# or
mvn spring-boot:run
```

Server runs at `http://localhost:8080`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/validate?smiles=...` | Validate single SMILES |
| POST | `/api/v1/validate` | Validate SMILES (JSON body) |
| POST | `/api/v1/validate/batch` | Batch validation |
| GET | `/api/v1/features?smiles=...` | 163D molecule-level features |
| POST | `/api/v1/generate` | Mutation generate (`{"n":50}`) |
| GET | `/api/v1/criteria` | CRAM criteria |

### Example Requests

```bash
# Validate molecule
curl "http://localhost:8080/api/v1/validate?smiles=CC1CCC(CC1)(C(=O)O)C(=O)O"

curl "http://localhost:8080/api/v1/features?smiles=CC1CCC(CC1)(C(=O)O)C(=O)O"
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"n": 20}'
```

### Example Response

```json
{
  "smiles": "CC1CCC(CC1)(C(=O)O)C(=O)O",
  "canonicalSmiles": "CC1CCC(C(=O)O)(CC1)C(=O)O",
  "valid": true,
  "carboxylCount": 2,
  "hasAlicyclicRing": true,
  "aliphaticPercent": 100.0,
  "molecularWeight": 186.21,
  "cramScore": 0.95,
  "properties": {
    "atomCount": 12,
    "bondCount": 12,
    "hcRatio": 1.75,
    "ocRatio": 0.5
  }
}
```

## Project Structure

```
java/
├── pom.xml                          # Maven configuration
├── README.md                        # This file
├── src/main/java/com/cram/generator/
│   ├── CramGeneratorApplication.java   # Main application
│   ├── CramCli.java                    # CLI commands
│   ├── model/
│   │   ├── Molecule.java               # Molecule entity
│   │   ├── ValidationResult.java       # Validation result
│   │   └── CramCriteria.java           # CRAM criteria constants
│   ├── service/
│   │   ├── CramValidator.java          # Validation logic
│   │   ├── FeatureExtractor.java       # 163D CDK features
│   │   └── MutationGenerator.java      # SMILES mutation (not CVAE)
│   └── controller/
│       └── CramController.java         # REST API endpoints
└── src/main/resources/
    └── application.properties          # Configuration
```

## CRAM Criteria

| Criterion | Requirement |
|-----------|-------------|
| Carboxyl groups | ≥2 (COOH or COO⁻) |
| Ring structure | Alicyclic (non-aromatic) |
| Aliphatic carbon | >60% |
| Molecular weight | 100-750 Da |

## Dependencies

- **Spring Boot 3.4** - Web framework
- **CDK 2.9** - Chemistry Development Kit for molecule handling
- **Picocli** - CLI framework
- **Lombok** - Boilerplate reduction

## Docker

```bash
cd java
docker build -t cram-generator-java .
docker run -p 8080:8080 cram-generator-java
```

## License

MIT License — see [../LICENSE](../LICENSE)
