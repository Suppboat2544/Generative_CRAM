# Java edition

Spring Boot REST API and Picocli for CRAM validation and 163D feature extraction (CDK).

Sibling editions: [Python CVAE](../python/README.md) · [Julia CLI](../julia/README.md) · [repo root](../README.md)

SMILES tables for batch validation: download **`cram_data`** from [Google Drive](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing) ([DATA.md](../DATA.md)).

## Requirements

- Java 21+
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
| GET | `/api/v1/criteria` | Get CRAM criteria info |

### Example Requests

```bash
# Validate molecule
curl "http://localhost:8080/api/v1/validate?smiles=CC1CCC(CC1)(C(=O)O)C(=O)O"

# Batch validation
curl -X POST http://localhost:8080/api/v1/validate/batch \
  -H "Content-Type: application/json" \
  -d '{"smilesList": ["CC(=O)O", "CC1CCC(CC1)(C(=O)O)C(=O)O"]}'
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
│   │   └── FeatureExtractor.java       # 163D feature extraction
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
