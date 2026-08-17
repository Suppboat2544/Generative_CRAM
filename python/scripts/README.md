# Scripts

Standalone Python utilities.

| Script | Purpose |
|--------|---------|
| `train_cvae.py` | Train the property-conditioned CVAE (`latent_dim=64`) |
| `pubchem_cram_search.py` | Search PubChem for CRAM-like seeds |
| `test_feature_dimensions.py` | Check 163D atom-feature concatenation |
| `van_krevelen_ocean_*.py` | Van Krevelen / ocean DOM plots |
| `cram_visualization_enhanced.py` | Publication-style CRAM figures |
| `generate_results_figures.py` | Result-panel figures |
| `visualize_*structures.py` | Draw representative molecules |
| `export_*` | Table / batch-stat export helpers |

Training CSV and checkpoints come from [Google Drive](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing) (`cram_data`, `models`, `Generated_Structures`). See [DATA.md](../../DATA.md).

Train:

```bash
cd python
python scripts/train_cvae.py --smiles-csv data/cram_molecules.csv --epochs 50 --latent-dim 64
```
