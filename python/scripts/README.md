# Scripts

Standalone Python utilities.

| Script | Purpose |
|--------|---------|
| `train_cvae.py` | Train the property-conditioned CVAE (`latent_dim=64`) |
| `paper_blocking_criticals.py` | Paper A1–A3: encoder μ∈ℝ⁶⁴ Dual-z vs Morgan 128-D PCA, per-dim KL / collapse, 10M funnel. Writes MISSING instead of restoring 72.3 / 69.7 / 87.5. |
| `pubchem_cram_search.py` | Search PubChem for CRAM-like seeds |
| `compare_mutation_smoke.py` | Java vs Julia mutation vs Hertkorn window. Expected ~30% env.; `--seed 42` Java: 50% / 0.05 s CDK, 56% / 0.35 s RDKit. Trained CVAE is better. |
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
python scripts/paper_blocking_criticals.py \
  --checkpoint models/cvae_best.pt \
  --smiles-csv data/cram_molecules.csv \
  --latent-128 data/latent_vectors_128d.npz \
  --out-dir outputs/paper_blocking
```

`paper_blocking_criticals.py` needs the **GRU CVAE** (`decoder.gru`, `concat(z,c)=96`), not Transformer `best_cram_model.pt`. Dual-z is encoder **μ ∈ ℝ⁶⁴** vs the deposited **Morgan residual PCA** (`latent_vectors_128d.npz`), not a PCA of the 256-D graph hidden. Table 1 A is n=10,000 only. Funnel integers come from logs; `10⁷×0.576×0.700 = 4,032,000`, not 5.04M.
