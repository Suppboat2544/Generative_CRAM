#!/usr/bin/env python3
"""Validate example CRAM / non-CRAM SMILES."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validator import CRAMValidator

EXAMPLES = [
    "CC1CCC(CC1)(C(=O)O)C(=O)O",  # alicyclic dicarboxylic acid
    "OC(=O)C1CCCCC1C(=O)O",
    "c1ccccc1",                   # benzene
    "CC(=O)O",                    # acetic acid
]


def main():
    validator = CRAMValidator()
    for smiles in EXAMPLES:
        r = validator.validate_smiles(smiles)
        flag = "CRAM" if r["is_cram"] else "not CRAM"
        print(f"{smiles}")
        print(
            f"  {flag} | COOH={r['n_cooh']} | alicyclic={r['has_alicyclic']} "
            f"| aliphatic={r['aliphatic_pct']:.0%} | MW={r['mw']:.1f}"
        )
        if r["failures"]:
            print("  " + "; ".join(r["failures"]))
        print()


if __name__ == "__main__":
    main()
