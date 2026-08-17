#!/usr/bin/env python3
"""Compare Java vs Julia mutation smoke tests.

Both editions generate CRAM-valid SMILES by mutation (not CVAE).
Environmentally plausible = structural CRAM plus Hertkorn van Krevelen:

    O/C 0.40–0.60, H/C 1.20–1.70

Mutation typically lands only ~30% of CRAM-valid outputs in that window
(this smoke often lands 30–50%; Java --seed 42 recorded 50% CDK / 56% RDKit).
The trained Python CVAE (64D) is better: property-conditioned, samples
marine-like CRAM directly, and is ~100× faster (no reject loop).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JAVA_DIR = REPO / "java"
JULIA_DIR = REPO / "julia"
JAR = JAVA_DIR / "target" / "cram-generator-1.0.0.jar"

ENV_OC = (0.40, 0.60)
ENV_HC = (1.20, 1.70)
EXPECTED_ENV_PCT = 30
SEED = "CC1CCC(CC1)(C(=O)O)C(=O)O"
CVAE_NOTE = (
    "The trained Python CVAE (64D) is better than Java/Julia mutation: "
    "property-conditioned marine CRAM, ~100× faster, no reject loop."
)


def score_smiles(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    n_o = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    if n_c == 0:
        return None
    formula = rdMolDescriptors.CalcMolFormula(mol)
    n_h = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            n_h += 1
        else:
            n_h += atom.GetTotalNumHs()
    oc = n_o / n_c
    hc = n_h / n_c
    env = ENV_OC[0] <= oc <= ENV_OC[1] and ENV_HC[0] <= hc <= ENV_HC[1]
    return {
        "smiles": smiles,
        "formula": formula,
        "oc": oc,
        "hc": hc,
        "mw": Descriptors.MolWt(mol),
        "env": env,
    }


def load_smiles(path: Path) -> list[str]:
    text = path.read_text().strip().splitlines()
    if not text:
        return []
    if "," in text[0] or text[0].lower().startswith("smiles"):
        with path.open() as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames and "smiles" in [c.lower() for c in reader.fieldnames]:
                key = next(c for c in reader.fieldnames if c.lower() == "smiles")
                return [row[key].strip() for row in reader if row.get(key)]
    return [line.strip() for line in text if line.strip() and not line.startswith("#")]


def timed_run(cmd: list[str], cwd: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=cwd)
    return time.perf_counter() - t0


def run_java(n: int, out_csv: Path, seed: int) -> float | None:
    if not JAR.exists():
        print(f"skip Java: missing {JAR} (run: cd java && mvn package -DskipTests)")
        return None
    cmd = [
        "java", "-jar", str(JAR), "generate",
        "-n", str(n), "-o", str(out_csv),
        "--max-attempts", "8000",
        "--seed", str(seed),
    ]
    return timed_run(cmd, JAVA_DIR)


def run_julia(n: int, out_csv: Path, seed: int) -> float | None:
    if shutil.which("julia") is None:
        print("skip Julia: julia not on PATH")
        return None
    cmd = [
        "julia", "--project=.", "cli.jl", "generate",
        "-n", str(n), "-o", str(out_csv),
        "--max-attempts", "8000",
        "--seed", str(seed),
    ]
    return timed_run(cmd, JULIA_DIR)


def summarize(name: str, smiles: list[str], seconds: float) -> dict:
    rows = []
    skipped = 0
    for s in smiles:
        scored = score_smiles(s)
        if scored is None:
            skipped += 1
            continue
        rows.append(scored)
    env = sum(1 for r in rows if r["env"])
    n = len(rows)
    pct = 0.0 if n == 0 else 100.0 * env / n
    rate = 0.0 if seconds <= 0 else len(smiles) / seconds
    print(f"\n{name}")
    print(f"  CRAM-valid generated : {len(smiles)}")
    print(f"  scored (RDKit)       : {n}  (unparseable {skipped})")
    print(f"  environmentally ok   : {env}/{n} = {pct:.0f}%")
    print(f"  time                 : {seconds:.2f} s ({rate:.1f} mol/s)")
    print(f"  expected for mutation: ~{EXPECTED_ENV_PCT}% (this smoke often lands 30–50%)")
    return {"name": name, "n": n, "env": env, "pct": pct, "seconds": seconds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--java-only", action="store_true")
    parser.add_argument("--julia-only", action="store_true")
    args = parser.parse_args()

    seed_score = score_smiles(SEED)
    if seed_score is None or not seed_score["env"]:
        print("reference seed failed RDKit environmental window", file=sys.stderr)
        return 1
    print("Reference seed in Hertkorn window:", SEED)
    print(f"  O/C={seed_score['oc']:.2f}  H/C={seed_score['hc']:.2f}  MW={seed_score['mw']:.1f}")
    print(
        f"\nMutation (Java/Julia) is expected to pass only ~{EXPECTED_ENV_PCT}% "
        "environmentally plausible."
    )
    print(CVAE_NOTE)

    summaries = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if not args.julia_only:
            java_csv = tmp_path / "java.csv"
            elapsed = run_java(args.n, java_csv, args.seed)
            if elapsed is not None and java_csv.exists():
                summaries.append(summarize("Java mutation", load_smiles(java_csv), elapsed))
        if not args.java_only:
            julia_csv = tmp_path / "julia.csv"
            elapsed = run_julia(args.n, julia_csv, args.seed)
            if elapsed is not None and julia_csv.exists():
                summaries.append(summarize("Julia mutation", load_smiles(julia_csv), elapsed))

    if not summaries:
        print("No editions ran. Build the Java JAR and/or install Julia.", file=sys.stderr)
        return 1

    print("\nComparison")
    print(f"{'edition':<18} {'CRAM n':>8} {'env n':>8} {'env %':>8} {'time s':>8}  expected")
    for row in summaries:
        print(
            f"{row['name']:<18} {row['n']:>8} {row['env']:>8} {row['pct']:>7.0f}% "
            f"{row['seconds']:>7.2f}  ~{EXPECTED_ENV_PCT}%"
        )
    print("\nMutation generators typically pass only ~30% environmentally plausible.")
    print(CVAE_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
