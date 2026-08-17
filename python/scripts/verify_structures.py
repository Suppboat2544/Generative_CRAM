#!/usr/bin/env python3
"""
Verify molecular properties for representative CRAM structures
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import pandas as pd

def count_carboxyl_groups(mol):
    """Count carboxyl groups (-COOH)"""
    carboxyl_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    if carboxyl_pattern:
        matches = mol.GetSubstructMatches(carboxyl_pattern)
        return len(matches)
    return 0

def calculate_aliphatic_percentage(mol):
    """Calculate percentage of aliphatic (non-aromatic) carbons"""
    total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    if total_carbons == 0:
        return 0.0
    aromatic_carbons = sum(1 for atom in mol.GetAtoms() 
                          if atom.GetSymbol() == 'C' and atom.GetIsAromatic())
    aliphatic_carbons = total_carbons - aromatic_carbons
    return (aliphatic_carbons / total_carbons) * 100

def count_alicyclic_rings(mol):
    """Count alicyclic (non-aromatic) rings"""
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    
    alicyclic_count = 0
    for ring in atom_rings:
        # Check if all atoms in ring are non-aromatic
        is_alicyclic = all(not mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        if is_alicyclic:
            alicyclic_count += 1
    
    return alicyclic_count

def calculate_oc_ratio(mol):
    """Calculate O/C ratio"""
    c_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    o_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    
    if c_count == 0:
        return 0.0
    return o_count / c_count

def count_rings(mol):
    """Count total number of rings"""
    return rdMolDescriptors.CalcNumRings(mol)

def analyze_molecule(smiles, example_name):
    """Analyze a single molecule and return properties"""
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return {
            'Example': example_name,
            'SMILES': smiles,
            'Valid': False,
            'Error': 'Invalid SMILES'
        }
    
    # Calculate all properties
    mw = Descriptors.MolWt(mol)
    cooh = count_carboxyl_groups(mol)
    aliphatic_pct = calculate_aliphatic_percentage(mol)
    alicyclic_rings = count_alicyclic_rings(mol)
    oc_ratio = calculate_oc_ratio(mol)
    num_rings = count_rings(mol)
    
    # Get molecular formula
    formula = rdMolDescriptors.CalcMolFormula(mol)
    
    return {
        'Example': example_name,
        'SMILES': smiles,
        'Valid': True,
        'Formula': formula,
        'MW (g/mol)': f"{mw:.2f}",
        'COOH': cooh,
        'Aliphatic %': f"{aliphatic_pct:.1f}%",
        'Alicyclic Rings': alicyclic_rings,
        'O/C Ratio': f"{oc_ratio:.2f}",
        'Total Rings': num_rings
    }

# Define the three example structures from the manuscript
structures = [
    {
        'name': 'Example 1: Tricyclohexane-1,3,5-tricarboxylic Acid',
        'smiles': 'C1CC(C(=O)O)C2CC(C(=O)O)CC(C(=O)O)C2C1'
    },
    {
        'name': 'Example 2: Octahydronaphthalene-2,6-dicarboxylic Acid',
        'smiles': 'C2CCC1CCC(C(=O)O)CC1C(C2)C(=O)O'
    },
    {
        'name': 'Example 3: Cyclopentanepentaacetic Acid (CPPA-like)',
        'smiles': 'C1(C(=O)O)C(C(=O)O)C(C(=O)O)C(C(=O)O)C1C(=O)OC2(C(=O)O)C(C(=O)O)C(C(=O)O)C(C(=O)O)C2C(=O)O'
    }
]

print("=" * 100)
print("CRAM Representative Structures - Property Verification")
print("=" * 100)
print()

results = []
for struct in structures:
    result = analyze_molecule(struct['smiles'], struct['name'])
    results.append(result)
    
    print(f"\n{result['Example']}")
    print("-" * 100)
    print(f"SMILES: {result['SMILES']}")
    
    if result['Valid']:
        print(f"Formula: {result['Formula']}")
        print(f"Molecular Weight: {result['MW (g/mol)']} g/mol")
        print(f"Carboxyl Groups (COOH): {result['COOH']}")
        print(f"Aliphatic Content: {result['Aliphatic %']}")
        print(f"Alicyclic Rings: {result['Alicyclic Rings']}")
        print(f"O/C Ratio: {result['O/C Ratio']}")
        print(f"Total Rings: {result['Total Rings']}")
    else:
        print(f"ERROR: {result.get('Error', 'Unknown error')}")

print("\n" + "=" * 100)
print("Summary Table")
print("=" * 100)

# Create DataFrame for nice table display
df_data = []
for result in results:
    if result['Valid']:
        df_data.append({
            'Example': result['Example'].split(':')[0],
            'MW': result['MW (g/mol)'],
            'COOH': result['COOH'],
            'Aliphatic': result['Aliphatic %'],
            'Alicyclic Rings': result['Alicyclic Rings'],
            'O/C': result['O/C Ratio'],
            'Rings': result['Total Rings']
        })

df = pd.DataFrame(df_data)
print(df.to_string(index=False))

print("\n" + "=" * 100)
print("CRAM Compliance Check")
print("=" * 100)

for result in results:
    if result['Valid']:
        cooh_ok = result['COOH'] >= 2
        aliphatic_ok = float(result['Aliphatic %'].rstrip('%')) >= 70.0
        oc_ratio = float(result['O/C Ratio'])
        oc_ok = 0.4 <= oc_ratio <= 0.6
        rings_ok = result['Alicyclic Rings'] >= 1
        
        compliant = all([cooh_ok, aliphatic_ok, oc_ok, rings_ok])
        
        print(f"\n{result['Example']}")
        print(f" COOH ≥ 2: {result['COOH']} {'' if cooh_ok else ''}")
        print(f" Aliphatic ≥ 70%: {result['Aliphatic %']} {'' if aliphatic_ok else ''}")
        print(f" O/C 0.4-0.6: {result['O/C Ratio']} {'' if oc_ok else ''}")
        print(f" Alicyclic Rings ≥ 1: {result['Alicyclic Rings']} {'' if rings_ok else ''}")
        print(f" → CRAM Compliant: {'YES ' if compliant else 'NO '}")

print("\n" + "=" * 100)
