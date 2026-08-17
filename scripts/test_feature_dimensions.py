#!/usr/bin/env python3
"""
Test script to verify DeepCRAMFeatureExtractor produces 163D features.

This script checks:
1. EnhancedOxidationFeatures is available
2. DeepCRAMFeatureExtractor can be instantiated
3. Feature dimensions are correct (163D)
4. Features can be extracted from a test molecule
"""

import sys

print("="*70)
print("FEATURE DIMENSION TEST")
print("="*70)

# Test 1: Check if EnhancedOxidationFeatures exists
print("\n[1/4] Checking EnhancedOxidationFeatures...")
try:
    from rdkit import Chem
    import numpy as np
    
    # Check if class is defined (you need to run cell 72 first)
    if 'EnhancedOxidationFeatures' in dir():
        print("   ✅ EnhancedOxidationFeatures found in namespace")
    else:
        print("   ❌ EnhancedOxidationFeatures NOT FOUND")
        print("   → You need to run Cell 72 first!")
        sys.exit(1)
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check if DeepCRAMFeatureExtractor exists
print("\n[2/4] Checking DeepCRAMFeatureExtractor...")
if 'DeepCRAMFeatureExtractor' in dir():
    print("   ✅ DeepCRAMFeatureExtractor found in namespace")
else:
    print("   ❌ DeepCRAMFeatureExtractor NOT FOUND")
    print("   → You need to run Cell 67 first!")
    sys.exit(1)

# Test 3: Instantiate and check dimensions
print("\n[3/4] Instantiating DeepCRAMFeatureExtractor...")
try:
    extractor = DeepCRAMFeatureExtractor()
    total_dim = extractor.TOTAL_ATOM_DIM
    
    print(f"   Total feature dimension: {total_dim}D")
    print(f"   Feature breakdown:")
    for name, dim in extractor.FEATURE_DIMS.items():
        print(f"      - {name}: {dim}D")
    
    if total_dim == 163:
        print("   ✅ Correct dimension (163D)")
    else:
        print(f"   ❌ WRONG dimension! Expected 163D, got {total_dim}D")
        print("   → EnhancedOxidationFeatures may not be integrated")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Error instantiating: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Extract features from test molecule
print("\n[4/4] Testing feature extraction...")
test_smiles = "CC(C(=O)O)CC(C(=O)O)O"  # Test CRAM molecule
try:
    mol = Chem.MolFromSmiles(test_smiles)
    if mol is None:
        print(f"   ❌ Could not parse SMILES: {test_smiles}")
        sys.exit(1)
    
    features = extractor.extract_all_features(mol)
    atom_features = features['atom_features']
    
    print(f"   Test molecule: {test_smiles}")
    print(f"   Number of atoms: {mol.GetNumAtoms()}")
    print(f"   Feature shape: {atom_features.shape}")
    
    if atom_features.shape[1] == 163:
        print("   ✅ Features extracted successfully (163D per atom)")
    else:
        print(f"   ❌ Wrong feature dimension! Expected 163D, got {atom_features.shape[1]}D")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Error extracting features: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYour DeepCRAMFeatureExtractor is correctly configured for 163D features.")
print("You can now run the training cell.")
