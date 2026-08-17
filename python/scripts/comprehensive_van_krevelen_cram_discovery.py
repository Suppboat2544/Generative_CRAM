#!/usr/bin/env python3
"""
Comprehensive Van Krevelen Analysis: CRAM Discovery in Ocean DOM
Creates new CRAM candidate database from compounds in CRAM region
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

# Publication quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5

def smiles_to_molecular_properties(smiles):
    """
    Convert SMILES to molecular properties including H/C and O/C ratios
    Returns: dict with properties or None if invalid
    """
    if pd.isna(smiles) or smiles == ' ' or smiles == 'N/A':
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Count atoms
        formula = rdMolDescriptors.CalcMolFormula(mol)
        
        # Parse formula
        C_count = 0
        H_count = 0
        O_count = 0
        N_count = 0
        
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'C':
                C_count += 1
            elif symbol == 'H':
                H_count += atom.GetTotalNumHs() + 1
            elif symbol == 'O':
                O_count += 1
            elif symbol == 'N':
                N_count += 1
        
        # Fix hydrogen count
        H_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
        
        if C_count == 0:
            return None
        
        # Calculate ratios
        H_C = H_count / C_count
        O_C = O_count / C_count
        
        # Additional CRAM-relevant properties
        num_rings = Descriptors.RingCount(mol)
        num_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        
        # Count carboxyl groups (COOH)
        carboxyl_pattern = Chem.MolFromSmarts('C(=O)[OH]')
        num_carboxyl = len(mol.GetSubstructMatches(carboxyl_pattern)) if carboxyl_pattern else 0
        
        # Aliphatic percentage
        aliphatic_pct = (num_aliphatic_rings / num_rings * 100) if num_rings > 0 else 0
        
        return {
            'SMILES': smiles,
            'Formula': formula,
            'C': C_count,
            'H': H_count,
            'O': O_count,
            'N': N_count,
            'H/C': H_C,
            'O/C': O_C,
            'MW': Descriptors.MolWt(mol),
            'Rings': num_rings,
            'Aliphatic_Rings': num_aliphatic_rings,
            'Aromatic_Rings': num_aromatic_rings,
            'COOH_groups': num_carboxyl,
            'Aliphatic_pct': aliphatic_pct
        }
    except:
        return None

def is_cram_candidate(props):
    """
    Strict CRAM criteria classification
    Returns: (is_CRAM, CRAM_score, category)
    """
    if props is None:
        return False, 0, 'Invalid'
    
    H_C = props['H/C']
    O_C = props['O/C']
    
    # CRAM region criteria
    in_OC_range = 0.4 <= O_C <= 0.6
    in_HC_range = 1.2 <= H_C <= 2.0
    has_carboxyl = props['COOH_groups'] >= 2
    has_rings = props['Rings'] >= 1
    aliphatic_rich = props['Aliphatic_Rings'] >= 1
    
    # Calculate CRAM score (0-100)
    score = 0
    if in_OC_range:
        score += 30
    if in_HC_range:
        score += 30
    if has_carboxyl:
        score += 20
    if has_rings:
        score += 10
    if aliphatic_rich:
        score += 10
    
    # Classification
    if score >= 80:
        category = 'High-confidence CRAM'
        is_cram = True
    elif score >= 60:
        category = 'Probable CRAM'
        is_cram = True
    elif score >= 40:
        category = 'CRAM-like'
        is_cram = False
    else:
        category = 'Non-CRAM'
        is_cram = False
    
    return is_cram, score, category

def load_and_process_ocean_data(tsv_file, label):
    """
    Load ocean sample data and extract molecular properties from SMILES
    """
    print(f"\n{'='*70}")
    print(f"Processing: {label}")
    print(f"File: {tsv_file}")
    print(f"{'='*70}")
    
    df = pd.read_csv(tsv_file, sep='\t')
    print(f"Total rows: {len(df)}")
    
    # Check for Smiles column
    if 'Smiles' not in df.columns:
        print(f"  ⚠️  WARNING: No 'Smiles' column found!")
        print(f"  Available columns: {', '.join(df.columns[:10])}...")
        return pd.DataFrame()
    
    results = []
    valid_count = 0
    
    for idx, row in df.iterrows():
        smiles = row.get('Smiles', None)
        props = smiles_to_molecular_properties(smiles)
        
        if props:
            valid_count += 1
            is_cram, cram_score, category = is_cram_candidate(props)
            
            result = props.copy()
            result.update({
                'Source': label,
                'Compound_Name': row.get('Compound_Name', 'Unknown'),
                'MQScore': row.get('MQScore', 0),
                'is_CRAM': is_cram,
                'CRAM_score': cram_score,
                'CRAM_category': category,
                'SpecMZ': row.get('SpecMZ', 0)
            })
            results.append(result)
    
    result_df = pd.DataFrame(results)
    
    print(f"\n📊 Statistics:")
    print(f"  Valid SMILES processed: {valid_count}")
    print(f"  High-confidence CRAM: {(result_df['CRAM_category'] == 'High-confidence CRAM').sum()}")
    print(f"  Probable CRAM: {(result_df['CRAM_category'] == 'Probable CRAM').sum()}")
    print(f"  CRAM-like: {(result_df['CRAM_category'] == 'CRAM-like').sum()}")
    print(f"  Non-CRAM: {(result_df['CRAM_category'] == 'Non-CRAM').sum()}")
    
    return result_df

def create_comprehensive_van_krevelen(surface_df, deep_df, output_file):
    """
    Create comprehensive Van Krevelen diagram with CRAM candidate identification
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Define axes
    ax1 = fig.add_subplot(gs[0, :2])  # Main Van Krevelen
    ax2 = fig.add_subplot(gs[0, 2])   # CRAM score distribution
    ax3 = fig.add_subplot(gs[1, 0])   # Surface detail
    ax4 = fig.add_subplot(gs[1, 1])   # Deep detail
    ax5 = fig.add_subplot(gs[1, 2])   # Statistics comparison
    
    # Combine data
    all_data = pd.concat([surface_df, deep_df], ignore_index=True)
    
    # Color mapping
    colors = {
        'High-confidence CRAM': '#E63946',
        'Probable CRAM': '#F77F00',
        'CRAM-like': '#FCBF49',
        'Non-CRAM': '#90A4AE'
    }
    
    # SUBPLOT 1: Main Van Krevelen with all data
    print("\n📊 Plotting main Van Krevelen diagram...")
    
    for category in ['Non-CRAM', 'CRAM-like', 'Probable CRAM', 'High-confidence CRAM']:
        cat_data = all_data[all_data['CRAM_category'] == category]
        if len(cat_data) > 0:
            size = 20 if category == 'Non-CRAM' else 60
            marker = 'o' if category == 'Non-CRAM' else 'D'
            alpha = 0.4 if category == 'Non-CRAM' else 0.8
            zorder = 1 if category == 'Non-CRAM' else 5
            
            ax1.scatter(cat_data['O/C'], cat_data['H/C'],
                       c=colors[category], s=size, alpha=alpha,
                       marker=marker, edgecolors='white', linewidths=0.5,
                       label=f'{category} (n={len(cat_data)})',
                       zorder=zorder)
    
    # CRAM region box
    cram_box = Rectangle((0.4, 1.2), 0.2, 0.8, 
                         linewidth=3, edgecolor='red', 
                         facecolor='yellow', alpha=0.15, linestyle='--')
    ax1.add_patch(cram_box)
    
    # Add region labels
    ax1.text(0.15, 2.3, 'Lipids', fontsize=11, style='italic', alpha=0.6, fontweight='bold')
    ax1.text(0.65, 2.3, 'Proteins', fontsize=11, style='italic', alpha=0.6, fontweight='bold')
    ax1.text(0.5, 1.6, 'CRAM\nREGION', fontsize=13, fontweight='bold', 
            color='red', ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax1.text(0.75, 0.8, 'Lignin', fontsize=11, style='italic', alpha=0.6, fontweight='bold')
    ax1.text(0.95, 1.4, 'Tannins', fontsize=11, style='italic', alpha=0.6, fontweight='bold')
    
    ax1.set_xlabel('O/C Ratio', fontsize=14, fontweight='bold')
    ax1.set_ylabel('H/C Ratio', fontsize=14, fontweight='bold')
    ax1.set_title('(A) Van Krevelen Diagram: Ocean DOM with CRAM Candidate Identification', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlim(0, 1.3)
    ax1.set_ylim(0.5, 2.6)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # SUBPLOT 2: CRAM Score Distribution
    cram_candidates = all_data[all_data['CRAM_score'] >= 40]
    if len(cram_candidates) > 0:
        ax2.hist(cram_candidates['CRAM_score'], bins=20, 
                color='#F77F00', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=80, color='red', linestyle='--', linewidth=2, label='High-confidence (≥80)')
        ax2.axvline(x=60, color='orange', linestyle='--', linewidth=2, label='Probable (≥60)')
    
    ax2.set_xlabel('CRAM Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('(B) CRAM Score Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # SUBPLOT 3: Surface Ocean Detail
    if len(surface_df) > 0:
        for category in ['Non-CRAM', 'CRAM-like', 'Probable CRAM', 'High-confidence CRAM']:
            cat_data = surface_df[surface_df['CRAM_category'] == category]
            if len(cat_data) > 0:
                ax3.scatter(cat_data['O/C'], cat_data['H/C'],
                           c=colors[category], s=50, alpha=0.7,
                           marker='D' if 'CRAM' in category else 'o',
                           edgecolors='white', linewidths=1)
    
    ax3.add_patch(Rectangle((0.4, 1.2), 0.2, 0.8, 
                            linewidth=2, edgecolor='red', 
                            facecolor='yellow', alpha=0.2))
    ax3.set_xlabel('O/C', fontsize=11, fontweight='bold')
    ax3.set_ylabel('H/C', fontsize=11, fontweight='bold')
    ax3.set_title(f'(C) Surface Ocean (n={len(surface_df)})', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1.2)
    ax3.set_ylim(0.8, 2.4)
    ax3.grid(True, alpha=0.3)
    
    # SUBPLOT 4: Deep Ocean Detail
    if len(deep_df) > 0:
        for category in ['Non-CRAM', 'CRAM-like', 'Probable CRAM', 'High-confidence CRAM']:
            cat_data = deep_df[deep_df['CRAM_category'] == category]
            if len(cat_data) > 0:
                ax4.scatter(cat_data['O/C'], cat_data['H/C'],
                           c=colors[category], s=50, alpha=0.7,
                           marker='D' if 'CRAM' in category else 'o',
                           edgecolors='white', linewidths=1)
    
    ax4.add_patch(Rectangle((0.4, 1.2), 0.2, 0.8, 
                            linewidth=2, edgecolor='red', 
                            facecolor='yellow', alpha=0.2))
    ax4.set_xlabel('O/C', fontsize=11, fontweight='bold')
    ax4.set_ylabel('H/C', fontsize=11, fontweight='bold')
    ax4.set_title(f'(D) Deep Ocean (n={len(deep_df)})', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 1.2)
    ax4.set_ylim(0.8, 2.4)
    ax4.grid(True, alpha=0.3)
    
    # SUBPLOT 5: Comparative Statistics
    categories = ['High-confidence\nCRAM', 'Probable\nCRAM', 'CRAM-like', 'Non-CRAM']
    
    surface_counts = [
        (surface_df['CRAM_category'] == 'High-confidence CRAM').sum(),
        (surface_df['CRAM_category'] == 'Probable CRAM').sum(),
        (surface_df['CRAM_category'] == 'CRAM-like').sum(),
        (surface_df['CRAM_category'] == 'Non-CRAM').sum()
    ]
    
    deep_counts = [
        (deep_df['CRAM_category'] == 'High-confidence CRAM').sum(),
        (deep_df['CRAM_category'] == 'Probable CRAM').sum(),
        (deep_df['CRAM_category'] == 'CRAM-like').sum(),
        (deep_df['CRAM_category'] == 'Non-CRAM').sum()
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, surface_counts, width, label='Surface', 
                    color='#2E86AB', edgecolor='black', linewidth=1.5)
    bars2 = ax5.bar(x + width/2, deep_counts, width, label='Deep', 
                    color='#A23B72', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax5.set_ylabel('Number of Compounds', fontsize=12, fontweight='bold')
    ax5.set_title('(E) CRAM Distribution Comparison', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, fontsize=9)
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Van Krevelen Analysis: CRAM Discovery in Ocean Dissolved Organic Matter', 
                fontsize=17, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {output_file}")
    plt.close()

def create_cram_database(all_data, output_file):
    """
    Create new CRAM candidate database
    """
    print("\n" + "="*70)
    print("CREATING NEW CRAM CANDIDATE DATABASE")
    print("="*70)
    
    # Extract CRAM candidates (score >= 60)
    cram_candidates = all_data[all_data['CRAM_score'] >= 60].copy()
    
    # Sort by CRAM score (highest first)
    cram_candidates = cram_candidates.sort_values('CRAM_score', ascending=False)
    
    # Save to CSV
    cram_candidates.to_csv(output_file, index=False)
    
    print(f"\n📊 NEW CRAM DATABASE STATISTICS:")
    print(f"  Total CRAM candidates: {len(cram_candidates)}")
    print(f"  High-confidence CRAM: {(cram_candidates['CRAM_category'] == 'High-confidence CRAM').sum()}")
    print(f"  Probable CRAM: {(cram_candidates['CRAM_category'] == 'Probable CRAM').sum()}")
    print(f"\n  Average CRAM score: {cram_candidates['CRAM_score'].mean():.1f}")
    print(f"  Average O/C ratio: {cram_candidates['O/C'].mean():.3f}")
    print(f"  Average H/C ratio: {cram_candidates['H/C'].mean():.3f}")
    print(f"  Average COOH groups: {cram_candidates['COOH_groups'].mean():.1f}")
    print(f"  Average molecular weight: {cram_candidates['MW'].mean():.1f} Da")
    
    print(f"\n✅ CRAM database saved: {output_file}")
    print("="*70)
    
    return cram_candidates

def main():
    print("\n" + "="*70)
    print("VAN KREVELEN ANALYSIS: CRAM DISCOVERY IN OCEAN DOM")
    print("="*70)
    
    # Load data
    surface_df = load_and_process_ocean_data(
        'DOM_data/surface_results.tsv',
        'Surface Ocean'
    )
    
    deep_df = load_and_process_ocean_data(
        'DOM_data/deepocean_results.tsv',
        'Deep Ocean'
    )
    
    if len(surface_df) == 0 and len(deep_df) == 0:
        print("\n❌ ERROR: No valid SMILES data found in either file!")
        return
    
    # Combine datasets
    all_data = pd.concat([surface_df, deep_df], ignore_index=True)
    
    # Create visualization
    create_comprehensive_van_krevelen(
        surface_df, 
        deep_df,
        'DOM_data/van_krevelen_cram_discovery.png'
    )
    
    # Create CRAM candidate database
    cram_db = create_cram_database(
        all_data,
        'DOM_data/NEW_CRAM_CANDIDATE_DATABASE.csv'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Surface Ocean compounds analyzed: {len(surface_df)}")
    print(f"Deep Ocean compounds analyzed: {len(deep_df)}")
    print(f"Total compounds analyzed: {len(all_data)}")
    print(f"\n🎯 NEW CRAM CANDIDATES IDENTIFIED: {len(cram_db)}")
    print(f"   - From Surface: {len(cram_db[cram_db['Source'] == 'Surface Ocean'])}")
    print(f"   - From Deep: {len(cram_db[cram_db['Source'] == 'Deep Ocean'])}")
    
    if len(cram_db) > 0:
        print(f"\n📈 CRAM Enrichment:")
        surface_cram = len(cram_db[cram_db['Source'] == 'Surface Ocean'])
        deep_cram = len(cram_db[cram_db['Source'] == 'Deep Ocean'])
        
        if surface_cram > 0 and len(surface_df) > 0:
            surface_pct = surface_cram / len(surface_df) * 100
            print(f"   Surface Ocean: {surface_pct:.2f}% CRAM candidates")
        
        if deep_cram > 0 and len(deep_df) > 0:
            deep_pct = deep_cram / len(deep_df) * 100
            print(f"   Deep Ocean: {deep_pct:.2f}% CRAM candidates")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
