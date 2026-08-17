#!/usr/bin/env python3
"""
Van Krevelen Diagram: Ocean Surface vs Deep Sea CRAM Analysis (MS1-based)

This script:
1. Parses MGF files to extract MS1 precursor masses
2. Uses GNPS identifications to get molecular formulas
3. Calculates H/C and O/C ratios for Van Krevelen diagram
4. Identifies CRAM-like compounds and compares ocean vs deep sea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Rectangle
from collections import defaultdict

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5

def parse_mgf(mgf_file):
    """
    Parse MGF file to extract precursor masses
    Returns: list of precursor m/z values
    """
    print(f"Parsing {mgf_file}...")
    precursor_masses = []
    
    with open(mgf_file, 'r') as f:
        for line in f:
            if line.startswith('PEPMASS='):
                mass = float(line.split('=')[1].strip().split()[0])
                precursor_masses.append(mass)
    
    print(f" Found {len(precursor_masses)} MS1 spectra")
    return precursor_masses

def parse_molecular_formula(formula):
    """Parse molecular formula to count atoms"""
    if pd.isna(formula) or formula == 'N/A':
        return None
    
    formula = formula.split('[')[0].split(']')[0]
    atom_pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(atom_pattern, formula)
    
    atom_counts = {}
    for atom, count in matches:
        if atom:
            count = int(count) if count else 1
            atom_counts[atom] = atom_counts.get(atom, 0) + count
    
    return atom_counts

def calculate_ratios(atom_counts):
    """Calculate H/C and O/C ratios"""
    if atom_counts is None:
        return None, None
    
    C = atom_counts.get('C', 0)
    H = atom_counts.get('H', 0)
    O = atom_counts.get('O', 0)
    
    if C == 0:
        return None, None
    
    return H / C, O / C

def is_cram_like(H_C, O_C):
    """Determine if compound is CRAM-like"""
    if H_C is None or O_C is None:
        return False
    return (0.4 <= O_C <= 0.6) and (1.2 <= H_C <= 2.0)

def load_gnps_identifications(tsv_file):
    """Load GNPS identification results"""
    print(f"\nLoading GNPS identifications from {tsv_file}...")
    
    df = pd.read_csv(tsv_file, sep='\t')
    print(f" Total identifications: {len(df)}")
    
    results = []
    for idx, row in df.iterrows():
        formula = row.get('molecular_formula', 'N/A')
        atom_counts = parse_molecular_formula(formula)
        
        if atom_counts:
            H_C, O_C = calculate_ratios(atom_counts)
            if H_C is not None and O_C is not None:
                if 0 <= H_C <= 3.0 and 0 <= O_C <= 1.5:
                    results.append({
                        'Formula': formula,
                        'H/C': H_C,
                        'O/C': O_C,
                        'is_CRAM': is_cram_like(H_C, O_C),
                        'Compound_Name': row.get('Compound_Name', 'Unknown'),
                        'SpecMZ': row.get('SpecMZ', 0)
                    })
    
    result_df = pd.DataFrame(results)
    print(f" Valid formulas: {len(result_df)}")
    print(f" CRAM-like: {result_df['is_CRAM'].sum()}")
    
    return result_df

def create_van_krevelen_comparison(surface_gnps, surface_mgf, deep_mgf, output_file):
    """
    Create Van Krevelen diagram with CRAM region analysis
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0]) # Surface identified
    ax2 = fig.add_subplot(gs[0, 1]) # CRAM region zoom
    ax3 = fig.add_subplot(gs[0, 2]) # Statistics bar chart
    ax4 = fig.add_subplot(gs[1,:]) # Mass distribution comparison
    
    # Colors
    surface_color = '#2E86AB'
    cram_color = '#F18F01'
    deep_color = '#A23B72'
    
    # SUBPLOT 1: Van Krevelen with identified compounds
    if len(surface_gnps) > 0:
        surface_non_cram = surface_gnps[~surface_gnps['is_CRAM']]
        surface_cram = surface_gnps[surface_gnps['is_CRAM']]
        
        ax1.scatter(surface_non_cram['O/C'], surface_non_cram['H/C'],
                   c=surface_color, alpha=0.6, s=40, edgecolors='white', linewidths=0.5,
                   label=f'Non-CRAM (n={len(surface_non_cram)})')
        
        if len(surface_cram) > 0:
            ax1.scatter(surface_cram['O/C'], surface_cram['H/C'],
                       c=cram_color, alpha=0.9, s=80, edgecolors='darkred', linewidths=1.5,
                       marker='D', label=f'CRAM-like (n={len(surface_cram)})', zorder=5)
    
    # CRAM region box
    ax1.add_patch(Rectangle((0.4, 1.2), 0.2, 0.8, 
                           linewidth=3, edgecolor='red', 
                           facecolor='yellow', alpha=0.1, linestyle='--'))
    
    # Add region labels
    ax1.text(0.15, 2.2, 'Lipids', fontsize=9, style='italic', alpha=0.7)
    ax1.text(0.5, 2.2, 'Carbohydrates', fontsize=9, style='italic', alpha=0.7)
    ax1.text(0.7, 0.8, 'Lignin', fontsize=9, style='italic', alpha=0.7)
    ax1.text(0.5, 1.6, 'CRAM', fontsize=11, fontweight='bold', color='red')
    
    ax1.set_xlabel('O/C Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('H/C Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Surface Ocean - Van Krevelen Space', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0.5, 2.5)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # SUBPLOT 2: CRAM Region Zoom
    if len(surface_cram) > 0:
        ax2.scatter(surface_cram['O/C'], surface_cram['H/C'],
                   c=cram_color, alpha=0.9, s=120, edgecolors='darkred', linewidths=2,
                   marker='D')
        
        # Add compound names
        for idx, row in surface_cram.iterrows():
            name = row['Compound_Name'][:20] # Truncate long names
            ax2.annotate(name, (row['O/C'], row['H/C']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, alpha=0.8)
    
    ax2.add_patch(Rectangle((0.4, 1.2), 0.2, 0.8, 
                           linewidth=2, edgecolor='red', 
                           facecolor='yellow', alpha=0.2))
    
    ax2.set_xlabel('O/C Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('H/C Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('(B) CRAM Region - Detailed View', fontsize=13, fontweight='bold')
    ax2.set_xlim(0.35, 0.65)
    ax2.set_ylim(1.1, 2.1)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    # SUBPLOT 3: CRAM Dominance Statistics
    total = len(surface_gnps)
    cram_count = surface_gnps['is_CRAM'].sum() if len(surface_gnps) > 0 else 0
    non_cram_count = total - cram_count
    cram_pct = (cram_count / total * 100) if total > 0 else 0
    
    categories = ['Total\nIdentified', 'CRAM-like', 'Non-CRAM']
    counts = [total, cram_count, non_cram_count]
    colors_bar = ['gray', cram_color, surface_color]
    
    bars = ax3.bar(categories, counts, color=colors_bar, 
                   edgecolor='black', linewidth=2, alpha=0.8)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (count / total * 100) if total > 0 else 0
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Number of Compounds', fontsize=12, fontweight='bold')
    ax3.set_title('(C) CRAM Identification Statistics', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, max(counts) * 1.2)
    ax3.grid(axis='y', alpha=0.3, linestyle=':')
    
    # SUBPLOT 4: MS1 Mass Distribution Comparison
    surface_masses = parse_mgf(surface_mgf)
    deep_masses = parse_mgf(deep_mgf)
    
    # Create histograms
    bins = np.linspace(100, 1000, 100)
    
    ax4.hist(surface_masses, bins=bins, alpha=0.5, label=f'Surface Ocean (n={len(surface_masses)})',
            color=surface_color, edgecolor='black', linewidth=0.5)
    ax4.hist(deep_masses, bins=bins, alpha=0.5, label=f'Deep Sea (n={len(deep_masses)})',
            color=deep_color, edgecolor='black', linewidth=0.5)
    
    # Add CRAM mass range (typical CRAM: 200-400 Da)
    ax4.axvspan(200, 400, color='yellow', alpha=0.2, label='Typical CRAM m/z range')
    
    ax4.set_xlabel('Precursor m/z', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('(D) MS1 Mass Distribution: Surface Ocean vs Deep Sea', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle=':')
    ax4.set_xlim(100, 1000)
    
    plt.suptitle('Van Krevelen Analysis: Ocean DOM Composition and CRAM Dominance', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n Van Krevelen diagram saved: {output_file}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("CRAM DOMINANCE ANALYSIS - SURFACE OCEAN (GNPS Identifications)")
    print("="*70)
    print(f"Total identified compounds: {total}")
    print(f"CRAM-like compounds: {cram_count} ({cram_pct:.2f}%)")
    print(f"Non-CRAM compounds: {non_cram_count} ({100-cram_pct:.2f}%)")
    print(f"\nMS1 Data:")
    print(f"Surface Ocean spectra: {len(surface_masses)}")
    print(f"Deep Sea spectra: {len(deep_masses)}")
    print(f"Deep/Surface ratio: {len(deep_masses)/len(surface_masses):.2f}x")
    
    # Mass range analysis
    surface_cram_range = sum(1 for m in surface_masses if 200 <= m <= 400)
    deep_cram_range = sum(1 for m in deep_masses if 200 <= m <= 400)
    print(f"\nCompounds in CRAM m/z range (200-400 Da):")
    print(f"Surface: {surface_cram_range} ({surface_cram_range/len(surface_masses)*100:.2f}%)")
    print(f"Deep Sea: {deep_cram_range} ({deep_cram_range/len(deep_masses)*100:.2f}%)")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("Van Krevelen Diagram: Ocean vs Deep Sea CRAM Analysis")
    print("="*70 + "\n")
    
    # Load GNPS identifications (surface ocean only - has molecular formulas)
    surface_gnps = load_gnps_identifications('DOM_data/surface_results.tsv')
    
    # Create comprehensive visualization
    if len(surface_gnps) > 0:
        create_van_krevelen_comparison(
            surface_gnps,
            'DOM_data/Nelha_openocean_surface.mgf',
            'DOM_data/Nelha_openocean_deep.mgf',
            'DOM_data/van_krevelen_ocean_cram_comparison.png'
        )
        
        # Save data
        surface_gnps.to_csv('DOM_data/van_krevelen_identified_compounds.csv', index=False)
        print(f" Data saved: DOM_data/van_krevelen_identified_compounds.csv")
    else:
        print("\nERROR: No valid molecular formulas found!")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
