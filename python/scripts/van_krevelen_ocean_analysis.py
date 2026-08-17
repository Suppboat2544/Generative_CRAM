#!/usr/bin/env python3
"""
Van Krevelen Diagram: Ocean Surface vs Deep Sea CRAM Analysis

This script:
1. Reads MS data from surface and deep ocean samples
2. Extracts molecular formulas from identified compounds
3. Calculates H/C and O/C ratios for Van Krevelen diagram
4. Identifies CRAM-like compounds (O/C: 0.4-0.6, H/C: 1.2-2.0, alicyclic)
5. Visualizes CRAM dominance in surface vs deep sea environments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import re
from collections import Counter

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5

def parse_molecular_formula(formula):
    """
    Parse molecular formula string to count atoms
    Returns: dict with atom counts
    """
    if pd.isna(formula) or formula == 'N/A':
        return None
    
    # Remove charge indicators
    formula = formula.split('[')[0].split(']')[0]
    
    # Parse formula using regex
    atom_pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(atom_pattern, formula)
    
    atom_counts = {}
    for atom, count in matches:
        if atom: # Skip empty matches
            count = int(count) if count else 1
            atom_counts[atom] = atom_counts.get(atom, 0) + count
    
    return atom_counts

def calculate_ratios(atom_counts):
    """
    Calculate H/C and O/C ratios from atom counts
    Returns: (H_C_ratio, O_C_ratio)
    """
    if atom_counts is None:
        return None, None
    
    C = atom_counts.get('C', 0)
    H = atom_counts.get('H', 0)
    O = atom_counts.get('O', 0)
    
    if C == 0:
        return None, None
    
    H_C = H / C
    O_C = O / C
    
    return H_C, O_C

def is_cram_like(H_C, O_C):
    """
    Determine if compound is CRAM-like based on ratios
    CRAM criteria: O/C = 0.4-0.6, H/C = 1.2-2.0 (alicyclic range)
    """
    if H_C is None or O_C is None:
        return False
    
    # CRAM region in Van Krevelen space
    return (0.4 <= O_C <= 0.6) and (1.2 <= H_C <= 2.0)

def load_and_process_data(tsv_file, label):
    """
    Load TSV file and extract molecular formulas with ratios
    """
    print(f"Loading {label} data from {tsv_file}...")
    
    df = pd.read_csv(tsv_file, sep='\t')
    print(f" Total identifications: {len(df)}")
    
    # Extract molecular formulas
    if 'molecular_formula' not in df.columns:
        print(f" WARNING: No molecular_formula column found!")
        return pd.DataFrame()
    
    results = []
    for idx, row in df.iterrows():
        formula = row.get('molecular_formula', 'N/A')
        atom_counts = parse_molecular_formula(formula)
        
        if atom_counts:
            H_C, O_C = calculate_ratios(atom_counts)
            if H_C is not None and O_C is not None:
                # Filter reasonable ranges for visualization
                if 0 <= H_C <= 3.0 and 0 <= O_C <= 1.5:
                    results.append({
                        'Formula': formula,
                        'H/C': H_C,
                        'O/C': O_C,
                        'C_count': atom_counts.get('C', 0),
                        'H_count': atom_counts.get('H', 0),
                        'O_count': atom_counts.get('O', 0),
                        'N_count': atom_counts.get('N', 0),
                        'S_count': atom_counts.get('S', 0),
                        'is_CRAM': is_cram_like(H_C, O_C),
                        'Source': label,
                        'Compound_Name': row.get('Compound_Name', 'Unknown'),
                        'MQScore': row.get('MQScore', 0)
                    })
    
    result_df = pd.DataFrame(results)
    print(f" Valid formulas with ratios: {len(result_df)}")
    print(f" CRAM-like compounds: {result_df['is_CRAM'].sum()}")
    
    return result_df

def create_van_krevelen_plot(surface_df, deep_df, output_file):
    """
    Create Van Krevelen diagram comparing surface and deep ocean
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define CRAM region
    cram_box = Rectangle((0.4, 1.2), 0.2, 0.8, 
                         linewidth=2, edgecolor='red', 
                         facecolor='none', linestyle='--',
                         label='CRAM Region')
    
    # Color schemes
    surface_color = '#2E86AB' # Blue for surface
    deep_color = '#A23B72' # Purple for deep
    cram_color = '#F18F01' # Orange for CRAM
    
    # SUBPLOT 1: Surface Ocean
    ax1 = axes[0]
    if len(surface_df) > 0:
        # Non-CRAM compounds
        surface_non_cram = surface_df[~surface_df['is_CRAM']]
        surface_cram = surface_df[surface_df['is_CRAM']]
        
        ax1.scatter(surface_non_cram['O/C'], surface_non_cram['H/C'],
                   c=surface_color, alpha=0.6, s=30, edgecolors='white', linewidths=0.5,
                   label=f'Non-CRAM (n={len(surface_non_cram)})')
        
        ax1.scatter(surface_cram['O/C'], surface_cram['H/C'],
                   c=cram_color, alpha=0.8, s=50, edgecolors='darkred', linewidths=1,
                   marker='D', label=f'CRAM-like (n={len(surface_cram)})')
    
    ax1.add_patch(Rectangle((0.4, 1.2), 0.2, 0.8, 
                           linewidth=2, edgecolor='red', 
                           facecolor='none', linestyle='--'))
    
    ax1.set_xlabel('O/C Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('H/C Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Surface Ocean', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0.5, 2.5)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # SUBPLOT 2: Deep Sea
    ax2 = axes[1]
    if len(deep_df) > 0:
        deep_non_cram = deep_df[~deep_df['is_CRAM']]
        deep_cram = deep_df[deep_df['is_CRAM']]
        
        ax2.scatter(deep_non_cram['O/C'], deep_non_cram['H/C'],
                   c=deep_color, alpha=0.6, s=30, edgecolors='white', linewidths=0.5,
                   label=f'Non-CRAM (n={len(deep_non_cram)})')
        
        ax2.scatter(deep_cram['O/C'], deep_cram['H/C'],
                   c=cram_color, alpha=0.8, s=50, edgecolors='darkred', linewidths=1,
                   marker='D', label=f'CRAM-like (n={len(deep_cram)})')
    
    ax2.add_patch(Rectangle((0.4, 1.2), 0.2, 0.8, 
                           linewidth=2, edgecolor='red', 
                           facecolor='none', linestyle='--'))
    
    ax2.set_xlabel('O/C Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('H/C Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Deep Ocean', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(0.5, 2.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # SUBPLOT 3: Comparison - CRAM Dominance
    ax3 = axes[2]
    
    # Calculate percentages
    surface_total = len(surface_df)
    surface_cram_count = surface_df['is_CRAM'].sum()
    surface_cram_pct = (surface_cram_count / surface_total * 100) if surface_total > 0 else 0
    
    deep_total = len(deep_df)
    deep_cram_count = deep_df['is_CRAM'].sum()
    deep_cram_pct = (deep_cram_count / deep_total * 100) if deep_total > 0 else 0
    
    # Bar chart
    environments = ['Surface\nOcean', 'Deep\nSea']
    cram_counts = [surface_cram_count, deep_cram_count]
    non_cram_counts = [surface_total - surface_cram_count, deep_total - deep_cram_count]
    
    x_pos = np.arange(len(environments))
    width = 0.6
    
    bars1 = ax3.bar(x_pos, non_cram_counts, width, 
                    label='Non-CRAM', color='lightgray', edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x_pos, cram_counts, width, 
                    bottom=non_cram_counts,
                    label='CRAM-like', color=cram_color, edgecolor='darkred', linewidth=1.5)
    
    # Add percentage labels
    for i, (env, cram_cnt, total) in enumerate(zip(environments, cram_counts, [surface_total, deep_total])):
        pct = (cram_cnt / total * 100) if total > 0 else 0
        ax3.text(i, total + 5, f'{pct:.1f}%\nCRAM', 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkred')
    
    ax3.set_ylabel('Number of Compounds', fontsize=12, fontweight='bold')
    ax3.set_title('(C) CRAM Dominance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(environments, fontsize=11)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n Van Krevelen diagram saved: {output_file}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("CRAM DOMINANCE ANALYSIS")
    print("="*60)
    print(f"\nSURFACE OCEAN:")
    print(f" Total compounds: {surface_total}")
    print(f" CRAM-like: {surface_cram_count} ({surface_cram_pct:.2f}%)")
    print(f" Non-CRAM: {surface_total - surface_cram_count} ({100-surface_cram_pct:.2f}%)")
    
    print(f"\nDEEP SEA:")
    print(f" Total compounds: {deep_total}")
    print(f" CRAM-like: {deep_cram_count} ({deep_cram_pct:.2f}%)")
    print(f" Non-CRAM: {deep_total - deep_cram_count} ({100-deep_cram_pct:.2f}%)")
    
    print(f"\nCRAM ENRICHMENT:")
    if deep_cram_pct > surface_cram_pct:
        enrichment = deep_cram_pct / surface_cram_pct if surface_cram_pct > 0 else float('inf')
        print(f" Deep sea has {enrichment:.2f}x more CRAM dominance than surface")
    elif surface_cram_pct > deep_cram_pct:
        enrichment = surface_cram_pct / deep_cram_pct if deep_cram_pct > 0 else float('inf')
        print(f" Surface has {enrichment:.2f}x more CRAM dominance than deep sea")
    else:
        print(f" Similar CRAM dominance in both environments")
    print("="*60)

def main():
    print("\n" + "="*60)
    print("Van Krevelen Diagram: Ocean vs Deep Sea CRAM Analysis")
    print("="*60 + "\n")
    
    # Load data
    surface_df = load_and_process_data(
        'DOM_data/surface_results.tsv',
        'Surface Ocean'
    )
    
    deep_df = load_and_process_data(
        'DOM_data/deepocean_results.tsv',
        'Deep Sea'
    )
    
    # Create visualization
    if len(surface_df) > 0 or len(deep_df) > 0:
        create_van_krevelen_plot(
            surface_df, 
            deep_df,
            'DOM_data/van_krevelen_ocean_comparison.png'
        )
        
        # Save combined data
        combined_df = pd.concat([surface_df, deep_df], ignore_index=True)
        combined_df.to_csv('DOM_data/van_krevelen_data.csv', index=False)
        print(f" Data saved: DOM_data/van_krevelen_data.csv")
    else:
        print("\nERROR: No valid data found in either file!")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
