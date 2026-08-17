#!/usr/bin/env python3
"""
Generate individual high-resolution visualizations for each representative CRAM structure
Creates separate 700 DPI images for flexible use in presentations and publications
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
import numpy as np

# Set publication-quality settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5

def count_carboxyl_groups(mol):
    """Count carboxyl groups"""
    carboxyl_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    if carboxyl_pattern:
        return len(mol.GetSubstructMatches(carboxyl_pattern))
    return 0

def calculate_aliphatic_percentage(mol):
    """Calculate aliphatic percentage"""
    total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    if total_carbons == 0:
        return 0.0
    aromatic_carbons = sum(1 for atom in mol.GetAtoms() 
                          if atom.GetSymbol() == 'C' and atom.GetIsAromatic())
    return ((total_carbons - aromatic_carbons) / total_carbons) * 100

def count_alicyclic_rings(mol):
    """Count alicyclic rings"""
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    alicyclic_count = 0
    for ring in atom_rings:
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

def create_individual_structure_plot(mol_data, properties, output_path, color_scheme):
    """Create individual structure plot with properties"""
    
    fig = plt.figure(figsize=(8, 10), dpi=700)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.3,
                  left=0.1, right=0.9, top=0.92, bottom=0.08)
    
    # Title
    fig.suptitle(mol_data['name'], fontsize=16, fontweight='bold', y=0.96)
    
    # Top panel: Molecule structure
    ax_mol = fig.add_subplot(gs[0, 0])
    mol = Chem.MolFromSmiles(mol_data['smiles'])
    
    if mol is not None:
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(600, 500), kekulize=True)
        ax_mol.imshow(img)
    
    ax_mol.axis('off')
    
    # Bottom panel: Properties table
    ax_props = fig.add_subplot(gs[1, 0])
    ax_props.axis('off')
    
    # Create property table
    table_data = [
        ['Molecular Formula', properties['formula']],
        ['Molecular Weight', f"{properties['mw']:.2f} g/mol"],
        ['Carboxyl Groups (COOH)', str(properties['cooh'])],
        ['Aliphatic Content', f"{properties['aliphatic']:.1f}%"],
        ['O/C Ratio', f"{properties['oc_ratio']:.2f}"],
        ['Alicyclic Rings', str(properties['rings'])],
    ]
    
    # Add SMILES
    smiles_wrapped = mol_data['smiles']
    if len(smiles_wrapped) > 50:
        # Wrap long SMILES
        smiles_lines = [smiles_wrapped[i:i+50] for i in range(0, len(smiles_wrapped), 50)]
        smiles_wrapped = '\n'.join(smiles_lines)
    table_data.append(['SMILES', smiles_wrapped])
    
    # Create table
    table = ax_props.table(cellText=table_data,
                           colWidths=[0.35, 0.65],
                           cellLoc='left',
                           loc='center',
                           bbox=[0.05, 0.0, 0.9, 1.0])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style the table
    for i, (key, value) in enumerate(table_data):
        # Header cells (property names)
        cell = table[(i, 0)]
        cell.set_facecolor('#ecf0f1')
        cell.set_text_props(weight='bold', color='#2c3e50')
        cell.set_edgecolor('#bdc3c7')
        cell.set_linewidth(1.5)
        
        # Value cells
        cell = table[(i, 1)]
        cell.set_facecolor('white')
        cell.set_text_props(color='#34495e', family='monospace')
        cell.set_edgecolor('#bdc3c7')
        cell.set_linewidth(1.5)
    
    # Add compliance badge
    if properties['compliant']:
        badge_text = " CRAM COMPLIANT"
        badge_color = '#27ae60'
        badge_text_color = 'white'
    else:
        badge_text = " PARTIAL COMPLIANCE"
        badge_color = '#e67e22'
        badge_text_color = 'white'
    
    # Draw compliance badge
    badge = mpatches.FancyBboxPatch((0.25, -0.25), 0.5, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor=badge_color,
                                     edgecolor=badge_color,
                                     linewidth=3,
                                     transform=ax_props.transAxes,
                                     zorder=10)
    ax_props.add_patch(badge)
    ax_props.text(0.5, -0.175, badge_text, transform=ax_props.transAxes,
                  fontsize=11, fontweight='bold', ha='center', va='center',
                  color=badge_text_color, zorder=11)
    
    # Save figure
    plt.savefig(output_path, dpi=700, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f" Saved: {output_path}")
    plt.close()

def create_radar_chart_only(properties_list, names, output_path):
    """Create standalone radar chart for property comparison"""
    
    fig = plt.figure(figsize=(10, 10), dpi=700)
    ax = fig.add_subplot(111, projection='polar')
    
    # Properties to compare (normalized 0-1)
    categories = ['COOH\nGroups', 'Aliphatic\nContent %', 'O/C Ratio\n(normalized)', 
                  'Alicyclic\nRings', 'Molecular\nWeight']
    N = len(categories)
    
    # Normalize values
    data = []
    for props in properties_list:
        normalized = [
            props['cooh'] / 5.0, # Max 5 COOH
            props['aliphatic'] / 100.0, # Max 100%
            props['oc_ratio'] / 0.6, # Target max 0.6
            props['rings'] / 4.0, # Max 4 rings
            props['mw'] / 400.0 # Max 400 Da
        ]
        data.append(normalized)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    data = [d + [d[0]] for d in data] # Close the plot
    angles += angles[:1]
    
    # Plot
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    markers = ['o', 's', '^']
    
    for i, (d, name, color, marker) in enumerate(zip(data, names, colors, markers)):
        ax.plot(angles, d, marker=marker, linewidth=3, label=name, 
                color=color, markersize=10, markeredgewidth=2, 
                markeredgecolor='white')
        ax.fill(angles, d, alpha=0.2, color=color)
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=9, color='gray')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    
    # Add title
    ax.set_title('CRAM Property Comparison\n(Normalized to Target Ranges)', 
                 fontsize=14, fontweight='bold', pad=30, y=1.1)
    
    # Add legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), 
                      fontsize=11, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.5)
    
    # Add CRAM criteria reference
    criteria_text = (
        "CRAM Criteria:\n"
        "━━━━━━━━━━━━━━\n"
        "• COOH ≥ 2\n"
        "• Aliphatic ≥ 70%\n"
        "• O/C: 0.4-0.6\n"
        "• Rings ≥ 1\n"
        "• MW: 150-400 Da"
    )
    
    fig.text(0.15, 0.15, criteria_text, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', 
                      edgecolor='#34495e', linewidth=2, alpha=0.95),
             verticalalignment='top', fontfamily='monospace', fontweight='bold')
    
    # Save
    plt.savefig(output_path, dpi=700, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    print(f" Saved: {output_path}")
    plt.close()

# Define the three example molecules
molecules = [
    {
        'name': 'Example 1',
        'short_name': 'Example 1',
        'smiles': 'C1CC(C(=O)O)C2CC(C(=O)O)CC(C(=O)O)C2C1',
        'color': '#3498db'
    },
    {
        'name': 'Example 2',
        'short_name': 'Example 2',
        'smiles': 'C2CCC1CCC(C(=O)O)CC1C(C2)C(=O)O',
        'color': '#e74c3c'
    },
    {
        'name': 'Example 3',
        'short_name': 'Example 3',
        'smiles': 'C1CC2CC(C(=O)O)C1C(C(=O)O)C2C(=O)O',
        'color': '#2ecc71'
    }
]

# Calculate properties for all molecules
properties_list = []
for mol_data in molecules:
    mol = Chem.MolFromSmiles(mol_data['smiles'])
    props = {
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'mw': Descriptors.MolWt(mol),
        'cooh': count_carboxyl_groups(mol),
        'aliphatic': calculate_aliphatic_percentage(mol),
        'oc_ratio': calculate_oc_ratio(mol),
        'rings': count_alicyclic_rings(mol)
    }
    # Check compliance
    props['compliant'] = (props['cooh'] >= 2 and 
                         props['aliphatic'] >= 70 and 
                         0.4 <= props['oc_ratio'] <= 0.6 and
                         props['rings'] >= 1 and
                         150 <= props['mw'] <= 400)
    properties_list.append(props)

print("\n" + "="*80)
print("Generating individual high-resolution structure visualizations (700 DPI)")
print("="*80 + "\n")

# Create individual structure plots
for i, (mol_data, props) in enumerate(zip(molecules, properties_list), 1):
    output_path = f'figures/representative_structure_example{i}_700dpi.png'
    create_individual_structure_plot(mol_data, props, output_path, mol_data['color'])

# Create standalone radar chart
print("\nGenerating radar chart comparison...")
names = [m['short_name'] for m in molecules]
create_radar_chart_only(properties_list, names, 
                        'figures/representative_structures_radar_700dpi.png')

# Create a combined comparison view (side by side)
print("\nGenerating side-by-side comparison...")
fig = plt.figure(figsize=(20, 7), dpi=700)
gs = GridSpec(1, 3, figure=fig, wspace=0.25,
              left=0.05, right=0.95, top=0.88, bottom=0.12)

fig.suptitle('CRAM Representative Structures Comparison', 
             fontsize=18, fontweight='bold', y=0.96)

for i, (mol_data, props) in enumerate(zip(molecules, properties_list)):
    ax = fig.add_subplot(gs[0, i])
    
    # Draw molecule
    mol = Chem.MolFromSmiles(mol_data['smiles'])
    if mol is not None:
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(500, 400), kekulize=True)
        ax.imshow(img)
    
    ax.axis('off')
    
    # Title
    ax.set_title(mol_data['name'], fontsize=12, fontweight='bold', pad=10)
    
    # Properties text below
    props_text = (
        f"Formula: {props['formula']}\n"
        f"MW: {props['mw']:.2f} g/mol\n"
        f"COOH: {props['cooh']} | Aliphatic: {props['aliphatic']:.0f}%\n"
        f"O/C: {props['oc_ratio']:.2f} | Rings: {props['rings']}"
    )
    
    ax.text(0.5, -0.05, props_text, transform=ax.transAxes,
            fontsize=9, ha='center', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='gray', linewidth=1.5, alpha=0.9))
    
    # Compliance badge
    if props['compliant']:
        badge_text = " COMPLIANT"
        badge_color = '#27ae60'
    else:
        badge_text = " PARTIAL"
        badge_color = '#e67e22'
    
    ax.text(0.5, -0.18, badge_text, transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='center', va='top',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=badge_color,
                     edgecolor=badge_color, linewidth=2, alpha=1.0))

plt.savefig('figures/representative_structures_comparison_700dpi.png', 
            dpi=700, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f" Saved: figures/representative_structures_comparison_700dpi.png")
plt.close()

print("\n" + "="*80)
print("All individual visualizations complete!")
print("="*80)
print("\nGenerated files:")
print(" 1. figures/representative_structure_example1_700dpi.png")
print(" 2. figures/representative_structure_example2_700dpi.png")
print(" 3. figures/representative_structure_example3_700dpi.png")
print(" 4. figures/representative_structures_radar_700dpi.png")
print(" 5. figures/representative_structures_comparison_700dpi.png")
print("="*80 + "\n")
