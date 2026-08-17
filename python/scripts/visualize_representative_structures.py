#!/usr/bin/env python3
"""
Generate visualization for Representative CRAM Structures
Shows molecular structures with property annotations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
import numpy as np

# Set publication-quality settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
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

def draw_molecule_with_properties(ax, smiles, title, properties):
    """Draw molecule structure with property table"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
        return
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Draw molecule
    img = Draw.MolToImage(mol, size=(400, 300), kekulize=True)
    
    # Display molecule
    ax.imshow(img)
    ax.axis('off')
    
    # Add title
    ax.text(0.5, 1.02, title, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', ha='center', va='bottom')
    
    # Add property table below molecule
    props_text = []
    props_text.append(f"Formula: {properties['formula']}")
    props_text.append(f"MW: {properties['mw']:.2f} g/mol")
    props_text.append(f"COOH: {properties['cooh']}")
    props_text.append(f"Aliphatic: {properties['aliphatic']:.0f}%")
    props_text.append(f"O/C: {properties['oc_ratio']:.2f}")
    props_text.append(f"Alicyclic Rings: {properties['rings']}")
    
    # Compliance status
    if properties['compliant']:
        status_text = " CRAM Compliant"
        status_color = '#27ae60'
    else:
        status_text = " Partial Compliance"
        status_color = '#e67e22'
    
    props_text.append(f"{status_text}")
    
    # Position text below molecule
    y_start = -0.15
    for i, text in enumerate(props_text[:-1]):
        color = 'black'
        ax.text(0.5, y_start - i*0.05, text, transform=ax.transAxes,
                fontsize=9, ha='center', va='top', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.7))
    
    # Status text with special styling
    ax.text(0.5, y_start - len(props_text[:-1])*0.05 - 0.03, props_text[-1], 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='center', va='top', color=status_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=status_color, linewidth=2, alpha=0.9))

def create_property_radar(ax, properties_list, names):
    """Create radar chart comparing properties"""
    # Properties to compare (normalized 0-1)
    categories = ['COOH', 'Aliphatic\n%', 'O/C\n(norm)', 'Rings', 'MW\n(norm)']
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
    ax = plt.subplot(111, projection='polar')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, (d, name, color) in enumerate(zip(data, names, colors)):
        ax.plot(angles, d, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, d, alpha=0.15, color=color)
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=7)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title('Property Comparison', fontsize=11, fontweight='bold', pad=20)
    
    return ax

# Define the three example molecules
molecules = [
    {
        'name': 'Example 1:\nTricyclic Tricarboxylic Acid',
        'smiles': 'C1CC(C(=O)O)C2CC(C(=O)O)CC(C(=O)O)C2C1'
    },
    {
        'name': 'Example 2:\nDecalin-type Dicarboxylic Acid',
        'smiles': 'C2CCC1CCC(C(=O)O)CC1C(C2)C(=O)O'
    },
    {
        'name': 'Example 3:\nNorbornane Tricarboxylic Acid',
        'smiles': 'C1CC2CC(C(=O)O)C1C(C(=O)O)C2C(=O)O'
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

# Create the figure
fig = plt.figure(figsize=(16, 12), dpi=300)
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3,
              left=0.05, right=0.95, top=0.94, bottom=0.06)

# Add main title
fig.suptitle('Representative Generated CRAM Structures', 
             fontsize=18, fontweight='bold', y=0.98)

# Plot each molecule
for i, (mol_data, props) in enumerate(zip(molecules, properties_list)):
    ax = fig.add_subplot(gs[i,:2])
    draw_molecule_with_properties(ax, mol_data['smiles'], 
                                  mol_data['name'], props)

# Add radar chart in the right column
ax_radar = fig.add_subplot(gs[:, 2], projection='polar')
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Radar chart data
categories = ['COOH', 'Aliphatic\n%', 'O/C\n(norm)', 'Rings', 'MW\n(norm)']
N = len(categories)

data = []
for props in properties_list:
    normalized = [
        props['cooh'] / 5.0,
        props['aliphatic'] / 100.0,
        props['oc_ratio'] / 0.6,
        props['rings'] / 4.0,
        props['mw'] / 400.0
    ]
    data.append(normalized)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
data = [d + [d[0]] for d in data]
angles += angles[:1]

for i, (d, mol_data, color) in enumerate(zip(data, molecules, colors)):
    label = mol_data['name'].replace('\n', ' ')
    ax_radar.plot(angles, d, 'o-', linewidth=2.5, label=f"Ex {i+1}", color=color, markersize=6)
    ax_radar.fill(angles, d, alpha=0.15, color=color)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, size=10, fontweight='bold')
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(['25%', '50%', '75%', '100%'], size=8, color='gray')
ax_radar.grid(True, linestyle='--', alpha=0.6, linewidth=1.2)
ax_radar.set_title('Property Comparison\n(Normalized to Target Ranges)', 
                   fontsize=12, fontweight='bold', pad=25)

# Add legend
legend = ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), 
                        fontsize=10, frameon=True, fancybox=True, shadow=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Add CRAM criteria reference box
criteria_text = (
    "CRAM Criteria:\n"
    "• COOH ≥ 2\n"
    "• Aliphatic ≥ 70%\n"
    "• O/C: 0.4-0.6\n"
    "• Rings ≥ 1\n"
    "• MW: 150-400 Da"
)

fig.text(0.78, 0.12, criteria_text, fontsize=9, 
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', 
                  edgecolor='#34495e', linewidth=2, alpha=0.9),
         verticalalignment='top', fontfamily='monospace')

# Add statistics summary
summary_text = (
    f"Compliance Summary:\n"
    f"Example 1: {' Full' if properties_list[0]['compliant'] else ' Partial'}\n"
    f"Example 2: {' Full' if properties_list[1]['compliant'] else ' Partial'}\n"
    f"Example 3: {' Full' if properties_list[2]['compliant'] else ' Partial'}"
)

fig.text(0.78, 0.30, summary_text, fontsize=9,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#e8f5e9',
                  edgecolor='#27ae60', linewidth=2, alpha=0.9),
         verticalalignment='top', fontfamily='monospace')

# Save figure
output_path = 'figures/representative_structures.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f" Saved: {output_path}")

# Also save high-res version
output_path_hires = 'figures/representative_structures_highres.png'
plt.savefig(output_path_hires, dpi=700, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f" Saved high-resolution: {output_path_hires}")

plt.close()

print("\n" + "="*80)
print("Representative structures visualization complete!")
print("="*80)
