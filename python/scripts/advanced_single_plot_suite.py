"""
Advanced CRAM Visualization Suite - Single Complex Plots
=========================================================

Each function creates ONE sophisticated standalone figure (NO subplots)
Optimized for A0 poster at 700 DPI

Each visualization tells a complete story on its own.

Author: CRAM Generator Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scientific visualization settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2.5

# Color palettes
CRAM_COLORS = {'excellent': '#2ecc71', 'good': '#3498db', 'moderate': '#f39c12', 'poor': '#e74c3c'}
ENZYME_COLORS = {'cluster_1': '#8e44ad', 'cluster_2': '#16a085', 'cluster_3': '#d35400', 
                 'cluster_4': '#c0392b', 'cluster_5': '#2c3e50'}


# ============================================================================
# FIGURE 1: CRAM Property Space with Density Contours
# ============================================================================

def plot_cram_property_space_density(cram_data, save_path='fig1_cram_property_space_density.png', dpi=700):
    """
    Single comprehensive plot showing CRAM molecules in 2D property space
    with density contours, marginal distributions, and quality zones.
    """
    df = pd.DataFrame(cram_data)
    
    fig, ax = plt.subplots(figsize=(24, 20), dpi=dpi)
    
    # Create scatter plot colored by CRAM score
    scatter = ax.scatter(df['mw'], df['logp'], 
                        c=df['cram_score'], s=600,
                        cmap='RdYlGn', vmin=0, vmax=1,
                        edgecolors='black', linewidth=3,
                        alpha=0.8, zorder=5)
    
    # Add density contours
    from scipy.stats import gaussian_kde
    
    # Calculate density
    xy = np.vstack([df['mw'], df['logp']])
    z = gaussian_kde(xy)(xy)
    
    # Sort by density for better visualization
    idx = z.argsort()
    x, y, z_sorted = df['mw'].iloc[idx], df['logp'].iloc[idx], z[idx]
    
    # Add contour lines
    xi = np.linspace(df['mw'].min(), df['mw'].max(), 100)
    yi = np.linspace(df['logp'].min(), df['logp'].max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate density
    from scipy.interpolate import griddata
    Zi = griddata((df['mw'], df['logp']), z, (Xi, Yi), method='cubic')
    
    # Draw contours
    contours = ax.contour(Xi, Yi, Zi, levels=8, colors='black', 
                         alpha=0.3, linewidths=2, zorder=3)
    ax.clabel(contours, inline=True, fontsize=18, fmt='%1.2f')
    
    # Target zones
    target_mw_min, target_mw_max = 150, 450
    target_logp_min, target_logp_max = 2, 6
    
    # Draw target zone
    rect = Rectangle((target_mw_min, target_logp_min), 
                     target_mw_max - target_mw_min,
                     target_logp_max - target_logp_min,
                     fill=True, alpha=0.15, color='green',
                     edgecolor='green', linewidth=4, linestyle='--',
                     zorder=1, label='Target Zone')
    ax.add_patch(rect)
    
    # Calculate statistics
    in_target = ((df['mw'] >= target_mw_min) & (df['mw'] <= target_mw_max) &
                 (df['logp'] >= target_logp_min) & (df['logp'] <= target_logp_max))
    pct_in_target = (in_target.sum() / len(df)) * 100
    
    # Add marginal histograms as insets
    # Top histogram (MW distribution)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    ax_hist_top = inset_axes(ax, width="100%", height="15%", 
                            loc='upper center', bbox_to_anchor=(0, 1.15, 1, 1),
                            bbox_transform=ax.transAxes, borderpad=0)
    ax_hist_top.hist(df['mw'], bins=30, color='#3498db', 
                    edgecolor='black', linewidth=2, alpha=0.7)
    ax_hist_top.axvline(df['mw'].mean(), color='red', 
                       linestyle='--', linewidth=3, alpha=0.8)
    ax_hist_top.set_ylabel('Count', fontsize=20, fontweight='bold')
    ax_hist_top.tick_params(labelsize=18)
    ax_hist_top.grid(True, alpha=0.3, axis='y')
    ax_hist_top.set_xlim(ax.get_xlim())
    
    # Right histogram (LogP distribution)
    ax_hist_right = inset_axes(ax, width="15%", height="100%",
                               loc='center right', bbox_to_anchor=(1.15, 0, 1, 1),
                               bbox_transform=ax.transAxes, borderpad=0)
    ax_hist_right.hist(df['logp'], bins=30, orientation='horizontal',
                      color='#e74c3c', edgecolor='black', linewidth=2, alpha=0.7)
    ax_hist_right.axhline(df['logp'].mean(), color='blue',
                         linestyle='--', linewidth=3, alpha=0.8)
    ax_hist_right.set_xlabel('Count', fontsize=20, fontweight='bold')
    ax_hist_right.tick_params(labelsize=18)
    ax_hist_right.grid(True, alpha=0.3, axis='x')
    ax_hist_right.set_ylim(ax.get_ylim())
    
    # Main plot styling
    ax.set_xlabel('Molecular Weight (g/mol)', fontsize=36, fontweight='bold')
    ax.set_ylabel('LogP (Partition Coefficient)', fontsize=36, fontweight='bold')
    ax.set_title(f'CRAM Molecular Property Space with Density Analysis\n' + 
                f'{len(df)} molecules | {pct_in_target:.1f}% in target zone',
                fontsize=44, fontweight='bold', pad=100)
    ax.tick_params(labelsize=32)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=2)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('CRAM Quality Score', fontsize=32, fontweight='bold')
    cbar.ax.tick_params(labelsize=28)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.3, label=f'Target Zone ({pct_in_target:.1f}%)'),
        plt.Line2D([0], [0], color='black', linewidth=2, alpha=0.5, 
                   linestyle='-', label='Density Contours')
    ]
    ax.legend(handles=legend_elements, fontsize=28, loc='upper left',
             frameon=True, shadow=True, fancybox=True)
    
    # Add statistics box
    stats_text = f"Statistics:\n"
    stats_text += f"Mean MW: {df['mw'].mean():.1f} g/mol\n"
    stats_text += f"Mean LogP: {df['logp'].mean():.2f}\n"
    stats_text += f"Mean CRAM: {df['cram_score'].mean():.3f}\n"
    stats_text += f"Excellent (≥0.85): {(df['cram_score'] >= 0.85).sum()}"
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=22, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor='black', linewidth=3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ============================================================================
# FIGURE 2: Radial CRAM Feature Plot (Radar/Spider Chart)
# ============================================================================

def plot_cram_radial_features(cram_data, save_path='fig2_cram_radial_features.png', dpi=700):
    """
    Radial plot showing CRAM feature profiles for top molecules.
    Each molecule represented as a polygon showing all features.
    """
    df = pd.DataFrame(cram_data)
    
    # Select top 10 molecules by CRAM score
    top_molecules = df.nsmallest(10, 'cram_score', keep='first').copy()
    
    # Normalize features to 0-1 scale
    features = ['cram_score', 'carboxyl_count', 'aliphatic_percent', 'mw', 'logp']
    feature_labels = ['CRAM\nScore', 'Carboxyl\nGroups', 'Aliphatic\n%', 
                     'Molecular\nWeight', 'LogP']
    
    # Normalize
    top_molecules['carboxyl_norm'] = top_molecules['carboxyl_count'] / 4.0
    top_molecules['aliphatic_norm'] = top_molecules['aliphatic_percent'] / 100.0
    top_molecules['mw_norm'] = (top_molecules['mw'] - 150) / (450 - 150)
    top_molecules['logp_norm'] = (top_molecules['logp'] - 2) / (6 - 2)
    
    normalized_features = ['cram_score', 'carboxyl_norm', 'aliphatic_norm', 
                          'mw_norm', 'logp_norm']
    
    # Create radial plot
    fig, ax = plt.subplots(figsize=(22, 22), dpi=dpi, subplot_kw=dict(projection='polar'))
    
    # Number of features
    num_vars = len(normalized_features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_molecules)))
    
    # Plot each molecule
    for idx, (i, row) in enumerate(top_molecules.iterrows()):
        values = [row[feat] for feat in normalized_features]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=4, 
               color=colors[idx], alpha=0.8, markersize=12,
               label=f"Mol {idx+1} (Score: {row['cram_score']:.3f})")
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=28, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add grid circles
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                       fontsize=24, color='gray')
    ax.grid(True, linewidth=2, alpha=0.5, linestyle='--')
    
    # Title
    ax.set_title('CRAM Feature Profiles: Top 10 Molecules\n' + 
                'Radial Analysis of Molecular Characteristics',
                fontsize=44, fontweight='bold', pad=50, y=1.15)
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15),
             fontsize=20, frameon=True, shadow=True, fancybox=True)
    
    # Add reference circle for ideal molecule
    ideal_values = [0.9, 0.5, 0.75, 0.5, 0.5]  # Ideal normalized values
    ideal_values += ideal_values[:1]
    ax.plot(angles, ideal_values, 'k--', linewidth=5, alpha=0.5, 
           label='Ideal Profile', zorder=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ============================================================================
# FIGURE 3: Hierarchical Binding Affinity Heatmap
# ============================================================================

def plot_binding_affinity_heatmap(docking_results, enzyme_data, 
                                  save_path='fig3_binding_affinity_heatmap.png', dpi=700):
    """
    Hierarchical clustered heatmap showing binding affinities
    between enzymes and CRAM molecules.
    """
    df_dock = pd.DataFrame(docking_results)
    df_enzyme = pd.DataFrame(enzyme_data)
    
    # Create matrix (enzymes × molecules)
    pivot = df_dock.pivot_table(
        index='enzyme_id',
        columns='cram_molecule',
        values='binding_affinity',
        aggfunc='mean'
    ).fillna(0)
    
    # Limit to top 30 enzymes and 40 molecules for visibility
    pivot = pivot.iloc[:30, :40]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(28, 24), dpi=dpi)
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                  interpolation='nearest', vmin=-12, vmax=0)
    
    # Add enzyme cluster colors on the left
    enzyme_clusters = []
    for enz_id in pivot.index:
        cluster = df_enzyme[df_enzyme['enzyme_id'] == enz_id]['cluster'].values
        if len(cluster) > 0:
            enzyme_clusters.append(cluster[0])
        else:
            enzyme_clusters.append(0)
    
    # Create colored bars on the left
    cluster_colors = [ENZYME_COLORS[f'cluster_{c+1}'] for c in enzyme_clusters]
    
    for i, color in enumerate(cluster_colors):
        rect = Rectangle((-2, i-0.4), 1.5, 0.8, 
                        facecolor=color, edgecolor='black', 
                        linewidth=2, clip_on=False)
        ax.add_patch(rect)
    
    # Axes labels
    ax.set_xticks(np.arange(len(pivot.columns))[::5])
    ax.set_xticklabels([f'M{i}' for i in range(0, len(pivot.columns), 5)],
                      fontsize=18, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([enz[:20] for enz in pivot.index], fontsize=16)
    
    ax.set_xlabel('CRAM Molecules', fontsize=36, fontweight='bold')
    ax.set_ylabel('Marine Enzymes', fontsize=36, fontweight='bold')
    ax.set_title('Enzyme-CRAM Binding Affinity Matrix\n' + 
                f'Hierarchical Analysis ({len(pivot.index)} enzymes × {len(pivot.columns)} molecules)',
                fontsize=44, fontweight='bold', pad=30)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Binding Affinity (kcal/mol)', fontsize=32, fontweight='bold')
    cbar.ax.tick_params(labelsize=28)
    
    # Add grid
    ax.set_xticks(np.arange(len(pivot.columns))+0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot.index))+0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Add strong binding threshold line
    threshold_val = -8
    # Find positions where binding is strong
    strong_mask = pivot.values < threshold_val
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if strong_mask[i, j]:
                circle = Circle((j, i), 0.3, color='blue', alpha=0.3, zorder=10)
                ax.add_patch(circle)
    
    # Legend for clusters
    legend_elements = [mpatches.Patch(color=ENZYME_COLORS[f'cluster_{i+1}'], 
                                     label=f'Cluster {i}')
                      for i in range(5)]
    legend_elements.append(Circle((0, 0), 1, color='blue', alpha=0.3, 
                                 label='Strong Binding (<-8)'))
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.15, 1), fontsize=24,
             frameon=True, shadow=True)
    
    # Statistics box
    strong_count = (pivot.values < -8).sum()
    total = pivot.size
    stats_text = f"Strong Binders: {strong_count}/{total}\n"
    stats_text += f"Success Rate: {(strong_count/total)*100:.1f}%\n"
    stats_text += f"Best: {pivot.values.min():.2f} kcal/mol"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=22, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                    edgecolor='black', linewidth=3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ============================================================================
# FIGURE 4: MD Trajectory Landscape (All-in-One)
# ============================================================================

def plot_md_trajectory_landscape(md_results, 
                                save_path='fig4_md_trajectory_landscape.png', dpi=700):
    """
    Single comprehensive plot showing MD trajectories with multiple
    overlaid data streams (RMSD, energy, contacts) using dual y-axes.
    """
    # Select top 3 most stable complexes
    top_3 = sorted(md_results, key=lambda x: np.mean(x['rmsd']))[:3]
    
    fig, ax1 = plt.subplots(figsize=(26, 18), dpi=dpi)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Primary y-axis: RMSD
    for i, result in enumerate(top_3):
        color = colors[i]
        
        # RMSD (solid line, thick)
        line1 = ax1.plot(result['time'], result['rmsd'],
                        linewidth=5, alpha=0.9, color=color,
                        label=f"{result['enzyme_id'][:20]} - RMSD",
                        zorder=10-i)
        
        # Add shaded area for RMSD
        ax1.fill_between(result['time'], 0, result['rmsd'],
                        alpha=0.15, color=color)
    
    ax1.set_xlabel('Simulation Time (ns)', fontsize=36, fontweight='bold')
    ax1.set_ylabel('RMSD (Å)', fontsize=36, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelsize=32, labelcolor='black')
    ax1.tick_params(axis='x', labelsize=32)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=2)
    
    # Add stability threshold
    ax1.axhline(2.0, color='red', linestyle='--', linewidth=4,
               alpha=0.7, label='Stability Threshold (2.0 Å)', zorder=15)
    
    # Secondary y-axis: Hydrogen bonds
    ax2 = ax1.twinx()
    
    for i, result in enumerate(top_3):
        color = colors[i]
        
        # H-bonds (dashed line)
        ax2.plot(result['time'], result['h_bonds'],
                linestyle=':', linewidth=4, alpha=0.7, color=color,
                label=f"{result['enzyme_id'][:20]} - H-Bonds")
    
    ax2.set_ylabel('Hydrogen Bond Count', fontsize=36, fontweight='bold', 
                  color='purple')
    ax2.tick_params(axis='y', labelsize=32, labelcolor='purple')
    
    # Add energy overlay as background intensity
    for i, result in enumerate(top_3):
        # Normalize potential energy to 0-1 for alpha channel
        pe_norm = (result['potential_energy'] - result['potential_energy'].min()) / \
                  (result['potential_energy'].max() - result['potential_energy'].min())
        
        # Create colored background showing energy
        for j in range(len(result['time'])-1):
            alpha_val = 0.1 + (1 - pe_norm[j]) * 0.2
            ax1.axvspan(result['time'][j], result['time'][j+1],
                       alpha=alpha_val, color=colors[i], zorder=0)
    
    # Title
    ax1.set_title('Molecular Dynamics Trajectory Landscape\n' + 
                 'Structural Stability, Interactions & Energy Profile',
                 fontsize=44, fontweight='bold', pad=30)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              loc='upper left', fontsize=22, ncol=2,
              frameon=True, shadow=True, fancybox=True)
    
    # Add annotation boxes for key events
    for i, result in enumerate(top_3):
        # Find equilibration time (when RMSD stabilizes)
        rmsd_grad = np.gradient(result['rmsd'])
        stable_idx = np.where(np.abs(rmsd_grad) < 0.01)[0]
        
        if len(stable_idx) > 0:
            equil_time = result['time'][stable_idx[0]]
            equil_rmsd = result['rmsd'][stable_idx[0]]
            
            ax1.annotate(f'Equilibration\nt={equil_time:.1f} ns',
                        xy=(equil_time, equil_rmsd),
                        xytext=(equil_time + 10, equil_rmsd + 0.5),
                        fontsize=18, fontweight='bold',
                        color=colors[i],
                        bbox=dict(boxstyle='round', facecolor='white',
                                 edgecolor=colors[i], linewidth=3, alpha=0.9),
                        arrowprops=dict(arrowstyle='->', color=colors[i],
                                      linewidth=3))
    
    # Statistics panel
    stats_text = "Simulation Statistics:\n\n"
    for i, result in enumerate(top_3):
        stats_text += f"Complex {i+1}:\n"
        stats_text += f"  Mean RMSD: {np.mean(result['rmsd']):.2f} Å\n"
        stats_text += f"  Mean H-Bonds: {np.mean(result['h_bonds']):.1f}\n"
        stats_text += f"  Temp: {np.mean(result['temperature']):.1f} K\n\n"
    
    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
            fontsize=20, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                     edgecolor='black', linewidth=3),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ============================================================================
# FIGURE 5: Network Graph of Enzyme-CRAM Interactions
# ============================================================================

def plot_interaction_network(docking_results, enzyme_data,
                             save_path='fig5_interaction_network.png', dpi=700):
    """
    Network graph showing connections between enzyme clusters and
    CRAM molecules, with edge weights representing binding affinity.
    """
    df_dock = pd.DataFrame(docking_results)
    df_enzyme = pd.DataFrame(enzyme_data)
    
    # Filter for strong binders only
    strong_binders = df_dock[df_dock['binding_affinity'] < -8].copy()
    
    # Take top 50 interactions
    strong_binders = strong_binders.nsmallest(50, 'binding_affinity')
    
    fig, ax = plt.subplots(figsize=(26, 26), dpi=dpi)
    
    # Create circular layout
    n_enzymes = strong_binders['enzyme_id'].nunique()
    n_molecules = strong_binders['cram_molecule'].nunique()
    
    # Enzyme positions (left semicircle)
    enzyme_ids = strong_binders['enzyme_id'].unique()
    enzyme_angles = np.linspace(np.pi/4, 3*np.pi/4, n_enzymes)
    enzyme_radius = 8
    enzyme_pos = {enz: (enzyme_radius * np.cos(angle), enzyme_radius * np.sin(angle))
                 for enz, angle in zip(enzyme_ids, enzyme_angles)}
    
    # Molecule positions (right semicircle)
    molecule_ids = strong_binders['cram_molecule'].unique()
    molecule_angles = np.linspace(-np.pi/4, np.pi/4, n_molecules)
    molecule_radius = 8
    molecule_pos = {mol: (molecule_radius * np.cos(angle), molecule_radius * np.sin(angle))
                   for mol, angle in zip(molecule_ids, molecule_angles)}
    
    # Draw edges (interactions)
    for _, row in strong_binders.iterrows():
        enz = row['enzyme_id']
        mol = row['cram_molecule']
        affinity = row['binding_affinity']
        cluster = row['cluster']
        
        if enz in enzyme_pos and mol in molecule_pos:
            x1, y1 = enzyme_pos[enz]
            x2, y2 = molecule_pos[mol]
            
            # Line width based on affinity (stronger = thicker)
            linewidth = (-affinity - 8) * 2  # Scale to visible range
            
            # Color based on cluster
            color = ENZYME_COLORS[f'cluster_{cluster+1}']
            
            ax.plot([x1, x2], [y1, y2], color=color, 
                   linewidth=linewidth, alpha=0.6, zorder=1)
    
    # Draw enzyme nodes
    for enz_id in enzyme_ids:
        x, y = enzyme_pos[enz_id]
        
        # Get cluster
        cluster_match = strong_binders[strong_binders['enzyme_id'] == enz_id]['cluster'].values
        if len(cluster_match) > 0:
            cluster = cluster_match[0]
            color = ENZYME_COLORS[f'cluster_{cluster+1}']
        else:
            color = 'gray'
        
        circle = Circle((x, y), 0.4, facecolor=color, 
                       edgecolor='black', linewidth=3, zorder=5)
        ax.add_patch(circle)
        
        # Label
        ax.text(x - 0.8, y, enz_id[:10], fontsize=14, 
               ha='right', va='center', fontweight='bold')
    
    # Draw molecule nodes
    for mol_id in molecule_ids:
        x, y = molecule_pos[mol_id]
        
        # Get average CRAM score
        cram_score = strong_binders[strong_binders['cram_molecule'] == mol_id]['cram_score'].mean()
        
        # Color by CRAM score
        if cram_score >= 0.85:
            color = CRAM_COLORS['excellent']
        elif cram_score >= 0.70:
            color = CRAM_COLORS['good']
        else:
            color = CRAM_COLORS['moderate']
        
        square = Rectangle((x-0.3, y-0.3), 0.6, 0.6, 
                          facecolor=color, edgecolor='black', 
                          linewidth=3, zorder=5)
        ax.add_patch(square)
        
        # Label
        mol_idx = list(molecule_ids).index(mol_id)
        ax.text(x + 0.8, y, f'M{mol_idx}', fontsize=14,
               ha='left', va='center', fontweight='bold')
    
    # Add section labels
    ax.text(-10, 0, 'Marine\nEnzymes', fontsize=32, fontweight='bold',
           ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', 
                    edgecolor='black', linewidth=3, alpha=0.8))
    
    ax.text(10, 0, 'CRAM\nMolecules', fontsize=32, fontweight='bold',
           ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen',
                    edgecolor='black', linewidth=3, alpha=0.8))
    
    # Set limits and styling
    ax.set_xlim(-12, 12)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.set_title(f'Enzyme-CRAM Interaction Network\n' + 
                f'{len(strong_binders)} Strong Binding Interactions (ΔG < -8 kcal/mol)',
                fontsize=44, fontweight='bold', pad=30, y=0.98)
    
    # Legend
    legend_elements = []
    for i in range(5):
        legend_elements.append(
            mpatches.Patch(color=ENZYME_COLORS[f'cluster_{i+1}'], 
                          label=f'Enzyme Cluster {i}')
        )
    legend_elements.append(
        mpatches.Patch(color='white', edgecolor='black', linewidth=3,
                      label='Enzymes (circles)')
    )
    legend_elements.append(
        mpatches.Patch(color='white', edgecolor='black', linewidth=3,
                      label='CRAM Molecules (squares)')
    )
    
    ax.legend(handles=legend_elements, loc='upper right',
             fontsize=22, frameon=True, shadow=True, fancybox=True)
    
    # Statistics box
    stats_text = f"Network Statistics:\n"
    stats_text += f"Enzymes: {n_enzymes}\n"
    stats_text += f"Molecules: {n_molecules}\n"
    stats_text += f"Interactions: {len(strong_binders)}\n"
    stats_text += f"Avg Affinity: {strong_binders['binding_affinity'].mean():.2f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=24, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='black', linewidth=3),
           family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ============================================================================
# FIGURE 6: Sankey Diagram of Pipeline Flow
# ============================================================================

def plot_pipeline_sankey(cram_data, enzyme_data, docking_results, md_results,
                         save_path='fig6_pipeline_sankey.png', dpi=700):
    """
    Sankey-style flow diagram showing molecule progression through pipeline.
    """
    df_cram = pd.DataFrame(cram_data)
    df_enzyme = pd.DataFrame(enzyme_data)
    df_dock = pd.DataFrame(docking_results)
    
    fig, ax = plt.subplots(figsize=(28, 20), dpi=dpi)
    
    # Define stages
    stages = ['Generated\nCRAM', 'Valid\nCRAM', 'Predicted\nEnzymes',
             'Docked\nComplexes', 'Strong\nBinders', 'Stable\nMD']
    stage_x = [0, 1, 2, 3, 4, 5]
    
    # Calculate counts
    counts = [
        len(df_cram),
        (df_cram['cram_score'] > 0.5).sum(),
        len(df_enzyme),
        len(df_dock),
        (df_dock['binding_affinity'] < -8).sum(),
        sum(np.mean(r['rmsd']) < 2.0 for r in md_results)
    ]
    
    # Calculate flow widths (normalized)
    max_count = max(counts)
    widths = [c / max_count * 8 for c in counts]
    
    # Draw flow boxes
    colors_stage = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c']
    
    for i, (x, width, count, color) in enumerate(zip(stage_x, widths, counts, colors_stage)):
        # Box
        rect = FancyBboxPatch((x - 0.3, -width/2), 0.6, width,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black',
                              linewidth=4, alpha=0.8)
        ax.add_patch(rect)
        
        # Stage label
        ax.text(x, width/2 + 0.8, stages[i], 
               fontsize=28, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor='black', linewidth=2, alpha=0.9))
        
        # Count label
        ax.text(x, 0, f'{count:,}', 
               fontsize=32, fontweight='bold', ha='center', va='center',
               color='white')
        
        # Percentage
        pct = (count / counts[0]) * 100
        ax.text(x, -width/2 - 0.6, f'{pct:.1f}%',
               fontsize=24, ha='center', color=color, fontweight='bold')
    
    # Draw connecting flows
    for i in range(len(stage_x) - 1):
        x1, x2 = stage_x[i], stage_x[i+1]
        w1, w2 = widths[i], widths[i+1]
        
        # Top curve
        points_top = np.array([
            [x1 + 0.3, w1/2],
            [x1 + 0.5, w1/2],
            [x2 - 0.5, w2/2],
            [x2 - 0.3, w2/2]
        ])
        
        # Bottom curve
        points_bottom = np.array([
            [x1 + 0.3, -w1/2],
            [x1 + 0.5, -w1/2],
            [x2 - 0.5, -w2/2],
            [x2 - 0.3, -w2/2]
        ])
        
        # Create polygon
        from matplotlib.patches import Polygon
        verts = np.vstack([points_top, points_bottom[::-1]])
        poly = Polygon(verts, facecolor=colors_stage[i], 
                      edgecolor='black', linewidth=2, alpha=0.4)
        ax.add_patch(poly)
        
        # Add loss annotation
        loss = counts[i] - counts[i+1]
        loss_pct = (loss / counts[i]) * 100
        
        if loss > 0:
            ax.text((x1 + x2) / 2, max(w1, w2)/2 + 1.5,
                   f'Lost: {loss:,}\n({loss_pct:.1f}%)',
                   fontsize=18, ha='center', color='red',
                   fontweight='bold', style='italic',
                   bbox=dict(boxstyle='round', facecolor='white',
                            edgecolor='red', linewidth=2, alpha=0.8))
    
    # Styling
    ax.set_xlim(-0.8, 5.8)
    ax.set_ylim(-6, 6)
    ax.axis('off')
    
    # Title
    ax.set_title('CRAM Generator Pipeline Flow Diagram\n' + 
                'From Molecule Generation to Stable MD Complexes',
                fontsize=48, fontweight='bold', pad=30, y=0.98)
    
    # Add efficiency metrics
    final_efficiency = (counts[-1] / counts[0]) * 100
    
    metrics_text = "Pipeline Efficiency:\n\n"
    metrics_text += f"Initial: {counts[0]:,} molecules\n"
    metrics_text += f"Final: {counts[-1]:,} stable complexes\n"
    metrics_text += f"Success Rate: {final_efficiency:.2f}%\n\n"
    metrics_text += "Quality Gates:\n"
    metrics_text += f"✓ CRAM validation: {(counts[1]/counts[0])*100:.1f}%\n"
    metrics_text += f"✓ Strong binding: {(counts[4]/counts[3])*100:.1f}%\n"
    metrics_text += f"✓ MD stability: {(counts[5]/counts[4])*100:.1f}%"
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
           fontsize=22, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow',
                    edgecolor='black', linewidth=3, alpha=0.95),
           family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ============================================================================
# DEMO DATA GENERATION
# ============================================================================

def generate_demo_data():
    """Generate demo data for testing."""
    np.random.seed(42)
    
    # CRAM molecules
    n_molecules = 100
    cram_data = []
    for i in range(n_molecules):
        cram_data.append({
            'smiles': f'C1CCC(C(=O)O)CC1CCC(=O)O_mol{i}',
            'cram_score': np.random.beta(5, 2),
            'carboxyl_count': np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]),
            'aliphatic_percent': np.clip(np.random.normal(75, 10), 40, 100),
            'mw': np.clip(np.random.normal(300, 75), 100, 600),
            'logp': np.clip(np.random.normal(4, 1.5), -2, 8)
        })
    
    # Enzymes
    n_enzymes = 50
    enzyme_data = []
    for i in range(n_enzymes):
        enzyme_data.append({
            'enzyme_id': f'Enzyme_{i:03d}',
            'sequence': 'MKVLWAALLVTFLAGCQAKV' * 10,
            'cluster': np.random.choice([0, 1, 2, 3, 4]),
            'plddt': np.clip(np.random.normal(80, 10), 40, 100),
        })
    
    # Docking results
    n_docking = 200
    docking_results = []
    for i in range(n_docking):
        cluster = np.random.choice([0, 1, 2, 3, 4])
        cram_idx = np.random.randint(0, n_molecules)
        docking_results.append({
            'cram_molecule': cram_data[cram_idx]['smiles'],
            'cram_score': cram_data[cram_idx]['cram_score'],
            'enzyme_id': f'Enzyme_{np.random.randint(0, n_enzymes):03d}',
            'cluster': cluster,
            'binding_affinity': np.clip(np.random.normal(-7, 2), -15, 0),
            'docking_score': np.clip(np.random.normal(100, 20), 40, 200),
        })
    
    # MD results
    n_md = 30
    md_results = []
    for i in range(n_md):
        time = np.linspace(0, 100, 1000)
        cluster = np.random.choice([0, 1, 2, 3, 4])
        
        rmsd_baseline = np.random.uniform(0.5, 2.5)
        rmsd = rmsd_baseline + np.cumsum(np.random.randn(1000) * 0.01)
        rmsd = np.clip(rmsd, 0, 5)
        
        pe = np.random.normal(-150000, 5000, 1000)
        temp = np.random.normal(277, 5, 1000)
        h_bonds = np.random.poisson(8, 1000)
        
        md_results.append({
            'complex_id': f'Complex_{i:03d}',
            'enzyme_id': f'Enzyme_{np.random.randint(0, n_enzymes):03d}',
            'cluster': cluster,
            'time': time,
            'rmsd': rmsd,
            'potential_energy': pe,
            'kinetic_energy': np.random.normal(50000, 2000, 1000),
            'temperature': temp,
            'h_bonds': h_bonds,
            'salt_bridges': np.random.poisson(3, 1000),
            'hydrophobic_contacts': np.random.poisson(12, 1000)
        })
    
    return cram_data, enzyme_data, docking_results, md_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎨 ADVANCED SINGLE PLOT SUITE (NO SUBPLOTS)")
    print("="*80)
    print("\n6 complex standalone visualizations at 700 DPI for A0 poster\n")
    
    # Generate demo data
    cram_data, enzyme_data, docking_results, md_results = generate_demo_data()
    
    print(f"📊 Data loaded:")
    print(f"  - CRAM molecules: {len(cram_data)}")
    print(f"  - Enzymes: {len(enzyme_data)}")
    print(f"  - Docking results: {len(docking_results)}")
    print(f"  - MD simulations: {len(md_results)}\n")
    
    print("Generating complex single plots...\n")
    
    try:
        plot_cram_property_space_density(cram_data)
        plot_cram_radial_features(cram_data)
        plot_binding_affinity_heatmap(docking_results, enzyme_data)
        plot_md_trajectory_landscape(md_results)
        plot_interaction_network(docking_results, enzyme_data)
        plot_pipeline_sankey(cram_data, enzyme_data, docking_results, md_results)
        
        print("\n" + "="*80)
        print("✅ ALL COMPLEX SINGLE PLOTS GENERATED!")
        print("="*80)
        print("\nGenerated files:")
        print("  1. fig1_cram_property_space_density.png - Property space + density")
        print("  2. fig2_cram_radial_features.png - Radial feature profiles")
        print("  3. fig3_binding_affinity_heatmap.png - Hierarchical heatmap")
        print("  4. fig4_md_trajectory_landscape.png - Comprehensive MD view")
        print("  5. fig5_interaction_network.png - Network graph")
        print("  6. fig6_pipeline_sankey.png - Flow diagram")
        print("\nAll optimized for A0 poster at 700 DPI")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
