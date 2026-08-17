"""
CRAM Generator: Individual Plot Generator
==========================================

Generates each visualization as a separate high-resolution file (700 DPI)
Optimized for A0 poster presentation

Each function creates ONE standalone figure.

Author: CRAM Generator Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Scientific visualization settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

# Color palettes
CRAM_COLORS = {
    'excellent': '#2ecc71',
    'good': '#3498db',
    'moderate': '#f39c12',
    'poor': '#e74c3c'
}

ENZYME_COLORS = {
    'cluster_1': '#8e44ad',
    'cluster_2': '#16a085',
    'cluster_3': '#d35400',
    'cluster_4': '#c0392b',
    'cluster_5': '#2c3e50'
}


# ===========================================================================
# PLOT 1: CRAM Score Distribution
# ===========================================================================

def plot_cram_score_distribution(cram_data, save_path='01_cram_score_distribution.png', dpi=700):
    """CRAM quality score distribution with statistics."""
    df = pd.DataFrame(cram_data)
    
    fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi)
    
    n, bins, patches = ax.hist(df['cram_score'], bins=40, 
                                edgecolor='black', linewidth=2, alpha=0.8)
    
    # Color by quality
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center >= 0.85:
            patch.set_facecolor(CRAM_COLORS['excellent'])
        elif bin_center >= 0.70:
            patch.set_facecolor(CRAM_COLORS['good'])
        elif bin_center >= 0.50:
            patch.set_facecolor(CRAM_COLORS['moderate'])
        else:
            patch.set_facecolor(CRAM_COLORS['poor'])
    
    # Statistics
    mean_score = df['cram_score'].mean()
    median_score = df['cram_score'].median()
    
    ax.axvline(mean_score, color='red', linestyle='--', linewidth=4,
              label=f'Mean: {mean_score:.3f}', alpha=0.8)
    ax.axvline(median_score, color='blue', linestyle='--', linewidth=4,
              label=f'Median: {median_score:.3f}', alpha=0.8)
    
    ax.set_xlabel('CRAM Score', fontsize=32, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=32, fontweight='bold')
    ax.set_title('Distribution of CRAM Quality Scores\n' + 
                f'n={len(df)} molecules',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=28, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=28)
    
    # Add quality zones
    ax.axvspan(0.85, 1.0, alpha=0.1, color=CRAM_COLORS['excellent'])
    ax.axvspan(0.70, 0.85, alpha=0.1, color=CRAM_COLORS['good'])
    ax.axvspan(0.50, 0.70, alpha=0.1, color=CRAM_COLORS['moderate'])
    ax.axvspan(0, 0.50, alpha=0.1, color=CRAM_COLORS['poor'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 2: Carboxyl Group Distribution
# ===========================================================================

def plot_carboxyl_distribution(cram_data, save_path='02_carboxyl_distribution.png', dpi=700):
    """Distribution of carboxyl groups in CRAM molecules."""
    df = pd.DataFrame(cram_data)
    
    fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi)
    
    counts = df['carboxyl_count'].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, 
                  color='#e74c3c', edgecolor='black', linewidth=3,
                  width=0.7)
    
    # Add percentages
    total = len(df)
    for bar, count in zip(bars, counts.values):
        height = bar.get_height()
        pct = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=24, fontweight='bold')
    
    ax.set_xlabel('Number of Carboxyl Groups (-COOH)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Count', fontsize=32, fontweight='bold')
    ax.set_title('Carboxyl Group Distribution in Generated CRAM Molecules',
                fontsize=40, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=28)
    ax.set_xticks(counts.index)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 3: Aliphatic Character Distribution
# ===========================================================================

def plot_aliphatic_distribution(cram_data, save_path='03_aliphatic_distribution.png', dpi=700):
    """Aliphatic character percentage distribution."""
    df = pd.DataFrame(cram_data)
    
    fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi)
    
    ax.hist(df['aliphatic_percent'], bins=35, 
           color='#2ecc71', edgecolor='black', linewidth=2, alpha=0.8)
    
    # Target line
    ax.axvline(70, color='red', linestyle='--', linewidth=4,
              label='Target: ≥70%', alpha=0.8)
    
    # Mean
    mean_val = df['aliphatic_percent'].mean()
    ax.axvline(mean_val, color='blue', linestyle='--', linewidth=4,
              label=f'Mean: {mean_val:.1f}%', alpha=0.8)
    
    # Achievement rate
    above_target = (df['aliphatic_percent'] >= 70).sum()
    pct_above = (above_target / len(df)) * 100
    
    ax.set_xlabel('Aliphatic Character (%)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=32, fontweight='bold')
    ax.set_title(f'Aliphatic Character Distribution\n' + 
                f'{above_target}/{len(df)} molecules (% {pct_above:.1f}%) meet ≥70% target',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=28, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=28)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 4: Molecular Weight vs LogP
# ===========================================================================

def plot_mw_logp_space(cram_data, save_path='04_mw_logp_space.png', dpi=700):
    """Molecular property space colored by CRAM score."""
    df = pd.DataFrame(cram_data)
    
    fig, ax = plt.subplots(figsize=(18, 14), dpi=dpi)
    
    scatter = ax.scatter(df['mw'], df['logp'],
                        c=df['cram_score'], s=300,
                        cmap='RdYlGn', edgecolors='black', linewidth=2,
                        alpha=0.7, vmin=0, vmax=1)
    
    # Target regions
    ax.axvspan(150, 450, alpha=0.1, color='green', label='Target MW: 150-450')
    ax.axhspan(2, 6, alpha=0.1, color='blue', label='Target LogP: 2-6')
    
    ax.set_xlabel('Molecular Weight (g/mol)', fontsize=32, fontweight='bold')
    ax.set_ylabel('LogP (Partition Coefficient)', fontsize=32, fontweight='bold')
    ax.set_title('Molecular Property Space of Generated CRAM Molecules',
                fontsize=40, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=28)
    ax.legend(fontsize=24, frameon=True, shadow=True, loc='upper left')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('CRAM Score', fontsize=28, fontweight='bold')
    cbar.ax.tick_params(labelsize=24)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 5: Enzyme Cluster Pie Chart
# ===========================================================================

def plot_enzyme_clusters_pie(enzyme_data, save_path='05_enzyme_clusters_pie.png', dpi=700):
    """Enzyme distribution across k-means clusters."""
    df = pd.DataFrame(enzyme_data)
    
    fig, ax = plt.subplots(figsize=(16, 16), dpi=dpi)
    
    cluster_sizes = df['cluster'].value_counts().sort_index()
    colors = [ENZYME_COLORS[f'cluster_{i+1}'] for i in cluster_sizes.index]
    labels = [f'Cluster {i}\n({count} enzymes)' 
             for i, count in zip(cluster_sizes.index, cluster_sizes.values)]
    
    wedges, texts, autotexts = ax.pie(cluster_sizes.values, labels=labels,
                                        colors=colors, autopct='%1.1f%%',
                                        startangle=90, 
                                        wedgeprops={'linewidth': 4, 'edgecolor': 'black'},
                                        textprops={'fontsize': 28, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(32)
        autotext.set_fontweight('bold')
    
    ax.set_title('Marine Enzyme Distribution Across K-Means Clusters',
                fontsize=44, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 6: pLDDT Confidence Score Distribution
# ===========================================================================

def plot_plddt_distribution(enzyme_data, save_path='06_plddt_distribution.png', dpi=700):
    """AlphaFold structure prediction confidence distribution."""
    df = pd.DataFrame(enzyme_data)
    
    fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi)
    
    # Histogram by cluster
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        color = ENZYME_COLORS[f'cluster_{cluster_id+1}']
        ax.hist(cluster_df['plddt'], bins=25, alpha=0.6,
               label=f'Cluster {cluster_id} (n={len(cluster_df)})',
               color=color, edgecolor='black', linewidth=2)
    
    # Confidence threshold
    ax.axvline(70, color='red', linestyle='--', linewidth=4,
              label='Confidence Threshold (70)', alpha=0.8)
    
    # Mean
    mean_plddt = df['plddt'].mean()
    ax.axvline(mean_plddt, color='blue', linestyle='--', linewidth=4,
              label=f'Mean: {mean_plddt:.1f}', alpha=0.8)
    
    # Success rate
    above_threshold = (df['plddt'] >= 70).sum()
    pct_above = (above_threshold / len(df)) * 100
    
    ax.set_xlabel('pLDDT Score (AlphaFold Confidence)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=32, fontweight='bold')
    ax.set_title(f'Structure Prediction Confidence Distribution\n' + 
                f'{above_threshold}/{len(df)} structures ({pct_above:.1f}%) above threshold',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=22, frameon=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=28)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 7: Binding Affinity Distribution
# ===========================================================================

def plot_binding_affinity_histogram(docking_results, save_path='07_binding_affinity_histogram.png', dpi=700):
    """Binding affinity distribution across all docking results."""
    df = pd.DataFrame(docking_results)
    
    fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi)
    
    ax.hist(df['binding_affinity'], bins=40, 
           color='#f39c12', edgecolor='black', linewidth=2, alpha=0.8)
    
    # Strong binding threshold
    ax.axvline(-8, color='red', linestyle='--', linewidth=4,
              label='Strong Binding Threshold (-8 kcal/mol)', alpha=0.8)
    
    # Mean
    mean_ba = df['binding_affinity'].mean()
    ax.axvline(mean_ba, color='blue', linestyle='--', linewidth=4,
              label=f'Mean: {mean_ba:.2f} kcal/mol', alpha=0.8)
    
    # Success rate
    strong_binders = (df['binding_affinity'] < -8).sum()
    pct_strong = (strong_binders / len(df)) * 100
    
    ax.set_xlabel('Binding Affinity (kcal/mol)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=32, fontweight='bold')
    ax.set_title(f'Binding Affinity Distribution\n' + 
                f'{strong_binders}/{len(df)} complexes ({pct_strong:.1f}%) with strong binding',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=28, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=28)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 8: Top 20 Binding Affinities
# ===========================================================================

def plot_top20_binding_affinities(docking_results, save_path='08_top20_binding_affinities.png', dpi=700):
    """Top 20 most stable enzyme-CRAM complexes."""
    df = pd.DataFrame(docking_results)
    top_20 = df.nsmallest(20, 'binding_affinity')
    
    fig, ax = plt.subplots(figsize=(18, 14), dpi=dpi)
    
    colors_bars = [ENZYME_COLORS[f'cluster_{c+1}'] for c in top_20['cluster'].values]
    
    y_pos = np.arange(len(top_20))
    bars = ax.barh(y_pos, top_20['binding_affinity'],
                   color=colors_bars, edgecolor='black', linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['enzyme_id'][:20]}" 
                        for _, row in top_20.iterrows()],
                       fontsize=20)
    ax.set_xlabel('Binding Affinity (kcal/mol)', fontsize=32, fontweight='bold')
    ax.set_title('Top 20 Most Stable CRAM-Enzyme Complexes',
                fontsize=40, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(labelsize=24)
    ax.axvline(-8, color='red', linestyle='--', linewidth=3, alpha=0.7)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_20['binding_affinity'])):
        ax.text(value - 0.3, i, f'{value:.2f}',
               va='center', ha='right', fontsize=18,
               fontweight='bold', color='white')
    
    # Legend
    legend_elements = [mpatches.Patch(color=ENZYME_COLORS[f'cluster_{i+1}'], 
                                     label=f'Cluster {i}')
                      for i in sorted(top_20['cluster'].unique())]
    ax.legend(handles=legend_elements, fontsize=24, frameon=True, 
             shadow=True, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 9: Binding Affinity by Cluster (Boxplot)
# ===========================================================================

def plot_binding_affinity_by_cluster(docking_results, save_path='09_binding_affinity_by_cluster.png', dpi=700):
    """Binding affinity distribution across enzyme clusters."""
    df = pd.DataFrame(docking_results)
    
    fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi)
    
    cluster_data = []
    cluster_labels = []
    cluster_colors = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_data.append(cluster_df['binding_affinity'].values)
        cluster_labels.append(f'Cluster {cluster_id}\n(n={len(cluster_df)})')
        cluster_colors.append(ENZYME_COLORS[f'cluster_{cluster_id+1}'])
    
    bp = ax.boxplot(cluster_data, labels=cluster_labels, patch_artist=True,
                   widths=0.6,
                   boxprops=dict(linewidth=3),
                   medianprops=dict(color='black', linewidth=4),
                   whiskerprops=dict(linewidth=3),
                   capprops=dict(linewidth=3),
                   flierprops=dict(marker='o', markersize=8, alpha=0.5))
    
    for patch, color in zip(bp['boxes'], cluster_colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Binding Affinity (kcal/mol)', fontsize=32, fontweight='bold')
    ax.set_title('Binding Affinity Distribution by Enzyme Cluster',
                fontsize=40, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=28)
    ax.axhline(-8, color='red', linestyle='--', linewidth=3, 
              alpha=0.7, label='Strong Binding Threshold')
    ax.legend(fontsize=24)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 10: Docking Score vs Binding Affinity
# ===========================================================================

def plot_docking_score_vs_affinity(docking_results, save_path='10_docking_score_vs_affinity.png', dpi=700):
    """Correlation between docking score and binding affinity."""
    df = pd.DataFrame(docking_results)
    
    fig, ax = plt.subplots(figsize=(18, 14), dpi=dpi)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        color = ENZYME_COLORS[f'cluster_{cluster_id+1}']
        
        ax.scatter(cluster_df['docking_score'],
                  cluster_df['binding_affinity'],
                  c=color, s=300, alpha=0.7,
                  edgecolors='black', linewidth=2,
                  label=f'Cluster {cluster_id} (n={len(cluster_df)})')
    
    # Trend line
    slope, intercept, r_value, _, _ = linregress(df['docking_score'],
                                                  df['binding_affinity'])
    x_trend = np.linspace(df['docking_score'].min(), 
                         df['docking_score'].max(), 100)
    y_trend = slope * x_trend + intercept
    ax.plot(x_trend, y_trend, 'k--', linewidth=4, alpha=0.7,
           label=f'Linear Fit: R² = {r_value**2:.3f}')
    
    ax.set_xlabel('Docking Score (a.u.)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Binding Affinity (kcal/mol)', fontsize=32, fontweight='bold')
    ax.set_title('Docking Score vs Binding Affinity\nCorrelation Analysis',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=22, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=28)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 11: MD RMSD Trajectories
# ===========================================================================

def plot_md_rmsd_trajectories(md_results, save_path='11_md_rmsd_trajectories.png', dpi=700):
    """RMSD over time for top complexes."""
    top_6 = sorted(md_results, key=lambda x: np.mean(x['rmsd']))[:6]
    
    fig, ax = plt.subplots(figsize=(20, 12), dpi=dpi)
    
    for result in top_6:
        cluster_color = ENZYME_COLORS[f'cluster_{result["cluster"]+1}']
        ax.plot(result['time'], result['rmsd'],
               linewidth=4, alpha=0.8, color=cluster_color,
               label=f"{result['enzyme_id'][:20]} (Cluster {result['cluster']})")
    
    ax.axhline(2.0, color='red', linestyle='--', linewidth=4,
              alpha=0.7, label='Stability Threshold (2.0 Å)')
    
    ax.set_xlabel('Time (ns)', fontsize=32, fontweight='bold')
    ax.set_ylabel('RMSD (Å)', fontsize=32, fontweight='bold')
    ax.set_title('Molecular Dynamics: RMSD Trajectories\nStructural Stability Over Time',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=22, frameon=True, fancybox=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=28)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 12: MD Potential Energy
# ===========================================================================

def plot_md_potential_energy(md_results, save_path='12_md_potential_energy.png', dpi=700):
    """Potential energy evolution during MD simulation."""
    top_6 = sorted(md_results, key=lambda x: np.mean(x['rmsd']))[:6]
    
    fig, ax = plt.subplots(figsize=(20, 12), dpi=dpi)
    
    for result in top_6:
        cluster_color = ENZYME_COLORS[f'cluster_{result["cluster"]+1}']
        ax.plot(result['time'], result['potential_energy'],
               linewidth=3, alpha=0.8, color=cluster_color,
               label=f"{result['enzyme_id'][:20]}")
    
    ax.set_xlabel('Time (ns)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Potential Energy (kJ/mol)', fontsize=32, fontweight='bold')
    ax.set_title('Molecular Dynamics: Potential Energy Evolution',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=24, frameon=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=28)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 13: MD Hydrogen Bond Dynamics
# ===========================================================================

def plot_md_hydrogen_bonds(md_results, save_path='13_md_hydrogen_bonds.png', dpi=700):
    """Hydrogen bond count over MD simulation time."""
    top_6 = sorted(md_results, key=lambda x: np.mean(x['rmsd']))[:6]
    
    fig, ax = plt.subplots(figsize=(20, 12), dpi=dpi)
    
    for result in top_6:
        cluster_color = ENZYME_COLORS[f'cluster_{result["cluster"]+1}']
        ax.plot(result['time'], result['h_bonds'],
               linewidth=3, alpha=0.8, color=cluster_color,
               label=f"{result['enzyme_id'][:20]} (mean: {np.mean(result['h_bonds']):.1f})")
    
    ax.set_xlabel('Time (ns)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Hydrogen Bond Count', fontsize=32, fontweight='bold')
    ax.set_title('Molecular Dynamics: Hydrogen Bond Dynamics',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=22, frameon=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=28)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 14: MD Temperature Control
# ===========================================================================

def plot_md_temperature(md_results, save_path='14_md_temperature.png', dpi=700):
    """Temperature control during deep-sea simulation."""
    top_6 = sorted(md_results, key=lambda x: np.mean(x['rmsd']))[:6]
    
    fig, ax = plt.subplots(figsize=(20, 12), dpi=dpi)
    
    for result in top_6:
        cluster_color = ENZYME_COLORS[f'cluster_{result["cluster"]+1}']
        ax.plot(result['time'], result['temperature'],
               linewidth=3, alpha=0.8, color=cluster_color,
               label=f"{result['enzyme_id'][:20]}")
    
    ax.axhline(277, color='blue', linestyle='--', linewidth=4,
              alpha=0.8, label='Target Temperature: 277 K (Deep-Sea)')
    
    ax.set_xlabel('Time (ns)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Temperature (K)', fontsize=32, fontweight='bold')
    ax.set_title('Molecular Dynamics: Temperature Control\nDeep-Sea Environment (4°C / 277 K)',
                fontsize=40, fontweight='bold', pad=20)
    ax.legend(fontsize=22, frameon=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=28)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# PLOT 15: Overall Pipeline Success Metrics
# ===========================================================================

def plot_pipeline_success_metrics(cram_data, enzyme_data, docking_results, md_results,
                                  save_path='15_pipeline_success_metrics.png', dpi=700):
    """Overall success rates across the entire pipeline."""
    df_cram = pd.DataFrame(cram_data)
    df_enzyme = pd.DataFrame(enzyme_data)
    df_docking = pd.DataFrame(docking_results)
    
    fig, ax = plt.subplots(figsize=(20, 14), dpi=dpi)
    
    metrics = {
        'Valid\nCRAM': (df_cram['cram_score'] > 0.5).sum() / len(df_cram) * 100,
        'Excellent\nCRAM': (df_cram['cram_score'] > 0.85).sum() / len(df_cram) * 100,
        'Confident\nStructures': (df_enzyme['plddt'] > 70).sum() / len(df_enzyme) * 100,
        'Strong\nBinders': (df_docking['binding_affinity'] < -8).sum() / len(df_docking) * 100,
        'Stable\nComplexes': sum(np.mean(r['rmsd']) < 2.0 for r in md_results) / len(md_results) * 100
    }
    
    colors_metrics = ['#2ecc71', '#27ae60', '#3498db', '#f39c12', '#e74c3c']
    bars = ax.bar(range(len(metrics)), list(metrics.values()),
                  color=colors_metrics, edgecolor='black', linewidth=4,
                  width=0.7)
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics.keys()), fontsize=32, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=36, fontweight='bold')
    ax.set_title('CRAM Generator Pipeline: Overall Success Metrics',
                fontsize=44, fontweight='bold', pad=20)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=32)
    
    # Add value labels
    for bar, (name, value) in zip(bars, metrics.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{value:.1f}%',
               ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    # Add count labels
    counts = [
        f'{(df_cram["cram_score"] > 0.5).sum()}/{len(df_cram)}',
        f'{(df_cram["cram_score"] > 0.85).sum()}/{len(df_cram)}',
        f'{(df_enzyme["plddt"] > 70).sum()}/{len(df_enzyme)}',
        f'{(df_docking["binding_affinity"] < -8).sum()}/{len(df_docking)}',
        f'{sum(np.mean(r["rmsd"]) < 2.0 for r in md_results)}/{len(md_results)}'
    ]
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., 5,
               count,
               ha='center', va='bottom', fontsize=24, fontweight='bold',
               color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ===========================================================================
# DEMO DATA GENERATION
# ===========================================================================

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
            'sequence': 'MKVLWAALLVTFLAGCQAKV' * 10,
            'cluster': np.random.choice([0, 1, 2, 3, 4]),
            'plddt': np.clip(np.random.normal(80, 10), 40, 100),
            'active_site_residues': list(range(50, 70))
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
        ke = np.random.normal(50000, 2000, 1000)
        temp = np.random.normal(277, 5, 1000)
        h_bonds = np.random.poisson(8, 1000)
        
        md_results.append({
            'complex_id': f'Complex_{i:03d}',
            'enzyme_id': f'Enzyme_{np.random.randint(0, n_enzymes):03d}',
            'cluster': cluster,
            'time': time,
            'rmsd': rmsd,
            'potential_energy': pe,
            'kinetic_energy': ke,
            'temperature': temp,
            'h_bonds': h_bonds,
            'salt_bridges': np.random.poisson(3, 1000),
            'hydrophobic_contacts': np.random.poisson(12, 1000)
        })
    
    return cram_data, enzyme_data, docking_results, md_results


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎨 CRAM GENERATOR: INDIVIDUAL PLOTS (700 DPI for A0 Poster)")
    print("="*80)
    
    # Generate demo data
    cram_data, enzyme_data, docking_results, md_results = generate_demo_data()
    
    print(f"\n📊 Data loaded:")
    print(f"  - CRAM molecules: {len(cram_data)}")
    print(f"  - Enzymes: {len(enzyme_data)}")
    print(f"  - Docking results: {len(docking_results)}")
    print(f"  - MD simulations: {len(md_results)}\n")
    
    print("Generating individual plots...\n")
    
    # Generate all individual plots
    plot_cram_score_distribution(cram_data)
    plot_carboxyl_distribution(cram_data)
    plot_aliphatic_distribution(cram_data)
    plot_mw_logp_space(cram_data)
    plot_enzyme_clusters_pie(enzyme_data)
    plot_plddt_distribution(enzyme_data)
    plot_binding_affinity_histogram(docking_results)
    plot_top20_binding_affinities(docking_results)
    plot_binding_affinity_by_cluster(docking_results)
    plot_docking_score_vs_affinity(docking_results)
    plot_md_rmsd_trajectories(md_results)
    plot_md_potential_energy(md_results)
    plot_md_hydrogen_bonds(md_results)
    plot_md_temperature(md_results)
    plot_pipeline_success_metrics(cram_data, enzyme_data, docking_results, md_results)
    
    print("\n" + "="*80)
    print("✅ ALL INDIVIDUAL PLOTS GENERATED!")
    print("="*80)
    print("\n15 individual high-resolution figures created (700 DPI)")
    print("Ready for A0 poster presentation")
    print("="*80 + "\n")
