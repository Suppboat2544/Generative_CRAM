import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

np.random.seed(42)

# =============== 1. REALISTIC ENVIRONMENT-SPECIFIC DOM WITH STRICT VALIDATION =================

def make_cho_dom(n_target, env, oc_center, hc_center):
    """Generate formulas clustered around specific O/C and H/C centers with strict validation."""
    rows, trials = [], 0
    while len(rows) < n_target and trials < n_target * 60:
        trials += 1

        C = np.random.randint(8, 45)

        if env == "deep":
            oc_target = np.random.normal(oc_center, 0.08)
            hc_target = np.random.normal(hc_center, 0.15)
        elif env == "surface":
            oc_target = np.random.normal(oc_center, 0.10)
            hc_target = np.random.normal(hc_center, 0.18)
        else: # coastal
            oc_target = np.random.normal(oc_center, 0.12)
            hc_target = np.random.normal(hc_center, 0.20)

        O = int(oc_target * C)
        H = int(hc_target * C)
        
        # Strict bounds for O and H
        O = max(1, min(O, int(0.8*C)))
        H = max(int(0.5*C), min(H, int(2.2*C)))

        O_C = O / C
        H_C = H / C

        # Seven Golden Rules validation
        # Rule 1: Element ratios must be in valid range
        if not (0 <= O_C <= 1.2 and 0.2 <= H_C <= 3.1):
            continue
        
        # Rule 2: DOM-specific window (marine organic matter)
        if not (0.05 <= O_C <= 0.9 and 0.4 <= H_C <= 2.3):
            continue

        # Rule 3: DBE (Double Bond Equivalent) must be non-negative
        # DBE = 1 + C - H/2 + N/2 - X/2 (for CHO, N=X=0)
        dbe = 1 + C - H/2
        if dbe < 0:
            continue
        
        # Rule 4: DBE must be reasonable (not too high = impossible)
        # Maximum DBE roughly = C + 1 (all carbon in rings/double bonds)
        if dbe > (C + 1):
            continue
        
        # Rule 5: Hydrogen deficiency index (H must be compatible with C and O)
        # For organic molecules: H <= 2*C + 2 (saturated alkane upper limit)
        if H > (2*C + 2):
            continue
        
        # Rule 6: Senior's theorem - check for valence validity
        # Sum of valences must be even: 4*C + 1*H + 2*O must be even
        if (4*C + H + 2*O) % 2 != 0:
            continue

        rows.append({
            "formula": f"C{C}H{H}O{O}_{env[0]}{len(rows)}",
            "env": env,
            "C": C, "H": H, "O": O,
            "O_C": O_C, "H_C": H_C,
            "DBE": dbe
        })

    return pd.DataFrame(rows)

# Generate formulas with distinct distributions
print("\nGenerating validated CHO formulas...")
df_deep = make_cho_dom(2500, "deep", oc_center=0.35, hc_center=1.05)
df_surface = make_cho_dom(2500, "surface", oc_center=0.45, hc_center=1.30)
df_coastal = make_cho_dom(2500, "coastal", oc_center=0.55, hc_center=1.55)

print(f" Deep sea: {len(df_deep)} valid formulas generated")
print(f" Surface sea: {len(df_surface)} valid formulas generated")
print(f" Coastal sea: {len(df_coastal)} valid formulas generated")

def add_shared(df_list, frac=0.15):
    base = df_list[0]
    n_shared = int(frac * len(base))
    shared = base.sample(n_shared, random_state=10).copy()
    shared_ids = [f"Fshared_{i}" for i in range(n_shared)]
    shared["formula"] = shared_ids
    for i, df in enumerate(df_list):
        rep = shared.copy()
        idx = df.sample(min(n_shared, len(df)), random_state=20+i).index
        rep.index = idx
        df.loc[idx] = rep

add_shared([df_deep, df_surface, df_coastal], frac=0.15)

df = pd.concat([df_deep, df_surface, df_coastal], ignore_index=True)

# =============== 2. GROUP LABELS ===============

g_counts = df.groupby("formula")["env"].nunique()
common_ids = set(g_counts[g_counts == 3].index)

deep_only = set(df_deep.loc[~df_deep["formula"].isin(common_ids), "formula"])
surface_only = set(df_surface.loc[~df_surface["formula"].isin(common_ids), "formula"])
coastal_only = set(df_coastal.loc[~df_coastal["formula"].isin(common_ids), "formula"])

def group_label(f):
    if f in common_ids:
        return "common"
    if f in deep_only:
        return "deep"
    if f in surface_only:
        return "surface"
    if f in coastal_only:
        return "coastal"
    return "other"

df["group"] = df["formula"].map(group_label)

# Print formula distribution
print("\n" + "="*75)
print("Formula Distribution Across Environments:")
print("="*75)
print(f"Coastal sea only: {len(df[df['group']=='coastal'])} formulas")
print(f"Surface sea only: {len(df[df['group']=='surface'])} formulas")
print(f"Deep sea only: {len(df[df['group']=='deep'])} formulas")
print(f"Common to all: {len(df[df['group']=='common'])} formulas (found in all 3 environments)")
print("="*75)

# =============== 3. CRAM REGION ====================

cram_poly = np.array([
    [0.20, 1.55],
    [0.42, 1.75],
    [0.68, 1.45],
    [0.58, 0.98],
    [0.25, 0.93],
])
poly_path = Path(cram_poly)
df["in_cram"] = poly_path.contains_points(df[["O_C", "H_C"]].values)

# =============== 4. ISOMER MODELS =========

df["mass"] = 12*df["C"] + df["H"] + 16*df["O"]
base_iso = (df["mass"] / 350.0)**1.15 * (df["C"] / 20.0)**0.3

df["isomers_pubchem"] = base_iso.clip(0.5, 5.0)
df["isomers_pubchem_cram"] = df["isomers_pubchem"].copy()

cram_boost = {"deep": 5.5, "surface": 2.8, "coastal": 2.2}
for env, boost in cram_boost.items():
    mask = (df["env"] == env) & df["in_cram"]
    df.loc[mask, "isomers_pubchem_cram"] *= boost

df["size_pubchem"] = df["isomers_pubchem"] * 10
df["size_pubchem_cram"] = df["isomers_pubchem_cram"] * 10

# =============== 5. VALIDATE FINAL DATASET ===============

print("\n" + "="*75)
print("Final Dataset Validation:")
print("="*75)
print(f"Total formulas: {len(df)}")
print(f"O/C range: {df['O_C'].min():.3f} - {df['O_C'].max():.3f}")
print(f"H/C range: {df['H_C'].min():.3f} - {df['H_C'].max():.3f}")
print(f"DBE range: {df['DBE'].min():.1f} - {df['DBE'].max():.1f}")
print(f"Mass range: {df['mass'].min():.1f} - {df['mass'].max():.1f} Da")
print(f"All DBE >= 0: {(df['DBE'] >= 0).all()}")
print(f"CRAM formulas: {df['in_cram'].sum()}")
print("="*75)

# =============== 6. SAVE DATASET ===============

print("\nSaving dataset to CSV...")
df.to_csv("dom_cram_dataset.csv", index=False)
print(" Saved: dom_cram_dataset.csv")

# Print CRAM statistics
print("\n" + "="*75)
print("ISOMER IDENTIFICATION: PubChem vs PubChem+CRAM Database")
print("="*75)

comparison_stats = []
for env in ["coastal", "surface", "deep"]:
    sub_cram = df[(df["env"] == env) & df["in_cram"]]
    
    n_cram = sub_cram["formula"].nunique()
    pub_only = sub_cram["isomers_pubchem"].mean() if len(sub_cram) > 0 else 0
    pub_cram = sub_cram["isomers_pubchem_cram"].mean() if len(sub_cram) > 0 else 0
    increase = ((pub_cram - pub_only) / pub_only * 100) if pub_only > 0 else 0
    
    comparison_stats.append({
        "Environment": env.capitalize(),
        "CRAM formulas": n_cram,
        "PubChem": f"{pub_only:.2f}",
        "PubChem+CRAM": f"{pub_cram:.2f}",
        "Increase": f"+{increase:.0f}%"
    })

comp_df = pd.DataFrame(comparison_stats)
print("\n" + comp_df.to_string(index=False))
print("="*75 + "\n")

# =============== 7. COLORS =======================================

palette = {
    "deep": "#f4a442",
    "surface": "#5ba3d0",
    "coastal": "#3d9970",
    "common": "#3a3a3a",
}

# =============== 8. PLOTTING FUNCTION ================

sns.set_theme(style="white", context="talk")

def plot_vkd_with_marginals(size_col, title, filename, panel_label):
    """Create VKD with marginals and save."""
    
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05,
                          height_ratios=[1, 5, 0.1],
                          width_ratios=[0.1, 5, 1])
    
    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    
    # Add panel label (a, b, c)
    fig.text(0.02, 0.98, panel_label, fontsize=18, weight='bold', 
             va='top', ha='left')
    
    # Plot common (background)
    common = df[df["group"] == "common"]
    if len(common) > 0:
        common_plot = common.sample(min(300, len(common)), random_state=5)
        ax_main.scatter(
            common_plot["O_C"], common_plot["H_C"],
            s=common_plot[size_col] * 0.25,
            c=palette["common"],
            alpha=0.15,
            edgecolors="none",
            zorder=1
        )
    
    # Plot unique formulas
    plot_order = [("coastal", 3), ("surface", 4), ("deep", 5)]
    
    for env, z in plot_order:
        sub = df[df["group"] == env]
        if len(sub) > 0:
            cram_sub = sub[sub["in_cram"]]
            non_cram_sub = sub[~sub["in_cram"]]
            
            if len(non_cram_sub) > 600:
                non_cram_sub = non_cram_sub.sample(600, random_state=z)
            
            sub_plot = pd.concat([cram_sub, non_cram_sub])
            
            ax_main.scatter(
                sub_plot["O_C"], sub_plot["H_C"],
                s=sub_plot[size_col],
                c=palette[env],
                alpha=0.75,
                edgecolors="none",
                zorder=z
            )
    
    # CRAM polygon
    cp = np.vstack([cram_poly, cram_poly[0]])
    ax_main.plot(cp[:,0], cp[:,1], "--", color="#d62728", linewidth=2.5, zorder=10)
    
    # CRAM annotation for PubChem+CRAM plot
    if "cram" in size_col:
        ax_main.text(0.44, 1.37, "+ CRAM\nDatabase", ha='center', va='center',
                     fontsize=10, color='#d62728', weight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6', 
                              edgecolor='#d62728', linewidth=2, alpha=0.92))
    
    ax_main.set_xlabel("O/C", fontsize=12, weight='bold')
    ax_main.set_ylabel("H/C", fontsize=12, weight='bold')
    ax_main.set_xlim(0.0, 1.0)
    ax_main.set_ylim(0.3, 2.3)
    ax_main.set_title(title, fontsize=14, weight='bold', pad=15)
    ax_main.tick_params(labelsize=10)
    
    # Marginals (NO LABELS)
    for env, col in [("coastal", palette["coastal"]),
                     ("surface", palette["surface"]),
                     ("deep", palette["deep"])]:
        sub = df[df["env"] == env]
        
        sns.kdeplot(data=sub, x="O_C", ax=ax_top, fill=True,
                    color=col, alpha=0.6, linewidth=2.5, bw_adjust=0.8)
        
        sns.kdeplot(data=sub, y="H_C", ax=ax_right, fill=True,
                    color=col, alpha=0.6, linewidth=2.5, bw_adjust=0.8)
    
    # Remove all labels and ticks from marginals
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_top.set_yticks([])
    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(labelbottom=False, labelleft=False)
    ax_top.spines['left'].set_visible(False)
    
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")
    ax_right.set_xticks([])
    ax_right.set_xlim(left=0)
    ax_right.tick_params(labelleft=False, labelright=False)
    ax_right.spines['bottom'].set_visible(False)
    
    # Clearer legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["coastal"],
               markersize=7, label='Coastal only', alpha=0.75),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["surface"],
               markersize=7, label='Surface only', alpha=0.75),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["deep"],
               markersize=7, label='Deep only', alpha=0.75),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["common"],
               markersize=6, label='All 3 environments', alpha=0.4),
    ]
    
    leg = ax_main.legend(handles=legend_elements, loc="upper left", fontsize=7,
                         frameon=True, framealpha=0.95, edgecolor='lightgrey',
                         title="Formula occurrence", title_fontsize=7.5)
    leg.get_title().set_fontweight('bold')
    
    for ax in [ax_main, ax_top, ax_right]:
        sns.despine(fig=fig, ax=ax, trim=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=700, bbox_inches='tight')
    print(f" Saved: {filename}")
    plt.close()

# =============== 9. SAVE INDIVIDUAL PLOTS ================

print("\nGenerating and saving plots at 700 DPI...")

# Plot a): PubChem only
plot_vkd_with_marginals("size_pubchem", "PubChem Database", 
                         "vkd_pubchem_only.png", "a)")

# Plot b): PubChem + CRAM
plot_vkd_with_marginals("size_pubchem_cram", "PubChem + CRAM Database",
                         "vkd_pubchem_cram.png", "b)")

# Plot c): Isomer comparison
fig, ax = plt.subplots(figsize=(10, 5))

# Add panel label
fig.text(0.02, 0.98, "c)", fontsize=18, weight='bold', 
         va='top', ha='left', transform=fig.transFigure)

env_order = [("coastal", palette["coastal"]),
             ("surface", palette["surface"]),
             ("deep", palette["deep"])]

# PubChem only (dashed)
for env, col in env_order:
    sub_cram = df[(df["env"] == env) & df["in_cram"]]
    if len(sub_cram) > 0:
        sns.kdeplot(data=sub_cram, x="isomers_pubchem", ax=ax,
                    fill=False, color=col, alpha=0.8, linewidth=2.5,
                    linestyle='--')

# PubChem + CRAM (solid, filled)
for env, col in env_order:
    sub_cram = df[(df["env"] == env) & df["in_cram"]]
    if len(sub_cram) > 0:
        sns.kdeplot(data=sub_cram, x="isomers_pubchem_cram", ax=ax,
                    fill=True, color=col, alpha=0.5, linewidth=3.5)

ax.set_xlabel("Isomers per Formula in CRAM Region", fontsize=12, weight='bold')
ax.set_ylabel("Density", fontsize=11, weight='bold')
ax.set_xlim(0, 32)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=10)

# Clearer legend
legend_elements = [
    Line2D([0], [0], color=palette["coastal"], linewidth=2.5, linestyle='--',
           label='Coastal (PubChem)', alpha=0.8),
    Patch(facecolor=palette["coastal"], alpha=0.5, label='Coastal (PubChem+CRAM)'),
    Line2D([0], [0], color=palette["surface"], linewidth=2.5, linestyle='--',
           label='Surface (PubChem)', alpha=0.8),
    Patch(facecolor=palette["surface"], alpha=0.5, label='Surface (PubChem+CRAM)'),
    Line2D([0], [0], color=palette["deep"], linewidth=2.5, linestyle='--',
           label='Deep (PubChem)', alpha=0.8),
    Patch(facecolor=palette["deep"], alpha=0.5, label='Deep (PubChem+CRAM)'),
]

leg = ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
                frameon=True, framealpha=0.95, ncol=3,
                title="CRAM Isomer Distribution", title_fontsize=7.5)
leg.get_title().set_fontweight('bold')

ax.set_title("Impact of CRAM Database on Isomer Identification",
             fontsize=13, weight='bold', pad=12)

sns.despine(fig=fig, ax=ax, trim=True)
plt.tight_layout()
plt.savefig("isomer_comparison.png", dpi=700, bbox_inches='tight')
print(" Saved: isomer_comparison.png")
plt.close()

print("\n" + "="*75)
print("COMPLETE! Generated files:")
print(" 1. dom_cram_dataset.csv - Full validated CHO dataset")
print(" 2. vkd_pubchem_only.png - a) PubChem database VKD (700 DPI)")
print(" 3. vkd_pubchem_cram.png - b) PubChem+CRAM database VKD (700 DPI)")
print(" 4. isomer_comparison.png - c) Isomer distribution (700 DPI)")
print("="*75 + "\n")
