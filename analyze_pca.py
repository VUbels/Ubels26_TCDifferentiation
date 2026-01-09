# analyze_pca.py
"""
PCA analysis of fitted parameters across CD3/CD28 conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from pathlib import Path

from constants import CD3_CD28_COMBINATIONS, OUTPUT_DIR


# =============================================================================
# CONSISTENT COLOR SCHEME (IFNg red, IL2 blue, IL4 green, IL21 purple + orange, teal)
# =============================================================================

# Colors assigned by CD3 value (main differentiator)
def get_condition_color(cd3, cd28):
    """Get color based on CD3/CD28 values."""
    # Map based on CD3/CD28 combination
    color_map = {
        (0.01, 1.0): '#e41a1c',   # red
        (0.1, 0.1): '#377eb8',    # blue
        (0.1, 1.0): '#4daf4a',    # green
        (1.0, 1.0): '#984ea3',    # purple
        (5.0, 1.0): '#ff7f00',    # orange
        (10.0, 10.0): '#1b9e77',  # teal
    }
    return color_map.get((cd3, cd28), '#808080')


# Ordered list for consistent plotting
CONDITION_ORDER = [
    (0.01, 1.0),
    (0.1, 0.1),
    (0.1, 1.0),
    (1.0, 1.0),
    (5.0, 1.0),
    (10.0, 10.0),
]

GROUP_COLORS = {
    'Production': '#e41a1c',
    'Degradation': '#377eb8',
    'CD3-linked': '#4daf4a',
    'CD28-linked': '#984ea3',
    'IL2 interactions': '#ff7f00',
    'IFNg interactions': '#1b9e77',
    'IL21 interactions': '#a65628',
    'IL4 interactions': '#f781bf',
    'Hill K': '#999999',
    'Hill h': '#666666',
    'IL4-IL12': '#000000',
}

PARAM_GROUPS = {
    'Production': ['log_k_prod_IL2', 'log_k_prod_IFNg', 'log_k_prod_IL21', 'log_k_prod_IL4'],
    'Degradation': ['log_k_deg_IL2', 'log_k_deg_IFNg', 'log_k_deg_IL21', 'log_k_deg_IL4'],
    'CD3-linked': ['log_k1', 'log_k2', 'log_k8', 'log_k15', 'log_k23'],
    'CD28-linked': ['log_k3', 'log_k9', 'log_k16', 'log_k24', 'log_k25'],
    'IL2 interactions': ['log_k4', 'log_k5', 'log_k6', 'log_k7'],
    'IFNg interactions': ['log_k10', 'log_k11', 'log_k12', 'log_k13', 'log_k14'],
    'IL21 interactions': ['log_k17', 'log_k18', 'log_k19', 'log_k20', 'log_k21'],
    'IL4 interactions': ['log_k22', 'log_k26', 'log_k27', 'log_k28', 'log_k29', 'log_k30', 'log_k31', 'log_k32'],
    'Hill K': ['log_K10', 'log_K12', 'log_K17', 'log_K18', 'log_K21', 'log_K22', 'log_K25', 'log_K27', 'log_K30'],
    'Hill h': ['log_h10_m1', 'log_h12_m1', 'log_h17_m1', 'log_h18_m1', 'log_h21_m1', 'log_h22_m1', 'log_h25_m1', 'log_h27_m1', 'log_h30_m1'],
    'IL4-IL12': ['log_K_IL4inh'],
}


def get_param_group(param: str) -> str:
    for group, params in PARAM_GROUPS.items():
        if param in params:
            return group
    return 'Other'


def load_top_fits(results_dir: Path, model_name: str, top_n: int = 10) -> pd.DataFrame:
    """Load top N fits from each condition."""
    all_data = []
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    print(f"\nLoading from: {results_dir}")
    
    for cd3, cd28 in CD3_CD28_COMBINATIONS:
        csv_path = results_dir / f"params_{safe_name}_CD3_{cd3}_CD28_{cd28}.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            top_df = df.nsmallest(top_n, 'loss').copy()
            # Store numeric values for color lookup
            top_df['cd3_val'] = cd3
            top_df['cd28_val'] = cd28
            all_data.append(top_df)
            print(f"  ✓ CD3={cd3}, CD28={cd28}: {len(top_df)} fits")
        else:
            print(f"  ✗ Missing: {csv_path.name}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal: {len(combined)} fits from {len(all_data)} conditions")
        return combined
    return pd.DataFrame()


def get_param_columns(df: pd.DataFrame) -> list:
    exclude = ['loss', 'cd3', 'cd28', 'model', 'batch_id', 'fit_id', 
               'condition', 'cd3_val', 'cd28_val', 'total_loss', 'avg_loss', 'n_conditions']
    return [c for c in df.columns if c not in exclude]


def draw_confidence_ellipse(ax, x, y, color, n_std=1.5):
    if len(x) < 3:
        return
    try:
        cov = np.cov(x, y)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            return
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * n_std * np.sqrt(max(eigenvalues[0], 1e-10))
        height = 2 * n_std * np.sqrt(max(eigenvalues[1], 1e-10))
        ellipse = Ellipse(
            xy=(np.mean(x), np.mean(y)),
            width=width, height=height, angle=angle,
            facecolor=color, alpha=0.2, edgecolor=color, linewidth=2
        )
        ax.add_patch(ellipse)
    except:
        pass


def plot_pca_scatter(ax, scores, df, var_explained):
    """Panel A: PCA scatter plot with consistent colors."""
    for cd3, cd28 in CONDITION_ORDER:
        mask = (df['cd3_val'] == cd3) & (df['cd28_val'] == cd28)
        if mask.sum() == 0:
            continue
        
        color = get_condition_color(cd3, cd28)
        label = f'{cd3}/{cd28}'
        
        x = scores[mask, 0]
        y = scores[mask, 1]
        
        ax.scatter(x, y, c=color, s=10, alpha=0.7, edgecolor='white',
                   linewidth=1, label=label, zorder=3)
        draw_confidence_ellipse(ax, x, y, color, n_std=1.5)
        ax.scatter(np.mean(x), np.mean(y), c=color, s=250, marker='o',
                   edgecolor='black', linewidth=2, zorder=4)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(f'Dim1 ({var_explained[0]:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Dim2 ({var_explained[1]:.1f}%)', fontsize=12)
    ax.legend(title='anti-CD3/anti-CD28', loc='upper left', framealpha=0.9, fontsize=9)
    ax.set_title('a', fontsize=14, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)


def plot_pca_biplot(ax, loadings, param_cols, var_explained):
    """Panel B: PCA biplot with parameter loadings."""
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='-', linewidth=1)
    ax.add_patch(circle)
    
    max_loading = np.max(np.abs(loadings[:, :2]))
    scale = 0.9 / max_loading
    
    plotted_groups = set()
    
    for i, param in enumerate(param_cols):
        x = loadings[i, 0] * scale
        y = loadings[i, 1] * scale
        
        group = get_param_group(param)
        color = GROUP_COLORS.get(group, 'black')
        
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
        
        label = param.replace('log_', '').replace('_m1', '').replace('k_prod_', 'p_').replace('k_deg_', 'd_')
        offset = 0.08
        ha = 'left' if x >= 0 else 'right'
        va = 'bottom' if y >= 0 else 'top'
        
        text = ax.text(x + (offset if x >= 0 else -offset), 
                       y + (offset if y >= 0 else -offset), 
                       label, fontsize=7, color=color, ha=ha, va=va, fontweight='bold')
        text.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])
        plotted_groups.add(group)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(f'Dim1 ({var_explained[0]:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Dim2 ({var_explained[1]:.1f}%)', fontsize=12)
    ax.set_title('b', fontsize=14, fontweight='bold', loc='left')
    ax.set_aspect('equal')
    
    legend_handles = [plt.Line2D([0], [0], marker='>', color=GROUP_COLORS.get(g, 'black'), 
                                  linestyle='', markersize=8, label=g) for g in plotted_groups]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.9, title='Parameter Type')


def plot_pca_analysis(df: pd.DataFrame, output_dir: Path, top_n: int = 10):
    """Create main PCA plots."""
    param_cols = get_param_columns(df)
    X = df[param_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    var_explained = pca.explained_variance_ratio_ * 100
    
    print(f"\nPCA Variance Explained:")
    for i in range(min(5, len(var_explained))):
        print(f"  PC{i+1}: {var_explained[i]:.1f}%")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_pca_scatter(axes[0], scores, df, var_explained)
    plot_pca_biplot(axes[1], loadings, param_cols, var_explained)
    plt.tight_layout()
    
    save_path = output_dir / "plots" / "pca_analysis.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()
    
    return pca, loadings, param_cols, scaler


def plot_parameter_boxplots_raw(df: pd.DataFrame, output_dir: Path):
    """
    Plot parameter distributions as RAW values with LOG y-axis.
    Uses numeric cd3_val/cd28_val for matching.
    """
    param_cols = get_param_columns(df)
    n_cond = len(CONDITION_ORDER)
    
    # Get colors for each condition
    cond_colors = [get_condition_color(cd3, cd28) for cd3, cd28 in CONDITION_ORDER]
    cond_labels = [f'{cd3}/{cd28}' for cd3, cd28 in CONDITION_ORDER]
    
    groups = {
        'Production & Degradation': [p for p in param_cols if 'prod' in p or 'deg' in p],
        'CD3-CD28 linked': [f'log_k{i}' for i in [1,2,3,8,9,15,16,23,24,25] if f'log_k{i}' in param_cols],
        'Cytokine interactions': [p for p in param_cols if p.startswith('log_k') and 
                                   p not in [f'log_k{i}' for i in [1,2,3,8,9,15,16,23,24,25]] and 
                                   'prod' not in p and 'deg' not in p],
        'Hill K thresholds': [p for p in param_cols if p.startswith('log_K')],
        'Hill h coefficients': [p for p in param_cols if 'log_h' in p],
    }
    
    for group_name, group_params in groups.items():
        if not group_params:
            continue
        
        n_params = len(group_params)
        fig_width = max(16, n_params * 3)
        fig, ax = plt.subplots(figsize=(fig_width, 7))
        
        box_width = 0.7
        param_gap = 4
        
        positions = []
        data_to_plot = []
        colors_to_plot = []
        tick_positions = []
        tick_labels = []
        
        pos = 0
        for param in group_params:
            param_start = pos
            for i, (cd3, cd28) in enumerate(CONDITION_ORDER):
                cond_mask = (df['cd3_val'] == cd3) & (df['cd28_val'] == cd28)
                if cond_mask.sum() == 0:
                    continue
                
                raw_values = np.exp(df.loc[cond_mask, param].values)
                data_to_plot.append(raw_values)
                positions.append(pos)
                colors_to_plot.append(cond_colors[i])
                pos += 1
            
            tick_positions.append((param_start + pos - 1) / 2)
            tick_labels.append(param.replace('log_', '').replace('_m1', ''))
            pos += param_gap
        
        if not data_to_plot:
            plt.close()
            continue
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=box_width, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for element in ['whiskers', 'caps']:
            for line in bp[element]:
                line.set_color('gray')
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(2)
        
        # LOG SCALE Y-AXIS
        ax.set_yscale('log')
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Raw parameter value (log scale)', fontsize=12)
        ax.set_title(f'{group_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Reference line at 1
        ax.axhline(1, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Legend
        existing_colors = []
        existing_labels = []
        for i, (cd3, cd28) in enumerate(CONDITION_ORDER):
            if ((df['cd3_val'] == cd3) & (df['cd28_val'] == cd28)).sum() > 0:
                existing_colors.append(cond_colors[i])
                existing_labels.append(cond_labels[i])
        
        legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.7) 
                          for c in existing_colors]
        ax.legend(legend_patches, existing_labels, title='CD3/CD28', loc='upper right',
                  fontsize=10, ncol=2)
        
        ax.set_xlim(-2, pos)
        
        plt.tight_layout()
        
        safe_group = group_name.replace(' ', '_').replace('/', '_').replace('-', '_')
        save_path = output_dir / "plots" / f"params_raw_{safe_group}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def plot_dendrogram(loadings: np.ndarray, param_cols: list, output_dir: Path):
    """Plot clean dendrogram."""
    loading_subset = loadings[:, :5]
    distances = pdist(loading_subset, metric='euclidean')
    Z = linkage(distances, method='ward')
    
    fig, ax = plt.subplots(figsize=(18, 8))
    labels = [p.replace('log_', '').replace('_m1', '') for p in param_cols]
    
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=9, ax=ax,
               color_threshold=0.7 * max(Z[:, 2]))
    
    ax.set_xlabel('Parameter', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering of Parameters', fontsize=14)
    
    plt.tight_layout()
    save_path = output_dir / "plots" / "parameter_clustering.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def analyze_cd3_cd28_compensation(df: pd.DataFrame, output_dir: Path):
    """Check if CD3/CD28 parameters compensate."""
    cd3_params = ['log_k1', 'log_k2', 'log_k8', 'log_k15', 'log_k23']
    cd28_params = ['log_k3', 'log_k9', 'log_k16', 'log_k24', 'log_k25']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # CD3 parameters
    ax = axes[0]
    for param in cd3_params:
        if param not in df.columns:
            continue
        effective_values = []
        cond_labels = []
        for cd3, cd28 in CONDITION_ORDER:
            mask = (df['cd3_val'] == cd3) & (df['cd28_val'] == cd28)
            if mask.sum() == 0:
                continue
            k_values = np.exp(df.loc[mask, param].values)
            effective = k_values * cd3
            effective_values.append(np.median(effective))
            cond_labels.append(f'{cd3}/{cd28}')
        if effective_values:
            ax.plot(range(len(cond_labels)), effective_values, 'o-', 
                    label=param.replace('log_', ''), markersize=10, linewidth=2)
    
    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_xlabel('Condition (CD3/CD28)', fontsize=12)
    ax.set_ylabel('Effective term (k × CD3)', fontsize=12)
    ax.set_title('CD3-linked parameters', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # CD28 parameters
    ax = axes[1]
    cond_labels = []
    for param in cd28_params:
        if param not in df.columns:
            continue
        effective_values = []
        cond_labels = []
        for cd3, cd28 in CONDITION_ORDER:
            mask = (df['cd3_val'] == cd3) & (df['cd28_val'] == cd28)
            if mask.sum() == 0:
                continue
            k_values = np.exp(df.loc[mask, param].values)
            effective = k_values * cd28
            effective_values.append(np.median(effective))
            cond_labels.append(f'{cd3}/{cd28}')
        if effective_values:
            ax.plot(range(len(cond_labels)), effective_values, 'o-', 
                    label=param.replace('log_', ''), markersize=10, linewidth=2)
    
    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_xlabel('Condition (CD3/CD28)', fontsize=12)
    ax.set_ylabel('Effective term (k × CD28)', fontsize=12)
    ax.set_title('CD28-linked parameters', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / "plots" / "compensation_check.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    model_name = "Full Model - SD (59 params)"
    top_n = 10
    
    folder_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    results_dir = OUTPUT_DIR / folder_name
    
    print(f"Results directory: {results_dir}")
    
    df = load_top_fits(results_dir, model_name, top_n=top_n)
    
    if len(df) == 0:
        print("\nNo results found!")
        exit(1)
    
    # Debug: show what conditions were loaded
    print("\nConditions in data:")
    for cd3 in df['cd3_val'].unique():
        for cd28 in df[df['cd3_val'] == cd3]['cd28_val'].unique():
            count = ((df['cd3_val'] == cd3) & (df['cd28_val'] == cd28)).sum()
            print(f"  CD3={cd3}, CD28={cd28}: {count} fits")
    
    # PCA
    pca, loadings, param_cols, scaler = plot_pca_analysis(df, results_dir, top_n=top_n)
    
    # Dendrogram
    plot_dendrogram(loadings, param_cols, results_dir)
    
    # Raw boxplots (more data)
    df_full = load_top_fits(results_dir, model_name, top_n=10)
    plot_parameter_boxplots_raw(df_full, results_dir)
    
    # Compensation check
    analyze_cd3_cd28_compensation(df_full, results_dir)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)