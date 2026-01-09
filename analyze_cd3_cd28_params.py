# analyze_cd3_cd28_params.py
"""
Analyze CD3/CD28-linked parameter distributions across conditions.

After running run_all_conditions with model_4, this script:
1. Loads all 6 condition results
2. Extracts CD3/CD28-linked parameters (k1,k2,k3,k8,k9,k15,k16,k23,k24,k25)
3. Computes statistics (median, IQR, etc.)
4. Checks for compensation (k * CD3 = constant?)
5. Suggests constrained bounds for global fitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from constants import CD3_CD28_COMBINATIONS, OUTPUT_DIR

# CD3/CD28-linked parameters
CD3_PARAMS = ['log_k1', 'log_k2', 'log_k8', 'log_k15', 'log_k23']  # multiply CD3
CD28_PARAMS = ['log_k3', 'log_k9', 'log_k16', 'log_k24', 'log_k25']  # multiply CD28

ALL_STIMULI_PARAMS = CD3_PARAMS + CD28_PARAMS


def load_all_condition_results(results_dir: Path, model_name: str) -> dict:
    """Load results from all 6 conditions."""
    results = {}
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    for cd3, cd28 in CD3_CD28_COMBINATIONS:
        csv_path = results_dir / f"params_{safe_name}_CD3_{cd3}_CD28_{cd28}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            results[(cd3, cd28)] = df
            print(f"Loaded {len(df)} fits for CD3={cd3}, CD28={cd28}")
        else:
            print(f"WARNING: Missing {csv_path}")
    
    return results


def analyze_stimuli_params(results: dict, top_n: int = 100):
    """
    Analyze CD3/CD28-linked parameters across conditions.
    
    Args:
        results: dict of (cd3, cd28) -> DataFrame
        top_n: Use top N fits per condition
    
    Returns:
        DataFrame with statistics
    """
    stats = []
    
    for (cd3, cd28), df in results.items():
        # Get top fits
        top_df = df.nsmallest(top_n, 'loss')
        
        for param in ALL_STIMULI_PARAMS:
            if param not in top_df.columns:
                continue
            
            values = top_df[param].values
            
            # Compute statistics
            stat = {
                'param': param,
                'cd3': cd3,
                'cd28': cd28,
                'median': np.median(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
            }
            
            # For CD3 params, compute k * CD3 (the effective term)
            if param in CD3_PARAMS:
                effective = np.exp(values) * cd3
                stat['effective_median'] = np.median(effective)
                stat['effective_std'] = np.std(effective)
            
            # For CD28 params, compute k * CD28
            if param in CD28_PARAMS:
                effective = np.exp(values) * cd28
                stat['effective_median'] = np.median(effective)
                stat['effective_std'] = np.std(effective)
            
            stats.append(stat)
    
    return pd.DataFrame(stats)


def check_compensation(stats_df: pd.DataFrame):
    """
    Check if parameters compensate for CD3/CD28 changes.
    
    If k * CD3 is constant across conditions, the parameter is compensating.
    """
    print("\n" + "="*70)
    print("COMPENSATION CHECK: Is k * stimulus constant across conditions?")
    print("="*70)
    
    for param in ALL_STIMULI_PARAMS:
        param_stats = stats_df[stats_df['param'] == param]
        if len(param_stats) == 0:
            continue
        
        if 'effective_median' not in param_stats.columns:
            continue
        
        effective_values = param_stats['effective_median'].values
        cv = np.std(effective_values) / np.mean(effective_values) if np.mean(effective_values) > 0 else float('inf')
        
        # Also check the raw parameter variation
        raw_values = param_stats['median'].values
        raw_range = raw_values.max() - raw_values.min()
        
        compensating = cv < 0.5  # Effective term is relatively constant
        
        print(f"\n{param}:")
        print(f"  Raw parameter range: {raw_range:.2f} log-units")
        print(f"  Effective term CV: {cv:.2f}")
        print(f"  Compensating: {'YES ⚠️' if compensating else 'No'}")
        
        if compensating:
            print(f"  -> This parameter varies to keep k*stimulus ≈ {np.mean(effective_values):.3f}")


def suggest_constrained_bounds(stats_df: pd.DataFrame, margin: float = 1.0):
    """
    Suggest constrained bounds for global fitting based on per-condition results.
    
    Args:
        stats_df: DataFrame with parameter statistics
        margin: How many IQRs to extend beyond median
    
    Returns:
        dict of param -> (lower, upper) bounds
    """
    print("\n" + "="*70)
    print("SUGGESTED CONSTRAINED BOUNDS FOR GLOBAL FITTING")
    print("="*70)
    
    bounds = {}
    
    for param in ALL_STIMULI_PARAMS:
        param_stats = stats_df[stats_df['param'] == param]
        if len(param_stats) == 0:
            continue
        
        # Aggregate across conditions
        all_medians = param_stats['median'].values
        all_q25 = param_stats['q25'].values
        all_q75 = param_stats['q75'].values
        
        # Overall statistics
        overall_median = np.median(all_medians)
        overall_iqr = np.median(all_q75 - all_q25)
        
        # Suggested bounds: median ± margin * IQR
        lower = overall_median - margin * max(overall_iqr, 0.5)
        upper = overall_median + margin * max(overall_iqr, 0.5)
        
        bounds[param] = (lower, upper)
        
        print(f"{param}:")
        print(f"  Median across conditions: {overall_median:.2f}")
        print(f"  Typical IQR: {overall_iqr:.2f}")
        print(f"  Suggested bounds: [{lower:.2f}, {upper:.2f}]")
    
    return bounds


def plot_stimuli_params(results: dict, stats_df: pd.DataFrame, output_dir: Path, top_n: int = 100):
    """Plot CD3/CD28 parameter distributions across conditions."""
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for ax, param in zip(axes, ALL_STIMULI_PARAMS):
        data_by_condition = []
        labels = []
        
        for (cd3, cd28), df in results.items():
            top_df = df.nsmallest(top_n, 'loss')
            if param in top_df.columns:
                data_by_condition.append(top_df[param].values)
                labels.append(f'{cd3}/{cd28}')
        
        if data_by_condition:
            bp = ax.boxplot(data_by_condition, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(param.replace('log_', ''))
            ax.set_xlabel('CD3/CD28')
            ax.set_ylabel('Log value')
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'CD3/CD28-Linked Parameters Across Conditions (top {top_n} fits)', fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / "plots" / "cd3_cd28_param_analysis.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()
    
    # Plot effective terms (k * stimulus)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for ax, param in zip(axes, ALL_STIMULI_PARAMS):
        effective_by_condition = []
        labels = []
        
        for (cd3, cd28), df in results.items():
            top_df = df.nsmallest(top_n, 'loss')
            if param in top_df.columns:
                k_values = np.exp(top_df[param].values)
                if param in CD3_PARAMS:
                    effective = k_values * cd3
                else:
                    effective = k_values * cd28
                effective_by_condition.append(effective)
                labels.append(f'{cd3}/{cd28}')
        
        if effective_by_condition:
            bp = ax.boxplot(effective_by_condition, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
            ax.set_title(f'{param.replace("log_", "")} × stimulus')
            ax.set_xlabel('CD3/CD28')
            ax.set_ylabel('Effective value')
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Effective Terms (k × stimulus) - Should be CONSTANT if not compensating', fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / "plots" / "cd3_cd28_effective_terms.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Settings
    model_name = "Full Model - SD (59 params)"
    folder_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    results_dir = OUTPUT_DIR / folder_name
    
    print(f"Loading results from: {results_dir}")
    
    # Load results
    results = load_all_condition_results(results_dir, model_name)
    
    if not results:
        print("No results found. Run run_all_conditions first.")
        exit(1)
    
    # Analyze
    stats_df = analyze_stimuli_params(results, top_n=100)
    
    # Check for compensation
    check_compensation(stats_df)
    
    # Suggest bounds
    bounds = suggest_constrained_bounds(stats_df, margin=1.5)
    
    # Plot
    plot_stimuli_params(results, stats_df, results_dir, top_n=100)
    
    # Save statistics
    stats_path = results_dir / "cd3_cd28_param_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistics saved: {stats_path}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the plots to see if parameters are compensating")
    print("2. If effective terms (k × stimulus) are constant → parameters compensating")
    print("3. Use suggested bounds to create constrained global fit")
    print("4. Or fix parameters to their median values")