# visualization.py
"""
Visualization functions for PSO fitting results.

Two main functions:
- generate_single_condition_plots: For single condition results
- generate_all_conditions_plots: For full 6-condition results
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchdiffeq import odeint

from constants import CD3_CD28_COMBINATIONS, CYTOKINE_NAMES, CYTOKINE_COLORS


# =============================================================================
# SINGLE CONDITION PLOTS (for model comparison)
# =============================================================================

def generate_single_condition_plots(
    model_class,
    data: dict,
    device: torch.device,
    results_dir: Path,
    cd3: float = 1.0,
    cd28: float = 1.0,
    top_n: int = 10,
):
    """
    Generate plots for a single condition fit.
    
    Args:
        model_class: The model class used for fitting
        data: Dictionary of ConditionData
        device: torch device
        results_dir: Path to results folder (e.g., results/Full_Model_59_params/)
        cd3, cd28: The condition
        top_n: Number of top fits to use for parameter distributions
    """
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Get model info
    temp_model = model_class(n_batch=1, device=device)
    model_name = temp_model.model_name
    del temp_model
    
    param_names = model_class.get_param_names()
    
    # Find the results CSV
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    csv_path = results_dir / f"params_{safe_name}_CD3_{cd3}_CD28_{cd28}.csv"
    
    if not csv_path.exists():
        print(f"Results not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path).sort_values('loss')
    cond = data[(cd3, cd28)]
    
    print(f"\n{'='*60}")
    print(f"Generating plots for {model_name}")
    print(f"Condition: CD3={cd3}, CD28={cd28}")
    print(f"Loaded {len(df)} fits")
    print('='*60)
    
    # Plot 1: Best fit vs data
    _plot_best_fit(model_class, df, cond, device, param_names, model_name,
                   plots_dir / "best_fit.png")
    
    # Plot 2: Top N fits comparison
    _plot_top_n_fits(model_class, df, cond, device, param_names, model_name,
                     n_values=[1, 10, 100, 1000],
                     save_path=plots_dir / "top_n_fits.png")
    
    # Plot 3: Loss distribution
    _plot_loss_distribution(df, model_name, plots_dir / "loss_distribution.png")
    
    # Plot 4: Parameter distributions
    _plot_parameter_distributions(df, param_names, model_name, top_n,
                                   plots_dir / "parameter_distributions.png")
    
    print(f"\nPlots saved to: {plots_dir}")


# =============================================================================
# ALL CONDITIONS PLOTS (for full model)
# =============================================================================

def generate_all_conditions_plots(
    model_class,
    data: dict,
    device: torch.device,
    results_dir: Path,
    top_n: int = 10,
):
    """
    Generate plots for all 6 conditions.
    
    Args:
        model_class: The model class used for fitting
        data: Dictionary of ConditionData
        device: torch device
        results_dir: Path to results folder
        top_n: Number of top fits for parameter distributions
    """
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Get model info
    temp_model = model_class(n_batch=1, device=device)
    model_name = temp_model.model_name
    del temp_model
    
    param_names = model_class.get_param_names()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    print(f"\n{'='*60}")
    print(f"Generating plots for {model_name}")
    print(f"All 6 conditions")
    print('='*60)
    
    # Load all results
    all_results = {}
    for cd3, cd28 in CD3_CD28_COMBINATIONS:
        csv_path = results_dir / f"params_{safe_name}_CD3_{cd3}_CD28_{cd28}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path).sort_values('loss')
            all_results[(cd3, cd28)] = df
            print(f"Loaded {len(df)} fits for CD3={cd3}, CD28={cd28}")
        else:
            print(f"NOT FOUND: {csv_path}")
    
    if not all_results:
        print("No results found!")
        return
    
    # Plot 1: All conditions best fits (6x4 grid)
    _plot_all_conditions_grid(model_class, all_results, data, device, param_names,
                               model_name, plots_dir / "all_conditions_best_fits.png")
    
    # Plot 2: Loss comparison across conditions
    _plot_loss_comparison(all_results, model_name, plots_dir / "loss_comparison.png")
    
    # Plot 3: Cross-condition parameter comparison
    _plot_cross_condition_parameters(all_results, param_names, model_name, top_n,
                                      plots_dir)
    
    # Per-condition plots
    for cd3, cd28 in CD3_CD28_COMBINATIONS:
        if (cd3, cd28) not in all_results:
            continue
        
        df = all_results[(cd3, cd28)]
        cond = data[(cd3, cd28)]
        
        print(f"\n--- CD3={cd3}, CD28={cd28} ---")
        
        # Top N fits
        _plot_top_n_fits(model_class, df, cond, device, param_names, model_name,
                         n_values=[1, 10, 100, 1000],
                         save_path=plots_dir / f"top_n_fits_CD3_{cd3}_CD28_{cd28}.png")
        
        # Loss distribution
        _plot_loss_distribution(df, f"{model_name} CD3={cd3} CD28={cd28}",
                                plots_dir / f"loss_dist_CD3_{cd3}_CD28_{cd28}.png")
    
    print(f"\nAll plots saved to: {plots_dir}")


# =============================================================================
# INTERNAL PLOTTING FUNCTIONS
# =============================================================================

def _plot_best_fit(model_class, df, cond, device, param_names, model_name, save_path):
    """Plot the single best fit vs experimental data."""
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Get best parameters
    best_row = df.iloc[0]
    best_params = torch.tensor(best_row[param_names].values, dtype=torch.float32, device=device)
    best_loss = best_row['loss']
    
    # Simulate
    t_fine = torch.linspace(3, 11, 100, device=device)
    
    model = model_class(n_batch=1, device=device)
    model.set_condition(cd3=cond.cd3, cd28=cond.cd28)
    model_class.params_to_model(best_params.unsqueeze(0), model)
    
    y0 = cond.y0.unsqueeze(0).to(device)
    
    with torch.no_grad():
        solution = odeint(model, y0, t_fine, method='dopri5', options={'max_num_steps': 5000})
    
    t_fine_np = t_fine.cpu().numpy()
    pred = solution[:, 0, :].cpu().numpy()
    t_data = cond.t.cpu().numpy()
    y_mean = cond.y_mean.cpu().numpy()
    y_std = cond.y_std.cpu().numpy()
    
    for i, (ax, name, color) in enumerate(zip(axes, CYTOKINE_NAMES, CYTOKINE_COLORS)):
        ax.errorbar(t_data, y_mean[:, i], yerr=y_std[:, i],
                   fmt='o', markersize=8, capsize=5, color=color, label='Data')
        ax.plot(t_fine_np, pred[:, i], '-', linewidth=2, color=color, label='Model')
        ax.set_xlabel('Day')
        ax.set_ylabel('% CD4+')
        ax.set_title(name)
        ax.set_xlim(2.5, 11.5)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(f'{model_name}\nCD3={cond.cd3}, CD28={cond.cd28}, Loss={best_loss:.4f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def _plot_top_n_fits(model_class, df, cond, device, param_names, model_name, n_values, save_path):
    """Plot comparison of top N fits."""
    
    fig, axes = plt.subplots(len(n_values), 4, figsize=(16, 4*len(n_values)))
    
    t_fine = torch.linspace(3, 11, 100, device=device)
    t_data = cond.t.cpu().numpy()
    y_mean = cond.y_mean.cpu().numpy()
    y_std = cond.y_std.cpu().numpy()
    
    for row_idx, n in enumerate(n_values):
        n_actual = min(n, len(df))
        top_df = df.head(n_actual)
        
        # Collect predictions
        all_preds = []
        for _, row in top_df.iterrows():
            params = torch.tensor(row[param_names].values, dtype=torch.float32, device=device)
            
            model = model_class(n_batch=1, device=device)
            model.set_condition(cd3=cond.cd3, cd28=cond.cd28)
            model_class.params_to_model(params.unsqueeze(0), model)
            
            y0 = cond.y0.unsqueeze(0).to(device)
            
            try:
                with torch.no_grad():
                    solution = odeint(model, y0, t_fine, method='dopri5', options={'max_num_steps': 5000})
                all_preds.append(solution[:, 0, :].cpu().numpy())
            except:
                pass
        
        if not all_preds:
            continue
        
        all_preds = np.array(all_preds)
        pred_mean = np.mean(all_preds, axis=0)
        pred_min = np.min(all_preds, axis=0)
        pred_max = np.max(all_preds, axis=0)
        
        t_fine_np = t_fine.cpu().numpy()
        ax_row = axes[row_idx] if len(n_values) > 1 else axes
        
        for col_idx, (name, color) in enumerate(zip(CYTOKINE_NAMES, CYTOKINE_COLORS)):
            ax = ax_row[col_idx]
            
            ax.errorbar(t_data, y_mean[:, col_idx], yerr=y_std[:, col_idx],
                       fmt='o', markersize=8, capsize=5, color='black', zorder=5)
            ax.fill_between(t_fine_np, pred_min[:, col_idx], pred_max[:, col_idx],
                           alpha=0.3, color=color, label='Range')
            ax.plot(t_fine_np, pred_mean[:, col_idx], '-', linewidth=2, color=color, label='Mean')
            
            ax.set_xlim(2.5, 11.5)
            ax.set_ylim(0, None)
            ax.grid(True, alpha=0.3)
            
            if row_idx == 0:
                ax.set_title(name)
            if col_idx == 0:
                ax.set_ylabel(f'Top {n_actual}')
            if row_idx == len(n_values) - 1:
                ax.set_xlabel('Day')
    
    plt.suptitle(f'{model_name}: CD3={cond.cd3}, CD28={cond.cd28}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def _plot_loss_distribution(df, title, save_path):
    """Plot histogram of loss values."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    losses = df['loss'].values
    
    # Histogram
    axes[0].hist(losses, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(losses.min(), color='red', linestyle='--', label=f'Best: {losses.min():.4f}')
    axes[0].axvline(np.median(losses), color='orange', linestyle='--', label=f'Median: {np.median(losses):.4f}')
    axes[0].set_xlabel('Loss')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Loss Distribution')
    axes[0].legend()
    
    # Log scale
    axes[1].hist(losses, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Loss')
    axes[1].set_ylabel('Count (log)')
    axes[1].set_title('Loss Distribution (log scale)')
    axes[1].set_yscale('log')
    
    # Cumulative
    sorted_losses = np.sort(losses)
    cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    axes[2].plot(sorted_losses, cumulative, 'b-', linewidth=2)
    axes[2].axhline(0.01, color='red', linestyle=':', alpha=0.7, label='Top 1%')
    axes[2].axhline(0.10, color='orange', linestyle=':', alpha=0.7, label='Top 10%')
    axes[2].set_xlabel('Loss')
    axes[2].set_ylabel('Cumulative Fraction')
    axes[2].set_title('Cumulative Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} (n={len(df)})', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def _plot_parameter_distributions(df, param_names, model_name, top_n, save_path):
    """Plot boxplots of parameter values."""
    
    top_df = df.head(top_n)
    
    # Group parameters by type
    param_groups = {
        'Production': [c for c in param_names if 'k_prod' in c],
        'Degradation': [c for c in param_names if 'k_deg' in c],
        'Interaction (k)': [c for c in param_names if c.startswith('log_k') and 'prod' not in c and 'deg' not in c],
        'Hill K': [c for c in param_names if c.startswith('log_K') and 'IL4inh' not in c],
        'Hill h': [c for c in param_names if 'log_h' in c],
    }
    
    # Filter empty groups
    param_groups = {k: v for k, v in param_groups.items() if v}
    
    n_groups = len(param_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 3*n_groups))
    if n_groups == 1:
        axes = [axes]
    
    for ax, (group_name, cols) in zip(axes, param_groups.items()):
        data_to_plot = [top_df[c].values for c in cols]
        labels = [c.replace('log_', '') for c in cols]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Log value')
        ax.set_title(group_name)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{model_name}: Parameter Distributions (top {top_n})', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def _plot_all_conditions_grid(model_class, all_results, data, device, param_names, model_name, save_path):
    """Plot best fit for all 6 conditions in a grid."""
    
    fig, axes = plt.subplots(6, 4, figsize=(16, 20))
    
    for row_idx, (cd3, cd28) in enumerate(CD3_CD28_COMBINATIONS):
        if (cd3, cd28) not in all_results:
            continue
        
        df = all_results[(cd3, cd28)]
        cond = data[(cd3, cd28)]
        
        best_row = df.iloc[0]
        best_params = torch.tensor(best_row[param_names].values, dtype=torch.float32, device=device)
        best_loss = best_row['loss']
        
        t_fine = torch.linspace(3, 11, 100, device=device)
        
        model = model_class(n_batch=1, device=device)
        model.set_condition(cd3=cd3, cd28=cd28)
        model_class.params_to_model(best_params.unsqueeze(0), model)
        
        y0 = cond.y0.unsqueeze(0).to(device)
        
        try:
            with torch.no_grad():
                solution = odeint(model, y0, t_fine, method='dopri5', options={'max_num_steps': 5000})
            
            t_fine_np = t_fine.cpu().numpy()
            pred = solution[:, 0, :].cpu().numpy()
            t_data = cond.t.cpu().numpy()
            y_mean = cond.y_mean.cpu().numpy()
            y_std = cond.y_std.cpu().numpy()
            
            for col_idx, (name, color) in enumerate(zip(CYTOKINE_NAMES, CYTOKINE_COLORS)):
                ax = axes[row_idx, col_idx]
                ax.errorbar(t_data, y_mean[:, col_idx], yerr=y_std[:, col_idx],
                           fmt='o', markersize=6, capsize=3, color=color)
                ax.plot(t_fine_np, pred[:, col_idx], '-', linewidth=2, color=color)
                ax.set_xlim(2.5, 11.5)
                ax.set_ylim(0, None)
                ax.grid(True, alpha=0.3)
                
                if row_idx == 0:
                    ax.set_title(name)
                if col_idx == 0:
                    ax.set_ylabel(f'CD3={cd3}\nCD28={cd28}\nL={best_loss:.3f}', fontsize=9)
                if row_idx == 5:
                    ax.set_xlabel('Day')
        except:
            print(f"Failed: CD3={cd3}, CD28={cd28}")
    
    plt.suptitle(f'{model_name}: Best Fits All Conditions', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def _plot_loss_comparison(all_results, model_name, save_path):
    """Compare losses across conditions."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    conditions = []
    best_losses = []
    median_losses = []
    all_losses_list = []
    
    for cd3, cd28 in CD3_CD28_COMBINATIONS:
        if (cd3, cd28) not in all_results:
            continue
        df = all_results[(cd3, cd28)]
        conditions.append(f'{cd3}/{cd28}')
        best_losses.append(df['loss'].min())
        median_losses.append(df['loss'].median())
        all_losses_list.append(df['loss'].values)
    
    x = np.arange(len(conditions))
    
    axes[0].bar(x, best_losses, color='steelblue', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions)
    axes[0].set_xlabel('CD3/CD28')
    axes[0].set_ylabel('Best Loss')
    axes[0].set_title('Best Loss per Condition')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, median_losses, color='coral', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions)
    axes[1].set_xlabel('CD3/CD28')
    axes[1].set_ylabel('Median Loss')
    axes[1].set_title('Median Loss per Condition')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    bp = axes[2].boxplot(all_losses_list, labels=conditions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    axes[2].set_xlabel('CD3/CD28')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Loss Distribution')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{model_name}: Loss Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def _plot_cross_condition_parameters(all_results, param_names, model_name, top_n, plots_dir):
    """Compare parameters across conditions."""
    
    top_dfs = {k: v.head(top_n) for k, v in all_results.items()}
    
    param_groups = {
        'production_rates': [c for c in param_names if 'k_prod' in c],
        'degradation_rates': [c for c in param_names if 'k_deg' in c],
        'interaction_k': [c for c in param_names if c.startswith('log_k') and 'prod' not in c and 'deg' not in c],
        'hill_K': [c for c in param_names if c.startswith('log_K') and 'IL4inh' not in c],
        'hill_h': [c for c in param_names if 'log_h' in c],
    }
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_dfs)))
    
    for group_name, group_params in param_groups.items():
        if not group_params:
            continue
        
        n_params = len(group_params)
        fig, axes = plt.subplots(1, n_params, figsize=(min(4*n_params, 20), 5))
        if n_params == 1:
            axes = [axes]
        
        for ax, param in zip(axes, group_params):
            data_by_cond = [top_dfs[k][param].values for k in CD3_CD28_COMBINATIONS if k in top_dfs]
            labels = [f'{cd3}/{cd28}' for cd3, cd28 in CD3_CD28_COMBINATIONS if (cd3, cd28) in top_dfs]
            
            bp = ax.boxplot(data_by_cond, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('CD3/CD28')
            ax.set_ylabel('Log value')
            ax.set_title(param.replace('log_', ''))
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'{model_name}: {group_name.replace("_", " ").title()} (top {top_n})', fontsize=12)
        plt.tight_layout()
        save_path = plots_dir / f"cross_condition_{group_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()