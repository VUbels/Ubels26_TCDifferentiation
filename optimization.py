# optimization.py
"""
PSO optimization for T cell differentiation models.

Contains:
- MultiSwarmPSO: The particle swarm optimizer
- compute_loss: Loss function
- evaluate_particles: Batch evaluation with rk4
- run_single_condition: Fit one model on CD3=1, CD28=1
- run_all_conditions: Fit one model on all 6 conditions
"""

import gc
import time
import torch
import pandas as pd
from pathlib import Path
from torchdiffeq import odeint

from constants import CD3_CD28_COMBINATIONS


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def compute_loss(solution, y_mean, y_std):
    """
    Compute SD-weighted mean squared error.
    
    Args:
        solution: [n_timepoints, n_batch, 4] model predictions
        y_mean: [n_timepoints, 4] experimental means
        y_std: [n_timepoints, 4] experimental standard deviations
    
    Returns:
        losses: [n_batch] loss per particle
    """
    # Add batch dimension to data
    y_mean = y_mean.unsqueeze(1)  # [n_timepoints, 1, 4]
    y_std = y_std.unsqueeze(1)    # [n_timepoints, 1, 4]
    
    # Weighted squared error
    squared_error = ((solution - y_mean) / y_std) ** 2
    
    # Mean over timepoints and cytokines, keep batch dimension
    losses = squared_error.mean(dim=(0, 2))  # [n_batch]
    
    return losses


# =============================================================================
# PARTICLE SWARM OPTIMIZER
# =============================================================================

class MultiSwarmPSO:
    """
    Multiple independent PSO swarms running in parallel on GPU.
    
    Each swarm searches independently. This gives diversity in solutions.
    """
    
    def __init__(
        self,
        n_swarms: int,
        n_particles_per_swarm: int,
        n_params: int,
        bounds: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
        w: float = 0.7,      # Inertia weight
        c1: float = 1.5,     # Cognitive (personal best) weight
        c2: float = 1.5,     # Social (global best) weight
    ):
        self.n_swarms = n_swarms
        self.n_particles = n_particles_per_swarm
        self.n_params = n_params
        self.device = device
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Total particles across all swarms
        self.total_particles = n_swarms * n_particles_per_swarm
        
        # Parameter bounds
        self.lb, self.ub = bounds[0].to(device), bounds[1].to(device)
        
        # Initialize positions randomly within bounds
        self.positions = self.lb + (self.ub - self.lb) * torch.rand(
            self.total_particles, n_params, device=device
        )
        
        # Initialize velocities to zero
        self.velocities = torch.zeros(self.total_particles, n_params, device=device)
        
        # Personal best (per particle)
        self.pbest_positions = self.positions.clone()
        self.pbest_scores = torch.full((self.total_particles,), float('inf'), device=device)
        
        # Global best (per swarm)
        self.gbest_positions = self.positions[:n_swarms].clone()
        self.gbest_scores = torch.full((n_swarms,), float('inf'), device=device)
        
        # Track which swarm each particle belongs to
        self.swarm_ids = torch.arange(n_swarms, device=device).repeat_interleave(n_particles_per_swarm)
    
    def update(self, scores: torch.Tensor):
        """
        Update particle positions based on new scores.
        
        Args:
            scores: [total_particles] loss for each particle
        """
        # Update personal bests
        improved = scores < self.pbest_scores
        self.pbest_positions[improved] = self.positions[improved].clone()
        self.pbest_scores[improved] = scores[improved]
        
        # Update global bests (per swarm)
        for s in range(self.n_swarms):
            mask = self.swarm_ids == s
            swarm_scores = scores[mask]
            min_idx_local = swarm_scores.argmin()
            min_idx_global = mask.nonzero()[min_idx_local]
            
            if scores[min_idx_global] < self.gbest_scores[s]:
                self.gbest_scores[s] = scores[min_idx_global].clone()
                self.gbest_positions[s] = self.positions[min_idx_global].clone().squeeze(0)
        
        # Get each particle's swarm's global best
        gbest_for_particles = self.gbest_positions[self.swarm_ids]
        
        # Random factors
        r1 = torch.rand(self.total_particles, self.n_params, device=self.device)
        r2 = torch.rand(self.total_particles, self.n_params, device=self.device)
        
        # Velocity update
        cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
        social = self.c2 * r2 * (gbest_for_particles - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social
        
        # Position update (clamped to bounds)
        self.positions = torch.clamp(self.positions + self.velocities, self.lb, self.ub)


# =============================================================================
# PARTICLE EVALUATION
# =============================================================================

def evaluate_particles(model_class, positions, cond, device):
    """
    Evaluate all particles using rk4 (fast, for PSO iterations).
    
    Args:
        model_class: The model class (e.g., FullModel)
        positions: [n_particles, n_params] parameter values
        cond: ConditionData object
        device: torch device
    
    Returns:
        losses: [n_particles] loss per particle
    """
    n_particles = positions.shape[0]
    
    # Create model with batch size = number of particles
    model = model_class(n_batch=n_particles, device=device)
    model.set_condition(cd3=cond.cd3, cd28=cond.cd28)
    model_class.params_to_model(positions, model)
    
    # Prepare data
    t = cond.t.to(device)
    y_mean = cond.y_mean.to(device)
    y_std = cond.y_std.to(device)
    y0 = cond.y0.unsqueeze(0).expand(n_particles, -1).to(device)
    
    with torch.no_grad():
        # Integrate ODE
        solution = odeint(model, y0, t, method='rk4')
        
        # Compute loss
        losses = compute_loss(solution, y_mean, y_std)
        
        # Mark invalid solutions
        has_nan = torch.isnan(solution).any(dim=(0, 2))
        has_inf = torch.isinf(solution).any(dim=(0, 2))
        has_negative = (solution < -1).any(dim=(0, 2))
        invalid = has_nan | has_inf | has_negative
        
        # Set invalid losses to penalty
        losses = torch.where(invalid, torch.full_like(losses, 1e6), losses)
        losses = torch.where(torch.isfinite(losses), losses, torch.full_like(losses, 1e6))
    
    return losses


def evaluate_single_dopri5(model_class, position, cond, device):
    """
    Evaluate a single solution using dopri5 (accurate, for final refinement).
    
    Args:
        model_class: The model class
        position: [n_params] single parameter vector
        cond: ConditionData object
        device: torch device
    
    Returns:
        loss: float, the loss value (1e6 if failed)
    """
    pos = position.unsqueeze(0)  # Add batch dimension
    
    model = model_class(n_batch=1, device=device)
    model.set_condition(cd3=cond.cd3, cd28=cond.cd28)
    model_class.params_to_model(pos, model)
    
    y0 = cond.y0.unsqueeze(0).to(device)
    t = cond.t.to(device)
    y_mean = cond.y_mean.to(device)
    y_std = cond.y_std.to(device)
    
    try:
        with torch.no_grad():
            solution = odeint(model, y0, t, method='dopri5', options={'max_num_steps': 5000})
            loss = compute_loss(solution, y_mean, y_std)
            if torch.isfinite(loss):
                return loss.item()
    except:
        pass
    
    return 1e6


# =============================================================================
# RUN SINGLE CONDITION (CD3=1, CD28=1 only)
# =============================================================================

def run_single_condition(
    model_class,
    data: dict,
    device: torch.device,
    output_dir: Path,
    target_valid_fits: int = 10000,
    n_swarms_per_batch: int = 300,
    n_particles: int = 200,
    n_iterations: int = 130,
    cd3: float = 1.0,
    cd28: float = 1.0,
    verbose: bool = True,
):
    """
    Run PSO optimization for a single CD3/CD28 condition.
    
    Args:
        model_class: The model class to use (e.g., FullModel)
        data: Dictionary of ConditionData objects
        device: torch device
        output_dir: Where to save results
        target_valid_fits: Stop when this many valid fits found
        n_swarms_per_batch: Number of swarms per batch
        n_particles: Particles per swarm
        n_iterations: PSO iterations per batch
        cd3: CD3 concentration (default 1.0)
        cd28: CD28 concentration (default 1.0)
        verbose: Print progress
    
    Returns:
        DataFrame with valid parameter sets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model info
    temp_model = model_class(n_batch=1, device=device)
    model_name = temp_model.model_name
    n_params = temp_model.n_params
    del temp_model
    
    # Get model-specific bounds and param names
    lb, ub = model_class.get_param_bounds(device)
    param_names = model_class.get_param_names()
    
    # Output file
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = output_dir / f"params_{safe_name}_CD3_{cd3}_CD28_{cd28}.csv"
    
    if output_path.exists():
        output_path.unlink()
    
    # Get condition data
    cond = data[(cd3, cd28)]
    
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} ({n_params} parameters)")
    print(f"CONDITION: CD3={cd3}, CD28={cd28}")
    print(f"TARGET: {target_valid_fits} valid fits")
    print(f"SETTINGS: {n_swarms_per_batch} swarms × {n_particles} particles × {n_iterations} iters")
    print('='*70)
    
    # Prepare tensors
    t = cond.t.to(device)
    y_mean = cond.y_mean.to(device)
    y_std = cond.y_std.to(device)
    
    total_valid = 0
    batch_num = 0
    first_batch = True
    start_time = time.time()
    
    while total_valid < target_valid_fits:
        batch_num += 1
        batch_start = time.time()
        
        if verbose:
            print(f"\n  Batch {batch_num}: {total_valid}/{target_valid_fits} valid so far")
        
        # Create PSO
        pso = MultiSwarmPSO(n_swarms_per_batch, n_particles, n_params, (lb, ub), device)
        
        # Run PSO with rk4
        for i in range(n_iterations):
            losses = evaluate_particles(model_class, pso.positions, cond, device)
            pso.update(losses)
            
            if verbose and (i == 0 or i == n_iterations - 1):
                n_valid = (pso.gbest_scores < 1e5).sum().item()
                print(f"    Iter {i:3d}: best={pso.gbest_scores.min().item():.4f}  valid={n_valid}/{n_swarms_per_batch}")
        
        # Refine with dopri5 (only rk4-valid solutions)
        rk4_valid_mask = pso.gbest_scores < 1e5
        rk4_valid_indices = rk4_valid_mask.nonzero().squeeze(-1)
        if rk4_valid_indices.dim() == 0:
            rk4_valid_indices = rk4_valid_indices.unsqueeze(0)
        
        n_rk4_valid = len(rk4_valid_indices)
        if verbose:
            print(f"    Refining {n_rk4_valid} with dopri5...")
        
        final_losses = torch.full((n_swarms_per_batch,), 1e6, device=device)
        
        for idx in rk4_valid_indices:
            idx = idx.item()
            loss = evaluate_single_dopri5(model_class, pso.gbest_positions[idx], cond, device)
            final_losses[idx] = loss
        
        # Save valid results
        valid_indices = (final_losses < 1e5).nonzero().squeeze(-1)
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)
        
        n_dopri5_valid = (final_losses < 1e5).sum().item()
        
        if len(valid_indices) > 0:
            df = pd.DataFrame(
                pso.gbest_positions[valid_indices].cpu().numpy(),
                columns=param_names
            )
            df['loss'] = final_losses[valid_indices].cpu().numpy()
            df['cd3'] = cd3
            df['cd28'] = cd28
            df['model'] = model_name
            df['batch_id'] = batch_num
            df['fit_id'] = range(total_valid, total_valid + len(valid_indices))
            
            df.to_csv(output_path, mode='a', header=first_batch, index=False)
            first_batch = False
            total_valid += len(df)
        
        batch_time = time.time() - batch_start
        if verbose:
            best = final_losses[final_losses < 1e5].min().item() if n_dopri5_valid > 0 else float('inf')
            print(f"    Done: {n_rk4_valid} rk4 → {n_dopri5_valid} dopri5, best={best:.4f}, time={batch_time:.1f}s")
        
        # Cleanup
        del pso, losses, final_losses
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    total_time = time.time() - start_time
    final_df = pd.read_csv(output_path)
    
    print(f"\n{'='*70}")
    print(f"COMPLETE: {model_name}")
    print('='*70)
    print(f"  Valid fits: {len(final_df)}")
    print(f"  Best loss: {final_df['loss'].min():.4f}")
    print(f"  Median loss: {final_df['loss'].median():.4f}")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"  Saved: {output_path}")
    
    return final_df


# =============================================================================
# RUN ALL CONDITIONS
# =============================================================================

def run_all_conditions(
    model_class,
    data: dict,
    device: torch.device,
    output_dir: Path,
    target_valid_fits: int = 10000,
    n_swarms_per_batch: int = 500,
    n_particles: int = 200,
    n_iterations: int = 150,
    verbose: bool = True,
):
    """
    Run PSO optimization for all 6 CD3/CD28 conditions.
    
    Args:
        model_class: The model class to use (e.g., FullModel)
        data: Dictionary of ConditionData objects
        device: torch device
        output_dir: Where to save results
        target_valid_fits: Valid fits per condition
        n_swarms_per_batch: Number of swarms per batch
        n_particles: Particles per swarm
        n_iterations: PSO iterations per batch
        verbose: Print progress
    
    Returns:
        Summary DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model info
    temp_model = model_class(n_batch=1, device=device)
    model_name = temp_model.model_name
    n_params = temp_model.n_params
    del temp_model
    
    # Get model-specific bounds and param names
    lb, ub = model_class.get_param_bounds(device)
    param_names = model_class.get_param_names()
    
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} ({n_params} parameters)")
    print(f"TARGET: {target_valid_fits} valid fits per condition")
    print(f"CONDITIONS: {len(CD3_CD28_COMBINATIONS)}")
    print('='*70)
    
    summary = []
    
    for cd3, cd28 in CD3_CD28_COMBINATIONS:
        print(f"\n{'='*70}")
        print(f"CONDITION: CD3={cd3}, CD28={cd28}")
        print(f"Target: {target_valid_fits} valid fits")
        print('='*70)
        
        cond = data[(cd3, cd28)]
        
        # Output file for this condition
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        output_path = output_dir / f"params_{safe_name}_CD3_{cd3}_CD28_{cd28}.csv"
        
        if output_path.exists():
            output_path.unlink()
        
        # Prepare tensors
        t = cond.t.to(device)
        y_mean = cond.y_mean.to(device)
        y_std = cond.y_std.to(device)
        
        total_valid = 0
        batch_num = 0
        first_batch = True
        condition_start = time.time()
        
        while total_valid < target_valid_fits:
            batch_num += 1
            batch_start = time.time()
            
            if verbose:
                print(f"\n  Batch {batch_num}: {total_valid}/{target_valid_fits} valid so far")
            
            pso = MultiSwarmPSO(n_swarms_per_batch, n_particles, n_params, (lb, ub), device)
            
            # rk4 PSO
            for i in range(n_iterations):
                losses = evaluate_particles(model_class, pso.positions, cond, device)
                pso.update(losses)
                
                if verbose and (i == 0 or i == n_iterations - 1):
                    n_valid = (pso.gbest_scores < 1e5).sum().item()
                    print(f"    Iter {i:3d}: best={pso.gbest_scores.min().item():.4f}  valid={n_valid}/{n_swarms_per_batch}")
            
            # dopri5 refinement
            rk4_valid_mask = pso.gbest_scores < 1e5
            rk4_valid_indices = rk4_valid_mask.nonzero().squeeze(-1)
            if rk4_valid_indices.dim() == 0:
                rk4_valid_indices = rk4_valid_indices.unsqueeze(0)
            
            n_rk4_valid = len(rk4_valid_indices)
            if verbose:
                print(f"    Refining {n_rk4_valid} with dopri5...")
            
            final_losses = torch.full((n_swarms_per_batch,), 1e6, device=device)
            
            for idx in rk4_valid_indices:
                idx = idx.item()
                loss = evaluate_single_dopri5(model_class, pso.gbest_positions[idx], cond, device)
                final_losses[idx] = loss
            
            # Save valid
            valid_indices = (final_losses < 1e5).nonzero().squeeze(-1)
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            n_dopri5_valid = (final_losses < 1e5).sum().item()
            
            if len(valid_indices) > 0:
                df = pd.DataFrame(
                    pso.gbest_positions[valid_indices].cpu().numpy(),
                    columns=param_names
                )
                df['loss'] = final_losses[valid_indices].cpu().numpy()
                df['cd3'] = cd3
                df['cd28'] = cd28
                df['model'] = model_name
                df['batch_id'] = batch_num
                df['fit_id'] = range(total_valid, total_valid + len(valid_indices))
                
                df.to_csv(output_path, mode='a', header=first_batch, index=False)
                first_batch = False
                total_valid += len(df)
            
            batch_time = time.time() - batch_start
            if verbose:
                best = final_losses[final_losses < 1e5].min().item() if n_dopri5_valid > 0 else float('inf')
                print(f"    Done: {n_rk4_valid} rk4 → {n_dopri5_valid} dopri5, best={best:.4f}, time={batch_time:.1f}s")
            
            del pso, losses, final_losses
            torch.cuda.empty_cache()
            gc.collect()
        
        condition_time = time.time() - condition_start
        final_df = pd.read_csv(output_path)
        
        print(f"\n  CONDITION COMPLETE: CD3={cd3}, CD28={cd28}")
        print(f"  Valid fits: {len(final_df)}")
        print(f"  Loss range: {final_df['loss'].min():.4f} - {final_df['loss'].max():.4f}")
        print(f"  Time: {condition_time/60:.1f} minutes")
        
        summary.append({
            'model': model_name,
            'cd3': cd3,
            'cd28': cd28,
            'n_valid_fits': len(final_df),
            'best_loss': final_df['loss'].min(),
            'median_loss': final_df['loss'].median(),
            'time_minutes': condition_time / 60
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / f"fitting_summary_{safe_name}.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print("ALL CONDITIONS COMPLETE")
    print('='*70)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")
    
    return summary_df

# =============================================================================
# GLOBAL FIT (all conditions simultaneously, shared parameters)
# =============================================================================

def evaluate_particles_global(model_class, positions, all_cond_data, device):
    """
    Evaluate particles across ALL conditions with shared parameters.
    
    Args:
        model_class: The model class
        positions: [n_particles, n_params] parameter values
        all_cond_data: dict of (cd3, cd28) -> ConditionData
        device: torch device
    
    Returns:
        losses: [n_particles] total loss across all conditions
    """
    n_particles = positions.shape[0]
    total_losses = torch.zeros(n_particles, device=device)
    
    for (cd3, cd28), cond in all_cond_data.items():
        # Create model with batch size = number of particles
        model = model_class(n_batch=n_particles, device=device)
        model.set_condition(cd3=cd3, cd28=cd28)
        model_class.params_to_model(positions, model)
        
        # Prepare data
        t = cond.t.to(device)
        y_mean = cond.y_mean.to(device)
        y_std = cond.y_std.to(device)
        y0 = cond.y0.unsqueeze(0).expand(n_particles, -1).to(device)
        
        with torch.no_grad():
            # Integrate ODE
            solution = odeint(model, y0, t, method='rk4')
            
            # Compute loss for this condition
            losses = compute_loss(solution, y_mean, y_std)
            
            # Mark invalid solutions
            has_nan = torch.isnan(solution).any(dim=(0, 2))
            has_inf = torch.isinf(solution).any(dim=(0, 2))
            has_negative = (solution < -1).any(dim=(0, 2))
            invalid = has_nan | has_inf | has_negative
            
            # Set invalid losses to penalty
            losses = torch.where(invalid, torch.full_like(losses, 1e6), losses)
            losses = torch.where(torch.isfinite(losses), losses, torch.full_like(losses, 1e6))
            
            # Accumulate
            total_losses += losses
    
    return total_losses


def evaluate_single_dopri5_global(model_class, position, all_cond_data, device):
    """
    Evaluate a single solution using dopri5 across ALL conditions.
    
    Args:
        model_class: The model class
        position: [n_params] single parameter vector
        all_cond_data: dict of (cd3, cd28) -> ConditionData
        device: torch device
    
    Returns:
        total_loss: float, sum of losses across all conditions (1e6 if any failed)
    """
    pos = position.unsqueeze(0)  # Add batch dimension
    total_loss = 0.0
    
    for (cd3, cd28), cond in all_cond_data.items():
        model = model_class(n_batch=1, device=device)
        model.set_condition(cd3=cd3, cd28=cd28)
        model_class.params_to_model(pos, model)
        
        y0 = cond.y0.unsqueeze(0).to(device)
        t = cond.t.to(device)
        y_mean = cond.y_mean.to(device)
        y_std = cond.y_std.to(device)
        
        try:
            with torch.no_grad():
                solution = odeint(model, y0, t, method='dopri5', options={'max_num_steps': 5000})
                loss = compute_loss(solution, y_mean, y_std)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                else:
                    return 1e6  # Any invalid condition fails the whole fit
        except:
            return 1e6
    
    return total_loss


def run_global_fit(
    model_class,
    data: dict,
    device: torch.device,
    output_dir: Path,
    target_valid_fits: int = 10000,
    n_swarms_per_batch: int = 300,
    n_particles: int = 200,
    n_iterations: int = 130,
    verbose: bool = True,
):
    """
    Run PSO optimization with SHARED parameters across all 6 conditions.
    
    Each parameter set is evaluated on ALL conditions simultaneously.
    Total loss = sum of losses across all conditions.
    This ensures parameters are globally consistent and identifiable.
    
    Args:
        model_class: The model class to use (e.g., FullModel_4_SD)
        data: Dictionary of ConditionData objects (all 6 conditions)
        device: torch device
        output_dir: Where to save results
        target_valid_fits: Stop when this many valid fits found
        n_swarms_per_batch: Number of swarms per batch
        n_particles: Particles per swarm
        n_iterations: PSO iterations per batch
        verbose: Print progress
    
    Returns:
        DataFrame with valid parameter sets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model info
    temp_model = model_class(n_batch=1, device=device)
    model_name = temp_model.model_name
    n_params = temp_model.n_params
    del temp_model
    
    # Get model-specific bounds and param names
    lb, ub = model_class.get_param_bounds(device)
    param_names = model_class.get_param_names()
    
    # Output file
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = output_dir / f"params_{safe_name}_GLOBAL.csv"
    
    if output_path.exists():
        output_path.unlink()
    
    print(f"\n{'='*70}")
    print(f"GLOBAL FIT: {model_name} ({n_params} parameters)")
    print(f"Fitting ALL {len(data)} conditions simultaneously")
    print(f"TARGET: {target_valid_fits} valid fits")
    print(f"SETTINGS: {n_swarms_per_batch} swarms × {n_particles} particles × {n_iterations} iters")
    print('='*70)
    print("\nConditions:")
    for cd3, cd28 in data.keys():
        print(f"  CD3={cd3}, CD28={cd28}")
    print()
    
    total_valid = 0
    batch_num = 0
    first_batch = True
    start_time = time.time()
    
    while total_valid < target_valid_fits:
        batch_num += 1
        batch_start = time.time()
        
        if verbose:
            print(f"\n  Batch {batch_num}: {total_valid}/{target_valid_fits} valid so far")
        
        # Create PSO
        pso = MultiSwarmPSO(n_swarms_per_batch, n_particles, n_params, (lb, ub), device)
        
        # Run PSO with rk4 - evaluate across ALL conditions
        for i in range(n_iterations):
            losses = evaluate_particles_global(model_class, pso.positions, data, device)
            pso.update(losses)
            
            if verbose and (i == 0 or i == n_iterations - 1):
                # Valid means total loss across all 6 conditions < 6e5 (avg < 1e5 per condition)
                n_valid = (pso.gbest_scores < 6e5).sum().item()
                best = pso.gbest_scores.min().item()
                print(f"    Iter {i:3d}: best={best:.4f} (total), valid={n_valid}/{n_swarms_per_batch}")
        
        # Refine with dopri5 (only rk4-valid solutions)
        # Threshold: total loss < 6e5 means average per condition < 1e5
        rk4_valid_mask = pso.gbest_scores < 6e5
        rk4_valid_indices = rk4_valid_mask.nonzero().squeeze(-1)
        if rk4_valid_indices.dim() == 0:
            rk4_valid_indices = rk4_valid_indices.unsqueeze(0)
        
        n_rk4_valid = len(rk4_valid_indices)
        if verbose:
            print(f"    Refining {n_rk4_valid} with dopri5 (all conditions)...")
        
        final_losses = torch.full((n_swarms_per_batch,), 1e7, device=device)
        
        for idx in rk4_valid_indices:
            idx = idx.item()
            loss = evaluate_single_dopri5_global(model_class, pso.gbest_positions[idx], data, device)
            final_losses[idx] = loss
        
        # Save valid results (total loss < 6e5)
        valid_mask = final_losses < 6e5
        valid_indices = valid_mask.nonzero().squeeze(-1)
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)
        
        n_dopri5_valid = valid_mask.sum().item()
        
        if len(valid_indices) > 0:
            df = pd.DataFrame(
                pso.gbest_positions[valid_indices].cpu().numpy(),
                columns=param_names
            )
            df['total_loss'] = final_losses[valid_indices].cpu().numpy()
            df['avg_loss'] = df['total_loss'] / len(data)  # Average per condition
            df['model'] = model_name
            df['n_conditions'] = len(data)
            df['batch_id'] = batch_num
            df['fit_id'] = range(total_valid, total_valid + len(valid_indices))
            
            df.to_csv(output_path, mode='a', header=first_batch, index=False)
            first_batch = False
            total_valid += len(df)
        
        batch_time = time.time() - batch_start
        if verbose:
            best = final_losses[valid_mask].min().item() if n_dopri5_valid > 0 else float('inf')
            avg_best = best / len(data) if n_dopri5_valid > 0 else float('inf')
            print(f"    Done: {n_rk4_valid} rk4 → {n_dopri5_valid} dopri5")
            print(f"    Best total={best:.4f}, avg={avg_best:.4f}, time={batch_time:.1f}s")
        
        # Cleanup
        del pso, losses, final_losses
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    total_time = time.time() - start_time
    final_df = pd.read_csv(output_path)
    
    print(f"\n{'='*70}")
    print(f"GLOBAL FIT COMPLETE: {model_name}")
    print('='*70)
    print(f"  Valid fits: {len(final_df)}")
    print(f"  Best total loss: {final_df['total_loss'].min():.4f}")
    print(f"  Best avg loss: {final_df['avg_loss'].min():.4f}")
    print(f"  Median avg loss: {final_df['avg_loss'].median():.4f}")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"  Saved: {output_path}")
    
    return final_df