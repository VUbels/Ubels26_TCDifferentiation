# run_all_conditions.py
"""
Run a single model on all 6 CD3/CD28 conditions.
Used for full model fitting.

Usage:
    1. Change SELECTED_MODEL below to the model you want
    2. Adjust settings if needed
    3. Run: python run_all_conditions.py
"""

import torch
from pathlib import Path

from data import prepare_all_conditions
from optimisation import run_all_conditions

# =============================================================================
# SELECT YOUR MODEL HERE
# =============================================================================

from models.model_4_full import FullModel_4_SD
SELECTED_MODEL = FullModel_4_SD


# =============================================================================
# SETTINGS
# =============================================================================

BASE_OUTPUT_DIR = Path("/mnt/d/UVA/TCR/2026_Updated/PSO_results")

TARGET_FITS = 10000
N_SWARMS = 300
N_PARTICLES = 200
N_ITERATIONS = 130

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get model name for output folder
    temp_model = SELECTED_MODEL(n_batch=1, device=device)
    model_name = temp_model.model_name
    del temp_model
    
    # Create model-specific output directory
    folder_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_dir = BASE_OUTPUT_DIR / folder_name
    
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    
    # Load data
    print("Loading data...")
    data = prepare_all_conditions(device=device)
    
    # Run
    summary = run_all_conditions(
        model_class=SELECTED_MODEL,
        data=data,
        device=device,
        output_dir=output_dir,
        target_valid_fits=TARGET_FITS,
        n_swarms_per_batch=N_SWARMS,
        n_particles=N_PARTICLES,
        n_iterations=N_ITERATIONS,
    )
    
    print("\nDone!")
