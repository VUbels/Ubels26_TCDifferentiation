# run_all_conditions.py
"""
Run global fitting: shared parameters across all 6 CD3/CD28 conditions.
"""

import torch
from pathlib import Path

from data import prepare_all_conditions
from optimization import run_global_fit

# =============================================================================
# SELECT YOUR MODEL HERE
# =============================================================================

from models.model_4_full import FullModel_4_SD
SELECTED_MODEL = FullModel_4_SD

# =============================================================================
# SETTINGS
# =============================================================================

BASE_OUTPUT_DIR = Path(__file__).parent / "results"

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
    
    temp_model = SELECTED_MODEL(n_batch=1, device=device)
    model_name = temp_model.model_name
    del temp_model
    
    folder_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_dir = BASE_OUTPUT_DIR / folder_name
    
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    
    print("Loading data...")
    data = prepare_all_conditions(device=device)
    
    results = run_global_fit(
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