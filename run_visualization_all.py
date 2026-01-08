# run_visualization_all.py
"""Generate plots for all conditions results."""

import torch
from pathlib import Path

from data import prepare_all_conditions
from visualization import generate_all_conditions_plots

from models.model_3_full.py import FullModel_3_SD
SELECTED_MODEL = FullModel

BASE_OUTPUT_DIR = Path(__file__).parent / "results"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get model folder
    temp_model = SELECTED_MODEL(n_batch=1, device=device)
    model_name = temp_model.model_name
    del temp_model
    
    folder_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    results_dir = BASE_OUTPUT_DIR / folder_name
    
    # Load data
    data = prepare_all_conditions(device=device)
    
    # Generate plots
    generate_all_conditions_plots(
        model_class=SELECTED_MODEL,
        data=data,
        device=device,
        results_dir=results_dir,
        top_n=10,
    )
