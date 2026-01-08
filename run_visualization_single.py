# run_visualization_single.py
"""Generate plots for single condition results."""

import torch
from pathlib import Path

from data import prepare_all_conditions
from visualization import generate_single_condition_plots

from models.model_1_literature import LiteratureModel_1_S
SELECTED_MODEL = LiteratureModel_1_S

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
    generate_single_condition_plots(
        model_class=SELECTED_MODEL,
        data=data,
        device=device,
        results_dir=results_dir,
        cd3=1.0,
        cd28=1.0,
        top_n=10,
    )