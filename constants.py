# constants.py
"""Constants shared across all models."""

import numpy as np
from pathlib import Path

# Data paths
EXCEL_PATH = Path("/mnt/d/UVA/TCR/Data/Donors_Table_3.xlsx")
OUTPUT_DIR = Path("/mnt/d/UVA/TCR/2026_Updated/PSO_results")

# Experimental setup
SHEET_NAMES = ["BUF18-0048", "BUF18-0050", "BUF18-0052"]
CYTOKINE_COLS = ["IFN-g+", "IL-2+", "IL-4+", "IL-21+"]
CD4_COUNT_COL = "CD4 T cell count"
TIMEPOINTS = np.array([3, 5, 7, 9, 11])

CD3_CD28_COMBINATIONS = [
    (0.01, 1.0),
    (0.1, 0.1),
    (0.1, 1.0),
    (1.0, 1.0),
    (5.0, 1.0),
    (10.0, 10.0),
]

# Model constant
IL12_0 = 10.0  # IL-12 supplementation in arbitrary units

# Display
CYTOKINE_NAMES = ['IFN-γ+', 'IL-2+', 'IL-4+', 'IL-21+']
CYTOKINE_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
