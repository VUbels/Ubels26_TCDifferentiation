import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Callable
from pathlib import Path
from scipy.interpolate import PchipInterpolator

from constants import (
    EXCEL_PATH, SHEET_NAMES, CYTOKINE_COLS, CD4_COUNT_COL,
    TIMEPOINTS, CD3_CD28_COMBINATIONS
)


@dataclass
class ConditionData:
    """Data container for a single CD3/CD28 condition."""
    cd3: float
    cd28: float
    t: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    y0: torch.Tensor
    cd4_interpolator: Callable
    
    def cd4(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate CD4 count at time t."""
        t_np = t.detach().cpu().numpy()
        cd4_np = self.cd4_interpolator(t_np)
        return torch.tensor(cd4_np, dtype=t.dtype, device=t.device)


def load_control_data(excel_path: Path = EXCEL_PATH) -> pd.DataFrame:
    """Load and concatenate control data from all patient sheets."""
    frames = []
    for sheet in SHEET_NAMES:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df["Patient"] = sheet
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined[combined["Cytokine"] == "Control"].copy()


def prepare_condition_data(
    df: pd.DataFrame,
    cd3: float,
    cd28: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> ConditionData:
    """Prepare all data for a single CD3/CD28 condition."""
    
    subset = df[
        (np.isclose(df["anti-CD3 (ug/mL)"], cd3)) &
        (np.isclose(df["anti-CD28 (ug/mL)"], cd28))
    ]
    
    n_timepoints = len(TIMEPOINTS)
    n_cytokines = len(CYTOKINE_COLS)
    
    cytokine_per_patient = np.zeros((n_timepoints, n_cytokines, len(SHEET_NAMES)))
    cd4_per_patient = np.zeros((n_timepoints, len(SHEET_NAMES)))
    
    for t_idx, day in enumerate(TIMEPOINTS):
        day_data = subset[subset["Day"] == day]
        for p_idx, patient in enumerate(SHEET_NAMES):
            patient_data = day_data[day_data["Patient"] == patient]
            for c_idx, cyt_col in enumerate(CYTOKINE_COLS):
                cytokine_per_patient[t_idx, c_idx, p_idx] = patient_data[cyt_col].mean()
            cd4_per_patient[t_idx, p_idx] = patient_data[CD4_COUNT_COL].mean()
    
    y_mean = np.mean(cytokine_per_patient, axis=2)
    y_std = np.maximum(np.std(cytokine_per_patient, axis=2, ddof=1), 1e-6)
    cd4_mean = np.mean(cd4_per_patient, axis=1)
    
    return ConditionData(
        cd3=cd3,
        cd28=cd28,
        t=torch.tensor(TIMEPOINTS, dtype=dtype, device=device),
        y_mean=torch.tensor(y_mean, dtype=dtype, device=device),
        y_std=torch.tensor(y_std, dtype=dtype, device=device),
        y0=torch.tensor(y_mean[0, :], dtype=dtype, device=device),
        cd4_interpolator=PchipInterpolator(TIMEPOINTS, cd4_mean),
    )


def prepare_all_conditions(
    excel_path: Path = EXCEL_PATH,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> dict[tuple[float, float], ConditionData]:
    """Load and prepare data for all CD3/CD28 conditions."""
    df = load_control_data(excel_path)
    return {
        (cd3, cd28): prepare_condition_data(df, cd3, cd28, device, dtype)
        for cd3, cd28 in CD3_CD28_COMBINATIONS
    }