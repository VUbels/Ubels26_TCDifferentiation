# models/base.py
"""Base class that all T cell models must inherit from."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseTcellModel(ABC, nn.Module):
    """
    Abstract base class for T cell differentiation models.
    
    All models must:
    - Define n_params property
    - Define model_name attribute
    - Implement forward(t, y) method
    - Implement get_param_bounds(device) static method
    - Implement get_param_names() static method
    - Implement params_to_model(positions, model) static method
    """
    
    def __init__(self, n_batch: int, device: torch.device):
        super().__init__()
        self.n_batch = n_batch
        self.device = device
        self.cd3 = None
        self.cd28 = None
    
    def set_condition(self, cd3: float, cd28: float):
        """Set the CD3/CD28 stimulation condition."""
        self.cd3 = cd3
        self.cd28 = cd28
    
    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of parameters in this model."""
        pass
    
    @abstractmethod
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ODE right-hand side. Returns dy/dt."""
        pass
    
    @staticmethod
    @abstractmethod
    def get_param_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lower_bounds, upper_bounds) tensors."""
        pass
    
    @staticmethod
    @abstractmethod
    def get_param_names() -> list[str]:
        """Return list of parameter names in order."""
        pass
    
    @staticmethod
    @abstractmethod
    def params_to_model(positions: torch.Tensor, model: 'BaseTcellModel'):
        """Map PSO position vector to model parameters."""
        pass