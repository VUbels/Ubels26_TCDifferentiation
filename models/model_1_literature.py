# models/model_1_literature.py
"""
Literature-based model.

Model summary
1. Negative terms inhibit differentiation (subtract from production)
2. No Hill functions
3. Reduced parameter set based on literature support
4. Only basal degradation (k_deg * species)

Parameters: 28 total
- Basal production: 4
- Basal degradation: 4
- Interaction k: 19 (k1,k3,k5,k6,k7,k8,k9,k11,k12,k14,k20,k22,k24,k26,k28,k29,k30,k31,k32)
- IL4 inhibition of IL12: 1
"""

import torch
import torch.nn as nn
from .base import BaseTcellModel
from constants import IL12_0


class LiteratureModel_1_S(BaseTcellModel):
    """
    Literature-based 28-parameter model.
    
    Simplified interactions based on well-established literature.
    No Hill functions - all interactions are mass-action.
    Negative terms inhibit differentiation (subtract from production).
    """
    
    model_name = "Literature Model (28 params)"
    
    def __init__(self, n_batch: int = 1, device: torch.device = torch.device("cpu")):
        super().__init__(n_batch, device)
        
        # Basal rates (8)
        self.log_k_prod = nn.Parameter(torch.zeros(n_batch, 4, device=device))
        self.log_k_deg = nn.Parameter(torch.zeros(n_batch, 4, device=device))
        
        # IL-2 equation: k1, k3, k5, k6, k7 (5)
        self.log_k1 = nn.Parameter(torch.zeros(n_batch, device=device))   # CD3 -> IL2
        self.log_k3 = nn.Parameter(torch.zeros(n_batch, device=device))   # CD28 -> IL2
        self.log_k5 = nn.Parameter(torch.zeros(n_batch, device=device))   # IL21 inhibits IL2 diff
        self.log_k6 = nn.Parameter(torch.zeros(n_batch, device=device))   # IL4 inhibits IL2 diff
        self.log_k7 = nn.Parameter(torch.zeros(n_batch, device=device))   # IL12 -> IL2
        
        # IFNg equation: k8, k9, k11, k12, k14 (5)
        self.log_k8 = nn.Parameter(torch.zeros(n_batch, device=device))   # CD3 -> IFNg
        self.log_k9 = nn.Parameter(torch.zeros(n_batch, device=device))   # CD28 -> IFNg
        self.log_k11 = nn.Parameter(torch.zeros(n_batch, device=device))  # IL2*IL12 -> IFNg
        self.log_k12 = nn.Parameter(torch.zeros(n_batch, device=device))  # IFNg positive feedback
        self.log_k14 = nn.Parameter(torch.zeros(n_batch, device=device))  # IL4 inhibits IFNg diff
        
        # IL-21 equation: k20 only (1)
        self.log_k20 = nn.Parameter(torch.zeros(n_batch, device=device))  # IL2 inhibits IL21 diff
        
        # IL-4 equation: k22, k24, k26, k28, k29, k30, k31, k32 (8)
        self.log_k22 = nn.Parameter(torch.zeros(n_batch, device=device))  # Basal IL4 production
        self.log_k24 = nn.Parameter(torch.zeros(n_batch, device=device))  # CD28 -> IL4
        self.log_k26 = nn.Parameter(torch.zeros(n_batch, device=device))  # IL2 -> IL4
        self.log_k28 = nn.Parameter(torch.zeros(n_batch, device=device))  # IL21 -> IL4
        self.log_k29 = nn.Parameter(torch.zeros(n_batch, device=device))  # IL21 inhibits IL4 diff
        self.log_k30 = nn.Parameter(torch.zeros(n_batch, device=device))  # IL4 positive feedback
        self.log_k31 = nn.Parameter(torch.zeros(n_batch, device=device))  # IFNg inhibits IL4 diff
        self.log_k32 = nn.Parameter(torch.zeros(n_batch, device=device))  # IL12 inhibits IL4 diff
        
        # IL-4 inhibition of IL-12 (1)
        self.log_K_IL4inh = nn.Parameter(torch.zeros(n_batch, device=device))
    
    @property
    def n_params(self) -> int:
        return 28
    
    def il12_effective(self, IL4):
        """IL-12 effectiveness reduced by IL-4."""
        K_IL4inh = torch.exp(self.log_K_IL4inh)
        return IL12_0 / (1 + K_IL4inh * IL4)
    
    def forward(self, t, y):
        IL2, IFNg, IL21, IL4 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        
        # Get positive parameters
        k_prod = torch.exp(self.log_k_prod)
        k_deg = torch.exp(self.log_k_deg)
        
        k1 = torch.exp(self.log_k1)
        k3 = torch.exp(self.log_k3)
        k5 = torch.exp(self.log_k5)
        k6 = torch.exp(self.log_k6)
        k7 = torch.exp(self.log_k7)
        
        k8 = torch.exp(self.log_k8)
        k9 = torch.exp(self.log_k9)
        k11 = torch.exp(self.log_k11)
        k12 = torch.exp(self.log_k12)
        k14 = torch.exp(self.log_k14)
        
        k20 = torch.exp(self.log_k20)
        
        k22 = torch.exp(self.log_k22)
        k24 = torch.exp(self.log_k24)
        k26 = torch.exp(self.log_k26)
        k28 = torch.exp(self.log_k28)
        k29 = torch.exp(self.log_k29)
        k30 = torch.exp(self.log_k30)
        k31 = torch.exp(self.log_k31)
        k32 = torch.exp(self.log_k32)
        
        IL12_eff = self.il12_effective(IL4)
        
        # IL-2: CD3, CD28, IL12 promote; IL21, IL4 inhibit differentiation
        dIL2 = (
            k_prod[:, 0]
            + k1 * self.cd3
            + k3 * self.cd28
            + k7 * IL12_eff
            - k5 * IL21        # IL21 inhibits differentiation (not * IL2)
            - k6 * IL4         # IL4 inhibits differentiation (not * IL2)
            - k_deg[:, 0] * IL2
        )
        
        # IFNg: CD3, CD28, IL2*IL12, self-feedback promote; IL4 inhibits differentiation
        dIFNg = (
            k_prod[:, 1]
            + k8 * self.cd3
            + k9 * self.cd28
            + k11 * IL2 * IL12_eff
            + k12 * IFNg       # Positive feedback
            - k14 * IL4        # IL4 inhibits differentiation (not * IFNg)
            - k_deg[:, 1] * IFNg
        )
        
        # IL-21: Only basal production; IL2 inhibits differentiation
        dIL21 = (
            k_prod[:, 2]
            - k20 * IL2        # IL2 inhibits differentiation (not * IL21)
            - k_deg[:, 2] * IL21
        )
        
        # IL-4: CD28, IL2, IL21, self-feedback promote; IL21, IFNg, IL12 inhibit differentiation
        dIL4 = (
            k_prod[:, 3]
            + k22                # Basal
            + k24 * self.cd28
            + k26 * IL2
            + k28 * IL21
            + k30 * IL4          # Positive feedback
            - k29 * IL21         # IL21 inhibits differentiation (not * IL4)
            - k31 * IFNg         # IFNg inhibits differentiation (not * IL4)
            - k32 * IL12_eff     # IL12 inhibits differentiation (not * IL4)
            - k_deg[:, 3] * IL4
        )
        
        return torch.stack([dIL2, dIFNg, dIL21, dIL4], dim=1)
    
    @staticmethod
    def get_param_bounds(device):
        """Parameter bounds for 28 parameters."""
        lb = torch.zeros(28)
        ub = torch.zeros(28)
        
        # Basal production [0:4]
        lb[0:4] = -4.0
        ub[0:4] = 2.0
        
        # Basal degradation [4:8]
        lb[4:8] = -4.0
        ub[4:8] = 1.0
        
        # Interaction k values [8:27]: 0.0001 to 100
        lb[8:27] = -9.2
        ub[8:27] = 4.6
        
        # K_IL4inh [27]
        lb[27] = -4.0
        ub[27] = 0.0
        
        return lb.to(device), ub.to(device)
    
    @staticmethod
    def get_param_names():
        """Ordered list of 28 parameter names."""
        return (
            # Basal rates (8)
            ['log_k_prod_IL2', 'log_k_prod_IFNg', 'log_k_prod_IL21', 'log_k_prod_IL4'] +
            ['log_k_deg_IL2', 'log_k_deg_IFNg', 'log_k_deg_IL21', 'log_k_deg_IL4'] +
            # IL2 interactions (5)
            ['log_k1', 'log_k3', 'log_k5', 'log_k6', 'log_k7'] +
            # IFNg interactions (5)
            ['log_k8', 'log_k9', 'log_k11', 'log_k12', 'log_k14'] +
            # IL21 interactions (1)
            ['log_k20'] +
            # IL4 interactions (8)
            ['log_k22', 'log_k24', 'log_k26', 'log_k28', 'log_k29', 'log_k30', 'log_k31', 'log_k32'] +
            # IL4 inhibition of IL12 (1)
            ['log_K_IL4inh']
        )
    
    @staticmethod
    def params_to_model(positions, model):
        """Map PSO positions to model parameters."""
        idx = 0
        with torch.no_grad():
            # Basal rates
            model.log_k_prod.data = positions[:, idx:idx+4]; idx += 4
            model.log_k_deg.data = positions[:, idx:idx+4]; idx += 4
            
            # IL2: k1, k3, k5, k6, k7
            model.log_k1.data = positions[:, idx]; idx += 1
            model.log_k3.data = positions[:, idx]; idx += 1
            model.log_k5.data = positions[:, idx]; idx += 1
            model.log_k6.data = positions[:, idx]; idx += 1
            model.log_k7.data = positions[:, idx]; idx += 1
            
            # IFNg: k8, k9, k11, k12, k14
            model.log_k8.data = positions[:, idx]; idx += 1
            model.log_k9.data = positions[:, idx]; idx += 1
            model.log_k11.data = positions[:, idx]; idx += 1
            model.log_k12.data = positions[:, idx]; idx += 1
            model.log_k14.data = positions[:, idx]; idx += 1
            
            # IL21: k20
            model.log_k20.data = positions[:, idx]; idx += 1
            
            # IL4: k22, k24, k26, k28, k29, k30, k31, k32
            model.log_k22.data = positions[:, idx]; idx += 1
            model.log_k24.data = positions[:, idx]; idx += 1
            model.log_k26.data = positions[:, idx]; idx += 1
            model.log_k28.data = positions[:, idx]; idx += 1
            model.log_k29.data = positions[:, idx]; idx += 1
            model.log_k30.data = positions[:, idx]; idx += 1
            model.log_k31.data = positions[:, idx]; idx += 1
            model.log_k32.data = positions[:, idx]; idx += 1
            
            # K_IL4inh
            model.log_K_IL4inh.data = positions[:, idx]; idx += 1
        
        assert idx == 28, f"Expected 28 parameters, got {idx}"