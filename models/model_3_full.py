# models/model_full.py

import torch
import torch.nn as nn
from .base import BaseTcellModel
from ..constants import IL12_0


class FullModel_3_SD(BaseTcellModel):
    """
    Full 59-parameter model with Hill functions and all interactions.
    """
    
    model_name = "Full Model (59 params)"
    
    def __init__(self, n_batch: int = 1, device: torch.device = torch.device("cpu")):
        super().__init__(n_batch, device)
        
        # Basal rates (8)
        self.log_k_prod = nn.Parameter(torch.zeros(n_batch, 4, device=device))
        self.log_k_deg = nn.Parameter(torch.zeros(n_batch, 4, device=device))
        
        # IL-2 equation: k1-k7
        self.log_k1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k2 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k3 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k4 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k5 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k6 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k7 = nn.Parameter(torch.zeros(n_batch, device=device))
        
        # IFNg equation: k8-k14
        self.log_k8 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k9 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k10 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k11 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k12 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k13 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k14 = nn.Parameter(torch.zeros(n_batch, device=device))
        
        # IL-21 equation: k15-k21
        self.log_k15 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k16 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k17 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k18 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k19 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k20 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k21 = nn.Parameter(torch.zeros(n_batch, device=device))
        
        # IL-4 equation: k22-k32
        self.log_k22 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k23 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k24 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k25 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k26 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k27 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k28 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k29 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k30 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k31 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_k32 = nn.Parameter(torch.zeros(n_batch, device=device))
        
        # Hill K thresholds (9)
        self.log_K10 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_K12 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_K17 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_K18 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_K21 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_K22 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_K25 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_K27 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_K30 = nn.Parameter(torch.zeros(n_batch, device=device))
        
        # Hill h coefficients (9)
        self.log_h10_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_h12_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_h17_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_h18_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_h21_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_h22_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_h25_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_h27_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        self.log_h30_minus1 = nn.Parameter(torch.zeros(n_batch, device=device))
        
        # IL-4 inhibition (1)
        self.log_K_IL4inh = nn.Parameter(torch.zeros(n_batch, device=device))
    
    @property
    def n_params(self) -> int:
        return 59
    
    def hill_activation(self, X, K, h):
        X_safe = X.clamp(min=1e-8)
        K_safe = K.clamp(min=1e-8)
        X_h = X_safe.pow(h)
        K_h = K_safe.pow(h)
        return X_h / (K_h + X_h)
    
    def hill_inhibition(self, X, K, h):
        X_safe = X.clamp(min=1e-8)
        K_safe = K.clamp(min=1e-8)
        X_h = X_safe.pow(h)
        K_h = K_safe.pow(h)
        return K_h / (K_h + X_h)
    
    def il12_effective(self, IL4):
        K_IL4inh = torch.exp(self.log_K_IL4inh)
        return IL12_0 / (1 + K_IL4inh * IL4)
    
    def forward(self, t, y):
        IL2, IFNg, IL21, IL4 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        
        k_prod = torch.exp(self.log_k_prod)
        k_deg = torch.exp(self.log_k_deg)
        
        k1 = torch.exp(self.log_k1)
        k2 = torch.exp(self.log_k2)
        k3 = torch.exp(self.log_k3)
        k4 = torch.exp(self.log_k4)
        k5 = torch.exp(self.log_k5)
        k6 = torch.exp(self.log_k6)
        k7 = torch.exp(self.log_k7)
        k8 = torch.exp(self.log_k8)
        k9 = torch.exp(self.log_k9)
        k10 = torch.exp(self.log_k10)
        k11 = torch.exp(self.log_k11)
        k12 = torch.exp(self.log_k12)
        k13 = torch.exp(self.log_k13)
        k14 = torch.exp(self.log_k14)
        k15 = torch.exp(self.log_k15)
        k16 = torch.exp(self.log_k16)
        k17 = torch.exp(self.log_k17)
        k18 = torch.exp(self.log_k18)
        k19 = torch.exp(self.log_k19)
        k20 = torch.exp(self.log_k20)
        k21 = torch.exp(self.log_k21)
        k22 = torch.exp(self.log_k22)
        k23 = torch.exp(self.log_k23)
        k24 = torch.exp(self.log_k24)
        k25 = torch.exp(self.log_k25)
        k26 = torch.exp(self.log_k26)
        k27 = torch.exp(self.log_k27)
        k28 = torch.exp(self.log_k28)
        k29 = torch.exp(self.log_k29)
        k30 = torch.exp(self.log_k30)
        k31 = torch.exp(self.log_k31)
        k32 = torch.exp(self.log_k32)
        
        K10 = torch.exp(self.log_K10)
        K12 = torch.exp(self.log_K12)
        K17 = torch.exp(self.log_K17)
        K18 = torch.exp(self.log_K18)
        K21 = torch.exp(self.log_K21)
        K22 = torch.exp(self.log_K22)
        K25 = torch.exp(self.log_K25)
        K27 = torch.exp(self.log_K27)
        K30 = torch.exp(self.log_K30)
        
        h10 = 1 + torch.exp(self.log_h10_minus1)
        h12 = 1 + torch.exp(self.log_h12_minus1)
        h17 = 1 + torch.exp(self.log_h17_minus1)
        h18 = 1 + torch.exp(self.log_h18_minus1)
        h21 = 1 + torch.exp(self.log_h21_minus1)
        h22 = 1 + torch.exp(self.log_h22_minus1)
        h25 = 1 + torch.exp(self.log_h25_minus1)
        h27 = 1 + torch.exp(self.log_h27_minus1)
        h30 = 1 + torch.exp(self.log_h30_minus1)
        
        IL12_eff = self.il12_effective(IL4)
        
        cd3_t = torch.full((self.n_batch,), self.cd3, device=self.device)
        cd28_t = torch.full((self.n_batch,), self.cd28, device=self.device)
        
        H_inh_CD28_k10 = self.hill_inhibition(cd28_t, K10, h10)
        H_act_CD3_k12 = self.hill_activation(cd3_t, K12, h12)
        H_inh_CD28_k17 = self.hill_inhibition(cd28_t, K17, h17)
        H_act_IL21_k18 = self.hill_activation(IL21, K18, h18)
        H_inh_CD3_k21 = self.hill_inhibition(cd3_t, K21, h21)
        H_inh_CD3_k22 = self.hill_inhibition(cd3_t, K22, h22)
        H_inh_IL2_k25 = self.hill_inhibition(IL2, K25, h25)
        H_act_CD28_k27 = self.hill_activation(cd28_t, K27, h27)
        H_inh_CD3_k30 = self.hill_inhibition(cd3_t, K30, h30)
        
        dIL2 = (
            k_prod[:, 0]
            + k1 * self.cd3 + k3 * self.cd28 + k4 * IL21 + k7 * IL12_eff
            - k2 * self.cd3 * IL2 - k5 * IL21 * IL2 - k6 * IL4 * IL2
            - k_deg[:, 0] * IL2
        )
        
        dIFNg = (
            k_prod[:, 1]
            + k8 * self.cd3 + k9 * self.cd28
            + k10 * IL2 * H_inh_CD28_k10 + k11 * IL2 * IL12_eff
            + k12 * IFNg * H_act_CD3_k12 + k13 * IL12_eff
            - k14 * IL4 * IFNg
            - k_deg[:, 1] * IFNg
        )
        
        dIL21 = (
            k_prod[:, 2]
            + k15 * self.cd3 + k16 * self.cd28
            + k17 * IL2 * H_inh_CD28_k17 + k18 * H_act_IL21_k18 + k19 * IL4
            - k20 * IL2 * IL21 - k21 * IL4 * IL21 * H_inh_CD3_k21
            - k_deg[:, 2] * IL21
        )
        
        dIL4 = (
            k_prod[:, 3]
            + k22 * H_inh_CD3_k22 + k24 * self.cd28
            + k25 * self.cd28 * H_inh_IL2_k25 + k26 * IL2 + k28 * IL21
            + k30 * IL4 * H_inh_CD3_k30
            - k23 * self.cd3 * IL4 - k27 * IL2 * IL4 * H_act_CD28_k27
            - k29 * IL21 * IL4 - k31 * IFNg * IL4 - k32 * IL12_eff * IL4
            - k_deg[:, 3] * IL4
        )
        
        return torch.stack([dIL2, dIFNg, dIL21, dIL4], dim=1)
    
    @staticmethod
    def get_param_bounds(device):
        lb = torch.zeros(59)
        ub = torch.zeros(59)
        
        lb[0:4] = -4.0;   ub[0:4] = 2.0    # production
        lb[4:8] = -4.0;   ub[4:8] = 1.0    # degradation
        lb[8:40] = -9.2;  ub[8:40] = 4.6   # interaction k (0.0001 to 100)
        lb[40:49] = -2.0; ub[40:49] = 3.0  # Hill K
        lb[49:58] = -2.0; ub[49:58] = 1.5  # Hill h
        lb[58] = -4.0;    ub[58] = 0.0     # IL4 inhibition
        
        return lb.to(device), ub.to(device)
    
    @staticmethod
    def get_param_names():
        return (
            [f'log_k_prod_{i}' for i in ['IL2', 'IFNg', 'IL21', 'IL4']] +
            [f'log_k_deg_{i}' for i in ['IL2', 'IFNg', 'IL21', 'IL4']] +
            [f'log_k{i}' for i in range(1, 33)] +
            ['log_K10', 'log_K12', 'log_K17', 'log_K18', 'log_K21', 'log_K22', 'log_K25', 'log_K27', 'log_K30'] +
            ['log_h10_m1', 'log_h12_m1', 'log_h17_m1', 'log_h18_m1', 'log_h21_m1', 'log_h22_m1', 'log_h25_m1', 'log_h27_m1', 'log_h30_m1'] +
            ['log_K_IL4inh']
        )
    
    @staticmethod
    def params_to_model(positions, model):
        idx = 0
        with torch.no_grad():
            model.log_k_prod.data = positions[:, idx:idx+4]; idx += 4
            model.log_k_deg.data = positions[:, idx:idx+4]; idx += 4
            for i in range(1, 33):
                getattr(model, f'log_k{i}').data = positions[:, idx]; idx += 1
            for name in ['K10', 'K12', 'K17', 'K18', 'K21', 'K22', 'K25', 'K27', 'K30']:
                getattr(model, f'log_{name}').data = positions[:, idx]; idx += 1
            for name in ['h10', 'h12', 'h17', 'h18', 'h21', 'h22', 'h25', 'h27', 'h30']:
                getattr(model, f'log_{name}_minus1').data = positions[:, idx]; idx += 1
            model.log_K_IL4inh.data = positions[:, idx]; idx += 1