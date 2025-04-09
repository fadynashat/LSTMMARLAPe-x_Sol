import torch
import torch.nn as nn
from mamba_ssm import Mamba

class TemporalFusion(nn.Module):
    def __init__(self, input_dim=23, lstm_dim=128, mamba_dim=64):
        super().__init__()
        # BiLSTM Pathway
        self.lstm = nn.LSTM(input_dim, lstm_dim//2, num_layers=4, 
                           bidirectional=True, dropout=0.1)
        
        # Mamba SSM Pathway
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=64,
            d_conv=4,
            expand=2
        )
        self.mamba_proj = nn.Linear(input_dim, mamba_dim)
        
        # Dynamic Fusion Gate
        self.gate = nn.Sequential(
            nn.LayerNorm(lstm_dim + mamba_dim),
            nn.Linear(lstm_dim + mamba_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(x)  # (seq_len, batch, lstm_dim)
        
        mamba_proj = self.mamba_proj(x)  # (seq_len, batch, mamba_dim)
        mamba_out = self.mamba(mamba_proj)
        
        # Dynamic fusion
        combined = torch.cat([lstm_out, mamba_out], dim=-1)
        gate_weight = self.gate(combined)
        output = gate_weight * lstm_out + (1 - gate_weight) * mamba_out
        
        return output