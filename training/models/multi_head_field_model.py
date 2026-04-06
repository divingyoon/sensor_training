
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadFieldModel(nn.Module):
    """
    위치(xyz), 3축 힘(Fx, Fy, Fz), 그리고 연속적인 Force Field(Heatmap)를 동시에 학습하는 최종형 모델.
    """
    def __init__(self, seq_len=50):
        super().__init__()
        # Backbone (Spatial-Temporal)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(64*4*4, 128, batch_first=True)
        
        # Head 1: 3-Axis Force Vector at Contact Center [x, y, z, Fx, Fy, Fz]
        self.head_force = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Head 2: 2D Force Distribution Field (25x25)
        self.head_field = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 25 * 25),
            nn.Sigmoid()
        )

    def forward(self, grid_seq):
        # grid_seq: (B, T, 1, 4, 4)
        B, T, C, H, W = grid_seq.shape
        cnn_out = self.cnn(grid_seq.view(B * T, C, H, W)).view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_out)
        feat = lstm_out[:, -1, :] # Last hidden state
        
        force_vec = self.head_force(feat)
        field_map = self.head_field(feat).view(B, 1, 25, 25)
        
        return force_vec, field_map
