
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadFieldModel(nn.Module):
    """
    위치(xyz), 3축 힘(Fx, Fy, Fz), 그리고 2D heatmap(분포)을 동시에 예측하는 모델.
    - heatmap 출력은 logits로 내보내 BCEWithLogitsLoss에 바로 사용할 수 있게 함.
    """

    def __init__(self, seq_len: int = 50, heatmap_size: int = 40):
        super().__init__()
        self.heatmap_size = heatmap_size

        # Backbone (Spatial-Temporal)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(64 * 4 * 4, 128, batch_first=True)
        
        # Head 1: 3-Axis Force Vector at Contact Center [x, y, z, Fx, Fy, Fz]
        self.head_force = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Head 2: 2D Heatmap logits (e.g., 40x40 grid)
        self.head_field = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, heatmap_size * heatmap_size),
        )

    def forward(self, grid_seq):
        # grid_seq: (B, T, 1, 4, 4)
        B, T, C, H, W = grid_seq.shape
        cnn_out = self.cnn(grid_seq.view(B * T, C, H, W)).view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_out)
        feat = lstm_out[:, -1, :]  # Last hidden state
        
        force_vec = self.head_force(feat)
        field_map = self.head_field(feat).view(B, 1, self.heatmap_size, self.heatmap_size)
        
        return force_vec, field_map
