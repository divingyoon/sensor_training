"""
cnnlstm_sr.py

CNN-LSTM SR 모델.
  - 깊이(z) 방향 시계열 처리
  - Per-frame: 4×4 grid CNN → 64-dim feature
  - LSTM: z 방향 temporal dependency 학습
  - Many-to-many: 각 timestep마다 (x, y, z, fz) 예측

Input:
  s14  : (B, T, 14) - T 프레임의 live channel s_norm
  diam : (B, T, 1)  - diameter_norm (시퀀스 전체 동일값)

Output:
  (B, T, 4) - 각 timestep의 [x_mm, y_mm, z_depth_mm, fz]
"""

import torch
import torch.nn as nn



class _FrameEncoder(nn.Module):
    """단일 프레임 (B*T, 1, 4, 4) → (B*T, 64)"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B,16,4,4)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B,32,4,4)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),                       # (B,32,2,2)
        )
        self.fc = nn.Linear(32 * 4, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))  # (B, 64)


class CNNLSTMSR(nn.Module):
    """
    CNN-LSTM for depth-axis sequence regression.

    Args:
        lstm_hidden : LSTM 은닉 차원
        lstm_layers : LSTM 레이어 수
        dropout     : LSTM dropout (layers > 1일 때만 적용)
        out_dim     : 출력 차원 (4)
    """

    def __init__(
        self,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        out_dim: int = 4,
    ):
        super().__init__()
        self.encoder = _FrameEncoder()
        # CNN 64-dim + diameter 1-dim → LSTM input 64-dim
        self.proj = nn.Linear(64 + 1, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_hidden, out_dim)

    def _to_grid(self, s16_flat: torch.Tensor) -> torch.Tensor:
        """s16_flat: (N, 16) → (N, 1, 4, 4)"""
        return s16_flat.view(-1, 1, 4, 4)

    def forward(self, s16: torch.Tensor, diam: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s16  : (B, T, 16)
            diam : (B, T, 1)
        Returns:
            (B, T, 4)
        """
        B, T, _ = s16.shape

        # Per-frame CNN encoding
        grid = self._to_grid(s16.reshape(B * T, 16))  # (B*T, 1, 4, 4)
        feat = self.encoder(grid)                       # (B*T, 64)
        feat = feat.view(B, T, 64)

        # Concat diameter, project
        feat = torch.cat([feat, diam], dim=-1)          # (B, T, 65)
        feat = self.proj(feat)                           # (B, T, 64)

        # LSTM
        lstm_out, _ = self.lstm(feat)                   # (B, T, hidden)
        return self.head(lstm_out)                       # (B, T, 4)
