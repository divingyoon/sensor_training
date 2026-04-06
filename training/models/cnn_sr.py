"""
cnn_sr.py

2D CNN SR 모델.
  - 16채널 → 4×4 그리드 직접 reshape
  - Conv2d 3층으로 공간 패턴 추출
  - diameter_norm concat 후 FC → 4 outputs

센서 그리드 레이아웃 (row-major, 0-indexed):
  [0,0]=Skin1  [0,1]=Skin2  [0,2]=Skin3   [0,3]=Skin4
  [1,0]=Skin5  [1,1]=Skin6  [1,2]=Skin7   [1,3]=Skin8
  [2,0]=Skin9  [2,1]=Skin10 [2,2]=Skin11  [2,3]=Skin12
  [3,0]=Skin13 [3,1]=Skin14 [3,2]=Skin15  [3,3]=Skin16
"""

import torch
import torch.nn as nn


class CNNSR(nn.Module):
    """
    2D CNN for tactile super-resolution.

    Args:
        out_dim: 출력 차원 (x, y, z, fz = 4)
    """

    def __init__(self, out_dim: int = 4):
        super().__init__()

        # 4×4 grid conv: (B,1,4,4) → (B,128,3,3)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,32,4,4)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,4,4)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2),             # (B,128,3,3)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # flatten(128*9=1152) + diam(1) → FC
        self.fc = nn.Sequential(
            nn.Linear(128 * 9 + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def _to_grid(self, s16: torch.Tensor) -> torch.Tensor:
        """(B, 16) → (B, 1, 4, 4)"""
        return s16.view(-1, 1, 4, 4)

    def forward(self, s16: torch.Tensor, diam: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s16  : (B, 16)
            diam : (B, 1)
        Returns:
            (B, 4)
        """
        grid = self._to_grid(s16)          # (B, 1, 4, 4)
        feat = self.conv(grid).flatten(1)  # (B, 1152)
        x    = torch.cat([feat, diam], dim=1)  # (B, 1153)
        return self.fc(x)
