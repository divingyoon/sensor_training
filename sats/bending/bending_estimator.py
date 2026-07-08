"""BendingEstimator — 센서 시계열에서 밴딩 곡률(signed deg)을 추정.

구조: 공유 LSTM(16채널 시계열 인코딩, 이력/드리프트 포착) → 마지막 유효 hidden →
MLP head → **부호 있는** 스칼라 deg. 양/음 방향을 모두 표현하도록 출력에 활성화 없음.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import BendingConfig


class BendingEstimator(nn.Module):
    def __init__(self, cfg: BendingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.n_sensors,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1),   # signed deg (활성화 없음)
        )

    def _last_hidden(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)                      # [B, T, H]
        idx = (lengths - 1).clamp(min=0)
        b = torch.arange(seq.shape[0], device=seq.device)
        return out[b, idx]                           # [B, H]

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """[B, T, 16] → signed deg [B]. deg_scale로 정규화된 회귀."""
        h = self._last_hidden(seq, lengths)
        deg = self.head(h).squeeze(-1) * float(self.cfg.deg_scale)
        return deg
