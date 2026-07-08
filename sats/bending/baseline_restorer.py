"""BaselineRestorer — 밴딩 유발 오프셋을 예측해 flat 등가 신호를 복원.

가정: 밴딩 상태 신호 ≈ flat 접촉 신호 + 밴딩 오프셋(곡률 의존). 곡률(signed deg)을
조건으로 채널별·시점별 오프셋을 예측하고 빼서 flat 등가 신호를 만든다. 동결 SATS의
입력 분포(flat)와 맞추는 것이 목적.

zero-init 마지막 층 → 오프셋 0 → 복원=원신호(항등 웜스타트, 안전).
부호 있는 deg → 오프셋 방향이 밴딩 방향(양/음)에 따라 달라짐.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import BendingConfig


class BaselineRestorer(nn.Module):
    def __init__(self, cfg: BendingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.n_sensors + 1, cfg.mlp_hidden),   # +1 = 정규화 deg(부호 포함)
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, cfg.n_sensors),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, seq: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """[B, T, 16] + signed deg[B] → flat 등가 [B, T, 16] = seq − 오프셋."""
        b, t, _ = seq.shape
        deg_n = (deg / float(self.cfg.deg_scale)).view(b, 1, 1).expand(b, t, 1)
        x = torch.cat([seq, deg_n], dim=-1)             # [B, T, 17]
        offset = self.net(x)                            # [B, T, 16]
        return seq - offset
