"""BaselineRestorer — 밴딩 유발 오프셋을 예측해 flat 등가 신호를 복원.

가정: 밴딩 상태 신호 ≈ flat 접촉 신호 + 밴딩 오프셋(곡률 의존). 곡률(signed deg)을
조건으로 오프셋을 예측하고 빼서 flat 등가 신호를 만든다. 동결 SATS 입력 분포와 정합.

★ 두 가지 모드(`cfg.restorer_mode`):
- **deg_only(기본)**: 오프셋 = f(deg) 만의 함수(입력 seq 비의존). §5.3 Δp_bend≈κ·k_i z_i =
  곡률×고정 공간패턴. 입력에 안 의존하므로 **접촉을 건드릴 수 없음(접촉 보존)**.
- seq_deg(레거시): (seq+deg) MLP. 표현력↑이나 무접촉 학습 시 offset≈seq 로 오학습해
  **접촉을 파괴**함이 준-합성 검증에서 드러남(eval_contact_preservation).

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
        self.mode = getattr(cfg, "restorer_mode", "deg_only")
        if self.mode not in ("deg_only", "seq_deg"):
            raise ValueError(f"restorer_mode must be deg_only/seq_deg, got {self.mode!r}")
        in_dim = 1 if self.mode == "deg_only" else cfg.n_sensors + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, cfg.n_sensors),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, seq: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """[B, T, 16] + signed deg[B] → flat 등가 [B, T, 16] = seq − 오프셋."""
        b, t, _ = seq.shape
        deg_n = deg / float(self.cfg.deg_scale)
        if self.mode == "deg_only":
            offset = self.net(deg_n.view(b, 1)).unsqueeze(1)   # [B,1,16] broadcast (seq 비의존)
        else:
            dg = deg_n.view(b, 1, 1).expand(b, t, 1)
            offset = self.net(torch.cat([seq, dg], dim=-1))    # [B, T, 16]
        return seq - offset
