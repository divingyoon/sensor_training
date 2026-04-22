"""
sats/training/cnn_module.py

SATS CNN Refining 모듈.

논문 근거 (Note S3 / "The DNN models" 섹션)
-------------------------------------------
Local Map Construction 이후 전체 merged map에
3×3 커널 2층 CNN을 적용하여 더 부드럽고 정밀한 압력 분포를 생성한다.

  Layer 1: Conv2d(1 → C, 3×3, padding=1) + LeakyReLU
  Layer 2: Conv2d(C → 1, 3×3, padding=1)  (activation 없음)

입력  : merged_map [B, H, W]  (Local Map Decoder 출력)
출력  : refined_map [B, H, W] (같은 공간 크기 유지)

학습 전략
----------
train_cnn.py에서 SATSLocalMapStage 체크포인트로부터
encoder + attention + local_map_decoder 가중치를 로드 후 동결,
cnn_refiner만 학습한다.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import SATSConfig
from .lstm_module import SensorLSTMEncoder
from .attention_module import SATSSelfAttention
from .local_map_module import SATSLocalMapDecoder


# ─────────────────────────────────────────────────────────────────────────────
# SATSCNNRefiner — 2층 CNN 정제 모듈
# ─────────────────────────────────────────────────────────────────────────────

class SATSCNNRefiner(nn.Module):
    """
    2층 CNN으로 merged_map을 정제한다.

    Architecture (논문 직역)
    -------------------------
    Conv2d(1 → hidden_channels, 3×3, padding=1) + LeakyReLU(0.2)
    Conv2d(hidden_channels → 1, 3×3, padding=1)

    Parameters
    ----------
    grid_size       : 전체 맵 한 변 크기 (기본 40) — 현재는 메타데이터용
    hidden_channels : 중간 채널 수 (기본 16; 논문 미명시)
    """

    def __init__(self, grid_size: int = 40, hidden_channels: int = 16) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.hidden_channels = hidden_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, merged_map: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        merged_map : [B, H, W]

        Returns
        -------
        refined_map : [B, H, W]
        """
        x = merged_map.unsqueeze(1)          # [B, 1, H, W]
        out = self.net(x)                    # [B, 1, H, W]
        return out.squeeze(1)                # [B, H, W]

    def extra_repr(self) -> str:
        return (
            f"grid_size={self.grid_size}, "
            f"hidden_channels={self.hidden_channels}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SATSCNNStage — 전체 파이프라인 통합 모듈
# ─────────────────────────────────────────────────────────────────────────────

class SATSCNNStage(nn.Module):
    """
    LSTM 인코더 + Self-Attention + Local Map Decoder + CNN Refiner.

    학습 전략
    ----------
    train_cnn.py에서 SATSLocalMapStage 체크포인트의
    encoder + attention + local_map_decoder 가중치를 로드 후 동결,
    cnn_refiner만 학습한다.

    Input
    -----
    sensor_seq : [B, T, 16]
    lengths    : [B]

    Output
    ------
    refined_map  : [B, 40, 40]   ← MSE 손실 대상
    merged_map   : [B, 40, 40]   ← CNN 이전 맵 (디버깅/보조 손실용)
    """

    def __init__(self, cfg: SATSConfig) -> None:
        super().__init__()

        self.encoder = SensorLSTMEncoder(
            n_sensors     = cfg.n_sensors,
            hidden_dim    = cfg.hidden_dim,
            num_layers    = cfg.num_layers,
            dropout       = cfg.dropout,
            bidirectional = cfg.bidirectional,
        )
        self.attention = SATSSelfAttention(
            in_dim    = self.encoder.out_dim,
            attn_dim  = cfg.attn_dim,
            n_sensors = cfg.n_sensors,
        )
        combined_dim = self.encoder.out_dim + cfg.attn_dim
        self.local_map_decoder = SATSLocalMapDecoder(
            combined_dim      = combined_dim,
            local_map_size    = cfg.local_map_size,
            grid_size         = cfg.grid_size,
            n_sensors         = cfg.n_sensors,
            grid_min_mm       = cfg.grid_min_mm,
            sensor_spacing_mm = cfg.sensor_spacing_mm,
            grid_step_mm      = cfg.grid_step_mm,
        )
        self.cnn_refiner = SATSCNNRefiner(
            grid_size       = cfg.grid_size,
            hidden_channels = cfg.cnn_hidden_channels,
        )

    def forward(
        self,
        sensor_seq: torch.Tensor,   # [B, T, 16]
        lengths: torch.Tensor,      # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_feat    = self.encoder(sensor_seq, lengths)            # [B, 16, lstm_out]
        agg_feat      = self.attention(local_feat)                   # [B, 16, attn_dim]
        combined_feat = torch.cat([local_feat, agg_feat], dim=-1)   # [B, 16, combined]
        merged_map    = self.local_map_decoder(combined_feat)        # [B, 40, 40]
        refined_map   = self.cnn_refiner(merged_map)                 # [B, 40, 40]
        return refined_map, merged_map
