"""
sats/training/attention_module.py

SATS Self-Attention 모듈.

논문 근거 (Note S3)
-------------------
e_{ij}  = a([W·h_i ∥ W·h_j]),  ∀ j ∈ N_i          (S4)
α_{ij}  = softmax_j(LeakyReLU(e_{ij}))               (S5)
h'_i    = ELU(Σ_{j∈N_i} α_{ij} · W·h_j)             (S6)

센서 레이아웃 (4×4 grid, 6.5mm 간격)
--------------------------------------
인덱스 규칙: sensor i → row = i // 4,  col = 3 - (i % 4)

  col→  0      1      2      3        (x 방향: -9.75 → +9.75)
row 0:  S4     S3     S2     S1    (y=-9.75mm)
row 1:  S8     S7     S6     S5    (y=-3.25mm)
row 2: S12    S11    S10     S9    (y=+3.25mm)
row 3: S16    S15    S14    S13    (y=+9.75mm)

* S1~S4는 x 내림차순(S1=+9.75mm, S4=-9.75mm)이므로 col = 3-(i%4)
* 인접 행렬은 상대 거리 기반이므로 col 방향 반전이 연결 구조에 영향 없음
8-connected 인접 + self-loop: max(|dr|, |dc|) ≤ 1
  · 코너 센서: 4개 이웃
  · 엣지 센서: 6개 이웃
  · 내부 센서: 9개 이웃
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SATSConfig
from .lstm_module import SensorLSTMEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 인접 행렬
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency_4x4(n_sensors: int = 16) -> torch.Tensor:
    """
    4×4 그리드의 8-connected 인접 행렬(self-loop 포함)을 반환한다.

    Returns
    -------
    adj : BoolTensor[n_sensors, n_sensors]
    """
    rows = torch.arange(n_sensors) // 4   # [16]
    cols = torch.arange(n_sensors) % 4    # [16]

    # 브로드캐스팅으로 모든 쌍(i, j)의 행·열 차이 계산
    dr = (rows.unsqueeze(1) - rows.unsqueeze(0)).abs()   # [16, 16]
    dc = (cols.unsqueeze(1) - cols.unsqueeze(0)).abs()   # [16, 16]

    adj = torch.maximum(dr, dc) <= 1   # Chebyshev distance ≤ 1
    return adj.bool()


# ─────────────────────────────────────────────────────────────────────────────
# Self-Attention 모듈
# ─────────────────────────────────────────────────────────────────────────────

class SATSSelfAttention(nn.Module):
    """
    Graph Attention Network(GAT) 방식의 Self-Attention.

    4×4 센서 그리드에서 인접 센서 간 정보를 집계한다.

    Parameters
    ----------
    in_dim    : LSTM 인코더 출력 차원 (= hidden_dim or hidden_dim×2)
    attn_dim  : 선형 투영 차원 (W: in_dim → attn_dim)
    n_sensors : 센서 수 (기본 16)
    leaky_slope : LeakyReLU negative slope

    Input
    -----
    local_feat : Tensor[B, n_sensors, in_dim]

    Output
    ------
    agg_feat : Tensor[B, n_sensors, attn_dim]
    """

    def __init__(
        self,
        in_dim: int,
        attn_dim: int = 64,
        n_sensors: int = 16,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.n = n_sensors
        self.attn_dim = attn_dim

        # W: 선형 투영 (편향 없음 — 논문 표준 GAT)
        self.W = nn.Linear(in_dim, attn_dim, bias=False)
        # a: 연결된 feature → scalar attention score
        self.a = nn.Linear(2 * attn_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.elu = nn.ELU()

        # 인접 행렬 — 학습 파라미터 아님
        adj = build_adjacency_4x4(n_sensors)
        self.register_buffer("adj", adj)   # [n, n] bool

    # ─────────────────────────────────────────────────────────────────────

    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        local_feat : [B, n, in_dim]

        Returns
        -------
        agg_feat : [B, n, attn_dim]
        """
        B, n, _ = local_feat.shape

        # 1. 선형 투영: Wh [B, n, attn_dim]
        Wh = self.W(local_feat)

        # 2. 모든 쌍(i, j)에 대한 attention score 계산 — 벡터화
        #    Wh_i [B, n, 1, attn_dim], Wh_j [B, 1, n, attn_dim]
        #    브로드캐스팅으로 [B, n, n, attn_dim] 생성
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, n, -1)   # [B, n, n, attn_dim]
        Wh_j = Wh.unsqueeze(1).expand(-1, n, -1, -1)   # [B, n, n, attn_dim]

        # e [B, n, n]: e_ij = a([Wh_i ∥ Wh_j])
        e = self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)   # [B, n, n]

        # 3. LeakyReLU 적용 후 비인접 위치를 -inf 마스킹
        e = self.leaky_relu(e)
        mask = self.adj.unsqueeze(0).expand(B, -1, -1)    # [B, n, n]
        e = e.masked_fill(~mask, float("-inf"))

        # 4. Softmax 정규화 → α [B, n, n]
        alpha = F.softmax(e, dim=-1)

        # 5. 집계: h'_i = ELU(Σ_j α_ij · Wh_j)
        #    [B, n, n] × [B, n, attn_dim] → [B, n, attn_dim]
        h_agg = torch.bmm(alpha, Wh)   # [B, n, attn_dim]
        return self.elu(h_agg)

    # ─────────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in_dim→attn_dim={self.W.in_features}→{self.attn_dim}, "
            f"n_sensors={self.n}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 임시 프록시 디코더 (Self-Attention 단독 학습용)
# ─────────────────────────────────────────────────────────────────────────────

class _AttentionProxyDecoder(nn.Module):
    """
    [B, n, combined_dim] → flatten → MLP → [B, 40, 40]

    combined_dim = lstm_out_dim + attn_dim  (concat of local + agg features)

    이 디코더는 Self-Attention 학습 단계에서만 사용되며,
    이후 Local Map 모듈로 교체된다.
    """

    def __init__(
        self,
        n_sensors: int,
        combined_dim: int,
        grid_size: int = 40,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        in_features = n_sensors * combined_dim

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, grid_size * grid_size),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        combined : [B, n_sensors, combined_dim]

        Returns
        -------
        pred_map : [B, grid_size, grid_size]
        """
        B = combined.size(0)
        x = combined.reshape(B, -1)
        return self.head(x).view(B, self.grid_size, self.grid_size)


# ─────────────────────────────────────────────────────────────────────────────
# 학습용 통합 모듈
# ─────────────────────────────────────────────────────────────────────────────

class SATSAttentionStage(nn.Module):
    """
    LSTM 인코더 + Self-Attention + 임시 디코더.

    학습 전략
    ----------
    train_attention.py에서 LSTM 인코더 가중치를 로드 후 동결,
    Self-Attention + 디코더만 학습한다.

    Input
    -----
    sensor_seq : [B, T, 16]
    lengths    : [B]

    Output
    ------
    pred_map  : [B, 40, 40]
    agg_feat  : [B, 16, attn_dim]   ← 다음 단계(Local Map) 연계용
    """

    def __init__(self, cfg: SATSConfig) -> None:
        super().__init__()

        self.encoder = SensorLSTMEncoder(
            n_sensors    = cfg.n_sensors,
            hidden_dim   = cfg.hidden_dim,
            num_layers   = cfg.num_layers,
            dropout      = cfg.dropout,
            bidirectional = cfg.bidirectional,
        )
        self.attention = SATSSelfAttention(
            in_dim    = self.encoder.out_dim,
            attn_dim  = cfg.attn_dim,
            n_sensors = cfg.n_sensors,
        )
        combined_dim = self.encoder.out_dim + cfg.attn_dim
        self.decoder = _AttentionProxyDecoder(
            n_sensors    = cfg.n_sensors,
            combined_dim = combined_dim,
            grid_size    = cfg.grid_size,
        )

    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        sensor_seq: torch.Tensor,   # [B, T, 16]
        lengths: torch.Tensor,      # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_feat = self.encoder(sensor_seq, lengths)          # [B, 16, lstm_out]
        agg_feat   = self.attention(local_feat)                 # [B, 16, attn_dim]
        combined   = torch.cat([local_feat, agg_feat], dim=-1) # [B, 16, combined]
        pred_map   = self.decoder(combined)                     # [B, 40, 40]
        return pred_map, agg_feat
