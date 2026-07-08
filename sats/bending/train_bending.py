"""밴딩 모듈 학습 드라이버 (Phase 0 스켈레톤).

2단계 (SATS는 항상 동결):
- Phase 1 — BendingEstimator: 밴딩 데이터 → signed deg 회귀(MSE/MAE). ★데이터 오면 실행 가능.
- Phase 2 — BaselineRestorer: flat 등가 복원. 학습 신호 2안(데이터 취득 방식에 따라 선택):
    (A) 오프셋 지도: bending-only(무접촉) 신호 = 순수 밴딩 오프셋 → restorer가 재현하도록
        학습 → bending+contact에서 빼면 contact-only(flat 등가). 중첩 선형성 가정.
    (B) end-to-end: 밴딩+접촉 → restorer → ❄️SATS → 압력맵 손실을 flat 기준 대비 최소화.
        SATS 동결이라 grad는 restorer까지만 전파(pipeline.forward 참조).
- Phase 3 — Pipeline 검증: 밴딩 하 SATS 정확도 vs flat 기준(재학습 0).

여기서는 Phase 1 estimator 학습을 제공(데이터 배열만 있으면 동작). Phase 2/3은 취득 후.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .bending_estimator import BendingEstimator
from .config import BendingConfig


def train_estimator(
    cfg: BendingConfig,
    windows: np.ndarray,      # [M, W, 16]
    degs: np.ndarray,         # [M] signed
    *,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> BendingEstimator:
    """BendingEstimator를 signed deg 회귀로 학습. 지표 = deg MAE."""
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = BendingEstimator(cfg).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(windows.astype(np.float32))
    y = torch.from_numpy(degs.astype(np.float32))
    lengths = torch.full((X.shape[0],), X.shape[1], dtype=torch.long)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            xb = X[b].to(device); yb = y[b].to(device); lb = lengths[b].to(device)
            pred = model(xb, lb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        with torch.no_grad():
            mae = (model(X.to(device), lengths.to(device)) - y.to(device)).abs().mean().item()
        print(f"[estimator] ep{ep + 1}/{epochs} mse={tot / n:.4f} deg_MAE={mae:.3f}")
    return model.eval()


# Phase 2/3 (restorer 학습·pipeline 검증)는 밴딩 데이터 취득 후 구현.
