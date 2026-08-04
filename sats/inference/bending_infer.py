"""실시간 밴딩 추론 — estimator(theta) + restorer(flat 복원). 버클 방식(D).

★입력 규약(중요): estimator 는 **원시 센서 윈도우를 표준화**해 받는다(pct 아님).
실시간 리더는 pct(=(raw−base)/base×100)를 주므로, flat baseline 으로 raw 를 복원:
  raw = base × (1 + pct/100).
restorer 는 pct(+theta) 를 받아 flat 등가 pct 를 돌려준다(동결 SATS 입력).

밴딩 방식은 v6 buckling(Y구동 δ=Y−18)로 estimator/restorer 학습됨. 순수 jig-bend 는
신호가 약해 estimator 포화 → 데모는 buckling 지그 전제([[v6-test-eval]]).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sats.bending.config import BendingConfig
from sats.bending.train_bending import load_estimator


def pct_to_raw(pct_window: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """pct 윈도우[W,16] + flat baseline[16] → 원시 센서 윈도우[W,16] 복원."""
    return (baseline[None, :] * (1.0 + pct_window / 100.0)).astype(np.float32)


class BendingInference:
    """estimator theta + (선택) restorer 복원. window 크기는 estimator cfg 기준."""

    def __init__(self, estimator_ckpt: str | Path, device: str = "cpu",
                 restorer=None, cfg: BendingConfig | None = None) -> None:
        self.device = device
        self.est, self.stats = load_estimator(estimator_ckpt, device=device)
        self.cfg = cfg or BendingConfig()
        self.W = self.cfg.window_size
        self._mean = torch.tensor(np.asarray(self.stats.mean), device=device)
        self._std = torch.tensor(np.asarray(self.stats.std), device=device)
        self.restorer = restorer

    @torch.no_grad()
    def theta_from_raw(self, raw_window: np.ndarray) -> float:
        """원시 윈도우[W,16] → 밴딩각 theta(deg, signed)."""
        std = (torch.from_numpy(raw_window.astype(np.float32)).to(self.device) - self._mean) / self._std
        L = torch.full((1,), std.shape[0], dtype=torch.long, device=self.device)
        deg = self.est(std.unsqueeze(0), L)
        return float(deg.reshape(-1)[0].cpu())

    @torch.no_grad()
    def restore(self, pct_window: np.ndarray, theta_deg: float) -> np.ndarray:
        """pct 윈도우[W,16] + theta → flat 등가 pct[W,16] (restorer 없으면 그대로)."""
        if self.restorer is None:
            return pct_window.astype(np.float32)
        x = torch.from_numpy(pct_window.astype(np.float32)).to(self.device).unsqueeze(0)
        deg = torch.full((1,), float(theta_deg), device=self.device)
        return self.restorer(x, deg)[0].cpu().numpy().astype(np.float32)


def load_restorer(sats_run: str | Path, bending_dir: str | Path,
                  device: str = "cpu", cfg: BendingConfig | None = None,
                  epochs: int = 120):
    """v6 buckling 밴딩으로 restorer 학습(데모 셋업 시 1회, ~1분)."""
    from sats.bending.eval_pipeline import _train_pct_restorer
    from sats.bending.pipeline import load_frozen_sats
    cfg = cfg or BendingConfig()
    sats = load_frozen_sats(sats_run, device)
    trials = sorted(p.stem for p in Path(bending_dir).glob("*.npz"))
    if not trials:
        raise FileNotFoundError(f"밴딩 npz 없음: {bending_dir}")
    return _train_pct_restorer(sats, Path(bending_dir), trials, cfg, device, epochs, 1e-3)
