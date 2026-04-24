"""
SATS 학습 설정.

trial_id 명명 규칙: "ecomesh_d{D}_z{Z}_test{N}"
  - D : 인덴터 직경 (mm), 예) 5, 10
  - Z : z_max (mm), 예) 1 → 1.0mm, 1.5 → 1.5mm
  - N : 반복 번호

파일 경로 매핑:
  raw CSV   : raw_data/ecomesh/d{D}/z_{Z_full}mm/test{N}/ecomesh_d{D}_z{Z_full}_test{N}_merged.csv
  baseline  : raw_data/ecomesh/d{D}/z_{Z_full}mm/test{N}/ecomesh_d{D}_z{Z_full}_test{N}_baseline.json
  GT npy    : sats/preprocessing/gt_output_v1/{trial_id}_targets.npy
  GT index  : sats/preprocessing/gt_output_v1/dataset_index.json

baseline JSON 키: Skin1_mean ~ Skin16_mean  (merged CSV 컬럼: s1 ~ s16)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ──────────────────────────────────────────────
# trial_id 파싱 / 경로 헬퍼
# ──────────────────────────────────────────────

def _parse_trial_id(trial_id: str) -> dict:
    """
    "ecomesh_d10_z1.5_test2" → {"material": "ecomesh", "d": 10, "z": 1.5, "n": 2}
    "ecomesh_d5_z1_test1"   → {"material": "ecomesh", "d": 5,  "z": 1.0, "n": 1}
    """
    m = re.fullmatch(r"([^_]+)_d(\d+)_z([0-9.]+)_test(\d+)", trial_id)
    if m is None:
        raise ValueError(f"trial_id 파싱 실패: {trial_id!r}")
    return {
        "material": m.group(1),
        "d": int(m.group(2)),
        "z": float(m.group(3)),
        "n": int(m.group(4)),
    }


def trial_id_to_paths(trial_id: str, raw_dir: str = "raw_data") -> dict:
    """
    trial_id → merged CSV / baseline JSON 절대 경로 반환.

    Returns
    -------
    {
        "merged_csv" : Path,
        "baseline_json" : Path,
    }
    """
    p = _parse_trial_id(trial_id)
    mat  = p["material"]                       # ecomesh
    d    = p["d"]                              # 10
    z    = p["z"]                              # 1.0
    n    = p["n"]                              # 1

    # z_full: 1.0 → "1.0", 1.5 → "1.5"
    z_full = f"{z:.1f}" if z == int(z) else f"{z}"

    base = Path(raw_dir) / mat / f"d{d}" / f"z_{z_full}mm" / f"test{n}"
    stem = f"{mat}_d{d}_z{z_full}_test{n}"

    return {
        "merged_csv":    base / f"{stem}_merged.csv",
        "baseline_json": base / f"{stem}_baseline.json",
    }


# ──────────────────────────────────────────────
# 메인 설정 dataclass
# ──────────────────────────────────────────────

@dataclass
class SATSConfig:
    """SATS 전체 학습 파이프라인 설정."""

    # ── 경로 ──────────────────────────────────────────────────────────────────
    raw_dir: str = "raw_data"
    gt_dir: str = "sats/preprocessing/gt_output_v1"
    dataset_index_path: str = "sats/preprocessing/gt_output_v1/dataset_index.json"
    out_dir: str = "sats/training/runs"
    run_name: str = "lstm_v1"

    # ── 소재 / trial 선택 ─────────────────────────────────────────────────────
    material: str = "ecomesh"

    # validation에 사용할 trial_id 목록 (나머지는 train).
    # d10 제거 시: d5 기준 두 z 조건을 모두 커버하도록 설정.
    val_trials: List[str] = field(default_factory=lambda: [
        "ecomesh_d5_z1_test3",
        "ecomesh_d5_z1.5_test9",
    ])

    # 학습/검증 풀에서 제외할 인덴터 직경(mm) 목록.
    # 예) [10] → d10 trial 전체 제거. val_trials에 포함된 d10 trial도 함께 제거됨.
    exclude_diameters: List[int] = field(default_factory=list)

    # ── 데이터 필터 (GT generation과 동일 기준) ───────────────────────────────
    grid_step_mm: float  = 0.5      # 그리드 간격
    grid_tol_mm: float   = 0.05     # on-grid 허용 오차
    grid_min_mm: float   = -9.75    # 그리드 범위 [min, max]
    grid_max_mm: float   = 9.75
    fz_min_abs_n: float  = 0.1      # |Fz| < 이 값이면 GT = 0 (영압 행)
    grid_size: int       = 40       # GT 맵 한 변 크기

    # ── 시퀀스 구성 ───────────────────────────────────────────────────────────
    # 한 (x_mm, y_mm) 위치의 압입 사이클 = 하나의 시퀀스.
    # 시퀀스 내 timestep은 z_mm 오름차순(loading) → 내림차순(unloading) 순서.
    seq_len: int     = 400      # max timestep (초과하면 앞부분 사용; 실측 중앙값 ~343)
    min_seq_len: int = 3        # 이보다 짧은 시퀀스는 학습에서 제외

    # ── 센서 ──────────────────────────────────────────────────────────────────
    n_sensors: int = 16
    # merged CSV 컬럼명
    sensor_cols: List[str] = field(
        default_factory=lambda: [f"s{i}" for i in range(1, 17)]
    )
    # baseline JSON 키 (Skin1_mean ~ Skin16_mean)
    baseline_mean_keys: List[str] = field(
        default_factory=lambda: [f"Skin{i}_mean" for i in range(1, 17)]
    )

    # ── LSTM 하이퍼파라미터 ───────────────────────────────────────────────────
    hidden_dim: int    = 64      # LSTM hidden size (per sensor)
    num_layers: int    = 2       # LSTM 깊이
    dropout: float     = 0.1    # LSTM 내부 dropout (num_layers > 1 일 때 유효)
    bidirectional: bool = False  # 단방향: 실시간 추론 호환

    # ── Self-Attention 하이퍼파라미터 ─────────────────────────────────────────
    attn_dim: int      = 64      # GAT 선형 투영 차원 (논문: 125)
    n_gat_layers: int  = 2       # GAT 레이어 수 (논문: 2)

    # ── Local Map 하이퍼파라미터 ──────────────────────────────────────────────
    local_map_size: int       = 15    # 각 센서의 local map 한 변 크기 (홀수 권장)
    sensor_spacing_mm: float  = 6.5   # 센서 간 물리 간격 (mm)

    # ── CNN 하이퍼파라미터 ────────────────────────────────────────────────────
    cnn_hidden_channels: int = 16  # CNN Refiner 중간 채널 수 (논문 미명시 → 기본 16)

    # ── 체크포인트 연계 ───────────────────────────────────────────────────────
    lstm_ckpt: str      = ""     # 사전학습된 LSTM 체크포인트 경로 (빈 문자열=미사용)
    attn_ckpt: str      = ""     # 사전학습된 Self-Attention 체크포인트 경로
    local_map_ckpt: str = ""     # 사전학습된 Local Map 체크포인트 경로

    # ── 학습 ──────────────────────────────────────────────────────────────────
    # 논문 기준: batch_size=2048, lr=0.0064 (고정, 스케줄러 없음)
    batch_size: int  = 64
    lr: float        = 1e-3
    weight_decay: float = 1e-5
    epochs: int      = 50
    lr_patience: int = 5        # ReduceLROnPlateau patience
    lr_factor: float = 0.5
    clip_grad: float = 1.0      # gradient clipping (None 이면 비활성)
    use_lr_scheduler: bool = True  # False → 고정 LR (논문 방식)
    num_workers: int = 4
    seed: int        = 42
    save_every: int  = 10       # N epoch마다 체크포인트 저장

    # ── 윈도우 데이터셋 설정 ──────────────────────────────────────────────────
    # 논문 방식: loading phase만, window_size=10 슬라이딩 윈도우
    window_size: int = 10
    use_window_dataset: bool = False  # True → SATSWindowDataset (논문 방식)

    # ── 계산 장치 ─────────────────────────────────────────────────────────────
    device: str = "cuda"        # "cuda" | "cpu"

    # ─────────────────────────────────────────────────────────────────────────
    # 경로 유틸리티 (dataclass 메서드)
    # ─────────────────────────────────────────────────────────────────────────

    def run_dir(self) -> Path:
        """학습 결과 저장 디렉터리."""
        return Path(self.out_dir) / self.run_name

    def gt_npy_path(self, trial_id: str) -> Path:
        return Path(self.gt_dir) / f"{trial_id}_targets.npy"

    def trial_paths(self, trial_id: str) -> dict:
        """trial_id → merged_csv / baseline_json 경로."""
        return trial_id_to_paths(trial_id, raw_dir=self.raw_dir)

    def effective_device(self) -> str:
        """cuda 요청이지만 불가능하면 cpu로 폴백."""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device
