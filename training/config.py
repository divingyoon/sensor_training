"""
config.py

학습 설정 기본값. argparse와 함께 사용.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainConfig:
    # 경로
    data_dir: Path = Path("/home/user/skin_ws/preprocessing/preprocessing_data")
    out_dir: Path = Path("/home/user/skin_ws/training/runs")

    # 모델 phase
    phase: int = 1  # 1: MLP+CNN baseline, 2: 1D CNN + FiLM

    # 데이터
    batch_size: int = 64
    num_workers: int = 0
    min_depth_mm: float = 0.5
    data_phase: str = "loading"
    val_ratio: float = 0.15
    seed: int = 42

    # 학습
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # 모델 구조
    latent_dim: int = 128
    map_size: int = 64
    n_tactile: int = 14    # dead ch 제거 후
    n_aux: int = 4         # [fx, fy, depth_mm, radius_mm]

    # Loss 가중치
    lambda_map: float = 1.0
    lambda_sensor: float = 0.5
    lambda_fz: float = 0.01   # Fz 스케일 차이로 인해 작게 설정
    lambda_smooth: float = 0.1

    # 센서 레이아웃
    sensor_spacing_mm: float = 6.5
    sensor_origin_x_mm: float = 0.0
    sensor_origin_y_mm: float = 0.0
    canvas_size_mm: float = 25.0

    # 체크포인트
    save_best: bool = True
    log_interval: int = 10  # epoch 단위

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir)
