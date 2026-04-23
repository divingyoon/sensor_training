"""
sats/training/dataset.py

SATSSequenceDataset
-------------------
각 압입 사이클 (trial 내 한 grid 위치의 time-series)을 하나의 학습 샘플로 구성한다.

데이터 흐름
-----------
raw merged CSV  →  on-grid 필터  →  s_norm 계산  →  (x,y)별 그룹화
                                                         │
GT targets.npy (mmap) ──────────────────────────────────┘
                                 __getitem__: [sensor_seq, gt_seq]

반환 형식 (__getitem__)
-----------------------
sensor_seq : Tensor[T, 16]       s_norm 시계열, float32
gt_seq     : Tensor[T, 40, 40]   GT 압력맵 시계열, float32
length     : int                 실제 시퀀스 길이 T

반환 형식 (collate_fn)
-----------------------
sensor_batch : Tensor[B, T_max, 16]
gt_batch     : Tensor[B, T_max, 40, 40]
lengths      : Tensor[B]  int64

온-그리드 필터 (GT generation과 동일 기준)
------------------------------------------
그리드: np.linspace(-9.75, 9.75, 40), step=0.5mm
스냅:   xs = lo + round((x - lo) / step) * step   (lo=-9.75 기준 offset 사용)
조건:   |x - xs| < tol  AND  |y - ys| < tol
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import SATSConfig

log = logging.getLogger(__name__)

# ── 센서 컬럼 상수 ────────────────────────────────────────────────────────────
_SENSOR_COLS = [f"s{i}" for i in range(1, 17)]          # merged CSV 컬럼명
_SKIN_MEAN_KEYS = [f"Skin{i}_mean" for i in range(1, 17)]  # baseline JSON 키


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _load_baseline(path: Path) -> np.ndarray:
    """baseline JSON → [16] float64 (Skin1_mean ~ Skin16_mean)."""
    with open(path) as f:
        data = json.load(f)
    return np.array([data[k] for k in _SKIN_MEAN_KEYS], dtype=np.float64)


def _on_grid_mask(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    step: float = 0.5,
    tol: float  = 0.05,
    lo: float   = -9.75,
    hi: float   = 9.75,
) -> np.ndarray:
    """
    on-grid bool mask.

    올바른 스냅 공식: lo를 기준으로 offset을 계산하여 반올림.
    (단순 round(x/step)*step 은 그리드 원점 오프셋을 무시하므로 잘못된 결과를 냄)

    예: lo=-9.75, step=0.5 → grid = [-9.75, -9.25, ..., 9.75]
    x = -9.75: (x-lo)/step = 0.0 → snap = -9.75 ✓
    x = -9.50: (x-lo)/step = 0.5 → round=0 → snap=-9.75, diff=0.25 > tol → 제외 ✓
    """
    n_steps = round((hi - lo) / step)         # 39 for default params
    kx = np.clip(np.round((x_arr - lo) / step), 0, n_steps)
    ky = np.clip(np.round((y_arr - lo) / step), 0, n_steps)
    xs = lo + kx * step
    ys = lo + ky * step
    return (np.abs(x_arr - xs) < tol) & (np.abs(y_arr - ys) < tol)


def _snap_coords(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    step: float = 0.5,
    lo: float   = -9.75,
    hi: float   = 9.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """on-grid 통과 후 좌표를 그리드 좌표로 스냅 (float 비교 안정성)."""
    n_steps = round((hi - lo) / step)
    xs = lo + np.clip(np.round((x_arr - lo) / step), 0, n_steps) * step
    ys = lo + np.clip(np.round((y_arr - lo) / step), 0, n_steps) * step
    return xs, ys


def _load_trial(
    trial_id: str,
    cfg: SATSConfig,
) -> Optional[Tuple[List[dict], np.ndarray]]:
    """
    한 trial을 처리하여 (시퀀스 목록, GT mmap)을 반환한다.

    Parameters
    ----------
    trial_id : str  예) "ecomesh_d5_z1_test1"
    cfg      : SATSConfig

    Returns
    -------
    (sequences, gt_mmap) 또는 None (파일 오류 시)

    sequences: List[dict], 각 원소 =
        {
          "trial_id"   : str,
          "x_mm"       : float,
          "y_mm"       : float,
          "sensor_seq" : np.ndarray [T, 16] float32,
          "row_indices": np.ndarray [T]     int64,
        }

    gt_mmap: np.ndarray (mmap read-only) shape [N_total, grid_size, grid_size]
    """
    paths = cfg.trial_paths(trial_id)
    merged_csv    = paths["merged_csv"]
    baseline_json = paths["baseline_json"]
    gt_npy        = cfg.gt_npy_path(trial_id)

    # 파일 존재 확인
    for label, p in [("merged CSV", merged_csv),
                     ("baseline JSON", baseline_json),
                     ("GT npy", gt_npy)]:
        if not p.exists():
            log.warning("[%s] %s 없음: %s", trial_id, label, p)
            return None

    # ── 1. baseline 로드 ──────────────────────────────────────────────────
    baseline = _load_baseline(baseline_json)   # [16] float64

    # ── 2. merged CSV 로드 (필요 컬럼만) ─────────────────────────────────
    usecols = _SENSOR_COLS + ["x_mm", "y_mm", "Fz"]
    df = pd.read_csv(merged_csv, usecols=usecols, dtype=np.float64)

    x_arr = df["x_mm"].to_numpy()
    y_arr = df["y_mm"].to_numpy()

    # ── 3. on-grid 필터 ────────────────────────────────────────────────────
    mask = _on_grid_mask(
        x_arr, y_arr,
        step=cfg.grid_step_mm,
        tol=cfg.grid_tol_mm,
        lo=cfg.grid_min_mm,
        hi=cfg.grid_max_mm,
    )
    n_grid = int(mask.sum())
    log.info("[%s] on-grid: %d / %d rows", trial_id, n_grid, len(df))

    if n_grid == 0:
        log.warning("[%s] on-grid rows 없음, 건너뜀", trial_id)
        return None

    # ── 4. GT mmap 로드 & 행 수 검증 ─────────────────────────────────────
    gt_mmap = np.load(str(gt_npy), mmap_mode="r")  # [N_total, 40, 40]
    if gt_mmap.shape[0] != n_grid:
        log.warning(
            "[%s] GT 행 수 불일치 (GT=%d, on-grid=%d). trial 건너뜀.",
            trial_id, gt_mmap.shape[0], n_grid,
        )
        return None

    df_grid = df[mask].reset_index(drop=True)

    # ── 5. s_norm 계산 및 100배 스케일 업 ───────────────────────────────
    s_raw  = df_grid[_SENSOR_COLS].to_numpy(dtype=np.float64)   # [N, 16]
    s_norm = (((s_raw - baseline) / baseline) * 100.0).astype(np.float32)  # [N, 16]

    # ── 6. 그리드 좌표로 스냅 → (x,y) 별 그룹화 ─────────────────────────
    xg, yg = _snap_coords(
        df_grid["x_mm"].to_numpy(),
        df_grid["y_mm"].to_numpy(),
        step=cfg.grid_step_mm,
        lo=cfg.grid_min_mm,
        hi=cfg.grid_max_mm,
    )
    # 부동소수점 키: 4자리 반올림으로 안정화
    groups: Dict[Tuple[float, float], List[int]] = defaultdict(list)
    for i in range(n_grid):
        key = (round(float(xg[i]), 4), round(float(yg[i]), 4))
        groups[key].append(i)

    # ── 7. 시퀀스 생성 ─────────────────────────────────────────────────────
    sequences = []
    skipped_short = 0
    for (x, y), idxs in groups.items():
        T = len(idxs)
        if T < cfg.min_seq_len:
            skipped_short += 1
            continue
        # seq_len 초과 시 앞부분(loading phase 포함) 사용
        if T > cfg.seq_len:
            idxs = idxs[: cfg.seq_len]
            T = cfg.seq_len
        row_idx = np.array(idxs, dtype=np.int64)
        sequences.append({
            "trial_id":    trial_id,
            "x_mm":        x,
            "y_mm":        y,
            "sensor_seq":  s_norm[row_idx],    # [T, 16] float32 - pre-loaded
            "row_indices": row_idx,             # [T] int64 - for GT lookup
        })

    log.info(
        "[%s] 시퀀스: %d개 (건너뜀=%d)",
        trial_id, len(sequences), skipped_short,
    )
    return sequences, gt_mmap


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SATSSequenceDataset(Dataset):
    """
    SATS 학습용 시퀀스 데이터셋.

    각 샘플 = 한 grid 위치에서의 압입 사이클 전체 시계열.

    Parameters
    ----------
    trial_ids : List[str]
        학습(또는 검증)에 사용할 trial_id 목록.
    cfg : SATSConfig

    사용 예시
    ---------
    >>> cfg = SATSConfig()
    >>> train_trials = [t for t in all_trials if t not in cfg.val_trials]
    >>> train_ds = SATSSequenceDataset(train_trials, cfg)
    >>> loader = DataLoader(train_ds, batch_size=32, collate_fn=sats_collate_fn)
    """

    def __init__(self, trial_ids: List[str], cfg: SATSConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self._index: List[dict] = []                    # 시퀀스 목록
        self._gt_mmaps: Dict[str, np.ndarray] = {}     # trial_id → GT mmap

        n_failed = 0
        for tid in trial_ids:
            result = _load_trial(tid, cfg)
            if result is None:
                n_failed += 1
                continue
            seqs, gt_mmap = result
            self._index.extend(seqs)
            self._gt_mmaps[tid] = gt_mmap

        log.info(
            "Dataset 완료: %d 시퀀스, %d trial 성공, %d 실패",
            len(self._index), len(self._gt_mmaps), n_failed,
        )
        if len(self._index) == 0:
            raise RuntimeError("유효한 시퀀스가 없습니다. 경로와 설정을 확인하세요.")

    # ── Dataset 인터페이스 ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns
        -------
        sensor_seq : Tensor[T, 16]      s_norm 시계열
        gt_seq     : Tensor[T, 40, 40]  GT 압력맵 시계열
        length     : int                실제 시퀀스 길이 T
        """
        item = self._index[idx]
        trial_id    = item["trial_id"]
        row_indices = item["row_indices"]              # [T] int64
        sensor_seq  = item["sensor_seq"]               # [T, 16] float32 (pre-loaded)

        # GT: mmap에서 해당 행만 읽어 복사 후 100배 스케일 업
        gt_seq = (self._gt_mmaps[trial_id][row_indices].copy() * 100.0)   # [T, 40, 40]

        return (
            torch.from_numpy(sensor_seq),           # [T, 16]
            torch.from_numpy(gt_seq),               # [T, 40, 40]
            len(row_indices),                       # T (int)
        )

    # ── 편의 메서드 ───────────────────────────────────────────────────────────

    def trial_ids(self) -> List[str]:
        """로드된 trial_id 목록."""
        return list(self._gt_mmaps.keys())

    def seq_lengths(self) -> List[int]:
        """전체 시퀀스 길이 목록 (통계 확인용)."""
        return [len(item["row_indices"]) for item in self._index]


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────────────────────────────────────

def sats_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    가변 길이 시퀀스를 배치로 패딩.

    Parameters
    ----------
    batch : List of (sensor_seq [T_i, 16], gt_seq [T_i, 40, 40], length T_i)

    Returns
    -------
    sensor_batch : Tensor[B, T_max, 16]        zero-padded
    gt_batch     : Tensor[B, T_max, 40, 40]    zero-padded
    lengths      : Tensor[B]  int64            실제 길이
    """
    sensor_seqs, gt_seqs, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    # 센서 시퀀스: pad_sequence 사용 ([T, 16] → [B, T_max, 16])
    sensor_batch = pad_sequence(
        sensor_seqs, batch_first=True, padding_value=0.0
    )  # [B, T_max, 16]

    # GT: 수동 패딩 (4D tensor)
    B = len(gt_seqs)
    T_max = int(lengths.max().item())
    G = gt_seqs[0].shape[-1]   # 40
    gt_batch = torch.zeros(B, T_max, G, G, dtype=torch.float32)
    for i, (gs, ln) in enumerate(zip(gt_seqs, lengths)):
        gt_batch[i, : int(ln)] = gs

    return sensor_batch, gt_batch, lengths


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: SATSConfig,
    all_trial_ids: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    train / val DataLoader를 생성한다.

    Parameters
    ----------
    cfg            : SATSConfig
    all_trial_ids  : 전체 trial_id 목록. None이면 dataset_index.json에서 자동 로드.

    Returns
    -------
    train_loader, val_loader
    """
    if all_trial_ids is None:
        with open(cfg.dataset_index_path) as f:
            idx = json.load(f)
        all_trial_ids = [t["trial_id"] for t in idx["trials"]]

    val_set   = set(cfg.val_trials)
    train_ids = [t for t in all_trial_ids if t not in val_set]
    val_ids   = [t for t in all_trial_ids if t in val_set]

    log.info("Train trials (%d): %s", len(train_ids), train_ids)
    log.info("Val   trials (%d): %s", len(val_ids),   val_ids)

    train_ds = SATSSequenceDataset(train_ids, cfg)
    val_ds   = SATSSequenceDataset(val_ids,   cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=sats_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.effective_device() == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=sats_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.effective_device() == "cuda"),
        drop_last=False,
    )
    return train_loader, val_loader
