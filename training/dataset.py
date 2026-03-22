"""
dataset.py

SkinDataset: preprocessing_data 디렉토리에서 샘플을 로드하는 PyTorch Dataset.

샘플 단위:
  tactile_lr_norm.npy  (16,) — baseline-subtracted, filtered, z-score normalized
  aux_feat.npy         (4,)  — [fx_N, fy_N, depth_mm, radius_mm]
  hr_contact_map.npy   (64, 64) float32 — soft Gaussian pseudo GT
  meta.json

출력 dict:
  "tactile"    (14,) float32 — dead ch 제거 후
  "tactile_raw" (16,) float32 — dead ch 포함 원본 (D(M) 일관성에 사용)
  "aux"        (4,)  float32
  "hr_map"     (1, 64, 64) float32
  "fz"         scalar float32
  "cx"         scalar float32 — contact center x (mm)
  "cy"         scalar float32 — contact center y (mm)
  "x_bounds"   (2,) float32  — canvas x bounds
  "y_bounds"   (2,) float32  — canvas y bounds
  "depth_mm"   scalar float32
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .sensor_layout import DEAD_CHANNEL_INDICES, LIVE_CHANNEL_INDICES


class SkinDataset(Dataset):
    """
    Args:
        data_dir:     preprocessing_data 최상위 디렉토리
        split:        "train", "val", "test", 또는 "all"
        phase:        "loading", "unloading", 또는 "all"
        val_ratio:    train/val 분할 비율 (test 제외)
        test_trials:  test set으로 고정할 trial_id 목록. None이면 자동 분할.
        val_trials:   val set으로 고정할 trial_id 목록. None이면 자동 분할.
        seed:         trial 분할 시 난수 시드
        min_depth_mm: 이 depth 이상인 샘플만 포함 (0.0 이하 샘플 제거)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "all",
        phase: str = "loading",
        val_ratio: float = 0.15,
        test_trials: Optional[List[str]] = None,
        val_trials: Optional[List[str]] = None,
        seed: int = 42,
        min_depth_mm: float = 0.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.phase_filter = phase

        index_path = self.data_dir / "dataset_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"dataset_index.json not found: {index_path}")

        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        all_samples: List[Dict] = index["samples"]

        # depth 필터
        if min_depth_mm > 0.0:
            all_samples = [s for s in all_samples if _safe_depth(s) >= min_depth_mm]

        # phase 필터
        if phase != "all":
            all_samples = [s for s in all_samples if s.get("phase") == phase]

        # trial 단위 split
        trial_ids = sorted({s["trial_id"] for s in all_samples})
        rng = random.Random(seed)
        rng.shuffle(trial_ids)

        if test_trials is not None:
            test_set = set(test_trials)
        else:
            n_test = max(0, min(1, len(trial_ids) - 2)) if len(trial_ids) >= 3 else 0
            test_set = set(trial_ids[-n_test:]) if n_test > 0 else set()
            trial_ids = trial_ids[: len(trial_ids) - n_test] if n_test > 0 else trial_ids

        if val_trials is not None:
            val_set = set(val_trials)
        else:
            n_val = max(0, min(max(1, int(len(trial_ids) * val_ratio)), len(trial_ids) - 1))
            val_set = set(trial_ids[-n_val:]) if n_val > 0 else set()
            trial_ids = trial_ids[: len(trial_ids) - n_val] if n_val > 0 else trial_ids

        train_set = set(trial_ids)

        # trial 부족 시 모든 split이 전체를 봄 (단일 trial 상황)
        if len(train_set) == 0 and len(val_set) == 0 and len(test_set) == 0:
            all_ids = sorted({s["trial_id"] for s in all_samples})
            train_set = val_set = test_set = set(all_ids)
        elif len(train_set) == 0:
            train_set = val_set | test_set
        elif len(val_set) == 0:
            val_set = train_set
        if len(test_set) == 0:
            test_set = val_set

        if split == "train":
            keep = train_set
        elif split == "val":
            keep = val_set
        elif split == "test":
            keep = test_set
        else:  # "all"
            keep = train_set | val_set | test_set

        self.samples: List[Dict] = [s for s in all_samples if s["trial_id"] in keep]
        self.live_idx = list(LIVE_CHANNEL_INDICES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.samples[idx]
        sample_dir = Path(info["sample_dir"])

        tactile_raw = np.load(sample_dir / "tactile_lr_norm.npy").astype(np.float32)
        aux = np.load(sample_dir / "aux_feat.npy").astype(np.float32)
        hr_map = np.load(sample_dir / "hr_contact_map.npy").astype(np.float32)

        with open(sample_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        tactile_live = tactile_raw[self.live_idx]

        x_bounds = np.array(meta["map_x_bounds_mm"], dtype=np.float32)
        y_bounds = np.array(meta["map_y_bounds_mm"], dtype=np.float32)

        return {
            "tactile": torch.from_numpy(tactile_live),
            "tactile_raw": torch.from_numpy(tactile_raw),
            "aux": torch.from_numpy(aux),
            "hr_map": torch.from_numpy(hr_map).unsqueeze(0),
            "fz": torch.tensor(meta["fz_N"], dtype=torch.float32),
            "cx": torch.tensor(meta["contact_center_x_mm"], dtype=torch.float32),
            "cy": torch.tensor(meta["contact_center_y_mm"], dtype=torch.float32),
            "depth_mm": torch.tensor(meta["depth_mm"], dtype=torch.float32),
            "x_bounds": torch.from_numpy(x_bounds),
            "y_bounds": torch.from_numpy(y_bounds),
        }

    def __repr__(self) -> str:
        return (
            f"SkinDataset(split={self.split!r}, phase={self.phase_filter!r}, "
            f"n_samples={len(self.samples)})"
        )


def _safe_depth(sample: Dict) -> float:
    return float(sample.get("depth_bin_mm", 0.0))


def build_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 0,
    phase: str = "loading",
    min_depth_mm: float = 0.0,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_trials: Optional[List[str]] = None,
    val_trials: Optional[List[str]] = None,
) -> Tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """train / val / test DataLoader 반환."""
    from torch.utils.data import DataLoader

    common = dict(
        data_dir=data_dir,
        phase=phase,
        min_depth_mm=min_depth_mm,
        seed=seed,
        val_ratio=val_ratio,
        test_trials=test_trials,
        val_trials=val_trials,
    )

    train_ds = SkinDataset(split="train", **common)
    val_ds = SkinDataset(split="val", **common)
    test_ds = SkinDataset(split="test", **common)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
