"""
dataset_zarr.py

Zarr 포맷 데이터셋 로더. GPU 학습 효율을 위해 최적화됨.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from training.data.sensor_layout import DEAD_CHANNEL_INDICES

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False


def _same_path(left: Union[str, Path], right: Union[str, Path]) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _index_candidates(zarr_path: Path) -> List[Path]:
    return [
        zarr_path / "dataset_index.json",
        zarr_path.parent / f"{zarr_path.stem}_index.json",
        zarr_path.parent / "dataset_index.json",
    ]


def _load_zarr_index(zarr_path: Path, zg) -> List[Dict]:
    index_path = next((p for p in _index_candidates(zarr_path) if p.exists()), None)
    if index_path is None:
        searched = ", ".join(str(p) for p in _index_candidates(zarr_path))
        raise FileNotFoundError(f"No dataset index found for {zarr_path}. Searched: {searched}")

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    all_samples = index_data.get("samples", [])
    if not isinstance(all_samples, list):
        raise ValueError(f"Invalid dataset index format: {index_path}")

    for sample in all_samples:
        sample_zarr = sample.get("zarr_path")
        if sample_zarr and not _same_path(sample_zarr, zarr_path):
            raise ValueError(
                f"Index {index_path} references a different zarr: {sample_zarr} != {zarr_path}"
            )
        zarr_index = int(sample["zarr_index"])
        n_rows = int(zg["tactile_lr_norm"].shape[0])
        if zarr_index < 0 or zarr_index >= n_rows:
            raise ValueError(
                f"Index {index_path} has zarr_index={zarr_index} outside {zarr_path} row count {n_rows}"
            )
    return all_samples


class ZarrDataset(Dataset):
    """
    preprocess.py에서 생성한 .zarr 데이터를 로드하는 PyTorch Dataset.
    aux_feat 마지막 컬럼이 직경(diameter_mm) 또는 접촉 반경(contact_radius_mm)일 수 있으며
    zarr attrs["aux_last_field"] 값으로 구분한다.
    
    Args:
        zarr_path: .zarr 디렉토리 경로
        split: "train", "val", "test", 또는 "all"
        phase: "loading", "unloading", 또는 "all"
        val_ratio: train/val 분할 비율
        seed: 난수 시드 (Trial 분할용)
    """

    def __init__(
        self,
        zarr_path: Union[str, Path],
        split: str = "all",
        phase: str = "loading",
        val_ratio: float = 0.2,
        seed: int = 42,
        drop_dead_channels: bool = False,
    ):
        if not ZARR_AVAILABLE:
            raise ImportError("zarr 라이브러리가 필요합니다. 'pip install zarr'를 실행하세요.")

        self.zarr_path = Path(zarr_path)
        self.split = split
        self.phase_filter = phase
        self.drop_dead_channels = drop_dead_channels
        
        # Zarr 데이터 열기
        zg = zarr.open_group(str(self.zarr_path), mode='r')
        self.aux_last_field = zg.attrs.get("aux_last_field", "diameter_mm")
        
        all_samples = _load_zarr_index(self.zarr_path, zg)
        
        # 1. Phase 필터링
        if phase != "all":
            all_samples = [s for s in all_samples if s["phase"] == phase]
            
        # 2. Trial 단위 분할
        trial_ids = sorted(list(set(s["trial_id"] for s in all_samples)))
        rng = np.random.default_rng(seed)
        rng.shuffle(trial_ids)
        
        n_val = int(len(trial_ids) * val_ratio)
        if n_val == 0 and len(trial_ids) > 1:
            n_val = 1
            
        val_trials = set(trial_ids[:n_val])
        train_trials = set(trial_ids[n_val:])
        
        if split == "train":
            keep_trials = train_trials
        elif split == "val":
            keep_trials = val_trials
        elif split == "all":
            keep_trials = set(trial_ids)
        else:
            keep_trials = val_trials

        self.samples = [s for s in all_samples if s["trial_id"] in keep_trials]
        
        # 3. 데이터 메모리 로드 (병목 해결 핵심)
        print(f"  [{split}] 데이터를 메모리에 로드 중... ({len(self.samples):,} 샘플)")
        indices = [s["zarr_index"] for s in self.samples]
        
        # 필요한 컬럼만 oindex를 사용하여 한 번에 NumPy로 로드
        self.tactile_data = torch.from_numpy(np.array(zg["tactile_lr_norm"].oindex[indices])).float()
        if self.drop_dead_channels:
            live_idx = [i for i in range(self.tactile_data.shape[1]) if i not in DEAD_CHANNEL_INDICES]
            self.tactile_data = self.tactile_data[:, live_idx]
        self.aux_data = torch.from_numpy(np.array(zg["aux_feat"].oindex[indices])).float()
        self.cx_data = torch.from_numpy(np.array(zg["cx"].oindex[indices])).float()
        self.cy_data = torch.from_numpy(np.array(zg["cy"].oindex[indices])).float()
        self.depth_data = torch.from_numpy(np.array(zg["depth_mm"].oindex[indices])).float()
        self.fz_data = torch.from_numpy(np.array(zg["fz"].oindex[indices])).float()

        self.trial_ids = [s["trial_id"] for s in self.samples]
        n_ch = self.tactile_data.shape[1]
        if self.drop_dead_channels:
            print(f"  [{split}] 로드 완료. (dead channel 제거 적용: tactile {n_ch}ch)")
        else:
            print(f"  [{split}] 로드 완료. (tactile {n_ch}ch)")
        
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.aux_last_field == "contact_radius_mm":
            radius_val = self.aux_data[idx, 3]
        else:
            radius_val = self.aux_data[idx, 3] / 2.0  # diameter → radius
        return {
            "tactile": self.tactile_data[idx],
            "radius":  radius_val,
            "target_sr": torch.stack([self.cx_data[idx], self.cy_data[idx], self.depth_data[idx]]),
            "target_fz": self.fz_data[idx:idx+1],
            "trial_id": self.trial_ids[idx]
        }
