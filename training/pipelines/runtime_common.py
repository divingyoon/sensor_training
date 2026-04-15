from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from training.data.dataset_zarr import ZarrDataset


@dataclass(frozen=True)
class TrialSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    train_trials: list[str]
    val_trials: list[str]
    test_trials: list[str]
    fold_index: int = 0
    num_folds: int = 1
    split_mode: str = "manual"


def resolve_zarr_path(data_dir: str, zarr_path: str = "") -> str:
    if zarr_path:
        return str(Path(zarr_path))
    p = Path(data_dir)
    if p.suffix == ".zarr":
        return str(p)
    cands = sorted((p / "zarr_data").glob("*.zarr")) + sorted(p.glob("*.zarr"))
    cands = sorted(set(cands))
    if len(cands) == 1:
        return str(cands[0])
    if len(cands) > 1:
        formatted = ", ".join(str(c) for c in cands)
        raise RuntimeError(
            "Found multiple .zarr datasets; pass --zarr-path explicitly or provide a single integrated zarr. "
            f"Candidates: {formatted}"
        )
    return ""


def parse_trial_list(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    if values is None:
        return None

    parsed: list[str] = []
    for raw in values:
        parsed.extend(token.strip() for token in str(raw).split(",") if token.strip())
    return parsed


def dataset_split_ids(dataset: Dataset) -> list[str]:
    explicit_ids = getattr(dataset, "sample_trial_ids", None)
    if explicit_ids is not None:
        if len(explicit_ids) != len(dataset):
            raise RuntimeError(
                f"Dataset split metadata length mismatch: {len(explicit_ids)} ids for {len(dataset)} samples"
            )
        return [str(x) for x in explicit_ids]

    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise RuntimeError("Dataset does not expose trial split metadata; expected sample_trial_ids or samples")

    split_ids = []
    for sample in samples:
        if isinstance(sample, dict) and "trial_id" in sample:
            split_ids.append(str(sample["trial_id"]))
        elif isinstance(sample, tuple) and sample:
            split_ids.append(str(sample[0]))
        else:
            raise RuntimeError(f"Cannot infer trial id for dataset sample: {sample!r}")
    return split_ids


def split_indices_for_trials(dataset: Dataset, train_trials: Sequence[str], val_trials: Sequence[str], test_trials: Sequence[str]) -> TrialSplit:
    sample_trials = dataset_split_ids(dataset)
    train_set = set(str(t) for t in train_trials)
    val_set = set(str(t) for t in val_trials)
    test_set = set(str(t) for t in test_trials)
    return TrialSplit(
        train_indices=[i for i, trial in enumerate(sample_trials) if trial in train_set],
        val_indices=[i for i, trial in enumerate(sample_trials) if trial in val_set],
        test_indices=[i for i, trial in enumerate(sample_trials) if trial in test_set],
        train_trials=sorted(train_set),
        val_trials=sorted(val_set),
        test_trials=sorted(test_set),
    )


def split_indices_by_trial(
    dataset: Dataset,
    seed: int,
    val_trials: Optional[Sequence[str]] = None,
    test_trials: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
) -> TrialSplit:
    sample_trials = dataset_split_ids(dataset)
    trial_ids = sorted(set(sample_trials))
    if len(trial_ids) < 2 and val_trials is None and test_trials is None:
        raise RuntimeError(
            "Trial-level split needs at least 2 distinct trials for train/val. "
            f"Found: {trial_ids}"
        )

    all_trials = set(trial_ids)
    requested_val = set(val_trials or [])
    requested_test = set(test_trials or [])
    unknown = (requested_val | requested_test) - all_trials
    if unknown:
        raise RuntimeError(f"Requested split trials are not present in dataset: {sorted(unknown)}")

    overlap = requested_val & requested_test
    if overlap:
        raise RuntimeError(f"Trials cannot be both val and test: {sorted(overlap)}")

    rng = np.random.default_rng(seed)
    remaining_trials = [t for t in trial_ids if t not in requested_val and t not in requested_test]

    if val_trials is None:
        if len(remaining_trials) < 2:
            raise RuntimeError(
                "Seed-based trial split needs at least 2 train/val candidate trials after test exclusion. "
                f"Candidates: {remaining_trials}"
            )
        shuffled = np.array(remaining_trials, dtype=object)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        requested_val = set(str(t) for t in shuffled[:n_val])
        remaining_trials = [str(t) for t in shuffled[n_val:]]

    train_trials = sorted(t for t in remaining_trials if t not in requested_val)
    val_trials_sorted = sorted(requested_val)
    test_trials_sorted = sorted(requested_test)

    if not train_trials:
        raise RuntimeError("Trial split leaves no train trials. Adjust --val-trials/--test-trials.")
    if not val_trials_sorted:
        raise RuntimeError("Trial split leaves no val trials. Provide --val-trials or more trials.")

    base = split_indices_for_trials(dataset, train_trials, val_trials_sorted, test_trials_sorted)
    return TrialSplit(
        train_indices=base.train_indices,
        val_indices=base.val_indices,
        test_indices=base.test_indices,
        train_trials=base.train_trials,
        val_trials=base.val_trials,
        test_trials=base.test_trials,
        fold_index=0,
        num_folds=1,
        split_mode="manual" if val_trials is not None or test_trials is not None else "single_split",
    )


def build_cv_splits(
    dataset: Dataset,
    seed: int,
    cv_folds: int,
    val_trials: Optional[Sequence[str]] = None,
    test_trials: Optional[Sequence[str]] = None,
) -> list[TrialSplit]:
    if val_trials is not None or test_trials is not None:
        return [split_indices_by_trial(dataset, seed, val_trials=val_trials, test_trials=test_trials)]

    sample_trials = dataset_split_ids(dataset)
    trial_ids = sorted(set(sample_trials))
    if len(trial_ids) < 2:
        raise RuntimeError(f"Trial-aware CV needs at least 2 distinct trials. Found: {trial_ids}")

    num_folds = max(2, min(cv_folds, len(trial_ids)))
    rng = np.random.default_rng(seed)
    shuffled = np.array(trial_ids, dtype=object)
    rng.shuffle(shuffled)
    trial_folds = [sorted(str(t) for t in fold.tolist()) for fold in np.array_split(shuffled, num_folds) if len(fold) > 0]

    splits: list[TrialSplit] = []
    for fold_index, val_fold in enumerate(trial_folds):
        train_trials = sorted(str(t) for t in trial_ids if str(t) not in set(val_fold))
        base = split_indices_for_trials(dataset, train_trials, val_fold, [])
        splits.append(
            TrialSplit(
                train_indices=base.train_indices,
                val_indices=base.val_indices,
                test_indices=base.test_indices,
                train_trials=base.train_trials,
                val_trials=base.val_trials,
                test_trials=base.test_trials,
                fold_index=fold_index,
                num_folds=len(trial_folds),
                split_mode="kfold",
            )
        )
    return splits


def save_cv_manifest(path: Path, splits: Sequence[TrialSplit]) -> None:
    payload = {
        "num_folds": len(splits),
        "folds": [
            {
                "fold_index": split.fold_index,
                "num_folds": split.num_folds,
                "split_mode": split.split_mode,
                "train_trials": split.train_trials,
                "val_trials": split.val_trials,
                "test_trials": split.test_trials,
                "train_count": len(split.train_indices),
                "val_count": len(split.val_indices),
                "test_count": len(split.test_indices),
            }
            for split in splits
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class ZarrSequenceDataset(Dataset):
    """
    Build sequence samples from preprocessed Zarr data.
    Sequence key: (trial_id, x_mm, y_mm), sorted by depth within the requested phase.
    """

    def __init__(self, zarr_path: str, seq_len: int = 50, stride: int = 5, phase: str = "all"):
        start = time.perf_counter()
        self.seq_len = seq_len
        self.stride = stride

        print(
            f"[INFO] Loading ZarrSequenceDataset from {zarr_path} "
            f"(phase={phase}, seq_len={seq_len}, stride={stride})",
            flush=True,
        )
        zds = ZarrDataset(zarr_path=zarr_path, split="all", phase=phase)
        tactile = zds.tactile_data.float()
        if getattr(zds, "aux_last_field", "diameter_mm") == "contact_radius_mm":
            radius = zds.aux_data[:, 3:4].float()
        else:
            radius = (zds.aux_data[:, 3:4] / 2.0).float()
        cx = zds.cx_data.float()
        cy = zds.cy_data.float()
        depth = zds.depth_data.float()
        fz = zds.fz_data.float()
        trial_ids = zds.trial_ids

        groups = defaultdict(list)
        for i in range(len(trial_ids)):
            key = (str(trial_ids[i]), round(float(cx[i].item()), 3), round(float(cy[i].item()), 3))
            groups[key].append(i)
        print(
            f"[INFO] Grouped {len(trial_ids):,} rows into {len(groups):,} trial/xy sequences",
            flush=True,
        )

        self.samples = []
        self.sample_trial_ids = []
        for key, idxs in groups.items():
            idxs = sorted(idxs, key=lambda j: float(depth[j].item()))
            trial_id = key[0]
            t = len(idxs)
            if t <= 0:
                continue
            if t <= seq_len:
                self.samples.append(idxs)
                self.sample_trial_ids.append(trial_id)
            else:
                max_start = t - seq_len
                for s in range(0, max_start + 1, stride):
                    self.samples.append(idxs[s : s + seq_len])
                    self.sample_trial_ids.append(trial_id)

        self.tactile = tactile
        self.radius = radius
        self.cx = cx
        self.cy = cy
        self.depth = depth
        self.fz = fz
        print(
            f"[INFO] ZarrSequenceDataset ready: {len(self.samples):,} samples built in "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idxs = self.samples[idx]
        t = len(idxs)
        s16 = self.tactile[idxs]
        r = self.radius[idxs]

        if t < self.seq_len:
            pad = self.seq_len - t
            s16 = torch.cat([s16, torch.zeros(pad, 16, dtype=s16.dtype)], dim=0)
            r = torch.cat([r, torch.zeros(pad, 1, dtype=r.dtype)], dim=0)

        s16 = s16[: self.seq_len]
        r = r[: self.seq_len]
        grid = s16.reshape(self.seq_len, 1, 4, 4)

        iso = torch.zeros(self.seq_len, 17, dtype=s16.dtype)
        iso[:, :16] = s16
        iso[:, 16:17] = r

        last_i = idxs[-1]
        tgt = torch.zeros(5, dtype=s16.dtype)
        tgt[0] = self.cx[last_i]
        tgt[1] = self.cy[last_i]
        tgt[2] = self.depth[last_i]
        tgt[3] = self.fz[last_i]
        tgt[4] = self.radius[last_i]
        return grid, iso, tgt
