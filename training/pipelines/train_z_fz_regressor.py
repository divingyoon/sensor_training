import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.models.multi_head_field_model import MultiHeadFieldModel
from training.models.z_fz_sequence_regressor import ZFzSequenceRegressor
from training.pipelines.train_comparison import (
    ZarrSequenceDataset,
    _decode_xy_from_heatmap,
    _parse_trial_list,
    _resolve_device,
    _resolve_zarr_path,
    _split_indices_by_trial,
)


XY_SCALE_MM = 10.0
RADIUS_SCALE_MM = 5.0


@dataclass(frozen=True)
class ScalarNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "ScalarNormalizer":
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp(min=1e-6)
        return cls(mean=mean, std=std)

    @classmethod
    def from_dict(cls, payload: dict, device: torch.device | None = None) -> "ScalarNormalizer":
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32, device=device),
            std=torch.tensor(payload["std"], dtype=torch.float32, device=device),
        )

    def to_dict(self) -> dict:
        return {
            "mean": [float(x) for x in self.mean.detach().cpu().tolist()],
            "std": [float(x) for x in self.std.detach().cpu().tolist()],
            "outputs": ["z", "fz"],
        }

    def to(self, device: torch.device) -> "ScalarNormalizer":
        return ScalarNormalizer(mean=self.mean.to(device), std=self.std.to(device))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values.device)) / self.std.to(values.device)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.device) + self.mean.to(values.device)


def build_condition_features(xy_mm: torch.Tensor, radius_mm: torch.Tensor) -> torch.Tensor:
    radius = radius_mm.view(-1, 1)
    return torch.cat([xy_mm / XY_SCALE_MM, radius / RADIUS_SCALE_MM], dim=1)


def metric_dict(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    diff = preds_np - targets_np
    mse = np.mean(diff * diff, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff), axis=0)
    names = ["z", "fz"]
    return {
        "metric_schema": {
            "outputs": names,
            "array_order": names,
            "units": {"z": "mm", "fz": "source_units"},
        },
        "output_names": names,
        "mse": mse.tolist(),
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "per_output": {
            name: {
                "mse": float(mse[i]),
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
            }
            for i, name in enumerate(names)
        },
    }


def _load_xy_model(args, device: torch.device) -> MultiHeadFieldModel | None:
    if not args.xy_checkpoint:
        return None
    model = MultiHeadFieldModel(seq_len=args.seq_len, heatmap_size=args.heatmap_size).to(device)
    ckpt = torch.load(args.xy_checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _xy_condition(
    grid: torch.Tensor,
    targets: torch.Tensor,
    xy_model: MultiHeadFieldModel | None,
    args,
) -> torch.Tensor:
    if xy_model is None:
        return targets[:, :2]
    with torch.no_grad():
        _, fmap = xy_model(grid)
        x_dec, y_dec = _decode_xy_from_heatmap(fmap, args.decode_xy)
        return torch.stack([x_dec, y_dec], dim=1)


def _batch_indices(indices: torch.Tensor, batch_size: int, shuffle: bool) -> list[torch.Tensor]:
    if shuffle:
        indices = indices[torch.randperm(indices.numel(), device=indices.device)]
    return [indices[i : i + batch_size] for i in range(0, indices.numel(), batch_size)]


def _evaluate(model, dataset, indices, normalizer, xy_model, args, device, use_predicted_xy: bool) -> dict:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx in _batch_indices(indices, args.batch_size, shuffle=False):
            grid = dataset.grid[batch_idx].to(device)
            tgt = dataset.tgt[batch_idx].to(device)
            xy_model_for_batch = xy_model if use_predicted_xy else None
            xy = _xy_condition(grid, tgt, xy_model_for_batch, args)
            cond = build_condition_features(xy, tgt[:, 4:5])
            pred_norm = model(grid, cond)
            preds.append(normalizer.denormalize(pred_norm))
            targets.append(tgt[:, 2:4])
    return metric_dict(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def _preload_dataset(ds: ZarrSequenceDataset, device: torch.device):
    index_matrix = torch.zeros((len(ds), ds.seq_len), dtype=torch.long)
    valid_mask = torch.zeros((len(ds), ds.seq_len), dtype=torch.bool)
    last_indices = torch.zeros(len(ds), dtype=torch.long)

    for row, sample_indices in enumerate(ds.samples):
        count = min(len(sample_indices), ds.seq_len)
        index_matrix[row, :count] = torch.tensor(sample_indices[:count], dtype=torch.long)
        valid_mask[row, :count] = True
        last_indices[row] = int(sample_indices[-1])

    s16 = ds.tactile[index_matrix]
    s16 = torch.where(valid_mask.unsqueeze(-1), s16, torch.zeros_like(s16))
    ds.grid = s16.reshape(len(ds), ds.seq_len, 1, 4, 4).to(device)
    ds.tgt = torch.stack(
        [
            ds.cx[last_indices],
            ds.cy[last_indices],
            ds.depth[last_indices],
            ds.fz[last_indices],
            ds.radius[last_indices, 0],
        ],
        dim=1,
    ).to(device)
    return ds


def _limit_samples_for_smoke(ds: ZarrSequenceDataset, max_samples: int) -> ZarrSequenceDataset:
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds

    by_trial: dict[str, list[int]] = {}
    for idx, trial_id in enumerate(ds.sample_trial_ids):
        by_trial.setdefault(str(trial_id), []).append(idx)

    selected = []
    trial_ids = sorted(by_trial)
    offset = 0
    while len(selected) < max_samples:
        added = False
        for trial_id in trial_ids:
            trial_samples = by_trial[trial_id]
            if offset < len(trial_samples):
                selected.append(trial_samples[offset])
                added = True
                if len(selected) >= max_samples:
                    break
        if not added:
            break
        offset += 1

    selected = sorted(selected)
    ds.samples = [ds.samples[i] for i in selected]
    ds.sample_trial_ids = [ds.sample_trial_ids[i] for i in selected]
    return ds


def train(args) -> dict:
    device = _resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = _resolve_zarr_path(args.data_dir, args.zarr_path)
    if not zarr_path:
        raise RuntimeError(f"No .zarr found under {args.data_dir}")

    ds = ZarrSequenceDataset(zarr_path=zarr_path, seq_len=args.seq_len, stride=args.stride, phase=args.phase)
    ds = _limit_samples_for_smoke(ds, args.max_samples)
    split = _split_indices_by_trial(
        ds,
        seed=args.seed,
        val_trials=_parse_trial_list(args.val_trials),
        test_trials=_parse_trial_list(args.test_trials),
    )
    ds = _preload_dataset(ds, device)
    train_idx = torch.tensor(split.train_indices, dtype=torch.long, device=device)
    val_idx = torch.tensor(split.val_indices, dtype=torch.long, device=device)

    normalizer = ScalarNormalizer.fit(ds.tgt[train_idx, 2:4]).to(device)
    xy_model = _load_xy_model(args, device)
    model = ZFzSequenceRegressor(seq_len=args.seq_len).to(device)
    criterion = nn.SmoothL1Loss(beta=args.huber_delta) if args.loss == "huber" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_metric = float("inf")
    best_metrics = {}

    history = {"train_loss": [], "val_gt_xy_mae": [], "val_pred_xy_mae": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_idx in _batch_indices(train_idx, args.batch_size, shuffle=True):
            grid = ds.grid[batch_idx].to(device)
            tgt = ds.tgt[batch_idx].to(device)
            xy = tgt[:, :2]
            if args.xy_noise_std_mm > 0:
                xy = xy + torch.randn_like(xy) * args.xy_noise_std_mm
            cond = build_condition_features(xy, tgt[:, 4:5])
            target_norm = normalizer.normalize(tgt[:, 2:4])

            optimizer.zero_grad()
            loss = criterion(model(grid, cond), target_norm)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        gt_metrics = _evaluate(model, ds, val_idx, normalizer, xy_model, args, device, use_predicted_xy=False)
        pred_metrics = _evaluate(
            model,
            ds,
            val_idx,
            normalizer,
            xy_model,
            args,
            device,
            use_predicted_xy=xy_model is not None,
        )
        history["train_loss"].append(float(np.mean(losses)))
        history["val_gt_xy_mae"].append(gt_metrics["mae"])
        history["val_pred_xy_mae"].append(pred_metrics["mae"])

        select_mae = pred_metrics["mae"][0] + pred_metrics["mae"][1]
        if select_mae < best_metric:
            best_metric = select_mae
            best_metrics = {"gt_xy": gt_metrics, "predicted_xy": pred_metrics}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_norm": normalizer.to_dict(),
                    "metrics": best_metrics,
                    "history": history,
                    "args": vars(args),
                    "split": {
                        "train_trials": split.train_trials,
                        "val_trials": split.val_trials,
                        "test_trials": split.test_trials,
                    },
                },
                args.out_dir / "best_z_fz_regressor.pth",
            )
            with (args.out_dir / "metrics_z_fz_regressor.json").open("w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)

        print(
            f"[EPOCH {epoch:03d}/{args.epochs}] "
            f"loss={history['train_loss'][-1]:.6f} "
            f"gt_xy_mae[z,fz]={gt_metrics['mae']} pred_xy_mae[z,fz]={pred_metrics['mae']}"
        )

    with (args.out_dir / "history_z_fz_regressor.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a separate z/Fz regressor from tactile sequence + xy/radius condition.")
    parser.add_argument("--data-dir", type=str, default="preprocessing/processed_data")
    parser.add_argument("--zarr-path", type=str, default="")
    parser.add_argument("--out-dir", type=Path, default=Path("training/runs_z_fz"))
    parser.add_argument("--xy-checkpoint", type=str, default="", help="Optional frozen multi_head_field checkpoint for end-to-end xy-conditioned eval")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--phase", choices=["loading", "unloading", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-trials", nargs="*", default=None)
    parser.add_argument("--test-trials", nargs="*", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--xy-noise-std-mm", type=float, default=0.5)
    parser.add_argument("--decode-xy", choices=["softargmax", "argmax_refine"], default="softargmax")
    parser.add_argument("--heatmap-size", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional balanced sample cap for quick smoke runs. 0 uses all samples.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
