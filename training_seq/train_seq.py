import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

try:
    import pandas as pd
except Exception as exc:
    raise RuntimeError("pandas is required for csv mode") from exc

try:
    import zarr

    ZARR_AVAILABLE = True
except Exception:
    ZARR_AVAILABLE = False


def _resolve_csv_files(data_dir: Path) -> List[Path]:
    if data_dir.is_file() and data_dir.suffix.lower() == ".csv":
        return [data_dir]
    return sorted(data_dir.rglob("*.csv"))


def _phase_mask_from_z(z: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("all", "both"):
        return np.ones_like(z, dtype=bool)
    dz = np.diff(z, prepend=z[0])
    eps = 1e-6
    loading = dz >= eps
    unloading = dz <= -eps
    if mode == "loading":
        return loading
    if mode == "unloading":
        return unloading
    return np.ones_like(z, dtype=bool)


def _normalize_seq_feature_per_channel(train_feat: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    x = train_feat.reshape(-1, train_feat.shape[-1])
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    y = feat.reshape(-1, feat.shape[-1])
    y = (y - mean) / std
    return y.reshape_as(feat)


def _build_windows_from_stream(
    tactile_2d: np.ndarray,
    aux_2d: np.ndarray,
    target_2d: np.ndarray,
    seq_len: int,
    stride: int,
    target_mode: str,
    baseline_frames: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    xs, as_, ys, bs = [], [], [], []
    n = tactile_2d.shape[0]
    if n < seq_len:
        return xs, as_, ys, bs
    bf = max(1, int(min(seq_len, baseline_frames)))

    for start in range(0, n - seq_len + 1, stride):
        end = start + seq_len
        tgt_win = target_2d[start:end]
        baseline = tgt_win[:bf].mean(axis=0)
        target_last = tgt_win[-1]
        if target_mode == "residual":
            y_out = target_last - baseline
        else:
            y_out = target_last

        xs.append(tactile_2d[start:end])
        as_.append(aux_2d[start:end])
        ys.append(y_out)
        bs.append(baseline)
    return xs, as_, ys, bs


def build_from_csv(
    data_dir: Path,
    phase: str,
    seq_len: int,
    stride: int,
    min_abs_z: float,
    csv_x_scale: float,
    csv_y_scale: float,
    csv_z_scale: float,
    csv_fx_scale: float,
    csv_fy_scale: float,
    csv_fz_scale: float,
    target_mode: str,
    baseline_frames: int,
    predict_fz: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    csv_files = _resolve_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No csv files found under: {data_dir}")

    all_x, all_aux, all_y, all_b, all_trial = [], [], [], [], []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        skin_cols = [f"Skin{i}" for i in range(1, 17)]
        required = ["X", "Y", "Z", "Fx", "Fy", "Fz", *skin_cols]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise KeyError(f"{csv_path} missing columns: {miss}")

        tactile = df[skin_cols].to_numpy(dtype=np.float32)
        x = df["X"].to_numpy(dtype=np.float32) * float(csv_x_scale)
        y = df["Y"].to_numpy(dtype=np.float32) * float(csv_y_scale)
        z = df["Z"].to_numpy(dtype=np.float32) * float(csv_z_scale)
        fx = df["Fx"].to_numpy(dtype=np.float32) * float(csv_fx_scale)
        fy = df["Fy"].to_numpy(dtype=np.float32) * float(csv_fy_scale)
        fz = df["Fz"].to_numpy(dtype=np.float32) * float(csv_fz_scale)

        mask_phase = _phase_mask_from_z(z, phase)
        mask_depth = np.abs(z) >= float(min_abs_z)
        mask = mask_phase & mask_depth
        if mask.sum() < seq_len:
            continue

        tactile = tactile[mask]
        aux = np.stack([fx[mask], fy[mask], fz[mask]], axis=1)

        if predict_fz:
            tgt = np.stack([x[mask], y[mask], z[mask], fz[mask]], axis=1)
        else:
            tgt = np.stack([x[mask], y[mask], z[mask]], axis=1)

        xs, as_, ys, bs = _build_windows_from_stream(
            tactile,
            aux,
            tgt,
            seq_len,
            stride,
            target_mode=target_mode,
            baseline_frames=baseline_frames,
        )
        all_x.extend(xs)
        all_aux.extend(as_)
        all_y.extend(ys)
        all_b.extend(bs)
        all_trial.extend([csv_path.stem] * len(xs))

    if not all_x:
        raise RuntimeError("No sequence windows produced from csv data")

    return (
        np.asarray(all_x, dtype=np.float32),
        np.asarray(all_aux, dtype=np.float32),
        np.asarray(all_y, dtype=np.float32),
        np.asarray(all_b, dtype=np.float32),
        np.asarray(all_trial),
    )


def _find_index_files(data_dir: Path) -> List[Path]:
    root_idx = data_dir / "dataset_index.json"
    if root_idx.exists():
        return [root_idx]
    return sorted(data_dir.rglob("dataset_index.json"))


def build_from_zarr(
    data_dir: Path,
    phase: str,
    seq_len: int,
    stride: int,
    min_depth_mm: float,
    target_mode: str,
    baseline_frames: int,
    predict_fz: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not ZARR_AVAILABLE:
        raise RuntimeError("zarr is required for zarr mode")

    index_files = _find_index_files(data_dir)
    if not index_files:
        raise FileNotFoundError(f"No dataset_index.json found under: {data_dir}")

    all_samples: List[Dict] = []
    for ip in index_files:
        with open(ip, "r", encoding="utf-8") as f:
            idx = json.load(f)
        all_samples.extend(idx.get("samples", []))
    if not all_samples:
        raise RuntimeError("No samples in dataset_index files")

    if phase not in ("all", "both"):
        all_samples = [s for s in all_samples if s.get("phase") == phase]
    all_samples = [s for s in all_samples if float(s.get("depth_bin_mm", 0.0)) >= float(min_depth_mm)]
    if not all_samples:
        raise RuntimeError("No samples after phase/depth filter")

    streams: Dict[Tuple[str, str], List[Dict]] = {}
    for s in all_samples:
        key = (str(s.get("trial_id", "unknown")), str(s.get("phase", "unknown")))
        streams.setdefault(key, []).append(s)

    zarr_cache: Dict[str, object] = {}
    all_x, all_aux, all_y, all_b, all_trial = [], [], [], [], []

    for (trial_id, _phase), recs in streams.items():
        recs_sorted = sorted(recs, key=lambda r: (int(r.get("sample_index", 0)), int(r.get("zarr_index", -1))))
        zpath = str(recs_sorted[0].get("zarr_path", ""))
        if not zpath:
            continue
        if zpath not in zarr_cache:
            zarr_cache[zpath] = zarr.open_group(zpath, mode="r")
        zg = zarr_cache[zpath]

        idx_arr = np.asarray([int(r["zarr_index"]) for r in recs_sorted], dtype=np.int64)
        tactile = np.asarray(zg["tactile_lr_norm"][idx_arr], dtype=np.float32)

        if "fx" in zg and "fy" in zg:
            fx = np.asarray(zg["fx"][idx_arr], dtype=np.float32)
            fy = np.asarray(zg["fy"][idx_arr], dtype=np.float32)
        else:
            aux_raw = np.asarray(zg["aux_feat"][idx_arr], dtype=np.float32)
            fx = aux_raw[:, 0]
            fy = aux_raw[:, 1]
        fz = np.asarray(zg["fz"][idx_arr], dtype=np.float32)
        aux = np.stack([fx, fy, fz], axis=1)

        if "x_mm" in zg and "y_mm" in zg:
            x = np.asarray(zg["x_mm"][idx_arr], dtype=np.float32)
            y = np.asarray(zg["y_mm"][idx_arr], dtype=np.float32)
        else:
            x = np.asarray(zg["cx"][idx_arr], dtype=np.float32)
            y = np.asarray(zg["cy"][idx_arr], dtype=np.float32)
        if "z_command_mm" in zg:
            z = np.asarray(zg["z_command_mm"][idx_arr], dtype=np.float32)
        else:
            z = np.asarray(zg["depth_mm"][idx_arr], dtype=np.float32)

        if predict_fz:
            tgt = np.stack([x, y, z, fz], axis=1)
        else:
            tgt = np.stack([x, y, z], axis=1)

        xs, as_, ys, bs = _build_windows_from_stream(
            tactile,
            aux,
            tgt,
            seq_len,
            stride,
            target_mode=target_mode,
            baseline_frames=baseline_frames,
        )
        all_x.extend(xs)
        all_aux.extend(as_)
        all_y.extend(ys)
        all_b.extend(bs)
        all_trial.extend([trial_id] * len(xs))

    if not all_x:
        raise RuntimeError("No sequence windows produced from zarr/index data")

    return (
        np.asarray(all_x, dtype=np.float32),
        np.asarray(all_aux, dtype=np.float32),
        np.asarray(all_y, dtype=np.float32),
        np.asarray(all_b, dtype=np.float32),
        np.asarray(all_trial),
    )


class CnnLstmPose(nn.Module):
    def __init__(self, lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.1, aux_dim: int = 3, out_dim: int = 3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=64 + int(aux_dim),
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden, int(out_dim)),
        )

    def forward(self, tactile_seq: torch.Tensor, aux_seq: torch.Tensor) -> torch.Tensor:
        b, t, _ = tactile_seq.shape
        x = tactile_seq.view(b * t, 1, 4, 4)
        s = self.cnn(x).view(b, t, -1)
        feat = torch.cat([s, aux_seq], dim=-1)
        out, _ = self.lstm(feat)
        return self.head(out[:, -1, :])


def _split_indices(trial_names: np.ndarray, mode: str, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(trial_names)
    if n < 2:
        idx = np.arange(n)
        return idx, idx

    if mode == "random":
        g = np.random.default_rng(seed)
        perm = np.arange(n)
        g.shuffle(perm)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        if len(train_idx) == 0:
            train_idx = val_idx
        return train_idx, val_idx

    uniq = sorted(set(trial_names.tolist()))
    rnd = random.Random(seed)
    rnd.shuffle(uniq)
    n_val_trial = max(1, int(len(uniq) * val_ratio))
    val_trials = set(uniq[:n_val_trial])

    train_idx = np.asarray([i for i, t in enumerate(trial_names.tolist()) if t not in val_trials], dtype=np.int64)
    val_idx = np.asarray([i for i, t in enumerate(trial_names.tolist()) if t in val_trials], dtype=np.int64)
    if len(train_idx) == 0:
        train_idx = val_idx
    if len(val_idx) == 0:
        val_idx = train_idx
    return train_idx, val_idx


def _gpu_mem_gib(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return f"alloc={alloc:.2f}GiB,resv={reserved:.2f}GiB"


def _make_loader(ds: TensorDataset, args, train: bool) -> DataLoader:
    kw = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "shuffle": train,
    }
    if args.num_workers > 0:
        kw["prefetch_factor"] = args.prefetch_factor
        kw["persistent_workers"] = args.persistent_workers
    return DataLoader(ds, **kw)


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    dx = (pred[:, 0] - target[:, 0]).abs().mean().item()
    dy = (pred[:, 1] - target[:, 1]).abs().mean().item()
    dz = (pred[:, 2] - target[:, 2]).abs().mean().item()
    xy = torch.sqrt((pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2).mean().item()
    out = {
        "x_mae_mm": dx,
        "y_mae_mm": dy,
        "z_mae_mm": dz,
        "xy_err_mm": xy,
    }
    if pred.shape[1] >= 4:
        out["fz_mae"] = (pred[:, 3] - target[:, 3]).abs().mean().item()
    return out


def _edge_sample_weight(target_abs: torch.Tensor, args) -> torch.Tensor:
    w = torch.ones(target_abs.shape[0], dtype=target_abs.dtype, device=target_abs.device)
    if args.edge_weight <= 1.0:
        return w
    if target_abs.shape[1] < 2:
        return w

    x = target_abs[:, 0]
    y = target_abs[:, 1]
    edge_x = x.abs() >= (args.sensor_x_limit - args.edge_margin_mm)
    edge_y = y.abs() >= (args.sensor_y_limit - args.edge_margin_mm)
    on_edge = edge_x | edge_y
    w = torch.where(on_edge, torch.full_like(w, args.edge_weight), w)

    corner_w = args.corner_weight if args.corner_weight is not None else args.edge_weight
    on_corner = edge_x & edge_y
    w = torch.where(on_corner, torch.full_like(w, corner_w), w)
    return w


def _compose_weighted_loss(pred: torch.Tensor, y: torch.Tensor, target_abs: torch.Tensor, args) -> torch.Tensor:
    per_sample = (
        args.w_x * nn.functional.smooth_l1_loss(pred[:, 0], y[:, 0], reduction="none")
        + args.w_y * nn.functional.smooth_l1_loss(pred[:, 1], y[:, 1], reduction="none")
        + args.w_z * nn.functional.smooth_l1_loss(pred[:, 2], y[:, 2], reduction="none")
    )
    if pred.shape[1] >= 4:
        per_sample = per_sample + args.w_fz * nn.functional.smooth_l1_loss(pred[:, 3], y[:, 3], reduction="none")

    sample_w = _edge_sample_weight(target_abs, args)
    return (per_sample * sample_w).mean()


def train_epoch(model, loader, optimizer, scaler, args, device, target_mean, target_std):
    model.train()
    loss_sum = 0.0
    n_steps = max(len(loader), 1)
    t0 = time.time()

    for step, (x_tac, x_aux, y_raw, baseline_raw) in enumerate(loader, 1):
        x_tac = x_tac.to(device, non_blocking=True)
        x_aux = x_aux.to(device, non_blocking=True)
        y_raw = y_raw.to(device, non_blocking=True)
        baseline_raw = baseline_raw.to(device, non_blocking=True)

        y = (y_raw - target_mean) / target_std
        target_abs = y_raw + baseline_raw if args.target_mode == "residual" else y_raw

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(args.amp and device.type == "cuda")):
            pred = model(x_tac, x_aux)
            loss = _compose_weighted_loss(pred, y, target_abs, args)

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        loss_sum += float(loss.item())
        if args.log_batch_every > 0 and (step % args.log_batch_every == 0 or step == n_steps):
            elapsed = time.time() - t0
            eta = (elapsed / step) * (n_steps - step)
            print(f"[TRAIN] {step}/{n_steps} loss={loss_sum/step:.4f} eta={eta/60:.1f}m")

    return loss_sum / n_steps


@torch.no_grad()
def validate_epoch(model, loader, args, device, target_mean, target_std):
    model.eval()
    loss_sum = 0.0
    n_steps = max(len(loader), 1)
    all_pred = []
    all_tgt = []

    for x_tac, x_aux, y_raw, baseline_raw in loader:
        x_tac = x_tac.to(device, non_blocking=True)
        x_aux = x_aux.to(device, non_blocking=True)
        y_raw = y_raw.to(device, non_blocking=True)
        baseline_raw = baseline_raw.to(device, non_blocking=True)

        y = (y_raw - target_mean) / target_std
        target_abs = y_raw + baseline_raw if args.target_mode == "residual" else y_raw

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(args.amp and device.type == "cuda")):
            pred = model(x_tac, x_aux)
            loss = _compose_weighted_loss(pred, y, target_abs, args)

        loss_sum += float(loss.item())

        pred_target = pred * target_std + target_mean
        if args.target_mode == "residual":
            pred_eval = pred_target + baseline_raw
            tgt_eval = y_raw + baseline_raw
        else:
            pred_eval = pred_target
            tgt_eval = y_raw

        all_pred.append(pred_eval.float().cpu())
        all_tgt.append(tgt_eval.float().cpu())

    pred = torch.cat(all_pred, dim=0)
    tgt = torch.cat(all_tgt, dim=0)
    return loss_sum / n_steps, _metrics(pred, tgt)


def parse_args():
    p = argparse.ArgumentParser(description="CNN-LSTM tactile sequence regressor")
    p.add_argument("--data-source", choices=["zarr", "csv"], default="zarr")
    p.add_argument("--data-dir", type=Path, default=Path("/home/user/sensor_training/preprocessing/preprocessing_data/eco20"))
    p.add_argument("--out-dir", type=Path, default=Path("/home/user/sensor_training/training_seq/runs"))

    p.add_argument("--phase", choices=["loading", "unloading", "all", "both"], default="both")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-depth-mm", type=float, default=0.5, help="zarr mode filter")
    p.add_argument("--min-abs-z", type=float, default=0.0, help="csv mode filter after scaling")

    p.add_argument("--csv-x-scale", type=float, default=1e-4)
    p.add_argument("--csv-y-scale", type=float, default=1e-4)
    p.add_argument("--csv-z-scale", type=float, default=1e-4)
    p.add_argument("--csv-fx-scale", type=float, default=1.0)
    p.add_argument("--csv-fy-scale", type=float, default=1.0)
    p.add_argument("--csv-fz-scale", type=float, default=1.0)

    p.add_argument("--split-mode", choices=["random", "trial"], default="trial")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--predict-fz", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--w-x", type=float, default=6.0)
    p.add_argument("--w-y", type=float, default=1.0)
    p.add_argument("--w-z", type=float, default=1.0)
    p.add_argument("--w-fz", type=float, default=1.0)

    p.add_argument("--edge-weight", type=float, default=1.0, help="sample loss weight for edge/corner regions")
    p.add_argument("--corner-weight", type=float, default=None, help="optional stronger weight when x/y are both near edge")
    p.add_argument("--sensor-x-limit", type=float, default=9.75)
    p.add_argument("--sensor-y-limit", type=float, default=9.75)
    p.add_argument("--edge-margin-mm", type=float, default=2.0)

    p.add_argument("--target-mode", choices=["absolute", "residual"], default="absolute")
    p.add_argument("--baseline-frames", type=int, default=4, help="first N frames in each window for baseline")

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log-batch-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    phase = "all" if args.phase == "both" else args.phase

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    t0 = time.time()
    if args.data_source == "csv":
        x_np, aux_np, y_np, b_np, trials_np = build_from_csv(
            data_dir=args.data_dir,
            phase=phase,
            seq_len=args.seq_len,
            stride=args.stride,
            min_abs_z=args.min_abs_z,
            csv_x_scale=args.csv_x_scale,
            csv_y_scale=args.csv_y_scale,
            csv_z_scale=args.csv_z_scale,
            csv_fx_scale=args.csv_fx_scale,
            csv_fy_scale=args.csv_fy_scale,
            csv_fz_scale=args.csv_fz_scale,
            target_mode=args.target_mode,
            baseline_frames=args.baseline_frames,
            predict_fz=args.predict_fz,
        )
    else:
        x_np, aux_np, y_np, b_np, trials_np = build_from_zarr(
            data_dir=args.data_dir,
            phase=phase,
            seq_len=args.seq_len,
            stride=args.stride,
            min_depth_mm=args.min_depth_mm,
            target_mode=args.target_mode,
            baseline_frames=args.baseline_frames,
            predict_fz=args.predict_fz,
        )
    print(f"[INFO] windows={len(x_np)} build_time={(time.time() - t0):.1f}s")

    train_idx, val_idx = _split_indices(trials_np, args.split_mode, args.val_ratio, args.seed)

    x = torch.from_numpy(x_np)
    aux = torch.from_numpy(aux_np)
    y = torch.from_numpy(y_np)
    b = torch.from_numpy(b_np)

    x_train = x[train_idx]
    aux_train = aux[train_idx]
    y_train = y[train_idx]
    b_train = b[train_idx]

    x_val = x[val_idx]
    aux_val = aux[val_idx]
    y_val = y[val_idx]
    b_val = b[val_idx]

    x_train_norm = _normalize_seq_feature_per_channel(x_train, x_train)
    x_val_norm = _normalize_seq_feature_per_channel(x_train, x_val)
    aux_train_norm = _normalize_seq_feature_per_channel(aux_train, aux_train)
    aux_val_norm = _normalize_seq_feature_per_channel(aux_train, aux_val)

    target_mean = y_train.mean(dim=0)
    target_std = y_train.std(dim=0).clamp_min(1e-6)

    train_ds = TensorDataset(x_train_norm, aux_train_norm, y_train, b_train)
    val_ds = TensorDataset(x_val_norm, aux_val_norm, y_val, b_val)

    train_loader = _make_loader(train_ds, args, train=True)
    val_loader = _make_loader(val_ds, args, train=False)

    print(
        f"[INFO] split={args.split_mode} train={len(train_ds)} val={len(val_ds)} "
        f"phase={phase} seq_len={args.seq_len} stride={args.stride} "
        f"target_mode={args.target_mode} predict_fz={args.predict_fz}"
    )

    model = CnnLstmPose(aux_dim=aux_train_norm.shape[-1], out_dim=y_train.shape[-1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    target_mean_dev = target_mean.to(device)
    target_std_dev = target_std.to(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_xy = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        e0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, scaler, args, device, target_mean_dev, target_std_dev)
        va_loss, m = validate_epoch(model, val_loader, args, device, target_mean_dev, target_std_dev)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, **m}
        history.append(row)

        msg = (
            f"[EPOCH {epoch:4d}/{args.epochs}] train={tr_loss:.4f} val={va_loss:.4f} "
            f"xy_err={m['xy_err_mm']:.3f}mm x_mae={m['x_mae_mm']:.3f} "
            f"y_mae={m['y_mae_mm']:.3f} z_mae={m['z_mae_mm']:.3f}"
        )
        if "fz_mae" in m:
            msg += f" fz_mae={m['fz_mae']:.3f}"
        msg += f" lr={scheduler.get_last_lr()[0]:.3e} time={(time.time() - e0)/60:.1f}m {_gpu_mem_gib(device)}"
        print(msg)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "target_mode": args.target_mode,
            "baseline_frames": int(args.baseline_frames),
            "out_dim": int(y_train.shape[-1]),
            "aux_dim": int(aux_train_norm.shape[-1]),
        }

        if va_loss < best_loss:
            best_loss = va_loss
            torch.save({**ckpt, "val_loss": best_loss}, args.out_dir / "best_loss.pt")
        if m["xy_err_mm"] < best_xy:
            best_xy = m["xy_err_mm"]
            torch.save({**ckpt, "xy_err_mm": best_xy}, args.out_dir / "best_xy.pt")

    torch.save({"epoch": args.epochs, "model": model.state_dict(), "args": vars(args)}, args.out_dir / "last.pt")
    with open(args.out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[DONE] best_loss={best_loss:.4f} best_xy={best_xy:.3f}mm out={args.out_dir}")


if __name__ == "__main__":
    main()
