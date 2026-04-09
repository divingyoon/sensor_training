
import os
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

# Avoid "Too many open files" when using many DataLoader workers / large preload.
# Use file_system sharing to reduce FD usage per worker on Linux.
mp.set_sharing_strategy("file_system")
from tqdm import tqdm
import json
import argparse
import numpy as np
from sklearn.metrics import r2_score

from training.data.dataset_unified import UnifiedTactileDataset
from training.data.dataset_zarr import ZarrDataset
from training.models.unified_model import UnifiedSensorModel
from training.models.mlp_baseline import MLPBaseline
from training.models.cnn_sr import CNNSR
from training.models.cnnlstm_sr import CNNLSTMSR
from training.models.cnn_bilstm import CNNBiLSTM
from training.models.sats_model import SATSModel
from training.models.sats_xy_multihead import SATSXYMultiHead
from training.models.tactile_transformer import TactileTransformer
from training.models.isoline_gnn import IsolineGNN
from training.models.tactile_gnn_gat import TactileGAT
from training.models.multi_head_field_model import MultiHeadFieldModel

def get_model(name, seq_len=50):
    if name == "unified": return UnifiedSensorModel(seq_len=seq_len)
    elif name == "mlp": return MLPBaseline()
    elif name == "cnn": return CNNSR()
    elif name == "cnnlstm": return CNNLSTMSR()
    elif name == "cnnbilstm": return CNNBiLSTM()
    elif name == "sats": return SATSModel()
    elif name == "sats_xy": return SATSXYMultiHead()
    elif name == "transformer": return TactileTransformer()
    elif name == "isoline_gnn": return IsolineGNN()
    elif name == "tactile_gnn_gat": return TactileGAT()
    elif name == "multi_head_field": return MultiHeadFieldModel(seq_len=seq_len)
    else: raise ValueError(f"Unknown model: {name}")

def calculate_metrics(preds, targets):
    mse = np.mean((preds - targets)**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets), axis=0)
    r2 = r2_score(targets, preds, multioutput='raw_values')
    return {"mse": mse.tolist(), "rmse": rmse.tolist(), "mae": mae.tolist(), "r2": r2.tolist()}


def apply_linear_calib(pred: torch.Tensor, args):
    """
    Optional post-calibration to correct systematic scale/offset between
    predicted [x,y] and ground truth. Coefficients are user-tunable.
    """
    if not getattr(args, "apply_linear_calib", False):
        return pred
    if pred.size(-1) < 2:
        return pred
    out = pred.clone()
    px = pred[:, 0]
    py = pred[:, 1]
    out[:, 0] = args.calib_x_ax * px + args.calib_x_by * py + args.calib_x_bias
    out[:, 1] = args.calib_y_ax * px + args.calib_y_by * py + args.calib_y_bias
    return out


def _forward_model(model_name, model, grid, iso):
    # grid: (B, T, 1, 4, 4), iso: (B, T, 17=[drift16 + radius1])
    if model_name == "unified":
        # Unified model expects iso dim 19; pad two zeros when metadata is unavailable.
        if iso.size(-1) < 19:
            pad = torch.zeros(iso.size(0), iso.size(1), 19 - iso.size(-1), device=iso.device, dtype=iso.dtype)
            iso_u = torch.cat([iso, pad], dim=-1)
        else:
            iso_u = iso[..., :19]
        res1, _ = model(grid, iso_u)
        return res1["xyz"] if isinstance(res1, dict) else res1

    if model_name == "cnnlstm":
        radius_seq = iso[:, :, -1:]
        return model(grid, radius_seq)

    if model_name in ["sats", "sats_xy"]:
        s16_seq = grid[:, :, 0].reshape(grid.size(0), grid.size(1), -1)
        return model(s16_seq)

    if model_name == "cnnbilstm":
        return model(grid)

    if model_name == "multi_head_field":
        force_vec, _ = model(grid)
        return force_vec

    if model_name == "mlp":
        s16 = grid[:, -1, 0].reshape(grid.size(0), -1)
        radius = iso[:, -1, -1:]
        return model(s16, radius)

    if model_name == "isoline_gnn":
        s16 = grid[:, -1, 0].reshape(grid.size(0), -1)
        radius = iso[:, -1, -1:]
        return model(s16, radius)

    if model_name == "tactile_gnn_gat":
        s16 = grid[:, -1, 0].reshape(grid.size(0), -1)
        return model(s16)

    if model_name in ["cnn", "transformer"]:
        s16 = grid[:, -1, 0].reshape(grid.size(0), -1)
        if model_name == "cnn":
            radius = iso[:, -1, -1:]
            return model(grid[:, -1], radius)
        return model(s16)

    raise ValueError(f"Unsupported model: {model_name}")


def _effective_batch_size(model_name: str, requested_bs: int) -> int:
    # LSTM-based models need much smaller batches than MLP/CNN.
    if model_name == "cnnlstm":
        return min(requested_bs, 1024)
    if model_name == "cnnbilstm":
        return min(requested_bs, 1024)
    if model_name == "multi_head_field":
        return min(requested_bs, 1024)
    if model_name == "unified":
        return min(requested_bs, 4096)
    if model_name in ["sats", "sats_xy"]:
        return min(requested_bs, 512)
    if model_name == "tactile_gnn_gat":
        return min(requested_bs, 2048)
    return requested_bs

def _resolve_device(force: str) -> torch.device:
    if force == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA device is not available")
        return torch.device("cuda")
    if force == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreloadedDataset(Dataset):
    def __init__(
        self,
        base_ds: Dataset,
        indices,
        store_device: torch.device,
        desc: str = "preload",
        preload_workers: int = 8,
        preload_batch_size: int = 2048,
    ):
        subset = torch.utils.data.Subset(base_ds, indices)
        loader_kwargs = {
            "batch_size": preload_batch_size,
            "shuffle": False,
            "num_workers": preload_workers,
            "pin_memory": (store_device.type == "cuda"),
        }
        if preload_workers > 0:
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(subset, **loader_kwargs)

        grid_chunks, iso_chunks, tgt_chunks = [], [], []
        for grid, iso, tgt in tqdm(loader, desc=desc):
            grid_chunks.append(grid.to(store_device, non_blocking=True))
            iso_chunks.append(iso.to(store_device, non_blocking=True))
            tgt_chunks.append(tgt.to(store_device, non_blocking=True))

        self.grid = torch.cat(grid_chunks, dim=0).contiguous()
        self.iso = torch.cat(iso_chunks, dim=0).contiguous()
        self.tgt = torch.cat(tgt_chunks, dim=0).contiguous()

    def __len__(self):
        return self.grid.size(0)

    def __getitem__(self, idx):
        return self.grid[idx], self.iso[idx], self.tgt[idx]


class ZarrSequenceDataset(Dataset):
    """
    Build sequence samples from preprocessed Zarr data.
    Sequence key: (trial_id, x_mm, y_mm), sorted by depth.
    """

    def __init__(self, zarr_path: str, seq_len: int = 50, stride: int = 5, phase: str = "all"):
        self.seq_len = seq_len
        self.stride = stride

        zds = ZarrDataset(zarr_path=zarr_path, split="all", phase=phase)
        tactile = zds.tactile_data.float()  # (N,16)
        radius = (zds.aux_data[:, 3:4] / 2.0).float()  # (N,1)
        cx = zds.cx_data.float()
        cy = zds.cy_data.float()
        depth = zds.depth_data.float()
        fz = zds.fz_data.float()
        trial_ids = zds.trial_ids

        groups = defaultdict(list)
        for i in range(len(trial_ids)):
            key = (str(trial_ids[i]), round(float(cx[i].item()), 3), round(float(cy[i].item()), 3))
            groups[key].append(i)

        self.samples = []
        for _, idxs in groups.items():
            idxs = sorted(idxs, key=lambda j: float(depth[j].item()))
            t = len(idxs)
            if t <= 0:
                continue
            if t <= seq_len:
                self.samples.append(idxs)
            else:
                max_start = t - seq_len
                for s in range(0, max_start + 1, stride):
                    self.samples.append(idxs[s : s + seq_len])

        self.tactile = tactile
        self.radius = radius
        self.cx = cx
        self.cy = cy
        self.depth = depth
        self.fz = fz

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idxs = self.samples[idx]
        t = len(idxs)

        s16 = self.tactile[idxs]  # (t,16)
        r = self.radius[idxs]     # (t,1)

        if t < self.seq_len:
            pad = self.seq_len - t
            s16 = torch.cat([s16, torch.zeros(pad, 16, dtype=s16.dtype)], dim=0)
            r = torch.cat([r, torch.zeros(pad, 1, dtype=r.dtype)], dim=0)

        s16 = s16[: self.seq_len]
        r = r[: self.seq_len]
        grid = s16.reshape(self.seq_len, 1, 4, 4)

        # iso layout expected by current model router: [:16]=aux seq, [-1]=radius
        iso = torch.zeros(self.seq_len, 17, dtype=s16.dtype)
        iso[:, :16] = s16
        iso[:, 16:17] = r

        last_i = idxs[-1]
        tgt = torch.zeros(6, dtype=s16.dtype)
        tgt[0] = self.cx[last_i]
        tgt[1] = self.cy[last_i]
        tgt[2] = self.depth[last_i]
        tgt[5] = self.fz[last_i]
        return grid, iso, tgt


def _resolve_zarr_path(data_dir: str, zarr_path: str = ""):
    if zarr_path:
        p = Path(zarr_path)
        return str(p)
    p = Path(data_dir)
    if p.suffix == ".zarr":
        return str(p)
    cands = sorted((p / "zarr_data").glob("*.zarr"))
    if cands:
        return str(cands[0])
    cands = sorted(p.glob("*.zarr"))
    if cands:
        return str(cands[0])
    return ""


def build_shared_data(args, device: torch.device):
    base_ds = None
    if args.data_source in ["auto", "zarr"]:
        zarr_path = _resolve_zarr_path(args.data_dir, args.zarr_path)
        if zarr_path:
            print(f"[INFO] data source: zarr ({zarr_path})")
            base_ds = ZarrSequenceDataset(
                zarr_path=zarr_path,
                seq_len=args.seq_len,
                stride=args.stride,
                phase=args.phase,
            )
        elif args.data_source == "zarr":
            raise RuntimeError(f"Requested --data-source zarr but no .zarr found under {args.data_dir}")

    if base_ds is None:
        print(f"[INFO] data source: csv ({args.data_dir})")
        base_ds = UnifiedTactileDataset(args.data_dir, seq_len=args.seq_len, augment=False)

    if len(base_ds) == 0:
        raise RuntimeError(f"Dataset at {args.data_dir} is empty. Check your data path!")

    total = len(base_ds)
    split = int(0.8 * total)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(total, dtype=np.int64)
    rng.shuffle(indices)
    train_indices = indices[:split].tolist()
    val_indices = indices[split:].tolist()

    if args.preload_vram and device.type != "cuda":
        raise RuntimeError("--preload-vram requires CUDA device")
    preload_device = device if args.preload_vram else torch.device("cpu")
    where = "VRAM" if args.preload_vram else "RAM"
    print(f"[INFO] Indexed {total:,} samples")
    print(f"[INFO] Preloading all samples to {where} once...")
    preloaded_ds = PreloadedDataset(
        base_ds,
        indices,
        store_device=preload_device,
        desc="preload all",
        preload_workers=args.preload_workers,
        preload_batch_size=args.preload_batch_size,
    )

    idx_device = preload_device
    train_idx = torch.as_tensor(train_indices, dtype=torch.long, device=idx_device)
    val_idx = torch.as_tensor(val_indices, dtype=torch.long, device=idx_device)
    print(
        f"[INFO] Train/Val split: {split:,}/{total-split:,} | "
        f"batch_size={args.batch_size}, preload_vram={args.preload_vram}, direct_batching=True"
    )
    return preloaded_ds, train_idx, val_idx


def _iter_batches(preloaded_ds, index_tensor, batch_size, shuffle, device, augment):
    if shuffle:
        order = torch.randperm(index_tensor.numel(), device=index_tensor.device)
        idx = index_tensor[order]
    else:
        idx = index_tensor

    num_batches = (idx.numel() + batch_size - 1) // batch_size
    for i in range(num_batches):
        bi = idx[i * batch_size : (i + 1) * batch_size]
        src_dev = preloaded_ds.grid.device
        if bi.device != src_dev:
            bi = bi.to(src_dev, non_blocking=True)

        grid = torch.index_select(preloaded_ds.grid, 0, bi)
        iso = torch.index_select(preloaded_ds.iso, 0, bi)
        tgt = torch.index_select(preloaded_ds.tgt, 0, bi)

        if augment:
            bsz = grid.size(0)
            flip_x = torch.rand(bsz, device=grid.device) > 0.5
            flip_y = torch.rand(bsz, device=grid.device) > 0.5
            if flip_x.any():
                grid[flip_x] = torch.flip(grid[flip_x], dims=[-1])
                tgt[flip_x, 0] = -tgt[flip_x, 0]
                tgt[flip_x, 3] = -tgt[flip_x, 3]
            if flip_y.any():
                grid[flip_y] = torch.flip(grid[flip_y], dims=[-2])
                tgt[flip_y, 1] = -tgt[flip_y, 1]
                tgt[flip_y, 4] = -tgt[flip_y, 4]
            grid = grid + torch.randn_like(grid) * 0.01

        if grid.device != device:
            grid = grid.to(device, non_blocking=True)
            iso = iso.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
        yield grid, iso, tgt, num_batches


def train_one_model(model_name, args, device, preloaded_ds, train_idx, val_idx):
    print(f"\n--- Training Model: {model_name} ---")
    print(f"  [INFO] device: {device}")
    effective_bs = _effective_batch_size(model_name, args.batch_size)
    if effective_bs != args.batch_size:
        print(f"  [INFO] batch_size override for {model_name}: {args.batch_size} -> {effective_bs} (OOM prevention)")

    model = get_model(model_name, args.seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_mae = float('inf')
    best_metrics = {}

    for epoch in range(args.epochs):
        model.train()
        train_iter = _iter_batches(
            preloaded_ds,
            train_idx,
            batch_size=effective_bs,
            shuffle=True,
            device=device,
            augment=True,
        )
        train_bar = tqdm(
            train_iter,
            total=(train_idx.numel() + effective_bs - 1) // effective_bs,
            desc=f"{model_name} train {epoch+1}/{args.epochs}",
            leave=False,
        )
        for grid, iso, tgt, _ in train_bar:
            optimizer.zero_grad()
            
            pred = _forward_model(model_name, model, grid, iso)
            loss = criterion(pred[:, :3], tgt[:, :3]) # Position loss
            if args.lambda_offdiag > 0.0:
                xc = pred[:, 0] - pred[:, 0].mean()
                yc = pred[:, 1] - pred[:, 1].mean()
                cov = (xc * yc).mean()
                loss = loss + args.lambda_offdiag * (cov * cov)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation & Metrics
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            val_iter = _iter_batches(
                preloaded_ds,
                val_idx,
                batch_size=effective_bs,
                shuffle=False,
                device=device,
                augment=False,
            )
            val_bar = tqdm(
                val_iter,
                total=(val_idx.numel() + effective_bs - 1) // effective_bs,
                desc=f"{model_name} val   {epoch+1}/{args.epochs}",
                leave=False,
            )
            for grid, iso, tgt, _ in val_bar:
                pred = _forward_model(model_name, model, grid, iso)
                pred_use = apply_linear_calib(pred, args)
                all_preds.append(pred_use[:, :3].cpu().numpy())
                all_targets.append(tgt[:, :3].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = calculate_metrics(all_preds, all_targets)
        avg_mae = np.mean(metrics["mae"])

        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            best_metrics = metrics
            save_path = os.path.join(args.out_dir, f"best_{model_name}.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "metrics": metrics,
                "model_name": model_name
            }, save_path)

        print(
            f"  [EPOCH {epoch+1:03d}/{args.epochs}] "
            f"val_mae={avg_mae:.4f}mm (best={best_val_mae:.4f}mm)"
        )

    print(f"Best MAE for {model_name}: {best_val_mae:.4f} mm")

    # Free model-specific memory before next experiment.
    del model
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="preprocessing/processed_data")
    parser.add_argument("--out-dir", type=str, default="training/runs_comparison")
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn", "unified", "cnnlstm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--preload-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--phase", choices=["loading", "unloading", "all"], default="all")
    # calibration (optional)
    parser.add_argument("--apply-linear-calib", action="store_true", help="Apply post linear calibration to [x,y] outputs")
    parser.add_argument("--calib-x-ax", type=float, default=1.0)
    parser.add_argument("--calib-x-by", type=float, default=0.23)
    parser.add_argument("--calib-x-bias", type=float, default=-1.06)
    parser.add_argument("--calib-y-ax", type=float, default=0.0)
    parser.add_argument("--calib-y-by", type=float, default=0.60)
    parser.add_argument("--calib-y-bias", type=float, default=1.80)
    # cross coupling regularization
    parser.add_argument("--lambda-offdiag", type=float, default=0.0, help="Penalty weight for cov(x_pred, y_pred)")
    parser.add_argument("--data-source", choices=["auto", "zarr", "csv"], default="auto")
    parser.add_argument("--zarr-path", type=str, default="")
    parser.add_argument("--preload-vram", action="store_true", default=True, help="Preload all samples into VRAM")
    parser.add_argument("--no-preload-vram", dest="preload_vram", action="store_false", help="Preload all samples into RAM")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="'auto'=cuda if available else cpu; 'cuda'=require GPU; 'cpu'=force cpu",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"[INFO] device: {device}")
    preloaded_ds, train_idx, val_idx = build_shared_data(args, device)

    results = {}
    for m in args.models:
        results[m] = train_one_model(m, args, device, preloaded_ds, train_idx, val_idx)
    
    with open(os.path.join(args.out_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)
