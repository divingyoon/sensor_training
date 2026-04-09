
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
import matplotlib.pyplot as plt

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
from training.utils.contact_geometry import contact_radius_tensor

def get_model(name, seq_len=50, heatmap_size=40):
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
    elif name == "multi_head_field": return MultiHeadFieldModel(seq_len=seq_len, heatmap_size=heatmap_size)
    else: raise ValueError(f"Unknown model: {name}")

GRID_STEP = 0.5
GRID_MIN = -9.75

def calculate_metrics(preds, targets):
    mse = np.mean((preds - targets)**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets), axis=0)
    r2 = r2_score(targets, preds, multioutput='raw_values')
    return {"mse": mse.tolist(), "rmse": rmse.tolist(), "mae": mae.tolist(), "r2": r2.tolist()}


def depth_bin_metrics(preds, targets, bin_edges):
    """
    preds, targets: np arrays (N,3) where [:,2]=depth
    bin_edges: list of edges including upper edge
    Returns list of dicts per bin.
    """
    results = []
    depth = targets[:, 2]
    xy_err = np.linalg.norm(preds[:, :2] - targets[:, :2], axis=1)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (depth >= lo) & (depth < hi)
        if mask.sum() == 0:
            results.append({"range": [lo, hi], "count": 0})
            continue
        mae = float(xy_err[mask].mean())
        succ = float((xy_err[mask] <= GRID_STEP).mean())
        results.append({"range": [float(lo), float(hi)], "count": int(mask.sum()), "xy_mae": mae, "success<=1cell": succ})
    return results


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


def _forward_model(model_name, model, grid, iso, return_field: bool = False):
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
        scalar_vec, field_map = model(grid)
        if return_field:
            return scalar_vec, field_map
        return scalar_vec

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
        return requested_bs  # allow full batch; user controls OOM
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


def _decode_xy_from_heatmap(fmap_logits: torch.Tensor, decode: str):
    """
    fmap_logits: (B,1,H,W)
    Returns (x_mm, y_mm) tensors (B,)
    """
    B, _, H, W = fmap_logits.shape
    device = fmap_logits.device
    dtype = fmap_logits.dtype
    xs = torch.arange(W, device=device, dtype=dtype) * GRID_STEP + GRID_MIN
    ys = torch.arange(H, device=device, dtype=dtype) * GRID_STEP + GRID_MIN

    if decode == "softargmax":
        prob = torch.softmax(fmap_logits.view(B, -1), dim=1)
        exp_x = torch.sum(prob * xs.view(1, 1, W).repeat(1, H, 1).view(1, -1), dim=1)
        exp_y = torch.sum(prob * ys.view(1, H, 1).repeat(1, 1, W).view(1, -1), dim=1)
        return exp_x, exp_y

    # argmax_refine: coarse argmax + local softmax 3x3 window
    flat_idx = torch.argmax(fmap_logits.view(B, -1), dim=1)
    iy = flat_idx // W
    ix = flat_idx % W
    # local window
    x0 = torch.clamp(ix - 1, 0, W - 1)
    x1 = torch.clamp(ix + 1, 0, W - 1)
    y0 = torch.clamp(iy - 1, 0, H - 1)
    y1 = torch.clamp(iy + 1, 0, H - 1)
    # gather 3x3
    coords_x = []
    coords_y = []
    weights = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            cx = torch.clamp(ix + dx, 0, W - 1)
            cy = torch.clamp(iy + dy, 0, H - 1)
            coords_x.append(xs[cx])
            coords_y.append(ys[cy])
            weights.append(fmap_logits[torch.arange(B, device=device), 0, cy, cx])
    w = torch.stack(weights, dim=1)
    prob_local = torch.softmax(w, dim=1)
    cx = torch.stack(coords_x, dim=1)
    cy = torch.stack(coords_y, dim=1)
    exp_x = (prob_local * cx).sum(dim=1)
    exp_y = (prob_local * cy).sum(dim=1)
    return exp_x, exp_y


def _build_soft_heatmap(
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    depth_mm: torch.Tensor,
    heatmap_size: int = 40,
    radius_model: str = "hertz",
    kernel: str = "gaussian",
    normalize: bool = False,
    indenter_radius_mm: float = 2.5,
    fallback_depth_mm: float = 1.0,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """
    Depth-aware soft target heatmap.
    Returns (B,1,H,W) on the same device/dtype as inputs.
    """
    device = x_mm.device
    dtype = x_mm.dtype
    xs = torch.arange(heatmap_size, device=device, dtype=dtype) * GRID_STEP + GRID_MIN
    ys = torch.arange(heatmap_size, device=device, dtype=dtype) * GRID_STEP + GRID_MIN
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    dx = xx.unsqueeze(0) - x_mm.view(-1, 1, 1)
    dy = yy.unsqueeze(0) - y_mm.view(-1, 1, 1)
    dist2 = dx * dx + dy * dy

    depth_eff = torch.where(depth_mm > 0, depth_mm, torch.full_like(depth_mm, fallback_depth_mm))
    a = contact_radius_tensor(depth_eff, R_mm=indenter_radius_mm, model=radius_model)

    a = a * sigma_scale

    if kernel == "linear":
        dist = torch.sqrt(dist2 + 1e-12)
        target = torch.relu(1.0 - dist / a.view(-1, 1, 1))
    else:
        denom = 2.0 * (a.view(-1, 1, 1) ** 2) + 1e-12
        target = torch.exp(-dist2 / denom)

    target = torch.where(depth_eff.view(-1, 1, 1) > 0, target, torch.zeros_like(target))

    if normalize:
        s = target.flatten(1).sum(dim=1, keepdim=True) + 1e-6
        target = target / s.view(-1, 1, 1)

    return target.unsqueeze(1)


def _save_overlay(batch_idx, fmap_logits, target_map, out_dir, prefix="val", max_samples=4):
    """
    Save overlay images (pred heatmap sigmoid vs target) for quick visual check.
    """
    os.makedirs(out_dir, exist_ok=True)
    prob = torch.sigmoid(fmap_logits).detach().cpu()
    tgt = target_map.detach().cpu()
    b = min(prob.size(0), max_samples)
    for i in range(b):
        plt.figure(figsize=(5, 2.2))
        plt.subplot(1, 2, 1)
        plt.title("pred")
        plt.imshow(prob[i, 0], origin="lower", cmap="inferno")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1, 2, 2)
        plt.title("target")
        plt.imshow(tgt[i, 0], origin="lower", cmap="viridis")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{prefix}_overlay_b{batch_idx}_i{i}.png")
        plt.savefig(path)
        plt.close()


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
        tgt = torch.zeros(4, dtype=s16.dtype)  # [x, y, z, fz]
        tgt[0] = self.cx[last_i]
        tgt[1] = self.cy[last_i]
        tgt[2] = self.depth[last_i]
        tgt[3] = self.fz[last_i]
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
            if flip_y.any():
                grid[flip_y] = torch.flip(grid[flip_y], dims=[-2])
                tgt[flip_y, 1] = -tgt[flip_y, 1]
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

    model = get_model(model_name, args.seq_len, args.heatmap_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bce_pos_weight = torch.tensor(args.fg_weight, device=device) if args.fg_weight > 0 else None
    bce_heatmap = None
    huber = None
    mse = nn.MSELoss()
    if model_name == "multi_head_field":
        if args.loss_xy == "bce":
            bce_heatmap = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        huber = nn.SmoothL1Loss(beta=args.huber_delta)

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

            if model_name == "multi_head_field":
                depth_mask = tgt[:, 2] > args.depth_min_for_label
                if not depth_mask.any():
                    continue  # skip batch with no contact

                grid_m = grid[depth_mask]
                iso_m = iso[depth_mask]
                tgt_m = tgt[depth_mask]

                pred, fmap = _forward_model(model_name, model, grid_m, iso_m, return_field=True)
                target_map = _build_soft_heatmap(
                    tgt_m[:, 0], tgt_m[:, 1], tgt_m[:, 2],
                    heatmap_size=args.heatmap_size,
                    radius_model=args.depth_radius_model,
                    kernel=args.depth_label_kernel,
                    normalize=args.normalize_heatmap,
                    indenter_radius_mm=args.indenter_radius_mm,
                    fallback_depth_mm=args.depth_fallback_mm,
                    sigma_scale=args.heatmap_sigma_scale,
                )
                if args.loss_xy == "bce" and args.normalize_heatmap:
                    maxv = target_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].clamp(min=1e-6)
                    target_map = target_map / maxv
                if args.loss_xy == "bce":
                    l_xy = bce_heatmap(fmap, target_map)
                else:
                    weight = 1.0 + args.wmse_alpha * target_map
                    l_xy = (weight * (fmap - target_map) ** 2).mean()

                pred_z = (pred[:, 0] - args.z_mean) / args.z_std
                tgt_z = (tgt_m[:, 2] - args.z_mean) / args.z_std
                pred_fz = (pred[:, 1] - args.fz_mean) / args.fz_std
                tgt_fz = (tgt_m[:, 3] - args.fz_mean) / args.fz_std

                if args.loss_z == "huber":
                    l_z = huber(pred_z, tgt_z)
                else:
                    l_z = mse(pred_z, tgt_z)
                if args.loss_fz == "huber":
                    l_fz = huber(pred_fz, tgt_fz)
                else:
                    l_fz = mse(pred_fz, tgt_fz)

                loss = args.lambda_xy * l_xy + args.lambda_z * l_z + args.lambda_fz * l_fz
            else:
                pred = _forward_model(model_name, model, grid, iso)
                loss = criterion(pred[:, :3], tgt[:, :3])  # Position loss
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
            val_losses = []
            for batch_idx, (grid, iso, tgt, _) in enumerate(val_bar):
                if model_name == "multi_head_field":
                    depth_mask = tgt[:, 2] > args.depth_min_for_label
                    if not depth_mask.any():
                        continue
                    grid_m = grid[depth_mask]
                    iso_m = iso[depth_mask]
                    tgt_m = tgt[depth_mask]

                    pred, fmap = _forward_model(model_name, model, grid_m, iso_m, return_field=True)
                    target_map = _build_soft_heatmap(
                        tgt_m[:, 0], tgt_m[:, 1], tgt_m[:, 2],
                        heatmap_size=args.heatmap_size,
                        radius_model=args.depth_radius_model,
                        kernel=args.depth_label_kernel,
                        normalize=args.normalize_heatmap,
                        indenter_radius_mm=args.indenter_radius_mm,
                        fallback_depth_mm=args.depth_fallback_mm,
                        sigma_scale=args.heatmap_sigma_scale,
                    )
                    if args.loss_xy == "bce" and args.normalize_heatmap:
                        maxv = target_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].clamp(min=1e-6)
                        target_map = target_map / maxv
                    if args.loss_xy == "bce":
                        l_xy = bce_heatmap(fmap, target_map)
                    else:
                        weight = 1.0 + args.wmse_alpha * target_map
                        l_xy = (weight * (fmap - target_map) ** 2).mean()
                    pred_z = (pred[:, 0] - args.z_mean) / args.z_std
                    tgt_z = (tgt_m[:, 2] - args.z_mean) / args.z_std
                    pred_fz = (pred[:, 1] - args.fz_mean) / args.fz_std
                    tgt_fz = (tgt_m[:, 3] - args.fz_mean) / args.fz_std
                    l_z = huber(pred_z, tgt_z) if args.loss_z == "huber" else mse(pred_z, tgt_z)
                    l_fz = huber(pred_fz, tgt_fz) if args.loss_fz == "huber" else mse(pred_fz, tgt_fz)
                    val_loss = args.lambda_xy * l_xy + args.lambda_z * l_z + args.lambda_fz * l_fz
                    if args.save_heatmap_overlay and batch_idx < args.overlay_batches:
                        _save_overlay(
                            batch_idx,
                            fmap,
                            target_map,
                            os.path.join(args.out_dir, "overlays"),
                            prefix=f"{model_name}_e{epoch+1}",
                            max_samples=args.overlay_samples,
                        )
                else:
                    pred = _forward_model(model_name, model, grid, iso)
                    val_loss = criterion(pred[:, :3], tgt[:, :3])

                # decode xy from heatmap for metrics if requested
                if model_name == "multi_head_field":
                    if args.decode_xy != "none":
                        x_dec, y_dec = _decode_xy_from_heatmap(fmap, args.decode_xy)
                        pred_xy = torch.stack([x_dec, y_dec], dim=1)
                    else:
                        flat = fmap.view(fmap.size(0), -1)
                        argmax = flat.argmax(dim=1)
                        iy = argmax // args.heatmap_size
                        ix = argmax % args.heatmap_size
                        xs = torch.arange(args.heatmap_size, device=fmap.device, dtype=fmap.dtype) * GRID_STEP + GRID_MIN
                        ys = torch.arange(args.heatmap_size, device=fmap.device, dtype=fmap.dtype) * GRID_STEP + GRID_MIN
                        pred_xy = torch.stack([xs[ix], ys[iy]], dim=1)
                    z_col = pred[:, 0:1]  # pred z
                    pred_concat = torch.cat([pred_xy, z_col], dim=1)
                else:
                    pred_concat = pred[:, :3]

                pred_use = apply_linear_calib(pred_concat, args)
                all_preds.append(pred_use[:, :3].cpu().numpy())
                all_targets.append(tgt_m[:, :3].cpu().numpy() if model_name == "multi_head_field" else tgt[:, :3].cpu().numpy())
                val_losses.append(val_loss.item())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = calculate_metrics(all_preds, all_targets)
        metrics["depth_bins"] = depth_bin_metrics(all_preds, all_targets, args.depth_bin_edges)
        avg_mae = np.mean(metrics["mae"])

        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            best_metrics = metrics
            tag = ""
            if args.use_depth_aware_label and model_name == "multi_head_field":
                tag = f"_dlabel-{args.depth_label_kernel}-{args.depth_radius_model}"
                tag += f"_xy{args.loss_xy}_z{args.loss_z}_fz{args.loss_fz}"
                if args.decode_xy != "none":
                    tag += f"_dec{args.decode_xy}"
                if args.normalize_heatmap:
                    tag += "_hnorm"
            save_path = os.path.join(args.out_dir, f"best_{model_name}{tag}.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "metrics": metrics,
                "model_name": model_name
            }, save_path)
            with open(os.path.join(args.out_dir, f"metrics_{model_name}{tag}.json"), "w") as f:
                json.dump(metrics, f, indent=2)

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
    # depth-aware heatmap option B
    parser.add_argument("--use-depth-aware-label", action="store_true", help="Use depth-aware soft heatmap (option B) for multi_head_field")
    parser.add_argument("--depth-label-kernel", choices=["gaussian", "linear"], default="gaussian")
    parser.add_argument("--depth-radius-model", choices=["hertz", "geom"], default="hertz")
    parser.add_argument("--indenter-radius-mm", type=float, default=2.5)
    parser.add_argument("--heatmap-size", type=int, default=40)
    parser.add_argument("--heatmap-sigma-scale", type=float, default=1.0, help="Scale factor applied to contact radius when building soft heatmap")
    parser.add_argument("--normalize-heatmap", action="store_true", help="Normalize soft heatmap to sum=1")
    parser.add_argument("--depth-fallback-mm", type=float, default=1.0, help="Depth to use when depth is missing/<=0")
    parser.add_argument("--depth-min-for-label", type=float, default=0.05, help="Ignore samples with depth <= this when computing heatmap losses/metrics")
    parser.add_argument("--fg-weight", type=float, default=5.0, help="Foreground weight for BCEWithLogitsLoss")
    parser.add_argument("--loss-xy", choices=["bce", "wmse"], default="bce")
    parser.add_argument("--wmse-alpha", type=float, default=4.0, help="weight = 1 + alpha * target for wmse")
    parser.add_argument("--loss-z", choices=["huber", "mse"], default="huber")
    parser.add_argument("--loss-fz", choices=["huber", "mse"], default="huber")
    parser.add_argument("--decode-xy", choices=["none", "softargmax", "argmax_refine"], default="none")
    parser.add_argument("--lambda-xy", type=float, default=1.0)
    parser.add_argument("--lambda-z", type=float, default=0.2)
    parser.add_argument("--lambda-fz", type=float, default=0.2)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--z-mean", type=float, default=0.0)
    parser.add_argument("--z-std", type=float, default=1.0)
    parser.add_argument("--fz-mean", type=float, default=0.0)
    parser.add_argument("--fz-std", type=float, default=1.0)
    parser.add_argument("--depth-bins", type=str, default="0.8,1.1,1.4,1.7", help="comma-separated depth bin edges (mm)")
    parser.add_argument("--save-heatmap-overlay", action="store_true", help="Save pred/target heatmap overlays on validation")
    parser.add_argument("--overlay-samples", type=int, default=4)
    parser.add_argument("--overlay-batches", type=int, default=1)
    parser.add_argument("--smoke-off-baseline", action="store_true", help="Run one forward path with depth-aware flag off to ensure compatibility")

    args = parser.parse_args()
    try:
        edges = [float(x) for x in args.depth_bins.split(",") if x.strip() != ""]
        if len(edges) >= 2:
            args.depth_bin_edges = edges + [float("inf")]
        else:
            args.depth_bin_edges = [0.0, float("inf")]
    except Exception:
        args.depth_bin_edges = [0.0, float("inf")]
    os.makedirs(args.out_dir, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"[INFO] device: {device}")
    preloaded_ds, train_idx, val_idx = build_shared_data(args, device)

    results = {}
    for m in args.models:
        results[m] = train_one_model(m, args, device, preloaded_ds, train_idx, val_idx)

        # quick smoke: flag off path still runs when depth-aware enabled
        if args.smoke_off_baseline and args.use_depth_aware_label and m == "multi_head_field":
            args.use_depth_aware_label = False
            try:
                # use a tiny subset to avoid heavy compute
                small_iter = _iter_batches(preloaded_ds, train_idx[:1], batch_size=1, shuffle=False, device=device, augment=False)
                grid, iso, tgt, _ = next(small_iter)
                _ = _forward_model(m, get_model(m, args.seq_len, args.heatmap_size).to(device), grid, iso)
                print("[SMOKE] flag-off forward pass succeeded.")
            except Exception as e:
                print(f"[SMOKE] flag-off forward failed: {e}")
            args.use_depth_aware_label = True
    
    with open(os.path.join(args.out_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)
