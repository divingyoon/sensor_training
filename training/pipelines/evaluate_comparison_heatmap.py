import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score

from training.data.dataset_unified import UnifiedTactileDataset
from training.pipelines.train_comparison import _forward_model, get_model, ZarrSequenceDataset, _resolve_zarr_path


GRID_STEP = 0.5
GRID_MIN = -9.75
GRID_MAX = 9.75
GRID_RANGE = np.arange(GRID_MIN, GRID_MAX + 1e-6, GRID_STEP)
N_GRID = len(GRID_RANGE)


def apply_linear_calib(pred: torch.Tensor, args):
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


def to_grid_idx(v: np.ndarray) -> np.ndarray:
    idx = np.round((v - GRID_MIN) / GRID_STEP).astype(np.int64)
    return np.clip(idx, 0, N_GRID - 1)


def evaluate_one(model_name: str, ckpt_path: Path, ds: UnifiedTactileDataset, val_idx: np.ndarray, batch_size: int, device: torch.device, args):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = get_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_map = np.full((N_GRID, N_GRID), np.nan, dtype=np.float32)
    y_map = np.full((N_GRID, N_GRID), np.nan, dtype=np.float32)
    z_map = np.full((N_GRID, N_GRID), np.nan, dtype=np.float32)
    xy_map = np.full((N_GRID, N_GRID), np.nan, dtype=np.float32)

    bucket = defaultdict(list)
    pred_list = []
    tgt_list = []
    with torch.no_grad():
        for s in tqdm(range(0, len(val_idx), batch_size), desc=f"{model_name} eval"):
            e = min(s + batch_size, len(val_idx))
            bi = val_idx[s:e]
            batch = [ds[int(i)] for i in bi]
            grid = torch.stack([b[0] for b in batch], dim=0).to(device)
            iso = torch.stack([b[1] for b in batch], dim=0).to(device)
            tgt = torch.stack([b[2] for b in batch], dim=0).to(device)

            pred = _forward_model(model_name, model, grid, iso)
            pred = apply_linear_calib(pred, args)
            err = (pred[:, :3] - tgt[:, :3]).abs().cpu().numpy()
            tgt_np = tgt[:, :3].cpu().numpy()
            pred_np = pred[:, :3].cpu().numpy()
            pred_list.append(pred_np)
            tgt_list.append(tgt_np)
            xi = to_grid_idx(tgt_np[:, 0])
            yi = to_grid_idx(tgt_np[:, 1])

            for i in range(err.shape[0]):
                bucket[(yi[i], xi[i])].append(err[i])

    rows = []
    for (yy, xx), vals in bucket.items():
        arr = np.stack(vals, axis=0)
        x_mae = float(arr[:, 0].mean())
        y_mae = float(arr[:, 1].mean())
        z_mae = float(arr[:, 2].mean())
        xy_err = float(np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2).mean())
        x_map[yy, xx] = x_mae
        y_map[yy, xx] = y_mae
        z_map[yy, xx] = z_mae
        xy_map[yy, xx] = xy_err
        rows.append(
            {
                "xi": int(xx),
                "yi": int(yy),
                "x_mm": float(GRID_RANGE[xx]),
                "y_mm": float(GRID_RANGE[yy]),
                "n_samples": int(arr.shape[0]),
                "x_mae": x_mae,
                "y_mae": y_mae,
                "z_mae": z_mae,
                "xy_err_mm": xy_err,
            }
        )

    pred_all = np.concatenate(pred_list, axis=0)
    tgt_all = np.concatenate(tgt_list, axis=0)
    diff = pred_all - tgt_all
    mse = np.mean(diff ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff), axis=0)
    r2 = r2_score(tgt_all, pred_all, multioutput="raw_values")
    xy_err = np.sqrt((diff[:, 0] ** 2) + (diff[:, 1] ** 2))
    global_metrics = {
        "mse_x": float(mse[0]),
        "mse_y": float(mse[1]),
        "mse_z": float(mse[2]),
        "rmse_x": float(rmse[0]),
        "rmse_y": float(rmse[1]),
        "rmse_z": float(rmse[2]),
        "mae_x": float(mae[0]),
        "mae_y": float(mae[1]),
        "mae_z": float(mae[2]),
        "mae_xyz_mean": float(np.mean(mae)),
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_z": float(r2[2]),
        "xy_err_mean": float(np.mean(xy_err)),
        "xy_err_p95": float(np.percentile(xy_err, 95)),
        "n_eval_samples": int(pred_all.shape[0]),
    }

    return {"x_mae": x_map, "y_mae": y_map, "z_mae": z_map, "xy_err": xy_map}, rows, global_metrics


def _fill_missing_neighbor_mean(arr: np.ndarray, max_iter: int = 200) -> np.ndarray:
    out = arr.copy()
    for _ in range(max_iter):
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            break
        updated = False
        ys, xs = np.where(nan_mask)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 1), min(out.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(out.shape[1], x + 2)
            nb = out[y0:y1, x0:x1]
            vals = nb[~np.isnan(nb)]
            if vals.size > 0:
                out[y, x] = float(vals.mean())
                updated = True
        if not updated:
            break
    return out


def save_heatmaps(
    maps: dict,
    out_dir: Path,
    prefix: str,
    fill_missing: str = "none",
    scale_limits: dict | None = None,
    error_vmin: float = 0.0,
    error_vmax: float = 1.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in maps.items():
        plot_arr = arr
        if fill_missing == "neighbor":
            plot_arr = _fill_missing_neighbor_mean(arr)
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#d9d9d9")
        cmap.set_over(color="red")
        vmin = error_vmin
        vmax = error_vmax
        if scale_limits is not None and key in scale_limits:
            vmin, vmax = scale_limits[key]
        if key == "z_mae":
            # Requested discrete bins for z:
            # 0.00~0.50 : 0.01 step, 0.50~1.00 : 0.1 step, >1.00 : red.
            b1 = np.arange(0.0, 0.5 + 1e-9, 0.01)
            b2 = np.arange(0.6, 1.0 + 1e-9, 0.1)
            boundaries = np.concatenate([b1, b2])
            norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        im = ax.imshow(
            plot_arr,
            origin="lower",
            cmap=cmap,
            extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
            norm=norm,
        )
        ax.set_title(f"{prefix} {key}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.colorbar(im, ax=ax, label="error")
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{prefix}_{key}.png", dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="preprocessing/processed_data")
    p.add_argument("--runs-dir", type=str, default="training/runs_comparison")
    p.add_argument("--models", nargs="+", default=["mlp", "cnnlstm", "sats"])
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--phase", choices=["loading", "unloading", "all"], default="loading")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--data-source", choices=["auto", "zarr", "csv"], default="auto")
    p.add_argument("--zarr-path", type=str, default="")
    p.add_argument("--eval-split", choices=["val", "all"], default="val")
    p.add_argument("--fill-missing", choices=["none", "neighbor"], default="none")
    p.add_argument("--shared-scale", action="store_true", default=True)
    p.add_argument("--no-shared-scale", dest="shared_scale", action="store_false")
    p.add_argument("--error-vmin", type=float, default=0.0)
    p.add_argument("--error-vmax", type=float, default=1.0)
    p.add_argument("--fixed-error-range", action="store_true", default=True)
    p.add_argument("--no-fixed-error-range", dest="fixed_error_range", action="store_false")
    # calibration
    p.add_argument("--apply-linear-calib", action="store_true", help="Apply post linear calibration to [x,y] outputs")
    p.add_argument("--calib-x-ax", type=float, default=1.0)
    p.add_argument("--calib-x-by", type=float, default=0.23)
    p.add_argument("--calib-x-bias", type=float, default=-1.06)
    p.add_argument("--calib-y-ax", type=float, default=0.0)
    p.add_argument("--calib-y-by", type=float, default=0.60)
    p.add_argument("--calib-y-bias", type=float, default=1.80)
    args = p.parse_args()

    device = torch.device(args.device)
    ds = None
    if args.data_source in ["auto", "zarr"]:
        zarr_path = _resolve_zarr_path(args.data_dir, args.zarr_path)
        if zarr_path:
            print(f"[INFO] data source: zarr ({zarr_path})")
            ds = ZarrSequenceDataset(
                zarr_path=zarr_path,
                seq_len=args.seq_len,
                stride=args.stride,
                phase=args.phase,
            )
        elif args.data_source == "zarr":
            raise RuntimeError(f"Requested --data-source zarr but no .zarr found under {args.data_dir}")

    if ds is None:
        print(f"[INFO] data source: csv ({args.data_dir})")
        ds = UnifiedTactileDataset(args.data_dir, seq_len=args.seq_len, augment=False)
    n = len(ds)
    split = int(0.8 * n)
    if args.eval_split == "all":
        val_idx = np.arange(0, n, dtype=np.int64)
    else:
        val_idx = np.arange(split, n, dtype=np.int64)

    out_root = Path(args.runs_dir) / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    model_maps = {}
    model_rows = {}
    model_global = {}

    for m in args.models:
        ckpt = Path(args.runs_dir) / f"best_{m}.pth"
        if not ckpt.exists():
            print(f"[WARN] skip {m}: checkpoint not found: {ckpt}")
            continue
        maps, rows, g = evaluate_one(m, ckpt, ds, val_idx, args.batch_size, device, args)
        model_maps[m] = maps
        model_rows[m] = rows
        model_global[m] = g
        summary[m] = {
            "n_grid_points": len(rows),
            "mean_xy_err_mm": float(np.nanmean(maps["xy_err"])),
            "mean_x_mae_mm": float(np.nanmean(maps["x_mae"])),
            "mean_y_mae_mm": float(np.nanmean(maps["y_mae"])),
            "mean_z_mae_mm": float(np.nanmean(maps["z_mae"])),
            **g,
        }
        print(f"[INFO] done: {m}")

    scale_limits = None
    if args.fixed_error_range:
        scale_limits = {k: (args.error_vmin, args.error_vmax) for k in ["x_mae", "y_mae", "z_mae", "xy_err"]}
    elif args.shared_scale and model_maps:
        keys = ["x_mae", "y_mae", "z_mae", "xy_err"]
        scale_limits = {}
        for k in keys:
            vals = []
            for m in model_maps:
                arr = model_maps[m][k]
                if args.fill_missing == "neighbor":
                    arr = _fill_missing_neighbor_mean(arr)
                v = arr[~np.isnan(arr)]
                if v.size:
                    vals.append(v)
            if vals:
                vv = np.concatenate(vals)
                scale_limits[k] = (float(np.nanmin(vv)), float(np.nanmax(vv)))

    import pandas as pd
    for m in model_maps:
        save_heatmaps(
            model_maps[m],
            out_root,
            m,
            fill_missing=args.fill_missing,
            scale_limits=scale_limits,
            error_vmin=args.error_vmin,
            error_vmax=args.error_vmax,
        )
        pd.DataFrame(model_rows[m]).to_csv(out_root / f"metrics_grid_{m}.csv", index=False)

    if model_global:
        rows = []
        for m in model_global:
            rows.append({"model": m, **model_global[m]})
        pd.DataFrame(rows).to_csv(out_root / "comparison_metrics.csv", index=False)

    with open(out_root / "summary_heatmap.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] saved: {out_root}")


if __name__ == "__main__":
    main()
