#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


REQUIRED_COLUMNS = ["X", "Y", "Z", "Fx", "Fy", "Fz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tactile preprocessing dataset from raw CSV files."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("/home/user/skin_ws/preprocessing/raw_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/user/skin_ws/preprocessing/preprocessing_data"))
    parser.add_argument("--glob", type=str, default="*.csv")
    parser.add_argument(
        "--xyz-scale",
        type=float,
        default=0.0,
        help="Scale factor for X/Y/Z. Use 0 for auto (1e-3 when values are large).",
    )
    parser.add_argument("--depth-step-mm", type=float, default=0.5)
    parser.add_argument("--radius-mm", type=float, default=3.0)
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--baseline-window", type=int, default=50)
    parser.add_argument("--filter-order", type=int, default=2)
    parser.add_argument("--filter-cutoff", type=float, default=0.08)
    parser.add_argument("--sigma-min-mm", type=float, default=0.3)
    parser.add_argument("--contact-threshold-ratio", type=float, default=0.2)
    parser.add_argument(
        "--canvas-size-mm",
        type=float,
        default=0.0,
        help="Fixed canvas half-size in mm (full canvas = 2x). 0 = auto (use full XY range). "
             "Set to sensor array half-size, e.g. 20.0 for a 40x40mm sensor.",
    )
    parser.add_argument(
        "--sensor-spacing-mm",
        type=float,
        default=6.5,
        help="Spacing between adjacent sensors in the 4x4 grid (mm). Default: 6.5.",
    )
    parser.add_argument(
        "--sensor-origin-x-mm",
        type=float,
        default=0.0,
        help="X position of Skin1 (top-left sensor) in stage frame (mm). Default: 0.0.",
    )
    parser.add_argument(
        "--sensor-origin-y-mm",
        type=float,
        default=0.0,
        help="Y position of Skin1 (top-left sensor) in stage frame (mm). Default: 0.0.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def detect_skin_columns(columns: List[str]) -> List[str]:
    skin_cols = [c for c in columns if c.lower().startswith("skin")]
    if not skin_cols:
        raise ValueError("No Skin* columns found in CSV.")
    return skin_cols


def check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def infer_material(trial_id: str) -> str:
    return trial_id.split("_")[0] if "_" in trial_id else trial_id


def resolve_xyz_scale(df: pd.DataFrame, user_scale: float) -> float:
    if user_scale > 0.0:
        return float(user_scale)
    max_abs = float(df[["X", "Y", "Z"]].abs().to_numpy(dtype=np.float64).max())
    return 1e-3 if max_abs > 1000.0 else 1.0


def lowpass_zero_phase(arr: np.ndarray, order: int, cutoff: float) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        return arr
    if arr.shape[0] < max(8, order * 4):
        return arr
    b, a = butter(order, cutoff, btype="low", analog=False)
    return filtfilt(b, a, arr, axis=0)


def split_loading_unloading(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    z = df["Z"].to_numpy(dtype=np.float64)
    if z.size == 0:
        return {"loading": df.copy(), "unloading": df.iloc[0:0].copy()}
    peak_idx = int(np.argmax(z))
    return {
        "loading": df.iloc[: peak_idx + 1].copy(),
        "unloading": df.iloc[peak_idx:].copy(),
    }


def format_depth_dir(depth_mm: float) -> str:
    return f"depth_{depth_mm:07.3f}mm"


def build_canvas_bounds(x_vals: np.ndarray, y_vals: np.ndarray, margin_mm: float = 6.0) -> Tuple[float, float, float, float]:
    xmin, xmax = float(np.min(x_vals)), float(np.max(x_vals))
    ymin, ymax = float(np.min(y_vals)), float(np.max(y_vals))

    if np.isclose(xmin, xmax):
        xmin -= margin_mm / 2.0
        xmax += margin_mm / 2.0
    else:
        xmin -= margin_mm
        xmax += margin_mm

    if np.isclose(ymin, ymax):
        ymin -= margin_mm / 2.0
        ymax += margin_mm / 2.0
    else:
        ymin -= margin_mm
        ymax += margin_mm

    return xmin, xmax, ymin, ymax


def build_fixed_canvas_bounds(
    x0_mm: float,
    y0_mm: float,
    canvas_size_mm: float,
) -> Tuple[float, float, float, float]:
    half = canvas_size_mm / 2.0
    return x0_mm - half, x0_mm + half, y0_mm - half, y0_mm + half


def build_sensor_positions(
    spacing_mm: float,
    origin_x_mm: float,
    origin_y_mm: float,
    n_rows: int = 4,
    n_cols: int = 4,
) -> List[List[float]]:
    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            positions.append([
                origin_x_mm + col * spacing_mm,
                origin_y_mm + row * spacing_mm,
            ])
    return positions


def make_hr_map(
    x0_mm: float,
    y0_mm: float,
    depth_mm: float,
    fz_n: float,
    radius_mm: float,
    map_size: int,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    sigma_min_mm: float,
) -> Tuple[np.ndarray, float, float]:
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    xs = np.linspace(x_min, x_max, map_size, dtype=np.float32)
    ys = np.linspace(y_min, y_max, map_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    d = float(max(depth_mm, 0.0))
    a = float(np.sqrt(max(2.0 * radius_mm * d - d * d, 0.0)))
    sigma = max(sigma_min_mm, 0.5 * a)
    p0 = float(max(fz_n, 0.0) if d > 0.0 else 0.0)

    rr2 = (xx - x0_mm) ** 2 + (yy - y0_mm) ** 2
    soft = p0 * np.exp(-rr2 / (2.0 * sigma * sigma))
    return soft.astype(np.float32), a, sigma


def estimate_contact_area_mm2(
    soft_map: np.ndarray,
    threshold_ratio: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> float:
    peak = float(np.max(soft_map))
    if peak <= 0.0:
        return 0.0

    th = peak * float(threshold_ratio)
    mask = soft_map >= th

    h, w = soft_map.shape
    dx = (x_bounds[1] - x_bounds[0]) / max(w - 1, 1)
    dy = (y_bounds[1] - y_bounds[0]) / max(h - 1, 1)
    return float(mask.sum() * dx * dy)


def normalize_channels(arr: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    norm = (arr - mu) / std
    return norm, mu, std


def make_phase_rows(
    phase_df: pd.DataFrame,
    skin_cols: List[str],
    baseline: np.ndarray,
    z_min_global: float,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if phase_df.empty:
        return pd.DataFrame(), np.array([]), np.array([])

    tactile_raw = phase_df[skin_cols].to_numpy(dtype=np.float64)
    tactile_bc = lowpass_zero_phase(tactile_raw - baseline, args.filter_order, args.filter_cutoff)

    aux_cols = ["X", "Y", "Fx", "Fy", "Fz"]
    aux = lowpass_zero_phase(phase_df[aux_cols].to_numpy(dtype=np.float64), args.filter_order, args.filter_cutoff)

    work = pd.DataFrame(
        np.concatenate([phase_df[["Z"]].to_numpy(dtype=np.float64), aux, tactile_bc], axis=1),
        columns=["Z", *aux_cols, *skin_cols],
    )

    work["depth_mm_raw"] = np.maximum(work["Z"] - z_min_global, 0.0)
    work["depth_mm"] = np.round(work["depth_mm_raw"] / args.depth_step_mm) * args.depth_step_mm
    work["repeat_idx"] = work.groupby("depth_mm", sort=False).cumcount()

    tactile_vals = work[skin_cols].to_numpy(dtype=np.float32)
    tactile_norm, tactile_mu, tactile_std = normalize_channels(tactile_vals)
    work["tactile_norm_blob"] = list(tactile_norm)
    return work.reset_index(drop=True), tactile_mu.astype(np.float32), tactile_std.astype(np.float32)


def save_vector_csv(path: Path, values: np.ndarray, header: List[str]) -> None:
    pd.DataFrame(np.asarray(values, dtype=np.float32).reshape(1, -1), columns=header).to_csv(path, index=False)


def save_map_csv(path: Path, values: np.ndarray) -> None:
    pd.DataFrame(np.asarray(values, dtype=np.float32)).to_csv(path, index=False, header=False)


def process_file(path: Path, args: argparse.Namespace, global_index_start: int) -> Tuple[int, List[Dict[str, object]]]:
    df = pd.read_csv(path)
    check_required_columns(df)
    skin_cols = detect_skin_columns(df.columns.tolist())

    df = df.dropna(subset=["Z", "X", "Y", "Fx", "Fy", "Fz", *skin_cols]).copy()
    if df.empty:
        return global_index_start, []

    xyz_scale = resolve_xyz_scale(df, args.xyz_scale)
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]].astype(np.float64) * xyz_scale

    phases = split_loading_unloading(df)
    z_min_global = float(df["Z"].min())

    base_n = min(max(1, args.baseline_window), len(df))
    baseline = df[skin_cols].iloc[:base_n].to_numpy(dtype=np.float64).mean(axis=0)

    # Global canvas bounds (fallback when canvas_size_mm == 0)
    x_min_global, x_max_global, y_min_global, y_max_global = build_canvas_bounds(
        df["X"].to_numpy(dtype=np.float32), df["Y"].to_numpy(dtype=np.float32)
    )

    # Sensor layout
    sensor_positions = build_sensor_positions(
        spacing_mm=args.sensor_spacing_mm,
        origin_x_mm=args.sensor_origin_x_mm,
        origin_y_mm=args.sensor_origin_y_mm,
    )
    # Detect dead channels (channels that are constant across entire file)
    skin_arr = df[skin_cols].to_numpy(dtype=np.float64)
    dead_ch_indices = [i for i, col in enumerate(skin_cols) if float(skin_arr[:, i].std()) < 1.0]

    trial_id = path.stem
    material = infer_material(trial_id)

    samples_written: List[Dict[str, object]] = []
    phase_stats: Dict[str, Dict[str, object]] = {}

    for phase_name in ["loading", "unloading"]:
        rows, tactile_mu, tactile_std = make_phase_rows(
            phase_df=phases[phase_name],
            skin_cols=skin_cols,
            baseline=baseline,
            z_min_global=z_min_global,
            args=args,
        )

        if rows.empty:
            phase_stats[phase_name] = {"num_samples": 0}
            continue

        phase_stats[phase_name] = {
            "num_samples": int(len(rows)),
            "depth_min_mm": float(rows["depth_mm"].min()),
            "depth_max_mm": float(rows["depth_mm"].max()),
            "tactile_mean": tactile_mu.tolist(),
            "tactile_std": tactile_std.tolist(),
        }

        for i in range(len(rows)):
            row = rows.iloc[i]
            sample_idx = global_index_start
            global_index_start += 1

            x = float(row["X"])
            y = float(row["Y"])
            depth_mm = float(row["depth_mm"])
            depth_raw_mm = float(row["depth_mm_raw"])
            repeat_idx = int(row["repeat_idx"])
            z_cmd = float(row["Z"])
            fx = float(row["Fx"])
            fy = float(row["Fy"])
            fz = float(row["Fz"])

            sample_dir = args.out_dir / material / trial_id / phase_name / format_depth_dir(depth_mm) / f"rep_{repeat_idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            if args.canvas_size_mm > 0.0:
                x_min, x_max, y_min, y_max = build_fixed_canvas_bounds(x, y, args.canvas_size_mm)
            else:
                x_min, x_max = x_min_global, x_max_global
                y_min, y_max = y_min_global, y_max_global

            hr_map, a_mm, sigma_mm = make_hr_map(
                x0_mm=x,
                y0_mm=y,
                depth_mm=depth_mm,
                fz_n=fz,
                radius_mm=args.radius_mm,
                map_size=args.map_size,
                x_bounds=(x_min, x_max),
                y_bounds=(y_min, y_max),
                sigma_min_mm=args.sigma_min_mm,
            )

            area_mm2 = estimate_contact_area_mm2(hr_map, args.contact_threshold_ratio, (x_min, x_max), (y_min, y_max))

            tactile_lr = row[skin_cols].to_numpy(dtype=np.float32)
            tactile_lr_norm = np.asarray(row["tactile_norm_blob"], dtype=np.float32)
            aux_feat = np.array([fx, fy, depth_mm, args.radius_mm], dtype=np.float32)

            np.save(sample_dir / "tactile_lr.npy", tactile_lr)
            np.save(sample_dir / "tactile_lr_norm.npy", tactile_lr_norm)
            np.save(sample_dir / "aux_feat.npy", aux_feat)
            np.save(sample_dir / "hr_contact_map.npy", hr_map)

            save_vector_csv(sample_dir / "tactile_lr.csv", tactile_lr, skin_cols)
            save_vector_csv(sample_dir / "tactile_lr_norm.csv", tactile_lr_norm, skin_cols)
            save_vector_csv(sample_dir / "aux_feat.csv", aux_feat, ["fx_N", "fy_N", "depth_mm", "indenter_radius_mm"])
            save_map_csv(sample_dir / "hr_contact_map.csv", hr_map)

            meta = {
                "material": material,
                "trial_id": trial_id,
                "phase": phase_name,
                "sample_index": int(sample_idx),
                "sample_in_phase": int(i),
                "depth_bin_mm": depth_mm,
                "repeat_idx": repeat_idx,
                "indenter_type": "sphere",
                "indenter_radius_mm": float(args.radius_mm),
                "depth_mm": depth_mm,
                "depth_raw_mm": depth_raw_mm,
                "depth_step_mm": float(args.depth_step_mm),
                "z_command_mm": z_cmd,
                "fx_N": fx,
                "fy_N": fy,
                "fz_N": fz,
                "contact_center_x_mm": x,
                "contact_center_y_mm": y,
                "contact_area_mm2": area_mm2,
                "pseudo_contact_radius_mm": a_mm,
                "pseudo_sigma_mm": sigma_mm,
                "xyz_scale_applied": xyz_scale,
                "skin_channels": skin_cols,
                "baseline_window": int(base_n),
                "map_size": int(args.map_size),
                "map_x_bounds_mm": [x_min, x_max],
                "map_y_bounds_mm": [y_min, y_max],
                "filter": {
                    "type": "butterworth_zero_phase" if SCIPY_AVAILABLE else "none",
                    "order": int(args.filter_order),
                    "cutoff": float(args.filter_cutoff),
                },
                "dead_channel_indices": dead_ch_indices,
                "sensor_grid_shape": [4, 4],
                "sensor_spacing_mm": float(args.sensor_spacing_mm),
                "sensor_origin_x_mm": float(args.sensor_origin_x_mm),
                "sensor_origin_y_mm": float(args.sensor_origin_y_mm),
                "sensor_positions_mm": sensor_positions,
                "canvas_size_mm": float(args.canvas_size_mm) if args.canvas_size_mm > 0.0 else None,
            }

            with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            samples_written.append(
                {
                    "material": material,
                    "trial_id": trial_id,
                    "phase": phase_name,
                    "sample_index": int(sample_idx),
                    "sample_in_phase": int(i),
                    "depth_bin_mm": depth_mm,
                    "repeat_idx": repeat_idx,
                    "sample_dir": str(sample_dir),
                }
            )

    trial_stats = {
        "material": material,
        "trial_id": trial_id,
        "source_csv": str(path),
        "num_samples": int(len(samples_written)),
        "num_skin_channels": int(len(skin_cols)),
        "skin_channels": skin_cols,
        "loading_rows_raw": int(len(phases["loading"])),
        "unloading_rows_raw": int(len(phases["unloading"])),
        "xyz_scale_applied": xyz_scale,
        "depth_step_mm": float(args.depth_step_mm),
        "phase_stats": phase_stats,
    }

    with open(args.out_dir / f"trial_stats_{trial_id}.json", "w", encoding="utf-8") as f:
        json.dump(trial_stats, f, ensure_ascii=False, indent=2)

    return global_index_start, samples_written


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for child in args.out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    csv_files = sorted(args.raw_dir.glob(args.glob))
    if not csv_files:
        raise FileNotFoundError(f"No files found: {args.raw_dir}/{args.glob}")

    next_idx = 0
    all_samples: List[Dict[str, object]] = []

    for csv_path in csv_files:
        next_idx, samples = process_file(csv_path, args, global_index_start=next_idx)
        all_samples.extend(samples)

    summary = {
        "num_trials": len(csv_files),
        "num_samples_total": len(all_samples),
        "raw_dir": str(args.raw_dir),
        "out_dir": str(args.out_dir),
        "radius_mm": args.radius_mm,
        "depth_step_mm": float(args.depth_step_mm),
        "map_size": args.map_size,
        "scipy_available": SCIPY_AVAILABLE,
        "samples": all_samples,
    }

    with open(args.out_dir / "dataset_index.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    by_material: Dict[str, List[Dict[str, object]]] = {}
    for s in all_samples:
        by_material.setdefault(s["material"], []).append(s)

    for material, samples in by_material.items():
        material_idx = {
            "material": material,
            "num_samples": len(samples),
            "trials": sorted({s["trial_id"] for s in samples}),
            "samples": samples,
        }
        with open(args.out_dir / f"material_index_{material}.json", "w", encoding="utf-8") as f:
            json.dump(material_idx, f, ensure_ascii=False, indent=2)

    print(f"[DONE] trials={len(csv_files)} samples={len(all_samples)} materials={len(by_material)} out={args.out_dir}")


if __name__ == "__main__":
    main()
