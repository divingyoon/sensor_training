"""
Train x_c, y_c, z, Fx, Fy, Fz regressors for a selected sensing point (p1/p3/p7/p9)
using merged_data.csv logs already generated in acc_v2/logs.

How to use
 - Set TARGET_POINT below (e.g., "p1", "p3", "p7", "p9").
 - Optionally adjust FEATURES_MODE ("sumdiff" or "all16") and RIDGE_ALPHA.
 - Run: python acc_v2/learning_based/train_fx_fy_fz.py

No external ML libs are required (uses numpy + pandas).
The model is ridge regression with closed-form solution.
"""

import os
import glob
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


# ==========================
# Global configuration
# ==========================

# Select which point to train: "p1", "p2", ..., "p9"
TARGET_POINT = "p9"

# Logs directory containing p*_*/merged_data.csv
LOGS_DIR = r"C:\\Users\\SORO2\\Desktop\\skin_ws\\acc_v2\\logs"

# Feature set: "sumdiff" (u,v,w from the point's 2x2 sensors)
#              "all16"   (all s1..s16)
FEATURES_MODE = "all16"

# Ridge regularization strength (L2). Use small positive value.
RIDGE_ALPHA = 1e-2

# Train/validation split ratio (by rows after concatenation)
VAL_RATIO = 0.2
RANDOM_SEED = 42

# Optional: quality filter on tz residual (if present). Set to None to disable.
TZ_RESID_ABS_MAX = 0.02  # e.g., 0.01~0.02 (units of torque)

# Where to save trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Convert laser z (mm) to meters for training target
CONVERT_Z_MM_TO_M = True

# Use AFD50 shear labels for fx, fy (instead of torque-derived fx,fy written in merged CSV)
USE_AFD50_SHEAR_LABELS = False

# Two-stage option: Stage1 predicts [x_c,y_c,z_m] from sensors; Stage2 predicts [fx,fy,fz]
# from sensors concatenated with (x_c,y_c). Training uses teacher forcing for Stage2.
USE_TWO_STAGE = True

# Filtering and feature augmentation options for comparison
# Minimum (|x_c|+|y_c|) in meters to keep for training; set None to disable
MIN_POS_SUM_M = 0.0008  # e.g., 0.0005
# Minimum (|fx|+|fy|) to keep for training; set None to disable
MIN_SHEAR_FORCE_SUM = 0.1 # e.g., 0.1
# Append uvw to all16 features
ADD_UVW_IN_ALL16 = True
# Add uvw*z interactions to Stage2 features (teacher forcing z)
ADD_Z_INTERACTIONS = True


# ==========================
# Point-to-sensor mapping
# ==========================

SENSOR_MAPPING: Dict[int, List[str]] = {
    1: ["s1",  "s2",  "s5",  "s6"],
    2: ["s2",  "s3",  "s6",  "s7"],
    3: ["s3",  "s4",  "s7",  "s8"],
    4: ["s5",  "s6",  "s9",  "s10"],
    5: ["s6",  "s7",  "s10", "s11"],
    6: ["s7",  "s8",  "s11", "s12"],
    7: ["s9",  "s10", "s13", "s14"],
    8: ["s10", "s11", "s14", "s15"],
    9: ["s11", "s12", "s15", "s16"],
}


# ==========================
# Utilities
# ==========================

def _parse_point_id(point_str: str) -> int:
    if not point_str.startswith("p"):
        raise ValueError(f"Invalid TARGET_POINT: {point_str}")
    pid = int(point_str[1:])
    if pid not in range(1, 10):
        raise ValueError(f"TARGET_POINT must be p1..p9, got {point_str}")
    return pid


def list_point_runs(logs_dir: str, point_str: str) -> List[str]:
    # Match directories starting with e.g., "p7_"
    patt = os.path.join(logs_dir, f"{point_str}_*")
    return [d for d in glob.glob(patt) if os.path.isdir(d)]


def load_merged_csvs(run_dirs: List[str]) -> pd.DataFrame:
    frames = []
    for d in sorted(run_dirs):
        csv_path = os.path.join(d, "merged_data.csv")
        if not os.path.exists(csv_path):
            print(f"  - Skip (no merged_data.csv): {d}")
            continue
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            frames.append(df)
            print(f"  - Loaded: {csv_path}  rows={len(df)}")
        except Exception as e:
            print(f"  - Error reading {csv_path}: {e}")
    if not frames:
        raise FileNotFoundError("No merged_data.csv files found for the selected point")
    out = pd.concat(frames, ignore_index=True)
    print(f"Total rows concatenated: {len(out)}")
    return out


@dataclass
class Standardizer:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.0
        return cls(mean, std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        return Xs * self.scale_ + self.mean_


def ridge_closed_form(X: np.ndarray, Y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Solve (X, Y) -> weights W and bias b for multiout ridge without penalizing bias.
    X: (n, d), Y: (n, k)
    Returns W: (d, k), b: (k,)
    """
    # Augment with ones for bias, but do not penalize bias
    n, d = X.shape
    X_aug = np.hstack([X, np.ones((n, 1))])  # (n, d+1)
    I = np.eye(d + 1)
    I[-1, -1] = 0.0  # do not penalize bias
    A = X_aug.T @ X_aug + alpha * I
    B = X_aug.T @ Y
    theta = np.linalg.pinv(A) @ B  # (d+1, k)
    W = theta[:-1, :]
    b = theta[-1, :]
    return W, b


def predict_linear(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


def build_features_sumdiff(df: pd.DataFrame, point_id: int) -> np.ndarray:
    sA, sB, sC, sD = SENSOR_MAPPING[point_id]
    a = df[sA].to_numpy(dtype=float)
    b = df[sB].to_numpy(dtype=float)
    c = df[sC].to_numpy(dtype=float)
    d = df[sD].to_numpy(dtype=float)
    # Sum/diff features to reduce collinearity
    u = a + b + c + d
    v = (b + d) - (a + c)
    w = (c + d) - (a + b)
    X = np.stack([u, v, w], axis=1)
    return X


def build_features_all16(df: pd.DataFrame) -> np.ndarray:
    s_cols = [f"s{i}" for i in range(1, 17)]
    Xs = df[s_cols].to_numpy(dtype=float)
    return Xs


def select_valid_rows(df: pd.DataFrame) -> pd.Series:
    # Drop rows with all-zero sensors
    s_cols = [f"s{i}" for i in range(1, 17)]
    sensors = df[s_cols].fillna(0.0)
    nonzero = sensors.sum(axis=1) != 0.0
    # Drop rows with missing targets (require x_c, y_c, z (laser), fx, fy, fz)
    z_col = "z_displacement_mm_laser"
    fx_col = "fx_afd50" if (USE_AFD50_SHEAR_LABELS and "fx_afd50" in df.columns) else "fx"
    fy_col = "fy_afd50" if (USE_AFD50_SHEAR_LABELS and "fy_afd50" in df.columns) else "fy"
    need_cols = ["x_c", "y_c", z_col, fx_col, fy_col, "fz"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns in merged CSV: {miss}")
    targets_ok = df[need_cols].notna().all(axis=1)
    mask = nonzero & targets_ok
    # Optional filter by tz residual (quality)
    if TZ_RESID_ABS_MAX is not None and "tz_resid" in df.columns:
        mask = mask & (df["tz_resid"].abs() <= TZ_RESID_ABS_MAX)
    return mask


def train_and_eval(X: np.ndarray, Y: np.ndarray, target_names: List[str], alpha: float, val_ratio: float, seed: int):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr, X_val = X[tr_idx], X[val_idx]
    Y_tr, Y_val = Y[tr_idx], Y[val_idx]

    Xs = Standardizer.fit(X_tr)
    Ys = Standardizer.fit(Y_tr)
    X_tr_s, X_val_s = Xs.transform(X_tr), Xs.transform(X_val)
    Y_tr_s = Ys.transform(Y_tr)

    W, b = ridge_closed_form(X_tr_s, Y_tr_s, alpha)

    # Predictions (inverse scale)
    Y_val_pred_s = X_val_s @ W + b
    Y_val_pred = Ys.inverse_transform(Y_val_pred_s)

    # Metrics
    metrics: Dict[str, Dict[str, Any]] = {}
    for k, name in enumerate(target_names):
        y_true = Y_val[:, k]
        y_pred = Y_val_pred[:, k]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan))
        metrics[name] = {"MAE": mae, "R2": r2, "N": int(len(y_true))}
        print(f"  {name}: MAE={mae:.4f}, R2={r2:.4f}, N={len(y_true)}")

    # Print brief model summary
    print("\nModel summary (standardized space):")
    print(f"  W shape: {W.shape}, b shape: {b.shape}, alpha={alpha}")

    return {
        "W": W,
        "b": b,
        "Xs": Xs,
        "Ys": Ys,
        "metrics": metrics,
    }


def train_two_stage(
    X: np.ndarray,
    Y_full: np.ndarray,
    alpha: float,
    val_ratio: float,
    seed: int,
    base_feature_names: List[str],
    uvw: np.ndarray | None = None,
):
    Y1 = Y_full[:, :3]  # [x_c, y_c, z_m]
    Y2 = Y_full[:, 3:]  # [fx, fy, fz]

    # Stage1: sensors -> [x_c,y_c,z]
    res1 = train_and_eval(X, Y1, ["x_c", "y_c", "z_m"], alpha, val_ratio, seed)

    # Stage2: sensors + (x_c,y_c) [+ uvw*z] -> [fx,fy,fz] (teacher forcing during training)
    parts = [X, Y1[:, :2]]
    if uvw is not None:
        z = Y1[:, 2:3]
        parts.append(uvw * z)
    X2 = np.hstack(parts)
    res2 = train_and_eval(X2, Y2, ["fx", "fy", "fz"], alpha, val_ratio, seed)

    model = {
        "stage1": {
            "feature_names": base_feature_names,
            "target_names": ["x_c", "y_c", "z_m"],
            "W": res1["W"].tolist(),
            "b": res1["b"].tolist(),
            "x_scaler": {"mean": res1["Xs"].mean_.tolist(), "scale": res1["Xs"].scale_.tolist()},
            "y_scaler": {"mean": res1["Ys"].mean_.tolist(), "scale": res1["Ys"].scale_.tolist()},
            "metrics": res1["metrics"],
        },
        "stage2": {
            "feature_names": base_feature_names + ["x_c", "y_c"],
            "target_names": ["fx", "fy", "fz"],
            "W": res2["W"].tolist(),
            "b": res2["b"].tolist(),
            "x_scaler": {"mean": res2["Xs"].mean_.tolist(), "scale": res2["Xs"].scale_.tolist()},
            "y_scaler": {"mean": res2["Ys"].mean_.tolist(), "scale": res2["Ys"].scale_.tolist()},
            "metrics": res2["metrics"],
        }
    }
    return model


def _feature_names(mode: str) -> List[str]:
    if mode == "sumdiff":
        return ["u", "v", "w"]
    elif mode == "all16":
        return [*(f"s{i}" for i in range(1, 17))]
    else:
        raise ValueError(f"Unknown FEATURES_MODE: {mode}")


def save_model_json(
    out_path: str,
    point_str: str,
    features_mode: str,
    feature_names: List[str],
    sensor_map: List[str],
    runs_used: List[str],
    n_rows: int,
    target_names: List[str] = None,
    W: np.ndarray = None,
    b: np.ndarray = None,
    Xs: Standardizer = None,
    Ys: Standardizer = None,
    alpha: float = RIDGE_ALPHA,
    metrics: Dict[str, Dict[str, Any]] = None,
    two_stage: Dict[str, Any] = None,
) -> None:
    payload = {
        "target_point": point_str,
        "features_mode": features_mode,
        "feature_names": feature_names,
        "sensor_mapping": sensor_map,
        "ridge_alpha": float(alpha),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs_used,
        "n_rows": int(n_rows),
    }
    if two_stage is not None:
        payload["model_type"] = "two_stage"
        payload["stages"] = two_stage
        payload["target_names"] = ["x_c", "y_c", "z_m", "fx", "fy", "fz"]
    else:
        payload["model_type"] = "single"
        payload["target_names"] = target_names
        payload["W"] = W.tolist()
        payload["b"] = b.tolist()
        payload["x_scaler"] = {"mean": Xs.mean_.tolist(), "scale": Xs.scale_.tolist()}
        payload["y_scaler"] = {"mean": Ys.mean_.tolist(), "scale": Ys.scale_.tolist()}
        payload["metrics"] = metrics
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved model: {out_path}")


def main() -> None:
    point_str = TARGET_POINT.lower()
    pid = _parse_point_id(point_str)
    print(f"Selected point: {point_str}  (id={pid})")

    runs = list_point_runs(LOGS_DIR, point_str)
    if not runs:
        raise RuntimeError(f"No runs found for {point_str} in {LOGS_DIR}")
    print(f"Found {len(runs)} run directories")

    df = load_merged_csvs(runs)

    # Build base features and uvw for augmentations
    uvw_all = build_features_sumdiff(df, pid)
    if FEATURES_MODE == "sumdiff":
        X = uvw_all
        feat_desc = "sumdiff[u,v,w]"
        sensor_map = SENSOR_MAPPING[pid]
    elif FEATURES_MODE == "all16":
        X = build_features_all16(df)
        if ADD_UVW_IN_ALL16:
            X = np.hstack([X, uvw_all])
            feat_desc = "all s1..s16 + uvw"
        else:
            feat_desc = "all s1..s16"
        sensor_map = [f"s{i}" for i in range(1, 17)]
    else:
        raise ValueError(f"Unknown FEATURES_MODE: {FEATURES_MODE}")

    # Targets: x_c [m], y_c [m], z [m], fx, fy, fz
    z_col = "z_displacement_mm_laser"
    needed = {"x_c", "y_c", z_col, "fz"}
    # Choose shear labels
    fx_col = "fx_afd50" if (USE_AFD50_SHEAR_LABELS and "fx_afd50" in df.columns) else "fx"
    fy_col = "fy_afd50" if (USE_AFD50_SHEAR_LABELS and "fy_afd50" in df.columns) else "fy"
    needed.update({fx_col, fy_col})
    if not needed.issubset(df.columns):
        raise RuntimeError(f"Missing required target columns: {sorted(list(needed - set(df.columns)))}")
    Xc = df[["x_c", "y_c"]].to_numpy(dtype=float)
    z_vals = df[[z_col]].to_numpy(dtype=float).reshape(-1)
    if CONVERT_Z_MM_TO_M:
        z_vals = z_vals / 1000.0
    F = df[[fx_col, fy_col, "fz"]].to_numpy(dtype=float)

    mask = select_valid_rows(df)
    m = mask.to_numpy()
    X = X[m]
    Xc = Xc[mask.to_numpy()]
    z_vals = z_vals[mask.to_numpy()]
    F = F[mask.to_numpy()]
    Y = np.column_stack([Xc, z_vals, F])  # [x_c, y_c, z_m, fx, fy, fz]
    target_names = ["x_c", "y_c", "z_m", "fx", "fy", "fz"]
    print(f"Valid samples: {len(Y)}  | Features: {feat_desc}")

    # Optional additional filtering for training
    if MIN_POS_SUM_M is not None:
        keep = (np.abs(Y[:, 0]) + np.abs(Y[:, 1])) >= float(MIN_POS_SUM_M)
        X, Y = X[keep], Y[keep]
        if uvw_all is not None:
            uvw_all = uvw_all[m][keep]
    else:
        if uvw_all is not None:
            uvw_all = uvw_all[m]
    if MIN_SHEAR_FORCE_SUM is not None:
        keep2 = (np.abs(Y[:, 3]) + np.abs(Y[:, 4])) >= float(MIN_SHEAR_FORCE_SUM)
        X, Y = X[keep2], Y[keep2]
        if uvw_all is not None:
            uvw_all = uvw_all[keep2]

    if len(Y) < 32:
        print("Warning: very few valid samples; metrics may be unreliable.")

    # Feature names for persistence and stage2 feature schema
    feature_names = _feature_names(FEATURES_MODE)

    if USE_TWO_STAGE:
        two_stage = train_two_stage(
            X, Y, alpha=RIDGE_ALPHA, val_ratio=VAL_RATIO, seed=RANDOM_SEED, base_feature_names=feature_names,
            uvw=uvw_all if ADD_Z_INTERACTIONS else None,
        )
        print("\nTwo-stage training completed.")
        print("Stage1 metrics:")
        for k, v in two_stage["stage1"]["metrics"].items():
            print(f"  {k}: MAE={v['MAE']:.4f}, R2={v['R2']:.4f}, N={v['N']}")
        print("Stage2 metrics:")
        for k, v in two_stage["stage2"]["metrics"].items():
            print(f"  {k}: MAE={v['MAE']:.4f}, R2={v['R2']:.4f}, N={v['N']}")
    else:
        result = train_and_eval(X, Y, target_names=target_names, alpha=RIDGE_ALPHA, val_ratio=VAL_RATIO, seed=RANDOM_SEED)

    # Persist model for later real-time inference
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{point_str}_{FEATURES_MODE}_{ts}.json"
    out_path = os.path.join(MODEL_DIR, fn)
    latest_path = os.path.join(MODEL_DIR, f"{point_str}_{FEATURES_MODE}_latest.json")

    if USE_TWO_STAGE:
        save_model_json(
            out_path=out_path,
            point_str=point_str,
            features_mode=FEATURES_MODE,
            feature_names=feature_names,
            sensor_map=sensor_map,
            runs_used=[os.path.basename(r) for r in runs],
            n_rows=len(df),
            two_stage=two_stage,
        )
    else:
        save_model_json(
            out_path=out_path,
            point_str=point_str,
            features_mode=FEATURES_MODE,
            feature_names=feature_names,
            sensor_map=sensor_map,
            target_names=target_names,
            W=result["W"],
            b=result["b"],
            Xs=result["Xs"],
            Ys=result["Ys"],
            alpha=RIDGE_ALPHA,
            metrics=result["metrics"],
            runs_used=[os.path.basename(r) for r in runs],
            n_rows=len(df),
        )
    # Also write/update latest
    if USE_TWO_STAGE:
        save_model_json(
            out_path=latest_path,
            point_str=point_str,
            features_mode=FEATURES_MODE,
            feature_names=feature_names,
            sensor_map=sensor_map,
            runs_used=[os.path.basename(r) for r in runs],
            n_rows=len(df),
            two_stage=two_stage,
        )
    else:
        save_model_json(
            out_path=latest_path,
            point_str=point_str,
            features_mode=FEATURES_MODE,
            feature_names=feature_names,
            sensor_map=sensor_map,
            target_names=target_names,
            W=result["W"],
            b=result["b"],
            Xs=result["Xs"],
            Ys=result["Ys"],
            alpha=RIDGE_ALPHA,
            metrics=result["metrics"],
            runs_used=[os.path.basename(r) for r in runs],
            n_rows=len(df),
        )


if __name__ == "__main__":
    main()
