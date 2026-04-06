"""Analyse symmetry of sensor outputs at a single sensing point.

For a given point (e.g., p1), this script groups samples according to the sign
of the commanded axis force (fx or fy) and reports statistics for the
corresponding coordinate/force pairs (x_c/fx or y_c/fy). Optionally the script
can run the saved PyTorch model to compare predicted outputs instead of raw
labels.

Example usage::

    python acc_v2/learning_based/train_fx_fy_fz/point_symmetry_analysis.py \
        --point p1 --axes x,y --use-model

"""

from __future__ import annotations

import argparse
import glob
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS_DIR = THIS_DIR.parents[1] / "logs"
DEFAULT_MODELS_DIR = THIS_DIR / "models"


AXIS_CONFIG = {
    'x': {'coord': 'x_c', 'force': 'fx', 'label': 'X axis'},
    'y': {'coord': 'y_c', 'force': 'fy', 'label': 'Y axis'},
    'z': {'coord': 'z', 'force': 'fz', 'label': 'Z axis'},
}

SENSOR_MAP: Dict[int, List[str]] = {
    1: ["s1", "s2", "s5", "s6"],
    2: ["s2", "s3", "s6", "s7"],
    3: ["s3", "s4", "s7", "s8"],
    4: ["s5", "s6", "s9", "s10"],
    5: ["s6", "s7", "s10", "s11"],
    6: ["s7", "s8", "s11", "s12"],
    7: ["s9", "s10", "s13", "s14"],
    8: ["s10", "s11", "s14", "s15"],
    9: ["s11", "s12", "s15", "s16"],
}

OUTPUT_COLUMNS = ["x_c", "y_c", "z", "fx", "fy", "fz"]


@dataclass
class Summary:
    mean: float
    std: float
    median: float
    p5: float
    p95: float


@dataclass
class ModelBundle:
    model: "torch.nn.Module"
    baseline: np.ndarray
    scaler_x: object
    scaler_y: object
    sensor_cols: List[str]


def _parse_point(point: str) -> int:
    if not point or not point.startswith('p'):
        raise ValueError(f"Point must be formatted as 'p1'..'p9': {point}")
    pid = int(point[1:])
    if pid not in SENSOR_MAP:
        raise ValueError(f"Point id out of range (1-9): {point}")
    return pid


def _load_point_dataframe(logs_dir: Path, point: int, limit: Optional[int]) -> pd.DataFrame:
    pattern = logs_dir / f"p{point}_*" / "merged_data.csv"
    csv_files = sorted(glob.glob(str(pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No merged_data.csv found for p{point} under {logs_dir}")

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"경고: '{csv_path}' 로드 실패: {exc}")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError(f"데이터를 로드하지 못했습니다 (p{point})")

    df = pd.concat(frames, ignore_index=True)
    df.rename(columns=lambda c: c.replace('_afd50', '').replace('_motor', ''), inplace=True)
    if 'z_displacement_mm_laser' in df.columns:
        df.rename(columns={'z_displacement_mm_laser': 'z'}, inplace=True)

    numeric_cols = [col for col in df.columns if col.startswith('s') or col in OUTPUT_COLUMNS]
    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_numeric.dropna(subset=OUTPUT_COLUMNS, inplace=True)
    if limit:
        df_numeric = df_numeric.head(limit)
    return df_numeric.reset_index(drop=True)


def _load_model_bundle(model_dir: Path, point_id: int) -> ModelBundle:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch가 필요합니다. --use-model 옵션을 사용하기 전에 설치하세요.") from exc

    path = model_dir / f"p{point_id}_6dof_model_gpu.pt"
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")

    bundle = torch.load(path, map_location='cpu', weights_only=False)

    class MLPRegressorTorch(nn.Module):
        def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for size in hidden_sizes:
                layers.append(nn.Linear(prev_dim, size))
                layers.append(nn.ReLU())
                prev_dim = size
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[override]
            return self.network(x)

    hidden_sizes = bundle.get('train_config', {}).get('hidden_sizes', [256, 128, 64])
    sensor_cols: List[str] = bundle['sensor_cols']
    output_cols: List[str] = bundle['output_cols']

    model = MLPRegressorTorch(len(sensor_cols), hidden_sizes, len(output_cols))
    model.load_state_dict(bundle['model_state_dict'])
    model.eval()

    baseline = np.array(bundle['baseline'], dtype=np.float32)

    return ModelBundle(
        model=model,
        baseline=baseline,
        scaler_x=bundle['scaler_x'],
        scaler_y=bundle['scaler_y'],
        sensor_cols=sensor_cols,
    )


def _predict_outputs(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    import torch

    missing = [col for col in bundle.sensor_cols if col not in df.columns]
    if missing:
        raise ValueError(f"입력 데이터에 필요한 센서 컬럼이 없습니다: {missing}")

    sensors = df[bundle.sensor_cols].to_numpy(dtype=np.float32)
    sensor_prime = sensors - bundle.baseline
    sensor_scaled = bundle.scaler_x.transform(sensor_prime)
    X_tensor = torch.from_numpy(sensor_scaled.astype(np.float32))
    with torch.no_grad():
        y_scaled = bundle.model(X_tensor).cpu().numpy()
    predictions = bundle.scaler_y.inverse_transform(y_scaled)
    return pd.DataFrame(predictions, columns=OUTPUT_COLUMNS)


def _summarise(values: Iterable[float]) -> Summary:
    data = [v for v in values if not math.isnan(v)]
    if not data:
        return Summary(float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))
    return Summary(
        mean=float(statistics.fmean(data)),
        std=float(statistics.pstdev(data)),
        median=float(statistics.median(data)),
        p5=float(np.percentile(data, 5)),
        p95=float(np.percentile(data, 95)),
    )


def _analyse_axis(df: pd.DataFrame, axis: str, min_force: float) -> None:
    cfg = AXIS_CONFIG[axis]
    coord_col = cfg['coord']
    force_col = cfg['force']
    label = cfg['label']

    if coord_col not in df.columns or force_col not in df.columns:
        print(f" - 축 {axis}: 필요한 컬럼이 없습니다 ({coord_col}, {force_col})")
        return

    df_filtered = df.copy()
    if min_force > 0:
        mask = df_filtered[force_col].abs() >= min_force
        df_filtered = df_filtered[mask]

    pos = df_filtered[df_filtered[force_col] >= 0]
    neg = df_filtered[df_filtered[force_col] < 0]

    print(f"축 {label} ({coord_col}/{force_col})")
    print(f" - 샘플 수: total={len(df_filtered)} | pos={len(pos)} | neg={len(neg)}")
    if pos.empty or neg.empty:
        print(" - 충분한 양/음 데이터가 없습니다.\n")
        return

    pos_coord = _summarise(pos[coord_col])
    neg_coord = _summarise(-neg[coord_col])  # negate for symmetry comparison
    pos_force = _summarise(pos[force_col])
    neg_force = _summarise(-neg[force_col])

    print("   좌우 비교 (좌측: +, 우측: - 반전)")
    header = "      {item:<6} | {pos:>12} | {neg:>12} | delta"
    print(header.format(item="metric", pos="pos", neg="neg"))

    def display(sym_pos: Summary, sym_neg: Summary, name: str) -> None:
        mean_delta = sym_pos.mean - sym_neg.mean
        print(header.format(
            item=name,
            pos=f"mean={sym_pos.mean: .5f}",
            neg=f"mean={sym_neg.mean: .5f}",
        ))
        print(f"        std : {sym_pos.std: .5f} vs {sym_neg.std: .5f}")
        print(f"        med : {sym_pos.median: .5f} vs {sym_neg.median: .5f}")
        print(f"        Δmean: {mean_delta: .6f}")

    display(pos_coord, neg_coord, coord_col)
    display(pos_force, neg_force, force_col)

    coord_residual = pos_coord.mean - neg_coord.mean
    force_residual = pos_force.mean - neg_force.mean
    print(f"   symmetry residual (mean_pos + mean_neg): coord={coord_residual: .6f}, force={force_residual: .6f}\n")


def analyse_point(
    point: int,
    axes: List[str],
    logs_dir: Path,
    use_model: bool,
    model_dir: Path,
    limit: Optional[int],
    min_force: float,
) -> None:
    df_raw = _load_point_dataframe(logs_dir, point, limit)
    if use_model:
        bundle = _load_model_bundle(model_dir, point)
        df_outputs = _predict_outputs(df_raw, bundle)
    else:
        df_outputs = df_raw[OUTPUT_COLUMNS]

    print(f"=== 포인트 p{point} | 데이터 소스: {'모델 추론' if use_model else '라벨'} ===")
    for axis in axes:
        _analyse_axis(df_outputs, axis, min_force)


def main() -> None:
    parser = argparse.ArgumentParser(description="한 포인트에서 축별 대칭성을 분석합니다")
    parser.add_argument("--point", required=True, help="분석할 포인트 (예: p1)")
    parser.add_argument("--axes", default="x,y", help="콤마로 구분된 축 목록 (x,y,z)")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR, help="로그 디렉토리")
    parser.add_argument("--use-model", action="store_true", help="라벨 대신 모델 예측을 사용")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS_DIR, help="모델 디렉토리")
    parser.add_argument("--limit", type=int, help="최대 샘플 수")
    parser.add_argument("--min-force", type=float, default=0.0, help="축 힘의 절댓값이 이 값 이상인 샘플만 사용")

    args = parser.parse_args()
    point_id = _parse_point(args.point)
    axes = [ax.strip() for ax in args.axes.split(',') if ax.strip()]
    for ax in axes:
        if ax not in AXIS_CONFIG:
            raise ValueError(f"지원하지 않는 축: {ax}")

    analyse_point(point_id, axes, args.logs_dir, args.use_model, args.model_dir, args.limit, args.min_force)


if __name__ == "__main__":
    main()
