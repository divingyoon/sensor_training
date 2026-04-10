#!/usr/bin/env python3
"""
새 전처리 파이프라인 (v2)
- 각 trial의 baseline 추출 → JSON 저장
- 그리드 포인트 행만 필터링 → grid CSV 저장 (raw ADC)
- SR 학습용 normalized features CSV 저장
- 전체 trial 통합 CSV 저장

사용법:
  python3 preprocessing/preprocess.py
  python3 preprocessing/preprocess.py --raw-dir preprocessing/raw_data --out-dir preprocessing/processed_data
"""

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.utils.contact_geometry import contact_radius

# ── 상수 ────────────────────────────────────────────────────────────────────
SKIN_COLS = [f"s{i}" for i in range(1, 17)]
NORM_SKIN_COLS = [f"s_norm_{i}" for i in range(1, 17)]

# 그리드 기준 (mm 단위)
GRID_STEP_MM = 0.5
GRID_MIN_MM = -9.75
GRID_MAX_MM = 9.75

TRIAL_RE = re.compile(
    r"^(?P<material>[^_]+)_d(?P<diameter>\d+(?:\.\d+)?)_(?P<trial_no>\d+)$",
    re.IGNORECASE,
)
NESTED_TRIAL_RE = re.compile(
    r"^(?P<material>[^_]+)_d(?P<diameter>\d+(?:\.\d+)?)_z(?P<z_max>\d+(?:\.\d+)?)_test(?P<trial_no>\d+)$",
    re.IGNORECASE,
)
INDENTER_DIR_RE = re.compile(r"^d(?P<diameter>\d+(?:\.\d+)?)$", re.IGNORECASE)
Z_DIR_RE = re.compile(r"^z_?(?P<z_max>\d+(?:\.\d+)?)mm$", re.IGNORECASE)
TEST_DIR_RE = re.compile(r"^test(?P<trial_no>\d+)$", re.IGNORECASE)


# ── 파일명 파싱 ──────────────────────────────────────────────────────────────
def parse_trial_name(stem: str) -> dict:
    base = stem.replace("_merged", "")

    nested = NESTED_TRIAL_RE.match(base)
    if nested:
        return {
            "material": nested.group("material").lower(),
            "diameter_mm": float(nested.group("diameter")),
            "z_max_indentation_mm": float(nested.group("z_max")),
            "trial_no": int(nested.group("trial_no")),
        }

    # trial_id가 폴더명과 동일한 경우 파싱
    m = TRIAL_RE.match(stem)
    if not m:
        # ecomesh_d5_1_merged 형태 대응
        m = TRIAL_RE.match(base)
    
    if not m:
        return {"material": base, "diameter_mm": None, "z_max_indentation_mm": None, "trial_no": None}
    return {
        "material": m.group("material").lower(),
        "diameter_mm": float(m.group("diameter")),
        "z_max_indentation_mm": None,
        "trial_no": int(m.group("trial_no")),
    }


def discover_merged_csvs(raw_dir: Path, glob_pattern: str) -> list[Path]:
    return sorted(raw_dir.glob(glob_pattern))


def _format_number_for_id(value: float) -> str:
    return f"{value:g}"


def parse_trial_csv_info(csv_path: Path, raw_dir: Path) -> tuple[str, dict]:
    trial_id = csv_path.stem.replace("_merged", "")
    info = parse_trial_name(trial_id)
    if info["diameter_mm"] is not None:
        return trial_id, info

    try:
        rel_parts = csv_path.parent.relative_to(raw_dir).parts
    except ValueError:
        rel_parts = csv_path.parent.parts

    if len(rel_parts) >= 4:
        material, indenter_dir, z_dir, test_dir = rel_parts[-4:]
        indenter_match = INDENTER_DIR_RE.match(indenter_dir)
        z_match = Z_DIR_RE.match(z_dir)
        test_match = TEST_DIR_RE.match(test_dir)
        if indenter_match and z_match and test_match:
            diameter_mm = float(indenter_match.group("diameter"))
            z_max_indentation_mm = float(z_match.group("z_max"))
            trial_no = int(test_match.group("trial_no"))
            material_id = material.lower()
            parsed_trial_id = (
                f"{material_id}_d{_format_number_for_id(diameter_mm)}_"
                f"z{z_match.group('z_max')}_test{trial_no}"
            )
            return parsed_trial_id, {
                "material": material_id,
                "diameter_mm": diameter_mm,
                "z_max_indentation_mm": z_max_indentation_mm,
                "trial_no": trial_no,
            }

    return trial_id, info


def _normalize_workers(worker_count: int, num_tasks: int) -> int:
    if worker_count <= 1 or num_tasks <= 1:
        return 1
    return min(worker_count, num_tasks)


# ── Baseline 추출 / 드리프트 보정 ────────────────────────────────────────────
def find_baseline_segments(
    df: pd.DataFrame,
    z_thresh: float = 0.001,
    force_thresh: float = 0.5,
    min_consec: int = 40,
) -> list[tuple[int, int]]:
    """
    z_mm가 거의 0이고(stage 정지) 힘이 작은 연속 구간을 baseline 후보로 찾는다.

    반환: [(start_idx, end_idx), ...] (end 포함)
    """

    dx = df["x_mm"].diff().abs().fillna(0.0)
    dy = df["y_mm"].diff().abs().fillna(0.0)

    no_load = (df["z_mm"].abs() <= z_thresh) & (dx < 0.01) & (dy < 0.01)
    if force_thresh > 0 and {"Fz"}.issubset(df.columns):
        no_load &= df["Fz"].abs() <= force_thresh

    segments: list[tuple[int, int]] = []
    start_idx: int | None = None
    arr = no_load.to_numpy()

    for idx, flag in enumerate(arr):
        if flag and start_idx is None:
            start_idx = idx
            continue
        if (not flag) and start_idx is not None:
            if idx - start_idx >= min_consec:
                segments.append((start_idx, idx - 1))
            start_idx = None

    if start_idx is not None and len(arr) - start_idx >= min_consec:
        segments.append((start_idx, len(arr) - 1))

    return segments


def compute_baseline_stats(
    rows: pd.DataFrame,
    trial_id: str,
    info: dict,
    baseline_id: int,
    start_idx: int,
    end_idx: int,
) -> dict:
    baseline = {
        "baseline_id": baseline_id,
        "trial_id": trial_id,
        "material": info["material"],
        "diameter_mm": info["diameter_mm"],
        "z_max_indentation_mm": info.get("z_max_indentation_mm"),
        "baseline_n_rows": int(len(rows)),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
    }

    if "timestep_sec" in rows.columns:
        baseline["start_time_sec"] = float(rows["timestep_sec"].iloc[0])
        baseline["end_time_sec"] = float(rows["timestep_sec"].iloc[-1])

    baseline["fz_mean"] = float(rows["Fz"].mean())
    baseline["fx_mean"] = float(rows["Fx"].mean())
    baseline["fy_mean"] = float(rows["Fy"].mean())
    for col in SKIN_COLS:
        baseline[f"{col}_mean"] = float(rows[col].mean())

    return baseline


def assign_baseline_ids(n_rows: int, segments: list[tuple[int, int]]) -> np.ndarray:
    """
    각 row에 가장 최근 baseline segment id를 할당한다.
    segment가 없으면 0으로 채운다.
    """

    if not segments:
        return np.zeros(n_rows, dtype=int)

    baseline_ids = np.zeros(n_rows, dtype=int)
    seg_ptr = 0
    seg_start, seg_end = segments[0]
    current_id = 0

    for i in range(n_rows):
        # 다음 baseline 구간 시작 시점 업데이트
        if seg_ptr < len(segments) and i >= segments[seg_ptr][0]:
            seg_start, seg_end = segments[seg_ptr]
            current_id = seg_ptr
        baseline_ids[i] = current_id
        # 다음 segment로 이동할 준비
        if i == seg_end and seg_ptr + 1 < len(segments):
            seg_ptr += 1

    return baseline_ids


def extract_baselines(
    df: pd.DataFrame,
    trial_id: str,
    info: dict,
    z_thresh: float = 0.001,
    force_thresh: float = 0.5,
    min_consec: int = 40,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    다중 baseline 구간을 찾아 각 구간별 평균을 반환한다.

    1) z_mm≈0 & 힘 작음 & 정지 상태 구간을 우선 사용
    2) 없으면 기존 방식(파일 앞 x=y=z=0 연속 구간)으로 폴백
    """

    segments = find_baseline_segments(df, z_thresh=z_thresh, force_thresh=force_thresh, min_consec=min_consec)

    # 기존 방식으로 폴백
    if not segments:
        mask = (df["x_mm"] == 0) & (df["y_mm"] == 0) & (df["z_mm"] == 0)
        inverse = ~mask
        if inverse.any():
            first_nonzero_pos = int(inverse.values.argmax())
        else:
            first_nonzero_pos = len(df)

        if first_nonzero_pos == 0 and not mask.values[0]:
            raise ValueError(f"[{trial_id}] baseline 구간을 찾을 수 없습니다.")
        segments = [(0, max(first_nonzero_pos - 1, 0))]

    baselines: list[dict] = []
    for b_id, (start_idx, end_idx) in enumerate(segments):
        rows = df.iloc[start_idx : end_idx + 1]
        baselines.append(compute_baseline_stats(rows, trial_id, info, b_id, start_idx, end_idx))

    return baselines, segments


# ── 그리드 행 필터링 ──────────────────────────────────────────────────────────
def filter_grid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """x_mm, y_mm가 0.5mm 그리드 포인트에 있고, 스테이지가 정지된 행만 선택."""
    # 1. 그리드 포인트 확인 (0.5mm 간격)
    grid_vals = np.round(np.arange(GRID_MIN_MM, GRID_MAX_MM + 0.001, GRID_STEP_MM), 3)
    x_rounded = np.round(df["x_mm"].values, 3)
    y_rounded = np.round(df["y_mm"].values, 3)
    on_grid = np.isin(x_rounded, grid_vals) & np.isin(y_rounded, grid_vals)

    # 2. 정지 상태 확인 (Stability Check)
    # 이전/다음 샘플과 좌표 차이가 매우 작아야 함 (0.001mm 미만)
    dx = df["x_mm"].diff().abs().fillna(0.0)
    dy = df["y_mm"].diff().abs().fillna(0.0)
    # 다음 샘플과의 차이도 확인하여 가감속 구간 제외
    dx_next = df["x_mm"].diff(-1).abs().fillna(0.0)
    dy_next = df["y_mm"].diff(-1).abs().fillna(0.0)

    is_stationary = (dx < 0.001) & (dy < 0.001) & (dx_next < 0.001) & (dy_next < 0.001)

    return df[on_grid & is_stationary].copy()


    # 필터 통과 직후 좌표를 그리드에 강제 스냅해 groupby 분할(미세 부동소수점) 방지
    out["x_mm"] = (
        np.round((out["x_mm"].to_numpy(dtype=np.float64) - GRID_MIN_MM) / GRID_STEP_MM) * GRID_STEP_MM
        + GRID_MIN_MM
    )
    out["y_mm"] = (
        np.round((out["y_mm"].to_numpy(dtype=np.float64) - GRID_MIN_MM) / GRID_STEP_MM) * GRID_STEP_MM
        + GRID_MIN_MM
    )
    out["x_mm"] = np.clip(out["x_mm"], GRID_MIN_MM, GRID_MAX_MM)
    out["y_mm"] = np.clip(out["y_mm"], GRID_MIN_MM, GRID_MAX_MM)
    out["x_mm"] = np.round(out["x_mm"], 3)
    out["y_mm"] = np.round(out["y_mm"], 3)
    return out


# ── z_depth 계산 ─────────────────────────────────────────────────────────────
def compute_z_depth(
    df: pd.DataFrame,
    contact_threshold: float = 0.01,
    n_consec: int = 3,
) -> pd.DataFrame:
    """
    z_depth를 contact 보정 없이 원본 z_mm로 사용.
    """
    out = df.copy()
    out["z_depth_mm"] = out["z_mm"].astype(np.float64)
    out = out[out["z_depth_mm"] >= 0].copy()
    return out.reset_index(drop=True)


# ── Phase 분류 ────────────────────────────────────────────────────────────────
def assign_phase(df: pd.DataFrame) -> pd.DataFrame:
    """각 (x_mm, y_mm) 그룹 내 z_mm 최대값 기준으로 loading(0) / unloading(1) 구분."""
    df = df.reset_index(drop=True)
    phase = np.zeros(len(df), dtype=np.int8)

    for (_, _), grp in df.groupby(["x_mm", "y_mm"], sort=False):
        if len(grp) < 2:
            continue
        peak_pos = grp["z_mm"].values.argmax()
        # peak 이후 행 → unloading
        after_peak = grp.iloc[peak_pos + 1 :].index
        phase[after_peak] = 1

    df["phase"] = phase
    return df


# ── Grid CSV 생성 (raw ADC) ──────────────────────────────────────────────────
def make_grid_df(
    df: pd.DataFrame,
    trial_id: str,
    info: dict,
    contact_threshold: float = 0.01,
) -> pd.DataFrame:
    """그리드 필터링 + z_depth + phase 포함 raw CSV 생성."""
    df = filter_grid_rows(df)
    if df.empty:
        return pd.DataFrame()

    df = compute_z_depth(df, contact_threshold=contact_threshold)
    if df.empty:
        return pd.DataFrame()

    df = assign_phase(df)

    n = len(df)
    data: dict = {
        "trial_id": np.full(n, trial_id),
        "material": np.full(n, info["material"]),
        "diameter_mm": np.full(n, info["diameter_mm"], dtype=np.float32),
        "baseline_id": df["baseline_id"].values if "baseline_id" in df.columns else np.zeros(n, dtype=int),
        "x_mm": df["x_mm"].values,
        "y_mm": df["y_mm"].values,
        "z_depth_mm": df["z_depth_mm"].values,
        "fz": np.round(df["Fz"].values, 2),
        "fx": np.round(df["Fx"].values, 2),
        "fy": np.round(df["Fy"].values, 2),
    }
    for col in SKIN_COLS:
        data[col] = df[col].values
    data["phase"] = df["phase"].values

    return pd.DataFrame(data)


# ── Normalized Features CSV 생성 (SR 및 Force Field 학습용) ───────────────────
def make_features_df(
    grid_df: pd.DataFrame,
    baselines: list[dict],
    max_diameter: float,
    z_bin_mm: float,
    min_signal: float,
    contact_radius_mm: np.ndarray | None = None,
    contact_radius_cell: np.ndarray | None = None,
) -> pd.DataFrame:
    """잔차 정규화 + diameter 정규화 + fz 정규화 → 학습 입력/타겟 CSV."""
    feat = pd.DataFrame()

    # baseline_id → baseline dict 매핑
    baseline_map = {b["baseline_id"]: b for b in baselines}
    if "baseline_id" not in grid_df.columns:
        grid_df = grid_df.copy()
        grid_df["baseline_id"] = 0

    # 입력 특징: s_norm_i = (s_i - baseline_i) / baseline_i (구간별 baseline 사용)
    bl_ids = grid_df["baseline_id"].to_numpy()
    for i, col in enumerate(SKIN_COLS, 1):
        bl_vals = np.array([baseline_map[b]["%s_mean" % col] for b in bl_ids], dtype=np.float64)
        norm_col = f"s_norm_{i}"
        col_vals = grid_df[col].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            feat[norm_col] = np.where(bl_vals == 0, 0.0, (col_vals - bl_vals) / bl_vals)

    # 입력 특징: diameter 정규화 (0~1)
    diam = grid_df["diameter_mm"].values[0] if "diameter_mm" in grid_df.columns else 0.0
    feat["diameter_norm"] = diam / max_diameter if max_diameter > 0 else 0.0

    # 위치 및 깊이 (SR 타겟이자 Force Field 입력)
    feat["x_mm"] = grid_df["x_mm"].values
    feat["y_mm"] = grid_df["y_mm"].values
    feat["z_depth_mm"] = grid_df["z_depth_mm"].values
    if contact_radius_mm is not None:
        feat["contact_radius_mm"] = contact_radius_mm
    if contact_radius_cell is not None:
        feat["contact_radius_cell"] = contact_radius_cell
    feat["baseline_id"] = bl_ids
    
    # 힘 (Force Field 타겟): Baseline 정정된 Fz (구간별 baseline 사용, 0.01 정밀도)
    fz_bl = np.array([baseline_map[b]["fz_mean"] for b in bl_ids], dtype=np.float64)
    feat["fz_bc"] = np.round(grid_df["fz"].values - fz_bl, 2)
    
    # 메타 데이터
    feat["fz_raw"] = grid_df["fz"].values
    feat["diameter_mm"] = grid_df["diameter_mm"].values
    feat["trial_id"] = grid_df["trial_id"].values
    feat["material"] = grid_df["material"].values
    feat["phase"] = grid_df["phase"].values
    feat = feat.reset_index(drop=True)

    if z_bin_mm > 0:
        feat["z_depth_mm"] = np.round(feat["z_depth_mm"].to_numpy(dtype=np.float64) / z_bin_mm) * z_bin_mm
        feat["z_depth_mm"] = np.maximum(feat["z_depth_mm"], 0.0)
        feat["z_depth_mm"] = np.round(feat["z_depth_mm"], 6)
        group_cols = ["trial_id", "material", "diameter_mm", "phase", "x_mm", "y_mm", "z_depth_mm"]
        agg_spec = {c: "median" for c in NORM_SKIN_COLS}
        agg_spec["diameter_norm"] = "first"
        agg_spec["fz_bc"] = "mean"
        agg_spec["fz_raw"] = "mean"
        if contact_radius_mm is not None and "contact_radius_mm" in feat.columns:
            agg_spec["contact_radius_mm"] = "mean"
        if contact_radius_cell is not None and "contact_radius_cell" in feat.columns:
            agg_spec["contact_radius_cell"] = "mean"
        feat = feat.groupby(group_cols, as_index=False).agg(agg_spec)

    if min_signal > 0:
        peak_signal = feat[NORM_SKIN_COLS].abs().max(axis=1)
        feat = feat.loc[peak_signal >= float(min_signal)].reset_index(drop=True)

    return feat.reset_index(drop=True)



# ── 단일 Trial 처리 ──────────────────────────────────────────────────────────
def process_trial(
    csv_path: Path, out_dir: Path, contact_threshold: float = 0.01
) -> tuple[list[dict], pd.DataFrame]:
    trial_id = csv_path.stem
    info = parse_trial_name(trial_id)

    print(f"  처리 중: {trial_id}")
    df = pd.read_csv(csv_path)

    # Baseline (다중 구간 지원)
    baselines, segments = extract_baselines(df, trial_id, info)
    baseline_ids = assign_baseline_ids(len(df), segments)
    df = df.copy()
    df["baseline_id"] = baseline_ids

    bl_path = out_dir / f"{trial_id}_baselines.json"
    with open(bl_path, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2, ensure_ascii=False)

    # Grid CSV (센서 반응 기반 접촉점 기준 z_depth)
    grid_df = make_grid_df(df, trial_id, info, contact_threshold)
    grid_path = out_dir / f"{trial_id}_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    n_pts = grid_df.groupby(["x_mm", "y_mm"]).ngroups
    n_load = (grid_df["phase"] == 0).sum()
    n_unload = (grid_df["phase"] == 1).sum()
    z_min = grid_df["z_depth_mm"].min()
    z_max = grid_df["z_depth_mm"].max()
    print(
        f"    → baselines {len(baselines)}개 | 그리드 {n_pts}포인트 | total {len(grid_df)}행 "
        f"(loading {n_load} / unloading {n_unload}) | z_depth [{z_min:.3f}, {z_max:.3f}]mm"
    )

    return baselines, grid_df


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="센서 압입 실험 전처리 (v3)")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("preprocessing/raw_data"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preprocessing/processed_data"),
    )
    parser.add_argument("--glob", type=str, default="**/*_merged.csv")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Trial CSV 병렬 처리 프로세스 수. 1이면 기존 단일 프로세스 실행.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.01,
        help="센서 접촉 판정 s_norm 임계값 (기본값: 0.01 = baseline 대비 1%% 변화)",
    )
    parser.add_argument(
        "--no-zarr",
        action="store_true",
        help="Zarr 포맷 저장을 건너뜁니다.",
    )
    parser.add_argument(
        "--z-bin-mm",
        type=float,
        default=0.02,
        help="z_depth를 이 간격(mm)으로 binning 후 (x,y,z_bin) 단위로 집계합니다. 0 이하면 비활성.",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=0.02,
        help="max(|s_norm_i|)가 이 값 미만인 저신호 샘플 제거. 0 이하면 비활성.",
    )
    parser.add_argument(
        "--min-reliable-s",
        type=float,
        default=0.005,
        help="일관성 필터에서 좌표별 최소 신호 임계값. 낮출수록 더 많은 좌표를 남김.",
    )
    parser.add_argument(
        "--baseline-z-thresh",
        type=float,
        default=0.001,
        help="baseline 탐색 시 |z_mm| 최대 허용치 (mm).",
    )
    parser.add_argument(
        "--baseline-force-thresh",
        type=float,
        default=0.5,
        help="baseline 탐색 시 |Fz| 최대 허용치 (N). 0이면 무시.",
    )
    parser.add_argument(
        "--baseline-min-consec",
        type=int,
        default=40,
        help="baseline으로 인정할 최소 연속 샘플 수.",
    )
    parser.add_argument(
        "--use-depth-aware-radius",
        action="store_true",
        help="깊이 기반 접촉 반경을 계산해 contact_radius_mm 필드를 생성합니다.",
    )
    parser.add_argument(
        "--radius-model",
        choices=["hertz", "geo"],
        default="hertz",
        help="접촉 반경 계산 모델: hertz(a=sqrt(R*δ)) 또는 geo(a=sqrt(2Rδ-δ^2)).",
    )
    parser.add_argument(
        "--indenter-radius-mm",
        type=float,
        default=2.5,
        help="인덴터 반경(mm). 기본 2.5 (지름 5mm).",
    )
    parser.add_argument(
        "--max-radius-mm",
        type=float,
        default=None,
        help="계산된 접촉 반경 상한(mm). None이면 제한 없음.",
    )
    parser.add_argument(
        "--fallback-depth-mode",
        choices=["none", "mean", "const"],
        default="none",
        help="깊이값이 0/음수인 경우 대체 규칙: none=0 유지, mean=양수 depth 평균으로 치환, const=fallback-depth-mm 사용.",
    )
    parser.add_argument(
        "--fallback-depth-mm",
        type=float,
        default=0.0,
        help="fallback-depth-mode=const 일 때 사용할 깊이(mm).",
    )
    parser.add_argument(
        "--export-label-heatmap",
        action="store_true",
        help="(옵션) 깊이 기반 라벨 히트맵을 PNG로 샘플 시각화합니다.",
    )
    parser.add_argument(
        "--label-samples",
        type=int,
        default=3,
        help="시각화할 샘플 수 (grid row 기준).",
    )
    parser.add_argument(
        "--label-kernel",
        choices=["gaussian", "linear"],
        default="gaussian",
        help="라벨 히트맵 커널 형태.",
    )
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=1.0,
        help="Gaussian 커널 시 sigma = a * scale 로 설정.",
    )
    return parser.parse_args(argv)


# ── Zarr Export ──────────────────────────────────────────────────────────────
def export_to_zarr(features_df: pd.DataFrame, zarr_path: Path, aux_last_field: str = "diameter_mm"):
    """
    SkinDataset에서 바로 로드 가능한 Zarr 구조로 저장.
    aux_last_field: aux_feat의 마지막 컬럼 의미를 명시 (\"diameter_mm\" | \"contact_radius_mm\")
    """
    try:
        import zarr
    except ImportError:
        print("  [경고] zarr 라이브러리가 없어 Zarr 저장을 건너뜁니다. (pip install zarr)")
        return

    print(f"  Zarr 저장 중: {zarr_path}")
    
    # 데이터 추출 및 변환
    n = len(features_df)
    s_norm_cols = [f"s_norm_{i}" for i in range(1, 17)]
    tactile_lr_norm = features_df[s_norm_cols].values.astype(np.float32)
    
    # SkinDataset 호환 aux_feat 구성: [fx_N, fy_N, depth_mm, last_field]
    # last_field는 diameter_mm 또는 contact_radius_mm 중 하나.
    aux_feat = np.zeros((n, 4), dtype=np.float32)
    aux_feat[:, 2] = features_df["z_depth_mm"].values
    if aux_last_field == "contact_radius_mm" and "contact_radius_mm" in features_df.columns:
        aux_feat[:, 3] = features_df["contact_radius_mm"].values
    else:
        aux_feat[:, 3] = features_df["diameter_mm"].values
    # contact_radius_cell은 현재 aux_feat에 넣지 않지만 메타로 남긴다.
    
    fz = features_df["fz_bc"].values.astype(np.float32)
    cx = features_df["x_mm"].values.astype(np.float32)
    cy = features_df["y_mm"].values.astype(np.float32)
    depth_mm = features_df["z_depth_mm"].values.astype(np.float32)
    
    # Canvas bounds (SkinDataset 요구사항 대응용 더미 또는 계산값)
    x_bounds = np.tile([GRID_MIN_MM, GRID_MAX_MM], (n, 1)).astype(np.float32)
    y_bounds = np.tile([GRID_MIN_MM, GRID_MAX_MM], (n, 1)).astype(np.float32)

    # Zarr 그룹 생성 및 데이터 저장 (Blosc 압축 기본 적용)
    root = zarr.open_group(str(zarr_path), mode='w')
    root.attrs["aux_last_field"] = aux_last_field
    root.create_dataset("tactile_lr_norm", data=tactile_lr_norm, chunks=(1000, 16))
    root.create_dataset("aux_feat", data=aux_feat, chunks=(1000, 4))
    root.create_dataset("fz", data=fz, chunks=(1000,))
    root.create_dataset("cx", data=cx, chunks=(1000,))
    root.create_dataset("cy", data=cy, chunks=(1000,))
    root.create_dataset("depth_mm", data=depth_mm, chunks=(1000,))
    root.create_dataset("x_bounds", data=x_bounds, chunks=(1000, 2))
    root.create_dataset("y_bounds", data=y_bounds, chunks=(1000, 2))
    
    # 메타데이터 (JSON 인덱스 생성을 위함)
    samples_info = []
    trial_ids = features_df["trial_id"].values
    phases = features_df["phase"].values
    
    has_contact_radius = aux_last_field == "contact_radius_mm" and "contact_radius_mm" in features_df.columns
    for i in range(n):
        sample = {
            "trial_id": str(trial_ids[i]),
            "phase": "loading" if phases[i] == 0 else "unloading",
            "zarr_path": str(zarr_path),
            "zarr_index": i,
            "depth_bin_mm": float(depth_mm[i]),
        }
        if has_contact_radius:
            sample["radius_mm"] = float(features_df["contact_radius_mm"].iloc[i])
            if "contact_radius_cell" in features_df.columns:
                sample["radius_cell"] = float(features_df["contact_radius_cell"].iloc[i])
        samples_info.append(sample)
    
    index_path = zarr_path / "dataset_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples_info}, f, indent=2)


def _trial_options_from_args(args: argparse.Namespace) -> dict:
    return {
        "contact_threshold": args.contact_threshold,
        "baseline_z_thresh": args.baseline_z_thresh,
        "baseline_force_thresh": args.baseline_force_thresh,
        "baseline_min_consec": args.baseline_min_consec,
        "use_depth_aware_radius": args.use_depth_aware_radius,
        "fallback_depth_mode": args.fallback_depth_mode,
        "fallback_depth_mm": args.fallback_depth_mm,
        "indenter_radius_mm": args.indenter_radius_mm,
        "radius_model": args.radius_model,
        "max_radius_mm": args.max_radius_mm,
        "z_bin_mm": args.z_bin_mm,
        "min_signal": args.min_signal,
    }


def _process_trial_csv(
    csv_path: Path,
    raw_dir: Path,
    dirs: dict[str, Path],
    max_diameter: float,
    options: dict,
) -> dict:
    trial_id, info = parse_trial_csv_info(csv_path, raw_dir)
    log_lines = [f"  처리 중: {trial_id}"]

    df = pd.read_csv(csv_path)

    try:
        baselines, segments = extract_baselines(
            df,
            trial_id,
            info,
            z_thresh=options["baseline_z_thresh"],
            force_thresh=options["baseline_force_thresh"],
            min_consec=options["baseline_min_consec"],
        )
    except ValueError as exc:
        log_lines.append(f"    [건너뜀] {exc}")
        return {"skipped": True, "trial_id": trial_id, "logs": log_lines}

    baseline_ids = assign_baseline_ids(len(df), segments)
    df = df.copy()
    df["baseline_id"] = baseline_ids

    bl_path = dirs["baselines"] / f"{trial_id}_baselines.json"
    with open(bl_path, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2, ensure_ascii=False)

    grid_df = make_grid_df(df, trial_id, info, options["contact_threshold"])
    if grid_df.empty:
        log_lines.append("    [건너뜀] 그리드 데이터가 없습니다.")
        return {"skipped": True, "trial_id": trial_id, "logs": log_lines}

    grid_path = dirs["grid"] / f"{trial_id}_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    contact_radius_arr = None
    contact_radius_cell = None
    if options["use_depth_aware_radius"]:
        depth_arr = grid_df["z_depth_mm"].to_numpy(dtype=np.float64)
        if options["fallback_depth_mode"] != "none":
            pos_depth = depth_arr[depth_arr > 0]
            fallback = options["fallback_depth_mm"]
            if options["fallback_depth_mode"] == "mean" and len(pos_depth) > 0:
                fallback = float(pos_depth.mean())
            depth_arr = np.where(depth_arr > 0, depth_arr, fallback)
        contact_radius_arr = np.array(
            [
                contact_radius(float(d), R_mm=options["indenter_radius_mm"], model=options["radius_model"])
                for d in depth_arr
            ],
            dtype=np.float64,
        )
        if options["max_radius_mm"] is not None:
            contact_radius_arr = np.minimum(contact_radius_arr, options["max_radius_mm"])
        contact_radius_cell = contact_radius_arr / GRID_STEP_MM
        grid_df = grid_df.copy()
        grid_df["contact_radius_mm"] = contact_radius_arr
        grid_df["contact_radius_cell"] = contact_radius_cell

    feat_df = make_features_df(
        grid_df,
        baselines,
        max_diameter,
        z_bin_mm=options["z_bin_mm"],
        min_signal=options["min_signal"],
        contact_radius_mm=contact_radius_arr,
        contact_radius_cell=contact_radius_cell,
    )
    feat_path = dirs["features"] / f"{trial_id}_features.csv"
    feat_df.to_csv(feat_path, index=False)

    mat = str(grid_df["material"].iloc[0])
    log_lines.append(f"    → total {len(grid_df):,}행 정제 완료")
    return {
        "skipped": False,
        "trial_id": trial_id,
        "material": mat,
        "feature_path": feat_path,
        "logs": log_lines,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    
    # 폴더 구조화
    base_out = args.out_dir
    dirs = {
        "baselines": base_out / "baselines",
        "grid": base_out / "grid",
        "features": base_out / "features",
        "zarr": base_out / "zarr_data"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    csv_files = discover_merged_csvs(args.raw_dir, args.glob)
    if not csv_files:
        print(f"[오류] {args.raw_dir}/{args.glob} 에서 파일을 찾을 수 없습니다.")
        return

    print(f"[전처리 시작] {len(csv_files)}개 파일 발견")

    # max_diameter 계산
    max_diameter = 0.0
    for f in csv_files:
        _, info = parse_trial_csv_info(f, args.raw_dir)
        if info["diameter_mm"] is not None:
            max_diameter = max(max_diameter, info["diameter_mm"])
    if max_diameter == 0:
        max_diameter = 1.0

    material_feature_paths: dict[str, list[Path]] = {}
    options = _trial_options_from_args(args)
    workers = _normalize_workers(args.workers, len(csv_files))
    if workers > 1:
        print(f"[병렬 처리] workers={workers}")
        results: list[dict | None] = [None] * len(csv_files)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_process_trial_csv, csv_path, args.raw_dir, dirs, max_diameter, options): idx
                for idx, csv_path in enumerate(csv_files)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                for line in results[idx]["logs"]:
                    print(line)
    else:
        results = [
            _process_trial_csv(csv_path, args.raw_dir, dirs, max_diameter, options)
            for csv_path in csv_files
        ]
        for result in results:
            for line in result["logs"]:
                print(line)

    for result in results:
        if result is None or result.get("skipped"):
            continue
        material_feature_paths.setdefault(result["material"], []).append(result["feature_path"])

    # 소재별 통합 및 Zarr 변환
    print()
    for mat in sorted(material_feature_paths):
        all_feat = pd.concat([pd.read_csv(path) for path in material_feature_paths[mat]], ignore_index=True)
        
        # ── [추가] 일관성 필터 (Consistency Filter) ──
        print(f"  [{mat}] 데이터 품질 검사 중...")
        # 1. 각 (x, y, trial_id) 별 최대 신호 강도 계산
        s_norm_cols = [f"s_norm_{i}" for i in range(1, 17)]
        all_feat["max_s"] = all_feat[s_norm_cols].abs().max(axis=1)
        
        # 각 좌표/Trial별 최대 신호
        trial_grid_quality = all_feat.groupby(["x_mm", "y_mm", "trial_id"])["max_s"].max().reset_index()
        
        # 각 좌표별로 모든 Trial 중 '가장 약한 신호'를 찾음
        grid_min_quality = trial_grid_quality.groupby(["x_mm", "y_mm"])["max_s"].min().reset_index()
        grid_min_quality.rename(columns={"max_s": "min_trial_s"}, inplace=True)
        
        # 2. 모든 Trial에서 최소 0.005 이상의 신호가 보장된 좌표만 선별
        MIN_RELIABLE_S = args.min_reliable_s
        good_grids = grid_min_quality[grid_min_quality["min_trial_s"] >= MIN_RELIABLE_S]
        
        n_total_grids = len(grid_min_quality)
        n_good_grids = len(good_grids)
        
        # 3. 데이터 필터링
        all_feat = pd.merge(all_feat, good_grids[["x_mm", "y_mm"]], on=["x_mm", "y_mm"], how="inner")
        
        print(
            f"    → 전체 {n_total_grids}개 그리드 중 {n_good_grids}개 유지 "
            f"({n_good_grids/n_total_grids*100:.1f}%) | min_reliable_s={MIN_RELIABLE_S}"
        )
        if n_total_grids > n_good_grids:
            removed = grid_min_quality[grid_min_quality["min_trial_s"] < MIN_RELIABLE_S]
            print(f"    → 제거된 그리드 수: {len(removed)}")
            print(f"    → 제거된 그리드 예시 (min_s < {MIN_RELIABLE_S}): {removed.head(3).values.tolist()}")
        
        # 통합 CSV 저장
        all_feat.drop(columns=["max_s"], inplace=True)
        all_feat.to_csv(base_out / f"{mat}_features.csv", index=False)
        
        # Zarr 저장
        if not args.no_zarr:
            zarr_dir = dirs["zarr"] / f"dataset_{mat}.zarr"
            aux_last_field = "contact_radius_mm" if args.use_depth_aware_radius and "contact_radius_mm" in all_feat.columns else "diameter_mm"
            export_to_zarr(all_feat, zarr_dir, aux_last_field=aux_last_field)

        print(f"  [{mat}] 최종 {len(all_feat):,}행 가공 완료")

    print(f"\n[완료] 결과물이 {base_out.resolve()} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
