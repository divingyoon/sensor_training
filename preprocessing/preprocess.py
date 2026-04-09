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
from pathlib import Path

import numpy as np
import pandas as pd

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


# ── 파일명 파싱 ──────────────────────────────────────────────────────────────
def parse_trial_name(stem: str) -> dict:
    # trial_id가 폴더명과 동일한 경우 파싱
    m = TRIAL_RE.match(stem)
    if not m:
        # ecomesh_d5_1_merged 형태 대응
        base = stem.replace("_merged", "")
        m = TRIAL_RE.match(base)
    
    if not m:
        return {"material": stem, "diameter_mm": None, "trial_no": None}
    return {
        "material": m.group("material").lower(),
        "diameter_mm": float(m.group("diameter")),
        "trial_no": int(m.group("trial_no")),
    }


# ── Baseline 추출 ────────────────────────────────────────────────────────────
def extract_baseline(df: pd.DataFrame, trial_id: str, info: dict) -> dict:
    """
    파일 맨 앞의 연속적인 x_mm=y_mm=z_mm=0 구간만 baseline으로 사용.
    """
    # raw_merge.py에서 생성된 컬럼명 사용 (x_mm, y_mm, z_mm)
    mask = (df["x_mm"] == 0) & (df["y_mm"] == 0) & (df["z_mm"] == 0)

    # 파일 앞에서 첫 번째 비-0 행 위치 탐색
    first_nonzero_pos = int((~mask).values.argmax())
    if first_nonzero_pos == 0 and not mask.values[0]:
        # 파일 첫 행부터 비-0 -> baseline 없음
        raise ValueError(f"[{trial_id}] 파일 시작에 x=y=z=0 baseline 구간이 없습니다.")

    rows = df.iloc[:first_nonzero_pos]
    if len(rows) == 0:
        raise ValueError(f"[{trial_id}] x=y=z=0인 baseline 행이 없습니다.")

    baseline = {
        "trial_id": trial_id,
        "material": info["material"],
        "diameter_mm": info["diameter_mm"],
        "baseline_n_rows": int(len(rows)),
        "fz_mean": float(rows["Fz"].mean()),
        "fx_mean": float(rows["Fx"].mean()),
        "fy_mean": float(rows["Fy"].mean()),
    }
    for col in SKIN_COLS:
        baseline[f"{col}_mean"] = float(rows[col].mean())

    return baseline


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
    baseline: dict,
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
    baseline: dict,
    contact_threshold: float = 0.01,
) -> pd.DataFrame:
    """그리드 필터링 + z_depth + phase 포함 raw CSV 생성."""
    df = filter_grid_rows(df)
    if df.empty:
        return pd.DataFrame()
        
    df = compute_z_depth(df, baseline, contact_threshold=contact_threshold)
    if df.empty:
        return pd.DataFrame()
        
    df = assign_phase(df)

    n = len(df)
    data: dict = {
        "trial_id": np.full(n, trial_id),
        "material": np.full(n, info["material"]),
        "diameter_mm": np.full(n, info["diameter_mm"], dtype=np.float32),
        "x_mm": df["x_mm"].values,
        "y_mm": df["y_mm"].values,
        "z_depth_mm": df["z_depth_mm"].values,
        "fz": df["Fz"].values,
        "fx": df["Fx"].values,
        "fy": df["Fy"].values,
    }
    for col in SKIN_COLS:
        data[col] = df[col].values
    data["phase"] = df["phase"].values

    return pd.DataFrame(data)


# ── Normalized Features CSV 생성 (SR 및 Force Field 학습용) ───────────────────
def make_features_df(
    grid_df: pd.DataFrame, baseline: dict, max_diameter: float, z_bin_mm: float, min_signal: float
) -> pd.DataFrame:
    """잔차 정규화 + diameter 정규화 + fz 정규화 → 학습 입력/타겟 CSV."""
    feat = pd.DataFrame()

    # 입력 특징: s_norm_i = (s_i - baseline_i) / baseline_i
    for i, col in enumerate(SKIN_COLS, 1):
        bl_val = baseline[f"{col}_mean"]
        norm_col = f"s_norm_{i}"
        if bl_val == 0:
            feat[norm_col] = 0.0
        else:
            feat[norm_col] = (grid_df[col].values - bl_val) / bl_val

    # 입력 특징: diameter 정규화 (0~1)
    diam = grid_df["diameter_mm"].values[0] if "diameter_mm" in grid_df.columns else 0.0
    feat["diameter_norm"] = diam / max_diameter if max_diameter > 0 else 0.0

    # 위치 및 깊이 (SR 타겟이자 Force Field 입력)
    feat["x_mm"] = grid_df["x_mm"].values
    feat["y_mm"] = grid_df["y_mm"].values
    feat["z_depth_mm"] = grid_df["z_depth_mm"].values
    
    # 힘 (Force Field 타겟): Baseline 정정된 Fz
    fz_bl = baseline.get("fz_mean", 0.0)
    feat["fz_bc"] = grid_df["fz"].values - fz_bl
    
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
        feat = feat.groupby(group_cols, as_index=False).agg(agg_spec)

    if min_signal > 0:
        peak_signal = feat[NORM_SKIN_COLS].abs().max(axis=1)
        feat = feat.loc[peak_signal >= float(min_signal)].reset_index(drop=True)

    return feat.reset_index(drop=True)



# ── 단일 Trial 처리 ──────────────────────────────────────────────────────────
def process_trial(
    csv_path: Path, out_dir: Path, contact_threshold: float = 0.01
) -> tuple[dict, pd.DataFrame]:
    trial_id = csv_path.stem
    info = parse_trial_name(trial_id)

    print(f"  처리 중: {trial_id}")
    df = pd.read_csv(csv_path)

    # Baseline
    baseline = extract_baseline(df, trial_id, info)
    bl_path = out_dir / f"{trial_id}_baseline.json"
    with open(bl_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    # Grid CSV (센서 반응 기반 접촉점 기준 z_depth)
    grid_df = make_grid_df(df, trial_id, info, baseline, contact_threshold)
    grid_path = out_dir / f"{trial_id}_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    n_pts = grid_df.groupby(["x_mm", "y_mm"]).ngroups
    n_load = (grid_df["phase"] == 0).sum()
    n_unload = (grid_df["phase"] == 1).sum()
    z_min = grid_df["z_depth_mm"].min()
    z_max = grid_df["z_depth_mm"].max()
    print(
        f"    → baseline {baseline['baseline_n_rows']}행 | "
        f"그리드 {n_pts}포인트 | "
        f"total {len(grid_df)}행 (loading {n_load} / unloading {n_unload}) | "
        f"z_depth [{z_min:.3f}, {z_max:.3f}]mm"
    )

    return baseline, grid_df


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


# ── Zarr Export ──────────────────────────────────────────────────────────────
def export_to_zarr(features_df: pd.DataFrame, zarr_path: Path):
    """
    SkinDataset에서 바로 로드 가능한 Zarr 구조로 저장.
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
    
    # SkinDataset 호환 aux_feat 구성: [fx_N, fy_N, depth_mm, radius_mm]
    # 현재 fx, fy는 features_df에 없으므로 (grid_df에는 있음) 0으로 채우거나 
    # 필요시 features_df 구성을 수정해야 함. 여기서는 일단 [0, 0, z_depth, diameter]
    aux_feat = np.zeros((n, 4), dtype=np.float32)
    aux_feat[:, 2] = features_df["z_depth_mm"].values
    aux_feat[:, 3] = features_df["diameter_mm"].values
    
    fz = features_df["fz_bc"].values.astype(np.float32)
    cx = features_df["x_mm"].values.astype(np.float32)
    cy = features_df["y_mm"].values.astype(np.float32)
    depth_mm = features_df["z_depth_mm"].values.astype(np.float32)
    
    # Canvas bounds (SkinDataset 요구사항 대응용 더미 또는 계산값)
    x_bounds = np.tile([GRID_MIN_MM, GRID_MAX_MM], (n, 1)).astype(np.float32)
    y_bounds = np.tile([GRID_MIN_MM, GRID_MAX_MM], (n, 1)).astype(np.float32)

    # Zarr 그룹 생성 및 데이터 저장 (Blosc 압축 기본 적용)
    root = zarr.open_group(str(zarr_path), mode='w')
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
    
    for i in range(n):
        samples_info.append({
            "trial_id": str(trial_ids[i]),
            "phase": "loading" if phases[i] == 0 else "unloading",
            "zarr_path": str(zarr_path),
            "zarr_index": i,
            "depth_bin_mm": float(depth_mm[i])
        })
    
    index_path = zarr_path.parent / "dataset_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples_info}, f, indent=2)


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

    csv_files = sorted(args.raw_dir.glob(args.glob))
    if not csv_files:
        print(f"[오류] {args.raw_dir}/{args.glob} 에서 파일을 찾을 수 없습니다.")
        return

    print(f"[전처리 시작] {len(csv_files)}개 파일 발견")

    # max_diameter 계산
    max_diameter = 0.0
    for f in csv_files:
        info = parse_trial_name(f.stem)
        if info["diameter_mm"] is not None:
            max_diameter = max(max_diameter, info["diameter_mm"])
    if max_diameter == 0:
        max_diameter = 1.0

    material_features: dict[str, list] = {}

    for csv_path in csv_files:
        trial_id = csv_path.stem.replace("_merged", "")
        info = parse_trial_name(trial_id)
        
        print(f"  처리 중: {trial_id}")
        df = pd.read_csv(csv_path)

        # Baseline
        try:
            baseline = extract_baseline(df, trial_id, info)
        except ValueError as e:
            print(f"    [건너뜀] {e}")
            continue
            
        bl_path = dirs["baselines"] / f"{trial_id}_baseline.json"
        with open(bl_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)

        # Grid CSV
        grid_df = make_grid_df(df, trial_id, info, baseline, args.contact_threshold)
        if grid_df.empty:
            print("    [건너뜀] 그리드 데이터가 없습니다.")
            continue
        grid_path = dirs["grid"] / f"{trial_id}_grid.csv"
        grid_df.to_csv(grid_path, index=False)

        # Features CSV
        feat_df = make_features_df(
            grid_df,
            baseline,
            max_diameter,
            z_bin_mm=args.z_bin_mm,
            min_signal=args.min_signal,
        )
        feat_path = dirs["features"] / f"{trial_id}_features.csv"
        feat_df.to_csv(feat_path, index=False)

        mat = grid_df["material"].iloc[0]
        material_features.setdefault(mat, []).append(feat_df)
        
        print(f"    → total {len(grid_df):,}행 정제 완료")

    # 소재별 통합 및 Zarr 변환
    print()
    for mat in sorted(material_features):
        all_feat = pd.concat(material_features[mat], ignore_index=True)
        
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
            export_to_zarr(all_feat, zarr_dir)

        print(f"  [{mat}] 최종 {len(all_feat):,}행 가공 완료")

    print(f"\n[완료] 결과물이 {base_out.resolve()} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
