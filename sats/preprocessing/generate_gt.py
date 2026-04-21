#!/usr/bin/env python3
"""
Boussinesq Ground Truth (GT) 압력맵 생성기
==========================================

merged CSV 의 각 row 에서 (x_mm, y_mm, Fz) 와 인덴터 직경을 읽어,
Boussinesq 탄성 반공간 모델 기반 40×40 수직응력 맵을 생성한다.

    GT(u, v) = Σᵢ  3·Fᵢ·z_s³ / [2π·Rᵢ^5]
    Rᵢ = sqrt((u−xᵢ)² + (v−yᵢ)² + z_s²)
    Fᵢ = (Fz / πa²) · Δstep²   (균일 원형 압력 × 면적)

GT 는 Fz 에 대해 선형이므로, 단위 커널(Fz=1) 을 기반으로 계산.

── 최적화 전략 (base kernel) ──────────────────────────────────────────
  K(u,v; cx,cy) = K₀(u−cx, v−cy)  (순수 평행이동 불변성)

  1. 79×79 extended grid (du ∈ [-19.5, 19.5] step 0.5 mm) 위에서
     원점 기준 base kernel K₀ 를 직경당 1회만 계산.
  2. (40,40,40,40) lookup table 구축:
         all_kernels[i_cx, j_cy] = base_kernel[39-i_cx:79-i_cx, 39-j_cy:79-j_cy]
  3. 각 row: nearest grid index 스냅 후 lookup + Fz 스칼라 곱.
     벡터화 청킹으로 메모리 제한.

출력:
  {trial_id}_targets.npy   — (N, 40, 40) float32
  {trial_id}_gt_meta.json  — 메타데이터
  dataset_index.json       — 전체 trial 인덱스

사용법:
  python3 generate_gt.py
  python3 generate_gt.py --raw-dir ../../raw_data --out-dir ./gt_output
  python3 generate_gt.py --z-s 2.5 --patch-step 0.05
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # type: ignore[misc]
        return it

# ── 상수 ──────────────────────────────────────────────────────────────────────
GRID_SIZE   = 40
GRID_MIN_MM = -9.75
GRID_MAX_MM =  9.75
GRID_STEP_MM = 0.5
EXT_SIZE    = 79        # 2*GRID_SIZE - 1 = 79  (extended kernel grid)
EXT_HALF    = 39        # EXT_SIZE // 2
ASSEMBLY_CHUNK = 20_000 # GT 조립 청크 크기 (메모리 상한 조정 가능)

_INDENTER_RE = re.compile(r"^d(?P<d>\d+(?:\.\d+)?)$",    re.IGNORECASE)
_Z_RE        = re.compile(r"^z_?(?P<z>\d+(?:\.\d+)?)mm$", re.IGNORECASE)
_TEST_RE     = re.compile(r"^test(?P<n>\d+)$",             re.IGNORECASE)


def compute_beta(
    p_kpa: np.ndarray,
    beta_mode: str,
    beta_coeffs: tuple[float, float, float],
    beta_min: float,
    beta_max: float,
) -> np.ndarray:
    """
    논문 S9의 rectification factor β(p) 근사.
    p_kpa: 평균 접촉 압력 [kPa]
    """
    if beta_mode == "none":
        return np.ones_like(p_kpa, dtype=np.float64)
    if beta_mode == "poly2":
        c0, c1, c2 = beta_coeffs
        beta = c0 + c1 * p_kpa + c2 * (p_kpa ** 2)
        return np.clip(beta, beta_min, beta_max)
    raise ValueError(f"알 수 없는 beta_mode: {beta_mode}")


def build_z0_map_from_trial(
    z_mm: np.ndarray,
    fz_raw: np.ndarray,
    i_cx: np.ndarray,
    j_cy: np.ndarray,
    on_grid: np.ndarray,
    force_thresh: float,
    estimator: str = "p05",
) -> np.ndarray:
    """
    (x, y)별 접촉 시작 높이 z0 추정 맵 (40x40).
    추정 기준:
      - |Fz| >= force_thresh 조건을 만족하는 샘플들
      - estimator: min / p05 / first
    """
    z0_map = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float64)
    valid = on_grid & np.isfinite(z_mm) & np.isfinite(fz_raw) & (np.abs(fz_raw) >= force_thresh)
    if not valid.any():
        return z0_map

    ii = i_cx[valid]
    jj = j_cy[valid]
    zz = z_mm[valid]

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            sel = (ii == i) & (jj == j)
            if not sel.any():
                continue
            z_vals = zz[sel]
            if estimator == "min":
                z0_map[i, j] = float(np.min(z_vals))
            elif estimator == "p05":
                z0_map[i, j] = float(np.percentile(z_vals, 5.0))
            elif estimator == "first":
                z0_map[i, j] = float(z_vals[0])
            else:
                raise ValueError(f"알 수 없는 z0 estimator: {estimator}")

    return z0_map


# ── 그리드 좌표 ───────────────────────────────────────────────────────────────

def make_grid_coords(
    size: int   = GRID_SIZE,
    gmin: float = GRID_MIN_MM,
    gmax: float = GRID_MAX_MM,
    mode: str   = "scan_points",
) -> tuple[np.ndarray, np.ndarray]:
    """
    GT 좌표축 (1-D) 반환.
    mode="scan_points":
      - 스캔 포인트와 동일한 40점 좌표 사용
      - [-9.75, -9.25, ..., 9.25, 9.75]
    mode="cell_centers":
      - 40개 cell 중심 좌표 사용
      - [-9.50, -9.00, ..., 9.00, 9.50]
    """
    if mode == "scan_points":
        c = np.linspace(gmin, gmax, size)
    elif mode == "cell_centers":
        c = np.linspace(gmin + GRID_STEP_MM / 2.0, gmax - GRID_STEP_MM / 2.0, size)
    else:
        raise ValueError(f"알 수 없는 grid mode: {mode}")
    return c, c  # (grid_x, grid_y)


def make_extended_grid() -> np.ndarray:
    """
    79점 extended grid: du ∈ [-19.5, -19.0, …, 0, …, 19.0, 19.5]  (0.5 mm 간격)
    base kernel 계산에 사용.
    """
    return np.linspace(-19.5, 19.5, EXT_SIZE)


# ── 원형 패치 이산화 ───────────────────────────────────────────────────────────

def make_patch_template(radius: float, step: float) -> np.ndarray:
    """
    원점 기준 원형 패치를 step 간격 격자로 이산화한다.

    n_half = floor(radius/step) 로 대칭 정수 격자를 만들어
    부동소수점 오차에 의한 비대칭 문제를 방지한다.

    Parameters
    ----------
    radius : 인덴터 반경 [mm]
    step   : 이산화 간격 [mm]

    Returns
    -------
    pts : (M, 2) float64  — 원 내부 점들의 (Δx, Δy) offset
    """
    if radius <= 0:
        return np.zeros((1, 2))
    n_half = int(radius / step)
    offsets = np.arange(-n_half, n_half + 1) * step          # 대칭 보장
    gx, gy  = np.meshgrid(offsets, offsets)
    inside  = gx ** 2 + gy ** 2 <= radius ** 2 + 1e-9        # 경계점 포함
    pts = np.column_stack([gx[inside], gy[inside]])
    return pts if len(pts) > 0 else np.zeros((1, 2))


# ── 단위 커널 계산 (핵심 벡터화 함수) ──────────────────────────────────────────

def compute_unit_kernel(
    cx: float,
    cy: float,
    radius: float,
    patch_tpl: np.ndarray,
    patch_step: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z_s: float,
    chunk: int = 4096,
) -> np.ndarray:
    """
    단위 하중(Fz = 1 N) 에 대한 G×G Boussinesq 수직응력 맵 반환.
    실제 GT 는 ``kernel * Fz`` 로 구한다.

    알고리즘
    --------
    σ_zz(u,v) = Σᵢ  3·Fᵢ·z_s³ / [2π·Rᵢ^5]
    Fz=1 이면  Fᵢ = step² / (π·a²)

    따라서 prefactor = 3·z_s³·step² / (2π²·a²)
    그리고  Σᵢ 1/Rᵢ^5 를 numpy 브로드캐스팅으로 계산.

    Parameters
    ----------
    cx, cy      : 인덴터 중심 위치 [mm]
    radius      : 인덴터 반경 a = d/2 [mm]
    patch_tpl   : (M, 2) 원점 기준 패치 offset
    patch_step  : 패치 이산화 간격 [mm]
    grid_x/y    : (G,) GT 셀 중심 좌표 [mm]
    z_s         : 센서 유효 깊이 [mm]
    chunk       : 메모리 절약용 청크 크기

    Returns
    -------
    kernel : (G, G) float32
    """
    patch_xy = patch_tpl + np.array([cx, cy])  # (M, 2)  실제 패치 좌표
    M        = len(patch_xy)
    G        = len(grid_x)
    z_sq     = z_s * z_s

    # Fz=1 기준 각 이산 점의 힘:  Fi = step² / (π·a²)
    # prefactor = 3·Fi·z_s³ / (2π) = 3·z_s³·step² / (2π²·a²)
    pre = (3.0 * (z_s ** 3) * (patch_step ** 2)) / (2.0 * np.pi ** 2 * radius ** 2)

    inv_r5_sum = np.zeros((G, G), dtype=np.float64)

    for s in range(0, M, chunk):
        e  = min(s + chunk, M)
        px = patch_xy[s:e, 0]  # (C,)
        py = patch_xy[s:e, 1]  # (C,)

        # 브로드캐스팅: (G, 1, C) + (1, G, C) → (G, G, C)
        r_sq = (
            (grid_x[:, None, None] - px[None, None, :]) ** 2
          + (grid_y[None, :, None] - py[None, None, :]) ** 2
          + z_sq
        )
        inv_r5_sum += np.sum(r_sq ** (-2.5), axis=2)

    return (pre * inv_r5_sum).astype(np.float32)


# ── Base kernel & lookup table ─────────────────────────────────────────────────

def compute_base_kernel(
    radius: float,
    patch_step: float,
    z_s: float,
    chunk: int = 4096,
) -> np.ndarray:
    """
    79×79 base kernel K₀(du, dv) 계산 (원점 기준, 직경당 1회).

    K₀ 는 원점 (cx=0, cy=0) 에서의 단위 커널을 extended grid 위에서 계산한 것.
    임의 위치 (cx, cy) 의 kernel = K₀(u-cx, v-cy) 이므로
    base kernel 슬라이싱만으로 모든 위치의 kernel 을 구할 수 있다.

    Returns
    -------
    base_kernel : (79, 79) float32
    """
    ext_grid  = make_extended_grid()
    patch_tpl = make_patch_template(radius, patch_step)
    return compute_unit_kernel(
        0.0, 0.0, radius, patch_tpl, patch_step,
        ext_grid, ext_grid, z_s, chunk,
    )


def build_all_kernels(base_kernel: np.ndarray) -> np.ndarray:
    """
    (79, 79) base kernel → (40, 40, 40, 40) lookup table.

    all_kernels[i_cx, j_cy, :, :] = kernel for contact at grid index (i_cx, j_cy)

    슬라이싱 공식:
        all_kernels[i, j] = base_kernel[EXT_HALF-i : EXT_HALF-i+40,
                                        EXT_HALF-j : EXT_HALF-j+40]
    i.e., base_kernel[39-i:79-i, 39-j:79-j]

    메모리: 40^4 × 4 bytes = 10.24 MB
    """
    G    = GRID_SIZE
    H    = EXT_HALF  # 39
    all_k = np.empty((G, G, G, G), dtype=np.float32)
    for i in range(G):
        for j in range(G):
            all_k[i, j] = base_kernel[H - i : H - i + G, H - j : H - j + G]
    return all_k


# ── 경로 파싱 & 파일 탐색 ──────────────────────────────────────────────────────

def discover_merged_csvs(raw_dir: Path) -> list[Path]:
    """raw_dir 하위의 *_merged.csv 파일을 모두 반환."""
    return sorted(raw_dir.glob("**/*_merged.csv"))


def parse_trial_info(csv_path: Path, raw_dir: Path) -> dict:
    """
    CSV 경로에서 (material, diameter_mm, z_max_mm, trial_no) 추출.
    지원 구조: raw_data/<material>/d<X>/z_<Z>mm/test<N>/
    """
    try:
        parts = csv_path.parent.relative_to(raw_dir).parts
    except ValueError:
        parts = csv_path.parent.parts

    if len(parts) >= 4:
        material, d_dir, z_dir, t_dir = parts[-4:]
        dm = _INDENTER_RE.match(d_dir)
        zm = _Z_RE.match(z_dir)
        tm = _TEST_RE.match(t_dir)
        if dm and zm and tm:
            return {
                "material":    material.lower(),
                "diameter_mm": float(dm.group("d")),
                "z_max_mm":    float(zm.group("z")),
                "trial_no":    int(tm.group("n")),
            }

    # fallback: 파일명에서 추출 불가
    stem = csv_path.stem.replace("_merged", "")
    return {"material": stem, "diameter_mm": None, "z_max_mm": None, "trial_no": None}


def make_trial_id(info: dict) -> str:
    d = f"d{info['diameter_mm']:g}" if info["diameter_mm"] is not None else "dX"
    z = f"z{info['z_max_mm']:g}"    if info["z_max_mm"]    is not None else "zX"
    n = f"test{info['trial_no']}"   if info["trial_no"]    is not None else "testX"
    return f"{info['material']}_{d}_{z}_{n}"


# ── Trial 단위 처리 ────────────────────────────────────────────────────────────

def process_trial(
    csv_path:     Path,
    raw_dir:      Path,
    out_dir:      Path,
    grid_x:       np.ndarray,
    grid_y:       np.ndarray,
    kernel_cache: dict,
    z_s:          float,
    patch_step:   float,
    fz_min_abs:   float,
    fz_mode:      str,
    beta_mode:    str,
    beta_coeffs:  tuple[float, float, float],
    beta_min:     float,
    beta_max:     float,
    z_comp_mode:  str,
    z_contact_force_thresh: float,
    z0_estimator: str,
    z_k:          float,
    z_min:        float,
    z_max:        float,
    z_cache_step: float,
    grid_tol_mm:  float,
    drop_offgrid: bool,
) -> dict | None:
    """
    단일 trial merged CSV → (N, 40, 40) targets.npy 저장.

    최적화 (base kernel + lookup table):
      1. 직경당 79×79 base kernel 을 1회 계산 (trial 간 공유)
      2. (40,40,40,40) lookup table 빌드 후 position snap → O(1) kernel 조회
      3. 벡터화 청킹으로 GT 조립 (메모리 상한: ASSEMBLY_CHUNK × 40×40 × 4 bytes)

    Parameters
    ----------
    kernel_cache : {diameter_mm: {"all_kernels": ndarray(40,40,40,40)}}
    """
    info     = parse_trial_info(csv_path, raw_dir)
    trial_id = make_trial_id(info)
    d_mm     = info["diameter_mm"]

    if d_mm is None:
        print(f"  [건너뜀] {trial_id}: diameter 파싱 실패")
        return None

    print(f"  [{trial_id}] 로드 중…")
    df = pd.read_csv(csv_path)

    req_cols = ["x_mm", "y_mm", "Fz"]
    if z_comp_mode == "xy_contact":
        req_cols.append("z_mm")
    for col in req_cols:
        if col not in df.columns:
            print(f"  [건너뜀] {trial_id}: 컬럼 '{col}' 없음")
            return None

    radius = d_mm / 2.0

    # ── 위치 추출 ──────────────────────────────────────────────────────────────
    x_arr_raw  = df["x_mm"].to_numpy(dtype=np.float64)
    y_arr_raw  = df["y_mm"].to_numpy(dtype=np.float64)
    fz_arr_raw = df["Fz"].to_numpy(dtype=np.float64)
    z_arr_raw = df["z_mm"].to_numpy(dtype=np.float64) if "z_mm" in df.columns else None
    G      = GRID_SIZE

    # raw_merge 결과에는 이동 구간 샘플이 섞일 수 있으므로,
    # 40x40 스캔 그리드에 충분히 가까운 샘플만 통과시킨다.
    i_cx_arr = np.rint((x_arr_raw - grid_x[0]) / GRID_STEP_MM).astype(np.int32)
    j_cy_arr = np.rint((y_arr_raw - grid_y[0]) / GRID_STEP_MM).astype(np.int32)

    x_snap = grid_x[0] + i_cx_arr.astype(np.float64) * GRID_STEP_MM
    y_snap = grid_y[0] + j_cy_arr.astype(np.float64) * GRID_STEP_MM
    in_bounds = (i_cx_arr >= 0) & (i_cx_arr < G) & (j_cy_arr >= 0) & (j_cy_arr < G)
    on_grid = in_bounds & (np.abs(x_arr_raw - x_snap) <= grid_tol_mm) & (np.abs(y_arr_raw - y_snap) <= grid_tol_mm)

    n_input_rows = len(df)
    n_offgrid = int((~on_grid).sum())
    if drop_offgrid:
        keep = on_grid
        i_cx_arr = i_cx_arr[keep]
        j_cy_arr = j_cy_arr[keep]
        fz_arr = fz_arr_raw[keep]
    else:
        i_cx_arr = np.clip(i_cx_arr, 0, G - 1)
        j_cy_arr = np.clip(j_cy_arr, 0, G - 1)
        fz_arr = fz_arr_raw
    if z_arr_raw is not None:
        z_arr = z_arr_raw[keep] if drop_offgrid else z_arr_raw
    else:
        z_arr = None

    if fz_mode == "positive_only":
        fz_eff = fz_arr
        active_mask = fz_eff > 0.0
    elif fz_mode == "abs":
        fz_eff = np.abs(fz_arr)
        active_mask = fz_eff > 0.0
    elif fz_mode == "signed":
        fz_eff = fz_arr
        active_mask = fz_eff != 0.0
    else:
        raise ValueError(f"알 수 없는 fz_mode: {fz_mode}")
    if fz_min_abs > 0:
        active_mask &= np.abs(fz_arr) >= fz_min_abs

    N = len(fz_eff)
    n_positive = int((fz_arr > 0).sum())
    n_negative = int((fz_arr < 0).sum())
    n_active = int(active_mask.sum())

    # ── (x,y)별 z_s 보정: z_eff = clip(z_s + z_k * (z0(x,y)-median(z0)), z_min, z_max)
    z0_map = np.full((G, G), np.nan, dtype=np.float64)
    z_eff_arr = np.full(N, z_s, dtype=np.float64)
    if z_comp_mode == "xy_contact":
        assert z_arr is not None
        # on_grid는 원본 길이 기준이므로 보정된 길이로 맞춰서 재생성
        on_grid_local = np.ones(N, dtype=bool) if drop_offgrid else on_grid
        z0_map = build_z0_map_from_trial(
            z_arr,
            fz_arr,
            i_cx_arr,
            j_cy_arr,
            on_grid_local,
            z_contact_force_thresh,
            estimator=z0_estimator,
        )
        valid_z0 = np.isfinite(z0_map)
        if valid_z0.any():
            z0_med = float(np.median(z0_map[valid_z0]))
            dz0_map = np.zeros_like(z0_map)
            dz0_map[valid_z0] = z0_map[valid_z0] - z0_med
            dz = dz0_map[i_cx_arr, j_cy_arr]
            z_eff_arr = np.clip(z_s + z_k * dz, z_min, z_max)
        else:
            z_eff_arr = np.full(N, np.clip(z_s, z_min, z_max), dtype=np.float64)
    else:
        z_eff_arr = np.full(N, np.clip(z_s, z_min, z_max), dtype=np.float64)

    # z_eff별 kernel key 양자화
    z_eff_key = np.round(z_eff_arr / z_cache_step) * z_cache_step
    z_eff_key = np.clip(z_eff_key, z_min, z_max)

    # p = |Fz|/(pi*a^2), N/mm^2 -> MPa 이므로 kPa 변환 x1000
    p_kpa = (np.abs(fz_arr) / (np.pi * radius * radius)) * 1000.0
    beta_arr = compute_beta(p_kpa, beta_mode, beta_coeffs, beta_min, beta_max)

    # ── GT 조립 (벡터화 청킹) ──────────────────────────────────────────────────
    print(f"  [{trial_id}] GT 조립: {N:,}행 / 입력 {n_input_rows:,}행 "
          f"(off-grid 제외: {n_offgrid:,}, 활성:{n_active:,}, Fz>0:{n_positive:,}, Fz<0:{n_negative:,}), "
          f"청크={ASSEMBLY_CHUNK:,}, fz_mode={fz_mode}, fz_min_abs={fz_min_abs}, "
          f"beta_mode={beta_mode}, z_comp_mode={z_comp_mode}")

    targets = np.zeros((N, G, G), dtype=np.float32)
    t_asm   = time.perf_counter()

    for s in range(0, N, ASSEMBLY_CHUNK):
        e    = min(s + ASSEMBLY_CHUNK, N)
        fz_c = fz_eff[s:e]              # (C,)
        mask = active_mask[s:e]
        if not mask.any():
            continue

        ics  = i_cx_arr[s:e][mask]      # (K,)
        jcs  = j_cy_arr[s:e][mask]      # (K,)
        fzs  = fz_c[mask].astype(np.float32)  # (K,)
        betas = beta_arr[s:e][mask].astype(np.float32)  # (K,)
        zkeys = z_eff_key[s:e][mask].astype(np.float32)  # (K,)

        local_idx = np.where(mask)[0]
        # z_eff가 다른 샘플은 서로 다른 커널셋을 사용해야 하므로 key별로 분리 계산
        for z_key in np.unique(zkeys):
            z_key_f = float(z_key)
            kkey = (float(d_mm), z_key_f)
            if kkey not in kernel_cache:
                t0 = time.perf_counter()
                base_k = compute_base_kernel(radius, patch_step, z_key_f)
                all_k  = build_all_kernels(base_k)
                kernel_cache[kkey] = {"all_kernels": all_k}
                t1 = time.perf_counter()
                print(f"  [d={d_mm}mm,z={z_key_f:.3f}] kernel cache 생성 {t1-t0:.1f}s")
            all_kernels = kernel_cache[kkey]["all_kernels"]

            sub = (zkeys == z_key)
            kernels = all_kernels[ics[sub], jcs[sub]]  # (Kz,40,40)
            scale = (fzs[sub] * betas[sub])[:, None, None]
            targets_sub = kernels * scale
            targets[s + local_idx[sub]] = targets_sub

    t_asm2 = time.perf_counter()
    print(f"  [{trial_id}] 조립 완료: {t_asm2-t_asm:.1f}s")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    npy_path  = out_dir / f"{trial_id}_targets.npy"
    meta_path = out_dir / f"{trial_id}_gt_meta.json"

    np.save(npy_path, targets)
    size_mb = npy_path.stat().st_size / 1e6

    meta: dict = {
        "trial_id":       trial_id,
        "material":       info["material"],
        "diameter_mm":    d_mm,
        "z_max_mm":       info["z_max_mm"],
        "trial_no":       info["trial_no"],
        "z_s_mm":         z_s,
        "patch_step_mm":  patch_step,
        "fz_min_abs_n":   fz_min_abs,
        "n_input_rows":   n_input_rows,
        "n_total_rows":   N,
        "n_offgrid_rows": n_offgrid,
        "drop_offgrid":   drop_offgrid,
        "grid_tol_mm":    grid_tol_mm,
        "fz_mode":        fz_mode,
        "beta_mode":      beta_mode,
        "beta_coeffs":    list(beta_coeffs),
        "beta_clip_min":  beta_min,
        "beta_clip_max":  beta_max,
        "p_kpa_mean":     float(np.mean(p_kpa)) if len(p_kpa) > 0 else 0.0,
        "beta_mean":      float(np.mean(beta_arr)) if len(beta_arr) > 0 else 1.0,
        "z_comp_mode":    z_comp_mode,
        "z_contact_force_thresh_n": z_contact_force_thresh,
        "z0_estimator":   z0_estimator,
        "z_k":            z_k,
        "z_min_mm":       z_min,
        "z_max_mm":       z_max,
        "z_cache_step_mm": z_cache_step,
        "z_eff_mean_mm":  float(np.mean(z_eff_arr)) if len(z_eff_arr) > 0 else z_s,
        "z_eff_min_mm":   float(np.min(z_eff_arr)) if len(z_eff_arr) > 0 else z_s,
        "z_eff_max_mm":   float(np.max(z_eff_arr)) if len(z_eff_arr) > 0 else z_s,
        "z0_valid_cells": int(np.isfinite(z0_map).sum()),
        "n_negative_fz":  n_negative,
        "n_active_rows":  n_active,
        "n_positive_fz":  n_positive,
        "n_zero_gt":      N - n_active,
        "gt_shape":       [N, G, G],
        "gt_dtype":       "float32",
        "file_size_mb":   round(size_mb, 1),
        "targets_file":   npy_path.name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  [{trial_id}] → {npy_path.name}  ({size_mb:.0f} MB)")
    return meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    THIS = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description="Boussinesq GT 압력맵 생성기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=THIS.parents[2] / "raw_data",
        help="raw_data 루트 (하위에서 *_merged.csv 를 재귀 탐색)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=THIS.parent / "gt_output",
        help="GT npy 저장 디렉토리",
    )
    parser.add_argument(
        "--z-s",
        type=float,
        default=2.0,
        help="센서 유효 깊이 z_s [mm]",
    )
    parser.add_argument(
        "--patch-step",
        type=float,
        default=0.1,
        help="인덴터 패치 이산화 간격 [mm]",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help="GT 맵 한 변 크기 (현재 40만 지원)",
    )
    parser.add_argument(
        "--grid-mode",
        type=str,
        choices=["scan_points", "cell_centers"],
        default="scan_points",
        help="GT 좌표축 정의 방식",
    )
    parser.add_argument(
        "--fz-mode",
        type=str,
        choices=["positive_only", "abs", "signed"],
        default="positive_only",
        help="Fz를 GT 하중으로 변환하는 방식",
    )
    parser.add_argument(
        "--fz-min-abs",
        type=float,
        default=0.05,
        help="유효 Fz 최소 절대값 [N]. 이 값 미만은 비접촉/노이즈로 간주",
    )
    parser.add_argument(
        "--beta-mode",
        type=str,
        choices=["none", "poly2"],
        default="none",
        help="S9 rectification factor β(p) 적용 방식",
    )
    parser.add_argument("--beta-c0", type=float, default=1.0, help="β(p)=c0+c1*p+c2*p^2 계수 c0 (p in kPa)")
    parser.add_argument("--beta-c1", type=float, default=0.0, help="β(p)=c0+c1*p+c2*p^2 계수 c1 (p in kPa)")
    parser.add_argument("--beta-c2", type=float, default=0.0, help="β(p)=c0+c1*p+c2*p^2 계수 c2 (p in kPa)")
    parser.add_argument("--beta-min", type=float, default=0.5, help="β 클램프 최소값")
    parser.add_argument("--beta-max", type=float, default=2.0, help="β 클램프 최대값")
    parser.add_argument(
        "--z-comp-mode",
        type=str,
        choices=["none", "xy_contact"],
        default="none",
        help="(x,y)별 접촉 높이 편차 기반 z_s 보정 모드",
    )
    parser.add_argument(
        "--z-contact-force-thresh",
        type=float,
        default=0.2,
        help="z0 추정 시 접촉 판정용 |Fz| 최소값 [N]",
    )
    parser.add_argument(
        "--z0-estimator",
        type=str,
        choices=["min", "p05", "first"],
        default="p05",
        help="(x,y)별 접촉 시작 높이 z0 추정 방식",
    )
    parser.add_argument(
        "--z-k",
        type=float,
        default=1.0,
        help="z_eff = z_s + z_k * (z0(x,y)-median(z0)) 계수",
    )
    parser.add_argument("--z-min", type=float, default=1.5, help="z_eff 최소값 [mm]")
    parser.add_argument("--z-max", type=float, default=3.5, help="z_eff 최대값 [mm]")
    parser.add_argument("--z-cache-step", type=float, default=0.05, help="z_eff 커널 캐시 양자화 step [mm]")
    parser.add_argument(
        "--grid-tol-mm",
        type=float,
        default=0.05,
        help="그리드 정합 허용 오차 [mm]",
    )
    parser.add_argument(
        "--keep-offgrid",
        action="store_true",
        help="오프그리드 row를 제외하지 않고 최근접 그리드로 스냅",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Boussinesq GT 압력맵 생성기  [base kernel 최적화]")
    print("=" * 60)
    print(f"  raw_dir    : {args.raw_dir}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  z_s        : {args.z_s} mm")
    print(f"  patch_step : {args.patch_step} mm")
    print(f"  grid_size  : {args.grid_size}×{args.grid_size}")
    print(f"  grid_mode  : {args.grid_mode}")
    print(f"  fz_mode    : {args.fz_mode}")
    print(f"  fz_min_abs : {args.fz_min_abs} N")
    print(f"  beta_mode  : {args.beta_mode}")
    if args.beta_mode == "poly2":
        print(f"  beta(p)    : {args.beta_c0} + {args.beta_c1}*p + {args.beta_c2}*p^2  (p in kPa)")
        print(f"  beta clip  : [{args.beta_min}, {args.beta_max}]")
    print(f"  grid_tol   : {args.grid_tol_mm} mm")
    print(f"  off-grid   : {'keep+snap' if args.keep_offgrid else 'drop'}")
    print(f"  z_comp_mode: {args.z_comp_mode}")
    if args.z_comp_mode == "xy_contact":
        print(f"  z0 cfg     : force>={args.z_contact_force_thresh}N, estimator={args.z0_estimator}")
        print(f"  z_eff cfg  : z_k={args.z_k}, clip=[{args.z_min},{args.z_max}]mm, cache_step={args.z_cache_step}mm")
    print()

    if args.grid_size != GRID_SIZE:
        print(f"[오류] grid_size={args.grid_size}는 아직 지원하지 않습니다. 현재는 {GRID_SIZE}만 지원합니다.")
        return
    grid_x, grid_y = make_grid_coords(args.grid_size, GRID_MIN_MM, GRID_MAX_MM, args.grid_mode)

    csv_files = discover_merged_csvs(args.raw_dir)
    if not csv_files:
        print(f"[오류] {args.raw_dir} 에서 *_merged.csv 를 찾을 수 없습니다.")
        return
    print(f"발견된 trial CSV: {len(csv_files)}개\n")

    # 직경별 lookup table 캐시 (trial 간 공유)
    kernel_cache: dict = {}
    all_metas:    list[dict] = []
    t_total = time.perf_counter()

    for csv_path in tqdm(csv_files, desc="trials", unit="trial"):
        meta = process_trial(
            csv_path, args.raw_dir, args.out_dir,
            grid_x, grid_y, kernel_cache,
            args.z_s, args.patch_step,
            args.fz_min_abs,
            args.fz_mode, args.beta_mode, (args.beta_c0, args.beta_c1, args.beta_c2),
            args.beta_min, args.beta_max,
            args.z_comp_mode, args.z_contact_force_thresh, args.z0_estimator,
            args.z_k, args.z_min, args.z_max, args.z_cache_step,
            args.grid_tol_mm, not args.keep_offgrid,
        )
        if meta:
            all_metas.append(meta)

    # ── 데이터셋 인덱스 저장 ──────────────────────────────────────────────────
    total_rows = sum(m["n_total_rows"] for m in all_metas)
    total_mb   = sum(m["file_size_mb"] for m in all_metas)
    index = {
        "z_s_mm":        args.z_s,
        "patch_step_mm": args.patch_step,
        "grid_size":     args.grid_size,
        "grid_mode":     args.grid_mode,
        "grid_min_mm":   GRID_MIN_MM,
        "grid_max_mm":   GRID_MAX_MM,
        "grid_step_mm":  GRID_STEP_MM,
        "grid_tol_mm":   args.grid_tol_mm,
        "drop_offgrid":  not args.keep_offgrid,
        "fz_mode":       args.fz_mode,
        "fz_min_abs_n":  args.fz_min_abs,
        "beta_mode":     args.beta_mode,
        "beta_coeffs":   [args.beta_c0, args.beta_c1, args.beta_c2],
        "beta_clip_min": args.beta_min,
        "beta_clip_max": args.beta_max,
        "z_comp_mode":   args.z_comp_mode,
        "z_contact_force_thresh_n": args.z_contact_force_thresh,
        "z0_estimator":  args.z0_estimator,
        "z_k":           args.z_k,
        "z_min_mm":      args.z_min,
        "z_max_mm":      args.z_max,
        "z_cache_step_mm": args.z_cache_step,
        "total_trials":  len(all_metas),
        "total_rows":    total_rows,
        "total_size_mb": round(total_mb, 1),
        "trials":        all_metas,
    }
    idx_path = args.out_dir / "dataset_index.json"
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t_total
    print()
    print("=" * 60)
    print(f"완료: {len(all_metas)}개 trial, {total_rows:,}행, {total_mb:.0f} MB")
    print(f"총 소요 시간: {elapsed:.0f}s")
    print(f"인덱스: {idx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
