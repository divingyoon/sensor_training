#!/usr/bin/env python3
"""미세 스텝 위치 분해능 분석 (D10, 오프라인) — 모터 상대변위 vs SATS centroid 상대변위.

★좌표 규약(중요): 모터 원점과 센서 원점은 정렬 불가 → **상대 변위**로 평가한다.
  - pos_mm = 모터 **절대 좌표**(원점 임의). 첫 압입점(모터 0,0 등)이 기준점.
  - 예측·GT 모두 기준점 대비 변위(Δ)로 바꿔 비교 → 원점 불일치(상수 offset)는
    소거된다. slope·R²·σ·분해능은 원래 offset-불변(회귀 절편이 흡수).

취득 절차(사용자 프로토콜): 모터 (0,0)에서 1회 압입(기준) → 절대 좌표로 0.1mm 등
스텝 이동 → 동일 z 압입 반복. 각 점 dwell 중 ≥20프레임 기록.

산출(SPEC.md 갱신 근거):
  - slope·R²      : 예측 변위가 실제 모터 변위를 얼마나 추종하나 (1=완벽)
  - σ (per-pos)   : 같은 위치 반복 예측의 흔들림 = 반복 노이즈
  - rel err       : 기준점 정렬 후 |예측Δ−GTΔ| median (상대 정확도)
  - resolution    : 2σ/slope = 두 위치를 분간 가능한 최소 간격(분해능)

────────────────────────────────────────────────────────────────────────
입력 형식 (둘 중 하나):
  A) --data-dir DIR : 위치당 npz 1개(파일명 순=취득 순, 첫 파일=기준점). 각 npz =
        sensor   float32 [F,16]  (F≥window_size, pct 상대% = 라이브 데모 윈도우값)
        pos_mm   float32 [2]     (모터 절대 x,y[mm] — 센서 원점과 무관)
        baseline float32 [16]    (선택; sensor가 raw면 --input raw 로 pct 변환)
  B) --data FILE.npz : 전 프레임 한 파일.
        sensor   [N,16],  pos_mm [N,2]   (동일 위치 연속 프레임끼리 묶음)
  지름은 D10 고정(--diameter 10).
────────────────────────────────────────────────────────────────────────

실행(4090):
  .venv/bin/python scripts/analyze_localization_resolution.py --data-dir learning_data/xy_fine/d10
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sats.inference.inference_engine import SATSInferenceEngine       # noqa: E402
from sats.tools.multicontact_metrics import detect_peaks              # noqa: E402


def _to_pct(sensor: np.ndarray, baseline: np.ndarray | None, is_raw: bool) -> np.ndarray:
    """raw면 baseline으로 pct(상대%) 변환, 이미 pct면 그대로."""
    if not is_raw:
        return sensor.astype(np.float32)
    if baseline is None:
        raise ValueError("--input raw 인데 baseline 없음")
    return ((sensor / baseline[None, :] - 1.0) * 100.0).astype(np.float32)


def _load_positions(args) -> list[tuple[np.ndarray, np.ndarray]]:
    """(sensor_pct[F,16], pos_mm[2]) 리스트로 정규화 로드."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    if args.data_dir:
        files = sorted(glob.glob(str(Path(args.data_dir) / "*.npz")))
        if not files:
            raise FileNotFoundError(f"npz 없음: {args.data_dir}")
        for f in files:
            z = np.load(f)
            sensor = _to_pct(z["sensor"].astype(np.float32),
                             z["baseline"].astype(np.float32) if "baseline" in z else None,
                             args.input == "raw")
            pos = np.asarray(z["pos_mm"], np.float32).reshape(-1)[:2]
            out.append((sensor, pos))
    else:
        z = np.load(args.data)
        sensor = _to_pct(z["sensor"].astype(np.float32),
                         z["baseline"].astype(np.float32) if "baseline" in z else None,
                         args.input == "raw")
        pos = np.asarray(z["pos_mm"], np.float32).reshape(-1, 2)
        # 연속 동일 위치(반올림) 구간으로 묶음
        key = np.round(pos / args.group_round).astype(np.int64)
        start = 0
        for i in range(1, len(pos) + 1):
            if i == len(pos) or np.any(key[i] != key[start]):
                out.append((sensor[start:i], pos[start:i].mean(0)))
                start = i
    return out


def _predict_positions(engine, sensor_pct, subpixel):
    """윈도우 슬라이딩 → 각 윈도우 centroid(또는 argmax) 예측 [P,2]."""
    W = engine.window_size
    if len(sensor_pct) < W:
        return np.zeros((0, 2))
    preds = []
    for i in range(W - 1, len(sensor_pct)):
        pmap = engine.predict(sensor_pct[i - W + 1:i + 1])
        pk = detect_peaks(pmap, grid_min_mm=engine.grid_min_mm, grid_step_mm=engine.grid_step_mm,
                          max_peaks=1, subpixel=subpixel)
        if len(pk):
            preds.append(pk[0, :2])
    return np.asarray(preds, float).reshape(-1, 2)


def analyze(positions, engine):
    """상대변위 기반 분석 — 모터·센서 원점 불일치를 기준점 정렬로 소거.

    GT/예측 모두 첫 위치(기준 압입점) 대비 Δ로 비교. slope·R²·σ·분해능은
    원래 offset-불변(회귀 절편), rel err 는 기준점 정렬 후 상대 정확도.
    """
    gt = np.array([p for _, p in positions], float)                    # [K,2] 모터 절대
    axis = gt[-1] - gt[0]
    axis = axis / (np.linalg.norm(axis) + 1e-9)                        # 스윕 방향 단위벡터
    rows = []
    for sensor, pos in positions:
        pr = _predict_positions(engine, sensor, subpixel=True)
        if len(pr) == 0:
            continue
        mu = pr.mean(0)
        sig_axis = float(np.std(pr @ axis))                           # 스윕축 방향 반복 노이즈
        rows.append({"gt": pos, "mu": mu, "sig": sig_axis,
                     "gt_proj": float(pos @ axis), "pred_proj": float(mu @ axis), "n": len(pr)})
    # ★기준점 정렬: 첫 위치를 Δ=0 으로 — 원점 불일치(상수 offset) 소거
    gt0, mu0 = rows[0]["gt"], rows[0]["mu"]
    for r in rows:
        r["gt_d"] = r["gt"] - gt0                                     # 모터 상대변위
        r["mu_d"] = r["mu"] - mu0                                     # 예측 상대변위
        r["rel"] = float(np.linalg.norm(r["mu_d"] - r["gt_d"]))       # 상대 정확도
    gp = np.array([r["gt_proj"] for r in rows]); gp -= gp[0]          # 투영도 상대화
    pp = np.array([r["pred_proj"] for r in rows]); pp -= pp[0]
    slope, intercept = np.polyfit(gp, pp, 1)
    resid = pp - (slope * gp + intercept)
    r2 = 1.0 - np.sum(resid ** 2) / max(np.sum((pp - pp.mean()) ** 2), 1e-12)
    sig_med = float(np.median([r["sig"] for r in rows]))
    rel_med = float(np.median([r["rel"] for r in rows[1:]])) if len(rows) > 1 else float("nan")
    res_2s = 2.0 * sig_med / max(abs(slope), 1e-6)
    return {"rows": rows, "axis": axis, "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2), "sig_med": sig_med, "rel_med": rel_med,
            "resolution_2sigma": float(res_2s), "gp": gp, "pp": pp}


def make_figure(res, out_png, diameter):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    gp, pp = res["gp"], res["pp"]
    sig = np.array([r["sig"] for r in rows])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.5))
    a0.errorbar(gp, pp, yerr=sig, fmt="o", ms=4, capsize=2, color="#0a7", label="predicted (centroid)")
    lo, hi = gp.min(), gp.max()
    a0.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal (slope 1)")
    a0.set_xlabel("motor displacement from ref (mm)"); a0.set_ylabel("predicted displacement (mm)")
    a0.set_title(f"D{diameter:g}  slope={res['slope']:.3f}  R2={res['r2']:.4f}")
    a0.legend(fontsize=8)
    a1.plot(gp, (pp - (res["slope"] * gp + res["intercept"])), "o-", ms=3, color="#c33")
    a1.axhline(0, color="k", lw=0.6)
    a1.set_xlabel("motor displacement (mm)"); a1.set_ylabel("residual (mm)")
    a1.set_title(f"sigma_med={res['sig_med']:.3f}mm  resolution(2σ)={res['resolution_2sigma']:.3f}mm")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="미세 스텝 위치 분해능 분석(D10)")
    ap.add_argument("--data-dir", default=None, help="위치당 npz 디렉토리(형식 A)")
    ap.add_argument("--data", default=None, help="전 프레임 단일 npz(형식 B)")
    ap.add_argument("--run-dir", default=str(_ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"))
    ap.add_argument("--diameter", type=float, default=10.0)
    ap.add_argument("--input", choices=["pct", "raw"], default="pct", help="sensor가 pct인지 raw인지")
    ap.add_argument("--group-round", type=float, default=0.05, help="[B] 위치 묶기 반올림(mm)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(_ROOT / "history/fig_data/experiments_archive/reeval/loc_resolution_d10.png"))
    args = ap.parse_args()
    if not args.data_dir and not args.data:
        ap.error("--data-dir 또는 --data 필요")

    engine = SATSInferenceEngine(args.run_dir, device=args.device, indenter_diameter_mm=args.diameter)
    positions = _load_positions(args)
    print(f"[분석] {len(positions)} 위치, window={engine.window_size}, D{args.diameter:g}")
    res = analyze(positions, engine)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make_figure(res, args.out, args.diameter)

    print("\n" + "=" * 60)
    print(f"  위치 분해능 — 상대변위 기준 (D{args.diameter:g}, {len(res['rows'])} 위치)")
    print("=" * 60)
    print(f"  추종성   slope = {res['slope']:.3f}   R2 = {res['r2']:.4f}")
    print(f"  반복노이즈 sigma(축) median = {res['sig_med']:.3f} mm")
    print(f"  상대정확도 |predΔ − motorΔ| median = {res['rel_med']:.3f} mm  (기준점 정렬 후)")
    print(f"  ★분해능  2σ/slope = {res['resolution_2sigma']:.3f} mm  "
          f"(이보다 가까운 두 위치는 노이즈에 묻힘)")
    print("=" * 60)
    print(f"  그림: {args.out}")
    print("  → SPEC.md §3 x/y resolution 을 이 수치로 갱신.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
