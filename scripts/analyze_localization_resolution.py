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


def _read_em_v2(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """EM_V2 bin → (t[s], xyzu[N,4]).  x,y,z,u 단위=μm."""
    import struct
    rows = []
    with open(path, "rb") as f:
        f.readline()                               # magic line "EM_V2"
        while True:
            d = f.read(8 + 32)
            if len(d) < 40:
                break
            rows.append(struct.unpack("<Qdddd", d))
    a = np.asarray(rows, float)
    if len(a) == 0:
        raise ValueError(f"EM_V2 레코드 없음: {path}")
    return a[:, 0] / 1e9, a[:, 1:]


def _read_due_v2(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """DUE_V2 bin → (t[s] per frame, raw[N,16]). burst=16센서×10프레임."""
    import struct
    rec = 8 + 16 * 10 * 4
    ts, vals = [], []
    with open(path, "rb") as f:
        f.readline()                               # magic line "DUE_V2"
        while True:
            d = f.read(rec)
            if len(d) < rec:
                break
            ns = struct.unpack("<Q", d[:8])[0]
            v = np.frombuffer(d[8:], dtype="<u4").reshape(16, 10).T   # [frame,sensor]
            ts.append(ns / 1e9)
            vals.append(v)
    if not ts:
        raise ValueError(f"DUE_V2 레코드 없음: {path}")
    t_burst = np.asarray(ts)
    raw = np.concatenate(vals).astype(np.float64)
    # burst 내 10프레임 시각을 선형 보간(200Hz, burst당 50ms)
    t = np.repeat(t_burst, 10) + np.tile(np.arange(10) * 0.005, len(t_burst))
    return t, raw


def _load_v2_session(data_dir: Path, *, hold_min_s: float = 0.3,
                     trim_frac: float = 0.25) -> list[tuple[np.ndarray, np.ndarray]]:
    """V2 취득 세션(due+em bin) → 압입(hold)별 (sensor_pct[F,16], pos_mm[2]).

    - z 정지 구간 중 z가 압입 깊이(최댓값 근처)인 세그먼트 = 압입 hold.
    - hold 앞뒤 trim_frac 씩 잘라 과도(진입/이탈) 제거.
    - baseline = 최초 비압입 대기 구간(z 하위) 채널 평균 → pct 변환.
    - 같은 (x,y) 반복 압입은 프레임을 합쳐 한 위치로(반복 노이즈 포함 측정).
    """
    due = sorted(data_dir.glob("due_v2_*.bin"))
    em = sorted(data_dir.glob("em_v2_*.bin"))
    if not due or not em:
        raise FileNotFoundError(f"due_v2/em_v2 bin 없음: {data_dir}")
    t_s, raw = _read_due_v2(due[0])
    t_e, xyz = _read_em_v2(em[0])
    z = xyz[:, 2]
    z_press = z.max() - 0.25 * (z.max() - z.min())         # 압입=최대 z 근처(상위 25%)만
    # 정지 세그먼트(위치 반올림 불변)
    key = np.round(xyz[:, :3], 1)
    change = np.any(np.diff(key, axis=0) != 0, axis=1)
    starts = np.concatenate([[0], np.where(change)[0] + 1])
    ends = np.concatenate([np.where(change)[0], [len(xyz) - 1]])
    # baseline: 최초 비압입 정지 구간(≥1s)
    base = None
    for s, e in zip(starts, ends):
        if z[s] < z_press and t_e[e] - t_e[s] >= 1.0:
            m = (t_s >= t_e[s] + 0.2) & (t_s <= t_e[e] - 0.2)
            if m.sum() >= 50:
                base = raw[m].mean(0)
                break
    if base is None:
        raise ValueError("baseline(비압입 대기 ≥1s) 구간 없음")
    # 압입 hold 수집 → (x,y)별 pct 프레임 합침
    groups: dict[tuple[float, float], list[np.ndarray]] = {}
    order: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        dur = t_e[e] - t_e[s]
        if z[s] < z_press or dur < hold_min_s:
            continue
        trim = dur * trim_frac
        m = (t_s >= t_e[s] + trim) & (t_s <= t_e[e] - trim)
        if m.sum() < 12:
            continue
        pct = ((raw[m] / base[None, :] - 1.0) * 100.0).astype(np.float32)
        pos = (round(xyz[s, 0] / 1000.0, 4), round(xyz[s, 1] / 1000.0, 4))   # μm→mm
        if pos not in groups:
            groups[pos] = []
            order.append(pos)
        groups[pos].append(pct)
    if not groups:
        raise ValueError("압입 hold 구간 없음(z 왕복 확인)")
    out = [(np.concatenate(groups[p]), np.asarray(p, np.float32)) for p in order]
    n_press = sum(len(v) for v in groups.values())
    print(f"[v2] {len(out)} 위치, 압입 {n_press}회, baseline {base.mean():.0f} "
          f"(z 대기 {z.min()/1000:.1f} / 압입 {z.max()/1000:.1f} mm)")
    return out


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
    if args.v2_dir:
        return _load_v2_session(Path(args.v2_dir))
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
    # 스윕 방향 = GT 분산 주성분(PCA) — σ(반복 노이즈) 측정 축
    c = gt - gt.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
    rows = []
    for sensor, pos in positions:
        pr = _predict_positions(engine, sensor, subpixel=True)
        if len(pr) == 0:
            continue
        mu = pr.mean(0)
        sig_axis = float(np.std(pr @ axis))                           # 스윕축 방향 반복 노이즈
        rows.append({"gt": pos, "mu": mu, "sig": sig_axis,
                     "gt_proj": float(pos @ axis), "pred_proj": float(mu @ axis), "n": len(pr)})
    # ★2D 유사변환 정합(스케일드 프로크루스테스, 반사 허용): pred ≈ s·R·gt + t
    #   모터축↔센서축은 원점·회전·반전이 모두 다를 수 있음(마운팅). 이를 적합해
    #   소거한 뒤의 잔차·스케일이 진짜 추종성/정확도.
    G = np.array([r["gt"] for r in rows]); P = np.array([r["mu"] for r in rows])
    Gc, Pc = G - G.mean(0), P - P.mean(0)
    U, S, Vt = np.linalg.svd(Gc.T @ Pc)
    R = (U @ Vt).T                                                    # 회전(+반사 허용)
    denom = float((Gc ** 2).sum())
    scale = float(S.sum() / max(denom, 1e-12))                        # 예측/모터 스케일(1=정확)
    fit = (scale * (R @ Gc.T)).T                                      # 정합된 GT
    resid2d = Pc - fit
    resid_rms = float(np.sqrt((resid2d ** 2).sum(1).mean()))
    rel_med = float(np.median(np.sqrt((resid2d ** 2).sum(1))))
    ss_tot = float((Pc ** 2).sum())
    r2 = 1.0 - float((resid2d ** 2).sum()) / max(ss_tot, 1e-12)
    rot_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    reflected = bool(np.linalg.det(R) < 0)
    sig_med = float(np.median([r["sig"] for r in rows]))
    res_2s = 2.0 * sig_med / max(abs(scale), 1e-6)
    # 그림용 1D 투영(정합 후)
    gp = (scale * (R @ Gc.T)).T @ axis
    pp = Pc @ axis
    for r, res in zip(rows, np.sqrt((resid2d ** 2).sum(1))):
        r["rel"] = float(res)
    return {"rows": rows, "axis": axis, "scale": scale, "rot_deg": rot_deg,
            "reflected": reflected, "r2": float(r2), "sig_med": sig_med,
            "rel_med": rel_med, "resid_rms": resid_rms,
            "resolution_2sigma": float(res_2s), "gp": np.asarray(gp), "pp": np.asarray(pp)}


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
    a0.set_xlabel("aligned motor displacement (mm)"); a0.set_ylabel("predicted displacement (mm)")
    a0.set_title(f"D{diameter:g}  scale={res['scale']:.3f}  R2={res['r2']:.4f}"
                 f"  rot={res['rot_deg']:.1f}deg" + ("  reflected" if res["reflected"] else ""))
    a0.legend(fontsize=8)
    a1.plot(gp, pp - gp, "o-", ms=3, color="#c33")
    a1.axhline(0, color="k", lw=0.6)
    a1.set_xlabel("aligned motor displacement (mm)"); a1.set_ylabel("residual (mm)")
    a1.set_title(f"sigma_med={res['sig_med']:.3f}mm  resolution(2σ)={res['resolution_2sigma']:.3f}mm")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="미세 스텝 위치 분해능 분석(D10)")
    ap.add_argument("--data-dir", default=None, help="위치당 npz 디렉토리(형식 A)")
    ap.add_argument("--data", default=None, help="전 프레임 단일 npz(형식 B)")
    ap.add_argument("--v2-dir", default=None,
                    help="V2 취득 세션 디렉토리(due_v2_*.bin + em_v2_*.bin) — 압입 자동 추출")
    ap.add_argument("--run-dir", default=str(_ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"))
    ap.add_argument("--diameter", type=float, default=10.0)
    ap.add_argument("--input", choices=["pct", "raw"], default="pct", help="sensor가 pct인지 raw인지")
    ap.add_argument("--group-round", type=float, default=0.05, help="[B] 위치 묶기 반올림(mm)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(_ROOT / "history/fig_data/experiments_archive/reeval/loc_resolution_d10.png"))
    args = ap.parse_args()
    if not args.data_dir and not args.data and not args.v2_dir:
        ap.error("--data-dir / --data / --v2-dir 중 하나 필요")

    engine = SATSInferenceEngine(args.run_dir, device=args.device, indenter_diameter_mm=args.diameter)
    positions = _load_positions(args)
    print(f"[분석] {len(positions)} 위치, window={engine.window_size}, D{args.diameter:g}")
    res = analyze(positions, engine)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make_figure(res, args.out, args.diameter)

    flip = "  (반사=축 반전 포함)" if res["reflected"] else ""
    print("\n" + "=" * 60)
    print(f"  위치 분해능 — 2D 정합(회전·반전 소거) 기준 (D{args.diameter:g}, {len(res['rows'])} 위치)")
    print("=" * 60)
    print(f"  추종성   scale = {res['scale']:.3f} (1=정확)   R2 = {res['r2']:.4f}")
    print(f"  축 정렬  rot = {res['rot_deg']:+.1f}°{flip}")
    print(f"  반복노이즈 sigma(축) median = {res['sig_med']:.3f} mm")
    print(f"  상대정확도 잔차 median = {res['rel_med']:.3f} mm  (RMS {res['resid_rms']:.3f})")
    print(f"  ★분해능  2σ/scale = {res['resolution_2sigma']:.3f} mm  "
          f"(이보다 가까운 두 위치는 노이즈에 묻힘)")
    print("=" * 60)
    print(f"  그림: {args.out}")
    print("  → SPEC.md §3 x/y resolution 을 이 수치로 갱신.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
