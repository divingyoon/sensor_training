"""밴딩 raw(due+ethermotion bin) 센서 신호 vs 변형량(δ) 비교 분석.

구·신 센서/취득 비교용. 각 세션의 due_raw_burst.bin(16ch) + ethermotion(Y구동)을
파싱해, flat baseline 대비 상대변화 |Δp|(%)를 변형량 δ(=Y−Y_min) 구간별로 집계.
핵심 질문: 신 데이터가 (a) 더 큰 신호, (b) 저-δ에서 신호, (c) 넓은 관측범위를 갖는가.

출력: JSON(세션·집계). 텍스트/그래프 없음(양 머신 동일 실행 후 수치 대조).
실행: .venv/bin/python scripts/compare_bending_sensor.py --dir skin_ws/raw_data/bending/v6 [--label local]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo 루트 → sats 임포트
from sats.preprocessing.bin_merge import load_due_bin, load_ethermotion_bin

FLAT_TOL_MM = 0.5


def _session_stats(test_dir: Path) -> dict | None:
    due_g = glob.glob(str(test_dir / "due_raw_burst_*.bin"))
    eth_g = glob.glob(str(test_dir / "ethermotion_encoder_*.bin"))
    if not due_g or not eth_g:
        return None
    due = load_due_bin(due_g[0])
    eth = load_ethermotion_bin(eth_g[0])
    s = np.asarray(due.sensors, np.float64)             # [N,16]
    ts = np.asarray(due.time_s, np.float64)
    ey, et = np.asarray(eth.y_mm, np.float64), np.asarray(eth.time_s, np.float64)
    # 시간 최근접으로 Y 정렬
    idx = np.searchsorted(et, ts).clip(0, len(et) - 1)
    y = ey[idx]
    delta = y - y.min()                                # 변형량(δ, flat=min Y)
    flat = delta < FLAT_TOL_MM
    if flat.sum() < 5:
        flat = delta < np.percentile(delta, 5)
    base = s[flat].mean(0)
    safe = np.where(np.abs(base) < 1e-9, 1e-9, base)
    dp = np.abs((s - base) / safe * 100.0).mean(1)     # 프레임별 16ch 평균 |Δp|%
    # δ 구간별 신호
    bins = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 35]
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (delta >= lo) & (delta < hi)
        if m.sum() > 10:
            curve.append({"delta_lo": lo, "n": int(m.sum()),
                          "signal_pct": round(float(dp[m].mean()), 3)})
    return {
        "session": test_dir.name,
        "n_frames": int(len(s)),
        "delta_max_mm": round(float(delta.max()), 1),
        "baseline_mean": round(float(base.mean()), 0),
        "flat_noise_pct": round(float(dp[flat].mean()), 3),
        "signal_max_pct": round(float(np.percentile(dp, 99)), 3),
        "curve": curve,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="밴딩 버전 폴더(하위에 test*/ due+ether bin)")
    p.add_argument("--label", default="")
    args = p.parse_args()
    root = Path(args.dir)
    tests = sorted([d for d in root.rglob("*") if d.is_dir() and glob.glob(str(d / "due_raw_burst_*.bin"))])
    sessions = [st for d in tests if (st := _session_stats(d))]
    # 집계: δ구간별 신호 평균(세션 평균)
    agg: dict[int, list] = {}
    for s in sessions:
        for c in s["curve"]:
            agg.setdefault(c["delta_lo"], []).append(c["signal_pct"])
    agg_curve = [{"delta_lo": k, "signal_pct": round(float(np.mean(v)), 3), "n_sess": len(v)}
                 for k, v in sorted(agg.items())]
    out = {
        "label": args.label, "dir": str(root), "n_sessions": len(sessions),
        "baseline_mean_all": round(float(np.mean([s["baseline_mean"] for s in sessions])), 0) if sessions else None,
        "flat_noise_pct_all": round(float(np.mean([s["flat_noise_pct"] for s in sessions])), 3) if sessions else None,
        "signal_max_pct_all": round(float(np.mean([s["signal_max_pct"] for s in sessions])), 3) if sessions else None,
        "delta_max_mm_all": round(float(np.mean([s["delta_max_mm"] for s in sessions])), 1) if sessions else None,
        "signal_vs_delta": agg_curve,
        "sessions": sessions,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
