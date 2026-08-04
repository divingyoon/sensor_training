#!/usr/bin/env python3
"""v6 실시간 추론 데모 (학회 데모세션) — 터미널 우선.

모드:
  contacts : x,y,z,fz 실시간 출력 (단일=--contacts 1, 다중=--contacts 3). [산출물 1]
  theta    : 밴딩각 theta 출력 (estimator).                              [산출물 3]
  bending  : SATS+밴딩 통합 상태머신(플랫→장착→밴딩→복원→SATS).           [산출물 4]

리더: --mock(하드웨어 없이) 또는 --port(시리얼). z 는 z_calibration LUT(A1) 근사.
시각화(산출물 2)는 run_realtime 의 2d/3d 재사용 또는 --viz(후속).

예:
  .venv/bin/python -m sats.inference.run_demo --mode contacts --contacts 3 --diameter 10 --mock
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import collections

import numpy as np

from sats.inference.demo_contacts import FrameLatch, extract_contacts, format_contacts
from sats.inference.inference_engine import SATSInferenceEngine
from sats.inference.z_calibration import ZCalibration

_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v6 실시간 추론 데모", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["contacts", "theta", "bending"], default="contacts")
    p.add_argument("--run-dir", default=str(_ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"))
    p.add_argument("--diameter", type=float, default=10.0, help="인덴터 지름(mm) — size 조건 + z 보정")
    p.add_argument("--contacts", type=int, default=3, help="최대 접촉 수(1=단일, 3=다중)")
    p.add_argument("--z-calib", default=None, help="z 보정 json (기본=z_calibration_v6.json)")
    p.add_argument("--min-distance-mm", type=float, default=3.0)
    p.add_argument("--rel-threshold", type=float, default=0.3)
    p.add_argument("--report-interval", type=float, default=2.0, help="최적 프레임 요약 주기(초)")
    p.add_argument("--viz", choices=["none", "2d", "3d", "both"], default="none", help="[산출물2] heatmap 시각화")
    p.add_argument("--viz-fps", type=float, default=10.0)
    p.add_argument("--infer-max-fps", type=float, default=20.0)
    p.add_argument("--device", default="auto")
    # 리더
    p.add_argument("--mock", action="store_true")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=2000000)
    p.add_argument("--baseline-seconds", type=float, default=5.0)
    p.add_argument("--startup-delay", type=float, default=2.0)
    p.add_argument("--protocol", default="v2")
    return p


def _start_reader(args, window_size: int):
    """mock 또는 serial 리더 시작(+baseline 대기). (reader, ok)."""
    if args.mock:
        from sats.inference.mock_reader import MockSensorReader
        reader = MockSensorReader(window_size=window_size)
        reader.start()
        print("[reader] mock 시작 (baseline 즉시)\n")
        return reader
    from sats.inference.serial_reader import SensorSerialReader
    reader = SensorSerialReader(port=args.port, baudrate=args.baudrate, window_size=window_size,
                                baseline_seconds=args.baseline_seconds, startup_delay=args.startup_delay,
                                protocol=args.protocol)
    reader.start()
    print(f"[reader] serial {args.port}@{args.baudrate}. 센서에서 손 떼고 baseline 대기...")
    t0 = time.time()
    while not reader.baseline_ready:
        if getattr(reader, "error_message", None):
            print(f"\n[오류] {reader.error_message}"); reader.stop(); sys.exit(1)
        print(f"\r  [{time.time()-t0:5.1f}s] baseline {reader.baseline_progress*100:.0f}%   ", end="", flush=True)
        time.sleep(0.3)
    print("\n[reader] baseline 완료\n")
    return reader


def _get_window(reader, last_seq: int):
    if hasattr(reader, "get_latest_window_with_seq"):
        return reader.get_latest_window_with_seq()
    return reader.get_latest_window(), last_seq + 1


def _make_viz(args, engine):
    """--viz 에 따라 2D/3D 데모 뷰 생성(없으면 []). [산출물 2]"""
    if args.viz == "none":
        return []
    from sats.inference.demo_viz import DemoViz2D, DemoViz3D
    vs = []
    if args.viz in ("2d", "both"):
        vs.append(DemoViz2D(engine.grid_min_mm, engine.grid_max_mm))
    if args.viz in ("3d", "both"):
        vs.append(DemoViz3D(engine.grid_min_mm, engine.grid_max_mm, engine.grid_size))
    return vs


def run_contacts(args, engine, z_calib, reader) -> None:
    """산출물 1: x,y,z,fz 단일/다중 실시간 터미널 + 최적(최대 fz) 프레임 latch."""
    kind = "단일" if args.contacts == 1 else f"다중(최대 {args.contacts})"
    print(f"[contacts] {kind}  d={args.diameter:g}mm  (Ctrl+C 종료)\n")
    latch = FrameLatch()
    viz = _make_viz(args, engine)
    viz_interval = 0.0 if args.viz_fps <= 0 else 1.0 / args.viz_fps
    last_seq, frame, last_infer, last_report, last_viz = 0, 0, 0.0, time.time(), 0.0
    infer_interval = 0.0 if args.infer_max_fps <= 0 else 1.0 / args.infer_max_fps
    try:
        while True:
            now = time.time()
            if now - last_infer < infer_interval:
                time.sleep(0.001); continue
            win, seq = _get_window(reader, last_seq)
            if win is None or seq == last_seq:
                time.sleep(0.002); continue
            last_seq, last_infer = seq, now
            frame += 1
            pmap = engine.predict(win)
            contacts = extract_contacts(pmap, grid_min_mm=engine.grid_min_mm, grid_step_mm=engine.grid_step_mm,
                                        taxel_area=engine.taxel_area, diameter_mm=args.diameter,
                                        max_contacts=args.contacts, min_distance_mm=args.min_distance_mm,
                                        rel_threshold=args.rel_threshold, z_calib=z_calib)
            if contacts:
                print(format_contacts(frame, contacts))
                latch.update(frame, pmap, contacts)
            if viz and now - last_viz >= viz_interval:
                for v in viz:
                    v.update(pmap, contacts, best_map=latch.pred_map, best_contacts=latch.contacts)
                last_viz = now
            if now - last_report >= args.report_interval and latch.contacts:
                print(f"  ★ 최적 프레임 {latch.frame_idx} (총 fz={latch.best_total:.2f}N):")
                print(format_contacts(latch.frame_idx, latch.contacts))
                last_report = now
    except KeyboardInterrupt:
        print("\n\n=== 종료 ===")
        if latch.contacts:
            print(f"최적(최대 fz) 프레임 {latch.frame_idx}:")
            print(format_contacts(latch.frame_idx, latch.contacts))


def run_theta(args, engine, bi, reader) -> None:
    """산출물 3: 밴딩각 theta 실시간 터미널 (estimator). raw=base×(1+pct/100) 복원."""
    from sats.inference.bending_infer import pct_to_raw
    base = reader.baseline
    if base is None:
        print("[theta] flat baseline 없음(mock 등) → theta 추정 불가. 실센서 필요."); return
    base = np.asarray(base, float)
    print("[theta] 밴딩각 실시간 출력 (Ctrl+C 종료)\n")
    hist: collections.deque = collections.deque(maxlen=15)
    last_seq, last_infer = 0, 0.0
    interval = 0.0 if args.infer_max_fps <= 0 else 1.0 / args.infer_max_fps
    try:
        while True:
            now = time.time()
            if now - last_infer < interval:
                time.sleep(0.001); continue
            win, seq = _get_window(reader, last_seq)
            if win is None or seq == last_seq:
                time.sleep(0.002); continue
            last_seq, last_infer = seq, now
            theta = bi.theta_from_raw(pct_to_raw(win, base))
            hist.append(theta)
            print(f"\r  theta = {theta:+7.1f} deg   (smoothed {np.median(hist):+7.1f})     ", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n\n=== 종료 ===  최종 smoothed theta = {np.median(hist):+.1f} deg" if hist else "\n종료")


def run_bending(args, engine, z_calib, bi, reader) -> None:
    """산출물 4: SATS+밴딩 통합 상태머신. 플랫 baseline→(장착·밴딩)→theta→복원→SATS."""
    from sats.inference.bending_infer import pct_to_raw
    base = reader.baseline
    if base is None:
        print("[bending] flat baseline 없음(mock 등) → 통합 모드 불가. 실센서 필요."); return
    base = np.asarray(base, float)
    print("\n[bending] === SATS+밴딩 통합 모드 (버클 방식) ===")
    print("  1) 지금 플랫 baseline 캡처됨.  2) 지그에 장착·밴딩 후 Enter를 누르세요.")
    input("  >> 밴딩 완료했으면 Enter: ")
    # REARM: 밴딩 무접촉 상태서 theta 고정 추정(최근 몇 프레임 중앙값)
    thetas = []
    for _ in range(15):
        win, _ = _get_window(reader, -1)
        if win is not None:
            thetas.append(bi.theta_from_raw(pct_to_raw(win, base)))
        time.sleep(0.03)
    theta = float(np.median(thetas)) if thetas else 0.0
    print(f"  ★ 밴딩 곡률 theta = {theta:+.1f} deg  (복원 조건 고정)\n")
    print("  접촉 press 시작 (Ctrl+C 종료)\n")
    latch = FrameLatch()
    viz = _make_viz(args, engine)
    viz_interval = 0.0 if args.viz_fps <= 0 else 1.0 / args.viz_fps
    last_seq, frame, last_infer, last_report, last_viz = 0, 0, 0.0, time.time(), 0.0
    interval = 0.0 if args.infer_max_fps <= 0 else 1.0 / args.infer_max_fps
    try:
        while True:
            now = time.time()
            if now - last_infer < interval:
                time.sleep(0.001); continue
            win, seq = _get_window(reader, last_seq)
            if win is None or seq == last_seq:
                time.sleep(0.002); continue
            last_seq, last_infer = seq, now
            frame += 1
            restored = bi.restore(win, theta)            # 밴딩→flat 등가 복원
            pmap = engine.predict(restored)               # 동결 SATS
            contacts = extract_contacts(pmap, grid_min_mm=engine.grid_min_mm, grid_step_mm=engine.grid_step_mm,
                                        taxel_area=engine.taxel_area, diameter_mm=args.diameter,
                                        max_contacts=args.contacts, min_distance_mm=args.min_distance_mm,
                                        rel_threshold=args.rel_threshold, z_calib=z_calib)
            if contacts:
                print(format_contacts(frame, contacts, theta_deg=theta))
                latch.update(frame, pmap, contacts)
            if viz and now - last_viz >= viz_interval:
                for v in viz:
                    v.update(pmap, contacts, theta_deg=theta, best_map=latch.pred_map, best_contacts=latch.contacts)
                last_viz = now
            if now - last_report >= args.report_interval and latch.contacts:
                print(f"  ★ 최적 프레임 {latch.frame_idx} (총 fz={latch.best_total:.2f}N, theta={theta:+.1f}):")
                print(format_contacts(latch.frame_idx, latch.contacts, theta_deg=theta))
                last_report = now
    except KeyboardInterrupt:
        print("\n\n=== 종료 ===")
        if latch.contacts:
            print(format_contacts(latch.frame_idx, latch.contacts, theta_deg=theta))


def main() -> None:
    args = _build_parser().parse_args()
    print(f"[1/2] 엔진 로드: {args.run_dir}")
    engine = SATSInferenceEngine(args.run_dir, device=args.device, indenter_diameter_mm=args.diameter)
    z_path = args.z_calib or (Path(__file__).resolve().parent / "z_calibration_v6.json")
    z_calib = ZCalibration.load(z_path) if Path(z_path).exists() else None
    if z_calib is None:
        print(f"  [경고] z 보정 없음({z_path}) → z=n/a. z_calibration 생성 권장.")
    # theta/bending 모드는 estimator(+restorer) 로드
    bi = None
    if args.mode in ("theta", "bending"):
        from sats.bending.config import BendingConfig
        from sats.inference.bending_infer import BendingInference, load_restorer
        est_ckpt = _ROOT / "sats/bending/runs/estimator_v6"
        cfg = BendingConfig()
        restorer = None
        if args.mode == "bending":
            print("[bending] restorer 학습(v6 buckling, ~1분)...")
            restorer = load_restorer(args.run_dir, _ROOT / "learning_data/bending/v6",
                                     device=engine.device, cfg=cfg)
        bi = BendingInference(est_ckpt, device=engine.device, restorer=restorer, cfg=cfg)

    reader = _start_reader(args, engine.window_size)
    try:
        if args.mode == "contacts":
            run_contacts(args, engine, z_calib, reader)
        elif args.mode == "theta":
            run_theta(args, engine, bi, reader)
        elif args.mode == "bending":
            run_bending(args, engine, z_calib, bi, reader)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
