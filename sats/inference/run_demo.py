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

from sats.inference.demo_contacts import (
    FrameLatch, LiveDisplay, extract_contacts, format_contacts, format_contacts_block,
)
from sats.inference.inference_engine import SATSInferenceEngine
from sats.inference.z_calibration import ZCalibration

_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v6 실시간 추론 데모", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["contacts", "theta", "bending"], default="contacts")
    p.add_argument("--run-dir", default=str(_ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"))
    p.add_argument("--diameter", type=float, default=10.0, help="인덴터 지름(mm) — size 조건 + z 보정")
    p.add_argument("--contacts", type=int, default=None,
                   help="최대 접촉 수(1=단일, 3=다중). 미지정 시 bending=1(단일)·contacts=3")
    p.add_argument("--z-calib", default=None, help="z 보정 json (기본=z_calibration_v6.json)")
    p.add_argument("--min-distance-mm", type=float, default=None,
                   help="접촉 간 최소 간격(mm). 미지정 시 지름값으로 자동 — "
                        "같은 지름 두 접촉은 중심간 지름보다 가까울 수 없으므로 단일접촉 peak-split 방지")
    p.add_argument("--rel-threshold", type=float, default=0.3)
    p.add_argument("--min-fz", type=float, default=0.2, help="무접촉 게이트: 접촉별 fz(N) 하한(약한 스퓨리어스 제거)")
    p.add_argument("--report-interval", type=float, default=2.0, help="최적 프레임 요약 주기(초)")
    p.add_argument("--viz", choices=["none", "2d", "3d", "both"], default="none", help="[산출물2] heatmap 시각화")
    p.add_argument("--show-units", action="store_true", help="실시간 16-taxel 센싱유닛 heatmap(SATS 입력 원본) 창 추가")
    # theta 히스테리시스 필터(점탄성 잔차 자동 영점) — release 감지 기반(유효범위 20~150° 보존)
    p.add_argument("--auto-rezero", action=argparse.BooleanOptionalAction, default=True,
                   help="theta: 더 큰 밴딩서 풀려 정지한 잔차만 서서히 0으로(held 밴딩은 보존)")
    p.add_argument("--rezero-release-deg", type=float, default=12.0,
                   help="잔차 판정: 최근 peak보다 이만큼↓ 낮게 정지 시 잔차로 간주(작을수록 민감)")
    p.add_argument("--rezero-tau", type=float, default=2.5, help="자동 영점 시간상수(초). 클수록 느리게 복원")
    p.add_argument("--viz-fps", type=float, default=10.0)
    p.add_argument("--infer-max-fps", type=float, default=20.0)
    p.add_argument("--device", default="auto")
    # 리더
    p.add_argument("--list-ports", action="store_true", help="시리얼 포트 목록 출력 후 종료(기본 셋업)")
    p.add_argument("--probe", action="store_true", help="포트에서 raw 바이트 몇 개 읽어 연결 확인 후 종료")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=250000)   # vensor2.ino Serial.begin(250000)
    p.add_argument("--baseline-seconds", type=float, default=5.0)
    p.add_argument("--startup-delay", type=float, default=2.0)
    p.add_argument("--protocol", choices=["auto", "binary", "csv"], default="auto",
                   help="vensor2.ino=바이너리(0xAA..0x55). auto=자동감지(→binary 폴백)")
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


def list_serial_ports() -> None:
    """연결된 시리얼 포트 목록 출력 (기본 셋업 — 센서 포트 찾기)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        print("pyserial 미설치 (.venv 확인)"); return
    ports = list(list_ports.comports())
    if not ports:
        print("시리얼 포트 없음. 센서 USB 연결·전원 확인, 권한(dialout) 확인."); return
    print(f"발견된 시리얼 포트 {len(ports)}개:")
    for p in ports:
        print(f"  {p.device:16s}  {p.description}  [{p.hwid}]")
    print("\n→ 센서 포트를 --port 로 지정 (예: --port " + ports[0].device + ")")


def probe_port(port: str, baudrate: int, n: int = 64) -> None:
    """포트에서 raw 바이트 읽어 연결·데이터 흐름 확인(기본 셋업)."""
    try:
        import serial
    except ImportError:
        print("pyserial 미설치"); return
    try:
        s = serial.Serial(port, baudrate, timeout=1.0)
    except Exception as e:
        print(f"[probe] {port} 열기 실패: {e}\n  권한? sudo usermod -aG dialout $USER (재로그인) 또는 sudo chmod 666 {port}")
        return
    data = s.read(n); s.close()
    print(f"[probe] {port}@{baudrate}: {len(data)} bytes 수신" + ("  ✅ 데이터 흐름 OK" if data else "  ⚠ 0 bytes(전원/보드레이트/포트 확인)"))
    if data:
        print("  head:", data[:24].hex(" "))


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


def _make_units_viz(args):
    """--show-units 시 실시간 16-taxel 센싱유닛 뷰 생성(없으면 None)."""
    if not getattr(args, "show_units", False):
        return None
    from sats.inference.demo_viz import SensingUnitViz
    return SensingUnitViz()


def run_contacts(args, engine, z_calib, reader) -> None:
    """산출물 1: x,y,z,fz 단일/다중 실시간 터미널 + 최적(최대 fz) 프레임 latch."""
    kind = "단일" if args.contacts == 1 else f"다중(최대 {args.contacts})"
    print(f"[contacts] {kind}  d={args.diameter:g}mm  (Ctrl+C 종료)\n")
    latch = FrameLatch()
    viz = _make_viz(args, engine)
    units_viz = _make_units_viz(args)
    disp = LiveDisplay()
    viz_interval = 0.0 if args.viz_fps <= 0 else 1.0 / args.viz_fps
    last_seq, frame, last_infer, last_viz, t_fps, fps = 0, 0, 0.0, 0.0, time.time(), 0.0
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
            if frame % 10 == 0:
                fps = 10.0 / max(now - t_fps, 1e-6); t_fps = now
            pmap = engine.predict(win)
            contacts = extract_contacts(pmap, grid_min_mm=engine.grid_min_mm, grid_step_mm=engine.grid_step_mm,
                                        taxel_area=engine.taxel_area, diameter_mm=args.diameter,
                                        max_contacts=args.contacts, min_distance_mm=args.min_distance_mm,
                                        rel_threshold=args.rel_threshold, min_fz=args.min_fz, z_calib=z_calib)
            if contacts:
                latch.update(frame, pmap, contacts)
            disp.render(format_contacts_block(contacts, frame=frame, fps=fps, mode="contacts"))
            if (viz or units_viz) and now - last_viz >= viz_interval:
                for v in viz:
                    v.update(pmap, contacts, best_map=latch.pred_map, best_contacts=latch.contacts)
                if units_viz:
                    units_viz.update(win)                 # 16-taxel 센싱유닛(입력 원본)
                last_viz = now
    except KeyboardInterrupt:
        print("\n\n=== 종료 ===")
        if latch.contacts:
            print(f"최적(최대 fz) 프레임 {latch.frame_idx} (총 fz={latch.best_total:.2f}N):")
            print(format_contacts(latch.frame_idx, latch.contacts))


def _enter_pressed() -> bool:
    """비차단 stdin 확인 — Enter 입력 있으면 소비하고 True(재영점 트리거)."""
    import select
    if not sys.stdin.isatty():
        return False
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        sys.stdin.readline()
        return True
    return False


def _capture_theta0(bi, reader, base, seconds: float) -> tuple[float, int]:
    """flat 구간 theta 중앙값 = 영점. (theta0, last_seq)."""
    t0s, last_seq = [], 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        win, seq = _get_window(reader, last_seq)
        if win is not None and seq != last_seq:
            last_seq = seq
            t0s.append(bi.theta_from_pct(win, demo_baseline=base))
        time.sleep(0.02)
    return (float(np.median(t0s)) if t0s else 0.0), last_seq


def run_theta(args, engine, bi, reader) -> None:
    """산출물 3: 밴딩각 theta 실시간 터미널 (estimator).

    ★재앵커: 학습 참조 baseline(ref)에 pct를 얹어 기압 무관하게 추정.
    ★영점보정: flat 구간 theta를 0으로 잡고 상대 밴딩각(Δθ) 표시.
    ★재영점: baseline 드리프트(손 타기·발열)로 flat인데 값이 뜨면 Enter로 현재 상태를 0으로 재설정.
    """
    if reader.baseline is None:
        print("[theta] flat baseline 없음(mock 등) → theta 추정 불가. 실센서 필요."); return
    if bi.ref_baseline is None:
        print("[theta] ⚠ ref_baseline.npy 없음 → 데모 baseline 폴백(기압 민감). 재생성 권장.")
    base = reader.baseline
    print("[theta] flat 영점 캡처 중 — 센서 평평·무접촉 유지 (2초)...")
    theta0, last_seq = _capture_theta0(bi, reader, base, 2.0)
    print(f"[theta] flat 영점 theta0 = {theta0:+.1f} deg  → 이후 Δθ(상대) 표시  (유효 밴딩 ~20-150°)")
    ar = "ON" if args.auto_rezero else "OFF"
    print(f"  ★ Enter=수동 재영점.  자동 영점(release 기반)={ar} "
          f"(최근 peak−{args.rezero_release_deg:g}° 아래로 정지=잔차 → τ={args.rezero_tau:g}s로 0; held 밴딩 보존). (Ctrl+C 종료)\n")
    hist: collections.deque = collections.deque(maxlen=20)    # ~1s @20fps (평활·움직임)
    peak_hist: collections.deque = collections.deque(maxlen=80)  # ~4s (최근 peak)
    last_infer, last_frame = 0.0, time.time()
    interval = 0.0 if args.infer_max_fps <= 0 else 1.0 / args.infer_max_fps
    try:
        while True:
            now = time.time()
            if _enter_pressed():                      # 수동 재영점(현재 flat 상태를 0으로)
                theta0, last_seq = _capture_theta0(bi, reader, base, 0.6)
                hist.clear(); peak_hist.clear(); last_frame = time.time()
                print(f"\n[theta] ★ 재영점 완료 theta0 = {theta0:+.1f} deg\n")
                continue
            if now - last_infer < interval:
                time.sleep(0.001); continue
            win, seq = _get_window(reader, last_seq)
            if win is None or seq == last_seq:
                time.sleep(0.002); continue
            last_seq, last_infer = seq, now
            theta_abs = bi.theta_from_pct(win, demo_baseline=base)
            theta_rel = theta_abs - theta0
            hist.append(theta_rel); peak_hist.append(theta_rel)
            smoothed = float(np.median(list(hist)[-7:]))
            # 움직임 감지(median, 노이즈 강건): 활성 밴딩 중엔 자동영점 미적용.
            moving = True
            if len(hist) >= 15:
                moving = abs(float(np.median(list(hist)[-5:])) - float(np.median(list(hist)[:5]))) > 4.0
            # ★release 감지: 최근 peak(~4s)보다 release_deg 이상 낮게 정지 = 히스테리시스 잔차.
            #   held 밴딩은 자기 peak≈현재라 미적용 → 유효 20-150° 보존.
            recent_max = max(peak_hist)
            released = (recent_max - smoothed) > args.rezero_release_deg
            adapting = args.auto_rezero and (not moving) and released
            if adapting:
                alpha = min(1.0, (now - last_frame) / max(args.rezero_tau, 1e-3))
                theta0 += alpha * (theta_abs - theta0)    # baseline 견인 → 잔차 Δθ→0
            last_frame = now
            tag = "~0(residual)" if adapting else ("bending    " if moving else "hold        ")
            print(f"\r  Δtheta = {smoothed:+7.1f} deg  {tag} [Enter=재영점] ", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n\n=== 종료 ===  최종 Δtheta = {float(np.median(list(hist)[-7:])):+.1f} deg" if hist else "\n종료")


def run_bending(args, engine, z_calib, bi, reader) -> None:
    """산출물 4: SATS+밴딩 통합 상태머신. 플랫 baseline→(장착·밴딩)→theta→복원→SATS."""
    base = reader.baseline
    if base is None:
        print("[bending] flat baseline 없음(mock 등) → 통합 모드 불가. 실센서 필요."); return
    base = np.asarray(base, float)
    print("\n[bending] === SATS+밴딩 통합 모드 (버클 방식) ===")
    print("  1) 지금 플랫 baseline 캡처됨.  2) 지그에 장착·밴딩 후 Enter를 누르세요.")
    input("  >> 밴딩 완료했으면 Enter: ")
    # REARM: 밴딩 무접촉 상태서 theta 고정 추정(재앵커, 최근 몇 프레임 중앙값)
    thetas = []
    for _ in range(15):
        win, _ = _get_window(reader, -1)
        if win is not None:
            thetas.append(bi.theta_from_pct(win, demo_baseline=base))
        time.sleep(0.03)
    theta = float(np.median(thetas)) if thetas else 0.0
    print(f"  ★ 밴딩 곡률 theta = {theta:+.1f} deg  (복원 조건 고정)\n")
    print("  접촉 press 시작 (Ctrl+C 종료)\n")
    latch = FrameLatch()
    viz = _make_viz(args, engine)
    units_viz = _make_units_viz(args)
    disp = LiveDisplay()
    viz_interval = 0.0 if args.viz_fps <= 0 else 1.0 / args.viz_fps
    last_seq, frame, last_infer, last_viz, t_fps, fps = 0, 0, 0.0, 0.0, time.time(), 0.0
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
            if frame % 10 == 0:
                fps = 10.0 / max(now - t_fps, 1e-6); t_fps = now
            restored = bi.restore(win, theta)            # 밴딩→flat 등가 복원
            pmap = engine.predict(restored)               # 동결 SATS
            contacts = extract_contacts(pmap, grid_min_mm=engine.grid_min_mm, grid_step_mm=engine.grid_step_mm,
                                        taxel_area=engine.taxel_area, diameter_mm=args.diameter,
                                        max_contacts=args.contacts, min_distance_mm=args.min_distance_mm,
                                        rel_threshold=args.rel_threshold, min_fz=args.min_fz, z_calib=z_calib)
            if contacts:
                latch.update(frame, pmap, contacts)
            disp.render(format_contacts_block(contacts, theta_deg=theta, frame=frame, fps=fps, mode="bending"))
            if (viz or units_viz) and now - last_viz >= viz_interval:
                for v in viz:
                    v.update(pmap, contacts, theta_deg=theta, best_map=latch.pred_map, best_contacts=latch.contacts)
                if units_viz:
                    units_viz.update(win)                 # 밴딩 상태 16-taxel(복원 전 원본)
                last_viz = now
    except KeyboardInterrupt:
        print("\n\n=== 종료 ===")
        if latch.contacts:
            print(f"최적 프레임 {latch.frame_idx} (총 fz={latch.best_total:.2f}N, theta={theta:+.1f}):")
            print(format_contacts(latch.frame_idx, latch.contacts, theta_deg=theta))


def main() -> None:
    args = _build_parser().parse_args()
    # 기본 셋업 유틸(모델 로드 불필요) — 먼저 처리 후 종료
    if args.list_ports:
        list_serial_ports(); return
    if args.probe:
        probe_port(args.port, args.baudrate); return
    # 접촉 수 미지정 시: bending=1(단일, 표준 시나리오), contacts=3(다중)
    if args.contacts is None:
        args.contacts = 1 if args.mode == "bending" else 3
    if args.mode == "bending" and args.contacts != 1:
        print(f"[setup] bending 모드 접촉 수 = {args.contacts} (단일만 필요하면 --contacts 1)")
    # 접촉 최소 간격 미지정 시 지름값으로 — 단일접촉이 인접 peak로 쪼개지는 것 방지
    if args.min_distance_mm is None:
        args.min_distance_mm = args.diameter
        print(f"[setup] min-distance = 지름 {args.diameter:g}mm (단일접촉 peak-split 방지)")
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
        # 신규 v6(y23-33·G1 MAE 1.78°) 우선, 없으면 구 v6 폴백
        est_ckpt = _ROOT / "sats/bending/runs/estimator_v6new/best.pt"
        if not est_ckpt.exists():
            est_ckpt = _ROOT / "sats/bending/runs/estimator_v6"
        bend_dir = _ROOT / "learning_data/bending/v6_new"
        if not bend_dir.exists():
            bend_dir = _ROOT / "learning_data/bending/v6"
        print(f"[bending] estimator={est_ckpt.parent.name if est_ckpt.name=='best.pt' else est_ckpt.name}  data={bend_dir.name}")
        cfg = BendingConfig()
        restorer = None
        if args.mode == "bending":
            print("[bending] restorer 학습(신규 v6 buckling, ~1분)...")
            restorer = load_restorer(args.run_dir, bend_dir, device=engine.device, cfg=cfg)
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
