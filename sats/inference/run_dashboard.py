#!/usr/bin/env python3
"""통합 데모 대시보드 — 3센서(각각 contacts·theta·bending) 1창 동시 시연.

레이아웃(1 figure, GridSpec 2x3):
  [ contacts heatmap ] [ theta gauge ] [ bending→SATS heatmap ]
  [ ------ spec footer (D10) ------- ] [ 16-taxel raw inset   ]

설계:
  - SATS 엔진 1개 공유(predict 상태무관) → GPU 1회 로드로 3센서 처리.
  - 센서별 SerialReader. 포트 'none'이면 해당 패널 OFFLINE(현재 1센서만 연결돼도 동작).
  - 포트 고정 권장: /dev/serial/by-id/*  (ttyACM 번호 뒤바뀜 방지).
  - 무접촉/접촉 판정은 demo_contacts.state_banner 로 3패널 통일.

키:
  b = bending 채널 재장착(밴딩 무접촉 상태서 theta+bent-baseline 캡처, ~1s)
  z = theta 채널 flat 재영점
  q = 종료

예(4090, 1센서를 bending 패널로 테스트):
  .venv/bin/python -m sats.inference.run_dashboard --contacts-port none --bending-port auto
예(3센서 전시):
  .venv/bin/python -m sats.inference.run_dashboard \
    --contacts-port /dev/serial/by-id/A --theta-port /dev/serial/by-id/B \
    --bending-port /dev/serial/by-id/C
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from sats.inference.demo_contacts import extract_contacts, state_banner
from sats.inference.inference_engine import SATSInferenceEngine
from sats.inference.run_demo import auto_detect_port
from sats.inference.z_calibration import ZCalibration

_ROOT = Path(__file__).resolve().parents[2]
_ROLES = ("contacts", "theta", "bending")


def _estimator_ckpt() -> Path:
    """v6_2(clean) 우선 → v6new → v6 폴백."""
    return next((p for p in [
        _ROOT / "sats/bending/runs/estimator_v6_2/best.pt",
        _ROOT / "sats/bending/runs/estimator_v6new/best.pt",
        _ROOT / "sats/bending/runs/estimator_v6",
    ] if p.exists()), _ROOT / "sats/bending/runs/estimator_v6")


class SensorChannel:
    """한 센서(=한 역할)의 리더 + 상태 + 매 틱 표시 payload 산출.

    role: contacts | theta | bending. reader None 이면 OFFLINE.
    """

    def __init__(self, role: str, reader, engine, bi, z_calib, args) -> None:
        self.role = role
        self.reader = reader
        self.engine = engine
        self.bi = bi
        self.z_calib = z_calib
        self.args = args
        self.connected = reader is not None
        self.last_seq = 0
        # theta 상태
        self.theta0 = 0.0
        self.hist: collections.deque = collections.deque(maxlen=max(20, args.theta_smooth))
        # bending 상태
        self.armed = False
        self.theta_fixed = 0.0
        self.bent_ref = np.zeros(16, np.float32)

    # ── 데이터 취득 ──────────────────────────────────────────────────────────
    def _latest(self):
        if self.reader is None:
            return None
        if hasattr(self.reader, "get_latest_window_with_seq"):
            win, seq = self.reader.get_latest_window_with_seq()
        else:
            win, seq = self.reader.get_latest_window(), self.last_seq + 1
        if win is None or seq == self.last_seq:
            return None
        self.last_seq = seq
        return win

    @property
    def baseline(self):
        return None if self.reader is None else self.reader.baseline

    # ── 재영점 / 재장착(키 이벤트) ────────────────────────────────────────────
    def rezero_theta(self) -> None:
        """flat 구간 theta 중앙값을 0으로(theta 채널)."""
        if self.bi is None or self.baseline is None:
            return
        vals = []
        t_end = time.time() + 0.6
        while time.time() < t_end:
            win = self._latest()
            if win is not None:
                vals.append(self.bi.theta_from_pct(win, demo_baseline=self.baseline))
            time.sleep(0.02)
        if vals:
            self.theta0 = float(np.median(vals))
            self.hist.clear()

    def arm_bending(self) -> None:
        """밴딩 무접촉 상태서 theta 고정 + bent-baseline(밴딩 패턴) 캡처(~1s)."""
        if self.bi is None or self.baseline is None:
            return
        base = np.asarray(self.baseline, float)
        thetas, frames = [], []
        for _ in range(30):
            win = self._latest()
            if win is not None:
                thetas.append(self.bi.theta_from_pct(win, demo_baseline=base))
                frames.append(np.asarray(win, np.float32).mean(0))
            time.sleep(0.03)
        self.theta_fixed = float(np.median(thetas)) if thetas else 0.0
        self.bent_ref = (np.median(frames, axis=0).astype(np.float32)
                         if frames else np.zeros(16, np.float32))
        self.armed = True

    # ── 매 틱 payload ─────────────────────────────────────────────────────────
    def poll(self) -> dict:
        """현재 표시 payload. kind ∈ {heatmap, theta}."""
        if not self.connected:
            banner = state_banner(None, connected=False)
            return {"kind": "heatmap" if self.role != "theta" else "theta",
                    "banner": banner, "pred_map": None, "contacts": [],
                    "theta": None, "noise": None, "units": None}
        win = self._latest()
        if self.role == "contacts":
            return self._poll_heatmap(win, theta_deg=None, bent=False)
        if self.role == "bending":
            return self._poll_heatmap(win, theta_deg=self.theta_fixed if self.armed else None,
                                      bent=self.armed)
        return self._poll_theta(win)

    def _poll_heatmap(self, win, *, theta_deg, bent: bool) -> dict:
        a = self.args
        if win is None:
            banner = state_banner([], theta_deg=theta_deg, theta_band_deg=a.theta_deadband)
            return {"kind": "heatmap", "banner": banner, "pred_map": None,
                    "contacts": [], "theta": theta_deg, "units": None}
        frame = (win - self.bent_ref[None, :]).astype(np.float32) if bent else win
        pmap = self.engine.predict(frame)
        contacts = extract_contacts(
            pmap, grid_min_mm=self.engine.grid_min_mm, grid_step_mm=self.engine.grid_step_mm,
            taxel_area=self.engine.taxel_area, diameter_mm=a.diameter,
            max_contacts=a.contacts, min_distance_mm=a.min_distance_mm,
            rel_threshold=a.rel_threshold, min_fz=a.min_fz, z_calib=self.z_calib)
        banner = state_banner(contacts, theta_deg=theta_deg, theta_band_deg=a.theta_deadband)
        return {"kind": "heatmap", "banner": banner, "pred_map": pmap,
                "contacts": contacts, "theta": theta_deg,
                "units": win if self.role == "bending" else None}

    def _poll_theta(self, win) -> dict:
        a = self.args
        if win is None or self.bi is None or self.baseline is None:
            banner = state_banner([], connected=self.connected)
            return {"kind": "theta", "banner": banner, "theta": None, "noise": None}
        theta_abs = self.bi.theta_from_pct(win, demo_baseline=self.baseline)
        self.hist.append(theta_abs - self.theta0)
        sm = max(1, int(a.theta_smooth))
        recent = list(self.hist)[-sm:]
        smoothed = float(np.median(recent))
        noise = float(np.std(recent))
        display = 0.0 if abs(smoothed) < a.theta_deadband else smoothed   # 관측 하한 dead-band
        banner = state_banner([], theta_deg=display, theta_band_deg=a.theta_deadband)
        return {"kind": "theta", "banner": banner, "theta": display, "noise": noise}


def _build_reader(port: str, args, window_size: int):
    """port=='none' → None. 'auto' → 자동탐지. 그 외 → 실포트. (reader or None)."""
    if port == "none":
        return None
    if args.mock:
        from sats.inference.mock_reader import MockSensorReader
        r = MockSensorReader(window_size=window_size)
        r.start()
        return r
    resolved = auto_detect_port(args.baudrate) if port == "auto" else port
    if resolved is None:
        print(f"[dashboard] 포트 자동탐지 실패({port}) → 해당 패널 OFFLINE")
        return None
    from sats.inference.serial_reader import SensorSerialReader
    r = SensorSerialReader(port=resolved, baudrate=args.baudrate, window_size=window_size,
                           baseline_seconds=args.baseline_seconds, startup_delay=args.startup_delay,
                           protocol=args.protocol)
    r.start()
    print(f"[dashboard] {resolved}@{args.baudrate} 시작")
    return r


def _wait_baselines(channels, timeout: float = 30.0) -> None:
    """실센서 리더들의 baseline 준비 대기(손 떼고)."""
    readers = [c.reader for c in channels if c.reader is not None
               and hasattr(c.reader, "baseline_ready")]
    if not readers:
        return
    print("[dashboard] baseline 대기 — 모든 센서 손 떼고 평평 유지...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(getattr(r, "baseline_ready", True) for r in readers):
            print("[dashboard] baseline 완료\n")
            return
        for r in readers:
            if getattr(r, "error_message", None):
                print(f"[dashboard] 리더 오류: {r.error_message}")
        time.sleep(0.3)
    print("[dashboard] baseline 타임아웃 — 진행(일부 패널 부정확 가능)\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="통합 데모 대시보드(3센서 1창)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--contacts-port", default="auto", help="contacts 센서 포트('none'=비활성)")
    p.add_argument("--theta-port", default="none", help="theta 센서 포트")
    p.add_argument("--bending-port", default="none", help="bending 센서 포트")
    p.add_argument("--run-dir", default=str(_ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"))
    p.add_argument("--diameter", type=float, default=10.0, help="D10 기준(고정)")
    p.add_argument("--contacts", type=int, default=3, help="최대 접촉 수(다중=3)")
    p.add_argument("--z-calib", default=None)
    p.add_argument("--min-distance-mm", type=float, default=None)
    p.add_argument("--rel-threshold", type=float, default=0.3)
    p.add_argument("--min-fz", type=float, default=0.2)
    p.add_argument("--theta-smooth", type=int, default=7)
    p.add_argument("--theta-deadband", type=float, default=20.0)
    p.add_argument("--viz-fps", type=float, default=10.0)
    p.add_argument("--infer-max-fps", type=float, default=20.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--mock", action="store_true", help="하드웨어 없이(연결된 포트 자리에 mock)")
    p.add_argument("--baudrate", type=int, default=250000)
    p.add_argument("--baseline-seconds", type=float, default=5.0)
    p.add_argument("--startup-delay", type=float, default=2.0)
    p.add_argument("--protocol", choices=["auto", "binary", "csv"], default="binary")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.min_distance_mm is None:
        args.min_distance_mm = args.diameter        # 단일접촉 peak-split 방지
    ports = {"contacts": args.contacts_port, "theta": args.theta_port, "bending": args.bending_port}
    if all(v == "none" for v in ports.values()):
        ports["contacts"] = "auto"                  # 기본: 1센서 → contacts 패널
        print("[dashboard] 포트 미지정 → contacts=auto 로 진행")

    print(f"[1/3] SATS 엔진 로드(공유): {args.run_dir}")
    engine = SATSInferenceEngine(args.run_dir, device=args.device, indenter_diameter_mm=args.diameter)
    z_path = args.z_calib or (Path(__file__).resolve().parent / "z_calibration_v6.json")
    z_calib = ZCalibration.load(z_path) if Path(z_path).exists() else None

    bi = None
    if ports["theta"] != "none" or ports["bending"] != "none":
        from sats.bending.config import BendingConfig
        from sats.inference.bending_infer import BendingInference
        est = _estimator_ckpt()
        print(f"[2/3] bending estimator 로드: {est.parent.name if est.name=='best.pt' else est.name}")
        bi = BendingInference(est, device=engine.device, restorer=None, cfg=BendingConfig())

    print("[3/3] 리더 시작")
    channels = [SensorChannel(role, _build_reader(ports[role], args, engine.window_size),
                              engine, bi, z_calib, args) for role in _ROLES]
    _wait_baselines(channels)

    from sats.inference.dashboard import Dashboard
    dash = Dashboard(channels, engine, args)
    try:
        dash.run()
    finally:
        for c in channels:
            if c.reader is not None:
                c.reader.stop()


if __name__ == "__main__":
    main()
