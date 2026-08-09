"""Tkinter 통합 데모 UI — 패널별 센서 컨트롤(포트 선택·연결/끊기·리셋) + 캔버스.

레이아웃(1창):
  [ 상단바 ] 제목 · 접촉 수(1/2/3) · 인덴터(D5/D10) · 종료
  [ 3패널 ] S1 SATS contacts | S2 bending theta(live) | S3 bending(jig)+SATS
            각 패널 = 상태배너 + matplotlib 캔버스 + (포트▾ · 연결/끊기 · 리셋)
  [ 하단바 ] 스펙 요약(D10 · 41²/81² · 200Hz …)

리셋 의미: S1=접촉 필터 초기화 · S2=flat 재영점 · S3=재장착(무접촉 유지, θ+bent-baseline).
S2/S3 리셋은 ~1초 캡처라 스레드로 실행(UI 비차단, 배너에 진행 표시).
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sats.inference.dashboard_panels import HeatmapPanel, ThetaGaugePanel, UnitsInset

_BG = "#f4f5f7"
_ACCENT = "#0a7a4f"


def list_port_choices() -> list[str]:
    """포트 드롭다운 후보 — auto + 현재 보이는 시리얼 포트."""
    try:
        from serial.tools import list_ports
        devs = [p.device for p in list_ports.comports()
                if any(k in p.device for k in ("ttyACM", "ttyUSB"))]
    except ImportError:
        devs = []
    return ["auto"] + devs


class PanelUI(ttk.Frame):
    """패널 1개 = 배너 + 캔버스 + 컨트롤(포트·연결·리셋) + 설정 요약."""

    def __init__(self, master, channel, title: str, flip_x: bool, flip_y: bool,
                 grid_min: float, grid_max: float, engines=None, sensors=None) -> None:
        super().__init__(master, padding=6)
        self.channel = channel
        self.title = title
        self.engines = engines                      # EngineCache — 센서 교체용

        # 상태 배너
        self.banner_var = tk.StringVar(value="SENSOR OFFLINE")
        self.banner = tk.Label(self, textvariable=self.banner_var, font=("TkDefaultFont", 11, "bold"),
                               fg="#888888", bg=_BG, anchor="center")
        self.banner.pack(fill="x", pady=(0, 4))

        # matplotlib 캔버스(역할별 구성)
        self.fig = Figure(figsize=(4.6, 4.8), dpi=90, facecolor=_BG)
        role = channel.role
        if role == "theta":
            # ★S2 = theta 게이지(위) + [변형 raw %(4×4) | 복원 후 SATS 맵](아래).
            #   복원 결과는 taxel 값이 아니라 SATS 압력맵(run_demo 2d 와 동일 형태)으로 —
            #   "복원하면 SATS 가 flat 을 본다(유령 없음)"가 그대로 보인다.
            gs = self.fig.add_gridspec(2, 2, height_ratios=[1.6, 1.5], hspace=0.45, wspace=0.2)
            self.p_theta = ThetaGaugePanel(self.fig.add_subplot(gs[0, :]), title)
            self.p_units = UnitsInset(self.fig.add_subplot(gs[1, 0]),
                                      title="raw (deformed) %", flip_x=flip_x, flip_y=flip_y)
            self.p_restored_map = HeatmapPanel(self.fig.add_subplot(gs[1, 1]),
                                               grid_min, grid_max, "restored -> SATS",
                                               draw_without_contacts=True,
                                               flip_x=flip_x, flip_y=flip_y)
            self.p_units_restored = None
            self.p_heat = None
        else:
            # S1(FLAT)·S3(DEFORMED) = SATS 맵 전체 크기
            self.p_heat = HeatmapPanel(self.fig.add_subplot(111), grid_min, grid_max, title,
                                       flip_x=flip_x, flip_y=flip_y)
            self.p_units = self.p_theta = None
            self.p_units_restored = None
            self.p_restored_map = None
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ★컨트롤 2행 구성 — 한 줄에 다 넣으면 패널 폭을 넘겨 뒤쪽 버튼이 잘린다.
        #   1행(row): port · 연결 · 리셋 (동작)
        #   2행(sel): v* · run/est · restore (모델 선택)
        row = ttk.Frame(self); row.pack(fill="x", pady=(6, 0))
        sel = ttk.Frame(self); sel.pack(fill="x", pady=(3, 0))
        self.sensor_var = self.run_var = None
        self.run_box = None
        ttk.Label(row, text="port").pack(side="left")
        self.port_var = tk.StringVar(value="auto")
        self.port_box = ttk.Combobox(row, textvariable=self.port_var, width=16,
                                     values=list_port_choices(), state="readonly")
        self.port_box.pack(side="left", padx=4)
        self.port_box.bind("<Button-1>", lambda e: self.port_box.configure(values=list_port_choices()))
        if engines is not None and sensors and role == "theta":
            # ★S2: 다른 센서의 theta + deform 복원기 선택
            from sats.inference.run_dashboard import available_estimators, available_restorers
            ttk.Label(sel, text="v*").pack(side="left")
            self.sensor_var = tk.StringVar(value="")
            sbox = ttk.Combobox(sel, textvariable=self.sensor_var, width=3,
                                values=list(sensors), state="readonly")
            sbox.pack(side="left", padx=2)
            sbox.bind("<<ComboboxSelected>>", lambda e: self._set_theta_sensor())
            ttk.Label(sel, text="est").pack(side="left", padx=(6, 0))
            self.est_var = tk.StringVar(value="")
            ebox = ttk.Combobox(sel, textvariable=self.est_var, width=13,
                                values=available_estimators(), state="readonly")
            ebox.pack(side="left", padx=2)
            ebox.bind("<<ComboboxSelected>>", lambda e: self._apply_estimator())
            ttk.Label(sel, text="restore").pack(side="left", padx=(6, 0))
            self.restore_var = tk.StringVar(value="")
            rbox = ttk.Combobox(sel, textvariable=self.restore_var, width=14,
                                values=available_restorers(), state="readonly")
            rbox.pack(side="left", padx=2)
            rbox.bind("<<ComboboxSelected>>", lambda e: self._apply_restorer())
        if engines is not None and sensors and role in ("contacts", "bending"):
            from sats.inference.run_dashboard import available_runs
            self._available_runs = available_runs
            ttk.Label(sel, text="v*").pack(side="left")
            self.sensor_var = tk.StringVar(value="")
            sbox = ttk.Combobox(sel, textvariable=self.sensor_var, width=3,
                                values=list(sensors), state="readonly")
            sbox.pack(side="left", padx=2)
            sbox.bind("<<ComboboxSelected>>", lambda e: self._on_sensor_selected())
            ttk.Label(sel, text="run").pack(side="left", padx=(6, 0))
            self.run_var = tk.StringVar(value="")
            self.run_box = ttk.Combobox(sel, textvariable=self.run_var, width=18,
                                        values=[], state="readonly")
            self.run_box.pack(side="left", padx=2)
            self.run_box.bind("<<ComboboxSelected>>", lambda e: self._apply_run())
            if role == "bending":
                from sats.inference.run_dashboard import available_restorers
                ttk.Label(sel, text="restore").pack(side="left", padx=(6, 0))
                self.restore_var = tk.StringVar(value="")
                rbox = ttk.Combobox(sel, textvariable=self.restore_var, width=14,
                                    values=available_restorers(), state="readonly")
                rbox.pack(side="left", padx=2)
                rbox.bind("<<ComboboxSelected>>", lambda e: self._apply_restorer())
        self.btn_conn = ttk.Button(row, text="연결", width=6, command=self._toggle_connect)
        self.btn_conn.pack(side="left", padx=2)
        reset_label = {"contacts": "리셋", "theta": "재영점", "bending": "재장착"}[role]
        self.btn_reset = ttk.Button(row, text=reset_label, width=7, command=self._reset_async)
        self.btn_reset.pack(side="left", padx=2)
        # ★S3 변형 복원 토글 — 데모의 '변형 → 정지 → 복원' 구분 동작
        self.btn_restore = None
        if role == "bending" and getattr(channel, "restorer", None) is not None:
            self.btn_restore = ttk.Button(row, width=9, command=self._toggle_restore)
            self.btn_restore.pack(side="left", padx=2)
            self._sync_restore_label()

        # 설정 요약(파트별 세팅 정리)
        self.info_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.info_var, font=("TkDefaultFont", 8)).pack(fill="x", pady=(3, 0))

    # ── 컨트롤 동작 ──────────────────────────────────────────────────────────
    def _toggle_connect(self) -> None:
        if (not self.channel.connected and self.channel.role in ("contacts", "bending")
                and self.channel.engine is None):
            print(f"[dashboard] {self.channel.role}: 먼저 v* 와 run 을 선택하세요")
            return
        ch = self.channel
        if ch.connected:
            ch.disconnect()
            self.btn_conn.configure(text="연결")
            return
        err = ch.connect(self.port_var.get())
        if err:
            self.banner_var.set(err)
            self.banner.configure(fg="#c22")
        else:
            self.btn_conn.configure(text="끊기")

    def _set_theta_sensor(self) -> None:
        err = self.channel.apply_theta_sensor(self.sensor_var.get(), self.engines)
        print(f"[dashboard] theta 센서 {self.sensor_var.get()}"
              + (f" — {err}" if err else " (복원 SATS 맵 활성)"))

    def _apply_estimator(self) -> None:
        err = self.channel.apply_estimator(self.est_var.get())
        print(f"[dashboard] {self.channel.role}: estimator {self.est_var.get()}"
              + (f" — {err}" if err else " 적용"))

    def _apply_restorer(self) -> None:
        err = self.channel.apply_restorer(self.restore_var.get())
        if err:
            print(f"[dashboard] {self.channel.role}: {err}")
        if self.btn_restore is not None:
            self.btn_restore.configure(state="normal" if self.channel.restorer else "disabled")
            self._sync_restore_label()

    def _on_sensor_selected(self) -> None:
        """v* 선택 → 그 센서의 추론 run 목록을 채운다(첫 항목 자동 선택)."""
        sensor = self.sensor_var.get()
        runs = self._available_runs(sensor)
        self.run_box.configure(values=runs)
        if runs:
            self.run_var.set(runs[0])
            self._apply_run()
        else:
            self.run_var.set("")
            print(f"[dashboard] {sensor}: 배포 run 없음 — 학습이 되어 있는지 확인")

    def _apply_run(self) -> None:
        """run(추론 파일) 확정 — 엔진 로드(캐시) + 복원기 유효성 갱신."""
        sensor, run = self.sensor_var.get(), self.run_var.get()
        if not sensor or not run:
            return
        err = self.channel.apply_run(sensor, run, self.engines)
        print(f"[dashboard] {self.channel.role}: {sensor}/{run}"
              + (f" — {err}" if err else " 적용"))
        if self.btn_restore is not None:
            state = "normal" if self.channel.restorer is not None else "disabled"
            self.btn_restore.configure(state=state)
            self._sync_restore_label()

    def _sync_restore_label(self) -> None:
        if self.btn_restore is not None:
            self.btn_restore.configure(
                text="복원 ON" if self.channel.restore_on else "복원 OFF")

    def _toggle_restore(self) -> None:
        self.channel.toggle_restore()
        self._sync_restore_label()

    def _reset_async(self) -> None:
        if not self.channel.connected or self.channel.busy:
            return
        threading.Thread(target=self.channel.reset, daemon=True).start()

    # ── 렌더 ─────────────────────────────────────────────────────────────────
    def render(self, payload: dict, args) -> None:
        b = payload["banner"]
        self.banner_var.set(b.label)
        self.banner.configure(fg=b.color)
        role = self.channel.role
        if role == "theta":
            self.p_theta.update(payload.get("theta"), payload.get("noise"), b)
            if self.p_units is not None:
                self.p_units.update(payload.get("units"))          # 센서 % 그대로(자체 스케일)
            if getattr(self, "p_restored_map", None) is not None:
                self.p_restored_map.update(payload.get("restored_map"), [], b)
            self.info_var.set(f"smooth={args.theta_smooth}  deadband={args.theta_deadband:g}°"
                              f"  (valid 20–150°)")
        else:
            theta = payload.get("theta") if role == "bending" else None
            self.p_heat.update(payload.get("pred_map"), payload.get("contacts", []), b,
                               theta_deg=theta, note=payload.get("note", ""))
            if self.p_units is not None:
                u = payload.get("units")
                shared = (max(float(abs(__import__("numpy").asarray(u, float)).max()), 1e-6)
                          if u is not None else None)   # raw 기준 공유 스케일
                self.p_units.update(u, vmax=shared)
                if getattr(self, "p_units_restored", None) is not None:
                    ur = payload.get("units_restored")
                    self.p_units_restored.update(ur, vmax=shared)
                    if ur is not None:
                        import numpy as _np
                        self.p_units_restored.ax.set_title(
                            f"restored (res {float(_np.abs(ur).max()):.2f}%)", fontsize=8)
            extra = (f"  armed θ={self.channel.theta_fixed:+.1f}°" if role == "bending"
                     and self.channel.armed and self.channel.restorer is None else "")
            self.info_var.set(f"contacts≤{args.contacts}  D{int(args.diameter)}"
                              f"  fz_on={args.fz_on:g}N{extra}")
        if self.channel.connected:
            self.btn_conn.configure(text="끊기")
        else:
            self.btn_conn.configure(text="연결")
        self.canvas.draw_idle()


class DashboardApp:
    """Tk 루트 — 3 PanelUI + 전역 설정(접촉 수·인덴터) + 폴링 루프."""

    def __init__(self, channels, engine, args, engines=None) -> None:
        from sats.inference.run_dashboard import available_sensors
        self.engines = engines
        self.sensors = available_sensors() if engines is not None else []
        self.args = args
        self.engine = engine
        self.channels = {c.role: c for c in channels}
        self.root = tk.Tk()
        self.root.title("SATS v6 — unified demo dashboard")
        self.root.configure(bg=_BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=_BG)
        style.configure("TButton", padding=4)

        # 상단바
        top = ttk.Frame(self.root, padding=(10, 8)); top.pack(fill="x")
        ttk.Label(top, text="SATS realtime demo", font=("TkDefaultFont", 13, "bold"),
                  foreground=_ACCENT).pack(side="left")
        ttk.Button(top, text="종료", command=self.root.destroy).pack(side="right")
        # 인덴터 토글
        self.dia_var = tk.StringVar(value=f"D{int(args.diameter)}")
        for d in ("D10", "D5"):
            ttk.Radiobutton(top, text=d, value=d, variable=self.dia_var,
                            command=self._set_diameter).pack(side="right", padx=2)
        ttk.Label(top, text="indenter:").pack(side="right", padx=(12, 2))
        # 접촉 수 토글
        self.n_var = tk.IntVar(value=args.contacts)
        for n in (3, 2, 1):
            ttk.Radiobutton(top, text=str(n), value=n, variable=self.n_var,
                            command=self._set_contacts).pack(side="right", padx=2)
        ttk.Label(top, text="contacts:").pack(side="right", padx=(12, 2))

        # 3패널
        mid = ttk.Frame(self.root, padding=(6, 0)); mid.pack(fill="both", expand=True)
        gmin, gmax = ((engine.grid_min_mm, engine.grid_max_mm) if engine is not None
                      else (-10.0, 10.0))          # 물리 범위는 모든 run 동일
        fx, fy = getattr(args, "flip_x", False), getattr(args, "flip_y", True)
        titles = {"contacts": "S1  FLAT SATS", "theta": "S2  theta (deg) + restore",
                  "bending": "S3  DEFORMED SATS"}
        self.panels: dict[str, PanelUI] = {}
        for i, role in enumerate(("contacts", "theta", "bending")):
            p = PanelUI(mid, self.channels[role], titles[role], fx, fy, gmin, gmax,
                        engines=self.engines, sensors=self.sensors)
            p.grid(row=0, column=i, sticky="nsew", padx=4)
            mid.columnconfigure(i, weight=1)
            self.panels[role] = p
        mid.rowconfigure(0, weight=1)

        # 하단 스펙바
        grid_txt = (f"{engine.grid_size}x{engine.grid_size}@{engine.grid_step_mm:g}mm"
                    if engine is not None else "run 선택 대기")
        spec = (f"SATS {grid_txt} (20x20mm) · raw 16-taxel 200Hz/250k · infer 129Hz"
                f" · Fz 0-3.9N (D10) · theta 20-150° (MAE 1.78°)")
        ttk.Label(self.root, text=spec, font=("TkDefaultFont", 8), padding=(10, 4)).pack(fill="x")

        self._tick_ms = max(30, int(1000 / max(args.viz_fps, 1)))

    # ── 전역 설정 ────────────────────────────────────────────────────────────
    def _set_contacts(self) -> None:
        self.args.contacts = int(self.n_var.get())

    def _set_diameter(self) -> None:
        d = 5.0 if self.dia_var.get() == "D5" else 10.0
        self.args.diameter = d
        self.args.min_distance_mm = d
        engines = ({self.engine} if self.engine is not None else set()) \
            | {c.engine for c in self.channels.values() if c.engine is not None}
        for e in engines:                            # ★패널별 엔진 전부에 적용
            setf = getattr(e, "set_diameter", None)
            if setf is not None:
                setf(d)

    # ── 루프 ─────────────────────────────────────────────────────────────────
    def _tick(self) -> None:
        for role, ch in self.channels.items():
            try:
                payload = ch.poll()
            except Exception as e:                 # 리더 급단선 등 — 패널만 오프라인
                ch.disconnect()
                from sats.inference.demo_contacts import state_banner
                payload = {"kind": "heatmap", "banner": state_banner(None, connected=False),
                           "pred_map": None, "contacts": [], "theta": None,
                           "noise": None, "units": None}
                print(f"[ui] {role} 리더 오류 → 연결 해제: {e}")
            self.panels[role].render(payload, self.args)
        self.root.after(self._tick_ms, self._tick)

    def run(self) -> None:
        self.root.after(self._tick_ms, self._tick)
        self.root.mainloop()
        for ch in self.channels.values():
            ch.disconnect()
