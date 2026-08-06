"""실시간 데모 시각화 — 다중접촉(최대3) heatmap + 최적(최대 fz) 프레임. 2D/3D.

live 맵(왼쪽) + 최적 프레임(오른쪽) 나란히. 접촉별 마커·라벨(x,y,z,fz)·theta 표시.
show=False(Agg)면 창 없이 save()만 — 헤드리스 테스트용. 실데모는 show=True(대화형 백엔드).
"""
from __future__ import annotations

import numpy as np

from sats.inference.demo_contacts import Contact

_COLORS = ["#00e5ff", "#ffea00", "#ff5cf0"]   # 접촉 #1,#2,#3


class DemoViz2D:
    def __init__(self, grid_min_mm: float, grid_max_mm: float, vmax: float | None = None,
                 title: str = "SATS v6 realtime demo", show: bool = True) -> None:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.show = show
        self.ext = [grid_min_mm, grid_max_mm, grid_min_mm, grid_max_mm]
        self.vmax = vmax
        self.fig, (self.ax_live, self.ax_best) = plt.subplots(1, 2, figsize=(12, 6))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.im_live = self._init_ax(self.ax_live, "live")
        self.im_best = self._init_ax(self.ax_best, "optimized frame (max Fz)")
        self._markers: list = []
        if show:
            plt.ion(); plt.show(block=False)

    def _init_ax(self, ax, name):
        im = ax.imshow(np.zeros((2, 2)), origin="lower", extent=self.ext, cmap="turbo",
                       vmin=0, vmax=(self.vmax or 1.0), aspect="equal", interpolation="bilinear")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        return im

    def _draw_contacts(self, ax, contacts: list[Contact]):
        for txt in list(ax.texts):
            txt.remove()
        for ln in list(ax.lines):
            ln.remove()
        for i, c in enumerate(contacts[:3]):
            col = _COLORS[i]
            ax.plot([c.x_mm], [c.y_mm], marker="+", color=col, markersize=16, markeredgewidth=2.5, lw=0)
            z = "n/a" if np.isnan(c.z_mm) else f"{c.z_mm:.2f}"
            ax.annotate(f"#{i+1} ({c.x_mm:+.1f},{c.y_mm:+.1f})\n z{z} fz{c.fz_n:.2f}N",
                        (c.x_mm, c.y_mm), color=col, fontsize=8, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    def _set_map(self, im, ax, pred_map, contacts):
        pm = np.clip(pred_map, 0, None) / 100.0     # 학습스케일 → N/mm²
        im.set_data(pm)
        if self.vmax is None:
            im.set_clim(0, max(pm.max(), 1e-6))
        self._draw_contacts(ax, contacts)

    def _set_blank(self, im, ax):
        """무접촉: 빈(0) 화면 — 노이즈 auto-scale로 뭔가 보이는 것 방지."""
        im.set_data(np.zeros((2, 2)))
        im.set_clim(0, self.vmax or 1.0)
        for txt in list(ax.texts):
            txt.remove()
        for ln in list(ax.lines):
            ln.remove()

    def update(self, pred_map, contacts, *, theta_deg=None, best_map=None, best_contacts=None):
        if contacts:
            self._set_map(self.im_live, self.ax_live, pred_map, contacts)
        else:
            self._set_blank(self.im_live, self.ax_live)   # 무접촉 → 빈 화면
        title = "live" if contacts else "live (no contact)"
        if theta_deg is not None:
            title += f"   theta={theta_deg:+.1f} deg"
        self.ax_live.set_title(title, fontsize=11)
        if best_map is not None:
            self._set_map(self.im_best, self.ax_best, best_map, best_contacts or [])
        if self.show:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)

    def save(self, path):
        self.fig.savefig(path, dpi=110, bbox_inches="tight")


class DemoViz3D:
    """3D 표면 + 접촉 수직선 마커(최대3). 실시간은 2D보다 무거움 → viz-fps 낮게 권장."""

    def __init__(self, grid_min_mm: float, grid_max_mm: float, grid_size: int,
                 vmax: float | None = None, title: str = "SATS v6 realtime 3D", show: bool = True) -> None:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        self.plt = plt; self.show = show; self.vmax = vmax
        coords = np.linspace(grid_min_mm, grid_max_mm, grid_size)
        self.XX, self.YY = np.meshgrid(coords, coords)
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(title)
        self._surf = None
        if show:
            plt.ion(); plt.show(block=False)

    def update(self, pred_map, contacts, *, theta_deg=None, **_):
        pm = np.clip(pred_map, 0, None) / 100.0
        if not contacts:                          # 무접촉 → 평평한 0 표면
            pm = np.zeros_like(pm)
        self.ax.clear()
        self.ax.plot_surface(self.XX, self.YY, pm, cmap="turbo", linewidth=0, antialiased=False)
        for i, c in enumerate(contacts[:3]):
            self.ax.plot([c.x_mm, c.x_mm], [c.y_mm, c.y_mm], [0, max(c.peak_val / 100.0, 1e-6)],
                         color=_COLORS[i], lw=2)
        t = "" if theta_deg is None else f"  theta={theta_deg:+.1f}"
        self.ax.set_title(f"3D pressure{t}")
        self.ax.set_xlabel("x (mm)"); self.ax.set_ylabel("y (mm)")
        if self.show:
            self.plt.pause(0.001)

    def save(self, path):
        self.fig.savefig(path, dpi=110, bbox_inches="tight")
