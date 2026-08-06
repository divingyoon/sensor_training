"""실시간 데모 시각화 — 단일 패널 live heatmap(단색 초록). 2D/3D.

접촉별 마커·라벨(x,y,z,fz)·theta 표시. 색은 **프레임 peak 기준 상대 정규화**
(0=연한 초록 → 그 접촉의 peak=어두운 초록)라, 손가락 등 약한 접촉도 항상 또렷하게 보인다.
show=False(Agg)면 창 없이 save()만 — 헤드리스 테스트용. 실데모는 show=True(대화형 백엔드).
"""
from __future__ import annotations

import numpy as np

from sats.inference.demo_contacts import Contact

_MARKER_COLORS = ["#d81e00", "#0033cc", "#111111"]   # 접촉 #1,#2,#3 (초록 배경 대비 고대비)


def _green_cmap():
    """0=연한 초록 → 최고=어두운 초록, 밝기가 고르게 변하는 램프.

    matplotlib 'Greens'(지각적 균일)를 흰색 대신 연한 초록(0.12)부터 잘라 씀 →
    peak에서 바깥으로 매끄럽게 감쇠하는 압력장이 전 구간에서 그라데이션으로 보임.
    """
    import numpy as _np
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    base = mpl.colormaps["Greens"]        # 지각적 균일 초록(deprecated cm.get_cmap 회피)
    return ListedColormap(base(_np.linspace(0.12, 1.0, 256)), name="demo_green")


class DemoViz2D:
    """단일 패널 live heatmap — 단색 초록. 프레임 peak=1 상대 정규화(약접촉도 진한 초록으로).

    vmax 지정 시 그 값(학습스케일)을 1.0으로 하는 절대 스케일, None이면 매 프레임 peak 상대.
    """

    def __init__(self, grid_min_mm: float, grid_max_mm: float, vmax: float | None = None,
                 title: str = "SATS v6 realtime demo", show: bool = True) -> None:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.show = show
        self.ext = [grid_min_mm, grid_max_mm, grid_min_mm, grid_max_mm]
        self.vmax = vmax                                  # None=상대(peak), 값=절대(학습스케일)
        self.cmap = _green_cmap()                         # 밝기 균일 초록 램프(감쇠 그라데이션)
        self.fig, self.ax_live = plt.subplots(1, 1, figsize=(7, 6))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.im_live = self._init_ax(self.ax_live, "live")
        cbar = self.fig.colorbar(self.im_live, ax=self.ax_live, fraction=0.046, pad=0.04)
        label = "relative intensity (0 = light → peak = dark green)" if vmax is None \
            else "intensity (0 = light → max = dark green)"
        cbar.set_label(label, fontsize=9)
        if show:
            plt.ion(); plt.show(block=False)

    def _init_ax(self, ax, name):
        im = ax.imshow(np.zeros((2, 2)), origin="lower", extent=self.ext, cmap=self.cmap,
                       vmin=0, vmax=1.0, aspect="equal", interpolation="bicubic")  # 표시는 0~1 정규화, 매끄러운 감쇠
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        return im

    def _draw_contacts(self, ax, contacts: list[Contact]):
        for txt in list(ax.texts):
            txt.remove()
        for ln in list(ax.lines):
            ln.remove()
        for i, c in enumerate(contacts[:3]):
            col = _MARKER_COLORS[i]
            ax.plot([c.x_mm], [c.y_mm], marker="+", color=col, markersize=16, markeredgewidth=2.5, lw=0)
            z = "n/a" if np.isnan(c.z_mm) else f"{c.z_mm:.2f}"
            ax.annotate(f"#{i+1} ({c.x_mm:+.1f},{c.y_mm:+.1f})\n z{z} fz{c.fz_n:.2f}N",
                        (c.x_mm, c.y_mm), color=col, fontsize=8, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    def _set_map(self, im, ax, pred_map, contacts):
        pm = np.clip(pred_map, 0, None)
        denom = self.vmax if self.vmax else max(float(pm.max()), 1e-6)   # 절대(고정) 또는 peak 상대
        im.set_data(np.clip(pm / denom, 0, 1))            # 0~1 정규화 → peak=어두운 초록
        self._draw_contacts(ax, contacts)

    def _set_blank(self, im, ax):
        """무접촉: 0(연한 초록 균일) — 노이즈로 뭔가 보이는 것 방지."""
        im.set_data(np.zeros((2, 2)))
        for txt in list(ax.texts):
            txt.remove()
        for ln in list(ax.lines):
            ln.remove()

    def update(self, pred_map, contacts, *, theta_deg=None, **_):
        if contacts:
            self._set_map(self.im_live, self.ax_live, pred_map, contacts)
        else:
            self._set_blank(self.im_live, self.ax_live)   # 무접촉 → 빈 화면
        title = "live" if contacts else "live (no contact)"
        if theta_deg is not None:
            title += f"   theta={theta_deg:+.1f} deg"
        self.ax_live.set_title(title, fontsize=11)
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
        self.plt = plt; self.show = show
        self.vmax = vmax                                  # None=상대(peak), 값=절대(학습스케일)
        self.cmap = _green_cmap()
        coords = np.linspace(grid_min_mm, grid_max_mm, grid_size)
        self.XX, self.YY = np.meshgrid(coords, coords)
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(title)
        self._surf = None
        if show:
            plt.ion(); plt.show(block=False)

    def update(self, pred_map, contacts, *, theta_deg=None, **_):
        pm = np.clip(pred_map, 0, None)
        if not contacts:                          # 무접촉 → 평평한 0 표면
            pm = np.zeros_like(pm)
        else:
            denom = self.vmax if self.vmax else max(float(pm.max()), 1e-6)   # 절대 또는 peak 상대
            pm = np.clip(pm / denom, 0, 1)        # 0~1 정규화(peak=1)
        self.ax.clear()
        self.ax.plot_surface(self.XX, self.YY, pm, cmap=self.cmap, vmin=0, vmax=1.0,
                             linewidth=0, antialiased=False)
        self.ax.set_zlim(0, 1.0)
        for i, c in enumerate(contacts[:3]):
            self.ax.plot([c.x_mm, c.x_mm], [c.y_mm, c.y_mm], [0, 1.0],
                         color=_MARKER_COLORS[i], lw=2)
        t = "" if theta_deg is None else f"  theta={theta_deg:+.1f}"
        self.ax.set_title(f"3D pressure{t}")
        self.ax.set_xlabel("x (mm)"); self.ax.set_ylabel("y (mm)")
        if self.show:
            self.plt.pause(0.001)

    def save(self, path):
        self.fig.savefig(path, dpi=110, bbox_inches="tight")
