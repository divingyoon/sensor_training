"""통합 대시보드 패널 렌더러 — 주어진 matplotlib ax 위에 그린다.

3종:
  HeatmapPanel  : SATS 초록 압력맵 + 접촉 마커(x,y,z,fz) + 상태 배너. (contacts·bending 공용)
  ThetaGaugePanel: 밴딩각 게이지(유효 20–150° 밴드) + live 값·노이즈. (theta)
  UnitsInset    : 16-taxel(4×4) 원본 미니맵(bending 패널 부속, 선택).

색·마커는 demo_viz 와 통일(초록 램프·고대비 마커). 상태 배너는 demo_contacts.state_banner.
show=False(Agg) 헤드리스에서도 예외 없이 동작(테스트).
"""
from __future__ import annotations

import numpy as np

from sats.bending.geometry import TAXEL_XY_MM
from sats.inference.demo_contacts import Contact, ContactState, StateBanner
from sats.inference.demo_viz import _MARKER_COLORS, _green_cmap, taxel_grid

_THETA_FULL = (0.0, 160.0)     # 게이지 전체 범위
_THETA_VALID = (20.0, 150.0)   # 유효 관측 밴드


def _clear_dynamic(ax) -> None:
    """동적 아티스트(마커·라벨) 제거 — imshow(image)는 유지."""
    for txt in list(ax.texts):
        txt.remove()
    for ln in list(ax.lines):
        ln.remove()


def _banner_title(ax, banner: StateBanner, prefix: str) -> None:
    """상태 배너를 색 있는 제목으로."""
    ax.set_title(f"{prefix}\n{banner.label}", fontsize=10, color=banner.color,
                 fontweight="bold")


class HeatmapPanel:
    """SATS 초록 압력맵 + 접촉 마커. contacts/bending 공용(bending은 theta 병기).

    flip_y(기본 True): 센서 +y가 물리적으로 아래쪽이면 화면도 아래로 → 손 이동과 마커
    이동 방향 일치(마운팅 방향 보정). flip_x 는 좌우 반전.
    """

    def __init__(self, ax, grid_min_mm: float, grid_max_mm: float, prefix: str = "SATS",
                 draw_without_contacts: bool = False,
                 flip_x: bool = False, flip_y: bool = True, compact: bool = False) -> None:
        self.ax = ax
        self.prefix = prefix
        self.draw_without_contacts = draw_without_contacts
        self.gmin, self.gmax = grid_min_mm, grid_max_mm
        self.ext = [grid_min_mm, grid_max_mm, grid_min_mm, grid_max_mm]
        self.cmap = _green_cmap()
        self.im = ax.imshow(np.zeros((2, 2)), origin="lower", extent=self.ext, cmap=self.cmap,
                            vmin=0, vmax=1.0, aspect="equal", interpolation="bicubic")
        # 물리 마운팅 방향에 맞춰 축 반전(이미지+마커 함께 반전 → 방향 일치)
        ax.set_xlim(grid_max_mm, grid_min_mm) if flip_x else ax.set_xlim(grid_min_mm, grid_max_mm)
        ax.set_ylim(grid_max_mm, grid_min_mm) if flip_y else ax.set_ylim(grid_min_mm, grid_max_mm)
        if compact:
            # ★S2 미니맵용 — 전체 크기 패널의 축 라벨("y (mm) (down = +)")이
            #   옆의 raw 미니맵을 침범한다(실측). 미니맵은 라벨·눈금 없이 그린다.
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)  (down = +)" if flip_y else "y (mm)")
        _banner_title(ax, StateBanner(ContactState.OFFLINE, "SENSOR OFFLINE",
                                      "#bbbbbb"), prefix)

    def _draw_contacts(self, contacts: list[Contact]) -> None:
        for i, c in enumerate(contacts[:3]):
            col = _MARKER_COLORS[i]
            self.ax.plot([c.x_mm], [c.y_mm], marker="+", color=col, markersize=15,
                         markeredgewidth=2.4, lw=0)
            z = "n/a" if np.isnan(c.z_mm) else f"{c.z_mm:.2f}"
            self.ax.annotate(f"#{i+1} ({c.x_mm:+.1f},{c.y_mm:+.1f})\n z{z} fz{c.fz_n:.2f}N",
                             (c.x_mm, c.y_mm), color=col, fontsize=8, ha="left", va="bottom",
                             xytext=(3, 3), textcoords="offset points")

    def update(self, pred_map, contacts: list[Contact], banner: StateBanner,
               theta_deg: float | None = None, note: str = "") -> None:
        _clear_dynamic(self.ax)
        prefix = self.prefix if theta_deg is None else f"{self.prefix}   θ={theta_deg:+.0f}°"
        if note:
            prefix = f"{prefix}   {note}"
        blank = (banner.state is ContactState.OFFLINE or pred_map is None
                 or (not contacts and not self.draw_without_contacts))
        if blank:
            self.im.set_data(np.zeros((2, 2)))     # OFFLINE·무접촉 → blank(연한 초록)
        else:
            pm = np.clip(pred_map, 0, None)
            denom = max(float(pm.max()), 1e-6)     # 프레임 peak 상대 정규화(약접촉도 또렷)
            self.im.set_data(np.clip(pm / denom, 0, 1))
            if contacts:
                self._draw_contacts(contacts)
        _banner_title(self.ax, banner, prefix)


class ThetaGaugePanel:
    """밴딩각 게이지 — 유효밴드(20–150°) 음영 + 현재값 마커 + 큰 텍스트."""

    def __init__(self, ax, prefix: str = "bending θ (live)") -> None:
        self.ax = ax
        self.prefix = prefix
        ax.set_xlim(*_THETA_FULL); ax.set_ylim(0, 1)
        ax.set_yticks([])
        # xlabel 없음 — 제목에 단위가 있고, 라벨이 아래 행(restored 맵 제목)과 겹친다(실측)
        ax.axvspan(*_THETA_VALID, color="#0033cc", alpha=0.10, zorder=0)  # 유효 관측 밴드
        ax.axvline(_THETA_VALID[0], color="#0033cc", lw=1, ls="--", alpha=0.5)
        ax.axvline(_THETA_VALID[1], color="#0033cc", lw=1, ls="--", alpha=0.5)
        _banner_title(ax, StateBanner(ContactState.OFFLINE, "SENSOR OFFLINE", "#bbbbbb"), prefix)

    def update(self, theta_deg: float | None, noise: float | None, banner: StateBanner) -> None:
        _clear_dynamic(self.ax)
        if banner.state is ContactState.OFFLINE or theta_deg is None:
            _banner_title(self.ax, banner, self.prefix)
            return
        th = float(np.clip(theta_deg, *_THETA_FULL))
        col = banner.color
        self.ax.plot([th, th], [0, 1], color=col, lw=4, solid_capstyle="round")  # 현재각 바
        self.ax.plot([th], [1.02], marker="v", color=col, markersize=12, clip_on=False)
        ntxt = "" if noise is None else f"  (±{noise:.1f})"
        self.ax.text(0.5, 0.5, f"{theta_deg:+.1f}°{ntxt}", transform=self.ax.transAxes,
                     ha="center", va="center", fontsize=22, fontweight="bold", color=col)
        _banner_title(self.ax, banner, self.prefix)


class UnitsInset:
    """16-taxel(4×4) 원본 미니맵 — bending 패널 부속(밴딩 공간 gradient 관찰)."""

    def __init__(self, ax, title: str = "16-taxel (raw)",
                 flip_x: bool = False, flip_y: bool = True) -> None:
        self.ax = ax
        self.im = ax.imshow(np.zeros((4, 4)), origin="lower", cmap="RdBu_r",
                            vmin=-1, vmax=1, extent=[-13, 13, -13, 13], interpolation="nearest")
        ax.set_xlim(13, -13) if flip_x else ax.set_xlim(-13, 13)   # heatmap 과 방향 일치
        ax.set_ylim(13, -13) if flip_y else ax.set_ylim(-13, 13)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        self._ema = None          # 표시 평활(★수시로 색이 바뀌는 깜빡임 억제)
        self._ema_vmax = 1e-6

    def update(self, pct_window, vmax: float | None = None) -> None:
        """vmax: 컬러 스케일 고정값. ★before/after 비교 시 두 미니맵이 같은 스케일을
        써야 한다 — 복원 결과(≈0)를 자체 스케일로 그리면 잔여 노이즈가 꽉 찬 것처럼
        보여 복원이 안 된 것으로 오독된다."""
        _clear_dynamic(self.ax)
        if pct_window is None:
            # ★빈 흰 판은 "복원이 잘 됨"과 구분되지 않는다 — 미장착임을 명시.
            self.im.set_data(np.zeros((4, 4)))
            self.ax.text(0, 0, "select restore", ha="center", va="center",
                         fontsize=7, color="#999999")
            return
        dp = np.asarray(pct_window, float).mean(0)      # [16] 윈도우 평균 Δp%
        # EMA(α=0.25): 프레임 노이즈로 색·숫자가 수시로 바뀌지 않게. 스케일도 함께
        # 평활 — 자동 스케일이 매 프레임 튀면 값이 같아도 색이 흔들린다.
        self._ema = dp if self._ema is None else 0.75 * self._ema + 0.25 * dp
        dp = self._ema
        raw_vmax = vmax if vmax else max(float(np.abs(dp).max()), 1e-6)
        self._ema_vmax = max(0.9 * self._ema_vmax, raw_vmax)   # 상승 즉시·하강 완만
        vmax = self._ema_vmax
        self.im.set_data(taxel_grid(dp)); self.im.set_clim(-vmax, vmax)
        for i in range(16):
            x, y = TAXEL_XY_MM[i + 1]
            self.ax.text(x, y, f"{dp[i]:+.1f}", ha="center", va="center", fontsize=6,
                         color="k" if abs(dp[i]) < vmax * 0.6 else "w")
