"""실시간 데모: SATS 압력맵 → 접촉별 (x,y,z,fz) 추출 + 터미널 포매팅 + 최적프레임 latch.

단일/다중(최대 3) 공통. z 는 z_calibration(A1) LUT로 근사, fz 는 접촉별 Voronoi 적분.
좌표/면적은 엔진 grid(config) 기준(0.5/0.25/0.1mm 출력 모델 모두 지원).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sats.tools.multicontact_metrics import detect_peaks


@dataclass(frozen=True)
class Contact:
    """단일 접촉 추정치."""
    x_mm: float
    y_mm: float
    z_mm: float
    fz_n: float
    peak_val: float


def extract_contacts(pred_map: np.ndarray, *, grid_min_mm: float, grid_step_mm: float,
                     taxel_area: float, diameter_mm: float, max_contacts: int = 3,
                     min_distance_mm: float = 3.0, rel_threshold: float = 0.3,
                     min_fz: float = 0.0, min_peak_val: float = 0.0,
                     z_calib=None) -> list[Contact]:
    """압력맵 → 접촉 리스트(총 fz 내림차순). z_calib 없으면 z=nan.

    fz: 양압 셀을 최근접 peak에 Voronoi 배정 후 셀합×area/100 (총합 보존, 다접촉 분할).
    ★무접촉 게이트: peak_val<min_peak_val 프레임은 빈 리스트, 접촉별 fz<min_fz 는 제거.
    (detect_peaks 는 상대임계라 무접촉 노이즈도 전역최대를 잡음 → 절대 하한 필요.)
    """
    if float(np.clip(pred_map, 0, None).max()) < min_peak_val:
        return []
    peaks = detect_peaks(pred_map, grid_min_mm=grid_min_mm, grid_step_mm=grid_step_mm,
                         min_distance_mm=min_distance_mm, rel_threshold=rel_threshold,
                         max_peaks=max_contacts)
    if len(peaks) == 0:
        return []
    h, w = pred_map.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cell_x = grid_min_mm + cc * grid_step_mm
    cell_y = grid_min_mm + rr * grid_step_mm
    pos = np.clip(pred_map, 0, None)
    dists = np.stack([np.hypot(cell_x - px, cell_y - py) for px, py, _ in peaks], axis=0)  # [K,H,W]
    assign = dists.argmin(0)
    contacts: list[Contact] = []
    for k, (px, py, pv) in enumerate(peaks):
        fz = float(pos[assign == k].sum()) * taxel_area / 100.0
        if fz < min_fz:                                  # 약한 스퓨리어스(무접촉) 제거
            continue
        z = float(z_calib.z_from_peak(pv, diameter_mm)) if z_calib is not None else float("nan")
        contacts.append(Contact(x_mm=float(px), y_mm=float(py), z_mm=z, fz_n=fz, peak_val=float(pv)))
    return sorted(contacts, key=lambda c: -c.fz_n)


def format_contacts(frame_idx: int, contacts: list[Contact], theta_deg: float | None = None) -> str:
    """터미널 한 줄(들). theta 지정 시 헤더에 표시."""
    head = f"[frame {frame_idx:6d}]"
    if theta_deg is not None:
        head += f"  theta={theta_deg:+6.1f} deg"
    head += f"  {len(contacts)} contact(s)"
    lines = [head]
    for i, c in enumerate(contacts, 1):
        z = "  n/a" if np.isnan(c.z_mm) else f"{c.z_mm:4.2f}mm"
        lines.append(f"    #{i}  x={c.x_mm:+6.1f}  y={c.y_mm:+6.1f}  z={z}  fz={c.fz_n:6.2f}N")
    return "\n".join(lines)


class FrameLatch:
    """총 fz 최대 프레임을 유지(최적/최대 프레임 = B1)."""

    def __init__(self) -> None:
        self.best_total: float = -1.0
        self.frame_idx: int | None = None
        self.pred_map: np.ndarray | None = None
        self.contacts: list[Contact] | None = None

    def update(self, frame_idx: int, pred_map: np.ndarray, contacts: list[Contact]) -> float:
        total = float(sum(c.fz_n for c in contacts))
        if total > self.best_total:
            self.best_total = total
            self.frame_idx = frame_idx
            self.pred_map = pred_map.copy()
            self.contacts = list(contacts)
        return total

    def reset(self) -> None:
        self.best_total = -1.0
        self.frame_idx = None
        self.pred_map = None
        self.contacts = None
