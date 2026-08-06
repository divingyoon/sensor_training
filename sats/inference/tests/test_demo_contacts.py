"""demo_contacts 순수 로직 테스트 (합성 압력맵)."""
import numpy as np

from sats.inference.demo_contacts import Contact, FrameLatch, extract_contacts, format_contacts

GRID, GMIN, STEP = 41, -10.0, 0.5
AREA = STEP ** 2


def _bump(cx, cy, amp=100.0, sig=1.5):
    idx = np.arange(GRID); coord = GMIN + idx * STEP
    xx, yy = np.meshgrid(coord, coord)
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))


def _extract(pmap, k):
    return extract_contacts(pmap, grid_min_mm=GMIN, grid_step_mm=STEP, taxel_area=AREA,
                            diameter_mm=5.0, max_contacts=k, min_distance_mm=3.0, rel_threshold=0.3)


def test_single_contact_position():
    c = _extract(_bump(4.0, -4.0), 1)
    assert len(c) == 1
    assert abs(c[0].x_mm - 4.0) <= STEP and abs(c[0].y_mm - (-4.0)) <= STEP
    assert c[0].fz_n > 0


def test_single_contact_not_split_when_min_distance_is_diameter():
    """한 접촉 blob 안의 국소 최대 2개(지름 이내)는 min_distance=지름이면 하나로 병합."""
    # d5 단일접촉: 주 peak + 2.5mm 옆의 약한 2차 lobe(saddle) — SATS 맵 peak-split 재현
    pmap = _bump(8.0, 8.0, amp=100.0, sig=1.2) + _bump(8.0, 5.5, amp=60.0, sig=1.2)
    split = extract_contacts(pmap, grid_min_mm=GMIN, grid_step_mm=STEP, taxel_area=AREA,
                             diameter_mm=5.0, max_contacts=2, min_distance_mm=2.0, rel_threshold=0.3)
    merged = extract_contacts(pmap, grid_min_mm=GMIN, grid_step_mm=STEP, taxel_area=AREA,
                              diameter_mm=5.0, max_contacts=2, min_distance_mm=5.0, rel_threshold=0.3)
    assert len(split) == 2      # 3mm 간격이면 두 개로 쪼개짐(회귀 대상)
    assert len(merged) == 1     # 지름(5mm) 간격이면 하나로 병합


def test_two_contacts_detected_and_split():
    pmap = _bump(-5.0, 0.0) + _bump(5.0, 0.0)
    c = _extract(pmap, 3)
    assert len(c) == 2
    xs = sorted(cc.x_mm for cc in c)
    assert abs(xs[0] - (-5.0)) <= STEP and abs(xs[1] - 5.0) <= STEP
    # 두 대칭 접촉 → fz 비슷
    assert abs(c[0].fz_n - c[1].fz_n) / max(c[0].fz_n, 1e-9) < 0.2


def test_three_contacts():
    pmap = _bump(-6, -6) + _bump(6, -6) + _bump(0, 6)
    c = _extract(pmap, 3)
    assert len(c) == 3


def test_max_contacts_cap():
    pmap = _bump(-6, -6) + _bump(6, -6) + _bump(0, 6) + _bump(6, 6)
    c = _extract(pmap, 2)
    assert len(c) == 2  # 최대 2개로 제한


def test_z_calibration_used():
    class FakeCal:
        def z_from_peak(self, pv, d):
            return 1.5
    c = extract_contacts(_bump(0, 0), grid_min_mm=GMIN, grid_step_mm=STEP, taxel_area=AREA,
                         diameter_mm=5.0, max_contacts=1, z_calib=FakeCal())
    assert abs(c[0].z_mm - 1.5) < 1e-9


def test_empty_map():
    assert _extract(np.zeros((GRID, GRID)), 3) == []


def test_frame_latch_keeps_max():
    latch = FrameLatch()
    latch.update(1, _bump(0, 0, amp=50), [Contact(0, 0, 1.0, 2.0, 50)])
    latch.update(2, _bump(0, 0, amp=100), [Contact(0, 0, 1.5, 5.0, 100)])
    latch.update(3, _bump(0, 0, amp=30), [Contact(0, 0, 0.8, 1.0, 30)])
    assert latch.frame_idx == 2 and abs(latch.best_total - 5.0) < 1e-9


def test_format_contacts_has_theta_and_xyz():
    line = format_contacts(7, [Contact(1.0, 2.0, 1.2, 3.4, 80.0)], theta_deg=12.3)
    assert "theta=" in line and "x=" in line and "fz=" in line and "frame" in line
