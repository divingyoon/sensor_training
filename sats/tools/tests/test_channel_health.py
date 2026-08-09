"""16채널 건강도 진단 — 죽은/약한 채널과 세션 중 사망 시각 탐지."""
import numpy as np
import pytest

from sats.tools.channel_health import analyze_channels, dead_indices

FS = 200.0


def _raw(n=6000, n_ch=16, seed=0):
    """정상 센서 raw: 채널마다 다른 진폭의 변동 + 노이즈."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    base = 6.9e6
    sig = np.sin(2 * np.pi * t / 3.0)[:, None] * (base * 0.02)
    return base + sig * np.linspace(0.8, 1.2, n_ch)[None, :] + rng.normal(0, 300, (n, n_ch))


def test_all_healthy_reports_ok():
    r = analyze_channels(_raw(), fs=FS)
    assert all(c.label == "ok" for c in r.channels)
    assert dead_indices(r) == []


def test_fully_dead_channel_detected():
    x = _raw()
    x[:, 10] = 6.9e6                       # S11 완전 고착
    r = analyze_channels(x, fs=FS)
    assert r.channels[10].label == "dead"
    assert dead_indices(r) == [10]
    assert all(r.channels[i].label == "ok" for i in range(16) if i != 10)


def test_death_midway_reports_time():
    """★세션 중간 사망 — 시각을 알려줘야 앞부분만 살려 쓸 수 있다."""
    x = _raw(n=6000)
    x[3000:, 14] = x[3000, 14]             # 15초 지점에서 S15 정지
    r = analyze_channels(x, fs=FS)
    c = r.channels[14]
    assert c.label == "died_mid"
    assert c.death_t_s is not None and abs(c.death_t_s - 15.0) < 2.0


def test_weak_channel_flagged_not_dead():
    x = _raw()
    x[:, 5] = 6.9e6 + (x[:, 5] - 6.9e6) * 0.10   # 진폭 10% 로 약화
    r = analyze_channels(x, fs=FS)
    assert r.channels[5].label == "weak"
    assert dead_indices(r) == []                 # weak 는 마스킹 대상 아님


def test_multiple_dead_channels():
    x = _raw()
    x[:, 10] = 6.9e6
    x[:, 14] = 6.9e6
    assert dead_indices(analyze_channels(x, fs=FS)) == [10, 14]


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="16"):
        analyze_channels(np.zeros((100, 8)), fs=FS)


def test_short_input_rejected():
    with pytest.raises(ValueError, match="너무 짧"):
        analyze_channels(_raw(n=50), fs=FS)


def test_summary_line_lists_dead_channels():
    x = _raw()
    x[:, 10] = 6.9e6
    s = analyze_channels(x, fs=FS).summary()
    assert "S11" in s and "dead" in s
