"""변형 크기별 평가 — 평균 하나로 가려지는 성능 차이를 드러낸다."""
import numpy as np
import torch

from sats.bending.train_deform_restorer import _MAG_BINS, evaluate_by_magnitude

W, C = 10, 16


class _IdentitySats:
    """SATS 대역: 입력 pct 를 그대로 작은 맵으로 펼치는 최소 대역품."""

    def __call__(self, x):
        b = x.shape[0]
        return x.mean(dim=1).reshape(b, 4, 4)


class _NoopRestorer(torch.nn.Module):
    def forward(self, x):
        return x * 0.5                      # 절반만 억제하는 더미


def _windows(mags):
    """지정한 |pct|max 를 갖는 윈도우들."""
    rng = np.random.default_rng(0)
    out = []
    for m in mags:
        w = rng.normal(0, 1, (W, C)).astype(np.float32)
        out.append((w / np.abs(w).max() * m).astype(np.float32))
    return np.stack(out)


def _run(val, pool):
    import sats.bending.eval_contact_preservation as E
    orig_map, orig_peak = E._sats_map, E._peak_xy
    E._sats_map = lambda sats, x: sats(x)
    E._peak_xy = lambda m: np.zeros((len(m), 2))
    try:
        return evaluate_by_magnitude(_NoopRestorer(), val, pool, _IdentitySats(), "cpu")
    finally:
        E._sats_map, E._peak_xy = orig_map, orig_peak


def test_splits_into_magnitude_bins():
    val = _windows([5.0] * 20 + [30.0] * 20)          # 두 구간에만 표본
    res = _run(val, _windows([2.0] * 40))
    assert [(r["mag_lo"], r["mag_hi"]) for r in res] == [(0.0, 10.0), (25.0, 50.0)]
    assert all(r["n_windows"] == 20 for r in res)


def test_skips_bins_with_too_few_samples():
    """표본 8개 미만 구간은 수치가 의미 없으므로 보고하지 않는다."""
    val = _windows([5.0] * 20 + [30.0] * 3)
    res = _run(val, _windows([2.0] * 40))
    assert [(r["mag_lo"], r["mag_hi"]) for r in res] == [(0.0, 10.0)]


def test_bins_cover_all_magnitudes():
    assert _MAG_BINS[0][0] == 0.0 and np.isinf(_MAG_BINS[-1][1])
    for (_, hi), (lo, _) in zip(_MAG_BINS[:-1], _MAG_BINS[1:]):
        assert hi == lo                                # 경계에 빈틈 없음
