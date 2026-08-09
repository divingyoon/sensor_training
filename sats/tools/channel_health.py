"""16채널 건강도 진단 — 파손된 taxel 을 취득 직후에 잡아낸다.

barometric taxel 은 과도한 변형·압박으로 파손되어 신호를 멈출 수 있다. 이를 모르고
학습하면 두 가지가 조용히 망가진다:

  1. **드리프트 보정 오염** — 세션 *중간*에 죽으면 앞/뒤 baseline 이 크게 달라져,
     선형 보정이 그 채널에 존재하지 않는 램프를 만들어 넣는다(가짜 변형 신호).
  2. **학습/추론 분포 불일치** — 16채널 정상으로 학습한 SATS 에 14채널만 들어오면
     죽은 taxel 근처 접촉이 살아있는 이웃 쪽으로 끌려가 위치가 편향된다.

그래서 "죽었는가"뿐 아니라 **언제 죽었는가**를 함께 보고한다.

사용:
    python -m sats.tools.channel_health skin_ws/raw_data/deform/v7
    python -m sats.tools.channel_health learning_data/sensor_raw_bin/ecomesh_v7_xy1
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_SENSORS = 16
_WIN_SEC = 5.0            # 활동도 판정 창
_DEAD_RATIO = 0.02        # 채널 중앙값 대비 이 이하면 신호 없음
_WEAK_RATIO = 0.20        # 살아는 있으나 현저히 약함
_MIN_FRAMES = 200


@dataclass(frozen=True)
class ChannelStatus:
    """채널 1개 진단. label ∈ {ok, weak, dead, died_mid}."""
    index: int                       # 0-based
    label: str
    activity: float                  # 채널 중앙값 대비 변동 비율
    death_t_s: float | None = None   # died_mid 일 때 사망 추정 시각(초)

    @property
    def name(self) -> str:
        return f"S{self.index + 1:02d}"


@dataclass(frozen=True)
class HealthReport:
    channels: tuple[ChannelStatus, ...]
    duration_s: float

    def summary(self) -> str:
        bad = [c for c in self.channels if c.label != "ok"]
        if not bad:
            return f"16채널 정상 ({self.duration_s:.0f}s)"
        parts = []
        for c in bad:
            when = f"@{c.death_t_s:.0f}s" if c.death_t_s is not None else ""
            parts.append(f"{c.name} {c.label}{when}(활동 {c.activity:.2f})")
        return f"★이상 {len(bad)}개 — " + " · ".join(parts)


def dead_indices(report: HealthReport) -> list[int]:
    """마스킹해야 할 채널(완전 사망 + 중도 사망). weak 는 제외 — 살아있으므로."""
    return [c.index for c in report.channels if c.label in ("dead", "died_mid")]


def _window_activity(x: np.ndarray, win: int) -> np.ndarray:
    """[W, 16] 창별 표준편차 — 창 단위로 신호 유무를 본다."""
    n_win = len(x) // win
    if n_win < 1:
        return x.std(0)[None, :]
    return x[: n_win * win].reshape(n_win, win, x.shape[1]).std(axis=1)


def analyze_channels(raw: np.ndarray, *, fs: float = 200.0, win_sec: float = _WIN_SEC,
                     dead_ratio: float = _DEAD_RATIO, weak_ratio: float = _WEAK_RATIO
                     ) -> HealthReport:
    """raw[N, 16] → 채널별 진단.

    절대 임계가 아니라 **같은 세션 다른 채널 대비 상대 활동도**로 판정한다. 센서·조건마다
    신호 크기가 달라 절대값 기준은 오탐하기 때문이다.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != N_SENSORS:
        raise ValueError(f"raw 는 [N, {N_SENSORS}] 여야 함 — 받은 shape {raw.shape}")
    if len(raw) < _MIN_FRAMES:
        raise ValueError(f"프레임이 너무 짧음({len(raw)} < {_MIN_FRAMES})")

    win = max(int(win_sec * fs), 10)
    act_w = _window_activity(raw, win)                       # [W, 16]
    ref_w = np.median(act_w, axis=1, keepdims=True)          # 창별 기준(중앙 채널)
    ratio_w = act_w / np.maximum(ref_w, 1e-12)               # [W, 16] 상대 활동도
    overall = raw.std(0) / max(float(np.median(raw.std(0))), 1e-12)

    out = []
    for i in range(N_SENSORS):
        alive = ratio_w[:, i] >= dead_ratio
        if alive.all():
            label = "weak" if overall[i] < weak_ratio else "ok"
            out.append(ChannelStatus(i, label, float(overall[i])))
            continue
        if not alive.any():                                  # 처음부터 끝까지 무신호
            out.append(ChannelStatus(i, "dead", float(overall[i])))
            continue
        # 마지막으로 살아있던 창 다음부터 끝까지 죽어 있으면 '중도 사망'
        last_alive = int(np.flatnonzero(alive)[-1])
        if last_alive < len(alive) - 1:
            out.append(ChannelStatus(i, "died_mid", float(overall[i]),
                                     death_t_s=(last_alive + 1) * win / fs))
        else:                                                # 중간중간 끊김 — 접촉 불량
            out.append(ChannelStatus(i, "weak", float(overall[i])))
    return HealthReport(tuple(out), duration_s=len(raw) / fs)


# ── CLI ────────────────────────────────────────────────────────────────────
def _iter_inputs(root: Path):
    """(이름, raw[N,16], fs) 를 yield — deform 세션과 merged bin 을 모두 지원."""
    from sats.bending.deform_data import discover_sessions, read_due_raw
    try:
        for d in discover_sessions(root):
            t, raw = read_due_raw(d)
            fs = len(t) / max(float(t[-1] - t[0]), 1e-9)
            yield d.name, raw, fs
        return
    except FileNotFoundError:
        pass
    from sats.preprocessing.merged_bin import open_merged_bin
    bins = sorted(root.rglob("*_merged.bin"))
    if not bins:
        raise FileNotFoundError(f"deform 세션도 merged bin 도 없음: {root}")
    for b in bins:
        _, arr = open_merged_bin(b)
        raw = np.stack([np.asarray(arr[f"s{i}"], float) for i in range(1, 17)], 1)
        yield b.stem, raw, 200.0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    dead_union: set[int] = set()
    for name, raw, fs in _iter_inputs(root):
        rep = analyze_channels(raw, fs=fs)
        print(f"[{name}] {rep.summary()}")
        dead_union |= set(dead_indices(rep))
    if dead_union:
        names = ",".join(f"S{i + 1:02d}" for i in sorted(dead_union))
        print(f"\n★마스킹 대상(합집합): {names}")
        print(f"  학습·추론 양쪽에 동일하게 적용할 것 — "
              f"--dead-channels {','.join(str(i + 1) for i in sorted(dead_union))}")
    else:
        print("\n전 세션 16채널 정상.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
