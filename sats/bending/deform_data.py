"""각도-프리 변형 세션 로더 — due bin 만으로 학습 데이터 구성 (Phase 1).

기존 `prepare_bending_data.py` 는 ethermotion δ → 원호 기하 → 각도 라벨에 의존한다.
변형 복원(latent restorer)은 각도를 쓰지 않으므로 **due 16채널만** 있으면 된다.

취득 프로토콜(세션 1개, Enter 로 진행):
    [앞 baseline] 무접촉·무하중 ~10초        → Enter
    [변형 구간]   손으로 순수 변형(가변 길이)  → Enter
    [뒤 baseline] 무접촉·무하중 10초          → 자동 종료
★단계 길이가 가변이므로 실제 경계는 session_meta.json 의 `stage_times_s` 를 신뢰한다
(없으면 baseline_sec 고정값으로 폴백 — 구 데이터 호환).

이 로더가 하는 일:
  1. due bin(V1 `due_raw_burst_*` / V2 `due_v2_*`) 자동 감지·로드
  2. 앞/뒤 baseline 구간 검출 → **선형 드리프트 보정**(열·기압 표류 제거)
  3. pct(상대변화%) 변환 + 변형 구간만 윈도우로 잘라 반환

★드리프트 보정이 핵심: 앞뒤 두 무접촉 구간의 채널 평균을 잇는 직선을 시간축 baseline 으로
써서, 세션 내내 서서히 변하는 기압/온도 성분을 제거한다(각도 방식의 재영점 대체).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_BASELINE_SEC = 10.0          # 프로토콜상 앞뒤 무접촉 길이
_EDGE_TRIM_SEC = 1.0          # baseline 구간 가장자리 트림(진입/이탈 과도 제거)


@dataclass(frozen=True)
class DeformSession:
    """한 변형 세션. sensor_pct = 드리프트 보정된 상대변화%[N,16]."""
    name: str
    sensor_pct: np.ndarray        # [N, 16] float32
    t: np.ndarray                 # [N] 초
    baseline_head: np.ndarray     # [16] 앞 baseline 채널 평균(raw)
    baseline_tail: np.ndarray     # [16] 뒤 baseline 채널 평균(raw)
    flags_s: tuple = ()           # (하위호환) 구간 마커 시각(초)
    segments: tuple = ()          # ({"t_s","label"}, ...) 라벨 있는 변형 구간 마커
    stage_times: dict | None = None   # 프로토콜 단계 전환 시각 {"BASE_HEAD":0.0, "DEFORM":10.0, ...}

    @property
    def drift_pct(self) -> float:
        """세션 동안의 baseline 표류 크기(%) — 취득 품질 점검용."""
        d = (self.baseline_tail - self.baseline_head) / np.maximum(np.abs(self.baseline_head), 1e-9)
        return float(np.abs(d).mean() * 100.0)


def _read_due(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """세션 폴더에서 due bin 자동 감지 → (t[N], raw[N,16]). V1/V2 모두 지원."""
    v2 = sorted(session_dir.glob("due_v2_*.bin"))
    if v2:
        import struct
        rec = 8 + 16 * 10 * 4
        ts, vals = [], []
        with open(v2[0], "rb") as f:
            f.readline()                              # magic "DUE_V2"
            while True:
                d = f.read(rec)
                if len(d) < rec:
                    break
                ns = struct.unpack("<Q", d[:8])[0]
                vals.append(np.frombuffer(d[8:], dtype="<u4").reshape(16, 10).T)
                ts.append(ns / 1e9)
        if not ts:
            raise ValueError(f"DUE_V2 레코드 없음: {v2[0]}")
        t_burst = np.asarray(ts)
        raw = np.concatenate(vals).astype(np.float64)
        t = np.repeat(t_burst, 10) + np.tile(np.arange(10) * 0.005, len(t_burst))
        return t, raw
    v1 = sorted(session_dir.glob("due_raw_burst_*.bin"))
    if not v1:
        raise FileNotFoundError(f"due bin 없음(V2·V1 모두): {session_dir}")
    from sats.preprocessing.bin_merge import load_due_bin
    d = load_due_bin(v1[0])
    return np.asarray(d.time_s, float), np.asarray(d.sensors, float)


def read_due_raw(session_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """세션 폴더 → (t[N], raw[N,16]). 드리프트 보정 전 원신호(채널 건강도 진단용)."""
    return _read_due(Path(session_dir))


def stage_bounds(session: DeformSession, baseline_sec: float = _BASELINE_SEC
                 ) -> tuple[float, float]:
    """(변형 시작, 변형 끝) 초. stage_times_s 우선, 없으면 고정 baseline_sec 폴백."""
    dur = float(session.t[-1])
    st = session.stage_times or {}
    lo = float(st.get("DEFORM", baseline_sec))
    hi = float(st.get("BASE_TAIL", dur - baseline_sec))
    if not (0 <= lo < hi <= dur):                    # 메타 이상 시 안전 폴백
        lo, hi = baseline_sec, dur - baseline_sec
    return lo, hi


def load_deform_session(session_dir: str | Path, *, baseline_sec: float = _BASELINE_SEC,
                        trim_sec: float = _EDGE_TRIM_SEC,
                        dead_channels: Sequence[int] = ()) -> DeformSession:
    """변형 세션 → 드리프트 보정된 pct. 앞뒤 baseline 은 프로토콜 길이로 잘라 쓴다.

    dead_channels(0-based): 파손된 taxel. **pct 를 0 으로 고정**한다 — 제거가 아니라
    0 고정인 이유는 SATS 가 16채널 입력을 요구하기 때문이고, 0 = "baseline 대비 변화 없음"
    이라 복원기·SATS 모두에게 "이 taxel 에는 아무 일도 없다"로 일관되게 읽힌다.
    ★고장 채널을 그대로 두면 앞/뒤 baseline 차이로 선형 드리프트 보정이 가짜 램프를
    주입하므로, 마스킹은 pct 계산 직후에 적용해 그 램프까지 함께 지운다.
    """
    session_dir = Path(session_dir)
    t, raw = _read_due(session_dir)
    t = t - t[0]
    dur = float(t[-1])
    if dur < 2 * baseline_sec + 5.0:
        raise ValueError(f"세션이 너무 짧음({dur:.0f}s): baseline {baseline_sec}s×2 + 변형 필요")
    # 메타의 실제 단계 경계 우선(가변 길이 프로토콜) — 없으면 고정 baseline_sec
    meta_pre = session_dir / "session_meta.json"
    st_pre: dict = {}
    if meta_pre.exists():
        try:
            import json as _json
            st_pre = _json.loads(meta_pre.read_text(encoding="utf-8")).get("stage_times_s") or {}
        except Exception:
            st_pre = {}
    head_end = float(st_pre.get("DEFORM", baseline_sec))
    tail_start = float(st_pre.get("BASE_TAIL", dur - baseline_sec))
    if not (0 < head_end < tail_start <= dur):
        head_end, tail_start = baseline_sec, dur - baseline_sec
    head_m = (t >= trim_sec) & (t <= head_end - trim_sec)
    tail_m = (t >= tail_start + trim_sec) & (t <= dur - trim_sec)
    if head_m.sum() < 50 or tail_m.sum() < 50:
        raise ValueError(f"baseline 구간 프레임 부족(head {head_m.sum()}, tail {tail_m.sum()})")
    b_head, b_tail = raw[head_m].mean(0), raw[tail_m].mean(0)
    # 세션 메타(로거 저장)에서 Enter flag 로드 — 없으면 빈 튜플
    meta_p = session_dir / "session_meta.json"
    flags: tuple = (); segments: tuple = (); stage_times = None
    if meta_p.exists():
        try:
            import json
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            flags = tuple(meta.get("flags_s", []))
            segments = tuple(meta.get("segments", []))
            stage_times = meta.get("stage_times_s")
        except Exception:
            pass
    # ★선형 드리프트 baseline: 앞 baseline 중앙 시각 → 뒤 baseline 중앙 시각 사이를 보간
    t0, t1 = float(t[head_m].mean()), float(t[tail_m].mean())
    w = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)[:, None]
    base_t = b_head[None, :] * (1 - w) + b_tail[None, :] * w        # [N,16]
    pct = ((raw / np.where(np.abs(base_t) < 1e-9, 1e-9, base_t) - 1.0) * 100.0).astype(np.float32)
    for ch in dead_channels:                          # 파손 taxel → 변화 없음으로 고정
        if not 0 <= int(ch) < pct.shape[1]:
            raise ValueError(f"dead_channels 범위 밖: {ch} (0~{pct.shape[1] - 1})")
        pct[:, int(ch)] = 0.0
    return DeformSession(name=session_dir.name, sensor_pct=pct, t=t,
                         baseline_head=b_head, baseline_tail=b_tail, flags_s=flags,
                         segments=segments, stage_times=stage_times)


def deform_windows(session: DeformSession, window_size: int, *,
                   baseline_sec: float = _BASELINE_SEC, stride: int = 1,
                   include_baseline: bool = False) -> np.ndarray:
    """세션 → 윈도우[K, W, 16].

    include_baseline=False(기본): **변형 구간만**(앞뒤 baseline 제외) — L_suppress 용.
    True: 앞뒤 baseline 구간만 — L_identity(변형 없으면 그대로) 용.
    """
    t, pct = session.t, session.sensor_pct
    lo, hi = stage_bounds(session, baseline_sec)      # ★실제 단계 경계(메타 우선)
    if include_baseline:
        m = (t <= lo) | (t >= hi)
    else:
        m = (t > lo) & (t < hi)
    idx = np.where(m)[0]
    if len(idx) < window_size:
        return np.zeros((0, window_size, pct.shape[1]), np.float32)
    # 연속 구간별로 윈도우 생성(경계를 넘는 윈도우 방지)
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    out = []
    for seg in splits:
        if len(seg) < window_size:
            continue
        starts = np.arange(0, len(seg) - window_size + 1, max(1, stride))
        out.append(np.stack([pct[seg[s:s + window_size]] for s in starts]))
    return np.concatenate(out).astype(np.float32) if out else \
        np.zeros((0, window_size, pct.shape[1]), np.float32)


def segment_windows(session: DeformSession, window_size: int, *,
                    baseline_sec: float = _BASELINE_SEC, stride: int = 1
                    ) -> dict[str, np.ndarray]:
    """라벨별 윈도우 {label: [K,W,16]} — "비틀림 구간만 억제율" 식 구간별 분석용.

    segments 의 각 마커 시각부터 다음 마커(또는 변형 구간 끝)까지를 그 라벨의 구간으로 본다.
    마커 이전(변형 시작~첫 마커)은 "(unlabeled)" 로 묶는다.
    """
    t = session.t
    lo, hi = stage_bounds(session, baseline_sec)
    marks = [(float(g["t_s"]), str(g.get("label") or "(unlabeled)"))
             for g in session.segments if lo <= float(g["t_s"]) <= hi]
    marks.sort()
    bounds = [(lo, "(unlabeled)")] + marks + [(hi, None)]
    out: dict[str, list] = {}
    for (t0, label), (t1, _) in zip(bounds[:-1], bounds[1:]):
        if label is None or t1 - t0 <= 0:
            continue
        m = (t > t0) & (t < t1)
        idx = np.where(m)[0]
        if len(idx) < window_size:
            continue
        seg = np.stack([session.sensor_pct[idx[s:s + window_size]]
                        for s in range(0, len(idx) - window_size + 1, max(1, stride))])
        out.setdefault(label, []).append(seg.astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def discover_sessions(root: str | Path) -> list[Path]:
    """root 아래에서 due bin 을 가진 세션 폴더를 찾는다(root 자신도 후보)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"경로 없음: {root}")
    found = []
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if any(d.glob("due_v2_*.bin")) or any(d.glob("due_raw_burst_*.bin")):
            found.append(d)
    return found


def load_all(root: str | Path, **kw) -> list[DeformSession]:
    """root 아래 모든 변형 세션 로드(품질 요약 출력). kw 는 load_deform_session 으로 전달."""
    sessions = []
    for d in discover_sessions(root):
        try:
            s = load_deform_session(d, **kw)
        except Exception as e:                       # 세션 하나 실패가 전체를 막지 않게
            print(f"  [skip] {d.name}: {e}")
            continue
        sessions.append(s)
        span = float(np.abs(s.sensor_pct).max())
        print(f"  [{s.name}] {len(s.sensor_pct):6d} frames  {s.t[-1]:5.0f}s  "
              f"drift {s.drift_pct:5.2f}%  |pct|max {span:5.1f}%  segments {len(s.segments)}")
    if not sessions:
        raise FileNotFoundError(f"유효한 변형 세션 없음: {root}")
    return sessions
