# SATS v6 Real-time Demo — Sensor & Inference Spec (D10)

전시/학회 데모용 스펙 시트. **모든 수치는 지름 10 mm 인덴터(D10) 기준.**
표는 포스터에 그대로 쓸 수 있게 영문 라벨로 정리. 주석은 한글.

> ⚠ `[measure@4090]` 표시 항목은 gitignore된 eval 산출물·GPU가 4090에만 있어
> `scripts/measure_demo_spec.py` 를 4090에서 실행해 채운다(추정치 날조 금지).

---

## 1. Raw sensing unit (per sensor)

| Item | Value | Source |
|---|---|---|
| Taxels | 16 (4×4) | vensor2.ino |
| Sampling rate | **200 Hz** | firmware |
| MCU link | Arduino **Due**, binary `0xAA … 0x55` | serial_reader |
| Baud rate | **250000** | vensor2.ino `Serial.begin(250000)` |
| Transducer | BMP384 absolute pressure | — |

## 2. SATS super-resolution output (D10)

| Item | Value | Source |
|---|---|---|
| Output map | **41 × 41** | deploy config `grid_size` |
| Grid pitch | **0.5 mm** | config `grid_step_mm` |
| Sensing area | **20 × 20 mm** (−10 … +10 mm) | config `grid_min/max_mm` |
| Super-resolution | 4×4 taxels → 41×41 (≈ **105× cells**) | 16 → 1681 |
| Window size | 10 frames | config `window_size` |
| Size conditioning | ON, fixed **d = 10 mm** | `use_indenter_size_input` |
| Inference throughput | **129 Hz** (7.75 ms/frame, RTX GPU) | measured@4090 |
| Demo output rate | cap **20 Hz** (`--infer-max-fps`, 상향 가능) | run_demo |

## 3. Force / position ranges & resolution (size-conditioned, D5 & D10)

| Quantity | Range | Resolution (localization err) | Source |
|---|---|---|---|
| **x, y (D5)** | −10 … +10 mm | **≈ 0.5 mm** (1 grid cell) | map_quality reeval (size model) |
| **x, y (D10)** | −10 … +10 mm | **≈ 1.0 mm** (2 grid cells) | map_quality reeval (size model) |
| **z (D10)** | **0.48 … 2.0 mm** | LUT (peak→z), r² = 0.59 (coarse) | z_calibration_v6.json `"10.0"` |
| **z (D5)** | 0.72 … 2.0 mm | LUT, r² = 0.87 (더 신뢰) | z_calibration_v6.json `"5.0"` |
| **Fz (D10)** | **0 … 3.9 N** | noise-floor gate `[measure@4090]` | GT-integral upper (memory) |
| **Fz (D5)** | 0 … 1.5 N | noise-floor gate `[measure@4090]` | GT-integral upper (memory) |

- localization = argmax(pred) vs GT 거리 median. grid pitch 0.5 mm 가 이론 하한.
- **다중접촉(2·3)은 D5 권장**: D10 은 min-distance 10 mm 라 가까운 접촉이 병합됨(대시보드 `[d]` 토글).
- **Fz 분해능 = 무접촉 노이즈 플로어**(필터 임계 `fz_on`/`fz_off` 근거) → `scripts/measure_resolution.py`(4090) 실측.
- Fz = Σ(clip≥0 pred) × taxel_area / 100 [N] (해상도 불변 적분, `inference_engine.get_fz`).

## 3b. Visualization filtering (contact_filter.py)

무접촉/접촉 안정화 — 임계는 위 분해능·노이즈 플로어로 튜닝:
- **히스테리시스**: 총 fz ≥ `fz_on`(0.30 N) → ON, < `fz_off`(0.15 N) → OFF (깜빡임 방지).
- **디바운스**: `on_frames`(2)/`off_frames`(5) 연속 만족 시 전환.
- **무접촉 리셋**: OFF 확정 시 표시 blank + 트랙 초기화(release).
- **위치 스무딩**: `pos_smooth`(3) median — 분해능 한계(≈0.5–1 mm) 내 떨림 제거.

## 4. Bending angle (θ) — jig curvature (D10)

| Item | Value | Source |
|---|---|---|
| Observable range | **~20 … 150°** (유효 관측창) | memory bending-observability |
| Accuracy (G1 leave-one-out) | **MAE 1.78°**, ρ ≈ 1.0 | memory v6 G1 |
| Per-frame resolution | std **~0.4 … 1.0°** | memory theta resolution |
| Dead-band | |θ| < 20° → 0 (관측 불가 → flat) | run_demo `--theta-deadband` |

## 5. Multi-sensor demo (3 sensors, one desktop)

- 통합 대시보드 1창(`run_dashboard.py`), 3열 패널: **contacts | theta | bending→SATS**.
- SATS 엔진 1개 공유(predict는 상태무관) → GPU 1회 로드로 3센서 처리.
- 포트 고정: **`/dev/serial/by-id/*`** symlink 사용(ttyACM 번호 재연결마다 뒤바뀜 방지).
- 부하: 소형 CNN 3-추론 + 3 × 200 Hz × 16 taxel @ 250000 baud(각 USB) → 여유.

---

_생성: `sats/inference/SPEC.md`. 수치 갱신은 `scripts/measure_demo_spec.py`(4090) 재실행._
