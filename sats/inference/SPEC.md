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
| Inference output rate | cap **20 Hz** (`--infer-max-fps`); GPU-bound higher | `[measure@4090]` |

## 3. Force / position ranges & resolution (D10)

| Quantity | Range | Resolution | Source |
|---|---|---|---|
| **x** | −10 … +10 mm | 0.5 mm grid pitch; localization err ≈ 1.4 mm | `[measure@4090]` deploy eval |
| **y** | −10 … +10 mm | 0.5 mm grid pitch; localization err ≈ 1.4 mm | `[measure@4090]` deploy eval |
| **z** (indent depth) | **0.48 … 2.0 mm** | LUT (peak→z), r² = 0.59 | z_calibration_v6.json `"10.0"` |
| **Fz** (normal force) | **0 … 3.9 N** | `[measure@4090]` | GT-integral upper (memory) |

- localization err ≈ 1.4 mm 는 홀드아웃 전이 실험(v3, d10) 값 — **deploy 모델 자체 eval로 확정 예정**.
- Fz = Σ(clip≥0 pred) × taxel_area / 100 [N] (해상도 불변 적분, `inference_engine.get_fz`).

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
