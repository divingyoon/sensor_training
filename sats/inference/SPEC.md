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
| Super-resolution | taxel pitch **~6.5 mm → 0.5 mm map** (≈13× linear, 105× cells) | 16 → 1681 |
| Window size | 10 frames | config `window_size` |
| Size conditioning | ON, fixed **d = 10 mm** | `use_indenter_size_input` |
| Inference throughput | **129 Hz** (7.75 ms/frame, RTX GPU) | measured@4090 |
| Demo output rate | cap **20 Hz** (`--infer-max-fps`, 상향 가능) | run_demo |

## 3. Force / position ranges & resolution (size-conditioned, D5 & D10)

**★ 위치 분해능 — 모터 실측 확정(2026-08-08)**: 0.1μm 스테이지로 0.1 mm 스텝 31점을
랜덤 순서·3회 반복 압입(D10, 깊이 1.5 mm)해 측정(`scripts/analyze_localization_resolution.py
--v2-dir`, 데이터 `skin_ws/raw_data/v6_finestep_2`, 모델 `ecomesh_v6_deploy_g025`).

| 측정 항목 | 실측값 | 비고 |
|---|---|---|
| 반복 노이즈 σ | **0.028 mm** | 같은 점 반복 예측의 흔들림 |
| 0.1 mm 스텝 구분 성공률 | **40 %** | 인접 위치가 2σ를 넘어 구분되는 비율 |
| **실효 분해능 (플래토 폭)** | **median 0.20 mm, worst 0.50 mm** | ★ 실제로 분간 가능한 최소 간격 |
| 추종 gain | **1.80** | 모터 1 mm 이동 → 예측 1.8 mm (과대응답, §3c 보정) |
| 직선 잔차 RMS | 0.48 mm | 계통 비선형 |

- **σ(0.028 mm)를 분해능으로 읽으면 안 된다**: 응답이 계단형이라 플래토 안에서 안정적일 뿐
  위치를 구분하지 못한다. 정직한 스펙은 **실효 분해능 0.2 mm**.
- 계단의 원인은 **학습 라벨 간격**(v6 = xy **1 mm** 격자)으로 추정. 출력 격자를 0.25 mm로
  낮춰도(81²) 라벨 사이는 보간이라 계단이 남는다. warm-start(xy0.5 가중치)는 초기값일 뿐
  fine-tuning으로 새 라벨 분포에 적응하므로 0.5 mm 해상 능력이 보존되지 않는다.
- 그래도 **taxel pitch ~6.5 mm → 0.2 mm 분간 = 약 32× super-resolution**.
- 측정 조건 한계: **x축 ±1.5 mm 구간·D10·깊이 1.5 mm**. 전면(±10 mm) 지도·y축은 미측정.

| Quantity | Range | Resolution | Source |
|---|---|---|---|
| **x, y** | −10 … +10 mm | **실효 0.20 mm** (σ 0.028 mm, grid 0.25/0.5 mm) | motor sweep measured |
| **z (D10)** | **0.48 … 2.0 mm** | LUT (peak→z), r² = 0.59 (coarse) | z_calibration_v6.json `"10.0"` |
| **z (D5)** | 0.72 … 2.0 mm | LUT, r² = 0.87 (더 신뢰) | z_calibration_v6.json `"5.0"` |
| **Fz (D10)** | **0 … 3.9 N** | 무접촉 노이즈 플로어 **p99 0.003 N** (사실상 0) | measured@4090 |
| **Fz (D5)** | 0 … 1.5 N | 노이즈 플로어 measured@4090 | GT-integral upper |

- **Fz 노이즈가 사실상 0**(p99 0.003 N) → 필터 임계는 *노이즈*가 아니라 *의미있는 접촉*
  기준(가벼운 손가락 ~0.2 N)으로 잡음: 기본 `fz_on 0.20 / fz_off 0.10`.
- **다중접촉(2·3)은 D5 권장**: D10 은 min-distance 10 mm 라 가까운 접촉이 병합됨(`[d]` 토글).
- Fz = Σ(clip≥0 pred) × taxel_area / 100 [N] (해상도 불변 적분, `inference_engine.get_fz`).

## 3c. Position gain correction (과대응답 보정)

실측 gain **1.80** = 접촉이 실제보다 중심에서 멀게 추정됨. 역보정:
`s = c + (p − c)/gain` (`demo_contacts.apply_position_gain`, 중심 c=(0,0)).

- 활성화: `--position-gain 1.8` (run_demo / run_dashboard). **기본 1.0 = off.**
- ⚠ 기본 off 인 이유: "중심에서는 정확하다"는 가정이 **x축 ±1.5 mm 한 구간에서만**
  검증됨. 전면 2D 격자로 gain map을 측정하기 전에는 전역 적용이 오히려 왜곡을 만들 수 있다.
- 보정해도 **분해능(0.2 mm)은 개선되지 않는다** — gain 은 계통 스케일 오차만 줄인다.

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
