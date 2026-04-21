# Preprocessing

센서 압입 실험의 raw 데이터로부터 SATS 학습용 데이터셋을 만드는 두 파이프라인.

---

## 파일 구조

```
raw_data/
└── <material>/              # 소재명 (예: ecomesh)
    └── d<D>/                # 인덴터 직경 (예: d5, d10)
        └── z_<Z>mm/         # 최대 압입 깊이 (예: z_1.0mm, z_1.5mm)
            └── test<N>/     # 반복 실험 번호
                └── *_merged.csv   ← 입력 파일

sats/preprocessing/
├── preprocess.py      # 센서 신호 전처리 (baseline 보정, 정규화, Zarr 저장)
└── generate_gt.py     # Boussinesq 물리 모델 기반 GT 압력맵 생성
```

`*_merged.csv` 컬럼: `timestep_sec, s1–s16, x_mm, y_mm, z_mm, Fx, Fy, Fz, ...`
(raw_data/raw_merge.py 가 3개 스트림—due/ethermotion/afd—을 타임스탬프 기준으로 병합)

---

## 1. preprocess.py — 센서 신호 전처리

### 목적

raw ADC 센서 신호를 baseline 보정·정규화해 학습 입력 피처와 Zarr 데이터셋으로 저장한다.

### 실행

```bash
python3 sats/preprocessing/preprocess.py \
    --raw-dir raw_data \
    --out-dir sats/preprocessing/processed_data
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--z-bin-mm` | 0.02 | z_depth 빈 간격 (0 이하면 비활성) |
| `--min-signal` | auto | 저신호 샘플 제거 임계값. 미지정 시 trial별 자동 추정 |
| `--min-reliable-s` | 0.005 | 일관성 필터: 모든 trial에서 이 값 미만인 그리드 좌표 제거 |
| `--workers` | 1 | 병렬 처리 프로세스 수 |
| `--use-depth-aware-radius` | off | Hertz/기하 모델로 접촉 반경 계산 |

### 처리 단계

```
merged CSV
    │
    ├─ 1. Baseline 추출
    │       z_mm ≈ 0 & Fz 작음 & 정지 상태인 연속 구간을 baseline으로 지정.
    │       구간이 없으면 파일 앞 x=y=z=0 구간으로 폴백.
    │       구간별 s1–s16 평균을 JSON으로 저장.
    │
    ├─ 2. 그리드 행 필터링 (filter_grid_rows)
    │       x_mm, y_mm 가 0.5 mm 그리드 위에 있고
    │       이전/다음 샘플과 좌표 변화 < 0.001 mm (정지 상태)인 행만 선택.
    │
    ├─ 3. z_depth 계산 (compute_z_depth)
    │       z_stage_mm : 원본 z_mm (음수 제거)
    │       z_contact_mm : 각 (x,y)에서 최초 접촉 시점의 z를 0으로 정렬한 값
    │
    ├─ 4. Phase 분류 (assign_phase)
    │       각 (x,y) 그룹에서 z_mm 피크 이전=loading(0), 이후=unloading(1)
    │
    ├─ 5. Grid CSV 저장  → {trial_id}_grid.csv
    │
    ├─ 6. 정규화 피처 생성 (make_features_df)
    │       s_norm_i = (s_i − baseline_i) / baseline_i  (구간별 baseline 사용)
    │       z_bin_mm 간격으로 빈 집계 (같은 위치·깊이 → 중앙값/평균)
    │       min_signal 미만 행 제거
    │       → {trial_id}_features.csv
    │
    └─ 7. 소재별 통합 → 일관성 필터 → Zarr 저장
            모든 trial에서 min_reliable_s 미만 신호가 있는 그리드 좌표 제거.
            최종 → {material}_features.csv + dataset_{material}.zarr
```

### 출력

```
processed_data/
├── baselines/   {trial_id}_baselines.json
├── grid/        {trial_id}_grid.csv
├── features/    {trial_id}_features.csv
├── peak_summary/ {material}_peak_summary.csv
├── zarr_data/   dataset_{material}.zarr/
└── {material}_features.csv   (소재별 통합)
```

---

## 2. generate_gt.py — Boussinesq GT 압력맵 생성

### 목적

각 row의 `(x_mm, y_mm, Fz, diameter)` 로부터 40×40 수직응력 맵(GT)을 계산해 npy로 저장한다.
GT는 SATS 모델의 학습 타겟으로 쓰인다.

### 실행

```bash
python3 sats/preprocessing/generate_gt.py \
    --raw-dir raw_data \
    --out-dir sats/preprocessing/gt_output \
    --z-s 2.0 \
    --patch-step 0.1
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--z-s` | 2.0 | 센서 유효 깊이 z_s [mm] |
| `--patch-step` | 0.1 | 인덴터 패치 이산화 간격 [mm] |
| `--grid-size` | 40 | GT 맵 한 변 크기 |

### GT 정의

한 row = 한 장의 40×40 수직응력 맵.

```
GT(u, v) = σ_zz(u, v, z_s | X, Y, Fz, d)
```

- `(u, v)`: 40×40 GT 셀 중심 좌표 (−9.75, −9.25, …, 9.75 mm, 0.5 mm 간격)
- `z_s`: 센서 유효 깊이 [mm] (권장 범위: 1.5 – 3.5 mm)
- `(X, Y)`: 인덴터 접촉 중심
- `Fz`: 총 수직 하중 [N]
- `d`: 인덴터 직경 [mm]

### 물리 모델 (Boussinesq 탄성 반공간)

**Step 1. 균일 원형 압력 분포**

접촉 반경 `a = d/2`, 접촉 면적 `A = πa²`.
패치 내부 압력은 균일하다고 가정:

```
p₀ = Fz / (π · a²)
```

**Step 2. 패치 이산화**

원형 패치를 `patch_step` 간격 격자로 분할.
각 점 `(xᵢ, yᵢ)` 의 하중:

```
Fᵢ = p₀ · patch_step²  =  Fz · patch_step² / (π · a²)
```

대칭 격자를 사용해 부동소수점 오차로 인한 비대칭을 방지한다
(`n_half = floor(a / step)`, offsets = `[-n_half, …, +n_half] × step`).

**Step 3. Boussinesq 수직응력 합산**

GT 셀 `(u, v)` 에서의 수직응력:

```
σ_zz(u, v) = Σᵢ  3·Fᵢ·z_s³ / (2π · Rᵢ⁵)

Rᵢ = sqrt((u − xᵢ)² + (v − yᵢ)² + z_s²)
```

정리하면 prefactor `C = 3·z_s³·patch_step² / (2π²·a²)` 를 빼고:

```
σ_zz(u, v) = C · Fz · Σᵢ  1/Rᵢ⁵
```

### 선형성과 단위 커널

`σ_zz ∝ Fz` (선형)이므로 **단위 커널** `K(u,v; cx,cy)`를 `Fz=1` 로 한 번만 계산한 뒤 실제 Fz를 곱하면 된다:

```
GT = K(u, v; X, Y) × Fz
```

### base kernel 최적화

커널이 순수 평행이동 불변성을 갖는다:

```
K(u, v; cx, cy) = K₀(u − cx, v − cy)
```

이를 이용해 직경당 계산량을 크게 줄인다.

**1. 79×79 base kernel 계산 (직경당 1회)**

`du, dv ∈ [−19.5, −19.0, …, 19.0, 19.5]` (79점, 0.5 mm 간격) 위에서
원점 기준 단위 커널 `K₀(du, dv)` 를 계산한다.

**2. (40, 40, 40, 40) lookup table 구축**

접촉 위치 `(cx, cy)` 가 40×40 그리드의 인덱스 `(i, j)` 로 스냅되면:

```
all_kernels[i, j] = base_kernel[39−i : 79−i, 39−j : 79−j]
```

메모리: 40⁴ × 4 bytes = **10.2 MB**

**3. 벡터화 GT 조립**

```python
# 각 row의 접촉 위치를 nearest grid index로 스냅
i_cx = argmin(|grid_x − x_mm|)
j_cy = argmin(|grid_y − y_mm|)

# 청크 단위 벡터화
targets[rows] = all_kernels[i_cx[rows], j_cy[rows]] × Fz[rows, None, None]
```

| 방법 | d=10 mm (1607 위치) | 전체 16 trial |
|---|---|---|
| 기존 per-position | ~354 s | ~95 분 (추정) |
| base kernel 최적화 | ~0.8 s + 조립 ~5 s | **~3 분** |
| 속도 향상 | **57× ** | — |

### 출력

```
gt_output/
├── {trial_id}_targets.npy     (N, 40, 40) float32
├── {trial_id}_gt_meta.json    trial별 메타데이터
└── dataset_index.json         전체 인덱스
```

`dataset_index.json` 항목:

```json
{
  "z_s_mm": 2.0,
  "patch_step_mm": 0.1,
  "grid_size": 40,
  "total_trials": 16,
  "total_rows": 10451288,
  "trials": [
    {
      "trial_id": "ecomesh_d10_z1_test1",
      "diameter_mm": 10.0,
      "n_total_rows": 863841,
      "n_positive_fz": 155869,
      "gt_shape": [863841, 40, 40],
      ...
    },
    ...
  ]
}
```

### z_s 튜닝 방법

`z_s` 는 물리적으로는 센서 유효 깊이(폴리머 두께 + 다이어프램 위치)이며,
최적값은 **GT의 16개 센서 위치 응력값과 실제 s1–s16 패턴의 상관성**이 최대가 되는 값으로 선택한다.

```bash
# z_s sweep 예시
for ZS in 1.5 2.0 2.5 3.0 3.5; do
    python3 sats/preprocessing/generate_gt.py --z-s $ZS --out-dir gt_zs_${ZS}
done
```

센서 좌표 (±3.25 mm, ±9.75 mm) 4×4 격자에서 GT 값과 실제 센서값의 Pearson 상관 계수가 높은 `z_s` 를 선택.

---

## 센서 좌표 참고

| 센서 | x [mm] | y [mm] |
|---|---|---|
| S1 | +9.75 | −9.75 |
| S2 | +3.25 | −9.75 |
| S3 | −3.25 | −9.75 |
| S4 | −9.75 | −9.75 |
| S5 | +9.75 | −3.25 |
| … | … | … |
| S13 | +9.75 | +9.75 |
| S14 | +3.25 | +9.75 |
| S15 | −3.25 | +9.75 |
| S16 | −9.75 | +9.75 |

16개 센서가 ±3.25/±9.75 mm 의 4×4 격자에 배치. 간격 6.5 mm.
