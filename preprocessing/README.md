# 센서 데이터 전처리 (Preprocessing)

이 디렉토리는 촉각 센서의 원시 데이터를 분석 및 기계 학습(CNN, MLP 등)에 적합한 형태로 가공하는 전처리 파이프라인을 포함합니다.

## 1. 데이터 구조 및 명명 규칙

### 입력 데이터 (`raw_data/`)
실험 데이터는 다음의 명명 규칙을 따르는 폴더에 저장됩니다:
- **형식**: `{소재}_{인덴터지름}_{시행번호}` (예: `ecomesh_d5_1`)
- `manifest.csv`(선택): `trial_id, material, diameter_mm, z_plan_mm, pause_sec, notes` 메타 기록

### 출력 데이터 구조 (`processed_data/`)
가공된 데이터는 효율적인 관리를 위해 다음과 같이 폴더별로 구분되어 저장됩니다:

- **`baselines/`**: 각 시행별 초기 무부하 상태 평균 (`.json`)
- **`grid/`**: 0.5 mm 격자 필터링 + `z_depth` 포함 원시 ADC (`.csv`)
- **`features/`**: 정규화 특징 및 메타 (`.csv`, `contact_radius_mm`, `contact_radius_cell` 포함 가능)
- **`zarr_data/`**: GPU 학습 최적화 포맷 (`dataset_{소재}.zarr`, `dataset_index.json`)
- **`label_preview/`**: 깊이 기반 라벨(히트맵) 시각화 PNG (옵션)

## 2. 전처리 파이프라인

전처리는 크게 두 단계로 진행됩니다.

### 단계 1: 데이터 병합 (`raw_merge.py`)
각 장치에서 독립적으로 수집된 데이터를 타임스탬프 기준으로 정렬하고 병합합니다.
- **입력**: `due_data` (센서 s1~s16), `ethermotion_data` (X, Y, Z 좌표), `afd50_data` (Fx, Fy, Fz 힘)
- **주요 기능**: 타임스탬프 동기화, 시행별 통합 CSV 생성

### 단계 2: 특징량 추출 및 GPU 최적화 (`preprocess.py`)
병합된 데이터를 학습 목적에 맞게 가공하고 고속 로딩 포맷으로 변환합니다.
- Baseline correction, grid filtering(0.5 mm), z_depth 재계산
- **Depth-aware 접촉 반경 옵션**: `--use-depth-aware-radius` 사용 시
  - 반경 모델: `--radius-model {hertz,geo}` (기본 hertz, R=2.5 mm)
  - 반경 상한: `--max-radius-mm`
  - 폴백 규칙: `--fallback-depth-mode {none,mean,const}` + `--fallback-depth-mm`
  - 산출 컬럼: `contact_radius_mm`, `contact_radius_cell`
  - Zarr `aux_last_field=contact_radius_mm`로 저장
- Zarr 변환: PyTorch Dataset 고속 로딩
- (옵션) 라벨 프리뷰: `--export-label-heatmap`, `--label-samples N`, `--label-kernel`, `--sigma-scale`

## 3. 학습 태스크 정의

### 1) Super-resolution (SR)
- **Input**: `s_norm_1` ~ `16` (+ 선택적으로 `contact_radius_mm/cell`)
- **Output**: `x`, `y`, `z_depth`

### 2) Force Field
- **Input**: `s_norm_1` ~ `16`, `x`, `y`, `z_depth`
- **Output**: `fz_bc` (정제된 수직력)

---
**실행 방법**:
```bash
# (선택) 병합
python3 preprocessing/raw_merge.py

# 전처리 + 깊이 기반 반경/라벨 옵션
python3 preprocessing/preprocess.py \
  --raw-dir preprocessing/raw_data \
  --out-dir preprocessing/processed_data \
  --use-depth-aware-radius \
  --radius-model hertz \
  --indenter-radius-mm 2.5 \
  --max-radius-mm 2.0 \
  --fallback-depth-mode none \
  --label-kernel gaussian --sigma-scale 1.0 \
  --export-label-heatmap --label-samples 3

# 라벨 프리뷰만 별도 실행 가능
python3 preprocessing/label_preview.py \
  --grid-file preprocessing/processed_data/grid/ecemesh_d5_1_grid.csv \
  --samples 3 --kernel gaussian --sigma-scale 1.0
```
