# 센서 데이터 전처리 (Preprocessing)

이 디렉토리는 촉각 센서의 원시 데이터를 분석 및 기계 학습(CNN, MLP 등)에 적합한 형태로 가공하는 전처리 파이프라인을 포함합니다.

## 1. 데이터 구조 및 명명 규칙

### 입력 데이터 (`raw_data/`)
실험 데이터는 다음의 명명 규칙을 따르는 폴더에 저장됩니다:
- **형식**: `{소재}_{인덴터반경}_{시행번호}`
- **예시**: `ecomesh_d5_1` (Ecomesh 소재, 5mm 반경 인덴터, 1번 시행)

### 출력 데이터 구조 (`processed_data/`)
가공된 데이터는 효율적인 관리를 위해 다음과 같이 폴더별로 구분되어 저장됩니다:

- **`baselines/`**: 각 시행별 초기 무부하 상태의 센서 및 힘 센서 평균값 (`.json`)
- **`grid/`**: 0.5mm 격자점 필터링 및 `z_depth`가 계산된 원시 ADC 데이터 (`.csv`)
- **`features/`**: 각 시행별 정규화된 특징량 데이터 (`.csv`)
- **`zarr_data/`**: GPU 학습에 최적화된 고속 로딩 포맷 (`.zarr`)
  - `dataset_{소재}.zarr`: 고성능 학습용 데이터셋
  - `dataset_index.json`: PyTorch `SkinDataset`에서 샘플을 참조하기 위한 통합 인덱스 파일

## 2. 전처리 파이프라인

전처리는 크게 두 단계로 진행됩니다.

### 단계 1: 데이터 병합 (`raw_merge.py`)
각 장치에서 독립적으로 수집된 데이터를 타임스탬프 기준으로 정렬하고 병합합니다.
- **입력**: `due_data` (센서 s1~s16), `ethermotion_data` (X, Y, Z 좌표), `afd50_data` (Fx, Fy, Fz 힘)
- **주요 기능**: 타임스탬프 동기화, 시행별 통합 CSV 생성

### 단계 2: 특징량 추출 및 GPU 최적화 (`preprocess.py`)
병합된 데이터를 학습 목적에 맞게 가공하고 고속 로딩 포맷으로 변환합니다.
- **Baseline Correction**: 센서 값과 힘(Fz)에서 무부하 평균값을 제거하여 정규화
- **Grid Filtering**: 로봇이 정지한 0.5mm 격자점 데이터만 추출하여 노이즈 억제
- **Z-depth 계산**: 센서 반응이 시작되는 지점을 0으로 설정하여 압입 깊이 재계산
- **Zarr 변환**: 대용량 CSV의 로딩 병목을 해결하기 위해 PyTorch와 호환되는 Zarr 포맷으로 저장

## 3. 학습 태스크 정의

### 1) Super-resolution (SR)
- **Input**: `s_norm_1` ~ `16` (정규화된 특징량)
- **Output**: `x`, `y`, `z_depth`

### 2) Force Field
- **Input**: `s_norm_1` ~ `16`, `x`, `y`, `z_depth`
- **Output**: `fz_bc` (정제된 수직력)

---
**실행 방법**:
```bash
# 단계 1: 병합
python3 preprocessing/raw_merge.py

# 단계 2: 전처리 및 GPU용 데이터셋 생성
python3 preprocessing/preprocess.py
```
