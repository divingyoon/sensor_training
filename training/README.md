# Tactile Sensor Training Framework (v2.0)

본 프레임워크는 16채널 바이브로 타일 촉각 센서의 슈퍼 해상도(SR) 및 수직 항력(Fz) 예측 모델 학습을 위한 고속 파이프라인입니다.

## 1. 학습 아키텍처 개요

### 핵심 기술
*   **VRAM Caching**: Zarr 데이터셋 전체를 GPU 메모리에 미리 로드하여 DataLoader 병목을 완전히 제거 (에포크당 ~1초).
*   **Stationary Filtering**: 스테이지가 이동 중인 오염된 데이터를 제거하고 완전 정지 상태의 데이터만 학습에 사용.
*   **Target Normalization**: X, Y, Z, Fz 데이터를 [0, 1] 범위로 정규화하여 학습 안정성 및 정밀도 극대화.
*   **Weighted Loss**: 오차가 상대적으로 높은 X축에 2.0배 이상의 가중치를 부여하여 축별 정밀도 균형 확보.

### 모델 구성
1.  **SR Model (MLPSR)**: 16ch 신호 + 인덴터 반지름 → (x, y, z_depth) 추론.
2.  **FF Model (MLPFF)**: 16ch 신호 + 인덴터 반지름 + (x, y, z_depth) → Fz(Newton) 추론.

---

## 2. 사용 방법 (Terminal Usage)

### Step 1: 데이터 병합 (Raw Merge)
타임스탬프를 기준으로 센서와 스테이지 좌표를 동기화합니다.
```bash
python3 preprocessing/raw_merge.py \
    --raw-root preprocessing/raw_data \
    --align-mode resample --sync-ref ethermotion \
    --resample-hz 100 --window-ms 20 --window-agg mean --max-dt-ms 8
```

### Step 2: 전처리 및 정제 (Preprocess)
정지 상태 데이터 필터링 및 Zarr 데이터셋을 생성합니다.
```bash
python3 preprocessing/preprocess.py \
    --raw-dir preprocessing/raw_data \
    --out-dir preprocessing/processed_data \
    --contact-threshold 0.01
```

### Step 3: SR 모델 학습 (Super Resolution)
고정밀 위치 추론 모델을 학습합니다.
```bash
python3 -m training.train_sr_zarr \
    --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
    --out-dir training/runs_sr_ecomesh \
    --epochs 200 --batch-size 16384 --lr 1e-3 --seed 44
```

### Step 4: Force Field 학습 (Force Prediction)
SR 결과를 바탕으로 수직 항력을 예측하는 모델을 학습합니다.
```bash
python3 -m training.train_ff_zarr \
    --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
    --sr-model-path training/runs_sr_ecomesh/best_model.pt \
    --out-dir training/runs_ff_zarr \
    --epochs 100
```

---

## 3. 분석 및 시각화 도구

### 그리드별 오차 히트맵 (Heatmap Visualization)
격자점별 MAE 분포를 PNG로 생성하여 하드웨어적 데드존이나 이상 지점을 파악합니다.

**SR 오차 시각화 (X, Y, Z):**
```bash
python3 -m training.visualize_grid_errors \
    --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
    --model-path training/runs_sr_ecomesh/best_model.pt
```
*   결과물: `training/runs_sr_ecomesh/grid_errors_v3.png`

**Force Field 오차 시각화 (Fz):**
```bash
python3 -m training.visualize_ff_grid_errors \
    --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
    --ff-model-path training/runs_ff_zarr/best_model.pt \
    --sr-model-path training/runs_sr_ecomesh/best_model.pt
```
*   결과물: `training/runs_ff_zarr/ff_grid_errors.png`

---

## 4. 모델 성능 리포트 (최신 기준)
*   **SR MAE**: X: 0.6mm / Y: 0.3mm / Z: 0.08mm
*   **FF MAE**: Fz: 0.6N
*   **특징**: 가장자리 영역 오차를 제외한 중앙부 정밀도는 상기 수치보다 약 2배 우수함.
