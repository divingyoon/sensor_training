# 16-Channel Tactile Intelligence Framework

16채널 기압 기반 촉각 센서 데이터로 XY 위치, contact 기준 Z depth, Fz를 학습하기 위한 파이프라인입니다.  
루트 README는 전체 실행 순서와 대표 명령만 다룹니다. 세부 옵션은 [preprocessing/README.md](/home/user/sensor_training/preprocessing/README.md:1), [training/README.md](/home/user/sensor_training/training/README.md:1)를 참조하세요.

## Workflow
1. `raw_merge.py`로 DUE, Ethermotion, AFD 로그를 trial별 merged CSV로 정렬합니다.
2. `preprocess.py`로 baseline 보정, grid 필터링, feature CSV, Zarr dataset을 만듭니다.
3. `train_z_fz_regressor.py`로 GT XY 조건 기반 Z/Fz 회귀를 학습하고, 필요하면 frozen XY checkpoint 기준 validation도 같이 봅니다. 이 경로는 hysteresis 보존을 위해 loading phase만 사용합니다.
4. 공식 학습/평가는 trial-aware 5-fold CV를 기본으로 사용합니다.
5. 필요할 때만 `train_comparison.py` Stage1/2/3 또는 평가 스크립트로 XY heatmap 실험을 재현합니다.

## Data Assumptions
- 원천 데이터 루트: `preprocessing/raw_data`
- 전처리 산출물 루트: `preprocessing/processed_data`
- 학습용 Zarr는 여러 개가 있을 수 있으므로 `--zarr-path`를 명시하는 것을 권장합니다.
- `preprocess.py`는 `*_merged.csv`를 읽으므로 `raw_merge.py`가 선행되어야 합니다.

## Step 1. Raw Merge
언제 쓰는가:
원시 장비 로그를 trial별 공통 타임라인 CSV로 만들 때 사용합니다.

대표 명령:
```bash
python3 preprocessing/raw_merge.py \
  --raw-root preprocessing/raw_data \
  --align-mode resample \
  --resample-hz 100 \
  --min-match-ratio 0.9 \
  --force-round-dp 2
```

입력:
- `preprocessing/raw_data/**/due*.csv`
- `preprocessing/raw_data/**/ethermotion*.csv`
- `preprocessing/raw_data/**/afd*.csv`

주요 출력:
- trial별 `*_merged.csv`
- 동기화 확인용 PNG
- baseline/summary JSON

다음 단계로 넘어가는 조건:
- 각 trial에 merged CSV가 생성되고, 동기화 품질이 `--min-match-ratio` 기준을 만족해야 합니다.

자세한 옵션 설명:
- [preprocessing/README.md](/home/user/sensor_training/preprocessing/README.md:1)

## Step 2. Preprocess
언제 쓰는가:
merged CSV를 baseline-corrected feature, grid CSV, Zarr dataset으로 바꿀 때 사용합니다.

대표 명령:
```bash
python3 preprocessing/preprocess.py \
  --raw-dir preprocessing/raw_data \
  --out-dir preprocessing/processed_data \
  --use-depth-aware-radius \
  --radius-model hertz \
  --z-bin-mm 0.001 \
  --min-reliable-s 0.001 \
  --baseline-z-thresh 0.001 \
  --baseline-force-thresh 0.5 \
  --baseline-min-consec 40 \
  --fallback-depth-mode none
```

입력:
- `preprocessing/raw_data/**/*_merged.csv`

주요 출력:
- `processed_data/baselines/*_baselines.json`
- `processed_data/grid/*_grid.csv`
- `processed_data/features/*_features.csv`
- `processed_data/zarr_data/dataset_<material>.zarr`
- `processed_data/peak_summary/*_peak_summary.csv`

주요 조정 포인트:
- `--min-signal`: 미지정 시 baseline noise 기반 자동 샘플 제거 기준
- `--min-reliable-s`: 좌표 유지 기준
- `--baseline-*`: baseline 자동 탐색 규칙
- `--use-depth-aware-radius`: `d5`, `d10` 폴더명에서 자동 반경을 계산해 Zarr의 마지막 aux field 의미를 바꿈

다음 단계로 넘어가는 조건:
- 원하는 필터 기준으로 feature/Zarr가 생성되고, 학습에 사용할 dataset 경로가 확정되어야 합니다.

자세한 옵션 설명:
- [preprocessing/README.md](/home/user/sensor_training/preprocessing/README.md:1)

## Step 3. Z/Fz Regressor
언제 쓰는가:
XY 위치는 주어진 조건으로 보고 contact 기준 Z/Fz만 별도 회귀할 때 사용합니다.

대표 명령:
```bash
python -m training.pipelines.train_z_fz_regressor \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --out-dir training/runs_z_fz \
  --xy-checkpoint training/runs_comparison/folds/fold_0/best_multi_head_field_stage2_dlabel-gaussian-hertz_xybce1_zoff_fzoff_decsoftargmax.pth \
  --decode-xy softargmax \
  --xy-noise-std-mm 0.5 \
  --cv-folds 5 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 1024 \
  --device cuda
```

동작 의미:
- 학습은 항상 GT XY를 조건으로 사용합니다.
- 전처리 zarr의 `depth_mm`는 현재 `z_contact_mm` 기준으로 저장됩니다.
- Z/Fz sequence 회귀는 loading phase만 사용합니다. unloading/all은 이 경로에서 지원하지 않습니다.
- 공식 학습 경로는 trial-aware 5-fold CV를 기본으로 사용합니다.
- `--xy-noise-std-mm`는 학습 시 GT XY에 노이즈를 섞어 XY 오차에 대한 강건성을 높이는 옵션입니다.
- `--xy-checkpoint`를 주면 validation에서 frozen XY heatmap checkpoint를 decode한 `predicted_xy` 지표도 같이 계산합니다.
- 이 `predicted_xy` 평가는 radius를 여전히 GT에서 가져오므로 완전한 end-to-end 평가는 아닙니다.

주요 출력:
- `training/runs_z_fz/best_z_fz_regressor.pth`
- `training/runs_z_fz/metrics_z_fz_regressor.json`
- `training/runs_z_fz/history_z_fz_regressor.json`

자세한 옵션 설명:
- [training/README.md](/home/user/sensor_training/training/README.md:1)

## Optional. XY Heatmap Comparison
`multi_head_field` Stage1/2/3를 다시 만들거나 다른 모델과 비교하려면 `train_comparison.py`를 사용합니다.

Stage2 예시:
```bash
python -m training.pipelines.train_comparison \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --models multi_head_field \
  --cv-folds 5 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --use-depth-aware-label \
  --depth-label-kernel gaussian \
  --depth-radius-model hertz \
  --heatmap-size 40 \
  --fg-weight 8.0 \
  --heatmap-sigma-scale 0.35 \
  --lambda-z 0.0 \
  --lambda-fz 0.0 \
  --decode-xy softargmax \
  --depth-fallback-mm 1.0 \
  --depth-min-for-label 0.05 \
  --save-heatmap-overlay \
  --overlay-batches 1 \
  --overlay-samples 4 \
  --epochs 100 \
  --batch-size 1024
```

평가 예시:
```bash
python3 -m training.pipelines.evaluate_comparison_heatmap \
  --runs-dir training/runs_comparison \
  --models multi_head_field \
  --batch-size 512 \
  --device cuda \
  --eval-split all \
  --fill-missing neighbor
```

세부 실험 순서와 옵션 설명:
- [training/README.md](/home/user/sensor_training/training/README.md:1)

## Directory Map
- `preprocessing/`: raw merge, preprocess, label preview
- `training/`: dataset, model, training/evaluation pipeline
- `inference/`: 저장된 checkpoint 기반 추론 및 overlay 확인
- `md/`: 연구/설계 메모
