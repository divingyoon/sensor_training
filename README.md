# 16-Channel Tactile Intelligence Framework

본 프로젝트는 16채널 기압 기반 촉각 센서의 **서브 밀리미터 위치 추론(SR)** 및 **3축 힘(3-Axis Force) 감지**를 위한 통합 인공지능 프레임워크입니다.

---

## 1. 전체 구조 및 데이터 흐름 (System Architecture)

본 시스템은 원천 데이터로부터 최적의 신경망 모델을 도출하기까지 3단계의 핵심 파이프라인을 거칩니다.

1.  **Raw Data Processing**: 아두이노(DUE), 스테이지(Ethermotion), 상용 힘 센서(AFD) 데이터를 타임스탬프 기준으로 병합합니다.
2.  **Preprocessing**: 물리적 노이즈를 제거하고 학습에 최적화된 Zarr 또는 Feature 형태의 데이터셋을 생성합니다.
3.  **Advanced Training**: CNN-LSTM, Transformer, Attention 등 고도화된 모델을 통해 위치와 힘을 학습합니다.

---

## 2. 핵심 파이프라인 사용법 (Core Pipeline)

### Step 1: 데이터 병합 (Raw Merge)
분산된 센서 로그를 하나의 시계열 CSV로 통합합니다.
```bash
# 대표 예시: 100Hz 리샘플링 및 미세 시간 지연 보정 적용
python3 preprocessing/raw_merge.py \
    --raw-root preprocessing/raw_data \
    --align-mode resample --resample-hz 100 \
    --min-match-ratio 0.9 \
    --force-round-dp 2

```

### Step 2: 전처리 및 정제 (Preprocess)
스테이지 정지 구간만 남기고, 드리프트를 구간별 baseline으로 보정하여 학습 특성을 생성합니다.
```bash
# 기본값 예시
python3 preprocessing/preprocess.py \
  --raw-dir preprocessing/raw_data \
  --out-dir preprocessing/processed_data \
  --min-signal 0.02 \
  --min-reliable-s 0.001 \
  --baseline-z-thresh 0.001 \
  --baseline-force-thresh 0.5 \
  --baseline-min-consec 40
  
python3 preprocessing/preprocess.py \
  --raw-dir preprocessing/raw_data \
  --out-dir preprocessing/processed_data_min10 \
  --min-signal 0.01 \
  --min-reliable-s 0.001\
  --baseline-z-thresh 0.001 \
  --baseline-force-thresh 0.5 \
  --baseline-min-consec 40

python3 preprocessing/preprocess.py \
  --raw-dir preprocessing/raw_data \
  --out-dir preprocessing/processed_data_min8 \
  --min-signal 0.008 \
  --min-reliable-s 0.001\
  --baseline-z-thresh 0.001 \
  --baseline-force-thresh 0.5 \
  --baseline-min-consec 40
```

전처리 세부 단계
- 다중 baseline 자동 탐색: z≈0 & |Fz|이 낮고 이동이 멈춘 구간을 전체 시퀀스에서 찾아 baseline_id를 부여합니다(없으면 파일 선두 구간으로 폴백).
- 좌표/정지 필터: 0.5 mm 그리드에 스냅되고 앞뒤 샘플까지 정지한 행만 유지.
- 깊이·위상 계산: z_depth_mm=원본 z_mm(음수 제거), 각 (x,y) 그룹의 최대 깊이 이후를 unloading(phase=1)으로 표기.
- 센서 정규화: s_norm_i = (s_i - baseline_i) / baseline_i 를 baseline_id별로 적용, Fz는 구간별 fz_mean을 빼서 보정.
- z-bin 및 신호 필터: z_bin_mm 간격으로 binning 후 max|s_norm| < min_signal 행 제거; min_reliable_s로 모든 실험에서 일정 신호가 확보된 좌표만 유지.
- 출력물: `processed_data/baselines/*_baselines.json`, `grid/*_grid.csv`, `features/*_features.csv`, 소재별 통합 `*_features.csv`, 선택 시 `zarr_data/dataset_<mat>.zarr`.


### Step 3: 통합 및 비교 학습 (Unified & Comparison Training)
권장 기본 실험은 depth-aware heatmap 기반의 `multi_head_field` 3단계(Stage1~3)입니다.
- **Stage1 (baseline, point label)**  
```
python -m training.pipelines.train_comparison \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --models multi_head_field \
  --epochs 100 --batch-size 1024 --seq-len 50 \
  --lambda-z 0.0 --lambda-fz 0.0 \
  --decode-xy none
```
- **Stage2 (soft label, xy만)**  
```
python -m training.pipelines.train_comparison \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --models multi_head_field \
  --use-depth-aware-label \
  --depth-label-kernel gaussian --depth-radius-model hertz \
  --heatmap-size 40 --fg-weight 8.0 --heatmap-sigma-scale 0.35 \
  --lambda-z 0.0 --lambda-fz 0.0 \
  --decode-xy softargmax \
  --depth-fallback-mm 1.0 --depth-min-for-label 0.05 \
  --save-heatmap-overlay --overlay-batches 1 --overlay-samples 4 \
  --epochs 100 --batch-size 1024
```
- **Stage3 (soft label + z/Fz 보조)**  
```
python -m training.pipelines.train_comparison \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --models multi_head_field \
  --use-depth-aware-label \
  --loss-xy bce --loss-z huber --loss-fz huber \
  --lambda-xy 1.0 --lambda-z 0.2 --lambda-fz 0.2 \
  --depth-label-kernel gaussian --depth-radius-model hertz \
  --heatmap-size 40 --fg-weight 8.0 --heatmap-sigma-scale 0.35 \
  --decode-xy softargmax \
  --depth-fallback-mm 1.0 --depth-min-for-label 0.05 \
  --save-heatmap-overlay --overlay-batches 1 --overlay-samples 4 \
  --epochs 100 --batch-size 1024
```

여전히 다른 모델(MFP 외 MLP, CNNLSTM, SATS 등)을 비교하고 싶다면 `--models` 목록에 추가해 동일 스크립트로 함께 학습하면 됩니다.

모델별 배치 상한(메모리 보호용 클램프)
- `cnnlstm`, `cnnbilstm`: 최대 1024
- `sats`, `sats_xy`: 최대 512
- `unified`: 최대 4096
- `multi_head_field`: 최대 1024
- 기타: 요청한 `--batch-size` 그대로 적용

성능/속도 참고
- `--preload-vram`을 켜면 전체 데이터를 VRAM에 적재해 I/O 대기 없이 빠르게 학습합니다. VRAM 부족 시 끄거나 `--preload-batch-size`를 더 줄이세요.
- 데이터 건수가 약 1.9만(예: min20)일 경우 CNN-LSTM은 에폭당 스텝이 ~19개라 에폭이 매우 빠르게 끝나는 것이 정상입니다. 더 촘촘한 학습이 필요하면 데이터 필터를 완화하거나 에폭 수를 늘리세요.

### Step 4: 셀별 히트맵 평가 (Comparison Heatmap)
모델별 X/Y/Z/XY 오차를 동일 그리드에서 시각화합니다.
```bash
python3 -m training.pipelines.evaluate_comparison_heatmap \
  --runs-dir training/runs \
  --models mlp cnn cnnlstm cnnbilstm sats transformer unified isoline_gnn tactile_gnn_gat multi_head_field \
  --batch-size 512 \
  --device cuda \
  --eval-split all \
  --fill-missing neighbor
```

---

## 3. 주요 디렉토리 안내 (Directory Map)

*   **`preprocessing/`**: 데이터 병합 및 정제 로직. (상세 내용은 `preprocessing/README.md` 참조)
*   **`training/`**: 모델 정의, 증강, 비교 학습 및 평가 도구. (상세 내용은 `training/README.md` 참조)
*   **`inference/`**: 학습된 체크포인트로 xy/z/Fz 추론 및 heatmap overlay를 확인하는 스크립트.
*   **`md/`**: 연구 보고서, 최신 논문 기술 분석 및 학습 구조 설계 문서.
*   **`learning_based/`**: (Old) 초기 단계의 CNN-LSTM 및 6자유도 실험 코드.

---

## 4. 분석 및 시각화 (Analysis)

학습이 완료된 모델은 다음 도구를 통해 검증할 수 있습니다.
*   **오차 분석**: `training/utils/visualize_grid_errors.py`를 통해 X/Y/Z 축별 오차 히트맵 생성.
*   **실시간 확인**: `training/pipelines/visualize_realtime`을 통해 3D 히트맵 및 힘 벡터 시각화.
