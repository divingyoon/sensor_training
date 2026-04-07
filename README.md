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
python3 preprocessing/raw_merge.py --align-mode resample --resample-hz 100
```

### Step 2: 전처리 및 정제 (Preprocess)
스테이지가 이동 중인 오염된 데이터를 필터링하고 학습용 정규화를 수행합니다.
```bash
# 대표 예시: 0.01 임계값 이상의 유효 접촉 데이터만 추출
python3 preprocessing/preprocess.py --contact-threshold 0.01 --z-bin-mm 0.02
```

### Step 3: 통합 및 비교 학습 (Unified & Comparison Training)
다양한 아키텍처를 테스트하고 최적의 모델을 선정합니다.
```bash
# 대표 예시: MLP, CNN-LSTM, SATS 모델을 동시에 학습하여 성능 비교
python3 -m training.pipelines.train_comparison \
 --models mlp cnn cnnlstm cnnbilstm sats transformer unified isoline_gnn tactile_gnn_gat multi_head_field \
 --epochs 100
```

### Step 4: 셀별 히트맵 평가 (Comparison Heatmap)
모델별 X/Y/Z/XY 오차를 동일 그리드에서 시각화합니다.
```bash
python3 -m training.pipelines.evaluate_comparison_heatmap \
  --runs-dir training/runs_comparison \
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
*   **`md/`**: 연구 보고서, 최신 논문 기술 분석 및 학습 구조 설계 문서.
*   **`learning_based/`**: (Old) 초기 단계의 CNN-LSTM 및 6자유도 실험 코드.

---

## 4. 분석 및 시각화 (Analysis)

학습이 완료된 모델은 다음 도구를 통해 검증할 수 있습니다.
*   **오차 분석**: `training/utils/visualize_grid_errors.py`를 통해 X/Y/Z 축별 오차 히트맵 생성.
*   **실시간 확인**: `training/pipelines/visualize_realtime`을 통해 3D 히트맵 및 힘 벡터 시각화.

---

## 5. 재현용 명령어 모음 (Used Commands)

아래 순서가 현재 비교 실험의 기본 워크플로우입니다.

```bash
# 1) Raw 병합 (100Hz 리샘플)
python3 preprocessing/raw_merge.py --align-mode resample --resample-hz 100

# 2) 전처리 (그리드/정지구간/품질 필터 + Zarr 생성)
python3 preprocessing/preprocess.py --contact-threshold 0.01 --z-bin-mm 0.02

# 3) 모델 비교 학습 (processed_data에서 Zarr 우선 사용)
python3 -m training.pipelines.train_comparison --models mlp cnnlstm sats --epochs 50

# 4) 히트맵 비교
python3 -m training.pipelines.evaluate_comparison_heatmap \
  --runs-dir training/runs_comparison \
  --models mlp cnnlstm sats \
  --batch-size 512 \
  --device cuda \
  --eval-split all \
  --fill-missing neighbor
```

추가 권장 옵션 (샘플 수 보강):
```bash
python3 -m training.pipelines.train_comparison \
  --models mlp cnnlstm sats \
  --epochs 50 \
  --phase all \
  --stride 1 \
  --seq-len 32
```

---
**문의 및 보고**: Gemini CLI Tactile Engineering Team (2026-04-06)
