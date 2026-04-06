# Tactile Sensor Training Framework (v4.0)

본 프레임워크는 16채널 촉각 센서의 **3D Super-Resolution(SR)**, **3축 힘(Fx, Fy, Fz) 추정**, 그리고 **연속적인 Force Field 재구성**을 위한 고도화된 연구 및 학습 파이프라인입니다.

---

## 1. 디렉토리 구조 (Directory Structure)

```text
training/
├── data/           # 데이터셋 로더 및 증강 (dataset_unified.py, sensor_layout.py)
├── models/         # 모델 정의 (10종 이상의 아키텍처)
│   ├── mlp_baseline.py         # 비교용 기본 MLP
│   ├── cnn_sr.py / cnnlstm_sr.py # 공간/시간 기반 SR
│   ├── sats_model.py           # Self-Attention 기반 (X축 오차 특화)
│   ├── supertac_vae.py         # VAE + Deconv (고해상도 맵 생성)
│   ├── tactile_transformer.py   # Transformer 기반 전역 상관관계 학습
│   └── tactile_gnn_gat.py      # Graph Attention Network 기반
├── utils/          # 손실 함수, 설정, 분석 도구 (loss.py, visualize_grid_errors.py)
└── pipelines/      # 실행 스크립트 (train_unified.py, train_comparison.py)
```

---

## 2. 모델 라인업 및 특징

| 모델명 | 핵심 기술 | 주요 용도 |
| :--- | :--- | :--- |
| **MLP Baseline** | Simple MLP | 성능 비교를 위한 기준점 |
| **CNN-LSTM** | Spatial + Temporal | 히스테리시스 보정 및 동적 정밀도 향상 |
| **SATS** | Self-Attention | 센서 간 비대칭성(X축 오차) 집중 해결 |
| **SuperTac** | VAE + Upsampling | 25x25 이상의 고해상도 Force Field 생성 |
| **Transformer** | Global Attention | 대량의 데이터에서 최고의 일반화 성능 확보 |
| **Tactile GAT** | Graph Attention | 센서 배치의 기하학적 구조를 명시적으로 반영 |

---

## 3. 사용 방법 (Usage Guide)

### 3.1 모델 비교 학습 (Comparison Pipeline)
여러 모델의 성능을 동시에 측정하고 최적의 구조를 찾습니다.
```bash
# MLP, CNN, CNN-LSTM, SATS 모델을 동시에 30에포크 학습 및 비교
python3 -m training.pipelines.train_comparison \
    --models mlp cnn cnnlstm sats \
    --epochs 30 --batch-size 64 --seq-len 50
```
*   **평가 지표**: MSE, RMSE, MAE, **$R^2$ (결정계수)** 가 자동으로 산출되어 `comparison_results.json`에 저장됩니다.

### 3.2 통합 모델 학습 (Unified Training)
위치와 힘을 동시에 학습하는 최종형 모델을 실행합니다.
```bash
python3 -m training.pipelines.train_unified \
    --data-dir preprocessing/raw_data \
    --out-dir training/runs_unified \
    --epochs 100
```

### 3.3 데이터 증강 (Data Augmentation)
과적합을 방지하기 위해 학습 시 `--augment` 옵션을 활성화할 수 있습니다 (코드 내 기본 설정 가능).
*   **Spatial Flip**: 센서 그리드 좌우/상하 반전 및 좌표 부호 보정.
*   **Gaussian Noise**: 신호에 1% 미세 노이즈 추가로 강인성 확보.

---

## 4. 분석 및 시각화 (Analysis)

### 그리드 오차 히트맵 생성
특정 모델이 X축 오차를 얼마나 줄였는지 시각적으로 검증합니다.
```bash
python3 -m training.utils.visualize_grid_errors \
    --model-path training/runs_comparison/best_sats.pth \
    --out-path training/runs_comparison/sats_error_map.png
```
*   **결과**: X, Y, Z축 각각의 MAE 분포가 PNG 히트맵으로 출력됩니다.

### 학습 결과 해석 (Metrics)
*   **MAE (Mean Absolute Error)**: 실제 거리(mm) 또는 힘(N) 단위의 평균 오차.
*   **$R^2$ Score**: 1.0에 가까울수록 모델이 센서 데이터의 물리적 거동을 완벽하게 이해했음을 의미합니다. (0.95 이상 권장)

---
**Last Updated**: 2026-04-06
**Version**: 4.0 (Advanced Tactile Intelligence Framework)
