# Skin Tactile SR Training

이 문서는 `/home/user/skin_ws/training/` 의 학습 구조 사용법, 파일 구성, 실행 방법을 설명합니다.

---

## 1. 목적

저해상도 barometric tactile array (16ch, 4×4 그리드) 의 응답으로부터 구형 인덴터에 의한 **고해상도 contact/pressure map (64×64)** 을 복원하는 physics-informed SR 문제.

학습 목표:

- 주과제: 고해상도 contact map 복원
- 부과제: contact map 으로부터 $x, y, F_z, A_c$ 계산 또는 회귀

supervision 구성:

| Loss 항 | 역할 |
|---------|------|
| $\mathcal{L}_{map}$ | analytic pseudo GT map 과 L1 비교 |
| $\mathcal{L}_{sensor}$ | 복원 map 을 센서 grid 로 내렸을 때 실제 sensor 응답과 일치 |
| $\mathcal{L}_{F_z}$ | map 적분값과 측정 $F_z$ 일치 |
| $\mathcal{L}_{smooth}$ | Total Variation (고주파 artifact 억제) |

---

## 2. 전제 조건

preprocessing 이 완료되어 있어야 합니다.

```bash
python3 /home/user/skin_ws/preprocessing/build_preprocessing_dataset.py \
  --canvas-size-mm 25.0 \
  --sensor-spacing-mm 6.5 \
  --overwrite
```

preprocessing_data 위치:

```
/home/user/skin_ws/preprocessing/preprocessing_data/
  dataset_index.json
  eco20/
    eco20_1/
      loading/
        depth_001.000mm/
          rep_0000/
            tactile_lr_norm.npy   (16,)
            aux_feat.npy          (4,)
            hr_contact_map.npy    (64, 64)
            meta.json
```

---

## 3. 파일 구성

```text
training/
  __init__.py
  config.py           하이퍼파라미터 기본값 (TrainConfig dataclass)
  sensor_layout.py    4×4 센서 레이아웃, D(M) operator
  dataset.py          SkinDataset, build_loaders
  loss.py             SkinLoss (복합 physics-informed loss)
  model_baseline.py   Phase 1: MLP encoder + CNN decoder
  model_main.py       Phase 2: 1D CNN encoder + FiLM + 2D UNet-lite decoder
  train.py            학습 루프 (Phase 1 / 2 공용)
  evaluate.py         평가 지표 함수군
  runs/               학습 결과 저장 폴더
    phase1/
      best.pt
      last.pt
      history.json
    phase2/
      best.pt
      last.pt
      history.json
```

---

## 4. 센서 레이아웃

센서 16채널 (Skin1 ~ Skin16) 의 물리적 위치 (센서 프레임 기준, mm):

```
      Col0    Col1    Col2    Col3
Row0  Skin1   Skin2   Skin3   Skin4     y = 0.0 mm
Row1  Skin5   Skin6   Skin7   Skin8     y = 6.5 mm
Row2  Skin9   Skin10  Skin11  Skin12    y = 13.0 mm
Row3  Skin13  Skin14  Skin15  Skin16    y = 19.5 mm

x = 0.0 / 6.5 / 13.0 / 19.5 mm
```

- 간격: 6.5 mm 균등 그리드
- 센서 어레이 전체 크기: 19.5 × 19.5 mm
- Dead channel: Skin2 (index 1), Skin9 (index 8) — 항상 0, 학습 시 제외
- **유효 채널 수: 14개** (dead ch 제거 후)

Canvas (HR map 공간 좌표):

- 크기: 25 × 25 mm (고정)
- 중심: 각 샘플의 contact center (X, Y)
- x 범위: `[cx - 12.5, cx + 12.5]` mm
- y 범위: `[cy - 12.5, cy + 12.5]` mm

---

## 5. 모델 구조

### Phase 1 — Baseline (`model_baseline.py`)

**MLP Encoder + CNN Decoder**

```
Input
  tactile   (B, 14)   dead ch 제거된 z-score 정규화 tactile
  aux       (B,  4)   [fx_N, fy_N, depth_mm, radius_mm]

MLP Encoder
  Linear(14, 128) → BN → ReLU
  Linear(128, 256) → BN → ReLU
  Linear(256, 128) → BN → ReLU       → latent (B, 128)

Auxiliary Encoder
  Linear(4, 32) → ReLU               → aux_enc (B, 32)

Fusion
  cat(latent, aux_enc)                → (B, 160)
  Linear(160, 512) → ReLU
  reshape                             → (B, 8, 8, 8)

CNN Decoder  [8×8 → 64×64, 3단 upsample]
  ConvTranspose2d(8,  64, 4, 2, 1) → BN → ReLU   (B, 64, 16, 16)
  ConvTranspose2d(64, 32, 4, 2, 1) → BN → ReLU   (B, 32, 32, 32)
  ConvTranspose2d(32, 16, 4, 2, 1) → BN → ReLU   (B, 16, 64, 64)
  Conv2d(16, 1, 3, 1, 1) → Sigmoid                (B,  1, 64, 64)

Output
  hr_map    (B, 1, 64, 64)   [0, 1] 범위 pressure map
```

파라미터 수: **201,089**

---

### Phase 2 — Main (`model_main.py`)

**1D CNN Encoder + FiLM Conditioning + 2D UNet-lite Decoder**

```
Input
  tactile   (B, 14)      단일 프레임
              (B, K, 14)  depth 시퀀스 (K: depth step 수)
  aux       (B,  4)      [fx_N, fy_N, depth_mm, radius_mm]

1D CNN Encoder  [K-axis Conv]
  Conv1d(14, 64,  3, pad=1) → ReLU
  Conv1d(64, 128, 3, pad=1) → ReLU
  Conv1d(128, 256, 3, pad=1) → ReLU
  AdaptiveAvgPool1d(1) → squeeze          → latent (B, 256)

FiLM Conditioning
  Linear(4, 512) → split → γ (B, 256), β (B, 256)
  latent = γ * latent + β                 → (B, 256)

Reshape
  (B, 256) → (B, 16, 4, 4)

2D UNet-lite Decoder  [4×4 → 64×64, 4단 upsample]
  ConvTranspose2d(16, 64, 4, 2, 1) → BN → ReLU   (B, 64,  8,  8)
  ConvTranspose2d(64, 32, 4, 2, 1) → BN → ReLU   (B, 32, 16, 16)
  ConvTranspose2d(32, 16, 4, 2, 1) → BN → ReLU   (B, 16, 32, 32)
  ConvTranspose2d(16,  8, 4, 2, 1) → BN → ReLU   (B,  8, 64, 64)
  Conv2d(8, 1, 1) → Sigmoid                        (B,  1, 64, 64)

Output
  hr_map    (B, 1, 64, 64)
```

파라미터 수: **188,337**

단일 프레임 입력 `(B, 14)` 은 내부에서 `(B, 1, 14)` 로 자동 변환됩니다.

---

## 6. Loss 구성 (`loss.py`)

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{map} + \lambda_2 \mathcal{L}_{sensor} + \lambda_3 \mathcal{L}_{F_z} + \lambda_4 \mathcal{L}_{smooth}
$$

| 항 | 수식 | 기본 λ |
|----|------|--------|
| $\mathcal{L}_{map}$ | `L1(M_pred, M_pseudo)` | 1.0 |
| $\mathcal{L}_{sensor}$ | `L1(D(M_pred), s_measured)` | 0.5 |
| $\mathcal{L}_{F_z}$ | `L1(sum(M_pred) * pixel_area, Fz)` | 0.01 |
| $\mathcal{L}_{smooth}$ | `TV(M_pred)` (anisotropic) | 0.1 |

**$\mathcal{L}_{F_z}$ λ 주의**: map integral 값이 N 단위 Fz 대비 수백 배 크므로 λ 를 0.01 이하로 유지합니다.

D(M) operator (sensor_layout.py):

- HR map `(B, 1, H, W)` → sensor response `(B, 16)`
- 각 센서 위치에서 `F.grid_sample` bilinear interpolation
- canvas bounds `(B, 2)` 가 배치마다 달라도 벡터화 처리 (GPU 효율)

---

## 7. Dataset (`dataset.py`)

### SkinDataset

```python
from training.dataset import SkinDataset

ds = SkinDataset(
    data_dir="/home/user/skin_ws/preprocessing/preprocessing_data",
    split="train",        # "train" | "val" | "test" | "all"
    phase="loading",      # "loading" | "unloading" | "all"
    min_depth_mm=0.5,     # 0.0 이하 depth 샘플 제외
    val_ratio=0.15,
    seed=42,
)
```

각 샘플 출력 dict:

| 키 | shape | 설명 |
|----|-------|------|
| `tactile` | (14,) | dead ch 제거, z-score 정규화 |
| `tactile_raw` | (16,) | dead ch 포함 전체 채널 (D(M) 일관성 loss 에 사용) |
| `aux` | (4,) | `[fx_N, fy_N, depth_mm, radius_mm]` |
| `hr_map` | (1, 64, 64) | pseudo GT contact map |
| `fz` | scalar | 측정 $F_z$ [N] |
| `cx`, `cy` | scalar | contact center [mm] |
| `depth_mm` | scalar | depth bin [mm] |
| `x_bounds` | (2,) | canvas x 범위 [mm] |
| `y_bounds` | (2,) | canvas y 범위 [mm] |

Split 방식:

- **trial_id 단위** 분리 (데이터 누수 방지)
- trial 수가 적을 경우 (1개 trial 등) 자동으로 전체 사용하는 fallback 적용

### build_loaders

```python
from training.dataset import build_loaders

train_loader, val_loader, test_loader = build_loaders(
    data_dir="/home/user/skin_ws/preprocessing/preprocessing_data",
    batch_size=64,
    phase="loading",
    min_depth_mm=0.5,
)
```

---

## 8. 학습 실행 (`train.py`)

### Phase 1 — Baseline

```bash
cd /home/user/skin_ws
python -m training.train \
  --phase 1 \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3
```

### Phase 2 — Main

```bash
python -m training.train \
  --phase 2 \
  --epochs 100 \
  --batch-size 64 \
  --lr 5e-4
```

### 전체 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--phase` | `1` | 모델 Phase (1: baseline, 2: main) |
| `--data-dir` | `preprocessing_data` 경로 | 데이터 디렉토리 |
| `--out-dir` | `training/runs` | 체크포인트 저장 경로 |
| `--epochs` | `100` | 총 학습 epoch |
| `--lr` | `1e-3` | 초기 learning rate |
| `--batch-size` | `64` | 배치 크기 |
| `--lambda-map` | `1.0` | $\mathcal{L}_{map}$ 가중치 |
| `--lambda-sensor` | `0.5` | $\mathcal{L}_{sensor}$ 가중치 |
| `--lambda-fz` | `0.01` | $\mathcal{L}_{F_z}$ 가중치 |
| `--lambda-smooth` | `0.1` | $\mathcal{L}_{smooth}$ 가중치 |
| `--resume` | — | 체크포인트 경로 (이어서 학습) |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |

Optimizer / Scheduler:

- **AdamW**: `lr=1e-3`, `weight_decay=1e-4`
- **CosineAnnealingLR**: `T_max=epochs`, `eta_min=lr*0.01`
- Gradient clipping: `max_norm=1.0`

체크포인트:

- `runs/phase{N}/best.pt` — val loss 기준 최적 모델
- `runs/phase{N}/last.pt` — 최종 epoch 모델
- `runs/phase{N}/history.json` — epoch별 loss 및 metric 기록

---

## 9. 평가 지표 (`evaluate.py`)

| 지표 | 설명 |
|------|------|
| `centroid_error_mm` | 예측 map weighted centroid vs GT $(x,y)$ 거리 [mm] |
| `fz_mae` | map 적분 기반 예측 $F_z$ vs 측정 $F_z$ MAE [N] |
| `iou` | threshold (peak × 0.2) 기준 IoU |
| `dice` | Dice coefficient |
| `sensor_l1` | $D(\hat{M})$ vs 실제 sensor 응답 L1 |
| `map_l1` | $\hat{M}$ vs pseudo GT L1 (참고용) |

```python
from training.evaluate import compute_metrics

metrics = compute_metrics(
    pred,        # (B, 1, 64, 64)
    target_map,  # (B, 1, 64, 64)
    tactile_raw, # (B, 16)
    fz,          # (B,)
    cx, cy,      # (B,)
    x_bounds,    # (B, 2)
    y_bounds,    # (B, 2)
)
# metrics: dict with "centroid_error_mm", "fz_mae", "iou", "dice", "sensor_l1", "map_l1"
```

---

## 10. 실험 순서

### Phase 1 — SR localization

목표: contact map + centroid 복원

```bash
python -m training.train --phase 1 --epochs 100
```

평가 핵심 지표: `centroid_error_mm`, `iou`

---

### Phase 2 — + force consistency

목표: contact map + $F_z$ 일관성 추가

```bash
python -m training.train --phase 2 --epochs 100 --lr 5e-4 \
  --lambda-fz 0.05
```

평가 핵심 지표: `centroid_error_mm`, `iou`, `fz_mae`

---

### Phase 3 — + area consistency (추후)

`evaluate.py` 의 `iou`, `dice` 로 contact area 품질 평가.
area head 추가 시 모델에 `sub_head` 를 붙이는 방향으로 확장.

---

## 11. GPU 사용

- `device = cuda` 자동 감지 (`torch.cuda.is_available()`)
- 모델 파라미터, 배치 텐서 전체 `.to(device)` 처리
- `sensor_layout.py` D(M) operator: `sensor_positions.to(hr_map.device)` 자동 추적
- loss, evaluate 내부: 배치 루프 없이 `F.grid_sample` 벡터화 처리 (GPU 효율)

GPU 메모리 (RTX 5080 기준, batch_size=64):

| Phase | VRAM |
|-------|------|
| Phase 1 (MLP + CNN) | ~17 MB |
| Phase 2 (1D CNN + FiLM) | ~14 MB |

---

## 12. 빠른 체크 명령

### 학습된 모델로 단일 예측

```python
import torch
from training.model_baseline import BaselineModel

model = BaselineModel()
ckpt = torch.load("training/runs/phase1/best.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

tactile = torch.zeros(1, 14)    # (1, 14) live ch
aux     = torch.tensor([[0.0, 0.0, 1.0, 3.0]])  # [fx, fy, depth_mm, R_mm]

with torch.no_grad():
    hr_map = model(tactile, aux)  # (1, 1, 64, 64)
```

### history 시각화 (간단)

```bash
python3 - << 'PY'
import json, matplotlib.pyplot as plt
with open("training/runs/phase1/history.json") as f:
    h = json.load(f)
epochs = [r["epoch"] for r in h]
plt.plot(epochs, [r["train_loss"] for r in h], label="train")
plt.plot(epochs, [r["val_loss"] for r in h], label="val")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
plt.savefig("training/runs/phase1/loss_curve.png", dpi=120)
print("saved")
PY
```

### 체크포인트 정보 확인

```bash
python3 - << 'PY'
import torch
ckpt = torch.load("training/runs/phase1/best.pt", map_location="cpu")
print("epoch:", ckpt["epoch"])
print("val_loss:", ckpt["val_loss"])
PY
```

---

## 13. 주의사항

- **pseudo GT** (`hr_contact_map.npy`) 는 analytic Gaussian prior 기반 라벨입니다. Ecoflex 실제 변형과 완전히 일치하지 않으므로 $\mathcal{L}_{sensor}$ 와 $\mathcal{L}_{F_z}$ 를 함께 사용하는 것을 권장합니다.
- **$\lambda_{F_z}$** 는 map integral 스케일이 크므로 기본값 `0.01` 을 유지하거나, 학습 초반 `0.0` 으로 시작 후 점진적으로 올리는 것을 권장합니다.
- **센서 원점** (`sensor_origin_x_mm`, `sensor_origin_y_mm`) 이 스테이지 프레임과 다를 경우 `config.py` 에서 오프셋을 수정해야 합니다. 현재 기본값은 `0.0` (스테이지 원점 = 센서 Skin1 위치).
- **Phase 2 depth block 입력**: 현재 단일 프레임 `(B, 14)` 로 동작합니다. 동일 XY 위치에서 여러 depth 의 시퀀스 데이터가 확보되면 `(B, K, 14)` 로 바로 확장 가능합니다.
- **trial split**: 현재 데이터 1개 trial 이므로 train/val/test 가 동일 trial 을 공유합니다. trial 수가 늘어나면 자동으로 올바른 split 이 적용됩니다.
