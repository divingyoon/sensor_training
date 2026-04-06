# training_seq (CNN-LSTM)

새로 분리한 시퀀스 학습 폴더입니다.
기존 training 코드는 유지하고, 여기서는 시간축 정보를 쓰는 CNN-LSTM으로 (x,y,z)를 직접 회귀합니다.

## 핵심
- 입력: seq_len 길이의 연속 데이터
  - tactile: Skin1~Skin16 (또는 zarr tactile_lr_norm)
  - force: Fx, Fy, Fz (또는 zarr fx,fy,fz 또는 aux_feat+fz)
- 출력: 마지막 시점의 (x,y,z)
- 손실: weighted SmoothL1 (w_x, w_y, w_z)

## 데이터 소스
1. --data-source zarr
- dataset_index.json + dataset_*.zarr 사용
- --data-dir는 preprocessing_data/eco20 또는 그 상위 경로

2. --data-source csv
- raw csv 직접 사용
- 필요한 컬럼: X,Y,Z,Fx,Fy,Fz,Skin1..Skin16
- --data-dir는 csv 파일 하나 또는 csv들이 있는 폴더

## 추천 실행 (eco20, zarr)
```bash
python3 -m training_seq.train_seq \
  --data-source zarr \
  --data-dir /home/user/sensor_training/preprocessing/preprocessing_data/eco20 \
  --out-dir /home/user/sensor_training/training_seq/runs_eco20_seq \
  --phase both \
  --split-mode trial \
  --seq-len 16 \
  --stride 1 \
  --min-depth-mm 0.5 \
  --device cuda \
  --batch-size 2048 \
  --epochs 160 \
  --num-workers 12 \
  --prefetch-factor 4 \
  --pin-memory \
  --persistent-workers \
  --amp \
  --w-x 6.0 --w-y 1.0 --w-z 1.0
```

## 추천 실행 (raw csv)
```bash
python3 -m training_seq.train_seq \
  --data-source csv \
  --data-dir /home/user/sensor_training/preprocessing/raw_data \
  --out-dir /home/user/sensor_training/training_seq/runs_csv_seq \
  --phase both \
  --split-mode trial \
  --seq-len 16 \
  --stride 1 \
  --min-abs-z 0.0 \
  --device cuda \
  --batch-size 2048 \
  --epochs 160 \
  --num-workers 12 \
  --prefetch-factor 4 \
  --pin-memory \
  --persistent-workers \
  --amp
```

## 참고
- split-mode trial: trial 단위 분리 (일반화 확인용)
- split-mode random: 샘플 랜덤 분리 (점수는 보통 더 좋게 나옴)
- 체크포인트:
  - best_loss.pt
  - best_xy.pt
  - last.pt
  - history.json

## Residual 타깃 학습 (baseline 보정)
- `--target-mode residual`: 타깃을 절대값이 아니라 `target - baseline` 잔차로 학습
- `--baseline-frames N`: 각 윈도우 첫 N프레임 평균을 baseline으로 사용 (입력 feature에는 넣지 않음)
- 검증 지표 출력은 내부적으로 `pred_residual + baseline`으로 복원한 절대값 기준(mm)

예시:
```bash
python3 -m training_seq.train_seq \
  --data-source zarr \
  --data-dir /home/user/sensor_training/preprocessing/preprocessing_data/eco20 \
  --out-dir /home/user/sensor_training/training_seq/runs_eco20_seq_residual \
  --phase both \
  --split-mode trial \
  --seq-len 16 \
  --stride 1 \
  --target-mode residual \
  --baseline-frames 4
```

## 추론 (센서 연결 전 오프라인/스트림 공용)
`training_seq.infer_seq`는 체크포인트가 `(x,y,z)`인지 `(x,y,z,fz)`인지 자동 인식해 추론하고, `fz`에는 `deadband + hysteresis + EMA`를 적용해 출력합니다.

```bash
python3 -m training_seq.infer_seq \
  --ckpt /home/user/sensor_training/training_seq/runs_eco20_seq_residual/best_xy.pt \
  --in-csv /path/to/input.csv \
  --out-csv /path/to/output.csv \
  --baseline-x 0 --baseline-y 0 --baseline-z 0 --baseline-fz 0 \
  --fz-deadband 0.05 --fz-on-th 0.15 --fz-off-th 0.08 --fz-ema-alpha 0.25
```

- 입력 CSV 필수 컬럼: `Skin1..Skin16,Fx,Fy,Fz`
- 출력 CSV 컬럼: `x,y,z,fz,fz_raw,fz_source,in_contact`
- `--z-contact-gate`가 켜져 있으면 비접촉(`in_contact=0`)에서 `z=0`으로 게이팅
- residual 체크포인트(`target_mode=residual`)는 절대좌표 복원을 위해 `--baseline-x/y/z(/fz)`를 설정

## xyz+fz 학습
`--predict-fz`를 켜면 모델 출력이 `(x,y,z,fz)`가 됩니다.

```bash
python3 -m training_seq.train_seq \
  --data-source zarr \
  --data-dir /home/user/sensor_training/preprocessing/preprocessing_data/eco20 \
  --out-dir /home/user/sensor_training/training_seq/runs_eco20_seq_xyzfz \
  --phase both --split-mode trial --seq-len 16 --stride 1 \
  --target-mode residual --baseline-frames 4 \
  --predict-fz --w-fz 1.0
```

## 가장자리 성능 보정 (edge-aware loss)
말씀한 외곽 영역(`x/y`가 ±9.75 근처) 성능 저하를 줄이기 위해 샘플 가중치를 줄 수 있습니다.

- `--edge-weight`: 외곽 영역 가중치(기본 1.0, 예: 1.5~2.0)
- `--corner-weight`: 코너(외곽 x와 y 동시) 추가 가중치
- `--edge-margin-mm`: 외곽 판정 마진(mm), 기본 2.0

예시:
```bash
python3 -m training_seq.train_seq \
  ... \
  --edge-weight 1.8 --corner-weight 2.2 --edge-margin-mm 2.0
```
