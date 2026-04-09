# Tactile Sensor Training (2026-04-09)

16ch 촉각 센서로 다음을 학습·검증하기 위한 파이프라인입니다.
- XY 위치(0.5 mm grid), Z depth, Fz 회귀
- 선택적 depth-aware soft heatmap + z/fz 보조 헤드 (옵션 B)
- Force Field 재구성(heatmap) 멀티태스크

## 구조
```
training/
├─ data/       # Dataset 로더 (zarr/csv)
├─ models/     # MLP, CNN/LSTM, SATS, Transformer, GNN, Multi-head 등
├─ pipelines/  # 학습/평가 스크립트
├─ utils/      # 기하/라벨/시각화 유틸
└─ runs*/      # 결과 저장 경로
```

## 주요 파이프라인
- `pipelines/train_comparison.py` : 여러 모델 동시 학습/비교 (zarr 또는 csv)
- `pipelines/train_sr_zarr.py`   : SR 회귀(x,y,z) 단일 모델 (zarr)
- `pipelines/train_ff_zarr.py`   : Force Field/Fz 전용 (zarr)
- `pipelines/train_unified.py`   : 시퀀스 기반 통합 모델
- `pipelines/evaluate_comparison_heatmap.py`, `evaluate_sr.py` : 평가/히트맵 출력

## 데이터 요구
- 기본 경로: `preprocessing/processed_data/` 또는 `preprocessing/zarr_data/*.zarr`
- Zarr 필수 필드: `tactile_lr_norm`(N,16), `aux_feat[:,3]=diameter`, `cx, cy, depth_mm, fz`
- CSV 경로 사용 시 `x_mm, y_mm, z_mm, Fz` 컬럼 포함 권장

## 옵션 B (depth-aware soft heatmap) 사용법
- 플래그 (multi_head_field 전용):
  - 라벨 옵션: `--use-depth-aware-label`, `--depth-label-kernel gaussian|linear`, `--depth-radius-model hertz|geom`, `--heatmap-size 40`, `--depth-fallback-mm 1.0`, `--normalize-heatmap`
  - 손실/가중치: `--loss-xy bce|wmse --loss-z huber|mse --loss-fz huber|mse`, `--lambda-xy 1.0 --lambda-z 0.2 --lambda-fz 0.2`, `--fg-weight 5.0`, `--huber-delta 1.0`
  - 보조헤드 정규화: `--z-mean/--z-std`, `--fz-mean/--fz-std`
  - 디코드/시각화: `--decode-xy softargmax|argmax_refine`, `--save-heatmap-overlay`, `--overlay-batches`, `--overlay-samples`
  - 지표 bin: `--depth-bins "0.8,1.1,1.4,1.7"`
- 출력: `xy_heatmap logits`, `z_depth`, `fz`; decode 옵션 사용 시 xy는 heatmap에서 softargmax/argmax_refine로 계산해 검증.
- ckpt/tag: `dlabel-<kernel>-<radius>_xy*_z*_fz*_dec*_hnorm` 형태로 자동 부여, metrics JSON도 동일 태그로 저장.

## 권장 실험 순서 (Ablation)
1) Baseline: point label + xy only (`--use-depth-aware-label` off)
2) Stage2: depth-aware soft label + xy only (`--use-depth-aware-label` on, λ_z=λ_fz=0 또는 헤드 off)
3) Stage3: soft label + z/fz heads (멀티태스크, λ=1/0.2/0.2)
4) Optional: depth/force 입력 conditioning 추가 후 비교

## 예시 명령 (추천)
- Stage1 (baseline, point label)
```
python -m training.pipelines.train_comparison \
  --models multi_head_field \
  --epochs 100 --batch-size 512 --seq-len 50 \
  --decode-xy none
```
- Stage2 (soft label, xy만)
```
python -m training.pipelines.train_comparison \
  --models multi_head_field \
  --use-depth-aware-label \
  --depth-label-kernel gaussian --depth-radius-model hertz \
  --heatmap-size 40 --fg-weight 8.0 \
  --heatmap-sigma-scale 0.35 \
  --lambda-z 0.0 --lambda-fz 0.0 \
  --decode-xy softargmax \
  --depth-fallback-mm 1.0 \
  --depth-min-for-label 0.05 \
  --normalize-heatmap \
  --save-heatmap-overlay --overlay-batches 1 --overlay-samples 4 \
  --epochs 100 --batch-size 512
```
- Stage3 (soft label + z/fz 보조 헤드)
```
python -m training.pipelines.train_comparison \
  --models multi_head_field \
  --use-depth-aware-label \
  --loss-xy bce --loss-z huber --loss-fz huber \
  --lambda-xy 1.0 --lambda-z 0.2 --lambda-fz 0.2 \
  --depth-label-kernel gaussian --depth-radius-model hertz \
  --heatmap-size 40 --fg-weight 8.0 \
  --heatmap-sigma-scale 0.35 \
  --decode-xy softargmax \
  --depth-fallback-mm 1.0 \
  --depth-min-for-label 0.05 \
  --normalize-heatmap \
  --save-heatmap-overlay --overlay-batches 1 --overlay-samples 4 \
  --epochs 100 --batch-size 512
```

## 검증 체크리스트
- A/B: 기존 라벨 vs depth-aware 라벨 (동일 모델)
- 깊이 구간별 지표: MAE/RMSE, 성공률(≤1 cell), `metrics_*.json`에 bin별 기록
- 히트맵 overlay PNG 저장(`out_dir/overlays`)
- 데이터 순서 무작위화, 드리프트 보정 여부 확인
- Ablation 순서 기록: (1) point → (2) soft → (3) soft+z/fz → (4) conditioning

## 출력/로그
- ckpt: `training/runs_comparison/best_<model>[dlabel-*].pth`
- 지표: `comparison_results.json`
- 히트맵/시각화: `runs_comparison/heatmaps/*` (evaluate_comparison_heatmap)

## 추론(inference)
- 스크립트: `inference/run_inference.py`
- 사용 예:
```
python inference/run_inference.py \
  --checkpoint training/runs_comparison/best_multi_head_field_dlabel-gaussian-hertz_xybce_zhuber_fzhuber_decsoftargmax.pth \
  --data-dir preprocessing/processed_data \
  --decode-xy softargmax \
  --heatmap-sigma-scale 0.35 \
  --normalize-heatmap \
  --depth-fallback-mm 1.0 --depth-min-for-label 0.05 \
  --batch-size 256 --max-batches 2 \
  --save-heatmap-overlay --overlay-batches 1 --overlay-samples 4
```
- 출력: 표준출력에 MAE/RMSE[x,y,z,fz], `training/runs_comparison/inference_overlays/`에 pred/target heatmap PNG 저장.

## 버전
- Last Updated: 2026-04-09
- Version: 4.1 (Depth-Aware Option B)
