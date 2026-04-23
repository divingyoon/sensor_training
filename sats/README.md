# SATS

SATS는 sparse tactile sensor array의 16개 물리 센서 신호로부터 40x40 virtual tactile grid 압력맵을 추론하기 위한 실험 파이프라인이다. 구조는 논문 *Super-resolution tactile sensor arrays with sparse units enabled by deep learning*의 SATS 흐름을 현재 센서 데이터 구조에 맞춘 것이다.

현재 구현은 크게 세 단계로 나뉜다.

1. `raw_data`의 실험 스트림을 trial별 `*_merged.csv`로 병합한다.
2. 병합 CSV에서 baseline, on-grid row, 정규화 feature, Boussinesq GT 압력맵을 만든다.
3. `sats.training` 모듈로 LSTM, self-attention, local map, CNN refining 단계를 순차 학습한다.

## 센서 및 데이터 전제

센서는 25 x 25 x 3 mm 구조의 FPCB 위에 BMP384 16개를 4x4 배열로 배치한 tactile sensor array이다. 물리 센서 간 중심 간격은 6.5 mm이며, 전체 sensing area는 다음 좌표계를 사용한다.

```text
x = [-9.75, 9.75] mm
y = [-9.75, 9.75] mm
scan grid step = 0.5 mm
virtual grid = 40 x 40
```

센서 번호와 좌표는 `sats/ref/20260420_Explain Sensor.md` 기준이다.

| row | 센서 | y [mm] | x [mm] |
| --- | --- | --- | --- |
| 1 | S4, S3, S2, S1 | -9.75 | -9.75, -3.25, 3.25, 9.75 |
| 2 | S8, S7, S6, S5 | -3.25 | -9.75, -3.25, 3.25, 9.75 |
| 3 | S12, S11, S10, S9 | 3.25 | -9.75, -3.25, 3.25, 9.75 |
| 4 | S16, S15, S14, S13 | 9.75 | -9.75, -3.25, 3.25, 9.75 |

실험은 3-axis stage로 `x, y`를 0.5 mm 간격으로 이동하며 각 grid point에서 `z` 방향 압입을 수행한다. `z_max`는 보통 1.0 mm 또는 1.5 mm이고, 인덴터 직경은 trial 경로의 `d5`, `d10`처럼 관리한다.

권장 raw data 구조는 다음과 같다.

```text
raw_data/
└── <material>/
    └── d<D>/
        └── z_<Z>mm/
            └── test<N>/
                ├── due*.csv
                ├── ethermotion*.csv 또는 eithermotion*.csv
                ├── afd*.csv
                ├── <trial_id>_merged.csv
                ├── <trial_id>_baseline.json
                └── <trial_id>_merge_summary.json
```

예:

```text
raw_data/ecomesh/d5/z_1.5mm/test9/ecomesh_d5_z1.5_test9_merged.csv
```

학습에서 사용하는 `trial_id`는 다음 형식이다.

```text
<material>_d<D>_z<Z>_test<N>
예: ecomesh_d10_z1_test3, ecomesh_d5_z1.5_test9
```

## 전체 구조

```text
sats/
├── README.md
├── ref/
│   ├── 20260420_Explain Sensor.md
│   ├── 20260420_GT.md
│   └── Super-resolution tactile sensor arrays with sparse units enabled by deep learning/
├── preprocessing/
│   ├── README.md
│   ├── raw_merge.py
│   ├── preprocess.py
│   ├── generate_gt.py
│   ├── label_preview.py
│   └── center_probe_report.csv
└── training/
    ├── config.py
    ├── dataset.py
    ├── lstm_module.py
    ├── attention_module.py
    ├── local_map_module.py
    ├── cnn_module.py
    ├── train_lstm.py
    ├── train_attention.py
    ├── train_local_map.py
    ├── train_cnn.py
    ├── tests/
    └── runs/
```

## `sats/ref`

참고 문서 디렉터리이다.

`20260420_Explain Sensor.md`는 센서 사양, 16개 센서 좌표, 데이터 취득 방식, SATS 구성 목표를 정리한다. README의 센서 좌표와 데이터 구조 설명은 이 문서를 기준으로 한다.

`20260420_GT.md`는 현재 실험 데이터에서 GT를 어떻게 정의할지 설명한다. 핵심은 한 측정 row마다 `40 x 40` 수직응력맵을 만들고, 이 맵을 SATS 학습 label로 사용하는 것이다.

논문 정리 문서는 LSTM, self-attention, elastic half-space model, Boussinesq 기반 GT 생성, rectification factor `beta(p)`의 배경을 설명한다. 현재 `generate_gt.py`의 `--beta-mode poly2` 옵션은 이 S9 rectification 개념을 코드 옵션으로 둔 것이다.

## `sats/preprocessing`

전처리와 GT 생성을 담당한다.

### `raw_merge.py`

`due`, `ethermotion`, `afd` 세 raw stream을 timestamp 기준으로 정렬해 trial별 `*_merged.csv`를 생성한다.

입력:

```text
due*.csv          센서 raw count, Skin1~Skin16
ethermotion*.csv 3-axis stage position
afd*.csv         force/torque sensor, Fx/Fy/Fz
```

출력:

```text
<trial_id>_merged.csv
<trial_id>_baseline.json
<trial_id>_merge_summary.json
```

기본 실행:

```bash
python3 sats/preprocessing/raw_merge.py \
  --raw-root raw_data
```

resample 기반 동기화와 품질 게이트를 명시한 실행:

```bash
python3 sats/preprocessing/raw_merge.py \
  --raw-root raw_data \
  --align-mode resample \
  --sync-ref ethermotion \
  --resample-hz 100 \
  --window-ms 10 \
  --window-agg median \
  --max-dt-ms 10 \
  --min-match-ratio 0.9 \
  --force-round-dp 2
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--raw-root` | `/home/user/sensor_training/preprocessing/raw_data` | trial 폴더가 들어있는 루트. 이 repo에서는 보통 `raw_data`를 사용한다. |
| `--sync-ref` | `ethermotion` | 병합 timeline 기준 stream. `due`, `ethermotion`, `afd` 중 선택. |
| `--align-mode` | `resample` | `nearest`는 최근접 timestamp 병합, `resample`은 공통 sampling rate로 보간. |
| `--resample-hz` | `100.0` | resample 기준 Hz. AFD가 100 Hz이면 100 권장. |
| `--window-ms` | `10.0` | resample 이후 centered rolling smoothing window. `0`이면 비활성. |
| `--window-agg` | `median` | rolling window 집계 방식. |
| `--max-dt-ms` | `10.0` | source stream과의 최근접 시간 차이가 이 값을 넘는 row 제거. |
| `--lag-due-ms` | `0.0` | DUE stream 전역 lag 보정. |
| `--lag-ethermotion-ms` | `0.0` | Ethermotion stream 전역 lag 보정. |
| `--lag-afd-ms` | `0.0` | AFD stream 전역 lag 보정. |
| `--baseline-fallback-sec` | `2.0` | XYZ=0 head block이 없을 때 baseline으로 사용할 앞부분 시간. |
| `--min-match-ratio` | `0.9` | due/ethermotion/afd match ratio 품질 게이트. |
| `--force-round-dp` | `2` | `Fx`, `Fy`, `Fz` 반올림 자릿수. 음수면 비활성. |

### `preprocess.py`

`*_merged.csv`에서 baseline 보정, on-grid 필터링, `z_depth`, loading/unloading phase, 정규화 feature, 소재별 Zarr 데이터셋을 만든다.

입력:

```text
raw_data/**/**/*_merged.csv
```

출력:

```text
sats/preprocessing/processed_data/
├── baselines/       <trial_id>_baselines.json
├── grid/            <trial_id>_grid.csv
├── features/        <trial_id>_features.csv
├── peak_summary/    <material>_peak_summary.csv
├── zarr_data/       dataset_<material>.zarr/
└── <material>_features.csv
```

실행:

```bash
python3 sats/preprocessing/preprocess.py \
  --raw-dir raw_data \
  --out-dir sats/preprocessing/processed_data
```

병렬 처리와 depth-aware contact radius를 켠 실행:

```bash
python3 sats/preprocessing/preprocess.py \
  --raw-dir raw_data \
  --out-dir sats/preprocessing/processed_data \
  --workers 4 \
  --z-bin-mm 0.02 \
  --min-reliable-s 0.005 \
  --use-depth-aware-radius \
  --radius-model hertz
```

현재 `sats/preprocessing/preprocess.py`는 `training.utils.contact_geometry`를 import한다. 해당 유틸은 현재 repo의 `hitmap/training/utils`에 있으므로, import 오류가 나면 `hitmap` 쪽 전처리 모듈을 사용하거나 import 경로를 정리해야 한다.

주요 처리 단계:

```text
merged CSV
  ├── baseline 구간 탐색
  ├── x/y 0.5 mm grid + stage stationary row 필터링
  ├── z_stage_mm, z_contact_mm, z_depth_mm 계산
  ├── loading/unloading phase 분류
  ├── raw grid CSV 저장
  ├── s_norm_i = (s_i - baseline_i) / baseline_i 생성
  ├── z binning 및 저신호 row 제거
  └── trial별 feature CSV, 소재별 통합 CSV, Zarr 저장
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--raw-dir` | `preprocessing/raw_data` | `*_merged.csv`를 찾을 루트. 이 repo에서는 보통 `raw_data` 지정. |
| `--out-dir` | `preprocessing/processed_data` | 전처리 결과 저장 디렉터리. |
| `--glob` | `**/*_merged.csv` | 탐색할 merged CSV glob. |
| `--workers` | `1` | trial 병렬 처리 프로세스 수. |
| `--contact-threshold` | `0.01` | 접촉 판정용 `s_norm` 임계값. |
| `--no-zarr` | off | Zarr 저장 생략. |
| `--z-bin-mm` | `0.02` | `z_depth` binning 간격. `0` 이하면 비활성. |
| `--min-signal` | auto | 저신호 row 제거 임계값. 미지정 시 baseline noise 기반 자동 추정. |
| `--min-reliable-s` | `0.005` | 소재별 일관성 필터에서 좌표별 최소 신호 기준. |
| `--baseline-z-thresh` | `0.001` | baseline 탐색 시 허용 `|z_mm|`. |
| `--baseline-force-thresh` | `0.5` | baseline 탐색 시 허용 `|Fz|`. `0`이면 무시. |
| `--baseline-min-consec` | `40` | baseline으로 인정할 최소 연속 row 수. |
| `--use-depth-aware-radius` | off | `contact_radius_mm`, `contact_radius_cell` 생성. |
| `--radius-model` | `hertz` | 접촉 반경 계산 모델. `hertz` 또는 `geo`. |
| `--fallback-depth-mode` | `none` | depth가 0 이하일 때 대체 방식. `none`, `mean`, `const`. |
| `--export-label-heatmap` | off | depth 기반 label heatmap PNG 샘플 생성. |

### `generate_gt.py`

각 merged CSV row의 `(x_mm, y_mm, Fz, d)`로부터 Boussinesq elastic half-space 기반 `40 x 40` GT 압력맵을 생성한다.

출력:

```text
sats/preprocessing/gt_output_v1/
├── <trial_id>_targets.npy
├── <trial_id>_gt_meta.json
└── dataset_index.json
```

기본 실행:

```bash
python3 -m sats.preprocessing.generate_gt \
  --raw-dir raw_data \
  --out-dir sats/preprocessing/gt_output_v1 \
  --z-s 2.0 \
  --patch-step 0.1
```

논문 S9의 rectification factor 근사와 위치별 유효 깊이 보정을 켠 실행:

```bash
python3 -m sats.preprocessing.generate_gt \
  --raw-dir raw_data \
  --out-dir sats/preprocessing/gt_output_v2 \
  --z-s 2.0 \
  --patch-step 0.1 \
  --fz-mode abs \
  --fz-min-abs 0.05 \
  --beta-mode poly2 \
  --beta-c0 1.0 \
  --beta-c1 0.0 \
  --beta-c2 0.0 \
  --z-comp-mode xy_contact \
  --z-contact-force-thresh 0.2 \
  --z0-estimator p05
```

GT 정의:

```text
GT_t(u, v) = sigma_zz(u, v, z_s)
```

여기서 `(u, v)`는 40x40 virtual grid 위치, `z_s`는 센서 유효 깊이, `sigma_zz`는 Boussinesq 수직응력이다. 원형 인덴터 patch 내부 압력을 균일하다고 두고 patch를 작은 점하중으로 이산화한 뒤 각 grid cell에서 응력을 합산한다.

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--raw-dir` | repo `raw_data` | `*_merged.csv`를 재귀 탐색할 루트. |
| `--out-dir` | `sats/preprocessing/gt_output` | GT npy와 meta 저장 디렉터리. 학습 기본값은 `gt_output_v1`이므로 명시 권장. |
| `--z-s` | `2.0` | 센서 유효 깊이 [mm]. |
| `--patch-step` | `0.1` | 원형 접촉 patch 이산화 간격 [mm]. 작을수록 정밀하지만 느림. |
| `--grid-size` | `40` | GT 맵 한 변 크기. 현재 40만 지원. |
| `--grid-mode` | `scan_points` | `scan_points`: -9.75부터 0.5 mm 간격. `cell_centers`: cell 중심 좌표. |
| `--fz-mode` | `positive_only` | GT 하중 변환 방식. `positive_only`, `abs`, `signed`. |
| `--fz-min-abs` | `0.05` | 이보다 작은 `|Fz|`는 비접촉/노이즈로 간주해 GT 0. |
| `--beta-mode` | `none` | `none` 또는 `poly2`. `poly2`는 `beta(p)=c0+c1*p+c2*p^2`. |
| `--z-comp-mode` | `none` | `xy_contact` 사용 시 위치별 접촉 시작 높이로 `z_eff` 보정. |
| `--z0-estimator` | `p05` | `xy_contact`에서 z0 추정 방식. `min`, `p05`, `first`. |
| `--grid-tol-mm` | `0.05` | scan grid 정합 허용 오차. |
| `--keep-offgrid` | off | off-grid row를 버리지 않고 최근접 grid로 snap. |

### `label_preview.py`

`*_grid.csv`에서 깊이가 큰 sample을 골라 depth 기반 contact radius label heatmap을 PNG로 저장하는 검증용 스크립트이다. 주 파이프라인의 GT는 `generate_gt.py`가 만든다.

실행:

```bash
python3 sats/preprocessing/label_preview.py \
  --grid-file sats/preprocessing/processed_data/grid/ecomesh_d5_z1.5_test9_grid.csv \
  --out-dir sats/preprocessing/processed_data/label_preview \
  --samples 3 \
  --kernel gaussian \
  --radius-model hertz \
  --indenter-radius-mm 2.5
```

## `sats/training`

SATS 학습 모듈이다. 현재 구현은 논문 흐름에 맞춰 네 단계로 나뉜다.

```text
sensor_seq [B, T, 16]
  ├── LSTM encoder
  ├── Self-Attention
  ├── Local Map Decoder
  └── CNN Refiner
      └── refined_map [B, 40, 40]
```

LSTM stage와 self-attention stage는 중간 단계 학습을 위해 proxy decoder로 `40 x 40` peak GT map을 직접 예측한다. Local map stage부터는 논문 구조에 가까운 방식으로 센서별 local pressure map을 만들고, CNN stage에서 병합된 pressure map을 2-layer CNN으로 정제한다.

### `config.py`

공통 설정 dataclass `SATSConfig`와 trial path helper를 제공한다.

중요 기본값:

```text
raw_dir = raw_data
gt_dir = sats/preprocessing/gt_output_v1
dataset_index_path = sats/preprocessing/gt_output_v1/dataset_index.json
out_dir = sats/training/runs
grid = 40 x 40, [-9.75, 9.75], step 0.5 mm
seq_len = 400
local_map_size = 15
cnn_hidden_channels = 16
val_trials = ecomesh_d10_z1_test3, ecomesh_d5_z1.5_test9
device = cuda
```

주의: `trial_id_to_paths()`는 baseline을 raw trial 폴더의 `<trial_id>_baseline.json`에서 읽는다. `raw_merge.py`가 만든 baseline JSON이 있어야 학습 dataset이 로드된다.

### `dataset.py`

`dataset_index.json`의 trial 목록을 읽고, 각 trial을 `(x_mm, y_mm)` grid 위치별 시계열 sample로 만든다.

한 sample:

```text
sensor_seq: [T, 16]      s_norm 시계열
gt_seq:     [T, 40, 40]  같은 row index의 GT 압력맵 시계열
length:     T
```

batch:

```text
sensor_batch: [B, T_max, 16]
gt_batch:     [B, T_max, 40, 40]
lengths:      [B]
```

학습 target은 전체 시계열 중 GT 총합이 가장 큰 timestep의 `40 x 40` map이다.

### `lstm_module.py`

16개 센서 각각에 독립 LSTM encoder를 둔다. 센서마다 응답 특성과 hysteresis가 다르다는 SATS 논문 가정을 반영한 구조이다.

```text
sensor_seq [B, T, 16]
  └── LSTM_1 ... LSTM_16
      └── local_feat [B, 16, hidden_dim]
          └── proxy decoder
              └── pred_map [B, 40, 40]
```

### `attention_module.py`

4x4 센서 grid에서 8-connected neighbor와 self-loop를 사용하는 graph attention 방식 self-attention을 구현한다.

```text
local_feat [B, 16, hidden_dim]
  ├── W projection
  ├── neighbor attention alpha_ij
  └── agg_feat [B, 16, attn_dim]
```

현재 attention stage는 LSTM encoder checkpoint를 로드하고 encoder를 freeze한 뒤, self-attention과 proxy decoder를 학습한다.

### `local_map_module.py`

LSTM feature와 self-attention feature를 센서별로 concat한 뒤, 공유 MLP 디코더로 센서별 local pressure map을 만든다. 16개 local map은 센서 물리 좌표에 맞춰 `40 x 40` 전체 맵에 배치되고 합산된다.

```text
local_feat [B, 16, hidden_dim]
agg_feat   [B, 16, attn_dim]
  └── combined_feat [B, 16, hidden_dim + attn_dim]
      └── shared MLP g_phi
          └── local_maps [B, 16, local_map_size, local_map_size]
              └── placement + sum
                  └── merged_map [B, 40, 40]
```

현재 기본 `local_map_size`는 `15`이다. 센서 중심은 6.5 mm 간격의 4x4 물리 배열을 `40 x 40` virtual grid index로 변환해 사용한다. 경계 센서의 local map이 전체 맵 범위를 벗어나는 부분은 clip하고, 실제 겹치는 영역만 합산한다.

학습 시 `train_local_map.py`는 attention checkpoint에서 `encoder.*`, `attention.*` 가중치를 로드한 뒤 두 모듈을 freeze하고 `local_map_decoder`만 학습한다.

### `cnn_module.py`

Local Map Decoder가 만든 `merged_map`을 입력으로 받아 2-layer CNN으로 최종 pressure map을 정제한다.

```text
merged_map [B, 40, 40]
  └── Conv2d(1 -> C, 3x3, padding=1) + LeakyReLU
      └── Conv2d(C -> 1, 3x3, padding=1)
          └── refined_map [B, 40, 40]
```

현재 기본 `cnn_hidden_channels`는 `16`이다. 학습 시 `train_cnn.py`는 local map checkpoint에서 `encoder.*`, `attention.*`, `local_map_decoder.*` 가중치를 로드한 뒤 세 모듈을 freeze하고 `cnn_refiner`만 학습한다.

## 학습 CLI

### LSTM stage 학습

먼저 GT를 만든 뒤 LSTM stage를 학습한다.

```bash
python3 -m sats.training.train_lstm \
  --raw-dir raw_data \
  --gt-dir sats/preprocessing/gt_output_v1 \
  --out-dir sats/training/runs \
  --run-name lstm_v1 \
  --epochs 50 \
  --batch-size 64 \
  --hidden-dim 64 \
  --num-layers 2 \
  --seq-len 400 \
  --device cuda \
  --val-trials ecomesh_d10_z1_test3 ecomesh_d5_z1.5_test9
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--raw-dir` | `raw_data` | merged CSV와 baseline JSON이 있는 raw data 루트. |
| `--gt-dir` | `sats/preprocessing/gt_output_v1` | `<trial_id>_targets.npy`와 `dataset_index.json` 위치. |
| `--out-dir` | `sats/training/runs` | run 결과 저장 루트. |
| `--run-name` | `lstm_v1` | 저장 run 이름. |
| `--epochs` | `50` | 학습 epoch 수. |
| `--batch-size` | `64` | DataLoader batch size. |
| `--lr` | `1e-3` | Adam learning rate. |
| `--hidden-dim` | `64` | 센서별 LSTM hidden size. |
| `--num-layers` | `2` | LSTM layer 수. |
| `--dropout` | `0.1` | LSTM dropout. |
| `--seq-len` | `400` | 위치별 시계열 최대 길이. 초과 시 앞부분 사용. |
| `--num-workers` | `4` | DataLoader worker 수. |
| `--device` | `cuda` | `cuda` 또는 `cpu`. CUDA 불가 시 config에서 CPU로 폴백. |
| `--val-trials` | 2개 기본 hold-out | 검증 trial 목록. |

출력:

```text
sats/training/runs/lstm_v1/
├── config.json
├── best_model.pt
├── last_model.pt
├── epoch_0010.pt
└── history.json
```

### Self-attention stage 학습

LSTM stage의 `best_model.pt`를 encoder 초기값으로 사용한다.

```bash
python3 -m sats.training.train_attention \
  --lstm-ckpt sats/training/runs/lstm_v1/best_model.pt \
  --raw-dir raw_data \
  --gt-dir sats/preprocessing/gt_output_v1 \
  --out-dir sats/training/runs \
  --run-name attn_v1 \
  --epochs 50 \
  --batch-size 64 \
  --hidden-dim 64 \
  --attn-dim 64 \
  --num-layers 2 \
  --seq-len 400 \
  --device cuda \
  --val-trials ecomesh_d10_z1_test3 ecomesh_d5_z1.5_test9
```

추가 주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--lstm-ckpt` | `""` | 사전학습된 LSTM checkpoint. 지정하면 encoder weight를 로드하고 freeze. |
| `--attn-dim` | `64` | self-attention projection dimension. |
| `--run-name` | `attn_v1` | attention run 저장 이름. |

출력:

```text
sats/training/runs/attn_v1/
├── config.json
├── best_model.pt
├── last_model.pt
├── epoch_0010.pt
└── history.json
```

### Local map stage 학습

Self-attention stage의 `best_model.pt`를 encoder와 attention 초기값으로 사용한다.

```bash
python3 -m sats.training.train_local_map \
  --attn-ckpt sats/training/runs/attn_v1/best_model.pt \
  --raw-dir raw_data \
  --gt-dir sats/preprocessing/gt_output_v1 \
  --out-dir sats/training/runs \
  --run-name local_map_v1 \
  --epochs 50 \
  --batch-size 64 \
  --hidden-dim 64 \
  --attn-dim 64 \
  --local-map-size 15 \
  --num-layers 2 \
  --seq-len 400 \
  --device cuda \
  --val-trials ecomesh_d10_z1_test3 ecomesh_d5_z1.5_test9
```

추가 주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--attn-ckpt` | `""` | 사전학습된 attention checkpoint. 지정하면 encoder와 attention weight를 로드하고 freeze. |
| `--local-map-size` | `15` | 센서별 local map 한 변 크기. 중앙 정렬을 위해 홀수 권장. |
| `--run-name` | `local_map_v1` | local map run 저장 이름. |

출력:

```text
sats/training/runs/local_map_v1/
├── config.json
├── best_model.pt
├── last_model.pt
├── epoch_0010.pt
└── history.json
```

### CNN refining stage 학습

Local map stage의 `best_model.pt`를 encoder, attention, local map decoder 초기값으로 사용한다.

```bash
python3 -m sats.training.train_cnn \
  --local-map-ckpt sats/training/runs/local_map_v1/best_model.pt \
  --raw-dir raw_data \
  --gt-dir sats/preprocessing/gt_output_v1 \
  --out-dir sats/training/runs \
  --run-name cnn_v1 \
  --epochs 50 \
  --batch-size 64 \
  --hidden-dim 64 \
  --attn-dim 64 \
  --local-map-size 15 \
  --cnn-hidden-channels 16 \
  --num-layers 2 \
  --seq-len 400 \
  --device cuda \
  --val-trials ecomesh_d10_z1_test3 ecomesh_d5_z1.5_test9
```

추가 주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--local-map-ckpt` | `""` | 사전학습된 local map checkpoint. 지정하면 encoder, attention, local map decoder weight를 로드하고 freeze. |
| `--cnn-hidden-channels` | `16` | CNN refiner의 중간 channel 수. |
| `--run-name` | `cnn_v1` | CNN run 저장 이름. |

출력:

```text
sats/training/runs/cnn_v1/
├── config.json
├── best_model.pt
├── last_model.pt
├── epoch_0010.pt
└── history.json
```

## 권장 실행 순서

환경 준비:

```bash
conda create -n sensor python=3.10
conda activate sensor
pip install -r requirements.txt
```

1. raw stream 병합:

```bash
python3 sats/preprocessing/raw_merge.py \
  --raw-root raw_data \
  --align-mode resample \
  --sync-ref ethermotion \
  --resample-hz 100 \
  --window-ms 10 \
  --window-agg median \
  --max-dt-ms 10
```

2. GT 생성:

```bash
python3 -m sats.preprocessing.generate_gt \
  --raw-dir raw_data \
  --out-dir sats/preprocessing/gt_output_v1 \
  --z-s 2.0 \
  --patch-step 0.1 \
  --fz-mode positive_only \
  --fz-min-abs 0.05
```

3. LSTM stage 학습:

```bash
python3 -m sats.training.train_lstm \
  --raw-dir raw_data \
  --gt-dir sats/preprocessing/gt_output_v1 \
  --run-name lstm_v1 \
  --epochs 50 \
  --device cuda
```

4. Self-attention stage 학습:

```bash
python3 -m sats.training.train_attention \
  --lstm-ckpt sats/training/runs/lstm_v1/best_model.pt \
  --raw-dir raw_data \
  --gt-dir sats/preprocessing/gt_output_v1 \
  --run-name attn_v1 \
  --epochs 50 \
  --device cuda
```

5. Local map stage 학습:

```bash
python3 -m sats.training.train_local_map \
  --attn-ckpt sats/training/runs/attn_v1/best_model.pt \
  --raw-dir raw_data \
  --gt-dir sats/preprocessing/gt_output_v1 \
  --run-name local_map_v1 \
  --epochs 50 \
  --device cuda
```

6. CNN refining stage 학습:

```bash
python3 -m sats.training.train_cnn \
  --local-map-ckpt sats/training/runs/local_map_v1/best_model.pt \
  --raw-dir raw_data \
  --gt-dir sats/preprocessing/gt_output_v1 \
  --run-name cnn_v1 \
  --epochs 50 \
  --device cuda
```

7. smoke test:

```bash
python3 -m pytest sats/training/tests
```

## 자주 확인할 문제

`dataset_index.json`과 `*_targets.npy`는 같은 `gt_dir`에 있어야 한다. 학습 기본값은 `sats/preprocessing/gt_output_v1`이므로 GT 생성 때 `--out-dir sats/preprocessing/gt_output_v1`을 명시하는 것이 안전하다.

학습 dataset은 raw trial 폴더의 `<trial_id>_merged.csv`, `<trial_id>_baseline.json`과 GT 폴더의 `<trial_id>_targets.npy`를 함께 요구한다. 셋 중 하나라도 없으면 해당 trial은 건너뛴다.

GT 생성에서 `--keep-offgrid`를 켜면 off-grid row도 최근접 grid로 snap되어 row 수가 달라질 수 있다. `dataset.py`는 on-grid mask로 raw CSV를 다시 필터링하므로, 학습 row 수 일치를 위해 기본값인 off-grid drop을 권장한다.

`z` 좌표 정의는 두 가지가 있다. `z_stage_mm`는 stage 원본 z이고, `z_contact_mm`는 각 `(x, y)` 위치별 접촉 시작점을 0으로 맞춘 depth이다. 전처리 feature/Zarr에서는 `z_contact_mm`를 우선 depth로 사용하지만, GT 생성 기본값은 일정한 `z_s`를 사용한다.

현재 attention module의 sensor indexing은 `S1, S2, S3, S4` 순서의 4x4 grid를 가정한다. 참고 문서의 좌표 표는 행 안에서 `S4, S3, S2, S1`처럼 좌표순으로 적혀 있으므로, sensor layout을 바꿀 때는 `attention_module.py`의 adjacency 가정과 입력 channel 순서를 같이 검토해야 한다.
