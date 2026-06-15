# 16-Channel Tactile Intelligence Framework

16채널 기압 기반 tactile sensor array 데이터로 sparse-to-dense pressure map,
XY heatmap, contact 기준 Z/Fz 회귀를 실험하는 workspace입니다.

현재 mk555 SATS 데이터의 공식 경로는 `skin_ws/raw_data`의 raw BIN archive를
`learning_data`로 정리한 뒤 `sats` 학습을 돌리는 흐름입니다. 기존
CSV/Zarr 기반 `hitmap` 경로는 XY/Z/Fz 회귀 실험용으로 유지됩니다.

## Current Official SATS Flow

```text
skin_ws/raw_data/sats/eco20 + mesh/d5/testN/
  ├── due_raw_burst_*.bin
  ├── ethermotion_encoder_*.bin
  └── loadcell_raw_*.bin

learning_data/
  ├── sensor_raw_bin/ecomesh/d5/z_2.5mm/testN/*_merged.bin
  └── gt/ecomesh_d5_z2.5_testN_targets.npy  # legacy/precomputed GT

sats/training/
  ├── train_lstm.py
  ├── train_attention.py
  ├── train_local_map.py
  └── train_cnn.py
```

## Data Alignment Policy

- DUE effective sensor stream is 200 Hz. One raw DUE burst contains 10 FIFO
  frames, so `bin_merge.py` expands bursts into the 200 Hz tactile stream.
- Loadcell is aligned to the same 200 Hz common timeline and converted with
  `Fz = (kg - kg_baseline) * 9.80665`.
- EtherMotion is logged much faster, around 1000 Hz or higher, and provides
  high-resolution `x/y/z/u` labels. It is interpolated onto the 200 Hz common
  timeline; the final training row count is not expanded to EtherMotion rate.
- `z_stage_mm` and `z_depth_mm` preserve EtherMotion command precision at about
  `0.0001 mm`; do not bin Z coarsely when the goal is `0.xx mm` behavior.
- `u_mm` is a node-internal wait/virtual axis. It is not physical shear and is
  not the depth source. SATS uses all rows by default.

This means the recommended learning row is:

```text
input[t] = DUE s1..s16 at 200 Hz
Fz[t]    = loadcell Fz interpolated to t
Z[t]     = EtherMotion z interpolated to t
GT[t]    = Boussinesq pressure map from x/y/Fz/diameter
```

## GT Modes

Two GT paths are available:

```text
precomputed  : legacy path, stores learning_data/gt/*_targets.npy
on_the_fly   : CPU worker path, generates dense GT during DataLoader fetch
gpu_on_the_fly: optimized path, sends compact GT metadata and builds batch GT on GPU
```

The default remains `precomputed`, so existing runs are unchanged. The new
on-the-fly modes avoid writing dense GT files of about 18 GB per d5 trial.
For `0.25 mm`, `0.2 mm`, or `0.1 mm` output grids, prefer
`gpu_on_the_fly`; the CPU `on_the_fly` mode is mainly a compatibility/debug
path for the original `41 x 41` grid.

Current on-the-fly GT assumptions:

```text
z_s_mm                  = 2.0 fixed
beta/FEM correction     = none
indenter model          = spherical
contact radius          = sqrt(R * z_depth_mm), R = diameter / 2
contact starts at d5    = z_depth_mm > 0.001
minimum contact radius  = 0.05 mm
z-depth sample bins     = 0.005 mm
pressure map grid       = 41 x 41, 0.5 mm XY spacing
```

The high-resolution EtherMotion Z signal is preserved in `merged.bin` at about
`0.0001 mm`, but training samples are balanced by `0.005 mm` z-depth bins to
avoid over-weighting repeated plateau rows.

Example on-the-fly training command:

First build compact metadata cache from merged BIN data. This does not save
dense pressure maps; it stores normalized sensor windows and compact
`diameter/x/y/z_depth/Fz` metadata.

```bash
python3 -m sats.training.build_gt_meta_cache \
  --raw-dir learning_data/sensor_raw_bin \
  --out-dir learning_data/gt_meta_cache \
  --exclude-diameters 10 \
  --grid-step-mm 0.5
```

Then train from that cache:

```bash
python3 -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir learning_data/gt_meta_cache \
  --z-depth-min-mm 0.001 \
  --z-balance-bin-width-mm 0.005 \
  --min-contact-radius-mm 0.05 \
  --exclude-diameters 10 \
  --run-name e2e_d5_onthefly
```

High-resolution grid examples:

```bash
# 0.25 mm virtual grid, 81 x 81 output
python3 -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir learning_data/gt_meta_cache \
  --exclude-diameters 10 \
  --grid-step-mm 0.25 \
  --batch-size 2048 \
  --num-workers 6 \
  --prefetch-factor 2 \
  --run-name e2e_d5_gpu_gt_g025

# 0.1 mm virtual grid, 201 x 201 output. Start smaller because target memory
# grows about 24x compared with 41 x 41.
python3 -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir learning_data/gt_meta_cache \
  --exclude-diameters 10 \
  --grid-step-mm 0.1 \
  --batch-size 1024 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --run-name e2e_d5_gpu_gt_g010
```

Use a denser virtual XY pressure map by changing the GT/model grid, not by
recollecting raw data. For example, `81 x 81` over the same `[-10, 10] mm` area
gives `0.25 mm` virtual taxel spacing. The current raw scan centers are still
collected every `0.5 mm`, so finer XY output improves map resolution but does
not create new sub-0.5mm press-center labels by itself.

In `train_e2e.py`, `--local-map-size 0` is the default. It automatically scales
the local decoder's physical footprint with `--grid-step-mm`: `0.5 mm -> 15`,
`0.25 mm -> 29`, `0.2 mm -> 37`, and `0.1 mm -> 71`.

## Build Learning Data

Run from repository root:

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --learning-root learning_data
```

The script discovers raw BIN trial folders, assigns stable `testN` numbers via
`learning_data/trial_registry.json`, writes merged BIN files, and generates
Boussinesq pressure-map GT files under `learning_data/gt`.

To preview what will be processed without writing:

```bash
python3 sats/preprocessing/prepare_learning_data.py --dry-run --stage all
```

To process only one new trial, use the planned output from dry-run and run the
merge/GT tools against that specific trial, or run `prepare_learning_data.py`
after confirming the registry mapping. Existing `testN` numbers are append-only.

For on-the-fly GT training, dense `learning_data/gt/*_targets.npy` files are not
required. You still need the merged BIN and baseline artifacts under
`learning_data/sensor_raw_bin`.

## Raw BIN Sufficiency Check

Before running long training, inspect whether the raw BIN archive has enough
usable data and consistent coverage.

Quick file/record check:

```bash
python3 sats/tools/analyze_raw_bins.py \
  --source-root skin_ws/raw_data \
  --source-material "eco20 + mesh" \
  --diameter d5 \
  --out sats/tools/raw_bin_sufficiency_quick.csv
```

Full distribution check:

```bash
python3 sats/tools/analyze_raw_bins.py \
  --source-root skin_ws/raw_data \
  --source-material "eco20 + mesh" \
  --diameter d5 \
  --full \
  --out sats/tools/raw_bin_sufficiency_full.csv
```

The full report builds the same 200 Hz merged rows in memory and summarizes:

```text
source stream row counts and rates
merged row count and duration
covered XY cells
per-XY sequence length distribution
z_depth range
Fz distribution
active contact row ratio
```

Use this report to confirm that each trial reaches the expected `2.5 mm` d5
depth, has comparable force distribution, covers all `41 x 41` XY points, and
does not have abnormal sequence lengths or missing streams.

## Current mk555 d5 Dataset

As of 2026-06-09, the current SATS d5 learning data contains:

```text
ecomesh_d5_z2.5_test1
ecomesh_d5_z2.5_test2
```

Both trials have been checked for SATS training alignment:

```text
test1 on-grid rows = 2,743,978 = GT rows
test2 on-grid rows = 2,743,016 = GT rows
trial count        = 2
sequence count     = 3,362  # 1,681 XY points per trial
```

GT peak positions match the row `(x_mm, y_mm)` coordinates in sampled checks.

## SATS Training

Default training follows the paper-style SATS data contract:

```text
raw cycle cap:  seq_len = 1000      # keeps loading peak around timestep 820-860
sample input:   sensor_window [B, 10, 16]
sample target:  pressure map  [B, 41, 41] at the window's last timestep
split:          random sequence-level train/val split, val_ratio = 0.2
```

`--use-window-dataset` is enabled by default. Use `--no-use-window-dataset`
only for the older peak-map experiment:

```text
old mode: sensor_seq [B, 1000, 16] -> peak pressure map [B, 41, 41]
```

Smoke test:

```bash
python3 -m sats.training.train_lstm \
  --run-name smoke_lstm_window10 \
  --epochs 1 \
  --batch-size 256 \
  --num-workers 0 \
  --device cuda
```

Recommended LSTM training on RTX 4090:

```bash
python3 -m sats.training.train_lstm \
  --run-name lstm_d5_test12_window10_bs2048 \
  --epochs 50 \
  --batch-size 2048 \
  --num-workers 2 \
  --device cuda
```

Why `batch-size=2048`: window mode transfers only one `[41,41]` target map per
sample instead of a full `[1000,41,41]` GT sequence, so the 4090 has enough VRAM
headroom. Keep `num-workers=2` as the default while the dataset grows to 10
sets; more workers can increase mmap/RAM pressure without improving epoch time.

Continue the staged SATS pipeline:

```bash
python3 -m sats.training.train_attention \
  --lstm-ckpt sats/training/runs/lstm_d5_test12_window10_bs2048/best_model.pt \
  --run-name attn_d5_test12_window10_bs2048 \
  --epochs 50 \
  --batch-size 2048 \
  --num-workers 2 \
  --device cuda

python3 -m sats.training.train_local_map \
  --attn-ckpt sats/training/runs/attn_d5_test12_window10_bs2048/best_model.pt \
  --run-name local_map_d5_test12_window10_bs2048 \
  --epochs 50 \
  --batch-size 2048 \
  --num-workers 2 \
  --device cuda

python3 -m sats.training.train_cnn \
  --local-map-ckpt sats/training/runs/local_map_d5_test12_window10_bs2048/best_model.pt \
  --run-name cnn_d5_test12_window10_bs2048 \
  --epochs 50 \
  --batch-size 2048 \
  --num-workers 2 \
  --device cuda
```

Batch tuning rule for 2 to 10 sets:

```text
default:       batch_size=2048, num_workers=2
RAM pressure:  batch_size=1024, num_workers=1 or 0
stable/idle:   batch_size=4096, num_workers=2
```

Judge tuning by epoch time and RAM stability, not by whether instantaneous GPU
utilization stays fixed at 90%.

## Legacy / Alternate Paths

- `hitmap/` contains the newer CSV/Zarr based XY heatmap and Z/Fz regressor
  experiments.
- `sats/preprocessing/raw_merge.py` and CSV exports are compatibility surfaces.
  For current mk555 SATS data, prefer raw BIN -> merged BIN.

## Directory Map

- `skin_ws/`: raw acquisition archive, node files, acquisition scripts.
- `learning_data/`: managed SATS merged BIN and GT workspace.
- `sats/`: SATS preprocessing, GT generation, training, and references.
- `hitmap/`: Zarr/heatmap/Z-Fz experimental training pipelines.
- `history/`: dated implementation and experiment notes.
