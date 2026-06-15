# SATS Learning Data

`learning_data` is the managed SATS training-data workspace. Raw acquisition
archives stay under `skin_ws/raw_data`; SATS GT generation and training should
read only the artifacts collected here.

## Layout

```text
learning_data/
├── sensor_raw_bin/
│   └── ecomesh/
│       └── d5/
│           └── z_2.5mm/
│               ├── test1/
│               │   ├── ecomesh_d5_z2.5_test1_merged.bin
│               │   ├── ecomesh_d5_z2.5_test1_baseline.json
│               │   └── ecomesh_d5_z2.5_test1_merge_summary.json
│               └── test2/
│                   ├── ecomesh_d5_z2.5_test2_merged.bin
│                   ├── ecomesh_d5_z2.5_test2_baseline.json
│                   └── ecomesh_d5_z2.5_test2_merge_summary.json
├── gt/
│   ├── ecomesh_d5_z2.5_test1_targets.npy
│   ├── ecomesh_d5_z2.5_test1_gt_meta.json
│   ├── ecomesh_d5_z2.5_test2_targets.npy
│   ├── ecomesh_d5_z2.5_test2_gt_meta.json
│   └── dataset_index.json
├── trial_registry.json
└── README.md
```

Large artifacts under `sensor_raw_bin/` and `gt/` are git-ignored.
`trial_registry.json` is small and tracked because it pins source folders to
stable `testN` numbers.

## Build

Build or update from the current mk555 source archive:

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --learning-root learning_data
```

Preview without writing:

```bash
python3 sats/preprocessing/prepare_learning_data.py --dry-run --stage all
```

The script creates trial-level merged BIN files and GT files. Full CSV export is
optional and should be avoided unless debugging.

## Current Trials

The current `eco20 + mesh` d5 archive has been mapped to:

```text
skin_ws/raw_data/sats/eco20 + mesh/d5/test1
  -> ecomesh_d5_z2.5_test1

skin_ws/raw_data/sats/eco20 + mesh/d5/test2
  -> ecomesh_d5_z2.5_test2
```

Verified row alignment:

```text
test1 merged rows  = 2,743,979
test1 on-grid rows = 2,743,978
test1 GT rows      = 2,743,978

test2 merged rows  = 2,743,017
test2 on-grid rows = 2,743,016
test2 GT rows      = 2,743,016
```

The one-row difference comes from the off-grid row dropped during GT generation
and SATS dataset loading. The drop policy is consistent, so training row
indices map 1:1 to GT rows.

## Alignment Principles

- `*_merged.bin` is the official input for both GT and SATS training.
- The merged timeline is 200 Hz, matching the meaningful DUE/loadcell rate.
- DUE raw bursts are expanded into effective 200 Hz `s1..s16` rows.
- EtherMotion can be 1000 Hz or higher and is interpolated onto the 200 Hz
  timeline. It provides high-resolution `x_mm`, `y_mm`, `z_stage_mm`, and
  `u_mm`; it does not define the final row rate.
- `z_depth_mm = max(z_stage_mm - z_start_mm, 0)`, with mk555 d5
  `z_start_mm = 13.0`.
- Z precision in the merged BIN is about `0.0001 mm`, so intermediate depths
  between 0.5 mm hold points are preserved.
- Force is single-axis loadcell:

```text
Fz = (kg - kg_baseline) * 9.80665
```

- `force_round_dp` is `null` for the current merged data, so Fz is not rounded
  to 0.01 N.
- `u_mm` is a node-internal wait/virtual axis and is not a physical shear label.
  Current SATS GT/training includes all rows by default.

## Stable Test Numbering

`trial_registry.json` maps source folders to `testN` numbers. Known source
folders keep their assigned number and only new folders are appended as
`max + 1`. Adding an earlier-dated acquisition later does not renumber existing
trials.

Do not delete `trial_registry.json` unless you intentionally want to rebuild the
entire trial numbering scheme.

## Training Note

Current d5 sequences include Z motion and 0.5 mm hold/plateau segments. Each
XY sequence is about 1600 timesteps, and peak contact is around timestep
820-860. Current SATS training defaults follow the paper-style sliding-window
setup:

```text
seq_len = 1000      # preserve the loading peak inside each raw cycle
window_size = 10    # one training sample = 10 timesteps
batch_size = 2048   # RTX 4090 default for window mode
num_workers = 2     # conservative default for 2-10 sets
```

Use `--no-use-window-dataset` only for the older peak-map experiment. That mode
is heavier because it moves full `[B, seq_len, 41, 41]` GT sequences through the
DataLoader.
