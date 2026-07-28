# 밴딩 보상 프론트엔드 (`sats/bending/`)

flat 학습 SATS를 **동결**한 채, 밴딩 상태 센서 신호에서 ①밴딩 곡률(부호 있는 deg)을 추정하고
②flat 등가 baseline을 복원해 ③동결 SATS에 넣어 **재학습 없이** 밴딩+압력 추론을 한다.
목적 = **곡률마다 접촉 데이터를 재수집하지 않는 것**(무접촉 밴딩만으로 보정).

## 구조
```
밴딩% [T,16] → LSTM → MLP-A → signed deg      (BendingEstimator)
밴딩% + deg  → MLP-B → 오프셋; %−오프셋 = flat등가  (BaselineRestorer)
flat등가 → ❄️Frozen SATS → 압력맵            (BendingPipeline)
```
★ **SATS 입력 = 상대변화% `(s_raw−baseline)/baseline×100`** (`training/dataset.py:248`).
raw(~7e6)를 넣으면 포화되므로 밴딩도 반드시 이 %표현으로 변환해 다룬다.

## 파일
| 파일 | 역할 |
|---|---|
| `config.py` | `BendingConfig` (LSTM/MLP, deg_scale, 동결 SATS 경로) |
| `dataset.py` | 로더·윈도잉 + **`valid_only` 필터·`NormStats`·`build_windows`** |
| `geometry.py` | 원호 좌굴 δ→κ,θ,R + 자가추정 κ̂ + 캘리브레이션(순수함수) |
| `bending_estimator.py` | LSTM+MLP → signed deg |
| `baseline_restorer.py` | deg 조건 오프셋 → flat 등가(zero-init 항등) |
| `pipeline.py` | estimator→restorer→동결 SATS, `load_frozen_sats()` |
| `train_bending.py` | **Phase1 estimator + Phase2 restorer 학습** + CLI, save/load |
| `eval_pipeline.py` | **Phase3** 검증 — 환각 억제·곡률별 오차(e2e) + 그림 |
| `eval_g1.py` | **G1 관측성 하네스** — leave-one-session-out MAE·Spearman·drift |
| `tests/` | Phase 0 TDD (합성) |

전처리(raw→npz)는 `sats/preprocessing/prepare_bending_data.py`. 취득 기구·곡률 정의·유효범위는
`history/fig_data/fig3_sats_bending/bending/G1_ACQUISITION_DESIGN.md`.

## 데이터 사양
trial별 `.npz`(`learning_data/bending/<ver>/`): `sensor`[N,16] raw, `baseline`[16],
`bend_deg`[N] signed θ, `kappa_geo`·`kappa_hat`·`delta_mm`·`valid`(|δ|≤상한). 곡률 GT는
**스테이지 압축 원호모델**(지그/IMU 아님). ★ **유효 |δ|≤~10mm** — 초과 시 센서 포화(κ̂ 비단조 붕괴).

## 단계 (진행 현황, v0 = 단일세션 무접촉 밴딩)
- **Phase 0** ✅ 스캐폴드·모듈·테스트.
- **Phase 1** ✅ estimator: valid+표준화+trial split. holdout θ **MAE 6.3°**(기준선 30°).
  `python -m sats.bending.train_bending --phase estimator --val-trials <holdout>`
- **Phase 2** ✅ restorer(A 오프셋지도): holdout 오프셋 제거 67.9%. `... --phase restorer`
- **Phase 3** ✅(예비) 밴딩→flat→동결SATS: **환각 88% 억제**, **e2e(예측곡률)=oracle 동등**.
  유효 |δ|≤5–6mm에서 가짜접촉 flat 수준 복원, 고곡률 잔여.
  `python -m sats.bending.eval_pipeline --analyze --estimator <ckpt> --figure out.png`
  ⚠ 무접촉이라 **접촉 보존**(진짜 접촉 살리기)은 미검증 = bending+contact(G2) 데이터 필요.
- **G1(관측성)**: 하네스 완비. **정식 판정 = remounting ≥3세션 무접촉 취득 필요**(v0는 스모크).
- **Phase 4**: figure/README 최종.

## 참고
- 그래프 텍스트는 **영어만**(matplotlib 한글 □ 깨짐).
- 원호 근사는 clamped-BC라 엄밀 균일호 아님 → κ̂ 교차검증이 유효범위 실증(valid R²=0.82).

## 재현
```bash
.venv/bin/python -m pytest sats/bending/tests/ -q
```
