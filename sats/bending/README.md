# 밴딩 보상 프론트엔드 (`sats/bending/`)

flat 학습 SATS를 **동결**한 채, 밴딩 상태 센서 신호에서 ①밴딩 곡률(부호 있는 deg)을 추정하고
②flat 등가 baseline을 복원해 ③동결 SATS에 넣어 **재학습 없이** 밴딩+압력 추론을 동시에 한다.

## 구조
```
밴딩 시퀀스[T,16] → LSTM 인코더 →┬ MLP-A → signed deg (BendingEstimator)
                                └ MLP-B → 밴딩 오프셋 (BaselineRestorer)
원신호 − 오프셋 = flat 등가 → ❄️Frozen SATS → 압력맵   (BendingPipeline)
```
- **부호 있는 deg**: 양/음 밴딩 방향 구분. estimator는 활성화 없는 signed 회귀, restorer 오프셋 방향은 부호에 따라 반전.

## 파일
| 파일 | 역할 |
|---|---|
| `config.py` | `BendingConfig` (LSTM/MLP 크기, deg_scale, 동결 SATS 경로) |
| `dataset.py` | 데이터 사양 + 로더 + 윈도잉 (`BendingTrial`, `make_windows`) |
| `bending_estimator.py` | LSTM+MLP → signed deg |
| `baseline_restorer.py` | deg 조건 채널별 오프셋 → flat 등가(zero-init 항등) |
| `pipeline.py` | estimator→restorer→동결 SATS. `load_frozen_sats()` |
| `train_bending.py` | Phase1 estimator 학습(데이터 오면 실행). P2/3 취득 후 |
| `tests/` | Phase 0 TDD (합성 데이터) |

## 데이터 사양 (곧 취득)
trial별 `.npz`: `sensor` float[N,16], `bend_deg` float[N] **(signed)**, 선택 `contact` float[N,3]=(x,y,fz).
모드: bending-only(무접촉) · bending+contact · flat 기준. 곡률 GT = 지그/IMU. 저장 위치 `learning_data/sensor_raw_bin/<mat>_bend/`.

## 단계 (로드맵)
- **Phase 0 (완료)**: 스캐폴드·데이터 사양·모듈·테스트. 데이터 불요.
- **Phase 1**: estimator 학습 → signed deg (지표 deg MAE). `train_estimator()`.
- **Phase 2**: restorer 학습 — (A)오프셋 지도(bending-only=순수 오프셋, 중첩 선형성 가정) 또는 (B)end-to-end(동결 SATS 통과 grad).
- **Phase 3**: pipeline 검증 — 밴딩 하 SATS 정확도 vs flat 기준(재학습 0).
- **Phase 4**: figure/README (곡률 정확도·복원 품질·밴딩 하 SATS 정확도), `history/fig_data/bending/`.

## 리스크
- 신호 분리 ill-posed(밴딩·접촉 중첩) → 시간 스케일 차·중첩 선형성 **데이터로 사전 검증** 권장.
- SATS 도메인 시프트 → end-to-end 검증 필수. 곡률 GT 신뢰도.

## 재현 (Phase 0 테스트)
```bash
.venv/bin/python -m pytest sats/bending/tests/ -q
```
