# Fig.3 밴딩 파트 — 데이터 취득 준비 (논문 §6 Fig.3 B–E)

> 상태: **데이터 취득 대기** (코드 Phase 0 + **합성 리허설 완료** — `rehearsal/rehearsal_report.md`).
> 리허설 결과(2026-07-13, `scripts/rehearse_bending_pipeline.py`): P1 deg MAE 0.77°(밴딩-only) ·
> P2 오프셋 제거율 98.5% · P3 동결 SATS(A, size 전달) 출력 회복률 97.8%.
> 발견 2건 수정/문서화: ①pipeline size 미전달 버그 fix ②B안 e2e 학습 시 cudnn RNN backward 제약
> (`torch.backends.cudnn.flags(enabled=False)` 필요). ⚠️ 밴딩+접촉 혼합 시 deg MAE 16.8°로 저하
> (밴딩-only 학습 estimator 기준) — 실데이터 사전검증 항목 그대로 유효.
> 목표: flat 학습 SATS를 동결한 채, 밴딩 baseline에서 곡률(signed deg)을 자가 추정하고
> flat 등가 신호를 복원해 밴딩 상태에서도 SR을 유지 (논문 C2).

## 논문 패널 매핑 (§6 Fig.3)

| 패널 | 내용 | 필요 데이터 | 생성 코드(예정) |
|---|---|---|---|
| (A) | jig·각도 정의 모식도 | 없음(모식도) | 신규 스크립트 or Figma |
| (B) | 각도별 Δbaseline vs z_i | 밴딩-only baseline | Phase 4 스크립트 |
| (C) | 곡률 θ 추정 회귀 (MAE/R²) | 밴딩-only baseline | `sats/bending/train_bending.py` → 평가 |
| (D) | flat vs bent SR 보정 전/후 RMSE/R² | 밴딩+접촉 | `sats/bending/pipeline.py` 평가 |
| (E) | 추론 vs GT SR 맵 | 밴딩+접촉 | 진단 덤프 + 맵 시각화 |

## 취득 스펙

**공통**: 시간별 sensor 16ch + 밴딩 deg(양/음 부호 구분). 온도 안정화 후, 무접촉 tare 구간 포함.

1. **밴딩-only baseline** (Phase 1·패널 B/C): jig 각도별(예: −40°~+40°, 10° 간격) 무하중 10 set.
   각 set = 해당 각도 유지 상태 시계열(수 초). flat(0°) 기준 포함.
2. **밴딩+접촉** (Phase 2·3·패널 D/E): 대표 각도(예: 0°, ±20°, ±40°)에서 d5 인덴터
   grid press(가능하면 flat 최종 취득과 동일 프로토콜 = xy0.5 계단식). 좌표·fz GT 동시 로깅.
3. **flat 기준**: 동일 세션 flat 데이터 1 set (성능 저하폭 비교 기준).

## npz 컨트랙트 (`sats/bending/dataset.py`)

trial별 `.npz`:
- `sensor` float[N,16] — 16ch 시계열
- `bend_deg` float[N] — **signed** 곡률 각도 GT (지그/IMU)
- `contact` float[N,3] — 선택, (x, y, fz). 밴딩-only는 생략
- 저장 위치: `learning_data/sensor_raw_bin/<mat>_bend/`

## 취득 후 실행 순서 (P1→P4)

1. P1 estimator: `train_bending.train_estimator()` → deg MAE 확인
2. P2 restorer: 오프셋 지도(A안, bending-only=순수 오프셋) 또는 end-to-end(B안)
3. P3 pipeline: 밴딩 하 동결 SATS 정확도 vs flat (재학습 0 확인)
4. P4 figure: 이 폴더에 패널 B–E 산출 + 생성 스크립트 매핑 기록

## 사전 검증 (리스크 방어)

- 밴딩·접촉 신호의 **시간 스케일 차이·중첩 선형성**을 취득 초기 데이터로 먼저 확인
  (분리 ill-posed 리스크). 곡률 GT(지그 각도) 신뢰도 체크.
