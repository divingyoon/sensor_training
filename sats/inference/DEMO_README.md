# v6 실시간 추론 데모 (학회 데모세션)

v6 학습 파일로 실시간 추론. **터미널 우선** + 2D/3D 시각화. 엔트리: `sats/inference/run_demo.py`.

## 모델·보정 자산
- SATS 배포: `sats/training/runs/ecomesh_v6_deploy_all4`
- 밴딩 estimator: `sats/bending/runs/estimator_v6`
- z 보정 LUT: `sats/inference/z_calibration_v6.json` (맵 peak→z_depth, d5 R²0.87·d10 R²0.59, **근사값**)

## 산출물 ↔ 실행

**환경**: RTX 5090 → `.venv/bin/python`. 시리얼 포트/보드레이트/protocol은 실센서에 맞게 조정.

```bash
# ① 단일접촉 x,y,z,fz (터미널)
.venv/bin/python -m sats.inference.run_demo --mode contacts --contacts 1 --diameter 5 --port /dev/ttyUSB0

# ① 다중접촉(최대 3) x,y,z,fz
.venv/bin/python -m sats.inference.run_demo --mode contacts --contacts 3 --diameter 10 --port /dev/ttyUSB0

# ② +시각화(2D/3D heatmap + 최적프레임). 위 명령에 --viz 추가
.venv/bin/python -m sats.inference.run_demo --mode contacts --contacts 3 --diameter 10 --viz 2d --port /dev/ttyUSB0
#   --viz {2d|3d|both}

# ③ 밴딩각 theta (터미널)
.venv/bin/python -m sats.inference.run_demo --mode theta --port /dev/ttyUSB0

# ④ SATS+밴딩 통합 (theta: x,y,z,fz)
.venv/bin/python -m sats.inference.run_demo --mode bending --contacts 3 --diameter 10 --viz 2d --port /dev/ttyUSB0
```

- `--mock`: 하드웨어 없이 파이프라인 확인(단 theta/bending은 flat baseline이 없어 비의미).
- `--report-interval`: 최적(최대 Fz) 프레임 요약 주기. `--infer-max-fps`/`--viz-fps`: 속도.

## 산출물별 동작
1. **contacts**: 매 프레임 접촉별 `x,y,z,fz` 출력 + 최대 Fz 프레임 latch. z=peak_val LUT 근사, fz=Voronoi 적분.
2. **viz**: `live | optimized frame` 나란히, 접촉 마커(최대3)·라벨·theta.
3. **theta**: raw=base×(1+pct/100) 복원 → estimator → 밴딩각(smoothed).
4. **bending 상태머신**:
   1. 시작 → **플랫 무접촉 baseline** 캡처(센서에서 손 떼고 대기).
   2. **지그 장착·밴딩** 후 **Enter** → 밴딩 무접촉서 theta 고정.
   3. 접촉 press → `restorer(pct,theta)` 복원 → 동결 SATS → `theta: x,y,z,fz`.

## ★ 내일 실기 체크리스트
- [ ] serial `--protocol`(현 기본 v2)·`--baudrate`·`--port` 실센서 확인.
- [ ] baseline 캡처 시 **무접촉 유지**(플랫). d5/d10 `--diameter` 지정(z·size 조건).
- [ ] theta 스케일: estimator는 **buckling(Y구동 δ=Y−18)** 학습 → 데모 지그도 **버클 방식**이어야 theta 유효. 순수 굽힘이면 정합 데이터 재학습 필요([[v6-test-eval]]).
- [ ] 다중접촉: 접촉을 **동등·firm하게** 눌러야 분리 선명(약접촉은 확산).
- [ ] z는 **근사**(특히 d10 R²0.59)임을 데모 설명에 명시.

## 신규 모듈
`z_calibration.py`(z LUT) · `demo_contacts.py`(추출·터미널·latch) · `bending_infer.py`(theta·복원) · `demo_viz.py`(2D/3D) · `run_demo.py`(엔트리). 테스트 `tests/test_demo_contacts.py`·`test_demo_viz.py`.
