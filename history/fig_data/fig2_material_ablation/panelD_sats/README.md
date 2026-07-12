# Fig.2 패널 D — 소재별 SATS 학습 성능 비교 (논문 §6 Fig.2D)

> eco20 / eco50 / ecomesh 를 **동일 조건(xy1, 크기입력 A 모델, 대표 healthy fold)** 으로 학습해
> SR 성능을 비교. 논문 C1("mesh 최종 선택")의 학습 성능 근거.
> 파일명 접두어 = `Fig2D_*` (2026-07-12 구명명 `Fig3*`에서 변경, figset `prefix="Fig2D_"`).

## 핵심 수치 (2026-07-07 A 모델 확정)

d10 상대오차(공정 지표): **ecomesh 0.182 < eco20 0.259 < eco50 0.336** — 가설(mesh 우세) 지지.
d5 비교는 xy1 전 소재 저force라 불신 → d10/force-matched 중심 해석.

## 파일 ↔ 생성 스크립트

| 파일 | 내용 | 생성 코드 |
|---|---|---|
| `xy1_material/Fig2D_B_material_compare.png` | 소재 비교 막대(대표 패널) | `visualizing_scripts/figure_set/generate_fig3_sats.py --figset xy1_material` |
| `xy1_material/Fig2D_{A,C,D,E,F}_*_{소재}.png` | 소재별 line profile·3D맵·오차 | 동일 스크립트 (패널 A~F) |
| `xy1_material/Fig2D_G_forcematched_d10.png` | force-matched d10 비교 | `generate_fig3_forcematched_d10.py` |
| `xy1_material/shared_axes/` | 소재 간 동일 축 버전 + `axis_limits.json` | 동일 스크립트 `--shared-axes` |
| `xy1_material/Fig2D_report.md` | 해석·재현 커맨드 상세 (구 Fig3_report) | — |

데이터 소스: `experiments_archive/sizeA_diag/` (진단 npz), 모델 `sats/training/runs/size_input_material/`.
