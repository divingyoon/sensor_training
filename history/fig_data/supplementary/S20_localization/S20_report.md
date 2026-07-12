# FigS20 — 위치추정 오차 (localization error)

논문 FigS20 대응. **pred 압력맵의 최대값 위치 = 추정 접촉위치, GT 맵 최대값 위치 = 실제 접촉위치**,
둘의 유클리드 거리(mm)를 위치추정 오차로 정의(논문 정의와 동일). 접촉 샘플(GT peak>0)만 집계.

## 결과 (평균 위치오차)

| 모델 | mean loc-error | n |
|---|---|---|
| **ecomesh_xy1** | **0.779 mm** | 274k |
| ecomesh_xy0p5_final | 0.830 mm | 1.93M |
| eco20_xy1 | 0.953 mm | 224k |
| eco50_xy1 | 1.312 mm | 259k |

- 논문 보고치(0.73 mm)와 **거의 일치** — 프레임워크·데이터 타당성 방증.
- **주의(eco50 재보정)**: eco50 d10 test3 로드셀 영점 교정·재학습 후에도 loc오차는 거의 불변(1.322→1.312).
  위치추정은 **peak 위치(argmax)** 라 force 크기 offset 에 둔감하기 때문. 교정 효과는 **Fig3B 상대오차**(eco50 d10 0.70→0.328)에 나타남.
- 소재 순위: **ecomesh < eco20 < eco50** → 가설(ecomesh 최상) 재확인(상대오차 지표와 일관).
- force↑ → 위치오차↓ (SNR 향상, 논문 FigS20B 경향과 동일). 저force(0–0.25N)에서 급증(near-zero 접촉 모호).
- 감지면 **가장자리에서 오차↑** (수용영역이 가장자리를 완전히 못 덮음, 논문 Discussion과 동일).

## 패널 (모델당 1 파일)

- 좌: 실제 위치별 평균 위치오차 2D 맵 (FigS20A)
- 우: force 구간별 평균 위치오차 + SEM (FigS20B)

## 공통 스케일 (소재 간 비교 가능하도록)

위치오차는 **mm 물리단위**라 소재 간 직접 비교 가능. 따라서 heatmap 컬러바 상한(vmax)과
막대 y축 상한을 **전 모델 공통값**으로 통일했다(기본 동작).
- 컬러바 vmax = 전 모델 위치오차 0.95분위의 최댓값(≈ eco50 기준) → 어느 소재도 포화되지 않음.
- 막대 y상한 = 전 모델 force-구간 (평균+SEM) 최댓값 × 1.12.
- 효과: 같은 초록 농도 = 같은 mm. eco50(짙음, 1.32mm) vs ecomesh(옅음, 0.78mm)가 한눈에 비교됨.
- `--per-model-scale` 로 예전 방식(모델별 자동 heatmap 스케일)도 선택 가능.

## 코드 (재현)

```bash
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_supp_localization.py \
    --models eco20_xy1 eco50_xy1 ecomesh_xy1 ecomesh_xy0p5_final
```

- 스크립트: `history/fig_data/visualizing_scripts/figure_set/generate_supp_localization.py`
- 위치추정 = `_peak_xy`(맵 argmax→mm), 오차 = `collect_localization`, 그림 = `plot_localization`.
- 별도 진단 npz 불필요(체크포인트에서 직접 추론).
