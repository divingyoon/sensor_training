# SATS map 품질 재평가 — 위치·형태·magnitude (2026-07-20)

> `scripts/reeval_map_quality.py`. 접촉 fz>0.3N. loc=argmax 위치오차, peak_corr=GT/pred peak 상관,
> peak_ratio=pred/GT peak median(1=정확, >1=과대예측). 스케일-무관 map 품질.

| 모델 | 지름 | loc(mm) | peak 상관 | peak 비율(pred/GT) | fz中 |
|---|---|---|---|---|---|
| ecomesh xy1 | d5 | 0.5 | 0.781 | 0.749 | 0.49 |
| ecomesh xy1 | d10 | 0.5 | 0.976 | 1.025 | 1.95 |
| eco20 xy1 | d10 | 0.707 | 0.944 | 1.038 | 1.5 |
| eco50 xy1 | d5 | 0.5 | 0.423 | 0.905 | 0.49 |
| eco50 xy1 | d10 | 1.0 | 0.901 | 0.982 | 2.15 |
| ecomesh xy0.5 (A) | d5 | 0.5 | 0.962 | 0.949 | 1.01 |
| ecomesh xy0.5 (A) | d10 | 1.0 | 0.491 | 1.205 | 1.35 |
| d5-only 0.5mm | d5 | 0.5 | 0.976 | 0.986 | 1.01 |

## 해석
- **loc 작고 peak_corr 높으면 = 위치·형태 정확** (rel 이 나빠도 map 은 좋음 = 지표 착시 확증).
- **peak_ratio > 1 = magnitude 과대예측** (d10 저force 에서 큼). 이건 실제 편향이나 위치와 분리됨.
