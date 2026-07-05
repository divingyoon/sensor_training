# ecomesh d10 pooling 실험 결과 (2026-07-06)

**가설**: xy0.5 d10 은 취득 3rep 뿐이라 약함 → xy1 d10 3rep 를 추가(pooling)하면 개선될 것.

**셋업**: train 14 (0p5 d5 test1–9 + 0p5 d10 test1–2 + **xy1 d10 test1–3**), holdout d5 test10 · **d10 test3**.
datarich(0p5-only, 11 trial) 와 동일 하이퍼파라미터 → 단일변수(xy1 d10 추가) 비교.
run: `sats/training/runs/pool_d10/ecomesh_pool_d10_val_d5t10_d10t3`.

## 결과: **개선 없음 (null result)**

| 홀드아웃 d10 test3 | datarich(0p5만) | pooled(+xy1 d10) |
|---|---|---|
| 상대오차 d10 | 0.710 | 0.709 |
| 절대 rmse d10 | 0.415 | 0.418 |

force 구간별로도 미세 등락뿐(0.25–0.5N 상대 1.74→1.57 소폭↓, 0.5–2N 소폭↑), **순변화 없음**.

## 해석

- **xy1(straight press) 과 xy0.5(계단식 하강)는 로딩 프로토콜이 달라 사실상 다른 도메인**.
  점탄성(rate-dependence) 확인됨: 같은 force 에서 센서 반응이 xy1(빠름) > xy0.5(느림).
  → xy1 d10 을 추가해도 xy0.5 저force 홀드아웃으로 **전이가 안 됨**.
- 홀드아웃 d10 test3 자체가 **저force**(fz_med 0.29) → 상대오차가 분모(작은 target)로 부풀려짐. 데이터량으로 안 풀림.

## 결론 / 다음

- **xy0.5 d10 개선의 진짜 해법 = xy1 추가가 아니라 xy0.5 d10(같은 프로토콜) 반복 취득 증가.**
- pooling 은 "더 많은 데이터가 항상 낫다"가 아니라 **도메인이 맞아야 이득**임을 보여준 유효한 음성 결과.
- eco50 재보정 재학습은 별개로 성공(d10 0.70→0.33, Fig3B 참조).

## 코드 (재현)
```bash
bash scratchpad_rerun_all.sh   # eco50 재학습 + ecomesh pooled + diag + figure 재생성
```
