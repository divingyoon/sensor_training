# 논문 투고 로드맵 & 마스터 체크리스트

> 기준 = 논문 md(`Development_...md`)의 기여 **C1~C4**, §6 Figure Plan, §7 데이터 체크리스트, §9 정량 수치.
> 갱신: **2026-07-12** (fig_data §6 재배치 완료 시점). 세부 데이터 위치는 `PROJECT_STRUCTURE.md`.
> 표기: ✅ 완료 · 🔄 진행/부분 · ⬜ 미착수 · ⚠️ **사용자 결정 필요**

---

## 0. 한눈에 보는 상태

| 기여 | 주장 | 상태 |
|---|---|---|
| **C1** mesh 수용장 중첩 → SR 향상 | 소재 비교 + SATS 성능으로 입증 | ✅ **데이터·figure 완비** (반복 3set만 미완) |
| **C2** 밴딩 자가추정 → bending-aware SR | 곡률 추정 + 잔차 보정 | 🔄 **코드 Phase 0 완료, 데이터 취득 0건** ← 투고 critical path |
| **C3** 다중 접점 + 동적 해석 | 단일→다점·시간 확장 | ⬜ **데이터·실험 전무** ⚠️ 유지/축소 결정 필요 |
| **C4** 사람손·로봇핸드 공용 플랫폼 | 곡면 부착 데모 | ⬜ 하드웨어 데모 미착수 (Fig.4) |

**투고 성립의 최소 조건 = C1(완료) + C2(밴딩 실증)**. C3·C4는 범위 결정에 따라 강도 조절.

---

## 1. P0 — Critical Path (이것 없이는 투고 불가)

### 1.1 밴딩 데이터 취득 (C2·Fig.3) — 최우선
- ⬜ jig 준비·각도 눈금 검증 (곡률 GT 신뢰도 — 리스크 항목)
- ⬜ **밴딩-only baseline**: 각도별(예: −40°~+40°, 10° 간격) 무하중 10 set, signed deg, flat(0°) 포함
- ⬜ **밴딩+접촉**: 대표 각도(0°, ±20°, ±40°) × d5 인덴터, **xy0.5 계단식 동일 프로토콜**(도메인 불일치 방지 — pool null result 교훈)
- ⬜ 같은 세션 flat 기준 1 set (성능 저하폭 비교 기준)
- ⬜ 취득 직후 **사전 검증**: 밴딩·접촉 신호 시간스케일 분리 가능성 + 중첩 선형성 (ill-posed 리스크 게이트)
- 스펙: `fig3_sats_bending/bending/README.md` / npz: `sensor[N,16] + bend_deg[N] (+contact[N,3])`

### 1.2 밴딩 모델 P1→P4 (`sats/bending`, Phase 0 완료)
- ⬜ P1 estimator 학습 → **곡률 MAE/R²** (§9 수치, Fig.3C)
- ⬜ P2 restorer (A안 오프셋 지도 / B안 end-to-end 중 데이터 보고 결정)
- ⬜ P3 pipeline 검증 → **flat vs bent 보정 전/후 RMSE/R²** (§9 수치, Fig.3D) — *"점진적 저하" 논거의 정량 근거*
- ⬜ P4 figure: Fig.3 패널 B/C/D/E 생성 스크립트 + report ([[figure-reproducible-code]] 원칙)

### 1.3 Fig.3 완성
- ⬜ 패널 (A) jig·각도 정의 모식도 (`algorithm/`의 알고리즘 구조도와 별개 — §6 정의 기준)
- ⬜ 패널 (B)~(E) ← 1.2 산출물
- ✅ flat 기준 패널 (`flat_sr/final_xy0p5/`, A 모델)

---

## 2. P1 — 주장 강화 (투고 전 완료 권장)

### 2.1 Fig.4 Application (C4)
- ⬜ 로봇핸드 부착 실시간 감지·제어 데모 (실시간 추론: `sats/inference` + `sats/bending/pipeline`)
- ⬜ 사람 손 곡면 부착 동작 시연
- ⬜ 데모 영상/스틸 → Fig.4 패널 구성

### 2.2 Fig.1 Concept
- ⬜ 컨셉 일러스트 제작 (후보 소재: `fig1_concept/README.md` — archive/사진·pptx)

### 2.3 성능 보강 취득
- ⬜ **저force d10 반복취득**(xy0.5 계단식) — d10 magnitude 과대예측(현 d10_rel 0.749)의 유일한 남은 해법. Fig.3E 맵 품질에 직결
- ⬜ 재학습(A 모델, 동일 하이퍼) → 진단 재덤프 → flat figure 재생성

---

## 3. ⚠️ 사용자 결정 필요 (범위 확정)

| # | 결정 | 옵션 A | 옵션 B |
|---|---|---|---|
| D1 | **C3(다중접점+동적) 유지 여부** | 유지 → 2·3점 zero-shot 취득(재학습 불필요) + 동적(load-unload/슬립) 지표 실험 추가 | 축소 → §2.6 C3·§5.5를 "향후 과제"로 강등(§8 한계로 이동), 투고 범위 = C1·C2·C4 |
| D2 | **Fig.2 소재별 3 set 반복** (§7, 통계 강건성) | 취득 → 오차막대 포함 재분석 | 현 1 set 유지 + 한계 명시 (리뷰 리스크 감수) |
| D3 | **molding 변경·아랫면 소재 통일** (§7) | 재제작 후 재취득 (Fig.2 전체 갱신) | 현 센서 유지 + §4.2에 현 사양 명시 |
| D4 | **§4.2 표기 확정**: "eco20/50/20"·"eco20/45"가 적층 순서인지 혼합비인지 | — 본문 명시 필수 (mix ratio·경화 조건 또는 sub-layer 두께) | |

> D1은 취득 장비가 이미 있으므로 2·3점 zero-shot만이라도 취득하면 C3를 "예비 실증"으로 유지 가능 (SATS 원논문 Fig4E 대응, `SATS_paper_implementability.md` §D).

---

## 4. §9 정량 수치 확보 현황 (Results 표)

| §9 항목 | 값 | 상태 / 출처 |
|---|---|---|
| Flat SR 위치오차 | ecomesh_xy1 **0.79 mm** (최종 xy0.5 1.13 mm) | ✅ `supplementary/S20_localization/loc_summary.csv` |
| Flat SR 상대 RMSE | 최종(A): d5_rel **0.188** / d10_rel 0.749(저force magnitude 잔존) | ✅ `experiments_archive/sizeA_final_xy0p5_diag/` (d10은 §2.3 보강으로 개선 여지) |
| SR scale factor | **≈105×** (41²/16; 다해상도 27×~2525× 안정 실증) | ✅ `supplementary/summary_metrics/` + `d5_final/` |
| 소재별 SR 비교 | d10_rel **ecomesh 0.182 < eco20 0.259 < eco50 0.336** · loc **0.79 < 1.02 < 1.27 mm** | ✅ `fig2_material_ablation/panelD_sats/` |
| 소재별 ΔS·확산·active taxel | §9.1 표 확보 (d5/d10) | ✅ `fig2_material_ablation/Analysis_Results/Fig2_report.md` |
| **곡률 추정 MAE(°)/R²** | — | ⬜ P0 1.2 |
| **Bent SR 보정 전/후 RMSE/R²·저하폭** | — | ⬜ P0 1.2 |
| 다중 접점 분리 / 동적(슬립) 지표 | — | ⚠️ D1 결정 후 |
| Abstract 정량 수치 채움 (§1 각주) | — | ⬜ 위 수치 확정 후 일괄 |

---

## 5. 원고(본문) 작업

- ⬜ §1 Abstract 정량 수치 채움 (§4 표 완성 후)
- ⬜ §4.2 재료 표기 확정 (D4) + mesh 사양(재질·피치) 기재
- ⬜ §5.4 처리 전략 (a)~(d) 중 실제 채택안으로 본문 확정 (P2 결과 반영)
- ⬜ §9 Results 표 완성 → 본문 §6 Figure 캡션과 수치 일치 검증
- ⬜ C3 범위 반영 (D1 결정 후 §2.6·§5.5·§8 수정)
- ⬜ Supplementary 재구성: 현 `supplementary/`는 SATS 원논문 번호(S19 등) → **우리 논문 Supp 번호로 재편성** + d5 다해상도(SR 해상도 자유 실증)를 Supp figure로 승격 검토
- ⬜ 참고문헌 정리 (`reference/` 선행논문 → 인용 목록화)

## 6. 투고 준비 (원고 완성 후)

- ⬜ 타깃 저널 확정 → figure 규격(포맷·해상도·컬러)·워드리밋 맞춤
- ⬜ figure 최종본: HTML figure_set → 저널 규격 개별 파일 export (생성 스크립트 경로 고정)
- ⬜ 재현성 패키지: 코드·모델·데이터 공개 범위 결정 (learning_data는 대용량 git-ignored)
- ⬜ 커버레터 / 하이라이트

---

## 7. 권장 실행 순서 (의존성 기준)

```
[지금] D1~D4 범위 결정
   ↓
① 밴딩 취득(1.1) ──→ ② P1~P4(1.2) ──→ ③ Fig.3 완성(1.3)     ← critical path
   ∥ (병렬 가능)
④ 저force d10 보강취득(2.3) → flat 재학습·figure 갱신
⑤ (D1=유지 시) 2·3점 zero-shot 취득 → C3 예비 실증
⑥ Fig.4 데모(2.1) · Fig.1 일러스트(2.2)
   ↓
⑦ §9 표·Abstract 수치 채움 → 본문 §5 확정(5장)
⑧ Supp 재편성 → 저널 규격 export → 투고(6장)
```

취득 세션 효율화: ①④⑤는 같은 로봇팔+jig 셋업 기간에 묶어서 취득 권장 (센서 탈부착·온도 조건 변화 최소화).
