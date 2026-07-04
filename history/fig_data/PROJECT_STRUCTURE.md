# fig_data 프로젝트 구조 (매 세션 탐색 단축용)

> 유연 촉각 센서(SATS 기반 super-resolution) 논문의 **Figure 제작 + 데이터 분석** 워크스페이스.
> 논문 본문: `Development_of_a_Flexible_Tactile_Sensor_with_Super-Resolution_Capability_for_Robust_Robotic_Manipulation.md`
> 센서 사양: SATS 기반 30×30×5.5mm, FPCB + MEMS Barometer, 3중층(Top: Ecoflex+mesh / Mid: MEMS embedded / Bot: 베이스). taxel 16개 = 4×4 격자(간격 6.5mm, ±9.75mm).

## 최상위 디렉토리

| 경로 | 역할 |
|---|---|
| `Development_...md` | 논문 working draft (Fig.1~4 계획은 §6, 데이터 체크리스트 §7) |
| `fig2_heatmap/` | **Fig.2(소재 ablation) 실험 데이터** (CenterLine / CenterPress / xy_1mm) |
| `fig1_concept/`, `fig3_sats and bending/`, `fig4_application/` | 각 Figure 자료 |
| `visualizing_scripts/` | Figure 생성 Python 스크립트 |
| `reference/` | 선행 논문(SATS, TVI/Barodome, barometric 등) |
| `sensor_training/` | **데이터 취득(skin_ws) + SATS 학습** 레포 (git repo) |

## Figure ↔ 데이터 매핑 (논문 §6)

- **Fig.1** Concept: 사람손/로봇핸드 공용 컨셉.
- **Fig.2** C1 소재 ablation: 인덴터 원형 d5/10/15·사각, xy 1mm 격자, 소재(eco20/eco50/ecomesh). 패널 (A)셋업 (B)ΔS heatmap (C)|ΔS|·확산·활성taxel (D)소재별 SATS RMSE/R². 결론=mesh20.
- **Fig.3** C2 SATS 최종 + 밴딩: ecomesh flat SR + 각도별 bending baseline.
- **Fig.4** Application: 로봇핸드/사람손 곡면 부착 데모.

## fig2_heatmap 데이터

세 실험 조건:
- `CenterLine/` — 1D 라인 스캔(y 고정, x 스위프). CSV 처리·PNG 6개 완료. `combined_centerline.csv` 병합본.
- `CenterPress/` — 중심 고정 깊이 스윕. **PNG 결과만** 존재.
- `xy_1mm/` — **2D 격자(±10mm, 1mm 간격) 압입. 논문 Fig.2B의 핵심 데이터.** 대부분 raw `.bin`, 일부만 CSV 변환됨.

소재/인덴터 폴더 규칙: `xy_1mm/{ec020|eco50|ecomesh}/d{5|10}/YYYYMMDD_testN/`
(주의: `ec020`은 eco20 오타. mesh는 `ecomesh`/`eco20 + mesh` 혼용.)

### CSV 포맷 (각 test 폴더)
- `due_data.csv`: `elapsed_ns,time_s,burst_index,frame_index,Skin1..Skin16` — 센서 raw ADC(~7.0e6 offset). **1 burst = 10 frame**.
- `ethermotion_data.csv`: `...,X,Y,Z,U,X_lCmd,Y_lCmd,Z_lCmd,U_lCmd,Z_lCmd_mm` — 인덴터 위치. **X/Y/Z 단위 = encoder pulse, ×1e-4 = mm** (±100000 pulse = ±10mm; Z max 14mm).
- `loadcell_data.csv`: `elapsed_ns,time_s,kg` (×9.80665 = N). xy_1mm baseline ~0.09kg, 접촉 peak 0.4~0.58kg.
- `afd50_data.csv`: 6축 F/T — **거의 미사용(0행)**.
- `manifest.json`: 변환 메타(일부 폴더). `due_payload_layout: sensor[16][frame10]`.

### ΔS(센서 응답) 정의
`ΔS_i = -(raw_i - baseline_i) / baseline_i * 100` (%). baseline = 무접촉 초기 구간 평균. 부호 반전으로 압력 증가를 양수로.

## visualizing_scripts

- `CenterLine/generate_plots.py` — **1D centerline heatmap**(센서index×x) + 곡선 + 메트릭. baseline/ΔS/동기화 로직 참고용.
- `CenterLine/centerline_merge.py` — raw 3 CSV → `unified.csv`(phase 라벨링: base/loading/holding).
- `CenterLine/consolidate.py` — 모든 unified → `consolidated.csv`(격자 정렬).
- `CenterLine/visualize.py` — 3D 막대 + 인덴터 곡면. `SENSOR_XY` 물리좌표.
- `xy_1mm/generate_panelC_metrics.py` — **★Fig.2C 정량 메트릭** (Total|ΔS|·Active taxels·Propagation σ·Entropy). 격자 셀별 **peak 압입**을 대표로, **|ΔS| 절댓값 + 절대 floor 0.5%** 로 노이즈 제거(2026-06-25 수정: 옛 상대임계 0.15·peak 은 d5 에서 σ·entropy 0 퇴화로 폐기). `--diameter` 지원, 출력 `Analysis_Results/{d}/Fig2C_metrics.{png,csv}`. `generate_2d_heatmap` 의 로딩/접촉 로직 import 재사용.
- `xy_1mm/generate_panelA_schematic.py` — **★Fig.2A 셋업 모식도**(데이터 아님): (A1)센서 평면도+16 taxel+xy격자+인덴터 풋프린트, (A2)3중층 단면. 출력 `Analysis_Results/Fig2A_schematic.png`.
- `xy_1mm/generate_2d_heatmap.py` — **★Fig.2B 2D 수용장 heatmap.** CLI `--diameter d5|d10|all`(기본 all). 산출은 **지름별 폴더** `xy_1mm/Analysis_Results/{d5,d10}/`에 `Fig2B_receptive.png`(단일 taxel 수용장 3소재), `Fig2B_radial.png`(반경 감쇠), `Fig2B_metrics.csv`. 격자는 **전체 샘플 셀별 max**로 구성(멀리=0으로 채움), **접촉 게이팅은 센서 응답 기반**(어떤 taxel이라도 ΔS>5%; loadcell 절대임계값은 d5/연질에서 0검출되어 폐기). **σ는 데이터 centroid 기준**(레이아웃 가정 비의존). 센서 레이아웃: `SENSOR_XY[Skin{r*4+c+1}]=(_xs[c], _xs[r])`, `_xs=[-9.75,-3.25,3.25,9.75]`. 소재↔test 폴더 매핑은 `DATASETS` dict.
- `0618_시각화 방향.md` — 시각화 전략(Heatmap A / Curves B / 2D map C / Radial D).

## sensor_training (취득 + 학습)

- `skin_ws/acquisition_code/` — 멀티센서 로깅.
  - `convert_bins.py` — **`.bin` → CSV 변환기.** CLI는 `raw_data/` 최신 폴더 자동 변환만 지원 → 임의 폴더는 `find_bin_set()`/`convert_set(inputs, out_dir, args)` 직접 호출(args=`SimpleNamespace(bias_samples=200, no_invert_fz=False)`).
  - `final_logger*.py` — DUE(200Hz)/Loadcell(200Hz)/EtherMotion(~1000Hz) 동기 로깅.
- `sats/` — **공식 SATS 파이프라인.** preprocessing(`bin_merge.py`→merged.bin, `generate_gt.py`→Boussinesq 41×41 GT) → training(LSTM→attention→localmap→CNN 4단계, `train_*.py`) → inference(realtime). 센서 normalize·on-grid 필터.
- `learning_data/` — 정규화 merged.bin + GT(대용량, git-ignored). `trial_registry.json`로 test 번호 추적.
- `hitmap/` — legacy XY heatmap + Z/Fz 회귀(데이터 컨트랙트 다름, 현재 우선순위 아님).
- `cnn_lstm/` — 사이드 프로젝트.

## 데이터 파이프라인 요약
```
취득(skin_ws final_logger) → raw .bin
  → convert_bins.py → due/ethermotion/loadcell CSV   (Figure 분석용)
  → sats/preprocessing/bin_merge.py → merged.bin → generate_gt.py → GT   (학습용)
  → SATS 4단계 학습 → inference
시각화: visualizing_scripts/* → fig2_heatmap/.../Analysis_Results/*.png
```

## Fig.2B 현재 상태 (2026-06-23)
- 변환 완료(.bin→CSV): d10 = eco20/test2, eco50/test2, ecomesh/test4 / d5 = eco20/test5, eco50/test1, ecomesh/test1.
- 산출물: `Analysis_Results/d10/`, `Analysis_Results/d5/` 각각 `Fig2B_receptive.png`·`Fig2B_radial.png`·`Fig2B_metrics.csv`.
- **d10 결과**: 수용장 σ = eco20 2.77 < eco50 3.56 < **ecomesh 3.77mm**, peak 모두 ~100% 포화 → 확산은 ecomesh 최대이나 소재 대비 약함.
- **d5 결과(대비 더 선명)**: peak = eco20 21% ≪ eco50 49% < **ecomesh 51%**, active = eco20 22, eco50 22, **ecomesh 24**. eco20 연질의 약한 신호(저SNR)가 직접 드러나고 ecomesh가 민감도+확산 동시 최대 → mesh20 선택 근거 강함.
- **주의**: d5의 eco20 σ(3.91)는 inflated — peak이 낮아 응답가중 centroid가 저신호 노이즈에 끌림. 저SNR 소재는 σ보다 **peak·active cells**가 신뢰지표.

### 패널 진행 현황
- **(B) 수용장 heatmap**: 완료 (`Fig2B_*`).
- **(C) 정량 메트릭**: 완료 (`Fig2C_metrics.{png,csv}`). d5 결과 단조: Total|ΔS| eco20 14.9 ≪ eco50 25.8 ≈ ecomesh 24.4(민감도 eco50급 유지), Active 1.00<1.03<**1.14**, σ_prop 0.00<0.07<**0.38**, Entropy 0<0.005<**0.027** → ecomesh 수용장 중첩 최대 + 민감도 유지 = mesh20 근거. d10 동일경향(대비 약함).
- **(A) 셋업 모식도**: 완료 (`Fig2A_schematic.png`).
- **(D) SATS RMSE/R²**: 미진행(모델 학습 필요). eco50 SNR 손실 직접 입증용 noise/SNR metric도 향후 고려.
- **통합 보고서**: `Analysis_Results/Fig2_report.md` — 패널 A/B/C PNG 임베드 + 표 + 종합 분석/한계. 논문 본문도 §9.1에 측정값 표 보충, §7 Fig.2 체크리스트 갱신.
- **center press(0,0) 깊이별**: `generate_centerpress_depth.py` → `Fig2_centerpress_{d5,d10}.png`. 센터 셀(|x|,|y|<0.7) 침투 ≈1mm / max(≈2mm)에서 16-taxel 응답 4×4 맵 + active·σ·peak 막대. **신호 = |ΔS| 절댓값** (baseline 차감 +/− 부호는 폴리머 인장/압축 측정방향이 불확실 → 크기로 비교). 실측 무접촉 노이즈 std≈0.01% → floor 0.1%. 컬러는 **소재별 자체 정규화**(각 패널 자기|peak|=1) → 절대세기 무관 '수용장 모양(퍼짐)' 공정비교(eco20 국소·eco50 집중·mesh 최광). 절대 민감도는 하단 회색 막대로 분리(total|ΔS|=eco50 최고). 약신호 패널(|peak|<0.6%)은 빨간 '(≈noise)' 표시(정규화 노이즈 증폭 방지). 각 깊이=그 구간 응답상위 샘플 대표값. **국소 drift 보정**: 글로벌 baseline 대비 ΔS엔 센터셀 도달까지 누적 drift가 섞여(약신호 d5에서 코너에 가짜 응답) → 셀 무접촉(총|ΔS| 하위30%) per-taxel 평균을 빼 압입 유발분만 남김(eco20 d5는 중앙=코너 평평한 drift라 보정후 ≈0=올바른 무응답). **d10**: ecomesh active 2<3<**8**, σ 3.2<3.7<**8.1mm**, |peak| 최저(7.2%) = 응답을 인접 taxel로 분산(수용장↑) → "왜 mesh" 명확. **d5**: ⌀5mm가 taxel 사이(pitch 6.5mm) 압입 → 약함(|peak|<2%), 약신호 중엔 eco50(강성)이 점하중 전달 최대라 mesh 우위 안 드러남 → 저진폭 경고배너, "d10/taxel-정렬(Fig.2B)에서 확인" 명시. *주의: 초기버전 버그 — 절대 floor 1.5%(노이즈 100×)+양수 clip 으로 d5 신호를 죽였음 → |ΔS|·floor 0.1%로 수정.*
- **between-taxel overlap**: `generate_betweentaxel.py` → `Fig2_betweentaxel_{d5,d10}.png`. (0,0)이 4개 대각 taxel서 멀어(4.6mm) 약하고 비대칭이라, **두 인접 taxel 중점**(각 3.25mm 등거리=SR 시나리오)을 누름. 인접쌍 4개 평균: overlap evenness(min/max), weaker-taxel|ΔS|, pair-sum|ΔS|. **mesh가 evenness 최고**(d10 0.87; 맵 Skin10|11=40.0|39.7 even 0.99 거의 완벽대칭, eco20 48.5|13.3 even 0.27 쏠림=undersampling). **d5에선 mesh가 3지표 모두 1등**(even 0.58·weaker 2.7%·sum 8.2%) → (0,0)서 못 본 d5 mesh 우위 명확. (0,0) 비대칭(eco20 좌/우 L/R 8.9, mesh 1.2)을 between-taxel evenness로 정량화. 맵=소재별 자체정규화, |ΔS| drift보정. (0,0) 자료는 그대로 유지.

## Fig.3 / Bending-aware SR (C2) — 설계+스캐폴드 (2026-06-24)
- **목표**: flat에서 학습한 SATS 코어를 **재학습 없이** 밴딩 상태에서 동작. 전략 = residual correction(잔차보정). 무접촉 baseline 이동에서 곡률 θ 자가추정 → Δp_bend 제거 → 동결 SATS.
- **설계문서**: `sensor_training/docs/superpowers/specs/2026-06-24-bending-aware-sr-design.md`.
- **코드 스캐폴드**: `sensor_training/sats/bending/` — `geometry.py`(taxel 좌표/z_i), `residual_corrector.py`(완전구현), `curvature_estimator.py`(Δp→θ̂ 회귀+closed-form+온라인EMA), `bending_baseline.py`(θ→Δp_bend fit), `bending_aware_sats.py`(동결 SATS wrapper), `datasets.py`/`train_curvature.py`/`eval_bent_sr.py`(데이터 취득 후 TODO). 단위테스트 `tests/test_residual_corrector.py` 6/6 통과(합성 데이터, pytest 미설치 시 직접 실행).
- **Fig.3 구조도**: `visualizing_scripts/fig3/generate_fig3_algorithm.py` → `Fig3A_algorithm.png` (matplotlib, 한글 폰트=Malgun Gothic). flat 학습→동결 + bending 경로 전체 알고리즘.
- **Fig.3 구조도 (논문용 Figma)**: Figma 디자인 파일 `Fig3 Bending-aware SR Algorithm` — https://www.figma.com/design/HcGcsnPYNawphYs1GLKl1Y (영문 라벨, 단방향 화살표, 2-레인 ①flat/②bending, frozen-weights 점선 엘보). use_figma로 박스=auto-layout, 화살표=Vector(정점별 ARROW cap, 단방향).
- **데이터 필요**: 각도별 무하중 baseline(jig) — merged.bin/baseline.json에 `bend_angle_deg` 라벨 추가 필요. 취득 후 train_curvature/eval_bent_sr 연결 → 패널 B/C/D/E.
- **가정/비목표**: 단일 축 굽힘·무하중 baseline 관측. 2축·비틀림·conditioning은 후속.

## 논문용 Figure Set (HTML, 2026-06-24 / 2026-06-29 pptx 통합 확장)
- **산출물**: `visualizing_scripts/figure_set/figure_set.html` — 단일 자가완결(이미지 base64 내장, ~3.2MB). **Fig.2·Fig.3·Fig.4 각각 3×3** 패널 set, 논문풍 serif 캡션 + 패널 레터 `(a)~(i)`.
- **생성 파이프라인**: `generate_atomic_panels.py`(xy_1mm → atomic PNG 8종) + `generate_benchmark_panels.py`(pptx 수치 → 막대 2종) + (pptx 슬라이드 PowerPoint COM 렌더 → PIL 크롭 `panels/pptx_*.png`) → `build_figure_set_html.py`(base64 + inline SVG + HTML 표 조립).
- **20260629.pptx 통합**: 기존 figure 작업(소재평가·SR 모델 벤치마크·Module A~D 밴딩 네트워크·정량 수치)을 흡수. pptx는 zip→`ppt/slides/*.xml`+`ppt/media/`, 슬라이드 렌더는 PowerPoint COM `Export`(LibreOffice 없음).
- **Fig.2 (C1 소재 ablation)**: (a)평면도 (b)단면 (c)mesh 메커니즘[SVG] (d–f)수용장 heatmap d5 (g)radial (h)정량 메트릭 (i)d10 σ 단조. *d5 eco20 σ는 저SNR inflated라 (i)는 d10 사용, radial 범례 σ→peak.*
- **Fig.3 (SR 학습구조 & 모델 벤치마크)**: (a)SATS 파이프라인[SVG hero] (b)X/Y/Z/Fz error map[pptx s17] (c)9-모델 localization map[pptx s26] (d)소재×인덴터 3D 수용장[pptx s9] (e)localization 리더보드 막대[SATS 0.58mm 최저] (f)리더보드 표(loc+z)[HTML] (g)소재별 SR R² 막대 (h)소재 SR 표 About[x]/[z][HTML] (i)요약 카드. 수치 출처=pptx slide 9/23/26.
- **Fig.4 (C2 Bending-aware SR)**: (a)**Module A~D 멀티모듈 네트워크[SVG hero, pptx s18/19 재작성]**(A:baseline→θ̂ / B:CNN encoder / C:SATS localization+lock / D:contact mechanics→z·Fz·Area), (b–e)미취득 패널 placeholder('DATA PENDING·jig'), (f)잔차분해 수식 카드(§5.4).
- **검증**: headless Chrome(Google Chrome 경로) 스크린샷으로 두 아키텍처 SVG·정량 표·전체 레이아웃 육안 확인. 미리보기 `preview_full.png`.
- **남은 것**: jig 데이터 취득 시 Fig.4 (b–e) placeholder를 실제 패널로 교체 → 재빌드. Fig.2 패널 D(소재별 SATS RMSE)는 Fig.3 벤치마크로 일부 충족.
