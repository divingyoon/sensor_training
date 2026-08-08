# 변형(deformation) 데이터 취득 — 각도-프리 baseline 복원용

각도 라벨을 쓰지 않으므로 **지그·모터·로드셀이 전부 불필요**하다. DUE 16채널만 있으면 된다.

## 1. 실행

```bash
# 취득 PC(Windows). 경로는 이 파일 위치 기준 상대해석 → 어느 폴더에서 실행해도 동일.
python deform_logger_gui.py --version v1                      # 기본: baseline 10s / 변형 240s
python deform_logger_gui.py --version v1 --deform-sec 300     # 변형 5분
python deform_logger_gui.py --version v1 --port COM11
python deform_logger_gui.py --version v1 --mock               # 하드웨어 없이 UI·경로 점검
```

`Space` 또는 **시작** 버튼 → 이후 **자동 진행**(단계 전환·카운트다운·저장 모두 자동).

## 2. 프로토콜 (GUI가 안내)

| 단계 | 시간 | 할 일 |
|---|---|---|
| ① `BASE_HEAD` | 10초 | **손 떼고 평평** — 무접촉·무하중 |
| ② `DEFORM` | 3~5분 | **손으로 순수 변형** (인덴터식 국소 압박 아님) |
| ③ `BASE_TAIL` | 10초 | **다시 손 떼고 평평** |

앞뒤 baseline 두 점으로 **선형 드리프트 보정**(열·기압 표류 제거)이 이루어진다. ③을 빠뜨리면
보정이 불가하므로 **끝까지 진행할 것**.

### ★ 변형 다양성이 성패를 가른다
장비를 줄인 대가로 **학습 분포 커버리지**가 전부다. GUI가 20초마다 가이드를 순환 표시한다.

- **방향**: 위/아래 휨 · 좌우 · **비틀림(torsion)** ← 각도 방식이 못 하던 변형, 여기서 강점
- **크기**: 약한 것 ~ **데모에서 쓸 최대보다 조금 더 강한 극단**
- **시간**: 연속 변형 / **변형 후 3~5초 정지 유지**(정적 프레임 필수) / 빠른 변형
- **부위**: 전체 휨 / 한쪽 끝만 / 국소

> **원칙: 데모에 나올 변형은 반드시 학습에 포함.** 학습 분포 밖에서 유령이 터진다.

세션 종료 시 `deform peak %` 가 표시된다. 세션마다 이 값이 비슷하면 변형이 단조로운 것이니
다음 세션에서 더 과감하게 변형할 것.

## 3. 저장 경로 (분석과 직결)

```
skin_ws/raw_data/deform/<version>/
  s01_20260808_143022/
    due_v2_20260808_143022.bin      ← DUE_V2 포맷(final_logger_gui 와 동일)
    session_meta.json               ← 프레임 수·deform peak·프로토콜 파라미터
  s02_.../
  ...
```

- **버전별 분리**: `--version v1` / `v2` … 센서나 취득 조건이 바뀌면 버전을 올린다.
- **세션 번호 자동 증가**: 같은 버전 폴더에 계속 쌓기만 하면 된다.
- 학습 스크립트가 하위 폴더를 **재귀 탐색**하므로 별도 정리·변환이 필요 없다.

## 4. 바로 분석

```bash
# 학습 PC(Linux). 취득 폴더를 그대로 가리키면 끝.
.venv/bin/python -m sats.bending.train_deform_restorer \
  --deform-root skin_ws/raw_data/deform/v1 \
  --contact-trial learning_data/sensor_raw_bin/ecomesh_v6_xy1/d5/z_2.5mm/test1 \
  --sats-run sats/training/runs/ecomesh_v6_deploy_g025 \
  --latent-dims 2 4 8
```

로드 즉시 세션별 품질 요약(프레임 수·드리프트·|pct|max)이 출력되므로 **취득 직후 점검**에도 쓴다:

```bash
.venv/bin/python -c "from sats.bending.deform_data import load_all; load_all('skin_ws/raw_data/deform/v1')"
```

## 5. 수량 · 검증 세션

- **학습용 6세션** — due만으로 충분
- **검증용 1~2세션** — 변형 상태에서 **실제 접촉**도 수행(학습 제외, 접촉 보존 실측 전용)
  - due만이면 **정성** 검증("유령 없고 접촉이 보인다")
  - **정량**(위치오차 mm, 논문 G2)을 원하면 접촉 위치 GT 필요 → **변형=손 / 접촉=모터 인덴터** 권장
  - 이 경우 세션 폴더에 ethermotion bin 이 같이 있어도 학습 로더는 무시하므로 무방하다

## 6. 참고

- 기존 통합 로거(`final_logger_gui.py`)는 모터·로드셀·AFD50까지 기록하는 xy/밴딩 취득용이다.
  변형 취득은 그 경로를 쓰지 않는다(각도·힘 라벨 불필요).
- 포맷은 동일한 `DUE_V2` 이므로 기존 변환 도구(`convert__bins_gui.py`)도 그대로 적용된다.
