# SATS mk555 기준 재수정 계획

작성일: 2026-06-09
대상: `/home/user/sensor_training/skin_ws`

## 기준

- 공식 수집기는 `skin_ws/acquisition_code/final_logger.py`이다.
- 현재 SATS 스캔 기준 node는 `skin_ws/node/SATS_d5_mk555.node`, `skin_ws/node/SATS_d10_mk555.node`이다.
- 실제 raw data archive 기준 예시는 `skin_ws/raw_data/sats/eco20 + mesh/d5/test1`이다.
- 현재 force 기준은 FT sensor가 아니라 load cell raw `kg` 값이다.
- `u`축은 실제 구동축이 아니라 node 내부 대기/가상 축으로 취급한다.

## 변경점 반영 기준

### 1. 센서 사양

- 센서 두께는 기존 `3.0mm`가 아니라 `5.5mm`로 고정한다.

### 2. 센서 배열

- 센서 좌표는 아래 4x4 배열로 고정한다.

```text
s1  = [-9.75, -9.75]   s2  = [-3.25, -9.75]   s3  = [ 3.25, -9.75]   s4  = [ 9.75, -9.75]
s5  = [-9.75, -3.25]   s6  = [-3.25, -3.25]   s7  = [ 3.25, -3.25]   s8  = [ 9.75, -3.25]
s9  = [-9.75,  3.25]   s10 = [-3.25,  3.25]   s11 = [ 3.25,  3.25]   s12 = [ 9.75,  3.25]
s13 = [-9.75,  9.75]   s14 = [-3.25,  9.75]   s15 = [ 3.25,  9.75]   s16 = [ 9.75,  9.75]
```

### 3. 측정 grid

- 측정 범위는 `x, y = -10.0 ~ 10.0mm`이다.
- XY step은 `0.5mm`이다.
- 전체 위치 수는 `41 x 41 = 1681`이다.
- 센서 좌표는 half-grid 위치이므로 grid index 매핑 시 nearest 규칙이 필요하다.
  - 예: `-9.75 -> -10.0`, `9.75 -> 10.0`

### 4. 측정 순서

- 측정 순서는 각 행마다 항상 `x=-10.0 -> 10.0` 정방향으로 진행한다.
- 한 행이 끝나면 `y`를 `0.5mm` 증가시키고 다시 `x=-10.0 -> 10.0`으로 진행한다.
- 즉 serpentine이 아니라 고정 raster 순서이다.

```text
[-10.0, -10.0] -> x +0.5 -> [10.0, -10.0]
[-10.0,  -9.5] -> x +0.5 -> [10.0,  -9.5]
...
[-10.0,  10.0] -> x +0.5 -> [10.0,  10.0]
```

### 5. force 기준 변경

- 기존 FT sensor `N` 기준 해석은 폐기한다.
- 현재 force 기준은 load cell `kg` raw 값이다.
- 후처리 force는 아래 식으로 재구성한다.

```text
Fz_N = (loadcell_kg - baseline_kg) * 9.80665
```

- `baseline_kg`는 각 trial 시작 idle 구간에서 추정한다.
- 인덴터는 기존과 동일하게 `sphere diameter 10.5mm`를 유지한다.
- 압입 하중이 증가할수록 `+N`으로 해석한다.

### 6. Z 압입 방법

- 현재 압입은 `0.5mm` 간격으로 잠깐 멈추고 다시 내려가는 계단식 탐색이다.
- 센서 접촉 여부와 관계없이 최대 `15.5mm`까지 내려간다.
- unloading은 중간 정지 없이 한 번에 올라온다.
- 따라서 depth 해석은 `u`가 아니라 `z` command 기준으로 재구성해야 한다.

## mk555 Node 확인값

| 항목 | d5 | d10 |
| --- | ---: | ---: |
| X/Y 범위 | -10.0 ~ 10.0mm | -10.0 ~ 10.0mm |
| X/Y step | 0.5mm | 0.5mm |
| X/Y 위치 수 | 41 x 41 = 1681 | 41 x 41 = 1681 |
| 시작 Z | 13.0mm | 12.0mm |
| 최대 Z | 15.5mm | 15.5mm |
| 유효 Z depth | 2.5mm | 3.5mm |
| U 값 범위 | 0.0 ~ 3.0 | 0.0 ~ 4.0 |

각 XY 위치별 node 패턴은 아래처럼 반복된다.

- `d5`
  - `(13.0, 0.0) -> (13.0, 0.5) -> (13.5, 0.5) -> ... -> (15.5, 3.0) -> (15.5, 0.0) -> (13.0, 0.0)`
- `d10`
  - `(12.0, 0.0) -> (12.0, 0.5) -> (12.5, 0.5) -> ... -> (15.5, 4.0) -> (15.5, 0.0) -> (12.0, 0.0)`

해석 규칙:

- 실제 압입 깊이는 `u`가 아니라 `z` 변화량으로 계산한다.
- depth 정의:
  - `d5`: `depth_mm = z_stage_mm - 13.0`
  - `d10`: `depth_mm = z_stage_mm - 12.0`
- `u`는 로그 컬럼으로 보존할 수 있지만, 실제 물리 depth나 GT 계산의 주 기준으로 사용하지 않는다.

## final_logger 기준 로깅 계약

`final_logger.py` 기준 현재 full stack logging 계약:

- DUE: `TARGET_HZ = 200`
- EtherMotion: `--ethermotion-hz` 독립 고속 폴링, 운영 기준 `1000 ~ 2000Hz`
- Load cell: 고정 Hz가 아니라 raw RS232 chunk logging

실로그 `skin_ws/raw_data/sats/eco20 + mesh/d5/test1` 확인값:

- DUE raw burst: `308229` records / `15411.6s` -> 약 `20 burst/s`
- DUE 1 burst 안에는 `16 sensors x 10 FIFO frames`가 들어 있으므로 유효 센서 샘플링은 `20 x 10 = 200Hz`
- EtherMotion encoder: `17147296` records / `15411.6s` -> 약 `1112.6 rec/s`
- EtherMotion header 설정값은 `2000Hz`였지만, 실제 병합은 실측 timestamp 기준으로 처리해야 한다.
- Load cell raw: `1175706` chunks / `15411.6s` -> 약 `76.3 chunk/s`

정리:

- DUE는 `logger burst rate`와 `effective sensor frame rate`를 분리해서 해석한다.
- EtherMotion은 고정 Hz 가정보다 실측 timestamp 정렬이 우선이다.
- Load cell은 고정 주파수 센서가 아니라 variable-size raw stream이다.

## 데이터 구조 기준

현재 raw input 기준 파일 계약:

```text
due_raw_burst_*.bin
ethermotion_encoder_*.bin
loadcell_raw_*.bin
```

현재 확인된 예시 session:

```text
skin_ws/raw_data/sats/eco20 + mesh/d5/test1/
  due_raw_burst_20260608_183223.bin
  ethermotion_encoder_20260608_183223.bin
  loadcell_raw_20260608_183223.bin
```

문서/전처리 수정 시 고정할 merged 출력 의미:

- `s1..s16`: DUE 센서 raw 또는 정규화 전 값
- `x_mm`, `y_mm`: EtherMotion command 기준 XY 위치
- `z_stage_mm`: EtherMotion command 기준 Z 위치
- `z_depth_mm`: node별 시작 Z를 뺀 실제 압입 깊이
- `u_mm`: 보조 로그 컬럼만 유지
- `Fz_N`: load cell `kg -> N` 변환 결과

## 이번 재수정에서 바꿔야 할 해석

- 기존 `mk777` 기준 문서는 폐기한다.
- 기존 `u_mm == 0` 행만 수직압 GT에 쓴다는 규칙은 폐기한다.
- 현재는 `z` 기반 계단식 압입 전체 구간을 depth 기준으로 재해석해야 한다.
- 기존 `d5=2.6mm`, `d10=1.6mm` 해석은 `mk555` 기준에서는 사용하지 않는다.
- force 주 입력은 load cell이며, FT sensor 기반 force 전처리 의존성은 제거한다.

## 검증 항목

- node parser 검증
  - `SATS_d5_mk555.node`, `SATS_d10_mk555.node`가 모두 `41 x 41`, `0.5mm step`, `-10.0 ~ 10.0mm`인지 확인
  - `d5` 시작 Z가 `13.0`, `d10` 시작 Z가 `12.0`인지 확인
  - 최대 Z가 모두 `15.5`인지 확인
- logger 검증
  - DUE burst 기준 `~20Hz`와 FIFO 10-frame 기준 `200Hz` 해석이 일치하는지 확인
  - EtherMotion은 실측 timestamp 기준 정렬인지 확인
  - Load cell은 raw chunk stream으로 처리되는지 확인
- semantics 검증
  - `u`를 실제 이동축이나 depth 기준으로 사용하지 않는지 확인
  - `z_depth_mm`가 node별 시작 Z 기준으로 계산되는지 확인
  - `Fz_N`가 `kg -> N`으로 변환되는지 확인
- raw data 검증
  - `skin_ws/raw_data/sats/eco20 + mesh/d5/test1`를 smoke dataset으로 사용 가능해야 한다.

## 구현 방향

- SATS 전처리와 GT 생성 문서는 모두 `mk555 + loadcell + z-depth` 기준으로 다시 맞춘다.
- training input 생성 시 force는 `loadcell_raw` 기반으로 재구성한다.
- stage label 생성 시 XY는 `41 x 41` raster, Z는 `d5/d10`별 시작점 보정 후 depth로 사용한다.
- `u`는 필요 시 저장만 하고 학습 핵심 feature/label 정의에서는 제외한다.
