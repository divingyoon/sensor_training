# Skin Sensor Preprocessing

이 문서는 `/home/user/skin_ws/preprocessing/build_preprocessing_dataset.py` 사용법과 데이터 구조를 설명합니다.

## 1. 목적

원본 CSV(`raw_data/*.csv`)를 학습용 샘플로 변환합니다.

핵심 처리:
- 소재(material) 기준 자동 분류 (`eco20`, `eco50`, `ours` 등)
- `loading / unloading` 분리
- 장비 depth 단위(기본 `0.5 mm`) 기준 depth binning
- baseline subtraction
- zero-phase filtering (Butterworth + filtfilt)
- 샘플별 `npy + csv` 동시 저장
- pseudo HR contact map 생성

---

## 2. 입력 데이터 규칙

원본 파일 위치:
- `/home/user/skin_ws/preprocessing/raw_data`

파일명 규칙(권장):
- `eco20_1.csv`, `eco20_2.csv`, `eco50_1.csv`, `ours_1.csv` ...

소재(material) 추출 규칙:
- 파일명에서 첫 `_` 앞 문자열을 material로 사용
- 예: `eco20_1` -> `eco20`, `ours_3` -> `ours`

CSV 필수 컬럼:
- `X`, `Y`, `Z`, `Fx`, `Fy`, `Fz`
- `Skin1 ... SkinN` (현재 데이터는 `Skin1..Skin16`)

---

## 3. 실행 방법

기본 실행:

```bash
python3 /home/user/skin_ws/preprocessing/build_preprocessing_dataset.py \
  --raw-dir /home/user/skin_ws/preprocessing/raw_data \
  --out-dir /home/user/skin_ws/preprocessing/preprocessing_data \
  --overwrite
```

기본값:
- `--depth-step-mm 0.5`
- `--radius-mm 3.0`
- `--map-size 64`

특정 옵션 포함 예시:

```bash
python3 /home/user/skin_ws/preprocessing/build_preprocessing_dataset.py \
  --raw-dir /home/user/skin_ws/preprocessing/raw_data \
  --out-dir /home/user/skin_ws/preprocessing/preprocessing_data \
  --depth-step-mm 0.5 \
  --radius-mm 3.0 \
  --xyz-scale 0 \
  --overwrite
```

`--xyz-scale` 설명:
- `0`이면 자동 스케일
- `|X,Y,Z|` 최대값이 큰 경우 자동으로 `1e-3` 적용 (예: um -> mm 추정)
- 장비 단위를 확정했다면 명시값 사용 권장 (예: `--xyz-scale 0.001`)

---

## 4. 출력 구조

출력 루트:
- `/home/user/skin_ws/preprocessing/preprocessing_data`

디렉터리 트리:

```text
preprocessing_data/
  dataset_index.json
  material_index_eco20.json
  material_index_eco50.json
  material_index_ours.json
  trial_stats_eco20_1.json
  trial_stats_eco20_2.json
  ...
  eco20/
    eco20_1/
      loading/
        depth_000.000mm/
          rep_0000/
            tactile_lr.npy
            tactile_lr.csv
            tactile_lr_norm.npy
            tactile_lr_norm.csv
            aux_feat.npy
            aux_feat.csv
            hr_contact_map.npy
            hr_contact_map.csv
            meta.json
          rep_0001/
          ...
        depth_000.500mm/
        ...
      unloading/
        depth_130.000mm/
          rep_0000/
          ...
  eco50/
    eco50_1/
    ...
  ours/
    ours_1/
    ...
```

---

## 5. sample이 나뉘는 방식

이 스크립트는 **시간 1프레임 = 1샘플** 방식입니다.

다만 폴더 구조는 depth 기준으로 보기 쉽게 정리됩니다:
1. trial에서 loading/unloading 분리
2. 각 프레임에 대해 `depth_mm_raw = Z - Z_min`
3. `depth_mm = round(depth_mm_raw / depth_step) * depth_step`
4. 같은 depth bin에 들어가는 프레임은 `rep_0000`, `rep_0001` ... 로 저장

즉,
- `depth_012.500mm/rep_0000`
- `depth_012.500mm/rep_0001`
는 같은 depth bin의 서로 다른 시간 프레임입니다.

---

## 6. 샘플 파일 설명

각 샘플 폴더(`.../rep_xxxx/`) 안:

- `tactile_lr.npy` / `tactile_lr.csv`
  - baseline + filtering 적용된 tactile 벡터 (Skin 채널)
- `tactile_lr_norm.npy` / `tactile_lr_norm.csv`
  - 샘플 단위가 아니라 phase 내 통계로 정규화된 tactile 벡터
- `aux_feat.npy` / `aux_feat.csv`
  - `[fx_N, fy_N, depth_mm, indenter_radius_mm]`
- `hr_contact_map.npy` / `hr_contact_map.csv`
  - 구형 인덴터 prior 기반 soft pseudo contact map (`H x W`)
- `meta.json`
  - material, trial, phase, depth_bin, repeat_idx, force, center, area 등 메타데이터

---

## 7. 인덱스 파일 설명

### `dataset_index.json`
- 전체 샘플 인덱스
- 각 샘플의 `material`, `trial_id`, `phase`, `depth_bin_mm`, `repeat_idx`, `sample_dir` 포함

### `material_index_{material}.json`
- 소재별 샘플 목록
- 해당 material의 trial 목록과 샘플 경로만 모아서 제공

### `trial_stats_{trial_id}.json`
- trial별 통계
- raw loading/unloading row 수
- phase별 샘플 수, depth 범위, tactile mean/std

---

## 8. 주요 옵션

- `--raw-dir`: 원본 CSV 폴더
- `--out-dir`: 전처리 결과 저장 폴더
- `--glob`: 파일 패턴 (기본 `*.csv`)
- `--overwrite`: 출력 폴더 초기화 후 재생성
- `--depth-step-mm`: depth bin 간격 (기본 `0.5`)
- `--xyz-scale`: `X,Y,Z` 스케일 (`0`=자동)
- `--radius-mm`: 구형 인덴터 반경
- `--map-size`: contact map 해상도 (`H=W`)
- `--baseline-window`: baseline 계산에 사용할 초기 프레임 수
- `--filter-order`, `--filter-cutoff`: zero-phase low-pass 필터 파라미터
- `--sigma-min-mm`: pseudo map 최소 sigma
- `--contact-threshold-ratio`: 면적 계산 threshold 비율

---

## 9. 권장 운용 방법

1. raw_data에 CSV 추가 (`eco20_*.csv`, `eco50_*.csv`, `ours_*.csv`)
2. `--overwrite`로 전체 재생성
3. `material_index_*.json`으로 소재별 dataset split 구성
4. 학습 코드는 `dataset_index.json` 또는 `material_index_*.json`을 기준으로 로딩

---

## 10. 빠른 체크 명령

전체 샘플 수 확인:

```bash
python3 - << 'PY'
import json
p='/home/user/skin_ws/preprocessing/preprocessing_data/dataset_index.json'
with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
print('num_samples_total =', d['num_samples_total'])
PY
```

소재 인덱스 파일 확인:

```bash
ls -1 /home/user/skin_ws/preprocessing/preprocessing_data/material_index_*.json
```

---

## 11. 주의사항

- unloading 데이터가 짧게 기록된 trial은 unloading 샘플이 적을 수 있습니다.
- `X,Y,Z` 단위가 장비마다 다르면 `--xyz-scale`을 고정값으로 맞추는 것을 권장합니다.
- pseudo HR map은 analytic prior 기반 라벨이므로, 추후 학습에서 sensor/force consistency loss와 함께 쓰는 것을 권장합니다.
