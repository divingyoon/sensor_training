# 고해상도 contact map 복원

# 결론

**이 기준으로 다시 진행하면 됨.**

다만 목표를 다음처럼 명확히 고정해야 함.

**입력**

- 저해상도 barometric tactile array
- 보조 입력: $(F_x, F_y)$ 참고값
- 압입 진행량 또는 depth-axis index

**최종 출력**

- 고해상도 contact map
- 부출력: $x, y, z, F_z, A_c$

**“저해상도 tactile response로부터 구형 인덴터에 의한 단일 접촉 pressure/contact field를 고해상도로 복원하는 문제”**

2022 RAL은 제한된 MEMS 센서 수로부터 **Gaussian 기반 pressure distribution**을 재구성하고 그로부터 위치와 normal force를 추정했으며 , 2024 Access는 **iterative fitting 기반 parameterized pressure distribution**으로 arbitrary pressure distribution reconstruction을 수행했음 . 2025 TRO는 저해상도 barometer spacing보다 훨씬 높은 localization precision을 얻기 위해 **MLP 기반 super-resolution localization**을 사용했음 .

---

# 1. 문제를 어떻게 다시 정의할지

## 기존 정의

- $x, y, z, F_z, A_c$ 회귀

## 수정 후 정의

- **주과제:** 고해상도 contact map 복원
- **부과제:** contact map으로부터 $x, y, z, F_z, A_c$ 계산 또는 회귀

이렇게 바꾸는 이유는 명확함.

1. 최종적으로 원하는 것이 scalar 5개가 아니라 **공간 분포**임
2. $x, y, A_c$는 contact map에서 자연스럽게 계산 가능함
3. $F_z$는 pressure map의 적분과 연결되는 물리량이므로 map 기반이 더 해석 가능함
4. 구형 인덴터는 contact shape prior를 줄 수 있어서 map reconstruction에 매우 유리함

즉 지금부터는 **direct regression 중심**이 아니라 **map reconstruction 중심 + geometry-aware supervision**으로 가는 것

---

# 2. 구형 인덴터라는 조건이 왜 중요한가

구형 인덴터이면 contact patch는 임의 shape가 아니라, 기본적으로 **축대칭 압력 분포** 또는 **원형 접촉영역**에 가까움. 이 말은 곧, 고해상도 contact map의 ground truth를 완전히 측정하지 못하더라도 아래 두 가지를 할 수 있다는 뜻임.

## 2-1. contact area prior를 줄 수 있음

구형 압입에서는 접촉면적이 임의 다각형처럼 나오지 않고, 대체로 **원형 또는 nearly circular patch**가 됨.

따라서 복원된 map에 대해 다음과 같은 제약을 줄 수 있음.

- 단일 connected component
- center가 하나
- isotropic 또는 near-isotropic spread
- radial monotonicity 경향

## 2-2. analytic pseudo-ground-truth 생성 가능

구형 인덴터 반경 (R), 압입 깊이 (\delta)가 있으면, 이상적인 rigid geometry 기준으로 contact radius는 대략 다음 형태로 둘 수 있음.

$$
a \approx \sqrt{2R\delta - \delta^2}
$$

작은 압입에서는 보통

$$
a \approx \sqrt{R\delta}
$$

형태의 근사도 많이 씀.

즉 각 depth step마다

- center $(x,y)$
- radius $a$
를 갖는 **원형 contact mask**
    
    또는
    
- 중심이 강하고 가장자리로 약해지는 **radial pressure map**
을 pseudo label로 만들 수 있음.

단, Ecoflex는 선형 탄성체가 아니고 large deformation 및 compressibility 영향이 있으므로, 이것을 **정답 그 자체**로 두면 안 됨.

Steck 논문도 Ecoflex의 응답이 large deformation에서 단순 모델로 충분히 설명되지 않으며 compressible/incompressible 조건에 따라 multiaxial response가 달라진다고 명시함 .
따라서 이 analytic map은 **hard GT**가 아니라 **shape prior**로 써야 함.

---

# 3. 추천하는 최종 학습 구조

## 가장 추천하는 구조

**Physics-informed SR reconstruction network**

### 입력

각 depth step 또는 depth block에서

- **baseline**-corrected tactile vector
- **tactile spatial feature**
- $F_x, F_y$ 참고 feature
- depth $z$
- 인덴터 반경 $R$ 또는 고정 인자

### 출력

- $\hat{M} \in \mathbb{R}^{H \times W}$: 고해상도 contact/pressure map

### 부출력

- $\hat{x}, \hat{y}$
- $\hat{F}_z$
- $\hat{A}_c$

여기서 (z)는 입력으로 넣는 것이 더 적절함.
이유는 지금 실험이 단순 압입이고, map의 크기와 spread는 depth에 강하게 종속되기 때문임. 즉 현재 과제에서 (z)는 예측 대상이기보다 **조건(conditioning variable)** 로 쓰는 편이 훨씬 유리함.

강한 의견으로, **이번 문제에서는 (z)를 출력으로 두지 말고 입력/조건으로 두는 것이 맞음.**

이유:

- 압입 실험에서 (z)는 이미 모터/스테이지 기준으로 알고 있음
- 모르는 값을 예측하는 것이 아니라, 알고 있는 물리 상태에서 contact field를 복원하는 것이 현재 목적임
- 이렇게 해야 네트워크가 불필요하게 inverse ambiguity를 떠안지 않음

---

# 4. 모델 아키텍처 구체안

## 4-1. baseline

**MLP encoder + CNN decoder**

### 구조

1. 저해상도 tactile vector를 MLP encoder로 latent화
2. depth $z$, $F_x,F_y$를 concat
3. latent를 작은 2D feature map으로 reshape
4. upsampling CNN decoder로 고해상도 map 생성

### 장점

- 구현이 가장 쉬움
- 현재 데이터 크기에서 안정적임
- TRO 2025에서 MLP가 SR localization에 이미 강한 baseline 역할을 했음

---

## 4-2. 추천 본선 모델

**1D CNN encoder + FiLM conditioning + 2D map decoder**

### 입력 형태

$$
X \in \mathbb{R}^{K \times C}
$$

- $K$: depth grid step 수
- $C$: tactile + 보조 feature 수

즉 한 점만 쓰지 않고 **loading 구간 전체 depth profile**을 사용함.

### 구조

- encoder: 1D CNN
- conditioning: $z$, $R$, optional $F_x,F_y$
- decoder: 2D CNN / UNet-lite
- output: $H \times W$ contact map

### 이유

- 단일 depth frame보다 depth progression 전체를 쓰면 노이즈에 강함
- same contact point라도 sensor response의 누적 패턴이 더 안정적임
- 하지만 시간축이 아니라 **depth-axis 시퀀스**여야 함

---

# 5. ground truth를 어떻게 만들지

고해상도 contact map이 최종 목표이면, GT 생성 방식이 가장 중요함.

## 권장안 A — analytic soft label

구형 인덴터 기반으로 depth마다 soft circular pressure prior를 생성함.

### 방법

각 sample에서

- 중심 $(x_0,y_0)$
- 반경 $a(z)$
- peak magnitude $p_0(F_z)$

를 정의하고, map을 다음처럼 생성함.

### binary contact map

$$
M(x,y) = \begin{cases} 1, & (x-x_0)^2 + (y-y_0)^2 \le a^2 \\ 0, & \text{otherwise} \end{cases}
$$

### soft radial map

$$
M(x,y) = p_0 \exp\left(-\frac{(x-x_0)^2+(y-y_0)^2}{2\sigma(z)^2}\right)
$$

혹은 truncated paraboloid 형태도 가능함.

### 장점

- 지금 바로 구축 가능
- 구형 인덴터 prior 반영 가능
- map learning을 시작할 수 있음

### 한계

- 실제 Ecoflex deformation과 완전히 일치하지 않음
- 따라서 “정답”이 아니라 **pseudo target**임

---

## 권장안 B — weak supervision + consistency

GT map을 완전히 신뢰하지 않고, 다음 제약을 함께 둠.

1. **sensor consistency**
    - 복원된 HR map을 sensor layout로 downsample 했을 때 실제 tactile reading과 일치해야 함
        
        $$
        \mathcal{L}{sensor} = | D(\hat{M}) - s{\text{measured}} |_1
        $$
        
    
    여기서 $D(\cdot)$는 고해상도 map을 저해상도 sensor response로 변환하는 operator임.
    
2. **shape prior**
    - 단일 원형성
    - connectedness
    - radial smoothness
3. **force consistency**
    - map 적분값이 measured $F_z$와 일치해야 함
        
        $$
        \mathcal{L}{F_z} = \left| \sum{x,y} \hat{M}(x,y)\Delta a - F_z \right|
        $$
        

이 방식이 가장 타당함.

즉 pseudo label만 믿지 말고, **measured sensor + measured force + geometric prior**를 같이 묶어야 함.

강한 의견으로,

**이번 과제는 supervised SR이라기보다 physics-informed weakly supervised SR reconstruction으로 정의하는 것이 가장 맞음.**

---

# 6. 최종 loss 설계

권장 loss는 아래처럼 구성함.

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{map}+\lambda_2 \mathcal{L}_{sensor}+\lambda_3 \mathcal{L}_{F_z}+\lambda_4 \mathcal{L}_{center}+\lambda_5 \mathcal{L}_{area}+\lambda_6 \mathcal{L}_{smooth}
$$

## 각 항 의미

$\mathcal{L}_{map}$ : analytic pseudo contact map과의 차이 [BCE 또는 L1]

$\mathcal{L}_{sensor}$ : 복원 map을 sensor grid로 내렸을 때 실제 sensor와 맞는지

$\mathcal{L}_{F_z}$ : map 적분과 measured normal force 일치

$\mathcal{L}_{center}$ : map centroid가 target (x,y)와 일치

$\mathcal{L}_{area}$ : map threshold area가 target area와 일치

$\mathcal{L}_{smooth}$ : 고주파 artifact 억제용 TV loss 또는 Laplacian smoothness

---

# 7. 데이터 전처리, 이제 어떻게 바꿔야 하는가

## 7-1. 유지할 것

- 시간축 직접 사용 안 함
- depth-axis 재파라미터화
- baseline subtraction
- zero-phase filtering
- loading phase 우선 사용

## 7-2. 새로 추가할 것

### A. center label 정밀화

각 trial/sample마다 접촉 중심 ((x,y))를 정확히 관리해야 함.

스테이지 위치 또는 indenter fixture 기준으로 ground truth center를 고정된 sensor-plane 좌표계로 변환해야 함.

즉 메타데이터에 반드시 필요함.

- sensor frame origin
- stage frame to sensor frame transform
- contact center label

### B. spherical prior parameter 저장

각 sample마다

- indenter radius (R)
- commanded depth (z)
- measured F_z

를 함께 저장함.

### C. HR map label 파일 생성

각 sample마다 아래 둘 중 하나를 생성함.

- binary contact mask png/npy
- soft radial pressure map npy

강한 권고:

**PNG보다 NPY 또는 float32 array로 저장하는 것이 맞음.**

이유는 pressure magnitude 정보와 soft label을 유지해야 하기 때문임.

---

# 8. 실제 데이터셋 포맷 추천

## sample 단위

하나의 sample은 depth step 하나 또는 depth block 하나로 저장

### 권장 파일 구조

```
sample_i/
  tactile_lr.npy
  tactile_lr_norm.npy
  aux_feat.npy
  hr_contact_map.npy
  meta.json
```

### meta.json 예시

```json
{
  "trial_id": "eco20_mesh_d6_rep03",
  "material": "eco20_mesh",
  "indenter_type": "sphere",
  "indenter_radius_mm": 3.0,
  "depth_mm": 1.0,
  "fx_N": 0.02,
  "fy_N": -0.01,
  "fz_N": 3.42,
  "contact_center_x_mm": 12.4,
  "contact_center_y_mm": 8.7,
  "contact_area_mm2": 6.21
}
```

---

# 9. 실험 순서

## Phase 1

**SR localization only**

- 출력: contact map + centroid
- 평가: center error

## Phase 2

**contact map + force consistency**

- 출력: contact map + $F_z$
- 평가: center error, $F_z$ MAE

## Phase 3

**contact map + area**

- 출력: contact map + $A_c$
- 평가: IoU, Dice, area MAE

이 순서가 맞음. 처음부터 area까지 동시에 정확히 잡으려 하면 불안정해짐.

---

# 10. 평가 지표

고해상도 contact map이 최종 목표이면, 평가는 아래처럼 해야 함.

## map 품질

- IoU
- Dice
- centroid error
- area error
- radial profile error

## 물리량 품질

- $F_z$ MAE
- center (x,y) RMSE
- $A_c$ MAE

## consistency 품질

- reconstructed map downsample 결과와 실제 sensor signal 간 L1/L2

---

# 11. 최종 추천안

## 최종 문제 정의

**구형 인덴터 단일 압입에서, depth-axis 정렬된 저해상도 barometric tactile response로부터 고해상도 contact map을 복원하는 physics-informed SR 문제**

## 최종 모델

**1D CNN encoder + conditioning(depth, radius, optional Fx/Fy) + 2D decoder**

## supervision

- analytic spherical pseudo map
- sensor consistency
- $F_z$ consistency
- centroid / area consistency

## 핵심 판단

- (z)는 출력보다 입력 조건으로 두는 것이 맞음
- (F_x,F_y)는 참고 feature만 유지하면 충분함
- 최종 목표는 scalar regression이 아니라 map reconstruction으로 고정해야 함

---

# 바로 실행해도 되는 실무형 정리

## 지금부터 해야 할 일

1. trial별로 depth-axis 정렬
2. 각 depth step의 ((x,y,z,F_z)) 메타데이터 정리
3. 구형 인덴터 반경 기반 pseudo HR map 생성
4. tactile LR 입력과 HR map target 쌍 생성
5. MLP encoder + CNN decoder baseline 먼저 학습
6. 이후 sensor consistency loss 추가
7. 마지막에 area/force head 추가

---