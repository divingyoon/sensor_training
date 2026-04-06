import pandas as pd
import numpy as np
import os
import glob
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- 설정: project-relative paths ---
LOG_DIR = str(Path(__file__).resolve().parents[1] / "logs")
MODEL_DIR = str(Path(__file__).resolve().parent / "models")

# --- MLP 하이퍼파라미터 설정 ---
MLP_CONFIG = {
    'hidden_layer_sizes': (256, 128, 64),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'early_stopping': True,
    'n_iter_no_change': 15,
    'verbose': True
}

# 각 포인트(P_i)별로 사용할 센서 목록
SENSOR_MAPPING = {
    1: ['s1', 's2', 's5', 's6'],
    2: ['s2', 's3', 's6', 's7'],
    3: ['s3', 's4', 's7', 's8'],
    4: ['s5', 's6', 's9', 's10'],
    5: ['s6', 's7', 's10', 's11'],
    6: ['s7', 's8', 's11', 's12'],
    7: ['s9', 's10', 's13', 's14'],
    8: ['s10', 's11', 's14', 's15'],
    9: ['s11', 's12', 's15', 's16'],
}

# --- 데이터 로딩 및 전처리 함수 ---
def _load_and_prepare_df(file_path, sensor_cols, all_output_cols):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"오류: '{os.path.basename(file_path)}' 로드 실패: {e}")
        return None

    df.rename(columns=lambda c: c.replace('_afd50', '').replace('_motor', ''), inplace=True)
    if 'z_displacement_mm_laser' in df.columns:
        df.rename(columns={'z_displacement_mm_laser': 'z'}, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]

    all_needed = sensor_cols + all_output_cols
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        return None

    for col in all_needed:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=all_needed, inplace=True)
    return df if not df.empty else None

def _compute_baseline(df, sensor_cols, force_cols, eps=0.1):
    try:
        no_load = (df[force_cols[0]].abs() < eps) & (df[force_cols[1]].abs() < eps) & (df[force_cols[2]].abs() < eps)
        if no_load.any():
            return df.loc[no_load, sensor_cols].mean()
    except Exception:
        pass
    return pd.Series(0, index=sensor_cols)

# --- 메인 로직 ---
def train_6dof_model(p_i):
    print(f"\n{'='*50}")
    print(f"Processing Point P{p_i} for 6-DOF Model")
    print(f"{ '='*50}\n")

    if p_i not in SENSOR_MAPPING:
        print(f"경고: P{p_i}에 대한 센서 매핑 정보가 없습니다. 건너뜁니다.")
        return

    sensor_cols = SENSOR_MAPPING[p_i]
    output_cols = ['fx', 'fy', 'fz', 'x_c', 'y_c', 'z']

    search_pattern = os.path.join(LOG_DIR, f"p{p_i}_*", "merged_data.csv")
    data_files = glob.glob(search_pattern)
    if not data_files:
        print(f"정보: P{p_i}에 대한 merged_data.csv 파일을 찾을 수 없습니다. 건너뜁니다.")
        return

    df_list = [_load_and_prepare_df(f, sensor_cols, output_cols) for f in sorted(data_files)]
    df_list = [df for df in df_list if df is not None]

    if not df_list:
        print(f"정보: P{p_i}에 대한 유효한 데이터를 로드하지 못했습니다. 건너뜁니다.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"정보: P{p_i}에 대한 총 {len(full_df)}개의 데이터 포인트를 로드했습니다.")

    baseline = _compute_baseline(full_df, sensor_cols, ['fx', 'fy', 'fz'])
    s_prime_df = full_df[sensor_cols] - baseline

    X = s_prime_df.values
    Y = full_df[output_cols].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    print("정보: 입력(X)과 출력(Y)에 대해 각각 표준 스케일링을 적용합니다...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    Y_train_scaled = scaler_y.fit_transform(Y_train)

    print("정보: 신경망 모델(MLPRegressor) 학습을 시작합니다...")
    model = MLPRegressor(random_state=42, **MLP_CONFIG)
    model.fit(X_train_scaled, Y_train_scaled)
    print("정보: 모델 학습 완료.")

    Y_pred_scaled = model.predict(X_test_scaled)
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled) # 원래 스케일로 복원

    print("\n--- 6-DOF 모델 성능 (Neural Network - MLP) ---")
    for i, name in enumerate(output_cols):
        r2 = r2_score(Y_test[:, i], Y_pred[:, i])
        print(f"{name} 예측 R² 점수: {r2:.4f}")
    print("---------------------------------------------------\n")

    print("정보: 학습된 모델과 스케일러를 파일로 저장합니다...")
    os.makedirs(MODEL_DIR, exist_ok=True) # 모델 저장 디렉토리 생성
    output_data = {'model': model, 'scaler_x': scaler_x, 'scaler_y': scaler_y}
    model_filename = os.path.join(MODEL_DIR, f'p{p_i}_6dof_model.joblib')
    joblib.dump(output_data, model_filename)
    print(f"성공: 모델을 '{model_filename}'에 저장했습니다.\n")

if __name__ == '__main__':
    # logs 폴더에서 p*_*... 형태의 모든 폴더를 찾아 자동으로 학습을 수행
    search_pattern = os.path.join(LOG_DIR, "p*_*")
    all_paths = glob.glob(search_pattern)
    p_folders = [path for path in all_paths if os.path.isdir(path)]

    point_ids = set()
    for folder in p_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        match = re.match(r"p(\d+)_", folder_name)
        if match:
            point_ids.add(int(match.group(1)))

    if not point_ids:
        print(f"오류: '{LOG_DIR}'에서 p*_*... 형태의 로그 폴더를 찾을 수 없습니다.")
    else:
        print(f"정보: 다음 포인트들에 대해 학습을 시작합니다: {sorted(list(point_ids))}")
        for p_id in sorted(list(point_ids)):
            train_6dof_model(p_id)
