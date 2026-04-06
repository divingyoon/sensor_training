import pandas as pd
import numpy as np
import os
import glob
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 설정: project-relative paths ---
LOG_DIR = str(Path(__file__).resolve().parents[1] / "logs")
MODEL_DIR = str(Path(__file__).resolve().parent / "models")

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

# --- 데이터 로딩 및 전처리 함수 (train_local_6dof_model.py에서 복사) ---
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

# --- 평가 로직 ---
def evaluate_model(p_i):
    print(f"\n{'='*50}")
    print(f"Evaluating Saved Model for Point P{p_i}")
    print(f"{'='*50}\n")

    model_filename = os.path.join(MODEL_DIR, f'p{p_i}_6dof_model.joblib')
    if not os.path.exists(model_filename):
        print(f"오류: 모델 파일 '{model_filename}'을 찾을 수 없습니다. 건너뜁니다.")
        return

    # 모델과 스케일러 로드
    saved_data = joblib.load(model_filename)
    model = saved_data['model']
    scaler_x = saved_data['scaler_x']
    scaler_y = saved_data['scaler_y']
    print(f"성공: '{model_filename}'에서 모델과 스케일러를 로드했습니다.")

    # 학습에 사용했던 것과 동일한 방식으로 데이터 로드 및 전처리
    sensor_cols = SENSOR_MAPPING.get(p_i)
    if not sensor_cols:
        print(f"경고: P{p_i}에 대한 센서 매핑 정보가 없습니다. 건너뜁니다.")
        return

    output_cols = ['fx', 'fy', 'fz', 'x_c', 'y_c', 'z']
    search_pattern = os.path.join(LOG_DIR, f"p{p_i}_*", "merged_data.csv")
    data_files = glob.glob(search_pattern)
    
    df_list = [_load_and_prepare_df(f, sensor_cols, output_cols) for f in sorted(data_files)]
    df_list = [df for df in df_list if df is not None]

    if not df_list:
        print(f"정보: P{p_i}에 대한 평가용 데이터를 찾을 수 없습니다.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    baseline = _compute_baseline(full_df, sensor_cols, ['fx', 'fy', 'fz'])
    s_prime_df = full_df[sensor_cols] - baseline

    X = s_prime_df.values
    Y = full_df[output_cols].values

    # 중요: 학습 시와 동일한 random_state로 테스트 데이터를 분리해야 동일한 평가가 가능
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # 평가 수행
    X_test_scaled = scaler_x.transform(X_test)
    Y_pred_scaled = model.predict(X_test_scaled)
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

    print("\n--- Saved 6-DOF Model Performance (Re-evaluation) ---")
    for i, name in enumerate(output_cols):
        r2 = r2_score(Y_test[:, i], Y_pred[:, i])
        print(f"{name} 예측 R² 점수: {r2:.4f}")
    print("-------------------------------------------------------\n")

if __name__ == '__main__':
    # 평가하고 싶은 포인트 ID를 여기에 넣으세요.
    points_to_evaluate = [1, 7, 9] # 예시: 1, 5, 9번 포인트 모델 평가
    
    print(f"정보: 다음 포인트들에 대해 저장된 모델 평가를 시작합니다: {points_to_evaluate}")
    for p_id in points_to_evaluate:
        evaluate_model(p_id)
