import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import re

def calculate_metrics(y_true, y_pred):
    """Calculate MSE, RMSE, MAE, R^2."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def process_test_directory(test_path, test_name):
    """
    Processes a single test directory to calculate and save a calibration matrix.
    """
    print(f"\n{'='*50}")
    print(f"Processing Test: {test_name}")
    print(f"{'='*50}")

    # --- 1. 데이터 로드 ---
    all_files = []
    for root, _, files in os.walk(test_path):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print(f"경고: '{test_name}' 폴더에 CSV 파일이 없습니다. 이 테스트를 건너뜁니다.")
        return

    # 파일 이름에서 P_i를 추출하여 데이터프레임에 추가
    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        match = re.search(r'p(\d+)', os.path.basename(f))
        if match:
            p_i = int(match.group(1))
            temp_df['P_i'] = p_i
            df_list.append(temp_df)

    if not df_list:
        print(f"오류: '{test_name}'에서 P_i 정보를 포함하는 유효한 데이터 파일을 찾을 수 없습니다. 이 테스트를 건너뜁니다.")
        return
        
    df = pd.concat(df_list, ignore_index=True)

    # --- 2. Baseline 보정 ---
    force_cols = ['afd_fx', 'afd_fy', 'afd_fz']
    no_load_condition = (df[force_cols[0]].abs() < 0.1) & (df[force_cols[1]].abs() < 0.1) & (df[force_cols[2]].abs() < 0.1)
    sensor_cols = [f's{i}' for i in range(1, 17)]
    
    if not no_load_condition.any():
        print("경고: Baseline을 계산할 무부하 상태 데이터를 찾을 수 없습니다. Baseline을 0으로 간주합니다.")
        baseline = pd.Series(0, index=sensor_cols)
    else:
        baseline = df[no_load_condition][sensor_cols].mean()

    S_prime_df = df[sensor_cols] - baseline

    # --- 3. 행렬 생성 ---
    F_matrix = df[force_cols].values.T
    S_prime_matrix = S_prime_df.values.T

    if S_prime_matrix.shape[1] == 0:
        print(f"오류: '{test_name}'에서 처리할 데이터가 없습니다. 이 테스트를 건너뜁니다.")
        return

    # --- 4. 데이터 유효성 검사 ---
    print("\n--- 데이터 유효성 검사 ---")
    cond_number = 0
    try:
        cond_number = np.linalg.cond(S_prime_matrix)
        print(f"데이터 행렬의 Condition Number: {cond_number:.2f}")
        if cond_number > 1000:
            print("경고: Condition Number가 매우 높습니다. (> 1000)")
            print("  - 데이터의 다중공선성 문제로 인해 교정 행렬의 신뢰도가 낮을 수 있습니다.")
        elif cond_number > 100:
            print("주의: Condition Number가 다소 높습니다. (> 100)")
        else:
            print("정보: Condition Number가 안정적인 수준입니다.")
    except np.linalg.LinAlgError as e:
        print(f"오류: Condition Number 계산 실패({e}). 이 테스트를 건너뜁니다.")
        return

    num_sensors = S_prime_matrix.shape[0]
    num_data_points = S_prime_matrix.shape[1]
    print(f"사용된 데이터 포인트 수: {num_data_points} (센서 수: {num_sensors})")
    if num_data_points < num_sensors * 2:
        print(f"경고: 데이터 포인트 수가 센서 수에 비해 충분하지 않을 수 있습니다. (최소 {num_sensors*2}개 권장)")
    print("--------------------------\n")

    # --- 5. 교정 행렬 계산 ---
    try:
        C_matrix = F_matrix @ np.linalg.pinv(S_prime_matrix)
    except np.linalg.LinAlgError as e:
        print(f"오류: 의사역행렬 계산 실패({e}). 이 테스트를 건너뜁니다.")
        return

    # --- 6. 오차 계산 ---
    F_predicted = C_matrix @ S_prime_matrix
    results = []
    for p_i in range(1, 10):
        p_indices = df.index[df['P_i'] == p_i].tolist()
        if not p_indices: continue
        true_forces_p = F_matrix[:, p_indices]
        pred_forces_p = F_predicted[:, p_indices]
        for i, axis in enumerate(force_cols):
            mse, rmse, mae, r2 = calculate_metrics(true_forces_p[i], pred_forces_p[i])
            results.append({'Point': f'P{p_i}', 'Axis': axis, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    metrics_df = pd.DataFrame(results)

    # --- 7. 결과 저장 ---
    c_df = pd.DataFrame(C_matrix, index=['C_fx', 'C_fy', 'C_fz'], columns=[f's{i}' for i in range(1, 17)])
    today_str = datetime.now().strftime('%y%m%d')
    output_filename = f'cal_{test_name}_{today_str}.csv'

    with open(output_filename, 'w', newline='') as f:
        f.write("Calibration Matrix (C),3x16\n")
        c_df.to_csv(f)
        f.write("\nError Metrics per Sensing Point\n")
        metrics_df.to_csv(f, index=False)
        f.write("\nData Validation Summary\n")
        f.write(f"Condition Number,{cond_number:.2f}\n")
        f.write(f"Number of Data Points,{num_data_points}\n")

    print(f"성공: 교정 행렬과 오차 지표를 '{output_filename}'에 저장했습니다.")


def main():
    """
    Main function to find and process each test directory for calibration.
    """
    data_root = 'data_calibration'
    
    try:
        test_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith('test')]
    except FileNotFoundError:
        print(f"오류: '{data_root}' 폴더를 찾을 수 없습니다.")
        return

    if not test_dirs:
        print(f"오류: '{data_root}' 폴더 안에 'test'로 시작하는 하위 폴더가 없습니다.")
        return

    for test_name in sorted(test_dirs):
        test_path = os.path.join(data_root, test_name)
        process_test_directory(test_path, test_name)

if __name__ == '__main__':
    main()
