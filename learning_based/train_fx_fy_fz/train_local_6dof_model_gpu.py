import glob
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
MODEL_DIR = Path(__file__).resolve().parent / "models"


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


TRAIN_CONFIG = {
    'hidden_sizes': [256, 128, 64],
    'batch_size': 256,
    'epochs': 250,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 25,
    'min_delta': 1e-4,
}


def _load_and_prepare_df(file_path, sensor_cols, all_output_cols):
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"오류: '{os.path.basename(file_path)}' 로드 실패: {exc}")
        return None

    df.rename(columns=lambda c: c.replace('_afd50', '').replace('_motor', ''), inplace=True)
    if 'z_displacement_mm_laser' in df.columns:
        df.rename(columns={'z_displacement_mm_laser': 'z'}, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]

    all_needed = sensor_cols + all_output_cols
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        print(f"경고: '{os.path.basename(file_path)}'에 필요한 컬럼이 없습니다: {missing}")
        return None

    for col in all_needed:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=all_needed, inplace=True)
    return df if not df.empty else None


def _compute_baseline(df, sensor_cols, force_cols, eps=0.1):
    try:
        mask = (df[force_cols[0]].abs() < eps) & (df[force_cols[1]].abs() < eps) & (df[force_cols[2]].abs() < eps)
        if mask.any():
            return df.loc[mask, sensor_cols].mean()
    except Exception:
        pass
    return pd.Series(0, index=sensor_cols)


class MLPRegressorTorch(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            prev_dim = size
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def _train_torch_model(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features.size(0)

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss + TRAIN_CONFIG['min_delta'] < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= TRAIN_CONFIG['patience']:
                print("조기 종료 조건을 만족하여 학습을 종료합니다.")
                break

    model.load_state_dict(best_state)
    return best_val_loss


def _prepare_dataloaders(X_train, Y_train, X_val, Y_val):
    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(Y_train.astype(np.float32))
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(Y_val.astype(np.float32))
    )

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, drop_last=False)
    return train_loader, val_loader


def train_6dof_model_gpu(p_i):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"\n{'=' * 50}")
    print(f"Processing Point P{p_i} for 6-DOF Model (PyTorch GPU)")
    print(f"{'=' * 50}\n")

    if p_i not in SENSOR_MAPPING:
        print(f"경고: P{p_i}에 대한 센서 매핑 정보가 없습니다. 건너뜁니다.")
        return

    sensor_cols = SENSOR_MAPPING[p_i]
    output_cols = ['fx', 'fy', 'fz', 'x_c', 'y_c', 'z']

    search_pattern = LOG_DIR / f"p{p_i}_*" / "merged_data.csv"
    data_files = glob.glob(str(search_pattern))
    if not data_files:
        print(f"정보: P{p_i}에 대한 merged_data.csv 파일을 찾을 수 없습니다. 건너뜁니다.")
        return

    df_list = [_load_and_prepare_df(path, sensor_cols, output_cols) for path in sorted(data_files)]
    df_list = [df for df in df_list if df is not None]

    if not df_list:
        print(f"정보: P{p_i}에 대한 유효한 데이터를 로드하지 못했습니다. 건너뜁니다.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"정보: P{p_i}에 대한 총 {len(full_df)}개의 데이터 포인트를 로드했습니다.")

    baseline = _compute_baseline(full_df, sensor_cols, ['fx', 'fy', 'fz'])
    s_prime_df = full_df[sensor_cols] - baseline

    X = s_prime_df.to_numpy()
    Y = full_df[output_cols].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    Y_train_scaled = scaler_y.fit_transform(Y_train)
    Y_test_scaled = scaler_y.transform(Y_test)

    X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(
        X_train_scaled, Y_train_scaled, test_size=0.2, random_state=42
    )

    train_loader, val_loader = _prepare_dataloaders(X_train_split, Y_train_split, X_val_split, Y_val_split)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"정보: 학습에 사용할 디바이스: {device}")

    model = MLPRegressorTorch(
        input_dim=len(sensor_cols),
        hidden_sizes=TRAIN_CONFIG['hidden_sizes'],
        output_dim=len(output_cols)
    ).to(device)

    start_time = time.time()
    best_val_loss = _train_torch_model(model, train_loader, val_loader, device)
    elapsed = time.time() - start_time
    print(f"정보: 학습 완료 (소요 시간: {elapsed:.2f}초)")
    print(f"최적 검증 손실(MSE): {best_val_loss:.6f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32)).to(device)
        Y_pred_scaled = model(X_test_tensor).cpu().numpy()

    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

    print("\n--- 6-DOF 모델 성능 (PyTorch GPU) ---")
    metrics_per_output = {}
    for idx, name in enumerate(output_cols):
        mse = mean_squared_error(Y_test[:, idx], Y_pred[:, idx])
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(Y_test[:, idx], Y_pred[:, idx])
        r2 = r2_score(Y_test[:, idx], Y_pred[:, idx])
        metrics_per_output[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }
        print(f"{name}: R²={r2:.4f} | MSE={mse:.6f} | RMSE={rmse:.6f} | MAE={mae:.6f}")

    mse_overall = mean_squared_error(Y_test, Y_pred, multioutput='uniform_average')
    rmse_overall = math.sqrt(mse_overall)
    mae_overall = mean_absolute_error(Y_test, Y_pred, multioutput='uniform_average')
    r2_overall = r2_score(Y_test, Y_pred, multioutput='uniform_average')

    metrics_overall = {
        'mse': mse_overall,
        'rmse': rmse_overall,
        'mae': mae_overall,
        'r2': r2_overall,
        'best_val_loss_mse': best_val_loss,
    }

    print("---------------------------------------------------")
    print(f"Overall: R²={r2_overall:.4f} | MSE={mse_overall:.6f} | RMSE={rmse_overall:.6f} | MAE={mae_overall:.6f}")
    print("---------------------------------------------------\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / f"p{p_i}_6dof_model_gpu.pt"
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'sensor_cols': sensor_cols,
            'output_cols': output_cols,
            'baseline': baseline.to_numpy(dtype=np.float32),
            'train_config': TRAIN_CONFIG,
            'metrics': {
                'per_output': metrics_per_output,
                'overall': metrics_overall,
            },
        },
        save_path
    )
    print(f"성공: 모델을 '{save_path}'에 저장했습니다.\n")

    metrics_path = MODEL_DIR / f"p{p_i}_6dof_model_gpu_metrics.json"
    with metrics_path.open('w', encoding='utf-8') as fp:
        json.dump(
            {
                'point_id': p_i,
                'per_output': metrics_per_output,
                'overall': metrics_overall,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
            },
            fp,
            ensure_ascii=False,
            indent=2
        )
    print(f"정보: 평가 지표를 '{metrics_path}'에 저장했습니다.\n")


def _collect_point_ids():
    search_pattern = str(LOG_DIR / "p*_*")
    all_paths = glob.glob(search_pattern)
    point_ids = set()
    for path in all_paths:
        if not os.path.isdir(path):
            continue
        folder_name = os.path.basename(os.path.normpath(path))
        match = re.match(r"p(\d+)_", folder_name)
        if match:
            point_ids.add(int(match.group(1)))
    return sorted(point_ids)


if __name__ == '__main__':
    point_ids = _collect_point_ids()
    if not point_ids:
        print(f"오류: '{LOG_DIR}'에서 p*_*... 형태의 로그 폴더를 찾을 수 없습니다.")
    else:
        print(f"정보: 다음 포인트들에 대해 학습을 시작합니다: {point_ids}")
        for point_id in point_ids:
            train_6dof_model_gpu(point_id)
