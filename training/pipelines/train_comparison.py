
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
import numpy as np
from sklearn.metrics import r2_score

from training.data.dataset_unified import UnifiedTactileDataset
from training.models.unified_model import UnifiedSensorModel
from training.models.mlp_baseline import MLPBaseline
from training.models.cnn_sr import CNNSR
from training.models.cnnlstm_sr import CNNLSTMSR
from training.models.sats_model import SATSModel
from training.models.tactile_transformer import TactileTransformer

def get_model(name, seq_len=50):
    if name == "unified": return UnifiedSensorModel(seq_len=seq_len)
    elif name == "mlp": return MLPBaseline()
    elif name == "cnn": return CNNSR()
    elif name == "cnnlstm": return CNNLSTMSR()
    elif name == "sats": return SATSModel()
    elif name == "transformer": return TactileTransformer()
    else: raise ValueError(f"Unknown model: {name}")

def calculate_metrics(preds, targets):
    mse = np.mean((preds - targets)**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets), axis=0)
    r2 = r2_score(targets, preds, multioutput='raw_values')
    return {"mse": mse.tolist(), "rmse": rmse.tolist(), "mae": mae.tolist(), "r2": r2.tolist()}

def _resolve_device(force: str) -> torch.device:
    if force == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA device is not available")
        return torch.device("cuda")
    if force == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_model(model_name, args):
    print(f"\n--- Training Model: {model_name} ---")
    device = _resolve_device(args.device)
    print(f"  [INFO] device: {device}")
    
    # Dataset with augmentation for training
    train_ds = UnifiedTactileDataset(args.data_dir, seq_len=args.seq_len, augment=True)
    val_ds = UnifiedTactileDataset(args.data_dir, seq_len=args.seq_len, augment=False)
    
    if len(train_ds) == 0:
        print(f"  [ERROR] Dataset at {args.data_dir} is empty. Check your data path!")
        return None

    print(f"  [INFO] Loaded {len(train_ds)} samples for {model_name}")

    indices = list(range(len(train_ds)))
    split = int(0.8 * len(train_ds))
    train_loader = DataLoader(torch.utils.data.Subset(train_ds, indices[:split]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, indices[split:]), batch_size=args.batch_size, shuffle=False)

    model = get_model(model_name, args.seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_mae = float('inf')
    best_metrics = {}

    for epoch in range(args.epochs):
        model.train()
        for grid, iso, tgt in train_loader:
            grid, iso, tgt = grid.to(device), iso.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            # 모델에 따라 입력 형태 다르게 처리
            if model_name in ["unified", "cnnlstm", "sats"]:
                res1, res2 = model(grid, iso) if model_name == "unified" else (model(grid, iso), None)
            elif model_name in ["mlp", "isoline_gnn"]:
                res1 = model(grid[:, -1].view(grid.size(0), -1), iso[:, -1, -1:]) # s16 + radius
            else: # cnn, transformer 등
                res1 = model(grid[:, -1], iso[:, -1, -1:])

            # Loss calculation (Simplified for comparison)
            pred = res1["xyz"] if isinstance(res1, dict) else res1
            loss = criterion(pred[:, :3], tgt[:, :3]) # Position loss
            loss.backward()
            optimizer.step()

        # Validation & Metrics
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for grid, iso, tgt in val_loader:
                grid, iso, tgt = grid.to(device), iso.to(device), tgt.to(device)
                if model_name == "unified": res1, _ = model(grid, iso)
                elif model_name in ["cnnlstm", "sats"]: res1 = model(grid, iso)
                elif model_name in ["mlp"]: res1 = model(grid[:, -1].view(grid.size(0), -1), iso[:, -1, -1:])
                else: res1 = model(grid[:, -1], iso[:, -1, -1:])
                
                pred = res1["xyz"] if isinstance(res1, dict) else res1
                all_preds.append(pred[:, :3].cpu().numpy())
                all_targets.append(tgt[:, :3].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = calculate_metrics(all_preds, all_targets)
        avg_mae = np.mean(metrics["mae"])

        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            best_metrics = metrics
            save_path = os.path.join(args.out_dir, f"best_{model_name}.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "metrics": metrics,
                "model_name": model_name
            }, save_path)

    print(f"Best MAE for {model_name}: {best_val_mae:.4f} mm")
    return best_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="preprocessing/raw_data")
    parser.add_argument("--out-dir", type=str, default="training/runs_comparison")
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn", "unified", "cnnlstm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="'auto'=cuda if available else cpu; 'cuda'=require GPU; 'cpu'=force cpu",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for m in args.models:
        results[m] = train_one_model(m, args)
    
    with open(os.path.join(args.out_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)
