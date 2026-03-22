"""
train.py

Phase 1 / Phase 2 공용 학습 루프.

Usage:
  # Phase 1 (MLP + CNN decoder)
  python -m training.train --phase 1 --epochs 100

  # Phase 2 (1D CNN + FiLM)
  python -m training.train --phase 2 --epochs 100 --lr 5e-4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 상위 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import TrainConfig
from training.dataset import build_loaders
from training.loss import SkinLoss
from training.evaluate import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train skin tactile SR model.")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lambda-map", type=float, default=None)
    parser.add_argument("--lambda-sensor", type=float, default=None)
    parser.add_argument("--lambda-fz", type=float, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_model(cfg: TrainConfig, phase: int) -> torch.nn.Module:
    if phase == 1:
        from training.model_baseline import BaselineModel
        return BaselineModel(
            n_tactile=cfg.n_tactile,
            n_aux=cfg.n_aux,
            latent_dim=cfg.latent_dim,
        )
    else:
        from training.model_main import MainModel
        return MainModel(
            n_tactile=cfg.n_tactile,
            n_aux=cfg.n_aux,
            latent_dim=256,
        )


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip):
    model.train()
    total_loss = 0.0
    for batch in loader:
        tactile = batch["tactile"].to(device)
        aux = batch["aux"].to(device)
        hr_map = batch["hr_map"].to(device)
        tactile_raw = batch["tactile_raw"].to(device)
        fz = batch["fz"].to(device)
        x_bounds = batch["x_bounds"].to(device)
        y_bounds = batch["y_bounds"].to(device)

        optimizer.zero_grad()
        pred = model(tactile, aux)
        loss, _ = loss_fn(pred, hr_map, tactile_raw, fz, x_bounds, y_bounds)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg):
    model.eval()
    total_loss = 0.0
    all_metrics = []
    for batch in loader:
        tactile = batch["tactile"].to(device)
        aux = batch["aux"].to(device)
        hr_map = batch["hr_map"].to(device)
        tactile_raw = batch["tactile_raw"].to(device)
        fz = batch["fz"].to(device)
        cx = batch["cx"].to(device)
        cy = batch["cy"].to(device)
        x_bounds = batch["x_bounds"].to(device)
        y_bounds = batch["y_bounds"].to(device)

        pred = model(tactile, aux)
        loss, _ = loss_fn(pred, hr_map, tactile_raw, fz, x_bounds, y_bounds)
        total_loss += loss.item()

        metrics = compute_metrics(
            pred, hr_map, tactile_raw, fz, cx, cy, x_bounds, y_bounds,
            sensor_spacing_mm=cfg.sensor_spacing_mm,
            sensor_origin_x_mm=cfg.sensor_origin_x_mm,
            sensor_origin_y_mm=cfg.sensor_origin_y_mm,
        )
        all_metrics.append(metrics)

    avg_loss = total_loss / max(len(loader), 1)
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    return avg_loss, avg_metrics


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    # CLI 오버라이드
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lambda_map is not None:
        cfg.lambda_map = args.lambda_map
    if args.lambda_sensor is not None:
        cfg.lambda_sensor = args.lambda_sensor
    if args.lambda_fz is not None:
        cfg.lambda_fz = args.lambda_fz
    if args.lambda_smooth is not None:
        cfg.lambda_smooth = args.lambda_smooth
    cfg.phase = args.phase

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Phase {cfg.phase} | device: {device} | epochs: {cfg.epochs}")

    # output dir
    run_dir = cfg.out_dir / f"phase{cfg.phase}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # data
    train_loader, val_loader, _ = build_loaders(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        phase=cfg.data_phase,
        min_depth_mm=cfg.min_depth_mm,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
    )
    print(f"[INFO] train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)}")

    # model
    model = build_model(cfg, cfg.phase).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model params: {total_params:,}")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] resumed from {args.resume}")

    # optimizer / scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    # loss
    loss_fn = SkinLoss(
        lambda_map=cfg.lambda_map,
        lambda_sensor=cfg.lambda_sensor,
        lambda_fz=cfg.lambda_fz,
        lambda_smooth=cfg.lambda_smooth,
        sensor_spacing_mm=cfg.sensor_spacing_mm,
        sensor_origin_x_mm=cfg.sensor_origin_x_mm,
        sensor_origin_y_mm=cfg.sensor_origin_y_mm,
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg.grad_clip)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device, cfg)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)

        if epoch % cfg.log_interval == 0 or epoch == 1:
            print(
                f"[{epoch:4d}/{cfg.epochs}] "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"centroid={val_metrics['centroid_error_mm']:.2f}mm  "
                f"IoU={val_metrics['iou']:.3f}  "
                f"Fz_mae={val_metrics['fz_mae']:.3f}"
            )

        if cfg.save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss, "cfg": cfg.__dict__},
                run_dir / "best.pt",
            )

    # 마지막 체크포인트
    torch.save(
        {"epoch": cfg.epochs, "model": model.state_dict(), "val_loss": val_loss, "cfg": cfg.__dict__},
        run_dir / "last.pt",
    )

    # history 저장
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[DONE] best val_loss={best_val_loss:.4f} → {run_dir}/best.pt")


if __name__ == "__main__":
    main()
