#!/usr/bin/env python3
"""
sats/training/train_e2e.py

SATS End-to-End 학습 스크립트.

학습 전략
----------
LSTM 인코더 → Self-Attention → Local Map Decoder → CNN Refiner 전체를
처음부터(또는 staged 체크포인트 초기화 후) 함께 학습한다.

staged 방식(4단계 순차 frozen 학습)과 ablation 비교용으로 설계됐다.

실행 예시
----------
# 완전 처음부터 E2E 학습
python3 -m sats.training.train_e2e \\
    --run-name e2e_v1 \\
    --epochs 100

# staged 체크포인트로 초기화 후 fine-tuning
python3 -m sats.training.train_e2e \\
    --init-ckpt sats/training/runs/04.24-sats-test2/cnn_v2/best_model.pt \\
    --run-name e2e_finetune_v1 \\
    --epochs 50 --lr 1e-4

# d10 제외 + d5 전용 val trial
python3 -m sats.training.train_e2e \\
    --run-name e2e_d5only_v1 \\
    --exclude-diameters 10 \\
    --val-trials ecomesh_d5_z1.5_test9 \\
    --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import SATSConfig
from .dataset import build_dataloaders
from .cnn_module import SATSCNNStage
from .train_lstm import find_peak_gt, get_target, set_seed, save_checkpoint, write_history

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# staged 체크포인트에서 전체 가중치 초기화 (선택적)
# ─────────────────────────────────────────────────────────────────────────────

def init_from_staged_ckpt(ckpt_path: Union[str, Path], model: SATSCNNStage) -> None:
    """
    SATSCNNStage 형식의 체크포인트로 전체 모델을 초기화한다.

    staged train_cnn.py의 best_model.pt와 동일한 키 구조를 사용하므로
    그대로 load_state_dict 가능.

    모든 파라미터는 requires_grad=True 상태를 유지한다 (E2E 학습).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    log.info("staged 체크포인트로 초기화 완료: %s", ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# 학습 / 검증 루프
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: SATSCNNStage,
    loader,
    optimizer,
    device: str,
    cfg: SATSConfig,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for sensor_b, gt_b, lengths in loader:
        sensor_b = sensor_b.to(device, non_blocking=True)
        gt_b     = gt_b.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)

        target = get_target(gt_b, lengths).detach()          # [B, 40, 40]

        refined_map, _ = model(sensor_b, lengths)            # [B, 40, 40]
        loss = F.mse_loss(refined_map, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def val_epoch(
    model: SATSCNNStage,
    loader,
    device: str,
) -> dict:
    model.eval()
    total_mse = 0.0
    n_batches = 0

    for sensor_b, gt_b, lengths in loader:
        sensor_b = sensor_b.to(device, non_blocking=True)
        gt_b     = gt_b.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)

        target = get_target(gt_b, lengths)
        refined_map, _ = model(sensor_b, lengths)

        total_mse += F.mse_loss(refined_map, target).item()
        n_batches += 1

    mse  = total_mse / max(n_batches, 1)
    rmse = math.sqrt(mse)
    return {"mse": mse, "rmse": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: SATSConfig, init_ckpt: str = "") -> None:
    set_seed(cfg.seed)
    device = cfg.effective_device()
    log.info("device: %s", device)

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))

    log.info("데이터 로더 구성 중...")
    train_loader, val_loader = build_dataloaders(cfg)

    model = SATSCNNStage(cfg).to(device)

    if init_ckpt:
        init_from_staged_ckpt(init_ckpt, model)
    else:
        log.info("무작위 초기화로 E2E 학습 시작")

    n_total = sum(p.numel() for p in model.parameters())
    log.info("전체 파라미터(모두 학습 가능): %d", n_total)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (
        ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
        )
        if cfg.use_lr_scheduler else None
    )
    if not cfg.use_lr_scheduler:
        log.info("고정 LR 모드 (lr=%.6f)", cfg.lr)

    history   = []
    best_rmse = float("inf")
    best_ckpt = run_dir / "best_model.pt"
    last_ckpt = run_dir / "last_model.pt"
    hist_path = run_dir / "history.json"

    log.info("학습 시작 (epochs=%d)", cfg.epochs)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device, cfg)
        val_metrics   = val_epoch(model, val_loader, device)

        lr_now = optimizer.param_groups[0]["lr"]
        if cfg.use_lr_scheduler and scheduler is not None:
            scheduler.step(val_metrics["mse"])

        elapsed = time.time() - t0
        row = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "val_mse":    val_metrics["mse"],
            "val_rmse":   val_metrics["rmse"],
            "lr":         lr_now,
            "elapsed_s":  elapsed,
        }
        history.append(row)
        write_history(hist_path, history)

        log.info(
            "Epoch %3d/%d  train_loss=%.6f  val_rmse=%.6f  lr=%.2e  (%.1fs)",
            epoch, cfg.epochs,
            row["train_loss"], row["val_rmse"], lr_now, elapsed,
        )

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, row)
            log.info("  ★ best val_rmse=%.6f → %s", best_rmse, best_ckpt)

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:04d}.pt",
                epoch, model, optimizer, scheduler, row,
            )

    save_checkpoint(last_ckpt, cfg.epochs, model, optimizer, scheduler, history[-1])
    log.info("완료. best val_rmse=%.6f", best_rmse)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SATS End-to-End 학습 (전체 파이프라인 동시 학습)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--init-ckpt",           default="",
                   help="staged 체크포인트로 초기화 (SATSCNNStage 형식). 비우면 무작위 초기화.")
    p.add_argument("--raw-dir",             default="raw_data")
    p.add_argument("--gt-dir",              default="sats/preprocessing/gt_output_v1")
    p.add_argument("--out-dir",             default="sats/training/runs")
    p.add_argument("--run-name",            default="e2e_v1")
    p.add_argument("--epochs",              type=int,   default=100)
    p.add_argument("--batch-size",          type=int,   default=64)
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--hidden-dim",          type=int,   default=64)
    p.add_argument("--attn-dim",            type=int,   default=64)
    p.add_argument("--local-map-size",      type=int,   default=15)
    p.add_argument("--cnn-hidden-channels", type=int,   default=16)
    p.add_argument("--num-layers",          type=int,   default=2)
    p.add_argument("--dropout",             type=float, default=0.1)
    p.add_argument("--seq-len",             type=int,   default=400)
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--device",              default="cuda")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--val-trials",          nargs="+",
                   default=["ecomesh_d5_z1_test3", "ecomesh_d5_z1.5_test9"])
    p.add_argument("--exclude-diameters",   nargs="+", type=int, default=[],
                   help="학습/검증 풀에서 제외할 인덴터 직경(mm). 예: --exclude-diameters 10")
    p.add_argument("--window-size",         type=int,   default=10)
    p.add_argument("--use-window-dataset",  action="store_true")
    p.add_argument("--no-lr-scheduler",     action="store_true")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()

    cfg = SATSConfig(
        raw_dir             = args.raw_dir,
        gt_dir              = args.gt_dir,
        dataset_index_path  = f"{args.gt_dir}/dataset_index.json",
        out_dir             = args.out_dir,
        run_name            = args.run_name,
        epochs              = args.epochs,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        hidden_dim          = args.hidden_dim,
        attn_dim            = args.attn_dim,
        local_map_size      = args.local_map_size,
        cnn_hidden_channels = args.cnn_hidden_channels,
        num_layers          = args.num_layers,
        dropout             = args.dropout,
        seq_len             = args.seq_len,
        num_workers         = args.num_workers,
        device              = args.device,
        seed                = args.seed,
        val_trials          = args.val_trials,
        exclude_diameters   = args.exclude_diameters,
        window_size         = args.window_size,
        use_window_dataset  = args.use_window_dataset,
        use_lr_scheduler    = not args.no_lr_scheduler,
    )
    train(cfg, init_ckpt=args.init_ckpt)


if __name__ == "__main__":
    main()
