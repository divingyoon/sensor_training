#!/usr/bin/env python3
"""
sats/training/train_e2e.py

SATS End-to-End 학습 스크립트.

학습 전략
----------
LSTM 인코더 → Self-Attention → Local Map Decoder → CNN Refiner 전체를
처음부터(또는 staged 체크포인트 초기화 후) 함께 학습한다.

기본 데이터셋은 논문 방식의 `window_size=10` sliding window다.
각 sample은 sensor window와 window 마지막 timestep의 pressure-map GT를 사용한다.

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

# d10 제외 + d5 전용 val trial split
python3 -m sats.training.train_e2e \\
    --run-name e2e_d5only_v1 \\
    --exclude-diameters 10 \\
    --val-ratio 0 \\
    --val-trials ecomesh_d5_z2.5_test2 \\
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
from tqdm.auto import tqdm

from .config import SATSConfig
from .dataset import build_dataloaders
from .cnn_module import SATSCNNStage
from .gt_gpu import BatchGPUTargetGenerator
from .train_lstm import get_target, set_seed, save_checkpoint, write_history

log = logging.getLogger(__name__)


def _loader_total(loader) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def _progress(loader, desc: str | None):
    if desc is None:
        return loader
    return tqdm(loader, total=_loader_total(loader), desc=desc, dynamic_ncols=True, leave=False)


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
    target_generator: BatchGPUTargetGenerator | None = None,
    progress_desc: str | None = None,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    progress_iter = _progress(loader, progress_desc)
    for sensor_b, gt_b, lengths in progress_iter:
        sensor_b = sensor_b.to(device, non_blocking=True)
        gt_b     = gt_b.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)

        if target_generator is None:
            target = get_target(gt_b, lengths).detach()      # [B, grid, grid]
        else:
            target = target_generator(gt_b).detach()         # [B, grid, grid]

        refined_map, _ = model(sensor_b, lengths)            # [B, grid, grid]
        loss = F.mse_loss(refined_map, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        if progress_desc is not None:
            progress_iter.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def val_epoch(
    model: SATSCNNStage,
    loader,
    device: str,
    target_generator: BatchGPUTargetGenerator | None = None,
    progress_desc: str | None = None,
) -> dict:
    model.eval()
    total_mse = 0.0
    n_batches = 0

    progress_iter = _progress(loader, progress_desc)
    for sensor_b, gt_b, lengths in progress_iter:
        sensor_b = sensor_b.to(device, non_blocking=True)
        gt_b     = gt_b.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)

        if target_generator is None:
            target = get_target(gt_b, lengths)
        else:
            target = target_generator(gt_b)
        refined_map, _ = model(sensor_b, lengths)

        total_mse += F.mse_loss(refined_map, target).item()
        n_batches += 1
        if progress_desc is not None:
            running_mse = total_mse / n_batches
            progress_iter.set_postfix(mse=f"{running_mse:.6f}", rmse=f"{math.sqrt(running_mse):.6f}")

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
    target_generator = (
        BatchGPUTargetGenerator(cfg, device)
        if cfg.gt_mode == "gpu_on_the_fly"
        else None
    )
    if target_generator is not None:
        log.info(
            "GPU on-the-fly GT 활성화: grid=%dx%d step=%.4fmm meta-only DataLoader",
            cfg.grid_size,
            cfg.grid_size,
            cfg.grid_step_mm,
        )

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

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg,
            target_generator=target_generator,
            progress_desc=f"train {epoch}/{cfg.epochs}",
        )
        val_metrics = val_epoch(
            model,
            val_loader,
            device,
            target_generator=target_generator,
            progress_desc=f"val   {epoch}/{cfg.epochs}",
        )

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
    p.add_argument("--raw-dir",             default="learning_data/sensor_raw_bin")
    p.add_argument("--gt-dir",              default="learning_data/gt")
    p.add_argument("--gt-mode",             choices=["precomputed", "on_the_fly", "gpu_on_the_fly"], default="precomputed",
                   help="GT 공급 방식. precomputed=기존 npy, on_the_fly=CPU 즉석 생성, gpu_on_the_fly=GPU batch 즉석 생성")
    p.add_argument("--gt-meta-cache-dir",   default="learning_data/gt_meta_cache",
                   help="gpu/on-the-fly용 compact sensor/meta cache 디렉터리")
    p.add_argument("--use-gt-meta-cache",   action=argparse.BooleanOptionalAction, default=True,
                   help="on-the-fly 계열에서 compact meta cache를 우선 사용")
    p.add_argument("--out-dir",             default="sats/training/runs")
    p.add_argument("--run-name",            default="e2e_v1")
    p.add_argument("--epochs",              type=int,   default=100)
    p.add_argument("--batch-size",          type=int,   default=2048)
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--weight-decay",        type=float, default=1e-5,
                   help="Adam weight decay (L2 정규화). overfit 완화용")
    p.add_argument("--hidden-dim",          type=int,   default=64)
    p.add_argument("--attn-dim",            type=int,   default=64)
    p.add_argument("--local-map-size",      type=int,   default=0,
                   help="각 센서 local map 한 변 크기. 0이면 grid-step에 맞춰 기존 0.5mm/15셀 물리 범위를 유지")
    p.add_argument("--cnn-hidden-channels", type=int,   default=16)
    p.add_argument("--num-layers",          type=int,   default=2)
    p.add_argument("--dropout",             type=float, default=0.1)
    p.add_argument("--seq-len",             type=int,   default=1000)
    p.add_argument("--grid-step-mm",        type=float, default=0.5,
                   help="GT/output grid 간격(mm). 0.5=41x41, 0.25=81x81, 0.2=101x101, 0.1=201x201")
    p.add_argument("--grid-size",           type=int, default=0,
                   help="GT/output grid 한 변 크기. 0이면 grid range와 step으로 자동 계산")
    p.add_argument("--num-workers",         type=int,   default=2)
    p.add_argument("--prefetch-factor",     type=int,   default=4,
                   help="DataLoader worker당 미리 준비할 batch 수(num_workers>0일 때)")
    p.add_argument("--persistent-workers",  action=argparse.BooleanOptionalAction, default=True,
                   help="DataLoader worker를 epoch 사이에 유지")
    p.add_argument("--device",              default="cuda")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--include-materials",   nargs="+", default=[],
                   help="학습/검증에 포함할 material key 목록. 예: --include-materials eco20_xy1")
    p.add_argument("--val-trials",          nargs="+", default=[])
    p.add_argument("--val-ratio",           type=float, default=0.2,
                   help=">0: 논문 방식 랜덤 sequence-level split. 0: --val-trials 기반 trial split.")
    p.add_argument("--exclude-diameters",   nargs="+", type=int, default=[],
                   help="학습/검증 풀에서 제외할 인덴터 직경(mm). 예: --exclude-diameters 10")
    p.add_argument("--window-size",         type=int,   default=10)
    p.add_argument("--on-the-fly-patch-step-mm", type=float, default=0.1,
                   help="on_the_fly GT 원형 접촉면 이산화 간격")
    p.add_argument("--contact-radius-step-mm", type=float, default=0.05,
                   help="구형 인덴터 접촉 반경 커널 캐시 양자화 간격")
    p.add_argument("--min-contact-radius-mm", type=float, default=0.05,
                   help="z_depth가 작을 때 사용할 최소 접촉 반경")
    p.add_argument("--z-depth-min-mm", type=float, default=0.001,
                   help="이 이하 z_depth는 비접촉 GT zero로 처리")
    p.add_argument("--z-balance-bin-width-mm", type=float, default=0.005,
                   help="on_the_fly balanced_contact에서 z_depth 균형 샘플링 bin 폭")
    p.add_argument("--plateau-stride", type=int, default=10,
                   help="on_the_fly balanced_contact에서 plateau/static 샘플 downsample stride")
    p.add_argument("--loading-stride", type=int, default=1,
                   help="on_the_fly balanced_contact에서 loading 샘플 stride")
    p.add_argument("--saturation-stride", type=int, default=2,
                   help="on_the_fly balanced_contact에서 high-force/saturation 샘플 stride")
    p.add_argument("--use-window-dataset",  action=argparse.BooleanOptionalAction, default=True,
                   help="윈도우 데이터셋 사용 (논문 방식). 끄려면 --no-use-window-dataset")
    p.add_argument("--no-lr-scheduler",     action="store_true")
    return p


def _config_from_args(args: argparse.Namespace) -> SATSConfig:
    grid_size = args.grid_size
    if grid_size <= 0:
        grid_size = int(round((20.0 / args.grid_step_mm))) + 1
    local_map_size = args.local_map_size
    if local_map_size <= 0:
        local_half_width_mm = 7 * 0.5
        local_half_cells = max(1, int(round(local_half_width_mm / args.grid_step_mm)))
        local_map_size = 2 * local_half_cells + 1

    cfg = SATSConfig(
        raw_dir             = args.raw_dir,
        gt_dir              = args.gt_dir,
        dataset_index_path  = f"{args.gt_dir}/dataset_index.json",
        out_dir             = args.out_dir,
        run_name            = args.run_name,
        epochs              = args.epochs,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        hidden_dim          = args.hidden_dim,
        attn_dim            = args.attn_dim,
        local_map_size      = local_map_size,
        cnn_hidden_channels = args.cnn_hidden_channels,
        num_layers          = args.num_layers,
        dropout             = args.dropout,
        seq_len             = args.seq_len,
        grid_step_mm        = args.grid_step_mm,
        grid_size           = grid_size,
        num_workers         = args.num_workers,
        dataloader_prefetch_factor = args.prefetch_factor,
        persistent_workers  = args.persistent_workers,
        device              = args.device,
        seed                = args.seed,
        include_materials   = args.include_materials,
        val_trials          = args.val_trials,
        exclude_diameters   = args.exclude_diameters,
        window_size         = args.window_size,
        use_window_dataset  = args.use_window_dataset,
        use_lr_scheduler    = not args.no_lr_scheduler,
        val_ratio           = args.val_ratio,
        gt_mode             = args.gt_mode,
        gt_meta_cache_dir   = args.gt_meta_cache_dir,
        use_gt_meta_cache   = args.use_gt_meta_cache,
        on_the_fly_patch_step_mm = args.on_the_fly_patch_step_mm,
        contact_radius_step_mm   = args.contact_radius_step_mm,
        min_contact_radius_mm    = args.min_contact_radius_mm,
        z_depth_min_mm           = args.z_depth_min_mm,
        z_balance_bin_width_mm   = args.z_balance_bin_width_mm,
        plateau_stride           = args.plateau_stride,
        loading_stride           = args.loading_stride,
        saturation_stride        = args.saturation_stride,
    )
    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()
    cfg = _config_from_args(args)
    train(cfg, init_ckpt=args.init_ckpt)


if __name__ == "__main__":
    main()
