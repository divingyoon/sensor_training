#!/usr/bin/env python3
"""
sats/tools/visualize.py

예측 맵 vs GT 맵 시각화 도구.

val 샘플에서 모델 추론 후 pred / GT / error heatmap을 저장한다.
각 단계(lstm, attn, local_map, cnn) 모델을 선택해서 비교 가능.

실행:
    cd /home/user/sensor_training

    # CNN 최종 모델로 val 샘플 5개 시각화
    python3 -m sats.tools.visualize --stage cnn --ckpt sats/training/runs/cnn_v1/best_model.pt

    # Local Map 모델 비교
    python3 -m sats.tools.visualize --stage local_map --ckpt sats/training/runs/local_map_v1/best_model.pt

    # 출력: sats/tools/viz_output/ 에 PNG 저장
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# matplotlib backend: 화면 없이 파일 저장
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 프로젝트 루트를 sys.path에 추가
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sats.training.config import SATSConfig
from sats.training.dataset import build_dataloaders, sats_collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로더
# ─────────────────────────────────────────────────────────────────────────────

def load_model(stage: str, ckpt_path: Path, cfg: SATSConfig, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if stage == "lstm":
        from sats.training.lstm_module import SATSLSTMStage
        model = SATSLSTMStage(cfg)
    elif stage == "attn":
        from sats.training.attention_module import SATSAttentionStage
        model = SATSAttentionStage(cfg)
    elif stage == "local_map":
        from sats.training.local_map_module import SATSLocalMapStage
        model = SATSLocalMapStage(cfg)
    elif stage == "cnn":
        from sats.training.cnn_module import SATSCNNStage
        model = SATSCNNStage(cfg)
    else:
        raise ValueError(f"알 수 없는 stage: {stage}")

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# find_peak_gt (train_lstm.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def find_peak_gt(gt_batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T_max = gt_batch.shape[:2]
    device = gt_batch.device
    gt_sum = gt_batch.sum(dim=(-2, -1))
    time_idx = torch.arange(T_max, device=device).unsqueeze(0)
    pad_mask = time_idx >= lengths.unsqueeze(1)
    gt_sum = gt_sum.masked_fill(pad_mask, -1.0)
    peak_t = gt_sum.argmax(dim=1)
    return gt_batch[torch.arange(B, device=device), peak_t]


# ─────────────────────────────────────────────────────────────────────────────
# 시각화 (한 샘플)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample(
    pred: np.ndarray,     # [40, 40]
    gt:   np.ndarray,     # [40, 40]
    title: str,
    save_path: Path,
) -> None:
    error = pred - gt
    vmax  = max(float(gt.max()), float(pred.max()), 1e-9)
    err_abs = float(np.abs(error).max())

    fig = plt.figure(figsize=(14, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    ax_gt   = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1])
    ax_err  = fig.add_subplot(gs[2])
    ax_info = fig.add_subplot(gs[3])

    extent = [-9.75, 9.75, -9.75, 9.75]

    im_gt = ax_gt.imshow(gt, origin="lower", extent=extent,
                         vmin=0, vmax=vmax, cmap="hot")
    ax_gt.set_title("GT (peak timestep)", fontsize=9)
    ax_gt.set_xlabel("x [mm]"); ax_gt.set_ylabel("y [mm]")
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046)

    im_pred = ax_pred.imshow(pred, origin="lower", extent=extent,
                             vmin=0, vmax=vmax, cmap="hot")
    ax_pred.set_title("Prediction", fontsize=9)
    ax_pred.set_xlabel("x [mm]")
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046)

    im_err = ax_err.imshow(error, origin="lower", extent=extent,
                           vmin=-err_abs, vmax=err_abs, cmap="bwr")
    ax_err.set_title("Error (pred − GT)", fontsize=9)
    ax_err.set_xlabel("x [mm]")
    plt.colorbar(im_err, ax=ax_err, fraction=0.046)

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae  = float(np.mean(np.abs(error)))
    rel  = rmse / (float(gt.max()) + 1e-12) * 100
    info_text = (
        f"GT peak:  {gt.max():.5f} N/mm²\n"
        f"Pred peak:{pred.max():.5f} N/mm²\n"
        f"RMSE:     {rmse:.6f}\n"
        f"MAE:      {mae:.6f}\n"
        f"RMSE/peak:{rel:.1f}%"
    )
    ax_info.axis("off")
    ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=9, verticalalignment="center",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}  (RMSE={rmse:.6f}, RMSE/peak={rel:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SATS 예측 시각화")
    parser.add_argument("--stage",    default="cnn",
                        choices=["lstm","attn","local_map","cnn"])
    parser.add_argument("--ckpt",     required=True, help="체크포인트 파일 경로")
    parser.add_argument("--n-samples",type=int, default=6)
    parser.add_argument("--out-dir",  default="sats/tools/viz_output")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--raw-dir",  default="raw_data")
    parser.add_argument("--gt-dir",   default="sats/preprocessing/gt_output_v1")
    parser.add_argument("--val-trials", nargs="+",
                        default=["ecomesh_d10_z1_test3","ecomesh_d5_z1.5_test9"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA 없음, CPU 사용")

    cfg = SATSConfig(
        raw_dir    = args.raw_dir,
        gt_dir     = args.gt_dir,
        val_trials = args.val_trials,
        device     = device,
    )

    print(f"모델 로드: {args.ckpt}")
    model = load_model(args.stage, Path(args.ckpt), cfg, device)

    print("Val 데이터 로더 구성 중...")
    _, val_loader = build_dataloaders(cfg)

    # 시각화 시 다양한 위치를 보기 위해 shuffle=True 적용
    val_loader = torch.utils.data.DataLoader(
        val_loader.dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=sats_collate_fn,
    )

    count = 0
    rmse_list = []

    with torch.no_grad():
        for sensor_b, gt_b, lengths in val_loader:
            sensor_b = sensor_b.to(device)
            gt_b     = gt_b.to(device)
            lengths  = lengths.to(device)

            target = find_peak_gt(gt_b, lengths)   # [B, 40, 40]

            # 모델 추론
            out = model(sensor_b, lengths)
            pred_map = out[0] if isinstance(out, tuple) else out

            for b in range(pred_map.shape[0]):
                if count >= args.n_samples:
                    break

                pred_np = pred_map[b].cpu().numpy()
                gt_np   = target[b].cpu().numpy()
                rmse = float(np.sqrt(np.mean((pred_np - gt_np) ** 2)))
                rmse_list.append(rmse)

                save_path = out_dir / f"{args.stage}_sample{count:03d}.png"
                plot_sample(
                    pred_np, gt_np,
                    title=f"Stage={args.stage}  sample={count}",
                    save_path=save_path,
                )
                count += 1

            if count >= args.n_samples:
                break

    if rmse_list:
        print(f"\n샘플 {len(rmse_list)}개 평균 RMSE: {np.mean(rmse_list):.6f}")
        print(f"출력 디렉터리: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
