#!/usr/bin/env python3
"""
sats/tools/analyze_taxel_rmse.py

논문 Fig.4.b/c 스타일 per-taxel RMSE 분석 도구.

기능
----
val_trials 전체 순회 → 각 샘플의 peak timestep에서 예측 & GT 추출
→ 40×40 per-taxel RMSE heatmap (Fig.4.b)
→ RMSE 분포 히스토그램 + 통계 (Fig.4.c)

지원 stage: lstm, attn, local_map, cnn, e2e (CNN 체크포인트는 cnn/e2e 모두 동일 구조)

실행 예시
----------
# CNN staged 모델 단일 분석
python3 -m sats.tools.analyze_taxel_rmse \\
    --ckpt sats/training/runs/cnn_paper/best_model.pt \\
    --stage cnn

# E2E 모델 단일 분석
python3 -m sats.tools.analyze_taxel_rmse \\
    --ckpt sats/training/runs/e2e_paper/best_model.pt \\
    --stage e2e

# 여러 모델 side-by-side 비교
python3 -m sats.tools.analyze_taxel_rmse \\
    --ckpts sats/training/runs/cnn_paper/best_model.pt \\
            sats/training/runs/e2e_paper/best_model.pt \\
    --stages cnn e2e \\
    --labels "Staged-CNN" "E2E"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sats.training.config import SATSConfig
from sats.training.dataset import build_dataloaders
from sats.training.train_lstm import get_target


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로더 / config 복원
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
    elif stage in ("cnn", "e2e"):
        from sats.training.cnn_module import SATSCNNStage
        model = SATSCNNStage(cfg)
    else:
        raise ValueError(f"알 수 없는 stage: {stage!r}")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)


def load_cfg_from_ckpt(ckpt_path: Path, overrides: dict) -> SATSConfig:
    """체크포인트 디렉터리의 config.json 복원 후 overrides 적용."""
    cfg_path = ckpt_path.parent / "config.json"
    d: dict = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = json.load(f)
        valid = {f.name for f in dataclasses.fields(SATSConfig)}
        d = {k: v for k, v in raw.items() if k in valid}
    else:
        print(f"[경고] config.json 없음: {cfg_path}  — 기본값 사용")
    d.update(overrides)
    return SATSConfig(**d)


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 RMSE 계산
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_taxel_rmse(
    model,
    val_loader,
    device: str,
    grid_size: int = 40,
) -> Tuple[np.ndarray, int, List[float]]:
    """
    val 데이터 전체에서 taxel별 squared error를 누적 → per-taxel RMSE 반환.

    Returns
    -------
    per_taxel_rmse  : [H, W] float64 ndarray
    n_samples       : 처리된 샘플 수
    per_sample_rmse : List[float]  샘플별 전체 RMSE
    """
    sq_err_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    n_samples = 0
    per_sample_rmse: List[float] = []

    model.eval()
    for sensor_b, gt_b, lengths in val_loader:
        sensor_b = sensor_b.to(device)
        gt_b     = gt_b.to(device)
        lengths  = lengths.to(device)

        target = get_target(gt_b, lengths)                          # [B, H, W]
        out    = model(sensor_b, lengths)
        pred   = out[0] if isinstance(out, tuple) else out          # [B, H, W]

        err = (pred - target).cpu().numpy()                         # [B, H, W]
        sq_err_sum += (err ** 2).sum(axis=0)
        for b in range(err.shape[0]):
            per_sample_rmse.append(float(np.sqrt(np.mean(err[b] ** 2))))
        n_samples += err.shape[0]

    per_taxel_rmse = np.sqrt(sq_err_sum / max(n_samples, 1))
    return per_taxel_rmse, n_samples, per_sample_rmse


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def _extent() -> list:
    return [-9.75, 9.75, -9.75, 9.75]


def plot_heatmap_comparison(
    results: List[Tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    """모델별 per-taxel RMSE heatmap을 가로 배열 (논문 Fig.4.b 스타일)."""
    n    = len(results)
    vmax = max(r.max() for _, r in results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, rmse_map) in zip(axes, results):
        im = ax.imshow(
            rmse_map, origin="lower", extent=_extent(),
            cmap="hot", vmin=0, vmax=vmax,
        )
        ax.set_title(
            f"{label}\nmean={rmse_map.mean():.4f}  max={rmse_map.max():.4f}",
            fontsize=9,
        )
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="RMSE (N/mm²)")

    fig.suptitle("Per-taxel RMSE Heatmap", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap 저장: {out_path}")


def plot_distribution_comparison(
    results: List[Tuple[str, np.ndarray, List[float]]],
    out_path: Path,
) -> None:
    """per-taxel / per-sample RMSE 분포를 겹쳐 비교 (논문 Fig.4.c 스타일)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for i, (label, rmse_map, sample_rmse) in enumerate(results):
        c    = colors[i % len(colors)]
        flat = rmse_map.flatten()

        axes[0].hist(flat, bins=50, alpha=0.6, color=c, label=label, density=True)
        axes[0].axvline(flat.mean(), color=c, linestyle="--", linewidth=1.5,
                        label=f"{label} mean={flat.mean():.4f}")

        axes[1].hist(sample_rmse, bins=50, alpha=0.6, color=c, label=label, density=True)
        axes[1].axvline(float(np.mean(sample_rmse)), color=c, linestyle="--", linewidth=1.5,
                        label=f"{label} mean={np.mean(sample_rmse):.4f}")

    axes[0].set_title("Per-taxel RMSE Distribution")
    axes[0].set_xlabel("RMSE (N/mm²)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Per-sample RMSE Distribution")
    axes[1].set_xlabel("RMSE (N/mm²)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  분포 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 통계 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(label: str, rmse_map: np.ndarray, sample_rmse: List[float]) -> None:
    flat = rmse_map.flatten()
    print(f"\n{'─'*54}")
    print(f"  모델: {label}")
    print(f"{'─'*54}")
    print(f"  처리 샘플 수:        {len(sample_rmse)}")
    print(f"  Per-taxel RMSE:")
    print(f"    평균              {flat.mean():.6f}")
    print(f"    중앙값            {float(np.median(flat)):.6f}")
    print(f"    최대              {float(flat.max()):.6f}")
    print(f"    90th percentile   {float(np.percentile(flat, 90)):.6f}")
    print(f"  Per-sample RMSE:")
    print(f"    평균              {float(np.mean(sample_rmse)):.6f}")
    print(f"    중앙값            {float(np.median(sample_rmse)):.6f}")
    print(f"    90th percentile   {float(np.percentile(sample_rmse, 90)):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 분석 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    ckpt_paths: List[Path],
    stages: List[str],
    labels: List[str],
    cfg_overrides: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_data: List[Tuple[str, np.ndarray]] = []
    dist_data: List[Tuple[str, np.ndarray, List[float]]] = []

    for ckpt_path, stage, label in zip(ckpt_paths, stages, labels):
        print(f"\n[{label}]  ckpt: {ckpt_path}")
        cfg    = load_cfg_from_ckpt(ckpt_path, cfg_overrides)
        device = cfg.effective_device()

        _, val_loader = build_dataloaders(cfg)
        model = load_model(stage, ckpt_path, cfg, device)

        print(f"  val 샘플 순회 중 (stage={stage}, device={device})...")
        rmse_map, n, sample_rmse = compute_taxel_rmse(
            model, val_loader, device, cfg.grid_size,
        )
        print_stats(label, rmse_map, sample_rmse)

        safe = label.replace(" ", "_").replace("/", "_")
        np.save(out_dir / f"{safe}_taxel_rmse.npy", rmse_map)

        heatmap_data.append((label, rmse_map))
        dist_data.append((label, rmse_map, sample_rmse))

    plot_heatmap_comparison(heatmap_data, out_dir / "taxel_rmse_heatmap.png")
    plot_distribution_comparison(dist_data, out_dir / "taxel_rmse_dist.png")
    print(f"\n출력 디렉터리: {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SATS per-taxel RMSE 분석 (논문 Fig.4.b/c)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 단일 모델
    p.add_argument("--ckpt",  default="",
                   help="단일 모델 체크포인트 경로")
    p.add_argument("--stage", default="cnn",
                   choices=["lstm", "attn", "local_map", "cnn", "e2e"],
                   help="모델 종류 (cnn / e2e 는 동일 구조 SATSCNNStage)")
    p.add_argument("--label", default="",
                   help="표시 이름 (기본: 체크포인트 폴더명)")

    # 다중 모델 비교
    p.add_argument("--ckpts",  nargs="+", default=[],
                   help="비교할 체크포인트 경로 목록")
    p.add_argument("--stages", nargs="+", default=[],
                   help="각 체크포인트의 stage (기본: 모두 cnn)")
    p.add_argument("--labels", nargs="+", default=[],
                   help="각 체크포인트의 표시 이름 (기본: 폴더명)")

    # 데이터 설정 (config.json 값을 오버라이드)
    p.add_argument("--raw-dir",    default="raw_data")
    p.add_argument("--gt-dir",     default="sats/preprocessing/gt_output_v1")
    p.add_argument("--val-trials", nargs="+",
                   default=["ecomesh_d5_z1_test3", "ecomesh_d5_z1.5_test9"])
    p.add_argument("--exclude-diameters", nargs="+", type=int, default=[])

    # 출력 / 추론 설정
    p.add_argument("--out-dir",    default="sats/tools/viz_output/taxel_rmse")
    p.add_argument("--batch-size", type=int, default=256,
                   help="추론 배치 크기 (메모리 조절용)")
    p.add_argument("--device",     default="cuda")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg_overrides = {
        "raw_dir":            args.raw_dir,
        "gt_dir":             args.gt_dir,
        "dataset_index_path": f"{args.gt_dir}/dataset_index.json",
        "val_trials":         args.val_trials,
        "exclude_diameters":  args.exclude_diameters,
        "batch_size":         args.batch_size,
        "device":             args.device,
    }

    if args.ckpts:
        ckpt_paths = [Path(c) for c in args.ckpts]
        stages = args.stages if args.stages else ["cnn"] * len(ckpt_paths)
        labels = args.labels if args.labels else [p.parent.name for p in ckpt_paths]
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt 또는 --ckpts 를 지정하세요.")
        ckpt_paths = [Path(args.ckpt)]
        stages     = [args.stage]
        labels     = [args.label or Path(args.ckpt).parent.name]

    if len(stages) != len(ckpt_paths) or len(labels) != len(ckpt_paths):
        raise SystemExit("--ckpts / --stages / --labels 의 개수가 일치해야 합니다.")

    run_analysis(ckpt_paths, stages, labels, cfg_overrides, Path(args.out_dir))


if __name__ == "__main__":
    main()
