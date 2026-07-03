#!/usr/bin/env python3
"""요약 지표 + SR scale factor(Note S1). 모델별 핵심 지표를 한 장에 통합.

입력: diag_summary.csv(상대오차) + loc_summary.csv(위치오차). 재학습·추가추론 없음.

산출: summary_metrics.png (좌: d5/d10 상대오차 + 위치오차 막대, 우: 지표 표).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / "history/fig_data/sats_supplementary/summary_metrics"
DIAG_XY1 = REPO / "history/fig_data/sats_experiments/fig3_diag/diag_summary.csv"
DIAG_XY0P5 = REPO / "history/fig_data/sats_experiments/final_xy0p5_diag/diag_summary.csv"
LOC_CSV = REPO / "history/fig_data/sats_supplementary/S20_localization/loc_summary.csv"

GRID_SIZE = 41            # virtual taxel 격자
N_PHYSICAL = 16           # physical taxel 수
SR_SCALE = GRID_SIZE * GRID_SIZE / N_PHYSICAL  # Note S1: N_v / N_r

# (label, diag run key, diag csv)
MODELS = [
    ("eco20_xy1", "xy1_d5d10_eco20_xy1_fold2_e2e_g05", DIAG_XY1),
    ("eco50_xy1", "xy1_d5d10_eco50_xy1_fold1_e2e_g05", DIAG_XY1),
    ("ecomesh_xy1", "xy1_d5d10_ecomesh_xy1_fold3_e2e_g05", DIAG_XY1),
    ("ecomesh_xy0p5_final", "ecomesh_xy0p5_datarich_val_d5test10_d10test3", DIAG_XY0P5),
]
COLOR = {"eco20_xy1": "#e07b39", "eco50_xy1": "#5b8def",
         "ecomesh_xy1": "#2ca25f", "ecomesh_xy0p5_final": "#8856a7"}


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with open(path) as f:
        return {r[list(r)[0]]: r for r in csv.DictReader(f)}


def collect() -> list[dict]:
    loc = _read_csv(LOC_CSV)
    diag_cache = {DIAG_XY1: _read_csv(DIAG_XY1), DIAG_XY0P5: _read_csv(DIAG_XY0P5)}
    rows = []
    for label, run, csvp in MODELS:
        d = diag_cache[csvp][run]
        rows.append({
            "label": label,
            "d5_rel": float(d["d5_rel_rmse"]),
            "d10_rel": float(d["d10_rel_rmse"]),
            "loc_mm": float(loc[label]["mean_loc_error_mm"]),
        })
    return rows


def main() -> None:
    rows = collect()
    labels = [r["label"] for r in rows]
    fig, (axb, axt) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # 좌: 상대오차(d5/d10) + 위치오차 막대
    x = np.arange(len(labels)); w = 0.26
    axb.bar(x - w, [r["d5_rel"] for r in rows], w, label="d5 rel-RMSE",
            color=[COLOR[r["label"]] for r in rows], alpha=0.55, edgecolor="k")
    axb.bar(x, [r["d10_rel"] for r in rows], w, label="d10 rel-RMSE",
            color=[COLOR[r["label"]] for r in rows], alpha=0.9, edgecolor="k", hatch="//")
    axb2 = axb.twinx()
    axb2.plot(x, [r["loc_mm"] for r in rows], "o-", c="0.15", label="loc-error [mm]")
    axb2.set_ylabel("localization error [mm]")
    axb2.set_ylim(0, max(r["loc_mm"] for r in rows) * 1.3)
    axb.set_xticks(x); axb.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axb.set_ylabel("relative RMSE")
    axb.set_title("Key metrics by model")
    axb.legend(loc="upper left", fontsize=8, frameon=False)
    axb2.legend(loc="upper right", fontsize=8, frameon=False)
    axb.grid(axis="y", ls=":", alpha=0.4)

    # 우: 지표 표 + SR scale factor
    axt.axis("off")
    table = [["model", "d5 rel", "d10 rel", "loc [mm]"]]
    for r in rows:
        table.append([r["label"], f"{r['d5_rel']:.3f}", f"{r['d10_rel']:.3f}", f"{r['loc_mm']:.2f}"])
    t = axt.table(cellText=table, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    for c in range(4):
        t[0, c].set_facecolor("#dddddd"); t[0, c].set_text_props(weight="bold")
    axt.set_title(f"SR scale factor (Note S1) = {GRID_SIZE}²/{N_PHYSICAL} ≈ {SR_SCALE:.0f}\n"
                  f"(virtual {GRID_SIZE*GRID_SIZE} / physical {N_PHYSICAL} taxels)", fontsize=10)

    fig.suptitle("SATS summary — relative error, localization, super-resolution scale", y=1.02)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "summary_metrics.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path, f"| SR scale factor ≈ {SR_SCALE:.0f}")


if __name__ == "__main__":
    main()
