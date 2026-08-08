"""각도-프리 변형 복원기 학습 (Phase 2·3) — due 데이터만으로 self-supervised.

3중 손실(설계 확정, Notion 토글 기록):
  L_suppress = ||restore(변형 무접촉)||²              → 유령 억제(무료 라벨)
  L_identity = ||restore(baseline) − baseline||²      → 변형 없으면 그대로(과잉보정 방지)
  L_contact  = ||restore(변형+접촉) − 접촉||²         → ★접촉 보존(붕괴 방지 핵심)

L_contact 의 접촉은 **기존 flat xy 취득 데이터**의 pct 를 재사용(§5.4 가법 가정).
실측 검증(2026-08-08, v6_new): L_contact 없으면 억제 36%·loc 4.26mm,
있으면 **87%·1.00mm** — 이 손실이 붕괴 방지의 결정적 요인.

leave-one-session-out 으로 미학습 변형에서의 억제율·접촉보존을 평가한다.

예:
  .venv/bin/python -m sats.bending.train_deform_restorer \\
    --deform-root skin_ws/raw_data/deform_v1 \\
    --contact-trial learning_data/sensor_raw_bin/ecomesh_v6_xy1/d5/z_2.5mm/test1 \\
    --sats-run sats/training/runs/ecomesh_v6_deploy_g025 --latent-dims 2 4 8
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .baseline_restorer import BaselineRestorer
from .config import BendingConfig
from .deform_data import deform_windows, load_all

_REPO = Path(__file__).resolve().parents[2]


def _contact_pool(trial_dir: Path, window: int, n: int, fz_min: float) -> np.ndarray:
    """flat 접촉 trial → pct 윈도우[K,W,16] (L_contact 용). eval_contact_preservation 과 동일 규약."""
    from .eval_contact_preservation import contact_windows
    win, _ = contact_windows(trial_dir, window, n, fz_min)
    return win


def train_deform_restorer(
    train_win: np.ndarray, base_win: np.ndarray, contact_pool: np.ndarray | None,
    cfg: BendingConfig, *, device: str = "cpu", epochs: int = 120, lr: float = 1e-3,
    lam_contact: float = 1.0, lam_identity: float = 0.5, batch: int = 512,
) -> BaselineRestorer:
    """3중 손실로 latent restorer 학습. train_win=변형 구간, base_win=앞뒤 baseline 구간."""
    X = torch.from_numpy(train_win).to(device)
    B = torch.from_numpy(base_win).to(device) if len(base_win) else None
    C = torch.from_numpy(contact_pool).to(device) if contact_pool is not None else None
    model = BaselineRestorer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]]
            loss = nn.functional.mse_loss(model(xb), torch.zeros_like(xb))       # L_suppress
            if C is not None:                                                    # L_contact
                cb = C[torch.randint(0, len(C), (len(xb),), device=device)]
                loss = loss + lam_contact * nn.functional.mse_loss(model(xb + cb), cb)
            if B is not None and lam_identity > 0:                                # L_identity
                bb = B[torch.randint(0, len(B), (min(len(xb), len(B)),), device=device)]
                loss = loss + lam_identity * nn.functional.mse_loss(model(bb), bb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def evaluate(model: BaselineRestorer, val_win: np.ndarray, contact_pool: np.ndarray,
             sats, device: str) -> dict:
    """홀드아웃 변형에서 억제율 + 준-합성 접촉 보존."""
    from .eval_contact_preservation import _peak_xy, _sats_map
    V = torch.from_numpy(val_win).to(device)
    n = min(len(V), len(contact_pool))
    idx = np.linspace(0, len(V) - 1, n).astype(int)
    vb = V[idx]
    C = torch.from_numpy(contact_pool[:n]).to(device)
    flat = _sats_map(sats, torch.zeros(1, vb.shape[1], vb.shape[2], device=device))[0]
    # 1) 유령 억제: 무접촉 변형 → SATS 맵이 flat 에 얼마나 가까운가
    m_unc = _sats_map(sats, vb)
    m_cor = _sats_map(sats, model(vb))
    du = (m_unc - flat).abs().mean(dim=(1, 2)).mean().item()
    dc = (m_cor - flat).abs().mean(dim=(1, 2)).mean().item()
    # 2) 접촉 보존: 변형+접촉 복원 후 접촉만 맵과 비교
    ref = _sats_map(sats, C)
    unc = _sats_map(sats, vb + C)
    cor = _sats_map(sats, model(vb + C))
    eu = (unc - ref).abs().mean(dim=(1, 2)).mean().item()
    ec = (cor - ref).abs().mean(dim=(1, 2)).mean().item()
    xy_ref, xy_unc, xy_cor = _peak_xy(ref), _peak_xy(unc), _peak_xy(cor)
    return {
        "suppression": 1.0 - dc / du if du > 0 else 0.0,
        "ghost_dev_uncorrected": du, "ghost_dev_corrected": dc,
        "contact_recovery": 1.0 - ec / eu if eu > 0 else 0.0,
        "loc_uncorrected_mm": float(np.linalg.norm(xy_unc - xy_ref, axis=1).mean()),
        "loc_corrected_mm": float(np.linalg.norm(xy_cor - xy_ref, axis=1).mean()),
        "n": int(n),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="각도-프리 변형 복원기 학습(due 데이터만)")
    p.add_argument("--deform-root", required=True, help="변형 세션 루트(due bin 폴더들)")
    p.add_argument("--contact-trial", type=Path,
                   default=_REPO / "learning_data/sensor_raw_bin/ecomesh_v6_xy1/d5/z_2.5mm/test1",
                   help="L_contact·평가용 flat 접촉 trial")
    p.add_argument("--sats-run", type=Path,
                   default=_REPO / "sats/training/runs/ecomesh_v6_deploy_g025")
    p.add_argument("--latent-dims", type=int, nargs="+", default=[2, 4, 8],
                   help="잠재 차원 스윕(억제율 vs 접촉보존 트레이드오프)")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--n-contact", type=int, default=300)
    p.add_argument("--fz-min", type=float, default=0.5)
    p.add_argument("--no-contact-loss", action="store_true", help="L_contact 제거(ablation)")
    p.add_argument("--out", type=Path, default=_REPO / "sats/bending/runs/deform_restorer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg0 = BendingConfig()
    W = cfg0.window_size
    print(f"[1/3] 변형 세션 로드: {args.deform_root}")
    sessions = load_all(args.deform_root)
    if len(sessions) < 2:
        raise SystemExit("leave-one-session-out 평가에 최소 2세션 필요")

    print(f"[2/3] 접촉 pool 로드: {args.contact_trial.name}")
    pool = _contact_pool(args.contact_trial, W, args.n_contact, args.fz_min)
    print(f"  접촉 윈도우 {len(pool)}개")

    from .pipeline import load_frozen_sats
    sats = load_frozen_sats(args.sats_run, args.device)

    print(f"[3/3] leave-one-session-out × latent_dim {args.latent_dims}")
    results: dict[str, list] = {}
    for k in args.latent_dims:
        cfg = replace(cfg0, restorer_mode="latent", latent_dim=k)
        per_session = []
        for i, held in enumerate(sessions):
            tr = [s for j, s in enumerate(sessions) if j != i]
            train_win = np.concatenate([deform_windows(s, W) for s in tr])
            base_win = np.concatenate([deform_windows(s, W, include_baseline=True) for s in tr])
            val_win = deform_windows(held, W)
            if len(val_win) == 0:
                continue
            model = train_deform_restorer(
                train_win, base_win, None if args.no_contact_loss else pool,
                cfg, device=args.device, epochs=args.epochs)
            r = evaluate(model, val_win, pool, sats, args.device)
            r["held_out"] = held.name
            per_session.append(r)
            print(f"  z={k} held={held.name:12s} 억제 {r['suppression']*100:5.1f}%  "
                  f"접촉회복 {r['contact_recovery']*100:5.1f}%  "
                  f"loc {r['loc_uncorrected_mm']:.2f}→{r['loc_corrected_mm']:.2f}mm")
        if per_session:
            results[f"latent:{k}"] = per_session
            sup = np.mean([r["suppression"] for r in per_session])
            rec = np.mean([r["contact_recovery"] for r in per_session])
            loc = np.mean([r["loc_corrected_mm"] for r in per_session])
            print(f"  ★ z={k} 평균: 억제 {sup*100:.1f}%  접촉회복 {rec*100:.1f}%  loc {loc:.2f}mm")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "loso_results.json").write_text(
        json.dumps({"sessions": [s.name for s in sessions],
                    "contact_loss": not args.no_contact_loss,
                    "results": results}, indent=2), encoding="utf-8")
    # 최종 배포 모델: 전 세션 학습(최고 z)
    if results:
        best_k = max(results, key=lambda kk: np.mean([r["suppression"] for r in results[kk]]))
        k = int(best_k.split(":")[1])
        cfg = replace(cfg0, restorer_mode="latent", latent_dim=k)
        all_win = np.concatenate([deform_windows(s, W) for s in sessions])
        all_base = np.concatenate([deform_windows(s, W, include_baseline=True) for s in sessions])
        model = train_deform_restorer(all_win, all_base,
                                      None if args.no_contact_loss else pool,
                                      cfg, device=args.device, epochs=args.epochs)
        torch.save({"model": model.state_dict(), "latent_dim": k,
                    "restorer_mode": "latent"}, args.out / "best.pt")
        print(f"\n배포 모델 저장: {args.out/'best.pt'} (latent_dim={k}, 전 세션 학습)")
    print(f"결과: {args.out/'loso_results.json'}")


if __name__ == "__main__":
    main()
