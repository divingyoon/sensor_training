import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import torch

from training_seq.train_seq import CnnLstmPose

SKIN_COLS = [f"Skin{i}" for i in range(1, 17)]


class RunningEma:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.state: Optional[float] = None

    def update(self, x: float) -> float:
        if self.state is None:
            self.state = float(x)
        else:
            self.state = self.alpha * float(x) + (1.0 - self.alpha) * self.state
        return self.state


class ContactFzFilter:
    def __init__(self, deadband: float, on_th: float, off_th: float, ema_alpha: float):
        if off_th > on_th:
            raise ValueError("Require fz_off_th <= fz_on_th")
        self.deadband = float(deadband)
        self.on_th = float(on_th)
        self.off_th = float(off_th)
        self.ema = RunningEma(ema_alpha)
        self.in_contact = False

    def update(self, fz_raw: float) -> Dict[str, float]:
        fz_db = 0.0 if abs(fz_raw) < self.deadband else float(fz_raw)

        mag = abs(fz_db)
        if self.in_contact:
            if mag < self.off_th:
                self.in_contact = False
        else:
            if mag >= self.on_th:
                self.in_contact = True

        fz_smooth = self.ema.update(fz_db)
        fz_out = float(fz_smooth) if self.in_contact else 0.0
        return {
            "fz_raw": float(fz_raw),
            "fz_out": float(fz_out),
            "in_contact": float(1.0 if self.in_contact else 0.0),
        }


class SeqInfer:
    def __init__(self, ckpt_path: Path, device: str, baseline_vals: List[float]):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.args = ckpt.get("args", {})
        self.target_mode = ckpt.get("target_mode", self.args.get("target_mode", "absolute"))
        self.seq_len = int(self.args.get("seq_len", 16))

        tm = ckpt.get("target_mean", [0.0, 0.0, 0.0])
        ts = ckpt.get("target_std", [1.0, 1.0, 1.0])
        self.out_dim = int(ckpt.get("out_dim", len(tm)))
        self.aux_dim = int(ckpt.get("aux_dim", 3))

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = CnnLstmPose(aux_dim=self.aux_dim, out_dim=self.out_dim)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)

        self.target_mean = torch.tensor(tm, dtype=torch.float32, device=self.device)
        self.target_std = torch.tensor(ts, dtype=torch.float32, device=self.device)

        base = np.asarray(baseline_vals, dtype=np.float32)
        if base.shape[0] < self.out_dim:
            base = np.pad(base, (0, self.out_dim - base.shape[0]), mode="constant", constant_values=0.0)
        self.baseline = torch.tensor(base[: self.out_dim], dtype=torch.float32, device=self.device)

        self.tactile_buf: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.aux_buf: Deque[np.ndarray] = deque(maxlen=self.seq_len)

        self._x_mu: Optional[np.ndarray] = None
        self._x_var: Optional[np.ndarray] = None
        self._a_mu: Optional[np.ndarray] = None
        self._a_var: Optional[np.ndarray] = None
        self._n = 0

    def _update_running_stats(self, tac: np.ndarray, aux: np.ndarray) -> None:
        self._n += 1
        if self._x_mu is None:
            self._x_mu = tac.astype(np.float64)
            self._x_var = np.zeros_like(self._x_mu)
            self._a_mu = aux.astype(np.float64)
            self._a_var = np.zeros_like(self._a_mu)
            return

        dx = tac - self._x_mu
        self._x_mu = self._x_mu + dx / self._n
        self._x_var = self._x_var + dx * (tac - self._x_mu)

        da = aux - self._a_mu
        self._a_mu = self._a_mu + da / self._n
        self._a_var = self._a_var + da * (aux - self._a_mu)

    def _normalize(self, x_seq: np.ndarray, a_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._n < 2 or self._x_mu is None or self._a_mu is None or self._x_var is None or self._a_var is None:
            return x_seq, a_seq
        x_std = np.sqrt(np.clip(self._x_var / (self._n - 1), 1e-6, None))
        a_std = np.sqrt(np.clip(self._a_var / (self._n - 1), 1e-6, None))
        x_norm = (x_seq - self._x_mu.reshape(1, -1)) / x_std.reshape(1, -1)
        a_norm = (a_seq - self._a_mu.reshape(1, -1)) / a_std.reshape(1, -1)
        return x_norm.astype(np.float32), a_norm.astype(np.float32)

    @torch.no_grad()
    def step(self, skin16: List[float], fx: float, fy: float, fz: float) -> Optional[Dict[str, float]]:
        tac = np.asarray(skin16, dtype=np.float32)
        if tac.shape[0] != 16:
            raise ValueError("skin16 must have length 16")

        aux = np.asarray([fx, fy, fz], dtype=np.float32)
        if self.aux_dim < 3:
            aux = aux[: self.aux_dim]

        self._update_running_stats(tac, aux)
        self.tactile_buf.append(tac)
        self.aux_buf.append(aux)
        if len(self.tactile_buf) < self.seq_len:
            return None

        x_seq = np.stack(self.tactile_buf, axis=0)
        a_seq = np.stack(self.aux_buf, axis=0)
        x_seq, a_seq = self._normalize(x_seq, a_seq)

        x_t = torch.from_numpy(x_seq).unsqueeze(0).to(self.device)
        a_t = torch.from_numpy(a_seq).unsqueeze(0).to(self.device)

        pred_n = self.model(x_t, a_t).squeeze(0)
        pred_t = pred_n * self.target_std + self.target_mean
        if self.target_mode == "residual":
            pred_abs = pred_t + self.baseline
        else:
            pred_abs = pred_t

        out = {
            "x": float(pred_abs[0].item()),
            "y": float(pred_abs[1].item()),
            "z": float(pred_abs[2].item()),
            "fz_pred": float(pred_abs[3].item()) if self.out_dim >= 4 else None,
            "fz_meas": float(fz),
        }
        return out


def parse_args():
    p = argparse.ArgumentParser(description="Realtime/offline inference for training_seq checkpoint")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--in-csv", type=Path, required=True, help="input csv with Skin1..16,Fx,Fy,Fz")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--baseline-x", type=float, default=0.0)
    p.add_argument("--baseline-y", type=float, default=0.0)
    p.add_argument("--baseline-z", type=float, default=0.0)
    p.add_argument("--baseline-fz", type=float, default=0.0)

    p.add_argument("--fz-deadband", type=float, default=0.05)
    p.add_argument("--fz-on-th", type=float, default=0.15)
    p.add_argument("--fz-off-th", type=float, default=0.08)
    p.add_argument("--fz-ema-alpha", type=float, default=0.25)
    p.add_argument("--z-contact-gate", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    infer = SeqInfer(
        args.ckpt,
        args.device,
        [args.baseline_x, args.baseline_y, args.baseline_z, args.baseline_fz],
    )
    fz_filter = ContactFzFilter(args.fz_deadband, args.fz_on_th, args.fz_off_th, args.fz_ema_alpha)

    with open(args.in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        miss = [c for c in [*SKIN_COLS, "Fx", "Fy", "Fz"] if c not in (reader.fieldnames or [])]
        if miss:
            raise KeyError(f"Missing required columns: {miss}")

        rows_out = []
        for row in reader:
            skin = [float(row[c]) for c in SKIN_COLS]
            fx = float(row["Fx"])
            fy = float(row["Fy"])
            fz = float(row["Fz"])

            pred = infer.step(skin, fx, fy, fz)
            if pred is None:
                continue

            fz_source = pred["fz_pred"] if pred["fz_pred"] is not None else pred["fz_meas"]
            ff = fz_filter.update(float(fz_source))
            z_out = pred["z"] if (not args.z_contact_gate or ff["in_contact"] >= 0.5) else 0.0

            rows_out.append(
                {
                    "x": pred["x"],
                    "y": pred["y"],
                    "z": z_out,
                    "fz": ff["fz_out"],
                    "fz_raw": ff["fz_raw"],
                    "fz_source": "pred" if pred["fz_pred"] is not None else "meas",
                    "in_contact": int(ff["in_contact"] > 0.5),
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "z", "fz", "fz_raw", "fz_source", "in_contact"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(json.dumps({"saved": str(args.out_csv), "rows": len(rows_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
