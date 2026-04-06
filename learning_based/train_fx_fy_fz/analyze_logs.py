import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


# Resolve logs directory relative to this file (../../logs)
LOGS_DIR = str(Path(__file__).resolve().parents[1] / "logs")

# Point-to-sensor mapping (2x2 block per sensing point, sensor names)
SENSOR_MAPPING: Dict[int, List[str]] = {
    1: ["s1",  "s2",  "s5",  "s6"],
    2: ["s2",  "s3",  "s6",  "s7"],
    3: ["s3",  "s4",  "s7",  "s8"],
    4: ["s5",  "s6",  "s9",  "s10"],
    5: ["s6",  "s7",  "s10", "s11"],
    6: ["s7",  "s8",  "s11", "s12"],
    7: ["s9",  "s10", "s13", "s14"],
    8: ["s10", "s11", "s14", "s15"],
    9: ["s11", "s12", "s15", "s16"],
}


def parse_point_id_from_dir(run_dir: str) -> int:
    base = os.path.basename(run_dir)
    # expected like p1_* or p7_*
    if not base.startswith('p'):
        return -1
    try:
        pid = int(base[1:base.find('_')] if '_' in base else base[1:])
        return pid if 1 <= pid <= 9 else -1
    except Exception:
        return -1


def compute_stats(x: np.ndarray) -> Dict[str, Any]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    q = np.percentile(x, [5, 50, 95])
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q05": float(q[0]),
        "q50": float(q[1]),
        "q95": float(q[2]),
        "max": float(np.max(x)),
    }


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float('nan')
    try:
        c = np.corrcoef(a[m], b[m])[0, 1]
        return float(c)
    except Exception:
        return float('nan')


def cond_number(X: np.ndarray) -> float:
    try:
        # standardize columns to unit variance to avoid scale effects
        Xc = X - np.nanmean(X, axis=0, keepdims=True)
        s = np.nanstd(Xc, axis=0, keepdims=True)
        s[s == 0.0] = 1.0
        Xs = Xc / s
        u, svals, vh = np.linalg.svd(np.nan_to_num(Xs), full_matrices=False)
        if svals.min() == 0:
            return float('inf')
        return float(svals.max() / svals.min())
    except Exception:
        return float('nan')


def build_uvw(df: pd.DataFrame, pid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sA, sB, sC, sD = SENSOR_MAPPING[pid]
    a = df[sA].to_numpy(dtype=float)
    b = df[sB].to_numpy(dtype=float)
    c = df[sC].to_numpy(dtype=float)
    d = df[sD].to_numpy(dtype=float)
    u = a + b + c + d
    v = (b + d) - (a + c)
    w = (c + d) - (a + b)
    return u, v, w


def analyze_run(run_dir: str) -> Dict[str, Any]:
    csv_path = os.path.join(run_dir, 'merged_data.csv')
    if not os.path.exists(csv_path):
        return {"run": os.path.basename(run_dir), "error": "merged_data.csv not found"}

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    pid = parse_point_id_from_dir(run_dir)
    point_key = f"p{pid}" if pid != -1 else "unknown"

    out: Dict[str, Any] = {
        "run": os.path.basename(run_dir),
        "point": point_key,
        "rows_total": int(len(df)),
    }

    # Basic masks
    sensor_cols = [f"s{i}" for i in range(1, 17)]
    sensors_ok = df[sensor_cols].fillna(0.0).sum(axis=1) != 0.0
    have_cols = all(c in df.columns for c in ["x_c", "y_c", "fx", "fy", "fz"]) and ("z_displacement_mm_laser" in df.columns)
    if have_cols:
        label_ok = df[["x_c", "y_c", "fx", "fy", "fz", "z_displacement_mm_laser"]].notna().all(axis=1)
    else:
        label_ok = pd.Series([False] * len(df))
    valid = sensors_ok & label_ok

    out["rows_nonzero_sensors"] = int(sensors_ok.sum())
    out["rows_valid_labels"] = int(label_ok.sum())
    out["rows_valid_both"] = int(valid.sum())

    # tz residual stats if present
    if "tz_resid" in df.columns:
        tz = df["tz_resid"].astype(float)
        out["tz_resid_stats"] = compute_stats(tz.to_numpy())
        out["tz_resid_frac_gt_0p01"] = float((tz.abs() > 0.01).mean())
        out["tz_resid_frac_gt_0p02"] = float((tz.abs() > 0.02).mean())

    # Target stats on valid rows
    if valid.any():
        sub = df[valid]
        z_m = sub["z_displacement_mm_laser"].astype(float).to_numpy() / 1000.0
        for name in ("x_c", "y_c", "fx", "fy", "fz"):
            out[f"stats_{name}"] = compute_stats(sub[name].astype(float).to_numpy())
        out["stats_z_m"] = compute_stats(z_m)

        # Shear vs normal magnitude ratios
        shear = np.abs(sub["fx"].astype(float).to_numpy()) + np.abs(sub["fy"].astype(float).to_numpy())
        norm = np.abs(sub["fz"].astype(float).to_numpy()) + 1e-9
        rat = shear / norm
        out["shear_over_normal_stats"] = compute_stats(rat)

        # uvw correlations if point is known
        if pid in SENSOR_MAPPING:
            u, v, w = build_uvw(sub, pid)
            out["corr_u_fz"] = corr_safe(u, sub["fz"].astype(float).to_numpy())
            out["corr_v_fx"] = corr_safe(v, sub["fx"].astype(float).to_numpy())
            out["corr_w_fy"] = corr_safe(w, sub["fy"].astype(float).to_numpy())
            # inter-sensor correlation and condition number for the 4 sensors
            s4 = sub[SENSOR_MAPPING[pid]].astype(float).to_numpy()
            out["s4_condition_number"] = cond_number(s4)
            try:
                cmat = np.corrcoef(s4, rowvar=False)
                # store upper triangle without diagonal
                tri = []
                k = s4.shape[1]
                for i in range(k):
                    for j in range(i + 1, k):
                        tri.append(float(cmat[i, j]))
                out["s4_corr_pairs"] = tri
            except Exception:
                pass

    return out


def analyze_all(logs_dir: str) -> List[Dict[str, Any]]:
    runs = [d for d in glob.glob(os.path.join(logs_dir, "p*_*")) if os.path.isdir(d)]
    results = []
    for d in sorted(runs):
        try:
            res = analyze_run(d)
            results.append(res)
        except Exception as e:
            results.append({"run": os.path.basename(d), "error": str(e)})
    return results


def summarize_by_point(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate key metrics per point p1..p9
    agg: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        p = r.get("point", "unknown")
        if p not in agg:
            agg[p] = {"rows": [], "corr_v_fx": [], "corr_w_fy": [], "corr_u_fz": [],
                      "s4_cond": [], "shear_over_normal_mean": []}
        rows = r.get("rows_valid_both") or 0
        agg[p]["rows"].append(float(rows))
        for key, dst in (("corr_v_fx", "corr_v_fx"), ("corr_w_fy", "corr_w_fy"), ("corr_u_fz", "corr_u_fz")):
            if key in r and np.isfinite(r[key]):
                agg[p][dst].append(float(r[key]))
        if "s4_condition_number" in r and np.isfinite(r["s4_condition_number"]):
            agg[p]["s4_cond"].append(float(r["s4_condition_number"]))
        s_stats = r.get("shear_over_normal_stats") or {}
        if "mean" in s_stats:
            agg[p]["shear_over_normal_mean"].append(float(s_stats["mean"]))

    # Reduce to means
    out: Dict[str, Any] = {}
    for p, d in agg.items():
        out[p] = {
            "runs": int(len(d["rows"])),
            "avg_valid_rows": float(np.mean(d["rows"])) if d["rows"] else 0.0,
            "avg_corr_v_fx": float(np.mean(d["corr_v_fx"])) if d["corr_v_fx"] else None,
            "avg_corr_w_fy": float(np.mean(d["corr_w_fy"])) if d["corr_w_fy"] else None,
            "avg_corr_u_fz": float(np.mean(d["corr_u_fz"])) if d["corr_u_fz"] else None,
            "avg_s4_condition": float(np.mean(d["s4_cond"])) if d["s4_cond"] else None,
            "avg_shear_over_normal": float(np.mean(d["shear_over_normal_mean"])) if d["shear_over_normal_mean"] else None,
        }
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=LOGS_DIR, help="Path to logs folder containing p*_* runs")
    ap.add_argument("--out_json", default=None, help="Optional path to save per-run JSON results")
    ap.add_argument("--out_csv", default=None, help="Optional path to save per-run CSV key metrics")
    ap.add_argument("--summary_json", default=None, help="Optional path to save aggregated summary by point")
    args = ap.parse_args()

    results = analyze_all(args.logs)

    # Print concise per-run lines
    for r in results:
        if "error" in r:
            print(f"{r['run']}: ERROR {r['error']}")
            continue
        p = r.get("point")
        rows = r.get("rows_valid_both")
        c_vfx = r.get("corr_v_fx")
        c_wfy = r.get("corr_w_fy")
        c_ufz = r.get("corr_u_fz")
        s4c = r.get("s4_condition_number")
        print(f"{r['run']} [{p}] valid={rows} corr(v,fx)={c_vfx:.3f} corr(w,fy)={c_wfy:.3f} corr(u,fz)={c_ufz:.3f} cond4={s4c:.2f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved per-run JSON: {args.out_json}")

    if args.out_csv:
        rows = []
        for r in results:
            row = {
                "run": r.get("run"),
                "point": r.get("point"),
                "rows_total": r.get("rows_total"),
                "rows_valid_both": r.get("rows_valid_both"),
                "corr_v_fx": r.get("corr_v_fx"),
                "corr_w_fy": r.get("corr_w_fy"),
                "corr_u_fz": r.get("corr_u_fz"),
                "s4_condition_number": r.get("s4_condition_number"),
                "tz_resid_frac_gt_0p01": r.get("tz_resid_frac_gt_0p01"),
                "tz_resid_frac_gt_0p02": r.get("tz_resid_frac_gt_0p02"),
            }
            rows.append(row)
        out_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        out_df.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
        print(f"Saved per-run CSV: {args.out_csv}")

    if args.summary_json:
        summary = summarize_by_point(results)
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        with open(args.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
