import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_model(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # Convert lists to numpy arrays for math
    if m.get("model_type") == "two_stage" or "stages" in m:
        st = m.get("stages", m)
        for key in ("stage1", "stage2"):
            if key in st:
                stg = st[key]
                stg["W"] = np.asarray(stg["W"], dtype=float)
                stg["b"] = np.asarray(stg["b"], dtype=float)
                stg["x_scaler"]["mean"] = np.asarray(stg["x_scaler"]["mean"], dtype=float)
                stg["x_scaler"]["scale"] = np.asarray(stg["x_scaler"]["scale"], dtype=float)
                stg["y_scaler"]["mean"] = np.asarray(stg["y_scaler"]["mean"], dtype=float)
                stg["y_scaler"]["scale"] = np.asarray(stg["y_scaler"]["scale"], dtype=float)
        m["stages"] = st
        m["model_type"] = "two_stage"
    else:
        m["W"] = np.asarray(m["W"], dtype=float)
        m["b"] = np.asarray(m["b"], dtype=float)
        m["x_scaler"]["mean"] = np.asarray(m["x_scaler"]["mean"], dtype=float)
        m["x_scaler"]["scale"] = np.asarray(m["x_scaler"]["scale"], dtype=float)
        m["y_scaler"]["mean"] = np.asarray(m["y_scaler"]["mean"], dtype=float)
        m["y_scaler"]["scale"] = np.asarray(m["y_scaler"]["scale"], dtype=float)
    return m


def _features_from_sensors(model: Dict[str, Any], sensors_16: List[float], x_c: float = 0.0, y_c: float = 0.0) -> np.ndarray:
    mode = model.get("features_mode", "sumdiff")
    s = sensors_16
    if mode == "sumdiff":
        # sensor_mapping holds names like ["s9","s10","s13","s14"]
        names = model["sensor_mapping"]
        def get_val(name: str) -> float:
            if not name.startswith("s"):
                return 0.0
            idx = int(name[1:]) - 1
            return float(s[idx]) if 0 <= idx < 16 else 0.0
        a, b, c, d = (get_val(n) for n in names)
        u = a + b + c + d
        v = (b + d) - (a + c)
        w = (c + d) - (a + b)
        X = np.array([u, v, w], dtype=float)
        return X
    elif mode == "all16":
        X = np.asarray(sensors_16, dtype=float)
        return X
    else:
        raise ValueError(f"Unknown features_mode: {mode}")


def predict_from_sensors(model: Dict[str, Any], sensors_16: List[float]) -> np.ndarray:
    if model.get("model_type") == "two_stage":
        stages = model["stages"]
        # Stage1
        Xb = _features_from_sensors(model, sensors_16)
        Xm1 = stages["stage1"]["x_scaler"]["mean"]
        Xs1 = stages["stage1"]["x_scaler"]["scale"]
        X1z = (Xb - Xm1) / Xs1
        Y1z = X1z @ stages["stage1"]["W"] + stages["stage1"]["b"]
        Ym1 = stages["stage1"]["y_scaler"]["mean"]
        Ys1 = stages["stage1"]["y_scaler"]["scale"]
        Y1 = Y1z * Ys1 + Ym1  # [x_c, y_c, z_m]
        # Stage2 features: base + predicted (x_c,y_c)
        X2 = np.hstack([Xb, Y1[:2]])
        Xm2 = stages["stage2"]["x_scaler"]["mean"]
        Xs2 = stages["stage2"]["x_scaler"]["scale"]
        X2z = (X2 - Xm2) / Xs2
        Y2z = X2z @ stages["stage2"]["W"] + stages["stage2"]["b"]
        Ym2 = stages["stage2"]["y_scaler"]["mean"]
        Ys2 = stages["stage2"]["y_scaler"]["scale"]
        Y2 = Y2z * Ys2 + Ym2  # [fx, fy, fz]
        return np.hstack([Y1, Y2])
    else:
        # Single-stage
        X = _features_from_sensors(model, sensors_16)
        Xm = model["x_scaler"]["mean"]
        Xs = model["x_scaler"]["scale"]
        Xz = (X - Xm) / Xs
        Yz = Xz @ model["W"] + model["b"]
        Ym = model["y_scaler"]["mean"]
        Ys = model["y_scaler"]["scale"]
        Y = Yz * Ys + Ym
        return Y


def predict_from_sensors_dict(model: Dict[str, Any], sensors_16: List[float]) -> Dict[str, float]:
    y = predict_from_sensors(model, sensors_16)
    names = model.get("target_names") or (["x_c","y_c","z_m","fx","fy","fz"] if model.get("model_type") == "two_stage" else ["fx","fy","fz"])
    out = {names[i]: float(y[i]) for i in range(min(len(names), len(y)))}
    return out
