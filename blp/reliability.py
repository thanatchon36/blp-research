"""วัดความน่าเชื่อถือของการลงรหัส (บทที่ 4.4): คน-คน และ โมเดล-gold standard

ค่า NaN (ไม่กล่าวถึง) ถูกนับเป็นหมวด "NA" เพราะการตรวจจับว่ามีหัวข้อหรือไม่ก็เป็นส่วนหนึ่งของความถูกต้อง
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

NA_CODE = 9.0


def _codes(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.where(np.isnan(a), NA_CODE, a)


def cohen_kappa(a, b) -> float:
    return float(cohen_kappa_score(_codes(a), _codes(b)))


def krippendorff_alpha(data: np.ndarray, level: str = "nominal") -> float:
    """data: coders x units (NaN = ไม่มีค่า). level = 'nominal' หรือ 'interval'"""
    data = np.asarray(data, dtype=float)
    values = np.unique(data[~np.isnan(data)])
    index = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for unit in data.T:
        vals = unit[~np.isnan(unit)]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[index[vals[i]], index[vals[j]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    delta = 1.0 - np.eye(len(values)) if level == "nominal" else (values[:, None] - values[None, :]) ** 2
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1.0 - d_o / d_e) if d_e > 0 else float("nan")


def classification_metrics(gold, pred) -> dict[str, float]:
    g, p = _codes(gold), _codes(pred)
    return {"n": int(len(g)), "accuracy": float(accuracy_score(g, p)),
            "macro_f1": float(f1_score(g, p, average="macro")), "kappa": float(cohen_kappa_score(g, p))}


def compare_to_gold(scored: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """scored: note_id, topic, score | gold: note_id, topic, label -> ตารางรายหัวข้อ + ภาพรวม"""
    m = gold.merge(scored, on=["note_id", "topic"], how="inner")
    rows = [{"topic": t, **classification_metrics(g["label"], g["score"])} for t, g in m.groupby("topic")]
    rows.append({"topic": "ALL", **classification_metrics(m["label"], m["score"])})
    return pd.DataFrame(rows)
