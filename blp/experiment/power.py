"""Power analysis แบบ simulation สำหรับ H5: เวลาลดลงเกิน 20% (บทที่ 7.4)

แบบจำลองเวลา: log T_ij = mu + beta*AI + task_k + learning*(session-1) + u_i + w_i*AI + e_ij
ทดสอบด้านเดียวระดับบุคคล: d_i = mean(logT | AI) - mean(logT | CTRL), H0: E[d] >= ln(0.8)
ค่าความแปรปรวนตั้งต้นเป็นสมมติฐาน ต้องปรับด้วยข้อมูลจาก pilot และ time-motion study (บทที่ 6.1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def simulate_d(n: int, true_reduction: float, n_tasks: int = 4, n_sims: int = 2000, sd_task: float = 0.15,
               sd_resid: float = 0.30, sd_ai_slope: float = 0.15, learning: float = -0.03,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """คืน d_i ขนาด (n_sims, n) ภายใต้การสลับ ABAB/BABA และลำดับงานสุ่มต่อคน"""
    if n_tasks % 2:
        raise ValueError("n_tasks ต้องเป็นเลขคู่")
    rng = rng or np.random.default_rng()
    beta = np.log(1.0 - true_reduction)
    ai = np.tile(np.arange(n_tasks) % 2 == 0, (n, 1)).astype(float)       # ABAB...
    ai[1::2] = 1.0 - ai[1::2]                                                # ครึ่งหนึ่งเป็น BABA
    session = np.arange(n_tasks)[None, None, :]
    task_order = np.argsort(rng.random((n_sims, n, n_tasks)), axis=2)      # ลำดับงานต่อคน
    task_fx = rng.normal(0, sd_task, (n_sims, 1, n_tasks))
    task_term = np.take_along_axis(np.broadcast_to(task_fx, (n_sims, n, n_tasks)), task_order, axis=2)
    slope = rng.normal(0, sd_ai_slope, (n_sims, n, 1))
    log_t = (beta + slope) * ai + task_term + learning * session + rng.normal(0, sd_resid, (n_sims, n, n_tasks))
    half = n_tasks / 2
    return (log_t * ai).sum(axis=2) / half - (log_t * (1 - ai)).sum(axis=2) / half


def one_sided_p(d: np.ndarray, threshold: float = 0.20) -> np.ndarray:
    n = d.shape[-1]
    t = (d.mean(axis=-1) - np.log(1 - threshold)) / (d.std(axis=-1, ddof=1) / np.sqrt(n))
    return stats.t.cdf(t, df=n - 1)


def power_curve(ns=(10, 15, 20, 25, 30, 40), reductions=(0.25, 0.30, 0.35), n_tasks: int = 4,
                n_sims: int = 2000, alpha: float = 0.05, threshold: float = 0.20, seed: int = 11,
                **kw) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for red in reductions:
        for n in ns:
            p = one_sided_p(simulate_d(n, red, n_tasks=n_tasks, n_sims=n_sims, rng=rng, **kw), threshold)
            rows.append({"n_participants": n, "n_tasks": n_tasks, "true_reduction": red,
                         "power": float((p < alpha).mean())})
    return pd.DataFrame(rows)


def required_n(curve: pd.DataFrame, reduction: float, target: float = 0.80) -> int | None:
    ok = curve[(curve["true_reduction"] == reduction) & (curve["power"] >= target)]
    return int(ok["n_participants"].min()) if not ok.empty else None
