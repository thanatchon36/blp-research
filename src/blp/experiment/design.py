"""ออกแบบการทดลองแบบสุ่มไขว้ (randomized crossover) สำหรับ Study 4 (บทที่ 7.1)

- ผู้เข้าร่วมทำ 4 งานที่ความยากเทียบกัน สลับเงื่อนไข ABAB หรือ BABA (A = มี AI, B = ไม่มี AI)
- ลำดับงานใช้ Williams Latin square (สมดุล first-order carryover)
- สุ่มแบบชั้นภูมิตามประสบการณ์ ทำให้แต่ละงานอยู่ทั้งสองเงื่อนไขเท่ากัน
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WILLIAMS_4 = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
SEQUENCES = {"ABAB": (1, 0, 1, 0), "BABA": (0, 1, 0, 1)}


def make_participants(n: int, share_senior: float = 0.5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_senior = int(round(n * share_senior))
    exp = np.array(["senior"] * n_senior + ["junior"] * (n - n_senior))
    rng.shuffle(exp)
    return pd.DataFrame({"participant_id": [f"P{i + 1:02d}" for i in range(n)], "experience": exp})


def crossover_schedule(participants: pd.DataFrame, tasks=("T1", "T2", "T3", "T4"), seed: int = 2026) -> pd.DataFrame:
    if len(tasks) != 4:
        raise ValueError("โครงนี้ใช้ 4 งานต่อคน (Williams square ขนาด 4)")
    rng = np.random.default_rng(seed)
    rows, used = [], {"ABAB": 0, "BABA": 0}     # นับแถว Latin square ต่อ sequence ข้ามทุกชั้นภูมิ
    for stratum, grp in participants.groupby("experience"):
        ids = grp["participant_id"].to_numpy().copy()
        rng.shuffle(ids)
        first = int(rng.integers(0, 2))                  # สุ่มว่าชั้นภูมินี้เริ่มด้วย ABAB หรือ BABA
        for i, pid in enumerate(ids):
            seq_name = ("ABAB", "BABA")[(i + first) % 2]
            square_row = WILLIAMS_4[used[seq_name] % 4]
            used[seq_name] += 1
            for session, (ai, task_idx) in enumerate(zip(SEQUENCES[seq_name], square_row), start=1):
                rows.append({"participant_id": pid, "experience": stratum, "sequence": seq_name,
                             "session": session, "task": tasks[task_idx], "ai": ai,
                             "condition": "AI" if ai else "CTRL"})
    return pd.DataFrame(rows).sort_values(["participant_id", "session"]).reset_index(drop=True)


def balance_table(schedule: pd.DataFrame) -> pd.DataFrame:
    """ตรวจว่าแต่ละงานถูกทำภายใต้ AI และ CTRL จำนวนใกล้เคียงกัน"""
    return pd.crosstab(schedule["task"], schedule["condition"])
