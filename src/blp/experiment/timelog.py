"""บันทึกเวลาทำงานต่อ session ลงไฟล์ JSONL (ผลลัพธ์หลักของ Study 4 และ field pilot)

เวลาที่นับ = active time จาก start ถึง finish หักช่วง pause; งานนับว่าเสร็จเมื่อผ่าน checklist เท่านั้น
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class SessionTimer:
    def __init__(self, log_path: str | Path, participant_id: str, task: str, condition: str):
        self.path = Path(log_path)
        self.meta = {"participant_id": participant_id, "task": task, "condition": condition}

    def _write(self, event: str, **extra) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **self.meta, **extra}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def start(self) -> None:
        self._write("start")

    def pause(self) -> None:
        self._write("pause")

    def resume(self) -> None:
        self._write("resume")

    def finish(self, checklist_passed: bool) -> None:
        self._write("finish", checklist_passed=bool(checklist_passed))


def summarize_log(log_path: str | Path) -> pd.DataFrame:
    """คืนนาทีที่ทำงานจริงต่อ (participant, task, condition)"""
    ev = pd.read_json(log_path, lines=True)
    ev["ts"] = pd.to_datetime(ev["ts"])
    rows = []
    for key, g in ev.sort_values("ts").groupby(["participant_id", "task", "condition"]):
        active, running_since, passed = 0.0, None, None
        for _, r in g.iterrows():
            if r["event"] in ("start", "resume"):
                running_since = r["ts"]
            elif r["event"] in ("pause", "finish") and running_since is not None:
                active += (r["ts"] - running_since).total_seconds()
                running_since = None
            if r["event"] == "finish":
                passed = bool(r.get("checklist_passed"))
        rows.append(dict(zip(("participant_id", "task", "condition"), key), minutes=active / 60, checklist_passed=passed))
    return pd.DataFrame(rows)
