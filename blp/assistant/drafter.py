"""ร่างรายงานภาวะเศรษฐกิจภูมิภาคทีละหัวข้อ (บทที่ 6.2)

หลักการออกแบบ: (1) ทุกประโยคต้องอ้างอิง [N:<note_id>] หรือ [D:<key>] (2) ตัวเลขมาจากตาราง DATA เท่านั้น
(3) หนึ่งประโยคต่อหนึ่งบรรทัด เพื่อให้ verifier ตรวจได้ทีละประโยค (4) ไม่มี LLM = ใช้ extractive baseline
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ..preprocess import split_clauses
from ..scoring import detect_topic

SECTIONS: dict[str, dict] = {
    "ภาพรวมภาวะธุรกิจ": {"topics": ["sales", "employment", "investment"], "indicators": ["BLP_RAI", "UNCERTAINTY"],
                         "query": "ภาวะธุรกิจ ยอดขาย คำสั่งซื้อ การจ้างงาน การลงทุน"},
    "อุปสงค์และยอดขาย": {"topics": ["sales"], "indicators": ["sales_adj"],
                         "query": "ยอดขาย คำสั่งซื้อ กำลังซื้อ นักท่องเที่ยว อัตราการเข้าพัก"},
    "การจ้างงาน": {"topics": ["employment"], "indicators": ["employment_adj"],
                   "query": "การจ้างงาน พนักงาน แรงงาน ชั่วโมงทำงาน"},
    "การลงทุน": {"topics": ["investment"], "indicators": ["investment_adj"],
                 "query": "การลงทุน เครื่องจักร ขยายกำลังการผลิต งบลงทุน"},
    "ราคาและต้นทุน": {"topics": ["price", "cost"], "indicators": ["PRICE", "COST"],
                      "query": "ราคาขาย ต้นทุน วัตถุดิบ ค่าแรง พลังงาน"},
    "แนวโน้มข้างหน้า": {"topics": ["outlook"], "indicators": ["OUTLOOK"],
                        "query": "คาดว่า แนวโน้ม ไตรมาสหน้า ระยะข้างหน้า"},
}
INDICATOR_LABELS = {"BLP_RAI": "ดัชนีภาวะธุรกิจ BLP-RAI", "UNCERTAINTY": "ดัชนีความไม่แน่นอน",
                    "sales_adj": "ดัชนียอดขาย", "employment_adj": "ดัชนีการจ้างงาน",
                    "investment_adj": "ดัชนีการลงทุน", "PRICE": "ดัชนีราคาขาย", "COST": "ดัชนีต้นทุน",
                    "OUTLOOK": "ดัชนีแนวโน้มข้างหน้า"}

SYSTEM_PROMPT = """คุณเป็นผู้ช่วยนักเศรษฐศาสตร์ของธนาคารกลาง เขียนร่างรายงานภาวะเศรษฐกิจภูมิภาคภาษาไทยแบบทางการ กระชับ และเป็นกลาง
กฎบังคับ:
1. ใช้เฉพาะข้อมูลใน DATA และ EVIDENCE
2. เขียนหนึ่งประโยคต่อหนึ่งบรรทัด และทุกบรรทัดต้องลงท้ายด้วยการอ้างอิง [D:<key>] หรือ [N:<id>] อย่างน้อยหนึ่งรายการ
3. ตัวเลขทุกตัวต้องคัดลอกจาก DATA ตามที่แสดง ห้ามคำนวณ ปัดเศษ หรือประมาณเอง
4. ห้ามระบุชื่อบริษัท บุคคล หรือรายละเอียดที่ทำให้ระบุตัวตนได้
5. ถ้าหลักฐานไม่พอ ให้เขียนว่า "ข้อมูลไม่เพียงพอ" พร้อมอ้างอิงตารางที่ใช้ตรวจ"""


def cite(e: dict) -> str:
    return "".join(f"[N:{i}]" for i in e["ids"])


def quarter_th(q: pd.Period) -> str:
    return f"ไตรมาส {q.quarter} ปี {q.year + 543}"


@dataclass
class DraftSection:
    section: str
    sentences: list[str]
    data_table: dict[str, dict]
    evidence: list[dict]
    prompt: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


class ReportDrafter:
    def __init__(self, retriever, indicators: pd.DataFrame, llm=None, k: int = 12, max_evidence: int = 6):
        self.retriever, self.ind, self.llm = retriever, indicators, llm
        self.k, self.max_evidence = k, max_evidence

    def data_table(self, region: str, quarter: pd.Period, keys: list[str]) -> dict[str, dict]:
        rows = self.ind[self.ind["region"] == region].set_index("quarter")
        cur = rows.loc[quarter]
        prev = rows.loc[quarter - 1] if (quarter - 1) in rows.index else None
        table = {}
        for i, key in enumerate(keys, 1):
            v = round(float(cur[key]), 1)
            p = round(float(prev[key]), 1) if prev is not None else None
            table[f"D{i}"] = {"label": INDICATOR_LABELS[key], "value": v, "prev": p,
                              "change": round(v - p, 1) if p is not None else None}
        return table

    def evidence(self, region: str, quarter: pd.Period, section: str) -> list[dict]:
        spec = SECTIONS[section]
        hits = self.retriever.search(spec["query"], k=self.k, filters={"region": region, "quarter": str(quarter)})
        out: list[dict] = []
        for _, row in hits.iterrows():
            clauses = [re.sub(r"^ระบุว่า", "", c) for c in split_clauses(row["text_body"]) if detect_topic(c) in spec["topics"]]
            if not clauses:
                continue
            snippet = " ".join(clauses)
            same = next((e for e in out if e["snippet"] == snippet and e["sector"] == row["sector"]), None)
            if same is not None:                       # หลายบริษัทพูดเหมือนกัน -> รวมเป็นหลักฐานเดียว อ้างทุกบันทึก
                same["ids"].append(str(row["note_id"]))
                same["dates"].append(row["interview_date"])
            elif len(out) < self.max_evidence:
                out.append({"ids": [str(row["note_id"])], "dates": [row["interview_date"]],
                            "sector": row["sector"], "snippet": snippet})
        return out

    def _prompt(self, region: str, quarter: pd.Period, section: str, dt: dict, ev: list[dict]) -> str:
        data = "\n".join(f"[D:{k}] {d['label']} = {d['value']:.1f}" +
                         (f" (ไตรมาสก่อน {d['prev']:.1f}, เปลี่ยนแปลง {d['change']:.1f})" if d["prev"] is not None else "")
                         for k, d in dt.items())
        evidence = "\n".join(f"{cite(e)} (สาขา{e['sector']}) {e['snippet']}" for e in ev) or "(ไม่มี)"
        return (f"หัวข้อ: {section} | ภาค{region} | {quarter_th(quarter)}\n"
                "ค่าดัชนีเป็น net balance: ค่าบวก = ผู้ตอบว่าดีขึ้นมากกว่าแย่ลง\n\n"
                f"DATA:\n{data}\n\nEVIDENCE:\n{evidence}\n\nเขียน 3–5 ประโยค")

    @staticmethod
    def _extractive(dt: dict, ev: list[dict]) -> list[str]:
        out = []
        for key, d in dt.items():
            label = d["label"] + (" " if d["label"][-1].isascii() else "")
            if d["prev"] is None:
                out.append(f"{label}อยู่ที่ {d['value']:.1f} [D:{key}]")
            else:
                trend = "ปรับเพิ่มขึ้น" if d["change"] > 0 else ("ปรับลดลง" if d["change"] < 0 else "ทรงตัว")
                out.append(f"{label}{trend}จาก {d['prev']:.1f} ในไตรมาสก่อนเป็น {d['value']:.1f} [D:{key}]")
        for e in ev[:3]:
            who = f"ผู้ประกอบการสาขา{e['sector']}" + (f" {len(e['ids'])} ราย" if len(e["ids"]) > 1 else "")
            out.append(f"{who}ระบุว่า{e['snippet']} {cite(e)}")
        return out

    def draft_section(self, region: str, quarter: pd.Period, section: str) -> DraftSection:
        spec = SECTIONS[section]
        dt = self.data_table(region, quarter, spec["indicators"])
        ev = self.evidence(region, quarter, section)
        if self.llm is None:
            return DraftSection(section, self._extractive(dt, ev), dt, ev, meta={"engine": "extractive_baseline"})
        prompt = self._prompt(region, quarter, section, dt, ev)
        raw = self.llm.complete(SYSTEM_PROMPT, prompt, max_tokens=900)
        lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in raw.splitlines()]
        return DraftSection(section, [ln for ln in lines if ln], dt, ev, prompt, meta={"engine": self.llm.name})

    def draft_report(self, region: str, quarter: pd.Period, sections: list[str] | None = None):
        drafts = [self.draft_section(region, quarter, s) for s in (sections or list(SECTIONS))]
        md = [f"# ร่างรายงานภาวะเศรษฐกิจภาค{region} {quarter_th(quarter)}", "",
              "> ร่างโดย AI assistant ต้องผ่านการตรวจทานของนักเศรษฐศาสตร์ก่อนเผยแพร่", ""]
        for d in drafts:
            md += [f"## {d.section}", "", d.text, ""]
        md += ["## แหล่งอ้างอิง", ""]
        seen = set()
        for d in drafts:
            for key, row in d.data_table.items():
                md.append(f"- [D:{key}] ({d.section}) {row['label']}")
            for e in d.evidence:
                for note_id, date in zip(e["ids"], e["dates"]):
                    if note_id not in seen:
                        seen.add(note_id)
                        md.append(f"- [N:{note_id}] บันทึก BLP สาขา{e['sector']} สัมภาษณ์ {date}")
        return "\n".join(md), drafts
