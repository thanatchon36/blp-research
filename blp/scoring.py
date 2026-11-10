"""ให้คะแนนบันทึก BLP เป็นทิศทาง -1/0/+1 รายหัวข้อ (บทที่ 4.2)

M1 LexiconScorer   : พจนานุกรมคำไทย + จัดการคำปฏิเสธ (baseline โปร่งใส)
M2 fine-tuned model: ฝึก WangchanBERTa ด้วย gold standard (ทำนอกไฟล์นี้ ใช้ interface เดียวกัน: score_text)
M3 LLMScorer       : LLM ตอบ JSON ตาม schema
M4 EnsembleScorer  : majority vote
ทุก scorer มีเมธอด score_text(text) -> {topic: -1/0/1 หรือ NaN}
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np
import pandas as pd

from .constants import TOPICS
from .preprocess import split_clauses

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "outlook": ["คาดว่า", "แนวโน้ม", "ไตรมาสหน้า", "ระยะข้างหน้า", "ปีหน้า", "มองว่า"],
    "price": ["ราคาขาย", "ปรับราคา", "ขึ้นราคา", "ลดราคา", "ราคาสินค้า", "ค่าห้อง"],
    "cost": ["ต้นทุน", "ค่าแรง", "ค่าวัตถุดิบ", "ราคาวัตถุดิบ", "ค่าไฟ", "ค่าพลังงาน", "ค่าขนส่ง"],
    "employment": ["จ้างงาน", "พนักงาน", "แรงงาน", "ลูกจ้าง", "ชั่วโมงทำงาน"],
    "investment": ["ลงทุน", "เครื่องจักร", "ขยายกำลังการผลิต", "ขยายสาขา"],
    "sales": ["ยอดขาย", "คำสั่งซื้อ", "ออเดอร์", "รายได้", "นักท่องเที่ยว", "อัตราการเข้าพัก", "ยอดจอง", "ยอดโอน"],
}
TOPIC_PRIORITY = ("outlook", "price", "cost", "employment", "investment", "sales")
UP = ["เพิ่มขึ้น", "ขยายตัว", "สูงขึ้น", "ดีขึ้น", "ฟื้นตัว", "ปรับขึ้น", "เร่งตัว", "เชิงบวก", "ขยาย",
      "รับพนักงานใหม่", "คึกคัก", "ขึ้นราคา"]
DOWN = ["ลดลง", "หดตัว", "ชะลอ", "แย่ลง", "ซบเซา", "เลื่อน", "ปรับลด", "ลด", "เชิงลบ", "อ่อนแอ", "กังวล", "ต่ำลง"]
FLAT = ["ทรงตัว", "คงที่", "ไม่เปลี่ยนแปลง", "ใกล้เคียง", "ตามปกติ"]
NEGATION = ["ไม่ได้", "ยังไม่", "ไม่"]
_UP, _DOWN = set(UP), set(DOWN)
_DIR_RE = re.compile("|".join(sorted(map(re.escape, UP + DOWN + FLAT), key=len, reverse=True)))


class Scorer(Protocol):
    name: str

    def score_text(self, text: str) -> dict[str, float]: ...


def detect_topic(clause: str) -> str | None:
    for topic in TOPIC_PRIORITY:
        if any(k in clause for k in TOPIC_KEYWORDS[topic]):
            return topic
    return None


def clause_direction(clause: str, window: int = 8) -> int:
    up = down = 0
    for m in _DIR_RE.finditer(clause):
        negated = any(n in clause[max(0, m.start() - window):m.start()] for n in NEGATION)
        if negated:
            continue                       # "ยังไม่ปรับขึ้น" = ไม่ใช่ขาขึ้น -> นับเป็นกลาง
        term = m.group(0)
        up += term in _UP
        down += term in _DOWN
    return int(np.sign(up - down))


class LexiconScorer:
    name = "M1_lexicon"

    def score_text(self, text: str) -> dict[str, float]:
        per_topic: dict[str, list[int]] = {}
        for clause in split_clauses(text):
            topic = detect_topic(clause)
            if topic is not None:
                per_topic.setdefault(topic, []).append(clause_direction(clause))
        return {t: float(np.sign(np.mean(per_topic[t]))) if t in per_topic else math.nan for t in TOPICS}


LLM_LABELS = {"up": 1.0, "flat": 0.0, "down": -1.0, "not_mentioned": math.nan}
LLM_SCHEMA = {
    "type": "object",
    "properties": {t: {"type": "string", "enum": list(LLM_LABELS)} for t in TOPICS},
    "required": list(TOPICS),
    "additionalProperties": False,
}
LLM_SYSTEM = ("คุณเป็นผู้ช่วยลงรหัสบันทึกสัมภาษณ์ผู้ประกอบการของธนาคารกลาง (Business Liaison Program) "
              "ตอบเป็น JSON ตาม schema เท่านั้น ห้ามอธิบายเพิ่ม")
LLM_PROMPT = """Codebook (เทียบกับไตรมาสก่อน เฉพาะข้อมูลของกิจการที่ให้สัมภาษณ์):
- sales: ยอดขาย คำสั่งซื้อ รายได้ ลูกค้า นักท่องเที่ยว อัตราการเข้าพัก
- employment: จำนวนพนักงาน การจ้างงาน ชั่วโมงทำงาน
- investment: การลงทุน เครื่องจักร การขยายกำลังการผลิตหรือสาขาของกิจการเอง
- price: ราคาขายสินค้าหรือบริการของกิจการ
- cost: ต้นทุนของกิจการ เช่น วัตถุดิบ ค่าแรง พลังงาน
- outlook: มุมมองต่อธุรกิจของกิจการในระยะข้างหน้า
ค่าที่ใช้: up / flat / down / not_mentioned
กฎ: ถ้าหัวข้อเดียวมีทั้งเพิ่มและลด ให้ตอบ flat | ข้อความเรื่องคู่แข่ง ภาครัฐ หรือตลาดโลก ไม่นับเป็นข้อมูลของกิจการ | "ยังไม่ปรับขึ้น" = flat

ตัวอย่าง: "ยอดขายในประเทศเพิ่มขึ้น แต่ยอดขายส่งออกลดลง คู่แข่งขยายสาขาเพิ่มขึ้น"
คำตอบ: {{"sales":"flat","employment":"not_mentioned","investment":"not_mentioned","price":"not_mentioned","cost":"not_mentioned","outlook":"not_mentioned"}}

บันทึก: <<<{text}>>>"""


class LLMScorer:
    def __init__(self, llm, max_tokens: int = 300):
        self.llm, self.max_tokens = llm, max_tokens
        self.name = f"M3_llm[{getattr(llm, 'name', 'llm')}]"

    def score_text(self, text: str) -> dict[str, float]:
        from .assistant.llm import extract_json
        raw = self.llm.complete(LLM_SYSTEM, LLM_PROMPT.format(text=text), max_tokens=self.max_tokens,
                                json_schema=LLM_SCHEMA)
        try:
            out = extract_json(raw)
        except ValueError:
            return {t: math.nan for t in TOPICS}           # นับเป็น parse failure ในการประเมิน
        return {t: LLM_LABELS.get(str(out.get(t, "not_mentioned")), math.nan) for t in TOPICS}


class EnsembleScorer:
    """M4: majority vote ของหลาย scorer (เสมอกันให้ใช้ผลของ scorer ตัวแรก)"""

    def __init__(self, scorers: list[Scorer]):
        self.scorers = scorers
        self.name = "M4_ensemble"

    def score_text(self, text: str) -> dict[str, float]:
        outs = [s.score_text(text) for s in self.scorers]
        result = {}
        for t in TOPICS:
            votes = Counter("NA" if math.isnan(o[t]) else o[t] for o in outs)
            top = votes.most_common()
            winner = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else ("NA" if math.isnan(outs[0][t]) else outs[0][t])
            result[t] = math.nan if winner == "NA" else float(winner)
        return result


def score_notes(notes: pd.DataFrame, scorer: Scorer, text_col: str = "text_body",
                id_col: str = "note_id") -> pd.DataFrame:
    """คืนตาราง long: note_id, topic, score"""
    rows = []
    for note_id, text in zip(notes[id_col], notes[text_col]):
        for topic, score in scorer.score_text(text).items():
            rows.append({id_col: note_id, "topic": topic, "score": score})
    return pd.DataFrame(rows)
