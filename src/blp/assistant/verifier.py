"""ตรวจร่างก่อนส่งให้นักเศรษฐศาสตร์ (บทที่ 6.5): ร่างที่ไม่ผ่านจะถูกบล็อก ไม่ใช่แค่เตือน

1) Numeric grounding : ตัวเลขทุกตัวต้องตรงกับ DATA, หลักฐานที่อ้าง, ค่าบริบท (ไตรมาส/ปี)
                      หรือจำนวนบันทึกที่อ้างในประโยคนั้น
2) Citation validity : [N:id] และ [D:key] ต้องมีอยู่จริงในชุดหลักฐานของหัวข้อนั้น
3) Citation coverage : สัดส่วนประโยคที่มีการอ้างอิง
4) Confidentiality   : ไม่มีชื่อบริษัท บุคคล เบอร์โทร หรืออีเมล
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ..preprocess import Anonymizer

CITE = re.compile(r"\[(N|D):([^\]\s]+)\]")
NUM = re.compile(r"(?<![A-Za-z0-9.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(m.group(0).replace(",", "").replace("−", "-")) for m in NUM.finditer(text)]


@dataclass
class VerificationReport:
    n_sentences: int = 0
    n_cited: int = 0
    unverified_numbers: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    pii_hits: list[str] = field(default_factory=list)
    min_coverage: float = 0.95

    @property
    def coverage(self) -> float:
        return self.n_cited / self.n_sentences if self.n_sentences else 0.0

    @property
    def passed(self) -> bool:
        return (not self.unverified_numbers and not self.invalid_citations and not self.pii_hits
                and self.coverage >= self.min_coverage)

    def to_dict(self) -> dict:
        return {**asdict(self), "coverage": self.coverage, "passed": self.passed}


def verify_sentences(sentences: list[str], data_table: dict[str, dict], evidence: list[dict],
                     extra_allowed=(), anonymizer: Anonymizer | None = None, tol: float = 0.051,
                     min_coverage: float = 0.95) -> VerificationReport:
    allowed: set[float] = {float(x) for x in extra_allowed}
    for d in data_table.values():
        for key in ("value", "prev", "change"):
            if d.get(key) is not None:
                allowed |= {d[key], abs(d[key])}
    for e in evidence:
        allowed |= set(_numbers(e["snippet"]))
    note_ids, data_keys = {i for e in evidence for i in e["ids"]}, set(data_table)
    anonymizer = anonymizer or Anonymizer()

    rep = VerificationReport(min_coverage=min_coverage)
    for s in sentences:
        rep.n_sentences += 1
        cites = CITE.findall(s)
        bad = [f"{kind}:{ref}" for kind, ref in cites
               if (kind == "N" and ref not in note_ids) or (kind == "D" and ref not in data_keys)]
        rep.invalid_citations += bad
        if cites and len(bad) < len(cites):
            rep.n_cited += 1
        else:
            rep.uncited_sentences.append(s)
        local = allowed | {float(sum(kind == "N" for kind, _ in cites))}   # "3 ราย" ต้องเท่ากับจำนวนบันทึกที่อ้าง
        for x in _numbers(CITE.sub(" ", s)):
            if not any(abs(x - a) <= tol for a in local):
                rep.unverified_numbers.append(f"{x:g}")
        rep.pii_hits += anonymizer.find_leaks(s)
    return rep
