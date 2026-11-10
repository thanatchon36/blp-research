"""เตรียมข้อมูล: ปิดบังข้อมูลที่ระบุตัวตน (anonymization) และตัดข้อความภาษาไทยเป็นวลี (บทที่ 3.4)

ลำดับความน่าเชื่อถือ: (1) รายชื่อจากฐานข้อมูลผู้ติดต่อ BLP แบบ exact match  (2) regex สำรอง
งานจริงควรเพิ่ม (3) NER ภาษาไทย แล้ววัด recall ของการปิดบังกับตัวอย่างที่คนตรวจ
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

THAI = "\u0E00-\u0E7F"
COMPANY_FULL = re.compile(r"(?:บริษัท|บจก\.|บมจ\.|ห้างหุ้นส่วนจำกัด|หจก\.)\s*\S+(?:\s+\S+){0,3}?\s+จำกัด(?:\s*\(มหาชน\))?")
COMPANY_SHORT = re.compile(r"(?:บริษัท|บจก\.|บมจ\.|หจก\.)\s+[^\s\]\[]+")
# "คุณ" ที่ไม่ใช่คำว่า คุณภาพ/คุณค่า/คุณสมบัติ ฯลฯ
PERSON = re.compile(rf"คุณ(?!ภาพ|ค่า|สมบัติ|ประโยชน์|ลักษณะ|ธรรม|วุฒิ|ูปการ)\s?[{THAI}]{{2,15}}")
PHONE = re.compile(r"(?<!\d)0\d{1,2}[-\s]?\d{3}[-\s]?\d{3,4}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PATTERNS = (("[อีเมล]", EMAIL), ("[โทรศัพท์]", PHONE), ("[บริษัท]", COMPANY_FULL),
            ("[บริษัท]", COMPANY_SHORT), ("[บุคคล]", PERSON))


@dataclass
class Anonymizer:
    entities: dict[str, str] = field(default_factory=dict)   # ชื่อจริง -> placeholder
    use_regex: bool = True

    def __post_init__(self) -> None:
        terms = sorted(self.entities, key=len, reverse=True)  # ยาวก่อน กันชื่อซ้อนกัน
        self._dict_re = re.compile("|".join(map(re.escape, terms))) if terms else None

    def anonymize(self, text: str) -> str:
        if self._dict_re is not None:
            text = self._dict_re.sub(lambda m: self.entities[m.group(0)], text)
        if self.use_regex:
            for placeholder, pattern in PATTERNS:
                text = pattern.sub(placeholder, text)
        return text

    def find_leaks(self, text: str) -> list[str]:
        """คืนข้อความที่ยังดูเหมือนข้อมูลระบุตัวตน ใช้ทั้งตอนตรวจผลปิดบังและใน verifier ของ AI assistant"""
        hits: list[str] = []
        if self._dict_re is not None:
            hits += [m.group(0) for m in self._dict_re.finditer(text)]
        for _, pattern in PATTERNS:
            hits += [m.group(0) for m in pattern.finditer(text)]
        return sorted(set(hits))


def body_text(text: str) -> str:
    """ตัดบรรทัดหัว (วันที่ ชื่อ สาขา ผู้ให้ข้อมูล) ออก เหลือเฉพาะเนื้อหาสัมภาษณ์"""
    return text.split("\n", 1)[1] if "\n" in text else text


def split_clauses(text: str) -> list[str]:
    """ภาษาไทยใช้ช่องว่างคั่นประโยคหรืออนุประโยค จึงตัดตามช่องว่าง (งานจริงเทียบกับ CRF sentence segmenter)"""
    return [c for c in re.split(r"\s+", text.strip()) if c]
