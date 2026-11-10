"""ข้อมูลจำลองของโครงการ BLP (synthetic data)

ใช้ทดสอบ pipeline แทนข้อมูลจริงซึ่งเป็นความลับ โดยจงใจใส่ปัญหาที่งานวิจัยต้องรับมือ:
1) ข้อความไทยมีทั้งประโยคชัดเจน ประโยคผสม (เพิ่ม+ลด) การปฏิเสธ และประโยคหลอก (คู่แข่ง/ภาครัฐ/ตลาดโลก)
2) composition bias: บริษัทที่เข้าโครงการช่วงหลังมองโลกในแง่ดีกว่า (optimism drift)
3) สัดส่วนสาขาในกลุ่มตัวอย่างไม่ตรงกับโครงสร้าง GRP จึงต้องถ่วงน้ำหนัก
4) หัวข้อ outlook มีข้อมูลล่วงหน้าของเศรษฐกิจไตรมาสถัดไปบางส่วน
ชื่อบริษัท บุคคล และเบอร์โทรทั้งหมดเป็นข้อมูลสมมติ
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import REGIONS, SECTOR_WEIGHTS, SECTORS, TOPICS

PROVINCES = {
    "เหนือ": ["เชียงใหม่", "เชียงราย", "ลำปาง", "พิษณุโลก", "นครสวรรค์"],
    "ตะวันออกเฉียงเหนือ": ["ขอนแก่น", "นครราชสีมา", "อุดรธานี", "อุบลราชธานี"],
    "ใต้": ["สงขลา", "ภูเก็ต", "สุราษฎร์ธานี", "นครศรีธรรมราช"],
}
# สัดส่วนสาขาในกลุ่มตัวอย่าง (purposive sample) ตั้งให้ต่างจากน้ำหนัก GRP โดยเจตนา
SAMPLE_MIX = {"เกษตรแปรรูป": 0.15, "อุตสาหกรรมการผลิต": 0.35, "ท่องเที่ยวและโรงแรม": 0.15,
              "ค้าปลีกค้าส่ง": 0.25, "อสังหาริมทรัพย์และก่อสร้าง": 0.10}

PHRASES: dict[str, dict[int, list[str]]] = {
    "sales": {
        1: ["ยอดขายปรับเพิ่มขึ้นจากไตรมาสก่อน", "คำสั่งซื้อจากลูกค้าเพิ่มขึ้นชัดเจน", "ยอดขายขยายตัวดีตามกำลังซื้อที่ฟื้นตัว"],
        0: ["ยอดขายทรงตัวใกล้เคียงไตรมาสก่อน", "ยอดขายโดยรวมไม่เปลี่ยนแปลงมากนัก"],
        -1: ["ยอดขายลดลงจากไตรมาสก่อน", "คำสั่งซื้อชะลอตัวลงตามกำลังซื้อที่อ่อนแอ", "ยอดขายหดตัวต่อเนื่อง"],
    },
    "employment": {
        1: ["มีการจ้างงานเพิ่มขึ้น", "รับพนักงานใหม่เพิ่มเพื่อรองรับคำสั่งซื้อ"],
        0: ["จำนวนพนักงานคงที่", "ยังไม่มีแผนลดพนักงาน"],
        -1: ["ลดจำนวนพนักงานลง", "ลดชั่วโมงทำงานล่วงเวลาและชะลอการจ้างงาน"],
    },
    "investment": {
        1: ["มีแผนลงทุนขยายกำลังการผลิตเพิ่มขึ้น", "ลงทุนเครื่องจักรใหม่เพิ่มขึ้น"],
        0: ["การลงทุนทรงตัว ลงทุนเพื่อซ่อมบำรุงตามปกติ"],
        -1: ["เลื่อนแผนการลงทุนออกไป", "ลดงบลงทุนลงจากปีก่อน"],
    },
    "price": {
        1: ["ปรับขึ้นราคาขายสินค้า", "ราคาขายปรับสูงขึ้นตามต้นทุน"],
        0: ["ราคาขายคงที่", "ยังไม่ปรับขึ้นราคาขาย"],
        -1: ["ลดราคาขายเพื่อกระตุ้นยอดขาย", "ราคาขายปรับลดลงจากการแข่งขันที่รุนแรง"],
    },
    "cost": {
        1: ["ต้นทุนวัตถุดิบปรับสูงขึ้น", "ต้นทุนพลังงานและค่าแรงเพิ่มขึ้น"],
        0: ["ต้นทุนทรงตัว"],
        -1: ["ต้นทุนวัตถุดิบลดลง", "ต้นทุนการผลิตปรับลดลง"],
    },
    "outlook": {
        1: ["คาดว่าธุรกิจจะปรับดีขึ้นในไตรมาสหน้า", "มองแนวโน้มธุรกิจในระยะข้างหน้าเชิงบวก"],
        0: ["คาดว่าธุรกิจจะทรงตัวในระยะข้างหน้า"],
        -1: ["กังวลว่าธุรกิจจะชะลอลงในไตรมาสหน้า", "มองแนวโน้มธุรกิจในระยะข้างหน้าเชิงลบ"],
    },
}
SECTOR_SALES = {
    "ท่องเที่ยวและโรงแรม": {
        1: ["จำนวนนักท่องเที่ยวและอัตราการเข้าพักเพิ่มขึ้น", "ยอดจองห้องพักล่วงหน้าเพิ่มขึ้นมาก"],
        0: ["อัตราการเข้าพักทรงตัว"],
        -1: ["นักท่องเที่ยวลดลงทำให้อัตราการเข้าพักต่ำลง", "ยอดจองห้องพักชะลอลง"],
    },
    "อสังหาริมทรัพย์และก่อสร้าง": {
        1: ["ยอดโอนกรรมสิทธิ์และยอดขายโครงการใหม่เพิ่มขึ้น"],
        0: ["ยอดขายโครงการทรงตัว"],
        -1: ["ยอดโอนกรรมสิทธิ์ลดลงจากการปฏิเสธสินเชื่อที่สูงขึ้น"],  # มีทั้งคำขึ้นและลง: กรณียาก
    },
}
MIXED = {  # ทิศทางผสมในหัวข้อเดียว -> gold = 0
    "sales": "ยอดขายในประเทศเพิ่มขึ้น แต่ยอดขายส่งออกลดลง",
    "employment": "จ้างพนักงานประจำเพิ่มขึ้น แต่ลดแรงงานชั่วคราวลง",
    "investment": "ลงทุนเครื่องจักรอัตโนมัติเพิ่มขึ้น แต่เลื่อนแผนขยายสาขาออกไป",
}
DISTRACTORS = [  # ไม่ใช่ข้อมูลของกิจการ แต่มีคำสำคัญที่ทำให้วิธี lexicon สับสน
    "คู่แข่งรายใหญ่ขยายสาขาเพิ่มขึ้นในพื้นที่",
    "ราคาน้ำมันในตลาดโลกปรับสูงขึ้นแต่ยังไม่กระทบต้นทุนของกิจการมากนัก",
    "ภาครัฐเร่งเบิกจ่ายงบลงทุนโครงการโครงสร้างพื้นฐาน",
    "ผู้ประกอบการกังวลเรื่องหนี้ครัวเรือนของลูกค้า",
    "ปัญหาขาดแคลนแรงงานทักษะยังมีอยู่",
]
_NAME_PRE = ["ล้านนา", "อีสาน", "ทักษิณ", "สยาม", "รุ่งเรือง", "เจริญ", "มั่นคง", "ศรีสุข"]
_NAME_SUF = ["ฟู้ด", "เทรดดิ้ง", "อุตสาหกรรม", "โฮเทล", "ก่อสร้าง", "การเกษตร", "พาณิชย์"]
_PERSONS = ["สมชาย", "สมหญิง", "วิไลวรรณ", "ประเสริฐ", "กมลชนก", "ธนพล", "อรุณี", "สุรเชษฐ์"]


@dataclass
class SyntheticBLP:
    notes: pd.DataFrame        # note_id, firm_id, region, province, sector, quarter, interview_date, raw_text
    gold: pd.DataFrame         # note_id, topic, label (-1/0/1 หรือ NaN = ไม่กล่าวถึง)
    hard: pd.DataFrame         # region, quarter, y (ข้อมูลทางการจำลอง), activity (ค่าจริงแฝง)
    entities: dict[str, str]   # รายชื่อบริษัท/ผู้ติดต่อ -> placeholder (เหมือนฐานข้อมูลผู้ติดต่อ BLP)
    sector_weights: dict[str, dict[str, float]]


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, sd)
    return x


def _direction(z: float, cut: float = 0.55) -> int:
    return 1 if z > cut else (-1 if z < -cut else 0)


def generate(seed: int = 42, start: str = "2012Q1", end: str = "2025Q4", firms_per_quarter: int = 45,
             retention: float = 0.80, optimism_drift: float = 0.8, p_mention: float = 0.85,
             p_mixed: float = 0.08, p_distractor: float = 0.25) -> SyntheticBLP:
    rng = np.random.default_rng(seed)
    quarters = pd.period_range(start, end, freq="Q")
    T = len(quarters)

    # ปัจจัยแฝง: เศรษฐกิจทั้งประเทศ (g), เงินเฟ้อ/ต้นทุน (pi), ภาค (f), shock รายสาขา
    g = _ar1(rng, T, 0.6, 0.55)
    pi = _ar1(rng, T, 0.7, 0.45)
    covid = pd.Period("2020Q2", freq="Q")
    tour = {s: np.zeros(T) for s in SECTORS}
    if covid in quarters:
        k = quarters.get_loc(covid)
        g[k] -= 3.0
        if k + 1 < T:
            g[k + 1] -= 1.0
        tour["ท่องเที่ยวและโรงแรม"][k:min(k + 7, T)] = -1.5   # ท่องเที่ยวฟื้นช้ากว่าสาขาอื่น
    loading = {"เหนือ": 1.0, "ตะวันออกเฉียงเหนือ": 0.8, "ใต้": 1.2}
    f = {r: loading[r] * g + _ar1(rng, T, 0.5, 0.4) for r in REGIONS}

    hard_rows, note_rows, gold_rows = [], [], []
    entities: dict[str, str] = {}
    firm_no = 0
    note_no = 0
    sectors = list(SAMPLE_MIX)
    mix = np.array([SAMPLE_MIX[s] for s in sectors])

    for r in REGIONS:
        w = SECTOR_WEIGHTS[r]
        activity = f[r] + sum(w[s] * tour[s] for s in SECTORS)
        y = 2.0 + 1.8 * activity + rng.normal(0.0, 0.6, T)          # อัตราการขยายตัว (%YoY) จำลอง
        hard_rows += [{"region": r, "quarter": q, "y": y[t], "activity": activity[t]} for t, q in enumerate(quarters)]

        active: list[dict] = []
        for t, q in enumerate(quarters):
            active = [firm for firm in active if rng.random() < retention]
            while len(active) < firms_per_quarter:
                firm_no += 1
                person = str(rng.choice(_PERSONS))
                firm = {
                    "firm_id": f"F{firm_no:05d}",
                    "sector": str(rng.choice(sectors, p=mix)),
                    "province": str(rng.choice(PROVINCES[r])),
                    # optimism drift: ผู้เข้าใหม่ช่วงหลังมองโลกในแง่ดีกว่า -> composition bias
                    "alpha": rng.normal(optimism_drift * t / max(T - 1, 1) - optimism_drift / 2, 0.5),
                    "name": f"บริษัท {rng.choice(_NAME_PRE)}{rng.choice(_NAME_SUF)}{firm_no} จำกัด",
                    "person": f"คุณ{person}",
                    "phone": f"08{rng.integers(1, 10)}-{rng.integers(100, 1000)}-{rng.integers(1000, 10000)}",
                }
                entities[firm["name"]] = "[บริษัท]"
                entities[firm["person"]] = "[บุคคล]"
                active.append(firm)

            nxt = min(t + 1, T - 1)
            for firm in active:
                note_no += 1
                note_id = f"N{note_no:06d}"
                s, a = firm["sector"], firm["alpha"]
                act = f[r][t] + tour[s][t]
                act_next = f[r][nxt] + tour[s][nxt]
                z = {
                    "sales": act + a + rng.normal(0, 0.7),
                    "employment": 0.6 * act + 0.7 * a + rng.normal(0, 0.7),
                    "investment": 0.7 * act + 0.8 * a + rng.normal(0, 0.8),
                    "price": 0.7 * pi[t] + 0.3 * act + 0.3 * a + rng.normal(0, 0.7),
                    "cost": 0.9 * pi[t] + rng.normal(0, 0.7),
                    "outlook": 0.6 * act_next + 0.3 * act + a + rng.normal(0, 0.7),
                }
                sentences, order = [], list(TOPICS[:-1])
                rng.shuffle(order)
                for topic in order + ["outlook"]:
                    p = 0.90 if topic == "outlook" else p_mention
                    if rng.random() >= p:
                        gold_rows.append({"note_id": note_id, "topic": topic, "label": np.nan})
                        continue
                    d = _direction(z[topic])
                    if topic in MIXED and rng.random() < p_mixed:
                        sentences.append(MIXED[topic])
                        d = 0
                    elif topic == "sales" and s in SECTOR_SALES and rng.random() < 0.7:
                        sentences.append(str(rng.choice(SECTOR_SALES[s][d])))
                    else:
                        sentences.append(str(rng.choice(PHRASES[topic][d])))
                    gold_rows.append({"note_id": note_id, "topic": topic, "label": float(d)})
                if rng.random() < p_distractor:
                    sentences.insert(int(rng.integers(0, len(sentences) + 1)), str(rng.choice(DISTRACTORS)))
                if sentences and rng.random() < 0.3:
                    sentences[0] = f"{firm['name']} ระบุว่า{sentences[0]}"
                date = (q.start_time + pd.Timedelta(days=int(rng.integers(5, 80)))).date().isoformat()
                header = (f"[{date}] {firm['name']} | สาขา{s} | จ.{firm['province']} | "
                          f"ผู้ให้ข้อมูล: {firm['person']} โทร {firm['phone']}")
                note_rows.append({"note_id": note_id, "firm_id": firm["firm_id"], "region": r,
                                  "province": firm["province"], "sector": s, "quarter": q,
                                  "interview_date": date, "raw_text": header + "\n" + " ".join(sentences)})

    return SyntheticBLP(notes=pd.DataFrame(note_rows), gold=pd.DataFrame(gold_rows),
                        hard=pd.DataFrame(hard_rows), entities=entities, sector_weights=SECTOR_WEIGHTS)


def simulate_coders(gold: pd.DataFrame, n_notes: int = 400, accuracy: float = 0.9,
                    seed: int = 7) -> pd.DataFrame:
    """จำลองผู้ลงรหัส 2 คน (A, B) ที่อ่านข้อความเดียวกันอย่างอิสระ เพื่อสาธิตการคำนวณ inter-coder reliability"""
    rng = np.random.default_rng(seed)
    ids = rng.choice(gold["note_id"].unique(), size=min(n_notes, gold["note_id"].nunique()), replace=False)
    sub = gold[gold["note_id"].isin(ids)].reset_index(drop=True)
    cats = [-1.0, 0.0, 1.0, np.nan]

    def code(label: float) -> float:
        if rng.random() < (0.97 if np.isnan(label) else accuracy):
            return label
        others = [c for c in cats if not (c == label or (np.isnan(c) and np.isnan(label)))]
        return others[int(rng.integers(0, len(others)))]

    return sub.assign(coder_a=[code(x) for x in sub["label"]], coder_b=[code(x) for x in sub["label"]])
